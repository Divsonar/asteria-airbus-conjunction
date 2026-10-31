"""Exploratory risk-review classifier; does not change production artifacts.

Uses the original train/validation/holdout event groups from training only.
It trains a fixed weighted point model and class-balanced classifier on the
training events, chooses a classifier threshold on validation misses, then
evaluates the previously inspected holdout. Test-file labels are never read.

The classifier's output is a score, not a calibrated probability. A proposed
flag is only a prompt for analyst review; it never changes an Airbus tier.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ARTIFACTS, DATA_DIR, clean_features, latest_message, load_messages, make_model, risk_sample_weights


def selection_threshold(frame: pd.DataFrame, point: np.ndarray, score: np.ndarray, target_recall: float = .95) -> float | None:
    """Largest threshold capturing target share of validation point misses."""
    truth = frame.risk.to_numpy(dtype=float)
    distance = frame.miss_distance.to_numpy(dtype=float)
    point, score = np.asarray(point, dtype=float), np.asarray(score, dtype=float)
    eligible = np.isfinite(truth) & (truth <= 0) & np.isfinite(distance) & (distance >= 1000)
    missed = eligible & (truth >= -4) & (~np.isfinite(point) | (point < -4))
    if not missed.any():
        return None
    if not 0 < target_recall <= 1:
        raise ValueError("Target recall must be >0 and <=1.")
    rank = math.ceil(target_recall * int(missed.sum()))
    values = score[missed & np.isfinite(score)]
    if len(values) < rank:
        raise ValueError("Missing review scores prevent the requested validation recall target.")
    return float(np.sort(values)[::-1][rank - 1])


def evaluate(frame: pd.DataFrame, point: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """Count added review flags, using truth solely for offline evaluation."""
    truth = frame.risk.to_numpy(dtype=float)
    distance = frame.miss_distance.to_numpy(dtype=float)
    point, score = np.asarray(point, dtype=float), np.asarray(score, dtype=float)
    eligible = np.isfinite(truth) & (truth <= 0) & np.isfinite(distance) & (distance >= 1000)
    critical = eligible & (truth >= -4)
    would_miss = ~np.isfinite(point) | (point < -4)
    miss = critical & would_miss
    review = eligible & would_miss & np.isfinite(score) & (score >= threshold)
    true_flags = review & critical
    false_flags = review & ~critical
    n = int(len(frame))
    count = lambda x: int(x.sum())
    unique = lambda mask: int(frame.loc[mask, "event_id"].nunique())
    return {
        "n_rows": n, "n_events": int(frame.event_id.nunique()),
        "eligible_rows": count(eligible), "unscorable_point_rows": count(eligible & ~np.isfinite(point)),
        "unscorable_review_score_rows": count(eligible & ~np.isfinite(score)),
        "pc_driven_positive_rows": count(critical), "pc_driven_positive_events": unique(critical),
        "point_model_missed_pc_rows": count(miss), "point_model_missed_pc_events": unique(miss),
        "additional_review_rows": count(review), "additional_review_events": unique(review),
        "additional_review_rate_all_rows": count(review)/n if n else None,
        "true_misses_flagged_rows": count(true_flags), "true_misses_flagged_events": unique(true_flags),
        "point_miss_recall": count(true_flags)/count(miss) if count(miss) else None,
        "false_review_rows": count(false_flags), "false_review_events": unique(false_flags),
        "review_precision": count(true_flags)/count(review) if count(review) else None,
        "remaining_pc_driven_missed_rows": count(miss & ~review),
    }


def run() -> dict:
    split = json.loads((ARTIFACTS / "split_events.json").read_text(encoding="utf-8"))
    rows = load_messages(DATA_DIR / "train_data.csv", training=True)
    rows = rows[np.isfinite(rows.risk) & (rows.risk <= 0)].copy()
    training = rows[rows.event_id.isin(split["train"])]
    validation = rows[rows.event_id.isin(split["validation"])]
    holdout = rows[rows.event_id.isin(split["holdout"])]
    assert set(training.event_id).isdisjoint(validation.event_id)
    assert set(training.event_id).isdisjoint(holdout.event_id)
    assert set(validation.event_id).isdisjoint(holdout.event_id)
    eligible_train = training[np.isfinite(training.miss_distance) & (training.miss_distance >= 1000)]
    labels = (eligible_train.risk >= -4).astype(int)
    if labels.nunique() != 2:
        raise ValueError("Training subset must contain both high- and low-risk messages.")
    print("Training exploratory point model and review-score classifier...", file=sys.stderr, flush=True)
    point_model = make_model()
    point_model.fit(clean_features(training), training.risk, estimator__sample_weight=risk_sample_weights(training))
    preprocessor = make_model().named_steps["preprocess"]
    classifier = Pipeline([("preprocess", preprocessor), ("estimator", HistGradientBoostingClassifier(
        learning_rate=.09, max_iter=180, max_leaf_nodes=31, l2_regularization=1.,
        early_stopping=False, class_weight="balanced", random_state=42))])
    classifier.fit(clean_features(eligible_train), labels)

    def predict(frame):
        x = clean_features(frame)
        return np.minimum(point_model.predict(x), 0.0), classifier.predict_proba(x)[:, 1]

    val_point, val_score = predict(validation)
    threshold = selection_threshold(validation, val_point, val_score)
    if threshold is None:
        raise ValueError("Validation has no Pc-driven point-model misses; threshold cannot be selected.")
    hold_point, hold_score = predict(holdout)
    latest = latest_message(holdout)
    latest_point, latest_score = predict(latest)
    early = holdout[holdout.time_to_tca >= 2]
    early_point, early_score = predict(early)
    early_latest = latest_message(early)
    early_latest_point, early_latest_score = predict(early_latest)
    sweep = []
    for target in (.95, .90, .80, .50):
        selected = selection_threshold(validation, val_point, val_score, target_recall=target)
        sweep.append({"target_validation_miss_recall": target, "threshold": selected,
                      "validation": evaluate(validation, val_point, val_score, selected),
                      "holdout_rows": evaluate(holdout, hold_point, hold_score, selected),
                      "holdout_latest_event": evaluate(latest, latest_point, latest_score, selected)})
    result = {
        "classifier": "class-balanced histogram gradient-boosting; score is uncalibrated",
        "point_model": "risk-weighted histogram gradient-boosting refit on training events only; not the production fitted artifact",
        "threshold": threshold,
        "threshold_selection": "Highest validation score threshold flagging at least 95% of Pc-driven critical messages missed by the point model; no parameter tuning on holdout.",
        "train_rows": int(len(training)), "train_events": int(training.event_id.nunique()),
        "classifier_train_rows": int(len(eligible_train)), "classifier_train_positives": int(labels.sum()),
        "validation": evaluate(validation, val_point, val_score, threshold),
        "holdout_rows": evaluate(holdout, hold_point, hold_score, threshold),
        "holdout_latest_event": evaluate(latest, latest_point, latest_score, threshold),
        "holdout_at_least_two_days": evaluate(early, early_point, early_score, threshold),
        "holdout_latest_at_least_two_days": evaluate(early_latest, early_latest_point, early_latest_score, threshold),
        "exploratory_validation_threshold_sweep": sweep,
        "limitations": [
            "The validation group has been used in prior model selection, and the holdout has been previously inspected; this is exploratory, not blind testing.",
            "The score is uncalibrated. The output is an additional review suggestion, not a risk probability, tier change, maneuver decision or safe clearance.",
            "Rows are repeated messages per event and are not independent. Event counts and latest-message results are included; the selected dataset is not population-representative.",
            "The validation threshold targets observed point-model misses only, without any guarantee on independent data.",
            "The additional threshold sweep was added after inspecting earlier results and is a post-hoc exploration, not a basis for selecting a deployable operating point.",
            "Training on allowed same-message features may partially reconstruct a risk value already reported in each CDM; this is not a forecast of final risk.",
            "The original test-file risk labels were not read.",
        ],
    }
    output = ROOT / "experiments" / "critical_review_results.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
