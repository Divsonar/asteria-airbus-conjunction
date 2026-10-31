"""Exploratory, event-separated interval study. Does not alter production files.

Fits on a subset of the original training events, calibrates on different
training events and scores the already-inspected holdout. No test CSV is
read. This study examines symmetric residual intervals for the reported
same-message log-risk target, not a physics-derived collision probability.

Example: ``python experiments/conformal_study.py --output experiments/conformal_results.json``
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ARTIFACTS, DATA_DIR, clean_features, latest_message, load_messages, make_model, risk_sample_weights


def conformal_quantile(scores, coverage: float = .8) -> float:
    """Finite-sample conformal order statistic; raises on impossible quantiles."""
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or not 0 < coverage < 1:
        raise ValueError("Calibration requires finite scores and coverage between 0 and 1.")
    # Unbounded split-conformal quantile when the target exceeds n/(n+1).
    rank = math.ceil((len(values) + 1) * coverage)
    if rank > len(values):
        raise ValueError("Not enough calibration units for this nominal coverage.")
    return float(np.partition(values, rank - 1)[rank - 1])


def summarize(frame: pd.DataFrame, predictions: np.ndarray, radius: float) -> dict:
    """Measure same-message coverage and alert-boundary usefulness."""
    y = frame["risk"].to_numpy(dtype=float)
    pred = np.asarray(predictions, dtype=float)
    distance = frame["miss_distance"].to_numpy(dtype=float)
    valid_truth = np.isfinite(y) & (y <= 0)
    valid = valid_truth & np.isfinite(pred)
    # Prediction values above log10(1) are physically impossible; the valid
    # label is <=0, so clipping the upper endpoint does not remove coverage.
    lower = pred - radius
    upper = np.minimum(pred + radius, 0.0)
    covered = valid & (y >= lower) & (y <= upper)
    widths = upper - lower
    pc_cases = valid_truth & np.isfinite(distance) & (distance >= 1000) & (y >= -4)
    predicted_low = ~np.isfinite(pred) | (pred < -4)
    missed_pc = pc_cases & predicted_low
    alert_ambiguous = valid & (lower < -4) & (upper >= -4)
    high_risk = valid_truth & (y >= -6)
    events = pd.DataFrame({"event": frame["event_id"].to_numpy()[valid_truth], "covered": covered[valid_truth]})
    if len(events):
        event_coverage = float(events.groupby("event")["covered"].all().mean())
        n_events = int(events.event.nunique())
    else:
        event_coverage, n_events = None, 0
    ratio = lambda numerator, denominator: float(numerator.sum() / denominator.sum()) if denominator.any() else None
    return {
        "n_rows": int(valid_truth.sum()), "n_events": n_events,
        "n_scored_rows": int(valid.sum()), "n_unscorable_predictions": int((valid_truth & ~np.isfinite(pred)).sum()),
        "row_coverage": ratio(covered, valid_truth), "all_messages_event_coverage": event_coverage,
        "mean_interval_width_log10": float(widths[valid].mean()) if valid.any() else None,
        "median_interval_width_log10": float(np.median(widths[valid])) if valid.any() else None,
        "critical_threshold_straddled_fraction": ratio(alert_ambiguous, valid),
        "high_risk_rows": int(high_risk.sum()), "high_risk_row_coverage": ratio(covered & high_risk, high_risk),
        "pc_driven_critical_rows": int(pc_cases.sum()), "pc_driven_critical_row_coverage": ratio(covered & pc_cases, pc_cases),
        "point_estimate_pc_driven_misses": int(missed_pc.sum()),
        "point_misses_with_interval_straddling_critical": int((missed_pc & alert_ambiguous).sum()),
    }


def run_study() -> dict:
    split = json.loads((ARTIFACTS / "split_events.json").read_text(encoding="utf-8"))
    train = load_messages(DATA_DIR / "train_data.csv", training=True)
    train = train[np.isfinite(train.risk) & (train.risk <= 0)].copy()
    eligible_fit = train[train.event_id.isin(split["train"])].copy()
    holdout = train[train.event_id.isin(split["holdout"])].copy()
    group_split = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=731)
    fit_index, calibration_index = next(group_split.split(eligible_fit, groups=eligible_fit.event_id))
    fitting = eligible_fit.iloc[fit_index]
    calibration = eligible_fit.iloc[calibration_index]
    assert set(fitting.event_id).isdisjoint(calibration.event_id)
    assert set(fitting.event_id).isdisjoint(holdout.event_id)
    assert set(calibration.event_id).isdisjoint(holdout.event_id)
    print("Fitting exploratory model on event-separated training data...", file=sys.stderr, flush=True)
    model = make_model()
    model.fit(clean_features(fitting), fitting.risk, estimator__sample_weight=risk_sample_weights(fitting))
    cal_pred = np.minimum(model.predict(clean_features(calibration)), 0.0)
    eval_pred = np.minimum(model.predict(clean_features(holdout)), 0.0)
    residual = np.abs(calibration.risk.to_numpy(dtype=float) - cal_pred)
    row_q = conformal_quantile(residual)
    event_residual = pd.DataFrame({"event_id": calibration.event_id.to_numpy(), "residual": residual}).groupby("event_id").residual.max()
    event_q = conformal_quantile(event_residual.to_numpy())
    latest = latest_message(holdout)
    latest_pred = np.minimum(model.predict(clean_features(latest)), 0.0)
    return {
        "nominal_coverage": .8,
        "sample": {"fit_events": int(fitting.event_id.nunique()), "calibration_events": int(calibration.event_id.nunique()),
                   "calibration_rows": int(len(calibration)), "holdout_events": int(holdout.event_id.nunique()),
                   "holdout_rows": int(len(holdout)), "split_seed": 731},
        "row_residual_calibration": {"radius_log10": row_q, "holdout_rows_and_events": summarize(holdout, eval_pred, row_q),
                                     "holdout_latest_events": summarize(latest, latest_pred, row_q)},
        "event_max_residual_calibration": {"radius_log10": event_q, "holdout_rows_and_events": summarize(holdout, eval_pred, event_q),
                                           "holdout_latest_events": summarize(latest, latest_pred, event_q)},
        "limits": [
            "Exploratory same-message risk intervals, not probabilities of collision or an operational safety guarantee.",
            "The holdout was previously inspected; the weighted model configuration was selected during prior development.",
            "Calibration and fitting event groups here do not overlap, but the selected historical sample may not be exchangeable with deployment data.",
            "Row-residual calibration does not establish event-cluster coverage; event-max calibration covers all observed messages in an event under exchangeability assumptions.",
            "Test-file labels were not read; the original challenge's future final-risk target was not evaluated.",
            "A substantial proportion of labels equal -30; the floor's semantics are not verified and heavily affect log-space intervals.",
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional experiment result JSON outside production artifacts")
    args = parser.parse_args()
    result = run_study()
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        if args.output.resolve().is_relative_to(ARTIFACTS.resolve()):
            parser.error("Do not write experimental output to production artifacts")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
