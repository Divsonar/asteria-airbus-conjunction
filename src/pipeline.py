"""Reproducible row-level risk surrogate and Airbus challenge rules.

This model estimates the risk recorded on the supplied conjunction message.
It does not forecast the last message's risk or calculate a physical Pc.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.risk import classify
from src.provenance import write_manifest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Challenge 1 Airbus_"
ARTIFACTS = ROOT / "artifacts"
FEATURES = [
    "time_to_tca", "miss_distance", "relative_speed", "relative_position_r",
    "relative_position_t", "relative_position_n", "c_object_type", "c_sigma_r",
    "t_sigma_r", "max_risk_estimate", "F10", "AP",
]
KEYS = ["event_id", *FEATURES]
NUMERIC_FEATURES = [f for f in FEATURES if f != "c_object_type"]
TIERS = ["Critical", "Elevated", "Safe", "Incomplete"]


def load_messages(path: Path, training: bool = False) -> pd.DataFrame:
    """Read explicitly allowed fields; test risk is never read into memory."""
    columns = KEYS + (["risk"] if training else [])
    frame = pd.read_csv(path, usecols=lambda col: col in columns, low_memory=False)
    missing = [col for col in columns if col not in frame]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {', '.join(missing)}")
    frame.insert(0, "source_row_id", np.arange(len(frame), dtype=np.int64))
    for col in NUMERIC_FEATURES + (["risk"] if training else []):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame["c_object_type"] = frame["c_object_type"].fillna("UNKNOWN").astype(str)
    return frame


def clean_features(frame: pd.DataFrame, feature_names=FEATURES) -> pd.DataFrame:
    """Accept telemetry with extra columns but pass only allowed inputs to a model."""
    missing = [col for col in feature_names if col not in frame]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    x = frame.loc[:, feature_names].copy()
    for col in feature_names:
        if col != "c_object_type":
            x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "c_object_type" in x:
        x["c_object_type"] = x["c_object_type"].fillna("UNKNOWN").astype(str)
    return x


def make_model(feature_names=FEATURES, baseline=False):
    numeric = [name for name in feature_names if name != "c_object_type"]
    transformers = [("numerical", SimpleImputer(strategy="median"), numeric)]
    if "c_object_type" in feature_names:
        transformers.append(("object", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["c_object_type"]))
    prep = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    estimator = DummyRegressor(strategy="median") if baseline else HistGradientBoostingRegressor(
        learning_rate=0.09, max_iter=180, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=False, random_state=42,
    )
    return Pipeline([("preprocess", prep), ("estimator", estimator)])


def risk_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Weights fixed before this validation experiment, after prior holdout inspection."""
    truth = frame["risk"].to_numpy(dtype=float)
    return np.where(truth >= -4, 20.0, np.where(truth >= -6, 8.0, 1.0))


def choose_weighted_candidate(unweighted: dict, weighted: dict) -> bool:
    """Use a weighted candidate only when the current experiment's guardrails pass.

    Validation decisions are based on event-disjoint validation data, after the
    holdout had already been inspected during earlier development.
    Precision and overall error are guardrails against buying recall at an
    unbounded false-alarm or regression-error cost.
    """
    keys = ["pc_driven_critical_recall", "pc_driven_critical_precision",
            "mae_log10", "mae_log10_risk_ge_minus6"]
    if any(unweighted.get(k) is None or weighted.get(k) is None for k in keys):
        return False
    return bool(
        weighted["pc_driven_critical_recall"] > unweighted["pc_driven_critical_recall"]
        and weighted["pc_driven_critical_precision"] >= max(0.0, unweighted["pc_driven_critical_precision"] - 0.05)
        and weighted["mae_log10_risk_ge_minus6"] < unweighted["mae_log10_risk_ge_minus6"]
        and weighted["mae_log10"] <= 1.25 * unweighted["mae_log10"]
    )


def predict_messages(frame: pd.DataFrame, model=None, model_path=ARTIFACTS / "risk_model.joblib") -> pd.DataFrame:
    """Predict for unseen messages and retain only allowed fields in output."""
    if model is None:
        model = joblib.load(model_path)
    missing = [k for k in KEYS if k not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    result = frame.loc[:, KEYS].copy()
    if "source_row_id" in frame:
        result.insert(0, "source_row_id", frame["source_row_id"].to_numpy())
    else:
        result.insert(0, "source_row_id", np.arange(len(frame)))
    x = clean_features(frame)
    # Return normalized numeric fields to ensure invalid upload text is reported
    # as missing telemetry rather than breaking downstream charts or filters.
    for name in NUMERIC_FEATURES:
        result[name] = x[name].to_numpy()
    raw_prediction = model.predict(x)
    # A probability cannot exceed one. Preserve the unmodified model output for audit.
    finite_prediction = np.isfinite(raw_prediction)
    display_prediction = np.where(finite_prediction, np.minimum(raw_prediction, 0.0), np.nan)
    result["raw_predicted_log10_pc"] = raw_prediction
    result["predicted_log10_pc"] = display_prediction
    result["estimated_pc"] = np.power(10.0, display_prediction)
    tiers, rules, quality = classify(display_prediction, result["miss_distance"])
    result["risk_tier"], result["triggered_rules"], result["data_quality"] = tiers, rules, quality
    # Imputation is explicit in the output, so users can see incomplete telemetry.
    feature_missing = x.isna()
    for pos in np.flatnonzero(feature_missing.any(axis=1).to_numpy()):
        names = ", ".join(feature_missing.columns[feature_missing.iloc[pos]].tolist())
        existing = result.iloc[pos, result.columns.get_loc("data_quality")]
        detail = f"Imputed model inputs: {names}"
        result.iloc[pos, result.columns.get_loc("data_quality")] = detail if existing == "Complete" else f"{existing}; {detail}"
    result["time_to_tca_days"] = pd.to_numeric(result["time_to_tca"], errors="coerce")
    result["miss_distance_m"] = pd.to_numeric(result["miss_distance"], errors="coerce")
    return result


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    truth = frame["risk"].to_numpy(dtype=float)
    distance = frame["miss_distance"].to_numpy(dtype=float)
    prediction = np.where(np.isfinite(prediction), np.minimum(prediction, 0.0), np.nan)
    finite = np.isfinite(truth) & (truth <= 0) & np.isfinite(prediction)
    true_tier, _, _ = classify(truth, distance)
    pred_tier, _, _ = classify(prediction, distance)
    true_tier = np.asarray(true_tier)
    pred_tier = np.asarray(pred_tier)
    critical = true_tier == "Critical"
    predicted_critical = pred_tier == "Critical"
    tp = int((critical & predicted_critical).sum())
    fn = int((critical & ~predicted_critical).sum())
    fp = int((~critical & predicted_critical).sum())
    valid_truth = np.isfinite(truth) & (truth <= 0)
    valid_prediction = np.isfinite(prediction)
    high_risk = valid_truth & (truth >= -4)
    high_risk_pred = valid_prediction & (prediction >= -4)
    risk_tp = int((high_risk & high_risk_pred).sum())
    risk_fn = int((high_risk & ~high_risk_pred).sum())
    # Distance independently triggers Critical below 1 km, so evaluate model
    # discrimination only where the probability estimate affects that outcome.
    # Rows with unusable distance or label are excluded from the denominator;
    # missing predictions on a known critical case are counted as misses.
    pc_scope = valid_truth & np.isfinite(distance) & (distance >= 1000.0)
    pc_truth = pc_scope & (truth >= -4)
    pc_pred = pc_scope & valid_prediction & (prediction >= -4)
    pc_tp = int((pc_truth & pc_pred).sum())
    pc_fn = int((pc_truth & ~pc_pred).sum())
    pc_fp = int((~pc_truth & pc_pred).sum())
    relevant_risk = finite & (truth >= -6)
    return {
        "n_rows": int(len(frame)), "n_events": int(frame["event_id"].nunique()),
        "n_valid_truth_rows": int(valid_truth.sum()), "n_scored_rows": int(finite.sum()),
        "n_unscorable_predictions_valid_truth": int((valid_truth & ~valid_prediction).sum()),
        "mae_log10": float(mean_absolute_error(truth[finite], prediction[finite])) if finite.any() else None,
        "rmse_log10": float(np.sqrt(mean_squared_error(truth[finite], prediction[finite]))) if finite.any() else None,
        "critical_recall": tp / (tp + fn) if tp + fn else None,
        "critical_precision": tp / (tp + fp) if tp + fp else None,
        "critical_true": tp + fn, "critical_tp": tp, "critical_fn": fn, "critical_fp": fp,
        "risk_only_critical_recall": risk_tp / (risk_tp + risk_fn) if risk_tp + risk_fn else None,
        "risk_only_critical_true": risk_tp + risk_fn, "risk_only_critical_fn": risk_fn,
        "risk_only_critical_unscorable": int((high_risk & ~valid_prediction).sum()),
        "pc_driven_critical_recall": pc_tp / (pc_tp + pc_fn) if pc_tp + pc_fn else None,
        "pc_driven_critical_precision": pc_tp / (pc_tp + pc_fp) if pc_tp + pc_fp else None,
        "pc_driven_critical_true": pc_tp + pc_fn, "pc_driven_critical_tp": pc_tp,
        "pc_driven_critical_fn": pc_fn, "pc_driven_critical_fp": pc_fp,
        "pc_driven_predicted_critical": pc_tp + pc_fp,
        "pc_driven_critical_unscorable": int((pc_truth & ~valid_prediction).sum()),
        "pc_driven_eligible_rows": int(pc_scope.sum()),
        "pc_driven_critical_events": int(frame.loc[pc_truth, "event_id"].nunique()),
        "mae_log10_risk_ge_minus6": float(mean_absolute_error(truth[relevant_risk], prediction[relevant_risk])) if relevant_risk.any() else None,
        "n_risk_ge_minus6": int(relevant_risk.sum()),
        "n_risk_ge_minus6_truth": int((valid_truth & (truth >= -6)).sum()),
        "n_risk_ge_minus6_unscorable": int((valid_truth & (truth >= -6) & ~valid_prediction).sum()),
        "confusion_matrix": {"labels": TIERS, "values": confusion_matrix(true_tier, pred_tier, labels=TIERS).tolist()},
    }


def distance_only_metrics(frame: pd.DataFrame) -> dict:
    """Evaluate distance-only tiers. This baseline does not estimate Pc.

    A hypothetical extremely low risk is used only to activate the exact same
    distance rules. Remove probability-error metrics: no Pc was predicted.
    """
    result = metrics(frame, np.full(len(frame), -30.0))
    for key in ("mae_log10", "rmse_log10", "mae_log10_risk_ge_minus6", "n_risk_ge_minus6",
                "n_risk_ge_minus6_unscorable", "n_scored_rows", "n_unscorable_predictions_valid_truth"):
        result.pop(key, None)
    result["description"] = "Distance-only tiers; does not predict Pc."
    return result


def pc_driven_recall_interval(frame: pd.DataFrame, prediction: np.ndarray,
                              iterations: int = 2000, seed: int = 43) -> dict | None:
    """Event-cluster bootstrap interval on this selected historical sample.

    Resample whole event IDs so repeated messages from the same event remain
    together. A single positive event cannot support a meaningful interval.
    This is not a claim of generalization to the operational population.
    """
    truth = frame["risk"].to_numpy(dtype=float)
    dist = frame["miss_distance"].to_numpy(dtype=float)
    pred = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & (truth <= 0) & np.isfinite(dist) & (dist >= 1000)
    positive = valid & (truth >= -4)
    true_positive = positive & np.isfinite(pred) & (pred >= -4)
    grouped = pd.DataFrame({"event_id": frame["event_id"].to_numpy(),
                            "positive": positive.astype(int), "tp": true_positive.astype(int)})
    grouped = grouped.groupby("event_id", sort=True, dropna=True)[["positive", "tp"]].sum()
    count = len(grouped)
    positive_events = int((grouped["positive"] > 0).sum())
    if count < 2 or positive_events < 2 or iterations < 1:
        return None
    rng = np.random.default_rng(seed)
    event_positive = grouped["positive"].to_numpy()
    event_tp = grouped["tp"].to_numpy()
    scores = []
    # Calculate in chunks to avoid a large n_events x iterations allocation.
    for _ in range(iterations):
        sample = rng.integers(0, count, size=count)
        denominator = int(event_positive[sample].sum())
        if denominator:
            scores.append(float(event_tp[sample].sum() / denominator))
    if not scores:
        return None
    lo, hi = np.quantile(scores, [0.025, 0.975])
    return {"lower": float(lo), "upper": float(hi), "positive_events": positive_events,
            "events_resampled": count, "valid_resamples": len(scores), "requested_resamples": iterations,
            "method": "Event-cluster bootstrap on the selected, previously inspected historical holdout; not a population guarantee."}


def bucket_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> list[dict]:
    """Row-level, retrospective days-to-TCA buckets (not concurrent alerts)."""
    tca = frame["time_to_tca"].to_numpy(dtype=float)
    buckets = np.full(len(frame), -1, dtype=int)
    valid = np.isfinite(tca) & (tca >= 0)
    buckets[valid] = np.floor(tca[valid]).astype(int)
    entries = []
    for bucket in sorted(np.unique(buckets[valid])):
        mask = valid & (buckets == bucket)
        subset = frame.loc[mask]
        entries.append({"bucket": f"{bucket}–{bucket + 1} days", "start_days": int(bucket),
                        "end_days": int(bucket + 1), "model": metrics(subset, prediction[mask]),
                        "distance_only": distance_only_metrics(subset)})
    return entries


def latest_message(frame: pd.DataFrame) -> pd.DataFrame:
    """Select one closest nonnegative snapshot per event; ties use row order."""
    eligible = frame.loc[frame["time_to_tca"] >= 0]
    return eligible.sort_values(["time_to_tca", "source_row_id"]).drop_duplicates("event_id")


def train_pipeline(train_path=DATA_DIR / "train_data.csv", test_path=DATA_DIR / "test_data.csv", output_dir=ARTIFACTS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_messages(Path(train_path), training=True)
    bad_label = ~np.isfinite(train["risk"]) | (train["risk"] > 0)
    dropped = int(bad_label.sum())
    train = train.loc[~bad_label].reset_index(drop=True)
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=42)
    train_idx, remainder_idx = next(split1.split(train, groups=train["event_id"]))
    remainder = train.iloc[remainder_idx]
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=43)
    val_idx, hold_idx = next(split2.split(remainder, groups=remainder["event_id"]))
    fitting = train.iloc[train_idx]
    validation = remainder.iloc[val_idx]
    holdout = remainder.iloc[hold_idx]
    print(f"Grouped split: {fitting.event_id.nunique()} train, {validation.event_id.nunique()} validation, {holdout.event_id.nunique()} holdout events", flush=True)
    base = make_model(baseline=True)
    base.fit(clean_features(fitting), fitting["risk"])
    baseline_validation = metrics(validation, base.predict(clean_features(validation)))
    print("Fitting gradient boosted model", flush=True)
    candidate = make_model()
    candidate.fit(clean_features(fitting), fitting["risk"])
    candidate_validation = metrics(validation, candidate.predict(clean_features(validation)))
    print("Fitting risk-weighted gradient boosted validation experiment", flush=True)
    weighted = make_model()
    weighted.fit(clean_features(fitting), fitting["risk"], estimator__sample_weight=risk_sample_weights(fitting))
    weighted_validation = metrics(validation, weighted.predict(clean_features(validation)))
    no_max = [c for c in FEATURES if c != "max_risk_estimate"]
    print("Fitting ablation model without max_risk_estimate", flush=True)
    ablation = make_model(no_max)
    ablation.fit(clean_features(fitting, no_max), fitting["risk"])
    ablation_validation = metrics(validation, ablation.predict(clean_features(validation, no_max)))
    selected_baseline = candidate_validation["mae_log10"] >= baseline_validation["mae_log10"]
    selected_weighted = not selected_baseline and choose_weighted_candidate(candidate_validation, weighted_validation)
    chosen_name = ("Median baseline" if selected_baseline else
                   "Risk-weighted histogram gradient-boosted regressor" if selected_weighted else
                   "Histogram gradient-boosted regressor")
    model = make_model(baseline=selected_baseline)
    train_and_val = pd.concat([fitting, validation], ignore_index=True)
    print(f"Refitting selected model: {chosen_name}", flush=True)
    fit_args = {"estimator__sample_weight": risk_sample_weights(train_and_val)} if selected_weighted else {}
    model.fit(clean_features(train_and_val), train_and_val["risk"], **fit_args)
    baseline_final = make_model(baseline=True)
    baseline_final.fit(clean_features(train_and_val), train_and_val["risk"])
    holdout_pred = model.predict(clean_features(holdout))
    holdout_metrics = metrics(holdout, holdout_pred)
    holdout_metrics["pc_driven_recall_event_bootstrap95"] = pc_driven_recall_interval(holdout, holdout_pred)
    holdout_baseline_metrics = metrics(holdout, baseline_final.predict(clean_features(holdout)))
    latest = latest_message(holdout)
    latest_predictions = model.predict(clean_features(latest))
    latest_metrics = metrics(latest, latest_predictions)
    latest_metrics["pc_driven_recall_event_bootstrap95"] = pc_driven_recall_interval(latest, latest_predictions)
    latest_baseline_metrics = metrics(latest, baseline_final.predict(clean_features(latest)))
    two_day = holdout.loc[holdout["time_to_tca"] >= 2]
    two_day_metrics = metrics(two_day, model.predict(clean_features(two_day)))
    two_day_baseline_metrics = metrics(two_day, baseline_final.predict(clean_features(two_day)))
    holdout_export = predict_messages(holdout, model=model)
    holdout_export["true_log10_pc"] = holdout["risk"].to_numpy()
    holdout_export["true_risk_tier"] = classify(holdout["risk"], holdout["miss_distance"])[0]
    holdout_export.to_csv(output_dir / "holdout_predictions.csv", index=False)
    joblib.dump(model, output_dir / "risk_model.joblib", compress=3)
    test = load_messages(Path(test_path), training=False)
    pred = predict_messages(test, model=model)
    pred.to_csv(output_dir / "test_predictions.csv", index=False)
    assert len(pred) == len(test) and pred.source_row_id.is_unique
    split = {
        "train_events": int(fitting.event_id.nunique()), "validation_events": int(validation.event_id.nunique()),
        "holdout_events": int(holdout.event_id.nunique()), "seed": 42,
        "method": "GroupShuffleSplit by event ID within the training file; 60/20/20 events",
    }
    split_ids = {name: sorted(group["event_id"].unique().tolist()) for name, group in
                 [("train", fitting), ("validation", validation), ("holdout", holdout)]}
    (output_dir / "split_events.json").write_text(json.dumps(split_ids, indent=2), encoding="utf-8")
    report = {
        "model": {"name": chosen_name, "target": "risk recorded on the same message (log10 Pc)",
                  "features": FEATURES, "selection": "Median baseline if unweighted model has no validation MAE improvement; weighted candidate must improve Pc-driven recall and relevant-risk MAE, retain precision within 5 percentage points, and keep overall MAE within 125% of unweighted model."},
        "data": {"train_rows": int(len(train)), "train_events": int(train.event_id.nunique()),
                 "test_rows": int(len(test)), "test_events": int(test.event_id.nunique()), "dropped_invalid_training_labels": dropped},
        "split": split,
        "validation": {"baseline": baseline_validation, "model": candidate_validation, "without_max_risk_estimate": ablation_validation,
                       "risk_weighted_experiment": weighted_validation, "risk_weighted_selected": selected_weighted,
                       "distance_only_rows": distance_only_metrics(validation)},
        "holdout": {"rows": holdout_metrics, "baseline_rows": holdout_baseline_metrics,
                    "latest_event": latest_metrics, "baseline_latest_event": latest_baseline_metrics,
                    "at_least_two_days": two_day_metrics, "baseline_at_least_two_days": two_day_baseline_metrics,
                    "distance_only_rows": distance_only_metrics(holdout),
                    "distance_only_latest_event": distance_only_metrics(latest),
                    "time_to_tca_buckets": bucket_metrics(holdout, holdout_pred)},
        "notes": ["Test risk was excluded from input and model selection.",
                  "Risk is a surrogate for the same message, not a prediction of final event risk.",
                  "Latest-event metrics use the closest nonnegative time-to-TCA snapshot in the held-out event.",
                  "Selected model refitted on training and validation groups before holdout evaluation.",
                  "The holdout has been inspected during prior development and is not a fresh blind evaluation.",
                  "Time-to-TCA buckets are historical relative message times, not a simultaneous live forecast.",
                  "Pc-driven critical is risk >= -4 with miss distance >= 1000 m; in that subset the distance rule does not independently force Critical."]
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    # Publish checksums only after the complete model/prediction bundle exists.
    # Checksums detect later mismatches; they do not establish model lineage.
    write_manifest(output_dir)
    print(json.dumps({"model": chosen_name, "holdout": holdout_metrics, "latest_event": latest_metrics}, indent=2), flush=True)
    return report
