"""Forecast organizer final-event tiers from CDMs at least two days before TCA.

Outputs are uncalibrated class probabilities, never physical collision Pc.
Training and headline evaluation use one latest eligible CDM per event.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.provenance import write_manifest
from src.validation import read_telemetry_csv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Challenge 1 Airbus_"
ARTIFACTS = ROOT / "artifacts"
LABELS_PATH = ROOT / "docs" / "challenge1_event_labels.xlsx"
FEATURES = [
    "time_to_tca", "miss_distance", "relative_speed", "relative_position_r",
    "relative_position_t", "relative_position_n", "c_object_type", "c_sigma_r",
    "t_sigma_r", "max_risk_estimate", "F10", "AP",
]
KEYS = ["event_id", *FEATURES]
NUMERIC_FEATURES = [f for f in FEATURES if f != "c_object_type"]
CLASS_LABELS = ["Critical", "Elevated", "Safe"]
TIERS = CLASS_LABELS + ["Incomplete"]
PROBABILITY_NOTE = "Uncalibrated final-tier class probabilities; not collision probability (Pc)."
SEED = 42


def load_messages(path: Path, training: bool = False) -> pd.DataFrame:
    """Read allowed telemetry, preserving string IDs; never load CSV risk.

    ``training`` remains for callers. Targets always come from the workbook.
    """
    frame = read_telemetry_csv(path)
    missing = [col for col in KEYS if col not in frame]
    if missing:
        raise ValueError(f"Missing required columns in {Path(path).name}: {', '.join(missing)}")
    if "source_row_id" in frame:
        frame = frame.drop(columns="source_row_id")
    frame.insert(0, "source_row_id", np.arange(len(frame), dtype=np.int64))
    for col in NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame["c_object_type"] = frame["c_object_type"].fillna("UNKNOWN").astype(str)
    return frame


def clean_features(frame: pd.DataFrame, feature_names=FEATURES) -> pd.DataFrame:
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


def load_event_labels(path: Path = LABELS_PATH) -> pd.DataFrame:
    """Read organizer train_event_labels with stdlib XML, without Excel deps."""
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(node.itertext()) for node in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next((node for node in workbook.findall("m:sheets/m:sheet", namespace)
                      if node.attrib.get("name") == "train_event_labels"), None)
        if sheet is None:
            raise ValueError("Organizer workbook is missing train_event_labels")
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(node.attrib["Target"] for node in relationships if node.attrib["Id"] == rel_id)
        member = target.lstrip("/") if target.startswith("/") else "xl/" + target
        root = ET.fromstring(archive.read(member))
        rows = {}
        for row in root.findall("m:sheetData/m:row", namespace):
            values = {}
            for cell in row.findall("m:c", namespace):
                column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                inline, numeric = cell.find("m:is", namespace), cell.find("m:v", namespace)
                value = "".join(inline.itertext()) if inline is not None else (numeric.text if numeric is not None else None)
                if cell.attrib.get("t") == "s" and value is not None:
                    value = shared[int(value)]
                values[column] = value
            rows[int(row.attrib["r"])] = values
    headers = rows.get(3, {})
    labels = pd.DataFrame([{headers[key]: value for key, value in row.items() if key in headers}
                           for number, row in rows.items() if number >= 4 and row])
    required = ["event_id", "final_tier", "usable_for_training", "cdm_count", "cdm_count_2d_or_more",
                "final_time_to_tca_days", "final_risk_log10_pc", "final_miss_distance_m"]
    if any(key not in labels for key in required):
        raise ValueError("Organizer label workbook schema does not match the updated challenge")
    labels["event_id"] = labels["event_id"].astype("string")
    if labels.event_id.isna().any() or labels.event_id.eq("").any() or labels.event_id.duplicated().any():
        raise ValueError("Organizer event labels require unique nonempty string event IDs")
    if not labels.final_tier.isin(CLASS_LABELS).all():
        raise ValueError("Unknown or missing organizer final tier")
    if not labels.usable_for_training.isin(["yes", "no"]).all():
        raise ValueError("Invalid organizer usable_for_training marker")
    for col in ["cdm_count", "cdm_count_2d_or_more", "final_time_to_tca_days", "final_risk_log10_pc", "final_miss_distance_m"]:
        labels[col] = pd.to_numeric(labels[col], errors="raise")
    return labels


def latest_message(frame: pd.DataFrame) -> pd.DataFrame:
    """Select minimum time_to_tca >= 2 per event; tied times use source order."""
    times = pd.to_numeric(frame["time_to_tca"], errors="coerce")
    eligible = frame.loc[np.isfinite(times) & times.ge(2)].copy()
    eligible["time_to_tca"] = times.loc[eligible.index]
    if "source_row_id" not in eligible:
        eligible["source_row_id"] = np.flatnonzero(np.isfinite(times) & times.ge(2))
    return eligible.sort_values(["time_to_tca", "source_row_id"], kind="stable").drop_duplicates("event_id")


def prepare_training_events(messages: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join labels to TRAIN events only; no test file is accepted here."""
    # Remove incidental target-like columns before any join, including those
    # that a caller may have attached for display or an audit.
    messages = messages.loc[:, [name for name in ["source_row_id", *KEYS] if name in messages]].copy()
    messages["event_id"] = messages.event_id.astype("string")
    labels = labels.copy()
    labels["event_id"] = labels.event_id.astype("string")
    if labels.event_id.duplicated().any():
        raise ValueError("Duplicate organizer event IDs")
    if set(messages.event_id) != set(labels.event_id):
        raise ValueError("Train event IDs do not exactly match organizer labels; never join test IDs to train labels")
    counts = messages.groupby("event_id").size()
    early_counts = messages.loc[messages.time_to_tca.ge(2)].groupby("event_id").size().reindex(counts.index, fill_value=0)
    indexed = labels.set_index("event_id").loc[counts.index]
    if not np.array_equal(indexed.cdm_count.to_numpy(), counts.to_numpy()):
        raise ValueError("Organizer CDM counts do not match the training file")
    if not np.array_equal(indexed.cdm_count_2d_or_more.to_numpy(), early_counts.to_numpy()):
        raise ValueError("Organizer early-CDM counts do not match the training file")
    if not np.array_equal(indexed.usable_for_training.eq("yes").to_numpy(), early_counts.gt(0).to_numpy()):
        raise ValueError("Organizer training eligibility does not match the >=2 day cutoff")
    latest = latest_message(messages)
    joined = latest.merge(labels[["event_id", "final_tier"]], on="event_id", how="left", validate="one_to_one")
    if joined.final_tier.isna().any():
        raise ValueError("Missing final tiers after joining train events")
    return joined.sort_values("event_id", kind="stable").reset_index(drop=True)


def split_training_events(events: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.event_id.duplicated().any():
        raise ValueError("Split input must contain exactly one row per event")
    fitting, remainder = train_test_split(events, test_size=.4, random_state=seed, stratify=events.final_tier)
    validation, holdout = train_test_split(remainder, test_size=.5, random_state=seed, stratify=remainder.final_tier)
    groups = [set(group.event_id) for group in [fitting, validation, holdout]]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    return fitting, validation, holdout


def make_model(feature_names=FEATURES, baseline=False, balanced=False):
    numeric = [name for name in feature_names if name != "c_object_type"]
    transformers = [("numerical", SimpleImputer(strategy="median", keep_empty_features=True), numeric)]
    if "c_object_type" in feature_names:
        transformers.append(("object", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["c_object_type"]))
    prep = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    estimator = DummyClassifier(strategy="prior") if baseline else HistGradientBoostingClassifier(
        learning_rate=.07, max_iter=160, max_leaf_nodes=15, l2_regularization=2.,
        early_stopping=False, random_state=SEED, class_weight="balanced" if balanced else None,
    )
    return Pipeline([("preprocess", prep), ("estimator", estimator)])


def class_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=float)
    classes = list(model.classes_)
    if set(classes) != set(CLASS_LABELS):
        raise ValueError("Forecast model must contain exactly the three organizer tier classes")
    aligned = raw[:, [classes.index(label) for label in CLASS_LABELS]]
    if aligned.shape != (len(features), 3) or not np.isfinite(aligned).all() or (aligned < 0).any() or not np.allclose(aligned.sum(axis=1), 1.):
        raise ValueError("Forecast model returned invalid class probabilities")
    return aligned


def predict_messages(frame: pd.DataFrame, model=None, model_path=ARTIFACTS / "risk_model.joblib") -> pd.DataFrame:
    """Preserve every row; forecast only eligible early messages.

    Rows under two days or with unknown TCA receive Incomplete and NaN
    probabilities. Replay scores eligible snapshots separately; headline
    evaluation uses only the latest eligible snapshot for each event.
    """
    if model is None:
        model = joblib.load(model_path)
    missing = [key for key in KEYS if key not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    result = frame.loc[:, KEYS].copy()
    result.insert(0, "source_row_id", frame.source_row_id.to_numpy() if "source_row_id" in frame else np.arange(len(frame)))
    x = clean_features(frame)
    for name in NUMERIC_FEATURES:
        result[name] = x[name].to_numpy()
    result["c_object_type"] = x.c_object_type.to_numpy()
    times = x.time_to_tca.to_numpy(dtype=float)
    eligible = np.isfinite(times) & (times >= 2)
    probabilities = np.full((len(frame), 3), np.nan)
    tiers = np.full(len(frame), "Incomplete", dtype=object)
    if eligible.any():
        probabilities[eligible] = class_probabilities(model, x.loc[eligible])
        tiers[eligible] = np.asarray(CLASS_LABELS)[probabilities[eligible].argmax(axis=1)]
    result["predicted_final_tier"] = tiers
    result["risk_tier"] = tiers
    for index, label in enumerate(CLASS_LABELS):
        result["prob_" + label.lower()] = probabilities[:, index]
    confidence = np.full(len(frame), np.nan)
    confidence[eligible] = probabilities[eligible].max(axis=1)
    result["prediction_confidence"] = confidence
    result["forecast_status"] = np.where(eligible, "Forecast available (>=2 days)", np.where(np.isfinite(times), "Outside forecast window (<2 days)", "Missing or invalid time_to_tca"))
    quality, explanation = [], []
    missing_fields = x.isna()
    for pos in range(len(frame)):
        names = missing_fields.columns[missing_fields.iloc[pos]].tolist()
        quality.append("Imputed model inputs: " + ", ".join(names) if eligible[pos] and names else "Complete" if eligible[pos] else str(result.forecast_status.iloc[pos]))
        explanation.append(f"Classifier forecasts final {tiers[pos]} tier from this >=2-day CDM; class probability {confidence[pos]:.1%} (uncalibrated, not Pc)." if eligible[pos] else "No forecast: only CDMs at least 2 days before TCA are eligible.")
    result["data_quality"] = quality
    result["triggered_rules"] = explanation
    result["time_to_tca_days"] = result.time_to_tca
    result["miss_distance_m"] = result.miss_distance
    return result


def metrics(frame: pd.DataFrame, predictions, probabilities=None) -> dict:
    """Score final tiers; probabilities are aligned Critical/Elevated/Safe."""
    truth = frame.final_tier.astype(str).to_numpy()
    predicted = np.asarray(predictions)
    if len(predicted) != len(truth) or not np.isin(predicted, CLASS_LABELS).all():
        raise ValueError("Evaluation requires one valid predicted tier per labeled event")
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=CLASS_LABELS, zero_division=0)
    positive, proposed = truth == "Critical", predicted == "Critical"
    result = {
        "n_rows": int(len(frame)), "n_events": int(frame.event_id.nunique()),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "critical_precision": float(precision[0]), "critical_recall": float(recall[0]), "critical_f1": float(f1[0]),
        "critical_true": int(positive.sum()), "critical_tp": int((positive & proposed).sum()),
        "critical_fn": int((positive & ~proposed).sum()), "critical_fp": int((~positive & proposed).sum()),
        "confusion_matrix": {"labels": CLASS_LABELS, "values": confusion_matrix(truth, predicted, labels=CLASS_LABELS).tolist()},
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(CLASS_LABELS)},
        "log_loss": None, "multiclass_brier": None,
    }
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=float)
        truth_one_hot = (truth[:, None] == np.asarray(CLASS_LABELS)[None, :]).astype(float)
        result["log_loss"] = float(log_loss(truth, probs, labels=CLASS_LABELS))
        result["multiclass_brier"] = float(np.mean(np.sum((probs - truth_one_hot) ** 2, axis=1)))
    return result


def score_model(frame, model):
    probs = class_probabilities(model, clean_features(frame))
    predictions = np.asarray(CLASS_LABELS)[probs.argmax(axis=1)]
    return metrics(frame, predictions, probs)


def distance_only_metrics(frame: pd.DataFrame) -> dict:
    distance = pd.to_numeric(frame.miss_distance, errors="coerce").to_numpy()
    if not np.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("Distance baseline requires valid nonnegative latest-eligible distances")
    prediction = np.select([distance < 1000, distance <= 5000], ["Critical", "Elevated"], "Safe")
    result = metrics(frame, prediction)
    result["description"] = "Latest eligible CDM distance alone: <1000 m Critical, <=5000 m Elevated, otherwise Safe; predicts final tier, no Pc."
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def train_pipeline(train_path=DATA_DIR / "train_data.csv", test_path=DATA_DIR / "test_data.csv", output_dir=ARTIFACTS, labels_path=LABELS_PATH):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, labels = load_messages(Path(train_path), training=True), load_event_labels(Path(labels_path))
    events = prepare_training_events(train, labels)
    fitting, validation, holdout = split_training_events(events)
    print(f"Stratified event split: {len(fitting)} train / {len(validation)} validation / {len(holdout)} holdout", flush=True)
    configurations = {"unweighted": {}, "balanced": {"balanced": True}, "majority": {"baseline": True}}
    validation_scores = {}
    for candidate, options in configurations.items():
        print(f"Fitting {candidate} final-tier classifier", flush=True)
        model = make_model(**options)
        model.fit(clean_features(fitting), fitting.final_tier)
        validation_scores[candidate] = score_model(validation, model)
    # Fixed selection rule; holdout labels never choose a configuration.
    selected = max(configurations, key=lambda candidate: (validation_scores[candidate]["macro_f1"], validation_scores[candidate]["critical_recall"]))
    model = make_model(**configurations[selected])
    refit = pd.concat([fitting, validation], ignore_index=True)
    model.fit(clean_features(refit), refit.final_tier)
    majority = make_model(baseline=True)
    majority.fit(clean_features(refit), refit.final_tier)
    holdout_scores = score_model(holdout, model)
    holdout_early = train.loc[train.event_id.isin(holdout.event_id) & train.time_to_tca.ge(2)].copy()
    holdout_export = predict_messages(holdout_early, model=model)
    holdout_export["true_final_tier"] = holdout_export.event_id.map(labels.set_index("event_id").final_tier)
    holdout_export.to_csv(output_dir / "holdout_predictions.csv", index=False)
    joblib.dump(model, output_dir / "risk_model.joblib", compress=3)
    test = load_messages(Path(test_path))
    predicted_test = predict_messages(test, model=model)
    predicted_test.to_csv(output_dir / "test_predictions.csv", index=False)
    assert len(predicted_test) == len(test) and predicted_test.source_row_id.is_unique
    split_ids = {name: sorted(group.event_id.astype(str).tolist()) for name, group in [("train", fitting), ("validation", validation), ("holdout", holdout)]}
    (output_dir / "split_events.json").write_text(json.dumps(split_ids, indent=2), encoding="utf-8")
    name = {"unweighted": "Histogram gradient-boosted final-tier classifier", "balanced": "Class-balanced histogram gradient-boosted final-tier classifier", "majority": "Majority-class final-tier baseline"}[selected]
    report = {
        "schema_version": 2,
        "model": {"name": name, "target": "Organizer final_tier for each event, using only CDMs with time_to_tca >= 2 days", "features": FEATURES,
                  "selection": "Highest validation event macro-F1; tie-break by Critical recall. Candidates and settings fixed before holdout scoring.",
                  "probability_note": PROBABILITY_NOTE, "selected_candidate": selected,
                  "evaluation_unit": "One latest eligible >=2-day CDM per event; each event has equal weight.",
                  "hyperparameters": {"learning_rate": .07, "max_iter": 160, "max_leaf_nodes": 15, "l2_regularization": 2., "random_state": SEED}},
        "data": {"train_rows": len(train), "train_events": int(train.event_id.nunique()), "eligible_train_rows": int(train.time_to_tca.ge(2).sum()),
                 "eligible_train_events": len(events), "excluded_no_early_events": int(train.event_id.nunique() - len(events)),
                 "test_rows": len(test), "test_events": int(test.event_id.nunique()), "test_all_at_least_two_days": bool(test.time_to_tca.ge(2).all()),
                 "final_tier_counts": labels.final_tier.value_counts().to_dict(), "eligible_final_tier_counts": events.final_tier.value_counts().to_dict(),
                 "final_recorded_message_still_at_least_two_days_events": int(labels.final_time_to_tca_days.ge(2).sum()),
                 "negative_final_recorded_time_events": int(labels.final_time_to_tca_days.lt(0).sum())},
        "split": {"train_events": len(fitting), "validation_events": len(validation), "holdout_events": len(holdout), "seed": SEED,
                  "method": "Stratified by final_tier on unique train event IDs: 60/20/20, random_state=42 in both stages; IDs are file-local."},
        "validation": {**validation_scores, "model": validation_scores[selected], "selected_candidate": selected, "distance_only": distance_only_metrics(validation)},
        "holdout": {"latest_event": holdout_scores, "majority_latest_event": score_model(holdout, majority), "distance_only_latest_event": distance_only_metrics(holdout)},
        "lineage": {"train_sha256": file_sha256(train_path), "test_sha256": file_sha256(test_path), "labels_sha256": file_sha256(labels_path),
                    "split_events_sha256": file_sha256(output_dir / "split_events.json"), "pipeline_sha256": file_sha256(Path(__file__)),
                    "labels_source": "Organizer challenge1_event_labels.xlsx / train_event_labels", "join_scope": "train_data only, one-to-one event label join; test IDs never joined to train labels"},
        "notes": [PROBABILITY_NOTE,
                  "No CSV risk or final_* field is a model input. max_risk_estimate is an organizer-recommended early-CDM feature.",
                  "Train and evaluation features are the latest eligible CDM per event (smallest time_to_tca >= 2); source-row order breaks ties.",
                  "The selected configuration is refitted on training plus validation events before holdout evaluation.",
                  "Early replay scores each eligible CDM separately; pooled replay-row accuracy is not the headline evaluation.",
                  "Organizer final tier means the final recorded message, including records whose last CDM is still >=2 days or slightly after TCA; it is not an observed collision outcome.",
                  "The original dataset was examined during previous development; this updated split is reproducible, not a never-inspected external test.",
                  "All test rows are retained for demo predictions; their true final outcomes are unavailable.",
                  "Multiclass Brier is mean sum of squared errors across three classes (range 0 to 2); probabilities are not calibrated."]
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_manifest(output_dir)
    print(json.dumps({"model": name, "holdout": holdout_scores}, indent=2), flush=True)
    return report
