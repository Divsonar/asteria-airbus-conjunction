"""Compare a reconstructed prior model and the current model on the same holdout.

The holdout was already inspected during development. This script only reads
training labels, and writes its exploratory result and reconstructed model
under experiments/; it does not touch the production artifact bundle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.legacy_pipeline import ARTIFACTS, DATA_DIR, clean_features, latest_message, load_messages, make_model, metrics
from src.provenance import verify_manifest


def run() -> dict:
    bundle = verify_manifest(ARTIFACTS)
    if bundle["status"] != "verified":
        raise ValueError("Production artifact hashes do not match their recorded bundle; comparison stopped.")
    split = json.loads((ARTIFACTS / "split_events.json").read_text(encoding="utf-8"))
    rows = load_messages(DATA_DIR / "train_data.csv", training=True)
    rows = rows[np.isfinite(rows.risk) & (rows.risk <= 0)].copy()
    train = rows[rows.event_id.isin(split["train"])]
    validation = rows[rows.event_id.isin(split["validation"])]
    holdout = rows[rows.event_id.isin(split["holdout"])]
    assert set(train.event_id).isdisjoint(validation.event_id)
    assert set(train.event_id).isdisjoint(holdout.event_id)
    assert set(validation.event_id).isdisjoint(holdout.event_id)
    fit = pd.concat([train, validation], ignore_index=True)
    latest = latest_message(holdout)
    print("Reconstructing prior unweighted model on train+validation events...", file=sys.stderr, flush=True)
    reference = make_model()
    reference.fit(clean_features(fit), fit.risk)
    output_dir = ROOT / "experiments"
    output_dir.mkdir(exist_ok=True)
    reference_path = output_dir / "unweighted_reference.joblib"
    joblib.dump(reference, reference_path, compress=3)
    current = joblib.load(ARTIFACTS / "risk_model.joblib")
    result = {
        "reference": "Reconstructed prior unweighted histogram gradient-boosting configuration, refit on the original train and validation event groups.",
        "current_model_sha256": bundle["model_sha256"],
        "limits": ["This holdout has been inspected previously; this is an exploratory same-sample comparison, not a fresh blind evaluation.",
                   "The current configuration was selected during development; performance on independent data is unknown.",
                   "Both models estimate same-message reported log-risk. The supplied test risk labels were not read."],
        "holdout_rows": {"reference": metrics(holdout, reference.predict(clean_features(holdout))),
                         "current": metrics(holdout, current.predict(clean_features(holdout)))},
        "holdout_latest_event": {"reference": metrics(latest, reference.predict(clean_features(latest))),
                                 "current": metrics(latest, current.predict(clean_features(latest)))},
    }
    (output_dir / "model_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
