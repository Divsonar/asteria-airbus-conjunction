"""Local analyst review records, not an operational authorization system.

The SQLite ledger records only a small whitelist of evidence. The caller
chooses its path, so a deployment can keep the file in local private storage.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import numbers
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

ACTIONS = ("Accept", "Override", "Escalate")
OVERRIDE_TIERS = ("Critical", "Elevated", "Safe", "Incomplete")
NUMERIC_EVIDENCE = (
    "input_row_position", "time_to_tca_days", "miss_distance_m", "relative_speed", "relative_position_r", "relative_position_t",
    "relative_position_n", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP",
    "predicted_log10_pc", "estimated_pc",
)
TEXT_EVIDENCE = ("c_object_type", "risk_tier", "rule_ids", "triggered_rules", "data_quality", "review_reasons")
EXPORT_FIELDS = ("record_id", "recorded_at_utc", "dataset_id", "event_id", "source_row_id",
                 "action", "override_tier", "reason", "analyst", "model_sha256",
                 "evidence_sha256", "evidence_json")

_SCHEMA = """CREATE TABLE IF NOT EXISTS decisions (
  record_id TEXT PRIMARY KEY,
  recorded_at_utc TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  source_row_id TEXT NOT NULL,
  action TEXT NOT NULL CHECK(action IN ('Accept','Override','Escalate')),
  override_tier TEXT,
  reason TEXT NOT NULL,
  analyst TEXT NOT NULL,
  model_sha256 TEXT NOT NULL,
  evidence_sha256 TEXT NOT NULL,
  evidence_json TEXT NOT NULL
)"""


def _text(value, label: str, max_length: int) -> str:
    if value is None:
        raise ValueError(f"{label} is required.")
    result = str(value).strip()
    if not result or len(result) > max_length:
        raise ValueError(f"{label} must be 1–{max_length} characters.")
    return result


def _identifier(value, label: str, max_length: int) -> str:
    """Preserve exact nonempty identifiers, including spaces and leading zeros."""
    if value is None or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be nonempty.")
    if isinstance(value, numbers.Real) and not math.isfinite(value):
        raise ValueError(f"{label} must be nonempty.")
    identifier = str(value)
    if not identifier.strip() or len(identifier) > max_length or (not isinstance(value, str) and identifier == "<NA>"):
        raise ValueError(f"{label} must be 1–{max_length} characters and nonempty.")
    return identifier


def _row_id(value) -> str:
    return _identifier(value, "Source row ID", 256)


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create schema or atomically migrate old integer source IDs to TEXT.

    Existing integer IDs are preserved as their decimal representation; any
    leading zeros already discarded by an older integer schema cannot be
    recovered. The caller holds the write transaction for the whole migration.
    """
    old_columns = connection.execute("PRAGMA table_info(decisions)").fetchall()
    if not old_columns:
        connection.execute(_SCHEMA)
        return
    columns = {row[1]: row[2].upper() for row in old_columns}
    if set(EXPORT_FIELDS) != set(columns):
        raise ValueError("Existing decision log has an incompatible schema; no changes made.")
    if columns["source_row_id"] == "TEXT":
        return
    legacy = "decisions_migration_" + uuid.uuid4().hex
    connection.execute(f'ALTER TABLE decisions RENAME TO "{legacy}"')
    connection.execute(_SCHEMA)
    selection = ["CAST(source_row_id AS TEXT)" if field == "source_row_id" else field for field in EXPORT_FIELDS]
    connection.execute(f'INSERT INTO decisions ({",".join(EXPORT_FIELDS)}) SELECT {",".join(selection)} FROM "{legacy}"')
    old_count = connection.execute(f'SELECT COUNT(*) FROM "{legacy}"').fetchone()[0]
    new_count = connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    if old_count != new_count:
        raise ValueError("Decision log migration could not verify the copied records; no changes made.")
    connection.execute(f'DROP TABLE "{legacy}"')


def _model_fingerprint(path: str | Path) -> str:
    if path is None:
        raise ValueError("A model artifact is required to record its fingerprint.")
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence_snapshot(evidence: Mapping | None) -> dict:
    """Copy only expected, bounded fields; never persist extra telemetry."""
    if evidence is None:
        evidence = {}
    if not isinstance(evidence, Mapping):
        raise ValueError("Evidence must be a mapping of the displayed fields.")
    snapshot = {}
    for key in NUMERIC_EVIDENCE:
        try:
            value = float(evidence.get(key))
            snapshot[key] = value if math.isfinite(value) else None
        except (TypeError, ValueError, OverflowError):
            snapshot[key] = None
    for key in TEXT_EVIDENCE:
        value = evidence.get(key)
        if value is None:
            snapshot[key] = ""
        elif isinstance(value, (list, tuple)):
            snapshot[key] = ", ".join(str(item) for item in value)[:1000]
        else:
            snapshot[key] = str(value)[:1000]
    return snapshot


def record_decision(db_path: str | Path, *, dataset_id, event_id, source_row_id,
                    action, reason, analyst, model_path: str | Path,
                    override_tier=None, evidence: Mapping | None = None, expected_model_sha256=None) -> dict:
    """Append one local review record and return it.

    ``Accept`` records an analyst's view of the assessment only. No operation
    is performed or authorized by any action. ``Override`` requires a tier.
    """
    dataset_id = _identifier(dataset_id, "Dataset ID", 256)
    event_id = _identifier(event_id, "Event ID", 128)
    row_id = _row_id(source_row_id)
    action = _text(action, "Action", 32)
    if action not in ACTIONS:
        raise ValueError("Action must be Accept, Override, or Escalate.")
    if action == "Override":
        if override_tier not in OVERRIDE_TIERS:
            raise ValueError("Choose a valid override tier.")
    elif override_tier not in (None, ""):
        raise ValueError("An override tier can only be provided for Override.")
    reason = _text(reason, "Reason", 1000)
    if len(reason) < 8 or sum(ch.isalpha() for ch in reason) < 2:
        raise ValueError("Give a reason of at least 8 characters, including letters.")
    analyst = _text(analyst, "Analyst", 100)
    fingerprint = _model_fingerprint(model_path)
    if expected_model_sha256 is not None and fingerprint != str(expected_model_sha256):
        raise ValueError("The model artifact changed after bundle verification; refresh before recording the decision.")
    snapshot = _evidence_snapshot(evidence)
    evidence_json = json.dumps(snapshot, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    evidence_hash = hashlib.sha256(evidence_json.encode("utf-8")).hexdigest()
    record = {
        "record_id": uuid.uuid4().hex,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_id": dataset_id, "event_id": event_id, "source_row_id": row_id,
        "action": action, "override_tier": override_tier or None,
        "reason": reason, "analyst": analyst, "model_sha256": fingerprint,
        "evidence_sha256": evidence_hash, "evidence_json": evidence_json,
    }
    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target, timeout=15) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection)
        connection.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(record[field] for field in EXPORT_FIELDS),
        )
    return record


def _spreadsheet_safe(value) -> str:
    """Prevent spreadsheet applications from executing user-supplied formulas."""
    if value is None:
        return ""
    text = str(value)
    # Tabs, CR and LF at the beginning can themselves be interpreted as
    # spreadsheet control characters; whitespace may also hide a formula prefix.
    if text.startswith(("\t", "\r", "\n")) or text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def list_decisions(db_path: str | Path, dataset_id=None, event_id=None, source_row_id=None) -> list[dict]:
    """Return local review records newest first without creating a database."""
    target = Path(db_path)
    if not target.is_file():
        return []
    query = "SELECT " + ",".join(EXPORT_FIELDS) + " FROM decisions"
    conditions, values = [], []
    for key, value in (("dataset_id", dataset_id), ("event_id", event_id), ("source_row_id", source_row_id)):
        if value is not None:
            conditions.append("CAST(source_row_id AS TEXT) = ?" if key == "source_row_id" else f"{key} = ?")
            values.append(_row_id(value) if key == "source_row_id" else
                          _identifier(value, "Dataset ID" if key == "dataset_id" else "Event ID", 256 if key == "dataset_id" else 128))
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY recorded_at_utc DESC, rowid DESC"
    with sqlite3.connect(target, timeout=15) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(query, values).fetchall()]


def export_decisions_csv(db_path: str | Path, dataset_id=None) -> bytes:
    """Export local records as UTF-8 CSV, optionally filtered by dataset ID."""
    target = Path(db_path)
    rows = []
    if target.is_file():
        with sqlite3.connect(target, timeout=15) as connection:
            connection.row_factory = sqlite3.Row
            query = "SELECT " + ",".join(EXPORT_FIELDS) + " FROM decisions"
            if dataset_id is None:
                rows = connection.execute(query + " ORDER BY recorded_at_utc, rowid").fetchall()
            else:
                rows = connection.execute(query + " WHERE dataset_id = ? ORDER BY recorded_at_utc, rowid", (str(dataset_id),)).fetchall()
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(EXPORT_FIELDS)
    for row in rows:
        writer.writerow([_spreadsheet_safe(row[field]) for field in EXPORT_FIELDS])
    return output.getvalue().encode("utf-8")
