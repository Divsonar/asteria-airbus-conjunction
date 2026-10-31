"""Local decision records retain traceable, bounded evidence safely."""
import csv
import hashlib
import io
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
import pandas as pd

from src.decisions import _SCHEMA, export_decisions_csv, list_decisions, record_decision


@pytest.fixture
def context(tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"deterministic-model-for-test")
    return tmp_path / "records.sqlite3", model


def write(context, **changes):
    database, model = context
    payload = dict(dataset_id="test:sha256:123", event_id="482", source_row_id=5430,
                   action="Accept", reason="Reviewed telemetry quality", analyst="Analyst A",
                   model_path=model, evidence={"risk_tier": "Critical", "miss_distance_m": 601,
                                               "predicted_log10_pc": -11.51, "input_row_position": 14,
                                               "private_unexpected_value": "must not persist",
                                               "risk": -3.5})
    payload.update(changes)
    return record_decision(database, **payload)


def test_record_contains_fingerprint_and_only_whitelisted_snapshot(context):
    record = write(context)
    database, model = context
    assert record["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    evidence = json.loads(record["evidence_json"])
    assert evidence["risk_tier"] == "Critical"
    assert evidence["miss_distance_m"] == 601
    assert evidence["input_row_position"] == 14
    assert "risk" not in evidence
    assert "private_unexpected_value" not in evidence
    expected = hashlib.sha256(record["evidence_json"].encode("utf-8")).hexdigest()
    assert record["evidence_sha256"] == expected
    assert record["recorded_at_utc"].endswith("+00:00")
    exported = list(csv.DictReader(io.StringIO(export_decisions_csv(database).decode("utf-8"))))
    assert len(exported) == 1
    assert exported[0]["record_id"] == record["record_id"]
    assert "must not persist" not in export_decisions_csv(database).decode("utf-8")


def test_invalid_records_are_rejected_before_writing(context):
    database, _ = context
    for changes in ({"reason": "n/a"}, {"analyst": " "}, {"action": "Burn"},
                    {"source_row_id": " "}, {"source_row_id": float("nan")},
                    {"source_row_id": pd.NA}, {"source_row_id": True}, {"source_row_id": "x" * 257},
                    {"action": "Override"}, {"action": "Accept", "override_tier": "Safe"}):
        with pytest.raises(ValueError):
            write(context, **changes)
    assert not database.exists()
    record = write(context, action="Override", override_tier="Elevated")
    assert record["override_tier"] == "Elevated"


def test_export_neutralizes_formula_fields_and_filters_dataset(context):
    database, _ = context
    write(context, dataset_id="=SUM(1,1)", event_id="@event", analyst="+test analyst",
          reason="=execute something")
    write(context, dataset_id="other", action="Escalate")
    values = list(csv.DictReader(io.StringIO(export_decisions_csv(database, "=SUM(1,1)").decode("utf-8"))))
    assert len(values) == 1
    assert values[0]["dataset_id"] == "'=SUM(1,1)"
    assert values[0]["event_id"] == "'@event"
    assert values[0]["analyst"] == "'+test analyst"
    assert values[0]["reason"] == "'=execute something"


def test_concurrent_local_writes_are_not_lost(context):
    with ThreadPoolExecutor(max_workers=6) as workers:
        records = list(workers.map(lambda index: write(context, event_id=str(index)), range(20)))
    assert len({record["record_id"] for record in records}) == 20
    exported = list(csv.DictReader(io.StringIO(export_decisions_csv(context[0]).decode("utf-8"))))
    assert len(exported) == 20
    assert {row["event_id"] for row in exported} == {str(index) for index in range(20)}


def test_text_row_ids_preserve_leading_zeros_and_legacy_migration(context):
    database, _ = context
    # Simulate an older local ledger with integer source-row affinity.
    with sqlite3.connect(database) as conn:
        conn.execute(_SCHEMA.replace("source_row_id TEXT NOT NULL", "source_row_id INTEGER NOT NULL"))
        conn.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            "old-record", "2026-10-31T00:00:00+00:00", "old-dataset", "42", 7,
            "Accept", None, "Reviewed existing evidence", "Analyst", "a" * 64, "b" * 64, "{}"))
    created = write(context, source_row_id="0007")
    assert created["source_row_id"] == "0007"
    write(context, source_row_id="message-A-002")
    rows = list(csv.DictReader(io.StringIO(export_decisions_csv(database).decode("utf-8"))))
    assert {row["source_row_id"] for row in rows} == {"7", "0007", "message-A-002"}
    with sqlite3.connect(database) as conn:
        info = conn.execute("PRAGMA table_info(decisions)").fetchall()
        assert next(x[2] for x in info if x[1] == "source_row_id").upper() == "TEXT"


def test_filtered_history_does_not_create_or_match_wrong_identifiers(context):
    database, _ = context
    assert list_decisions(database) == []
    assert not database.exists()
    write(context, dataset_id="source A", event_id=" event-1 ", source_row_id="0007")
    write(context, dataset_id="source A", event_id="event-1", source_row_id="7")
    results = list_decisions(database, dataset_id="source A", event_id=" event-1 ", source_row_id="0007")
    assert len(results) == 1
    assert results[0]["event_id"] == " event-1 "
    assert results[0]["source_row_id"] == "0007"
    assert list_decisions(database, dataset_id="source A", event_id=" event-1 ", source_row_id="7") == []
    write(context, dataset_id="source A", event_id="NA", source_row_id="<NA>")
    assert len(list_decisions(database, dataset_id="source A", event_id="NA", source_row_id="<NA>")) == 1


def test_changed_model_hash_prevents_record_creation(context):
    database, model = context
    with pytest.raises(ValueError, match="model artifact changed"):
        write(context, expected_model_sha256="0" * 64)
    assert not database.exists()
    correct = hashlib.sha256(model.read_bytes()).hexdigest()
    assert write(context, expected_model_sha256=correct)["model_sha256"] == correct
