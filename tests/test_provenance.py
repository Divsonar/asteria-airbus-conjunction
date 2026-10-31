"""Artifact checksums detect incomplete or mixed model-output bundles."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import src.provenance as provenance
from src.provenance import ARTIFACT_NAMES, MANIFEST_NAME, verify_manifest, write_manifest


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for name in ARTIFACT_NAMES:
            (self.directory / name).write_bytes((name + " contents").encode())

    def test_missing_manifest_is_unknown_and_matched_manifest_is_verified(self):
        self.assertEqual(verify_manifest(self.directory)["status"], "unknown")
        record = write_manifest(self.directory)
        result = verify_manifest(self.directory)
        self.assertEqual(result["status"], "verified")
        self.assertTrue(all(value == "matched" for value in result["file_status"].values()))
        self.assertEqual(result["model_sha256"], record["files_sha256"]["risk_model.joblib"])
        self.assertIn("do not establish lineage", result["details"])

    def test_changed_or_missing_artifact_is_never_verified(self):
        write_manifest(self.directory)
        (self.directory / "test_predictions.csv").write_text("changed", encoding="utf-8")
        result = verify_manifest(self.directory)
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["file_status"]["test_predictions.csv"], "mismatch")
        (self.directory / "risk_model.joblib").unlink()
        self.assertEqual(verify_manifest(self.directory)["status"], "missing")

    def test_invalid_or_partial_manifest_is_unknown(self):
        (self.directory / MANIFEST_NAME).write_text('{"schema_version": 1, "files_sha256": {}}', encoding="utf-8")
        self.assertEqual(verify_manifest(self.directory)["status"], "unknown")
        (self.directory / MANIFEST_NAME).write_text("{invalid", encoding="utf-8")
        self.assertEqual(verify_manifest(self.directory)["status"], "unknown")

    def test_manifest_write_requires_a_complete_bundle(self):
        (self.directory / "holdout_predictions.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            write_manifest(self.directory)
        self.assertFalse((self.directory / MANIFEST_NAME).exists())

    def test_detects_file_changed_after_its_individual_hash(self):
        write_manifest(self.directory)
        original = provenance._sha256_stable
        def replace_earlier(path):
            result = original(path)
            if path.name == "metrics.json":
                (self.directory / "risk_model.joblib").write_bytes(b"changed after model was hashed")
            return result
        with patch("src.provenance._sha256_stable", side_effect=replace_earlier):
            result = verify_manifest(self.directory)
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["file_status"]["risk_model.joblib"], "mismatch")


if __name__ == "__main__":
    unittest.main()
