"""Integrity records for a generated model, evaluation and prediction bundle.

An integrity manifest establishes only that files still match the recorded
hashes. It does not itself prove how any file was generated; verify prediction
reproduction before initially recording an already-existing bundle.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


ARTIFACT_NAMES = ("risk_model.joblib", "metrics.json", "test_predictions.csv", "holdout_predictions.csv")
MANIFEST_NAME = "artifact_manifest.json"


def _sha256_stable(path: Path) -> str:
    """Hash a file and reject it if it changes while being read."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(f"{path.name} changed during hashing")
    return digest.hexdigest()


def write_manifest(artifacts_dir: str | Path) -> dict:
    """Record stable current hashes; call only after an entire bundle is ready.

    The caller is responsible for establishing lineage. A new training run
    should call this only after it has written its model and all three outputs.
    """
    directory = Path(artifacts_dir)
    before = {name: (directory / name).stat() for name in ARTIFACT_NAMES}
    files = {name: _sha256_stable(directory / name) for name in ARTIFACT_NAMES}
    # Detect changes to an earlier file while a later file was being hashed.
    for name, prior in before.items():
        after = (directory / name).stat()
        if (prior.st_size, prior.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OSError(f"{name} changed while creating the artifact manifest")
    manifest = {
        "schema_version": 1,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "files_sha256": files,
        "scope": "Checksums identify file contents; they do not independently prove model or data lineage.",
    }
    directory.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".manifest-", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, directory / MANIFEST_NAME)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return manifest


def verify_manifest(artifacts_dir: str | Path) -> dict:
    """Check a bundle against the recorded snapshot, failing closed.

    Status 'verified' means **matched checksums**, not certified lineage.
    'unknown' means no valid manifest, 'missing' means a required artifact is
    absent, and 'mismatch' means a hash differs or a file changes during read.
    """
    directory = Path(artifacts_dir)
    path = directory / MANIFEST_NAME
    expected = None
    def snapshot(file_path: Path):
        try:
            item = file_path.stat()
            return (item.st_size, item.st_mtime_ns, item.st_ino)
        except OSError:
            return None

    manifest_before = snapshot(path)
    before = {name: snapshot(directory / name) for name in ARTIFACT_NAMES}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        candidate = manifest.get("files_sha256")
        if (manifest.get("schema_version") == 1 and isinstance(candidate, dict)
            and set(candidate) == set(ARTIFACT_NAMES)
            and all(isinstance(candidate[name], str) and len(candidate[name]) == 64
                    and all(ch in "0123456789abcdef" for ch in candidate[name])
                    for name in ARTIFACT_NAMES)):
            expected = candidate
    except (OSError, ValueError, TypeError, AttributeError):
        pass

    file_status: dict[str, str] = {}
    model_sha256 = None
    for name in ARTIFACT_NAMES:
        artifact = directory / name
        if not artifact.is_file():
            file_status[name] = "missing"
            continue
        if expected is None and name != "risk_model.joblib":
            file_status[name] = "unrecorded"
            continue
        try:
            actual = _sha256_stable(artifact)
            if name == "risk_model.joblib":
                model_sha256 = actual
            file_status[name] = "unrecorded" if expected is None else (
                "matched" if actual == expected[name] else "mismatch")
        except OSError:
            file_status[name] = "mismatch"
    # A file hashed early may be replaced while another file is being read.
    # Fail closed if any file or the manifest changed across the whole check.
    for name in ARTIFACT_NAMES:
        if before[name] != snapshot(directory / name):
            file_status[name] = "mismatch"
    manifest_changed = manifest_before != snapshot(path)
    states = set(file_status.values())
    if "missing" in states:
        status, details = "missing", "One or more expected artifacts are missing. Model lineage is unverified."
    elif "mismatch" in states or manifest_changed:
        status, details = "mismatch", "Artifact contents changed or could not be read consistently. Model lineage is unverified."
    elif expected is None:
        status, details = "unknown", "No valid artifact manifest is available. Model lineage is unverified."
    else:
        status, details = "verified", "All four files match recorded checksums. Checksums alone do not establish lineage."
    return {"status": status, "file_status": file_status, "model_sha256": model_sha256, "details": details}
