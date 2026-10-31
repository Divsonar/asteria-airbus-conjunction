"""Non-destructive data-quality checks for conjunction telemetry.

Review flags describe input quality, not model uncertainty or a safety
determination. Every input row stays in order and no value is replaced.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "event_id", "time_to_tca", "miss_distance", "relative_speed",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "c_object_type", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP",
)
NUMERIC_COLUMNS = tuple(c for c in REQUIRED_COLUMNS if c not in ("event_id", "c_object_type"))
OPTIONAL_MEASUREMENTS = frozenset(("F10", "AP"))
KNOWN_OBJECT_TYPES = frozenset(("DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN", "TBA"))
# This is a screening threshold, not a confirmed missing-value sentinel. Values
# are flagged and preserved; do not substitute or infer a covariance model.
LARGE_SIGMA_REVIEW_METRES = 60_000_000.0
# A display-range screening heuristic for this short-horizon demo, not a claim
# that longer physical prediction windows are impossible. Values are preserved.
LONG_TCA_REVIEW_DAYS = 365.0


@dataclass(frozen=True)
class ValidationResult:
    """flags aligns positionally with the input; summary contains plain data."""

    flags: pd.DataFrame
    summary: dict


def read_telemetry_csv(file_or_buffer) -> pd.DataFrame:
    """Read only permitted telemetry, preserving literal uploaded IDs.

    Default NA conversion still applies to measurement columns. IDs named
    ``NA`` or ``NULL`` are legitimate text; blank IDs remain empty and are
    surfaced by validation rather than silently converted or renumbered.
    Extra dtype/NA mapping entries are harmless when a required column is absent.
    """
    null_tokens = ["", "NA", "N/A", "NULL", "null", "NaN", "nan"]
    na_by_column = {name: null_tokens for name in (*NUMERIC_COLUMNS, "c_object_type")}
    permitted = frozenset((*REQUIRED_COLUMNS, "source_row_id"))
    frame = pd.read_csv(file_or_buffer, usecols=lambda name: name in permitted,
                        dtype={"event_id": "string", "source_row_id": "string"},
                        keep_default_na=False, na_values=na_by_column)
    # pandas returns a 0x0 frame if usecols matches nothing, even for a file
    # containing rows. Raise a schema error rather than claiming no data.
    if len(frame.columns) == 0:
        raise ValueError("Missing required columns: " + ", ".join(REQUIRED_COLUMNS))
    return frame


def spreadsheet_safe_csv(frame: pd.DataFrame) -> bytes:
    """Export CSV with dangerous user-supplied string cells neutralized.

    The original DataFrame is never changed. Number-valued cells stay numeric;
    string cells that a spreadsheet may interpret as formulas receive a
    leading apostrophe. This changes those exported string values and is not
    appropriate for machine scoring where IDs must match exactly.
    """
    safe = frame.copy(deep=True)
    def neutralize(value):
        if not isinstance(value, str):
            return value
        if value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value
    for col in safe.select_dtypes(include=["object", "string", "category"]).columns:
        safe[col] = safe[col].map(neutralize)
    return safe.to_csv(index=False).encode("utf-8")


def validate_telemetry(frame: pd.DataFrame) -> ValidationResult:
    """Inspect telemetry without changing, dropping, or reordering any rows.

    Flags can be attached to prediction rows *by position*, not by event ID.
    All input columns are required by the current model, although F10 and AP
    cells may be blank and are imputed without an analyst-review flag.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Telemetry must be a pandas DataFrame")
    n = len(frame)
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    # Use a positional index throughout, because callers can have duplicate or
    # non-integer DataFrame index labels.
    data = frame.reset_index(drop=True)
    false = np.zeros(n, dtype=bool)
    numeric: dict[str, pd.Series] = {}
    missing_counts: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}
    measurements_missing = false.copy()
    invalid_numeric = false.copy()

    for col in NUMERIC_COLUMNS:
        raw = data[col] if col in data else pd.Series([None] * n, dtype=object)
        # Blank cells count as missing; non-empty non-numeric and infinite
        # values count as invalid, separately.
        blank = raw.isna() | raw.astype("string").str.strip().eq("").fillna(False)
        converted = pd.to_numeric(raw, errors="coerce").astype(float)
        finite = np.isfinite(converted.to_numpy())
        invalid = ~blank.to_numpy(dtype=bool) & ~finite
        numeric[col] = converted.replace([np.inf, -np.inf], np.nan)
        missing_counts[col] = int(blank.sum())
        invalid_counts[col] = int(invalid.sum())
        invalid_numeric |= invalid
        if col not in OPTIONAL_MEASUREMENTS:
            measurements_missing |= blank.to_numpy(dtype=bool)

    invalid_domain = false.copy()
    for col in ("miss_distance", "relative_speed", "c_sigma_r", "t_sigma_r"):
        invalid_domain |= (numeric[col] < 0).fillna(False).to_numpy(dtype=bool)
    invalid_domain |= (numeric["max_risk_estimate"] > 0).fillna(False).to_numpy(dtype=bool)

    after_tca = (numeric["time_to_tca"] < 0).fillna(False).to_numpy(dtype=bool)
    long_tca = (numeric["time_to_tca"] > LONG_TCA_REVIEW_DAYS).fillna(False).to_numpy(dtype=bool)
    large_sigma = ((numeric["t_sigma_r"] >= LARGE_SIGMA_REVIEW_METRES) |
                   (numeric["c_sigma_r"] >= LARGE_SIGMA_REVIEW_METRES)).fillna(False).to_numpy(dtype=bool)
    zero_sigma = ((numeric["t_sigma_r"] == 0) | (numeric["c_sigma_r"] == 0)).fillna(False).to_numpy(dtype=bool)

    position = np.column_stack([numeric[c].to_numpy() for c in
                                ("relative_position_r", "relative_position_t", "relative_position_n")])
    dist = numeric["miss_distance"].to_numpy()
    norm = np.linalg.norm(position, axis=1)
    # A one-metre minimum tolerance avoids flagging expected coordinate rounding
    # for very small miss distances. Both values are the reported TCA geometry.
    tolerance = np.maximum(1.0, np.abs(dist) * 0.01)
    mismatch = np.isfinite(norm) & np.isfinite(dist) & (dist >= 0) & (np.abs(norm - dist) > tolerance)

    obj = data["c_object_type"] if "c_object_type" in data else pd.Series([None] * n, dtype=object)
    obj_raw = obj.astype("string")
    obj_normalized = obj_raw.str.strip().str.upper()
    missing_object = obj_raw.isna() | obj_normalized.eq("").fillna(False)
    # Inference uses the original categorical strings. Lowercase or extra
    # whitespace would otherwise appear valid here but encode as unknown.
    unseen_object = (~obj_raw.isin(KNOWN_OBJECT_TYPES)).fillna(False)
    unknown_object = (~missing_object & unseen_object).fillna(False).to_numpy(dtype=bool)
    measurements_missing |= missing_object.to_numpy(dtype=bool)
    missing_counts["c_object_type"] = int(missing_object.sum())

    ids = data["event_id"] if "event_id" in data else pd.Series([None] * n, dtype=object)
    event_ids = ids.astype("string")
    missing_id = (ids.isna() | event_ids.str.strip().eq("").fillna(False)).to_numpy(dtype=bool)
    missing_counts["event_id"] = int(missing_id.sum())
    invalid_event_type = ids.map(lambda value: isinstance(value, (bool, np.bool_)) or
                                 (isinstance(value, numbers.Real) and not math.isfinite(value))).to_numpy(dtype=bool)
    invalid_event_id = missing_id | event_ids.str.len().gt(128).fillna(False).to_numpy(dtype=bool) | invalid_event_type
    if "source_row_id" in data:
        source_ids = data["source_row_id"]
        row_ids = source_ids.astype("string")
        missing_row_id = row_ids.isna() | row_ids.str.strip().eq("").fillna(False)
        too_long = row_ids.str.len().gt(256).fillna(False)
        invalid_type = source_ids.map(lambda value: isinstance(value, (bool, np.bool_)) or
                                      (isinstance(value, numbers.Number) and not math.isfinite(value)))
        duplicates = row_ids.duplicated(keep=False)
        # Pandas may infer a string dtype for an empty map result; normalize
        # each mask explicitly so empty uploaded files validate correctly.
        invalid_row_id = (missing_row_id.to_numpy(dtype=bool) | too_long.to_numpy(dtype=bool) |
                          invalid_type.to_numpy(dtype=bool) | duplicates.to_numpy(dtype=bool))
    else:
        invalid_row_id = false.copy()

    checks = [
        ("R-ID", invalid_event_id, "Event identifier must be nonempty, finite and at most 128 characters for analyst review records."),
        ("R-ROW-ID", invalid_row_id, "Source-row identifier must be present, unique and at most 256 characters; use positional order to distinguish messages."),
        ("R-MISSING", measurements_missing, "One or more core telemetry measurements are missing."),
        ("R-NUMERIC", invalid_numeric, "A numeric field contains non-numeric or non-finite data."),
        ("R-RANGE", invalid_domain, "A measurement is outside its expected numerical range."),
        ("R-AFTER-TCA", after_tca, "The message is recorded after its closest-approach time."),
        ("R-TCA", long_tca, "Time to closest approach exceeds the 365-day display range; inspect the units and input. This is a demo screening heuristic."),
        ("R-RTN", mismatch, "RTN vector magnitude differs from the miss distance by over 1% and 1 metre."),
        ("R-OBJECT", unknown_object, "Object type does not match the model's known dataset categories exactly."),
        ("R-SIGMA", large_sigma, "A radial sigma is unusually large; inspect it. This is a screening heuristic, not a confirmed placeholder."),
        ("R-SIGMA-ZERO", zero_sigma, "A radial sigma is zero; inspect the uncertainty input."),
    ]
    codes: list[list[str]] = [[] for _ in range(n)]
    explanations: list[list[str]] = [[] for _ in range(n)]
    counts = {}
    for code, mask, reason in checks:
        counts[code] = int(mask.sum())
        for row in np.flatnonzero(mask):
            codes[row].append(code)
            explanations[row].append(reason)
    source = data["source_row_id"].to_numpy() if "source_row_id" in data else np.arange(n, dtype=np.int64)
    flagged = np.fromiter((bool(x) for x in codes), dtype=bool, count=n)
    flags = pd.DataFrame({
        "source_row_id": source,
        "review_required": flagged,
        "review_codes": ["; ".join(x) for x in codes],
        "review_reasons": [" ".join(x) for x in explanations],
    }, index=frame.index)
    summary = {
        "rows": n,
        "review_rows": int(flagged.sum()),
        "missing_columns": missing_columns,
        "missing_counts": missing_counts,
        "invalid_numeric_counts": invalid_counts,
        "issue_counts": counts,
        "notes": ["A review flag is a data-quality warning, not a calibrated uncertainty estimate.",
                  "Large-sigma screening is a heuristic; values are not replaced.",
                  "Blank F10 and AP cells are reported and imputed by the model without a review flag."],
    }
    return ValidationResult(flags=flags, summary=summary)
