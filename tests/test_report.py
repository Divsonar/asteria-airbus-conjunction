"""Check that assessment exports preserve evidence and avoid false claims."""
from io import BytesIO
import math

from pypdf import PdfReader
from src.report import create_event_assessment_pdf


def test_report_contains_selected_evidence_and_limits():
    row = {"event_id": 123, "source_row_id": 99, "input_row_position": 23, "risk_tier": "Critical",
           "prob_critical": .75, "prob_elevated": .20, "prob_safe": .05,
           "forecast_status": "Eligible early forecast",
           "miss_distance": 742, "time_to_tca": 2.1, "c_object_type": "DEBRIS",
           "triggered_rules": "Highest final-tier probability", "rule_ids": "F1",
           "review_required": True, "review_codes": "R-SIGMA",
           "review_reasons": "Unusually large radial sigma"}
    output = create_event_assessment_pdf(row, "Test set", {"name": "Example surrogate", "sha256": "abc123"})
    assert output.startswith(b"%PDF-")
    text = "\n".join(page.extract_text() for page in PdfReader(BytesIO(output)).pages)
    for part in ("123 / 99", "Zero-based input position", "23", "742.0 m", "2.100 days", "0.750 / 0.200 / 0.050", "Highest final-tier probability", "abc123",
                 "Required - R-SIGMA", "Unusually large radial sigma", "Not available in the permitted fields", "does not authorize"):
        assert part in text


def test_report_handles_missing_and_untrusted_text():
    output = create_event_assessment_pdf({"event_id": "<tag>&", "miss_distance": math.nan}, "Upload & review")
    text = "\n".join(page.extract_text() for page in PdfReader(BytesIO(output)).pages)
    assert "<tag>&" in text
    assert "Not available" in text
    assert "Upload & review" in text


def test_report_preserves_text_identifier_whitespace_and_unicode():
    output = create_event_assessment_pdf({"event_id": "  evt-\u4e2d  ", "source_row_id": "001"}, "Upload")
    text = "\n".join(page.extract_text() for page in PdfReader(BytesIO(output)).pages)
    assert '"\\u0020\\u0020evt-\\u4e2d\\u0020\\u0020"' in text
    assert '"001"' in text


def test_report_formats_extreme_measurements_compactly():
    output = create_event_assessment_pdf({"event_id": 1, "time_to_tca": 1e300, "miss_distance": 1e300}, "Upload")
    reader = PdfReader(BytesIO(output))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "1.00e+300" in text
    assert len(reader.pages) == 1
