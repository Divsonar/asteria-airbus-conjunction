"""Create a traceable, non-operational event-assessment PDF in memory."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from io import BytesIO
import json
import math
from typing import Any, Mapping

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _text(value: Any, default: str = "Not available") -> str:
    if value is None:
        return default
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return default
    except (TypeError, ValueError):
        pass
    rendered = str(value).strip()
    if not rendered or rendered.lower() in {"nan", "none", "nat"}:
        return default
    # The built-in PDF font supports Latin-1. Avoid missing-glyph boxes for
    # arbitrary user-supplied CSV fields while keeping PDF markup inert.
    return rendered.encode("latin-1", "replace").decode("latin-1")


def _number(value: Any, places: int = 2, scientific: bool = False) -> str:
    try:
        number = float(value)
        if not math.isfinite(number):
            return "Not available"
        if scientific or abs(number) >= 1e12 or (0 < abs(number) < 1e-4):
            return f"{number:.2e}"
        return f"{number:,.{places}f}"
    except (TypeError, ValueError, OverflowError):
        return "Not available"


def _with_unit(value: Any, unit: str, places: int = 2) -> str:
    rendered = _number(value, places=places)
    return rendered if rendered == "Not available" else f"{rendered} {unit}"


def _identifier(value: Any) -> str:
    """Represent text IDs losslessly, including whitespace and Unicode.

    JSON escaping uses ASCII glyphs available in the bundled PDF font and makes
    leading/trailing whitespace and special characters unambiguous.
    """
    if isinstance(value, str):
        # ReportLab collapses ordinary spaces in paragraphs. Encode them in
        # JSON so the displayed representation retains the exact identifier.
        return json.dumps(value, ensure_ascii=True).replace(" ", "\\u0020")
    return _text(value)


def create_event_assessment_pdf(
    row: Mapping[str, Any], dataset_label: str, model_info: Mapping[str, Any] | None = None
) -> bytes:
    """Return a PDF for one selected, already-scored historical message.

    Generation time is explicitly distinct from the unknown message creation
    time. Values are transcribed from the selected row; this function does not
    make a maneuver decision or call an LLM.
    """
    model_info = model_info or {}
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=(210 * mm, 297 * mm), leftMargin=19 * mm,
        rightMargin=19 * mm, topMargin=17 * mm, bottomMargin=18 * mm,
        title="Asteria event assessment", author="Asteria",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="AsteriaKicker", fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=colors.HexColor("#147d70"), spaceAfter=5))
    styles.add(ParagraphStyle(name="AsteriaTitle", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#102939"), spaceAfter=6))
    styles.add(ParagraphStyle(name="AsteriaSubtitle", fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#506875"), spaceAfter=13))
    styles.add(ParagraphStyle(name="AsteriaSection", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#102939"), spaceBefore=9, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name="AsteriaBody", fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#243e4a")))
    styles.add(ParagraphStyle(name="AsteriaLabel", fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=colors.HexColor("#526b77")))
    styles.add(ParagraphStyle(name="AsteriaValue", fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#152f3d")))
    styles.add(ParagraphStyle(name="AsteriaFoot", fontName="Helvetica", fontSize=7.5, leading=10, textColor=colors.HexColor("#556e77"), alignment=TA_RIGHT))

    def p(value: Any, style: str = "AsteriaValue") -> Paragraph:
        return Paragraph(escape(_text(value)), styles[style])

    def grid(items: list[tuple[str, Any]]) -> Table:
        cells = [[p(label, "AsteriaLabel"), p(value)] for label, value in items]
        table = Table(cells, colWidths=[48 * mm, 124 * mm], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f2f7f6"), colors.white]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return table

    event = _identifier(row.get("event_id"))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    review = row.get("review_reasons", row.get("data_quality"))
    if _text(review) == "Not available":
        review = row.get("data_quality", "Not available")
    review_codes = _text(row.get("review_codes"), default="")
    if review_codes:
        review = f"{review_codes}: {_text(review)}"
    if str(row.get("review_required", "")).strip().lower() in {"true", "1", "yes"}:
        review = f"Required - {_text(review)}"
    rule_ids = row.get("rule_ids", row.get("triggered_rule_ids"))
    rule_value = _text(rule_ids)
    if rule_value == "Not available":
        rule_value = "See trigger explanation"
    miss = row.get("miss_distance_m", row.get("miss_distance"))
    days = row.get("time_to_tca_days", row.get("time_to_tca"))
    story = [
        p("ASTERIA  /  CONJUNCTION INTELLIGENCE", "AsteriaKicker"),
        p("Event assessment", "AsteriaTitle"),
        p("Historical message review | Research prototype | No maneuver authorization", "AsteriaSubtitle"),
        grid([
            ("Dataset", dataset_label),
            ("Event / source row (text IDs JSON-quoted)", f"{event} / {_identifier(row.get('source_row_id'))}"),
            ("Zero-based input position", row.get("input_row_position")),
            ("Report generated", generated),
            ("Message timestamp", "Not available in the permitted fields"),
            ("Model", model_info.get("name", "Final-tier forecast classifier")),
            ("Artifact provenance", model_info.get("provenance_status", "Not verified")),
            ("Model version / SHA-256", model_info.get("artifact_sha256", model_info.get("sha256", model_info.get("version", "Not recorded")))),
        ]),
        p("Assessment and evidence", "AsteriaSection"),
        grid([
            ("Forecast final tier", row.get("predicted_final_tier", row.get("risk_tier"))),
            ("P(final Critical / Elevated / Safe)", " / ".join(_number(row.get(f"prob_{tier}"), places=3) for tier in ("critical", "elevated", "safe"))),
            ("Forecast status", row.get("forecast_status")),
            ("Miss distance", _with_unit(miss, "m", places=1)),
            ("Time to closest approach", _with_unit(days, "days from this message", places=3)),
            ("Object type", row.get("c_object_type")),
            ("Relative speed", _with_unit(row.get("relative_speed"), "m/s", places=1)),
            ("Relative position R / T / N", " / ".join(_number(row.get(f"relative_position_{axis}"), places=1) for axis in ("r", "t", "n")) + " m"),
            ("Radial sigma target / chaser", f"{_number(row.get('t_sigma_r'), places=1)} / {_number(row.get('c_sigma_r'), places=1)} m"),
            ("Max-risk input", _number(row.get("max_risk_estimate"), places=3)),
            ("Forecast method", row.get("triggered_rules")),
            ("Review / data quality", review),
            ("Model input quality", row.get("data_quality")),
        ]),
        p("Challenge rule reference", "AsteriaSection"),
        p("Updated Airbus problem statement, 31 October 2026, pages 1-2; official event-label workbook. "
          "Final labels: Critical if final Pc >= 1e-4 or final miss distance < 1,000 m; otherwise Elevated "
          "if final Pc >= 1e-6 or final miss distance <= 5,000 m; otherwise Safe. "
          "These rules define training labels, not a rule applied to this early message's distance. "
          "The risk=-30 placeholder means Pc approximately zero.", "AsteriaBody"),
        p("Interpretation and limitations", "AsteriaSection"),
        p("The classifier uses only telemetry at least 2 days before closest approach to forecast the "
          "organizer's final recorded tier. Class probabilities are uncalibrated model scores, not collision "
          "probability Pc. Evaluation uses one latest eligible message per held-out event; earlier replay "
          "messages are illustrative. Final recorded labels are not confirmed collision outcomes. "
          "Recorded miss distance is an early closest-approach estimate, not the final distance. "
          "A qualified analyst must evaluate the evidence and make all operational decisions. "
          "This document does not authorize or record a maneuver.", "AsteriaBody"),
    ]

    def footer(canvas, pdf):
        canvas.setStrokeColor(colors.HexColor("#c8dcda"))
        canvas.line(19 * mm, 13 * mm, 191 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#5b747c"))
        canvas.drawString(19 * mm, 9 * mm, "ASTERIA  |  RESEARCH DEMONSTRATION")
        canvas.drawRightString(191 * mm, 9 * mm, f"PAGE {pdf.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
