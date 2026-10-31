"""Asteria: an interactive, offline conjunction triage dashboard."""

from __future__ import annotations

import html
import json
import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.provenance import verify_manifest
from src.validation import read_telemetry_csv, spreadsheet_safe_csv, validate_telemetry


ROOT = Path(__file__).resolve().parent
APP_VERSION = "2026-10-31-final-tier"
ARTIFACTS = ROOT / "artifacts"
COLORS = {"Critical": "#ff6f73", "Elevated": "#f7bd68", "Safe": "#69e5c7", "Incomplete": "#9baabc"}
TEXT = "#deedf0"
MUTED = "#8196a4"
GRID = "rgba(135,169,186,.13)"

# Local component owns the first-visit preference and highlights real UI elements.
first_visit_component = components.declare_component(
    "asteria_first_visit", path=str(ROOT / "components" / "first_visit")
)

st.set_page_config(page_title="Asteria | Conjunction Intelligence", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {color-scheme:dark;}
html, body, [class*="css"], .stApp {font-family:'Segoe UI',sans-serif;}
.stApp {background:radial-gradient(ellipse at 77% -8%,#102b36 0%,transparent 38%),#050b14;color:#e7f0f2;}
.block-container {padding:2.2rem 3.2rem 4rem;max-width:1800px;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#0b1621,#08111b);border-right:1px solid #1c303b;}
section[data-testid="stSidebar"] .block-container {padding-top:2rem;}
header[data-testid="stHeader"] {background:transparent;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em;}
h1 {font-size:2.75rem!important;line-height:1.08!important;font-weight:600!important;}
h2 {font-size:1.25rem!important;font-weight:600!important;}
div[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#203440!important;background:rgba(12,25,37,.88)!important;border-radius:14px!important;}
.brand {display:flex;align-items:center;gap:11px;margin-bottom:1.5rem;}
.brand-mark {width:34px;height:34px;display:grid;place-items:center;border:1px solid #59dfcd;color:#72f5db;border-radius:50%;font-size:20px;box-shadow:0 0 20px #3bc8b53b;}
.brand-name {font-family:'Space Grotesk',sans-serif;font-size:19px;letter-spacing:.1em;font-weight:700;color:#f3ffff;}
.brand-sub {font-size:9px;letter-spacing:.19em;text-transform:uppercase;color:#7e9ba8;margin-top:1px;}
.eyebrow {color:#70e5d3;letter-spacing:.18em;text-transform:uppercase;font-size:10px;font-weight:700;}
.lede {font-size:15px;color:#9cb0bb;line-height:1.55;max-width:650px;margin-top:-.35rem;}
.status {display:inline-flex;align-items:center;gap:8px;border:1px solid #31504e;border-radius:99px;background:#0f2829;padding:7px 12px;font-size:10px;font-weight:700;letter-spacing:.12em;color:#8ee6d2;text-transform:uppercase;}
.dot {width:6px;height:6px;border-radius:50%;background:#6ee8ce;box-shadow:0 0 9px #6ee8ce;}
.hero {position:relative;overflow:hidden;border:1px solid #23404a;border-radius:16px;padding:25px 27px;background:radial-gradient(ellipse at 85% 50%,#143f42 0%,transparent 45%),linear-gradient(110deg,#102331,#0b1724 68%);margin:12px 0 18px;}
.hero:after {content:'';position:absolute;width:270px;height:270px;border:1px solid #72f0d02b;border-radius:50%;right:5%;top:-75px;box-shadow:0 0 0 40px #63d7c50c,0 0 0 80px #63d7c509,0 0 0 120px #63d7c506;pointer-events:none;}
.hero-label {color:#a9d1cb;font-size:10px;letter-spacing:.16em;font-weight:700;text-transform:uppercase;}
.hero-title {font-family:'Space Grotesk',sans-serif;font-size:25px;line-height:1.2;font-weight:600;color:#f3fffd;margin:7px 0 7px;letter-spacing:-.03em;}
.hero-copy {color:#a0b9bd;font-size:13px;max-width:590px;line-height:1.5;}
.metric {min-height:110px;border:1px solid #203641;border-radius:12px;background:linear-gradient(140deg,#101e2b,#0c1722);padding:16px 18px;position:relative;overflow:hidden;}
.metric:before {content:'';width:3px;height:28px;background:var(--accent);position:absolute;top:18px;left:0;border-radius:0 3px 3px 0;}
.metric-label {font-size:10px;color:#91a6b1;letter-spacing:.12em;text-transform:uppercase;font-weight:700;}
.metric-value {font-family:'Space Grotesk',sans-serif;font-size:30px;font-weight:600;color:#f2faf9;margin-top:6px;letter-spacing:-.04em;}
.metric-foot {font-size:11px;color:#8da4ae;margin-top:3px;}
.section-kicker {font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:#83a1ab;font-weight:700;margin-bottom:5px;}
.section-title {font-family:'Space Grotesk',sans-serif;font-size:20px;color:#edf8f7;font-weight:600;letter-spacing:-.025em;}
.section-caption {color:#90a5af;font-size:12px;line-height:1.5;margin:4px 0 8px;}
.divider {border-top:1px solid #1d303b;margin:24px 0 18px;}
.pill {display:inline-block;padding:5px 9px;border-radius:5px;font-size:10px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;border:1px solid currentColor;}
.detail-value {font-family:'Space Grotesk',sans-serif;font-size:22px;color:#ecf9f7;font-weight:600;}
.detail-label {font-size:10px;color:#8ba2ad;letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px;}
.detail-foot {font-size:11px;color:#91a6ae;margin-top:4px;}
.note {padding:12px 14px;background:#10232b;border:1px solid #294447;color:#a8c2c1;border-radius:9px;font-size:12px;line-height:1.5;}
.warning-note {padding:14px 17px;background:#2a1920;border:1px solid #79464d;border-left:3px solid #ff8085;color:#f2c8ca;border-radius:9px;font-size:12px;line-height:1.55;margin:12px 0 17px;}
.sidebar-foot {color:#79919d;font-size:11px;line-height:1.55;border-top:1px solid #203440;padding-top:16px;margin-top:25px;}
div[data-baseweb="select"] > div, div[data-baseweb="input"] {background:#101f2b!important;border-color:#2a414c!important;}
div[data-testid="stDataFrame"] {border:1px solid #203641;border-radius:10px;overflow:hidden;}
/* Use the explicit exports below so spreadsheet-safe downloads are available. */
div[data-testid="stDataFrame"] button[aria-label="Download as CSV"] {display:none!important;}
div.stDownloadButton button, div.stButton button {border-radius:8px;border:1px solid #457b72;background:#132e30;color:#b7f9e9;font-weight:600;}
div.stDownloadButton button:hover, div.stButton button:hover {border-color:#70efd5;color:#ecfff9;}
.smallprint {font-size:11px;color:#7f98a1;line-height:1.55;}
@media(max-width:800px){.block-container{padding:1.25rem 1rem 3rem;} h1{font-size:2rem!important;}.hero:after{opacity:.3;}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(path: str, modified_ns: int, size: int) -> pd.DataFrame:
    del modified_ns, size
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metrics(path: str, modified_ns: int, size: int) -> dict:
    del modified_ns, size
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metric(label: str, value: str, foot: str, color: str) -> None:
    st.markdown(f'<div class="metric" style="--accent:{color}"><div class="metric-label">{html.escape(label)}</div><div class="metric-value">{html.escape(value)}</div><div class="metric-foot">{html.escape(foot)}</div></div>', unsafe_allow_html=True)


def section(kicker: str, title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-kicker">{html.escape(kicker)}</div><div class="section-title">{html.escape(title)}</div><div class="section-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)


def prediction_downloads(safe_csv: bytes, exact_csv: bytes, filename: str) -> None:
    """Offer a spreadsheet-safe view and an exact, machine-scoring export."""
    st.download_button("↓  Download spreadsheet-safe predictions", data=safe_csv, file_name="spreadsheet_safe_" + filename, mime="text/csv", width="stretch")
    st.caption("This file prefixes formula-like text with an apostrophe. Use the exact file below for machine scoring.")
    with st.expander("Exact CSV for machine scoring"):
        st.caption("The exact CSV preserves original text and IDs for programmatic scoring; confirm the organiser's required columns and treat the file as data rather than opening untrusted text in a spreadsheet.")
        st.download_button("↓  Download exact predictions", data=exact_csv, file_name=filename, mime="text/csv", width="stretch")


def pill(tier: str) -> str:
    color = COLORS.get(str(tier).title(), COLORS["Incomplete"])
    return f'<span class="pill" style="color:{color}">{html.escape(str(tier))}</span>'


def num(value, fmt: str = ",.2f", fallback: str = "—") -> str:
    try:
        n = float(value)
        if np.isfinite(n) and abs(n) >= 1e12 and "e" not in fmt.lower():
            return format(n, ".3e")
        return format(n, fmt) if np.isfinite(n) else fallback
    except (ValueError, TypeError):
        return fallback


def event_sort_key(event_id):
    """Order integer event IDs numerically while preserving uploaded text IDs."""
    label = str(event_id)
    if re.fullmatch(r"[+-]?\d+", label.strip()):
        return (0, int(label), label)
    return (1, label.casefold(), label)


def latest_by_event(df: pd.DataFrame) -> pd.DataFrame:
    """Latest eligible early message per event, never across data sources."""
    eligible = df[pd.to_numeric(df["time_to_tca"], errors="coerce") >= 2].copy()
    return eligible.sort_values(["event_id", "time_to_tca", "input_row_position"], kind="stable").drop_duplicates("event_id", keep="first")


def chart_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=15, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, family="DM Sans, sans-serif", size=11), hoverlabel=dict(bgcolor="#142835", font_color=TEXT), legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig


def geometry_plot(row: pd.Series) -> go.Figure | None:
    vals = pd.to_numeric(row[["relative_position_r", "relative_position_t", "relative_position_n"]], errors="coerce").to_numpy(dtype=float) / 1000
    if not np.isfinite(vals).all():
        return None
    r, t, n = vals
    tier = str(row.get("risk_tier", "Incomplete")).title()
    color = COLORS.get(tier, COLORS["Incomplete"])
    fig = go.Figure()
    theta = np.linspace(0, 2*np.pi, 181)
    for radius, linecolor in [(1, "rgba(255,111,115,.48)"), (5, "rgba(247,189,104,.26)")]:
        fig.add_trace(go.Scatter3d(x=radius*np.cos(theta), y=radius*np.sin(theta), z=np.zeros_like(theta), mode="lines", line=dict(color=linecolor, width=2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, r], y=[0, t], z=[0, n], mode="lines", line=dict(color=color, width=5), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=7, color="#87f4e1", symbol="diamond"), name="Primary", hovertemplate="Primary reference<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[r], y=[t], z=[n], mode="markers", marker=dict(size=8, color=color, line=dict(color="#ffffff", width=1)), name="Relative position", hovertemplate="R %{x:.2f} km<br>T %{y:.2f} km<br>N %{z:.2f} km<extra>At predicted closest approach</extra>"))
    extent = max(6.0, np.max(np.abs(vals))*1.15)
    axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, zerolinecolor="#526a76", color=MUTED, range=[-extent,extent], showbackground=False)
    fig.update_layout(scene=dict(xaxis=dict(axis,title="R · km"), yaxis=dict(axis,title="T · km"), zaxis=dict(axis,title="N · km"), aspectmode="cube", camera=dict(eye=dict(x=1.6,y=1.5,z=.95)), bgcolor="rgba(0,0,0,0)"))
    return chart_layout(fig, 390)


def history_plot(group: pd.DataFrame, selected_row: pd.Series | None = None) -> go.Figure:
    h = group.sort_values("time_to_tca", ascending=False)
    fig = go.Figure()
    for tier in ["Critical", "Elevated", "Safe"]:
        fig.add_trace(go.Scatter(x=h["time_to_tca"], y=h[f"prob_{tier.lower()}"], mode="lines+markers", name=f"P(final {tier})", line=dict(color=COLORS[tier],width=2), marker=dict(size=5), hovertemplate="%{x:.2f} days to TCA<br>P(final " + tier + "): %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=h["time_to_tca"], y=pd.to_numeric(h["miss_distance"], errors="coerce") / 1000, mode="lines+markers", name="Recorded miss · km", yaxis="y2", line=dict(color="#85a8ef",width=1.5,dash="dot"), marker=dict(size=5,color="#85a8ef"), hovertemplate="%{x:.2f} days to TCA<br>Miss distance: %{y:.2f} km<extra></extra>"))
    if selected_row is not None and pd.notna(selected_row.get("time_to_tca")):
        fig.add_vline(x=float(selected_row["time_to_tca"]), line_dash="dash", line_color="rgba(238,255,252,.65)", annotation_text="Selected message", annotation_font_color="#c5eee5")
    fig.update_xaxes(title="Days to closest approach · earlier → later", autorange="reversed", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="Final-tier model probability", range=[0, 1.05], tickformat=".0%", gridcolor=GRID, zeroline=False)
    fig.update_layout(yaxis2=dict(title="Recorded miss · km", overlaying="y", side="right", showgrid=False, rangemode="tozero", color="#85a8ef"))
    return chart_layout(fig, 290)


def safe_text(value) -> str:
    return html.escape(str(value)) if pd.notna(value) else "Not available"


if "tour_launch_count" not in st.session_state:
    st.session_state.tour_launch_count = 0

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">✦</div><div><div class="brand-name">ASTERIA</div><div class="brand-sub">Conjunction intelligence</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Workspace / 01</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Mission control", "Model evidence", "About the data"], key="tour_navigation", label_visibility="collapsed")
    if st.button("✦  Guided tour", key="tour_launch", width="stretch", help="Explore the dashboard with a guided walkthrough"):
        st.session_state.tour_launch_count += 1
    st.markdown('<div class="sidebar-foot"><b style="color:#a9c7c8">Decision support, not flight control.</b><br>Risk estimates are research outputs. A qualified operator retains all maneuver decisions.</div>', unsafe_allow_html=True)

first_visit_component(key="asteria_first_visit", launch=st.session_state.tour_launch_count, default=None)

test_path = ARTIFACTS / "test_predictions.csv"
holdout_path = ARTIFACTS / "holdout_predictions.csv"
metrics_path = ARTIFACTS / "metrics.json"

def sensitivity_warning() -> None:
    """Surface the held-out safety limitation wherever estimates are displayed."""
    if not metrics_path.exists() or bundle["status"] != "verified":
        st.markdown('<div class="warning-note"><b>Research demonstration only.</b> Evaluation is unavailable. Do not rely on these estimates for operational decisions.</div>', unsafe_allow_html=True)
        return
    report = load_metrics(str(metrics_path), metrics_path.stat().st_mtime_ns, metrics_path.stat().st_size)
    rows = report.get("holdout", {}).get("latest_event", {})
    recall = rows.get("critical_recall")
    misses, total = rows.get("critical_fn", rows.get("fn")), rows.get("critical_true")
    if recall is None or misses is None or total is None:
        st.markdown('<div class="warning-note"><b>Research demonstration only.</b> Final-tier sensitivity has not been established. Qualified operators retain all maneuver decisions.</div>', unsafe_allow_html=True)
        return
    st.markdown(f'<div class="warning-note"><b>Early warning · measured limits.</b> Using messages at least 2 days before closest approach, the model identified {recall:.1%} of final Critical events and missed {misses} of {total} on the held-out event split. Probabilities describe final tiers, not collision probability, and are not operationally calibrated. <b>Qualified operators retain all maneuver decisions.</b></div>', unsafe_allow_html=True)

if not test_path.exists():
    st.markdown('<div class="eyebrow">Asteria / Initialization</div>', unsafe_allow_html=True)
    st.title("The mission is almost ready.")
    st.info("Prediction artifacts have not been generated yet. Run `python train.py` from the project directory, then reload this page.")
    st.stop()

bundle = verify_manifest(ARTIFACTS)
with st.sidebar:
    if bundle["status"] == "verified":
        st.caption("Artifact checksums match the recorded bundle. Checksums alone do not establish lineage.")
    else:
        st.warning("Prediction bundle integrity is not verified. Treat its model attribution as unknown.")

data_sources = {"Test set · early messages": test_path}
if holdout_path.exists():
    data_sources["Held-out training events · retrospective"] = holdout_path

if page == "Mission control":
    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown('<div class="eyebrow">Asteria / Conjunction intelligence</div>', unsafe_allow_html=True)
        st.title("See the close calls, sooner.")
        st.markdown('<p class="lede">Forecast the final risk tier at least two days ahead. Explore the telemetry, uncertainty and evidence behind each alert.</p>', unsafe_allow_html=True)
    with top_right:
        st.markdown('<div style="text-align:right;padding-top:18px"><span class="status"><span class="dot"></span> Historical data</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="hero-label">Early conjunction assessment / 2+ days ahead</div><div class="hero-title">A clearer picture of what needs attention.</div><div class="hero-copy">Replay early message updates, rotate the encounter geometry and see how the forecast changes. Three probabilities express the model’s view of the final tier. Every maneuver decision stays with the operator.</div></div>', unsafe_allow_html=True)
    sensitivity_warning()

    with st.sidebar:
        st.markdown('<div class="eyebrow" style="margin:22px 0 8px">Data & filters</div>', unsafe_allow_html=True)
        choices = list(data_sources)
        if (ARTIFACTS / "risk_model.joblib").exists():
            choices.append("Upload telemetry · estimate")
        source = st.selectbox("Dataset", choices, key="tour_dataset", help="Event IDs are scoped to each dataset. The test set has no messages within 24 hours of closest approach.")
    if source == "Upload telemetry · estimate":
        with st.sidebar:
            upload = st.file_uploader("Telemetry CSV", type=["csv"], help="One row per message. Include event_id and all 12 permitted model feature columns. Any risk column is ignored.")
        if upload is None:
            st.info("Choose a telemetry CSV in the sidebar to generate new estimates. It must contain `event_id` and the 12 permitted feature columns; any `risk` column is ignored.")
            st.stop()
        upload_bytes = upload.getvalue()
        model_stat = (ARTIFACTS / "risk_model.joblib").stat()
        input_key = ("upload", hashlib.sha256(upload_bytes).hexdigest(), model_stat.st_size, model_stat.st_mtime_ns)
        download_name = "uploaded_telemetry_predictions.csv"
    else:
        path = data_sources[source]
        path_stat = path.stat()
        input_key = ("file", str(path), path_stat.st_size, path_stat.st_mtime_ns)
        download_name = path.name
    # A session-local cache keeps uploaded telemetry private to this session and
    # avoids revalidation/CSV conversion on every slider or filter interaction.
    # Code and file versions form part of the key; bundle integrity is checked
    # separately on every run and decisions are never cached.
    code_paths = [ROOT / "app.py", ROOT / "src" / "pipeline.py", ROOT / "src" / "validation.py", ROOT / "src" / "risk.py"]
    code_version = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in code_paths)
    cache_key = (input_key, code_version)
    cached = st.session_state.get("prepared_telemetry")
    if not cached or cached["key"] != cache_key:
        try:
            validation_report = None
            if source == "Upload telemetry · estimate":
                from src.pipeline import predict_messages
                # Preserve user-provided IDs such as "001", "NA" and "A-12".
                raw = read_telemetry_csv(io.BytesIO(upload_bytes))
                validation_report = validate_telemetry(raw)
                if validation_report.summary.get("missing_columns"):
                    raise ValueError("Missing required columns: " + ", ".join(validation_report.summary["missing_columns"]))
                if len(raw) == 0:
                    raise ValueError("The CSV contains no telemetry rows. Add at least one message and try again.")
                if "event_id" in raw.columns:
                    ids = raw["event_id"].astype("string")
                    if (ids.isna() | ids.str.strip().eq("").fillna(False)).any():
                        raise ValueError("Every uploaded row must have a non-empty event_id for grouping. Correct the file and try again.")
                df = predict_messages(raw)
            else:
                df = load_csv(str(path), path.stat().st_mtime_ns, path.stat().st_size)
            required = {"source_row_id", "event_id", "time_to_tca", "miss_distance", "risk_tier", "predicted_final_tier", "prob_critical", "prob_elevated", "prob_safe", "prediction_confidence", "forecast_status", "relative_position_r", "relative_position_t", "relative_position_n"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError("Prediction file is missing required columns: " + ", ".join(sorted(missing)))
            if validation_report is None:
                validation_report = validate_telemetry(df)
            if validation_report.summary.get("missing_columns"):
                raise ValueError("The selected prediction file is missing telemetry columns: " + ", ".join(validation_report.summary["missing_columns"]))
            flags = validation_report.flags
            for col in ["review_required", "review_codes", "review_reasons"]:
                df[col] = flags[col].to_numpy()
            df["rule_ids"] = "F-2D; F-FINAL-TIER"
            # Physical input position is separate from any user-provided identifier.
            df["input_row_position"] = np.arange(len(df), dtype=np.int64)
            source_columns = ["source_row_id", "event_id", "time_to_tca", "miss_distance", "relative_speed", "relative_position_r", "relative_position_t", "relative_position_n", "c_object_type", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP"]
            source_bytes = upload_bytes if source == "Upload telemetry · estimate" else df[source_columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
            dataset_id = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
            download_bytes = df.to_csv(index=False).encode("utf-8")
            safe_download_bytes = spreadsheet_safe_csv(df)
            for col in ["time_to_tca", "miss_distance", "prob_critical", "prob_elevated", "prob_safe", "prediction_confidence", "relative_position_r", "relative_position_t", "relative_position_n"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df["risk_tier"] = df["risk_tier"].astype(str).str.title()
            cached = {"key": cache_key, "df": df, "validation_summary": validation_report.summary,
                      "dataset_id": dataset_id, "download_bytes": download_bytes, "safe_download_bytes": safe_download_bytes}
            st.session_state["prepared_telemetry"] = cached
        except Exception as exc:
            st.error(f"Could not process the telemetry: {exc}")
            st.stop()
    df = cached["df"]
    dataset_id = cached["dataset_id"]
    download_bytes = cached["download_bytes"]
    safe_download_bytes = cached["safe_download_bytes"]
    validation_summary = cached["validation_summary"]
    if source == "Upload telemetry · estimate":
        with st.expander("Upload validation report", expanded=True):
            st.write(f"{len(df):,} messages retained · {int(df['review_required'].sum()):,} flagged for analyst review")
            st.caption("Review flags identify questionable inputs, not calibrated uncertainty or evidence that unflagged messages are safe.")
            st.json(validation_summary, expanded=False)
    latest = latest_by_event(df)
    if latest.empty:
        st.warning("No eligible messages at least 2 days before closest approach are available. Later rows are retained in the export with an abstention; they are not used for forecasting.")
        prediction_downloads(safe_download_bytes, download_bytes, download_name)
        st.stop()
    with st.sidebar:
        all_tiers = [x for x in ["Critical", "Elevated", "Safe", "Incomplete"] if x in set(latest["risk_tier"])]
        selected_tiers = st.multiselect("Risk tier", all_tiers, default=all_tiers, key="tour_risk_filter")
        object_types = sorted(df["c_object_type"].fillna("UNKNOWN").astype(str).unique().tolist())
        object_type = st.selectbox("Object type", ["All object types", *object_types])
        review_only = st.checkbox("Analyst review only", help="Show events whose selected message has a data-quality review flag.")
        max_observed = pd.to_numeric(latest["time_to_tca"], errors="coerce").max()
        # Keep the slider usable even for implausibly large finite uploaded
        # values. Extended-time rows remain visible and exportable by default.
        max_day = max(1, int(np.ceil(min(float(max_observed), 365.0))))
        has_extended = bool((latest["time_to_tca"] > 365).any())
        days = st.slider("Time to closest approach (days)", 0.0, float(max_day), (0.0, float(max_day)), step=0.1)
        include_extended = st.checkbox("Include times over 365 days", value=True, help="These unusual values are included when the range reaches 365 days and grouped in the 365+ bucket.") if has_extended else False
        search = st.text_input("Find event ID", placeholder="e.g. 42")
    time_mask = latest["time_to_tca"].between(days[0], days[1])
    if has_extended and include_extended and days[1] == 365.0:
        time_mask |= latest["time_to_tca"] > 365
    filtered = latest[latest["risk_tier"].isin(selected_tiers) & time_mask].copy()
    if object_type != "All object types":
        filtered = filtered[filtered["c_object_type"].fillna("UNKNOWN").astype(str) == object_type]
    if review_only:
        filtered = filtered[filtered["review_required"]]
    if search.strip():
        filtered = filtered[filtered["event_id"].astype(str).str.contains(search.strip(), case=False, regex=False)]
    counts = filtered["risk_tier"].value_counts()
    cards = st.columns(4)
    with cards[0]: metric("Events in view", f"{len(filtered):,}", "Latest eligible message · ≥2 days", "#6de5cf")
    with cards[1]: metric("Critical", f"{counts.get('Critical', 0):,}", "Requires priority review", "#ff6f73")
    with cards[2]: metric("Elevated", f"{counts.get('Elevated', 0):,}", "Monitor and investigate", "#f7bd68")
    with cards[3]: metric("24h · early-data coverage", "No data", "Forecast inputs stop at T−2 days", "#86acf4")
    st.caption(f"{int(filtered['review_required'].sum()):,} selected event messages have data-quality review flags. Flags do not measure model uncertainty.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if filtered.empty:
        st.info("No events match the selected filters. Adjust the filters in the sidebar.")
        prediction_downloads(safe_download_bytes, download_bytes, download_name)
        st.stop()
    filtered["_priority"] = filtered["risk_tier"].map({"Critical":0,"Incomplete":1,"Elevated":2,"Safe":3}).fillna(4)
    filtered = filtered.sort_values(["_priority", "time_to_tca", "event_id"], kind="stable")
    overview, inspector = st.columns([1.05, .95], gap="large")
    with overview:
        section("01 / Prioritise", "Conjunction queue", "One latest eligible message per event. Tier is the forecast final outcome; distance is the current recorded estimate.")
        queue = filtered[["event_id", "risk_tier", "review_required", "time_to_tca", "miss_distance", "prob_critical"]].copy()
        queue.columns = ["Event", "Final tier", "Review", "TCA · days", "Miss · m", "P(final Critical)"]
        with st.container(key="tour_queue"):
            st.dataframe(queue, hide_index=True, width="stretch", height=355,
                         column_config={"TCA · days": st.column_config.NumberColumn(format="%.2f"), "Miss · m": st.column_config.NumberColumn(format="%.0f"), "P(final Critical)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})
        with st.expander("Download one forecast per event"):
            event_export = latest[["event_id", "predicted_final_tier", "prob_critical", "prob_elevated", "prob_safe", "source_row_id", "time_to_tca"]].copy()
            st.caption(f"{len(event_export):,} events from the full selected dataset, independent of queue filters. Each uses its latest eligible ≥2-day CDM. Probabilities describe final tiers; time_to_tca is in days.")
            st.download_button("↓  Event forecasts · spreadsheet-safe CSV", data=spreadsheet_safe_csv(event_export), file_name="spreadsheet_safe_event_forecasts.csv", mime="text/csv", width="stretch")
            st.download_button("↓  Event forecasts · exact machine CSV", data=event_export.to_csv(index=False).encode("utf-8"), file_name="event_forecasts.csv", mime="text/csv", width="stretch")
            st.caption("The exact file preserves original IDs for programmatic evaluation. The spreadsheet-safe file neutralizes formula-like text.")
        prediction_downloads(safe_download_bytes, download_bytes, download_name)
        st.markdown('<div class="smallprint">Both files contain every message in the selected dataset, independent of queue filters.</div>', unsafe_allow_html=True)
    with inspector:
        section("02 / Inspect", "Encounter geometry", "Relative position at predicted closest approach — not an orbit or trajectory.")
        event_options = sorted(filtered["event_id"].tolist(), key=event_sort_key)
        event_id = st.selectbox("Event to inspect", event_options, key="tour_event_select", format_func=lambda x: f"Event {x}", label_visibility="collapsed")
        group = df[(df["event_id"] == event_id) & (df["time_to_tca"] >= 2)].sort_values(["time_to_tca", "input_row_position"], ascending=[False, False], kind="stable")
        if group.empty:
            st.warning("No eligible early messages are available for this event.")
            st.stop()
        fingerprint = hashlib.sha256(download_bytes).hexdigest()[:12]
        selection = (source, str(event_id), fingerprint)
        if st.session_state.get("replay_selection") != selection:
            st.session_state["replay_selection"] = selection
            st.session_state["tour_replay"] = len(group) - 1
        elif "tour_replay" not in st.session_state or not 0 <= st.session_state["tour_replay"] < len(group):
            st.session_state["tour_replay"] = len(group) - 1
        if len(group) > 1:
            step = st.slider("Replay recorded messages", min_value=0, max_value=len(group)-1, key="tour_replay", help="Earlier to later eligible messages, all at least 2 days before TCA. Forecasts use the selected message only; later messages are never input features.")
        else:
            step = 0
            st.caption("This event has one eligible early message.")
        row = group.iloc[step]
        st.caption(f"Message {step + 1} of {len(group)} · T−{num(row['time_to_tca'])} days · queue shows the latest message. The plot shows each message's estimated position at TCA, not an orbit.")
        fig = geometry_plot(row)
        if fig is None:
            st.info("Relative position is incomplete for this message.")
        else:
            st.plotly_chart(fig, key="tour_geometry", width="stretch", config={"displayModeBar": False})
        st.markdown('<div class="smallprint">Drag to rotate · R / T / N are radial, along-track and cross-track axes. Rings are 1 km and 5 km guides in the R–T plane.</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    section("03 / Understand", f"Event {event_id}", "Final-tier probabilities from the selected early message. These describe model uncertainty, not the probability of a collision.")
    details, history = st.columns([.8, 1.2], gap="large")
    with details:
        with st.container(border=True):
            info = dict(load_metrics(str(metrics_path), metrics_path.stat().st_mtime_ns, metrics_path.stat().st_size).get("model", {})) if metrics_path.exists() and bundle["status"] == "verified" else {"name": "Not verified"}
            model_file = ARTIFACTS / "risk_model.joblib"
            info["provenance_status"] = "Bundle checksums match; lineage not independently proven" if bundle["status"] == "verified" else f"Not verified: {bundle['details']}"
            if bundle["status"] == "verified" and bundle.get("model_sha256"):
                info["artifact_sha256"] = bundle["model_sha256"]
            st.markdown(f'{pill(row["risk_tier"])} <span style="color:#8da7af;font-size:12px;margin-left:8px">MESSAGE #{safe_text(row["source_row_id"])}</span>', unsafe_allow_html=True)
            if bool(row["review_required"]):
                st.markdown(f'<div class="warning-note"><b>Analyst review · {safe_text(row["review_codes"])}</b><br>{safe_text(row["review_reasons"])}</div>', unsafe_allow_html=True)
            st.write("")
            a, b = st.columns(2)
            with a:
                st.markdown(f'<div class="detail-label">Final-tier confidence</div><div class="detail-value">{num(row["prediction_confidence"], ".1%")}</div><div class="detail-foot">Uncalibrated model probability</div>', unsafe_allow_html=True)
            with b:
                st.markdown(f'<div class="detail-label">Miss distance</div><div class="detail-value">{num(float(row["miss_distance"]) / 1000)} km</div><div class="detail-foot">{num(row["time_to_tca"])} days to TCA</div>', unsafe_allow_html=True)
            st.write("")
            for tier in ["Critical", "Elevated", "Safe"]:
                probability = float(row.get(f"prob_{tier.lower()}", float("nan")))
                if np.isfinite(probability):
                    st.progress(min(1.0, max(0.0, probability)), text=f"P(final {tier}) · {probability:.1%}")
            st.markdown(f'<div class="detail-label">Forecast basis</div><div style="color:#d4e6e5;font-size:13px;line-height:1.5">{safe_text(row.get("triggered_rules", "Latest eligible early telemetry → final-tier classifier"))}</div>', unsafe_allow_html=True)
            st.caption("F-2D: inputs ≥2 days before TCA · F-FINAL-TIER: organizer event labels · updated Airbus brief, pages 1–2. Classifier probabilities are not Pc.")
            if "data_quality" in row.index:
                st.markdown(f'<div class="detail-label" style="margin-top:16px">Model input status</div><div class="smallprint">{safe_text(row.get("data_quality"))}</div>', unsafe_allow_html=True)
            with st.expander("Message evidence"):
                st.caption(f"Source: {source} · row ID {row['source_row_id']} · zero-based input position {row['input_row_position']} · event IDs are scoped to the source dataset")
                st.caption(f"Dataset fingerprint: {dataset_id} ({'original upload bytes' if source == 'Upload telemetry · estimate' else 'canonical telemetry rows'})")
                st.caption(f"Model: {info.get('name', 'Not available')} · artifact SHA-256: {info.get('artifact_sha256', 'Not available')}")
                st.json({
                    "object_type": row.get("c_object_type"),
                    "relative_speed_m_per_s": row.get("relative_speed"),
                    "relative_position_rtn_m": [row.get("relative_position_r"), row.get("relative_position_t"), row.get("relative_position_n")],
                    "target_radial_sigma_m": row.get("t_sigma_r"),
                    "other_object_radial_sigma_m": row.get("c_sigma_r"),
                    "max_risk_estimate_as_provided": row.get("max_risk_estimate"),
                }, expanded=False)
                brief_path = ROOT / "docs" / "airbus-challenge-updated.pdf"
                if brief_path.exists():
                    st.download_button("↓  Airbus track brief · rule source", data=brief_path.read_bytes(), file_name=brief_path.name, mime="application/pdf", width="stretch")
            if (ROOT / "src" / "report.py").exists():
                try:
                    from src.report import create_event_assessment_pdf
                    report_pdf = create_event_assessment_pdf(row.to_dict(), dataset_label=f"{source} | {dataset_id}", model_info=info)
                    report_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(event_id))[:80]
                    st.download_button("↓  Download event assessment PDF", data=report_pdf, file_name=f"asteria_assessment_{report_id}.pdf", mime="application/pdf", width="stretch")
                except Exception as exc:
                    st.error(f"Could not generate the assessment report: {exc}")
            with st.expander("Record an analyst review"):
                st.caption("Records are stored on the app server and shared with its viewers. Cloud-hosted records may be lost on restart. They document an assessment and do not authorize or execute a maneuver.")
                identity_codes = set(str(row.get("review_codes", "")).split("; "))
                ambiguous_identity = bool(identity_codes & {"R-ROW-ID", "R-ID"})
                if bundle["status"] != "verified":
                    st.warning("Review recording is unavailable until the prediction bundle's checksums are verified.")
                if ambiguous_identity:
                    st.warning("This message has an invalid event ID or a missing, invalid or duplicate source-row ID, so a review record cannot uniquely identify it. Correct the IDs and upload the file again; all rows remain available in the prediction export.")
                with st.form("analyst_review_form", clear_on_submit=True):
                    analyst = st.text_input("Analyst name")
                    action = st.selectbox("Review action", ["Escalate", "Accept", "Override"])
                    override = st.selectbox("Override tier (only for Override)", ["Choose a tier", "Critical", "Elevated", "Safe", "Incomplete"])
                    reason = st.text_area("Reason", help="At least eight characters, including letters.")
                    submit = st.form_submit_button("Save review record", disabled=bundle["status"] != "verified" or ambiguous_identity)
                if submit:
                    from src.decisions import record_decision
                    try:
                        saved = record_decision(ARTIFACTS / "analyst_decisions.sqlite", dataset_id=dataset_id,
                                                event_id=row["event_id"], source_row_id=row["source_row_id"],
                                                action=action, reason=reason, analyst=analyst, model_path=model_file,
                                                override_tier=override if action == "Override" else None, evidence=row.to_dict(),
                                                expected_model_sha256=bundle["model_sha256"])
                        st.success(f"Recorded {saved['action'].lower()} for event {saved['event_id']} / row {saved['source_row_id']}. No maneuver was authorized.")
                    except Exception as exc:
                        st.error(f"Could not save the review record: {exc}")
                decisions_file = ARTIFACTS / "analyst_decisions.sqlite"
                if decisions_file.exists():
                    from src.decisions import export_decisions_csv, list_decisions
                    try:
                        if not ambiguous_identity:
                            records = list_decisions(decisions_file, dataset_id=dataset_id, event_id=row["event_id"], source_row_id=row["source_row_id"])
                            if records:
                                st.caption("Saved reviews for this message · newest first. Earlier model assessments are retained with their model version; they are not assessments of the current forecast.")
                                history_rows = [{"Recorded UTC": item["recorded_at_utc"], "Action": item["action"], "Override": item["override_tier"] or "—", "Model version": str(item.get("model_sha256", "Unknown"))[:12], "Current model": item.get("model_sha256") == bundle.get("model_sha256"), "Analyst": item["analyst"], "Reason": item["reason"]} for item in records[:10]]
                                st.dataframe(pd.DataFrame(history_rows), hide_index=True, width="stretch")
                        st.download_button("↓  Export this dataset's review records", data=export_decisions_csv(decisions_file, dataset_id=dataset_id), file_name="asteria_review_records.csv", mime="text/csv", width="stretch")
                    except Exception as exc:
                        st.error(f"Could not export the review records: {exc}")
    with history:
        st.plotly_chart(history_plot(group, selected_row=row), key="tour_history", width="stretch", config={"displayModeBar": False})
        st.markdown('<div class="smallprint">Coral, amber and mint show final Critical, Elevated and Safe probabilities (left axis). Blue shows recorded miss distance (right axis). Each forecast uses only its selected early message. The dashed line marks your replay position; inputs stop at T−2 days.</div>', unsafe_allow_html=True)
        with st.expander("Relative-position components through early messages"):
            rtn = go.Figure()
            ordered = group.sort_values("time_to_tca", ascending=False)
            for component, label, color in [("r", "Radial", "#6de5cf"), ("t", "Along-track", "#85a8ef"), ("n", "Cross-track", "#f7bd68")]:
                rtn.add_trace(go.Scatter(x=ordered["time_to_tca"], y=pd.to_numeric(ordered[f"relative_position_{component}"], errors="coerce") / 1000, mode="lines+markers", name=label, line=dict(color=color), hovertemplate="%{x:.2f} days to TCA<br>" + label + ": %{y:.3f} km<extra></extra>"))
            rtn.add_vline(x=float(row["time_to_tca"]), line_dash="dash", line_color="#bddbd6")
            rtn.update_xaxes(title="Days to closest approach", autorange="reversed", gridcolor=GRID)
            rtn.update_yaxes(title="Recorded relative position · km", gridcolor=GRID)
            st.plotly_chart(chart_layout(rtn, 270), key="rtn_time_history", width="stretch", config={"displayModeBar": False})
            st.caption("Each point is a CDM estimate of relative position at closest approach, indexed by the message’s days to TCA. It is not an object’s path through space.")
    st.markdown('<div class="note" style="margin-top:18px"><b>Why the 24-hour view has no data:</b> the official forecast task allows only messages at least 2 days before closest approach. An empty 0–1 day window means unavailable coverage, not zero danger. Historical messages have no common observation clock and are not a live warning feed.</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    bucket_caption = "One latest eligible early message per filtered event, grouped in 24-hour-wide intervals and colored by forecast final tier. These messages were not observed at a common time."
    if (filtered["time_to_tca"] >= 365).any():
        bucket_caption += " Unusually long values are grouped into a 365+ days bucket."
    section("04 / Look ahead", "Historical time-to-approach buckets", bucket_caption)
    buckets = filtered.copy()
    buckets["bucket_start"] = np.floor(np.minimum(buckets["time_to_tca"], 365)).astype(int)
    buckets["bucket"] = buckets["bucket_start"].map(lambda x: "365+ days" if x == 365 else f"{x}–{x+1} days")
    counts_by_bucket = buckets.groupby(["bucket_start", "bucket", "risk_tier"], observed=True).size().unstack(fill_value=0).reset_index().sort_values("bucket_start")
    bucket_plot = go.Figure()
    for tier in ["Critical", "Elevated", "Safe", "Incomplete"]:
        if tier in counts_by_bucket:
            bucket_plot.add_bar(name=tier, x=counts_by_bucket["bucket"], y=counts_by_bucket[tier], marker_color=COLORS[tier], hovertemplate="%{x}<br>" + tier + ": %{y} events<extra></extra>")
    bucket_plot.update_layout(barmode="stack")
    bucket_plot.update_yaxes(title="Events", gridcolor=GRID, zeroline=False)
    bucket_plot.update_xaxes(title="Recorded time to closest approach", showgrid=False)
    st.plotly_chart(chart_layout(bucket_plot, 300), key="historical_bucket_chart", width="stretch", config={"displayModeBar": False})
    distances = pd.to_numeric(buckets["miss_distance"], errors="coerce")
    buckets["under_1km"] = distances.ge(0) & distances.lt(1000)
    buckets["watch_1_to_5km"] = distances.ge(1000) & distances.le(5000)
    distance_counts = buckets.groupby(["bucket_start", "bucket"], observed=True).agg(events=("event_id", "size"), under_1km=("under_1km", "sum"), watch_1_to_5km=("watch_1_to_5km", "sum")).reset_index().sort_values("bucket_start")
    distance_counts = distance_counts.rename(columns={"bucket": "Time to TCA", "events": "Events", "under_1km": "Under 1 km", "watch_1_to_5km": "1–5 km"}).drop(columns="bucket_start")
    st.dataframe(distance_counts, hide_index=True, width="stretch")

elif page == "Model evidence":
    st.markdown('<div class="eyebrow">Asteria / Validation</div>', unsafe_allow_html=True)
    st.title("Evidence before confidence.")
    st.markdown('<p class="lede">A final-tier forecast tested on unseen events, with its baseline, missed cases and limitations in view.</p>', unsafe_allow_html=True)
    sensitivity_warning()
    if bundle["status"] != "verified":
        st.warning("The evaluation bundle is not verified and may not correspond to the displayed model.")
    if not metrics_path.exists():
        st.info("Evaluation metrics have not been generated. Run `python train.py` and reload the page.")
        st.stop()
    metrics = load_metrics(str(metrics_path), metrics_path.stat().st_mtime_ns, metrics_path.stat().st_size)
    st.markdown('<div class="note"><b>The official task:</b> predict the organizer’s final event tier using only telemetry at least 2 days before closest approach. Headline scores use one latest eligible message per held-out event. Final label fields are targets only; test-file outcomes are unavailable and never used.</div>', unsafe_allow_html=True)
    st.caption("Events are separated into training, validation and holdout groups. Candidate selection uses validation only. Repeated inspection during development makes this an exploratory evaluation, not a prospective operational trial.")
    st.write("")
    holdout = metrics.get("holdout", {})
    result = holdout.get("latest_event", {})
    def percent(value):
        return f"{100 * value:.1f}%" if value is not None else "—"
    section("01 / Measured performance", "Held-out event evaluation", "One prediction per eligible event. Recall reveals missed Critical outcomes; precision shows how many Critical alerts were correct.")
    if result:
        cards = st.columns(4)
        with cards[0]: metric("Critical recall", percent(result.get("critical_recall")), f"{result.get('critical_fn', '—')} missed of {result.get('critical_true', '—')} Critical events", "#ff6f73")
        with cards[1]: metric("Critical precision", percent(result.get("critical_precision")), f"{result.get('critical_fp', '—')} false Critical alerts", "#f7bd68")
        with cards[2]: metric("Macro F1", num(result.get("macro_f1"), ".3f"), "Equal weight to each final tier", "#6de5cf")
        with cards[3]: metric("Held-out events", f"{result.get('n_events', 0):,}", f"Accuracy {percent(result.get('accuracy'))}", "#86acf4")
        st.write("")
        left, right = st.columns(2, gap="large")
        comparisons = [("Majority baseline", holdout.get("majority_latest_event", {})), ("Early distance only", holdout.get("distance_only_latest_event", {})), ("Final-tier model", result)]
        with left:
            section("02 / Compare", "Forecast versus baseline", "All methods predict the same held-out final labels from the same early evaluation snapshots. Higher recall and macro F1 are better.")
            fig = go.Figure()
            for label, field, color in [("Critical recall", "critical_recall", "#ff8588"), ("Macro F1", "macro_f1", "#69e5c7")]:
                fig.add_bar(name=label, x=[x[0] for x in comparisons], y=[x[1].get(field) for x in comparisons], marker_color=color, hovertemplate="%{x}<br>" + label + ": %{y:.1%}<extra></extra>")
            fig.update_yaxes(title="Score", range=[0, 1.12], tickformat=".0%", gridcolor=GRID)
            fig.update_layout(barmode="group")
            st.plotly_chart(chart_layout(fig, 320), key="tour_error_chart", width="stretch", config={"displayModeBar": False})
        with right:
            section("03 / Inspect", "Final-tier outcomes", "Rows are the organizer’s final tiers; columns are forecast tiers. Every cell counts events, not repeated messages.")
            matrix = result.get("confusion_matrix", {})
            if matrix.get("labels") and matrix.get("values"):
                fig = go.Figure(go.Heatmap(z=matrix["values"], x=matrix["labels"], y=matrix["labels"], colorscale=[[0,"#132531"],[.5,"#245952"],[1,"#74e5cc"]], text=matrix["values"], texttemplate="%{text}", textfont=dict(color="#f0ffff"), hovertemplate="Final %{y}<br>Forecast %{x}<br>Events %{z}<extra></extra>", showscale=False))
                fig.update_xaxes(title="Forecast final tier")
                fig.update_yaxes(title="Official final tier", autorange="reversed")
                st.plotly_chart(chart_layout(fig, 320), key="tour_confusion_chart", width="stretch", config={"displayModeBar": False})
        st.dataframe(pd.DataFrame([{"Method": label, "Accuracy": percent(values.get("accuracy")), "Critical recall": percent(values.get("critical_recall")), "Critical precision": percent(values.get("critical_precision")), "Macro F1": num(values.get("macro_f1"), ".3f")} for label, values in comparisons]), hide_index=True, width="stretch")
        distance_reference = holdout.get("distance_only_latest_event", {})
        if distance_reference.get("macro_f1", -1) > result.get("macro_f1", -1):
            recall_change = 100 * (result.get("critical_recall", 0) - distance_reference.get("critical_recall", 0))
            st.markdown(f'<div class="note"><b>The simple baseline remains competitive:</b> early distance alone has higher macro F1 ({num(distance_reference.get("macro_f1"), ".4f")} versus {num(result.get("macro_f1"), ".4f")}). The classifier changes Critical recall by {recall_change:+.2f} percentage points, with Critical precision {percent(result.get("critical_precision"))} versus {percent(distance_reference.get("critical_precision"))}. This is a measured tradeoff, not an across-the-board improvement.</div>', unsafe_allow_html=True)
        st.caption(f"Probability scoring: multiclass log loss {num(result.get('log_loss'), '.3f')} · multiclass Brier score {num(result.get('multiclass_brier'), '.3f')}. Lower is better. These scores do not establish calibration or applicability to live operations.")
        if holdout_path.exists():
            examples = load_csv(str(holdout_path), holdout_path.stat().st_mtime_ns, holdout_path.stat().st_size).copy()
            examples["input_row_position"] = np.arange(len(examples))
            examples = latest_by_event(examples)
            if {"true_final_tier", "risk_tier"}.issubset(examples.columns):
                misses = examples[(examples["true_final_tier"] == "Critical") & (examples["risk_tier"] != "Critical")]
                if not misses.empty:
                    sample = misses.sort_values("prob_critical").iloc[0]
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    section("04 / A visible failure", f"Event {sample['event_id']}: a missed Critical outcome", "This early snapshot did not lead the model to the correct final tier. The final outcome is shown for retrospective evaluation only.")
                    c1, c2, c3 = st.columns(3)
                    with c1: metric("Official final tier", "Critical", "From organizer event labels", "#ff6f73")
                    with c2: metric("Forecast final tier", str(sample["risk_tier"]), f"P(final Critical) {percent(sample.get('prob_critical'))}", "#f7bd68")
                    with c3: metric("Early snapshot", f"T−{num(sample.get('time_to_tca'))} d", f"Recorded miss {num(sample.get('miss_distance'), ',.0f')} m", "#86acf4")
                    st.markdown('<div class="note" style="margin-top:14px">Early geometry and uncertainty estimates can change before the final CDM. This model uses a single eligible snapshot and cannot observe later changes. Its probability is not a safety guarantee; operators must review updated tracking information. The data does not establish the specific cause of this individual error.</div>', unsafe_allow_html=True)
                false_alarms = examples[(examples["true_final_tier"] != "Critical") & (examples["risk_tier"] == "Critical")]
                if not false_alarms.empty:
                    sample = false_alarms.iloc[0]
                    st.markdown(f'<div class="note" style="margin-top:14px"><b>False Critical alert:</b> event {safe_text(sample["event_id"])} · source row {safe_text(sample["source_row_id"])}. Forecast Critical, official final tier {safe_text(sample["true_final_tier"])}; P(final Critical) {percent(sample.get("prob_critical"))}.</div>', unsafe_allow_html=True)
    else:
        st.info("No final-tier holdout results were recorded in the metrics artifact.")
    for key, title in [("holdout", "Full held-out results"), ("validation", "Validation and model selection"), ("split", "Event split"), ("data", "Coverage and data profile"), ("model", "Model and features"), ("notes", "Limitations and notes")]:
        if key in metrics:
            with st.expander(title): st.json(metrics[key], expanded=2)
    st.markdown('<div class="note" style="margin-top:16px">Coverage matters: 11,942 of 13,154 labeled events contain an eligible early message. The other 1,212 cannot be forecast under the official cutoff and are excluded from model evaluation. The historical sample is not a live operational prevalence estimate.</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="eyebrow">Asteria / Method</div>', unsafe_allow_html=True)
    st.title("Clarity in every decision.")
    st.markdown('<p class="lede">The updated Airbus task: forecast the final tier, from early telemetry, with uncertainty and traceable evidence.</p>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        section("01 / Define the target", "The official final-tier rules", "These rules label final outcomes in the organizer’s workbook. The model forecasts that label from early telemetry; it does not apply these rules to classifier probabilities.")
        st.markdown("""
        | Final tier | Rule on final recorded telemetry |
        |:--|:--|
        | **Critical** | risk ≥ −4 **or** miss distance < 1,000 m |
        | **Elevated** | Not Critical, and risk ≥ −6 **or** miss distance ≤ 5,000 m |
        | **Safe** | risk < −6 **and** miss distance > 5,000 m |
        """)
        st.markdown('<div class="note"><b>Two different probabilities:</b> recorded risk is log₁₀(Pc), so Pc = 10<sup>risk</sup>. Our three outputs instead estimate P(final tier = Critical / Elevated / Safe). They are uncalibrated class probabilities and must never be interpreted as collision Pc.</div>', unsafe_allow_html=True)
        brief_path = ROOT / "docs" / "airbus-challenge-updated.pdf"
        if brief_path.exists():
            st.download_button("↓  Updated Airbus challenge · official source", data=brief_path.read_bytes(), file_name=brief_path.name, mime="application/pdf", width="stretch")
    with right:
        section("02 / Respect the cutoff", "What goes into a forecast", "One latest eligible CDM per event for training and headline evaluation. Replay forecasts use each selected early snapshot independently.")
        st.markdown("""
        - Only messages with **time_to_tca ≥ 2 days** are eligible. Later uploads receive an abstention.
        - Miss distance and R/T/N are **metres**; the 3D view displays kilometres. Relative speed is **m/s**.
        - The original 12 telemetry features are retained. `risk`, `event_id` and final label fields are not model inputs.
        - Train, validation and holdout contain separate event IDs. Train/test IDs restart independently and are never joined.
        - Test data contains only early messages and **no final outcomes**; it demonstrates forecasts, not measured accuracy.
        - No orbital elements or common clock times are available. Plots show relative geometry and time-to-TCA, not propagated orbits.
        """)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    section("03 / Know the coverage", "Official labels, explicit limits", "The new workbook is the source of truth for final outcomes.")
    a, b, c = st.columns(3)
    with a: metric("Labeled events", "13,154", "908 Critical · 2,403 Elevated · 9,843 Safe", "#6de5cf")
    with b: metric("Eligible early events", "11,942", "834 Critical · 2,162 Elevated · 8,946 Safe", "#f7bd68")
    with c: metric("Without early coverage", "1,212", "Excluded; no ≥2-day message exists", "#86acf4")
    st.write("")
    st.markdown('<div class="note"><b>Final means the organizer’s last available record:</b> the workbook selects the smallest time_to_tca. For 3,050 events that record is still ≥2 days out; 389 final timestamps are negative. We use the supplied labels as authoritative, without claiming that every final outcome was observed at TCA. Around 41% of original train rows have risk = −30, a placeholder for Pc ≈ 0, not a precisely measured probability.</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    section("04 / Responsible use", "Human judgment stays in the loop", "Forecasts support prioritization; a qualified person evaluates every operational maneuver.")
    st.markdown('<div class="note">This research prototype has no live tracking feed, certified maneuver policy or operational probability calibration. It cannot authorize or execute a maneuver. Model evidence includes baselines and real held-out errors so its limits stay visible.</div>', unsafe_allow_html=True)

st.caption(f"Asteria · {APP_VERSION} · Updated Airbus brief, 31 October 2026 12:45 PM")
