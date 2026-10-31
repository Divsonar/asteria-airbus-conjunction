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

from src.risk import classify_with_rule_ids
from src.provenance import verify_manifest
from src.validation import read_telemetry_csv, spreadsheet_safe_csv, validate_telemetry


ROOT = Path(__file__).resolve().parent
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


def latest_by_event(df: pd.DataFrame) -> pd.DataFrame:
    """Latest usable historical message per event, never across data sources."""
    nonnegative = df[pd.to_numeric(df["time_to_tca"], errors="coerce") >= 0].copy()
    return nonnegative.sort_values(["event_id", "time_to_tca", "input_row_position"], kind="stable").drop_duplicates("event_id", keep="first")


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
    fig.add_hrect(y0=-4, y1=0, fillcolor="rgba(255,111,115,.08)", line_width=0)
    fig.add_hline(y=-4, line_dash="dot", line_color="rgba(255,111,115,.7)", annotation_text="Critical · −4", annotation_font_color="#ff9a9c")
    fig.add_hline(y=-6, line_dash="dot", line_color="rgba(247,189,104,.65)", annotation_text="Elevated · −6", annotation_font_color="#edc583")
    fig.add_trace(go.Scatter(x=h["time_to_tca"], y=h["predicted_log10_pc"], mode="lines+markers", name="Estimated log₁₀(Pc)", line=dict(color="#6de5cf",width=2.5), marker=dict(size=7,color="#6de5cf"), hovertemplate="%{x:.2f} days to TCA<br>Estimated log₁₀(Pc): %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=h["time_to_tca"], y=pd.to_numeric(h["miss_distance"], errors="coerce") / 1000, mode="lines+markers", name="Recorded miss · km", yaxis="y2", line=dict(color="#85a8ef",width=1.5,dash="dot"), marker=dict(size=5,color="#85a8ef"), hovertemplate="%{x:.2f} days to TCA<br>Miss distance: %{y:.2f} km<extra></extra>"))
    if selected_row is not None and pd.notna(selected_row.get("time_to_tca")):
        fig.add_vline(x=float(selected_row["time_to_tca"]), line_dash="dash", line_color="rgba(238,255,252,.65)", annotation_text="Selected message", annotation_font_color="#c5eee5")
    fig.update_xaxes(title="Days to closest approach · earlier → later", autorange="reversed", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="Estimated log₁₀(Pc)", gridcolor=GRID, zeroline=False)
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
    rows = report.get("holdout", {}).get("rows", {})
    recall = rows.get("risk_only_critical_recall")
    misses, total = rows.get("risk_only_critical_fn"), rows.get("risk_only_critical_true")
    if recall is None or misses is None or total is None:
        st.markdown('<div class="warning-note"><b>Research demonstration only.</b> Risk-only sensitivity has not been established. Do not rely on these estimates for operational decisions.</div>', unsafe_allow_html=True)
        return
    st.markdown(f'<div class="warning-note"><b>Research demonstration only · limited sensitivity.</b> On held-out historical messages, the model identified {recall:.1%} of risk-only critical cases and missed {misses} of {total}. The combined-tier score benefits from the separate distance rule. <b>Do not rely on this model for operational decisions.</b></div>', unsafe_allow_html=True)

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
        st.markdown('<p class="lede">A clear view of conjunction messages, model-estimated collision risk and the rules behind every alert.</p>', unsafe_allow_html=True)
    with top_right:
        st.markdown('<div style="text-align:right;padding-top:18px"><span class="status"><span class="dot"></span> Historical data</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="hero-label">Conjunction assessment / decision support</div><div class="hero-title">A clearer picture of what needs attention.</div><div class="hero-copy">Explore relative encounter geometry, compare message updates and trace each classification to its threshold. Estimates support review; they do not authorize maneuvers.</div></div>', unsafe_allow_html=True)
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
            required = {"source_row_id", "event_id", "time_to_tca", "miss_distance", "risk_tier", "predicted_log10_pc", "estimated_pc", "relative_position_r", "relative_position_t", "relative_position_n"}
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
            _, _, _, rule_ids = classify_with_rule_ids(df["predicted_log10_pc"], df["miss_distance"])
            df["rule_ids"] = rule_ids
            # Physical input position is separate from any user-provided identifier.
            df["input_row_position"] = np.arange(len(df), dtype=np.int64)
            source_columns = ["source_row_id", "event_id", "time_to_tca", "miss_distance", "relative_speed", "relative_position_r", "relative_position_t", "relative_position_n", "c_object_type", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP"]
            source_bytes = upload_bytes if source == "Upload telemetry · estimate" else df[source_columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
            dataset_id = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
            download_bytes = df.to_csv(index=False).encode("utf-8")
            safe_download_bytes = spreadsheet_safe_csv(df)
            for col in ["time_to_tca", "miss_distance", "predicted_log10_pc", "estimated_pc", "relative_position_r", "relative_position_t", "relative_position_n"]:
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
        st.warning("No messages with a nonnegative time to closest approach are available in this dataset.")
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
    with cards[0]: metric("Events in view", f"{len(filtered):,}", "Latest recorded message per event", "#6de5cf")
    with cards[1]: metric("Critical", f"{counts.get('Critical', 0):,}", "Requires priority review", "#ff6f73")
    with cards[2]: metric("Elevated", f"{counts.get('Elevated', 0):,}", "Monitor and investigate", "#f7bd68")
    danger_distance = pd.to_numeric(filtered["miss_distance"], errors="coerce")
    danger_24h = filtered["time_to_tca"].between(0, 1) & danger_distance.ge(0) & danger_distance.lt(1000)
    with cards[3]: metric("24h · under 1 km", f"{int(danger_24h.sum()):,}", "Retrospective snapshots, not live", "#86acf4")
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
        section("01 / Prioritise", "Conjunction queue", "One latest available nonnegative-time message per event. Select an event to inspect its evidence.")
        queue = filtered[["event_id", "risk_tier", "review_required", "time_to_tca", "miss_distance", "estimated_pc"]].copy()
        queue.columns = ["Event", "Tier", "Review", "TCA · days", "Miss · m", "Estimated Pc"]
        with st.container(key="tour_queue"):
            st.dataframe(queue, hide_index=True, width="stretch", height=355,
                         column_config={"TCA · days": st.column_config.NumberColumn(format="%.2f"), "Miss · m": st.column_config.NumberColumn(format="%.0f"), "Estimated Pc": st.column_config.NumberColumn(format="%.2e")})
        prediction_downloads(safe_download_bytes, download_bytes, download_name)
        st.markdown('<div class="smallprint">Both files contain every message in the selected dataset, independent of queue filters.</div>', unsafe_allow_html=True)
    with inspector:
        section("02 / Inspect", "Encounter geometry", "Relative position at predicted closest approach — not an orbit or trajectory.")
        event_options = filtered["event_id"].tolist()
        event_id = st.selectbox("Event to inspect", event_options, key="tour_event_select", format_func=lambda x: f"Event {x}", label_visibility="collapsed")
        group = df[(df["event_id"] == event_id) & (df["time_to_tca"] >= 0)].sort_values(["time_to_tca", "input_row_position"], ascending=[False, False], kind="stable")
        if group.empty:
            st.warning("No nonnegative-time messages are available for this event.")
            st.stop()
        fingerprint = hashlib.sha256(download_bytes).hexdigest()[:12]
        selection = (source, str(event_id), fingerprint)
        if st.session_state.get("replay_selection") != selection:
            st.session_state["replay_selection"] = selection
            st.session_state["tour_replay"] = len(group) - 1
        elif "tour_replay" not in st.session_state or not 0 <= st.session_state["tour_replay"] < len(group):
            st.session_state["tour_replay"] = len(group) - 1
        if len(group) > 1:
            step = st.slider("Replay recorded messages", min_value=0, max_value=len(group)-1, key="tour_replay", help="Earlier to later historical messages. The latest snapshot is selected by default.")
        else:
            step = 0
            st.caption("This event has one nonnegative-time message.")
        row = group.iloc[step]
        st.caption(f"Message {step + 1} of {len(group)} · T−{num(row['time_to_tca'])} days · queue shows the latest message. The plot shows each message's estimated position at TCA, not an orbit.")
        fig = geometry_plot(row)
        if fig is None:
            st.info("Relative position is incomplete for this message.")
        else:
            st.plotly_chart(fig, key="tour_geometry", width="stretch", config={"displayModeBar": False})
        st.markdown('<div class="smallprint">Drag to rotate · R / T / N are radial, along-track and cross-track axes. Rings are 1 km and 5 km guides in the R–T plane.</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    section("03 / Understand", f"Event {event_id}", "Evidence from the selected historical message and the history of the model estimates.")
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
                st.markdown(f'<div class="detail-label">Estimated Pc</div><div class="detail-value">{num(row["estimated_pc"], ".2e")}</div><div class="detail-foot">log₁₀(Pc) {num(row["predicted_log10_pc"])}</div>', unsafe_allow_html=True)
            with b:
                st.markdown(f'<div class="detail-label">Miss distance</div><div class="detail-value">{num(float(row["miss_distance"]) / 1000)} km</div><div class="detail-foot">{num(row["time_to_tca"])} days to TCA</div>', unsafe_allow_html=True)
            st.write("")
            st.markdown(f'<div class="detail-label">Rule triggered</div><div style="color:#d4e6e5;font-size:13px;line-height:1.5">{safe_text(row.get("triggered_rules", "See threshold rules below"))}</div>', unsafe_allow_html=True)
            st.caption(f"Rule IDs: {row.get('rule_ids', '—')} · Airbus track brief, page 1 · challenge thresholds")
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
                brief_path = ROOT / "Autonomous SG 2026_Airbus Track.pdf"
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
                st.caption("Records are stored on this computer. They document an assessment and do not authorize or execute a maneuver.")
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
                                st.caption("Saved reviews for this message · newest first")
                                history_rows = [{"Recorded UTC": item["recorded_at_utc"], "Action": item["action"], "Override": item["override_tier"] or "—", "Analyst": item["analyst"], "Reason": item["reason"]} for item in records[:10]]
                                st.dataframe(pd.DataFrame(history_rows), hide_index=True, width="stretch")
                        st.download_button("↓  Export this dataset's review records", data=export_decisions_csv(decisions_file, dataset_id=dataset_id), file_name="asteria_review_records.csv", mime="text/csv", width="stretch")
                    except Exception as exc:
                        st.error(f"Could not export the review records: {exc}")
    with history:
        st.plotly_chart(history_plot(group, selected_row=row), key="tour_history", width="stretch", config={"displayModeBar": False})
        st.markdown('<div class="smallprint">Each point is a historical message. Mint shows estimated risk (left axis); blue shows recorded miss distance (right axis). The dashed vertical line marks the selected message. Neither line forecasts a later message.</div>', unsafe_allow_html=True)
    st.markdown('<div class="note" style="margin-top:18px"><b>Reading this view:</b> 24-hour counts refer to each message’s own time to closest approach. Messages are historical and do not share a verified observation time, so the count is not a live, simultaneous warning feed. Dataset selection over-represents high-risk events.</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    bucket_caption = "One latest available message per filtered event, grouped in 24-hour-wide intervals. These messages were not observed at a common time."
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
    st.markdown('<p class="lede">An honest view of what was measured, where the model misses, and what the numbers do not prove.</p>', unsafe_allow_html=True)
    sensitivity_warning()
    if bundle["status"] != "verified":
        st.warning("The evaluation bundle is not verified and may not correspond to the currently displayed prediction model.")
    if not metrics_path.exists():
        st.info("Evaluation metrics have not been generated. Run `python train.py` and reload the page.")
        st.stop()
    metrics = load_metrics(str(metrics_path), metrics_path.stat().st_mtime_ns, metrics_path.stat().st_size)
    st.markdown('<div class="note"><b>What is being predicted:</b> log₁₀(Pc) reported in the same message, using permitted telemetry. It is a learned surrogate, not a physics-derived collision probability or a forecast of final risk. Validation separates entire events; the test-file risk column is not used.</div>', unsafe_allow_html=True)
    st.caption("The training-file holdout has been inspected during development; these results are exploratory rather than a newly blinded final evaluation.")
    st.write("")
    section("01 / Measured performance", "Held-out event evaluation", "Metrics and denominators below come from the saved training run. Lower MAE and RMSE are better; critical recall exposes missed high-risk cases.")
    holdout = metrics.get("holdout", {})
    if holdout.get("rows"):
        result = holdout["rows"]
        baseline = holdout.get("baseline_rows", {})
        latest = holdout.get("latest_event", {})
        def percent(value):
            return f"{100 * value:.1f}%" if value is not None else "—"
        cards = st.columns(4)
        with cards[0]: metric("Log-risk MAE", num(result.get("mae_log10")), f"Median baseline {num(baseline.get('mae_log10'))}", "#6de5cf")
        with cards[1]: metric("Critical recall", percent(result.get("critical_recall")), f"{result.get('critical_fn', '—')} missed of {result.get('critical_true', '—')} critical rows", "#ff6f73")
        with cards[2]: metric("Risk-only recall", percent(result.get("risk_only_critical_recall")), f"{result.get('risk_only_critical_fn', '—')} missed of {result.get('risk_only_critical_true', '—')} high-risk rows", "#f7bd68")
        with cards[3]: metric("Held-out events", f"{result.get('n_events', 0):,}", f"{result.get('n_rows', 0):,} historical messages", "#86acf4")
        if "pc_driven_critical_recall" in result:
            st.write("")
            section("01B / Sensitivity", "Probability-driven critical cases", "Reference log-risk ≥ −4 and recorded miss distance ≥ 1 km. These are cases where distance alone does not trigger a critical tier.")
            pc_cards = st.columns(3)
            with pc_cards[0]: metric("Pc-driven recall · rows", percent(result.get("pc_driven_critical_recall")), f"{result.get('pc_driven_critical_fn', '—')} missed of {result.get('pc_driven_critical_true', '—')} qualifying messages", "#ff6f73")
            with pc_cards[1]: metric("Pc-driven precision · rows", percent(result.get("pc_driven_critical_precision")), f"{result.get('pc_driven_critical_fp', '—')} false positives in the evaluated subset", "#f7bd68")
            with pc_cards[2]: metric("Pc-driven recall · events", percent(latest.get("pc_driven_critical_recall")), f"{latest.get('pc_driven_critical_fn', '—')} missed of {latest.get('pc_driven_critical_true', '—')} qualifying events", "#86acf4")
            st.caption(f"Relevant-risk error: MAE {num(result.get('mae_log10_risk_ge_minus6'))} log₁₀ units over {result.get('n_risk_ge_minus6', 0):,} messages with recorded log-risk ≥ −6.")
            bootstrap = result.get("pc_driven_recall_event_bootstrap95")
            if bootstrap and bootstrap.get("lower") is not None and bootstrap.get("upper") is not None:
                st.caption(f"Exploratory event-resampled 95% interval for row-level Pc-driven recall: {percent(bootstrap['lower'])}–{percent(bootstrap['upper'])}; {bootstrap.get('positive_events', '—')} events contribute critical cases. The selected sample is not representative of live operations.")
            distance_only = holdout.get("distance_only_rows")
            if distance_only:
                comparison = pd.DataFrame([
                    {"Method": "Distance only", "Critical recall": percent(distance_only.get("critical_recall")), "Pc-driven recall": percent(distance_only.get("pc_driven_critical_recall")), "Pc-driven precision": percent(distance_only.get("pc_driven_critical_precision"))},
                    {"Method": "Model + distance", "Critical recall": percent(result.get("critical_recall")), "Pc-driven recall": percent(result.get("pc_driven_critical_recall")), "Pc-driven precision": percent(result.get("pc_driven_critical_precision"))},
                ])
                st.dataframe(comparison, hide_index=True, width="stretch")
        st.write("")
        left, right = st.columns([1, 1], gap="large")
        with left:
            section("02 / Compare", "Held-out prediction error", "Log₁₀ risk error, measured on the same held-out messages.")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Median baseline", "Selected model"], y=[baseline.get("mae_log10"), result.get("mae_log10")], marker_color=["#567384", "#69e5c7"], text=[num(baseline.get("mae_log10")), num(result.get("mae_log10"))], textposition="outside", hovertemplate="%{x}<br>MAE %{y:.2f}<extra></extra>"))
            fig.update_yaxes(title="MAE · log₁₀(Pc)", gridcolor=GRID, zeroline=False)
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(chart_layout(fig, 280), key="tour_error_chart", width="stretch", config={"displayModeBar": False})
        with right:
            section("03 / Inspect", "Classification outcomes", "Rows are reference tiers from the message's recorded risk; columns are predicted tiers. Both use the recorded miss distance.")
            matrix = result.get("confusion_matrix", {})
            if matrix.get("labels") and matrix.get("values"):
                fig = go.Figure(go.Heatmap(z=matrix["values"], x=matrix["labels"], y=matrix["labels"], colorscale=[[0,"#132531"],[.5,"#245952"],[1,"#74e5cc"]], text=matrix["values"], texttemplate="%{text}", textfont=dict(color="#f0ffff"), hovertemplate="Reference %{y}<br>Predicted %{x}<br>Messages %{z}<extra></extra>", showscale=False))
                fig.update_xaxes(title="Predicted tier", side="bottom")
                fig.update_yaxes(title="Reference tier", autorange="reversed")
                st.plotly_chart(chart_layout(fig, 280), key="tour_confusion_chart", width="stretch", config={"displayModeBar": False})
        st.markdown(f'<div class="note"><b>One message per held-out event:</b> critical recall {percent(latest.get("critical_recall"))}; {latest.get("critical_fn", "—")} missed of {latest.get("critical_true", "—")} critical events. The latest available nonnegative-time message represents each event.</div>', unsafe_allow_html=True)
        if holdout_path.exists():
            examples = load_csv(str(holdout_path), holdout_path.stat().st_mtime_ns, holdout_path.stat().st_size)
            if {"true_log10_pc", "predicted_log10_pc", "event_id", "source_row_id"}.issubset(examples.columns):
                misses = examples[(examples["true_log10_pc"] >= -4) & (examples["miss_distance"] >= 1000) & (examples["predicted_log10_pc"] < -4)]
                if not misses.empty:
                    sample = misses.iloc[0]
                    st.markdown(f'<div class="section-kicker" style="margin-top:22px">04 / A visible failure</div><div class="section-title">A probability-driven critical message the model missed</div><div class="section-caption">Held-out event {safe_text(sample["event_id"])} · source row {safe_text(sample["source_row_id"])}. The recorded miss distance does not independently trigger the critical rule.</div>', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    with c1: metric("Recorded log-risk", num(sample["true_log10_pc"]), "Reference from historical message", "#ff6f73")
                    with c2: metric("Model log-risk", num(sample["predicted_log10_pc"]), "Below the −4 risk threshold", "#f7bd68")
                    with c3: metric("Miss distance", f"{num(sample.get('miss_distance'), ',.0f')} m", "Used independently by tier rules", "#86acf4")
                false_alarms = examples[(examples["true_risk_tier"] != "Critical") & (examples["risk_tier"] == "Critical")]
                if not false_alarms.empty:
                    sample = false_alarms.iloc[0]
                    st.markdown(f'<div class="note" style="margin-top:14px"><b>False critical example:</b> held-out event {safe_text(sample["event_id"])} · row {safe_text(sample["source_row_id"])}. Recorded tier {safe_text(sample["true_risk_tier"])}, predicted tier {safe_text(sample["risk_tier"])}, recorded log-risk {num(sample["true_log10_pc"])}, predicted log-risk {num(sample["predicted_log10_pc"])}.</div>', unsafe_allow_html=True)
        if holdout.get("time_to_tca_buckets"):
            with st.expander("Performance by time to closest approach"):
                st.caption("Historical message-relative buckets; not a live forecast. See counts alongside rates.")
                st.json(holdout["time_to_tca_buckets"], expanded=False)
    else:
        st.info("No held-out results were recorded in the metrics artifact.")
    with st.expander("Full held-out metrics"):
        st.json(holdout, expanded=2)
    if metrics.get("validation"):
        with st.expander("Model selection and feature ablation"):
            st.json(metrics["validation"], expanded=2)
    comparison_path = ROOT / "experiments" / "model_comparison.json"
    if comparison_path.exists():
        with st.expander("Exploratory comparison with the earlier model"):
            st.caption("This comparison uses a reconstructed earlier model and the same previously inspected holdout; it is not an independent evaluation. The experiment file is separate from the checked artifact bundle.")
            try:
                with comparison_path.open(encoding="utf-8") as comparison_file:
                    comparison_data = json.load(comparison_file)
                current_rows = comparison_data["holdout_rows"]["current"]
                matches_bundle = bundle["status"] == "verified" and comparison_data.get("current_model_sha256") == bundle.get("model_sha256")
                matches_metrics = all(current_rows.get(k) == holdout.get("rows", {}).get(k) for k in ["n_rows", "n_events", "pc_driven_critical_tp", "pc_driven_critical_fn", "pc_driven_critical_fp"])
                if not matches_bundle or not matches_metrics:
                    st.warning("This experiment does not match the currently loaded model and metrics; comparison suppressed.")
                else:
                    table = []
                    for label, key in [("Earlier unweighted", "reference"), ("Current weighted", "current")]:
                        values = comparison_data["holdout_rows"][key]
                        event_values = comparison_data["holdout_latest_event"][key]
                        table.append({"Model": label,
                                      "Pc-driven recall · rows": percent(values.get("pc_driven_critical_recall")),
                                      "Missed / qualifying rows": f"{values.get('pc_driven_critical_fn', '—')} / {values.get('pc_driven_critical_true', '—')}",
                                      "Pc-driven precision · rows": percent(values.get("pc_driven_critical_precision")),
                                      "Relevant-risk MAE": num(values.get("mae_log10_risk_ge_minus6")),
                                      "Overall MAE": num(values.get("mae_log10")),
                                      "Pc-driven events found": f"{event_values.get('pc_driven_critical_tp', '—')} / {event_values.get('pc_driven_critical_true', '—')}"})
                    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
                    st.caption("Both models missed the only Pc-driven critical latest-event case. The event denominator is too small to support a reliable performance estimate.")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.warning(f"Could not read the comparison experiment: {exc}")
    for key, title in [("split", "Event split"), ("data", "Data profile"), ("model", "Model and features"), ("notes", "Limitations and notes")]:
        if key in metrics:
            with st.expander(title): st.json(metrics[key], expanded=2)
    st.markdown('<div class="note" style="margin-top:16px">Risk-tier metrics combine the predicted risk with observed miss distance. A distance trigger can mask a missed high-risk prediction, so inspect risk-only recall as well as the combined classification. These historical samples were selected and are not representative of real-world event prevalence.</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="eyebrow">Asteria / Method</div>', unsafe_allow_html=True)
    st.title("Clarity in every decision.")
    st.markdown('<p class="lede">The data, thresholds and boundaries of this demonstration are as important as its predictions.</p>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        section("01 / Classify", "The Airbus challenge rules", "The first matching rule takes precedence. Missing decisive information is marked incomplete.")
        st.markdown("""
        | Tier | Rule |
        |:--|:--|
        | **Critical** | Estimated log₁₀(Pc) ≥ −4 **or** miss distance < 1 km |
        | **Elevated** | Otherwise, estimated log₁₀(Pc) ≥ −6 **or** miss distance ≤ 5 km |
        | **Safe** | Estimated log₁₀(Pc) < −6 **and** miss distance > 5 km |
        | **Incomplete** | Missing inputs prevent a decision |
        """)
        st.markdown('<div class="note">These are the brief’s challenge thresholds, not a universal operational maneuver policy. A critical alert is a cue for qualified review.</div>', unsafe_allow_html=True)
    with right:
        section("02 / Interpret", "What the data can show", "The dataset has repeated conjunction data messages per event.")
        st.markdown("""
        - **Time to closest approach** is measured in days from each message’s creation time.
        - **Miss distance and R/T/N position** are measured in metres in the source file; the geometry view displays kilometres.
        - The test set contains messages at least two days before closest approach. It contains no 24-hour messages.
        - The model estimates the risk recorded in the same message, not a later final collision risk.
        - The test `risk` column is excluded from model inputs and model selection.
        - Event IDs are scoped to each file and cannot be used to match events between train and test.
        """)
        st.markdown('[ESA dataset documentation](https://kelvins.esa.int/collision-avoidance-challenge/data/) · [NASA conjunction assessment](https://www.nasa.gov/cara/step-2-close-approach-risk-assessment/)')
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    section("03 / Responsible use", "Human judgment stays in the loop", "A low estimated probability is not a guarantee of safety. The model has not been certified or calibrated for real operations.")
    st.markdown('<div class="note">This prototype is for demonstration and research. It does not issue maneuver commands, use a live tracking feed, reconstruct orbital trajectories, or substitute for a qualified conjunction assessment.</div>', unsafe_allow_html=True)
