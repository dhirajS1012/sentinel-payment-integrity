"""
Sentinel: Agentic Payment Integrity Console.

Stripe-inspired bright console for the POC. Four workspaces:
  1. Overview      pipeline health, top-line metrics
  2. Detector      IsolationForest scoring, top flagged claims
  3. Reasoning     Gemini agent decisions with policy citations
  4. Policy        the policy library the agent grounds decisions in
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent_reason import (  # noqa: E402
    MODEL,
    build_client,
    claim_to_context,
    load_policies,
    reason_about_claim,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BUILD_VERSION = "0.6.0"
ENVIRONMENT = "demo"
RATE_LIMIT_SLEEP_SEC = 13

# Stripe-aligned chart palette
COLOR_PRIMARY = "#635BFF"
COLOR_GREEN = "#0E7C3A"
COLOR_AMBER = "#B85C00"
COLOR_RED = "#B5341B"
COLOR_GRAY = "#A3ACB9"
COLOR_NAVY = "#1A1F36"
COLOR_GRID = "#E3E8EE"
COLOR_AXIS = "#697386"


def chart_theme() -> dict:
    return {
        "config": {
            "view": {"strokeWidth": 0, "continuousHeight": 280},
            "background": "#FFFFFF",
            "font": "Inter, -apple-system, sans-serif",
            "axis": {
                "labelColor": COLOR_AXIS,
                "titleColor": COLOR_AXIS,
                "labelFontSize": 11,
                "titleFontSize": 12,
                "titleFontWeight": 500,
                "gridColor": COLOR_GRID,
                "gridOpacity": 0.7,
                "domainColor": COLOR_GRID,
                "tickColor": COLOR_GRID,
            },
            "legend": {
                "labelColor": COLOR_NAVY,
                "titleColor": COLOR_AXIS,
                "labelFontSize": 12,
                "titleFontSize": 12,
            },
            "title": {"color": COLOR_NAVY, "fontSize": 14, "fontWeight": 600},
        }
    }


alt.themes.register("stripe", chart_theme)
alt.themes.enable("stripe")


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel | Payment Integrity Console",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---------- Reset Streamlit chrome ---------- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1280px; }

/* ---------- Canvas ---------- */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
    color: #1A1F36;
    background: #F6F9FC;
    font-size: 15px;
    line-height: 1.55;
}
.stApp { background: #F6F9FC; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E3E8EE;
}
section[data-testid="stSidebar"] * { color: #3C4257; }
section[data-testid="stSidebar"] .stSlider label { color: #1A1F36; font-weight: 500; font-size: 14px; }

/* ---------- Typography ---------- */
h1 { color: #1A1F36; font-size: 28px; font-weight: 700; letter-spacing: -0.02em; }
h2 { color: #1A1F36; font-size: 22px; font-weight: 600; letter-spacing: -0.01em; margin-top: 8px; }
h3 { color: #1A1F36; font-size: 17px; font-weight: 600; margin-top: 24px; margin-bottom: 12px; }
h4 { color: #1A1F36; font-size: 15px; font-weight: 600; }
p, .stMarkdown, span, div { color: #3C4257; }

/* ---------- Metric tiles ---------- */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E3E8EE;
    border-radius: 8px;
    padding: 18px 22px;
    box-shadow: 0 1px 2px rgba(60, 66, 87, 0.04);
}
div[data-testid="stMetricValue"] {
    color: #1A1F36;
    font-weight: 600;
    font-size: 26px;
    letter-spacing: -0.02em;
}
div[data-testid="stMetricLabel"] {
    color: #697386;
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---------- Cards ---------- */
.card {
    background: #FFFFFF;
    border: 1px solid #E3E8EE;
    border-radius: 8px;
    padding: 22px 26px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(60, 66, 87, 0.04);
}
.card-title { font-size: 15px; font-weight: 600; color: #1A1F36; margin-bottom: 4px; }
.card-sub { font-size: 13px; color: #697386; margin-bottom: 16px; }

/* ---------- Pills ---------- */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0;
}
.pill-green { background: #E6F6EE; color: #0E7C3A; }
.pill-amber { background: #FFF4E5; color: #B85C00; }
.pill-red   { background: #FCE8E8; color: #B5341B; }
.pill-blue  { background: #E7EEFF; color: #3B5BDB; }
.pill-gray  { background: #EEF1F6; color: #4F566B; }
.pill-purple{ background: #EFE9FF; color: #5A35DC; }

/* ---------- Brand bar ---------- */
.brand-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0 22px 0;
    border-bottom: 1px solid #E3E8EE;
    margin-bottom: 28px;
}
.brand-logo { display: flex; align-items: center; gap: 14px; }
.brand-mark {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #635BFF, #3B5BDB);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 16px;
    box-shadow: 0 2px 5px rgba(99, 91, 255, 0.25);
}
.brand-name { color: #1A1F36; font-size: 19px; font-weight: 600; letter-spacing: -0.01em; }
.brand-tag  { color: #697386; font-size: 13px; }

/* ---------- Decision cards ---------- */
.decision-card {
    background: #FFFFFF;
    border: 1px solid #E3E8EE;
    border-left: 3px solid #635BFF;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 14px;
    box-shadow: 0 1px 2px rgba(60, 66, 87, 0.04);
}
.decision-card.escalate    { border-left-color: #B5341B; }
.decision-card.investigate { border-left-color: #B85C00; }
.decision-card.clear       { border-left-color: #0E7C3A; }

.decision-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px;
}
.decision-id {
    font-family: 'JetBrains Mono', 'Menlo', monospace;
    color: #1A1F36; font-size: 14px; font-weight: 600;
}
.decision-conf { color: #697386; font-size: 13px; }

.decision-meta {
    font-size: 13px; color: #697386;
    font-family: 'JetBrains Mono', 'Menlo', monospace;
    margin-bottom: 10px;
}
.decision-rationale {
    color: #3C4257; font-size: 14px; line-height: 1.6;
    margin: 12px 0;
}
.decision-policies { margin-top: 12px; display: flex; gap: 6px; flex-wrap: wrap; }
.policy-chip {
    background: #EEF1F6;
    border: 1px solid #E3E8EE;
    color: #3B5BDB;
    padding: 2px 9px;
    border-radius: 4px;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Menlo', monospace;
    font-weight: 500;
}
.next-action {
    color: #697386; font-size: 13px;
    margin-top: 12px; padding-top: 12px;
    border-top: 1px solid #F0F3F8;
}
.next-action b { color: #1A1F36; }

/* ---------- Buttons ---------- */
.stButton > button {
    background: #635BFF;
    color: white !important;
    border: none;
    border-radius: 6px;
    padding: 9px 18px;
    font-weight: 500;
    font-size: 14px;
    transition: all 0.15s;
    box-shadow: 0 1px 2px rgba(99, 91, 255, 0.25);
}
.stButton > button:hover {
    background: #5147E5;
    transform: translateY(-1px);
    box-shadow: 0 3px 8px rgba(99, 91, 255, 0.3);
}
.stButton > button:disabled {
    background: #C7CCD6 !important;
    box-shadow: none;
}

/* ---------- Tabs ---------- */
div[data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #E3E8EE;
    margin-bottom: 24px;
}
button[data-baseweb="tab"] {
    color: #697386 !important;
    font-weight: 500;
    font-size: 14px;
    padding: 10px 16px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #635BFF !important;
    border-bottom-color: #635BFF !important;
}

/* ---------- Tables ---------- */
div[data-testid="stDataFrame"] {
    border: 1px solid #E3E8EE;
    border-radius: 8px;
    overflow: hidden;
    background: #FFFFFF;
}

/* ---------- Sliders ---------- */
.stSlider [data-baseweb="slider"] > div > div { background: #635BFF !important; }

/* ---------- Expander ---------- */
.streamlit-expanderHeader { background: #FFFFFF !important; border: 1px solid #E3E8EE !important; border-radius: 8px !important; }

/* ---------- Footer ---------- */
.app-footer {
    margin-top: 56px;
    padding-top: 18px;
    border-top: 1px solid #E3E8EE;
    font-size: 12px;
    color: #697386;
    display: flex;
    justify-content: space-between;
}

/* Section key/value lines in cards */
.kv { display: flex; justify-content: space-between; font-size: 13px; padding: 6px 0; border-bottom: 1px solid #F0F3F8; }
.kv:last-child { border-bottom: none; }
.kv .k { color: #697386; }
.kv .v { color: #1A1F36; font-weight: 500; font-family: 'JetBrains Mono', monospace; }

/* Status row */
.status-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #F0F3F8; font-size: 14px; }
.status-row:last-child { border-bottom: none; }
.status-row .label { color: #3C4257; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Brand bar
# ---------------------------------------------------------------------------
def _get_api_key() -> str | None:
    """Read the Gemini key from Streamlit secrets first, then env var."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, st.errors.StreamlitAPIException):
        pass
    return os.environ.get("GEMINI_API_KEY")


api_key = _get_api_key()
api_key_present = bool(api_key)
key_pill = (
    '<span class="pill pill-green">Connected</span>'
    if api_key_present
    else '<span class="pill pill-red">API key missing</span>'
)

st.markdown(
    f"""
    <div class="brand-bar">
        <div class="brand-logo">
            <div class="brand-mark">S</div>
            <div>
                <div class="brand-name">Sentinel</div>
                <div class="brand-tag">Agentic payment integrity console</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <span class="pill pill-purple">{ENVIRONMENT}</span>
            {key_pill}
            <span class="pill pill-gray">v{BUILD_VERSION}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 16px 0;">
            <div style="font-size:13px; color:#697386; font-weight:500;
                 text-transform:uppercase; letter-spacing:0.05em;">Workspace</div>
            <div style="font-size:15px; color:#1A1F36; font-weight:600; margin-top:2px;">
                Payment Integrity
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Run configuration")
    n_to_reason = st.slider(
        "Claims per agent run",
        min_value=1,
        max_value=4,
        value=4,
        help="Capped to stay within Gemini free-tier 5 requests/min.",
    )

    st.markdown("### Engine")
    st.markdown(
        f"""
        <div class="kv"><span class="k">Detector</span><span class="v">IsolationForest</span></div>
        <div class="kv"><span class="k">Reasoner</span><span class="v">{MODEL}</span></div>
        <div class="kv"><span class="k">Backend</span><span class="v">Google GenAI</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Pipeline")
    st.markdown(
        """
        <div style="font-size:13px; color:#3C4257; line-height:1.85;">
        <span style="color:#635BFF; font-weight:600;">1.</span>&nbsp; Ingest synthetic claims<br>
        <span style="color:#635BFF; font-weight:600;">2.</span>&nbsp; Score with IsolationForest<br>
        <span style="color:#635BFF; font-weight:600;">3.</span>&nbsp; Route flagged to agent<br>
        <span style="color:#635BFF; font-weight:600;">4.</span>&nbsp; Emit structured decision<br>
        <span style="color:#635BFF; font-weight:600;">5.</span>&nbsp; Hand off to analyst or SIU
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:12px; color:#697386; line-height:1.6;">
        <div style="color:#1A1F36; font-weight:600;">Dhiraj Salunkhe</div>
        University of Pennsylvania<br>
        Cotiviti Intern Assessment
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_claims() -> pd.DataFrame | None:
    p = DATA_DIR / "claims.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame | None:
    p = DATA_DIR / "scored_claims.csv"
    return pd.read_csv(p) if p.exists() else None


claims_df = load_claims()
scored = load_scored()

if claims_df is None or scored is None:
    with st.spinner("First-time setup: generating synthetic claims and scoring..."):
        from src import generate_data, detect_anomalies
        generate_data.main()
        detect_anomalies.main()
    load_claims.clear()
    load_scored.clear()
    claims_df = load_claims()
    scored = load_scored()

flagged = scored[scored["is_flagged"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_detector, tab_agent, tab_policy = st.tabs(
    ["Overview", "Detector", "Reasoning Agent", "Policy Library"]
)


# ============================================================
# Tab 1: Overview
# ============================================================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims ingested", f"{len(claims_df):,}")
    c2.metric("Flagged by detector", int(scored["is_flagged"].sum()))
    c3.metric(
        "Detector recall",
        f"{int(scored.loc[scored['is_flagged'], 'is_planted_anomaly'].sum())}"
        f" / {int(scored['is_planted_anomaly'].sum())}",
    )
    c4.metric("Active policies", "6")

    st.markdown("&nbsp;")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Pipeline status</div>
                <div class="card-sub">Last refresh: just now</div>
                <div class="status-row">
                    <span class="label">Synthetic claim ingestion</span>
                    <span class="pill pill-green">Operational</span>
                </div>
                <div class="status-row">
                    <span class="label">Feature engineering (7 features)</span>
                    <span class="pill pill-green">Operational</span>
                </div>
                <div class="status-row">
                    <span class="label">IsolationForest scoring (n_estimators=300)</span>
                    <span class="pill pill-green">Operational</span>
                </div>
                <div class="status-row">
                    <span class="label">Gemini reasoning agent</span>
                    <span class="pill pill-blue">Ready</span>
                </div>
                <div class="status-row">
                    <span class="label">Analyst handoff queue</span>
                    <span class="pill pill-gray">Idle</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Build</div>
                <div class="card-sub">Deployment metadata</div>
                <div class="kv"><span class="k">Version</span><span class="v">v{BUILD_VERSION}</span></div>
                <div class="kv"><span class="k">Env</span><span class="v">{ENVIRONMENT}</span></div>
                <div class="kv"><span class="k">Model</span><span class="v">{MODEL}</span></div>
                <div class="kv"><span class="k">Timestamp</span><span class="v">{datetime.now().strftime("%Y-%m-%d %H:%M")}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    g1, g2 = st.columns([1, 1])

    with g1:
        st.markdown('<div class="card-title" style="margin-bottom:8px;">Anomaly score distribution</div>'
                    '<div class="card-sub">Higher scores indicate stronger anomaly signal</div>',
                    unsafe_allow_html=True)
        hist = (
            alt.Chart(scored)
            .mark_bar(color=COLOR_PRIMARY, opacity=0.85)
            .encode(
                x=alt.X("anomaly_score:Q", bin=alt.Bin(maxbins=30), title="Anomaly score"),
                y=alt.Y("count():Q", title="Claims"),
                tooltip=[alt.Tooltip("count():Q", title="Claims")],
            )
            .properties(height=240)
        )
        st.altair_chart(hist, use_container_width=True)

    with g2:
        st.markdown('<div class="card-title" style="margin-bottom:8px;">Claim volume by CPT code</div>'
                    '<div class="card-sub">Distribution across procedure types</div>',
                    unsafe_allow_html=True)
        cpt_counts = (
            claims_df.groupby("cpt_code")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        bar = (
            alt.Chart(cpt_counts)
            .mark_bar(color=COLOR_PRIMARY, opacity=0.85, cornerRadiusEnd=2)
            .encode(
                x=alt.X("count:Q", title="Claims"),
                y=alt.Y("cpt_code:N", sort="-x", title="CPT"),
                tooltip=["cpt_code", "count"],
            )
            .properties(height=240)
        )
        st.altair_chart(bar, use_container_width=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-title">About this workspace</div>
            <div class="card-sub">Proof of concept for Cotiviti</div>
            <div style="font-size:14px; line-height:1.65; color:#3C4257;">
                Two-stage payment integrity. An unsupervised detector scores
                every claim. Flagged claims are routed to an LLM agent that
                grounds its decision in a policy document and returns
                structured output for analyst review or SIU referral. The
                pattern is "ML flags, agent triages, analyst decides", which
                matches Cotiviti's published stance on AI in payment integrity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Tab 2: Detector
# ============================================================
with tab_detector:
    m1, m2, m3 = st.columns(3)
    m1.metric("Total scored", f"{len(scored):,}")
    m2.metric("Flagged", int(scored["is_flagged"].sum()))
    m3.metric(
        "Planted anomalies caught",
        int(scored.loc[scored["is_flagged"], "is_planted_anomaly"].sum()),
    )

    d1, d2 = st.columns([3, 2])

    with d1:
        st.markdown('<div class="card-title" style="margin-bottom:8px;">Anomaly score, flagged vs normal</div>'
                    '<div class="card-sub">Threshold separates the flagged tail from the bulk</div>',
                    unsafe_allow_html=True)
        scored_for_chart = scored.copy()
        scored_for_chart["status"] = scored_for_chart["is_flagged"].map(
            {True: "Flagged", False: "Normal"}
        )
        layered = (
            alt.Chart(scored_for_chart)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("anomaly_score:Q", bin=alt.Bin(maxbins=30), title="Anomaly score"),
                y=alt.Y("count():Q", title="Claims"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["Normal", "Flagged"],
                        range=[COLOR_GRAY, COLOR_RED],
                    ),
                    legend=alt.Legend(title=None, orient="top-right"),
                ),
                tooltip=["status", alt.Tooltip("count():Q", title="Claims")],
            )
            .properties(height=260)
        )
        st.altair_chart(layered, use_container_width=True)

    with d2:
        st.markdown('<div class="card-title" style="margin-bottom:8px;">Top providers by flagged claims</div>'
                    '<div class="card-sub">Concentration is a fraud risk signal</div>',
                    unsafe_allow_html=True)
        prov = (
            flagged.groupby("provider_id")
            .size()
            .reset_index(name="flagged_count")
            .sort_values("flagged_count", ascending=False)
            .head(8)
        )
        prov_chart = (
            alt.Chart(prov)
            .mark_bar(color=COLOR_PRIMARY, opacity=0.9, cornerRadiusEnd=2)
            .encode(
                x=alt.X("flagged_count:Q", title="Flagged claims"),
                y=alt.Y("provider_id:N", sort="-x", title=None),
                tooltip=["provider_id", "flagged_count"],
            )
            .properties(height=260)
        )
        st.altair_chart(prov_chart, use_container_width=True)

    st.markdown("### Top 10 flagged claims by anomaly score")
    st.dataframe(
        flagged.head(10)[
            [
                "claim_id", "provider_id", "cpt_code", "cpt_description",
                "units", "billed_amount", "patient_age", "anomaly_score",
                "is_planted_anomaly", "planted_reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Full claims feed (first 50 rows)"):
        st.dataframe(claims_df.head(50), use_container_width=True, hide_index=True)


# ============================================================
# Tab 3: Reasoning Agent
# ============================================================
with tab_agent:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Agent run</div>
            <div class="card-sub">
                Sends the top {n_to_reason} flagged claims to {MODEL}, grounded in the
                payment integrity policy library, and stores structured decisions below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "decisions" not in st.session_state:
        st.session_state.decisions = []

    cta_col, info_col = st.columns([1, 3])
    with cta_col:
        run = st.button(
            "Run agent",
            type="primary",
            disabled=not api_key_present,
            use_container_width=True,
        )
    with info_col:
        st.markdown(
            f"""
            <div style="font-size:13px; color:#697386; padding-top:10px;">
            Agent calls are paced at {RATE_LIMIT_SLEEP_SEC}s intervals to respect the
            free-tier rate limit. A full run takes about {n_to_reason * RATE_LIMIT_SLEEP_SEC}s.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run:
        client = build_client(api_key)
        policies = load_policies()
        st.session_state.decisions = []

        progress = st.progress(0.0, text="Initialising agent...")
        for i, row in flagged.head(n_to_reason).iterrows():
            block = claim_to_context(row)
            try:
                decision = reason_about_claim(client, block, policies)
            except Exception as e:  # noqa: BLE001
                decision = {
                    "decision": "investigate",
                    "confidence": 0.0,
                    "rationale": f"Agent error: {e}",
                    "policy_refs": [],
                    "suggested_next_action": "manual review",
                }
            st.session_state.decisions.append(
                {"claim": row.to_dict(), "decision": decision}
            )
            progress.progress(
                (i + 1) / n_to_reason,
                text=f"Reasoning over claim {i + 1} of {n_to_reason}...",
            )
            if i + 1 < n_to_reason:
                time.sleep(RATE_LIMIT_SLEEP_SEC)
        progress.empty()

    if not st.session_state.decisions:
        st.markdown(
            """
            <div class="card" style="text-align:center; padding:56px 24px;">
                <div style="font-size:36px; color:#C7CCD6;">◆</div>
                <div style="color:#697386; font-size:14px; margin-top:14px;">
                    No agent runs yet. Click
                    <b style="color:#1A1F36;">Run agent</b>
                    to score the top flagged claims.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        counts = {"clear": 0, "investigate": 0, "escalate": 0}
        for item in st.session_state.decisions:
            d = item["decision"].get("decision", "")
            if d in counts:
                counts[d] += 1
        s1, s2, s3 = st.columns(3)
        s1.metric("Clear", counts["clear"])
        s2.metric("Investigate", counts["investigate"])
        s3.metric("Escalate", counts["escalate"])

        ch1, ch2 = st.columns([1, 1])

        with ch1:
            st.markdown('<div class="card-title" style="margin:14px 0 4px 0;">Decision mix</div>'
                        '<div class="card-sub">Triage routing summary</div>',
                        unsafe_allow_html=True)
            mix_df = pd.DataFrame(
                [{"decision": k.capitalize(), "count": v} for k, v in counts.items() if v > 0]
            )
            if len(mix_df) > 0:
                donut = (
                    alt.Chart(mix_df)
                    .mark_arc(innerRadius=60, stroke="#FFFFFF", strokeWidth=2)
                    .encode(
                        theta=alt.Theta("count:Q"),
                        color=alt.Color(
                            "decision:N",
                            scale=alt.Scale(
                                domain=["Clear", "Investigate", "Escalate"],
                                range=[COLOR_GREEN, COLOR_AMBER, COLOR_RED],
                            ),
                            legend=alt.Legend(title=None, orient="right"),
                        ),
                        tooltip=["decision", "count"],
                    )
                    .properties(height=240)
                )
                st.altair_chart(donut, use_container_width=True)

        with ch2:
            st.markdown('<div class="card-title" style="margin:14px 0 4px 0;">Confidence by claim</div>'
                        '<div class="card-sub">Agent self-reported certainty per decision</div>',
                        unsafe_allow_html=True)
            conf_rows = []
            for item in st.session_state.decisions:
                try:
                    conf_rows.append({
                        "claim_id": item["claim"]["claim_id"],
                        "confidence": float(item["decision"].get("confidence", 0)) * 100,
                        "decision": item["decision"].get("decision", "unknown").capitalize(),
                    })
                except (TypeError, ValueError):
                    pass
            if conf_rows:
                conf_df = pd.DataFrame(conf_rows)
                bars = (
                    alt.Chart(conf_df)
                    .mark_bar(opacity=0.9, cornerRadiusEnd=2)
                    .encode(
                        x=alt.X("confidence:Q", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("claim_id:N", sort="-x", title=None),
                        color=alt.Color(
                            "decision:N",
                            scale=alt.Scale(
                                domain=["Clear", "Investigate", "Escalate"],
                                range=[COLOR_GREEN, COLOR_AMBER, COLOR_RED],
                            ),
                            legend=None,
                        ),
                        tooltip=["claim_id", "decision", alt.Tooltip("confidence:Q", format=".0f")],
                    )
                    .properties(height=240)
                )
                st.altair_chart(bars, use_container_width=True)

        st.markdown("### Agent decisions")
        for item in st.session_state.decisions:
            c = item["claim"]
            d = item["decision"]
            decision_label = d.get("decision", "unknown")
            pill_class = {
                "clear": "pill-green",
                "investigate": "pill-amber",
                "escalate": "pill-red",
            }.get(decision_label, "pill-gray")
            policy_chips = "".join(
                f'<span class="policy-chip">{ref}</span>'
                for ref in d.get("policy_refs", []) or []
            )
            confidence = d.get("confidence", 0)
            try:
                confidence_str = f"{float(confidence) * 100:.0f}%"
            except (ValueError, TypeError):
                confidence_str = "n/a"

            st.markdown(
                f"""
                <div class="decision-card {decision_label}">
                    <div class="decision-header">
                        <div>
                            <span class="decision-id">{c['claim_id']}</span>
                            &nbsp;&nbsp;
                            <span class="pill {pill_class}">{decision_label.upper()}</span>
                        </div>
                        <div class="decision-conf">Confidence {confidence_str}</div>
                    </div>
                    <div class="decision-meta">
                        {c['cpt_code']} &middot; {c['cpt_description']} &middot;
                        {int(c['units'])} unit(s) &middot; ${c['billed_amount']:,.2f} &middot;
                        {c['provider_id']}
                    </div>
                    <div class="decision-rationale">{d.get('rationale', '')}</div>
                    <div class="decision-policies">{policy_chips}</div>
                    <div class="next-action"><b>Next action:</b> {d.get('suggested_next_action', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# Tab 4: Policy library
# ============================================================
with tab_policy:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Payment integrity policy library</div>
            <div class="card-sub">
                Synthetic policy clauses the agent grounds its decisions in.
                In a production deployment this would be backed by a retrieval
                layer over the real Cotiviti policy corpus.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    policy_text = load_policies()
    st.markdown(
        f'<div class="card"><pre style="background:transparent; color:#3C4257; '
        f'white-space:pre-wrap; font-size:14px; line-height:1.7; margin:0;">{policy_text}</pre></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-footer">
        <div>Sentinel v{BUILD_VERSION} &middot; {ENVIRONMENT}</div>
        <div>Synthetic data only. No PHI. Demo use only.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
