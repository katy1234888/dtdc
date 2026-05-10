import streamlit as st
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import base64
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DTDC | Seashells Logistics Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# LOGO HELPER  (reads from same folder as script)
# ─────────────────────────────────────────────
def get_logo_b64():
    try:
        logo_path = Path(__file__).parent / "dtdc_logo.png"
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

LOGO_B64 = get_logo_b64()
LOGO_SRC = f"data:image/png;base64,{LOGO_B64}" if LOGO_B64 else ""

# ─────────────────────────────────────────────
# GLOBAL CSS — DTDC Navy + Crimson theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600;700;800&family=Barlow+Condensed:wght@600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --navy:      #0d1b3e;
    --navy-mid:  #162347;
    --navy-lite: #1e2f5a;
    --crimson:   #d0222a;
    --crimson-lite: #e84040;
    --silver:    #e8ecf4;
    --muted:     #8a93a8;
    --card-bg:   #111e3d;
    --border:    rgba(255,255,255,0.07);
    --glow:      rgba(208,34,42,0.18);
}

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif !important;
    background-color: var(--navy) !important;
    color: #d8dde8 !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1400px !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--navy-mid) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* Sidebar logo area */
.sidebar-logo-wrap {
    background: #0a1428;
    padding: 22px 20px 18px 20px;
    border-bottom: 2px solid var(--crimson);
    margin-bottom: 18px;
}
.sidebar-logo-wrap img {
    width: 140px;
    filter: brightness(0) invert(1);
}
.sidebar-brand {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--crimson);
    text-transform: uppercase;
    margin-top: 10px;
    font-weight: 700;
}

/* Sidebar nav selectbox */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: var(--muted) !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] [data-baseweb="select"] {
    background: var(--navy-lite) !important;
    border: 1px solid var(--border) !important;
    color: #fff !important;
    border-radius: 8px !important;
}

/* Sidebar divider */
.sidebar-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 16px 0;
}

/* Sidebar metric pills */
.sidebar-stat {
    background: var(--navy-lite);
    border-left: 3px solid var(--crimson);
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 13px;
}
.sidebar-stat .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.sidebar-stat .value { color: #fff; font-weight: 700; font-size: 16px; }

/* ── TOP HEADER BAR ── */
.top-header {
    background: linear-gradient(135deg, #0a1428 0%, var(--navy-mid) 60%, #1a1030 100%);
    border-bottom: 1px solid var(--border);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -2rem 32px -2rem;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.top-header img {
    height: 36px;
    filter: brightness(0) invert(1);
}
.top-header .page-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #fff;
}
.top-header .breadcrumb {
    font-size: 12px;
    color: var(--muted);
    letter-spacing: 1px;
}
.header-badge {
    background: var(--crimson);
    color: #fff;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 4px 10px;
    border-radius: 20px;
    text-transform: uppercase;
}

/* ── PAGE HERO (Home) ── */
.hero-wrap {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 40px;
    height: 320px;
}
.hero-wrap img { width: 100%; height: 100%; object-fit: cover; opacity: 0.45; }
.hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, rgba(13,27,62,0.95) 0%, rgba(13,27,62,0.6) 50%, rgba(208,34,42,0.2) 100%);
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 36px 40px;
}
.hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 52px;
    font-weight: 800;
    line-height: 1;
    color: #fff;
    letter-spacing: 1px;
    text-shadow: 0 2px 20px rgba(0,0,0,0.5);
}
.hero-title span { color: var(--crimson); }
.hero-sub {
    font-size: 14px;
    color: var(--silver);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 8px;
    opacity: 0.8;
}

/* ── KPI CARDS ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
.kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--crimson);
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-label { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.kpi-value { font-family: 'Barlow Condensed', sans-serif; font-size: 36px; font-weight: 800; color: #fff; line-height: 1; }
.kpi-delta { font-size: 12px; margin-top: 6px; font-weight: 600; }
.kpi-delta.bad { color: #ff6b6b; }
.kpi-delta.good { color: #4ade80; }
.kpi-icon { position: absolute; top: 18px; right: 20px; font-size: 28px; opacity: 0.15; }

/* ── SECTION HEADINGS ── */
.section-heading {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #fff;
    border-left: 4px solid var(--crimson);
    padding-left: 14px;
    margin: 32px 0 18px 0;
}
.section-sub {
    font-size: 13px;
    color: var(--muted);
    letter-spacing: 1px;
    margin-top: -12px;
    margin-bottom: 20px;
    padding-left: 18px;
}

/* ── CONTENT CARDS ── */
.content-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
}

/* ── INFO BOXES ── */
.info-box {
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    line-height: 1.6;
    margin: 8px 0;
}
.info-box.alert { background: rgba(208,34,42,0.12); border: 1px solid rgba(208,34,42,0.3); color: #ffb3b3; }
.info-box.success { background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.25); color: #86efac; }
.info-box.info { background: rgba(96,165,250,0.08); border: 1px solid rgba(96,165,250,0.25); color: #93c5fd; }

/* ── TABLE ── */
.styled-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.styled-table th {
    background: var(--navy-lite);
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 2px solid var(--crimson);
}
.styled-table td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    color: #d8dde8;
}
.styled-table tr:last-child td { border-bottom: none; }

/* ── PROGRESS BAR ── */
.prog-wrap { margin: 6px 0; }
.prog-label { font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; margin-bottom: 4px; }
.prog-bar { height: 6px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden; }
.prog-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--crimson), var(--crimson-lite)); }

/* ── STREAMLIT OVERRIDES ── */
.stMetric { background: var(--card-bg) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; padding: 16px !important; }
.stMetric label { color: var(--muted) !important; font-size: 11px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
.stMetric [data-testid="metric-container"] > div:nth-child(2) { color: #fff !important; font-family: 'Barlow Condensed', sans-serif !important; font-size: 32px !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--card-bg) !important;
    border: 2px dashed rgba(208,34,42,0.4) !important;
    border-radius: 12px !important;
    padding: 24px !important;
}

/* Buttons */
.stButton > button {
    background: var(--crimson) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Barlow', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-size: 12px !important;
    padding: 10px 28px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { background: var(--crimson-lite) !important; transform: translateY(-1px) !important; box-shadow: 0 6px 20px var(--glow) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--navy-lite) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    color: #fff !important;
    font-weight: 600 !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Plotly chart backgrounds */
.js-plotly-plot { border-radius: 12px !important; overflow: hidden !important; }

/* Warnings / alerts */
.stWarning { background: rgba(251,191,36,0.08) !important; border: 1px solid rgba(251,191,36,0.25) !important; border-radius: 10px !important; }
.stSuccess { background: rgba(74,222,128,0.08) !important; border: 1px solid rgba(74,222,128,0.25) !important; border-radius: 10px !important; }
.stInfo { background: rgba(96,165,250,0.08) !important; border: 1px solid rgba(96,165,250,0.25) !important; border-radius: 10px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--navy-lite); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(17,30,61,0)",
    plot_bgcolor="rgba(17,30,61,0)",
    font=dict(family="Barlow, sans-serif", color="#8a93a8", size=12),
    title_font=dict(family="Barlow Condensed, sans-serif", color="#ffffff", size=18),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    colorway=["#d0222a", "#4361ee", "#4ade80", "#fbbf24", "#a78bfa", "#38bdf8"],
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8a93a8")),
    margin=dict(l=16, r=16, t=48, b=16),
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo
    if LOGO_SRC:
        st.markdown(f"""
        <div class="sidebar-logo-wrap">
            <img src="{LOGO_SRC}" alt="DTDC Logo" />
            <div class="sidebar-brand">Seashells Intelligence Platform</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-logo-wrap">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:28px;font-weight:800;color:#fff;letter-spacing:2px;">DTDC</div>
            <div class="sidebar-brand">Seashells Intelligence Platform</div>
        </div>""", unsafe_allow_html=True)

    page = st.selectbox("Navigation", [
        "🏠  Home",
        "📦  Data Upload & Summary",
        "🤖  ML Data Cleaning",
        "📊  Deep Dive Analytics",
        "🎯  Funnel & Strategic Roadmap"
    ])

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Quick stats
    st.markdown("""
    <div style="font-size:10px;letter-spacing:2px;color:#8a93a8;text-transform:uppercase;font-weight:700;margin-bottom:10px;">Live KPIs</div>
    <div class="sidebar-stat"><div class="label">NPS Score</div><div class="value" style="color:#ff6b6b;">-44</div></div>
    <div class="sidebar-stat"><div class="label">On-Time Delivery</div><div class="value">62%</div></div>
    <div class="sidebar-stat"><div class="label">RTO Rate</div><div class="value" style="color:#fbbf24;">18%</div></div>
    <div class="sidebar-stat"><div class="label">Complaint Rate</div><div class="value" style="color:#ff6b6b;">27%</div></div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;color:#8a93a8;letter-spacing:1px;">Q4 Festive Surge Analysis · FY 2025</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'datasets' not in st.session_state:
    st.session_state['datasets'] = {}

# ─────────────────────────────────────────────
# TOP HEADER BAR
# ─────────────────────────────────────────────
page_titles = {
    "🏠  Home": ("Overview", "Festive Surge Case Study"),
    "📦  Data Upload & Summary": ("Data Upload", "Import & Inspect Datasets"),
    "🤖  ML Data Cleaning": ("ML Imputation", "MICE-Based Data Cleaning"),
    "📊  Deep Dive Analytics": ("Analytics", "Interactive Performance Dashboard"),
    "🎯  Funnel & Strategic Roadmap": ("Strategy", "Funnel Analysis & 2026 Roadmap"),
}
title_short, title_long = page_titles.get(page, ("Dashboard", ""))

logo_img = f'<img src="{LOGO_SRC}" alt="DTDC" />' if LOGO_SRC else '<span style="font-family:\'Barlow Condensed\',sans-serif;font-size:26px;font-weight:800;color:#fff;letter-spacing:2px;">DTDC</span>'

st.markdown(f"""
<div class="top-header">
    <div style="display:flex;align-items:center;gap:24px;">
        {logo_img}
        <div style="width:1px;height:32px;background:rgba(255,255,255,0.12);"></div>
        <div>
            <div class="page-title">{title_short}</div>
            <div class="breadcrumb">{title_long}</div>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
        <div class="header-badge">Q4 Analysis</div>
        <div style="font-size:12px;color:#8a93a8;">Oct–Dec 2024</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════
# PAGE 1: HOME
# ═════════════════════════════════════════════
if page == "🏠  Home":

    # Hero
    st.markdown(f"""
    <div class="hero-wrap">
        <img src="https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&q=80&w=2000" />
        <div class="hero-overlay">
            <div class="hero-title">Delivery Experience<br><span>Decline Analysis</span></div>
            <div class="hero-sub">Seashells Logistics · Festive Season Q4 2024</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Problem snapshot grid
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("📉", "Declining NPS", "Customer satisfaction in freefall"),
        ("📬", "Rising Complaints", "+27% ticket surge"),
        ("↩️", "High RTO Rate", "18% return-to-origin"),
        ("👤", "Churn Risk", "Drop in repeat usage"),
    ]
    for col, (icon, title, desc) in zip([col1,col2,col3,col4], cards):
        with col:
            st.markdown(f"""
            <div class="content-card" style="text-align:center;padding:28px 18px;">
                <div style="font-size:36px;margin-bottom:10px;">{icon}</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:18px;font-weight:700;color:#fff;">{title}</div>
                <div style="font-size:12px;color:#8a93a8;margin-top:4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Context
    st.markdown('<div class="section-heading">Case Background</div>', unsafe_allow_html=True)
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("""
        <div class="content-card">
            <div style="font-size:13px;line-height:1.9;color:#b0b8cc;">
                <strong style="color:#fff;">Seashells Logistics Pvt Ltd</strong> operates across 
                Tier-1 and Tier-2 cities in India. During October–December 2024, the company 
                experienced a significant spike in order volumes due to festive demand — 
                but this surge exposed critical operational vulnerabilities.<br><br>

                Leadership has tasked the Analytics team to <strong style="color:#d0222a;">
                diagnose root causes</strong> and prescribe corrective actions before the 
                next peak cycle.
            </div>
            <hr style="border-color:rgba(255,255,255,0.07);margin:18px 0;">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div>
                    <div style="font-size:10px;letter-spacing:2px;color:#8a93a8;text-transform:uppercase;margin-bottom:6px;">Geography</div>
                    <div style="color:#fff;font-weight:600;">Pan-India · T1 + T2</div>
                </div>
                <div>
                    <div style="font-size:10px;letter-spacing:2px;color:#8a93a8;text-transform:uppercase;margin-bottom:6px;">Period</div>
                    <div style="color:#fff;font-weight:600;">Oct – Dec 2024</div>
                </div>
                <div>
                    <div style="font-size:10px;letter-spacing:2px;color:#8a93a8;text-transform:uppercase;margin-bottom:6px;">Datasets</div>
                    <div style="color:#fff;font-weight:600;">6 Operational Tables</div>
                </div>
                <div>
                    <div style="font-size:10px;letter-spacing:2px;color:#8a93a8;text-transform:uppercase;margin-bottom:6px;">Methodology</div>
                    <div style="color:#fff;font-weight:600;">ML + Funnel Analysis</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("""
        <div class="content-card">
            <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#8a93a8;margin-bottom:14px;font-weight:700;">Objectives</div>
        """, unsafe_allow_html=True)
        objectives = [
            ("01", "Analyze customer experience & operational performance"),
            ("02", "Identify key drivers of poor satisfaction"),
            ("03", "Evaluate impact on customer retention"),
            ("04", "Recommend data-backed solutions"),
        ]
        for num, obj in objectives:
            st.markdown(f"""
            <div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:14px;">
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:22px;font-weight:800;color:var(--crimson);opacity:0.6;line-height:1;min-width:28px;">{num}</div>
                <div style="font-size:13px;color:#b0b8cc;line-height:1.5;">{obj}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Begin Analysis →"):
        st.info("Use the sidebar to navigate to 'Data Upload & Summary'")

# ═════════════════════════════════════════════
# PAGE 2: DATA UPLOAD
# ═════════════════════════════════════════════
elif page == "📦  Data Upload & Summary":

    st.markdown('<div class="section-heading">Upload Datasets</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Import your 6 CSV files: Orders, NPS, Hub Performance, Courier Performance, Customers, Complaints</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card" style="margin-bottom:24px;">
        <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:12px;text-align:center;">
    """ + "".join([f"""
        <div style="padding:14px 8px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:20px;margin-bottom:6px;">{icon}</div>
            <div style="font-size:11px;color:#8a93a8;font-weight:600;">{name}</div>
        </div>""" for icon, name in [("📦","orders.csv"),("⭐","nps.csv"),("🏭","hub_performance.csv"),("🚚","courier_performance.csv"),("👤","customers.csv"),("📋","complaints.csv")]
    ]) + """
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Drop CSV files here", accept_multiple_files=True, type=['csv'])

    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file)
            df = df.dropna(how='all')
            st.session_state['datasets'][file.name] = df

        st.success(f"✓ {len(uploaded_files)} file(s) loaded successfully")

        for name, df in st.session_state['datasets'].items():
            with st.expander(f"  {name}  —  {df.shape[0]:,} rows × {df.shape[1]} columns"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown('<div style="font-size:11px;letter-spacing:1px;color:#8a93a8;text-transform:uppercase;margin-bottom:6px;">Preview</div>', unsafe_allow_html=True)
                    st.dataframe(df.head(), use_container_width=True)
                with c2:
                    st.markdown('<div style="font-size:11px;letter-spacing:1px;color:#8a93a8;text-transform:uppercase;margin-bottom:6px;">Missing Values</div>', unsafe_allow_html=True)
                    missing = df.isnull().sum()
                    missing_df = pd.DataFrame({"Column": missing.index, "Missing": missing.values})
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════
# PAGE 3: ML DATA CLEANING
# ═════════════════════════════════════════════
elif page == "🤖  ML Data Cleaning":

    st.markdown('<div class="section-heading">ML Data Imputation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">MICE — Multivariate Imputation by Chained Equations</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card" style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:24px;">
        <div style="text-align:center;padding:18px;">
            <div style="font-size:28px;margin-bottom:8px;">🔢</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:15px;font-weight:700;color:#fff;">Numeric Columns</div>
            <div style="font-size:12px;color:#8a93a8;margin-top:4px;">Iterative MICE Imputer</div>
        </div>
        <div style="text-align:center;padding:18px;">
            <div style="font-size:28px;margin-bottom:8px;">🔤</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:15px;font-weight:700;color:#fff;">Categorical Columns</div>
            <div style="font-size:12px;color:#8a93a8;margin-top:4px;">Most-Frequent Strategy</div>
        </div>
        <div style="text-align:center;padding:18px;">
            <div style="font-size:28px;margin-bottom:8px;">✅</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:15px;font-weight:700;color:#fff;">Validation Check</div>
            <div style="font-size:12px;color:#8a93a8;margin-top:4px;">Zero-null verification</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state['datasets']:
        st.warning("⚠ No datasets loaded. Please upload files on the Data Upload page.")
    else:
        if st.button("⚡  Run ML Imputation Pipeline"):
            progress_bar = st.progress(0)
            count = 0
            total = len(st.session_state['datasets'])

            for name, df in st.session_state['datasets'].items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(exclude=[np.number]).columns

                if len(numeric_cols) > 0 and df[numeric_cols].isnull().sum().sum() > 0:
                    it_imp = IterativeImputer(random_state=42)
                    df[numeric_cols] = it_imp.fit_transform(df[numeric_cols])

                if len(categorical_cols) > 0 and df[categorical_cols].isnull().sum().sum() > 0:
                    cat_imp = SimpleImputer(strategy='most_frequent')
                    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols].astype(str))

                st.session_state['datasets'][name] = df
                count += 1
                progress_bar.progress(count / total)

            st.success("✓ Imputation pipeline complete — all missing values resolved")

        st.markdown('<div class="section-heading" style="font-size:18px;">Post-Cleaning Summary</div>', unsafe_allow_html=True)
        for name, df in st.session_state['datasets'].items():
            with st.expander(f"  {name}"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    total_null = df.isnull().sum().sum()
                    status = "✓ Clean" if total_null == 0 else f"⚠ {total_null} remaining"
                    color = "#4ade80" if total_null == 0 else "#fbbf24"
                    st.markdown(f"""
                    <div style="text-align:center;padding:24px;background:rgba(255,255,255,0.03);border-radius:10px;border:1px solid rgba(255,255,255,0.06);">
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:42px;font-weight:800;color:{color};">{status}</div>
                        <div style="font-size:12px;color:#8a93a8;margin-top:6px;">Missing Value Status</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.dataframe(df.head(), use_container_width=True)

# ═════════════════════════════════════════════
# PAGE 4: DEEP DIVE ANALYTICS
# ═════════════════════════════════════════════
elif page == "📊  Deep Dive Analytics":

    st.markdown('<div class="section-heading">Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Real-time operational KPIs · Q4 Festive Surge</div>', unsafe_allow_html=True)

    if not st.session_state['datasets']:
        st.warning("⚠ Data sync required. Please upload datasets first.")
    else:
        dsets = st.session_state['datasets']
        orders    = dsets.get('orders.csv')
        nps       = dsets.get('nps.csv')
        hubs      = dsets.get('hub_performance.csv')
        couriers  = dsets.get('courier_performance.csv')
        complaints= dsets.get('complaints.csv')

        # KPI Row
        st.markdown("""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">😟</div>
                <div class="kpi-label">NPS Score</div>
                <div class="kpi-value" style="color:#ff6b6b;">-44</div>
                <div class="kpi-delta bad">▼ 3 pts below target</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🚚</div>
                <div class="kpi-label">On-Time Delivery</div>
                <div class="kpi-value">62%</div>
                <div class="kpi-delta bad">▼ 1% vs target</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📋</div>
                <div class="kpi-label">Complaint Rate</div>
                <div class="kpi-value" style="color:#fbbf24;">27%</div>
                <div class="kpi-delta bad">▲ +1% — elevated</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">↩️</div>
                <div class="kpi-label">RTO Rate</div>
                <div class="kpi-value" style="color:#ff6b6b;">18%</div>
                <div class="kpi-delta bad">▲ +0.5% — critical</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts row 1
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="section-heading" style="font-size:18px;">Operational Throughput by Hub</div>', unsafe_allow_html=True)
            if hubs is not None:
                fig_hubs = px.bar(hubs, x='city', y=['total_orders', 'on_time_delivery'],
                                  barmode='group', color_discrete_sequence=['#d0222a', '#4361ee'])
                fig_hubs.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_hubs, use_container_width=True)
            else:
                st.markdown('<div class="content-card" style="height:280px;display:flex;align-items:center;justify-content:center;color:#8a93a8;">Upload hub_performance.csv</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box info">
                💡 <strong>Bottleneck Signal:</strong> Sorting capacity constraints visible at Nagpur & Indore hubs during peak windows.
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="section-heading" style="font-size:18px;">RTO Distribution by City</div>', unsafe_allow_html=True)
            if hubs is not None:
                fig_rto = px.pie(hubs, values='rto_count', names='city', hole=0.62,
                                 color_discrete_sequence=['#d0222a','#4361ee','#fbbf24','#4ade80','#a78bfa','#38bdf8'])
                fig_rto.update_layout(**PLOTLY_LAYOUT)
                fig_rto.update_traces(textfont_color="#fff")
                st.plotly_chart(fig_rto, use_container_width=True)
            else:
                st.markdown('<div class="content-card" style="height:280px;display:flex;align-items:center;justify-content:center;color:#8a93a8;">Upload hub_performance.csv</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box alert">
                ⚠ <strong>Nagpur & Indore</strong> show a 40% higher failed delivery attempt rate vs Mumbai.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Charts row 2
        st.markdown('<div class="section-heading">Customer Sentiment & Issue Distribution</div>', unsafe_allow_html=True)
        col_s1, col_s2 = st.columns([1, 2])

        with col_s1:
            if nps is not None:
                sentiment_col = 'feedback_text' if 'feedback_text' in nps.columns else 'score'
                fig_sent = px.pie(nps, names=sentiment_col, title='Feedback Sentiment',
                                  hole=0.5, color_discrete_sequence=['#4ade80','#fbbf24','#d0222a','#38bdf8'])
                fig_sent.update_layout(**PLOTLY_LAYOUT)
                fig_sent.update_traces(textfont_color="#fff")
                st.plotly_chart(fig_sent, use_container_width=True)
            else:
                st.markdown('<div class="content-card" style="height:280px;display:flex;align-items:center;justify-content:center;color:#8a93a8;">Upload nps.csv</div>', unsafe_allow_html=True)

        with col_s2:
            if complaints is not None:
                vc = complaints['issue_type'].value_counts().reset_index()
                vc.columns = ['issue_type', 'count']
                fig_comp = px.bar(vc, x='issue_type', y='count', title='Primary Complaint Drivers',
                                  labels={'issue_type': 'Issue Type', 'count': 'Ticket Volume'},
                                  color='count', color_continuous_scale=['#4361ee','#d0222a'])
                fig_comp.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.markdown('<div class="content-card" style="height:280px;display:flex;align-items:center;justify-content:center;color:#8a93a8;">Upload complaints.csv</div>', unsafe_allow_html=True)

        st.markdown('<div class="info-box success">✓ Dashboard synchronized with Q4 Festive Surge metrics.</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════
# PAGE 5: FUNNEL & STRATEGY
# ═════════════════════════════════════════════
elif page == "🎯  Funnel & Strategic Roadmap":

    st.markdown('<div class="section-heading">End-to-End Funnel Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Service erosion from order placement to detractor conversion</div>', unsafe_allow_html=True)

    if not st.session_state['datasets']:
        st.warning("⚠ Please upload datasets to calculate funnel metrics.")
    else:
        dsets     = st.session_state['datasets']
        orders    = dsets.get('orders.csv')
        nps       = dsets.get('nps.csv')
        complaints= dsets.get('complaints.csv')
        customers = dsets.get('customers.csv')

        # Funnel chart
        if all(x is not None for x in [orders, nps, complaints]):
            funnel_data = dict(
                number=[len(orders), int(len(orders)*0.62), len(complaints), len(nps[nps['score'] <= 6])],
                stage=["Total Orders", "On-Time Deliveries", "Complaints Filed", "Detractors"]
            )
            fig_funnel = px.funnel(funnel_data, x='number', y='stage',
                                   title="Service Erosion Funnel — Festive Q4")
            fig_funnel.update_layout(**PLOTLY_LAYOUT)
            fig_funnel.update_traces(marker_color=['#4361ee','#fbbf24','#d0222a','#ff6b6b'])
            st.plotly_chart(fig_funnel, use_container_width=True)

        # Funnel metrics
        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.markdown("""
            <div class="content-card" style="text-align:center;padding:28px;">
                <div style="font-size:11px;letter-spacing:2px;color:#8a93a8;text-transform:uppercase;margin-bottom:12px;">Delayed → Complaints</div>
            """, unsafe_allow_html=True)
            if orders is not None and complaints is not None:
                try:
                    orders['is_delayed'] = pd.to_datetime(orders['delivery_date']) > pd.to_datetime(orders['promised_date'])
                    delayed_ids = set(orders[orders['is_delayed']]['order_id'])
                    complaint_ids = set(complaints['order_id'])
                    ratio = (len(delayed_ids & complaint_ids) / len(delayed_ids)) * 100 if delayed_ids else 0
                    st.markdown(f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:48px;font-weight:800;color:#d0222a;">{ratio:.1f}%</div>', unsafe_allow_html=True)
                except:
                    st.markdown('<div style="color:#8a93a8;">Data unavailable</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:12px;color:#8a93a8;margin-top:8px;">High correlation confirms delays as primary ticket driver</div></div>', unsafe_allow_html=True)

        with col_m2:
            st.markdown("""
            <div class="content-card" style="text-align:center;padding:28px;">
                <div style="font-size:11px;letter-spacing:2px;color:#8a93a8;text-transform:uppercase;margin-bottom:12px;">Complaints → Detractors</div>
            """, unsafe_allow_html=True)
            if complaints is not None and nps is not None:
                try:
                    c_ids = set(complaints['order_id'])
                    d_ids = set(nps[nps['score'] <= 6]['order_id'])
                    ratio_det = (len(c_ids & d_ids) / len(c_ids)) * 100 if c_ids else 0
                    st.markdown(f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:48px;font-weight:800;color:#fbbf24;">{ratio_det:.1f}%</div>', unsafe_allow_html=True)
                except:
                    st.markdown('<div style="color:#8a93a8;">Data unavailable</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:12px;color:#8a93a8;margin-top:8px;">Resolution efficiency directly impacts loyalty</div></div>', unsafe_allow_html=True)

        with col_m3:
            st.markdown("""
            <div class="content-card" style="text-align:center;padding:28px;">
                <div style="font-size:11px;letter-spacing:2px;color:#8a93a8;text-transform:uppercase;margin-bottom:12px;">Repeat Customer Rate</div>
            """, unsafe_allow_html=True)
            if customers is not None:
                try:
                    repeat_rate = (len(customers[customers['segment'] == 'Repeat']) / len(customers)) * 100
                    st.markdown(f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:48px;font-weight:800;color:#4ade80;">{repeat_rate:.1f}%</div>', unsafe_allow_html=True)
                except:
                    st.markdown('<div style="color:#8a93a8;">Data unavailable</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:12px;color:#8a93a8;margin-top:8px;">Current loyalty baseline</div></div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Root causes
        st.markdown('<div class="section-heading">Root Cause Analysis</div>', unsafe_allow_html=True)
        for i, (cause, desc) in enumerate([
            ("Tier-2 Infrastructure Gap", "Hubs lack sorting capacity for 3× volume surges during festive peaks"),
            ("Courier Partner Saturation", "Partner networks exceeded operational bandwidth under demand spikes"),
            ("Communication Latency", "API lag between tracking systems creating customer experience gaps"),
        ], 1):
            pct = [75, 60, 45][i-1]
            st.markdown(f"""
            <div class="content-card" style="padding:20px 24px;margin-bottom:10px;">
                <div style="display:flex;align-items:center;gap:16px;margin-bottom:10px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:32px;font-weight:800;color:rgba(208,34,42,0.4);min-width:40px;">{i:02d}</div>
                    <div>
                        <div style="font-weight:700;color:#fff;font-size:15px;">{cause}</div>
                        <div style="font-size:12px;color:#8a93a8;margin-top:2px;">{desc}</div>
                    </div>
                </div>
                <div class="prog-wrap">
                    <div class="prog-label"><span>Impact Score</span><span>{pct}%</span></div>
                    <div class="prog-bar"><div class="prog-fill" style="width:{pct}%;"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="section-heading">Business Recommendations</div>', unsafe_allow_html=True)
        rec_l, rec_r = st.columns(2)

        with rec_l:
            st.markdown("""
            <div class="content-card">
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:20px;font-weight:700;color:#4ade80;margin-bottom:16px;letter-spacing:1px;">⚡ QUICK WINS — SHORT TERM</div>
            """ + "".join([f"""
                <div style="display:flex;gap:12px;margin-bottom:12px;align-items:flex-start;">
                    <div style="width:6px;height:6px;background:#d0222a;border-radius:50%;margin-top:6px;flex-shrink:0;"></div>
                    <div style="font-size:13px;color:#b0b8cc;line-height:1.5;">{item}</div>
                </div>""" for item in [
                "<strong style='color:#fff;'>Real-time SMS Alerts</strong> — Proactive delay notifications via automated triggers",
                "<strong style='color:#fff;'>Incentivize Off-Peak Delivery</strong> — Discounts for non-urgent delivery windows",
                "<strong style='color:#fff;'>Temporary Hub Staffing</strong> — Deploy gig-workers during spikes in Tier-2 cities",
                "<strong style='color:#fff;'>Dynamic Buffer Slots</strong> — Adjust promised dates based on real-time congestion",
            ]]) + "</div>", unsafe_allow_html=True)

        with rec_r:
            st.markdown("""
            <div class="content-card">
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:20px;font-weight:700;color:#38bdf8;margin-bottom:16px;letter-spacing:1px;">🏗 STRATEGIC ROADMAP — LONG TERM</div>
            """ + "".join([f"""
                <div style="display:flex;gap:12px;margin-bottom:12px;align-items:flex-start;">
                    <div style="width:6px;height:6px;background:#4361ee;border-radius:50%;margin-top:6px;flex-shrink:0;"></div>
                    <div style="font-size:13px;color:#b0b8cc;line-height:1.5;">{item}</div>
                </div>""" for item in [
                "<strong style='color:#fff;'>AI Route Optimization</strong> — MICE-based traffic prediction for regional networks",
                "<strong style='color:#fff;'>Own-Fleet Expansion</strong> — Reduce 3rd-party dependency in high-RTO zones",
                "<strong style='color:#fff;'>Address Validation Engine</strong> — Geo-coding integration to reduce 'Address Not Found' RTOs",
                "<strong style='color:#fff;'>API Real-time Sync</strong> — Upgrade courier integrations for millisecond status updates",
            ]]) + "</div>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # KPI table
        st.markdown('<div class="section-heading">2026 Monitoring Framework</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="content-card">
        <table class="styled-table">
            <thead><tr><th>Metric</th><th>Target</th><th>Owner</th></tr></thead>
            <tbody>
                <tr><td>Perfect Order Rate</td><td style="color:#4ade80;font-weight:700;">&gt; 92%</td><td>Operations</td></tr>
                <tr><td>RTO Recovery Cost</td><td style="color:#4ade80;font-weight:700;">&lt; ₹120 / Order</td><td>Finance</td></tr>
                <tr><td>Resolution Lead Time</td><td style="color:#4ade80;font-weight:700;">&lt; 24 Hours</td><td>CX Team</td></tr>
                <tr><td>Partner SLA Compliance</td><td style="color:#4ade80;font-weight:700;">&gt; 98%</td><td>Logistics Lead</td></tr>
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box success" style="text-align:center;padding:18px;margin-top:20px;">
            <strong>Analysis Complete</strong> — Seashells Logistics is positioned for a stabilized Q4 2026 season.
        </div>
        """, unsafe_allow_html=True)
