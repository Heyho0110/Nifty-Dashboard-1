"""
═══════════════════════════════════════════════════════════════════════════════
  NSE NIFTY DASHBOARD — Streamlit Application
═══════════════════════════════════════════════════════════════════════════════

  Live dashboard for NSE Nifty broad-market and sectoral indices.
  Inspired by Google Finance's clean three-panel layout.

  Features:
    • OHLC candlestick charts with moving averages
    • Live price + intraday change for all major Nifty indices
    • P/E, P/B, Dividend Yield (sourced from niftyindices.com)
    • PE Percentile analysis (10-year historical distribution)
    • Multi-period returns (1D → 5Y, CAGR for longer periods)
    • Technical signals (SMA 20/50/100/200)
    • Valuation zone classification

  SETUP:
    pip install streamlit yfinance pandas numpy plotly requests
    streamlit run streamlit_app.py

═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import streamlit.components.v1 as components
import requests
import json
import logging
import warnings

warnings.filterwarnings("ignore")

# Module logger — emits an audit trail for the P/E normalization engine (STEP 10).
# Streamlit surfaces these on the server console; they are not shown to end users.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("nifty.pe_normalization")

# Make every plotly_dark figure blend with the dark app background regardless of the
# deploy environment. On Streamlit Cloud the Streamlit chart theme can otherwise
# re-skin charts to a light palette; we pair this with theme=None on each chart.
pio.templates["plotly_dark"].layout.paper_bgcolor = "rgba(0,0,0,0)"
pio.templates["plotly_dark"].layout.plot_bgcolor = "rgba(0,0,0,0)"
pio.templates["plotly_dark"].layout.font.color = "#e6e6e6"
pio.templates.default = "plotly_dark"

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NSE Nifty Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
#  STYLING — clean, Google Finance-inspired aesthetic
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Sidebar is permanently visible — hide all collapse/expand controls and
       lock it open so the index tiles are always available. */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] {
        transform: none !important;
        visibility: visible !important;
        min-width: 360px !important;
    }

    /* Dark app background */
    .stApp { background: #0e1117; }

    /* Main container */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 100%; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #11141b;
        border-right: 1px solid #232733;
        min-width: 360px;
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #cbd2e0; }

    /* Index tiles — neon-glow square cards (glow colour injected per-tile below) */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { gap: 12px; }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"] { gap: 12px; }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: linear-gradient(160deg, #12161f 0%, #0a0d13 100%);
        color: #eef2f8;
        border: 1.4px solid rgba(120,130,150,0.35);
        border-radius: 16px;
        text-align: left;
        padding: 13px 15px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.2px;
        line-height: 1.5;
        white-space: pre-line;
        height: auto;
        min-height: 94px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
        transition: transform 0.14s ease, box-shadow 0.14s ease, border-color 0.14s ease;
        will-change: box-shadow, transform;
    }
    /* First text line (index name) reads as a heading */
    section[data-testid="stSidebar"] button[kind="secondary"] p {
        white-space: pre-line;
        text-align: left;
        color: #f3f6fb;
        font-size: 12px;
        line-height: 1.55;
        font-weight: 600;
        margin: 0;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        transform: translateY(-2px);
        color: #ffffff;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:focus,
    section[data-testid="stSidebar"] button[kind="secondary"]:active { color: #ffffff; }

    /* Refresh button keeps a neutral dark look */
    section[data-testid="stSidebar"] .st-key-refresh_btn button {
        background: #1c2230;
        min-height: 0;
        box-shadow: none;
        border: 1px solid #2c3344;
        border-radius: 10px;
        font-weight: 500;
        flex-direction: row;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    /* Hero price */
    .hero-name { font-size: 1.1rem; color: #9aa3b5; margin: 0 0 4px 0; font-weight: 500; }
    .hero-price { font-size: 2.5rem; font-weight: 700; margin: 0; line-height: 1.1; color: #f5f7fa; }
    .hero-change { font-size: 1.1rem; font-weight: 500; margin-top: 6px; }
    .pos { color: #22c55e; }
    .neg { color: #ef4444; }
    .neutral { color: #9aa3b5; }
    .as-of { color: #6b7280; font-size: 0.85rem; margin-top: 4px; }

    /* Metric tweaks */
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; color: #f5f7fa; }
    div[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #9aa3b5; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stMetric"] {
        background: #161a23;
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #232733;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #232733; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px; font-size: 0.9rem; font-weight: 500;
        background: transparent; border-radius: 6px 6px 0 0; color: #9aa3b5;
    }
    .stTabs [aria-selected="true"] { background: #1c2230; color: #f5f7fa; }

    /* Range pills (radio buttons styled) */
    div[role="radiogroup"] { gap: 4px; }
    div[role="radiogroup"] label {
        background: #1c2230;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.15s;
        color: #cbd2e0;
    }
    div[role="radiogroup"] label:hover { background: #262d3d; }

    /* Chart-type dropdown — compact rounded pill to match the range buttons */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #1c2230;
        border: 1px solid #2c3344;
        border-radius: 999px;
        min-height: 34px;
        font-size: 0.85rem;
        color: #e6e6e6;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover { border-color: #3a4256; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] svg { fill: #9aa3b5; }

    /* Valuation zone badge */
    .zone-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
    }
    .zone-under { background: #14361f; color: #4ade80; }
    .zone-fair  { background: #16294d; color: #60a5fa; }
    .zone-over  { background: #3a1620; color: #f87171; }
    .zone-neutral { background: #1c2230; color: #cbd2e0; }

    /* Compare table */
    table.cmp-table { width: 100%; border-collapse: collapse; margin-top: 6px; }
    table.cmp-table th, table.cmp-table td {
        padding: 11px 16px; text-align: right; font-size: 0.92rem;
        border-bottom: 1px solid #1c212c;
    }
    table.cmp-table th { font-size: 0.95rem; font-weight: 600; border-bottom: 1px solid #2c3344; }
    table.cmp-table th:first-child, table.cmp-table td:first-child {
        text-align: left; color: #9aa3b5; font-weight: 500;
    }
    table.cmp-table td.cmp-num { color: #e6e6e6; font-variant-numeric: tabular-nums; }
    table.cmp-table td.cmp-pos { color: #22c55e; }
    table.cmp-table td.cmp-neg { color: #ef4444; }
    table.cmp-table tr:last-child td { border-bottom: none; }

    /* Section headers */
    h3 { font-size: 1.1rem; font-weight: 600; color: #f5f7fa; margin-top: 1.5rem; }
    h4 { font-size: 0.95rem; font-weight: 600; color: #cbd2e0; }

    /* Divider tweak */
    hr { margin: 1rem 0; border-color: #232733; }

    /* ── Market-status strip + last-updated ── */
    .mkt-bar {
        display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
        padding: 7px 14px; margin-bottom: 6px;
        background: #11151d; border: 1px solid #232733; border-radius: 10px;
        font-size: 0.82rem; color: #9aa3b5;
    }
    .mkt-item { display: inline-flex; align-items: center; gap: 6px; }
    .mkt-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .mkt-open  { background: #22c55e; box-shadow: 0 0 7px rgba(34,197,94,0.8); animation: pulse 1.8s ease-in-out infinite; }
    .mkt-closed{ background: #6b7280; }
    .mkt-label { font-weight: 600; color: #cbd2e0; letter-spacing: 0.3px; }
    .mkt-time  { color: #9aa3b5; }
    .mkt-stale { color: #f59e0b; font-weight: 600; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

    /* ── Source badges ── */
    .src-badge {
        display: inline-block; padding: 2px 9px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.2px;
        border: 1px solid #2c3344; background: #161a23; color: #9aa3b5;
    }
    .src-badge b { color: #cbd2e0; font-weight: 600; }

    /* ── Glossary info marker + native tooltip ── */
    .gloss {
        display: inline-block; width: 15px; height: 15px; line-height: 15px;
        text-align: center; font-size: 10px; font-weight: 700;
        border-radius: 50%; background: #2c3344; color: #cbd2e0;
        cursor: help; margin-left: 6px; vertical-align: middle;
    }
    .gloss:hover { background: #3a4256; color: #fff; }

    /* ── Sticky mini-header (appears as you scroll) ── */
    .sticky-hdr {
        position: sticky; top: 0; z-index: 999;
        display: flex; align-items: center; gap: 12px;
        padding: 8px 14px; margin: 0 0 8px 0;
        background: rgba(17,21,29,0.92); backdrop-filter: blur(8px);
        border: 1px solid #232733; border-radius: 10px;
        font-size: 0.9rem;
    }
    .sticky-hdr .sh-name { font-weight: 700; color: #f5f7fa; }
    .sticky-hdr .sh-price { font-weight: 700; color: #f5f7fa; font-variant-numeric: tabular-nums; }
    .sticky-hdr .sh-chg-pos { color: #22c55e; font-weight: 600; }
    .sticky-hdr .sh-chg-neg { color: #ef4444; font-weight: 600; }

    /* ── Unified notice / limited-state callout ── */
    .notice {
        display: flex; align-items: flex-start; gap: 11px;
        padding: 13px 16px; margin: 8px 0;
        background: #15171d; border: 1px solid #2c3344;
        border-left: 3px solid #f59e0b; border-radius: 10px;
    }
    .notice.info { border-left-color: #38bdf8; }
    .notice.muted { border-left-color: #6b7280; }
    .notice-ico { font-size: 1.1rem; line-height: 1.3; }
    .notice-body { font-size: 0.86rem; color: #cbd2e0; line-height: 1.5; }
    .notice-body .nt-title { font-weight: 600; color: #f5f7fa; display: block; margin-bottom: 2px; }

    /* ── Skeleton shimmer (sidebar tiles while quotes load) ── */
    .skel-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .skel-tile {
        height: 94px; border-radius: 16px;
        background: linear-gradient(100deg, #12161f 30%, #1b212d 50%, #12161f 70%);
        background-size: 200% 100%;
        animation: skel-shimmer 1.3s ease-in-out infinite;
        border: 1px solid rgba(120,130,150,0.18);
    }
    @keyframes skel-shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

    /* ── Hero price reveal (paired with count-up component) ── */
    .hero-price { animation: hero-rise 0.5s ease-out; }
    @keyframes hero-rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
#  INDEX REGISTRY — (display_name, yahoo_ticker, niftyindices_name)
# ═══════════════════════════════════════════════════════════════════════════
INDICES = {
    "Broad Market": [
        ("NIFTY 50",            "^NSEI",                  "NIFTY 50"),
        ("NIFTY 100",           "^CNX100",                "NIFTY 100"),
        ("NIFTY 200",           "^CNX200",                "NIFTY 200"),
        ("NIFTY 500",           "^CRSLDX",                "NIFTY 500"),
        ("NIFTY MIDCAP 150",    "NIFTYMIDCAP150.NS",      "NIFTY MIDCAP 150"),
        ("NIFTY SMALLCAP 250",  "^CNXSC",                 "NIFTY SMALLCAP 250"),
        ("NIFTY NEXT 50",       "^NSMIDCP",               "NIFTY NEXT 50"),
    ],
    "Sectoral": [
        ("NIFTY AUTO",          "^CNXAUTO",      "NIFTY AUTO"),
        ("NIFTY BANK",          "^NSEBANK",      "NIFTY BANK"),
        ("NIFTY ENERGY",        "^CNXENERGY",    "NIFTY ENERGY"),
        ("NIFTY FIN SERVICES",  "NIFTY_FIN_SERVICE.NS",   "NIFTY FINANCIAL SERVICES"),
        ("NIFTY FMCG",          "^CNXFMCG",      "NIFTY FMCG"),
        ("NIFTY INFRA",         "^CNXINFRA",     "NIFTY INFRASTRUCTURE"),
        ("NIFTY IT",            "^CNXIT",        "NIFTY IT"),
        ("NIFTY MEDIA",         "^CNXMEDIA",     "NIFTY MEDIA"),
        ("NIFTY METAL",         "^CNXMETAL",     "NIFTY METAL"),
        ("NIFTY PHARMA",        "^CNXPHARMA",    "NIFTY PHARMA"),
        ("NIFTY PSU BANK",      "^CNXPSUBANK",   "NIFTY PSU BANK"),
        ("NIFTY REALTY",        "^CNXREALTY",    "NIFTY REALTY"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
#  DATA FETCHING — yfinance for OHLC, niftyindices.com for PE/PB
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance with retry."""
    try:
        df = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        return df
    except Exception:
        return pd.DataFrame()


def _fetch_pe_year(nifty_name: str, year: int) -> pd.DataFrame:
    """
    One calendar year of daily PE / PB / Dividend Yield from niftyindices.com.

    The PE/PB endpoint (getpepbHistoricaldataDBtoString) collapses very long ranges
    to a single monthly snapshot, but returns FULL DAILY data for a one-year window —
    so we fetch year-by-year and stitch. Not cached individually (so transient
    failures aren't memoised); the stitched result is cached by fetch_pe_pb_data().
    Returns Date/PE/PB/DY (empty on failure or if the index has no PE history).
    """
    base = "https://www.niftyindices.com"
    url = f"{base}/Backpage.aspx/getpepbHistoricaldataDBtoString"
    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    headers = {
        "User-Agent": ua,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{base}/reports/historical-data",
        "Origin": base,
    }
    payload = {
        "cinfo": json.dumps(
            {
                "name": nifty_name,
                "startDate": f"01-Jan-{year}",
                "endDate": f"31-Dec-{year}",  # niftyindices clamps to latest available
                "indexName": nifty_name,
            }
        )
    }
    for _ in range(2):  # retry — endpoint is intermittently slow/flaky (fail fast; we have a fallback)
        try:
            session = requests.Session()
            session.get(base, headers={"User-Agent": ua}, timeout=8)
            r = session.post(url, headers=headers, json=payload, timeout=12)
            if r.status_code != 200:
                continue
            raw = r.json().get("d")
            records = json.loads(raw) if isinstance(raw, str) else raw
            if not records:
                continue
            df = pd.DataFrame(records)
            if "DATE" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["DATE"], errors="coerce")
            for src, dst in {"pe": "PE", "pb": "PB", "divYield": "DY"}.items():
                if src in df.columns:
                    df[dst] = pd.to_numeric(df[src], errors="coerce")
            keep = ["Date"] + [c for c in ("PE", "PB", "DY") if c in df.columns]
            df = df.dropna(subset=["Date"])[keep]
            if not df.empty and "PE" in df.columns:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def _fetch_pe_single(nifty_name: str, years: int = 10) -> pd.DataFrame:
    """
    Resilient FALLBACK: one full-range request. niftyindices collapses this to a
    sparse (~monthly) sample, but it's a single lightweight call that usually
    succeeds even when the heavier per-year daily stitch is rate-limited. Lets the
    valuation trend + percentile still render (monthly-basis) instead of vanishing.
    """
    base = "https://www.niftyindices.com"
    url = f"{base}/Backpage.aspx/getpepbHistoricaldataDBtoString"
    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    headers = {
        "User-Agent": ua, "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/json; charset=UTF-8", "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{base}/reports/historical-data", "Origin": base,
    }
    end = datetime.now()
    start = end - timedelta(days=365 * years)
    payload = {"cinfo": json.dumps({
        "name": nifty_name, "startDate": start.strftime("%d-%b-%Y"),
        "endDate": end.strftime("%d-%b-%Y"), "indexName": nifty_name})}
    for _ in range(2):
        try:
            session = requests.Session()
            session.get(base, headers={"User-Agent": ua}, timeout=8)
            r = session.post(url, headers=headers, json=payload, timeout=15)
            if r.status_code != 200:
                continue
            raw = r.json().get("d")
            records = json.loads(raw) if isinstance(raw, str) else raw
            if not records:
                continue
            df = pd.DataFrame(records)
            if "DATE" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["DATE"], errors="coerce")
            for src, dst in {"pe": "PE", "pb": "PB", "divYield": "DY"}.items():
                if src in df.columns:
                    df[dst] = pd.to_numeric(df[src], errors="coerce")
            keep = ["Date"] + [c for c in ("PE", "PB", "DY") if c in df.columns]
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)[keep]
            if not df.empty and "PE" in df.columns and not df["PE"].dropna().empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_pe_pb_data(nifty_name: str, years: int = 10, daily: bool = False) -> pd.DataFrame:
    """
    Historical PE / PB / Dividend Yield from niftyindices.com.

    daily=False (DEFAULT, lightweight): ONE full-range request. Fast and gentle on
    niftyindices (a single call), so it's used on every index view. niftyindices
    usually returns a dense series, occasionally a ~monthly sample — either way the
    percentile is honest, just possibly coarser.

    daily=True (opt-in "full daily"): fetch each of the last `years` calendar years
    separately and stitch them into the guaranteed full daily distribution
    (~2,200–2,500 points). Heavier (~10 requests) — only run on demand. Missing dates
    are simply absent (no interpolation); values are used as reported by NSE.

    Cached 24h per (index, daily). Returns Date/PE/PB/DY ascending, or empty if the
    index has no PE history.
    """
    if not daily:
        return _fetch_pe_single(nifty_name, years)

    current_year = datetime.now().year
    year_list = list(range(current_year - years + 1, current_year + 1))

    # Probe the current year first. If empty, it's EITHER a no-PE index (e.g. Smallcap)
    # OR niftyindices is throttling us — so try one lightweight full-range request
    # before concluding there's no history.
    probe = _fetch_pe_year(nifty_name, current_year)
    if probe.empty:
        return _fetch_pe_single(nifty_name, years)

    # Fetch the remaining years concurrently (modest pool to avoid rate-limiting).
    rest_years = year_list[:-1]
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            frames = list(ex.map(lambda y: _fetch_pe_year(nifty_name, y), rest_years))
    except Exception:
        frames = [_fetch_pe_year(nifty_name, y) for y in rest_years]

    frames = [probe] + [f for f in frames if f is not None and not f.empty]
    df = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["Date"])
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if df.empty or "PE" not in df.columns or df["PE"].dropna().empty:
        return _fetch_pe_single(nifty_name, years)
    # If the daily stitch came back thin (most years throttled), supplement with the
    # lightweight full-range sample so the percentile/trend still render.
    if len(df) < 60:
        alt = _fetch_pe_single(nifty_name, years)
        if len(alt) > len(df):
            return alt
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_nse_indices() -> dict:
    """
    Live P/E, P/B, Dividend Yield (+ last) for ALL indices from NSE's allIndices
    feed — one authoritative call. Returns {index_name: {pe, pb, dy, last}}.
    """
    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )

    def _f(x):
        try:
            v = float(x)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    try:
        s = requests.Session()
        s.headers.update({"User-Agent": ua, "Accept": "application/json",
                          "Accept-Language": "en-US,en;q=0.9"})
        s.get("https://www.nseindia.com", timeout=10)
        r = s.get("https://www.nseindia.com/api/allIndices", timeout=15)
        if r.status_code != 200:
            return {}
        out = {}
        for d in r.json().get("data", []):
            name = d.get("index")
            if name:
                out[name] = {"pe": _f(d.get("pe")), "pb": _f(d.get("pb")),
                             "dy": _f(d.get("dy")), "last": _f(d.get("last"))}
        return out
    except Exception:
        return {}


def nse_valuation(ni_name: str) -> tuple:
    """(pe, pb, dy) live from NSE allIndices for one index, or (None, None, None)."""
    rec = fetch_nse_indices().get(ni_name)
    if not rec:
        return (None, None, None)
    return (rec.get("pe"), rec.get("pb"), rec.get("dy"))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv_niftyindices(ni_name: str, days: int = 1825) -> pd.DataFrame:
    """
    Fallback OHLC source for indices Yahoo doesn't carry (e.g. NIFTY SMALLCAP 250).
    Pulls OHLC from niftyindices.com and returns a DataFrame shaped exactly like
    fetch_ohlcv() — DatetimeIndex with Open/High/Low/Close/Volume columns.

    The niftyindices endpoint can be slow/flaky on very long ranges, so we use a
    fresh session + short timeout + a few retries. Keep `days` modest (<= ~5y);
    10y requests reliably time out.
    """
    base = "https://www.niftyindices.com"
    url = f"{base}/Backpage.aspx/getHistoricaldatatabletoString"
    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    headers = {
        "User-Agent": ua,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{base}/reports/historical-data",
        "Origin": base,
    }
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    payload = {
        "cinfo": json.dumps(
            {
                "name": ni_name,
                "startDate": start_date.strftime("%d-%b-%Y"),
                "endDate": end_date.strftime("%d-%b-%Y"),
                "indexName": ni_name,
            }
        )
    }

    for _ in range(3):  # retry — server is intermittently slow
        try:
            session = requests.Session()
            session.get(base, headers={"User-Agent": ua}, timeout=10)
            r = session.post(url, headers=headers, json=payload, timeout=15)
            if r.status_code != 200:
                continue
            raw = r.json().get("d")
            records = json.loads(raw) if isinstance(raw, str) else raw
            if not records:
                continue
            df = pd.DataFrame(records)
            if "HistoricalDate" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["HistoricalDate"], errors="coerce")
            rename = {"OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close"}
            for src, dst in rename.items():
                if src in df.columns:
                    df[dst] = pd.to_numeric(df[src], errors="coerce")
            keep = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
            if "Close" not in keep:
                continue
            df = (
                df.dropna(subset=["Date", "Close"])
                .sort_values("Date")
                .set_index("Date")[keep]
            )
            # niftyindices often returns a dense recent daily block PLUS a few
            # isolated old points (year-long gaps). Keep only the most recent
            # continuous run so the chart never shows misleading gaps/clusters.
            if len(df) > 2:
                gap_days = df.index.to_series().diff().dt.days
                breaks = gap_days[gap_days > 7]
                if not breaks.empty:
                    df = df.loc[breaks.index[-1]:]
            df["Volume"] = 0  # niftyindices doesn't publish index volume
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def load_ohlcv(ticker: str, ni_name: str | None = None, period: str = "10y") -> pd.DataFrame:
    """
    Primary OHLC loader: Yahoo Finance first, with a niftyindices.com fallback
    for indices Yahoo doesn't cover. Returns whichever source has real history.
    """
    df = fetch_ohlcv(ticker, period=period)
    if (df is None or len(df) < 5) and ni_name:
        # niftyindices only reliably serves a short recent daily window; longer
        # ranges time out or come back clustered. Request a light window and let
        # fetch_ohlcv_niftyindices() trim to the continuous recent run.
        days = 20 if period == "5d" else 90
        alt = fetch_ohlcv_niftyindices(ni_name, days=days)
        if not alt.empty and len(alt) > (0 if df is None else len(df)):
            return alt
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fii_dii() -> dict:
    """
    Latest-day FII/FPI & DII cash-market activity from NSE (buy/sell/net, ₹ cr).
    NSE only serves the most recent day. Returns {category: {...}} or {}.
    """
    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    try:
        s = requests.Session()
        s.headers.update({"User-Agent": ua, "Accept": "application/json",
                          "Accept-Language": "en-US,en;q=0.9"})
        s.get("https://www.nseindia.com", timeout=10)
        r = s.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=15)
        if r.status_code != 200:
            return {}
        out = {}
        for row in r.json():
            cat = row.get("category", "")
            out[cat] = {
                "buy": float(row.get("buyValue", "nan")),
                "sell": float(row.get("sellValue", "nan")),
                "net": float(row.get("netValue", "nan")),
                "date": row.get("date", ""),
            }
        return out
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
#  CALCULATIONS — returns, moving averages, percentiles
# ═══════════════════════════════════════════════════════════════════════════
def calculate_returns(df: pd.DataFrame, close_col: str = "Close") -> dict:
    """Multi-period returns. Periods >1Y are annualized (CAGR)."""
    if df is None or df.empty:
        return {}
    s = df[close_col].dropna()
    latest = s.iloc[-1]

    periods_days = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
    cagr_periods = {"3Y": 3, "5Y": 5}

    out = {}
    for label, days in periods_days.items():
        if len(s) > days:
            past = s.iloc[-days - 1]
            out[label] = (latest / past - 1) * 100

    for label, years in cagr_periods.items():
        days = years * 252
        if len(s) > days:
            past = s.iloc[-days - 1]
            out[label] = ((latest / past) ** (1 / years) - 1) * 100

    # YTD
    try:
        idx = pd.to_datetime(s.index)
        current_year = idx[-1].year
        ytd_mask = idx.year == current_year
        if ytd_mask.sum() > 0:
            first_ytd = s[ytd_mask].iloc[0]
            out["YTD"] = (latest / first_ytd - 1) * 100
    except Exception:
        pass

    return out


def calculate_moving_averages(df: pd.DataFrame, close_col: str = "Close") -> dict:
    """SMA & EMA for 20, 50, 100, 200 day periods."""
    if df is None or df.empty:
        return {}
    s = df[close_col].dropna()
    out = {}
    for period in [20, 50, 100, 200]:
        if len(s) >= period:
            out[f"SMA_{period}"] = s.rolling(period).mean().iloc[-1]
            out[f"EMA_{period}"] = s.ewm(span=period, adjust=False).mean().iloc[-1]
    return out


def calculate_rsi(df: pd.DataFrame, period: int = 14, close_col: str = "Close") -> pd.Series:
    """Wilder's RSI as a time series. Returns an empty Series if there's too little data."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df[close_col].dropna()
    if len(s) <= period:
        return pd.Series(dtype=float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing (EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.dropna()


def calculate_fibonacci(df: pd.DataFrame, lookback: int = 252) -> dict:
    """
    Fibonacci retracement levels from the swing high/low over `lookback` sessions.
    Direction is inferred from whether the high or the low came last.
    """
    if df is None or df.empty:
        return {}
    w = df.tail(lookback)
    if w.empty or "High" not in w.columns or "Low" not in w.columns:
        return {}
    hi = float(w["High"].max())
    lo = float(w["Low"].min())
    if hi <= lo:
        return {}
    hi_idx = w["High"].idxmax()
    lo_idx = w["Low"].idxmin()
    uptrend = hi_idx >= lo_idx  # high made after low → measure retracement from the high
    diff = hi - lo
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {
        r: (hi - diff * r if uptrend else lo + diff * r)
        for r in ratios
    }
    return {
        "high": hi, "low": lo, "uptrend": uptrend,
        "hi_idx": hi_idx, "lo_idx": lo_idx, "levels": levels, "window": w,
    }


def pe_percentile(pe_series: pd.Series, current: float = None) -> tuple:
    """
    Return (current_pe, percentile, stats_dict). `pe_series` is the daily history
    (NSE data) used as the benchmark distribution; `current` optionally overrides the
    ranked value with the live NSE P/E (so the percentile ranks today's live figure
    against the full daily history). Falls back to the series' last point if omitted.
    """
    if pe_series is None or pe_series.empty:
        return None, None, None
    clean = pe_series.dropna()
    if clean.empty:
        return None, None, None
    cur = current if current is not None else clean.iloc[-1]
    pct = (clean < cur).sum() / len(clean) * 100
    stats = {
        "min": clean.min(),
        "max": clean.max(),
        "median": clean.median(),
        "mean": clean.mean(),
        "p25": clean.quantile(0.25),
        "p75": clean.quantile(0.75),
    }
    return cur, pct, stats


def classify_zone(percentile: float) -> tuple:
    """Return (zone_text, css_class) from PE percentile."""
    if percentile is None:
        return ("Unknown", "zone-neutral")
    if percentile < 20:
        return ("🟢 Deeply Undervalued", "zone-under")
    if percentile < 40:
        return ("🟢 Undervalued", "zone-under")
    if percentile < 60:
        return ("🔵 Fairly Valued", "zone-fair")
    if percentile < 80:
        return ("🟠 Overvalued", "zone-over")
    return ("🔴 Deeply Overvalued", "zone-over")


# ═══════════════════════════════════════════════════════════════════════════
#  VALUATION ENGINE — institution-grade percentile analytics
#  Percentile = % of daily historical observations strictly below the current
#  value, over the FULL daily history (no monthly/yearly sampling).
# ═══════════════════════════════════════════════════════════════════════════
# Valuation bands by percentile (for "higher = more expensive" metrics: PE, PB).
VAL_BANDS = [
    (0, 10, "🟢 Extremely Undervalued", "zone-under"),
    (10, 25, "🟢 Undervalued", "zone-under"),
    (25, 75, "🔵 Fair Value", "zone-fair"),
    (75, 90, "🟠 Expensive", "zone-over"),
    (90, 100.0001, "🔴 Extremely Expensive", "zone-over"),
]


def valuation_zone(percentile: float) -> tuple:
    """(label, css_class) from a percentile using the 5 institutional bands."""
    if percentile is None:
        return ("Unknown", "zone-neutral")
    for lo, hi, label, cls in VAL_BANDS:
        if lo <= percentile < hi:
            return (label, cls)
    return ("Unknown", "zone-neutral")


def _clean_val_series(series: pd.Series) -> pd.Series:
    """Data-quality gate: drop NaN/±inf and non-positive ratios."""
    if series is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return s[s > 0]


def percentile_of(hist: pd.Series, current: float) -> float:
    """% of daily historical observations strictly below `current`."""
    s = _clean_val_series(hist)
    if s.empty or current is None or current <= 0:
        return None
    return float((s < current).sum()) / len(s) * 100.0


def valuation_metrics(series: pd.Series, current: float = None, winsor: float = 0.01) -> dict:
    """
    Full distribution + percentile analytics for a daily valuation series.
    Returns current, count, min/max/median/mean/std, the 1/5/10/25/50/75/90/95/99
    percentile boundaries, raw + winsorized current-percentile, and the clean series.
    """
    s = _clean_val_series(series)
    if s.empty:
        return None
    cur = current if (current is not None and current > 0) else float(s.iloc[-1])
    raw_pct = float((s < cur).sum()) / len(s) * 100.0

    # Winsorized: cap 1% tails, then rank the (capped) current value
    lo_cap, hi_cap = float(s.quantile(winsor)), float(s.quantile(1 - winsor))
    sw = s.clip(lo_cap, hi_cap)
    cur_w = min(max(cur, lo_cap), hi_cap)
    wins_pct = float((sw < cur_w).sum()) / len(sw) * 100.0

    return {
        "current": cur,
        "n": int(len(s)),
        "min": float(s.min()), "max": float(s.max()),
        "median": float(s.median()), "mean": float(s.mean()), "std": float(s.std()),
        "pcts": {p: float(s.quantile(p / 100.0)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
        "raw_pct": raw_pct, "wins_pct": wins_pct,
        "series": s,
    }


def rolling_percentiles(df: pd.DataFrame, value_col: str, current: float) -> dict:
    """Percentile of `current` within 3Y / 5Y / 10Y / Full daily windows."""
    out = {"3Y": None, "5Y": None, "10Y": None, "Full": None}
    if df is None or df.empty or "Date" not in df.columns or value_col not in df.columns:
        return out
    d = df.dropna(subset=["Date"])
    if d.empty:
        return out
    end = d["Date"].max()
    for label, yrs in (("3Y", 3), ("5Y", 5), ("10Y", 10), ("Full", None)):
        sub = d if yrs is None else d[d["Date"] >= end - pd.Timedelta(days=365 * yrs)]
        out[label] = percentile_of(sub[value_col], current)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  P/E NORMALIZATION ENGINE — methodology-break adjustment
# ───────────────────────────────────────────────────────────────────────────
#  WHY THIS EXISTS
#  Around the close of FY2020-21, NSE/Nifty Indices switched the earnings basis
#  of its valuation ratios (P/E, P/B, dividend yield) from STANDALONE to
#  CONSOLIDATED earnings. The published historical P/E series SPLICES the two
#  regimes with no back-adjustment: pre-transition values sit on a standalone
#  basis, post-transition values on a consolidated basis. Ranking today's
#  (consolidated) P/E against a window that mixes both regimes is not
#  methodology-consistent.
#
#  WHAT THIS ENGINE DOES
#  It produces a NORMALIZED, consolidated-equivalent history by scaling the
#  pre-transition segment by an empirically derived multiplicative factor, while
#  leaving all raw data untouched. It does NOT rebuild earnings from constituents.
#
#  IDENTIFICATION (important caveat)
#  The data source exposes a single P/E per date — standalone and consolidated
#  P/E are NEVER published simultaneously, so there is no true "overlap" sample.
#  The adjustment factor is therefore identified from observations BRACKETING the
#  transition (the discontinuity), not from simultaneous standalone/consolidated
#  pairs. Where P/B is continuous across the break (book value largely unaffected),
#  we use the price-independent ratio R = P/E ÷ P/B (= Book/Earnings), whose jump
#  isolates the earnings-basis change and cancels any price drift in the window;
#  R_post / R_pre = Standalone-EPS / Consolidated-EPS, exactly the adjustment
#  factor. Where P/B ALSO shifted at the break, we fall back to raw P/E ratios over
#  a tight window. The result is an approximation — see the dashboard methodology
#  note and validation panel.
# ═══════════════════════════════════════════════════════════════════════════
# --- Configurable parameters (STEP 1) — adjust here, never hardcode downstream ---
PE_NORM_TRANSITION_START = "2020-01-01"   # outer bound of the transition era to consider
PE_NORM_TRANSITION_END = "2022-12-31"     # outer bound (inclusive)
PE_NORM_TRANSITION_ANCHOR = "2021-03-31"  # documented NSE FY21 standalone→consolidated switch
PE_NORM_DETECT_WINDOW_DAYS = 25           # ± calendar days around the anchor to locate the exact break day
PE_NORM_EST_WINDOW_DAYS = 25              # ± calendar days around the break used to estimate the factor
PE_NORM_MIN_BREAK_JUMP = 0.04             # min |Δln(P/E)| (~4%) on the break day to qualify as a real break
PE_NORM_WINDOW_LADDER = (25, 60, 120, 150)  # adaptive widening (days): start tight, widen if data is sparse/monthly
PE_NORM_PB_CONTINUITY_TOL = 0.03          # |ΔP/B| below this ⇒ P/B continuous ⇒ use price-independent P/E÷P/B
PE_NORM_IQR_K = 1.5                       # IQR multiplier for outlier fences (STEP 3)
PE_NORM_CONF_HIGH = 0.06                  # rel-dispersion (std/median) thresholds for the confidence label
PE_NORM_CONF_MODERATE = 0.15


def detect_transition_break(df: pd.DataFrame, value_col: str = "PE",
                            anchor: str = PE_NORM_TRANSITION_ANCHOR,
                            detect_days: int = PE_NORM_DETECT_WINDOW_DAYS,
                            start: str = PE_NORM_TRANSITION_START,
                            end: str = PE_NORM_TRANSITION_END) -> tuple:
    """
    STEP 1 — locate the methodology transition day.

    The transition is a documented, system-wide event (FY21 boundary, identical
    date across every index). Rather than taking the global maximum jump over the
    whole 2020-2022 window — which would catch COVID earnings-revision spikes — we
    ANCHOR on the documented date and snap to the largest single-day |Δln(P/E)|
    within ±detect_days of it. Anchor + window are both configurable, so the engine
    is reusable for any future methodology change or other instruments.

    Returns (break_timestamp | None, diagnostics_dict).
    """
    if df is None or df.empty or value_col not in df.columns or "Date" not in df.columns:
        return None, {"status": "no_data"}
    d = df.dropna(subset=[value_col]).sort_values("Date")
    d = d[d[value_col] > 0]
    a = pd.Timestamp(anchor)
    # Adaptive window: start tight (precise on daily data) and widen only if the
    # window is too sparse — so a monthly/sparse fetch still sees the month-to-month
    # step. We look at the largest jump between CONSECUTIVE points, which works for
    # both daily (single-day spike) and monthly (month-to-month step) series.
    ladder = sorted(set([detect_days, *PE_NORM_WINDOW_LADDER]))
    win = pd.DataFrame()
    lo = hi = None
    for w in ladder:
        lo = max(pd.Timestamp(start), a - pd.Timedelta(days=w))
        hi = min(pd.Timestamp(end), a + pd.Timedelta(days=w))
        win = d[(d["Date"] >= lo) & (d["Date"] <= hi)].copy()
        if len(win) >= 5:
            break
    if len(win) < 2:
        log.info("break-detect: insufficient observations (%d) even at widest window [%s, %s]",
                 len(win), lo.date() if lo else "?", hi.date() if hi else "?")
        return None, {"status": "insufficient_data", "n_in_window": int(len(win)),
                      "search_window": [str(lo.date()), str(hi.date())] if lo is not None else None}
    win["dln"] = np.log(win[value_col]).diff()
    idx = win["dln"].abs().idxmax()
    jump = float(win.loc[idx, "dln"])
    bdate = win.loc[idx, "Date"]
    if abs(jump) < PE_NORM_MIN_BREAK_JUMP:
        log.info("break-detect: largest jump %.3f%% below threshold — treating as no break",
                 (np.exp(jump) - 1) * 100)
        return None, {"status": "no_significant_break", "max_jump_pct": float(np.exp(jump) - 1),
                      "search_window": [str(lo.date()), str(hi.date())]}
    log.info("break-detect: break at %s (%.1f%% one-day P/E move)", bdate.date(), (np.exp(jump) - 1) * 100)
    return bdate, {"status": "detected", "break_date": str(bdate.date()),
                   "jump_pct": float(np.exp(jump) - 1),
                   "search_window": [str(lo.date()), str(hi.date())]}


def estimate_adjustment_factor(df: pd.DataFrame, break_date: pd.Timestamp,
                               window_days: int = PE_NORM_EST_WINDOW_DAYS,
                               iqr_k: float = PE_NORM_IQR_K) -> dict:
    """
    STEPS 2-4 — robustly estimate the adjustment factor at the break.

    Adjustment factor = P/E_consolidated ÷ P/E_standalone  ( = Standalone-EPS ÷
    Consolidated-EPS ). Pre-transition P/E is multiplied by this to reach a
    consolidated-equivalent level.

    Method:
      • Choose a price-independent basis when valid: if P/B is continuous across the
        break (|ΔP/B| < tol), use R = P/E ÷ P/B; otherwise use raw P/E over a tight
        window (price drift then small).
      • Build the factor-observation CLOUD as every post-break value ÷ every
        pre-break value within ±window_days (a Hodges-Lehmann-style two-sample ratio
        set). This yields many observations for the outlier/robust-stat machinery.
      • STEP 3 outlier removal: drop observations outside [Q1-k·IQR, Q3+k·IQR].
      • STEP 4 robust estimate: report median (production factor), mean, std, min,
        max, counts. The MEDIAN is used downstream — resistant to COVID distortions,
        one-off subsidiary gains/losses and reporting anomalies.

    Returns a diagnostics dict (status "ok" carries the factor).
    """
    d = df.dropna(subset=["PE"]).sort_values("Date")

    # Adaptive basis: prefer the price-independent ratio when P/B did not itself jump.
    # Compare the P/B values immediately bracketing the break (nearest point each side
    # within 45d) so this works on daily AND sparse/monthly series.
    basis = "PE"
    pb_jump = None
    if "PB" in d.columns and (d["PB"] > 0).sum() > 10:
        near = d[(d["Date"] >= break_date - pd.Timedelta(days=45)) &
                 (d["Date"] <= break_date + pd.Timedelta(days=45))]
        pb_b = near[near["Date"] < break_date]["PB"].dropna()
        pb_a = near[near["Date"] >= break_date]["PB"].dropna()
        if len(pb_b) and len(pb_a) and pb_b.iloc[-1] > 0:
            pb_jump = float(pb_a.iloc[0] / pb_b.iloc[-1] - 1)
            if abs(pb_jump) < PE_NORM_PB_CONTINUITY_TOL:
                basis = "PE/PB"

    src = (d["PE"] / d["PB"]) if basis == "PE/PB" else d["PE"]
    s = pd.DataFrame({"Date": d["Date"].values, "v": pd.to_numeric(src, errors="coerce").values}).dropna()
    # Adaptive window: widen until both sides have enough points (handles sparse data).
    ladder = sorted(set([window_days, *PE_NORM_WINDOW_LADDER]))
    pre = post = np.array([])
    for w in ladder:
        lo = break_date - pd.Timedelta(days=w)
        hi = break_date + pd.Timedelta(days=w)
        pre = s[(s["Date"] >= lo) & (s["Date"] < break_date)]["v"].values
        post = s[(s["Date"] >= break_date) & (s["Date"] <= hi)]["v"].values
        pre = pre[pre > 0]
        post = post[post > 0]
        if len(pre) >= 3 and len(post) >= 3:
            break
    if len(pre) < 2 or len(post) < 2:
        log.info("factor-estimate: insufficient window obs (pre=%d, post=%d) even at widest window",
                 len(pre), len(post))
        return {"status": "insufficient_window_obs", "n_pre": int(len(pre)), "n_post": int(len(post)),
                "basis": basis}

    cloud = (post[:, None] / pre[None, :]).ravel()  # all pairwise consolidated÷standalone ratios
    n_total = int(cloud.size)
    # STEP 3 — IQR outlier removal
    q1, q3 = np.percentile(cloud, [25, 75])
    iqr = q3 - q1
    lob, hib = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    kept = cloud[(cloud >= lob) & (cloud <= hib)]
    if kept.size == 0:
        kept = cloud
    n_used = int(kept.size)
    n_removed = n_total - n_used
    median = float(np.median(kept))
    rel_disp = float(kept.std() / median) if median else None
    log.info("factor-estimate: basis=%s factor(median)=%.4f obs=%d removed=%d reldisp=%.3f",
             basis, median, n_total, n_removed, rel_disp or 0.0)
    return {
        "status": "ok", "basis": basis, "pb_jump_pct": pb_jump,
        "factor": median, "median": median, "mean": float(kept.mean()), "std": float(kept.std()),
        "min": float(kept.min()), "max": float(kept.max()),
        "n_total": n_total, "n_removed": n_removed, "n_used": n_used,
        "n_pre": int(len(pre)), "n_post": int(len(post)), "rel_dispersion": rel_disp,
        "q1": float(q1), "q3": float(q3), "iqr_lower": float(lob), "iqr_upper": float(hib),
    }


def build_normalized_pe_series(df: pd.DataFrame, value_col: str = "PE", **cfg) -> tuple:
    """
    STEP 5 — produce the normalized, consolidated-equivalent P/E series.

    Preserves the raw column as `<value_col>_raw` and adds `<value_col>_normalized`
    (derived only — raw data is never modified). Pre-transition rows are scaled by
    the median adjustment factor; post-transition rows are copied through unchanged.
    Vectorized. Graceful no-op (normalized == raw) when no break is found or data is
    insufficient, so every index — including those with no transition — keeps working.

    Returns (dataframe_with_derived_columns, audit_dict). The audit dict carries the
    full diagnostics required by the validation panel (STEP 9).
    """
    raw_col = f"{value_col}_raw"
    norm_col = f"{value_col}_normalized"
    ts = ist_now().strftime("%Y-%m-%d %H:%M:%S IST")

    if df is None or df.empty or value_col not in df.columns:
        out = df.copy() if df is not None else pd.DataFrame()
        return out, {"normalized": False, "status": "no_data", "factor": 1.0, "timestamp": ts,
                     "transition_window": [PE_NORM_TRANSITION_START, PE_NORM_TRANSITION_END]}

    out = df.copy()
    out[raw_col] = out[value_col]  # STEP 4/10 — preserve raw, create derived only

    bdate, detect = detect_transition_break(out, value_col=value_col,
                                            anchor=cfg.get("anchor", PE_NORM_TRANSITION_ANCHOR),
                                            detect_days=cfg.get("detect_days", PE_NORM_DETECT_WINDOW_DAYS),
                                            start=cfg.get("start", PE_NORM_TRANSITION_START),
                                            end=cfg.get("end", PE_NORM_TRANSITION_END))
    base_audit = {"timestamp": ts, "value_col": value_col, "detect": detect,
                  "transition_window": [cfg.get("start", PE_NORM_TRANSITION_START),
                                        cfg.get("end", PE_NORM_TRANSITION_END)]}
    if bdate is None:
        out[norm_col] = out[value_col]  # identity — no comparable break to adjust
        log.info("normalize[%s]: no transition break — normalized series == raw", value_col)
        return out, {**base_audit, "normalized": False, "factor": 1.0,
                     "status": detect.get("status", "no_break")}

    est = estimate_adjustment_factor(out, bdate, window_days=cfg.get("est_days", PE_NORM_EST_WINDOW_DAYS),
                                     iqr_k=cfg.get("iqr_k", PE_NORM_IQR_K))
    if est.get("status") != "ok":
        out[norm_col] = out[value_col]
        return out, {**base_audit, "normalized": False, "factor": 1.0,
                     "break_date": str(bdate.date()), "estimate": est, "status": est.get("status")}

    factor = est["factor"]
    # STEP 5 — vectorized application: scale pre-transition, pass through post-transition.
    out[norm_col] = np.where(out["Date"] < bdate, out[value_col] * factor, out[value_col])

    rel = est.get("rel_dispersion") or 1.0
    confidence = ("high" if rel < PE_NORM_CONF_HIGH
                  else "moderate" if rel < PE_NORM_CONF_MODERATE else "low")
    log.info("normalize[%s]: applied factor %.4f to pre-%s rows (confidence=%s)",
             value_col, factor, bdate.date(), confidence)
    return out, {**base_audit, "normalized": True, "factor": factor, "break_date": str(bdate.date()),
                 "estimate": est, "confidence": confidence, "status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════
#  UI HELPERS — status strip · badges · notices · glossary · animated hero
# ═══════════════════════════════════════════════════════════════════════════
IST = timezone(timedelta(hours=5, minutes=30))


def ist_now():
    return datetime.now(IST)


def market_status():
    """(is_open, label) for the NSE cash market — Mon–Fri 09:15–15:30 IST.
    Holidays are not modelled (free data has no reliable holiday calendar)."""
    now = ist_now()
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    is_open = now.weekday() < 5 and open_t <= now <= close_t
    return is_open, ("Market open" if is_open else "Market closed")


@st.cache_data(ttl=300, show_spinner=False)
def quotes_last_updated():
    """Wall-clock IST time the live-quote cache cycle was (re)populated.
    Shares the 300s TTL with the quote fetch, so it tracks the same refresh cadence."""
    return ist_now()


def render_market_strip():
    """Thin status bar: market open/closed + when data was last refreshed."""
    is_open, label = market_status()
    updated_dt = quotes_last_updated()
    updated = updated_dt.strftime("%H:%M:%S")
    dot = "mkt-open" if is_open else "mkt-closed"
    stale = ""
    if is_open:
        age = (ist_now() - updated_dt).total_seconds()
        if age > 120:
            stale = f'<span class="mkt-stale">· delayed ~{int(age // 60)}m</span>'
    st.markdown(
        f'<div class="mkt-bar">'
        f'<span class="mkt-item"><span class="mkt-dot {dot}"></span>'
        f'<span class="mkt-label">{label}</span></span>'
        f'<span class="mkt-item mkt-time">🕒 Updated {updated} IST {stale}</span>'
        f'<span class="mkt-item">Live prices cached ~5 min</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def src_badge(source: str):
    """Small 'Source: X' pill for per-section data provenance."""
    st.markdown(f'<span class="src-badge">Source: <b>{source}</b></span>', unsafe_allow_html=True)


def section_header(title: str, tip: str = None):
    """An h4 section header with an optional glossary tooltip marker."""
    g = f'<span class="gloss" title="{tip}">i</span>' if tip else ""
    st.markdown(f"<h4>{title}{g}</h4>", unsafe_allow_html=True)


def data_notice(title: str, body: str, kind: str = "warn"):
    """Consistent styled callout for limited / empty / unavailable states."""
    icon = {"warn": "⚠️", "info": "ℹ️", "muted": "•"}.get(kind, "ℹ️")
    cls = {"warn": "", "info": "info", "muted": "muted"}.get(kind, "")
    st.markdown(
        f'<div class="notice {cls}"><div class="notice-ico">{icon}</div>'
        f'<div class="notice-body"><span class="nt-title">{title}</span>{body}</div></div>',
        unsafe_allow_html=True,
    )


# Self-contained animated hero (count-up price + inline sparkline). Rendered in an
# iframe via components.html so the count-up JS can run; styled to blend with the
# dark app background. Placeholders are substituted (avoids f-string brace clashes).
_HERO_TEMPLATE = """
<style>html,body{background:transparent;margin:0;padding:0;}</style>
<div style="font-family:'Source Sans 3','Source Sans Pro',-apple-system,BlinkMacSystemFont,sans-serif;">
  <div style="font-size:1.1rem;color:#9aa3b5;font-weight:500;margin-bottom:4px;">__NAME__</div>
  <div style="display:flex;align-items:flex-end;gap:18px;">
    <div id="hp" style="font-size:2.5rem;font-weight:700;color:#f5f7fa;line-height:1;">0.00</div>
    <svg id="hs" width="124" height="44" style="margin-bottom:6px;overflow:visible;"></svg>
  </div>
  <div style="font-size:1.1rem;font-weight:500;margin-top:8px;color:__CHGCOLOR__;">__ARROW__ __CHG__ (__CHGPCT__%) Today</div>
  <div style="color:#6b7280;font-size:0.85rem;margin-top:4px;">As of __ASOF__</div>
</div>
<script>
(function(){
  var target = __PRICE__, closes = __CLOSES__;
  var el = document.getElementById('hp');
  function fmt(n){return n.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});}
  var dur=900, t0=null;
  function step(ts){ if(!t0)t0=ts; var p=Math.min(1,(ts-t0)/dur); var e=1-Math.pow(1-p,3);
    el.textContent=fmt(target*e); if(p<1){requestAnimationFrame(step);} else {el.textContent=fmt(target);} }
  requestAnimationFrame(step);
  var svg=document.getElementById('hs'), W=124, H=44, pad=3;
  if(closes && closes.length>1){
    var mn=Math.min.apply(null,closes), mx=Math.max.apply(null,closes), rng=(mx-mn)||1;
    var pts=closes.map(function(c,i){var x=pad+(W-2*pad)*i/(closes.length-1);
      var y=pad+(H-2*pad)*(1-(c-mn)/rng); return x.toFixed(1)+','+y.toFixed(1);});
    var ns='http://www.w3.org/2000/svg';
    var poly=document.createElementNS(ns,'polyline');
    poly.setAttribute('points',pts.join(' ')); poly.setAttribute('fill','none');
    poly.setAttribute('stroke','__SPARK__'); poly.setAttribute('stroke-width','1.8');
    poly.setAttribute('stroke-linejoin','round'); poly.setAttribute('stroke-linecap','round');
    svg.appendChild(poly);
    var last=pts[pts.length-1].split(',');
    var dot=document.createElementNS(ns,'circle');
    dot.setAttribute('cx',last[0]); dot.setAttribute('cy',last[1]);
    dot.setAttribute('r','2.6'); dot.setAttribute('fill','__SPARK__');
    svg.appendChild(dot);
  }
})();
</script>
"""


def render_hero_left(name, current, prev, chg, chg_pct, as_of, closes):
    """Animated hero block: count-up price + inline sparkline + change + as-of."""
    pos = chg >= 0
    spark_up = len(closes) >= 2 and closes[-1] >= closes[0]
    repl = {
        "__NAME__": str(name),
        "__PRICE__": f"{current:.2f}",
        "__ARROW__": "▲" if pos else "▼",
        "__CHG__": f"{chg:+,.2f}",
        "__CHGPCT__": f"{chg_pct:+.2f}",
        "__CHGCOLOR__": "#22c55e" if pos else "#ef4444",
        "__ASOF__": str(as_of),
        "__CLOSES__": json.dumps([round(float(c), 2) for c in closes]),
        "__SPARK__": "#22c55e" if spark_up else "#ef4444",
    }
    html = _HERO_TEMPLATE
    for k, v in repl.items():
        html = html.replace(k, v)
    components.html(html, height=150)


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR — index selector as a grid of coloured square tiles
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("### 📋 Indices")
st.sidebar.caption("Tap any tile to open its dashboard")

if "selected_index" not in st.session_state:
    st.session_state.selected_index = ("NIFTY 50", "^NSEI", "NIFTY 50")

# P/E valuation basis: "Raw" (as-reported by NSE) or "Normalized" (methodology-break
# adjusted). Default Raw so existing behaviour is unchanged until the user opts in.
st.session_state.setdefault("pe_valuation_mode", "Raw")


def _refresh_data():
    """Refresh callback — runs before the rerun body, so one pass clears + reloads."""
    st.cache_data.clear()


# Refresh button
with st.sidebar.container(key="refresh_btn"):
    st.button("🔄 Refresh data", use_container_width=True, on_click=_refresh_data)

st.sidebar.markdown("")

# Flatten the registry so each index gets a stable tile id
_flat = [
    (category, display, ticker, ni_name)
    for category, items in INDICES.items()
    for (display, ticker, ni_name) in items
]
TILE_ID = {ticker: i for i, (_, _, ticker, _) in enumerate(_flat)}

# ── Neon glow palette: green = gainer, red = loser, slate = no data ──────────
GLOW = {
    "up":   {"line": "rgba(46,213,140,0.95)",  "g1": "rgba(46,213,140,0.50)",
             "g2": "rgba(46,213,140,0.30)",    "gin": "rgba(46,213,140,0.10)",
             "chg": "#34e0a1"},
    "down": {"line": "rgba(248,92,102,0.95)",  "g1": "rgba(248,92,102,0.50)",
             "g2": "rgba(248,92,102,0.30)",    "gin": "rgba(248,92,102,0.10)",
             "chg": "#f8606a"},
    "flat": {"line": "rgba(150,160,180,0.70)", "g1": "rgba(150,160,180,0.22)",
             "g2": "rgba(150,160,180,0.12)",   "gin": "rgba(150,160,180,0.06)",
             "chg": "#aab2c2"},
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_quotes(tickers: tuple) -> dict:
    """One batched 5-day download for every tile (~1s for all). Returns {ticker: (close, chg_pct)}."""
    out = {t: (None, None) for t in tickers}
    try:
        data = yf.download(
            list(tickers), period="5d", progress=False,
            auto_adjust=False, threads=True, group_by="ticker",
        )
    except Exception:
        return out
    for t in tickers:
        try:
            s = data[t]["Close"].dropna()
            if len(s) >= 2:
                close = float(s.iloc[-1])
                prev = float(s.iloc[-2])
                out[t] = (close, (close / prev - 1) * 100)
        except Exception:
            pass
    return out


def fetch_quote_fallback(ni_name: str) -> tuple:
    """niftyindices.com quote for tiles Yahoo can't supply (e.g. Smallcap 250).
    Not cached itself — it only derives from the already-cached fetch_ohlcv_niftyindices()."""
    alt = fetch_ohlcv_niftyindices(ni_name, days=20)
    if not alt.empty and len(alt) >= 2:
        close = float(alt["Close"].iloc[-1])
        prev = float(alt["Close"].iloc[-2])
        return close, (close / prev - 1) * 100
    return None, None


# Pull every tile's quote in one shot, fall back to niftyindices for any gaps.
# Show shimmering skeleton tiles while the (cold) fetch runs, then clear them.
with st.sidebar:
    _skel = st.empty()
    _skel.markdown(
        '<div class="skel-grid">' + ('<div class="skel-tile"></div>' * len(_flat)) + "</div>",
        unsafe_allow_html=True,
    )
    _quotes = fetch_all_quotes(tuple(t for _, _, t, _ in _flat))
    TILE_QUOTE = {}
    for _, _, ticker, _ni in _flat:
        close, chg = _quotes.get(ticker, (None, None))
        if close is None and _ni:
            close, chg = fetch_quote_fallback(_ni)
        if chg is None:
            state = "flat"
        elif chg >= 0:
            state = "up"
        else:
            state = "down"
        TILE_QUOTE[ticker] = (close, chg, state)
    _skel.empty()

# Inject per-tile neon glow (+ a brighter ring on the selected tile)
_selected_ticker = st.session_state.selected_index[1]
_tile_css = ["<style>"]
for ticker, i in TILE_ID.items():
    g = GLOW[TILE_QUOTE[ticker][2]]
    sel = ticker == _selected_ticker
    spread = "0 0 10px, 0 0 22px" if sel else "0 0 7px, 0 0 17px"
    _tile_css.append(
        f'.st-key-idx_{i} button[kind="secondary"] {{ '
        f'border-color: {g["line"]}; '
        f'box-shadow: 0 0 7px {g["g1"]}, 0 0 17px {g["g2"]}, '
        f'inset 0 0 12px {g["gin"]}{", 0 0 0 1.4px " + g["line"] if sel else ""}; }}'
    )
    _tile_css.append(
        f'.st-key-idx_{i} button[kind="secondary"]:hover {{ '
        f'box-shadow: 0 0 12px {g["g1"]}, 0 0 26px {g["g2"]}, inset 0 0 14px {g["gin"]}; }}'
    )
_tile_css.append("</style>")
st.sidebar.markdown("\n".join(_tile_css), unsafe_allow_html=True)


def _select_index(display, ticker, ni_name):
    """Tile-click callback — sets selection before the rerun body, so the main
    pane and the selected-ring update in the same pass (no extra full rerun)."""
    st.session_state.selected_index = (display, ticker, ni_name)


def render_index_tile(display, ticker, ni_name):
    """Render one neon tile: name, price, and glowing ▲/▼ change."""
    close, chg, _state = TILE_QUOTE[ticker]
    if chg is not None:
        arrow = "▲" if chg >= 0 else "▼"
        label = f"{display}\n{close:,.2f}\n{arrow} {chg:+.2f}%"
    else:
        label = f"{display}\n—\nno data"

    st.button(
        label, key=f"idx_{TILE_ID[ticker]}", use_container_width=True,
        on_click=_select_index, args=(display, ticker, ni_name),
    )


# Render each category as a 2-column grid of tiles
for category, items in INDICES.items():
    st.sidebar.markdown(f"**{category}**")
    for row_start in range(0, len(items), 2):
        cols = st.sidebar.columns(2)
        for col, (display, ticker, ni_name) in zip(cols, items[row_start:row_start + 2]):
            with col:
                render_index_tile(display, ticker, ni_name)
    st.sidebar.markdown("")

# ═══════════════════════════════════════════════════════════════════════════
#  VIEW MODE — Single dashboard  |  Compare two indices
# ═══════════════════════════════════════════════════════════════════════════
ALL_NAMES = [display for _, display, _, _ in _flat]
INDEX_BY_NAME = {display: (display, ticker, ni) for _, display, ticker, ni in _flat}

RANGE_LIST = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"]
RANGE_DAYS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260}


def slice_to_range(df, sel):
    """Trim an OHLC frame to the selected time range."""
    if df is None or df.empty:
        return df
    if sel == "YTD":
        return df[df.index.year == df.index[-1].year]
    if sel == "MAX":
        return df
    return df.tail(RANGE_DAYS[sel])


INTERVAL_RULE = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}


def resample_ohlc(df, interval):
    """Aggregate daily OHLCV into Weekly/Monthly candles (Daily passes through)."""
    rule = INTERVAL_RULE.get(interval)
    if df is None or df.empty or not rule:
        return df
    return (
        df.resample(rule)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )


view_mode = st.segmented_control(
    "View mode",
    ["📊 Single", "⚖️ Compare", "🔁 Market"],
    default="📊 Single",
    label_visibility="collapsed",
)

# Market open/closed + last-updated strip (shown across all views)
render_market_strip()

# ═══════════════════════════════════════════════════════════════════════════
#  COMPARE VIEW
# ═══════════════════════════════════════════════════════════════════════════
if view_mode and "Compare" in view_mode:
    st.markdown("### ⚖️ Compare Indices")

    COL_A, COL_B = "#22d3ee", "#fbbf24"  # cyan vs amber — clear on dark

    cur_name = st.session_state.selected_index[0]
    default_a = cur_name if cur_name in ALL_NAMES else ALL_NAMES[0]
    default_b = "NIFTY BANK" if "NIFTY BANK" in ALL_NAMES else ALL_NAMES[min(1, len(ALL_NAMES) - 1)]
    if default_b == default_a:
        default_b = next((n for n in ALL_NAMES if n != default_a), default_a)

    sc1, sc2 = st.columns(2)
    with sc1:
        name_a = st.selectbox("Index A", ALL_NAMES, index=ALL_NAMES.index(default_a), key="cmp_a")
    with sc2:
        name_b = st.selectbox("Index B", ALL_NAMES, index=ALL_NAMES.index(default_b), key="cmp_b")

    cr_col, ci_col = st.columns([5, 1])
    with cr_col:
        cmp_range = st.radio(
            "Range", RANGE_LIST, index=4, horizontal=True,
            label_visibility="collapsed", key="cmp_range",
        )
    with ci_col:
        cmp_interval = st.selectbox(
            "Interval", ["Daily", "Weekly", "Monthly"], index=0,
            label_visibility="collapsed", key="cmp_interval",
            help="Aggregate by Day, Week or Month",
        )

    a_disp, a_tk, a_ni = INDEX_BY_NAME[name_a]
    b_disp, b_tk, b_ni = INDEX_BY_NAME[name_b]

    with st.spinner("Loading comparison…"):
        da = load_ohlcv(a_tk, a_ni, period="10y")
        db = load_ohlcv(b_tk, b_ni, period="10y")
        # P/E for the table comes from NSE allIndices (live) — see _collect below.

    if da is None or da.empty or db is None or db.empty:
        st.error("Couldn't load price data for one of the selected indices. Try another pair.")
        st.stop()

    ca = resample_ohlc(slice_to_range(da, cmp_range), cmp_interval)
    cb = resample_ohlc(slice_to_range(db, cmp_range), cmp_interval)

    # ── Normalised performance chart (both rebased to 100) ──
    perf_fig = go.Figure()
    for cd, nm, color in [(ca, name_a, COL_A), (cb, name_b, COL_B)]:
        if cd is not None and not cd.empty:
            norm = cd["Close"] / float(cd["Close"].iloc[0]) * 100
            perf_fig.add_trace(
                go.Scatter(
                    x=cd.index, y=norm, name=nm,
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{nm}</b><br>%{{x|%b %d, %Y}}<br>%{{y:.1f}} (base 100)<extra></extra>",
                )
            )
    perf_fig.add_hline(y=100, line_dash="dot", line_color="#4b5563")
    perf_fig.update_layout(
        height=440, template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="left", x=0, font=dict(size=12)),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Indexed to 100",
    )
    perf_fig.update_xaxes(showgrid=False)
    perf_fig.update_yaxes(showgrid=True, gridcolor="#232733")
    st.plotly_chart(perf_fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    st.caption(
        "Both indices rebased to 100 at the start of the selected range — this compares "
        "**% performance**, not absolute levels."
    )

    if (ca is not None and len(ca) < 60) or (cb is not None and len(cb) < 60):
        st.caption("ℹ️ One index has limited daily history from free sources, so the overlay window may be shorter for it.")

    # ── Side-by-side metrics table ──
    def _collect(df, live_pe):
        if df is None or df.empty:
            return {}
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        r = calculate_returns(df)
        out = {
            "Last price": last,
            "Today %": (last / prev - 1) * 100 if prev else np.nan,
            "1M %": r.get("1M", np.nan),
            "6M %": r.get("6M", np.nan),
            "1Y %": r.get("1Y", np.nan),
            "3Y % (CAGR)": r.get("3Y", np.nan),
            "52W High": float(df["High"].tail(252).max()),
            "52W Low": float(df["Low"].tail(252).min()),
        }
        out["P/E"] = live_pe if live_pe is not None else np.nan
        return out

    ma, mb = _collect(da, nse_valuation(a_ni)[0]), _collect(db, nse_valuation(b_ni)[0])

    # Window (range) performance from the rebased series
    def _range_ret(cd):
        if cd is None or cd.empty:
            return np.nan
        return (float(cd["Close"].iloc[-1]) / float(cd["Close"].iloc[0]) - 1) * 100
    ma[f"{cmp_range} return %"] = _range_ret(ca)
    mb[f"{cmp_range} return %"] = _range_ret(cb)

    def _num(v, dp=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '<td class="cmp-num">—</td>'
        return f'<td class="cmp-num">{v:,.{dp}f}</td>'

    def _pct(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '<td class="cmp-num">—</td>'
        cls = "cmp-pos" if v >= 0 else "cmp-neg"
        return f'<td class="cmp-num {cls}">{v:+.2f}%</td>'

    row_order = [
        ("Last price", "num"),
        ("Today %", "pct"),
        (f"{cmp_range} return %", "pct"),
        ("1M %", "pct"),
        ("6M %", "pct"),
        ("1Y %", "pct"),
        ("3Y % (CAGR)", "pct"),
        ("52W High", "num"),
        ("52W Low", "num"),
        ("P/E", "num"),
    ]
    rows_html = []
    for label, kind in row_order:
        cell = _pct if kind == "pct" else _num
        rows_html.append(f"<tr><td>{label}</td>{cell(ma.get(label))}{cell(mb.get(label))}</tr>")

    st.markdown(
        f"""
        <table class="cmp-table">
            <tr>
                <th></th>
                <th style="color:{COL_A}">● {name_a}</th>
                <th style="color:{COL_B}">● {name_b}</th>
            </tr>
            {''.join(rows_html)}
        </table>
        """,
        unsafe_allow_html=True,
    )

    # ── One-line verdict over the selected window ──
    ra, rb = ma[f"{cmp_range} return %"], mb[f"{cmp_range} return %"]
    if not (np.isnan(ra) or np.isnan(rb)):
        if abs(ra - rb) < 0.05:
            st.info(f"Over **{cmp_range}**, {name_a} and {name_b} performed almost identically ({ra:+.2f}% vs {rb:+.2f}%).")
        else:
            winner, w, l = (name_a, ra, rb) if ra > rb else (name_b, rb, ra)
            st.success(f"Over **{cmp_range}**, **{winner}** outperformed by **{abs(w - l):.2f} pts** ({w:+.2f}% vs {l:+.2f}%).")

    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
#  MARKET VIEW — cap rotation + latest FII/DII flows
# ═══════════════════════════════════════════════════════════════════════════
if view_mode and "Market" in view_mode:
    st.markdown("### 🔁 Market — Cap Rotation & Institutional Flows")
    src_badge("Yahoo Finance")

    # ── Cap rotation ──
    CAP_SEGMENTS = [
        ("Largecap (Nifty 100)", "^CNX100", "#22d3ee"),
        ("Midcap (Nifty Midcap 150)", "NIFTYMIDCAP150.NS", "#a78bfa"),
        ("Smallcap (ETF proxy)", "HDFCSML250.NS", "#fbbf24"),
    ]
    mkt_range = st.radio(
        "Range", RANGE_LIST, index=5, horizontal=True,
        label_visibility="collapsed", key="mkt_range",
    )

    with st.spinner("Loading cap-segment data…"):
        closes = {}
        for nm, tk, _ in CAP_SEGMENTS:
            d = fetch_ohlcv(tk, period="10y")
            if d is not None and not d.empty:
                closes[nm] = d["Close"]

    if len(closes) < 2:
        st.error("Couldn't load cap-segment indices right now. Try refreshing.")
    else:
        # Align on common trading days, then slice to the selected range
        combined = pd.concat(closes, axis=1).dropna()
        combined = slice_to_range(combined, mkt_range)
        norm = combined / combined.iloc[0] * 100  # rebased to 100 at window start

        color_of = {nm: c for nm, _, c in CAP_SEGMENTS}

        # Normalised performance chart
        rot_fig = go.Figure()
        for nm in norm.columns:
            rot_fig.add_trace(
                go.Scatter(x=norm.index, y=norm[nm], name=nm,
                           line=dict(color=color_of.get(nm, "#e6e6e6"), width=2))
            )
        rot_fig.add_hline(y=100, line_dash="dot", line_color="#4b5563")
        rot_fig.update_layout(
            height=420, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0, font=dict(size=12)),
            hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Indexed to 100",
        )
        rot_fig.update_xaxes(showgrid=False)
        rot_fig.update_yaxes(showgrid=True, gridcolor="#232733")
        st.plotly_chart(rot_fig, use_container_width=True, theme=None, config={"displayModeBar": False})
        st.caption(
            f"All segments rebased to 100 at the start of the **{mkt_range}** window "
            f"({norm.index[0].strftime('%b %d, %Y')}). Smallcap uses the HDFC Smallcap 250 "
            f"ETF as a proxy (minor tracking error)."
        )

        # Relative strength vs Largecap (rising = that segment leading large-caps)
        large_col = "Largecap (Nifty 100)"
        if large_col in norm.columns:
            rs_fig = go.Figure()
            for nm in norm.columns:
                if nm == large_col:
                    continue
                rs = norm[nm] / norm[large_col] * 100
                rs_fig.add_trace(
                    go.Scatter(x=rs.index, y=rs, name=f"{nm.split(' (')[0]} ÷ Largecap",
                               line=dict(color=color_of.get(nm, "#e6e6e6"), width=2))
                )
            rs_fig.add_hline(y=100, line_dash="dot", line_color="#4b5563")
            rs_fig.update_layout(
                height=300, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="top", y=1.14, xanchor="left", x=0, font=dict(size=11)),
                hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Relative strength",
            )
            rs_fig.update_xaxes(showgrid=False)
            rs_fig.update_yaxes(showgrid=True, gridcolor="#232733")
            st.markdown("#### Relative Strength vs Largecap")
            st.plotly_chart(rs_fig, use_container_width=True, theme=None, config={"displayModeBar": False})
            st.caption("Above 100 = the segment has outperformed largecaps since the window start (money rotating into it).")

        # Leadership read for the window
        window_ret = (norm.iloc[-1] - 100).sort_values(ascending=False)
        leader = window_ret.index[0]
        rcols = st.columns(len(window_ret))
        for col, (nm, ret) in zip(rcols, window_ret.items()):
            col.metric(nm.split(" (")[0], f"{ret:+.2f}%", help="Return over the selected window")
        st.success(f"Over **{mkt_range}**, **{leader.split(' (')[0]}** is leading ({window_ret.iloc[0]:+.2f}%).")

    # ── FII / DII latest-day flows ──
    st.markdown("#### Institutional Flows (latest trading day)")
    src_badge("NSE")
    flows = fetch_fii_dii()
    fii = flows.get("FII/FPI") or flows.get("FII")
    dii = flows.get("DII")
    if fii or dii:
        as_of = (fii or dii).get("date", "")
        fcol, dcol = st.columns(2)
        for col, lbl, rec in [(fcol, "FII / FPI", fii), (dcol, "DII", dii)]:
            with col:
                if rec:
                    net = rec["net"]
                    tone = "🟢 net buyers" if net >= 0 else "🔴 net sellers"
                    col.metric(f"{lbl} — net", f"₹{net:,.0f} cr", delta=tone,
                               delta_color="normal" if net >= 0 else "inverse")
                    st.caption(f"Buy ₹{rec['buy']:,.0f} cr · Sell ₹{rec['sell']:,.0f} cr")
                else:
                    col.metric(f"{lbl} — net", "—")

        # Net-flow bar
        flow_fig = go.Figure()
        cats = [(lbl, rec) for lbl, rec in [("FII/FPI", fii), ("DII", dii)] if rec]
        flow_fig.add_trace(go.Bar(
            x=[lbl for lbl, _ in cats],
            y=[rec["net"] for _, rec in cats],
            marker_color=["#22c55e" if rec["net"] >= 0 else "#ef4444" for _, rec in cats],
            text=[f"₹{rec['net']:,.0f} cr" for _, rec in cats],
            textposition="outside",
        ))
        flow_fig.add_hline(y=0, line_color="#4b5563")
        flow_fig.update_layout(
            height=260, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False, yaxis_title="Net (₹ cr)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        flow_fig.update_xaxes(showgrid=False)
        flow_fig.update_yaxes(showgrid=True, gridcolor="#232733")
        st.plotly_chart(flow_fig, use_container_width=True, theme=None, config={"displayModeBar": False})
        st.caption(
            f"Cash-market provisional figures for **{as_of}** (source: NSE). "
            "Only the latest trading day is published free — no historical series."
        )
    else:
        data_notice(
            "FII/DII unavailable",
            "NSE may be rate-limiting right now. Try Refresh in the sidebar.",
            kind="warn",
        )

    # ── Sector Valuation Dashboard (opt-in: fetches per-sector daily history) ──
    st.markdown("#### Sector Valuation Dashboard")
    st.caption(
        "Live NSE P/E of each sector vs its own **10-year daily** P/E history. "
        "Building this fetches per-sector history (slow on first run, cached 24h)."
    )
    if st.button("📊 Build / refresh sector valuation table", key="build_sector_val"):
        st.session_state["_show_sector_val"] = True

    if st.session_state.get("_show_sector_val"):
        live_all = fetch_nse_indices()
        sectors = INDICES["Sectoral"]
        # Honour the same Raw/Normalized basis chosen in the valuation tab, so the
        # methodology-break normalization (with adaptive sparse-data detection) is
        # applied to EVERY index here too — each gets its own per-index factor.
        _sec_mode = st.session_state.get("pe_valuation_mode", "Raw")
        prog = st.progress(0.0, text="Fetching sector valuation history…")
        rows = []
        for i, (disp, _tk, ni) in enumerate(sectors):
            live_pe = (live_all.get(ni) or {}).get("pe")
            hist = fetch_pe_pb_data(ni, years=10)
            pctile = median = factor = conf = None
            if not hist.empty and "PE" in hist.columns:
                # Run the normalization engine per-index (adds PE_raw + PE_normalized).
                hist, _audit = build_normalized_pe_series(hist, value_col="PE")
                _col = ("PE_normalized" if (_sec_mode == "Normalized" and "PE_normalized" in hist.columns)
                        else "PE")
                if _audit.get("normalized"):
                    factor, conf = _audit.get("factor"), _audit.get("confidence")
                if live_pe:
                    pctile = rolling_percentiles(hist, _col, live_pe).get("10Y")
                median = float(_clean_val_series(hist[_col]).median())
            rows.append((disp, live_pe, pctile, median, factor, conf))
            prog.progress((i + 1) / len(sectors), text=f"Processed {disp}")
        prog.empty()

        # Sort most-expensive first (by 10Y percentile)
        rows.sort(key=lambda r: (r[2] is not None, r[2] if r[2] is not None else -1), reverse=True)

        def _c(v, fmt):
            return f'<td class="cmp-num">{format(v, fmt)}</td>' if v is not None else '<td class="cmp-num">—</td>'

        body = []
        for disp, live_pe, pctile, median, factor, conf in rows:
            pe_cell = _c(live_pe, ".2f")
            med_cell = _c(median, ".2f")
            if factor is not None:
                fac_cell = f'<td class="cmp-num">×{factor:.3f}<br><span style="font-size:0.72rem;color:#9aa3b5;">{conf or ""}</span></td>'
            else:
                fac_cell = '<td class="cmp-num">—</td>'
            if pctile is not None:
                z, zc = valuation_zone(pctile)
                pct_cell = f'<td class="cmp-num">{pctile:.0f}%</td>'
                zone_cell = f'<td><span class="zone-badge {zc}">{z}</span></td>'
            else:
                pct_cell = '<td class="cmp-num">—</td>'
                zone_cell = '<td class="cmp-num">no history</td>'
            body.append(f"<tr><td>{disp}</td>{pe_cell}{pct_cell}{med_cell}{fac_cell}{zone_cell}</tr>")
        st.markdown(
            '<table class="cmp-table"><tr><th>Sector</th><th>Current P/E</th>'
            '<th>10Y %ile</th><th>Median P/E</th><th>Adj. factor</th><th>Valuation zone</th></tr>'
            + "".join(body) + "</table>",
            unsafe_allow_html=True,
        )
        _basis_word = "Normalized" if _sec_mode == "Normalized" else "Raw"
        st.caption(
            f"Percentile = % of the sector's last-10Y daily P/E readings below today's live NSE P/E. "
            f"Basis: **{_basis_word} P/E** (per-index methodology-break factors shown in *Adj. factor* — "
            f"applied to the percentile only in Normalized mode; switch in the P/E Percentile tab)."
        )

        # Export the sector table (CSV + JSON)
        sec_records = [
            {"sector": disp, "current_pe": live_pe,
             "pctile_10Y": (round(pctile, 1) if pctile is not None else None),
             "median_pe": (round(median, 2) if median is not None else None),
             "pe_basis": ("normalized" if _sec_mode == "Normalized" else "raw"),
             "adjustment_factor": (round(factor, 4) if factor is not None else None),
             "normalization_confidence": conf,
             "zone": (valuation_zone(pctile)[0] if pctile is not None else None)}
            for disp, live_pe, pctile, median, factor, conf in rows
        ]
        sec_csv = pd.DataFrame(sec_records).to_csv(index=False).encode("utf-8")
        sec_json = json.dumps({"generated_at_ist": ist_now().strftime("%Y-%m-%d %H:%M:%S"),
                               "sectors": sec_records}, indent=2, default=str).encode("utf-8")
        sx1, sx2 = st.columns(2)
        sx1.download_button("⬇️ Sector table (CSV)", data=sec_csv,
                            file_name="sector_valuation.csv", mime="text/csv", use_container_width=True)
        sx2.download_button("⬇️ Sector table (JSON)", data=sec_json,
                            file_name="sector_valuation.json", mime="application/json", use_container_width=True)

    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PAGE (Single)
# ═══════════════════════════════════════════════════════════════════════════
display_name, yf_ticker, ni_name = st.session_state.selected_index

with st.spinner(f"Loading {display_name}…"):
    price_data = load_ohlcv(yf_ticker, ni_name, period="10y")
    # Lightweight (single-call) P/E by default; full daily stitch only when opted in.
    pe_data = fetch_pe_pb_data(ni_name, years=10, daily=st.session_state.get("pe_full_daily", False))

# Build the methodology-normalized P/E series (adds PE_raw + PE_normalized; raw is
# preserved). Cheap & vectorized, so computed every run. The audit dict feeds the
# validation panel. `pe_value_col` is the column every P/E analytic reads, switched
# by the Raw/Normalized toggle in the valuation tab.
pe_data, pe_norm_audit = build_normalized_pe_series(pe_data, value_col="PE")
_pe_mode = st.session_state.get("pe_valuation_mode", "Raw")
pe_value_col = "PE_normalized" if (_pe_mode == "Normalized" and "PE_normalized" in pe_data.columns) else "PE"

# Live P/E, P/B, Div Yield from NSE allIndices (authoritative current values).
nse_pe, nse_pb, nse_dy = nse_valuation(ni_name)

if price_data.empty or len(price_data) < 3:
    st.error(f"⚠️ Could not fetch enough price history for **{display_name}** ({yf_ticker}).")
    st.info(
        "Yahoo Finance may not carry this index and the niftyindices.com fallback "
        "returned too little data (or there's a temporary network issue). "
        "Try **🔄 Refresh data** in the sidebar, or pick another index."
    )
    st.stop()

# ───── HERO: name + current price + change ─────────────────────────────────
current = float(price_data["Close"].iloc[-1])
prev = float(price_data["Close"].iloc[-2]) if len(price_data) > 1 else current
chg = current - prev
chg_pct = (chg / prev) * 100 if prev != 0 else 0
chg_class = "pos" if chg >= 0 else "neg"
arrow = "▲" if chg >= 0 else "▼"
as_of = price_data.index[-1].strftime("%b %d, %Y")

hero_col_left, hero_col_right = st.columns([3, 1])
with hero_col_left:
    render_hero_left(
        display_name, current, prev, chg, chg_pct, as_of,
        price_data["Close"].tail(40).tolist(),
    )

with hero_col_right:
    # Quick valuation snapshot — live NSE P/E ranked against the daily NSE history.
    # Honours the Raw/Normalized basis chosen in the valuation tab; the live (always
    # consolidated) P/E remains the ranked value in both modes.
    if not pe_data.empty and pe_value_col in pe_data.columns:
        pe_now, pe_pct, _ = pe_percentile(pe_data[pe_value_col], current=nse_pe)
        if pe_now is not None:
            zone_txt, zone_cls = valuation_zone(pe_pct)
            _basis_tag = "normalized" if pe_value_col == "PE_normalized" else "raw"
            st.markdown(
                f"""
                <div style="text-align:right; padding-top: 0.5rem;">
                    <div class="hero-name">P/E (10Y percentile · {_basis_tag})</div>
                    <div style="font-size: 1.8rem; font-weight: 700;">{pe_now:.2f}</div>
                    <div style="margin-top: 6px;">
                        <span class="zone-badge {zone_cls}">{zone_txt}</span>
                    </div>
                    <div class="as-of">{pe_pct:.0f}th percentile of 10Y</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("<hr>", unsafe_allow_html=True)

# ───── TIME RANGE + CHART TYPE SELECTORS + PRICE CHART ──────────────────────
# Wrapped in a fragment: changing range / interval / chart-type reruns ONLY this
# block (chart + its controls), not the whole app. Nothing outside depends on
# these three widgets, so output is identical — only wasted full reruns are cut.
@st.fragment
def _render_price_chart():
    ranges = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"]
    range_days = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260}

    range_col, interval_col, type_col = st.columns([4, 1, 1])
    with range_col:
        selected_range = st.radio(
            "Time range",
            ranges,
            index=4,
            horizontal=True,
            label_visibility="collapsed",
        )
    with interval_col:
        chart_interval = st.selectbox(
            "Interval",
            ["Daily", "Weekly", "Monthly"],
            index=0,
            label_visibility="collapsed",
            help="Aggregate candles by Day, Week or Month",
        )
    with type_col:
        chart_type = st.selectbox(
            "Chart type",
            ["Candlestick", "Area", "Line", "OHLC Bars"],
            index=0,
            label_visibility="collapsed",
            help="Choose how to display the price series",
        )

    if selected_range == "YTD":
        cy = price_data.index[-1].year
        chart_data = price_data[price_data.index.year == cy]
    elif selected_range == "MAX":
        chart_data = price_data
    else:
        chart_data = price_data.tail(range_days[selected_range])

    # Resample to the chosen candle interval (Weekly/Monthly aggregate daily OHLCV)
    chart_data = resample_ohlc(chart_data, chart_interval)

    # ───── PRICE CHART (type per selection) + VOLUME ────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )

    GREEN, RED = "#22c55e", "#ef4444"

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=chart_data.index,
                open=chart_data["Open"],
                high=chart_data["High"],
                low=chart_data["Low"],
                close=chart_data["Close"],
                name="OHLC",
                increasing_line_color=GREEN,
                decreasing_line_color=RED,
                increasing_fillcolor=GREEN,
                decreasing_fillcolor=RED,
                showlegend=False,
            ),
            row=1, col=1,
        )
    elif chart_type == "OHLC Bars":
        fig.add_trace(
            go.Ohlc(
                x=chart_data.index,
                open=chart_data["Open"],
                high=chart_data["High"],
                low=chart_data["Low"],
                close=chart_data["Close"],
                name="OHLC",
                increasing_line_color=GREEN,
                decreasing_line_color=RED,
                showlegend=False,
            ),
            row=1, col=1,
        )
    else:  # Area or Line — colour by trend over the visible window
        trend_up = float(chart_data["Close"].iloc[-1]) >= float(chart_data["Close"].iloc[0])
        line_color = GREEN if trend_up else RED
        price_trace = go.Scatter(
            x=chart_data.index,
            y=chart_data["Close"],
            mode="lines",
            line=dict(color=line_color, width=2),
            name="Close",
            showlegend=False,
        )
        if chart_type == "Area":
            price_trace.update(
                fill="tozeroy",
                fillcolor="rgba(34,197,94,0.14)" if trend_up else "rgba(239,68,68,0.14)",
            )
        fig.add_trace(price_trace, row=1, col=1)
        # Tighten the y-axis so the fill reads as an area, not a thin sliver up top
        lo = float(chart_data["Low"].min())
        hi = float(chart_data["High"].max())
        pad = (hi - lo) * 0.06 or hi * 0.02
        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)

    # Moving averages on chart
    if len(chart_data) >= 50:
        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["Close"].rolling(50).mean(),
                name="SMA 50",
                line=dict(color="#f59e0b", width=1.4),
            ),
            row=1, col=1,
        )
    if len(chart_data) >= 200:
        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["Close"].rolling(200).mean(),
                name="SMA 200",
                line=dict(color="#3b82f6", width=1.4),
            ),
            row=1, col=1,
        )

    # Volume bars (vectorised colour mapping — identical hex/order to the prior loop)
    vol_colors = np.where(
        chart_data["Close"].values >= chart_data["Open"].values, "#16a34a", "#dc2626"
    ).tolist()
    fig.add_trace(
        go.Bar(
            x=chart_data.index,
            y=chart_data["Volume"],
            marker_color=vol_colors,
            name="Volume",
            showlegend=False,
            opacity=0.6,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="left", x=0, font=dict(size=11)),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#232733", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#232733", row=2, col=1, title="Volume", title_font=dict(size=10))

    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    # Limited-history notice (indices Yahoo doesn't carry fall back to niftyindices,
    # which only reliably serves a short recent daily window).
    if len(price_data) < 60:
        span = f"{price_data.index[0].strftime('%b %d, %Y')} – {price_data.index[-1].strftime('%b %d, %Y')}"
        data_notice(
            "Limited daily history",
            f"Showing the most recent continuous data for this index from free sources "
            f"({span}, {len(price_data)} sessions, source: niftyindices.com). "
            f"Range buttons above won't extend further back.",
            kind="info",
        )


_render_price_chart()

# ───── OVERVIEW METRICS ROW ────────────────────────────────────────────────
st.markdown("### Overview")
o = float(price_data["Open"].iloc[-1])
h = float(price_data["High"].iloc[-1])
l = float(price_data["Low"].iloc[-1])
high_52w = float(price_data["High"].tail(252).max())
low_52w = float(price_data["Low"].tail(252).min())

pct_from_high = (current / high_52w - 1) * 100
pct_from_low = (current / low_52w - 1) * 100

# Neon glow on the 52W boxes — green where it's a rise, red where it's a fall.
_glow_css = ["<style>"]
for key, val in (("m52high", pct_from_high), ("m52low", pct_from_low)):
    g = GLOW["up"] if val >= 0 else GLOW["down"]
    _glow_css.append(
        f'.st-key-{key} div[data-testid="stMetric"] {{ '
        f'border: 1px solid {g["line"]}; '
        f'box-shadow: 0 0 7px {g["g1"]}, 0 0 16px {g["g2"]}, inset 0 0 10px {g["gin"]}; }}'
    )
_glow_css.append("</style>")
st.markdown("\n".join(_glow_css), unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Open", f"{o:,.2f}")
c2.metric("High", f"{h:,.2f}")
c3.metric("Low", f"{l:,.2f}")
with c4:
    with st.container(key="m52high"):
        st.metric("52W High", f"{high_52w:,.2f}", delta=f"{pct_from_high:+.1f}% from high",
                  delta_color="normal")
with c5:
    with st.container(key="m52low"):
        st.metric("52W Low", f"{low_52w:,.2f}", delta=f"{pct_from_low:+.1f}% from low",
                  delta_color="normal")

# ═══════════════════════════════════════════════════════════════════════════
#  DEEP-DIVE TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_ret, tab_val, tab_tech, tab_pct = st.tabs(
    ["📈 Returns", "💰 Valuation", "📊 Technical", "🎯 PE Percentile"]
)

# ───── TAB 1: RETURNS ──────────────────────────────────────────────────────
with tab_ret:
    src_badge("Yahoo Finance")
    rets = calculate_returns(price_data)
    if rets:
        order = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"]
        rets_ordered = {k: rets[k] for k in order if k in rets}

        cols = st.columns(len(rets_ordered))
        for col, (label, val) in zip(cols, rets_ordered.items()):
            arrow_ret = "▲" if val >= 0 else "▼"
            col.metric(label, f"{val:+.2f}%", help="CAGR" if label in ["3Y", "5Y"] else "Absolute return")
        st.caption("ℹ️  3Y and 5Y returns are annualized (CAGR). All others are absolute returns.")

        # Returns bar chart
        st.markdown("#### Returns Comparison")
        ret_fig = go.Figure(
            go.Bar(
                x=list(rets_ordered.keys()),
                y=list(rets_ordered.values()),
                marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in rets_ordered.values()],
                text=[f"{v:+.1f}%" for v in rets_ordered.values()],
                textposition="outside",
            )
        )
        ret_fig.update_layout(
            height=300,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Return (%)",
            showlegend=False,
        )
        st.plotly_chart(ret_fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    else:
        st.info("Not enough data to compute returns.")

# ───── TAB 2: VALUATION ────────────────────────────────────────────────────
with tab_val:
    src_badge("NSE allIndices (current) · niftyindices (history)")
    # Current metrics from NSE allIndices (live) — available for ALL indices,
    # incl. ones with no daily history (e.g. Smallcap 250).
    latest_pe = nse_pe if nse_pe is not None else (pe_data["PE"].iloc[-1] if (not pe_data.empty and "PE" in pe_data.columns) else None)
    latest_pb = nse_pb if nse_pb is not None else (pe_data["PB"].iloc[-1] if (not pe_data.empty and "PB" in pe_data.columns) else None)
    latest_dy = nse_dy if nse_dy is not None else (pe_data["DY"].iloc[-1] if (not pe_data.empty and "DY" in pe_data.columns) else None)

    if any(v is not None and not pd.isna(v) for v in (latest_pe, latest_pb, latest_dy)):
        v1, v2, v3 = st.columns(3)
        if latest_pe is not None and not pd.isna(latest_pe):
            v1.metric("P/E Ratio", f"{latest_pe:.2f}")
        if latest_pb is not None and not pd.isna(latest_pb):
            v2.metric("P/B Ratio", f"{latest_pb:.2f}")
        if latest_dy is not None and not pd.isna(latest_dy):
            v3.metric("Dividend Yield", f"{latest_dy:.2f}%")

    if not pe_data.empty:
        st.markdown("#### Historical Valuation Trends (10Y)")

        val_fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("P/E Ratio", "P/B Ratio", "Dividend Yield (%)"),
            horizontal_spacing=0.08,
        )
        if "PE" in pe_data.columns:
            val_fig.add_trace(
                go.Scatter(x=pe_data["Date"], y=pe_data["PE"], line=dict(color="#6366f1", width=1.5), name="PE"),
                row=1, col=1,
            )
            # Add mean line
            val_fig.add_hline(y=pe_data["PE"].mean(), line_dash="dash", line_color="#9ca3af", row=1, col=1)
        if "PB" in pe_data.columns:
            val_fig.add_trace(
                go.Scatter(x=pe_data["Date"], y=pe_data["PB"], line=dict(color="#10b981", width=1.5), name="PB"),
                row=1, col=2,
            )
            val_fig.add_hline(y=pe_data["PB"].mean(), line_dash="dash", line_color="#9ca3af", row=1, col=2)
        if "DY" in pe_data.columns:
            val_fig.add_trace(
                go.Scatter(x=pe_data["Date"], y=pe_data["DY"], line=dict(color="#f59e0b", width=1.5), name="DY"),
                row=1, col=3,
            )
            val_fig.add_hline(y=pe_data["DY"].mean(), line_dash="dash", line_color="#9ca3af", row=1, col=3)

        val_fig.update_layout(
            height=320,
            template="plotly_dark",
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        val_fig.update_xaxes(showgrid=False)
        val_fig.update_yaxes(showgrid=True, gridcolor="#232733")
        st.plotly_chart(val_fig, use_container_width=True, theme=None, config={"displayModeBar": False})

        st.caption("Dashed line = 10-year mean. Current values: NSE · history: niftyindices.com")
    else:
        data_notice(
            "Historical valuation trend not available",
            "Current P/E / P/B / Dividend Yield above are live from NSE, but there's no daily "
            "valuation history to chart for this index (typical for some smallcap / sectoral "
            "indices on niftyindices.com). Try Refresh in the sidebar.",
            kind="muted",
        )

# ───── TAB 3: TECHNICAL ────────────────────────────────────────────────────
with tab_tech:
    src_badge("Yahoo Finance")
    mas = calculate_moving_averages(price_data)
    if mas:
        st.markdown("#### Moving Averages — Position Relative to Price")
        cols = st.columns(4)
        for i, period in enumerate([20, 50, 100, 200]):
            sma = mas.get(f"SMA_{period}")
            if sma:
                diff_pct = (current / sma - 1) * 100
                signal = "🟢 Above" if current > sma else "🔴 Below"
                cols[i].metric(
                    f"SMA {period}",
                    f"{sma:,.2f}",
                    delta=f"{diff_pct:+.2f}% ({signal})",
                    delta_color="normal" if current > sma else "inverse",
                )

        st.markdown("")
        cols = st.columns(4)
        for i, period in enumerate([20, 50, 100, 200]):
            ema = mas.get(f"EMA_{period}")
            if ema:
                diff_pct = (current / ema - 1) * 100
                cols[i].metric(f"EMA {period}", f"{ema:,.2f}", delta=f"{diff_pct:+.2f}%", delta_color="off")

        # Golden Cross / Death Cross detector
        if "SMA_50" in mas and "SMA_200" in mas:
            sma50 = mas["SMA_50"]
            sma200 = mas["SMA_200"]
            st.markdown("#### Trend Signal")
            if sma50 > sma200:
                st.success(f"🟢 **Bullish trend**: SMA 50 ({sma50:,.2f}) is above SMA 200 ({sma200:,.2f}) — Golden Cross territory")
            else:
                st.error(f"🔴 **Bearish trend**: SMA 50 ({sma50:,.2f}) is below SMA 200 ({sma200:,.2f}) — Death Cross territory")

        # Plot MAs as a chart
        st.markdown("#### Price with all Moving Averages (1Y)")
        ma_chart_data = price_data.tail(252)
        ma_fig = go.Figure()
        ma_fig.add_trace(go.Scatter(x=ma_chart_data.index, y=ma_chart_data["Close"], name="Close", line=dict(color="#e6e6e6", width=1.8)))
        for period, color in [(20, "#ef4444"), (50, "#f59e0b"), (100, "#10b981"), (200, "#3b82f6")]:
            ma_fig.add_trace(
                go.Scatter(
                    x=ma_chart_data.index,
                    y=ma_chart_data["Close"].rolling(period).mean(),
                    name=f"SMA {period}",
                    line=dict(color=color, width=1.2),
                )
            )
        ma_fig.update_layout(
            height=350,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.05, x=0),
        )
        ma_fig.update_xaxes(showgrid=False)
        ma_fig.update_yaxes(showgrid=True, gridcolor="#232733")
        st.plotly_chart(ma_fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    # ── RSI (Relative Strength Index) — shown for every index ──
    section_header(
        "Relative Strength Index (RSI 14)",
        "Momentum gauge from 0–100. Above 70 = overbought (often due for a pullback); "
        "below 30 = oversold (often due for a bounce); ~50 = neutral. Wilder's 14-period.",
    )
    rsi_series = calculate_rsi(price_data, period=14)
    if rsi_series is not None and not rsi_series.empty:
        rsi_now = float(rsi_series.iloc[-1])
        if rsi_now >= 70:
            rsi_zone, rsi_cls = "🔴 Overbought", "zone-over"
        elif rsi_now <= 30:
            rsi_zone, rsi_cls = "🟢 Oversold", "zone-under"
        else:
            rsi_zone, rsi_cls = "🔵 Neutral", "zone-fair"

        rc1, rc2 = st.columns([1, 3])
        with rc1:
            st.metric("RSI (14)", f"{rsi_now:.1f}")
            st.markdown(
                f'<span class="zone-badge {rsi_cls}" style="display:inline-block; margin-top:6px;">{rsi_zone}</span>',
                unsafe_allow_html=True,
            )
            st.caption("RSI > 70 = overbought · < 30 = oversold (Wilder's 14-period).")
        with rc2:
            rsi_plot = rsi_series.tail(252)
            rsi_fig = go.Figure()
            rsi_fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.10)", line_width=0)
            rsi_fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.10)", line_width=0)
            rsi_fig.add_trace(
                go.Scatter(
                    x=rsi_plot.index, y=rsi_plot, name="RSI",
                    line=dict(color="#a78bfa", width=1.6),
                )
            )
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="#ef4444")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="#22c55e")
            rsi_fig.add_hline(y=50, line_dash="dot", line_color="#4b5563")
            rsi_fig.update_layout(
                height=260, template="plotly_dark",
                margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                yaxis=dict(range=[0, 100], title="RSI"),
            )
            rsi_fig.update_xaxes(showgrid=False)
            rsi_fig.update_yaxes(showgrid=True, gridcolor="#232733")
            st.plotly_chart(rsi_fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    else:
        data_notice("RSI unavailable", "Not enough price history to compute RSI (needs > 14 sessions).", kind="muted")

    # ── Fibonacci Retracement — shown for every index ──
    section_header(
        "Fibonacci Retracement",
        "Horizontal levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) between a swing high and low — "
        "common zones where pullbacks may find support or resistance.",
    )
    fib = calculate_fibonacci(price_data, lookback=252)
    if fib:
        trend_txt = "uptrend (low → high)" if fib["uptrend"] else "downtrend (high → low)"
        st.caption(
            f"Swing over the last {len(fib['window'])} sessions — "
            f"High **{fib['high']:,.2f}** ({fib['hi_idx'].strftime('%b %d, %Y')}), "
            f"Low **{fib['low']:,.2f}** ({fib['lo_idx'].strftime('%b %d, %Y')}); {trend_txt}."
        )

        fib_colors = {
            0.0: "#94a3b8", 0.236: "#38bdf8", 0.382: "#22c55e", 0.5: "#eab308",
            0.618: "#f59e0b", 0.786: "#fb7185", 1.0: "#94a3b8",
        }
        w = fib["window"]
        fib_fig = go.Figure()
        fib_fig.add_trace(
            go.Scatter(x=w.index, y=w["Close"], name="Close", line=dict(color="#e6e6e6", width=1.6))
        )
        for r, price in fib["levels"].items():
            fib_fig.add_hline(
                y=price, line_dash="dash", line_width=1,
                line_color=fib_colors.get(r, "#94a3b8"),
                annotation_text=f"{r*100:.1f}%  {price:,.0f}",
                annotation_position="right",
                annotation_font_color=fib_colors.get(r, "#94a3b8"),
                annotation_font_size=10,
            )
        fib_fig.update_layout(
            height=380, template="plotly_dark",
            margin=dict(l=0, r=70, t=10, b=0), showlegend=False,
        )
        fib_fig.update_xaxes(showgrid=False)
        fib_fig.update_yaxes(showgrid=True, gridcolor="#232733")
        st.plotly_chart(fib_fig, use_container_width=True, theme=None, config={"displayModeBar": False})

        # Levels row
        lvl_cols = st.columns(len(fib["levels"]))
        for col, (r, price) in zip(lvl_cols, sorted(fib["levels"].items())):
            col.metric(f"{r*100:.1f}%", f"{price:,.2f}")

        # Where the current price sits between levels
        price_to_ratio = {p: r for r, p in fib["levels"].items()}
        below = max((p for p in price_to_ratio if p <= current), default=None)
        above = min((p for p in price_to_ratio if p >= current), default=None)
        if below is not None and above is not None and below != above:
            st.caption(
                f"Current **{current:,.2f}** sits between the "
                f"**{price_to_ratio[below]*100:.1f}%** ({below:,.2f}) and "
                f"**{price_to_ratio[above]*100:.1f}%** ({above:,.2f}) levels."
            )
        st.caption("Levels derived from the swing high/low; common support/resistance zones, not advice.")
    else:
        data_notice("Fibonacci unavailable", "Not enough data to compute Fibonacci levels.", kind="muted")

# ───── TAB 4: PE PERCENTILE ────────────────────────────────────────────────
with tab_pct:
    src_badge("NSE allIndices (current) · niftyindices (history)")
    section_header(
        "P/E Percentile — historical valuation engine",
        "Live NSE P/E ranked against the index's own historical P/E "
        "(% of observations cheaper than today). Data-quality gate drops NaN/∞ and "
        "non-positive P/E before computing.",
    )

    # ── STEP 7: Raw ⟷ Normalized basis toggle ──
    # Drives EVERY P/E analytic below (percentile, gauge, rolling windows, stats,
    # distribution, band chart, export) via `pe_value_col`, recomputed at page top.
    _norm_ok = bool(pe_norm_audit.get("normalized"))
    mode_col, badge_col = st.columns([2, 3])
    with mode_col:
        st.radio(
            "P/E basis",
            ["Raw", "Normalized"],
            key="pe_valuation_mode",
            horizontal=True,
            help="**Raw** — P/E exactly as reported by NSE (standalone basis before the "
                 "FY21 transition, consolidated after).\n\n**Normalized** — pre-transition "
                 "history scaled to a consolidated-equivalent basis using an empirically "
                 "derived adjustment factor, so the full window is methodology-consistent.",
        )
    with badge_col:
        if _norm_ok:
            _conf = pe_norm_audit.get("confidence", "—")
            _conf_cls = {"high": "zone-under", "moderate": "zone-fair", "low": "zone-over"}.get(_conf, "zone-neutral")
            st.markdown(
                f'<div style="padding-top:1.9rem;">'
                f'<span class="zone-badge {_conf_cls}">factor ×{pe_norm_audit["factor"]:.3f} · '
                f'break {pe_norm_audit.get("break_date")} · {_conf} confidence</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="padding-top:1.9rem;"><span class="zone-badge zone-neutral">'
                'No methodology break detected — Normalized ≡ Raw</span></div>',
                unsafe_allow_html=True,
            )

    # STEP 8 — methodology disclosure (shown when the normalized basis is in effect).
    if pe_value_col == "PE_normalized" and _norm_ok:
        st.info(
            "**Normalized P/E methodology.** Historical P/E values prior to the NSE earnings "
            "methodology transition have been normalized using a median adjustment factor derived "
            "from overlapping standalone and consolidated valuation observations. This adjustment "
            "improves comparability across time but should be considered an approximation rather "
            "than a full historical earnings restatement.",
            icon="ℹ️",
        )

    # Lightweight by default (one fast call); opt in to the heavier guaranteed full
    # daily 10-year stitch. The choice is read at page load to fetch pe_data.
    _full_daily = st.toggle(
        "🔬 Use full daily 10-year history (slower, ~10 requests)",
        key="pe_full_daily",
        help="Default uses one lightweight request (fast, gentle on the data source). "
             "Enable for the guaranteed full daily distribution — heavier and may be "
             "rate-limited if used a lot.",
    )
    _n_obs = int(pe_data[pe_value_col].dropna().shape[0]) if (not pe_data.empty and pe_value_col in pe_data.columns) else 0
    _basis_word = "Normalized" if pe_value_col == "PE_normalized" else "Raw"
    st.caption(
        ("**Full daily** basis — " if _full_daily else "**Lightweight** basis (single request) — ")
        + f"{_n_obs:,} observations · **{_basis_word} P/E**."
        + ("" if _full_daily else " Toggle above for the guaranteed full daily distribution.")
    )

    # All P/E analytics below read `pe_value_col` (Raw or Normalized). The ranked
    # `current` is always the live consolidated NSE P/E in both modes.
    m = (valuation_metrics(pe_data[pe_value_col], current=nse_pe)
         if (not pe_data.empty and pe_value_col in pe_data.columns) else None)
    if m:
        zone_txt, zone_cls = valuation_zone(m["raw_pct"])
        roll = rolling_percentiles(pe_data, pe_value_col, m["current"])

        # ── Current valuation + zone ──
        p1, p2, p3 = st.columns([1, 1, 2])
        p1.metric("Current P/E", f"{m['current']:.2f}")
        p2.metric("Percentile", f"{m['raw_pct']:.1f}%",
                  help="% of daily readings over the last 10Y that were cheaper than today")
        with p3:
            st.markdown(
                f'<div style="padding-top:8px;"><div class="hero-name">Valuation Zone</div>'
                f'<span class="zone-badge {zone_cls}" style="margin-top:8px;display:inline-block;">{zone_txt}</span></div>',
                unsafe_allow_html=True,
            )
        st.caption(
            f"Based on **{m['n']:,} daily observations** · as of {pe_data['Date'].max():%b %d, %Y}. "
            f"Winsorized percentile (1% tails capped): **{m['wins_pct']:.1f}%**."
        )

        # ── STEP 9: NORMALIZATION VALIDATION PANEL ──
        # Always show the raw-vs-normalized comparison + audit metrics so the
        # adjustment is fully transparent (and its uncertainty visible).
        with st.expander("🔎 Normalization validation & audit", expanded=(pe_value_col == "PE_normalized")):
            _m_raw = valuation_metrics(pe_data["PE"], current=nse_pe) if "PE" in pe_data.columns else None
            _m_norm = (valuation_metrics(pe_data["PE_normalized"], current=nse_pe)
                       if "PE_normalized" in pe_data.columns else None)
            if not pe_norm_audit.get("normalized"):
                st.caption(
                    "No methodology break was detected for this index within the configured "
                    f"transition window ({pe_norm_audit.get('transition_window')}), so the normalized "
                    "series is identical to the raw series. Reason: "
                    f"`{pe_norm_audit.get('status')}`."
                )
            vc1, vc2 = st.columns(2)
            with vc1:
                st.markdown("###### Before normalization (raw)")
                if _m_raw:
                    st.metric("Mean P/E", f"{_m_raw['mean']:.2f}")
                    st.metric("Median P/E", f"{_m_raw['median']:.2f}")
                    st.metric("Current P/E percentile", f"{_m_raw['raw_pct']:.1f}%")
            with vc2:
                st.markdown("###### After normalization")
                if _m_norm:
                    _d_mean = _m_norm["mean"] - (_m_raw["mean"] if _m_raw else _m_norm["mean"])
                    _d_med = _m_norm["median"] - (_m_raw["median"] if _m_raw else _m_norm["median"])
                    _d_pct = _m_norm["raw_pct"] - (_m_raw["raw_pct"] if _m_raw else _m_norm["raw_pct"])
                    st.metric("Mean P/E", f"{_m_norm['mean']:.2f}", delta=f"{_d_mean:+.2f}")
                    st.metric("Median P/E", f"{_m_norm['median']:.2f}", delta=f"{_d_med:+.2f}")
                    st.metric("Current P/E percentile", f"{_m_norm['raw_pct']:.1f}%",
                              delta=f"{_d_pct:+.1f} pts", delta_color="off")

            _est = pe_norm_audit.get("estimate", {}) or {}
            a1, a2, a3 = st.columns(3)
            a1.metric("Adjustment factor", f"×{pe_norm_audit.get('factor', 1.0):.4f}")
            a1.caption(f"Basis: `{_est.get('basis', 'n/a')}` · confidence: "
                       f"**{pe_norm_audit.get('confidence', '—')}**")
            a2.metric("Factor observations", f"{_est.get('n_used', 0):,}")
            a2.caption(f"Outliers removed (IQR): **{_est.get('n_removed', 0):,}** of "
                       f"{_est.get('n_total', 0):,}")
            a3.metric("Transition / break", f"{pe_norm_audit.get('break_date', '—')}")
            a3.caption(f"Window: {pe_norm_audit.get('transition_window', '—')}")
            if _est.get("status") == "ok":
                st.caption(
                    f"Factor stats — median **{_est['median']:.4f}** · mean {_est['mean']:.4f} · "
                    f"std {_est['std']:.4f} · range [{_est['min']:.4f}, {_est['max']:.4f}] · "
                    f"rel-dispersion {(_est.get('rel_dispersion') or 0):.3f}. "
                    f"Computed {pe_norm_audit.get('timestamp', '')}."
                )
            if pe_norm_audit.get("confidence") == "low":
                st.warning(
                    "⚠️ Low-confidence factor — wide dispersion across the transition window "
                    "(typically COVID-distorted or subsidiary-heavy indices). Treat the normalized "
                    "percentile as indicative only.",
                    icon="⚠️",
                )

        # ── Percentile gauge + rolling-window percentiles ──
        gcol, rcol = st.columns([1, 1])
        with gcol:
            gfig = go.Figure(go.Indicator(
                mode="gauge+number", value=m["raw_pct"],
                number={"suffix": "%", "font": {"size": 32}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#9aa3b5"},
                    "bar": {"color": "rgba(0,0,0,0)"},
                    "steps": [
                        {"range": [0, 10], "color": "#14532d"},
                        {"range": [10, 25], "color": "#166534"},
                        {"range": [25, 75], "color": "#1e3a5f"},
                        {"range": [75, 90], "color": "#7c2d12"},
                        {"range": [90, 100], "color": "#7f1d1d"},
                    ],
                    "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 0.85, "value": m["raw_pct"]},
                },
            ))
            gfig.update_layout(height=230, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=0),
                               paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6e6e6"})
            st.plotly_chart(gfig, use_container_width=True, theme=None, config={"displayModeBar": False})
        with rcol:
            st.markdown("###### Percentile across market regimes")
            rk = st.columns(4)
            for col, lbl in zip(rk, ["3Y", "5Y", "10Y", "Full"]):
                v = roll.get(lbl)
                col.metric(lbl, f"{v:.0f}%" if v is not None else "—")
            st.caption("Today's P/E ranked within each look-back window — compares across regimes.")

        # ── Distribution statistics ──
        st.markdown("#### Historical P/E Statistics")
        sc = st.columns(5)
        sc[0].metric("Min", f"{m['min']:.2f}")
        sc[1].metric("Median", f"{m['median']:.2f}")
        sc[2].metric("Mean", f"{m['mean']:.2f}")
        sc[3].metric("Std Dev", f"{m['std']:.2f}")
        sc[4].metric("Max", f"{m['max']:.2f}")

        # ── Percentile boundaries ──
        st.markdown("#### Percentile Boundaries — P/E at each percentile")
        pc = st.columns(9)
        for col, p in zip(pc, [1, 5, 10, 25, 50, 75, 90, 95, 99]):
            col.metric(f"{p}th", f"{m['pcts'][p]:.1f}")

        # ── Distribution histogram ──
        st.markdown("#### Distribution — where today sits")
        dist_fig = go.Figure()
        dist_fig.add_trace(go.Histogram(x=m["series"], nbinsx=60, marker_color="#6366f1",
                                        opacity=0.7, name="Historical PE"))
        dist_fig.add_vline(x=m["current"], line_dash="dash", line_color="#ef4444", line_width=2,
                           annotation_text=f"Today: {m['current']:.2f}", annotation_position="top")
        dist_fig.add_vline(x=m["median"], line_dash="dot", line_color="#22c55e",
                           annotation_text=f"Median: {m['median']:.2f}", annotation_position="bottom")
        dist_fig.update_layout(height=340, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0),
                               xaxis_title="P/E Ratio", yaxis_title="Frequency (days)", showlegend=False,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        dist_fig.update_xaxes(showgrid=False)
        dist_fig.update_yaxes(showgrid=True, gridcolor="#232733")
        st.plotly_chart(dist_fig, use_container_width=True, theme=None, config={"displayModeBar": False})

        # ── Time series with 5 / 50 / 95 percentile lines + current ──
        st.markdown("#### P/E Over Time with Percentile Bands")
        band_fig = go.Figure()
        band_fig.add_trace(go.Scatter(x=pe_data["Date"], y=pe_data[pe_value_col],
                                      line=dict(color="#6366f1", width=1.4), name="P/E"))
        band_fig.add_hline(y=m["pcts"][95], line_dash="dash", line_color="#ef4444", annotation_text="95th %ile")
        band_fig.add_hline(y=m["pcts"][50], line_dash="dash", line_color="#22c55e", annotation_text="Median")
        band_fig.add_hline(y=m["pcts"][5], line_dash="dash", line_color="#3b82f6", annotation_text="5th %ile")
        band_fig.add_hline(y=m["current"], line_color="#ffffff", line_width=1.2, annotation_text="Today")
        band_fig.update_layout(height=340, template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0),
                               yaxis_title="P/E Ratio", showlegend=False,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        band_fig.update_xaxes(showgrid=False)
        band_fig.update_yaxes(showgrid=True, gridcolor="#232733")
        st.plotly_chart(band_fig, use_container_width=True, theme=None, config={"displayModeBar": False})

        # ── Same framework for P/B and Dividend Yield ──
        st.markdown("#### Other Valuation Metrics — percentile & band")
        mb = valuation_metrics(pe_data["PB"], current=nse_pb) if "PB" in pe_data.columns else None
        md = valuation_metrics(pe_data["DY"], current=nse_dy) if "DY" in pe_data.columns else None
        oc = st.columns(2)
        with oc[0]:
            if mb:
                z, zc = valuation_zone(mb["raw_pct"])
                st.metric("P/B Ratio", f"{mb['current']:.2f}", help=f"{mb['raw_pct']:.0f}th percentile of 10Y daily")
                st.markdown(f'<span class="zone-badge {zc}">{z} · {mb["raw_pct"]:.0f}%ile</span>', unsafe_allow_html=True)
            else:
                st.metric("P/B Ratio", f"{nse_pb:.2f}" if nse_pb else "—")
        with oc[1]:
            if md:
                # Dividend yield is inverted (higher yield = cheaper) → invert the band
                z, zc = valuation_zone(100 - md["raw_pct"])
                st.metric("Dividend Yield", f"{md['current']:.2f}%", help=f"{md['raw_pct']:.0f}th percentile of 10Y daily — higher yield = cheaper")
                st.markdown(f'<span class="zone-badge {zc}">{z} · {md["raw_pct"]:.0f}%ile yield</span>', unsafe_allow_html=True)
            else:
                st.metric("Dividend Yield", f"{nse_dy:.2f}%" if nse_dy else "—")
        st.caption("EV/EBITDA and Price/Sales: not available for indices from current free data sources "
                   "(the engine accepts them automatically if a source is added).")

        # ── Export the full valuation-metrics object (CSV + JSON) ──
        st.markdown("#### Export")

        def _metric_export(metr, roll, inverted=False):
            if not metr:
                return None
            pct = metr["raw_pct"]
            zlabel, _ = valuation_zone(100 - pct if inverted else pct)
            return {
                "current": round(metr["current"], 4),
                "percentile_raw": round(pct, 2),
                "percentile_winsorized": round(metr["wins_pct"], 2),
                "valuation_zone": zlabel,
                "inverted_band": inverted,
                "n_observations": metr["n"],
                "min": round(metr["min"], 4), "max": round(metr["max"], 4),
                "median": round(metr["median"], 4), "mean": round(metr["mean"], 4),
                "std": round(metr["std"], 4),
                "percentile_boundaries": {str(k): round(v, 4) for k, v in metr["pcts"].items()},
                "rolling_percentile": {k: (round(v, 2) if v is not None else None) for k, v in roll.items()},
            }

        rb = rolling_percentiles(pe_data, "PB", nse_pb) if mb else {"3Y": None, "5Y": None, "10Y": None, "Full": None}
        rd = rolling_percentiles(pe_data, "DY", nse_dy) if md else {"3Y": None, "5Y": None, "10Y": None, "Full": None}
        export = {
            "index": display_name,
            "as_of": f"{pe_data['Date'].max():%Y-%m-%d}",
            "generated_at_ist": ist_now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "current value: NSE allIndices · daily history: niftyindices (NSE)",
            "methodology": ("percentile = % of daily historical observations strictly below the "
                            "current value, over full daily history; NaN/±inf/non-positive removed"),
            "pe_basis": ("normalized" if pe_value_col == "PE_normalized" else "raw"),
            # Full audit of the methodology-break normalization (STEP 9 export).
            "pe_normalization": {
                "applied": pe_value_col == "PE_normalized",
                "break_detected": bool(pe_norm_audit.get("normalized")),
                "adjustment_factor": round(float(pe_norm_audit.get("factor", 1.0)), 6),
                "break_date": pe_norm_audit.get("break_date"),
                "confidence": pe_norm_audit.get("confidence"),
                "transition_window": pe_norm_audit.get("transition_window"),
                "factor_stats": pe_norm_audit.get("estimate"),
                "status": pe_norm_audit.get("status"),
            },
            "metrics": {
                "PE": _metric_export(m, roll),
                "PB": _metric_export(mb, rb),
                "DividendYield": _metric_export(md, rd, inverted=True),
            },
        }

        # Flatten to tidy long-format CSV (metric, field, value) — agent-friendly
        flat = []
        for mname, md_ in export["metrics"].items():
            if not md_:
                continue
            for k, v in md_.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        flat.append({"metric": mname, "field": f"{k}.{sk}", "value": sv})
                else:
                    flat.append({"metric": mname, "field": k, "value": v})
        csv_bytes = pd.DataFrame(flat, columns=["metric", "field", "value"]).to_csv(index=False).encode("utf-8")
        json_bytes = json.dumps(export, indent=2, default=str).encode("utf-8")

        slug = display_name.replace(" ", "_")
        ec1, ec2 = st.columns(2)
        ec1.download_button("⬇️ Valuation metrics (CSV)", data=csv_bytes,
                            file_name=f"{slug}_valuation_metrics.csv", mime="text/csv",
                            use_container_width=True)
        ec2.download_button("⬇️ Valuation metrics (JSON)", data=json_bytes,
                            file_name=f"{slug}_valuation_metrics.json", mime="application/json",
                            use_container_width=True)
    else:
        if nse_pe:
            st.metric("Current P/E (NSE, live)", f"{nse_pe:.2f}")
        data_notice(
            "Percentile unavailable",
            "No daily P/E history for this index (typical for some smallcap / sectoral indices on "
            "niftyindices.com), so a historical percentile can't be computed. Any current value "
            "above is live from NSE. Try Refresh in the sidebar.",
            kind="muted",
        )

# ═══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
foot_l, foot_r = st.columns([3, 1])
with foot_l:
    st.caption(
        "📊 Data sources: **Yahoo Finance** (OHLC, Volume) · **niftyindices.com** (P/E, P/B, DY). "
        "Indicative only — not investment advice."
    )
with foot_r:
    # Download button for current data
    csv = price_data.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download OHLCV CSV",
        data=csv,
        file_name=f"{display_name.replace(' ', '_')}_ohlcv.csv",
        mime="text/csv",
        use_container_width=True,
    )
