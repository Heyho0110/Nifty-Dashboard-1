"""
═══════════════════════════════════════════════════════════════════════════════
  growth_data.py — Data adapter for the Growth engine
═══════════════════════════════════════════════════════════════════════════════

  Network + caching layer ONLY. It fetches:
    • index/sector constituent lists  (NSE archive CSVs)
    • per-stock fundamentals          (yfinance income statements)

  It returns normalized pandas Series (period-end → value, ascending) that the
  pure growth_engine consumes. No growth math lives here. Swapping in a different
  / real-time fundamentals source later means changing only this file.
═══════════════════════════════════════════════════════════════════════════════
"""

import io
import warnings

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# niftyindices name → NSE constituent-list CSV (verified live)
CONSTITUENT_FILES = {
    "NIFTY 50": "ind_nifty50list.csv",
    "NIFTY 100": "ind_nifty100list.csv",
    "NIFTY 200": "ind_nifty200list.csv",
    "NIFTY 500": "ind_nifty500list.csv",
    "NIFTY MIDCAP 150": "ind_niftymidcap150list.csv",
    "NIFTY SMALLCAP 250": "ind_niftysmallcap250list.csv",
    "NIFTY NEXT 50": "ind_niftynext50list.csv",
    "NIFTY AUTO": "ind_niftyautolist.csv",
    "NIFTY BANK": "ind_niftybanklist.csv",
    "NIFTY ENERGY": "ind_niftyenergylist.csv",
    "NIFTY FINANCIAL SERVICES": "ind_niftyfinancialservices25_50list.csv",
    "NIFTY FMCG": "ind_niftyfmcglist.csv",
    "NIFTY INFRASTRUCTURE": "ind_niftyinfralist.csv",
    "NIFTY IT": "ind_niftyitlist.csv",
    "NIFTY MEDIA": "ind_niftymedialist.csv",
    "NIFTY METAL": "ind_niftymetallist.csv",
    "NIFTY PHARMA": "ind_niftypharmalist.csv",
    "NIFTY PSU BANK": "ind_niftypsubanklist.csv",
    "NIFTY REALTY": "ind_niftyrealtylist.csv",
}


def has_constituents(ni_name: str) -> bool:
    return ni_name in CONSTITUENT_FILES


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_constituents(ni_name: str) -> list:
    """List of '<SYMBOL>.NS' tickers for an index/sector, from the NSE archive CSV."""
    fname = CONSTITUENT_FILES.get(ni_name)
    if not fname:
        return []
    try:
        s = requests.Session()
        s.headers.update({"User-Agent": _UA})
        r = s.get("https://archives.nseindia.com/content/indices/" + fname, timeout=15)
        if r.status_code != 200:
            return []
        df = pd.read_csv(io.StringIO(r.text))
        if "Symbol" not in df.columns:
            return []
        return [f"{str(sym).strip()}.NS" for sym in df["Symbol"].dropna()]
    except Exception:
        return []


def _row(stmt, name) -> pd.Series:
    """Extract one income-statement row as an ascending period→value Series."""
    try:
        if stmt is None or name not in stmt.index:
            return None
        s = pd.to_numeric(stmt.loc[name], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()].sort_index()
        return s if len(s) else None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stock_fundamentals(symbol: str) -> dict:
    """
    Per-stock Revenue & PAT (Net Income) as quarterly + annual ascending Series.
    Returns {revenue_q, revenue_a, pat_q, pat_a} with None where unavailable.
    Cached 24h (fundamentals change at most quarterly).
    """
    out = {"revenue_q": None, "revenue_a": None, "pat_q": None, "pat_a": None}
    try:
        t = yf.Ticker(symbol)
        qi, ai = t.quarterly_income_stmt, t.income_stmt
        out["revenue_q"] = _row(qi, "Total Revenue")
        out["revenue_a"] = _row(ai, "Total Revenue")
        out["pat_q"] = _row(qi, "Net Income")
        out["pat_a"] = _row(ai, "Net Income")
    except Exception:
        pass
    return out
