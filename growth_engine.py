"""
═══════════════════════════════════════════════════════════════════════════════
  growth_engine.py — Fundamental Growth Calculation Engine
═══════════════════════════════════════════════════════════════════════════════

  PURE calculation layer for fundamental growth metrics (Revenue, EBITDA, PAT,
  EPS, Operating Profit, Cash Flow). NO network I/O, NO Streamlit, NO data-source
  knowledge — it consumes normalized period→value series and returns numbers.

  This separation is deliberate (per spec): future real-time / paid data sources
  can be plugged into the data adapter without touching this engine.

  Input contract
  --------------
  A metric's history is a pandas Series indexed by period-end Timestamp, sorted
  ASCENDING (oldest → newest). Quarterly series for YoY/QoQ; annual series for
  CAGR. Missing periods may be absent or NaN — every function degrades to None
  rather than raising.

  All growth figures are returned in PERCENT.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Data-quality primitives
# ─────────────────────────────────────────────────────────────────────────────
def _num(x):
    """Coerce to a finite float, else None (handles None/NaN/inf/strings)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _clean_series(series: pd.Series) -> pd.Series:
    """Drop NaN/±inf, keep chronological order. Returns empty Series on bad input."""
    if series is None or not isinstance(series, pd.Series) or series.empty:
        return pd.Series(dtype=float)
    s = pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
    try:
        s = s.sort_index()
    except Exception:
        pass
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  Generic growth + CAGR (spec formulas)
# ─────────────────────────────────────────────────────────────────────────────
def growth_rate(current, previous, allow_negative_base: bool = False):
    """
    Growth% = (current - previous) / previous * 100.
    Returns None for a zero / missing / (by default) negative base — a % change
    from a negative or zero base (e.g. a prior-period loss) is not meaningful.
    Set allow_negative_base=True to compute it anyway.
    """
    c, p = _num(current), _num(previous)
    if c is None or p is None or p == 0:
        return None
    if p < 0 and not allow_negative_base:
        return None
    return (c - p) / p * 100.0


def cagr(beginning, ending, years):
    """
    CAGR% = (ending / beginning) ** (1/years) - 1, in percent.
    Returns None unless beginning > 0, ending > 0 and years > 0 (roots of
    non-positive values are undefined / not meaningful for earnings).
    """
    b, e = _num(beginning), _num(ending)
    y = _num(years)
    if b is None or e is None or y is None or b <= 0 or e <= 0 or y <= 0:
        return None
    return ((e / b) ** (1.0 / y) - 1.0) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
#  Series-based growth (YoY / QoQ / CAGR)
# ─────────────────────────────────────────────────────────────────────────────
def yoy(series: pd.Series, periods_per_year: int = 4, **kw):
    """YoY: latest vs the same period one year earlier (4 quarters back, default)."""
    s = _clean_series(series)
    if len(s) <= periods_per_year:
        return None
    return growth_rate(s.iloc[-1], s.iloc[-1 - periods_per_year], **kw)


def qoq(series: pd.Series, **kw):
    """QoQ: latest period vs the immediately preceding period."""
    s = _clean_series(series)
    if len(s) < 2:
        return None
    return growth_rate(s.iloc[-1], s.iloc[-2], **kw)


def cagr_series(annual_series: pd.Series, years: int):
    """CAGR over `years` using an ANNUAL series' endpoints (needs years+1 points)."""
    s = _clean_series(annual_series)
    if len(s) <= years:
        return None
    return cagr(s.iloc[-1 - years], s.iloc[-1], years)


def growth_trend(series: pd.Series, periods_per_year: int = 4, stable_band: float = 2.0):
    """
    Classify momentum for the heat-map: compares the latest YoY growth to the
    previous period's YoY growth.
      'accelerating' : YoY improved by more than stable_band points
      'declining'    : YoY worsened by more than stable_band points
      'stable'       : within ±stable_band
    Returns None if there isn't enough data.
    """
    s = _clean_series(series)
    if len(s) <= periods_per_year + 1:
        return None
    cur = growth_rate(s.iloc[-1], s.iloc[-1 - periods_per_year])
    prev = growth_rate(s.iloc[-2], s.iloc[-2 - periods_per_year])
    if cur is None or prev is None:
        return None
    diff = cur - prev
    if diff > stable_band:
        return "accelerating"
    if diff < -stable_band:
        return "declining"
    return "stable"


# ─────────────────────────────────────────────────────────────────────────────
#  Index / sector aggregation — SUM constituents first, THEN compute growth
#  (never average percentages, never market-cap weight)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_periodic(constituent_series: list) -> pd.Series:
    """
    Sum a metric across constituents per period (aligned on period-end index).
    A period's total is NaN only if no constituent reported it. Coverage should
    be reported separately so callers know how complete each period is.
    """
    frames = [_clean_series(s) for s in (constituent_series or [])]
    frames = [s for s in frames if not s.empty]
    if not frames:
        return pd.Series(dtype=float)
    wide = pd.concat(frames, axis=1)
    return wide.sum(axis=1, min_count=1).sort_index()


def coverage(constituent_series: list) -> float:
    """Fraction of constituents that supplied a usable series (0..1)."""
    total = len(constituent_series or [])
    if total == 0:
        return 0.0
    have = sum(1 for s in constituent_series if not _clean_series(s).empty)
    return have / total


# ─────────────────────────────────────────────────────────────────────────────
#  High-level metric block (what UI cards/tables consume)
# ─────────────────────────────────────────────────────────────────────────────
def metric_block(quarterly: pd.Series, annual: pd.Series = None, allow_negative_base: bool = False) -> dict:
    """
    Full growth summary for one metric. UI components read this dict; they never
    compute growth themselves.
    """
    q = _clean_series(quarterly)
    a = _clean_series(annual)
    return {
        "current": (float(q.iloc[-1]) if not q.empty else (float(a.iloc[-1]) if not a.empty else None)),
        "current_period": (str(q.index[-1].date()) if not q.empty and hasattr(q.index[-1], "date")
                           else (str(q.index[-1]) if not q.empty else None)),
        "yoy": yoy(q, allow_negative_base=allow_negative_base),
        "qoq": qoq(q, allow_negative_base=allow_negative_base),
        "cagr_3y": cagr_series(a, 3),
        "cagr_5y": cagr_series(a, 5),
        "trend": growth_trend(q),
        "n_quarters": int(len(q)),
        "n_years": int(len(a)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Self-test — run `python growth_engine.py`
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    assert round(growth_rate(110, 100), 2) == 10.0
    assert growth_rate(10, 0) is None                 # zero denominator
    assert growth_rate(10, -5) is None                # negative base → None
    assert round(cagr(100, 200, 5), 2) == 14.87       # standard CAGR
    assert cagr(0, 200, 5) is None and cagr(100, -5, 5) is None

    # spec aggregate example: sum first, then grow
    cur = aggregate_periodic([pd.Series({pd.Timestamp("2026-03-31"): v}) for v in (100, 200, 300)])
    prev = aggregate_periodic([pd.Series({pd.Timestamp("2025-03-31"): v}) for v in (90, 180, 250)])
    assert cur.iloc[-1] == 600 and prev.iloc[-1] == 520
    assert round(growth_rate(cur.iloc[-1], prev.iloc[-1]), 2) == 15.38

    # YoY / QoQ / trend on a quarterly series
    idx = pd.to_datetime(["2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30",
                          "2025-09-30", "2025-12-31", "2026-03-31"])
    rev = pd.Series([100, 105, 110, 115, 120, 130, 140], index=idx)
    assert round(yoy(rev), 2) == 27.27   # 140 vs 110 (4q back)
    assert round(qoq(rev), 2) == 7.69    # 140 vs 130
    assert growth_trend(rev) in {"accelerating", "stable", "declining"}

    print("growth_engine self-test: ALL PASSED")
