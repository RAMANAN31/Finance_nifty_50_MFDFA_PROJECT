"""Historical market data retrieval with an explicit local fallback."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_market_data(start="2021-01-01", end=None) -> tuple[pd.DataFrame, str, str]:
    """Try Yahoo Finance (^NSEI); fall back to the clearly labelled sample file."""
    end = end or (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        import yfinance as yf
        raw = yf.download("^NSEI", start=start, end=end, auto_adjust=False, progress=False, timeout=10)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index()
            from .preprocessing import clean_prices
            cleaned = clean_prices(raw, "Date", "Close")
            return cleaned, "Yahoo Finance (^NSEI)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    fallback = DATA_DIR / "nifty50_sample.csv"
    if not fallback.exists():
        raise RuntimeError("Yahoo Finance is unavailable and no fallback CSV is installed")
    from .preprocessing import clean_prices
    cleaned = clean_prices(pd.read_csv(fallback), "Date", "Close")
    return cleaned, "Bundled synthetic sample (illustrative only)", "Static sample; not a live retrieval"
