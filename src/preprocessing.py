"""Input validation and return construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_prices(data: pd.DataFrame, date_col: str = "Date", price_col: str = "Close") -> pd.DataFrame:
    """Return a date-indexed, sorted frame with valid positive closing prices.

    Duplicate dates keep the last available observation. Invalid prices are dropped;
    a wholly invalid input raises a ValueError rather than silently proceeding.
    """
    if date_col not in data.columns or price_col not in data.columns:
        raise ValueError(f"Input must contain {date_col!r} and {price_col!r} columns")
    out = data[[date_col, price_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna(subset=[date_col, price_col])
    out = out[np.isfinite(out[price_col]) & (out[price_col] > 0)]
    out = out.sort_values(date_col).drop_duplicates(date_col, keep="last")
    if out.empty:
        raise ValueError("No valid positive closing prices were found")
    return out.set_index(date_col).rename_axis("Date")


def add_log_returns(prices: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add continuously compounded returns ln(P_t / P_(t-1))."""
    if price_col not in prices.columns:
        raise ValueError(f"Missing price column {price_col!r}")
    out = prices.copy()
    out["Log Return"] = np.log(out[price_col] / out[price_col].shift(1))
    return out
