"""Rolling volatility and squared-return autocorrelation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS


def rolling_annualized_volatility(returns: pd.Series, window: int = 21, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    if window < 2:
        raise ValueError("window must be at least 2")
    r = pd.Series(returns, dtype=float)
    return r.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)


def squared_return_acf(returns: pd.Series, nlags: int = 20) -> pd.Series:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if nlags < 0:
        raise ValueError("nlags must be non-negative")
    if r.size <= nlags + 1:
        nlags = max(0, r.size - 1)
    squared = r.pow(2)
    if squared.nunique() <= 1:
        return pd.Series([1.0] + [0.0] * nlags, index=range(nlags + 1), name="ACF")
    centered = squared - squared.mean()
    denom = float(np.dot(centered, centered))
    vals = [1.0] + [float(np.dot(centered.iloc[lag:].to_numpy(), centered.iloc[:-lag].to_numpy()) / denom) for lag in range(1, nlags + 1)]
    return pd.Series(vals, index=range(nlags + 1), name="ACF")
