"""Descriptive and downside-risk metrics; returns are decimal log returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summarize_returns(returns: pd.Series, periods_per_year: int = TRADING_DAYS, target: float = 0.0) -> dict[str, float]:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        raise ValueError("At least one finite return is required")
    downside = np.minimum(r.to_numpy() - target, 0.0)
    return {
        "observations": int(r.size),
        "mean_daily_return": float(r.mean()),
        "annualized_volatility": float(r.std(ddof=1) * np.sqrt(periods_per_year)) if r.size > 1 else 0.0,
        "historical_var_95": float(max(0.0, -np.quantile(r, 0.05))),
        "downside_deviation": float(np.sqrt(np.mean(np.square(downside)))),
        "total_return": float(np.expm1(r.sum())),
        "skewness": float(r.skew()) if r.size > 2 else 0.0,
        "excess_kurtosis": float(r.kurt()) if r.size > 3 else 0.0,
    }


def drawdown_series(prices: pd.Series) -> pd.Series:
    p = pd.Series(prices, dtype=float)
    p = p.replace([np.inf, -np.inf], np.nan).dropna()
    if p.empty or (p <= 0).any():
        raise ValueError("Drawdown requires non-empty, positive index levels")
    return p / p.cummax() - 1.0


def maximum_drawdown(prices: pd.Series) -> float:
    return float(drawdown_series(prices).min())


def yearly_summary(frame: pd.DataFrame, return_col: str = "Log Return") -> pd.DataFrame:
    valid = frame.dropna(subset=[return_col]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["Year", "Observations", "Mean Return", "Std Dev", "Skewness", "Excess Kurtosis", "Annualized Volatility", "Maximum Drawdown"])
    valid["Year"] = valid.index.year
    rows = []
    for year, group in valid.groupby("Year", sort=True):
        stats = summarize_returns(group[return_col])
        rows.append({"Year": int(year), "Observations": stats["observations"], "Mean Return": stats["mean_daily_return"],
                     "Std Dev": float(group[return_col].std(ddof=1)) if len(group) > 1 else 0.0,
                     "Skewness": stats["skewness"], "Excess Kurtosis": stats["excess_kurtosis"],
                     "Annualized Volatility": stats["annualized_volatility"], "Maximum Drawdown": maximum_drawdown(group["Close"])})
    return pd.DataFrame(rows)
