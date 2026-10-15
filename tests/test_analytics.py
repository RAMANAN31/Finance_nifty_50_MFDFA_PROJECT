import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
import sys

from src.data_loader import load_market_data
from src.metrics import drawdown_series, maximum_drawdown, summarize_returns, yearly_summary
from src.mfdfa_analysis import mfdfa
from src.preprocessing import add_log_returns, clean_prices
from src.volatility import rolling_annualized_volatility


def test_clean_prices_sorts_deduplicates_and_drops_bad_prices():
    raw = pd.DataFrame({"Date": ["2021-01-02", "2021-01-01", "2021-01-02", "bad", "2021-01-03"], "Close": [12, 10, 13, 9, -1]})
    out = clean_prices(raw)
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert out["Close"].tolist() == [10, 13]


def test_log_return_matches_hand_calculation():
    prices = clean_prices(pd.DataFrame({"Date": ["2021-01-01", "2021-01-02", "2021-01-03"], "Close": [100, 110, 99]}))
    actual = add_log_returns(prices)["Log Return"].dropna().to_numpy()
    np.testing.assert_allclose(actual, [np.log(1.1), np.log(0.9)])


def test_risk_metrics_known_series():
    stats = summarize_returns(pd.Series([-0.10, 0.0, 0.10]))
    assert stats["historical_var_95"] == pytest.approx(0.09)
    assert stats["downside_deviation"] == pytest.approx(0.10 / np.sqrt(3))
    assert stats["annualized_volatility"] == pytest.approx(np.std([-0.1, 0, 0.1], ddof=1) * np.sqrt(252))
    assert maximum_drawdown(pd.Series([100, 120, 90, 99])) == pytest.approx(-0.25)
    assert drawdown_series(pd.Series([100, 120, 90, 99])).iloc[-1] == pytest.approx(-0.175)


def test_yearly_groups_and_rolling_volatility():
    idx = pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-03"])
    ret = pd.Series([0.01, -0.02, 0.03], index=idx)
    roll = rolling_annualized_volatility(ret, window=2)
    assert roll.iloc[0] != roll.iloc[0]
    assert roll.iloc[1] == pytest.approx(np.std([.01, -.02], ddof=1) * np.sqrt(252))
    frame = pd.DataFrame({"Close": [100, 101, 99], "Log Return": [np.nan, .01, -.02]}, index=idx)
    summary = yearly_summary(frame)
    assert summary["Year"].tolist() == [2021, 2022]
    assert summary["Observations"].tolist() == [1, 1]


def test_mfdfa_outputs_valid_shapes_and_increasing_scales():
    x = np.random.default_rng(7).normal(size=400)
    out = mfdfa(x)
    assert np.all(np.diff(out["scales"]) > 0)
    assert out["Fq"].shape == (len(out["q"]), len(out["scales"]))
    for key in ["Hq", "tau", "alpha", "f_alpha"]:
        assert out[key].shape == out["q"].shape and np.isfinite(out[key]).all()
    assert np.isfinite(out["spectrum_width"])


def test_mfdfa_rejects_short_samples_and_constant_prices_are_finite():
    with pytest.raises(ValueError):
        mfdfa(np.arange(20))
    prices = clean_prices(pd.DataFrame({"Date": pd.date_range("2021-01-01", periods=40), "Close": np.full(40, 100.)}))
    returns = add_log_returns(prices)["Log Return"].dropna()
    assert np.isfinite(returns).all() and (returns == 0).all()


def test_failed_download_uses_labeled_sample_fallback(monkeypatch):
    def fail_download(*args, **kwargs):
        raise RuntimeError("offline")
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fail_download))
    frame, source, retrieved = load_market_data()
    assert len(frame) == 10
    assert "synthetic" in source.lower()
    assert "static" in retrieved.lower()
