"""NIFTY 50 Market Risk & Volatility Intelligence Dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from importlib.metadata import PackageNotFoundError, version

from src.charts import acf_chart, distribution_chart, line_chart
from src.data_loader import load_market_data
from src.insights import risk_interpretations
from src.metrics import drawdown_series, maximum_drawdown, summarize_returns, yearly_summary
from src.mfdfa_analysis import mfdfa
from src.preprocessing import add_log_returns
from src.volatility import rolling_annualized_volatility, squared_return_acf

st.set_page_config(page_title="NIFTY 50 Risk Intelligence", page_icon="📈", layout="wide")
st.title("NIFTY 50 Market Risk & Volatility Intelligence Dashboard")
st.caption("Historical descriptive analytics and an optional MFDFA research lab. This dashboard does not forecast prices or provide trading recommendations.")


@st.cache_data(ttl=3600, show_spinner="Loading historical index data…")
def get_data():
    return load_market_data()


with st.sidebar:
    st.header("Add your own data")
    st.write("Upload a CSV to analyze your own index or asset price history. Your file is used for this session and is not saved by the app.")
    with st.expander("CSV format and how to use it", expanded=True):
        st.markdown("""
        1. Prepare a CSV with two columns named **Date** and **Close**.
        2. Put one observation per row and include at least two valid dates. Dates should be parseable (for example, `2024-01-31`); `Close` must be a positive numeric price or index level.
        3. Choose the CSV below. The app sorts dates, keeps the last row for duplicate dates, and drops rows with invalid dates or prices.
        4. Use the date filter below to choose the analysis period. Returns are calculated first, so the first selected return uses the prior available close.

        The uploaded data replaces the built-in source for the current session. Clear the upload to return to Yahoo Finance or the labelled sample fallback.
        """)
    uploaded_file = st.file_uploader("Choose a price-history CSV", type=["csv"], help="Required columns: Date and Close")

if uploaded_file is not None:
    try:
        raw_upload = pd.read_csv(uploaded_file)
        from src.preprocessing import clean_prices
        prices = clean_prices(raw_upload, "Date", "Close")
        if len(prices) < 2:
            raise ValueError("At least two valid dated price observations are required to calculate returns")
        data_source = f"User-uploaded CSV ({uploaded_file.name})"
        retrieved = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC; uploaded this session")
        st.sidebar.success(f"Loaded {len(prices):,} valid unique dates from {len(raw_upload):,} CSV rows.")
    except Exception as exc:
        st.error(f"Could not use the uploaded CSV: {exc}. Check that it has Date and Close columns and at least two valid positive prices.")
        st.stop()
else:
    try:
        prices, data_source, retrieved = get_data()
    except Exception as exc:
        st.error(f"Could not load market data: {exc}")
        st.stop()

base = add_log_returns(prices)
base["Drawdown"] = drawdown_series(base["Close"])
base["Rolling Volatility"] = rolling_annualized_volatility(base["Log Return"])
st.caption(f"Data source: **{data_source}** · Retrieval: **{retrieved}** · Annualization convention: **252 trading days**")
def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"
st.caption("Runtime versions: Python " + ".".join(map(str, __import__("sys").version_info[:3])) +
           " · pandas " + package_version("pandas") + " · NumPy " + package_version("numpy") +
           " · Plotly " + package_version("plotly") + " · Streamlit " + package_version("streamlit"))

min_date, max_date = base.index.min().date(), base.index.max().date()
with st.sidebar:
    st.header("Filters")
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        start_date = end_date = date_range
    else:
        start_date, end_date = date_range
    st.caption("Daily returns are calculated on the complete price series before filtering, so the first selected return uses the immediately preceding available close.")

selected = base.loc[(base.index.date >= start_date) & (base.index.date <= end_date)].copy()
ret = selected["Log Return"].dropna()
if ret.empty:
    st.warning("The selected date range needs at least two price observations to calculate return metrics.")
    st.stop()
stats = summarize_returns(ret)
max_dd = maximum_drawdown(selected["Close"])

overview, volatility, risk, compare, research = st.tabs(["Market Overview", "Volatility & Returns", "Risk Assessment", "Period Comparison", "MFDFA Research Lab"])

with overview:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Index change", f"{selected['Close'].iloc[-1] / selected['Close'].iloc[0] - 1:+.2%}")
    k2.metric("Mean daily log return", f"{stats['mean_daily_return']:+.3%}")
    k3.metric("Annualized volatility", f"{stats['annualized_volatility']:.2%}")
    k4.metric("Trading observations", f"{len(selected):,}")
    st.plotly_chart(line_chart(selected, "Close", "NIFTY 50 closing level", "Index points"), width="stretch")
    st.plotly_chart(line_chart(selected.dropna(subset=["Log Return"]), "Log Return", "Daily log returns", "Log return"), width="stretch")

with volatility:
    left, right = st.columns(2)
    left.plotly_chart(line_chart(selected, "Rolling Volatility", "21-trading-day rolling annualized volatility", "Annualized volatility"), width="stretch")
    right.plotly_chart(distribution_chart(ret), width="stretch")
    st.subheader("Year-wise return statistics")
    year_data = base.loc[(base.index.date >= start_date) & (base.index.date <= end_date)]
    st.dataframe(yearly_summary(year_data).style.format({"Mean Return": "{:.4%}", "Std Dev": "{:.4%}", "Skewness": "{:.3f}", "Excess Kurtosis": "{:.3f}", "Annualized Volatility": "{:.2%}", "Maximum Drawdown": "{:.2%}"}), width="stretch", hide_index=True)
    st.plotly_chart(acf_chart(squared_return_acf(ret)), width="stretch")
    st.caption("Positive autocorrelation in squared returns can be consistent with volatility clustering; it is a diagnostic, not a causal test.")

with risk:
    left, right = st.columns(2)
    left.metric("Maximum drawdown", f"{max_dd:.2%}")
    right.metric("Historical 95% VaR", f"{stats['historical_var_95']:.2%}", help="Positive empirical loss threshold: negative of the 5th percentile daily log return. It is not a maximum-loss guarantee.")
    left2, right2 = st.columns(2)
    left2.metric("Downside deviation", f"{stats['downside_deviation']:.2%}", help="Root mean square of daily returns below the zero-return target.")
    right2.metric("Cumulative return", f"{stats['total_return']:+.2%}")
    st.plotly_chart(line_chart(selected, "Drawdown", "Drawdown from running index peak", "Drawdown"), width="stretch")
    cumulative = np.expm1(selected["Log Return"].fillna(0).cumsum())
    st.plotly_chart(px.line(x=selected.index, y=cumulative, title="Cumulative return from log returns", labels={"x": "Date", "y": "Cumulative return"}), width="stretch")
    st.subheader("Metric interpretation")
    for name, explanation in risk_interpretations(stats, max_dd).items():
        st.markdown(f"**{name}.** {explanation}")

with compare:
    st.subheader("Calendar-year comparison")
    all_years = yearly_summary(base)
    if not all_years.empty:
        all_years["Volatility Rank"] = all_years["Annualized Volatility"].rank(ascending=False, method="min").astype(int)
        all_years["Drawdown Rank"] = all_years["Maximum Drawdown"].rank(ascending=True, method="min").astype(int)
    st.dataframe(all_years.style.format({"Mean Return": "{:.4%}", "Std Dev": "{:.4%}", "Skewness": "{:.3f}", "Excess Kurtosis": "{:.3f}", "Annualized Volatility": "{:.2%}", "Maximum Drawdown": "{:.2%}"}), width="stretch", hide_index=True)
    st.subheader("Selected-period summary")
    summary_table = pd.DataFrame([{"Start": str(start_date), "End": str(end_date), "Observations": stats["observations"], "Mean daily return": stats["mean_daily_return"], "Annualized volatility": stats["annualized_volatility"], "Maximum drawdown": max_dd, "Historical VaR 95%": stats["historical_var_95"], "Downside deviation": stats["downside_deviation"], "Cumulative return": stats["total_return"]}])
    st.dataframe(summary_table, width="stretch", hide_index=True)
    st.download_button("Download filtered prices and returns (CSV)", selected.reset_index().to_csv(index=False).encode("utf-8"), "nifty50_filtered_data.csv", "text/csv")
    st.download_button("Download selected-period summary (CSV)", summary_table.to_csv(index=False).encode("utf-8"), "nifty50_period_summary.csv", "text/csv")

with research:
    st.subheader("Multifractal Detrended Fluctuation Analysis")
    st.write("MFDFA estimates how fluctuation size scales with segment length. H(q) describes scaling at different fluctuation orders; q=2 is commonly used to discuss persistence. Values above 0.5 indicate persistence, below 0.5 anti-persistence, and near 0.5 random-walk-like scaling under the report's interpretation. These estimates do not prove or disprove market efficiency.")
    st.caption("Method: demeaned return profile, forward and reverse non-overlapping segments, polynomial detrending, log-log slope across geometrically spaced scales. Spectrum width depends on sample length, tails and these implementation choices.")
    if len(ret) < 32:
        st.info("MFDFA requires at least 32 finite returns. The core market-risk dashboard remains available.")
    else:
        order = st.selectbox("Detrending polynomial order", [1, 2, 3], index=0)
        try:
            result = mfdfa(ret.to_numpy(), order=order)
            idx2 = int(np.argmin(np.abs(result["q"] - 2)))
            m1, m2 = st.columns(2)
            m1.metric("H(2)", f"{result['Hq'][idx2]:.4f}")
            m2.metric("Estimated spectrum width Δα", f"{float(result['spectrum_width']):.4f}")
            hdf = pd.DataFrame({"q": result["q"], "H(q)": result["Hq"]})
            st.plotly_chart(px.line(hdf, x="q", y="H(q)", markers=True, title="Generalized Hurst exponent H(q)"), width="stretch")
            fqdf = pd.DataFrame(result["Fq"].T, columns=[f"q={q:g}" for q in result["q"]]); fqdf.insert(0, "Scale", result["scales"])
            st.plotly_chart(px.line(fqdf, x="Scale", y=fqdf.columns[1:], log_x=True, log_y=True, title="Scaling function Fq(s)"), width="stretch")
            tau_df = pd.DataFrame({"q": result["q"], "τ(q)": result["tau"]})
            st.plotly_chart(px.line(tau_df, x="q", y="τ(q)", markers=True, title="Mass exponent τ(q)"), width="stretch")
            spec_df = pd.DataFrame({"α": result["alpha"], "f(α)": result["f_alpha"]}).sort_values("α")
            st.plotly_chart(px.line(spec_df, x="α", y="f(α)", markers=True, title="Estimated multifractal spectrum"), width="stretch")
            mfdfa_table = pd.DataFrame({"q": result["q"], "H(q)": result["Hq"], "tau(q)": result["tau"], "alpha": result["alpha"], "f(alpha)": result["f_alpha"]})
            mfdfa_table["data_source"] = data_source
            mfdfa_table["retrieval"] = retrieved
            mfdfa_table["period_start"] = str(start_date)
            mfdfa_table["period_end"] = str(end_date)
            mfdfa_table["detrending_order"] = order
            mfdfa_table["scales_used"] = ";".join(map(str, result["scales"]))
            mfdfa_table["annualization_days"] = 252
            st.download_button("Download MFDFA outputs (CSV)", mfdfa_table.to_csv(index=False).encode(), "nifty50_mfdfa.csv", "text/csv")
            st.caption("Report comparison targets (initial reference only; not hardcoded into the analysis): 2021 H(2)=0.3237, Δα=0.5864; 2022 0.5434, 0.5019; 2023 0.5070, 0.1539; 2021–2023 0.4931, 0.4623. Differences should be investigated across data, scales, detrending and fitting choices.")
        except ValueError as exc:
            st.warning(f"MFDFA could not be estimated for this selection: {exc}")
