# NIFTY 50 Market Risk & Volatility Intelligence Dashboard

An interactive Streamlit application for descriptive analysis of historical NIFTY 50 returns, volatility, drawdowns and downside risk, with an optional Multifractal Detrended Fluctuation Analysis (MFDFA) research module.

## Features

- Yahoo Finance (`^NSEI`) retrieval with a clearly identified offline synthetic fallback.
- Upload your own CSV in the sidebar using `Date` and `Close` columns; the app explains the accepted format and data-cleaning behavior there.
- Date filters, index and log-return charts, 21-observation rolling annualized volatility, return distribution and squared-return autocorrelation.
- Annual and selected-period risk summaries, maximum drawdown calculated from index levels, historical 95% VaR displayed as a positive loss threshold, and zero-target downside deviation.
- Period comparison, CSV downloads, and optional MFDFA outputs for H(q), Fq(s), τ(q), f(α) and Δα.
- Metric definitions and limitations in the interface.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The data provider must be reachable for actual NIFTY 50 data. If retrieval fails, the app uses `data/nifty50_sample.csv`, a small synthetic illustrative sample. The source and retrieval status are shown in the dashboard. Do not present the fallback as market history.

To analyze your own price history, upload a CSV in the sidebar with at least two rows and columns named `Date` and `Close`. Dates must be parseable and prices positive numeric values. The app sorts rows, keeps the last row for duplicate dates, and drops invalid dates/prices before calculating returns. The upload replaces the default data for the current session.

## Calculations

- Daily return: `ln(P_t / P_(t-1))`; daily returns are stored in decimal units.
- Annualized volatility: sample standard deviation × `sqrt(252)`.
- Maximum drawdown: minimum of index level / running peak − 1.
- Historical 95% VaR: negative of the 5th percentile daily log return, floored at zero so the displayed threshold is a positive loss amount.
- Downside deviation: square root of the average squared shortfall below the zero daily-return target.
- MFDFA: integrate demeaned returns, detrend forward and reverse non-overlapping segments, estimate Fq(s), and regress log Fq against log scale. The estimated spectrum and Hurst exponents depend on sample, scale and detrending choices.

H(2) and Δα are scaling estimates, not conclusive market-efficiency tests. Historical VaR does not bound future losses. This project does not forecast prices or provide investment recommendations.

## Validation and reproducibility

Run `pytest`. The test suite checks sorting, duplicate handling, positive prices, hand-calculated log returns, annualization, drawdown, VaR, downside deviation and basic MFDFA output shape/finite-value conditions. Report reference values are comparison targets only and are not hardcoded into calculations.

Record the retrieval date/source displayed by the dashboard, Python and dependency versions, selected dates, trading-day convention, MFDFA q values, scales and polynomial order when comparing results. The bundled sample is synthetic and suitable only for a smoke demonstration.

## Deployment

Deploy `app.py` to Streamlit Community Cloud with `requirements.txt` as the dependency file. A deployed app can fall back to the included sample when Yahoo Finance cannot be reached; the source label makes this visible.
