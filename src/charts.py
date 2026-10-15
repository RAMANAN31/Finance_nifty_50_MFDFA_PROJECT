"""Reusable Plotly figures."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def line_chart(data, y, title, y_title=None):
    fig = px.line(data, x=data.index, y=y, title=title, labels={"x": "Date", y: y_title or y})
    fig.update_layout(hovermode="x unified", margin=dict(l=15, r=15, t=55, b=15))
    return fig


def distribution_chart(returns):
    fig = px.histogram(x=returns, nbins=50, title="Daily log-return distribution", labels={"x": "Daily log return", "y": "Days"})
    fig.update_layout(margin=dict(l=15, r=15, t=55, b=15))
    return fig


def acf_chart(acf: pd.Series):
    fig = go.Figure(go.Bar(x=acf.index, y=acf.values, name="ACF"))
    fig.update_layout(title="Squared-return autocorrelation", xaxis_title="Lag (trading days)", yaxis_title="Autocorrelation", margin=dict(l=15, r=15, t=55, b=15))
    return fig
