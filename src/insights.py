"""Short rule-based descriptions for historical descriptive metrics."""


def risk_interpretations(stats: dict, max_dd: float) -> dict[str, str]:
    vol = stats["annualized_volatility"]
    var = stats["historical_var_95"]
    down = stats["downside_deviation"]
    return {
        "Volatility": f"Daily return dispersion annualizes to {vol:.1%} using 252 trading days; this describes historical variability, not a forecast.",
        "Historical VaR (95%)": f"The empirical 5th-percentile daily return corresponds to a loss threshold of {var:.2%}; losses can exceed this amount.",
        "Maximum drawdown": f"The largest peak-to-trough decline in the selected index levels was {abs(max_dd):.2%}.",
        "Downside deviation": f"Returns below the zero daily target had a downside deviation of {down:.2%} per observation.",
    }
