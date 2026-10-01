"""Descriptive statistics for unmanaged and volatility-managed factors."""

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import pandas as pd


# =============================================================================
# 1. Return paths and volatility sorts
# =============================================================================
def cumulative_returns(panel, factor):
    """Compound managed and unmanaged returns over their common valid sample."""
    columns = ["date", factor, f"managed_{factor}"]
    common = panel[columns].dropna().copy()
    common["unmanaged_wealth"] = (1 + common[factor]).cumprod()
    common["managed_wealth"] = (1 + common[f"managed_{factor}"]).cumprod()
    return common


def market_volatility_sorts(panel, buckets=5):
    """Reproduce the four panels of Moreira–Muir Figure 1."""
    data = panel[["mkt_rf", "rv_mkt_rf_lag1", "recession"]].dropna().copy()
    data["volatility_bucket"] = pd.qcut(
        data["rv_mkt_rf_lag1"], buckets, labels=False, duplicates="raise"
    ) + 1
    grouped = data.groupby("volatility_bucket")
    result = grouped["mkt_rf"].agg(["count", "mean", "std", "var"]).reset_index()
    result["mean_annual_pct"] = result["mean"] * 1200
    result["volatility_annual_pct"] = result["std"] * np.sqrt(12) * 100
    result["mean_divided_by_variance"] = result["mean"] / result["var"]
    result["recession_probability"] = grouped["recession"].mean().to_numpy()
    return result
