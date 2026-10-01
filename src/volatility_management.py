"""Construct realized variance, managed weights, and managed returns."""

# =============================================================================
# 0. Libraries
# =============================================================================
import logging

import numpy as np


# =============================================================================
# 1. Realized variance and lagged-RV management
# =============================================================================
def realized_variance(values):
    """Calculate the demeaned sum of squared daily returns within one month."""
    values = values.dropna()
    return ((values - values.mean()) ** 2).sum() if len(values) >= 2 else np.nan


def add_managed_return(data, factor, sample_end):
    """Add a lagged inverse-variance weight and volatility-normalized return."""
    variance = f"rv_{factor}"
    lagged_variance = f"{variance}_lag1"
    weight = f"weight_{factor}"
    managed = f"managed_{factor}"

    data[lagged_variance] = data[variance].shift(1)
    unscaled = data[factor] / data[lagged_variance]
    sample = (data["date"] <= sample_end) & unscaled.notna() & data[factor].notna()
    if (data.loc[sample, lagged_variance] <= 0).any():
        raise ValueError(f"{factor}: nonpositive lagged realized variance")

    normalizer = data.loc[sample, factor].std(ddof=1) / unscaled.loc[sample].std(ddof=1)
    data[weight] = normalizer / data[lagged_variance]
    data[managed] = data[weight] * data[factor]

    if not np.isclose(
        data.loc[sample, factor].std(ddof=1),
        data.loc[sample, managed].std(ddof=1),
        rtol=1e-12,
    ):
        raise ValueError(f"{factor}: volatility normalization failed")
    logging.info("%s | n=%s | c=%.10g", factor, int(sample.sum()), normalizer)
    return normalizer
