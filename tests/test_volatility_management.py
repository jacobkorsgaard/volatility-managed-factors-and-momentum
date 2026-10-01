"""Tests for realized variance and lagged volatility management."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volatility_management import add_managed_return, realized_variance


def test_realized_variance_is_demeaned_daily_sum_of_squares():
    returns = pd.Series([0.01, 0.03, -0.02])
    expected = ((returns - returns.mean()) ** 2).sum()

    assert np.isclose(realized_variance(returns), expected)
    assert np.isnan(realized_variance(pd.Series([0.01])))


def test_managed_return_uses_lagged_variance_and_matches_sample_volatility():
    dates = pd.date_range("2000-01-31", periods=8, freq="ME")
    data = pd.DataFrame(
        {
            "date": dates,
            "factor": [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03, 0.02],
            "rv_factor": np.linspace(0.001, 0.004, len(dates)),
        }
    )

    normalizer = add_managed_return(data, "factor", dates[-1])
    valid = data["managed_factor"].notna()

    assert np.allclose(data["rv_factor_lag1"].iloc[1:], data["rv_factor"].iloc[:-1])
    assert np.allclose(
        data.loc[valid, "managed_factor"],
        data.loc[valid, "weight_factor"] * data.loc[valid, "factor"],
    )
    assert np.isclose(
        data.loc[valid, "factor"].std(ddof=1),
        data.loc[valid, "managed_factor"].std(ddof=1),
    )
    assert np.allclose(
        data.loc[valid, "weight_factor"] * data.loc[valid, "rv_factor_lag1"],
        normalizer,
    )
