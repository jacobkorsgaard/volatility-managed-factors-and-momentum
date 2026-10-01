"""Tests for the time-series factor-momentum timing convention."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_momentum import (
    daily_factor_momentum_proxy,
    factor_momentum_positions,
    factor_momentum_returns,
)


def test_signal_uses_previous_twelve_months_without_skip():
    returns = pd.DataFrame(
        {"positive": [0.01] * 12 + [-0.04], "negative": [-0.01] * 12 + [0.02]}
    )
    positions = factor_momentum_positions(returns)
    assert positions.iloc[:12].isna().all().all()
    assert positions.iloc[12].to_dict() == {"positive": 0.5, "negative": -0.5}
    assert np.isclose(factor_momentum_returns(returns).iloc[12], -0.03)


def test_missing_history_excludes_only_the_unavailable_factor():
    returns = pd.DataFrame(
        {
            "complete": [0.01] * 13,
            "late": [np.nan] + [0.02] * 12,
        }
    )
    positions = factor_momentum_positions(returns)
    assert positions.loc[12, "complete"] == 1.0
    assert np.isnan(positions.loc[12, "late"])


def test_current_return_does_not_change_current_signal():
    baseline = pd.DataFrame({"factor": [0.01] * 13})
    altered = baseline.copy()
    altered.loc[12, "factor"] = -0.99
    assert factor_momentum_positions(baseline).iloc[12, 0] == 1.0
    assert factor_momentum_positions(altered).iloc[12, 0] == 1.0


def test_signal_includes_the_immediately_preceding_month():
    returns = pd.DataFrame({"factor": [-0.01] * 11 + [0.20, 0.00]})
    assert factor_momentum_positions(returns).iloc[12, 0] == 1.0


def test_signal_uses_arithmetic_average_not_compounded_return():
    # The arithmetic sum is positive, while the compounded return is negative.
    returns = pd.DataFrame({"factor": [0.10] * 11 + [-0.90, 0.00]})
    assert returns.iloc[:12, 0].sum() > 0
    assert np.prod(1 + returns.iloc[:12, 0]) - 1 < 0
    assert factor_momentum_positions(returns).iloc[12, 0] == 1.0


def test_daily_proxy_uses_month_start_known_position_and_scaling_weight():
    monthly_dates = pd.date_range("2000-01-31", periods=13, freq="ME")
    monthly = pd.DataFrame(
        {"positive": [0.01] * 13, "negative": [-0.01] * 13},
        index=monthly_dates,
    )
    weights = pd.DataFrame(
        {"positive": [2.0] * 13, "negative": [3.0] * 13},
        index=monthly_dates,
    )
    daily = pd.DataFrame(
        {"positive": [0.02], "negative": [0.01]},
        index=pd.DatetimeIndex(["2001-01-15"]),
    )
    proxy = daily_factor_momentum_proxy(daily, monthly, weights)
    # Month 13 positions are +1/2 and -1/2, fixed using months 1 through 12.
    assert np.isclose(proxy.iloc[0], 0.5 * 2.0 * 0.02 - 0.5 * 3.0 * 0.01)
