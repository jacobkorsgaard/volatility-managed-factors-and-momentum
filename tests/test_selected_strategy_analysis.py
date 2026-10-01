"""Tests for selected-strategy risk and subperiod summaries."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selected_strategy_analysis import (
    equal_subperiod_statistics,
    native_performance_table,
    risk_management_table,
)


def test_empirical_var_es_and_drawdown_use_return_losses():
    returns = pd.Series([-0.20, -0.10, 0.00, 0.10, 0.20])
    data = pd.DataFrame({"strategy": returns})
    result = risk_management_table(data, {"strategy": "Strategy"}).iloc[0]

    threshold = returns.quantile(0.05)
    expected_shortfall = returns.loc[returns <= threshold].mean()
    wealth = (1 + returns).cumprod()
    expected_drawdown = (wealth / wealth.cummax() - 1).min()

    assert np.isclose(result["empirical_var_05_loss_pct"], -threshold * 100)
    assert np.isclose(result["empirical_es_05_loss_pct"], -expected_shortfall * 100)
    assert np.isclose(result["maximum_drawdown_pct"], expected_drawdown * 100)


def test_equal_subperiods_are_chronological_and_exhaustive():
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    data = pd.DataFrame({"date": dates, "strategy": np.arange(12) / 100})
    result = equal_subperiod_statistics(
        data, {"strategy": "Strategy"}, n_periods=3
    )

    assert result["observations"].tolist() == [4, 4, 4]
    assert result["start"].tolist() == [dates[0], dates[4], dates[8]]
    assert result["end"].tolist() == [dates[3], dates[7], dates[11]]
    assert result["observations"].sum() == len(data)


def test_standard_deviation_and_median_are_performance_columns_only():
    data = pd.DataFrame({"strategy": [-0.02, 0.01, 0.03, 0.04]})
    strategies = {"strategy": "Strategy"}

    performance = native_performance_table(data, strategies)
    risk = risk_management_table(data, strategies)

    assert {"std_monthly_pct", "median_pct"}.issubset(performance.columns)
    assert "std_monthly_pct" not in risk.columns
    assert "median_pct" not in risk.columns
