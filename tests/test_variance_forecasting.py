"""Tests for real-time AR(1) and GARCH variance calculations."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from variance_forecasting import (
    _filter_garch_variance,
    _forecast_monthly_garch_variance,
    add_expanding_ar1_forecast,
)


def _ar1_panel(observations=30):
    dates = pd.date_range("2000-01-31", periods=observations, freq="ME")
    return pd.DataFrame(
        {
            "date": dates,
            "factor": 0.01 + np.sin(np.arange(observations)) / 100,
            "rv_factor": np.exp(-7 + 0.02 * np.arange(observations)),
        }
    )


def test_expanding_ar1_starts_after_minimum_history_without_lookahead():
    baseline = _ar1_panel()
    altered = baseline.copy()
    altered.loc[20:, "rv_factor"] *= 100

    add_expanding_ar1_forecast(baseline, "factor", minimum_months=12)
    add_expanding_ar1_forecast(altered, "factor", minimum_months=12)
    forecast = "rv_forecast_ar1_factor"

    assert baseline[forecast].iloc[:12].isna().all()
    assert baseline[forecast].iloc[12:].notna().all()
    assert np.allclose(
        baseline.loc[12:20, forecast], altered.loc[12:20, forecast]
    )


def test_garch_filter_and_monthly_forecast_remain_positive():
    residuals = np.array([0.01, -0.02, 0.015, -0.005])
    variance = _filter_garch_variance(residuals, 1e-6, 0.08, 0.90)
    fit = {
        "omega": 1e-6,
        "alpha": 0.08,
        "beta": 0.90,
        "residuals": residuals,
        "variance": variance,
    }

    assert (variance > 0).all()
    assert _forecast_monthly_garch_variance(fit, trading_days=22) > 0
