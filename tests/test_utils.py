"""Tests for shared regression and return-statistics helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import fit_hac_regression, return_statistics


def test_hac_helper_returns_named_covariance_and_requested_sample():
    x = np.linspace(-1, 1, 60)
    data = pd.DataFrame({"x": x, "y": 0.01 + 2 * x})
    model, standard_errors, covariance, nobs = fit_hac_regression(
        data, "y", ["x"], hac_lags=12, use_correction=True
    )

    assert nobs == 60
    assert list(standard_errors.index) == ["const", "x"]
    assert covariance.index.tolist() == ["const", "x"]
    assert np.isclose(model.params["x"], 2)


def test_return_statistics_reports_loss_var_es_and_dates():
    returns = pd.Series([-0.2, -0.1, 0.0, 0.1, 0.2])
    dates = pd.Series(pd.date_range("2000-01-31", periods=5, freq="ME"))
    statistics = return_statistics(returns, dates=dates)

    threshold = returns.quantile(0.05)
    assert np.isclose(statistics["empirical_var_05_loss_pct"], -threshold * 100)
    assert statistics["worst_month"] == dates.iloc[0]
    assert statistics["best_month"] == dates.iloc[-1]
