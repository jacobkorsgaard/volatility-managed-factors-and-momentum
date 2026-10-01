"""Tests for spanning and recession-interaction regressions."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spanning import run_recession_beta_tests, run_spanning_tests


def test_own_factor_spanning_recovers_known_loading():
    rng = np.random.default_rng(4)
    factor = rng.normal(0.005, 0.04, 240)
    data = pd.DataFrame(
        {
            "mkt_rf": factor,
            "managed_mkt_rf": 0.002 + 1.5 * factor,
        }
    )
    result = run_spanning_tests(
        data, {"mkt_rf": "Market"}, "Own factor", hac_lags=12
    ).iloc[0]

    assert np.isclose(result["alpha_annual_pct"], 2.4)
    assert np.isclose(result["mkt_rf_loading"], 1.5)


def test_recession_beta_is_base_plus_interaction():
    rng = np.random.default_rng(8)
    factor = rng.normal(0.005, 0.04, 240)
    recession = (np.arange(240) % 7 == 0).astype(int)
    data = pd.DataFrame(
        {
            "factor": factor,
            "managed_factor": (1.2 - 0.7 * recession) * factor,
            "recession": recession,
        }
    )
    result = run_recession_beta_tests(
        data, {"factor": "Factor"}, hac_lags=12
    ).iloc[0]

    assert np.isclose(
        result["recession_beta"],
        result["nonrecession_beta"] + result["recession_beta_interaction"],
    )
