"""Tests for the momentum volatility-timing diagnostics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from momentum_timing_analysis import (
    build_momentum_timing_sample,
    equal_volatility_timing_payoff,
    momentum_distribution_table,
    recession_loading_table,
    recession_performance_table,
    rolling_strategy_relationships,
    strategy_comovement_table,
    timing_payoff_summary,
    timing_payoff_contribution_table,
    volatility_state_table,
)


def _example_panel(observations=120):
    dates = pd.date_range("2000-01-31", periods=observations, freq="ME")
    momentum = np.linspace(-0.08, 0.10, observations)
    variance = np.linspace(0.0004, 0.006, observations)
    weight = 0.0012 / variance
    return pd.DataFrame(
        {
            "date": dates,
            "mkt_rf": np.linspace(-0.05, 0.06, observations),
            "mom": momentum,
            "managed_mom": weight * momentum,
            "rv_mom_lag1": variance,
            "weight_mom": weight,
            "recession": np.arange(observations) % 7 == 0,
        }
    )


def test_momentum_sample_validates_timing_identity():
    panel = _example_panel()
    sample = build_momentum_timing_sample(
        panel, panel["date"].min(), panel["date"].max()
    )

    assert len(sample) == len(panel)
    assert np.allclose(sample["managed_mom"], sample["weight_mom"] * sample["mom"])
    assert (sample[["rv_mom_lag1", "weight_mom"]] > 0).all().all()


def test_distribution_and_volatility_states_cover_both_returns():
    panel = _example_panel()
    distribution = momentum_distribution_table(panel)
    states = volatility_state_table(panel)

    assert distribution["strategy"].tolist() == [
        "Momentum", "Lagged-RV volatility-timed momentum"
    ]
    assert states["volatility_quintile"].tolist() == [1, 2, 3, 4, 5]
    assert states["observations"].sum() == len(panel)
    assert {
        "mom_mean_monthly_pct", "managed_mean_monthly_pct",
        "mom_std_monthly_pct", "managed_std_monthly_pct",
    }.issubset(states.columns)


def test_timing_payoff_contributions_sum_to_full_sample_mean():
    panel = _example_panel()
    payoff, scale = equal_volatility_timing_payoff(panel)
    contributions = timing_payoff_contribution_table(payoff)

    assert scale > 0
    assert np.isclose(
        contributions["contribution_to_full_mean_pct"].sum(),
        payoff["timing_payoff"].mean() * 100,
    )
    summary = timing_payoff_summary(payoff)
    assert np.isclose(summary.loc[0, "mean_monthly_pct"], payoff["timing_payoff"].mean() * 100)
    assert summary.loc[0, "observations"] == len(payoff)
    assert np.isfinite(summary.loc[0, "nw12_t"])


def test_recession_loading_table_reports_full_and_interacted_models():
    panel = _example_panel(240)
    results = recession_loading_table(panel)

    assert results.groupby("relationship").size().to_dict() == {
        "Market": 5, "Mom": 5,
    }
    for relationship in ["Market", "Mom"]:
        relationship_results = results.loc[
            results["relationship"].eq(relationship)
        ].set_index("estimate")
        label = "market beta" if relationship == "Market" else "Mom loading"
        assert np.isclose(
            relationship_results.loc[
                f"Total {label} in recessions", "coefficient"
            ],
            relationship_results.loc[
                f"{label.capitalize()} in expansions", "coefficient"
            ]
            + relationship_results.loc[
                f"Change in {label} during recessions", "coefficient"
            ],
        )


def test_recession_performance_excludes_weight_and_comovement_is_conditional():
    panel = _example_panel()
    performance = recession_performance_table(panel)
    comovement = strategy_comovement_table(panel)

    assert "average_weight" not in performance.columns
    assert comovement["state"].tolist() == [
        "Full sample", "Expansion", "Recession"
    ]
    assert comovement["observations"].sum() == 2 * len(panel)
    assert (comovement["correlation"].abs() <= 1).all()


def test_rolling_relationships_use_trailing_window():
    panel = _example_panel()
    panel["managed_mom"] = 1.5 * panel["mom"]
    rolling = rolling_strategy_relationships(panel, window=24)

    assert len(rolling) == len(panel) - 24 + 1
    assert rolling["date"].iloc[0] == panel["date"].iloc[23]
    assert np.allclose(rolling["rolling_correlation"], 1.0)
    assert np.allclose(rolling["rolling_mom_loading"], 1.5)
