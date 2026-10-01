"""Diagnose how lagged-realized-variance timing changes momentum returns."""

# =============================================================================
# 0. Libraries and definitions
# =============================================================================
import numpy as np
import pandas as pd

from utils import fit_hac_regression, return_statistics


MOMENTUM_TIMING_COLUMNS = [
    "date", "mkt_rf", "mom", "managed_mom", "rv_mom_lag1", "weight_mom",
    "recession",
]


# =============================================================================
# 1. Analysis sample and exposure
# =============================================================================
def build_momentum_timing_sample(panel, start_date, end_date):
    """Return a validated momentum-timing sample over fixed comparison dates."""
    missing = sorted(set(MOMENTUM_TIMING_COLUMNS) - set(panel.columns))
    if missing:
        raise ValueError(f"Momentum-timing panel lacks columns: {missing}")
    if panel["date"].duplicated().any():
        raise ValueError("Momentum-timing panel contains duplicate dates")

    sample = (
        panel.loc[
            panel["date"].between(start_date, end_date),
            MOMENTUM_TIMING_COLUMNS,
        ]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
    if sample.empty:
        raise ValueError("Momentum-timing sample is empty")
    numeric = sample.drop(columns="date")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Momentum-timing sample contains non-finite values")
    if (sample[["rv_mom_lag1", "weight_mom"]] <= 0).any().any():
        raise ValueError("Momentum variance and weight must be strictly positive")
    if not sample["recession"].isin([0, 1]).all():
        raise ValueError("Recession indicator must contain only zero and one")
    if not np.allclose(
        sample["managed_mom"], sample["weight_mom"] * sample["mom"],
        rtol=1e-12, atol=1e-14,
    ):
        raise ValueError("Managed momentum does not equal weight times momentum")
    return sample


def exposure_summary_table(data):
    """Summarize the uncapped monthly momentum weight and its turnover."""
    weight = data["weight_mom"]
    normalizer = weight * data["rv_mom_lag1"]
    if not np.allclose(normalizer, normalizer.iloc[0], rtol=1e-10, atol=1e-14):
        raise ValueError("Lagged-RV normalization is not constant through time")
    metrics = {
        "Lagged-RV normalization constant": normalizer.iloc[0],
        "Mean weight": weight.mean(),
        "Median weight": weight.median(),
        "Weight standard deviation": weight.std(ddof=1),
        "Weight 5th percentile": weight.quantile(0.05),
        "Weight 95th percentile": weight.quantile(0.95),
        "Weight 99th percentile": weight.quantile(0.99),
        "Minimum weight": weight.min(),
        "Maximum weight": weight.max(),
        "Annualized weight turnover": weight.diff().abs().mean() * 12,
    }
    return pd.DataFrame(
        {"Metric": list(metrics), "Value": list(metrics.values())}
    )


# =============================================================================
# 2. Return distributions
# =============================================================================
def momentum_distribution_table(data):
    """Compare original and volatility-timed momentum distributions."""
    strategies = {
        "mom": "Momentum",
        "managed_mom": "Lagged-RV volatility-timed momentum",
    }
    rows = []
    for column, label in strategies.items():
        statistics = return_statistics(data[column], dates=data["date"])
        statistics["median_monthly_pct"] = statistics.pop("median_pct")
        rows.append(
            {
                "strategy": label,
                **statistics,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 3. Lagged-volatility states and timing payoff
# =============================================================================
def add_volatility_quintiles(data, groups=5):
    """Assign ex-post diagnostic groups using lagged momentum variance."""
    result = data.copy()
    result["volatility_quintile"] = (
        pd.qcut(result["rv_mom_lag1"], groups, labels=False, duplicates="raise") + 1
    )
    result["lagged_volatility_pct"] = np.sqrt(result["rv_mom_lag1"]) * 100
    return result


def volatility_state_table(data, groups=5):
    """Summarize next-month returns within lagged-volatility groups."""
    grouped_data = add_volatility_quintiles(data, groups=groups)
    rows = []
    for quintile, group in grouped_data.groupby("volatility_quintile", sort=True):
        row = {
            "volatility_quintile": int(quintile),
            "observations": len(group),
            "average_lagged_volatility_pct": group["lagged_volatility_pct"].mean(),
            "average_weight": group["weight_mom"].mean(),
            "recession_months_pct": group["recession"].mean() * 100,
            "negative_momentum_months_pct": group["mom"].lt(0).mean() * 100,
        }
        for column, prefix in [("mom", "mom"), ("managed_mom", "managed")]:
            returns = group[column]
            row[f"{prefix}_mean_monthly_pct"] = returns.mean() * 100
            row[f"{prefix}_std_monthly_pct"] = returns.std(ddof=1) * 100
            row[f"{prefix}_annualized_sharpe"] = (
                returns.mean() / returns.std(ddof=1) * np.sqrt(12)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def equal_volatility_timing_payoff(data):
    """Construct an ex-post equal-volatility managed-minus-original payoff."""
    scale = data["mom"].std(ddof=1) / data["managed_mom"].std(ddof=1)
    payoff = add_volatility_quintiles(data)
    payoff["managed_mom_equal_vol"] = payoff["managed_mom"] * scale
    payoff["timing_payoff"] = payoff["managed_mom_equal_vol"] - payoff["mom"]
    payoff["cumulative_timing_payoff_pct"] = payoff["timing_payoff"].cumsum() * 100
    return payoff, scale


def timing_payoff_summary(payoff, hac_lags=12):
    """Report the equal-volatility payoff mean with monthly HAC inference."""
    model, standard_errors, _, nobs = fit_hac_regression(
        payoff,
        "timing_payoff",
        [],
        hac_lags,
        use_correction=True,
    )
    return pd.DataFrame(
        {
            "mean_monthly_pct": [model.params["const"] * 100],
            "nw12_t": [model.params["const"] / standard_errors["const"]],
            "observations": [nobs],
        }
    )


def timing_payoff_contribution_table(payoff):
    """Decompose the average timing payoff across volatility quintiles."""
    total_observations = len(payoff)
    rows = []
    for quintile, group in payoff.groupby("volatility_quintile", sort=True):
        rows.append(
            {
                "volatility_quintile": int(quintile),
                "observations": len(group),
                "mean_timing_payoff_pct": group["timing_payoff"].mean() * 100,
                "contribution_to_full_mean_pct": (
                    group["timing_payoff"].sum() / total_observations * 100
                ),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 4. Recession diagnostics
# =============================================================================
def recession_performance_table(data):
    """Compare native momentum performance across NBER states."""
    states = {0: "Expansion", 1: "Recession"}
    strategies = {
        "mom": "Momentum",
        "managed_mom": "Lagged-RV volatility-timed momentum",
    }
    rows = []
    for state, state_label in states.items():
        state_data = data.loc[data["recession"].eq(state)]
        for column, strategy_label in strategies.items():
            returns = state_data[column]
            rows.append(
                {
                    "state": state_label,
                    "strategy": strategy_label,
                    "observations": len(state_data),
                    "mean_monthly_pct": returns.mean() * 100,
                    "median_monthly_pct": returns.median() * 100,
                    "std_monthly_pct": returns.std(ddof=1) * 100,
                    "annualized_sharpe": (
                        returns.mean() / returns.std(ddof=1) * np.sqrt(12)
                    ),
                    "negative_months_pct": returns.lt(0).mean() * 100,
                }
            )
    return pd.DataFrame(rows)


def strategy_comovement_table(data):
    """Describe how managed momentum comoves with Mom across NBER states."""
    states = {
        "Full sample": data,
        "Expansion": data.loc[data["recession"].eq(0)],
        "Recession": data.loc[data["recession"].eq(1)],
    }
    rows = []
    for label, state_data in states.items():
        original = state_data["mom"]
        managed = state_data["managed_mom"]
        correlation = original.corr(managed)
        beta = original.cov(managed) / original.var(ddof=1)
        rows.append(
            {
                "state": label,
                "observations": len(state_data),
                "correlation": correlation,
                "mom_loading": beta,
                "r_squared": correlation**2,
            }
        )
    return pd.DataFrame(rows)


def rolling_strategy_relationships(data, window=120):
    """Calculate trailing correlation and Mom loading for managed momentum."""
    if window < 2:
        raise ValueError("Rolling window must contain at least two months")
    if len(data) < window:
        raise ValueError("Rolling window exceeds the momentum sample")
    ordered = data.sort_values("date").reset_index(drop=True)
    original = ordered["mom"]
    managed = ordered["managed_mom"]
    rolling_variance = original.rolling(window).var(ddof=1)
    result = ordered[["date", "recession"]].copy()
    result["rolling_correlation"] = original.rolling(window).corr(managed)
    result["rolling_mom_loading"] = (
        original.rolling(window).cov(managed) / rolling_variance
    )
    result = result.dropna().reset_index(drop=True)
    if not np.isfinite(
        result[["rolling_correlation", "rolling_mom_loading"]].to_numpy()
    ).all():
        raise ValueError("Rolling momentum relationships contain non-finite values")
    return result


def recession_loading_table(data, hac_lags=12):
    """Report unconditional and NBER-interacted loadings on Market and Mom."""
    specifications = {
        "Market": ("mkt_rf", "market beta"),
        "Mom": ("mom", "Mom loading"),
    }
    rows = []
    for relationship, (regressor, loading_label) in specifications.items():
        regression_data = data[["managed_mom", regressor, "recession"]].dropna().copy()
        if regression_data.empty:
            raise ValueError(f"{relationship} loading sample is empty")
        regression_data["recession"] = regression_data["recession"].astype(int)

        full_model, full_se, _, full_nobs = fit_hac_regression(
            regression_data,
            "managed_mom",
            [regressor],
            hac_lags,
            use_correction=True,
        )
        rows.append(
            {
                "relationship": relationship,
                "model": "Unconditional full-period regression",
                "estimate": f"Full-period {loading_label}",
                "coefficient": full_model.params[regressor],
                "unit": "Loading",
                "nw12_t": full_model.params[regressor] / full_se[regressor],
                "observations": full_nobs,
                "r_squared": full_model.rsquared,
            }
        )

        interaction = f"{regressor}_x_recession"
        regression_data[interaction] = (
            regression_data[regressor] * regression_data["recession"]
        )
        regressors = [regressor, "recession", interaction]
        interaction_model, standard_errors, covariance, interaction_nobs = (
            fit_hac_regression(
                regression_data,
                "managed_mom",
                regressors,
                hac_lags,
                use_correction=True,
            )
        )
        recession_loading = (
            interaction_model.params[regressor]
            + interaction_model.params[interaction]
        )
        recession_loading_se = np.sqrt(
            covariance.loc[regressor, regressor]
            + covariance.loc[interaction, interaction]
            + 2 * covariance.loc[regressor, interaction]
        )
        interaction_rows = [
            (
                "Recession dummy (intercept shift)",
                interaction_model.params["recession"] * 100,
                "Monthly return (%)",
                interaction_model.params["recession"]
                / standard_errors["recession"],
            ),
            (
                f"{loading_label.capitalize()} in expansions",
                interaction_model.params[regressor],
                "Loading",
                interaction_model.params[regressor] / standard_errors[regressor],
            ),
            (
                f"Change in {loading_label} during recessions",
                interaction_model.params[interaction],
                "Loading",
                interaction_model.params[interaction]
                / standard_errors[interaction],
            ),
            (
                f"Total {loading_label} in recessions",
                recession_loading,
                "Loading",
                recession_loading / recession_loading_se,
            ),
        ]
        for estimate, coefficient, unit, t_statistic in interaction_rows:
            rows.append(
                {
                    "relationship": relationship,
                    "model": "Recession interaction regression",
                    "estimate": estimate,
                    "coefficient": coefficient,
                    "unit": unit,
                    "nw12_t": t_statistic,
                    "observations": interaction_nobs,
                    "r_squared": interaction_model.rsquared,
                }
            )
    return pd.DataFrame(rows)
