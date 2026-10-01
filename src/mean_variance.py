"""Transparent in-sample mean–variance frontier diagnostics."""

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import pandas as pd

from utils import fit_hac_regression
from volatility_management import realized_variance


# =============================================================================
# 1. Tangency portfolio
# =============================================================================
def tangency_portfolio(returns):
    """Estimate an unconstrained in-sample tangency direction."""
    clean = returns.dropna()
    mean = clean.mean().to_numpy()
    covariance = clean.cov().to_numpy()
    condition_number = np.linalg.cond(covariance)
    raw_weights = np.linalg.pinv(covariance) @ mean
    # Normalize gross exposure only to make displayed weights comparable; the
    # Sharpe ratio is invariant to positive rescaling of the weight vector.
    weights = raw_weights / np.abs(raw_weights).sum()
    portfolio = clean.to_numpy() @ weights
    mean_monthly = portfolio.mean()
    volatility_monthly = portfolio.std(ddof=1)
    return {
        "nobs": len(clean),
        "annualized_mean_pct": mean_monthly * 1200,
        "annualized_volatility_pct": volatility_monthly * np.sqrt(12) * 100,
        "annualized_sharpe": mean_monthly / volatility_monthly * np.sqrt(12),
        "condition_number": condition_number,
        "weights": dict(zip(clean.columns, weights)),
    }


# =============================================================================
# 2. Moreira–Muir Table II replication
# =============================================================================
def replicate_mve_volatility_timing(
    monthly,
    daily,
    opportunity_sets,
    sample_end,
    hac_lags,
):
    """Replicate Moreira-Muir Table II aggregate-MVE volatility timing."""
    rows = []
    weight_rows = []
    for name, factors in opportunity_sets.items():
        monthly_sample = monthly.loc[
            monthly["date"].le(sample_end), ["date", *factors]
        ].dropna()
        mve = tangency_portfolio(monthly_sample[factors])
        weights = pd.Series(mve["weights"])
        monthly_return = monthly_sample[factors] @ weights

        first_month = monthly_sample["date"].min().to_period("M")
        last_month = monthly_sample["date"].max().to_period("M")
        daily_sample = daily.loc[
            daily["date"].dt.to_period("M").between(first_month - 1, last_month),
            ["date", *factors],
        ].dropna()
        daily_return = daily_sample[factors] @ weights
        daily_mve = pd.DataFrame({"date": daily_sample["date"], "return": daily_return})
        daily_mve["month"] = daily_mve["date"].dt.to_period("M")
        variance = daily_mve.groupby("month")["return"].apply(realized_variance)

        strategy = pd.DataFrame(
            {
                "date": monthly_sample["date"],
                "mve_return": monthly_return,
            }
        )
        strategy["month"] = strategy["date"].dt.to_period("M")
        strategy["realized_variance"] = strategy["month"].map(variance)
        strategy["lagged_variance"] = strategy["month"].map(
            lambda month: variance.get(month - 1, np.nan)
        )
        strategy = strategy.dropna(subset=["mve_return", "lagged_variance"]).copy()
        unscaled = strategy["mve_return"] / strategy["lagged_variance"]
        normalization = strategy["mve_return"].std(ddof=1) / unscaled.std(ddof=1)
        strategy["managed_return"] = normalization * unscaled

        regression, standard_errors, _, nobs = fit_hac_regression(
            strategy,
            "managed_return",
            ["mve_return"],
            hac_lags,
            # Preserve the original Table II convention used by this project.
            use_correction=False,
        )
        residual_volatility = regression.resid.std(ddof=1) * np.sqrt(12)
        alpha = regression.params["const"] * 12
        rows.append(
            {
                "opportunity_set": name,
                "observations": nobs,
                "start": strategy["date"].min(),
                "end": strategy["date"].max(),
                "alpha_annual_pct": alpha * 100,
                "alpha_t_hac12": (
                    regression.params["const"] / standard_errors["const"]
                ),
                "r_squared": regression.rsquared,
                "original_sharpe": (
                    strategy["mve_return"].mean()
                    / strategy["mve_return"].std(ddof=1) * np.sqrt(12)
                ),
                "managed_sharpe": (
                    strategy["managed_return"].mean()
                    / strategy["managed_return"].std(ddof=1) * np.sqrt(12)
                ),
                "appraisal_ratio": alpha / residual_volatility,
            }
        )
        weight_rows.extend(
            {
                "opportunity_set": name,
                "factor": factor,
                "weight_gross_normalized": weight,
            }
            for factor, weight in weights.items()
        )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)
