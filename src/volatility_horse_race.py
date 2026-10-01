"""Common-sample performance and spanning tests for volatility-managed factors."""

# =============================================================================
# 0. Libraries and specifications
# =============================================================================
import numpy as np
import pandas as pd

from utils import fit_hac_regression


METHOD_COLUMNS = {
    "Original": lambda factor: factor,
    "Lagged RV": lambda factor: f"managed_{factor}",
    "AR(1)": lambda factor: f"managed_ar1_{factor}",
    "GARCH(1,1)": lambda factor: f"managed_garch_{factor}",
}

ALPHA_MODELS = {
    "CAPM": ["mkt_rf"],
    "FF3": ["mkt_rf", "smb", "hml"],
    "FF5": ["ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma"],
}

VOLATILITY_COLUMNS = {
    "Lagged RV": lambda factor: f"rv_{factor}_lag1",
    "AR(1)": lambda factor: f"rv_forecast_ar1_{factor}",
    "GARCH(1,1)": lambda factor: f"rv_forecast_garch_{factor}",
}


# =============================================================================
# 1. Common sample
# =============================================================================
def common_horse_race_sample(panel, factors):
    """Align every factor and managed method on one global date sample."""
    columns = []
    for factor in factors:
        columns.extend(builder(factor) for builder in METHOD_COLUMNS.values())
    columns = list(dict.fromkeys(columns))
    common = panel.dropna(subset=columns).copy()
    if common.empty:
        raise ValueError("No sample is shared by every horse-race return")
    return common


# =============================================================================
# 2. Performance
# =============================================================================
def volatility_summary_table(panel, factors):
    """Summarize the three monthly volatility estimates on the global sample."""
    common = common_horse_race_sample(panel, factors)
    rows = []
    for factor, factor_label in factors.items():
        for method, builder in VOLATILITY_COLUMNS.items():
            volatility = np.sqrt(common[builder(factor)]) * 100
            rows.append(
                {
                    "factor": factor_label,
                    "method": method,
                    "mean_monthly_vol_pct": volatility.mean(),
                    "std_monthly_vol_pct": volatility.std(ddof=1),
                    "median_monthly_vol_pct": volatility.median(),
                    "p95_monthly_vol_pct": volatility.quantile(0.95),
                }
            )
    return pd.DataFrame(rows)


def volatility_performance_table(panel, factors, hac_lags):
    """Compare the three managed returns on each factor's common sample."""
    rows = []
    managed_methods = {key: value for key, value in METHOD_COLUMNS.items() if key != "Original"}
    analysis = common_horse_race_sample(panel, factors)
    for factor, factor_label in factors.items():
        for method, builder in managed_methods.items():
            dependent = builder(factor)
            returns = analysis[dependent]
            _, mean_se, _, mean_n = fit_hac_regression(
                analysis, dependent, [], hac_lags
            )
            row = {
                "factor": factor_label,
                "method": method,
                "observations": mean_n,
                "mean_monthly_pct": returns.mean() * 100,
                "mean_se_nw12_monthly_pct": mean_se["const"] * 100,
                "mean_t_nw12": returns.mean() / mean_se["const"],
                "std_monthly_pct": returns.std(ddof=1) * 100,
                "annualized_sharpe": returns.mean() / returns.std(ddof=1) * np.sqrt(12),
            }
            for model_label, regressors in ALPHA_MODELS.items():
                model, standard_errors, _, nobs = fit_hac_regression(
                    analysis, dependent, regressors, hac_lags
                )
                key = model_label.lower()
                row[f"{key}_alpha_monthly_pct"] = model.params["const"] * 100
                row[f"{key}_alpha_se_nw12_monthly_pct"] = standard_errors["const"] * 100
                row[f"{key}_alpha_t_nw12"] = model.params["const"] / standard_errors["const"]
                row[f"{key}_nobs"] = nobs
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 3. Spanning tests
# =============================================================================
def original_factor_spanning_table(panel, factors, hac_lags):
    """Test whether each managed return is spanned by its original factor."""
    rows = []
    managed_methods = {key: value for key, value in METHOD_COLUMNS.items() if key != "Original"}
    common = common_horse_race_sample(panel, factors)
    for factor, factor_label in factors.items():
        for method, builder in managed_methods.items():
            dependent = builder(factor)
            model, standard_errors, _, nobs = fit_hac_regression(
                common, dependent, [factor], hac_lags
            )
            rows.append(
                {
                    "factor": factor_label,
                    "managed_method": method,
                    "observations": nobs,
                    "alpha_monthly_pct": model.params["const"] * 100,
                    "alpha_se_nw12_monthly_pct": standard_errors["const"] * 100,
                    "alpha_t_nw12": model.params["const"] / standard_errors["const"],
                    "original_factor_loading": model.params[factor],
                    "loading_se_nw12": standard_errors[factor],
                    "loading_t_nw12": model.params[factor] / standard_errors[factor],
                    "r_squared": model.rsquared,
                }
            )
    return pd.DataFrame(rows)


def pairwise_managed_spanning_table(panel, factors, hac_lags):
    """Run the three requested one-regressor managed-factor spanning tests."""
    rows = []
    managed_methods = {key: value for key, value in METHOD_COLUMNS.items() if key != "Original"}
    comparisons = [
        ("Lagged RV", "AR(1)"),
        ("Lagged RV", "GARCH(1,1)"),
        ("AR(1)", "GARCH(1,1)"),
    ]
    common = common_horse_race_sample(panel, factors)
    for factor, factor_label in factors.items():
        for dependent_method, spanning_method in comparisons:
            dependent = managed_methods[dependent_method](factor)
            regressor = managed_methods[spanning_method](factor)
            model, standard_errors, _, nobs = fit_hac_regression(
                common, dependent, [regressor], hac_lags
            )
            rows.append(
                {
                    "factor": factor_label,
                    "dependent_method": dependent_method,
                    "spanning_method": spanning_method,
                    "observations": nobs,
                    "alpha_monthly_pct": model.params["const"] * 100,
                    "alpha_se_nw12_monthly_pct": standard_errors["const"] * 100,
                    "alpha_t_nw12": model.params["const"] / standard_errors["const"],
                    "spanning_loading": model.params[regressor],
                    "loading_se_nw12": standard_errors[regressor],
                    "loading_t_nw12": model.params[regressor] / standard_errors[regressor],
                    "r_squared": model.rsquared,
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# 4. Subperiod robustness
# =============================================================================
def subperiod_original_spanning_table(panel, factors, hac_lags, periods=3):
    """Repeat original-factor spanning over equal-observation subperiods."""
    common = common_horse_race_sample(panel, factors).sort_values("date").reset_index(drop=True)
    managed_methods = {key: value for key, value in METHOD_COLUMNS.items() if key != "Original"}
    rows = []
    for period_number, positions in enumerate(np.array_split(np.arange(len(common)), periods), 1):
        subperiod = common.iloc[positions]
        period_label = f"{subperiod.date.min():%Y-%m} to {subperiod.date.max():%Y-%m}"
        for factor, factor_label in factors.items():
            for method, builder in managed_methods.items():
                dependent = builder(factor)
                model, standard_errors, _, nobs = fit_hac_regression(
                    subperiod, dependent, [factor], hac_lags
                )
                rows.append(
                    {
                        "period_number": period_number,
                        "period": period_label,
                        "factor": factor_label,
                        "managed_method": method,
                        "observations": nobs,
                        "alpha_monthly_pct": model.params["const"] * 100,
                        "alpha_t_nw12": model.params["const"] / standard_errors["const"],
                    }
                )
    return pd.DataFrame(rows)
