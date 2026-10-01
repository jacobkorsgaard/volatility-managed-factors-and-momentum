"""Shared statistical and presentation helpers used across analyses."""

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm


# =============================================================================
# 1. Monthly HAC regressions
# =============================================================================
def fit_hac_regression(
    data,
    dependent,
    regressors,
    hac_lags,
    use_correction=True,
):
    """Estimate OLS with explicit monthly Newey-West covariance settings."""
    columns = list(dict.fromkeys([dependent, *regressors]))
    regression = data[columns].dropna()
    if regression.empty:
        raise ValueError(f"No observations for {dependent} on {regressors}")
    design = sm.add_constant(
        regression[regressors], has_constant="add"
    )
    model = sm.OLS(regression[dependent], design).fit()
    robust = model.get_robustcov_results(
        cov_type="HAC",
        maxlags=hac_lags,
        use_correction=use_correction,
        use_t=False,
    )
    names = list(model.params.index)
    standard_errors = pd.Series(robust.bse, index=names)
    covariance = pd.DataFrame(
        robust.cov_params(), index=names, columns=names
    )
    return model, standard_errors, covariance, len(regression)


# =============================================================================
# 2. Return statistics
# =============================================================================
def return_statistics(returns, tail_probability=0.05, dates=None):
    """Calculate a common superset of performance and empirical risk measures."""
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("Cannot summarize an empty return series")
    if not 0 < tail_probability < 0.5:
        raise ValueError("tail_probability must lie between zero and one-half")

    threshold = returns.quantile(tail_probability)
    tail_returns = returns.loc[returns <= threshold]
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    statistics = {
        "mean_monthly_pct": returns.mean() * 100,
        "median_pct": returns.median() * 100,
        "std_monthly_pct": returns.std(ddof=1) * 100,
        "compounded_annual_equivalent_pct": (
            (1 + returns.mean()) ** 12 - 1
        ) * 100,
        "annualized_sharpe": (
            returns.mean() / returns.std(ddof=1) * np.sqrt(12)
        ),
        "quantile_01_pct": returns.quantile(0.01) * 100,
        "quantile_05_pct": returns.quantile(0.05) * 100,
        "quantile_25_pct": returns.quantile(0.25) * 100,
        "quantile_75_pct": returns.quantile(0.75) * 100,
        "quantile_95_pct": returns.quantile(0.95) * 100,
        "quantile_99_pct": returns.quantile(0.99) * 100,
        "empirical_var_05_loss_pct": -threshold * 100,
        "empirical_es_05_loss_pct": -tail_returns.mean() * 100,
        "skewness": returns.skew(),
        "excess_kurtosis": returns.kurt(),
        "positive_months_pct": returns.gt(0).mean() * 100,
        "maximum_drawdown_pct": drawdown.min() * 100,
    }
    if dates is not None:
        aligned_dates = dates.loc[returns.index]
        worst = returns.idxmin()
        best = returns.idxmax()
        statistics.update(
            {
                "worst_month": aligned_dates.loc[worst],
                "worst_return_pct": returns.loc[worst] * 100,
                "best_month": aligned_dates.loc[best],
                "best_return_pct": returns.loc[best] * 100,
            }
        )
    return statistics


# =============================================================================
# 3. Notebook table formatting
# =============================================================================
def format_estimate_with_t(
    data,
    estimate,
    t_statistic,
    missing_t="",
):
    """Format an estimate and t-statistic without changing stored results."""
    def format_row(row):
        if pd.isna(row[estimate]):
            return ""
        if pd.isna(row[t_statistic]):
            suffix = f" ({missing_t})" if missing_t else ""
            return f"{row[estimate]:.3f}{suffix}"
        return f"{row[estimate]:.3f} ({row[t_statistic]:.2f})"

    return data.apply(format_row, axis=1)
