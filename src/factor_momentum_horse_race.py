"""Selection-oriented horse race for factor-momentum strategy architectures."""

# =============================================================================
# 0. Libraries and strategy definitions
# =============================================================================
import numpy as np
import pandas as pd

from utils import fit_hac_regression


METHODS = {
    "lagged_rv": "Lagged RV",
    "ar1": "AR(1)",
    "garch": "GARCH(1,1)",
}

PARENT_STRATEGIES = {
    "factor_momentum_original": "Factor momentum",
    "factor_momentum_lagged_rv": "Factor momentum from lagged-RV-timed factors",
    "factor_momentum_ar1": "Factor momentum from AR(1)-timed factors",
    "factor_momentum_garch": "Factor momentum from GARCH-timed factors",
}

PORTFOLIO_TIMED_STRATEGIES = {
    f"factor_momentum_original_scaled_{method}":
        f"{label} volatility-timed factor momentum"
    for method, label in METHODS.items()
}

SINGLE_LAYER_STRATEGIES = {
    "factor_momentum_original": "Factor momentum",
    **PORTFOLIO_TIMED_STRATEGIES,
    **{key: value for key, value in PARENT_STRATEGIES.items() if key != "factor_momentum_original"},
}

COMPACT_SINGLE_LAYER_LABELS = {
    "factor_momentum_original": "FM",
    "factor_momentum_original_scaled_lagged_rv": "FM → VT-LRV",
    "factor_momentum_original_scaled_ar1": "FM → VT-AR1",
    "factor_momentum_original_scaled_garch": "FM → VT-GARCH",
    "factor_momentum_lagged_rv": "VT-LRV factors → FM",
    "factor_momentum_ar1": "VT-AR1 factors → FM",
    "factor_momentum_garch": "VT-GARCH factors → FM",
}

ALL_TIMED_STRATEGIES = []
for parent in PARENT_STRATEGIES:
    ALL_TIMED_STRATEGIES.append(parent)
    ALL_TIMED_STRATEGIES.extend(f"{parent}_scaled_{method}" for method in METHODS)

ALPHA_MODELS = {
    "momentum": ["mom"],
    "capm": ["mkt_rf"],
    "ff3": ["mkt_rf", "smb", "hml"],
    "ff5": ["ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma"],
    "ff5_mom": ["ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma", "mom"],
}

FACTOR_INPUT_SETS = {
    "factor_momentum_original": ["mkt_rf", "smb", "hml", "mom", "rmw", "cma"],
    "factor_momentum_lagged_rv": [
        "managed_mkt_rf", "managed_smb", "managed_hml",
        "managed_mom", "managed_rmw", "managed_cma",
    ],
    "factor_momentum_ar1": [
        "managed_ar1_mkt_rf", "managed_ar1_smb", "managed_ar1_hml",
        "managed_ar1_mom", "managed_ar1_rmw", "managed_ar1_cma",
    ],
    "factor_momentum_garch": [
        "managed_garch_mkt_rf", "managed_garch_smb", "managed_garch_hml",
        "managed_garch_mom", "managed_garch_rmw", "managed_garch_cma",
    ],
}

FACTOR_INPUT_LABELS = {
    "factor_momentum_original": "Original factors",
    "factor_momentum_lagged_rv": "Lagged-RV-timed factors",
    "factor_momentum_ar1": "AR(1)-timed factors",
    "factor_momentum_garch": "GARCH-timed factors",
}


# =============================================================================
# 1. Common sample and scaling
# =============================================================================
def prepare_factor_momentum_horse_race(strategies, managed_factors):
    """Impose one sample across all 16 factor-momentum strategies."""
    if strategies["date"].duplicated().any() or managed_factors["date"].duplicated().any():
        raise ValueError("Factor-momentum inputs contain duplicate dates")

    missing_strategies = sorted(set(ALL_TIMED_STRATEGIES) - set(strategies.columns))
    if missing_strategies:
        raise ValueError(f"Missing factor-momentum strategies: {missing_strategies}")
    common = strategies[["date", *ALL_TIMED_STRATEGIES]].dropna().copy()
    if common.empty:
        raise ValueError("No sample is shared by all factor-momentum strategies")

    controls = sorted({column for values in ALPHA_MODELS.values() for column in values})
    analysis = common.merge(
        managed_factors[["date", *controls]],
        on="date", how="left", validate="one_to_one",
    )
    return analysis.sort_values("date").reset_index(drop=True)


def equal_volatility_panel(data, return_columns, target_column="factor_momentum_original"):
    """Scale returns to the target's full-common-sample volatility."""
    columns = list(dict.fromkeys([target_column, *return_columns]))
    common = data[["date", *columns]].dropna().copy()
    target_volatility = common[target_column].std(ddof=1)
    scales = {}
    for column in return_columns:
        volatility = common[column].std(ddof=1)
        if not np.isfinite(volatility) or volatility <= 0:
            raise ValueError(f"{column}: invalid volatility for comparison scaling")
        scales[column] = target_volatility / volatility
        common[column] = common[column] * scales[column]
    return common, scales


# =============================================================================
# 2. Performance
# =============================================================================
def strategy_performance_table(
    data,
    strategies,
    hac_lags,
    target_column="factor_momentum_original",
):
    """Report equal-volatility performance and factor-model alphas."""
    scaled_returns, scales = equal_volatility_panel(
        data, list(strategies), target_column=target_column
    )
    controls = sorted({column for values in ALPHA_MODELS.values() for column in values})
    controls_to_add = [column for column in controls if column not in scaled_returns]
    analysis = scaled_returns.merge(
        data[["date", *controls_to_add]], on="date", how="left", validate="one_to_one"
    )
    rows = []
    for column, label in strategies.items():
        returns = analysis[column]
        _, mean_se, _, nobs = fit_hac_regression(analysis, column, [], hac_lags)
        row = {
            "strategy": label,
            "column": column,
            "observations": nobs,
            "mean_monthly_pct": returns.mean() * 100,
            "mean_t_nw12": returns.mean() / mean_se["const"],
            "std_monthly_pct": returns.std(ddof=1) * 100,
            "annualized_sharpe": returns.mean() / returns.std(ddof=1) * np.sqrt(12),
            "equal_volatility_scale": scales[column],
        }
        for model_name, regressors in ALPHA_MODELS.items():
            if column in regressors:
                model_nobs = len(analysis[[column, *regressors]].dropna())
                row[f"{model_name}_alpha_monthly_pct"] = 0.0
                row[f"{model_name}_alpha_t_nw12"] = np.nan
                row[f"{model_name}_nobs"] = model_nobs
                continue
            model, standard_errors, _, model_nobs = fit_hac_regression(
                analysis, column, regressors, hac_lags
            )
            row[f"{model_name}_alpha_monthly_pct"] = model.params["const"] * 100
            row[f"{model_name}_alpha_t_nw12"] = (
                model.params["const"] / standard_errors["const"]
            )
            row[f"{model_name}_nobs"] = model_nobs
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 3. Matched 1/N and single-layer spanning
# =============================================================================
def matched_one_over_n_spanning_table(data, managed_factors, hac_lags):
    """Compare each factor-momentum parent with its factor-input 1/N portfolio."""
    required_inputs = sorted(
        {column for columns in FACTOR_INPUT_SETS.values() for column in columns}
    )
    missing = sorted(set(required_inputs) - set(managed_factors.columns))
    if missing:
        raise ValueError(f"Managed-factor panel lacks 1/N inputs: {missing}")
    if managed_factors["date"].duplicated().any():
        raise ValueError("Managed-factor panel contains duplicate dates")

    benchmarks = managed_factors[["date"]].copy()
    benchmark_columns = {}
    for strategy, factor_columns in FACTOR_INPUT_SETS.items():
        benchmark = f"one_over_n_{strategy}"
        benchmark_columns[strategy] = benchmark
        benchmarks[benchmark] = managed_factors[factor_columns].mean(
            axis=1, skipna=True
        )

    columns = [*FACTOR_INPUT_SETS, *benchmark_columns.values()]
    comparison = data.merge(
        benchmarks, on="date", how="left", validate="one_to_one"
    )
    scaled, _ = equal_volatility_panel(
        comparison, columns, target_column="factor_momentum_original"
    )

    rows = []
    for strategy, benchmark in benchmark_columns.items():
        regression, standard_errors, _, nobs = fit_hac_regression(
            scaled, strategy, [benchmark], hac_lags
        )
        strategy_returns = comparison[strategy].dropna()
        benchmark_returns = comparison[benchmark].dropna()
        rows.append(
            {
                "factor_inputs": FACTOR_INPUT_LABELS[strategy],
                "strategy": PARENT_STRATEGIES[strategy],
                "observations": nobs,
                "factor_momentum_sharpe": (
                    strategy_returns.mean()
                    / strategy_returns.std(ddof=1)
                    * np.sqrt(12)
                ),
                "one_over_n_sharpe": (
                    benchmark_returns.mean()
                    / benchmark_returns.std(ddof=1)
                    * np.sqrt(12)
                ),
                "alpha_monthly_pct": regression.params["const"] * 100,
                "alpha_t_nw12": (
                    regression.params["const"] / standard_errors["const"]
                ),
                "one_over_n_loading": regression.params[benchmark],
                "loading_t_nw12": (
                    regression.params[benchmark] / standard_errors[benchmark]
                ),
                "r_squared": regression.rsquared,
            }
        )
    return pd.DataFrame(rows)


def pairwise_strategy_spanning(
    data,
    strategies,
    hac_lags,
    short_labels=None,
    target_column="factor_momentum_original",
):
    """Run every directional one-regressor spanning test among strategies."""
    scaled, _ = equal_volatility_panel(
        data, list(strategies), target_column=target_column
    )
    short_labels = short_labels or COMPACT_SINGLE_LAYER_LABELS
    rows = []
    for dependent, dependent_label in strategies.items():
        for spanning, spanning_label in strategies.items():
            if dependent == spanning:
                continue
            model, standard_errors, _, nobs = fit_hac_regression(
                scaled, dependent, [spanning], hac_lags
            )
            rows.append(
                {
                    "dependent": dependent,
                    "dependent_label": dependent_label,
                    "dependent_short": short_labels[dependent],
                    "spanning": spanning,
                    "spanning_label": spanning_label,
                    "spanning_short": short_labels[spanning],
                    "observations": nobs,
                    "alpha_monthly_pct": model.params["const"] * 100,
                    "alpha_t_nw12": model.params["const"] / standard_errors["const"],
                    "loading": model.params[spanning],
                    "loading_t_nw12": model.params[spanning] / standard_errors[spanning],
                    "r_squared": model.rsquared,
                }
            )
    return pd.DataFrame(rows)


def momentum_spanning_table(data, strategies, hac_lags):
    """Test each strategy against momentum alone and the full FF5 + Mom model."""
    scaled, _ = equal_volatility_panel(data, list(strategies))
    controls = sorted({column for values in ALPHA_MODELS.values() for column in values})
    controls_to_add = [column for column in controls if column not in scaled]
    analysis = scaled.merge(
        data[["date", *controls_to_add]], on="date", how="left", validate="one_to_one"
    )
    rows = []
    for column, label in strategies.items():
        momentum, momentum_se, _, momentum_nobs = fit_hac_regression(
            analysis, column, ALPHA_MODELS["momentum"], hac_lags
        )
        ff5_mom, ff5_mom_se, _, ff5_mom_nobs = fit_hac_regression(
            analysis, column, ALPHA_MODELS["ff5_mom"], hac_lags
        )
        rows.append(
            {
                "strategy": label,
                "column": column,
                "momentum_alpha_monthly_pct": momentum.params["const"] * 100,
                "momentum_alpha_t_nw12": momentum.params["const"] / momentum_se["const"],
                "momentum_loading": momentum.params["mom"],
                "momentum_loading_t_nw12": momentum.params["mom"] / momentum_se["mom"],
                "momentum_r_squared": momentum.rsquared,
                "momentum_nobs": momentum_nobs,
                "ff5_mom_alpha_monthly_pct": ff5_mom.params["const"] * 100,
                "ff5_mom_alpha_t_nw12": ff5_mom.params["const"] / ff5_mom_se["const"],
                "ff5_mom_r_squared": ff5_mom.rsquared,
                "ff5_mom_nobs": ff5_mom_nobs,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 4. Double timing
# =============================================================================
def double_timing_table(data, hac_lags):
    """Test each double-timed strategy against its factor-timed FM parent."""
    factor_timed_parents = {
        method: f"factor_momentum_{method}" for method in METHODS
    }
    columns = []
    for parent in factor_timed_parents.values():
        columns.append(parent)
        columns.extend(f"{parent}_scaled_{method}" for method in METHODS)
    scaled, _ = equal_volatility_panel(data, columns)
    controls = sorted({column for values in ALPHA_MODELS.values() for column in values})
    analysis = scaled.merge(
        data[["date", *controls]], on="date", how="left", validate="one_to_one"
    )

    rows = []
    for factor_method, parent in factor_timed_parents.items():
        parent_returns = analysis[parent]
        parent_sharpe = parent_returns.mean() / parent_returns.std(ddof=1) * np.sqrt(12)
        for portfolio_method in METHODS:
            child = f"{parent}_scaled_{portfolio_method}"
            child_returns = analysis[child]
            incremental, incremental_se, _, nobs = fit_hac_regression(
                analysis, child, [parent], hac_lags
            )
            ff5_mom, ff5_mom_se, _, ff5_mom_nobs = fit_hac_regression(
                analysis, child, ALPHA_MODELS["ff5_mom"], hac_lags
            )
            rows.append(
                {
                    "factor_timing": METHODS[factor_method],
                    "portfolio_timing": METHODS[portfolio_method],
                    "parent_column": parent,
                    "strategy_column": child,
                    "observations": nobs,
                    "parent_sharpe": parent_sharpe,
                    "strategy_sharpe": (
                        child_returns.mean() / child_returns.std(ddof=1) * np.sqrt(12)
                    ),
                    "incremental_alpha_monthly_pct": incremental.params["const"] * 100,
                    "incremental_alpha_t_nw12": (
                        incremental.params["const"] / incremental_se["const"]
                    ),
                    "ff5_mom_alpha_monthly_pct": ff5_mom.params["const"] * 100,
                    "ff5_mom_alpha_t_nw12": ff5_mom.params["const"] / ff5_mom_se["const"],
                    "ff5_mom_nobs": ff5_mom_nobs,
                }
            )
    return pd.DataFrame(rows)
