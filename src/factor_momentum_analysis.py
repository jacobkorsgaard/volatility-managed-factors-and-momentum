"""Performance, spanning, exposure, and turnover analysis for factor momentum."""

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import pandas as pd

from utils import fit_hac_regression


# =============================================================================
# 1. Regression helper
# =============================================================================
def _alpha(data, dependent, regressors, hac_lags):
    """Estimate annualized alpha with monthly Newey-West inference."""
    model, standard_errors, _, nobs = fit_hac_regression(
        data,
        dependent,
        regressors,
        hac_lags,
        # Preserve the build-table convention used before the shared helper.
        use_correction=False,
    )
    alpha_t = model.params["const"] / standard_errors["const"]
    return 1200 * model.params["const"], alpha_t, nobs


# =============================================================================
# 2. Common-sample results table
# =============================================================================
def factor_momentum_results_table(
    strategies,
    monthly_controls,
    parent_exposures,
    parent_strategies,
    analysis_end,
    hac_lags,
):
    """Compare base and second-stage strategies at common in-sample volatility."""
    columns = []
    metadata = []
    for parent in parent_strategies:
        columns.append(parent)
        metadata.append((parent, "None", parent))
        for method in ["lagged_rv", "ar1", "garch"]:
            column = f"{parent}_scaled_{method}"
            columns.append(column)
            metadata.append((parent, method, column))

    common = strategies.loc[
        strategies["date"].le(analysis_end), ["date", *columns]
    ].dropna()
    target_volatility = common["factor_momentum_original"].std(ddof=1)
    scaled = common[["date"]].copy()
    comparison_scales = {}
    for column in columns:
        comparison_scales[column] = target_volatility / common[column].std(ddof=1)
        scaled[column] = common[column] * comparison_scales[column]

    controls = monthly_controls[[
        "date", "mkt_rf", "smb", "hml", "mom",
        "ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma",
    ]]
    analysis = scaled.merge(controls, on="date", how="left", validate="one_to_one")
    exposure_years = (
        analysis["date"].max() - analysis["date"].min()
    ).days / 365.25

    rows = []
    for parent, method, column in metadata:
        returns = analysis[column]
        incremental_alpha = incremental_t = np.nan
        if method != "None":
            incremental_alpha, incremental_t, _ = _alpha(
                analysis, column, [parent], hac_lags
            )
        capm_alpha, capm_t, capm_n = _alpha(
            analysis, column, ["mkt_rf"], hac_lags
        )
        ff3_alpha, ff3_t, ff3_n = _alpha(
            analysis, column, ["mkt_rf", "smb", "hml"], hac_lags
        )
        ff5_alpha, ff5_t, ff5_n = _alpha(
            analysis, column,
            ["ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma"],
            hac_lags,
        )
        ff5_mom_alpha, ff5_mom_t, ff5_mom_n = _alpha(
            analysis, column,
            ["ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma", "mom"],
            hac_lags,
        )

        weight_column = None if method == "None" else f"weight_{parent}_scaled_{method}"
        if weight_column:
            weight_sample = strategies.loc[
                strategies["date"].isin(analysis["date"]), ["date", weight_column]
            ].dropna()
            weight_years = (
                weight_sample["date"].max() - weight_sample["date"].min()
            ).days / 365.25
        else:
            weight_sample = None
            weight_years = np.nan

        effective_exposures = parent_exposures[parent].reindex(analysis["date"])
        effective_exposures.index = analysis["date"]
        if weight_column:
            second_stage_weight = strategies.set_index("date")[weight_column].reindex(
                analysis["date"]
            )
            effective_exposures = effective_exposures.mul(
                second_stage_weight.to_numpy(), axis=0
            )
        total_gross = effective_exposures.abs().sum(axis=1, min_count=1)
        factor_turnover = effective_exposures.diff().abs().sum(axis=1, min_count=1)
        comparison_scale = comparison_scales[column]

        rows.append(
            {
                "parent_strategy": parent,
                "second_stage_timing": method,
                "start": analysis["date"].min(),
                "end": analysis["date"].max(),
                "observations": len(analysis),
                "annualized_mean_pct": returns.mean() * 1200,
                "annualized_volatility_pct": returns.std(ddof=1) * np.sqrt(12) * 100,
                "sharpe_ratio": returns.mean() / returns.std(ddof=1) * np.sqrt(12),
                "incremental_alpha_pct": incremental_alpha,
                "incremental_alpha_t_nw12": incremental_t,
                "capm_alpha_pct": capm_alpha,
                "capm_alpha_t_nw12": capm_t,
                "ff3_alpha_pct": ff3_alpha,
                "ff3_alpha_t_nw12": ff3_t,
                "ff5_alpha_pct": ff5_alpha,
                "ff5_alpha_t_nw12": ff5_t,
                "ff5_mom_alpha_pct": ff5_mom_alpha,
                "ff5_mom_alpha_t_nw12": ff5_mom_t,
                "capm_n": capm_n,
                "ff3_n": ff3_n,
                "ff5_n": ff5_n,
                "ff5_mom_n": ff5_mom_n,
                "mean_second_stage_weight": (
                    weight_sample[weight_column].mean() if weight_column else np.nan
                ),
                "p99_second_stage_weight": (
                    weight_sample[weight_column].quantile(0.99) if weight_column else np.nan
                ),
                "max_second_stage_weight": (
                    weight_sample[weight_column].max() if weight_column else np.nan
                ),
                "annualized_second_stage_turnover": (
                    weight_sample[weight_column].diff().abs().sum() / weight_years
                    if weight_column else np.nan
                ),
                "mean_native_total_gross_exposure": total_gross.mean(),
                "p99_native_total_gross_exposure": total_gross.quantile(0.99),
                "max_native_total_gross_exposure": total_gross.max(),
                "annualized_native_factor_turnover": factor_turnover.sum() / exposure_years,
                "comparison_equal_vol_scale": comparison_scale,
                "mean_equal_vol_total_gross_exposure": total_gross.mean() * comparison_scale,
                "p99_equal_vol_total_gross_exposure": total_gross.quantile(0.99) * comparison_scale,
                "max_equal_vol_total_gross_exposure": total_gross.max() * comparison_scale,
                "annualized_equal_vol_factor_turnover": (
                    factor_turnover.sum() / exposure_years * comparison_scale
                ),
            }
        )
    return pd.DataFrame(rows)
