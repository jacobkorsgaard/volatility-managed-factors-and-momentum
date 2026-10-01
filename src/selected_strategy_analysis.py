"""Build and compare the five selected momentum and factor-momentum strategies."""

# =============================================================================
# 0. Libraries and strategy definitions
# =============================================================================
import numpy as np
import pandas as pd

from utils import return_statistics


SELECTED_STRATEGIES = {
    "mom": "Momentum",
    "managed_mom": "Lagged-RV volatility-timed momentum",
    "factor_momentum_original": "Factor momentum",
    "factor_momentum_original_scaled_lagged_rv": (
        "Lagged-RV volatility-timed factor momentum"
    ),
    "factor_momentum_lagged_rv": (
        "Factor momentum from lagged-RV-timed factors"
    ),
}

SELECTED_SHORT_LABELS = {
    "mom": "Mom",
    "managed_mom": "VT-Mom",
    "factor_momentum_original": "FM",
    "factor_momentum_original_scaled_lagged_rv": "FM → VT-LRV",
    "factor_momentum_lagged_rv": "VT-LRV factors → FM",
}

SELECTED_PLOT_LABELS = {
    "mkt_rf": "Market",
    **SELECTED_STRATEGIES,
}

# =============================================================================
# 1. Selected strategy panel
# =============================================================================
def build_selected_strategy_panel(managed_factors, factor_momentum):
    """Merge and impose one common sample on the selected strategies."""
    if managed_factors["date"].duplicated().any() or factor_momentum["date"].duplicated().any():
        raise ValueError("Selected-strategy inputs contain duplicate dates")

    selected_factor_momentum = [
        column for column in SELECTED_STRATEGIES if column in factor_momentum.columns
    ]
    missing_selected = sorted(
        set(SELECTED_STRATEGIES) - set(managed_factors.columns) - set(factor_momentum.columns)
    )
    if missing_selected:
        raise ValueError(f"Missing selected strategies: {missing_selected}")

    controls = [
        "mkt_rf", "smb", "hml", "mom",
        "ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma",
    ]
    required_managed = ["date", "managed_mom", *controls]
    missing_managed = sorted(set(required_managed) - set(managed_factors.columns))
    if missing_managed:
        raise ValueError(f"Managed-factor panel lacks columns: {missing_managed}")
    panel = managed_factors[["date", "managed_mom", *controls]].merge(
        factor_momentum[["date", *selected_factor_momentum]],
        on="date", how="left", validate="one_to_one",
    )
    common = (
        panel.dropna(subset=list(SELECTED_STRATEGIES))
        .sort_values("date")
        .reset_index(drop=True)
    )
    if common.empty:
        raise ValueError("No sample is shared by the selected strategies")
    return common


# =============================================================================
# 2. Full-sample performance and risk
# =============================================================================
def native_performance_table(data, strategies):
    """Summarize native-scale central performance and dispersion."""
    rows = []
    for column, label in strategies.items():
        statistics = return_statistics(data[column])
        rows.append({"strategy": label, "column": column, **statistics})
    columns = [
        "strategy", "column", "mean_monthly_pct",
        "median_pct", "std_monthly_pct",
        "compounded_annual_equivalent_pct", "annualized_sharpe",
    ]
    return pd.DataFrame(rows)[columns]


def risk_management_table(data, strategies):
    """Summarize native-scale distribution and downside risk."""
    rows = []
    for column, label in strategies.items():
        statistics = return_statistics(data[column])
        rows.append({"strategy": label, "column": column, **statistics})
    columns = [
        "strategy", "column", "quantile_05_pct", "quantile_25_pct",
        "quantile_75_pct", "quantile_95_pct", "empirical_var_05_loss_pct",
        "empirical_es_05_loss_pct", "skewness", "excess_kurtosis",
        "maximum_drawdown_pct",
    ]
    return pd.DataFrame(rows)[columns]


# =============================================================================
# 3. Equal-length subperiods
# =============================================================================
def equal_subperiod_statistics(data, strategies, n_periods=3):
    """Summarize native returns over equal-count chronological subperiods."""
    ordered = data.sort_values("date").reset_index(drop=True)
    if ordered["date"].duplicated().any():
        raise ValueError("Selected-strategy panel contains duplicate dates")
    if n_periods < 1 or n_periods > len(ordered):
        raise ValueError("n_periods must be between one and the sample length")

    rows = []
    for period_number, indices in enumerate(
        np.array_split(np.arange(len(ordered)), n_periods), start=1
    ):
        period_data = ordered.iloc[indices]
        start = period_data["date"].iloc[0]
        end = period_data["date"].iloc[-1]
        period = f"{start:%Y-%m} to {end:%Y-%m}"
        for column, label in strategies.items():
            statistics = return_statistics(period_data[column])
            rows.append(
                {
                    "period_number": period_number,
                    "period": period,
                    "start": start,
                    "end": end,
                    "observations": len(period_data),
                    "strategy": label,
                    "column": column,
                    **statistics,
                }
            )
    return pd.DataFrame(rows)


def subperiod_display_tables(period_results):
    """Select and label the performance and risk columns used in Notebook 05."""
    performance = period_results[[
        "strategy", "mean_monthly_pct", "median_pct", "std_monthly_pct",
        "compounded_annual_equivalent_pct", "annualized_sharpe",
    ]].rename(columns={
        "strategy": "Strategy",
        "mean_monthly_pct": "Monthly mean (%)",
        "median_pct": "Monthly median (%)",
        "std_monthly_pct": "Monthly std (%)",
        "compounded_annual_equivalent_pct": "Compounded annual equivalent (%)",
        "annualized_sharpe": "Annualized SR",
    })
    risk = period_results[[
        "strategy", "quantile_05_pct", "quantile_25_pct", "quantile_75_pct",
        "quantile_95_pct", "empirical_var_05_loss_pct",
        "empirical_es_05_loss_pct", "maximum_drawdown_pct",
    ]].rename(columns={
        "strategy": "Strategy",
        "quantile_05_pct": "5th percentile (%)",
        "quantile_25_pct": "25th percentile (%)",
        "quantile_75_pct": "75th percentile (%)",
        "quantile_95_pct": "95th percentile (%)",
        "empirical_var_05_loss_pct": "5% VaR: loss (%)",
        "empirical_es_05_loss_pct": "5% ES: loss (%)",
        "maximum_drawdown_pct": "Maximum drawdown (%)",
    })
    return performance, risk
