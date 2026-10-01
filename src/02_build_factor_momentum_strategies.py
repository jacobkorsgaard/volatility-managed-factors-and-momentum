"""Stage 2: build factor-momentum and portfolio-level timing strategies."""

# =============================================================================
# 0. Libraries
# =============================================================================
import logging

import pandas as pd

from config import (
    ANALYSIS_END,
    AR1_MIN_MONTHS,
    FACTORS,
    FACTOR_MOMENTUM_PANEL,
    FACTOR_MOMENTUM_LOG,
    FACTOR_MOMENTUM_SETS,
    FACTOR_MOMENTUM_TABLE,
    FACTOR_MOMENTUM_WEIGHT_SETS,
    GARCH_MIN_MONTHS,
    HAC_LAGS,
    LOGS,
    MANAGED_FACTORS,
    TABLES,
)
from data import load_factor_returns
from factor_momentum import (
    daily_factor_momentum_proxy,
    factor_momentum_positions,
    factor_momentum_returns,
)
from factor_momentum_analysis import factor_momentum_results_table
from volatility_management import add_managed_return, realized_variance
from variance_forecasting import add_expanding_ar1_forecast, add_expanding_garch_forecast


# =============================================================================
# 1. Daily return proxies
# =============================================================================
def _daily_proxy(daily, monthly, strategy):
    """Build the daily proxy using month-start-known signals and scale weights."""
    signal_columns = FACTOR_MOMENTUM_SETS[strategy]
    signal_returns = monthly.set_index("date")[signal_columns].copy()
    signal_returns.columns = list(FACTORS)

    weight_columns = FACTOR_MOMENTUM_WEIGHT_SETS[strategy]
    if weight_columns is None:
        scaling_weights = None
    else:
        scaling_weights = monthly.set_index("date")[weight_columns].copy()
        scaling_weights.columns = list(FACTORS)

    daily_returns = daily.set_index("date")[list(FACTORS)].copy()
    proxy = daily_factor_momentum_proxy(
        daily_returns, signal_returns, scaling_weights, lookback=12
    )
    return proxy.dropna().rename(strategy).reset_index()


def _monthly_factor_exposures(monthly, strategy):
    """Return month-t exposures to each raw factor before second-stage timing."""
    signal_columns = FACTOR_MOMENTUM_SETS[strategy]
    signal_returns = monthly.set_index("date")[signal_columns].copy()
    signal_returns.columns = list(FACTORS)
    positions = factor_momentum_positions(signal_returns, lookback=12)
    weight_columns = FACTOR_MOMENTUM_WEIGHT_SETS[strategy]
    if weight_columns is None:
        return positions
    scaling_weights = monthly.set_index("date")[weight_columns].copy()
    scaling_weights.columns = list(FACTORS)
    return positions * scaling_weights


# =============================================================================
# 2. Second-stage volatility timing
# =============================================================================
def _time_one_strategy(daily_proxy, monthly, strategy):
    """Apply lagged RV, AR(1), and GARCH timing to one factor-momentum return."""
    strategy_monthly = monthly[["date", strategy]].copy()
    proxy = daily_proxy.copy()
    proxy["month"] = proxy["date"].dt.to_period("M")
    monthly_variance = (
        proxy.groupby("month")[strategy].apply(realized_variance).rename(f"rv_{strategy}")
    )
    strategy_monthly = strategy_monthly.merge(
        monthly_variance.rename_axis("month").reset_index().assign(
            date=lambda data: data["month"].dt.to_timestamp("M")
        ).drop(columns="month"),
        on="date", how="left", validate="one_to_one",
    )
    add_managed_return(strategy_monthly, strategy, ANALYSIS_END)
    add_expanding_ar1_forecast(strategy_monthly, strategy, AR1_MIN_MONTHS)
    add_expanding_garch_forecast(proxy, strategy_monthly, strategy, GARCH_MIN_MONTHS)
    return strategy_monthly


def _select_strategy_columns(timed, strategy):
    """Keep returns, weights, forecasts, and persistence diagnostics."""
    rename = {
        strategy: strategy,
        f"managed_{strategy}": f"{strategy}_scaled_lagged_rv",
        f"managed_ar1_{strategy}": f"{strategy}_scaled_ar1",
        f"managed_garch_{strategy}": f"{strategy}_scaled_garch",
        f"weight_{strategy}": f"weight_{strategy}_scaled_lagged_rv",
        f"weight_ar1_{strategy}": f"weight_{strategy}_scaled_ar1",
        f"weight_garch_{strategy}": f"weight_{strategy}_scaled_garch",
        f"rv_{strategy}": f"rv_{strategy}",
        f"rv_forecast_ar1_{strategy}": f"rv_forecast_ar1_{strategy}",
        f"rv_forecast_garch_{strategy}": f"rv_forecast_garch_{strategy}",
        f"ar1_persistence_{strategy}": f"ar1_persistence_{strategy}",
        f"garch_persistence_{strategy}": f"garch_persistence_{strategy}",
    }
    return timed[["date", *rename]].rename(columns=rename)


# =============================================================================
# 3. Run and save
# =============================================================================
def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(FACTOR_MOMENTUM_LOG, mode="w"),
            logging.StreamHandler(),
        ],
    )
    monthly = pd.read_parquet(MANAGED_FACTORS).sort_values("date").reset_index(drop=True)
    daily, _ = load_factor_returns()
    daily["month"] = daily["date"].dt.to_period("M")

    for strategy, factor_columns in FACTOR_MOMENTUM_SETS.items():
        monthly[strategy] = factor_momentum_returns(
            monthly[factor_columns], lookback=12, name=strategy
        )
        valid = monthly.loc[monthly[strategy].notna(), "date"]
        logging.info(
            "%s | first_return=%s | returns=%s",
            strategy, valid.min().date(), len(valid),
        )

    output = monthly[["date", *FACTOR_MOMENTUM_SETS]].copy()
    parent_exposures = {}
    for strategy in FACTOR_MOMENTUM_SETS:
        logging.info("Second-stage timing for %s", strategy)
        parent_exposures[strategy] = _monthly_factor_exposures(monthly, strategy)
        proxy = _daily_proxy(daily, monthly, strategy)
        timed = _time_one_strategy(proxy, monthly, strategy)
        selected = _select_strategy_columns(timed, strategy)
        output = output.drop(columns=strategy).merge(
            selected, on="date", how="left", validate="one_to_one"
        )

    output.to_parquet(FACTOR_MOMENTUM_PANEL, index=False)
    TABLES.mkdir(parents=True, exist_ok=True)
    results = factor_momentum_results_table(
        output,
        monthly,
        parent_exposures,
        list(FACTOR_MOMENTUM_SETS),
        ANALYSIS_END,
        HAC_LAGS,
    )
    results.to_csv(FACTOR_MOMENTUM_TABLE, index=False)
    logging.info("Saved %s", FACTOR_MOMENTUM_PANEL)
    logging.info("Saved %s", FACTOR_MOMENTUM_TABLE)


if __name__ == "__main__":
    main()
