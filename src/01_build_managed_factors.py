"""Stage 1: construct volatility-managed returns for six US factors."""

# =============================================================================
# 0. Libraries
# =============================================================================
import logging

from config import (
    FACTORS,
    MANAGED_FACTORS,
    PROCESSED,
    LOGS,
    BUILD_LOG,
    ANALYSIS_END,
    AR1_MIN_MONTHS,
    GARCH_MIN_MONTHS,
)
from data import load_factor_returns
from volatility_management import add_managed_return, realized_variance
from variance_forecasting import add_expanding_ar1_forecast, add_expanding_garch_forecast


# =============================================================================
# 1. Pipeline
# =============================================================================
def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(BUILD_LOG, mode="w"), logging.StreamHandler()],
    )
    daily, monthly = load_factor_returns()
    daily["month"] = daily["date"].dt.to_period("M")

    for factor in FACTORS:
        daily_valid = daily.loc[daily[factor].notna(), ["date", factor]]
        monthly_valid = monthly.loc[monthly[factor].notna(), ["date", factor]]
        logging.info(
            "%s inputs | daily_n=%s daily_range=%s:%s | monthly_n=%s monthly_range=%s:%s",
            factor,
            len(daily_valid),
            daily_valid["date"].min().date(),
            daily_valid["date"].max().date(),
            len(monthly_valid),
            monthly_valid["date"].min().date(),
            monthly_valid["date"].max().date(),
        )
        monthly_variance = (
            daily.groupby("month")[factor].apply(realized_variance).rename(f"rv_{factor}")
        )
        monthly = monthly.merge(
            monthly_variance.rename_axis("month").reset_index().assign(
                date=lambda x: x["month"].dt.to_timestamp("M")
            ).drop(columns="month"),
            on="date", how="left", validate="one_to_one",
        )
        add_managed_return(monthly, factor, ANALYSIS_END)
        add_expanding_ar1_forecast(monthly, factor, AR1_MIN_MONTHS)
        add_expanding_garch_forecast(daily, monthly, factor, GARCH_MIN_MONTHS)

    monthly.to_parquet(MANAGED_FACTORS, index=False)
    logging.info("Saved %s rows and %s columns to %s", len(monthly), len(monthly.columns), MANAGED_FACTORS)



if __name__ == "__main__":
    main()
