"""Read French source files and validate the processed monthly panel."""

# =============================================================================
# 0. Libraries and source paths
# =============================================================================
from io import StringIO
import re

import pandas as pd

from config import (
    FF3_DAILY,
    FF3_MONTHLY,
    FF5_DAILY,
    FF5_MONTHLY,
    MOM_DAILY,
    MOM_MONTHLY,
    NBER_RECESSIONS,
)


# =============================================================================
# 1. Source parsing and validation
# =============================================================================
def validate_unique_dates(data, name):
    """Return date-sorted data after checking missing and duplicate keys."""
    if data["date"].isna().any():
        raise ValueError(f"{name}: missing dates")
    if data["date"].duplicated().any():
        raise ValueError(f"{name}: duplicate dates")
    return data.sort_values("date").reset_index(drop=True)


def _read_metadata_wrapped_csv(path, frequency, columns):
    """Extract the requested frequency from a French metadata-wrapped CSV."""
    digits = 8 if frequency == "daily" else 6
    pattern = re.compile(rf"^\s*\d{{{digits}}},")
    lines = [line for line in path.read_text().splitlines() if pattern.match(line)]
    if not lines:
        raise ValueError(f"No {frequency} observations found in {path}")

    data = pd.read_csv(StringIO("date," + ",".join(columns) + "\n" + "\n".join(lines)))
    date_format = "%Y%m%d" if frequency == "daily" else "%Y%m"
    data["date"] = pd.to_datetime(data["date"].astype(str).str.strip(), format=date_format)
    if frequency == "monthly":
        data["date"] = data["date"].dt.to_period("M").dt.to_timestamp("M")
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="raise") / 100
        if data[column].isin([-0.9999, -9.99]).any():
            raise ValueError(f"{path}: missing-value codes in {column}")
    return validate_unique_dates(data, f"{path.name} {frequency}")


def read_french_factors(path, frequency, columns):
    """Read a daily or monthly multivariate French factor file."""
    return _read_metadata_wrapped_csv(path, frequency, columns)


def read_french_momentum(path, frequency):
    """Read a daily or monthly French momentum file."""
    return _read_metadata_wrapped_csv(path, frequency, ["mom"])


# =============================================================================
# 2. Raw factor panel
# =============================================================================
def load_factor_returns():
    """Load and merge official daily, monthly, and recession source series."""
    daily = read_french_factors(
        FF3_DAILY, "daily", ["mkt_rf", "smb", "hml", "rf"]
    ).drop(columns="rf")
    daily = daily.merge(
        read_french_momentum(MOM_DAILY, "daily"),
        on="date", how="outer", validate="one_to_one",
    )
    ff5_daily = read_french_factors(
        FF5_DAILY,
        "daily",
        ["ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma", "ff5_rf"],
    ).drop(columns="ff5_rf")
    daily = daily.merge(ff5_daily, on="date", how="outer", validate="one_to_one")
    daily = validate_unique_dates(daily, "combined daily factors")

    monthly = read_french_factors(
        FF3_MONTHLY, "monthly", ["mkt_rf", "smb", "hml", "rf"]
    ).drop(columns="rf")
    monthly = monthly.merge(
        read_french_momentum(MOM_MONTHLY, "monthly"),
        on="date", how="outer", validate="one_to_one",
    )
    ff5_monthly = read_french_factors(
        FF5_MONTHLY,
        "monthly",
        ["ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma", "ff5_rf"],
    ).drop(columns="ff5_rf")
    monthly = monthly.merge(ff5_monthly, on="date", how="outer", validate="one_to_one")

    recessions = pd.read_csv(NBER_RECESSIONS)
    recessions["date"] = (
        pd.to_datetime(recessions["date"], format="%Y-%m-%d")
        .dt.to_period("M").dt.to_timestamp("M")
    )
    recessions = validate_unique_dates(recessions, "NBER recessions")
    if not recessions["recession"].isin([0, 1]).all():
        raise ValueError("NBER recession indicator must contain only zero and one")
    monthly = monthly.merge(
        recessions, on="date", how="left", validate="one_to_one", indicator=True
    )
    if (monthly["_merge"] != "both").any():
        raise ValueError("Monthly factor observations lack NBER recession coverage")
    monthly = monthly.drop(columns="_merge")
    return daily, validate_unique_dates(monthly, "combined monthly factors")


# =============================================================================
# 3. Processed panels and coverage
# =============================================================================
def load_processed_panel(path, factors, end_date=None):
    """Load and structurally validate the analysis-ready monthly factor panel."""
    panel = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
    required = ["date", "recession", *factors, *[f"managed_{factor}" for factor in factors]]
    missing_columns = sorted(set(required) - set(panel.columns))
    if missing_columns:
        raise ValueError(f"Missing processed columns: {missing_columns}")
    panel = validate_unique_dates(panel, "processed factor panel")
    if end_date is not None:
        panel = panel.loc[panel["date"] <= end_date].copy()
    return panel


def series_coverage(data, column, prefix=None):
    """Summarize non-missing date coverage, optionally with prefixed keys."""
    valid = data.loc[data[column].notna(), "date"]
    statistics = {
        "series": column,
        "column": column,
        "start": valid.min(),
        "end": valid.max(),
        "observations": len(valid),
        "missing": int(data[column].isna().sum()),
    }
    if prefix is None:
        return statistics
    return {
        f"{prefix}_{key}": value
        for key, value in statistics.items()
        if key not in {"series", "column", "missing"}
    }
