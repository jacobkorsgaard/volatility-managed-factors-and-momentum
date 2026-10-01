"""Project-wide paths and factor-analysis settings."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"
LOGS = ROOT / "output" / "logs"

FF3_DAILY = RAW / "F-F_Research_Data_Factors_daily.csv"
FF3_MONTHLY = RAW / "F-F_Research_Data_Factors.csv"
FF5_DAILY = RAW / "F-F_Research_Data_5_Factors_2x3_daily.csv"
FF5_MONTHLY = RAW / "F-F_Research_Data_5_Factors_2x3.csv"
MOM_DAILY = RAW / "F-F_Momentum_Factor_daily.csv"
MOM_MONTHLY = RAW / "F-F_Momentum_Factor.csv"
NBER_RECESSIONS = RAW / "nber_recessions.csv"

MANAGED_FACTORS = PROCESSED / "managed_factors.parquet"
FACTOR_MOMENTUM_PANEL = PROCESSED / "factor_momentum_strategies.parquet"
SPANNING_TABLE = TABLES / "moreira_muir_table_i_spanning.csv"
MVE_TABLE = TABLES / "moreira_muir_table_ii_mve.csv"
FACTOR_MOMENTUM_TABLE = TABLES / "factor_momentum_build_diagnostics.csv"
BUILD_LOG = LOGS / "build_managed_factors.log"
FACTOR_MOMENTUM_LOG = LOGS / "build_factor_momentum_strategies.log"

# April 2015 is inferred from Table I's observation counts. The paper describes
# the endpoint only as 2015, so this is documented as an inferred vintage cutoff.
ANALYSIS_END = pd.Timestamp("2015-04-30")
HAC_LAGS = 12
AR1_MIN_MONTHS = 120
GARCH_MIN_MONTHS = 120

FACTORS = {
    "mkt_rf": "Mkt",
    "smb": "SMB",
    "hml": "HML",
    "mom": "Mom",
    "rmw": "RMW",
    "cma": "CMA",
}

FACTOR_MOMENTUM_SETS = {
    "factor_momentum_original": list(FACTORS),
    "factor_momentum_lagged_rv": [f"managed_{factor}" for factor in FACTORS],
    "factor_momentum_ar1": [f"managed_ar1_{factor}" for factor in FACTORS],
    "factor_momentum_garch": [f"managed_garch_{factor}" for factor in FACTORS],
}

FACTOR_MOMENTUM_WEIGHT_SETS = {
    "factor_momentum_original": None,
    "factor_momentum_lagged_rv": [f"weight_{factor}" for factor in FACTORS],
    "factor_momentum_ar1": [f"weight_ar1_{factor}" for factor in FACTORS],
    "factor_momentum_garch": [f"weight_garch_{factor}" for factor in FACTORS],
}

OPPORTUNITY_SETS = {
    "Mkt": ["mkt_rf"],
    "FF3": ["mkt_rf", "smb", "hml"],
    "FF3 + Mom": ["mkt_rf", "smb", "hml", "mom"],
    "FF5": ["mkt_rf", "smb", "hml", "rmw", "cma"],
    "FF5 + Mom": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"],
}

# Moreira-Muir Table II uses each factor model's own source definitions and
# volatility-times the aggregate in-sample MVE portfolio.
REPLICATION_MVE_SETS = {
    "Mkt": ["mkt_rf"],
    "FF3": ["mkt_rf", "smb", "hml"],
    "FF3 + Mom": ["mkt_rf", "smb", "hml", "mom"],
    "FF5": ["ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma"],
    "FF5 + Mom": ["ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma", "mom"],
}
