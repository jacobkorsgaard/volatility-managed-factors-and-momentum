"""Generic time-series factor-momentum signals and portfolio returns."""

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import pandas as pd


# =============================================================================
# 1. Signals and positions
# =============================================================================
def factor_momentum_positions(factor_returns, lookback=12):
    """Form equal-weight sign positions from the preceding 12 monthly returns.

    At month t, each signal uses only returns from t-lookback through t-1.
    A factor must have a complete lookback window and a nonmissing month-t
    return to enter the portfolio. An exactly zero trailing mean gives a zero
    position but remains in the equal-weight denominator.
    """
    if not isinstance(factor_returns, pd.DataFrame) or factor_returns.empty:
        raise ValueError("factor_returns must be a nonempty DataFrame")
    if lookback < 1:
        raise ValueError("lookback must be a positive integer")
    if factor_returns.columns.duplicated().any():
        raise ValueError("factor_returns contains duplicate column names")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in factor_returns.dtypes):
        raise TypeError("all factor-return columns must be numeric")

    # shift(1) makes the information set explicit: the month-t position cannot
    # use the month-t return. rolling requires 12 complete calendar rows.
    trailing_average = factor_returns.rolling(
        window=lookback, min_periods=lookback
    ).mean().shift(1)
    signals = np.sign(trailing_average)
    available = signals.notna() & factor_returns.notna()
    available_count = available.sum(axis=1).replace(0, np.nan)
    positions = signals.where(available).div(available_count, axis=0)
    return positions


# =============================================================================
# 2. Portfolio returns
# =============================================================================
def factor_momentum_returns(factor_returns, lookback=12, name="factor_momentum"):
    """Return the monthly equal-weight time-series factor-momentum portfolio."""
    positions = factor_momentum_positions(factor_returns, lookback)
    portfolio = (positions * factor_returns).sum(axis=1, min_count=1)
    portfolio.name = name
    return portfolio


# =============================================================================
# 3. Daily portfolio proxy for second-stage volatility timing
# =============================================================================
def daily_factor_momentum_proxy(
    daily_factor_returns,
    monthly_signal_returns,
    monthly_scaling_weights=None,
    lookback=12,
):
    """Construct daily portfolio returns from positions fixed before each month.

    The monthly signal-return and daily raw-return columns must identify the
    same factors. Optional scaling weights are also monthly and are multiplied
    by the factor-momentum positions before being applied to daily returns.
    """
    factors = list(monthly_signal_returns.columns)
    if set(daily_factor_returns.columns) != set(factors):
        raise ValueError("daily and monthly factor columns must match")
    daily_factor_returns = daily_factor_returns[factors]
    positions = factor_momentum_positions(monthly_signal_returns, lookback)
    if monthly_scaling_weights is None:
        exposures = positions
    else:
        if set(monthly_scaling_weights.columns) != set(factors):
            raise ValueError("scaling-weight and signal-return columns must match")
        exposures = positions * monthly_scaling_weights[factors]

    if not isinstance(monthly_signal_returns.index, pd.DatetimeIndex):
        raise TypeError("monthly_signal_returns must have a DatetimeIndex")
    if not isinstance(daily_factor_returns.index, pd.DatetimeIndex):
        raise TypeError("daily_factor_returns must have a DatetimeIndex")

    exposure_by_month = exposures.copy()
    exposure_by_month.index = exposure_by_month.index.to_period("M")
    daily_month = daily_factor_returns.index.to_period("M")
    daily_exposures = exposure_by_month.reindex(daily_month)
    daily_exposures.index = daily_factor_returns.index
    portfolio = (daily_exposures * daily_factor_returns).sum(axis=1, min_count=1)
    portfolio.name = "daily_factor_momentum"
    return portfolio
