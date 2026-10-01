"""Real-time variance forecasts and forecast-managed factor returns."""

# =============================================================================
# 0. Libraries
# =============================================================================
import logging

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter


# =============================================================================
# 1. Expanding log-realized-variance AR(1)
# =============================================================================
def _fit_log_variance_ar1(log_variance):
    """Estimate log(RV_t) on a constant and log(RV_{t-1}) by OLS."""
    dependent = log_variance[1:]
    regressors = np.column_stack([np.ones(len(log_variance) - 1), log_variance[:-1]])
    parameters = np.linalg.lstsq(regressors, dependent, rcond=None)[0]
    residuals = dependent - regressors @ parameters
    residual_variance = residuals @ residuals / (len(residuals) - len(parameters))
    return parameters[0], parameters[1], residual_variance, regressors @ parameters


def add_expanding_ar1_forecast(data, factor, minimum_months=120):
    """Add expanding AR(1) forecasts and returns without using future data."""
    variance_column = f"rv_{factor}"
    valid = data.loc[data[variance_column].notna(), ["date", variance_column]].copy()
    if len(valid) <= minimum_months:
        raise ValueError(f"{factor}: insufficient realized-variance history")
    if (valid[variance_column] <= 0).any():
        raise ValueError(f"{factor}: AR(1) requires strictly positive realized variance")

    log_variance = np.log(valid[variance_column].to_numpy())
    output_columns = {
        f"rv_forecast_ar1_{factor}": np.nan,
        f"rv_forecast_ar1_uncorrected_{factor}": np.nan,
        f"ar1_intercept_{factor}": np.nan,
        f"ar1_persistence_{factor}": np.nan,
        f"ar1_residual_variance_{factor}": np.nan,
        f"ar1_stationary_{factor}": False,
        f"rv_forecast_error_ar1_{factor}": np.nan,
    }
    for column, default in output_columns.items():
        data[column] = default

    # The initial 120-month window calibrates one fixed normalization constant.
    intercept, persistence, residual_variance, fitted_log = _fit_log_variance_ar1(
        log_variance[:minimum_months]
    )
    fitted_variance = np.exp(fitted_log + 0.5 * residual_variance)
    calibration_dates = valid["date"].iloc[1:minimum_months]
    calibration = data.set_index("date").loc[calibration_dates, [factor]].copy()
    calibration["unscaled"] = calibration[factor].to_numpy() / fitted_variance
    calibration = calibration.dropna()
    normalization = (
        calibration[factor].std(ddof=1) / calibration["unscaled"].std(ddof=1)
    )

    # Forecast the target month using only realized variances through the prior month.
    for target_position in range(minimum_months, len(valid)):
        training = log_variance[:target_position]
        intercept, persistence, residual_variance, _ = _fit_log_variance_ar1(training)
        forecast_log = intercept + persistence * training[-1]
        forecast_uncorrected = np.exp(forecast_log)
        forecast = np.exp(forecast_log + 0.5 * residual_variance)
        target_index = valid.index[target_position]
        data.loc[target_index, f"rv_forecast_ar1_{factor}"] = forecast
        data.loc[target_index, f"rv_forecast_ar1_uncorrected_{factor}"] = forecast_uncorrected
        data.loc[target_index, f"ar1_intercept_{factor}"] = intercept
        data.loc[target_index, f"ar1_persistence_{factor}"] = persistence
        data.loc[target_index, f"ar1_residual_variance_{factor}"] = residual_variance
        data.loc[target_index, f"ar1_stationary_{factor}"] = abs(persistence) < 1
        data.loc[target_index, f"rv_forecast_error_ar1_{factor}"] = (
            valid.iloc[target_position][variance_column] - forecast
        )

    forecast_column = f"rv_forecast_ar1_{factor}"
    data[f"weight_ar1_{factor}"] = normalization / data[forecast_column]
    data[f"managed_ar1_{factor}"] = data[f"weight_ar1_{factor}"] * data[factor]
    data[f"normalization_ar1_{factor}"] = normalization

    forecast_rows = data[forecast_column].notna()
    stationary_share = data.loc[forecast_rows, f"ar1_stationary_{factor}"].mean()
    logging.info(
        "%s AR(1) | first_forecast=%s | forecasts=%s | stationary_share=%.3f | c=%.10g",
        factor,
        data.loc[forecast_rows, "date"].min().date(),
        int(forecast_rows.sum()),
        stationary_share,
        normalization,
    )
    return normalization


# =============================================================================
# 2. GARCH(1,1) estimated on daily returns for monthly forecasts
# =============================================================================
def _transform_garch_parameters(raw_parameters):
    """Enforce only the nonnegativity conditions required for conditional variance."""
    omega, alpha, beta = np.exp(raw_parameters)
    return omega, alpha, beta


def _inverse_garch_parameters(omega, alpha, beta):
    """Map valid GARCH parameters to unconstrained optimizer values."""
    return np.log(np.maximum([omega, alpha, beta], 1e-12))


def _filter_garch_variance(residuals, omega, alpha, beta):
    """Filter h[d] from returns known through trading day d-1."""
    initial_variance = max(float(np.var(residuals, ddof=1)), 1e-12)
    innovations = np.empty(len(residuals))
    innovations[0] = omega
    innovations[1:] = omega + alpha * residuals[:-1] ** 2
    variance, _ = lfilter(
        [1.0], [1.0, -beta], innovations, zi=[beta * initial_variance]
    )
    return np.maximum(variance, 1e-12)


def _garch_objective(raw_parameters, residuals):
    """Calculate the average Gaussian quasi-negative-log-likelihood."""
    omega, alpha, beta = _transform_garch_parameters(raw_parameters)
    variance = _filter_garch_variance(residuals, omega, alpha, beta)
    if not np.isfinite(variance).all():
        return 1e100
    return 0.5 * np.mean(np.log(variance) + residuals**2 / variance)


def _fit_garch11(returns, starting_parameters=None):
    """Estimate a constant-mean Gaussian GARCH(1,1)."""
    returns = np.asarray(returns, dtype=float)
    mean = float(returns.mean())
    residuals = returns - mean
    unconditional_variance = max(float(np.var(residuals, ddof=1)), 1e-12)
    if starting_parameters is None:
        alpha, beta = 0.05, 0.90
        starting_parameters = _inverse_garch_parameters(
            unconditional_variance * (1 - alpha - beta), alpha, beta
        )
    result = minimize(
        _garch_objective,
        starting_parameters,
        args=(residuals,),
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"GARCH optimization failed: {result.message}")
    omega, alpha, beta = _transform_garch_parameters(result.x)
    return {
        "mean": mean,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "variance": _filter_garch_variance(residuals, omega, alpha, beta),
        "residuals": residuals,
        "raw_parameters": result.x,
    }


def _forecast_monthly_garch_variance(fit, trading_days):
    """Sum daily variance forecasts formed at the preceding month-end."""
    daily_forecast = (
        fit["omega"]
        + fit["alpha"] * fit["residuals"][-1] ** 2
        + fit["beta"] * fit["variance"][-1]
    )
    total_forecast = daily_forecast
    persistence = fit["alpha"] + fit["beta"]
    for _ in range(1, trading_days):
        daily_forecast = fit["omega"] + persistence * daily_forecast
        total_forecast += daily_forecast
    return total_forecast


def add_expanding_garch_forecast(daily, monthly, factor, minimum_months=120):
    """Add a real-time monthly GARCH forecast, weight, and managed factor return."""
    valid_daily = daily.loc[daily[factor].notna(), ["date", "month", factor]].copy()
    available_months = valid_daily["month"].drop_duplicates().sort_values()
    if len(available_months) <= minimum_months:
        raise ValueError(f"{factor}: insufficient daily history for GARCH")
    initial_month = available_months.iloc[minimum_months - 1]
    initial_daily = valid_daily.loc[valid_daily["month"] <= initial_month]
    initial_fit = _fit_garch11(initial_daily[factor].to_numpy())

    # Use fitted initial-sample daily variances only to fix the scale constant.
    calibration = initial_daily[["date", "month"]].copy()
    calibration["fitted_variance"] = initial_fit["variance"]
    calibration_variance = calibration.groupby("month")["fitted_variance"].sum()
    monthly_returns = monthly.loc[monthly[factor].notna(), ["date", factor]].copy()
    monthly_returns["month"] = monthly_returns["date"].dt.to_period("M")
    calibration_monthly = monthly_returns.loc[
        monthly_returns["month"] <= initial_month
    ].set_index("month")
    calibration_monthly["fitted_variance"] = calibration_variance
    calibration_monthly = calibration_monthly.dropna(subset=[factor, "fitted_variance"])
    unscaled = calibration_monthly[factor] / calibration_monthly["fitted_variance"]
    normalization = calibration_monthly[factor].std(ddof=1) / unscaled.std(ddof=1)

    columns = {
        f"rv_forecast_garch_{factor}": np.nan,
        f"garch_omega_{factor}": np.nan,
        f"garch_alpha_{factor}": np.nan,
        f"garch_beta_{factor}": np.nan,
        f"garch_persistence_{factor}": np.nan,
    }
    for column, default in columns.items():
        monthly[column] = default

    starting_parameters = initial_fit["raw_parameters"]
    target_months = available_months.loc[available_months > initial_month]
    for target_month in target_months:
        training = valid_daily.loc[valid_daily["month"] < target_month, factor].to_numpy()
        target_days = int((valid_daily["month"] == target_month).sum())
        fit = _fit_garch11(training, starting_parameters)
        starting_parameters = fit["raw_parameters"]
        forecast = _forecast_monthly_garch_variance(fit, target_days)
        target_date = target_month.to_timestamp("M")
        target_index = monthly.index[monthly["date"] == target_date]
        if len(target_index) != 1:
            raise ValueError(f"{factor}: missing or duplicate monthly return for {target_month}")
        index = target_index[0]
        monthly.loc[index, f"rv_forecast_garch_{factor}"] = forecast
        monthly.loc[index, f"garch_omega_{factor}"] = fit["omega"]
        monthly.loc[index, f"garch_alpha_{factor}"] = fit["alpha"]
        monthly.loc[index, f"garch_beta_{factor}"] = fit["beta"]
        monthly.loc[index, f"garch_persistence_{factor}"] = fit["alpha"] + fit["beta"]

    forecast_column = f"rv_forecast_garch_{factor}"
    monthly[f"weight_garch_{factor}"] = normalization / monthly[forecast_column]
    monthly[f"managed_garch_{factor}"] = monthly[f"weight_garch_{factor}"] * monthly[factor]
    monthly[f"normalization_garch_{factor}"] = normalization

    forecast_rows = monthly[forecast_column].notna()
    persistence = monthly.loc[forecast_rows, f"garch_persistence_{factor}"]
    if (monthly.loc[forecast_rows, forecast_column] <= 0).any():
        raise ValueError(f"{factor}: invalid GARCH forecast")
    logging.info(
        "%s GARCH | first_forecast=%s | forecasts=%s | persistence=%.4f:%.4f | c=%.10g",
        factor,
        monthly.loc[forecast_rows, "date"].min().date(),
        int(forecast_rows.sum()),
        persistence.min(), persistence.max(), normalization,
    )
    return normalization
