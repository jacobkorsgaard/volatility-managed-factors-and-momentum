"""Factor-spanning regressions with monthly Newey–West inference."""

# =============================================================================
# 0. Libraries and specifications
# =============================================================================
import numpy as np
import pandas as pd

from utils import fit_hac_regression


ALL_REGRESSORS = [
    "mkt_rf", "smb", "hml", "mom", "ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma"
]


# =============================================================================
# 1. Factor-spanning regressions
# =============================================================================
def estimate_spanning_regression(data, managed_factor, label, specification, regressors, hac_lags):
    """Estimate one monthly factor-spanning regression."""
    dependent = f"managed_{managed_factor}"
    model, standard_errors, _, nobs = fit_hac_regression(
        data, dependent, regressors, hac_lags, use_correction=True
    )
    row = {
        "managed_factor": label,
        "specification": specification,
        "alpha_annual_pct": model.params["const"] * 1200,
        "alpha_se_hac12_annual_pct": standard_errors["const"] * 1200,
        "alpha_t_hac12": model.params["const"] / standard_errors["const"],
        "nobs": nobs,
        "r_squared": model.rsquared,
    }
    for regressor in ALL_REGRESSORS:
        row[f"{regressor}_loading"] = model.params.get(regressor, np.nan)
        row[f"{regressor}_se_hac12"] = standard_errors.get(regressor, np.nan)
        row[f"{regressor}_t_hac12"] = (
            model.params[regressor] / standard_errors[regressor]
            if regressor in model.params else np.nan
        )
    return row


def run_spanning_tests(data, factors, specification, hac_lags):
    """Run one spanning specification across every managed factor."""
    rows = []
    ff5_controls = ["ff5_mkt_rf", "ff5_smb", "ff5_hml", "rmw", "cma"]
    ff5_members = {"mkt_rf", "smb", "hml", "rmw", "cma"}
    for factor, label in factors.items():
        if specification == "Own factor":
            regressors = [factor]
        elif specification == "FF3":
            regressors = list(dict.fromkeys([factor, "mkt_rf", "smb", "hml"]))
        elif specification == "FF5":
            regressors = ff5_controls if factor in ff5_members else [factor, *ff5_controls]
        else:
            raise ValueError(f"Unknown specification: {specification}")
        rows.append(
            estimate_spanning_regression(data, factor, label, specification, regressors, hac_lags)
        )
    return pd.DataFrame(rows)


# =============================================================================
# 2. Recession-interaction regressions
# =============================================================================
def run_recession_beta_tests(data, factors, hac_lags):
    """Estimate Moreira–Muir Table III recession-beta specifications."""
    rows = []
    for factor, label in factors.items():
        dependent = f"managed_{factor}"
        interaction = f"{factor}_x_recession"
        regression_data = data[[dependent, factor, "recession"]].dropna().copy()
        regression_data[interaction] = regression_data[factor] * regression_data["recession"]
        regressors = ["recession", factor, interaction]
        model, standard_errors, covariance, nobs = fit_hac_regression(
            regression_data,
            dependent,
            regressors,
            hac_lags,
            use_correction=True,
        )
        recession_beta = model.params[factor] + model.params[interaction]
        recession_beta_se = np.sqrt(
            covariance.loc[factor, factor]
            + covariance.loc[interaction, interaction]
            + 2 * covariance.loc[factor, interaction]
        )
        rows.append(
            {
                "managed_factor": label,
                "nonrecession_beta": model.params[factor],
                "nonrecession_beta_se_hac12": standard_errors[factor],
                "recession_beta_interaction": model.params[interaction],
                "recession_beta_interaction_se_hac12": standard_errors[interaction],
                "recession_beta": recession_beta,
                "recession_beta_se_hac12": recession_beta_se,
                "nobs": nobs,
                "r_squared": model.rsquared,
            }
        )
    return pd.DataFrame(rows)
