"""Consistent plots for factor-return and volatility diagnostics."""

# =============================================================================
# 0. Libraries and common helpers
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from descriptive_statistics import cumulative_returns


def _factor_axes(factors, title, ylabel):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)
    fig.suptitle(title)
    for axis, label in zip(axes.flat, factors.values()):
        axis.set_title(label)
        axis.set_ylabel(ylabel)
    return fig, axes.flat


def _shade_recessions(axis, panel):
    """Shade months classified as NBER recessions."""
    axis.fill_between(
        panel["date"], 0, 1, where=panel["recession"].eq(1),
        color="0.85", step="mid", transform=axis.get_xaxis_transform(), zorder=0,
    )


# =============================================================================
# 1. Moreira–Muir replication and volatility comparison
# =============================================================================
def plot_realized_volatility(panel, factors):
    """Plot monthly realized volatility, corresponding to the core of Figure 2."""
    fig, axes = _factor_axes(factors, "Monthly realized volatility", "Realized volatility")
    for axis, factor in zip(axes, factors):
        _shade_recessions(axis, panel)
        axis.plot(panel["date"], np.sqrt(panel[f"rv_{factor}"]), linewidth=0.7)
    fig.tight_layout()
    return fig


def _volatility_comparison(panel, factor):
    """Return aligned monthly volatility estimates in percentage units."""
    columns = {
        f"rv_{factor}_lag1": "Lagged realized",
        f"rv_forecast_ar1_{factor}": "AR(1)",
        f"rv_forecast_garch_{factor}": "GARCH(1,1)",
    }
    comparison = panel[["date", *columns]].dropna().copy()
    if comparison.empty:
        raise ValueError(f"{factor}: no common volatility-estimate sample")
    if (comparison[list(columns)] < 0).any().any():
        raise ValueError(f"{factor}: negative variance estimate")
    comparison[list(columns)] = np.sqrt(comparison[list(columns)]) * 100
    return comparison.rename(columns=columns)


def plot_volatility_estimates(panel, factors):
    """Compare lagged realized, AR(1), and GARCH monthly volatility."""
    fig, axes = _factor_axes(
        factors,
        "Monthly volatility estimates",
        "Monthly volatility (%)",
    )
    colors = {
        "Lagged realized": "0.55",
        "AR(1)": "#1f77b4",
        "GARCH(1,1)": "#d62728",
    }
    for axis, factor in zip(axes, factors):
        comparison = _volatility_comparison(panel, factor)
        for method, color in colors.items():
            axis.plot(
                comparison["date"], comparison[method],
                label=method, color=color, linewidth=0.7, alpha=0.9,
            )
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def plot_market_volatility_sorts(sort_results):
    """Plot the four panels of Moreira–Muir Figure 1."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = sort_results["volatility_bucket"]
    panels = [
        ("mean_annual_pct", "Next-month mean return", "Annualized percent"),
        ("volatility_annual_pct", "Next-month volatility", "Annualized percent"),
        ("mean_divided_by_variance", "Mean divided by variance", "Ratio"),
        ("recession_probability", "Probability of recession", "Probability"),
    ]
    for axis, (column, title, ylabel) in zip(axes.flat, panels):
        axis.bar(x, sort_results[column])
        axis.set(title=title, xlabel="Lagged-volatility quintile", ylabel=ylabel)
    fig.tight_layout()
    return fig


def plot_market_cumulative_performance(panel):
    """Replicate Figure 3: cumulative wealth, annual returns, and drawdowns."""
    cumulative = cumulative_returns(panel, "mkt_rf")
    cumulative["unmanaged_rolling_year"] = (
        (1 + cumulative["mkt_rf"]).rolling(12).apply(np.prod, raw=True) - 1
    )
    cumulative["managed_rolling_year"] = (
        (1 + cumulative["managed_mkt_rf"]).rolling(12).apply(np.prod, raw=True) - 1
    )
    cumulative["unmanaged_drawdown"] = (
        cumulative["unmanaged_wealth"] / cumulative["unmanaged_wealth"].cummax() - 1
    )
    cumulative["managed_drawdown"] = (
        cumulative["managed_wealth"] / cumulative["managed_wealth"].cummax() - 1
    )

    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2)
    wealth_axis = fig.add_subplot(grid[0, :])
    return_axis = fig.add_subplot(grid[1, 0])
    drawdown_axis = fig.add_subplot(grid[1, 1])

    wealth_axis.plot(cumulative["date"], cumulative["unmanaged_wealth"], label="Unmanaged Mkt")
    wealth_axis.plot(cumulative["date"], cumulative["managed_wealth"], label="Managed Mkt")
    wealth_axis.set(
        title="Cumulative market performance",
        ylabel="Growth of 1 (log scale)",
        yscale="log",
    )
    wealth_axis.legend(frameon=False)

    return_axis.plot(cumulative["date"], cumulative["unmanaged_rolling_year"], label="Unmanaged")
    return_axis.plot(cumulative["date"], cumulative["managed_rolling_year"], label="Managed")
    return_axis.axhline(0, color="black", linewidth=0.5)
    return_axis.set(title="Rolling one-year return", ylabel="Return")

    drawdown_axis.plot(cumulative["date"], cumulative["unmanaged_drawdown"], label="Unmanaged")
    drawdown_axis.plot(cumulative["date"], cumulative["managed_drawdown"], label="Managed")
    drawdown_axis.set(title="Drawdown", ylabel="Drawdown")
    fig.tight_layout()
    return fig


# =============================================================================
# 2. Factor-momentum horse race
# =============================================================================
def _annotated_heatmap(axis, values, row_labels, column_labels, title, cmap, vmin, vmax):
    """Draw one compact annotated heatmap."""
    masked = np.ma.masked_invalid(values)
    image = axis.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axis.set(
        title=title,
        xticks=np.arange(len(column_labels)),
        xticklabels=column_labels,
        yticks=np.arange(len(row_labels)),
        yticklabels=row_labels,
    )
    axis.tick_params(axis="x", rotation=45)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    return image


def plot_pairwise_spanning_heatmap(alpha_t_statistics):
    """Plot directional alpha t-statistics for pairwise spanning."""
    labels = list(alpha_t_statistics.index)
    alpha_values = alpha_t_statistics.to_numpy(dtype=float)
    alpha_limit = max(2.0, np.nanmax(np.abs(alpha_values)))

    fig, axis = plt.subplots(figsize=(10, 8))
    alpha_image = _annotated_heatmap(
        axis, alpha_values, labels, list(alpha_t_statistics.columns),
        "Alpha t-statistic (NW12)", "coolwarm", -alpha_limit, alpha_limit,
    )
    axis.set(xlabel="Spanning strategy", ylabel="Dependent strategy")
    fig.colorbar(alpha_image, ax=axis, shrink=0.75)
    fig.suptitle("Directional spanning among single-layer factor-momentum strategies")
    fig.tight_layout()
    return fig


def plot_double_timing_heatmap(incremental_alpha_t):
    """Plot incremental-alpha t-statistics for the 3-by-3 double-timing design."""
    values = incremental_alpha_t.to_numpy(dtype=float)
    limit = max(2.0, np.nanmax(np.abs(values)))
    fig, axis = plt.subplots(figsize=(8, 6))
    image = _annotated_heatmap(
        axis,
        values,
        list(incremental_alpha_t.index),
        list(incremental_alpha_t.columns),
        "Incremental alpha t-statistic (NW12)",
        "coolwarm",
        -limit,
        limit,
    )
    axis.set(
        xlabel="Portfolio-level timing",
        ylabel="Factor-level timing before factor momentum",
    )
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle("Does a second volatility-timing layer add value?")
    fig.tight_layout()
    return fig


# =============================================================================
# 3. Selected-strategy comparison
# =============================================================================
def plot_selected_strategy_paths(scaled_returns, strategy_labels):
    """Plot cumulative wealth and drawdowns for selected momentum strategies."""
    path_colors = {"factor_momentum_lagged_rv": "tab:purple"}
    momentum_columns = ["mkt_rf", "mom", "managed_mom"]
    factor_momentum_columns = [
        "mkt_rf",
        "factor_momentum_original",
        "factor_momentum_original_scaled_lagged_rv",
        "factor_momentum_lagged_rv",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for column in momentum_columns:
        wealth = (1 + scaled_returns[column]).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        axes[0, 0].plot(
            scaled_returns["date"], wealth, label=strategy_labels[column],
            color=path_colors.get(column),
        )
        axes[1, 0].plot(
            scaled_returns["date"], drawdown, label=strategy_labels[column],
            color=path_colors.get(column),
        )
    for column in factor_momentum_columns:
        wealth = (1 + scaled_returns[column]).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        axes[0, 1].plot(
            scaled_returns["date"], wealth, label=strategy_labels[column],
            color=path_colors.get(column),
        )
        axes[1, 1].plot(
            scaled_returns["date"], drawdown, label=strategy_labels[column],
            color=path_colors.get(column),
        )

    axes[0, 0].set(title="Momentum strategies", ylabel="Growth of 1", yscale="log")
    axes[0, 1].set(title="Factor-momentum strategies", ylabel="Growth of 1", yscale="log")
    axes[1, 0].set(title="Momentum drawdowns", ylabel="Drawdown")
    axes[1, 1].set(title="Factor-momentum drawdowns", ylabel="Drawdown")
    for axis in axes.flat:
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Selected strategies at conventional momentum volatility")
    fig.tight_layout()
    return fig


def _selected_comparison_colors():
    """Keep strategy colors fixed in the additional comparison figures."""
    return {
        "mkt_rf": "0.35",
        "mom": "tab:orange",
        "managed_mom": "tab:green",
        "factor_momentum_original": "tab:orange",
        "factor_momentum_original_scaled_lagged_rv": "tab:green",
        "factor_momentum_lagged_rv": "tab:purple",
    }


def _mark_subperiods(axis, subperiod_ranges, show_labels=False):
    """Mark the fixed equal-length periods on a full-history axis."""
    background_colors = ["#edf3f8", "#f5f1e8", "#f0edf5"]
    ordered = subperiod_ranges.sort_values("period_number")
    for position, period in enumerate(ordered.itertuples(index=False)):
        axis.axvspan(
            period.start, period.end,
            color=background_colors[position % len(background_colors)],
            alpha=0.4, linewidth=0, zorder=0,
        )
        if position > 0:
            axis.axvline(period.start, color="0.45", linewidth=0.7, linestyle="--")
        if show_labels:
            midpoint = period.start + (period.end - period.start) / 2
            axis.text(
                midpoint, 0.98, f"Period {period.period_number}",
                transform=axis.get_xaxis_transform(), ha="center", va="top",
                color="0.3", fontsize=8,
            )


def plot_selected_full_return_series(
    scaled_returns, strategy_labels, subperiod_ranges
):
    """Plot one full-history monthly-return figure with marked subperiods."""
    groups = [
        ("Momentum strategies", ["mom", "managed_mom"]),
        (
            "Factor-momentum strategies",
            [
                "factor_momentum_original",
                "factor_momentum_original_scaled_lagged_rv",
                "factor_momentum_lagged_rv",
            ],
        ),
    ]
    colors = _selected_comparison_colors()
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for axis, (title, columns) in zip(axes, groups):
        _mark_subperiods(axis, subperiod_ranges, show_labels=True)
        for column in columns:
            axis.plot(
                scaled_returns["date"], scaled_returns[column] * 100,
                label=strategy_labels[column], color=colors[column], linewidth=0.7,
            )
        axis.axhline(0, color="black", linewidth=0.5)
        axis.set(title=title, ylabel="Monthly return (%)")
        axis.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("Month")
    fig.suptitle("Full-sample monthly returns with fixed subperiods")
    fig.tight_layout()
    return fig


def plot_selected_subperiod_paths(scaled_returns, strategy_labels, period):
    """Plot reset wealth and drawdowns for one fixed subperiod."""
    groups = [
        ("Momentum strategies", ["mom", "managed_mom"]),
        (
            "Factor-momentum strategies",
            [
                "factor_momentum_original",
                "factor_momentum_original_scaled_lagged_rv",
                "factor_momentum_lagged_rv",
            ],
        ),
    ]
    colors = _selected_comparison_colors()
    period_returns = (
        scaled_returns.loc[scaled_returns["date"].between(period.start, period.end)]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(period_returns) != period.observations:
        raise ValueError(
            f"Period {period.period_number}: expected {period.observations} "
            f"observations, found {len(period_returns)}"
        )

    initial_date = period_returns["date"].iloc[0].to_period("M").start_time
    wealth_dates = [initial_date, *period_returns["date"].tolist()]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex="col")
    for column_number, (group_title, columns) in enumerate(groups):
        for column in ["mkt_rf", *columns]:
            returns = period_returns[column]
            wealth = (1 + returns).cumprod()
            wealth_with_initial = np.r_[1.0, wealth.to_numpy()]
            drawdown = (
                wealth_with_initial / np.maximum.accumulate(wealth_with_initial) - 1
            )
            line_style = "--" if column == "mkt_rf" else "-"
            axes[0, column_number].plot(
                wealth_dates, np.r_[1.0, wealth.to_numpy()],
                label=strategy_labels[column], color=colors[column],
                linestyle=line_style, linewidth=1,
            )
            axes[1, column_number].plot(
                wealth_dates, drawdown,
                label=strategy_labels[column], color=colors[column],
                linestyle=line_style, linewidth=1,
            )
        axes[0, column_number].set(
            title=f"{group_title}: cumulative wealth", yscale="log"
        )
        axes[1, column_number].set_title(f"{group_title}: drawdown")
        axes[1, column_number].axhline(0, color="black", linewidth=0.5)
        axes[0, column_number].legend(frameon=False, fontsize=8)
        axes[1, column_number].set_xlabel("Month")

    axes[0, 0].set_ylabel("Growth of 1 (log scale)")
    axes[1, 0].set_ylabel("Drawdown")
    fig.suptitle(
        f"Period {period.period_number}: {period.period} "
        f"({period.observations} months)", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_selected_strategy_correlation(data, strategy_labels):
    """Plot full-sample Pearson correlations among selected strategies."""
    columns = list(strategy_labels)
    labels = [strategy_labels[column] for column in columns]
    correlations = data[columns].corr().to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(8, 6.5))
    image = _annotated_heatmap(
        axis, correlations, labels, labels,
        "Monthly return correlation", "coolwarm", -1, 1,
    )
    axis.set(xlabel="Strategy", ylabel="Strategy")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle("Full-sample correlations among selected strategies")
    fig.tight_layout()
    return fig


# =============================================================================
# 4. Momentum volatility-timing diagnostics
# =============================================================================
def plot_momentum_timing_series(data):
    """Plot lagged volatility, timing exposure, and momentum with recessions."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for axis in axes:
        _shade_recessions(axis, data)

    axes[0].plot(
        data["date"], np.sqrt(data["rv_mom_lag1"]) * 100,
        color="tab:blue", linewidth=0.8,
    )
    axes[0].set(title="Lagged realized momentum volatility", ylabel="Monthly volatility (%)")

    axes[1].plot(
        data["date"], data["weight_mom"], color="tab:orange", linewidth=0.8,
    )
    axes[1].axhline(1, color="black", linewidth=0.5, linestyle="--")
    axes[1].set(title="Uncapped inverse-variance weight", ylabel="Weight", yscale="log")

    axes[2].plot(
        data["date"], data["mom"] * 100, color="tab:green", linewidth=0.7,
    )
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set(title="Original momentum return", ylabel="Monthly return (%)", xlabel="Month")

    fig.suptitle("Momentum volatility timing and NBER recessions")
    fig.tight_layout()
    return fig


def plot_momentum_return_distributions(data):
    """Overlay each return histogram with its kernel-density estimate."""
    series = {
        "Momentum": (data["mom"].to_numpy() * 100, "tab:orange"),
        "Lagged-RV volatility-timed momentum": (
            data["managed_mom"].to_numpy() * 100, "tab:green"
        ),
    }
    lower = min(values.min() for values, _ in series.values())
    upper = max(values.max() for values, _ in series.values())
    bins = np.linspace(lower, upper, 61)
    grid = np.linspace(lower, upper, 500)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
    for axis, (label, (values, color)) in zip(axes, series.items()):
        axis.hist(
            values, bins=bins, density=True, alpha=0.4,
            label="Histogram", color=color,
        )
        density = gaussian_kde(values)
        axis.plot(
            grid, density(grid), label="Kernel density",
            color="black", linewidth=1.4,
        )
        axis.axvline(0, color="black", linewidth=0.5)
        axis.axvline(
            np.mean(values), color=color, linewidth=0.9,
            linestyle="--", label="Mean",
        )
        axis.set(title=label, xlabel="Monthly return (%)")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Density")
    fig.suptitle("Monthly-return histograms with kernel densities")
    fig.tight_layout()
    return fig


def plot_momentum_return_cdf(data):
    """Compare empirical cumulative distributions of monthly returns."""
    fig, axis = plt.subplots(figsize=(11, 5.5))
    series = {
        "Momentum": (data["mom"].to_numpy() * 100, "tab:orange"),
        "Lagged-RV volatility-timed momentum": (
            data["managed_mom"].to_numpy() * 100, "tab:green"
        ),
    }
    for label, (values, color) in series.items():
        ordered = np.sort(values)
        probabilities = np.arange(1, len(ordered) + 1) / len(ordered) * 100
        axis.step(ordered, probabilities, where="post", label=label, color=color)
    axis.axvline(0, color="black", linewidth=0.5)
    axis.set(
        title="Empirical CDF of monthly returns",
        xlabel="Monthly return (%)",
        ylabel="Cumulative probability (%)",
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_momentum_volatility_states(state_table):
    """Compare returns, risk, and exposure across lagged-volatility quintiles."""
    x = state_table["volatility_quintile"].to_numpy()
    width = 0.36
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    axes[0].bar(
        x - width / 2, state_table["mom_mean_monthly_pct"], width,
        label="Momentum", color="tab:orange",
    )
    axes[0].bar(
        x + width / 2, state_table["managed_mean_monthly_pct"], width,
        label="Volatility-timed momentum", color="tab:green",
    )
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set(title="Mean next-month return", ylabel="Monthly mean (%)")
    axes[0].legend(frameon=False)

    axes[1].bar(
        x - width / 2, state_table["mom_std_monthly_pct"], width,
        label="Momentum", color="tab:orange",
    )
    axes[1].bar(
        x + width / 2, state_table["managed_std_monthly_pct"], width,
        label="Volatility-timed momentum", color="tab:green",
    )
    axes[1].set(title="Next-month return volatility", ylabel="Monthly std (%)")

    axes[2].bar(
        x - width / 2, state_table["mom_annualized_sharpe"], width,
        label="Momentum", color="tab:orange",
    )
    axes[2].bar(
        x + width / 2, state_table["managed_annualized_sharpe"], width,
        label="Volatility-timed momentum", color="tab:green",
    )
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set(title="Within-quintile Sharpe ratio", ylabel="Annualized Sharpe ratio")

    axes[3].plot(
        x, state_table["average_weight"], marker="o", color="tab:blue",
    )
    axes[3].axhline(1, color="black", linewidth=0.5, linestyle="--")
    axes[3].set(
        title="Average inverse-variance weight",
        xlabel="Lagged momentum-volatility quintile (low to high)",
        ylabel="Weight",
        xticks=x,
    )

    fig.suptitle("Momentum outcomes across lagged-volatility states")
    fig.tight_layout()
    return fig


def plot_momentum_rolling_relationships(rolling_relationships):
    """Plot trailing Mom correlation and loading for managed momentum."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for axis in axes:
        _shade_recessions(axis, rolling_relationships)

    axes[0].plot(
        rolling_relationships["date"],
        rolling_relationships["rolling_correlation"],
        color="tab:blue", linewidth=1,
    )
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set(title="Trailing 120-month return correlation", ylabel="Correlation")

    axes[1].plot(
        rolling_relationships["date"],
        rolling_relationships["rolling_mom_loading"],
        color="tab:purple", linewidth=1,
    )
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axhline(1, color="black", linewidth=0.5, linestyle="--")
    axes[1].set(
        title="Trailing 120-month loading on original momentum",
        ylabel="Mom loading", xlabel="Month",
    )
    fig.suptitle("Changing comovement between volatility-timed and original momentum")
    fig.tight_layout()
    return fig


def plot_momentum_timing_payoff(payoff):
    """Plot the cumulative equal-volatility timing payoff with recessions."""
    fig, axis = plt.subplots(figsize=(14, 5))
    _shade_recessions(axis, payoff)
    axis.plot(
        payoff["date"], payoff["cumulative_timing_payoff_pct"],
        color="tab:purple", linewidth=1,
    )
    axis.axhline(0, color="black", linewidth=0.5)
    axis.set(
        title="Cumulative equal-volatility timing payoff",
        xlabel="Month",
        ylabel="Cumulative return difference (percentage points)",
    )
    fig.tight_layout()
    return fig
