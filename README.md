# Volatility-Managed Factors and Momentum

Factor risk varies over time, but higher volatility need not be accompanied by proportionally higher expected returns. Reducing exposure after volatile months and increasing it after calm months may therefore improve risk-adjusted performance. This repository examines that proposition for US equity factors, with particular attention to momentum.

The analysis begins with a replication of Moreira and Muir (2017), then asks whether the results depend on the volatility estimate and whether volatility management adds to factor momentum. It distinguishes conventional equity momentum—the published winner-minus-loser factor—from time-series momentum across factors. A final diagnostic examines which volatility states account for the gains from managing conventional momentum and how those gains vary across market conditions and historical periods.

## Empirical approach

The project uses the market, size, value, profitability, investment, and momentum factors from the Kenneth R. French Data Library. The baseline strategy scales each monthly factor return inversely with the preceding month's realized variance, calculated from demeaned daily returns. Alternatives use expanding AR(1) forecasts of log realized variance and GARCH(1,1) forecasts estimated from daily returns, each initialized with 120 months of data and refitted monthly.

Factor momentum takes equally weighted long or short positions according to the sign of each available factor's average return over the preceding 12 months, without a skip month. Volatility timing is applied before factor-momentum construction, after portfolio aggregation, or at both stages. Comparisons use Sharpe ratios, downside-risk measures, and spanning regressions to assess whether a strategy adds returns beyond its benchmark. Monthly regression inference uses Newey–West standard errors with 12 lags; paper-specific inference is distinguished in the replication.

The replication ends in April 2015. Extensions use the available data through July 2026, with common samples defined within each comparison. The project covers factor portfolios and factor momentum; liquidity portfolios are outside its scope.

## Main findings

- **Momentum provides the clearest evidence for volatility management.** Its alpha relative to the original momentum factor is significant under all three volatility estimates and across the reported subperiods, although its magnitude declines in the later sample. The evidence is weaker and less persistent for other factors.
- **Lagged-variance timing improves momentum's historical risk-adjusted performance and downside outcomes.** Over August 1927–July 2026, its annualized Sharpe ratio rises from 0.45 to 0.94, while maximum drawdown falls from approximately 78% to 32% on the stored return scales.
- **Volatility timing also improves factor momentum, and timing order matters.** In the selected-strategy comparison, annualized Sharpe ratios are 0.49 for original factor momentum, 0.70 when lagged-variance timing is applied to the aggregate portfolio, and 0.77 when factor momentum is constructed from individually timed factors. The two managed constructions do not span one another. In the broader comparison, adding a second timing layer produces no incremental alpha significant at 5% relative to its corresponding single-layer parent.
- **Momentum's average timing gain comes mainly from increasing exposure in calm periods.** After equalizing full-sample volatility, the lowest lagged-volatility quintile contributes 0.603 percentage points to the average monthly payoff of 0.663%. Reducing exposure after volatile months contributes more clearly to downside protection. The payoff is historically uneven and much flatter after the late 1990s.

These are historical gross-return findings under the normalization and sample conventions below. The notebooks report the supporting estimates and diagnostics.

## Notebooks

Read the six notebooks in order. Embedded outputs allow the results to be inspected on GitHub without rerunning the analysis.

| Notebook | Coverage |
|---|---|
| [01 — Summary Statistics](notebooks/01_summary_statistics.ipynb) | Documents data sources, dimensions, factor and strategy coverage, and structural validation. |
| [02 — Moreira–Muir Replication](notebooks/02_moreira_muir_replication.ipynb) | Replicates the available parts of Figures 1–3 and Tables I–III, comparing selected estimates with published benchmarks. |
| [03 — Volatility Forecast Comparison](notebooks/03_volatility_horse_race.ipynb) | Compares lagged realized variance, AR(1), and GARCH timing using performance, spanning tests, and subperiod evidence. |
| [04 — Factor Momentum Comparison](notebooks/04_factor_momentum_horse_race.ipynb) | Evaluates timing before and after aggregation, matched equal-weight benchmarks, and the incremental contribution of a second timing layer. |
| [05 — Selected Strategy Comparison](notebooks/05_selected_strategy_comparison.ipynb) | Compares conventional and factor momentum with selected lagged-variance strategies through performance, downside risk, spanning, and subperiod analysis. |
| [06 — Momentum Volatility Timing](notebooks/06_momentum_volatility_timing.ipynb) | Examines exposure, return distributions, volatility-state payoff contributions, and recession-dependent comovement. |

## Data and reproducibility

Download the daily and monthly US factor files from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and prepare the monthly recession indicator from the [NBER chronology](https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions). The required filenames, recession convention, and input schema are documented in [data/README.md](data/README.md). Place these inputs in `data/raw/` without modifying the French source files. Raw data, processed panels, execution logs, and the local reference collection are excluded from the public repository. Recession labels are retrospective context, not trading inputs.

From the repository root, create the environment and register its Jupyter kernel:

```bash
conda env create -f environment.yml
conda activate anaconda-finance
python -m ipykernel install --user --name anaconda-finance --display-name "Python (anaconda-finance)"
```

For an existing environment, use `conda env update -f environment.yml` instead of the create command. Build the processed panels before executing the notebooks:

```bash
python src/01_build_managed_factors.py
python src/02_build_factor_momentum_strategies.py
jupyter lab
```

The first stage writes the original and volatility-managed factor panel; the second constructs factor-momentum portfolios and their portfolio-level timing variants. Both panels are written to `data/processed/`. Select the **Python (anaconda-finance)** kernel and run the notebooks in numerical order, executing cells from top to bottom. Expanding GARCH estimation can make the build computationally demanding. Run the automated checks with `python -m pytest -q`.

Reusable research code lives in `src/`, retained generated tables in `output/tables/`, and build diagnostics in `output/logs/`. [docs/methodology.md](docs/methodology.md) records definitions and implementation choices; [docs/decision_log.md](docs/decision_log.md) records research decisions.

## Interpretation and limitations

- **Normalization affects implementation claims.** The baseline scale is estimated through April 2015, and several comparisons equalize volatility ex post. Positive constant scaling leaves Sharpe ratios and alpha t-statistics unchanged, but return magnitudes and exposures are not real-time volatility targets. The volatility-forecast comparison retains different native scales, so magnitude differences also reflect normalization. The timing-payoff t-statistic is conditional on its estimated comparison scale.
- **Results exclude trading frictions and constraints.** Weights are uncapped, transaction costs are absent, and published factor returns do not account for real-time data availability. Subperiod comparisons and in-sample mean–variance portfolios are descriptive rather than out-of-sample investment tests; the many pairwise spanning tests are exploratory.
- **Forecasting and mechanism evidence remain qualified.** GARCH persistence is unrestricted and its forecast horizon uses the historical target month's trading-day count. Momentum data contain only the net factor, so separate winner- and loser-leg contributions and proposed holdings-level explanations are not tested.

## Reference and license

Moreira, A., and T. Muir (2017). “Volatility-Managed Portfolios.” *The Journal of Finance*, 72(4), 1611–1644. [doi:10.1111/jofi.12513](https://doi.org/10.1111/jofi.12513).

Selected published point estimates are included to evaluate the replication; the paper and its table layouts are not redistributed. The [MIT License](LICENSE) covers original software and documentation. Third-party data, publications, and reference material remain subject to their providers' terms.
