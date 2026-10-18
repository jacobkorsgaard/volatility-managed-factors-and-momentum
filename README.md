# Volatility-Managed Factors and Momentum

Factor risk varies over time, but higher volatility need not be accompanied by proportionally higher expected returns. Reducing exposure after volatile months and increasing it after calm months may therefore improve risk-adjusted performance. This repository examines that proposition for US equity factors, with particular attention to momentum.

The analysis replicates Moreira and Muir (2017), compares alternative volatility estimates, and studies whether volatility timing adds to factor momentum. It distinguishes conventional cross-sectional equity momentum from time-series momentum across factors, then examines which volatility states account for the gains from managing conventional momentum.

The baseline scales monthly factor returns inversely with the preceding month's realized variance. Extensions use expanding AR(1) and GARCH(1,1) forecasts and apply timing before or after factor-momentum construction. Factor momentum takes equally weighted long or short positions based on each available factor's average return over the preceding 12 months, without a skip month.

## Main findings

- Momentum provides the clearest evidence for volatility management across the three volatility estimates, although the gains weaken in the later sample. Over August 1927–July 2026, lagged-variance timing raises its annualized Sharpe ratio from 0.45 to 0.94 and reduces maximum drawdown from approximately 78% to 32% on the stored return scales.
- Volatility timing also improves factor momentum. Annualized Sharpe ratios are 0.49 for original factor momentum, 0.70 when timing the aggregate portfolio, and 0.77 when constructing momentum from individually timed factors. Timing order matters, while a second timing layer adds no statistically significant incremental alpha at 5% in the broader comparison.
- Momentum's average timing gain comes mainly from increasing exposure in calm periods. Reducing exposure after volatile months contributes more clearly to downside protection. The payoff is historically uneven and much flatter after the late 1990s.

These are historical gross-return comparisons with uncapped weights and no transaction costs. Several comparisons use ex-post volatility normalization. Detailed conventions and qualifications are recorded in [docs/methodology.md](docs/methodology.md).

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

The analysis uses daily and monthly US factors from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and retrospective recession dates from the [NBER chronology](https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions). Raw data and processed panels are excluded from the repository. Follow [data/README.md](data/README.md) for the required filenames, recession-indicator schema, and preparation steps. Place the inputs in `data/raw/` before running the analysis.

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

The two build stages write factor and factor-momentum panels to `data/processed/`. Select the **Python (anaconda-finance)** kernel and execute the notebooks in numerical order. Expanding GARCH estimation can be computationally demanding. Run automated checks with `python -m pytest -q`.

Reusable code is in `src/`, retained tables in `output/tables/`, and build diagnostics in `output/logs/`. Research decisions are documented in [docs/decision_log.md](docs/decision_log.md).

## Reference

Moreira, A., and T. Muir (2017). “Volatility-Managed Portfolios.” *The Journal of Finance*, 72(4), 1611–1644. [doi:10.1111/jofi.12513](https://doi.org/10.1111/jofi.12513).

Third-party data and publications remain subject to their providers' terms.
