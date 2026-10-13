# Volatility-managed factor portfolios

This repository studies volatility-managed factor portfolios, beginning with a replication of Moreira and Muir (2017) and extending the analysis to alternative volatility forecasts and factor momentum.

## Scope

The active analysis:

1. validates the available daily and monthly factor data;
2. replicates the available parts of Figures 1–3 and Tables I–III in Moreira and Muir (2017);
3. compares lagged realized variance, expanding AR(1), and monthly GARCH(1,1) timing;
4. compares alternative factor-momentum constructions; and
5. studies why lagged-realized-variance timing is particularly strong for momentum.

The completed scope covers factor portfolios and factor momentum. Liquidity portfolios are outside this scope.

## Main findings

- Momentum is the clearest case for volatility management in the volatility-forecast comparison: its own-factor spanning alpha is significant under lagged realized variance, AR(1), and GARCH timing and across the reported subperiods. The benefit is not universal across factors, and momentum's alpha magnitude declines in the later sample.
- For lagged-RV-managed momentum, the average equal-volatility timing payoff is 0.663% per month, with a Newey–West(12) t-statistic of 4.026 conditional on the ex-post scale. The lowest lagged-volatility quintile contributes 0.603 percentage points of that average gain. Scaling up calm momentum months drives most of the mean improvement; reducing exposure after volatile months contributes more clearly to lower downside risk.
- The timing payoff is historically uneven and much flatter after the late 1990s. The strategy's near-zero unconditional market beta also masks changing exposure across expansion and recession states.

These findings describe historical gross returns under the construction and comparison conventions below. The notebooks provide the full replication, factor-momentum comparisons, and supporting diagnostics; [`docs/methodology.md`](docs/methodology.md) records the definitions and implementation choices.

## Repository structure

- `data/raw/`: immutable French factor files and the NBER recession indicator.
- `data/processed/`: reproducible analysis-ready factor and strategy panels.
- `src/`: two numbered build stages plus reusable research modules.
- `notebooks/`: the ordered replication and comparative analysis.
- `output/tables/`: generated tables retained outside the notebooks.
- `output/logs/`: build diagnostics.
- `docs/methodology.md`: definitions, timing conventions, and possible research extensions.
- `docs/decision_log.md`: chronological research decisions and material findings.
- `references/`: read-only papers and project source material.

Third-party raw data, derived data panels, execution logs, and the local `references/` collection are excluded from the public repository. See [`data/README.md`](data/README.md) for the required source files and reconstruction steps.

## Data and benchmark attribution

Factor returns come from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Recession dates come from the [NBER US Business Cycle Expansions and Contractions](https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions) chronology.

The replication benchmarks are selected point estimates transcribed from:

> Moreira, A., and T. Muir (2017), “Volatility-Managed Portfolios,” *The Journal of Finance* 72(4), 1611–1644. [https://doi.org/10.1111/jofi.12513](https://doi.org/10.1111/jofi.12513)

The repository reproduces only the selected numerical benchmarks needed to evaluate the replication; it does not redistribute the paper or reproduce its table layouts.

The MIT license covers the repository's original software and documentation only. It does not grant rights to third-party data, publications, or reference material.

## Environment

Create the Conda environment and register its Jupyter kernel:

```bash
conda env create -f environment.yml
conda activate anaconda-finance
python -m ipykernel install --user --name anaconda-finance --display-name "Python (anaconda-finance)"
```

If `anaconda-finance` already exists, update it instead:

```bash
conda env update -f environment.yml
```

All notebooks are configured to use the `anaconda-finance` kernel.

## Rebuild the processed data

Run the two stages from the repository root:

```bash
python src/01_build_managed_factors.py
python src/02_build_factor_momentum_strategies.py
```

Stage 1 reads the immutable raw factor files and constructs the original, lagged-RV-, AR(1)-, and GARCH-managed factor panel. Stage 2 constructs factor-momentum portfolios and their portfolio-level volatility-managed variants.

## Notebook order

1. `01_summary_statistics.ipynb` — sources, dimensions, coverage, and structural validation.
2. `02_moreira_muir_replication.ipynb` — the available Moreira–Muir replication.
3. `03_volatility_horse_race.ipynb` — comparison of volatility definitions.
4. `04_factor_momentum_horse_race.ipynb` — factor-momentum architecture selection.
5. `05_selected_strategy_comparison.ipynb` — performance, risk, spanning, and subperiods.
6. `06_momentum_volatility_timing.ipynb` — momentum-specific timing diagnostics.

The notebooks contain embedded outputs so results can be inspected on GitHub without rerunning them. To reproduce the results, rebuild the processed panels first and then run the notebooks in numerical order.

## Important interpretation limits

- Monthly regressions use Newey–West HAC inference with 12 lags as the primary project convention. Paper-specific inference is reported separately where relevant.
- Several comparisons use positive ex-post equal-volatility constants. These do not affect Sharpe ratios or alpha t-statistics, but they are not implementable real-time volatility targets.
- Weights are uncapped and results exclude transaction costs.
- Subperiod results are descriptive rather than out-of-sample tests.
- GARCH persistence is not constrained below one.
- The available momentum data contain the net factor, not separate winner and loser legs.

## Tests

Run:

```bash
python -m pytest -q
```
