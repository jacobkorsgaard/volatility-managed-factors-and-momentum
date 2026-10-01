# Data sources

Third-party raw data and derived processed panels are deliberately excluded from the public repository. They are not covered by the repository's MIT software license.

## Required raw files

Download the CSV versions of the US research factors from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and place the following files, without editing them, in `data/raw/`:

- `F-F_Research_Data_Factors_daily.csv`
- `F-F_Research_Data_Factors.csv`
- `F-F_Research_Data_5_Factors_2x3_daily.csv`
- `F-F_Research_Data_5_Factors_2x3.csv`
- `F-F_Momentum_Factor_daily.csv`
- `F-F_Momentum_Factor.csv`

The repository uses both frequencies. Daily returns construct realized variance and GARCH forecasts; monthly files provide the official factor returns used by the strategies and regressions.

## NBER recession indicator

Create `data/raw/nber_recessions.csv` from the [NBER US Business Cycle Expansions and Contractions](https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions) chronology. The required schema is:

```text
date,recession
1926-07-01,0
1926-08-01,0
...
```

Use one row per calendar month. Set `recession` to one from the month after an NBER peak through the NBER trough month, inclusive, and zero otherwise. These are retrospective business-cycle classifications, not real-time strategy inputs.

## Rebuild derived data

From the repository root, run:

```bash
python src/01_build_managed_factors.py
python src/02_build_factor_momentum_strategies.py
```

This recreates:

- `data/processed/managed_factors.parquet`
- `data/processed/factor_momentum_strategies.parquet`

The notebooks retain embedded outputs for inspection on GitHub, but reproducing them requires the local source files and rebuilt processed panels.
