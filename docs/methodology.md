# Methodology

## Settled conventions

- Raw source files remain unchanged in `data/raw/`.
- Monthly regressions use Newey-West HAC standard errors with 12 lags unless a paper-specific replication convention is stated separately.
- The baseline Moreira-Muir managed return uses the preceding month's realized variance, so the weight is known before the managed return month.
- Following equation (2) of Moreira and Muir (2017), realized variance is the sum of squared daily factor returns after subtracting the within-month average daily return. The code's demeaning is therefore intentional and paper-consistent.
- Expanding AR(1) forecasts use 120 monthly observations initially and are refitted each month using only information available before the forecast month.

## Reproducible pipeline

`src/01_build_managed_factors.py` is the upstream factor stage: it loads validated raw sources and produces unmanaged, lagged-RV, AR(1), and GARCH factor returns in `data/processed/managed_factors.parquet`. `src/02_build_factor_momentum_strategies.py` is the downstream strategy stage: it constructs all factor-momentum parents from the managed-factor panel, applies second-stage timing, and writes `data/processed/factor_momentum_strategies.parquet`. Shared source loading lives in `src/data.py`; common HAC, return-statistic, and notebook-formatting helpers live in `src/utils.py`; reusable factor-momentum construction and evaluation live in `src/factor_momentum.py` and `src/factor_momentum_analysis.py`. The reproducible environment is recorded in `environment.yml`, and the working execution guide is in `README.md`.

The three retained generated CSVs have descriptive rather than notebook-order names: `moreira_muir_table_i_spanning.csv`, `moreira_muir_table_ii_mve.csv`, and `factor_momentum_build_diagnostics.csv`. Notebook-only display tables are regenerated in place and are not duplicated as separate files.

### Inference implementation

All active monthly regressions use a shared HAC implementation with 12 lags and asymptotic normal reference statistics (`use_t=False`). Existing horse-race, spanning, selected-strategy, and momentum-timing regressions explicitly retain Statsmodels' finite-sample HAC covariance correction. The original Table II replication and factor-momentum build diagnostics historically did not apply that correction; the cleanup preserves those outputs by setting `use_correction=False` explicitly. This distinction slightly affects standard errors and t-statistics, not OLS point estimates. Whether to standardize the two legacy outputs to the corrected project convention remains unresolved and will not be changed silently. The paper's heteroskedasticity-only inference remains a separate replication benchmark.

The active notebook layer contains `01_summary_statistics.ipynb`, which documents dimensions and coverage; `02_moreira_muir_replication.ipynb`, which consolidates Figures 1–3 and Tables I–III; `03_volatility_horse_race.ipynb`, which contains the volatility-timing horse race; `04_factor_momentum_horse_race.ipynb`, which compares factor-momentum architectures and their matched 1/N benchmarks for strategy selection; `05_selected_strategy_comparison.ipynb`, which studies the selected momentum strategies; and `06_momentum_volatility_timing.ipynb`, which diagnoses how lagged-RV timing changes momentum. Later extension notebooks will be added only when their analysis is developed.

## Exploratory volatility comparison

The volatility-comparison notebook uses the full processed-data vintage and one global July 1973-July 2026 sample in which the original return and all three managed returns are available for every factor. Stored monthly variances are converted to monthly volatility percentages by taking the square root and multiplying by 100. The correlation heatmap was removed; only the three volatility time series are displayed.

The performance horse race includes only the three volatility-managed returns and reports monthly mean, monthly standard deviation, annualized Sharpe ratio, and monthly CAPM, FF3, and FF5 alphas in separate factor panels. The notebook displays Newey-West(12) t-statistics, rather than standard errors, beside means, alphas, and reported regression loadings. One global date sample is imposed: every original and managed return for every factor must be observed, and that identical sample is used for volatility summaries, performance, and both spanning exercises. The original return is retained only as the benchmark in the original-factor spanning test. Return series retain their stored construction scales; the notebook does not impose ex-post equal-volatility rescaling.

Original-factor spanning regresses each managed return on its corresponding unmanaged factor. Managed-factor spanning is pairwise and directional, using the three pre-specified comparisons: lagged RV on AR(1), lagged RV on GARCH(1,1), and AR(1) on GARCH(1,1). The final robustness section divides the global sample mechanically into three nearly equal observation counts and repeats only the original-factor spanning regression. The breakpoints are not selected from results. These remain exploratory comparisons and do not select a volatility model.

The notebook's interpretation is deliberately qualified. The three volatility definitions usually produce the same qualitative factor ranking and mostly span one another. Momentum is the robust exception to original-factor spanning, with significant own-factor alpha under every method and in every subperiod, although its alpha magnitude declines after November 2008. RMW is significant over the full sample but not after November 2008. This supports a stronger pre-/post-GFC pattern without claiming that volatility timing works universally before the GFC. Loading differences are not treated as pure timing effects because native normalization differs across methods.

## Monthly GARCH variance forecasts

Monthly GARCH timing is implemented for Mkt, SMB, HML, Mom, RMW, and CMA. For each factor, a constant-mean Gaussian GARCH(1,1) is estimated on its daily returns:

`h[t] = omega + alpha * epsilon[t-1]^2 + beta * h[t-1]`.

Each factor has its own initial 120-calendar-month estimation and scaling period because factor availability differs. Thereafter, parameters are re-estimated at each month-end on an expanding sample containing only daily returns observed before the target month.

- The month-end forecast sums the expected daily conditional variances over the trading days in the target month.
- One inverse-forecast-variance weight is fixed before the target month and held for that month. It multiplies the official monthly French factor return.
- The scaling constant is fixed from the factor's initial 120-month period and matches the volatility of fitted managed monthly returns to the corresponding unmanaged monthly returns during calibration.
- No external data or additional package is used; estimation uses SciPy's optimizer.
- These GARCH-managed returns remain an exploratory forecasting extension. They do not replace the lagged-realized-variance replication or the expanding AR(1) forecasts.

## Unresolved decisions

- Whether monthly GARCH or the expanding AR(1) forecast should enter the later strategy.
- Whether the Gaussian innovation assumption should be compared with an alternative distribution before formal use.
- Whether GARCH should use the historical target month's realized number of trading days or a fixed ex-ante horizon such as 22 trading days.
- Whether the Table II replication and factor-momentum build diagnostics should adopt the same finite-sample HAC covariance correction as the rest of the project. Current results preserve their prior uncorrected-HAC implementation explicitly.
- Whether lagged-RV momentum should receive a separate constrained-weight robustness analysis. Moreira and Muir's Table IV considers maximum weights of 1.0 and 1.5; neither cap is part of the current uncapped baseline and no cap will be selected silently.
- The fixed 30/40/30 liquidity split is reserved for the later liquidity stage and has not been changed or applied here.
- Liquidity timing is outside the active analysis plan. No liquidity portfolio has been constructed; the previously specified 30/40/30 split remains unaltered in case that research stage is revisited.

## Implementation choices requiring awareness

- Baseline realized variance is currently the within-month sum of squared demeaned daily factor returns. Months require at least two daily observations.
- Baseline managed returns use the preceding month's realized variance. Their scale constant is estimated over the sample ending April 2015, so only the constant scale—not the time-varying signal—uses the full replication sample.
- AR(1) is estimated on log monthly realized variance with a constant, an expanding 120-month initial window, and monthly refits. Forecasts apply the lognormal correction `exp(forecast_log + residual_variance / 2)`; the uncorrected forecast is also retained.
- GARCH is a constant-mean Gaussian quasi-maximum-likelihood model implemented with SciPy rather than the external `arch` package. Conditional variance starts at the expanding sample variance. `omega`, `alpha`, and `beta` are constrained only to be nonnegative; `alpha + beta` is unrestricted.
- GARCH parameters are re-estimated monthly on expanding daily data. The next month's forecast sums expected daily variances over the number of trading observations present in that historical target month, then applies one weight to the official monthly return.
- AR(1) and GARCH scale constants use only their initial 120-month calibration samples, whereas the baseline replication normalization uses its full analysis sample. The current horse race deliberately displays native construction scales, so differences in mean, alpha magnitude, and original-factor loadings combine normalization and timing effects. A common-sample equal-volatility robustness comparison is still required before interpreting those magnitude differences as pure timing effects.
- No strategy currently includes weight caps, leverage constraints, transaction costs, or turnover penalties.
- The processed panel retains the full available data vintage. Existing replication notebooks apply the inferred April 2015 endpoint from the paper's observation counts.
- In-sample mean-variance results use unconstrained tangency portfolios, a covariance pseudo-inverse, and unit-gross displayed weights. They are diagnostics, not implementable portfolio rules.
- The lagged-RV normalization constant and equal-volatility comparison scales use the full analysis sample. They do not affect Sharpe ratios or alpha t-statistics, but they are ex-post scale choices and should be replaced by an initial-window or expanding target-volatility rule before presenting implementable return magnitudes and exposures.
- GARCH currently uses the historical target month's realized number of trading observations to set its finite forecast horizon. This does not use target-month returns, but a fixed ex-ante horizon such as 22 trading days would remove any calendar-information concern.

## Factor momentum

Factor momentum follows the baseline time-series specification described by Ehsani and Linnainmaa. For each factor in month `t`, the signal is the sign of the arithmetic average of its monthly returns over `t-12` through `t-1`. There is no skip month and returns are not compounded to construct the signal. Positive signals are long, negative signals are short, and an exact zero produces a zero position.

A factor enters only after 12 complete past monthly returns and when its current return is available. Positions are equal-weighted across factors with valid signals for that month and are rebalanced monthly. The signal calculation is applied separately to four sets: original factor returns, lagged-realized-variance-managed returns, AR(1)-managed returns, and GARCH-managed returns. Each portfolio therefore uses only the factors and histories available within its own return set.

### Factor-momentum horse race

The selection horse race uses one July 1947-July 2026 sample shared by all 16 original, single-timed, and double-timed factor-momentum returns. FF5 and FF5 + Mom regressions use the July 1963-July 2026 control-availability sample. All return strategies are rescaled ex post to the volatility of original factor momentum over the common sample. This scaling makes gross means and cumulative performance comparable but is descriptive rather than an implementable volatility target; alpha and mean t-statistics and Sharpe ratios are invariant to the positive constant scaling.

Four constructions are distinguished throughout:

- Factor momentum: original factors, then factor momentum.
- Volatility-timed factor momentum: factor momentum, then portfolio-level volatility timing.
- Factor momentum from volatility-timed factors: factor-level volatility timing, then factor momentum.
- Volatility-timed factor momentum from volatility-timed factors: factor-level timing, factor momentum, then portfolio-level timing.

Momentum spanning is reported both against standalone Mom and against FF5 + Mom. Pairwise spanning among the seven original and single-layer strategies is directional and contains 42 exploratory regressions; the displayed alpha-t-statistic matrix uses the row strategy as dependent and the column strategy as spanning return. Pairwise R-squared remains in the supporting table but is not graphed. The nine double-timed strategies are tested against their corresponding factor-momentum parent so that the intercept measures the incremental contribution of the second timing layer.

Notebook 04 also compares each of the four factor-momentum parents with a long-only 1/N portfolio formed from exactly the same original, lagged-RV-timed, AR(1)-timed, or GARCH-timed factor inputs. These tests isolate the signed factor-momentum rule relative to passive equal weighting within an input family. Notebook 05 does not repeat the 1/N spanning analysis.

### Selected momentum-strategy comparison

Notebook 05 compares conventional Mom, lagged-RV volatility-timed Mom, original factor momentum, lagged-RV volatility-timed factor momentum, and factor momentum from lagged-RV-timed factors over their common August 1927-July 2026 sample. Native-scale performance contains the monthly arithmetic mean, median, standard deviation, the mean's compounded annual equivalent, and the annualized Sharpe ratio; factor-model comparisons report monthly CAPM, FF3, and FF5 alphas with Newey-West(12) t-statistics. The compounded annual equivalent is `(1 + monthly arithmetic mean)^12 - 1`; it is not a realized geometric CAGR.

Risk management is reported separately on each strategy's native stored return scale. It contains the 5th, 25th, 75th, and 95th return percentiles, skewness, excess kurtosis, maximum drawdown, empirical 5% Value at Risk (VaR), and empirical 5% Expected Shortfall (ES). Monthly standard deviation and median are classified as performance statistics and are therefore not duplicated in the risk table. VaR is the negative of the 5th-percentile monthly return and ES is the negative mean return among months at or below that threshold, so both are displayed as positive loss percentages. These are historical empirical tail measures without a parametric distributional assumption. Maximum drawdown is classified as path-dependent risk rather than performance.

Comparative monthly-return paths, cumulative wealth, drawdowns, factor-model alphas, and directional spanning rescale each return by one positive constant to conventional Mom's common-sample volatility. The cumulative-wealth and drawdown panels additionally show the market excess return on this same equal-volatility scale as a standard benchmark. This is an ex-post descriptive comparison scale, not an implementable volatility target. Rolling Sharpe-ratio and rolling-volatility figures are not retained in notebook 05.

For descriptive stability checks, the 1,188-month common sample is divided mechanically into three consecutive blocks of 396 observations: August 1927-July 1960, August 1960-July 1993, and August 1993-July 2026. The subperiod section begins with one full-history monthly-return figure marking these blocks. Each subperiod then receives a separate cumulative-wealth and drawdown figure; wealth is reset to one at the period's beginning. Subperiod figures retain the full-sample equal-volatility comparison constants, so resetting cumulative wealth does not re-estimate strategy normalization.

Performance and downside-risk statistics are repeated using native stored returns, with separate performance and risk-management tables for each period. No subperiod heatmap is retained. The subperiod definitions do not depend on outcomes, but this is not an out-of-sample design: strategy constructions and normalization constants are not re-estimated separately within each block.

Full-sample directional spanning is presented before the subperiod analysis. It is accompanied by a Pearson correlation heatmap for the five selected monthly return series. Correlation is treated only as descriptive comovement and not as evidence that one strategy spans another.

### Momentum volatility-timing diagnostics

Notebook 06 uses the same August 1927-July 2026 sample as notebook 05 and only the existing processed columns for original momentum, lagged-RV-managed momentum, lagged momentum realized variance, the associated weight, and the NBER recession indicator. It does not alter the strategy or write another processed dataset. The identity `managed_mom = weight_mom * mom` is validated exactly, and the lagged variance and weight must be positive. Month `t` exposure is therefore fixed from daily momentum returns observed during month `t-1`; month `t` momentum is not used in forming its own weight.

Native-scale exposure and return-distribution summaries retain the stored April 2015 normalization constant. Weights are uncapped and unwinsorized. Reported scalar-weight turnover is the annualized average absolute monthly change in the multiplier; it is not transaction-cost-adjusted security-level turnover. Empirical VaR, ES, return quantiles, skewness, excess kurtosis, and maximum drawdown use the same definitions as notebook 05. For each strategy, its Gaussian kernel-density estimate is overlaid directly on its density-normalized histogram in a separate panel with common axes; the following figure reports empirical cumulative distribution functions. These figures are descriptive and do not impose a parametric return model.

For state diagnostics, the full sample is sorted ex post into five equal-count groups based on lagged momentum realized variance. Quintile 1 contains the lowest 20% and quintile 5 the highest 20%. These groups are descriptive bins rather than real-time estimated thresholds. Each bin reports its known lagged variance and weight together with the associated month `t` original and managed return mean, standard deviation, and within-bin annualized Sharpe ratio. The timing-payoff diagnostic additionally rescales managed momentum by one positive full-sample constant to original momentum's standard deviation, then defines the monthly payoff as equal-volatility managed momentum minus original momentum. Its displayed cumulative path is the arithmetic sum of monthly differences, not compounded wealth. The payoff's mean t-statistic uses Newey-West HAC inference with 12 lags, conditional on the chosen scale and without adjustment for estimating that scale. This second ex-post scaling isolates timing from unconditional leverage but is not an implementable target-volatility rule.

NBER states are used only as retrospective labels. Recession performance contains return statistics but not average weights. Because all volatility-management weights are positive, managed and original momentum have identical return signs and negative-month frequencies; timing changes the magnitude, not the direction, of each monthly return. Comovement and the formal Moreira-Muir recession interaction are retained as subsections under expansion and recession performance. The comovement subsection reports native-return correlation, the OLS loading of managed momentum on original Mom, and R-squared for the full sample, expansions, and recessions. This loading is an own-Mom exposure measure rather than a CAPM beta; unlike a formal spanning test, the descriptive table does not test its intercept. A trailing 120-month figure reports backward-looking correlation and Mom loading using the current and preceding 119 observations.

The recession-loading analysis is applied separately to Market and Mom with Newey-West(12) inference. For each regressor, an ordinary full-period regression first reports the unconditional loading. The full-sample interaction specification then regresses managed momentum on the regressor, an NBER recession dummy, and their interaction. Its dummy is the recession intercept shift, its base loading is the expansion loading, its interaction is the change during recessions, and their sum is the total recession loading. Showing the unconditional loading separately prevents the base coefficient from the interacted regression from being mislabeled as an unconditional beta. The lagged-RV rule reacts to preceding-month volatility; recession shading and interaction terms do not imply advance recession prediction.

A winner-versus-loser momentum-leg decomposition is outside the active analysis. The available source files contain the net Mom factor but not its underlying long and short portfolio returns, so their separate contributions cannot be recovered from the net series. No leg series will be inferred or downloaded without a later data decision.

The main descriptive result is that the average equal-volatility payoff is concentrated in the lowest lagged-volatility quintile: it contributes 0.603 percentage points to the 0.663% monthly full-sample mean, while the highest-volatility quintile contributes 0.015 percentage points. The mean's Newey-West(12) t-statistic is 4.026 conditional on the ex-post scale. Quintile 1 has the highest absolute Sharpe ratio for both returns; quintile 5 has the largest Sharpe-ratio improvement relative to raw Mom but remains a low-Sharpe state. Raw Mom's realized mean declines from 0.895% in quintile 1 to 0.002% in quintile 5 as monthly standard deviation rises from 1.882% to 8.180%. Thus the average gain is primarily associated with levering calm, subsequently profitable momentum months; reducing exposure after volatile months contributes more clearly to improved left-tail risk and drawdown.

Managed and original momentum correlation is 0.532 in expansions and 0.369 in recessions. The expansion Mom loading is 0.661, the recession interaction is -0.470 with a Newey-West(12) t-statistic of -4.470, and the resulting recession loading is 0.191. Trailing 120-month loadings are commonly near or above one from the 1950s through the 1990s and generally much lower after 2000, demonstrating that the relationship is not stable. The additive payoff is historically uneven and much flatter after the late 1990s. These are gross in-sample diagnostics under ex-post normalization, uncapped weights, and no transaction costs, not an out-of-sample or implementability claim.

The unconditional full-period market beta of managed momentum is -0.013 with a Newey-West(12) t-statistic of -0.523 and an R-squared of 0.0002. In the interacted specification, the expansion market beta is 0.028, the recession change is -0.109 (t = -2.316), and the total recession beta is -0.081. Thus the strategy has almost no unconditional linear market exposure but becomes mildly counter-market in recessions; this is not evidence that the strategy is riskless.

### Second-stage volatility timing

Each of the four factor-momentum portfolios is additionally evaluated with its own lagged-RV, AR(1), and GARCH monthly volatility multiplier. Daily portfolio proxies use factor-momentum positions and any first-stage factor-scaling weights fixed before the month begins, applied to raw daily factor returns within that month. These daily proxies estimate the aggregate portfolio's variance; the resulting second-stage weight multiplies the stored monthly factor-momentum return.

The second-stage comparison contains 12 timed strategies plus their four unscaled parents. Results use the common date range shared by all 16 portfolios through April 2015 and rescale every return to the original factor-momentum portfolio's volatility. Regressions report incremental alpha relative to the corresponding unscaled parent and CAPM, FF3, FF5, and FF5 + Mom alphas using Newey-West(12).

Performance and alpha columns use this equal-volatility comparison scale. Exposure and turnover columns retain each strategy's native construction scale so they describe the leverage and trading actually generated by its stored normalization constants.

Factor-momentum signals themselves are lagged correctly: month `t` uses only returns through `t-1`, and the daily proxy holds those positions during month `t`. The source factor panels have no internal missing observations after each factor's inception, so current-month missing-value filtering does not alter any historical position. Real-time trading would additionally require either contemporaneously reconstructing factor returns or allowing for the publication delay of vendor-provided monthly factor files.
