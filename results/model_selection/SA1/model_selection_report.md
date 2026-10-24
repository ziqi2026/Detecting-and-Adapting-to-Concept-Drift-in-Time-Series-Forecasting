# Forecaster selection — SA1

**Historical validation (October–December 2019), not the calibration window.**

Fit January 2018–September 2019; project calibration Jan–Feb 2020 and test from March 2020 are excluded. All forecasters use the existing `drift_lab` loader and registered architectures.

**Predeclared selection rule:** lowest mean validation MAE (MW) among candidates meeting the optional mean fit+predict runtime budget; runtime breaks ties. 7-day rolling MAE is a supplementary F1-aligned metric, not the selection metric.

Runtime budget: not specified seconds.

| Model | Validation MAE (MW) | 7-day rolling MAE (MW) | Fit (s) | Predict (s) | Total (s) | Seeds | Pick |
|---|---|---|---|---|---|---|---|
| Seasonal Naive | 179.57 | 173.58 | 0.00 | 0.00 | 0.00 | 1/1 |  |
| SARIMAX | 38.92 | 38.89 | 26.25 | 0.02 | 26.28 | 1/1 |  |
| Gradient Boosting | 174.47 ± 1.92 | 167.22 ± 1.81 | 0.15 ± 0.01 | 0.01 ± 0.00 | 0.16 ± 0.01 | 5/5 |  |
| PatchTST | 38.52 ± 3.10 | 38.30 ± 3.13 | 34.67 ± 6.00 | 0.14 ± 0.04 | 34.81 ± 6.00 | 5/5 | ✓ |

**Selected: PatchTST.**

Numbers are mean ± sample SD for five-seed models; deterministic models show one run. 7-day rolling MAE averages complete 336-observation rolling windows, and therefore does not equal overall MAE. Stochastic seeds are 1–5, deterministic seed 0.

Model choice does not by itself demonstrate degradation or guarantee that downstream adaptation uses that model. Region-specific choices must be agreed across teams for T7.

Raw data: `validation_runs.csv`; aggregate: `validation_summary.csv`; protocol: `selection.json`.

Potential caveat: results are causal one-step-ahead forecasts conditional on observed past values; they are not 48-step blind forecasts. Compare with Team 41 only if horizons match.
