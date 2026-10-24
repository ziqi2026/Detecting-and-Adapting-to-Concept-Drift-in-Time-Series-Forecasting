# Forecaster selection — NSW1

**Historical validation (October–December 2019), not the calibration window.**

Fit January 2018–September 2019; project calibration Jan–Feb 2020 and test from March 2020 are excluded. All forecasters use the existing `drift_lab` loader and registered architectures.

**Predeclared selection rule:** lowest mean validation MAE (MW) among candidates meeting the optional mean fit+predict runtime budget; runtime breaks ties. 7-day rolling MAE is a supplementary F1-aligned metric, not the selection metric.

Runtime budget: not specified seconds.

| Model | Validation MAE (MW) | 7-day rolling MAE (MW) | Fit (s) | Predict (s) | Total (s) | Seeds | Pick |
|---|---|---|---|---|---|---|---|
| Seasonal Naive | 449.70 | 457.59 | 0.00 | 0.00 | 0.00 | 1/1 |  |
| SARIMAX | 92.52 | 90.98 | 16.19 | 0.02 | 16.21 | 1/1 | ✓ |
| Gradient Boosting | 324.22 ± 3.77 | 326.98 ± 3.63 | 0.20 ± 0.10 | 0.01 ± 0.00 | 0.21 ± 0.10 | 5/5 |  |
| PatchTST | 117.20 ± 11.42 | 116.54 ± 11.91 | 18.10 ± 1.86 | 0.13 ± 0.01 | 18.23 ± 1.87 | 5/5 |  |

**Selected: SARIMAX.**

Numbers are mean ± sample SD for five-seed models; deterministic models show one run. 7-day rolling MAE averages complete 336-observation rolling windows, and therefore does not equal overall MAE. Stochastic seeds are 1–5, deterministic seed 0.

Model choice does not by itself demonstrate degradation or guarantee that downstream adaptation uses that model. Region-specific choices must be agreed across teams for T7.

Raw data: `validation_runs.csv`; aggregate: `validation_summary.csv`; protocol: `selection.json`.

Potential caveat: results are causal one-step-ahead forecasts conditional on observed past values; they are not 48-step blind forecasts. Compare with Team 41 only if horizons match.
