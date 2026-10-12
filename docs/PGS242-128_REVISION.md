# PGS242-128 — Current implementation details

[Navigation](PGS242-128_README.md)

## Upstream contract

The experiment consumes the frozen PR #29 changepoint table and validates each
SA1/NSW1 × ADWIN/Page-Hinkley/KSWIN sequence before writing output. Required
provenance includes:

- train-only deseasonalisation;
- train-only forecast-error Z-score parameters;
- 2,880 processed Calibration warm-up observations;
- a 14-day, 672-sample refractory period;
- one detector configuration per region/detector;
- changepoint indices that map exactly to chronological test timestamps.

The runner does not reproduce or modify the upstream detection pipeline.

`validate_experiment_decisions()` checks the shared decisions document, and
`validate_frozen_detector_changepoints()` checks each frozen detector sequence.
The execution chain is `run_multi_seed_adaptation_experiment()` →
`run_four_arm_adaptation_experiment()` → `run_regional_four_arm_comparison()`.
These names describe the functions' responsibilities; PR #29 remains the
historical source of the detection protocol.

## Forecasting invariants

- Every configuration starts from a copy of one fitted initial model state per
  region and forecasting seed.
- A retraining event at time `t` affects only forecasts after `t`.
- Arm C uses original training data plus causally observed test rows.
- Arm D uses only the configured recent causal window.
- Calibration is prediction context only and is excluded from every model fit.
- Timestamp-based lag lookup prevents lag values from crossing excluded gaps.
- A/B are detector-independent and execute once per region.
- For one detector, C and every D window use the identical trigger sequence.

## Configuration matrix

| Scope | Configurations |
| --- | --- |
| Detector-independent | A, B |
| ADWIN | C, D-30, D-60, D-120, D-180 |
| Page-Hinkley | C, D-30, D-60, D-120, D-180 |
| KSWIN | C, D-30, D-60, D-120, D-180 |

This gives 17 configurations per region and 34 summary rows across SA1 and
NSW1.

## Outputs

Each successful run directory contains:

- `forecasts_SA1.csv.gz` and `forecasts_NSW1.csv.gz` with timestamp-level
  `y_true`, `y_pred`, error, arm, detector, window, and retraining linkage;
- `retraining_events.csv` with trigger time, first affected forecast, training
  window, raw/effective samples, and fitting cost;
- `summary.csv` with full-test MAE and adaptation-cost fields;
- `window_sensitivity.csv` for all Arm D configurations;
- `detector_comparison.csv` for all C/D configurations;
- `runs.csv`, logged through the shared logger under
  `adaptive_forecasting_v3_pr29`;
- `manifest.json` with hashes, configuration, software versions, upstream
  metadata, and completion status.

`ADAPTATION_RUN_GROUP` and `ADAPTATION_SPLIT_ID` retain the existing persisted
values `adaptive_forecasting_v3_pr29` and
`aemo_train2018_2019_cal202001_202002_test202003_pr29`. Historical result folders,
CSV fields, and manifest values remain valid when the Python identifiers change.

The D-window comparison is sensitivity analysis on the test stream. The runner
does not automatically label the lowest-MAE window as an independently
validated optimum.

## Current verification state

The focused tests and read-only checks against the real PR #29 files pass. Run
`pr29_detector_sensitivity_20260920_retry1` completed with 34 summary rows,
667 retraining events, aligned SA1/NSW1 forecast streams, and a complete
manifest. Runtime was 458.157 seconds.

## Result interpretation

- Arm C reduced MAE against Arm A by 6.699–7.408% in SA1 and
  7.744–8.332% in NSW1.
- C-KSWIN achieved the minimum MAE in both regions: 136.381 MW in SA1 and
  341.527 MW in NSW1.
- C-ADWIN was only 0.303 MW and 1.104 MW above C-KSWIN while requiring fewer
  retrains, making it the preferred accuracy-cost compromise.
- C-Page-Hinkley used the fewest retrains but had the highest C-arm MAE.
- Arm B was competitive in MAE but required 46 retrains per region.
- Every 30/60/120/180-day Arm D configuration underperformed Arm A and Arm C.
  Longer windows generally reduced the damage, but no tested recent window is
  recommended for the current Gradient Boosting configuration.

The MAE differences between C-KSWIN and C-ADWIN are small and no forecasting
uncertainty interval was estimated. Report KSWIN as the strict MAE winner and
ADWIN as the operational recommendation rather than claiming a statistically
significant detector advantage. The interpretation remains conditional on the
PR #29 Calibration warm-up protocol, whose final team confirmation is still
recorded as pending in `docs/decisions.md`.
