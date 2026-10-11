# PGS242-128 — Four-arm adaptive forecasting

[Project home](../README.md) · [Implementation details](PGS242-128_REVISION.md)

## Status

The current local implementation is the PR #29 detector-sensitivity version.
It has passed the focused automated test suite and the real PR #29 input
contract checks. Run `pr29_detector_sensitivity_20260920_retry1` completed all
34 configurations in 458.157 seconds and wrote the final local output set.

## Experiment design

The runner evaluates the same chronological SA1 and NSW1 test streams with:

- Arm A: never retrain;
- Arm B: retrain every 30 days;
- Arm C: drift-triggered retraining with full causal history;
- Arm D: drift-triggered retraining with 30, 60, 120, and 180-day causal windows.

A and B run once per region. C and every D window run separately for ADWIN,
Page-Hinkley, and KSWIN using the frozen PR #29 changepoints. This produces 17
configurations per region. Within each detector scenario, C and all D variants
must use exactly the same trigger sequence.

## Files

- [Experiment runner](../experiments/adaptation/run_four_arm_adaptation.py)
- [Adaptation component](../src/drift_lab/adaptation.py)
- [Forecasting component](../src/drift_lab/forecasters.py)
- [Automated tests](../tests/adaptation/test_four_arm_adaptation.py)
- [PR #29 changepoints](../results/tables/detection/aemo_detection_changepoints.csv)
- [Shared decisions](decisions.md)
- [Result summary](../results/tables/adaptation/revised/pr29_detector_sensitivity_20260920_retry1/summary.csv)
- [Detector comparison](../results/tables/adaptation/revised/pr29_detector_sensitivity_20260920_retry1/detector_comparison.csv)
- [D-window sensitivity](../results/tables/adaptation/revised/pr29_detector_sensitivity_20260920_retry1/window_sensitivity.csv)
- [Retraining events](../results/tables/adaptation/revised/pr29_detector_sensitivity_20260920_retry1/retraining_events.csv)
- [Run manifest](../results/tables/adaptation/revised/pr29_detector_sensitivity_20260920_retry1/manifest.json)

## Results

| Region | A MAE | B MAE / retrains | C-ADWIN MAE / retrains | C-Page-Hinkley MAE / retrains | C-KSWIN MAE / retrains |
| --- | ---: | ---: | ---: | ---: | ---: |
| SA1 | 147.293 | 136.401 / 46 | 136.684 / 16 | 137.425 / 9 | **136.381 / 28** |
| NSW1 | 372.569 | 342.286 / 46 | 342.630 / 22 | 343.719 / 10 | **341.527 / 30** |

Every Arm C configuration improved on Arm A. KSWIN achieved the lowest MAE in
both regions and slightly outperformed Arm B with fewer retrains. ADWIN was
within 0.303 MW of KSWIN in SA1 and 1.104 MW in NSW1 while using 12 and 8 fewer
retraining events, respectively. Therefore, C-KSWIN is the strict MAE winner,
while C-ADWIN is the recommended accuracy-cost compromise.

All tested Arm D recent windows performed worse than A, B, and C. Increasing
the window generally improved D, but even the best D configuration,
Page-Hinkley with 180 days, had MAE 179.574 MW in SA1 and 486.790 MW in NSW1.
The result supports full-history drift-triggered retraining for the current
Gradient Boosting model and does not support selecting any tested D window.

## Review boundary

PGS242-128 consumes upstream detections. It does not rerun detectors, tune
parameters, apply a refractory filter, or select alarms using T1/T2 event
matches. Calibration observations may warm the upstream detector, but they
remain excluded from forecasting-model training and retraining.

The shared decisions document still records the exact detector warm-up period
as pending team confirmation. If the upstream team changes that decision, it
must regenerate the changepoint CSV before the final 128 run.

## Validation

```sh
PYTHONPATH=src python3 -m pytest -q tests/adaptation
```

The suite verifies causal forecast/refit ordering, calibration exclusion,
timestamp-aware lag behavior, training-window boundaries, shared initial model
state, PR #29 provenance, identical C/D schedules, output identity, logging,
and overwrite protection.

## Reproduction

The command-line entry point calls `run_multi_seed_adaptation_experiment()`.
For Python callers, the runner exposes these functional entry points:

| Function | Responsibility |
| --- | --- |
| `validate_experiment_decisions()` | Check the shared detection and adaptation decisions before running. |
| `validate_frozen_detector_changepoints()` | Check detector provenance and timestamp alignment for a region/detector. |
| `run_regional_four_arm_comparison()` | Compare A/B/C/D for one region and forecasting seed. |
| `run_four_arm_adaptation_experiment()` | Run both regions for one forecasting seed and save the results. |
| `run_multi_seed_adaptation_experiment()` | Run the requested seeds and save their combined summary and manifest. |

```sh
python3 experiments/adaptation/run_four_arm_adaptation.py \
  --detections results/tables/detection/aemo_detection_changepoints.csv \
  --decisions docs/decisions.md \
  --detectors adwin page_hinkley kswin \
  --run-id four_arm_adaptation_NEW_RUN_ID \
  --seeds 1 2 3 4 5
```

Each run writes a new directory under `results/tables/adaptation/revised/` and
refuses to overwrite an existing run ID. Per-seed forecasts are written to
`<run_id>/seed_<seed>/forecasts_SA1.csv.gz` and `forecasts_NSW1.csv.gz`.
The parent directory contains `summary_all_seeds.csv` and
`multi_seed_manifest.json`.
