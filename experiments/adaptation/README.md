# Model adaptation experiments

This directory runs the four-arm forecasting comparison (PGS242-128) and generates the F3, T3 and T4 evaluation outputs. Run commands from the repository root. The shared forecaster is the `gradient_boosted` entry in `MODEL_REGISTRY` (a `HistGradientBoostingRegressor`-based forecaster); the detector is an upstream input, not a different forecasting model.

## Design and inputs

The experiment uses half-hourly AEMO `TOTALDEMAND` (MW) for SA1 and NSW1. TRAIN (2018–2019) fits the initial forecaster; CALIBRATION (January–February 2020) belongs to upstream detector warm-up and is excluded from forecaster fitting; TEST starts in March 2020. Each forecast is made before the observation at that timestamp can contribute to a later refit.

| Arm | Update rule | Data available to a refit |
| --- | --- | --- |
| A | No retraining | Original TRAIN fit |
| B | Every 30 days | Full history available at the update |
| C | Accepted detector alarm | Full history available at the update |
| D | Same accepted alarm as C | Most recent 30, 60, 120 or 180 days |

D's day count specifies **history length**, not update frequency. A and B are shared controls, run once per region and forecasting seed. For each of ADWIN, Page-Hinkley and KSWIN, C and all four D windows use the same frozen alarm sequence. Thus each region/seed has 17 configurations: A, B, three C and twelve D. The standard evaluation uses seeds 1–5. Configuration IDs include `A`, `B`, `C_page_hinkley` and `D_180d_page_hinkley` (with analogous IDs for other detectors/windows).

The runner requires `data/raw/SA1/` and `data/raw/NSW1/`, `docs/decisions.md`, and the accepted upstream artifact `results/tables/detection/aemo_detection_changepoints.csv`. It validates detector provenance, warm-up and refractory metadata before running; it does not tune or recompute alarms or filter them using event matches. See [the implementation details](../../docs/PGS242-128_REVISION.md) and [shared decisions](../../docs/decisions.md).

## Run the experiment

Install the repository dependencies described in the [project README](../../README.md), then run:

```sh
python3 experiments/adaptation/run_four_arm_adaptation.py \
  --detections results/tables/detection/aemo_detection_changepoints.csv \
  --decisions docs/decisions.md \
  --run-id four_arm_NEW_RUN_ID \
  --seeds 1 2 3 4 5
```

The defaults include all three detectors and D windows 30/60/120/180. `--detectors`, `--windows`, `--data-path` and `--output-root` allow explicit subsets or alternative paths. Use a new `--run-id`: the runner refuses to overwrite a prior run.

The default output is `results/tables/adaptation/revised/<run-id>/`:

| File | Content |
| --- | --- |
| `multi_seed_manifest.json`, `summary_all_seeds.csv` | Seed list/status and combined per-configuration summaries |
| `seed_<n>/manifest.json` | Input/source hashes, settings, versions, status and output hashes |
| `seed_<n>/forecasts_SA1.csv.gz`, `forecasts_NSW1.csv.gz` | Timestamp-level actual demand and predictions for every configuration |
| `seed_<n>/summary.csv` | Full-test MAE, retrain counts, cumulative samples and timing |
| `seed_<n>/retraining_events.csv` | Refit timestamps, training boundaries and event costs |
| `seed_<n>/window_sensitivity.csv`, `detector_comparison.csv` | D-only and C/D subsets of the summary |
| `seed_<n>/runs.csv` | Per-seed metric log |

The runner also appends metrics to `results/tables/runs.csv`. The per-seed `overall_mae` is for the **whole TEST stream**; the regime-specific drift MAE used by T3/T4 is calculated separately. `train_samples_total` counts rows processed across fits, including rows reused in later refits. Wall-clock timings depend on the machine.

## Generate F3, T3 and T4

Set `RUN_DIR` to the five-seed output directory above. F3 draws seven-day (336 half-hour observation) rolling-MAE curves and retraining timestamps for A/B/C/D. Each plot keeps C and D on the **same detector**:

```sh
python3 experiments/adaptation/generate_f3_four_arm_comparison.py \
  --input-dir "$RUN_DIR" \
  --window-days 180 \
  --output-dir results/figures/adaptation/F3_four_arms
```

The F3 generator produces both regions and all three detector scenarios. For T3, each seed also needs `seed_<n>/uncertainty/aemo_prediction_intervals_90_summary.csv` with a `conformal_radius` for each region. Those seed-specific calibration radii supply T3's interval pinball-loss column; the adaptation runner itself does not create them.

```sh
python3 experiments/adaptation/generate_t3_forecast_accuracy.py \
  --run-dir "$RUN_DIR" \
  --events data/events.csv \
  --output-dir results/tables/adaptation/t3
```

T3 writes full-stream and event-local tables, seed metrics, regime-assignment audits, counts and `t3_manifest.json`. Full-stream assigns each TEST timestamp to pre-drift, drift or post-drift using the documented events and seven-day buffers. Event-local includes seven days before an event, its documented duration and seven days after; observations outside those windows are excluded. Both use mean ± sample SD across five seeds. The headline T3 table uses A, B, `C_kswin` and `D_180d_page_hinkley`; `t3_seed_metrics_full.csv` retains all configurations. A detector-filtered A/B/C/D comparison should choose the matching C and D IDs from that seed-level file.

T4 joins the **full-stream** T3 drift-regime seed metrics with the runner's cost summaries:

```sh
python3 experiments/adaptation/generate_t4_adaptation_cost.py \
  --run-dir "$RUN_DIR" \
  --t3-file results/tables/adaptation/t3/t3_seed_metrics_full.csv \
  --output-dir results/tables/adaptation/t4
```

T4 writes `t4_seed_level_metrics.csv`, `t4_all_configurations.csv` and the four-row-per-region headline `t4_adaptation_cost.csv`. Its signed `MAE gain vs arm A` is **strategy drift MAE minus Arm A drift MAE**, paired by region and seed: negative means lower error. `gain per retrain` divides that seed's signed difference by its retrain count before aggregation; Arm A has no such ratio. T4 reports retrain counts, cumulative training samples and retraining wall time alongside drift MAE differences. Its static headline uses the same mixed-detector C/D choices as headline T3; use the all-configuration/seed-level files for a single-detector comparison.

The checked-in [T3 results](../../results/tables/adaptation/t3/t3_full.md), [T4 results](../../results/tables/adaptation/t4/t4_adaptation_cost.csv) and [F3 figures](../../results/figures/adaptation/F3_four_arms/) are the existing published outputs. Re-running generators into their default folders can replace files; choose another output directory when preserving those artifacts.

## Validation

```sh
PYTHONPATH=src python3 -m pytest -q tests/adaptation
```

The tests cover forecast-before-refit ordering, training boundaries, calibration exclusion, shared initial state and C/D schedules, input provenance and output identity. Check each new run's manifests for `status: complete` and keep its input hashes with reported numbers.
