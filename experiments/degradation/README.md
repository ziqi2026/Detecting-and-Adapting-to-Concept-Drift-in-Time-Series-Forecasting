# Data Preparation and Forecast Degradation Experiments

[Project README](../../README.md) · [Shared decisions](../../docs/decisions.md)

## Purpose and scope

This directory evaluates how frozen forecasting models perform over the AEMO SA1 and NSW1 demand streams. It produces the timestamp-level results and seven-day rolling MAE used for F1, together with event-specific MAE records.

Data inspection and preparation are documented in the [preparation notebook](../../notebooks/01_AEMO_Data_Preparation.ipynb). Reusable loading and splitting are implemented in the [shared data loader](../../src/drift_lab/data_loader.py).

This workflow does not detect drift, retrain models during testing, or evaluate prediction intervals. Detector preprocessing and adaptive forecasting are separate stages.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| [run_baseline_degradation_evaluation.py](run_baseline_degradation_evaluation.py) | Fit registered forecasters, generate frozen test predictions, calculate rolling and event-window MAE, and log event metrics |
| [generate_f1_degradation_figure.py](generate_f1_degradation_figure.py) | Generate regional F1 figures from saved rolling-MAE results without fitting models |
| [patchtst_synthetic_degradation.py](patchtst_synthetic_degradation.py) | Run the separate controlled PatchTST degradation diagnostic |

Shared components:

- [Data loader](../../src/drift_lab/data_loader.py)
- [Forecasting models and registry](../../src/drift_lab/forecasters.py)
- [Evaluation metrics](../../src/drift_lab/evaluation.py)
- [Experiment logger](../../src/drift_lab/run_logger.py)

## Data preparation

### Inputs

Place the monthly regional CSV files under:

```text
data/raw/SA1/*.csv
data/raw/NSW1/*.csv
```

The loader combines files, parses `SETTLEMENTDATE`, sorts observations chronologically, and requires `SETTLEMENTDATE` and `TOTALDEMAND`. Demand is measured in MW. The experiment uses fixed-AEST market timestamps, not a conversion to local daylight-saving time.

### Frequency standardisation

The input series changes from 30-minute to five-minute observations at `2021-10-01 00:05:00`.

- Retain observations before this transition unchanged.
- Average the subsequent five-minute demand observations into right-closed, right-labelled 30-minute intervals.
- Require six non-missing observations per aggregated interval; otherwise the loader raises an error.

For example, observations from 00:05 through 00:30 are averaged and labelled 00:30. The result represents average demand, not a sum of MW values.

The preparation notebook reports no missing values, duplicate timestamps, or unexpected raw-data gaps in the inspected inputs. No imputation or duplicate removal was needed. These are findings for those inputs, not a general cleaning guarantee for replacement datasets. The loader checks group counts but does not independently establish that every group contains six distinct, correctly spaced timestamps.

### Chronological split

| Partition | Timestamp boundaries | Observations per region in the inspected dataset |
| --- | --- | ---: |
| TRAIN | `2018-01-01 <= t < 2020-01-01` | 35,039 |
| CALIBRATION | `2020-01-01 <= t < 2020-03-01` | 2,880 |
| TEST | `t >= 2020-03-01` | 67,249 |

The notebook records 105,168 standardised observations per region, spanning `2018-01-01 00:30` through `2024-01-01 00:00`. Counts and the test endpoint depend on the available source files; the loader imposes no fixed test end date.

The notebook exports `data/processed/SA1.csv` and `data/processed/NSW1.csv`. However, the degradation runner loads the raw monthly files through `load_and_split_aemo()` and performs standardisation in memory. Those processed exports are not prerequisites for running F1.

The notebook contains historical `src.data_loader` imports. For the packaged workflow, use `drift_lab.data_loader` as in the current runner.

## Forecasting protocol

For each region, the runner constructs all models available through `make_forecasters()` and `MODEL_REGISTRY`, fits them on TRAIN only, and evaluates them chronologically on TEST. CALIBRATION is supplied as historical prediction context, not as additional fitting data.

Frozen means that fitted parameters are not re-estimated during testing. It does not mean that prediction inputs remain fixed: past observed demand can enter later forecasts through each model's causal forecasting mechanism. This is a sequential forecasting experiment, not a forecast of the entire test period made at its start.

| Model | Role |
| --- | --- |
| Seasonal Naive | Required reference baseline using seasonal historical demand |
| SARIMAX | Required classical baseline with frozen fitted parameters |
| Gradient Boosting | Required machine-learning baseline; construction and lag settings come from the shared registry |
| PatchTST | Optional Transformer-based comparison, included when PyTorch is available |

The supplied AEMO runner records `SEED = 0` and is not a five-seed evaluation. Exact model settings should be read from the registry used at execution. The logger's experiment seed should not be assumed to override every model's own configuration.

### PatchTST brief

PatchTST is an additional patch-based Transformer forecaster, not a replacement for the three required baselines. It processes recent observed demand as historical context while keeping learned parameters frozen during test evaluation. Consequently, stable error after a change can reflect useful new context rather than model retraining.

PyTorch availability controls registration in the shared forecasting module. Without it, the AEMO experiment runs the required baselines only. Once predictions have been saved, the F1 figure generator can plot PatchTST results without PyTorch.

## Evaluation

### MAE and rolling error

MAE is the mean absolute difference between observed and forecast demand:

```text
MAE = mean(abs(y_true - y_pred))
```

F1 uses a trailing window of 336 observations: 48 half-hours per day multiplied by seven days. The first 335 rolling values are undefined because a full window is unavailable. These values should not be filled with zeros.

The runner prints overall test MAE and saves forecasts and rolling MAE. Its shared log contains event-specific MAE, not a separately logged overall-test MAE row.

### Event-local windows

For an event with start date `s` and inclusive final date `e`, the runner uses:

| Regime | Interval |
| --- | --- |
| pre-drift | `[s - 7 days, s)` |
| drift | `[s, e + 1 day)` |
| post-drift | `[e + 1 day, e + 1 day + 7 days)` |

Windows are left-inclusive and right-exclusive. An empty window returns `NaN` and is skipped by the logging helper. A partially available window is evaluated on its available test observations; it is not rejected for incomplete exposure. For example, a pre-event window before the test start may be unavailable.

Events are evaluated separately, so overlapping event windows may reuse observations. Do not sum these event records as though they partition the test stream. This differs from later mutually exclusive full-stream T3 regime evaluation.

### Event-version qualification

The supplied runner and figure generator each contain a hard-coded `EVENTS` dictionary; neither reads `data/events.csv`.

Their COVID period is March–April 2020, and both retain Five-Minute Settlement as a contextual marker. In contrast, the shared decisions specify COVID as 16 March–17 May 2020 and remove Five-Minute Settlement from the demand-drift target catalogue.

These are known implementation differences, not equivalent event definitions. Reproducing this script reproduces its embedded event list. For an aligned final analysis, reconcile both dictionaries with the approved catalogue, preserve the previous outputs, assign a new experiment identity, and regenerate the event metrics and figures. Changing event windows alone does not require refitting the saved forecasts, but the current runner combines forecasting and event evaluation.

Historical events provide context; their alignment with error changes does not prove concept drift or causation.

## Installation and execution

Run commands from the repository root. Follow the [project README](../../README.md) for the supported environment and dependencies.

```bash
python -m pip install -e ".[dev]"
```

For PatchTST, install the optional dependencies:

```bash
python -m pip install -e ".[patchtst]"
```

Generate predictions and metrics:

```bash
python experiments/degradation/run_baseline_degradation_evaluation.py
```

Then generate F1 from the saved results:

```bash
python experiments/degradation/generate_f1_degradation_figure.py
```

The supplied scripts use module-level paths and settings rather than an argument parser. They do not expose CLI switches for regions, models, dates, or output directories. The runner evaluates both regions and all registered models.

Review the output locations before rerunning: the source CSV and figure files are written to fixed paths and can replace earlier files. Shared-log duplicate behaviour depends on `run_logger.py`; do not assume reruns are automatically deduplicated.

## AEMO outputs

| Output | Contents |
| --- | --- |
| [F1_rolling_mae.csv](../../results/tables/degradation/F1_rolling_mae.csv) | Timestamp-level forecasts, errors, and rolling MAE for each evaluated region/model |
| [runs.csv](../../results/tables/runs.csv) | Event-regime MAE and experiment metadata |
| [SA1 F1 figure](../../results/figures/degradation/F1_SA1_forecasting_degradation.png) | Regional rolling-error trajectories and contextual markers |
| [NSW1 F1 figure](../../results/figures/degradation/F1_NSW1_forecasting_degradation.png) | Regional rolling-error trajectories and contextual markers |

The source CSV columns are:

```text
SETTLEMENTDATE, region, method, y_true, y_pred, error, rolling_mae
```

`error = y_true - y_pred`. PatchTST rows appear only if it was registered during execution.

Logged metadata includes model configuration, raw training-row count, experiment seed, event-specific split identifier, elapsed time, and `n_retrains = 0`. The runner uses group `degradation`, dataset `AEMO_TOTALDEMAND`, and base split identifier:

```text
aemo_sa1_nsw1_fixed_split_2018_2023_frozen_baselines_v2
```

Each event appends `|event=<event_id>` to that identifier. The elapsed time covers fitting, forecasting, and rolling-error computation, not fitting alone. The configuration identifier hashes registry settings; it is not a complete source-and-input provenance manifest.

The figure generator reads saved results, plots all available methods, and does not recompute MAE, fit models, or update the shared log. Use these script-defined output names rather than similarly named historical `*_baseline*` files.

## Separate PatchTST synthetic diagnostic

Run the optional diagnostic with:

```bash
python experiments/degradation/patchtst_synthetic_degradation.py
```

PyTorch and a registered `patchtst` model are required. The experiment generates 20,000 observations for each no-drift, sudden, gradual, and recurring scenario. Positional splits are:

- TRAIN: `[0, 4000)`
- CALIBRATION/history: `[4000, 5000)`
- TEST: `[5000, 20000)`

Synthetic data use seeds 1–5. The model is constructed from the registry without replacing its configured seed with the synthetic seed. Therefore, these five runs should not automatically be described as five different model-initialisation seeds.

For sudden and recurring changes, the comparison uses the week before the change, the first week after it, and the second week after it. Gradual drift uses the complete programmed transition and adjacent one-week windows. The no-drift control compares an early stable week with the final week.

Outputs:

- [Seed-level results](../../results/tables/degradation/patchtst_synthetic_degradation_results.csv)
- [Representative seed-1 figure](../../results/figures/degradation/PatchTST_synthetic_degradation_seed1.png)

The CSV records MAE, absolute and percentage degradation, sample counts, and elapsed time. Positive degradation means error increased. Undefined comparisons and percentage changes from a zero reference remain missing. The seed-1 figure is illustrative; quantitative results use all five synthetic seeds. This diagnostic does not write to the shared `runs.csv` and is separate from the T1 detector-calibration benchmark.

## Validation and reproducibility

The preparation notebook checks missingness, duplicate timestamps, cadence, regional timestamp agreement, and split separation. The loader enforces required columns and complete aggregation counts. The degradation runner prints forecast and timestamp diagnostics and raises an error unless each series has exactly 335 undefined rolling-MAE values. Printed diagnostics should not be mistaken for assertions covering every possible input defect.

Run the degradation test directory documented by the project:

```bash
python -m pytest -q tests/degradation
```

The tests were not executed as part of preparing this README. Before reporting a new run, retain:

- the repository commit and installed dependency versions;
- input-file identity and observed date ranges;
- the actual registered model list and model settings;
- the embedded event definitions and experiment identity;
- the saved forecasts, execution output, and event metrics.

No new standalone manifest is created by the supplied AEMO degradation runner. Archive fixed-path outputs before rerunning. Dependency availability can change the evaluated model set, while hardware and software versions can change runtime and numerical results.

## Interpretation boundaries

- Increasing rolling error indicates reduced forecasting accuracy, not independently verified concept drift.
- Causal historical inputs may help frozen models track demand changes without refitting.
- AEMO event windows are contextual and can overlap or be only partly covered.
- AEMO seed-0 results and five-seed synthetic diagnostics are different experiments.
- Unnormalised MW errors should be compared carefully across regions with different demand scales.
- This workflow evaluates historical forecasts, not a validated live operational service.
