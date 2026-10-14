# Detecting-and-Adapting-to-Concept-Drift-in-Time-Series-Forecasting
An adaptive forecasting system for detecting concept drift, updating models, and improving prediction reliability in changing environments.

## Project navigation

| Work area | Main entry point | Outputs and supporting files |
| --- | --- | --- |
| AEMO data preparation | [Preparation notebook](notebooks/01_AEMO_Data_Preparation.ipynb) | [Shared experimental decisions](docs/decisions.md) |
| Event catalogue | [Event catalogue](data/events.csv) | [Tiering criteria](docs/event_tiering_criteria.md) |
| Classical forecasting baseline | [SARIMAX notebook](notebooks/03.2_Classical_Baselines_SARIMAX.ipynb) | [Leakage tests](tests/degradation/test_sarimax_leakage.py) |
| F1 — Baseline degradation | [Baseline degradation notebook](notebooks/03.3_F1_Baseline_Degradation.ipynb) | [Rolling MAE](results/tables/degradation/F1_rolling_mae_baseline.csv) |
| T1 — Synthetic detection | [Table generator](experiments/detection/generate_t1_detection_table.py) | [Synthetic detection table](results/tables/detection/T1_detection_on_synthetic.csv) |
| T2 — AEMO event matching | [Detection table](results/tables/detection/T2_aemo_detection.md) | [Matching protocol](docs/aemo_matching_protocol.md) |
| PGS242-128 — Four-arm adaptation | [Adaptation README](docs/PGS242-128_README.md) | [Implementation and results](docs/PGS242-128_REVISION.md) |
| T3 — Forecast accuracy by regime | [Table generator](experiments/adaptation/generate_t3_forecast_accuracy.py) | [Stored table](results/tables/adaptation/T3_forecast_accuracy_by_regime.md) |
| T4 — Adaptation cost | [Cost-summary generator](experiments/adaptation/generate_t4_cost_summary.py) | [Stored cost summary](results/tables/adaptation/T4_adaptation_cost_summary.csv) |
| F3 — Four-arm comparison | [Figure generator](experiments/adaptation/generate_f3_four_arm_comparison.py) | [Comparison figures](results/figures/adaptation/F3_four_arms/) |
| T5 — Coverage and interval width | [T5 methodology](docs/T5_EVALUATION.md) | [Simulation README and tables](results/tables/uncertainty/t5_simulation/README.md) |

The table links to stored artifacts and their implementations; it does not
declare every historical result to be a newly validated final result. Follow
each task's documentation and manifest for its experiment version and status.

## PGS242-128 — Four-arm adaptive forecasting on SA1 and NSW1

**Completed local run:** the runner consumes the PR #29
provenance-complete changepoint artifact as three frozen detector scenarios.
It runs A/B once per region and C plus D-30/D-60/D-120/D-180 for each of
ADWIN, Page-Hinkley and KSWIN. It validates the recorded train-only
preprocessing, Calibration warm-up metadata and 14-day refractory setting;
it never reruns, filters or retunes a detector.

- [128 navigation and run instructions](docs/PGS242-128_README.md)
- [Current implementation details](docs/PGS242-128_REVISION.md)
- [Chronological experiment runner](experiments/adaptation/run_four_arm_adaptation.py)
- [Automated tests](tests/adaptation/test_four_arm_adaptation.py)
- [34-configuration result summary](results/tables/adaptation/revised/pr29_detector_sensitivity_20260920_retry1/summary.csv)
- [Detector comparison](results/tables/adaptation/revised/pr29_detector_sensitivity_20260920_retry1/detector_comparison.csv)
- [D-window sensitivity](results/tables/adaptation/revised/pr29_detector_sensitivity_20260920_retry1/window_sensitivity.csv)
- [Completed run manifest](results/tables/adaptation/revised/pr29_detector_sensitivity_20260920_retry1/manifest.json)

The completed run contains 34 configurations and finished in 458.157 seconds.
Arm C improved on Arm A in both regions for every detector. KSWIN produced the
lowest MAE (SA1 136.381 MW; NSW1 341.527 MW), while ADWIN achieved very similar
accuracy with fewer retrains and is the preferred accuracy-cost compromise.
Every tested Arm D recent window underperformed Arm A and Arm C. These findings
remain conditional on the PR #29 Calibration warm-up protocol recorded in
`docs/decisions.md`.

## T2 — Detection on AEMO vs documented events

The current T2 evaluates the stored Task 120 SA1/NSW1 detections using T1's
seven-day, inclusive, first-unused one-to-one matching rule. All five frozen
Tier 1 events are represented, and remaining detections are checked against
Tier 2 before being reported as unmatched. Historical events are contextual
references, not ground-truth changepoints.

- [Current T2 table](results/tables/detection/T2_aemo_detection.md)
- [Current T2 CSV](results/tables/detection/T2_aemo_detection.csv)
- [Matching protocol](docs/aemo_matching_protocol.md)
- [Input and implementation manifest](results/tables/detection/T2_manifest.json)

From the repository root:

```sh
python3 experiments/detection/run_aemo_t2_evaluation.py
python3 experiments/detection/run_aemo_t2_evaluation.py --from-logs
```

Metrics are recorded through `run_logger.py` in `results/runs.csv`.
The current table selects the manifest's configuration hashes and split ID
`aemo_fixed_train2018_2019_test202003_task120_t2_t1rule_v2`.
Earlier results in `results/tables/detection/archive/aemo_context_v1/` are
superseded historical records and must not be pooled with the current T2.

## T5 — Coverage and interval width by regime

T5 compares static and adaptive conformal intervals using coverage, mean
width, normalised width and worst rolling 24-hour coverage. Metrics are
calculated separately by seed and regime, then aggregated across seeds 1–5
with mean and sample standard deviation (`ddof=1`). Each region/configuration
has its own exact six-column table.

**Current status:** implementation and synthetic demonstration are available.
The ten real interval streams passed input checks, but formal AEMO T5 results
are pending team agreement on normalisation and regime/window policy.
The linked simulation tables are not AEMO model-performance results.

- [T5 methodology, input audit and formal-run instructions](docs/T5_EVALUATION.md)
- [Simulation README, current conventions and reproduction](results/tables/uncertainty/t5_simulation/README.md)
- [SA1 simulation table](results/tables/uncertainty/t5_simulation/tables/T5_SA1_A.csv)
- [NSW1 simulation table](results/tables/uncertainty/t5_simulation/tables/T5_NSW1_A.csv)
- [Exact simulation protocol](results/tables/uncertainty/t5_simulation/protocol.json)
- [Separate synthetic metric log](results/tables/uncertainty/t5_simulation/runs.csv)
- [T5 table generator](experiments/uncertainty/generate_t5_uncertainty_table.py)
- [Shared evaluation functions](src/drift_lab/evaluation.py)
- [T5 metric and generator tests](tests/uncertainty/test_t5_evaluation.py)

The current **changeable simulation conventions** are:

| Setting | Current simulation convention |
| --- | --- |
| Normalised width denominator | Fixed synthetic scale **10** per region, shared across methods, seeds and regimes. This is not TRAIN range/SD and must not be used as a formal AEMO denominator. |
| Regime scope | `full`, using the shared T3 fixed-AEST assignment and unioned seven-day buffered half-open event windows. |
| Cross-regime rolling windows | `within_regime`: all 48 half-hour observations in `(t-24h,t]` must belong to the same regime. Cross-boundary and incomplete windows are excluded; disjoint segments are not joined. |
| No eligible 24h window | Report N/A instead of substituting a shorter window. |

These settings may be revised after team feedback. A formal run must supply
regional scale values with provenance and the agreed regime/window policy in
an explicit protocol. Regenerate affected tables under a distinct protocol
identity when conventions change. Synthetic logs remain separate from the
canonical `results/runs.csv`; no model retraining is needed to change the
evaluation conventions.

## Baselines and earlier evaluation work

The following entry points provide direct navigation to earlier preparation,
forecasting and evaluation work alongside the T2 and T5 sections above.

- [AEMO preparation notebook](notebooks/01_AEMO_Data_Preparation.ipynb)
- [Event catalogue](data/events.csv) and [tiering criteria](docs/event_tiering_criteria.md)
- [Seasonal-naive baseline notebook](notebooks/03.1_Seasonal_Naive_Baseline.ipynb)
- [SARIMAX baseline notebook](notebooks/03.2_Classical_Baselines_SARIMAX.ipynb)
- [SARIMAX leakage tests](tests/degradation/test_sarimax_leakage.py)
- [F1 baseline-degradation notebook](notebooks/03.3_F1_Baseline_Degradation.ipynb)
- [F1 evaluation runner](experiments/degradation/run_baseline_degradation_evaluation.py)
- [F1 rolling-MAE data](results/tables/degradation/F1_rolling_mae_baseline.csv)
- [F1 SA1 figure](results/figures/degradation/F1_SA1_baseline_degradation.png)
- [F1 NSW1 figure](results/figures/degradation/F1_NSW1_baseline_degradation.png)
- [T3 stored accuracy table](results/tables/adaptation/T3_forecast_accuracy_by_regime.md),
  [generator](experiments/adaptation/generate_t3_forecast_accuracy.py) and
  [tests](tests/test_t3_evaluation.py)
- [T4 stored cost summary](results/tables/adaptation/T4_adaptation_cost_summary.csv),
  [generator](experiments/adaptation/generate_t4_cost_summary.py) and
  [cost-versus-accuracy figure](results/figures/adaptation/T4_cost_vs_accuracy.png)
- [F3 four-arm figures](results/figures/adaptation/F3_four_arms/) and
  [generator](experiments/adaptation/generate_f3_four_arm_comparison.py)
