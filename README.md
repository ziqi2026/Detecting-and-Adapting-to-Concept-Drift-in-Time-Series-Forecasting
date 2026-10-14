# Detecting-and-Adapting-to-Concept-Drift-in-Time-Series-Forecasting

An adaptive forecasting system for detecting concept drift, updating models,
and improving prediction reliability in changing environments.

## Project Navigation

| Work area | Main Entry Point | Outputs and Supporting Files |
| --- | --- | --- |
| AEMO data preparation | [Preparation notebook](notebooks/01_AEMO_Data_Preparation.ipynb) · [Shared decisions](docs/decisions.md) | [Processed SA1 data](data/processed/SA1.csv) · [Processed NSW1 data](data/processed/NSW1.csv) |
| Event catalogue | [Tiering criteria](docs/event_tiering_criteria.md) | [Event catalogue](data/events.csv) |
| Seasonal-naive baseline | [Baseline notebook](notebooks/03.1_Seasonal_Naive_Baseline.ipynb) | [F1 rolling-MAE data](results/tables/degradation/F1_rolling_mae_baseline.csv) |
| SARIMAX baseline | [Baseline notebook](notebooks/03.2_Classical_Baselines_SARIMAX.ipynb) · [Leakage tests](tests/degradation/test_sarimax_leakage.py) | [F1 rolling-MAE data](results/tables/degradation/F1_rolling_mae_baseline.csv) |
| F1 — Baseline degradation | [Analysis notebook](notebooks/03.3_F1_Baseline_Degradation.ipynb) · [Evaluation runner](experiments/degradation/run_baseline_degradation_evaluation.py) · [Figure generator](experiments/degradation/generate_f1_degradation_figure.py) | [Rolling-MAE data](results/tables/degradation/F1_rolling_mae_baseline.csv) · [SA1 figure](results/figures/degradation/F1_SA1_baseline_degradation.png) · [NSW1 figure](results/figures/degradation/F1_NSW1_baseline_degradation.png) |
| T1 — Synthetic detection | [Table generator](experiments/detection/generate_t1_detection_table.py) · [Tuning notebook](notebooks/04.1_Synthetic_Drift_Detector_Tuning_and_Evaluation.ipynb) | [Synthetic detection table](results/tables/detection/T1_detection_on_synthetic.csv) |
| T2 — AEMO detection vs documented events | [Matching protocol](docs/aemo_matching_protocol.md) · [Evaluation runner](experiments/detection/run_aemo_t2_evaluation.py) · [Tests](tests/detection/test_aemo_t2_evaluation.py) | [Table](results/tables/detection/T2_aemo_detection.md) · [CSV](results/tables/detection/T2_aemo_detection.csv) · [Matches](results/tables/detection/T2_detection_matches.csv) · [Summary](results/tables/detection/T2_detection_summary.csv) · [Manifest](results/tables/detection/T2_manifest.json) |
| PGS242-128 — Four-arm adaptation | [Adaptation README](docs/PGS242-128_README.md) · [Implementation details](docs/PGS242-128_REVISION.md) · [Experiment runner](experiments/adaptation/run_four_arm_adaptation.py) · [Tests](tests/adaptation/test_four_arm_adaptation.py) | [Five-seed summary](results/tables/adaptation/revised/pgs242_128_five_seed_final/summary_all_seeds.csv) · [Five-seed run artifacts](results/tables/adaptation/revised/pgs242_128_five_seed_final/) · [Historical single-seed summary](results/tables/adaptation/revised/pr29_detector_sensitivity_20260920_retry1/summary.csv) |
| T3 — Forecast accuracy by regime | [Table generator](experiments/adaptation/generate_t3_forecast_accuracy.py) · [Tests](tests/test_t3_evaluation.py) | [Stored regime table](results/tables/adaptation/T3_forecast_accuracy_by_regime.md) · [Five-seed full-test summary](results/tables/adaptation/revised/pgs242_128_five_seed_final/summary_all_seeds.csv) |
| T4 — Adaptation cost | [Cost-summary generator](experiments/adaptation/generate_t4_cost_summary.py) · [Adaptation-cost generator](experiments/adaptation/generate_t4_adaptation_cost.py) | [Stored cost summary](results/tables/adaptation/T4_adaptation_cost_summary.csv) · [Cost-versus-accuracy figure](results/figures/adaptation/T4_cost_vs_accuracy.png) |
| F3 — Four-arm comparison | [Figure generator](experiments/adaptation/generate_f3_four_arm_comparison.py) | [Comparison figures](results/figures/adaptation/F3_four_arms/) |
| T5 — Coverage and interval width by regime | [Methodology and data status](docs/T5_EVALUATION.md) · [Simulation README and changeable conventions](results/tables/uncertainty/t5_simulation/README.md) · [Table generator](experiments/uncertainty/generate_t5_uncertainty_table.py) · [Shared metrics](src/drift_lab/evaluation.py) · [Tests](tests/uncertainty/test_t5_evaluation.py) | [SA1 simulation table](results/tables/uncertainty/t5_simulation/tables/T5_SA1_A.csv) · [NSW1 simulation table](results/tables/uncertainty/t5_simulation/tables/T5_NSW1_A.csv) · [Simulation protocol](results/tables/uncertainty/t5_simulation/protocol.json) · [Separate simulation log](results/tables/uncertainty/t5_simulation/runs.csv) · [AEMO TRAIN means](results/tables/uncertainty/t5_training_mean_demand.json) |

T5 tables linked above contain synthetic demonstration data; formal AEMO T5
results are pending the agreed evaluation protocol. Stored tables and figures
may come from different experiment versions; follow their documentation and
manifests before comparing results.
