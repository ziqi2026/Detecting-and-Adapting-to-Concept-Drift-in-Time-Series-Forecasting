# Detecting-and-Adapting-to-Concept-Drift-in-Time-Series-Forecasting
An adaptive forecasting system for detecting concept drift, updating models, and improving prediction reliability in changing environments.

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
