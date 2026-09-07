# Detecting-and-Adapting-to-Concept-Drift-in-Time-Series-Forecasting
An adaptive forecasting system for detecting concept drift, updating models, and improving prediction reliability in changing environments.

## T2 — Detection on AEMO vs documented events

The current T2 evaluates the stored Task 120 SA1/NSW1 detections using T1's
seven-day, inclusive, first-unused one-to-one matching rule. All five frozen
Tier 1 events are represented, and remaining detections are checked against
Tier 2 before being reported as unmatched. Historical events are contextual
references, not ground-truth changepoints.

- [Current T2 table](results/tables/detection/T2_aemo_detection.md)
- [Current T2 CSV](results/tables/detection/T2_aemo_detection.csv)
- [Acceptance checklist and file locations](docs/T2_delivery_checklist.md)
- [Bilingual process and results / 中英过程说明](docs/T2_process_zh_en.md)
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
