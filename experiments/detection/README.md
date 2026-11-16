# Drift Detection and Evaluation Pipeline

This directory documents Team 42's forecast-error-based drift detection workflow: synthetic calibration, evaluation of frozen detectors, application to AEMO demand, event matching, and F2 visualisation.

The detectors are **ADWIN, KSWIN, and Page-Hinkley**. Real-world experiments cover **SA1 and NSW1**. The main deliverables are **T1** (synthetic detection evaluation), **T2** (AEMO detection-event matching), and **F2** (regional detection panels).

For project installation, architecture, and component attribution, see the [main README](../../README.md).

## Contents

- [Workflow and execution boundaries](#workflow-and-execution-boundaries)
- [Entry points](#entry-points)
- [Data and preprocessing](#data-and-preprocessing)
- [Frozen detector configurations](#frozen-detector-configurations)
- [Running the pipeline](#running-the-pipeline)
- [T2 event-matching rules](#t2-event-matching-rules)
- [Outputs and interpretation](#outputs-and-interpretation)
- [Reproducibility and troubleshooting](#reproducibility-and-troubleshooting)
- [Related documentation](#related-documentation)

## Workflow and execution boundaries

1. Evaluate candidate detector configurations on controlled synthetic streams.
2. Evaluate the configurations stored in `DETECTOR_REGISTRY` across synthetic seeds 1–5 and generate T1 from logged metrics.
3. Apply the frozen configurations to AEMO forecast-error streams, saving accepted detections and diagnostic traces.
4. Evaluate saved AEMO detections against the documented event catalogue to produce T2 and matching audits.
5. Generate F2 from saved diagnostics, T2 matches, and event dates.

Tuning writes candidate results and prints selected configurations; it does **not** automatically update `DETECTOR_REGISTRY`. The registry is the configuration source for the detection runners.

T2 performs evaluation of stored detections; it does not refit a forecaster or rerun a detector. F2 plots saved evidence without retraining, rerunning detectors, or rematching events. Downstream adaptation experiments consume accepted alarm timestamps separately.

## Entry points

| Script | Purpose |
| --- | --- |
| [`tune_detectors_deseasonalized_zscore.py`](tune_detectors_deseasonalized_zscore.py) | Evaluate the synthetic tuning grid and identify configurations satisfying the control and false-alarm constraints |
| [`run_synthetic_drift_detection.py`](run_synthetic_drift_detection.py) | Evaluate registry configurations on synthetic streams and log seed-level metrics |
| [`generate_t1_detection_table.py`](generate_t1_detection_table.py) | Produce the T1 display table from the shared experiment log |
| [`run_aemo_drift_detection.py`](run_aemo_drift_detection.py) | Generate accepted AEMO alarms and detector diagnostics |
| [`run_aemo_t2_evaluation.py`](run_aemo_t2_evaluation.py) | Validate saved AEMO detections, match events, log metrics, and produce T2 |
| [`plot_f2_detection_panel.py`](plot_f2_detection_panel.py) | Validate saved inputs and generate combined regional F2 panels |

Shared implementations are in [`detectors.py`](../../src/drift_lab/detectors.py), [`forecasters.py`](../../src/drift_lab/forecasters.py), [`deseasonalise.py`](../../src/drift_lab/deseasonalise.py), [`detector_diagnostics.py`](../../src/drift_lab/detector_diagnostics.py), and [`evaluation.py`](../../src/drift_lab/evaluation.py).

## Data and preprocessing

### Synthetic benchmark

Synthetic demand has a 30-minute cadence, with 35,040 TRAIN observations, 2,880 CALIBRATION observations, and 17,520 TEST observations. Experiments use seeds 1–5 and four scenarios:

| Scenario | TEST change structure |
| --- | --- |
| None | No programmed changepoint |
| Sudden | One changepoint at 50% of TEST |
| Gradual | A 30-day transition centred at 50% of TEST |
| Recurring | Changepoints at 35% and 65% of TEST |

The tuning script evaluates 140 candidate configurations. Passing configurations must have zero TEST detections on the no-drift control across all five seeds and mean false alarms of at most two per year for each positive-drift scenario. Passing candidates are ranked by missed detections, worst scenario-specific mean false-alarm rate, overall mean false-alarm rate, and mean detection delay. Candidates that cannot satisfy the constraints can be rejected early.

The tuning budget is expressed **per year**; the T1 display table reports false alarms **per 10,000 observations**. These are different reporting scales.

### AEMO experiment

Regional source data are expected under `data/raw/SA1/` and `data/raw/NSW1/`, loaded through the shared data loader. TRAIN covers 2018–2019, CALIBRATION covers January–February 2020, and TEST starts on 1 March 2020. Market timestamps use fixed AEST, UTC+10, without daylight-saving shifts.

The stored T2 manifest records 35,039 TRAIN, 2,880 CALIBRATION, and 67,249 TEST observations per detector-region configuration. Its TEST endpoints are `2020-03-01 00:00:00+10:00` and `2024-01-01 00:00:00+10:00`. These are artifact-specific counts; the synthetic TRAIN length is different.

### Forecast-error signal

The runners fit a seasonal profile on TRAIN using an overall mean, smoothed day-of-year effects, and day-of-week × half-hour effects. A temporary Gradient Boosting forecaster fitted on TRAIN Year 1 produces chronological Year-2 errors for fitting the global absolute-error Z-score transformation. The final forecaster is fitted on complete deseasonalised TRAIN with calendar features and lags of 48 and 336 observations.

ADWIN and KSWIN additionally use partial AR(1) whitening with `lambda = 0.80`; Page-Hinkley uses no whitening. AR(1) parameters are fitted using TRAIN Year-2 processed errors. Detector inputs therefore represent preprocessed forecast errors, not raw electricity demand or rolling MAE.

Processed CALIBRATION errors warm the detector, with warm-up alarms ignored. The same detector state continues into TEST. A **14-day refractory period (672 observations)** suppresses alarms occurring too soon after the last accepted TEST alarm. The detector still receives every observation during suppression.

## Frozen detector configurations

The reviewed [`DETECTOR_REGISTRY`](../../src/drift_lab/detectors.py) contains:

| Detector | Explicit parameters |
| --- | --- |
| ADWIN | `delta=1e-7` |
| Page-Hinkley | `min_instances=30`, `delta=0.001`, `threshold=700`, `alpha=0.999`, `mode="both"` |
| KSWIN | `alpha=5e-8`, `window_size=480`, `stat_size=120`, `seed=12` |

Unspecified parameters use the installed River version's defaults. The AEMO forecaster seed is 0. KSWIN's internal seed 12 is separate from the synthetic experiment seeds 1–5. T2 evaluates the stored AEMO runs; it is not a five-seed experiment and must not be presented with invented multi-seed standard deviations.

## Running the pipeline

Run the commands below from the **repository root**, in the project Python environment. This is required by the relative paths in the T1 table generator. Follow the [installation instructions](../../README.md#installation-and-quick-start), including Git LFS retrieval where applicable. These scripts depend on the repository's shared modules and data; downloading the scripts alone is insufficient.

### 1. Optional synthetic tuning

This stage repeats configuration search and is unnecessary for simply inspecting or plotting saved results.

```bash
python experiments/detection/tune_detectors_deseasonalized_zscore.py
```

Inspect the tuning output before proposing a registry change. Do not silently replace frozen configurations when reproducing existing results.

### 2. Synthetic evaluation and T1

```bash
python experiments/detection/run_synthetic_drift_detection.py
python experiments/detection/generate_t1_detection_table.py
```

The first command evaluates the registry configurations and writes metrics to `results/tables/runs.csv`. The second filters the matching synthetic split and formats mean ± sample SD values for the display table.

### 3. AEMO detection and diagnostics

```bash
python experiments/detection/run_aemo_drift_detection.py
```

This requires regional raw data and fits the forecasting/preprocessing components. It writes the accepted changepoint file and compressed diagnostic trace. Preserve published outputs before rerunning, because the script writes to the standard result paths.

### 4. T2 evaluation of saved detections

```bash
python experiments/detection/run_aemo_t2_evaluation.py
```

The normal evaluation path requires the saved changepoints, raw AEMO data for input/index validation, event catalogue, protocol files, and shared source modules. It records source hashes and produces matching audits, metrics, a manifest, and display tables.

To rebuild only the T2 display table from existing logs and manifest:

```bash
python experiments/detection/run_aemo_t2_evaluation.py --from-logs
```

This mode requires `results/tables/runs.csv` and `results/tables/detection/T2_manifest.json`, together with an installed project environment. It does not reconstruct missing detection diagnostics or matching audits.

### 5. F2 visualisation

```bash
python experiments/detection/plot_f2_detection_panel.py
```

Required inputs are `aemo_detection_diagnostics.csv.gz`, `T2_detection_matches.csv`, and `data/events.csv`. If these compatible saved artifacts are already available, F2 can be regenerated without repeating tuning, forecasting, or detection.

## T2 event-matching rules

The protocol is [`aemo_t1_7day_one_to_one_v2`](../../docs/aemo_matching_protocol.json). It distinguishes the seven-day matching tolerance from the fourteen-day alarm refractory period.

- Day-precision events use the inclusive window `[start_date, start_date + 7 days]`.
- Date-range or month-precision events use `[start_date, end_date + 7 days]`.
- A detection before an event start cannot match that event.
- Each detector-region combination is evaluated independently.
- Applicable events are ordered by tier, start date, and event ID. All Tier 1 assignments precede Tier 2 assignments.
- Each event receives the first unused chronological detection in its window. Each event and detection can be assigned at most once.
- Remaining detections are unmatched after both tier passes.

Tier 1 delay is measured in fractional days from the event's start anchor. Catalogue dates are contextual references, so these offsets do not estimate the exact physical onset of drift. The table distinguishes `not detected` from `not applicable`.

Catalogue-match proportion equals `(Tier 1 matches + Tier 2 matches) / total detections` and is undefined for zero detections. Tier 1 unmatched counts include both subsequent Tier 2 matches and final unmatched detections. Chance-match evaluation supplies a temporal-overlap reference under the shared evaluation implementation; it does not establish causation.

Event-relative descriptive regimes and the mutually exclusive full-stream accounting regimes used for logging serve different purposes. Overlapping event-window counts must not be added to obtain global totals. See the [full matching protocol](../../docs/aemo_matching_protocol.md) for boundaries and accounting rules.

## Outputs and interpretation

All table filenames below are relative to [`results/tables/detection/`](../../results/tables/detection/).

| Output | Contents |
| --- | --- |
| `detector_tuning_results.csv` | Candidate configurations, constraint outcomes, and selection information |
| `synthetic_detection_raw.csv` | Synthetic run-level results |
| `synthetic_detection_summary.csv` | Aggregated synthetic results |
| `T1_detection_on_synthetic.csv` | T1 display table from logged synthetic metrics |
| `aemo_detection_changepoints.csv` | Accepted TEST detections, indices, timestamps, and configuration metadata |
| `aemo_detection_diagnostics.csv.gz` | Detector input and diagnostic statistics over time |
| `T2_detection_matches.csv` | Detection-level event-assignment audit |
| `T2_tier1_event_details.csv` | Tier 1 applicability, detection status, and delay details |
| `T2_detection_summary.csv` | Detector-region matching totals |
| `T2_aemo_detection.csv` / `T2_aemo_detection.md` | T2 display tables |
| `T2_manifest.json` | Protocol, source hashes, configurations, sample counts, and log identity |

The central log is [`results/tables/runs.csv`](../../results/tables/runs.csv). F2 PNG and PDF outputs are saved as `F2_detection_panel_SA1` and `F2_detection_panel_NSW1` under [`results/figures/detection/`](../../results/figures/detection/).

F2 upper panels display detector inputs and accepted alarms: green indicates Tier 1 matches, orange Tier 2 matches, and red crosses unmatched detections. Shading marks documented Tier 1 periods. Lower panels show ADWIN adaptive window width, Page-Hinkley cumulative increase, and KSWIN `-log10(p-value)`. These diagnostics have different meanings and scales; ADWIN window width is not a detector decision threshold. Official accepted detections come from the shared pipeline outputs.

Synthetic changepoints provide controlled reference labels. AEMO events are an incomplete contextual catalogue: an unmatched detection is **not automatically a false positive**, and a matched detection is not proof that an event caused concept drift. Detection quality alone also does not establish that retraining improves forecasting performance.

## Reproducibility and troubleshooting

Keep detection files, diagnostics, matching audits, registry configurations, and manifests from compatible runs. T2 validates stored detection metadata and index mappings, checks source hashes during evaluation, and avoids appending duplicate metric identities. F2 validates input columns, match statuses, and alignment before plotting.

### Protocol ID versus log split ID

The reviewed stored manifest records protocol ID `aemo_t1_7day_one_to_one_v2`, but its log `split_id` is:

```text
aemo_fixed_train2018_2019_test202003_task120_t2_t1rule
```

The reviewed T2 runner uses this same split ID. The matching-protocol Markdown instead lists a split ID ending in `_v2`. These identifiers must not be conflated: the stored manifest confirms v2 matching despite the absence of that suffix in its log split ID. Until the documentation discrepancy is reconciled, use the exact manifest identity and a compatible runner when reconstructing the stored table. The reviewed `--from-logs` implementation filters with the runner's `SPLIT` constant and the manifest's configuration hashes; changing the suffix alone can prevent it from finding the required rows.

| Problem | Check |
| --- | --- |
| T1 reports no matching results | Run from the repository root and verify the synthetic runner/table generator split IDs and available logged metrics |
| T2 reconstruction reports missing metrics | Check the runner split ID, manifest configuration hashes, and corresponding `runs.csv` rows |
| T2 rejects metadata or timestamp mappings | Confirm the detections, raw data, registry, and protocol belong to a compatible revision |
| F2 reports missing files or columns | Obtain the diagnostics and matching audit from compatible upstream outputs; the display table alone is insufficient |
| Outputs differ between environments | Check dependency versions, configuration values, seeds, preprocessing, and source provenance before interpreting differences |

The commands above document the supplied implementations; they do not imply that a fresh full pipeline run has been performed during documentation preparation.

## Related documentation

- [Main project README](../../README.md)
- [AEMO matching protocol](../../docs/aemo_matching_protocol.md) and [machine-readable settings](../../docs/aemo_matching_protocol.json)
- [Event-tiering criteria](../../docs/event_tiering_criteria.md)
- [Shared experimental decisions](../../docs/decisions.md)
- [Adaptation experiment documentation](../../docs/PGS242-128_README.md)

For historical implementations, consult the archive directories. Use the entry points and matching provenance described above for the documented workflow rather than mixing archived and current outputs.
