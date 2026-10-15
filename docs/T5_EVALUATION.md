# T5 implementation and verification

## Status and evidence

The ten timestamp-level interval streams were added to PGS242-160 at
8987f528797c9c8982a99cac867521839569d07c and supplied as five seed archives.
All ten uploaded streams were read successfully. Each contains 17 configurations
with 67,249 consecutive half-hour observations per configuration, from
2020-03-01 00:00 through 2024-01-01 00:00 inclusive (fixed AEST).

Input checks passed for manifest seed/run/split identities, finite ordered
bounds, continuous unique timestamps, identical truth/grid across seeds and
configurations, stored coverage flags and widths, and recomputed full-test
coverage/mean width versus upstream summaries. The target is TOTALDEMAND in MW. A complete real AEMO T5 run has been generated locally; the input audit remains separate from the scope of numerical review.

## Current real AEMO run

[Results and interpretation](../results/tables/uncertainty/t5_aemo_post_event_grace/README.md) ·
[Tables and captions](../results/tables/uncertainty/t5_aemo_post_event_grace/tables/) ·
[Resolved protocol](../results/tables/uncertainty/t5_aemo_post_event_grace/protocol.json) ·
[Canonical log](../results/runs.csv)

The corrected local run completed with exit code 0, reusing stored intervals
without retraining. It generated 34 CSV and 34 Markdown tables, 4080 canonical
metric rows and 1020 window-audit records. All 816 mean ± sample SD cells agree
with independent aggregation of the log. Review did not include raw input streams. The aggregate verification described
here applies to the supplied pre-timing result package; the timed run has only
been checked through the user-reported exit code, row count, positive timing and
table counts. Its final snapshot/hash and aggregate checks remain to be repeated.

| Region | Regime | Observations | Complete 24h windows |
| --- | --- | ---: | ---: |
| SA1 | pre-drift | 720 | 673 |
| SA1 | drift | 16609 | 15716 |
| SA1 | post-drift | 49920 | 49074 |
| NSW1 | pre-drift | 720 | 673 |
| NSW1 | drift | 14304 | 13599 |
| NSW1 | post-drift | 52225 | 51520 |

All window counts exceed 10; sufficient support is not a coverage-quality claim.
The minimum of 10 remains an implementation choice, not a supervisor threshold.
The previous symmetric-buffer tables are historical. The full/within_regime choice
was reported as approved; separate approval of this correction is not asserted.

## Methodology: established versus unresolved

The source requirements specify six columns, nominal 90% coverage, regime
reporting, five seeds, mean plus sample SD, and runs.csv traceability.
The supervisor has specified the AEMO width denominator: the mean actual
TOTALDEMAND in the TRAIN window, computed separately for SA1 and NSW1 and
held fixed across methods, configurations, seeds and regimes. The generator
automatically calculates each value from its hashed regional data source using
TRAIN only: [2018-01-01, 2020-01-01), fixed AEST. Calibration/TEST values are
excluded. Each generated Markdown table caption reports the numeric regional
denominator in MW and the training window. Simulation retains scale 10.

T5 code supports two explicitly different scopes (event-local T3 behavior is retained):

- full: fixed AEST (Etc/GMT-10); NEM and region-specific events; union of
  [start, end-date + 1d + 7d); no pre-event buffer; pre before the first
  applicable event, post elsewhere after it. Overlaps are unioned.
- event-local: [start-7d,start) pre; [start,end-date+1d) drift;
  [end-date+1d,end-date+1d+7d) post. Drift wins overlapping assignments,
  then pre, then post. Outside windows remains unassigned and is excluded
  from T5 regimes, but retained in time order for rolling evaluation.

Both reuse the existing shared implementations without changing the catalogue.
The current AEMO run selects full; event-local is an alternative supported by code,
not the scope used for these uploaded results.

The generator requires an explicit protocol JSON. The current AEMO run uses
full/within_regime and threshold 10; the following describes supported controls:

- normalisation: `training_mean_actual_demand`, one positive TRAIN mean per
  region, automatically calculated from the regional TRAIN CSVs. No manual
  scale values are needed. The resolved protocol records value, observation
  count, source and hash; the generator verifies them before logging.
  The existing `synthetic-test-only` fixture is the sole exception (scale 10).
- window support: AEMO protocols must explicitly set a positive integer
  `minimum_complete_24h_windows`. The supervisor has not specified the numeric
  threshold, so no AEMO default is assumed. Regimes with fewer complete windows
  are marked `insufficient_windows` and their reported worst coverage is N/A.
  Zero windows are marked `no_complete_windows`. Counts/status are reported for
  every seed/regime, with no pooling or silent dropping of seeds. The fixture
  keeps its existing threshold of one and stored simulation outputs remain historical; regenerating under the corrected scope may change them.
- rolling attribution: within_regime (every observation must share the label)
  or ending_regime (a full window may cross boundaries, attributed to its end).
- scope: full or event-local.
- granularity: one table per region and explicitly listed configuration,
  with five-seed aggregation only. No pooling of regions/configurations.
- test_start/test_end: explicit inclusive timestamp grid endpoints. The stream
  must match every expected half-hour timestamp, in order.

Supported rolling convention is explicitly selected by the protocol:
(t-24h,t], 30min cadence, 48 observations including t. This represents
48 half-hour intervals even though the first-to-last timestamp separation is
23h30min. Only complete contiguous windows qualify. Missing or irregular
timestamps invalidate windows; duplicated or unsorted timestamps are rejected.
A regime below the explicit window-support threshold has NaN worst coverage,
rendered N/A. Rolling windows overlap; their count is not an independent-sample
count and a large count alone does not establish statistical reliability. The generator
rejects missing timestamps against the full test grid rather than silently
evaluating a smaller test population.

## Automatic AEMO TRAIN means

T5 computes the denominators itself; no manual mean values are required.
By default it reads the 24 original monthly CSV files for 2018–2019 under
`data/raw/SA1/` and `data/raw/NSW1/`, keeps only timestamps in TRAIN, and
calculates the arithmetic mean of actual `TOTALDEMAND`. The monthly files'
2020-01-01 endpoint is excluded. Each region contributes 35,039 observations.

| Region | Mean actual TRAIN demand (MW) |
| --- | ---: |
| SA1 | 1300.8730537401182 |
| NSW1 | 7982.25166471646 |

The calculation was independently checked using CSV parsing and decimal
arithmetic. Machine-readable values are in
[the TRAIN-mean reference](../results/tables/uncertainty/t5_training_mean_demand.json).
These are real training statistics, not formal T5 performance results.

Use `--training-data-dir /path/to/data/raw` for another data root with SA1/NSW1
subdirectories. A regional CSV can alternatively be supplied through its
`normalisation.<region>.source`. The generator replaces supplied scale values
with the computed means and records them in the resolved protocol, output
`t5_normalisation.json`, manifest and table captions. Directory provenance uses
a hash of sorted filenames and per-file UTF-8 hashes with canonical LF newlines,
so Git CRLF/LF checkout conversion does not alter the source identity. Single
CSV sources retain their raw SHA256. All 24 monthly TRAIN files must be present.

## Required protocol JSON

Create a protocol using the supervisor's training-mean denominator and the
team's remaining actual choices. The window-count threshold must be agreed
and supplied; null below deliberately fails validation. The following is a template,
not a runnable or approved protocol (placeholders deliberately fail validation):

```json
{
  "protocol_id": "TEAM-SUPPLIED-ID",
  "regime_scope": "TEAM-SELECTS-full-OR-event-local",
  "regime_days": 7,
  "pre_event_buffer_days": 0,
  "post_event_grace_days": 7,
  "training_window": {"start": "2018-01-01", "end_exclusive": "2020-01-01"},
  "minimum_complete_24h_windows": null,
  "rolling_boundary": "TEAM-SELECTS-within_regime-OR-ending_regime",
  "rolling_closed": "right",
  "incomplete_windows": "exclude",
  "cadence": "30min",
  "granularity": "region_configuration",
  "test_start": "ACTUAL-FIRST-TEST-TIMESTAMP",
  "test_end": "ACTUAL-LAST-TEST-TIMESTAMP",
  "configurations": ["EXPLICIT-CONFIGURATIONS-FROM-EXPERIMENT"],
  "normalisation": {}
}
```

The complete upstream configurations are A, B, C_adwin, C_kswin,
C_page_hinkley and D_<30|60|120|180>d_<adwin|kswin|page_hinkley>.
For all 17 configurations the output is 34 tables and 4,080 logged metric rows.
Every stream's configuration set must match the explicitly selected set.

## Execution and outputs

```bash
python experiments/uncertainty/generate_t5_uncertainty_table.py \
  --run-dir results/tables/adaptation/revised/pgs242_128_five_seed_final \
  --protocol results/tables/uncertainty/t5_aemo_post_event_grace/protocol.json \
  --training-data-dir data/raw \
  --output-dir results/tables/uncertainty/t5_aemo_post_event_grace/tables \
  --runs results/runs.csv
```

Provide the five upstream manifests, five UQ manifests, five UQ summaries and
ten compressed interval streams. Manifests must identify the same split and
matching seeds/upstream run IDs. Full-test coverage and width are independently
recomputed and compared with each summary before writes. Actual observations
and timestamp grids must be identical across seeds/configurations per region.

The default results/runs.csv is the specification's canonical location.
Existing upstream scripts use results/tables/runs.csv. --runs can explicitly
select that legacy location; the manifest records which file was used.
Historical rows are preserved, matching identical reruns do not append, and
conflicting values or duplicate identities fail. The shared log_run harness is
used to stage additions followed by atomic replacement. Run a single writer;
the existing shared harness has no cross-process locking.

Every T5_<region>_<configuration>.csv has exactly:
UQ method, regime, coverage, mean width, normalised width, worst 24h coverage.
Each cell uses the five seed metrics with ddof=1, not pooled observations.
Worst coverage is the minimum within each seed before seed aggregation.
If any seed is undefined, the displayed cell is N/A; no four-seed result is
silently substituted. Coverage and SD use fractions; width uses MW.

t5_seed_metrics.csv snapshots the exact selected runs.csv rows.
t5_window_audit.csv records seed-level metrics, observation counts, complete
window counts, the support threshold, and worst-window status. Each
T5_<region>_<configuration>.md contains a numeric denominator caption, the
same six-column table and complete-window counts/status for every seed/regime.
The CSV schema remains exactly six columns. t5_manifest.json records protocol, source hashes, upstream identities,
scale provenance, catalogue hash and log hash. The evaluation config_hash
incorporates source, UQ configuration, catalogue and protocol.
The timed generator measures wall_clock_s using time.perf_counter() around
metric evaluation for one seed/region/configuration/UQ method. The same measured
duration is repeated across its metric/regime rows: these are not additive costs.
It excludes input loading, table export and upstream model training. n_retrains
and train_samples retain upstream experiment metadata; their scope differs from
the evaluation duration. The shared log_run harness writes the log automatically.
The reported timed run contains 4080 positive durations, approximately
0.389472–0.563554 seconds. The previous zero-duration log is historical.

Tests use synthetic fixtures only; their tables are never real project results.

## Verification performed

Command: PYTHONPATH=src python -m pytest -q tests/uncertainty tests/test_t3_evaluation.py

Result: 48 passed after adding TRAIN-mean verification, caption/window-count
reporting, explicit AEMO threshold and sparse-window tests. Includes independently enumerated 48-point coverage,
right-open exclusion at t-24h, gaps, irregular cadence, DST-aware conversion,
duplicate/unsorted/missing timestamp rejection, disjoint regime segments,
overlap union, five-seed sample SD, full end-to-end synthetic generation,
log snapshot equality, idempotent reruns and preservation of historical bytes.
Corrupt truth/seed/summary and missing stream/timestamp fail before output writes.
git diff --check passed.

The existing T3 fixture omitted the frozen Tier 1 catalogue. The test now
includes its real Tier 1 rows alongside the synthetic event; production
catalogue validation is unchanged.

A broader detection-test collection was attempted but was blocked by missing
river and the legacy test's top-level evaluation import. Those tests were not
counted as passed. No claim is made about the full repository suite or real
T5 numerical validation.
