# T5 implementation and verification

## Status and evidence

Inspected main at d02c42528177ccbedc71ac7e50cad47600a4598d and
PGS242-160 at c924f38e5c6fbf83e03ba19ed39a3d3bc4d4a2b8 on 2026-10-14.
Neither complete Git tree contains
seed_<1..5>/uncertainty/aemo_conformal_intervals_<SA1|NSW1>.csv.gz.
These are the exact files written by run_aemo_adaptive_conformal.py.
They may exist on the original operator's machine, but are not available here.

The committed aemo_conformal_comparison_all_seeds.csv contains 340 rows:
17 configurations x 5 seeds x 2 regions x 2 methods (170 paired comparisons).
It is a full-test summary, not timestamp-level data. It cannot establish
worst 24h coverage or regime-level coverage. No real T5 has been generated.
No training or adaptive interval reconstruction has been performed.
The target in this upstream implementation is AEMO_TOTALDEMAND, measured in MW.

## Methodology: established versus unresolved

The source requirements specify six columns, nominal 90% coverage, regime
reporting, five seeds, mean plus sample SD, and runs.csv traceability.
docs/decisions.md does not define the width denominator or T5 rolling policy.
The source PDFs do not define these either.

Current T3 supports two explicitly different scopes:

- full: fixed AEST (Etc/GMT-10); NEM and region-specific events; union of
  [start - 7d, end-date + 1d + 7d); pre before the first overlapping buffered
  window; post elsewhere after it. Overlaps are unioned.
- event-local: [start-7d,start) pre; [start,end-date+1d) drift;
  [end-date+1d,end-date+1d+7d) post. Drift wins overlapping assignments,
  then pre, then post. Outside windows remains unassigned and is excluded
  from T5 regimes, but retained in time order for rolling evaluation.

Both reuse the existing shared implementations without changing the catalogue.
Selecting which scope is the official T5 scope remains a team decision.

The generator requires a supplied protocol JSON. It makes no claim that the
following choices have already been agreed:

- normalisation: one positive fixed regional scale, with statistic, source and
  source SHA256. Shared across methods, seeds, configurations and regimes.
  Values and provenance must be supplied; they are recorded, not independently
  derived or verified by this generator. A TRAIN-only regional statistic is a
  reasonable proposal, but no TRAIN range/SD/IQR has been chosen here.
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
An incomplete regime has NaN worst coverage, rendered N/A. The generator
rejects missing timestamps against the full test grid rather than silently
evaluating a smaller test population.

## Required protocol JSON

Create a protocol using the team's actual choices. The following is a template,
not a runnable or approved protocol (placeholders deliberately fail validation):

```json
{
  "protocol_id": "TEAM-SUPPLIED-ID",
  "regime_scope": "TEAM-SELECTS-full-OR-event-local",
  "regime_days": 7,
  "rolling_boundary": "TEAM-SELECTS-within_regime-OR-ending_regime",
  "rolling_closed": "right",
  "incomplete_windows": "exclude",
  "cadence": "30min",
  "granularity": "region_configuration",
  "test_start": "ACTUAL-FIRST-TEST-TIMESTAMP",
  "test_end": "ACTUAL-LAST-TEST-TIMESTAMP",
  "configurations": ["EXPLICIT-CONFIGURATIONS-FROM-EXPERIMENT"],
  "normalisation": {
    "SA1": {"value": null, "statistic": "TEAM-SUPPLIED",
            "source": "REFERENCE-ARTIFACT", "source_sha256": "ACTUAL-SHA256"},
    "NSW1": {"value": null, "statistic": "TEAM-SUPPLIED",
             "source": "REFERENCE-ARTIFACT", "source_sha256": "ACTUAL-SHA256"}
  }
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
  --protocol path/to/agreed_t5_protocol.json \
  --output-dir results/tables/uncertainty/t5_verified \
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
t5_window_audit.csv records seed-level metrics, observation and valid-window
counts. t5_manifest.json records protocol, source hashes, upstream identities,
scale provenance, catalogue hash and log hash. The evaluation config_hash
incorporates source, UQ configuration, catalogue and protocol.
wall_clock_s=0 documents that T5 does not record a new training run.

Tests use synthetic fixtures only; their tables are never real project results.

## Verification performed

Command: PYTHONPATH=src python -m pytest -q tests/uncertainty tests/test_t3_evaluation.py

Result: 41 passed. Includes independently enumerated 48-point coverage,
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
