# F4 — Coverage Trajectory

F4 plots trailing empirical coverage from stored PGS242-160 prediction intervals.
It does not retrain forecasts or reconstruct uncertainty outputs. The committed
results use the T5 PR #38 snapshot at
`b3beabc72b01e1a497e2764ec967a34080c45425`.

## Scope and figure definition

- Regions: SA1 and NSW1; seeds: 1–5.
- Forecast configurations: A, B, C_kswin, D_180d_page_hinkley, reported separately.
  These are four selected configurations, not all 17 configurations evaluated by T5.
- Two lines per figure: static and adaptive conformal, with nominal coverage 0.90.
- Rolling window: **336 half-hour observations (7 days), matching F1**.
- Each window ends at the current timestamp and contains no later observations.
  The first 335 timestamps have no estimate. Windows continue across regime boundaries.
- Lines show the equal-weight seed mean; shading around each line shows sample
  standard deviation across five seeds (`ddof=1`). Only display bands are clipped
  to [0,1]; exact SD values remain in the stored CSV files.
- Event shading uses the same buffered event periods as this T5 snapshot. Darker
  shading identifies Tier 1 events. These are documented contextual events, not
  proof that concept drift occurred at every shaded timestamp.

## T5 alignment

The full TEST stream is assigned to pre-drift, drift or post-drift, without
unassigned observations. For the inspected snapshot:

- Drift is the union of `[event_start - 7 days, event_end + 1 day + 7 days)`.
- Pre-drift precedes the first applicable buffered event window.
- Post-drift covers timestamps after a buffered window, until the next begins.
- Overlapping drift windows are treated as a union. Region/event applicability
  and labels are calculated using the snapshot's shared evaluation functions.

This describes the **current T5 implementation**; it does not assert final
supervisor approval of the event-buffer interpretation. If the agreed definition
changes, refresh the integration and regenerate the regime audits, shading and
T5 checks together. The continuous seven-day coverage values do not depend on labels.

Companion checks recalculate coverage **with interval width**, normalised width,
and worst 24-hour coverage directly from the stored interval bounds. Coverage
includes both endpoints. Regional TRAIN mean demand denominators are fixed:

| Region | TRAIN mean demand (MW) |
| --- | ---: |
| SA1 | 1300.8730537401182 |
| NSW1 | 7982.25166471646 |

TRAIN is `[2018-01-01, 2020-01-01)` in fixed AEMO UTC+10. The 24-hour check uses
48 observations in `(t-24h, t]`, requires a complete regular window entirely
within one regime, and excludes incomplete/boundary-crossing windows. Each
regime's observation count and complete-window count are checked against T5.
The snapshot requires at least 10 eligible windows per seed/regime to report
worst coverage; this number is an implementation choice. Below it, the metric
is undefined (`N/A` in T5). Overlapping windows are not independent samples.

**Do not compare T5 coverage to the simple average of the seven-day curve.**
T5 coverage is recalculated from pointwise covered indicators. T5's worst
24-hour metric is separately recalculated; it is not the minimum of F4's
seven-day curve. Worst coverage is calculated per seed before mean/SD aggregation.

## Prepare the pinned review reference

The following commands are for **PowerShell**, from the repository root, with
the project Python environment active. The reference export is a local dependency
and is deliberately separate from the team working files. It need not be committed.

```powershell
git fetch origin pull/38/head:refs/remotes/origin/pr-38
git cat-file -t b3beabc72b01e1a497e2764ec967a34080c45425
git archive --format=zip --output=T5_PR38_Review.zip b3beabc72b01e1a497e2764ec967a34080c45425 -- docs/T5_EVALUATION.md docs/decisions.md src/drift_lab/evaluation.py experiments/uncertainty/generate_t5_uncertainty_table.py experiments/uncertainty/audit_t5_deliverables.py results/tables/uncertainty/t5_aemo results/verification/t5_deliverable_audit.json
Expand-Archive -LiteralPath .\T5_PR38_Review.zip -DestinationPath .\T5_PR38_Review
git archive --format=zip --output=T5_Canonical_Log.zip b3beabc72b01e1a497e2764ec967a34080c45425 -- results/runs.csv
```

The `git cat-file` command should print `commit`. If the PR reference is already
available, fetching is unnecessary. If the PR history has changed and this commit
is missing, obtain the pinned commit from the team rather than silently evaluating
a different snapshot. Keep existing exports if they already match; do not extract
over them blindly. Proxy settings, if needed, are local environment configuration.

The generator deliberately verifies the inspected evaluation.py bytes; it rejects
a changed reference. A file-byte mismatch, including an export newline change,
requires checking the export before changing the guard. The canonical-log validator
accepts the manifest's LF hash after CRLF/LF normalisation, and also requires parsed
records to agree exactly. It verifies the canonical ZIP's git commit comment.

## Required stored experiment inputs

For each seed 1–5, the run directory must contain:

```text
seed_N/uncertainty/adaptive_conformal_manifest.json
seed_N/uncertainty/aemo_conformal_comparison_summary.csv
seed_N/uncertainty/aemo_conformal_intervals_SA1.csv.gz
seed_N/uncertainty/aemo_conformal_intervals_NSW1.csv.gz
```

The ten interval files are upstream PGS242-160 outputs and are not duplicated
by this F4 change. The generator requires their SHA256 hashes to match the T5
source manifest exactly. Use real Git LFS objects rather than pointer files.
The original shared `evaluation.py`, `data/events.csv`, F1 generator and T3
manifest must also be available in the main checkout. F1 window constants and
T3 alpha/window compatibility are checked before generation.

If the interval files are missing locally, obtain the published uncertainty
outputs from `pgs242-160-adaptive-conformal`. One way to inspect that branch
without changing the F4 checkout is a separate worktree:

```powershell
git fetch origin pgs242-160-adaptive-conformal
git worktree add --detach ..\f4-uq-inputs FETCH_HEAD
git -C ..\f4-uq-inputs lfs pull
```

Inspect the required files under that worktree's
`results/tables/adaptation/revised/pgs242_128_five_seed_final/`. The branch may
advance; exact interval hashes in the pinned T5 manifest, rather than the branch
name alone, establish compatibility. Do not regenerate model outputs to bypass
a mismatch. The generator can read that external directory via `--run-dir`;
provide the main checkout's T3 manifest with `--t3-manifest` and an explicit
`--output-dir` in the F4 checkout when using an external run directory.

## Run

With the required input files in the default project run directory:

```powershell
python tests/test_f4_coverage_trajectory.py
python experiments/uncertainty/generate_f4_coverage_trajectory.py --run-dir results/tables/adaptation/revised/pgs242_128_five_seed_final --configurations A B C_kswin D_180d_page_hinkley
python experiments/uncertainty/validate_f4_canonical_log.py
```

The T5 integration test class requires the extracted reference. Confirm that the
test output says **8 tests, OK**, with no skipped integration tests.
Default generated results are under:
`results/tables/adaptation/revised/pgs242_128_five_seed_final/f4_t5_review/`.
The log check updates `F4_validation.json` and saves canonical-log evidence.
Run it **after** the generator: a fresh generator run replaces the main report.

For a reproducibility check, use the same generator command with a separate
`--output-dir`, then run the log validator with `--f4-dir` pointing to that directory.
PNG rendering can vary with Matplotlib/fonts; compare numerical outputs and
provenance as well as image files. Runtime environment details should be retained
when exact image-byte reproduction is needed.

## Outputs and verified status

The result directory contains eight PNG figures; figure captions; 40 per-seed
trajectories; eight trajectory mean/SD files; event and regime audits; coverage
and width summaries; T5 recomputed metrics; and validation records. Paths in the
source records refer to the producing machine; SHA256 values identify the inputs.

On the completed AEMO run:

- 40 seed/region/configuration streams passed chronology, bounds, inclusive
  coverage flag, right-aligned window and prefix-invariance checks.
- 240 comparisons to upstream full-TEST summary values passed.
- 960 seed/regime T5 metric comparisons passed; observation and eligible-window
  counts also matched the T5 audits.
- 192 displayed T5 metric cells matched five-seed mean and sample SD.
- All 4,080 T5 seed-metric records matched the canonical `results/runs.csv`;
  all 960 selected F4 metric comparisons traced to that canonical log.
- Eight focused tests passed. A synthetic two-region/five-seed end-to-end check
  reproduced all 26 fixture output files byte for byte. This synthetic check is
  distinct from the real AEMO validation above.

Useful evidence files:

```text
F4_validation.json
F4_upstream_summary_validation.csv
F4_t5_recomputed_metrics.csv
F4_t5_seed_validation.csv
F4_t5_table_validation.csv
F4_canonical_log_validation.csv
F4_canonical_log_validation.json
F4_captions.txt
```

This change writes figure/validation outputs only. It does not edit T5 tables,
shared UQ logic, the team's decisions, or the canonical experiment log.
