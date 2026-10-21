# PGS242-163 escalation threshold sweep

This ticket adds a human-review escalation policy on top of the PGS242-160
conformal intervals. It measures the trade-off between review rate and
coverage on the forecasts the system keeps. PGS242-164 (T6 operating points)
and PGS242-165 (F5 escalation curve) read its outputs, so the schema below is
the contract for those tickets.

[Runner](../experiments/uncertainty/run_escalation_sweep.py) ·
[Functions](../src/drift_lab/uncertainty.py) ·
[Tests](../tests/uncertainty/test_escalation.py) ·
[Run summary](../results/series/escalation_summary.md)

## Policy

- **Score:** interval width `upper - lower` at step t. The width is fixed
  when the interval is issued, before y_t is observed, so it is a valid
  ex-ante signal. The current-step nonconformity `|y_t - yhat_t|` is never
  used, because y_t is unknown when the decision is made.
- **Rule:** `escalate(t) = score(t) > tau`. The inequality is strict: a step
  whose score equals tau is retained.
- **Tau grid:** for each (region, uq_method, seed),
  `tau = k × W_cal`, with k = 0.50, 0.55, ..., 3.00 (51 values) plus
  `-inf` (escalate everything) and `+inf` (escalate nothing). That gives 53
  tau points, ordered strict to loose. `W_cal` is the calibration-window
  (2020-01-01 to 2020-02-29) 90% conformal width, `2 × conformal_radius`
  from the PGS242-160 summary.
- **Width rounding:** widths and `W_cal` are rounded to 6 decimals (1e-6 MW)
  before they are compared with tau. Otherwise floating-point noise in
  `(f + r) - (f - r)` would split a constant static width around `tau = W_cal`.

### Why the grid is width multiples, not calibration quantiles

The original specification asked for quantiles p0, p5, ..., p95, p99, p100 of
calibration-window widths. PGS242-160 builds intervals for the TEST stream
only (from 2020-03-01) and does not save calibration residuals. That leaves
`W_cal` as the only calibration-window width on record, and quantiles of a
single value would collapse. Adaptive conformal starts the TEST stream at
exactly `W_cal` (alpha = 0.10), so multiples of `W_cal` express "escalate when
the interval is k times wider than its calibrated starting width". The grid
uses no TEST information. The k range was fixed before looking at any TEST
widths.

`calibration_tau_grid(cal_scores, quantiles)` is still available for
quantile grids if calibration-window score streams are saved in the future.

## Expected finding: static conformal collapses

Static split conformal has a constant width `W_cal`, so its sweep only
produces review rates of 0 (k >= 1, +inf) and 1 (k < 1, -inf). Width-based
escalation is informative only for adaptive conformal. Static rows are kept
in every output.

## Inputs

- `results/tables/adaptation/revised/pgs242_128_five_seed_final/seed_<n>/`
  - `manifest.json`
  - `uncertainty/adaptive_conformal_manifest.json`
  - `uncertainty/aemo_conformal_comparison_summary.csv`
  - `uncertainty/aemo_conformal_intervals_<region>.csv.gz` (Git LFS on
    `pgs242-160-adaptive-conformal`)
- `data/events.csv`, the frozen catalogue. Regimes use the T5 full-scope
  assignment with 7-day buffers.
- `results/tables/uncertainty/t5_aemo/tables/t5_manifest.json`. Each interval
  stream's sha256 must match the hash T5 recorded; otherwise it is skipped.

Each stream's recomputed coverage and mean width must match the PGS242-160
summary to 1e-10. Static widths must equal `W_cal`. A missing stream is
skipped and listed in the manifest and run summary. No values are filled in.

## Outputs

### results/series/escalation_sweep.csv

One row per (region, configuration, regime, uq_method, seed, tau). Rows are
ordered by region (SA1, NSW1), configuration, uq_method (static, adaptive),
seed, regime (pre-drift, drift, post-drift, all), then tau ascending.

| Column | Meaning |
| --- | --- |
| config_hash | sha256 of protocol, upstream uq_config_hash, interval sha256, catalogue sha256, regime, score, rule and grid; one per (region, configuration, uq_method, seed) |
| split_id | PGS242-160 split identifier |
| region | SA1 or NSW1 |
| configuration | PGS242-160/T5 configuration (A, B, C_adwin, ..., D_60d_page_hinkley) |
| regime | pre-drift, drift, post-drift or all |
| uq_method | static_conformal or adaptive_conformal |
| seed | forecast seed |
| tau | threshold in MW (-inf/+inf at the endpoints) |
| tau_quantile | grid label: -inf, k0.5, k0.55, ..., k3, +inf |
| tau_multiplier | k (-inf/+inf at the endpoints) |
| calibration_width | W_cal in MW |
| n_total, n_escalated, n_retained | step counts |
| review_rate | n_escalated / n_total |
| retained_coverage | mean(lower <= y_true <= upper) over retained steps; NaN if n_retained = 0 |
| escalated_coverage | the same over escalated steps; NaN if n_escalated = 0 |
| retained_mean_width | mean width of retained steps in MW; NaN if n_retained = 0 |
| retained_mae | mean \|y_true − point_forecast\| over retained steps in MW; NaN if n_retained = 0 |
| escalated_mae | the same over escalated steps; NaN if n_escalated = 0 |

`retained_mae` and `escalated_mae` were added in schema version 2. They are
evaluation-only: the point forecast never enters the escalation decision,
and they don't change `config_hash`. The runner checks that each persisted
`point_forecast` is the midpoint of both conformal intervals.

### results/series/escalation_by_regime.csv

One row per (region, configuration, uq_method, seed, regime), with the same
key columns plus:

- `n_total`, `overall_coverage`, `mean_width` and `overall_mae`: the +inf
  row.
- `n_distinct_review_rates`: 2 means the sweep collapsed to {0, 1}.
- For targets 5%, 10% and 20% (`05pct`, `10pct`, `20pct`):
  `tau_at_review_<t>`, `tau_quantile_at_review_<t>`,
  `review_rate_at_review_<t>`, `retained_coverage_at_review_<t>`,
  `retained_mae_at_review_<t>` and `escalated_mae_at_review_<t>`. These
  come from the grid row whose review rate is closest to the target. Ties go
  to the lower review rate, then the smaller tau. Check the achieved review
  rate: static conformal can only reach 0 or 1.
- For 90% retained coverage: `review_rate_for_retained_coverage_90pct`,
  `tau_for_...`, `tau_quantile_for_...` and
  `retained_coverage_for_retained_coverage_90pct`. These come from the
  lowest review rate whose retained coverage is >= 0.90, with ties going to
  the smaller tau. The values are NaN if the target is never reached.
- `n_retrains` and `train_samples` from the PGS242-160 summary.

### results/runs.csv

Rows use the existing shared schema (`drift_lab.run_logger.RUNS_COLUMNS`),
with `group = escalation_sweep`, `method = <configuration>|<uq_method>`,
`regime` as above and `wall_clock_s = 0` (no training). There are six
metrics per (region, configuration, uq_method, seed, regime):

- `retained_coverage_at_review_05pct`
- `retained_coverage_at_review_10pct`
- `retained_coverage_at_review_20pct`
- `review_rate_for_retained_coverage_90pct`
- `retained_mae_at_review_10pct` (schema version 2)
- `escalated_mae_at_review_10pct` (schema version 2; NaN when nothing is
  escalated, which is always the case for static conformal at its 0% point)

`config_hash` links each row to the series files. The append reuses T5's
`append_run_metrics_safely`, keyed on group, method, dataset, region, seed,
config_hash, split_id, metric_name and regime. A rerun with identical inputs
adds nothing. A rerun after a schema extension adds only the new metrics.
Historical bytes are preserved, and a conflicting value for an existing
identity fails before anything is written.

### results/series/escalation_manifest.json and escalation_summary.md

The manifest records the protocol, `schema_version`, input and catalogue
hashes, T5 verification status, skipped inputs, the runs.csv metrics and
the output hashes. The summary is generated. It reports the static {0, 1}
check, the A vs B vs C_* comparison at the ~10% operating point (coverage,
retained/escalated MAE and their ratio), the per-region adaptive MAE ratio,
and arm A operating points. Both files are deterministic for identical
inputs.

## Reproduction

```bash
PYTHONPATH=src python experiments/uncertainty/run_escalation_sweep.py
PYTHONPATH=src python -m pytest -q tests/uncertainty/test_escalation.py
```

Options: `--seeds`, `--regions`, `--run-dir`, `--events`, `--output-dir`,
`--runs`, `--t5-manifest` and `--skip-runs-log`.

## Backlog

- **Lagged-nonconformity escalation score (later sprint).** Add a
  separately named score option that uses only past nonconformity scores,
  for example a lagged or rolling mean of `|y_s − yhat_s|` over steps
  s < t, and compare it with width-based escalation. Never use the
  current-step score. The motivation is the PGS242-163 finding that, under
  adaptive conformal, escalating the widest intervals lowers retained
  coverage, because the widest adaptive intervals are the best covered.
  This is deferred by decision on PGS242-163 and still needs a ticket.
