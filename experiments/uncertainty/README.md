# Uncertainty Quantification and Escalation Experiments

[Project README](../../README.md) · [Shared decisions](../../docs/decisions.md) · [T5 protocol notes](../../docs/T5_EVALUATION.md) · [Escalation schema](../../docs/ESCALATION_SWEEP.md) · [F4 notes](../../docs/F4_COVERAGE_TRAJECTORY.md)

## Purpose and scope

This directory turns the saved point forecasts from the four-arm adaptation experiment into **nominal 90% prediction intervals**, then evaluates them in two ways:

1. **Uncertainty (T5, F4).** How well do static conformal, adaptive conformal inference (ACI) and AgACI-BOA intervals hold their coverage before, during and after documented drift events, and how wide are they?
2. **Escalation (T6, F5).** If a causal score flags some forecasts for human review, what coverage do the *unflagged* (retained) forecasts reach, and how much review does it take?

Every script here reads stored forecasts or stored interval streams. **None of them refits a forecaster or re-runs a drift detector.** Forecast generation is upstream in [`experiments/adaptation/`](../adaptation/); detector alarms are upstream in [`experiments/detection/`](../detection/).

The escalation analysis is an offline review-allocation study. It does not model what a human reviewer would change, so "retained coverage" is the coverage of the forecasts the system keeps, not the coverage of a human-corrected system.

## Files and responsibilities

### Interval construction

| File | Ticket | Responsibility |
| --- | --- | --- |
| [run_aemo_uncertainty.py](run_aemo_uncertainty.py) | PGS242-141 | Initial static 90% split-conformal baseline for the frozen Gradient Boosting forecaster. Historical; superseded by the two runners below |
| [run_aemo_adaptive_conformal.py](run_aemo_adaptive_conformal.py) | PGS242-160 | Static conformal and ACI intervals for all 17 adaptation configurations, five seeds, SA1 and NSW1 |
| [run_aemo_agaci.py](run_aemo_agaci.py) | PGS242-165 (PR #42) | Experimental three-method comparison: static conformal, ACI and AgACI-BOA on the same forecast streams. Produces the interval streams used by the final T5, F4, T6 and F5 |

### Evaluation, tables and figures

| File | Output | Responsibility |
| --- | --- | --- |
| [generate_t5_uncertainty_table.py](generate_t5_uncertainty_table.py) | T5 | Six-column coverage/width table per region and configuration, with five-seed mean ± sample SD |
| [audit_t5_deliverables.py](audit_t5_deliverables.py) | T5 check | Recomputes the stored two-method T5 tables from seed metrics; writes nothing |
| [generate_f4_coverage_trajectory.py](generate_f4_coverage_trajectory.py) | F4 | Trailing-window coverage and width over the TEST stream, by detector panel |
| [validate_f4_canonical_log.py](validate_f4_canonical_log.py) | F4 check | Checks F4 outputs against the stored T5 metrics and input hashes |
| [run_escalation_sweep.py](run_escalation_sweep.py) | T6/F5 input | Threshold sweep using **current interval width** as the review score |
| [run_escalation_lag1_nonconformity.py](run_escalation_lag1_nonconformity.py) | T6/F5 input | Threshold sweep using the **previous half-hour's absolute forecast error** as the review score |
| [generate_t6_escalation_table.py](generate_t6_escalation_table.py) | T6 | Operating points from a stored sweep: review needed for 90%/95% retained coverage, and best coverage under 5/10/20% review budgets |
| [generate_f5_escalation_curve.py](generate_f5_escalation_curve.py) | F5 | Retained coverage versus review rate, by regime, from a stored sweep |

Shared components:

- [Conformal intervals and escalation functions](../../src/drift_lab/uncertainty.py): `conformal_radius`, `prediction_intervals`, `adaptive_prediction_intervals`, `escalation_flags`, `retained_coverage`, `calibration_width_tau_grid`, `escalation_sweep`
- [AgACI-BOA aggregation](../../src/drift_lab/agaci.py): `agaci_prediction_intervals` and the BOA expert weights
- [Evaluation and regime labelling](../../src/drift_lab/evaluation.py): coverage, width and `assign_full_stream_forecast_regimes`
- [Experiment logger](../../src/drift_lab/run_logger.py)

## Data flow

```text
experiments/adaptation/run_four_arm_adaptation.py
  └─ results/tables/adaptation/revised/pgs242_128_five_seed_final/seed_<n>/
       manifest.json, summary.csv, forecasts_<region>.csv.gz
          │
          ├─ run_aemo_adaptive_conformal.py  ──► seed_<n>/uncertainty/aemo_conformal_intervals_<region>.csv.gz
          │                                       (static + ACI; T5 two-method run in t5_aemo/)
          │
          └─ run_aemo_agaci.py --output-dir results/tables/uncertainty/agaci_boa_fixed_calibration
               └─ seed_<n>/agaci_comparison_intervals_<region>.csv.gz   (Git LFS)
                    │
                    ├─ generate_t5_uncertainty_table.py --agaci-dir ...   ──► T5  (t5_agaci_full_post7/)
                    ├─ generate_f4_coverage_trajectory.py                 ──► F4
                    ├─ run_escalation_sweep.py --agaci-dir ...            ──► sweep (escalation_agaci_full_post7/)
                    └─ run_escalation_lag1_nonconformity.py --agaci-dir ...──► sweep (escalation_lag1_nonconformity_full_post7/)
                                                                               │
                                       generate_t6_escalation_table.py  ◄──────┤──► T6
                                       generate_f5_escalation_curve.py  ◄──────┘──► F5
```

## Data split and causality

The split is the shared one from [decisions.md](../../docs/decisions.md):

| Partition | Boundaries (fixed AEST) | Role in this stage |
| --- | --- | --- |
| TRAIN | `2018-01-01 <= t < 2020-01-01` | Upstream forecaster fitting; also the T5 width denominator |
| CALIBRATION | `2020-01-01 <= t < 2020-03-01` | Residuals that set the conformal radius and the escalation tau grid |
| TEST | `2020-03-01 00:00` to `2024-01-01 00:00` inclusive | Intervals, coverage and escalation are evaluated here only |

Each TEST stream has **67,249 half-hourly observations per configuration**.

The causal rules the runners enforce:

- Calibration residuals are **seed-specific** and come from the calibration period only. They are frozen across TEST.
- For ACI and AgACI-BOA, the interval for step `t` is built first; `y_t` updates the online state only afterwards.
- Online state is reset for every seed × region × configuration. No state carries between streams.
- The interval-width score is known when the interval is issued. The lagged-error score at `t` uses only `|y_(t-1) − ŷ_(t-1)|`; the first TEST step has no previous TEST error and gets score 0 (retained).
- The tau grid is built from the calibration radius only. Changing TEST truth does not change it ([test](../../tests/uncertainty/test_escalation.py) `test_runner_tau_grid_unchanged_when_test_truth_perturbed`).

## Uncertainty methods

All methods target 90% coverage (`alpha = 0.10`).

| Method | How the interval is built | Key settings |
| --- | --- | --- |
| **Static conformal** | Symmetric radius = the `ceil((n+1)(1−alpha))`-th smallest absolute calibration residual (finite-sample split conformal). Same width at every step | `DEFAULT_ALPHA = 0.10` |
| **ACI** | Starts at the static width, then updates its working miscoverage level online: `alpha_(t+1) = alpha_t + gamma × (target_alpha − error_t)` | `DEFAULT_GAMMA = 0.005` |
| **AgACI-BOA** | Runs several ACI experts with different learning rates and aggregates their bounds with BOA weights. Intervals may be asymmetric | Eight experts, `AGACI_GAMMAS = 0.001, 0.002, …, 0.128` (doubling) in [agaci.py](../../src/drift_lab/agaci.py) |

AgACI-BOA here uses **fixed calibration residuals**. It is not the rolling-retrained protocol of the original AgACI paper; the runner, its manifest and its protocol ID (`team42-experimental-agaci-boa-fixed-calibration-v1`) all say so.

## T5: interval reliability by regime

### Regimes

T5 uses the **full-stream scope with 7-day buffers**: every TEST timestamp is labelled `pre-drift`, `drift` or `post-drift` from the frozen catalogue [`data/events.csv`](../../data/events.csv). Overlapping buffered event windows are unioned. Event-local scope is supported by the code but was not used for the stored results.

### Metrics

| Column | Definition |
| --- | --- |
| coverage | Share of observations with `lower <= y <= upper` |
| mean width | Mean of `upper − lower`, in MW |
| normalised width | Mean width ÷ the region's mean actual TRAIN demand (SA1 1300.87 MW, NSW1 7982.25 MW) |
| worst 24h coverage | Lowest coverage over complete, contiguous 48-step windows that stay inside one regime |

Cells are the mean ± sample SD (`ddof=1`) of the five seed values. Worst-window coverage is taken within each seed before averaging. A regime with fewer than 10 complete windows shows N/A. Rolling windows overlap, so their count is not an independent sample size.

The generator needs an explicit protocol JSON. The one used for the stored results is [`results/tables/uncertainty/t5_aemo/protocol.json`](../../results/tables/uncertainty/t5_aemo/protocol.json) (`aemo-t5-full-within-regime-v1`).

## Escalation: T6 and F5

### Policy

- **Rule:** `escalate(t) = score(t) > tau`. Strict inequality, so a step with score equal to tau is retained.
- **Review rate:** share of steps escalated.
- **Retained coverage:** coverage of the steps not escalated. Undefined when nothing is retained.
- **Tau grid:** `tau = k × reference`, with `k = 0.50, 0.55, …, 3.00` (51 values) plus `−inf` (escalate all) and `+inf` (escalate none).

### The two scores

| Score | Signal at step `t` | Tau reference | Stored sweep |
| --- | --- | --- | --- |
| Interval width | `upper_t − lower_t` | `W_cal` = static calibration width = 2 × conformal radius | [`results/series/escalation_agaci_full_post7/`](../../results/series/escalation_agaci_full_post7/) |
| Lagged nonconformity (30 min) | `|y_(t−1) − ŷ_(t−1)|` | Calibration radius = `W_cal / 2` (same MW units as an absolute error) | [`results/series/escalation_lag1_nonconformity_full_post7/`](../../results/series/escalation_lag1_nonconformity_full_post7/) |

The two scores answer different questions. Width says how uncertain the method is right now. The lagged error says how badly the forecast just missed. Their operating points are not interchangeable.

### What the stored results show

- **Static conformal cannot be escalated by width.** Its width is constant, so the width sweep only produces review rates of 0 or 1. Static rows are kept in every output for completeness.
- **Width-based escalation works against ACI.** ACI widens the interval when it has recently missed, so the widest ACI intervals are often the best covered. Escalating them lowers retained coverage. This finding motivated the lagged-error score.
- **The lagged-error score gives usable operating points.** For example, NSW1 / Arm A / ACI over the whole stream retains 96.05% ± 0.12 pp coverage at a 10% review budget, and reaches 95% retained coverage at 8.04% ± 0.24 pp review ([T6, lagged error](../../results/tables/uncertainty/t6_nonconformity_30min/T6_operating_points.md)). With interval width, the same configuration stays at about 90.0% at every budget ([T6, width](../../results/tables/uncertainty/t6_interval_width/T6_operating_points.md)).

### T6 selection rules

From [generate_t6_escalation_table.py](generate_t6_escalation_table.py):

- **Coverage target (90%, 95%):** the smallest observed review rate whose retained coverage is at least the target. "Unreachable" if no grid point gets there. If only some seeds reach it, the cell says so (for example "partial: 3/5 seeds").
- **Review budget (5%, 10%, 20%):** the highest retained coverage among points with review rate ≤ the budget. Ties go to less review, then the smaller tau.
- Only stored grid points are used. There is no interpolation between thresholds.

F5 averages the five seeds at matching grid labels (`tau_quantile`), not at matching absolute tau values. It plots four arms × three regimes per region/detector; Arm D uses the 180-day window only.

## Reproduction

Run from the repository root. Install the package first (`python -m pip install -e '.[dev]'`) or prefix commands with `PYTHONPATH=src`.

**Prerequisites:**

- The five-seed adaptation run under `results/tables/adaptation/revised/pgs242_128_five_seed_final/` (see the [adaptation README](../adaptation/)).
- The AgACI interval streams `results/tables/uncertainty/agaci_boa_fixed_calibration/seed_<n>/agaci_comparison_intervals_<region>.csv.gz` are stored with **Git LFS** ([.gitattributes](../../.gitattributes)). Run `git lfs pull` before anything that reads them. Without LFS you only get pointer files and the T5/F4/T6 inputs fail their hash checks.
- `data/raw/SA1/` and `data/raw/NSW1/` for the T5 TRAIN-mean denominator.

Most generators refuse to overwrite an existing output folder. Use a new `--output-dir` for each rerun.

### 1. Build intervals

```bash
# Static + ACI (PGS242-160). Writes into each seed's uncertainty/ folder.
python experiments/uncertainty/run_aemo_adaptive_conformal.py \
  --run-dir results/tables/adaptation/revised/pgs242_128_five_seed_final \
  --skip-shared-log

# Static + ACI + AgACI-BOA (experimental three-method comparison)
python experiments/uncertainty/run_aemo_agaci.py \
  --run-dir results/tables/adaptation/revised/pgs242_128_five_seed_final \
  --output-dir results/tables/uncertainty/agaci_boa_fixed_calibration_NEW
```

### 2. T5 tables

```bash
python experiments/uncertainty/generate_t5_uncertainty_table.py \
  --agaci-dir results/tables/uncertainty/agaci_boa_fixed_calibration \
  --protocol results/tables/uncertainty/t5_aemo/protocol.json \
  --output-dir results/tables/uncertainty/t5_agaci_NEW
```

Use `--run-dir` instead of `--agaci-dir` for the original two-method (static + ACI) inputs. Exactly one of the two is required.

### 3. F4 coverage trajectories

```bash
python experiments/uncertainty/generate_f4_coverage_trajectory.py \
  --output-dir results/figures/uncertainty/f4_NEW
python experiments/uncertainty/validate_f4_canonical_log.py \
  --f4-dir results/figures/uncertainty/f4_NEW
```

F4 requires seeds 1–5 unless `--allow-preview` is passed. Its rolling window length is read from the F1 figure script so the two figures stay aligned.

### 4. Escalation sweeps

```bash
# Interval-width score
python experiments/uncertainty/run_escalation_sweep.py \
  --agaci-dir results/tables/uncertainty/agaci_boa_fixed_calibration \
  --t5-manifest results/tables/uncertainty/t5_agaci_full_post7/t5_manifest.json \
  --output-dir results/series/escalation_width_NEW --skip-runs-log

# Lagged 30-minute absolute-error score (requires --agaci-dir)
python experiments/uncertainty/run_escalation_lag1_nonconformity.py \
  --agaci-dir results/tables/uncertainty/agaci_boa_fixed_calibration \
  --t5-manifest results/tables/uncertainty/t5_agaci_full_post7/t5_manifest.json \
  --output-dir results/series/escalation_lag1_NEW --skip-runs-log
```

Each stream's SHA-256 must match the hash in the T5 manifest; a mismatched stream is skipped and listed in the sweep manifest rather than silently used. Leave out `--skip-runs-log` only when you mean to append to `results/runs.csv`.

### 5. T6 and F5 from a stored sweep

```bash
python experiments/uncertainty/generate_t6_escalation_table.py \
  --sweep results/series/escalation_lag1_nonconformity_full_post7/escalation_sweep.csv \
  --summary results/series/escalation_lag1_nonconformity_full_post7/escalation_by_regime.csv \
  --manifest results/series/escalation_lag1_nonconformity_full_post7/escalation_manifest.json \
  --output-dir results/tables/uncertainty/t6_NEW

python experiments/uncertainty/generate_f5_escalation_curve.py \
  --sweep results/series/escalation_lag1_nonconformity_full_post7/escalation_sweep.csv \
  --output-dir results/figures/uncertainty/f5_NEW \
  --score-label 'Previous 30-minute absolute error'
```

`--max-review` (default 0.25) sets the right edge of the F5 x-axis.

## Output map

| Result | Location |
| --- | --- |
| Static + ACI intervals (PGS242-160) | `results/tables/adaptation/revised/pgs242_128_five_seed_final/seed_<n>/uncertainty/` |
| Three-method interval streams and summaries | [`results/tables/uncertainty/agaci_boa_fixed_calibration/`](../../results/tables/uncertainty/agaci_boa_fixed_calibration/) |
| T5, two methods (static, ACI) | [`results/tables/uncertainty/t5_aemo/`](../../results/tables/uncertainty/t5_aemo/) |
| **T5, three methods (final)** | [`results/tables/uncertainty/t5_agaci_full_post7/`](../../results/tables/uncertainty/t5_agaci_full_post7/) |
| T5 TRAIN-mean denominators | [`results/tables/uncertainty/t5_training_mean_demand.json`](../../results/tables/uncertainty/t5_training_mean_demand.json) |
| F4 figures and data | [`results/figures/uncertainty/f4/`](../../results/figures/uncertainty/f4/) |
| Width sweep (two-method, original) | [`results/series/`](../../results/series/) `escalation_*.csv` |
| **Width sweep (three-method)** | [`results/series/escalation_agaci_full_post7/`](../../results/series/escalation_agaci_full_post7/) |
| **Lagged-error sweep (three-method)** | [`results/series/escalation_lag1_nonconformity_full_post7/`](../../results/series/escalation_lag1_nonconformity_full_post7/) |
| **T6, interval width** | [`results/tables/uncertainty/t6_interval_width/`](../../results/tables/uncertainty/t6_interval_width/) |
| **T6, lagged error** | [`results/tables/uncertainty/t6_nonconformity_30min/`](../../results/tables/uncertainty/t6_nonconformity_30min/) |
| **F5, interval width** | [`results/figures/uncertainty/f5_width/`](../../results/figures/uncertainty/f5_width/) |
| **F5, lagged error** | [`results/figures/uncertainty/f5_nonconformity/`](../../results/figures/uncertainty/f5_nonconformity/) |

Bold rows are the final results used in the report, presentation and dashboard.

## Tests

```bash
PYTHONPATH=src python -m pytest -q tests/uncertainty tests/test_f4_coverage_trajectory.py
```

| Test file | Covers |
| --- | --- |
| [test_uncertainty.py](../../tests/uncertainty/test_uncertainty.py) | Conformal radius, static and adaptive intervals, causal update order |
| [test_t5_evaluation.py](../../tests/uncertainty/test_t5_evaluation.py) | T5 windows, regimes, sample SD, traceability, corrupt-input rejection |
| [test_escalation.py](../../tests/uncertainty/test_escalation.py) | Strict rule, tau grid, operating points, idempotent `runs.csv` appends |
| [test_lag1_nonconformity_leakage.py](../../tests/uncertainty/test_lag1_nonconformity_leakage.py) | The lagged score never uses `y_t` at step `t` |
| [test_t6_escalation_table.py](../../tests/uncertainty/test_t6_escalation_table.py) | T6 target and budget selection |
| [test_f4_coverage_trajectory.py](../../tests/test_f4_coverage_trajectory.py) | F4 trailing windows and validation |

On 15 Nov 2026, with `data/events.csv` present, this command gave **136 passed, 4 skipped**. The tests use synthetic fixtures; they check the code, not the stored real-data numbers.

## Known issues

- The module docstring of `run_escalation_lag1_nonconformity.py` was copied from the width runner and still describes the score as interval width. The code, manifest and protocol ID use the lagged absolute error. Trust the code.
- Output defaults don't match where results are stored. `run_aemo_agaci.py` defaults to `results/experimental/...`, but the stored streams are in `results/tables/uncertainty/agaci_boa_fixed_calibration/`. `generate_f4_coverage_trajectory.py` defaults to `f4_agaci_full_post7`, `validate_f4_canonical_log.py` defaults to `f4_by_detector`, and the stored F4 is in `f4/`. Pass explicit paths, as in the commands above.
- `run_aemo_adaptive_conformal.py` logs to the legacy `results/tables/runs.csv`; T5 and the escalation runners use `results/runs.csv`.
- The example command in [docs/F4_COVERAGE_TRAJECTORY.md](../../docs/F4_COVERAGE_TRAJECTORY.md) uses `--configurations`, which the current F4 parser no longer accepts. Use `--detectors`.
- T5 manifest source paths are absolute paths from the machine that ran it. The SHA-256 values, not the paths, are what identify the inputs.

## Limitations

- Coverage is evaluated on one historical TEST stream per region. The F5 curves and T6 operating points are chosen on that same stream, so they describe what happened, not a guaranteed future operating point.
- The event catalogue is incomplete and dated at day or month precision. Regime labels are a reference, not ground truth for when drift happened.
- AgACI-BOA uses fixed calibration and is labelled experimental.
- Retained coverage leaves out escalated steps. It does not measure what a reviewer would have done with them.
