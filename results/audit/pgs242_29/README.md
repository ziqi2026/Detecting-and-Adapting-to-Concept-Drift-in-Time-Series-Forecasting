# PGS242-29 — End-to-end run and leakage audit

**Ticket:** PGS242-29 · **Run:** 15 Nov 2026 · **Script:** [`experiments/audit/run_end_to_end_audit.py`](../../../experiments/audit/run_end_to_end_audit.py) · **Full check table:** [AUDIT_REPORT.md](AUDIT_REPORT.md)

## Verdict

- **Leakage audit: passed.** Changing every demand value after 1 Jan 2022 left all alarms, forecasts, intervals and review scores issued before that point exactly unchanged (difference 0), in both regions and all four arms. TEST data never moved the calibration radius. Every retrain used only data up to its retrain time.
- **End-to-end run: passed and reproducible.** Re-running detection → adaptation → uncertainty from the raw AEMO files reproduced the committed change points (81 alarms, max difference 1.8e-15) and the committed seed-1 adaptation results exactly (MAE, retrain counts, training samples).
- **One reproducibility problem found (not leakage).** The committed three-method interval streams in `results/tables/uncertainty/agaci_boa_fixed_calibration/` were built with calibration residuals that the current code does not reproduce. See [Finding](#finding-agaci-calibration-residuals-do-not-reproduce). These streams feed the final T5, F4, T6 and F5.

32 of 34 checks pass; the 2 failures are this finding.

## What was run

The script makes two scratch copies of the repository and runs the real pipeline scripts in each, from `data/raw/`:

| Stage | Script | Settings |
| --- | --- | --- |
| Detection | `experiments/detection/run_aemo_drift_detection.py` | All three detectors, both regions, as committed |
| Adaptation | `experiments/adaptation/run_four_arm_adaptation.py` | Seed 1; arms A, B, C and D (180 days); Page-Hinkley triggers |
| Uncertainty | `experiments/uncertainty/run_aemo_agaci.py` | Static conformal, ACI and AgACI-BOA |
| Escalation score | Same formula as `run_escalation_lag1_nonconformity.py` | `\|y_(t−1) − ŷ_(t−1)\|` |

- **Baseline copy:** committed data, unchanged.
- **Perturbed copy:** every raw `TOTALDEMAND` value after 2022-01-01 00:00 multiplied by 1.25 and shifted by +300 MW (420,480 raw rows). Everything before that is byte-identical. The first affected half-hour is 2022-01-01 00:30.

The idea: in a causal pipeline, nothing decided at time `t` can depend on data after `t`. So everything up to the first perturbed half-hour must be identical between the two copies, and things after it must differ (which shows the perturbation actually reached the pipeline).

Each copy took about 6 minutes in the audit environment (detection ~3 min, adaptation ~1 min, uncertainty ~1 min).

## Checks and results

| Area | What was checked | Result |
| --- | --- | --- |
| Split | TRAIN < CALIBRATION < TEST with no overlap; 35,039 / 2,880 / 67,249 rows per region; continuous 30-minute grid; `data/processed/` equals the loader output | Pass (both regions) |
| Reproducibility | Re-run change points vs committed `aemo_detection_changepoints.csv` | Pass — 81 alarms, identical |
| Reproducibility | Re-run seed-1 adaptation vs committed `pgs242_128_five_seed_final/seed_1/summary.csv` | Pass — identical MAE, retrains and samples for all 8 region/arm runs |
| Reproducibility | Re-run static calibration width vs committed PGS242-160 | Pass — SA1 868.4158 MW, NSW1 2765.0036 MW, identical |
| Reproducibility | Re-run static calibration width vs committed AgACI comparison | **Fail** — see the finding below |
| Future perturbation | Accepted alarms before 2022-01-01 00:30 | Pass — identical (SA1 18, NSW1 27); after the cutoff they change |
| Future perturbation | Forecasts up to and including 2022-01-01 00:30, all four arms | Pass — 128,840 rows per region, difference 0 |
| Future perturbation | Static, ACI and AgACI-BOA intervals up to and including 2022-01-01 00:30 | Pass — difference 0 |
| Future perturbation | Calibration radius | Pass — unchanged |
| Future perturbation | Lagged review score | Pass — unchanged through 00:30, first changes at 01:00 (it uses `y` from 00:30) |
| Adaptation | Training window end ≤ retrain time, and first forecast from the refitted model > training window end | Pass — 56 retrain events per region, 0 violations |
| Single step | Change only `y_t`: ACI and AgACI-BOA intervals at `t` and before are unchanged; ACI reacts at `t+1`; lag score changes only at `t+1` | Pass |

### Repository test suite

On the same environment: `PYTHONPATH=src python -m pytest -q` → **209 passed, 4 skipped** (including the PatchTST tests with PyTorch installed). Without PyTorch, `tests/degradation/test_patchtst_forecaster.py` fails to import; the other 192 pass.

## Finding: AgACI calibration residuals do not reproduce

`run_aemo_agaci.py` and `run_aemo_adaptive_conformal.py` both rebuild calibration residuals the same way: fit a fresh `GradientBoostedForecaster(feature_mode="calendar+lags", lags=(48, 336), seed=seed)` on TRAIN and forecast CALIBRATION. In this environment that gives exactly the committed PGS242-160 radius, and the same model gives exactly the committed TEST forecasts.

The committed AgACI comparison streams use a different static width for every seed:

| Region | Seed | PGS242-160 width (MW) | Committed AgACI width (MW) | Difference |
| --- | ---: | ---: | ---: | ---: |
| SA1 | 1 | 868.42 | 856.10 | −1.42% |
| SA1 | 2 | 853.34 | 870.55 | +2.02% |
| SA1 | 3 | 899.73 | 850.05 | −5.52% |
| SA1 | 4 | 903.88 | 868.64 | −3.90% |
| SA1 | 5 | 897.87 | 855.81 | −4.68% |
| NSW1 | 1 | 2765.00 | 2755.38 | −0.35% |
| NSW1 | 2 | 2781.09 | 2810.92 | +1.07% |
| NSW1 | 3 | 2737.34 | 2792.50 | +2.02% |
| NSW1 | 4 | 2871.58 | 2771.79 | −3.48% |
| NSW1 | 5 | 2838.68 | 2839.79 | +0.04% |

**What it means.** The forecasts in those streams come from the committed adaptation run, but the calibration residuals came from a separately refitted model that did not match it. The most likely cause is that the AgACI run used a different software environment (for example another scikit-learn version), so the refitted calibration model differed. The committed manifests don't record package versions, so this can't be confirmed from the repository.

**Is it leakage?** No. The residuals still come from CALIBRATION only, and the perturbation test shows TEST data never reaches them. It is a consistency problem: the conformal radius should be calibrated on the same model whose TEST errors it covers.

**Impact.** The static width differs by −5.5% to +2.0%. ACI and AgACI-BOA start from this width and then adapt online, so their long-run coverage is affected less (in this re-run, seed-1 SA1 / Arm A ACI coverage is 0.8999 vs committed 0.8999; AgACI-BOA 0.9087 vs 0.9100). Static-conformal coverage and width in the final T5 change the most. Escalation tau grids are multiples of this width, so T6 thresholds shift with it.

**Suggested fix (needs a team decision).**

1. Record package versions in every manifest.
2. Either save the calibration residuals once, from the adaptation run's own initial model, and have every UQ runner read them instead of refitting, or regenerate `agaci_boa_fixed_calibration/` (five seeds) in one recorded environment and re-run T5, F4, T6 and F5 from it.
3. Until then, state in the report that T5/T6/F5 use a calibration model that differs slightly from the forecasting model.

## Reproduce

```bash
PYTHONPATH=src python experiments/audit/run_end_to_end_audit.py \
  --work-dir /tmp/pgs242_29 --report-dir results/audit/pgs242_29
```

Options: `--seed`, `--detector`, `--window`, `--cutoff`, `--scale`, `--shift`, `--reuse` (re-run the checks on existing outputs). Exit code is 1 if any check fails. Nothing under `results/` other than `--report-dir` is written.

Environment used: Python 3.13.18, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, statsmodels 0.15.0, river 0.27.0.

## Files

| File | Content |
| --- | --- |
| [AUDIT_REPORT.md](AUDIT_REPORT.md) | Generated: every check with its detail, timings, baseline outputs |
| [audit_report.json](audit_report.json) | The same, machine-readable, with environment versions |
| [end_to_end_adaptation_summary.csv](end_to_end_adaptation_summary.csv) | Baseline MAE and retrain count per region and arm |
| [end_to_end_uncertainty_summary.csv](end_to_end_uncertainty_summary.csv) | Baseline coverage and width per region, arm and UQ method |

## Scope and limits

- One seed (1), one detector (Page-Hinkley), one Arm D window (180 days). Reproducibility of the other seeds and configurations was checked for the committed calibration widths only (table above), not by re-running them.
- SARIMAX and PatchTST are not part of the adaptation pipeline and are not re-run here; their leakage tests are in `tests/degradation/`.
- The T5, T6 and F5 generators were not re-run, because their committed inputs are Git LFS streams; the leakage test covers the functions they rely on.
