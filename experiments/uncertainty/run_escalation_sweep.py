"""Sweep the human-review escalation threshold over PGS242-160 intervals.

PGS242-163

Protocol
--------
- Inputs: persisted PGS242-160 interval streams (no forecaster refit and no
  interval reconstruction), five seeds, SA1 and NSW1, all configurations.
- Score: interval width upper - lower, known before y_t is observed.
  Widths are rounded to WIDTH_DECIMALS so floating-point noise in
  (f + r) - (f - r) cannot split ties at tau.
- Rule: escalate(t) = score(t) > tau (strict); score == tau is retained.
- Tau grid, per (region, uq_method, seed): tau = k * W_cal, where W_cal is
  the calibration-window (2020-01-01 to 2020-02-29) 90% conformal width,
  2 * conformal_radius from the PGS242-160 summary. k = 0.50, 0.55, ...,
  3.00, plus -inf and +inf. PGS242-160 builds intervals for TEST only, so
  W_cal is the only calibration-window width persisted; adaptive conformal
  starts the TEST stream at exactly this width.
- Regimes: T5 full scope (7-day buffers) plus an "all" aggregate.
- Evaluation only: point-forecast MAE over retained and escalated steps.
  The persisted point_forecast never enters the escalation decision.
- Outputs: results/series/escalation_sweep.csv, escalation_by_regime.csv,
  escalation_manifest.json, escalation_summary.md, and idempotent scalar
  rows in results/runs.csv linked by config_hash.

See docs/ESCALATION_SWEEP.md for the output schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from drift_lab.data_loader import TEST_START
from drift_lab.evaluation import (
    AEMO_REGIMES, aemo_timestamp, assign_full_stream_forecast_regimes,
    build_forecast_event_windows, calculate_empirical_coverage,
    calculate_mean_interval_width,
)
from drift_lab.uncertainty import (
    calibration_width_tau_grid, escalation_sweep, interval_width,
)
from generate_t5_uncertainty_table import (
    append_run_metrics_safely, calculate_file_sha256, calculate_json_sha256,
)

TASK = "PGS242-163"
# The protocol id and config_hash describe the escalation policy and inputs.
# Evaluation-only columns (schema version 2 added MAE) do not change them, so
# earlier runs.csv rows stay linked and reruns append only new metrics.
PROTOCOL_ID = "pgs242-163-escalation-sweep-v1"
SCHEMA_VERSION = 2
GROUP = "escalation_sweep"
DATASET = "AEMO_TOTALDEMAND"
SEEDS = (1, 2, 3, 4, 5)
REGIONS = ("SA1", "NSW1")
UQ_METHODS = ("static_conformal", "adaptive_conformal")
REGIMES = (*AEMO_REGIMES, "all")
REGIME_SCOPE = "full"
REGIME_DAYS = 7
WIDTH_DECIMALS = 6
WIDTH_MULTIPLIERS = tuple(step / 100 for step in range(50, 301, 5))
REVIEW_TARGETS = (0.05, 0.10, 0.20)
COVERAGE_TARGET = 0.90
MAE_REVIEW_TARGET = 0.10
INTERVAL_COLUMNS = (
    "SETTLEMENTDATE", "seed", "region", "configuration", "y_true",
    "point_forecast", "static_lower_90", "static_upper_90",
    "adaptive_lower_90", "adaptive_upper_90",
)
SWEEP_COLUMNS = (
    "config_hash", "split_id", "region", "configuration", "regime",
    "uq_method", "seed", "tau", "tau_quantile", "tau_multiplier",
    "calibration_width", "n_total", "n_escalated", "n_retained",
    "review_rate", "retained_coverage", "escalated_coverage",
    "retained_mean_width", "retained_mae", "escalated_mae",
)
GROUP_KEYS = ("config_hash", "split_id", "region", "configuration",
              "uq_method", "seed", "regime")

DEFAULT_RUN_DIR = ROOT / "results/tables/adaptation/revised/pgs242_128_five_seed_final"
DEFAULT_EVENTS = ROOT / "data/events.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results/series"
DEFAULT_RUNS = ROOT / "results/runs.csv"
DEFAULT_T5_MANIFEST = ROOT / "results/tables/uncertainty/t5_aemo/tables/t5_manifest.json"


def target_label(value):
    return f"{round(100 * value):02d}pct"


def metric_names():
    """runs.csv metric names, in logging order."""
    mae_label = target_label(MAE_REVIEW_TARGET)
    return ([f"retained_coverage_at_review_{target_label(t)}" for t in REVIEW_TARGETS]
            + [f"review_rate_for_retained_coverage_{target_label(COVERAGE_TARGET)}",
               f"retained_mae_at_review_{mae_label}",
               f"escalated_mae_at_review_{mae_label}"])


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load_t5_hashes(path):
    """Expected interval sha256 per (seed, region) from the T5 manifest."""
    path = Path(path)
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text())
    return {(int(s["seed"]), s["region"]): s["sha256"] for s in manifest["sources"]}


def load_seed_inputs(run_dir, seed):
    """Load and cross-check the upstream and PGS242-160 manifests and summary."""
    base = Path(run_dir) / f"seed_{seed}"
    paths = [base / "manifest.json",
             base / "uncertainty/adaptive_conformal_manifest.json",
             base / "uncertainty/aemo_conformal_comparison_summary.csv"]
    missing = [relative(p) for p in paths if not p.is_file()]
    if missing:
        return None, f"missing {', '.join(missing)}"
    manifest = json.loads(paths[0].read_text())
    uq = json.loads(paths[1].read_text())
    summary = pd.read_csv(paths[2])
    if manifest["seed"] != seed or uq["seed"] != seed or set(summary.seed) != {seed}:
        raise ValueError(f"Seed mismatch in {relative(base)}.")
    if uq["task"] != "PGS242-160" or uq["alpha"] != 0.10:
        raise ValueError("Expected PGS242-160 nominal 90% intervals.")
    if uq["upstream_run_id"] != manifest["run_id"] or uq["split_id"] != manifest["split_id"]:
        raise ValueError("UQ manifest does not match upstream run.")
    if summary.duplicated(["region", "configuration", "method"]).any():
        raise ValueError("Duplicate UQ summary identities.")
    return {"split_id": uq["split_id"], "summary": summary,
            "summary_path": paths[2]}, None


def load_interval_stream(path, seed, region, configurations, test_size):
    """Read one persisted interval stream and validate its test grid."""
    frame = pd.read_csv(path, usecols=list(INTERVAL_COLUMNS))
    if frame.isna().any().any():
        raise ValueError(f"Missing interval fields in {relative(path)}.")
    if not (frame.seed.eq(seed).all() and frame.region.eq(region).all()):
        raise ValueError(f"Interval stream seed/region mismatch in {relative(path)}.")
    if set(frame.configuration) != set(configurations):
        raise ValueError(f"Interval configurations differ from the UQ summary in {relative(path)}.")
    frame["SETTLEMENTDATE"] = pd.to_datetime(frame.SETTLEMENTDATE)
    frame = frame.sort_values(["configuration", "SETTLEMENTDATE"], kind="mergesort")
    expected = pd.date_range(TEST_START, periods=test_size, freq="30min")
    reference = None
    for configuration, group in frame.groupby("configuration", sort=True):
        if not pd.DatetimeIndex(group.SETTLEMENTDATE).equals(expected):
            raise ValueError(f"{configuration} does not match the TEST grid in {relative(path)}.")
        truth = group.y_true.to_numpy(dtype=float)
        if reference is not None and not np.array_equal(truth, reference):
            raise ValueError("y_true differs across configurations.")
        reference = truth
    return frame.reset_index(drop=True), expected


def assign_regimes(timestamps, events_path, region):
    """T5 full-scope regime labels for the TEST timestamps."""
    windows = build_forecast_event_windows(events_path, region, regime_days=REGIME_DAYS)
    stamps = pd.DatetimeIndex([aemo_timestamp(value) for value in timestamps])
    labels = assign_full_stream_forecast_regimes(stamps, windows).regime.to_numpy()
    if not set(labels) <= set(AEMO_REGIMES):
        raise ValueError("Unexpected regime labels.")
    return labels


def calibration_width(summary, region):
    """2 * calibration conformal radius, shared by every configuration."""
    radii = summary[(summary.region == region) &
                    (summary.method == "static_conformal")].conformal_radius
    radii = radii.to_numpy(dtype=float)
    if not len(radii) or not np.isfinite(radii).all() or len(np.unique(radii)) != 1:
        raise ValueError(f"Expected one finite calibration radius for {region}.")
    return round(2.0 * float(radii[0]), WIDTH_DECIMALS)


def sweep_config_hash(*, upstream_uq_hash, interval_sha256, events_sha256):
    return calculate_json_sha256({
        "protocol": PROTOCOL_ID,
        "upstream_uq_config_hash": upstream_uq_hash,
        "interval_sha256": interval_sha256,
        "events_sha256": events_sha256,
        "regimes": {"scope": REGIME_SCOPE, "regime_days": REGIME_DAYS},
        "score": {"name": "interval_width", "decimals": WIDTH_DECIMALS},
        "rule": "score > tau",
        "tau_grid": {"reference": "2 * calibration conformal_radius",
                     "multipliers": list(WIDTH_MULTIPLIERS),
                     "endpoints": ["-inf", "+inf"]},
    })


def sweep_stream(frame, labels, summary, *, region, seed, split_id,
                 interval_sha256, events_sha256):
    """Sweep every configuration and UQ method of one seed/region stream.

    Returns the sweep rows and one metadata row per configuration/method.
    The tau grid uses only the calibration radius from the summary.
    """
    width_cal = calibration_width(summary, region)
    sweeps, metadata = [], []
    for method in UQ_METHODS:
        prefix = method.split("_")[0]
        grid = calibration_width_tau_grid(width_cal, WIDTH_MULTIPLIERS)
        parts, hashes = [], {}
        for configuration, group in frame.groupby("configuration", sort=True):
            meta = summary[(summary.region == region) &
                           (summary.configuration == configuration) &
                           (summary.method == method)]
            if len(meta) != 1:
                raise ValueError("Expected exactly one UQ summary row.")
            meta = meta.iloc[0]
            truth = group.y_true.to_numpy(dtype=float)
            forecast = group.point_forecast.to_numpy(dtype=float)
            lower = group[f"{prefix}_lower_90"].to_numpy(dtype=float)
            upper = group[f"{prefix}_upper_90"].to_numpy(dtype=float)
            # Both conformal methods are symmetric around the point forecast.
            if not np.allclose((lower + upper) / 2.0, forecast, rtol=0.0, atol=1e-6):
                raise ValueError("point_forecast is not the interval midpoint.")
            for name, value in (
                ("empirical_coverage", calculate_empirical_coverage(truth, lower, upper)),
                ("mean_interval_width", calculate_mean_interval_width(lower, upper)),
            ):
                if not np.isclose(value, meta[name], rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Stream {name} disagrees with the PGS242-160 summary.")
            score = np.round(interval_width(lower, upper), WIDTH_DECIMALS)
            if method == "static_conformal" and not (score == width_cal).all():
                raise ValueError("Static widths differ from 2 * calibration radius.")
            hashes[configuration] = sweep_config_hash(
                upstream_uq_hash=meta.uq_config_hash,
                interval_sha256=interval_sha256, events_sha256=events_sha256)
            part = pd.DataFrame({"configuration": configuration, "regime": labels,
                                 "score": score, "y_true": truth, "y_pred": forecast,
                                 "lower": lower, "upper": upper})
            parts.extend([part, part.assign(regime="all")])
            metadata.append({"config_hash": hashes[configuration], "split_id": split_id,
                             "region": region, "configuration": configuration,
                             "uq_method": method, "seed": seed,
                             "n_retrains": int(meta.n_retrains),
                             "train_samples": int(meta.train_samples_total)})
        sweep = escalation_sweep(pd.concat(parts, ignore_index=True), grid,
                                 ["configuration", "regime"], y_pred_col="y_pred")
        sweep["config_hash"] = sweep.configuration.map(hashes)
        sweeps.append(sweep.assign(split_id=split_id, region=region, uq_method=method,
                                   seed=seed, calibration_width=width_cal))
    return pd.concat(sweeps, ignore_index=True)[list(SWEEP_COLUMNS)], metadata


def order_rows(frame):
    """Deterministic order: region, configuration, method, seed, regime, tau."""
    keys = {"region": REGIONS, "uq_method": UQ_METHODS, "regime": REGIMES}
    ordered = frame.assign(**{f"_{k}": frame[k].map({v: i for i, v in enumerate(values)})
                              for k, values in keys.items()})
    sort = ["_region", "configuration", "_uq_method", "seed", "_regime"]
    sort += ["tau"] if "tau" in frame else []
    ordered = ordered.sort_values(sort, kind="mergesort")
    return ordered.drop(columns=[f"_{k}" for k in keys]).reset_index(drop=True)


def select_operating_points(group):
    """Grid rows nearest each review target, and the cheapest 90% point.

    Ties go to the lower review rate, then the smaller (stricter) tau.
    Rows with equal review rates escalate identical step sets.
    """
    rates = group.review_rate.to_numpy()
    taus = group.tau.to_numpy()
    plus_inf = group[np.isposinf(taus)].iloc[0]
    row = {"n_total": int(group.n_total.iloc[0]),
           "overall_coverage": plus_inf.retained_coverage,
           "mean_width": plus_inf.retained_mean_width,
           "overall_mae": plus_inf.retained_mae,
           "n_distinct_review_rates": int(len(np.unique(rates)))}
    for target in REVIEW_TARGETS:
        order = np.lexsort((taus, rates, np.abs(rates - target)))
        chosen = group.iloc[order[0]]
        label = target_label(target)
        row.update({f"tau_at_review_{label}": chosen.tau,
                    f"tau_quantile_at_review_{label}": chosen.tau_quantile,
                    f"review_rate_at_review_{label}": chosen.review_rate,
                    f"retained_coverage_at_review_{label}": chosen.retained_coverage,
                    f"retained_mae_at_review_{label}": chosen.retained_mae,
                    f"escalated_mae_at_review_{label}": chosen.escalated_mae})
    label = target_label(COVERAGE_TARGET)
    reached = group[group.retained_coverage >= COVERAGE_TARGET]
    if len(reached):
        chosen = reached.iloc[np.lexsort((reached.tau.to_numpy(),
                                          reached.review_rate.to_numpy()))[0]]
        values = (chosen.review_rate, chosen.tau, chosen.tau_quantile, chosen.retained_coverage)
    else:
        values = (np.nan, np.nan, "", np.nan)
    row.update(dict(zip((f"review_rate_for_retained_coverage_{label}",
                         f"tau_for_retained_coverage_{label}",
                         f"tau_quantile_for_retained_coverage_{label}",
                         f"retained_coverage_for_retained_coverage_{label}"), values)))
    return row


def summarise_by_regime(sweep, metadata):
    rows = [{**dict(zip(GROUP_KEYS, key)), **select_operating_points(group)}
            for key, group in sweep.groupby(list(GROUP_KEYS), sort=True)]
    summary = pd.DataFrame(rows).merge(
        pd.DataFrame(metadata), how="left", validate="many_to_one",
        on=["config_hash", "split_id", "region", "configuration", "uq_method", "seed"])
    return order_rows(summary)


def build_run_rows(by_regime):
    """Scalar runs.csv rows in the existing shared schema."""
    rows = []
    for record in by_regime.to_dict("records"):
        for metric in metric_names():
            rows.append({
                "group": GROUP, "method": f"{record['configuration']}|{record['uq_method']}",
                "dataset": DATASET, "region": record["region"], "seed": int(record["seed"]),
                "config_hash": record["config_hash"], "split_id": record["split_id"],
                "metric_name": metric, "metric_value": float(record[metric]),
                "regime": record["regime"], "n_retrains": int(record["n_retrains"]),
                "train_samples": int(record["train_samples"]), "wall_clock_s": 0.0,
            })
    return rows


def mean_sd(values, digits=4):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return "N/A"
    sd = values.std(ddof=1) if len(values) > 1 else float("nan")
    return f"{values.mean():.{digits}f} ± {sd:.{digits}f} (n={len(values)})"


def mae_ratio(frame):
    """Escalated / retained MAE at the ~10% review operating point."""
    label = target_label(MAE_REVIEW_TARGET)
    return (frame[f"escalated_mae_at_review_{label}"]
            / frame[f"retained_mae_at_review_{label}"])


def comparison_table(by_regime, region):
    """Coverage and MAE at ~10% review for A, B and C_* (regime all)."""
    label = target_label(MAE_REVIEW_TARGET)
    subset = by_regime[(by_regime.region == region) & (by_regime.regime == "all") &
                       by_regime.configuration.str.fullmatch(r"A|B|C_.*")]
    lines = ["| Configuration | UQ method | Achieved review rate | Retained coverage | "
             "Retained MAE (MW) | Escalated MAE (MW) | Escalated/retained MAE |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for (configuration, method), group in subset.groupby(
            ["configuration", subset.uq_method.map(UQ_METHODS.index)], sort=True):
        lines.append(f"| {configuration} | {UQ_METHODS[method]} | "
                     f"{mean_sd(group[f'review_rate_at_review_{label}'])} | "
                     f"{mean_sd(group[f'retained_coverage_at_review_{label}'])} | "
                     f"{mean_sd(group[f'retained_mae_at_review_{label}'], 1)} | "
                     f"{mean_sd(group[f'escalated_mae_at_review_{label}'], 1)} | "
                     f"{mean_sd(mae_ratio(group), 3)} |")
    return lines


def adaptive_mae_ratio_lines(by_regime):
    """Per-region adaptive MAE ratio over every configuration and seed."""
    lines = []
    for region in REGIONS:
        subset = by_regime[(by_regime.region == region) & (by_regime.regime == "all") &
                           (by_regime.uq_method == "adaptive_conformal")]
        if len(subset):
            ratio = mae_ratio(subset)
            lines.append(f"- {region}: {mean_sd(ratio, 3)}, "
                         f"range {ratio.min():.3f} to {ratio.max():.3f}, "
                         f"{int((ratio > 1).sum())} of {len(ratio)} groups above 1.")
    return lines


def arm_a_table(by_regime, region):
    subset = by_regime[(by_regime.region == region) & (by_regime.configuration == "A")]
    columns = ["overall_coverage", *[f"retained_coverage_at_review_{target_label(t)}"
                                     for t in REVIEW_TARGETS],
               f"review_rate_for_retained_coverage_{target_label(COVERAGE_TARGET)}"]
    lines = ["| UQ method | Regime | " + " | ".join(columns) + " |",
             "| --- | --- | " + " | ".join(["---"] * len(columns)) + " |"]
    for method in UQ_METHODS:
        for regime in REGIMES:
            group = subset[(subset.uq_method == method) & (subset.regime == regime)]
            if len(group):
                lines.append(f"| {method} | {regime} | "
                             + " | ".join(mean_sd(group[c]) for c in columns) + " |")
    return lines


def write_summary(path, sweep, by_regime, processed, skipped):
    static = sweep[sweep.uq_method == "static_conformal"]
    static_rates = sorted(set(static.review_rate.tolist()))
    adaptive = by_regime[by_regime.uq_method == "adaptive_conformal"]
    lines = [
        f"# {TASK} escalation threshold sweep", "",
        "Generated by experiments/uncertainty/run_escalation_sweep.py; do not edit by hand.",
        "Values are seed mean ± sample SD (ddof=1) over the seeds processed.", "",
        "## Protocol", "",
        f"- Score: interval width (upper - lower), rounded to {WIDTH_DECIMALS} decimals (MW).",
        "- Rule: escalate when score > tau (strict); a step with score == tau is retained.",
        "- Tau grid per (region, UQ method, seed): tau = k × W_cal, W_cal = 2 × calibration "
        "conformal radius (2020-01-01 to 2020-02-29), "
        f"k = {WIDTH_MULTIPLIERS[0]:.2f}, {WIDTH_MULTIPLIERS[1]:.2f}, ..., "
        f"{WIDTH_MULTIPLIERS[-1]:.2f}, plus -inf and +inf.",
        f"- Regimes: T5 {REGIME_SCOPE} scope ({REGIME_DAYS}-day buffers) plus 'all'.",
        f"- Operating points: grid row whose review rate is closest to each target "
        "(ties: lower review rate, then smaller tau); the 90% point is the lowest "
        "review rate with retained coverage >= 0.90 (NaN if never reached).",
        f"- Streams processed: {len(processed)} seed/region interval files.", "",
        "## Static conformal collapses to review rate {0, 1}", "",
        f"Distinct static review rates over all rows: {static_rates}.",
        "Static split-conformal widths are constant (2 × calibration radius), so every "
        "tau either escalates all steps or none. Width-based escalation is informative "
        "only for adaptive conformal.",
        f"Adaptive groups with more than two distinct review rates: "
        f"{int((adaptive.n_distinct_review_rates > 2).sum())} of {len(adaptive)}.", "",
        "## Coverage and MAE at the ~10% review operating point: A vs B vs C_*", "",
        "Regime 'all'. Static conformal cannot reach 10%; its nearest grid point is 0%, "
        "so it has no escalated steps and its escalated MAE is N/A. MAE is "
        "mean |y_true - point_forecast| (evaluation only; it does not affect escalation).", "",
    ]
    for region in REGIONS:
        if (by_regime.region == region).any():
            lines += [f"### {region}", "", *comparison_table(by_regime, region), ""]
    lines += ["### Adaptive conformal escalated/retained MAE ratio", "",
              "Regime 'all', ~10% review, every configuration and seed:", "",
              *adaptive_mae_ratio_lines(by_regime), ""]
    lines += ["## Arm A operating points", ""]
    for region in REGIONS:
        if (by_regime.region == region).any():
            lines += [f"### {region}", "", *arm_a_table(by_regime, region), ""]
    lines += ["## Skipped inputs", ""]
    lines += ([f"- seed {s['seed']}, {s['region']}: {s['reason']}" for s in skipped]
              or ["None."])
    Path(path).write_text("\n".join(lines) + "\n")


def run_sweep(*, run_dir, events_path, output_dir, runs_path, t5_manifest,
              seeds=SEEDS, regions=REGIONS, write_runs=True):
    """Validate inputs, compute every sweep, log scalars, then write series."""
    expected_hashes = load_t5_hashes(t5_manifest)
    events_sha256 = calculate_file_sha256(events_path)
    sweeps, metadata, processed, skipped = [], [], [], []
    split_ids = set()
    for seed in seeds:
        inputs, reason = load_seed_inputs(run_dir, seed)
        if inputs is None:
            skipped.extend({"seed": seed, "region": r, "reason": reason} for r in regions)
            continue
        summary = inputs["summary"]
        for region in regions:
            path = Path(run_dir) / f"seed_{seed}/uncertainty/aemo_conformal_intervals_{region}.csv.gz"
            if not path.is_file():
                skipped.append({"seed": seed, "region": region,
                                "reason": f"missing {relative(path)}"})
                continue
            sha256 = calculate_file_sha256(path)
            expected = expected_hashes.get((seed, region))
            if expected is not None and expected != sha256:
                skipped.append({"seed": seed, "region": region,
                                "reason": "sha256 differs from the T5 manifest"})
                continue
            region_summary = summary[summary.region == region]
            test_sizes = set(region_summary.test_size.astype(int))
            if len(test_sizes) != 1:
                raise ValueError("Expected one test_size per region.")
            frame, timestamps = load_interval_stream(
                path, seed, region, set(region_summary.configuration), test_sizes.pop())
            labels = assign_regimes(timestamps, events_path, region)
            sweep, meta = sweep_stream(
                frame, labels, summary, region=region, seed=seed,
                split_id=inputs["split_id"], interval_sha256=sha256,
                events_sha256=events_sha256)
            sweeps.append(sweep)
            metadata.extend(meta)
            split_ids.add(inputs["split_id"])
            processed.append({"seed": seed, "region": region, "path": relative(path),
                              "sha256": sha256, "t5_manifest_sha256": expected,
                              "verified_against_t5_manifest": expected == sha256,
                              "summary_sha256": calculate_file_sha256(inputs["summary_path"])})
            print(f"seed={seed} region={region}: {len(sweep)} sweep rows")
    if not sweeps:
        raise SystemExit("No PGS242-160 interval streams available; nothing written.\n"
                         + "\n".join(f"seed {s['seed']} {s['region']}: {s['reason']}"
                                     for s in skipped))
    if len(split_ids) != 1:
        raise ValueError("Processed streams do not share one split_id.")

    sweep = order_rows(pd.concat(sweeps, ignore_index=True))
    by_regime = summarise_by_regime(sweep, metadata)
    run_rows = build_run_rows(by_regime)
    # All validation is complete before any writes.
    if write_runs:
        append_run_metrics_safely(run_rows, runs_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "escalation_sweep.csv"
    by_regime_path = output_dir / "escalation_by_regime.csv"
    sweep.to_csv(sweep_path, index=False, lineterminator="\n")
    by_regime.to_csv(by_regime_path, index=False, lineterminator="\n")
    write_summary(output_dir / "escalation_summary.md", sweep, by_regime, processed, skipped)
    manifest = {
        "task": TASK, "protocol": PROTOCOL_ID, "schema_version": SCHEMA_VERSION,
        "split_id": split_ids.pop(),
        "score": {"name": "interval_width", "definition": "upper - lower",
                  "decimals": WIDTH_DECIMALS},
        "rule": "escalate when score > tau; score == tau is retained",
        "tau_grid": {"definition": "tau = k * calibration_width",
                     "calibration_width": "2 * PGS242-160 static conformal_radius "
                                          "(calibration window 2020-01-01 to 2020-02-29)",
                     "multipliers": list(WIDTH_MULTIPLIERS), "endpoints": ["-inf", "+inf"],
                     "order": "strict -> loose (tau ascending)"},
        "regimes": {"scope": REGIME_SCOPE, "regime_days": REGIME_DAYS,
                    "labels": list(REGIMES), "source": "T5 full-scope assignment"},
        "review_targets": list(REVIEW_TARGETS), "coverage_target": COVERAGE_TARGET,
        "evaluation_only": {"columns": ["retained_mae", "escalated_mae"],
                            "definition": "mean |y_true - point_forecast| over retained "
                                          "or escalated steps; NaN when empty",
                            "affects_escalation": False},
        "events": {"path": relative(events_path), "sha256": events_sha256},
        "inputs": processed, "skipped": skipped,
        "runs": {"path": relative(runs_path), "group": GROUP, "written": write_runs,
                 "metrics": metric_names(), "n_rows": len(run_rows)},
        "outputs": {path.name: calculate_file_sha256(path)
                    for path in (sweep_path, by_regime_path)},
    }
    (output_dir / "escalation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return sweep, by_regime, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--t5-manifest", type=Path, default=DEFAULT_T5_MANIFEST,
                        help="T5 manifest whose interval sha256 values are verified.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--regions", nargs="+", choices=REGIONS, default=list(REGIONS))
    parser.add_argument("--skip-runs-log", action="store_true",
                        help="Write series outputs without appending to runs.csv.")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.regions)) != len(args.regions):
        raise ValueError("--seeds and --regions must be unique.")
    _, _, skipped = run_sweep(
        run_dir=args.run_dir, events_path=args.events, output_dir=args.output_dir,
        runs_path=args.runs, t5_manifest=args.t5_manifest, seeds=tuple(args.seeds),
        regions=tuple(args.regions), write_runs=not args.skip_runs_log)
    for item in skipped:
        print(f"SKIPPED seed {item['seed']} {item['region']}: {item['reason']}")


if __name__ == "__main__":
    main()
