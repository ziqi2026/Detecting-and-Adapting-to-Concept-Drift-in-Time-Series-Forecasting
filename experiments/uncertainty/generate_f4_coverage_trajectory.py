"""F4: trailing coverage from stored PGS242-160 intervals. No model fitting.

Uses a separately exported T5 review snapshot for regimes and metric checks.
Configurations and regions are never pooled; no shared team files are modified.
"""

from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import textwrap
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from drift_lab.evaluation import (
    calculate_empirical_coverage, calculate_mean_interval_width,
    build_forecast_event_windows, assign_forecast_regimes,
)

METHODS = ("static", "adaptive")
COLORS = {"static": "#3564ae", "adaptive": "#c56b25"}


def f1_window(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value.value
    days = values.get("ROLLING_WINDOW_DAYS")
    daily = values.get("OBSERVATIONS_PER_DAY")
    if not isinstance(days, int) or not isinstance(daily, int) or min(days, daily) < 1:
        raise ValueError("Cannot read positive integer F1 rolling-window constants.")
    return days, daily, days * daily


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


T5_METRICS = ("empirical_coverage", "mean_interval_width",
              "normalised_interval_width", "worst_24h_coverage")
T5_COLUMNS = ("coverage", "mean width", "normalised width", "worst 24h coverage")
REGIMES = ("pre-drift", "drift", "post-drift")


def load_t5_reference(root):
    """Read the explicit review copy, without replacing shared evaluation.py."""
    base = root / "results/tables/uncertainty/t5_aemo"
    if sha256(root / "src/drift_lab/evaluation.py") != "a17bb869935154994f0db324e0c78ffd4dce71e6febcf8cdab0486540ccea290":
        raise ValueError("T5 evaluation.py differs from the inspected PR38 b3beabc snapshot; review the regime definition before updating F4.")
    spec = importlib.util.spec_from_file_location("f4_t5_shared_evaluation", root / "src/drift_lab/evaluation.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    protocol = json.loads((base / "protocol.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((base / "tables/t5_manifest.json").read_text(encoding="utf-8-sig"))
    if manifest["protocol"] != protocol:
        raise ValueError("T5 protocol and stored manifest disagree.")
    if (protocol["regime_scope"], protocol["rolling_boundary"], protocol["cadence"],
        protocol["rolling_closed"], protocol["incomplete_windows"]) != (
        "full", "within_regime", "30min", "right", "exclude"):
        raise ValueError("This F4 integration requires full regimes and complete within-regime 24h windows.")
    support = protocol["minimum_complete_24h_windows"]
    if isinstance(support, bool) or not isinstance(support, int) or support < 1:
        raise ValueError("Invalid T5 minimum window support.")
    metrics = pd.read_csv(base / "tables/t5_seed_metrics.csv")
    runs = pd.read_csv(base / "runs.csv")
    keys = ["group", "method", "dataset", "region", "seed", "config_hash",
            "split_id", "regime", "metric_name"]
    for frame in (metrics, runs):
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate T5 metric/log keys.")
    pd.testing.assert_frame_equal(metrics.sort_values(keys).reset_index(drop=True),
                                  runs.sort_values(keys).reset_index(drop=True))
    audit = pd.read_csv(base / "tables/t5_window_audit.csv")
    if audit.duplicated(["region", "seed", "method", "regime"]).any():
        raise ValueError("Duplicate T5 audit keys.")
    return evaluation, protocol, manifest, metrics, audit, base


def t5_regime_metrics(frame, assignments, method, evaluation, protocol, region):
    """Recompute T5 metrics from interval bounds, independently of the 7d curve."""
    actual, lower, upper = frame.y_true, frame[f"{method}_lower_90"], frame[f"{method}_upper_90"]
    rolling = evaluation.calculate_rolling_24h_coverage(
        frame.SETTLEMENTDATE, actual, lower, upper, cadence=protocol["cadence"],
        regimes=assignments.regime, boundary_policy=protocol["rolling_boundary"]).to_numpy()
    rows = []
    for regime in REGIMES:
        mask = assignments.regime.eq(regime).to_numpy()
        valid = rolling[mask & np.isfinite(rolling)]
        n = int(mask.sum())
        values = dict.fromkeys(T5_METRICS, float("nan"))
        if n:
            values.update(
                empirical_coverage=evaluation.calculate_empirical_coverage(actual[mask], lower[mask], upper[mask]),
                mean_interval_width=evaluation.calculate_mean_interval_width(lower[mask], upper[mask]),
                normalised_interval_width=evaluation.calculate_normalised_interval_width(
                    lower[mask], upper[mask], scale=protocol["normalisation"][region]["value"]),
                worst_24h_coverage=float(valid.min()) if len(valid) >= protocol["minimum_complete_24h_windows"] else float("nan"))
        rows.append(dict(regime=regime, n_observations=n, n_eligible_24h_windows=len(valid), **values))
    return rows


def check_t5_rows(recomputed, stored, audit, region, seed, config, method):
    records = []
    identity = f"{config}|{method}_conformal"
    selected = stored[(stored.region == region) & (stored.seed == seed) & (stored.method == identity)]
    audit = audit[(audit.region == region) & (audit.seed == seed) & (audit.method == identity)]
    for row in recomputed:
        for column in ("n_observations", "n_eligible_24h_windows"):
            expected = audit.loc[audit.regime.eq(row["regime"]), column]
            if len(expected) != 1 or int(expected.iloc[0]) != row[column]:
                raise ValueError(f"T5 audit mismatch: {region}/{seed}/{identity}/{row['regime']}/{column}")
        for metric in T5_METRICS:
            expected = selected.loc[selected.regime.eq(row["regime"]) & selected.metric_name.eq(metric), "metric_value"]
            if len(expected) != 1:
                raise ValueError("Missing or ambiguous T5 reference metric.")
            expected = float(expected.iloc[0])
            np.testing.assert_allclose(row[metric], expected, rtol=0, atol=1e-8, equal_nan=True,
                err_msg=f"{region}/{seed}/{identity}/{row['regime']}/{metric}")
            records.append(dict(region=region, seed=seed, configuration=config,
                method=f"{method}_conformal", regime=row["regime"], metric=metric,
                stored=expected, recomputed=row[metric], passed=True))
    return records


def check_t5_tables(rows, base, seeds):
    records = []
    if set(seeds) != {1, 2, 3, 4, 5}:
        return records
    for (region, config, method, regime), group in rows.groupby(
            ["region", "configuration", "method", "regime"]):
        table = pd.read_csv(base / "tables" / f"T5_{region}_{config}.csv", keep_default_na=False)
        ref = table[table["UQ method"].eq(method) & table.regime.eq(regime)]
        if len(ref) != 1 or len(group) != 5 or set(group.seed) != {1, 2, 3, 4, 5}:
            raise ValueError("T5 table aggregation is missing a method/regime or seed.")
        for metric, column in zip(T5_METRICS, T5_COLUMNS):
            v = group[metric].to_numpy(dtype=float)
            formatted = "N/A" if not np.isfinite(v).all() else f"{v.mean():.8f} ± {v.std(ddof=1):.8f}"
            if str(ref.iloc[0][column]) != formatted:
                raise ValueError(f"T5 displayed cell mismatch: {region}/{config}/{method}/{regime}/{column}")
            records.append(dict(region=region, configuration=config, method=method,
                                regime=regime, metric=metric, value=formatted, passed=True))
    return records


def boolean_values(series):
    mapped = series.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"Invalid coverage flags in {series.name}.")
    return mapped.to_numpy(dtype=bool)


def check_stream(frame, window, daily):
    # Require chronological stored order within one seed/region/configuration.
    if not frame.SETTLEMENTDATE.is_monotonic_increasing:
        raise ValueError("TEST observations must already be chronological.")
    frame = frame.reset_index(drop=True)
    timestamps = frame["SETTLEMENTDATE"]
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("Missing or duplicate timestamps.")
    expected_step = pd.Timedelta(days=1) / daily
    if not timestamps.diff().dropna().eq(expected_step).all():
        raise ValueError("Irregular TEST spacing: observation window would not equal F1 days.")
    if len(frame) < window:
        raise ValueError("TEST stream is shorter than the rolling window.")
    rolling = pd.DataFrame({"SETTLEMENTDATE": timestamps})
    metrics = {}
    for method in METHODS:
        lower = frame[f"{method}_lower_90"].to_numpy(dtype=float)
        upper = frame[f"{method}_upper_90"].to_numpy(dtype=float)
        actual = frame["y_true"].to_numpy(dtype=float)
        if not np.isfinite(np.column_stack([actual, lower, upper])).all():
            raise ValueError("Non-finite actual values or interval endpoints.")
        if np.any(lower > upper):
            raise ValueError("Reversed interval endpoints.")
        covered = (lower <= actual) & (actual <= upper)
        if not np.array_equal(covered, boolean_values(frame[f"{method}_covered_90"])):
            raise ValueError("Stored coverage flags disagree with inclusive interval endpoints.")
        trajectory = pd.Series(covered.astype(float)).rolling(
            window=window, min_periods=window, center=False
        ).mean()
        if not trajectory.iloc[:window - 1].isna().all():
            raise AssertionError("Unexpected partial-window estimates.")
        if not trajectory.dropna().between(0, 1).all():
            raise AssertionError("Rolling coverage outside [0,1].")
        # Check right-aligned windows against direct sums, including boundaries.
        positions = np.unique(np.linspace(window - 1, len(frame) - 1, 25, dtype=int))
        for position in positions:
            expected = covered[position - window + 1:position + 1].mean()
            np.testing.assert_allclose(trajectory.iloc[position], expected, rtol=0, atol=1e-12)
        # Removing all future observations must preserve the prefix trajectory.
        cutoff = min(len(frame), window + 1000)
        prefix = pd.Series(covered[:cutoff].astype(float)).rolling(
            window, min_periods=window, center=False
        ).mean()
        np.testing.assert_allclose(trajectory.iloc[:cutoff], prefix, rtol=0, atol=1e-12, equal_nan=True)
        rolling[f"{method}_coverage"] = trajectory
        metrics[method] = {
            "test_size": len(frame),
            "empirical_coverage": calculate_empirical_coverage(actual, lower, upper),
            "mean_interval_width": calculate_mean_interval_width(lower, upper),
            "minimum_rolling_coverage": float(trajectory.min()),
            "maximum_rolling_coverage": float(trajectory.max()),
        }
    return frame, rolling, metrics

def aggregate_trajectories(streams, seeds):
    """Align complete streams, then take equal-weight mean/sample SD by seed."""
    first = streams[0]["SETTLEMENTDATE"]
    if len(streams) != len(seeds):
        raise ValueError("Missing seed trajectory.")
    for frame in streams[1:]:
        if not frame["SETTLEMENTDATE"].equals(first):
            raise ValueError("TEST timestamps differ across seeds; cannot average.")
    output = pd.DataFrame({"SETTLEMENTDATE": first})
    for method in METHODS:
        values = pd.concat([f[f"{method}_coverage"] for f in streams], axis=1)
        if not values.isna().eq(values.iloc[:, 0].isna(), axis=0).all().all():
            raise ValueError("Rolling-window availability differs across seeds.")
        output[f"{method}_mean"] = values.mean(axis=1)
        output[f"{method}_sd"] = values.std(axis=1, ddof=1)
        output[f"{method}_n_seeds"] = values.notna().sum(axis=1)
    return output


def plot_trajectory(frame, windows, region, config, seeds, days, window, output):
    fig, ax = plt.subplots(figsize=(13, 5.8))
    start, end = frame.SETTLEMENTDATE.iloc[[0, -1]]
    for i, event in enumerate(windows.to_dict("records")):
        if event["post_drift_end"] <= start or event["pre_drift_start"] > end:
            continue
        ax.axvspan(max(start, event["pre_drift_start"]), min(end, event["post_drift_end"]),
                   color="#766786", alpha=0.16 if event["tier"] == 1 else 0.06,
                   linewidth=0, zorder=0)
    for method in METHODS:
        mean, sd = frame[f"{method}_mean"], frame[f"{method}_sd"]
        ax.plot(frame.SETTLEMENTDATE, mean, color=COLORS[method], linewidth=1,
                label=f"{method.title()} conformal")
        if len(seeds) > 1:
            ax.fill_between(frame.SETTLEMENTDATE, (mean-sd).clip(0, 1),
                            (mean+sd).clip(0, 1), color=COLORS[method], alpha=0.17)
    ax.axhline(.9, color="#222222", ls="--", lw=1.7,
               label="Target coverage: 0.90", zorder=5)
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#766786", alpha=.16))
    labels.append("T5 buffered event periods (provisional)")
    ax.legend(handles, labels, loc="lower left", ncol=2, fontsize=9)
    ax.set_ylim(.0, 1.02)
    ax.set_ylabel(f"{days}-day rolling empirical coverage")
    ax.set_xlabel("TEST date (AEMO fixed UTC+10)")
    tag = "PREVIEW | " if len(seeds) < 5 else ""
    ax.set_title(f"{tag}F4 Coverage Trajectory: {region} | {config}\n"
                 f"Seeds {', '.join(map(str, seeds))}", loc="left", fontsize=13)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    caption = (f"F4. Trailing {days}-day coverage ({window} half-hour observations; same as F1). "
               f"Each seed uses the complete chronological TEST stream; first {window-1} values "
               "are omitted. Lines show the seed mean; bands show +/- sample SD, clipped to [0,1] "
               "for display only. Shading uses the exported T5 full-stream buffered event definition "
               "(7 days before/after; darker: Tier 1); events are contextual, not proven drift. "
               "The 7-day curve continues across regime boundaries. Companion metrics use T5 complete "
               "within-regime 24h windows. Values are checked against the T5 review snapshot; "
               "the shared regime decision remains provisional.")
    if len(seeds) == 1:
        caption = caption.replace("Lines show the seed mean; bands show +/- sample SD, clipped to [0,1] "
                                  "for display only.", "Single-seed preview; sample SD is unavailable.")
    fig.text(.08, .025, textwrap.fill(caption, 150), fontsize=9, va="bottom")
    fig.subplots_adjust(left=.08, right=.98, top=.84, bottom=.27)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=PROJECT_ROOT / "data/events.csv")
    parser.add_argument("--t3-manifest", type=Path)
    parser.add_argument("--f1-script", type=Path, default=PROJECT_ROOT /
                        "experiments/degradation/generate_f1_degradation_figure.py")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--regions", nargs="+", choices=["SA1", "NSW1"], default=["SA1", "NSW1"])
    parser.add_argument("--configurations", nargs="+")
    parser.add_argument("--allow-preview", action="store_true")
    parser.add_argument("--t5-review-dir", type=Path, default=PROJECT_ROOT / "T5_PR38_Review")
    args = parser.parse_args()
    ev, t5_protocol, t5_manifest, t5_stored, t5_audit, t5_base = load_t5_reference(args.t5_review_dir)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds.")
    if set(args.seeds) != {1, 2, 3, 4, 5} and not args.allow_preview:
        parser.error("Reporting requires seeds 1-5; use --allow-preview for development only.")
    manifest_path = args.t3_manifest or args.run_dir / "t3/t3_manifest.json"
    protocol = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    days, daily, window = f1_window(args.f1_script)
    if protocol["rolling_window"] != window or not np.isclose(protocol["alpha"], .1):
        raise ValueError("F1/T3 window or nominal alpha mismatch.")
    configs = args.configurations or list(protocol["primary_configurations"].values())
    if len(set(configs)) != len(configs):
        raise ValueError("Duplicate configurations.")
    if not set(configs) <= set(t5_protocol["configurations"]):
        raise ValueError("Selected F4 configurations are absent from T5.")
    if daily != 48 or t5_protocol["regime_days"] != 7:
        raise ValueError("Cadence or regime span differs from the T5 snapshot.")
    if sha256(args.events) != t5_manifest["events_sha256"]:
        raise ValueError("F4 and T5 event files differ.")
    events = pd.read_csv(args.events)
    out = args.output_dir or args.run_dir / "f4_t5_review"
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "validation_in_progress", "seeds": args.seeds,
              "regions": args.regions, "configurations": configs,
              "rolling_window_observations": window, "rolling_window_days": days,
              "min_periods": window, "center": False, "sd_ddof": 1,
              "regime_definition": "T5 full-stream; drift=[event_start-7d,event_end+1d+7d); provisional review snapshot",
              "regime_days": protocol["regime_days"],
              "t5_validation": "in_progress",
              "runs_csv_validation": "T5 bundled runs.csv checked against seed metrics; canonical results/runs.csv not in review export",
              "sources": {str(p): sha256(p) for p in
                          [args.events, manifest_path, args.f1_script, Path(__file__),
                           PROJECT_ROOT / "src/drift_lab/evaluation.py"]}, "streams": []}
    report["t5_protocol"] = t5_protocol
    report["t5_reference_commit"] = "b3beabc72b01e1a497e2764ec967a34080c45425 (user-exported review snapshot)"
    report["regime_approval"] = "provisional; supervisor/team confirmation not asserted"
    for path in [args.t5_review_dir / "src/drift_lab/evaluation.py", t5_base / "protocol.json",
                 t5_base / "runs.csv", t5_base / "tables/t5_seed_metrics.csv",
                 t5_base / "tables/t5_window_audit.csv", t5_base / "tables/t5_manifest.json"]:
        report["sources"][str(path)] = sha256(path)
    metric_rows, comparison_rows, captions, t5_rows, t5_checks = [], [], [], [], []
    reference_manifest = None
    for region in args.regions:
        windows = ev.build_forecast_event_windows(events, region, t5_protocol["regime_days"])
        windows.to_csv(out / f"F4_{region}_event_windows.csv", index=False)
        assignments = None
        trajectories = {config: [] for config in configs}
        for seed in args.seeds:
            uq = args.run_dir / f"seed_{seed}/uncertainty"
            meta_path = uq / "adaptive_conformal_manifest.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            signature = (meta["alpha"], meta["gamma"], meta["split_id"])
            if reference_manifest is None:
                reference_manifest = signature
            if signature != reference_manifest or meta["seed"] != seed or signature[0] != .1:
                raise ValueError("Inconsistent seed UQ settings/split.")
            summary_path = uq / "aemo_conformal_comparison_summary.csv"
            summary = pd.read_csv(summary_path)
            keys = ["region", "seed", "configuration", "method"]
            if summary.duplicated(keys).any():
                raise ValueError("Duplicate summary keys.")
            summary = summary.set_index(keys)
            interval_path = uq / f"aemo_conformal_intervals_{region}.csv.gz"
            for p in [meta_path, summary_path, interval_path]:
                report["sources"][str(p)] = sha256(p)
            references = [r for r in t5_manifest["sources"] if r["seed"] == seed and r["region"] == region]
            if len(references) != 1 or sha256(interval_path) != references[0]["sha256"]:
                raise ValueError("Interval bytes differ from the T5 snapshot source.")
            ref_meta = references[0]["uq_manifest"]
            if any(meta[k] != ref_meta[k] for k in ("alpha", "gamma", "split_id", "seed")):
                raise ValueError("F4 uncertainty settings differ from T5.")
            # Keep only required configurations; do not average the 17 forecast variants.
            parts = []
            for chunk in pd.read_csv(interval_path, chunksize=100000):
                parts.append(chunk.loc[chunk.configuration.isin(configs)])
            data = pd.concat(parts, ignore_index=True)
            if set(data.region) != {region} or set(data.seed) != {seed}:
                raise ValueError("Interval region/seed mismatch.")
            data["SETTLEMENTDATE"] = pd.to_datetime(data.SETTLEMENTDATE, errors="raise")
            if data.SETTLEMENTDATE.dt.tz is None:
                data["SETTLEMENTDATE"] = data.SETTLEMENTDATE.dt.tz_localize("Etc/GMT-10")
            else:
                data["SETTLEMENTDATE"] = data.SETTLEMENTDATE.dt.tz_convert("Etc/GMT-10")
            for config in configs:
                selected = data.loc[data.configuration.eq(config)].copy()
                if selected.empty or not selected.SETTLEMENTDATE.is_monotonic_increasing:
                    raise ValueError("Missing configuration or nonchronological stored stream.")
                frame, rolling, metrics = check_stream(selected, window, daily)
                endpoints = frame.SETTLEMENTDATE.iloc[[0, -1]]
                if list(endpoints) != [ev.aemo_timestamp(t5_protocol[k]) for k in ("test_start", "test_end")]:
                    raise ValueError("F4 TEST bounds differ from T5.")
                if assignments is None:
                    print(f"Assigning shared T3 regimes for {region}...", flush=True)
                    assignments = ev.assign_full_stream_forecast_regimes(frame.SETTLEMENTDATE, windows)
                    if not set(assignments.regime) <= set(REGIMES):
                        raise ValueError("Full-stream regimes contain unassigned labels.")
                    assignments.to_csv(out / f"F4_{region}_regime_assignment.csv.gz", index=False,
                                       compression={"method": "gzip", "mtime": 0})
                if not assignments.SETTLEMENTDATE.equals(frame.SETTLEMENTDATE):
                    raise ValueError("Timestamps differ across seeds/configurations.")
                rolling["regime"] = assignments.regime
                rolling["event_id"] = assignments.event_id
                rolling.to_csv(out / f"F4_{region}_seed{seed}_{config}_rolling.csv.gz", index=False,
                               compression={"method": "gzip", "mtime": 0})
                trajectories[config].append(rolling)
                for method in METHODS:
                    ref = summary.loc[(region, seed, config, f"{method}_conformal")]
                    for metric in ["test_size", "empirical_coverage", "mean_interval_width"]:
                        expected, actual = float(ref[metric]), metrics[method][metric]
                        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
                        comparison_rows.append(dict(region=region, seed=seed, configuration=config,
                                                    method=f"{method}_conformal", metric=metric,
                                                    stored=expected, recomputed=actual, passed=True))
                    recomputed = t5_regime_metrics(frame, assignments, method, ev, t5_protocol, region)
                    t5_checks.extend(check_t5_rows(recomputed, t5_stored, t5_audit, region, seed, config, method))
                    t5_rows.extend(dict(region=region, seed=seed, configuration=config,
                                        method=f"{method}_conformal", **row) for row in recomputed)
                    for regime in ["overall", *REGIMES]:
                        mask = np.ones(len(frame), dtype=bool) if regime == "overall" else assignments.regime.eq(regime)
                        subset = frame.loc[mask].reset_index(drop=True)
                        n = len(subset)
                        coverage = calculate_empirical_coverage(subset.y_true, subset[f"{method}_lower_90"],
                                    subset[f"{method}_upper_90"]) if n else np.nan
                        width = calculate_mean_interval_width(subset[f"{method}_lower_90"],
                                    subset[f"{method}_upper_90"]) if n else np.nan
                        metric_rows.append(dict(region=region, seed=seed, configuration=config,
                                                method=f"{method}_conformal", regime=regime,
                                                n_observations=n, coverage=coverage, mean_interval_width=width))
                report["streams"].append(dict(region=region, seed=seed, configuration=config,
                                               chronology="passed", coverage_flags="passed", bounds="passed",
                                               trailing_window="passed", prefix_invariance="passed",
                                               upstream_summary="passed"))
                print(f"PASS {region} seed {seed} {config}", flush=True)
        for config, streams in trajectories.items():
            agg = aggregate_trajectories(streams, args.seeds)
            stem = f"F4_{region}_{config}"
            agg.to_csv(out / f"{stem}_mean_sd.csv.gz", index=False,
                       compression={"method": "gzip", "mtime": 0})
            captions.append(plot_trajectory(agg, windows, region, config, args.seeds, days, window,
                                            out / f"{stem}.png"))
    t5_rows = pd.DataFrame(t5_rows)
    table_checks = check_t5_tables(t5_rows, t5_base, args.seeds)
    t5_rows.to_csv(out / "F4_t5_recomputed_metrics.csv", index=False)
    pd.DataFrame(t5_checks).to_csv(out / "F4_t5_seed_validation.csv", index=False)
    pd.DataFrame(table_checks).to_csv(out / "F4_t5_table_validation.csv", index=False)
    for region in args.regions:
        for config in configs:
            path = t5_base / "tables" / f"T5_{region}_{config}.csv"
            report["sources"][str(path)] = sha256(path)
    report["status"] = "passed_against_t5_review_snapshot" if table_checks else "preview_passed_against_t5_review_snapshot"
    report["t5_validation"] = {"seed_metric_checks": len(t5_checks), "table_cells": len(table_checks),
        "source_interval_hashes": "passed", "bundled_t5_log": "passed",
        "scope": "selected F4 configurations only; no claim of final regime approval"}
    seed_metrics = pd.DataFrame(metric_rows)
    seed_metrics.to_csv(out / "F4_coverage_width_by_seed_regime.csv", index=False)
    seed_metrics.groupby(["region", "configuration", "method", "regime"], sort=True).agg(
        coverage_mean=("coverage", "mean"), coverage_sd=("coverage", "std"),
        mean_width_mean=("mean_interval_width", "mean"), mean_width_sd=("mean_interval_width", "std"),
        n_seeds=("seed", "nunique")).reset_index().to_csv(out / "F4_coverage_width_mean_sd.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(out / "F4_upstream_summary_validation.csv", index=False)
    (out / "F4_captions.txt").write_text("\n\n".join(captions)+"\n", encoding="utf-8")
    (out / "F4_validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"F4 generation complete: {out}\nPASS against exported T5 review snapshot: {len(t5_checks)} seed metrics, {len(table_checks)} table cells.\nRegime definition remains provisional; canonical root runs.csv was not in this export.")


if __name__ == "__main__":
    main()
