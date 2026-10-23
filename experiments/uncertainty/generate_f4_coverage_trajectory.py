"""F4: trailing coverage from stored PGS242-160 intervals. No model fitting.

Uses current T5 AgACI comparison interval streams and the shared evaluation code.
Configurations and regions are never pooled; no shared team files are modified.
"""

from pathlib import Path
import argparse
import ast
import hashlib
import json
import textwrap
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import drift_lab.evaluation as ev
from drift_lab.evaluation import (
    calculate_empirical_coverage, calculate_mean_interval_width,
    calculate_rolling_24h_coverage, calculate_normalised_interval_width,
    build_forecast_event_windows, assign_full_stream_forecast_regimes,
)

METHODS = ("static", "aci", "agaci")
COLORS = {"static": "#4a647f", "aci": "#c47f43", "agaci": "#248174"}
LABELS = {"static": "Static conformal", "aci": "Adaptive conformal (ACI)",
          "agaci": "AgACI-BOA"}
# Columns match run_aemo_agaci.py's interval comparison export.
INTERVAL_COLUMNS = ("SETTLEMENTDATE", "region", "seed", "configuration", "y_true",
                    "point_forecast", *[f"{method}_{suffix}" for method in METHODS
                    for suffix in ("lower", "upper", "covered", "width")])


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
REGIMES = ("pre-drift", "drift", "post-drift")


def read_current_intervals(path, region, seed, configurations):
    """Load only selected configurations; preserve chronological rows for each."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-observation T5 intervals: {path}")
    parts = []
    for chunk in pd.read_csv(path, chunksize=150000, usecols=list(INTERVAL_COLUMNS)):
        parts.append(chunk.loc[chunk.configuration.isin(configurations)])
    if not parts:
        raise ValueError(f"Empty interval file: {path}")
    data = pd.concat(parts, ignore_index=True)
    if data.empty or not data.region.eq(region).all() or not data.seed.eq(seed).all():
        raise ValueError(f"Region/seed mismatch or missing configurations in {path}")
    data["SETTLEMENTDATE"] = pd.to_datetime(data.SETTLEMENTDATE, errors="raise")
    # Original experiment stores timestamps in fixed AEMO UTC+10 wall time.
    if data.SETTLEMENTDATE.dt.tz is None:
        data["SETTLEMENTDATE"] = data.SETTLEMENTDATE.dt.tz_localize("Etc/GMT-10")
    else:
        data["SETTLEMENTDATE"] = data.SETTLEMENTDATE.dt.tz_convert("Etc/GMT-10")
    return data


def load_t5_reference(t5_dir):
    """Read current T5 outputs if present, without requiring old PR38 artefacts."""
    manifest_file = t5_dir / "t5_manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}
    seed_file = t5_dir / "t5_seed_metrics.csv"
    seed_metrics = pd.read_csv(seed_file) if seed_file.is_file() else None
    return manifest, seed_metrics


def t5_regime_metrics(frame, assignments, method, evaluation, protocol, region):
    """Retain the original within-regime 24-hour coverage audit."""
    actual, lower, upper = frame.y_true, frame[f"{method}_lower"], frame[f"{method}_upper"]
    rolling = evaluation.calculate_rolling_24h_coverage(
        frame.SETTLEMENTDATE, actual, lower, upper, cadence="30min",
        regimes=assignments.regime, boundary_policy="within_regime").to_numpy()
    scale = protocol.get("normalisation", {}).get(region, {})
    if isinstance(scale, dict):
        scale = scale.get("value")
    rows = []
    for regime in REGIMES:
        mask = assignments.regime.eq(regime).to_numpy()
        valid = rolling[mask & np.isfinite(rolling)]
        n = int(mask.sum())
        values = dict.fromkeys(T5_METRICS, float("nan"))
        if n:
            values["empirical_coverage"] = evaluation.calculate_empirical_coverage(
                actual[mask], lower[mask], upper[mask])
            values["mean_interval_width"] = evaluation.calculate_mean_interval_width(
                lower[mask], upper[mask])
            if scale is not None and np.isfinite(float(scale)) and float(scale) > 0:
                values["normalised_interval_width"] = evaluation.calculate_normalised_interval_width(
                    lower[mask], upper[mask], scale=float(scale))
            values["worst_24h_coverage"] = (float(valid.min()) if
                len(valid) >= protocol.get("minimum_complete_24h_windows", 1) else float("nan"))
        rows.append(dict(regime=regime, n_observations=n,
                         n_eligible_24h_windows=len(valid), **values))
    return rows


def check_t5_rows(recomputed, stored, region, seed, config, method):
    """Validate matching current T5 seed-level metrics when compatible rows exist.

    Unavailable/unknown schemas are reported as unverified, never as passing.
    """
    if stored is None:
        return [], "unavailable"
    required = {"region", "seed", "configuration", "method", "regime"}
    if not required <= set(stored.columns):
        return [], "incompatible_schema"
    subset = stored[stored.region.astype(str).eq(region) &
                    stored.seed.eq(seed) &
                    stored.configuration.astype(str).eq(config)]
    aliases = {"static": {"static", "static_conformal", "Static"},
               "aci": {"aci", "adaptive", "adaptive_conformal", "ACI"},
               "agaci": {"agaci", "agaci_boa", "AgACI-BOA"}}
    subset = subset[subset.method.astype(str).isin(aliases[method])]
    checks = []
    for item in recomputed:
        rows = subset[subset.regime.eq(item["regime"])]
        if "metric_name" in rows.columns and "metric_value" in rows.columns:
            for metric in T5_METRICS:
                match = rows[rows.metric_name.eq(metric)]
                if len(match) == 1:
                    expected = float(match.metric_value.iloc[0])
                    actual = item[metric]
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8,
                                               equal_nan=True)
                    checks.append(dict(region=region, seed=seed, configuration=config,
                                       method=method, regime=item["regime"], metric=metric,
                                       stored=expected, recomputed=actual, passed=True))
        elif len(rows) == 1:
            for metric in T5_METRICS:
                if metric in rows.columns:
                    expected = float(rows.iloc[0][metric])
                    actual = item[metric]
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8,
                                               equal_nan=True)
                    checks.append(dict(region=region, seed=seed, configuration=config,
                                       method=method, regime=item["regime"], metric=metric,
                                       stored=expected, recomputed=actual, passed=True))
    return checks, "verified" if checks else "not_matched"


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
        lower = frame[f"{method}_lower"].to_numpy(dtype=float)
        upper = frame[f"{method}_upper"].to_numpy(dtype=float)
        actual = frame["y_true"].to_numpy(dtype=float)
        if not np.isfinite(np.column_stack([actual, lower, upper])).all():
            raise ValueError("Non-finite actual values or interval endpoints.")
        if np.any(lower > upper):
            raise ValueError("Reversed interval endpoints.")
        widths = frame[f"{method}_width"].to_numpy(dtype=float)
        np.testing.assert_allclose(upper - lower, widths, rtol=0, atol=1e-6,
                                   err_msg=f"Stored width mismatch: {method}")
        covered = (lower <= actual) & (actual <= upper)
        if not np.array_equal(covered, boolean_values(frame[f"{method}_covered"])):
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


def detector_key(config):
    """Determine detector from C or 180-day D configuration names."""
    low = str(config).lower().replace("-", "_")
    if low.startswith("c_") or (low.startswith("d_") and ("180d" in low or "180_day" in low or "180days" in low)):
        if "page_hinkley" in low or "pagehinkley" in low or re.search(r"(?:^|_)ph(?:_|$)", low):
            return "page_hinkley"
        if "kswin" in low:
            return "kswin"
        if "adwin" in low:
            return "adwin"
    return None


def discover_configurations(path):
    """Read only configuration strings, not the full multi-GB interval table."""
    if not path.is_file():
        raise FileNotFoundError(path)
    found = set()
    for chunk in pd.read_csv(path, usecols=["configuration"], chunksize=250000):
        found.update(chunk.configuration.dropna().astype(str).unique())
    return found


def select_detector_panels(configurations):
    """One 4-arm panel set per detector; require matching C and D (180 days)."""
    names = set(configurations)
    if not {"A", "B"} <= names:
        raise ValueError("Arm A and B must be present in the interval files")
    groups = {}
    for detector in ("adwin", "kswin", "page_hinkley"):
        c = [x for x in names if x.lower().startswith("c_") and detector_key(x) == detector]
        d = [x for x in names if x.lower().startswith("d_") and detector_key(x) == detector]
        if len(c) != 1 or len(d) != 1:
            raise ValueError(f"Expected exactly one Arm C and 180-day Arm D for {detector}; found C={c}, D={d}. Available: {sorted(names)}")
        groups[detector] = ("A", "B", c[0], d[0])
    return groups


def plot_detector_figure(aggregates, windows, region, detector, configs, seeds, days, window, output):
    """One region/detector figure; 4 arms, 3 UQ methods in each panel."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.2), sharex=True, sharey=True)
    label_detector = {"adwin": "ADWIN", "kswin": "KSWIN", "page_hinkley": "Page–Hinkley"}[detector]
    titles = ("A  ·  No retraining", "B  ·  Periodic retraining (30 days)",
              f"C  ·  {label_detector}, full history", f"D  ·  {label_detector}, 180-day window")
    for ax, config, panel_title in zip(axes.flat, configs, titles):
        frame = aggregates[config]
        start, end = frame.SETTLEMENTDATE.iloc[[0, -1]]
        for event in windows.to_dict("records"):
            if event["post_drift_end"] <= start or event["drift_start"] > end:
                continue
            tier = str(event.get("tier", "")).lower().strip()
            alpha = .14 if tier in ("1", "tier 1", "tier1") else .055
            ax.axvspan(max(start, event["drift_start"]),
                        min(end, event["post_drift_end"]),
                        color="#77818d", alpha=alpha, linewidth=0, zorder=0)
        for method in METHODS:
            mean = frame[f"{method}_mean"].to_numpy(float)
            sd = frame[f"{method}_sd"].to_numpy(float)
            ax.plot(frame.SETTLEMENTDATE, mean, color=COLORS[method],
                    lw=1.25, label=LABELS[method], zorder=3)
            if len(seeds) > 1:
                ax.fill_between(frame.SETTLEMENTDATE,
                                np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                                color=COLORS[method], alpha=.10, linewidth=0)
        ax.axhline(.90, color="#26303c", ls="--", lw=1.25, zorder=4)
        ax.set_ylim(.45, 1.02)  # Coverage remains on its meaningful 0–1 scale
        ax.grid(axis="y", color="#cbd2d9", alpha=.45, lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(panel_title, fontsize=11.5, loc="left", pad=15)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Trailing {days}-day coverage")
    for ax in axes[-1, :]:
        ax.set_xlabel("Date")
    # Centred title and clearly separated subtitle, consistent with F5 presentation.
    fig.suptitle("Coverage trajectories by adaptation strategy", x=.5, y=.975,
                 ha="center", fontsize=22, fontweight="normal", fontfamily="serif")
    fig.text(.5, .932, f"F4  |  {region}  ·  {label_detector}  ·  {len(seeds)} seeds",
             ha="center", va="center", fontsize=15, color="#586474")
    handles = [Line2D([0], [0], color=COLORS[m], lw=2, label=LABELS[m]) for m in METHODS]
    handles.append(Line2D([0], [0], color="#26303c", ls="--", label="90% coverage target"))
    handles.append(Patch(facecolor="#77818d", alpha=.14,
                         label="Documented event window + 7-day post-event buffer"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .102),
               ncol=3, frameon=False, fontsize=10)
    caption = (f"Trailing {days}-day empirical coverage ({window} half-hour observations, as in F1). "
               f"Curves represent the mean of {len(seeds)} seeds; envelopes show ±1 sample SD. "
               "Documented event windows include a seven-day post-event buffer; events are not verified drift labels.")
    fig.text(.075, .038, textwrap.fill(caption, 145), fontsize=9,
             color="#586474", va="bottom")
    fig.subplots_adjust(left=.075, right=.975, top=.835, bottom=.22,
                        wspace=.17, hspace=.39)
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT /
                        "results/tables/uncertainty/agaci_boa_fixed_calibration")
    parser.add_argument("--t5-dir", type=Path, default=PROJECT_ROOT /
                        "results/tables/uncertainty/t5_agaci_full_post7")
    parser.add_argument("--events", type=Path, default=PROJECT_ROOT / "data/events.csv")
    parser.add_argument("--f1-script", type=Path, default=PROJECT_ROOT /
                        "experiments/degradation/generate_f1_degradation_figure.py")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT /
                        "results/figures/uncertainty/f4_agaci_full_post7")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--regions", nargs="+", choices=["SA1", "NSW1"], default=["SA1", "NSW1"])
    parser.add_argument("--detectors", nargs="+", choices=["adwin", "kswin", "page_hinkley"],
                        default=["adwin", "kswin", "page_hinkley"])
    parser.add_argument("--allow-preview", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds.")
    if set(args.seeds) != {1, 2, 3, 4, 5} and not args.allow_preview:
        parser.error("Final reporting requires seeds 1–5; use --allow-preview for development.")
    available = discover_configurations(args.run_dir / "seed_1" / "agaci_comparison_intervals_SA1.csv.gz")
    panel_groups = select_detector_panels(available)
    if len(set(args.detectors)) != len(args.detectors):
        parser.error("Duplicate detectors.")
    configs = list(dict.fromkeys(c for name in args.detectors for c in panel_groups[name]))
    days, daily, window = f1_window(args.f1_script)
    if daily != 48:
        raise ValueError("AEMO F4 expects 48 observations/day.")
    t5_manifest, t5_stored = load_t5_reference(args.t5_dir)
    protocol = t5_manifest.get("protocol", {})
    if isinstance(protocol, dict) and protocol.get("regime_days", 7) != 7:
        raise ValueError("The current T5 regime span is not 7 days.")
    events = pd.read_csv(args.events)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status="validation_in_progress", seeds=args.seeds,
                  regions=args.regions, configurations=configs,
                  detector_panels={k:list(panel_groups[k]) for k in args.detectors},
                  methods=list(METHODS), rolling_window_days=days,
                  rolling_window_observations=window,
                  window_source=str(args.f1_script),
                  regime_definition="Full-stream; documented event through event end plus 7 days; no pre-event buffer",
                  input_dir=str(args.run_dir), t5_dir=str(args.t5_dir),
                  sources={str(args.events): sha256(args.events),
                           str(args.f1_script): sha256(args.f1_script)}, streams=[])
    if (args.t5_dir / "t5_manifest.json").is_file():
        report["sources"][str(args.t5_dir / "t5_manifest.json")] = sha256(args.t5_dir / "t5_manifest.json")
    metric_rows, comparison_rows, captions, t5_rows, t5_checks = [], [], [], [], []
    for region in args.regions:
        windows = build_forecast_event_windows(events, region, regime_days=7)
        # Full-regime evaluation: no pre-event buffer (supervisor clarification).
        windows["pre_drift_start"] = windows["drift_start"]
        windows.to_csv(out / f"F4_{region}_event_windows.csv", index=False)
        assignments = None
        trajectories = {config: [] for config in configs}
        for seed in args.seeds:
            path = args.run_dir / f"seed_{seed}" / f"agaci_comparison_intervals_{region}.csv.gz"
            report["sources"][str(path)] = sha256(path)
            data = read_current_intervals(path, region, seed, configs)
            for config in configs:
                selected = data.loc[data.configuration.eq(config)].copy()
                if selected.empty or not selected.SETTLEMENTDATE.is_monotonic_increasing:
                    raise ValueError(f"Missing/nonchronological {region}/seed{seed}/{config} data")
                frame, rolling, metrics = check_stream(selected, window, daily)
                if assignments is None:
                    assignments = assign_full_stream_forecast_regimes(frame.SETTLEMENTDATE, windows)
                    if not set(assignments.regime) <= set(REGIMES):
                        raise ValueError("Full-stream regime assignment has invalid labels")
                    assignments.to_csv(out / f"F4_{region}_regime_assignment.csv.gz",
                                       index=False, compression={"method":"gzip", "mtime":0})
                if not assignments.SETTLEMENTDATE.equals(frame.SETTLEMENTDATE):
                    raise ValueError("Test timestamps vary across seeds or configurations")
                rolling["regime"] = assignments.regime
                rolling["event_id"] = assignments.event_id
                rolling.to_csv(out / f"F4_{region}_seed{seed}_{config}_rolling.csv.gz",
                               index=False, compression={"method":"gzip", "mtime":0})
                trajectories[config].append(rolling)
                for method in METHODS:
                    rec = t5_regime_metrics(frame, assignments, method,
                                            ev,
                                            protocol, region)
                    checks, result = check_t5_rows(rec, t5_stored, region, seed, config, method)
                    t5_checks.extend(checks)
                    t5_rows.extend(dict(region=region, seed=seed, configuration=config,
                                        method=method, **r) for r in rec)
                    for regime in ["overall", *REGIMES]:
                        mask = (np.ones(len(frame), dtype=bool) if regime == "overall"
                                else assignments.regime.eq(regime).to_numpy())
                        subset = frame.loc[mask]
                        metric_rows.append(dict(region=region, seed=seed,
                            configuration=config, method=method, regime=regime,
                            n_observations=len(subset),
                            coverage=(calculate_empirical_coverage(subset.y_true,
                                subset[f"{method}_lower"], subset[f"{method}_upper"])
                                if len(subset) else np.nan),
                            mean_interval_width=(calculate_mean_interval_width(
                                subset[f"{method}_lower"], subset[f"{method}_upper"])
                                if len(subset) else np.nan)))
                    # Source data and calculated empirical coverage must agree.
                    comparison_rows.append(dict(region=region, seed=seed,
                        configuration=config, method=method,
                        source_coverage=float(boolean_values(frame[f"{method}_covered"]).mean()),
                        computed_coverage=metrics[method]["empirical_coverage"], passed=True))
                report["streams"].append(dict(region=region, seed=seed,
                    configuration=config, chronological=True, interval_bounds=True,
                    coverage_flags=True, width_validation=True, trailing_window=True,
                    prefix_invariance=True))
                print(f"PASS {region} seed {seed} {config}", flush=True)
        aggregates = {}
        for config, streams in trajectories.items():
            agg = aggregate_trajectories(streams, args.seeds)
            aggregates[config] = agg
            stem = f"F4_{region}_{config}"
            agg.to_csv(out / f"{stem}_mean_sd.csv.gz", index=False,
                       compression={"method":"gzip", "mtime":0})

        for detector in args.detectors:
            panel_configs = panel_groups[detector]
            filename = out / f"F4_{region}_{detector}_four_arms.png"
            captions.append(plot_detector_figure(aggregates, windows, region, detector,
                            panel_configs, args.seeds, days, window, filename))
            print(f"Saved {filename}", flush=True)
    t5_rows = pd.DataFrame(t5_rows)
    t5_rows.to_csv(out / "F4_t5_recomputed_metrics.csv", index=False)
    pd.DataFrame(t5_checks).to_csv(out / "F4_t5_seed_validation.csv", index=False)
    seed_metrics = pd.DataFrame(metric_rows)
    seed_metrics.to_csv(out / "F4_coverage_width_by_seed_regime.csv", index=False)
    seed_metrics.groupby(["region", "configuration", "method", "regime"], sort=True).agg(
        coverage_mean=("coverage", "mean"), coverage_sd=("coverage", "std"),
        mean_width_mean=("mean_interval_width", "mean"),
        mean_width_sd=("mean_interval_width", "std"),
        n_seeds=("seed", "nunique")).reset_index().to_csv(
            out / "F4_coverage_width_mean_sd.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(out / "F4_upstream_summary_validation.csv", index=False)
    (out / "F4_captions.txt").write_text("\n\n".join(captions)+"\n")
    report["status"] = "passed_stream_checks"
    report["t5_validation"] = dict(status=("partially_verified" if t5_checks else "not_verified"),
                                    matched_seed_metrics=len(t5_checks),
                                    explanation="Stream checks passed; verify against current T5 separately")
    report["runs_csv_validation"] = "not_checked; run validate_f4_canonical_log.py"
    (out / "F4_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"F4 complete: {out}; streams={len(report['streams'])}; T5 metric comparisons={len(t5_checks)}")


if __name__ == "__main__":
    main()
