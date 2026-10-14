"""Generate exact six-column T5 tables from PGS242-160 interval streams.

No training, interval reconstruction, parameter tuning or implicit methodology.
See docs/T5_EVALUATION.md for the required explicit protocol JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drift_lab.evaluation import (
    aemo_timestamp, assign_forecast_regimes, assign_full_stream_forecast_regimes,
    build_forecast_event_windows, calculate_empirical_coverage,
    calculate_mean_interval_width, calculate_normalised_interval_width,
    calculate_rolling_24h_coverage,
)
from drift_lab.run_logger import RUNS_COLUMNS, log_run

SEEDS = (1, 2, 3, 4, 5)
REGIONS = ("SA1", "NSW1")
REGIMES = ("pre-drift", "drift", "post-drift")
METHODS = ("static_conformal", "adaptive_conformal")
METRICS = ("empirical_coverage", "mean_interval_width",
           "normalised_interval_width", "worst_24h_coverage")
T5_COLUMNS = ("UQ method", "regime", "coverage", "mean width",
              "normalised width", "worst 24h coverage")
GROUP = "t5_uncertainty"


def is_simulation(protocol):
    """Recognise the existing synthetic arithmetic fixture only."""
    return protocol["protocol_id"] == "synthetic-test-only"


def minimum_windows(protocol):
    """AEMO requires an explicit reporting threshold; the fixture keeps one."""
    value = protocol.get("minimum_complete_24h_windows", 1 if is_simulation(protocol) else None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Set minimum_complete_24h_windows to an agreed positive integer.")
    return value


def training_demand_mean(source, start, end_exclusive):
    """Calculate mean actual demand from a regional source within TRAIN only."""
    frame = pd.read_csv(source, usecols=["SETTLEMENTDATE", "TOTALDEMAND"])
    timestamps = frame.SETTLEMENTDATE.map(aemo_timestamp)
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("Training source has missing or duplicate timestamps.")
    start, end = aemo_timestamp(start), aemo_timestamp(end_exclusive)
    if start >= end:
        raise ValueError("Training window must have start < end_exclusive.")
    actual = frame.loc[(timestamps >= start) & (timestamps < end), "TOTALDEMAND"].to_numpy(dtype=float)
    if not len(actual) or not np.isfinite(actual).all():
        raise ValueError("Training actual demand must be nonempty and finite.")
    value = float(actual.mean())
    if value <= 0:
        raise ValueError("Training mean actual demand must be positive.")
    return value, len(actual)


def verify_aemo_normalisation(protocol):
    """Verify fixed regional TRAIN means and their source hashes before logging."""
    if is_simulation(protocol):
        return
    window = protocol.get("training_window", {})
    if not {"start", "end_exclusive"} <= window.keys():
        raise ValueError("AEMO needs training_window start and end_exclusive.")
    if (aemo_timestamp(window["start"]), aemo_timestamp(window["end_exclusive"])) != (
            aemo_timestamp("2018-01-01"), aemo_timestamp("2020-01-01")):
        raise ValueError("Use the agreed AEMO TRAIN window [2018-01-01, 2020-01-01).")
    for region in REGIONS:
        scale = protocol["normalisation"][region]
        if scale["statistic"] != "training_mean_actual_demand":
            raise ValueError("AEMO denominator must be training_mean_actual_demand.")
        if digest(scale["source"]) != scale["source_sha256"]:
            raise ValueError(f"Training source hash mismatch for {region}.")
        actual_mean, _ = training_demand_mean(scale["source"], **window)
        if not np.isclose(scale["value"], actual_mean, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Denominator differs from TRAIN mean actual demand for {region}.")


def digest(path):
    with Path(path).open("rb") as stream:
        result = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else None
        if result is not None:
            return result.hexdigest()
        stream.seek(0)
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_protocol(protocol):
    """Fail closed on missing or unsupported methodology."""
    required = {"protocol_id", "regime_scope", "regime_days", "rolling_boundary",
                "rolling_closed", "incomplete_windows", "cadence", "granularity",
                "normalisation", "configurations", "test_start", "test_end"}
    if required - protocol.keys():
        raise ValueError(f"Missing explicit protocol fields: {sorted(required - protocol.keys())}")
    if not protocol["protocol_id"]:
        raise ValueError("protocol_id is required.")
    minimum_windows(protocol)
    if protocol["regime_scope"] not in {"full", "event-local"}:
        raise ValueError("regime_scope must be full or event-local.")
    if protocol["regime_days"] != 7:
        raise ValueError("Shared T3 buffers are seven days.")
    if protocol["rolling_boundary"] not in {"within_regime", "ending_regime"}:
        raise ValueError("Unsupported rolling_boundary.")
    if (protocol["rolling_closed"], protocol["incomplete_windows"],
        protocol["cadence"], protocol["granularity"]) != (
            "right", "exclude", "30min", "region_configuration"):
        raise ValueError("Supported policy: (t-24h,t], complete 30min windows; separate region/configuration.")
    configs = protocol["configurations"]
    if not configs or len(configs) != len(set(configs)):
        raise ValueError("Explicit unique configurations are required.")
    if any(not isinstance(c, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in configs):
        raise ValueError("Configurations must be safe filename identifiers.")
    scales = protocol["normalisation"]
    if set(scales) != set(REGIONS):
        raise ValueError("Provide one fixed normalisation scale for each region.")
    for region, scale in scales.items():
        if not {"value", "statistic", "source", "source_sha256"} <= scale.keys():
            raise ValueError(f"Missing scale provenance for {region}.")
        if not all(scale[k] for k in ("statistic", "source", "source_sha256")):
            raise ValueError("Scale provenance must not be empty.")
        if not re.fullmatch(r"[0-9a-f]{64}", scale["source_sha256"]):
            raise ValueError("Scale source_sha256 must be a SHA256 digest.")
        calculate_normalised_interval_width([0], [1], scale=scale["value"])
    start, end = aemo_timestamp(protocol["test_start"]), aemo_timestamp(protocol["test_end"])
    if end < start or (end - start) % pd.Timedelta("30min"):
        raise ValueError("test_end must lie on the grid starting at test_start.")
    return protocol


def required_paths(run_dir):
    paths = []
    for seed in SEEDS:
        base = Path(run_dir) / f"seed_{seed}"
        paths.extend([base / "manifest.json",
                      base / "uncertainty/adaptive_conformal_manifest.json",
                      base / "uncertainty/aemo_conformal_comparison_summary.csv"])
        paths.extend(base / f"uncertainty/aemo_conformal_intervals_{r}.csv.gz"
                     for r in REGIONS)
    return paths


def preflight(run_dir):
    missing = [str(p) for p in required_paths(run_dir) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing PGS242-160 inputs (no T5 generated):\n" + "\n".join(missing))


def evaluate_stream(frame, labels, *, scale, protocol):
    """Compute each metric by regime for one seed/configuration/UQ method."""
    rolling = calculate_rolling_24h_coverage(
        frame.SETTLEMENTDATE, frame.y_true, frame.lower, frame.upper,
        cadence=protocol["cadence"], regimes=labels,
        boundary_policy=protocol["rolling_boundary"],
    ).to_numpy()
    rows = []
    required_windows = minimum_windows(protocol)
    for regime in REGIMES:
        mask = np.asarray(labels) == regime
        subset = frame.loc[mask]
        values = dict.fromkeys(METRICS, float("nan"))
        valid = rolling[mask & np.isfinite(rolling)]
        if len(subset):
            values.update(
                empirical_coverage=calculate_empirical_coverage(
                    subset.y_true, subset.lower, subset.upper),
                mean_interval_width=calculate_mean_interval_width(subset.lower, subset.upper),
                normalised_interval_width=calculate_normalised_interval_width(
                    subset.lower, subset.upper, scale=scale),
                worst_24h_coverage=float(valid.min()) if len(valid) >= required_windows else float("nan"),
            )
        rows.append({"regime": regime, "n_observations": len(subset),
                     "n_eligible_24h_windows": len(valid),
                     "minimum_complete_24h_windows": required_windows,
                     "worst_window_status": ("no_complete_windows" if not len(valid) else
                         "insufficient_windows" if len(valid) < required_windows else "sufficient"),
                     **values})
    return rows


def aggregate(logs):
    """Aggregate only complete five-seed metrics; undefined metrics stay N/A."""
    rows = []
    for (method, regime), group in logs.groupby(["method", "regime"], sort=True):
        row = {"UQ method": method.split("|", 1)[1], "regime": regime}
        for metric, column in zip(METRICS, T5_COLUMNS[2:]):
            selected = group[group.metric_name == metric]
            if len(selected) != 5 or set(selected.seed) != set(SEEDS):
                raise ValueError("Every T5 cell requires exactly seeds 1–5.")
            values = selected.metric_value.to_numpy(dtype=float)
            row[column] = ("N/A" if not np.isfinite(values).all() else
                           f"{values.mean():.8f} ± {values.std(ddof=1):.8f}")
        rows.append(row)
    return pd.DataFrame(rows, columns=T5_COLUMNS)


def write_table_report(output_dir, region, configuration, table, audit, protocol):
    """Keep the six-column CSV while publishing its caption and window support."""
    scale = protocol["normalisation"][region]
    selected = audit[(audit.region == region) &
                     audit.method.eq(configuration + "|static_conformal")].sort_values(["regime", "seed"])
    # Eligibility depends on timestamps/regimes, not interval coverage or UQ method.
    comparison = audit[(audit.region == region) &
                       audit.method.eq(configuration + "|adaptive_conformal")]
    counts = selected.merge(comparison, on=["region", "regime", "seed"], suffixes=("_static", "_adaptive"))
    if not (counts.n_eligible_24h_windows_static == counts.n_eligible_24h_windows_adaptive).all():
        raise ValueError("Static/adaptive complete-window counts differ.")
    caption = (f"T5 — {region}, {configuration}. Normalised width denominator: "
               f"{scale['value']:.17g} ({scale['statistic']}); fixed across all methods, seeds and regimes.")
    if not is_simulation(protocol):
        window = protocol["training_window"]
        caption += f" TRAIN actual demand in MW, [{window['start']}, {window['end_exclusive']}) in fixed AEST."
    else:
        caption += " Synthetic demonstration only."
    lines = [f"# T5 — {region} / {configuration}", "", caption, "",
             "| " + " | ".join(T5_COLUMNS) + " |",
             "| " + " | ".join(["---"] * len(T5_COLUMNS)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False, name=None)]
    lines += ["", "## Complete 24-hour windows", "",
              f"Minimum required per seed/regime: {minimum_windows(protocol)}. "
              "Worst coverage is N/A if any seed has too few windows. "
              "Counts are identical for static and adaptive intervals. "
              "These overlapping rolling windows are not independent samples.", "",
              "| Regime | Seed | Complete windows | Worst-window status |",
              "| --- | --- | --- | --- |"]
    lines += [f"| {row.regime} | {row.seed} | {row.n_eligible_24h_windows} | {row.worst_window_status} |"
              for row in selected.itertuples()]
    (output_dir / f"T5_{region}_{configuration}.md").write_text("\n".join(lines) + "\n")


def append_logs_safely(rows, runs_path):
    """Use shared harness; preserve historical bytes and reject collisions.

    Single writer required, like the existing shared logger. Identical reruns
    are idempotent. All checks precede the atomic replacement.
    """
    runs_path = Path(runs_path)
    existing = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame(columns=RUNS_COLUMNS)
    if list(existing.columns) != RUNS_COLUMNS:
        raise ValueError("runs.csv schema differs from shared harness.")
    keys = ["group", "method", "dataset", "region", "seed", "config_hash",
            "split_id", "metric_name", "regime"]
    additions = []
    for row in rows:
        matched = existing
        for key in keys:
            matched = matched[matched[key] == row[key]]
        if len(matched) > 1:
            raise ValueError("Duplicate matching metrics already exist in runs.csv.")
        if len(matched):
            value = float(matched.iloc[0].metric_value)
            if not np.isclose(value, row["metric_value"], rtol=1e-12, atol=1e-12, equal_nan=True):
                raise ValueError("Existing metric conflicts with recomputed value.")
        else:
            additions.append(row)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runs_path.parent) as tmp:
        stage = Path(tmp) / "runs.csv"
        if runs_path.exists():
            original = runs_path.read_bytes()
            if original and not original.endswith(b"\n"):
                raise ValueError("Historical runs.csv must end in a newline.")
            stage.write_bytes(original)
        for row in additions:
            log_run(results_path=stage, **row)
        if not stage.exists():
            from drift_lab.run_logger import ensure_runs_file
            ensure_runs_file(stage)
        recorded = pd.read_csv(stage)
        os.replace(stage, runs_path)
    return recorded


def generate(run_dir, events_path, output_dir, runs_path, protocol):
    validate_protocol(protocol)
    verify_aemo_normalisation(protocol)
    preflight(run_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing results are preserved.")
    expected = pd.date_range(aemo_timestamp(protocol["test_start"]),
                             aemo_timestamp(protocol["test_end"]), freq="30min")
    rows, audit, sources, identities = [], [], [], []
    references = {}
    split_id = None
    for seed in SEEDS:
        base = Path(run_dir) / f"seed_{seed}"
        manifest = json.loads((base / "manifest.json").read_text())
        uq = json.loads((base / "uncertainty/adaptive_conformal_manifest.json").read_text())
        if manifest["seed"] != seed or uq["seed"] != seed:
            raise ValueError("Manifest seed mismatch.")
        if uq["upstream_run_id"] != manifest["run_id"] or uq["split_id"] != manifest["split_id"]:
            raise ValueError("UQ manifest does not match upstream run.")
        if uq["task"] != "PGS242-160" or uq["alpha"] != 0.10:
            raise ValueError("Expected PGS242-160 nominal 90% intervals.")
        if split_id is not None and split_id != uq["split_id"]:
            raise ValueError("Seeds have different split_id values.")
        split_id = uq["split_id"]
        summary_path = base / "uncertainty/aemo_conformal_comparison_summary.csv"
        summary = pd.read_csv(summary_path)
        if summary.duplicated(["region", "configuration", "method"]).any():
            raise ValueError("Duplicate UQ summary identities.")
        for region in REGIONS:
            path = base / f"uncertainty/aemo_conformal_intervals_{region}.csv.gz"
            frame = pd.read_csv(path)
            required = {"SETTLEMENTDATE", "seed", "region", "configuration", "y_true",
                        "static_lower_90", "static_upper_90",
                        "adaptive_lower_90", "adaptive_upper_90"}
            if required - set(frame):
                raise ValueError(f"Missing interval columns: {sorted(required - set(frame))}")
            if frame[list(required)].isna().any().any():
                raise ValueError("Missing interval fields.")
            if not (frame.seed.eq(seed).all() and frame.region.eq(region).all()):
                raise ValueError("Interval stream seed/region mismatch.")
            if set(frame.configuration) != set(protocol["configurations"]):
                raise ValueError("Interval configurations differ from explicit protocol.")
            frame["SETTLEMENTDATE"] = frame.SETTLEMENTDATE.map(aemo_timestamp)
            windows = build_forecast_event_windows(events_path, region, regime_days=7)
            assign = (assign_full_stream_forecast_regimes if protocol["regime_scope"] == "full"
                      else assign_forecast_regimes)
            labels = assign(expected, windows).regime.to_numpy()
            source = {"seed": seed, "region": region, "path": str(path.resolve()),
                      "sha256": digest(path), "summary_sha256": digest(summary_path),
                      "upstream_run_id": manifest["run_id"], "uq_manifest": uq}
            sources.append(source)
            for config in protocol["configurations"]:
                group = frame[frame.configuration == config].reset_index(drop=True)
                if not pd.DatetimeIndex(group.SETTLEMENTDATE).equals(expected):
                    raise ValueError("Stream must match full explicit grid exactly, in order.")
                truth = group.y_true.to_numpy(dtype=float)
                if not np.isfinite(truth).all():
                    raise ValueError("Nonfinite y_true.")
                if region in references and not np.array_equal(truth, references[region]):
                    raise ValueError("Actual values differ across seeds/configurations.")
                references[region] = truth
                for method in METHODS:
                    prefix = method.split("_")[0]
                    bounds = group.rename(columns={
                        f"{prefix}_lower_90": "lower", f"{prefix}_upper_90": "upper"})
                    upstream = summary[(summary.region == region) &
                                       (summary.configuration == config) &
                                       (summary.method == method)]
                    if len(upstream) != 1:
                        raise ValueError("Expected exactly one UQ summary row.")
                    meta = upstream.iloc[0]
                    if int(meta.seed) != seed or meta.alpha != 0.10 or int(meta.test_size) != len(expected):
                        raise ValueError("UQ summary seed/alpha/test_size mismatch.")
                    for metric, value in (
                        ("empirical_coverage", calculate_empirical_coverage(truth, bounds.lower, bounds.upper)),
                        ("mean_interval_width", calculate_mean_interval_width(bounds.lower, bounds.upper)),
                    ):
                        if not np.isclose(value, meta[metric], rtol=1e-10, atol=1e-10):
                            raise ValueError("Stream metrics disagree with PGS242-160 summary.")
                    evaluation_hash = json_hash({
                        "protocol": protocol, "source": source, "configuration": config,
                        "method": method, "uq_config_hash": meta.uq_config_hash,
                        "events_sha256": digest(events_path),
                    })
                    identity = dict(group=GROUP, method=f"{config}|{method}",
                                    dataset="AEMO_TOTALDEMAND", region=region, seed=seed,
                                    config_hash=evaluation_hash, split_id=split_id)
                    identities.append(identity)
                    for evaluated in evaluate_stream(bounds, labels,
                            scale=protocol["normalisation"][region]["value"], protocol=protocol):
                        audit.append({**identity, **evaluated})
                        for metric in METRICS:
                            rows.append({**identity, "regime": evaluated["regime"],
                                         "metric_name": metric, "metric_value": evaluated[metric],
                                         "n_retrains": int(meta.n_retrains),
                                         "train_samples": int(meta.train_samples_total),
                                         "wall_clock_s": 0.0})
    # Data validation is complete before any writes.
    recorded = append_logs_safely(rows, runs_path)
    selected = recorded.merge(pd.DataFrame(identities).drop_duplicates(),
                              on=list(identities[0]), how="inner", validate="many_to_one")
    output_dir.mkdir(parents=True)
    selected.to_csv(output_dir / "t5_seed_metrics.csv", index=False)
    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(output_dir / "t5_window_audit.csv", index=False)
    for region in REGIONS:
        for config in protocol["configurations"]:
            subset = selected[(selected.region == region) &
                              selected.method.str.startswith(config + "|")]
            table = aggregate(subset)
            table.to_csv(output_dir / f"T5_{region}_{config}.csv", index=False)
            write_table_report(output_dir, region, config, table, audit_frame, protocol)
    (output_dir / "t5_manifest.json").write_text(json.dumps({
        "protocol": protocol, "sources": sources, "events_sha256": digest(events_path),
        "runs_path": str(Path(runs_path).resolve()), "runs_sha256": digest(runs_path),
        "metric_count": len(rows), "sd_ddof": 1,
        "units": "coverage fractions; mean width MW; normalised width dimensionless",
        "undefined": "N/A if any seed lacks observations or a qualifying full 24h window",
        "logging": "wall_clock_s=0: T5 evaluation does not rerun model training",
    }, indent=2) + "\n")
    return pd.DataFrame(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=ROOT / "data/events.csv")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=ROOT / "results/runs.csv")
    args = parser.parse_args()
    generate(args.run_dir, args.events, args.output_dir, args.runs,
             json.loads(args.protocol.read_text()))


if __name__ == "__main__":
    main()
