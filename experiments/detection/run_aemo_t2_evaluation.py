"""T2: evaluate stored Task 120 detections; never rerun a forecaster/detector.

Run from any directory. --from-logs reconstructs the final table solely from
results/tables/runs.csv and the saved manifest. See docs/aemo_matching_protocol.md.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drift_lab.data_loader import load_and_split_aemo
from drift_lab.detectors import DETECTOR_REGISTRY
from drift_lab.evaluation import (
    AEMO_REGIMES, AEMO_TIER1_COLUMNS, AEMO_TIMEZONE, aemo_timestamp,
    build_aemo_event_windows, evaluate_aemo_detections, load_aemo_events,
    validate_aemo_protocol,
)
from drift_lab.run_logger import log_run

INPUT = ROOT / "results/tables/detection/aemo_detection_changepoints.csv"
EVENTS = ROOT / "data/events.csv"
PROTOCOL = ROOT / "docs/aemo_matching_protocol.json"
RUNS = ROOT / "results/tables/runs.csv"
OUT = ROOT / "results/tables/detection"
MANIFEST = OUT / "T2_manifest.json"
TABLE = OUT / "T2_aemo_detection.csv"
GROUP = "aemo_t2"
SPLIT = "aemo_fixed_train2018_2019_test202003_task120_t2_t1rule_v2"
REGIONS = ("SA1", "NSW1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scalar(data, metric):
    rows = data[data.metric_name == metric]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one logged value for {metric}; got {len(rows)}.")
    return float(rows.metric_value.iloc[0])

def build_t2_from_runs(runs, configurations):

    """Reconstruct all five delays and ratio-of-counts precision from logs."""

    result = []

    for config in configurations:

        method, region = config["detector"], config["region"]

        data = runs[(runs.group == GROUP) & (runs.split_id == SPLIT)

                    & (runs.method == method) & (runs.region == region)

                    & (runs.config_hash == config["config_hash"])

                    & (runs.seed == config["seed"])]

        if data.duplicated(["metric_name", "regime"]).any():

            raise ValueError("Duplicate T2 log metrics.")

        row = {"detector": method, "region": region}

        for event_id, column in AEMO_TIER1_COLUMNS.items():

            prefix = f"t2.event.{event_id}"

            applies, detected = _scalar(
                data,
                prefix + ".applicable"
            ), _scalar(
                data,
                prefix + ".detected"
            )

            if applies not in (0, 1) or detected not in (0, 1):

                raise ValueError("Invalid logged event status.")

            if not applies:

                row[column] = "not applicable"

            elif not detected:

                row[column] = "not detected"

            else:

                delay = _scalar(data, prefix + ".delay_days")

                if not np.isfinite(delay) or delay < 0:

                    raise ValueError("Invalid logged event delay.")

                row[column] = round(delay, 3)

        counts = {}

        for name in (
            "detections",
            "tier1_matched",
            "tier2_matched",
            "unmatched",
        ):

            values = data[
                data.metric_name == f"t2.global.{name}"
            ]

            if (
                set(values.regime) != set(AEMO_REGIMES)
                or len(values) != 3
            ):

                raise ValueError(
                    f"Incomplete logged regimes for {name}."
                )

            if (
                (values.metric_value < 0)
                | (values.metric_value % 1 != 0)
            ).any():

                raise ValueError(
                    "Detection counts must be non-negative integers."
                )

            counts[name] = int(
                values.metric_value.sum()
            )

        if (
            counts["detections"]
            != counts["tier1_matched"]
            + counts["tier2_matched"]
            + counts["unmatched"]
        ):

            raise ValueError(
                "Logged detection accounting does not balance."
            )

        row["tier1_unmatched"] = (
            counts["detections"]
            - counts["tier1_matched"]
        )

        row["tier2_contextual"] = (
            counts["tier2_matched"]
        )

        row["unmatched"] = (
            counts["unmatched"]
        )

        row["precision"] = (
            (
                counts["tier1_matched"]
                + counts["tier2_matched"]
            )
            / counts["detections"]
            if counts["detections"]
            else float("nan")
        )

        result.append(row)

    return pd.DataFrame(result)


def prepare_inputs():
    """Verify Task 120 index mapping, normalization, configurations and coverage."""
    detections = pd.read_csv(INPUT)
    required = {"region", "detector", "detected_index", "SETTLEMENTDATE", "forecast_error",
                "normalized_abs_error", "calibration_abs_error_mean",
                "calibration_abs_error_std", "detector_config"}
    if not required.issubset(detections.columns) or detections[list(required)].isna().any().any():
        raise ValueError("Incomplete Task 120 input fields.")
    if not detections.region.isin(REGIONS).all() or not detections.detector.isin(DETECTOR_REGISTRY).all():
        raise ValueError("Unexpected detector or region in Task 120.")
    if detections.duplicated(["region", "detector", "detected_index"]).any():
        raise ValueError("Duplicate Task 120 indices.")
    values = detections[["forecast_error", "normalized_abs_error", "calibration_abs_error_mean",
                         "calibration_abs_error_std"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 3] <= 0).any():
        raise ValueError("Invalid Task 120 normalization values.")
    if not np.allclose(values[:, 1], (np.abs(values[:, 0]) - values[:, 2]) / values[:, 3]):
        raise ValueError("Task 120 normalized absolute error does not reproduce.")
    prepared = []
    for region in REGIONS:
        # Read/standardise the original stream only to validate indices and exposure.
        _, train, calibration, test = load_and_split_aemo(region, ROOT / "data/raw")
        times = pd.DatetimeIndex(test.SETTLEMENTDATE).tz_localize(AEMO_TIMEZONE)
        for method, entry in DETECTOR_REGISTRY.items():
            rows = detections[(detections.region == region) & (detections.detector == method)].copy()
            # Task 120's CSV has no run roster: absence is not proof of zero detections.
            if rows.empty:
                raise ValueError(f"Missing Task 120 run {method}/{region}; supply a run roster before treating it as zero.")
            indices = pd.to_numeric(rows.detected_index, errors="raise")
            if ((indices % 1 != 0) | (indices < 0) | (indices >= len(test))).any():
                raise ValueError("Task 120 index outside test stream.")
            actual = pd.DatetimeIndex([aemo_timestamp(t) for t in rows.SETTLEMENTDATE])
            if not (actual == times[indices.astype(int)]).all():
                raise ValueError("Task 120 index/timestamp mapping does not reproduce.")
            if not indices.is_monotonic_increasing:
                raise ValueError("Task 120 indices are not chronological.")
            if not all(json.loads(c) == entry["config"] for c in rows.detector_config):
                raise ValueError(f"Task 120 configuration differs from frozen registry: {method}.")
            if rows.calibration_abs_error_mean.nunique() != 1 or rows.calibration_abs_error_std.nunique() != 1:
                raise ValueError("Mixed Task 120 calibration configurations.")
            prepared.append({"detector": method, "region": region, "rows": rows,
                             "times": times, "train_samples": len(train),
                             "calibration_samples": len(calibration), "detector_config": entry["config"],
                             "seed": entry["config"].get("seed", 0)})
    return prepared


def metric_records(result, events, times, region, protocol):
    """Long-format event metrics and a non-overlapping global accounting layer."""
    records = []
    def add(name, value, regime):
        records.append({"metric_name": name, "metric_value": float(value), "regime": regime})
    for regime, metrics in result["global_metrics"].items():
        for name, value in metrics.items():
            add(f"t2.global.{name}", value, regime)
    for event in result["tier1_events"].to_dict("records"):
        prefix = f"t2.event.{event['event_id']}"
        add(prefix + ".applicable", event["applicable"], "drift")
        add(prefix + ".detected", event["detected"], "drift")
        add(prefix + ".delay_days", event["delay_days"], event["delay_regime"])
    audit = result["audit"]
    timestamps = pd.DatetimeIndex(audit.SETTLEMENTDATE)
    for event in events[events.region.isin(["NEM", region])].to_dict("records"):
        prefix = f"t2.window.{event['event_id']}"
        for regime, (start, end) in build_aemo_event_windows(event, regime_days=protocol["regime_days"]).items():
            window = audit[(timestamps >= start) & (timestamps < end)]
            n = len(window)
            n1, n2 = int((window.match_tier == 1).sum()), int((window.match_tier == 2).sum())
            values = {"observations": int(((times >= start) & (times < end)).sum()),
                      "detections": n, "tier1_matched": n1, "tier2_matched": n2,
                      "unmatched": n - n1 - n2, "precision": (n1 + n2) / n if n else float("nan"),
                      "assigned_to_event": int((window.matched_event_id == event["event_id"]).sum())}
            for name, value in values.items():
                add(prefix + "." + name, value, regime)
    return records


def append_metrics(records, common):
    """Skip identical completed runs; reject partial/conflicting runs."""
    expected = pd.DataFrame(records).sort_values(["metric_name", "regime"]).reset_index(drop=True)
    if RUNS.exists():
        existing = pd.read_csv(RUNS)
        subset = existing[(existing.group == GROUP) & (existing.split_id == SPLIT)
                          & (existing.method == common["method"]) & (existing.region == common["region"])
                          & (existing.config_hash == common["config_hash"]) & (existing.seed == common["seed"])]
        if len(subset):
            subset = subset[expected.columns].sort_values(["metric_name", "regime"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(subset, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
            return False
    for record in records:
        log_run(**record, **common, results_path=RUNS)
    return True


def markdown_table(frame):
    display = frame.fillna("N/A").astype(str)
    return "\n".join(["| " + " | ".join(display.columns) + " |",
                      "| " + " | ".join(["---"] * len(display.columns)) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in display.to_numpy()])


def write_table(table):
    table.to_csv(TABLE, index=False, na_rep="N/A")
    shown = table.copy()
    shown["precision"] = shown.precision.map(
        lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
    )
    (OUT / "T2_aemo_detection.md").write_text(
        "# T2 — Detection on AEMO vs documented events\n\n" + markdown_table(shown) +
        "\n\nDelays are fractional days from the frozen reference date, rounded to 3 decimals. "
        "Tier 1 unmatched counts detections that did not match a primary Tier 1 event. "
        "Tier 2 contextual reports how many of those detections were subsequently matched "
        "to Tier 2 events. Unmatched counts detections matched to neither tier. "
        "Precision is the proportion of detections matched to either Tier 1 or Tier 2. "
        "Historical events are contextual references; unmatched detections are not automatically false positives. "
        "See docs/aemo_matching_protocol.md and T2_manifest.json.\n",
        encoding="utf-8",
    )
    
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-logs", action="store_true")
    args = parser.parse_args()
    if args.from_logs:
        manifest = json.loads(MANIFEST.read_text())
        write_table(build_t2_from_runs(pd.read_csv(RUNS), manifest["configurations"]))
        print(f"Rebuilt from logged metrics: {TABLE}")
        return
    protocol = json.loads(PROTOCOL.read_text())
    validate_aemo_protocol(protocol)
    events = load_aemo_events(EVENTS)
    sources = [INPUT, EVENTS, PROTOCOL, ROOT / "docs/aemo_matching_protocol.md",
               ROOT / "docs/event_tiering_criteria.md", Path(__file__),
               ROOT / "src/drift_lab/evaluation.py", ROOT / "src/drift_lab/run_logger.py",
               ROOT / "src/drift_lab/data_loader.py", ROOT / "src/drift_lab/detectors.py",
               ROOT / "experiments/detection/run_aemo_detection.py",
               ROOT / "experiments/detection/run_synthetic_detection_evaluation_absnorm_v2.py"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    prepared = prepare_inputs()
    configurations, audits, event_rows, summaries = [], [], [], []
    OUT.mkdir(parents=True, exist_ok=True)
    for run in prepared:
        method, region = run["detector"], run["region"]
        metadata = {k: run[k] for k in ("detector", "region", "train_samples",
                                      "calibration_samples", "detector_config", "seed")}
        metadata.update({"test_samples": len(run["times"]),
                         "test_start": str(run["times"][0]),
                         "test_end": str(run["times"][-1])})
        identity = json.dumps({"hashes": hashes, "run": metadata}, sort_keys=True).encode()
        metadata["config_hash"] = hashlib.sha256(identity).hexdigest()
        started = perf_counter()
        result = evaluate_aemo_detections(run["rows"], events, region, protocol=protocol,
                                         stream_start=run["times"][0],
                                         stream_end_exclusive=run["times"][-1] + pd.Timedelta(minutes=30))
        records = metric_records(result, events, run["times"], region, protocol)
        elapsed = perf_counter() - started
        append_metrics(records, {"method": method, "region": region, "group": GROUP,
                                "dataset": "AEMO_TOTALDEMAND", "split_id": SPLIT,
                                "seed": run["seed"], "config_hash": metadata["config_hash"],
                                "train_samples": run["train_samples"], "n_retrains": 0,
                                "wall_clock_s": elapsed})
        configurations.append(metadata)
        audit = result["audit"].assign(config_hash=metadata["config_hash"])
        audits.append(audit)
        event_rows.append(result["tier1_events"].assign(detector=method, region=region))
        summaries.append({"detector": method, "region": region, **result["totals"]})
        print(f"{method}/{region}: {result['totals']}")
    if any(digest(ROOT / path) != value for path, value in hashes.items()):
        raise RuntimeError("Inputs or protocol changed during evaluation.")
    pd.concat(audits, ignore_index=True).to_csv(OUT / "T2_detection_matches.csv", index=False)
    pd.concat(event_rows, ignore_index=True).to_csv(OUT / "T2_tier1_event_details.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "T2_detection_summary.csv", index=False)
    table = build_t2_from_runs(pd.read_csv(RUNS), configurations)
    write_table(table)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                "source_sha256": hashes, "configurations": configurations,
                "runs_path": "results/tables/runs.csv", "group": GROUP, "split_id": SPLIT,
                "notes": "Stored Task 120 runs only; no model fitting, detector reruns, or inferred multi-seed uncertainty."}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(table)} rows to {TABLE}")


if __name__ == "__main__":
    main()
