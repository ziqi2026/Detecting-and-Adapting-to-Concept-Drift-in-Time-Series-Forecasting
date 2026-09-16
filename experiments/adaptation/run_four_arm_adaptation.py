"""Run the Sprint 4 four-arm adaptive forecasting experiment.

The runner uses one frozen forecasting configuration, the fixed AEMO split,
and one Task-120 detector sequence for both drift-triggered arms.  It writes
timestamp-level forecasts and separate retraining-event metadata so later
performance and adaptation-cost analyses do not need to rerun the experiment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drift_lab.adaptation import adaptive_forecast_stream
from drift_lab.data_loader import load_and_split_aemo
from drift_lab.evaluation import AEMO_TIMEZONE, build_aemo_event_windows, load_aemo_events, calculate_mae
from drift_lab.forecasters import MODEL_REGISTRY, make_forecasters
from drift_lab.run_logger import log_run

REGIONS = ("SA1", "NSW1")
ARMS = {
    "arm_a_never": "never",
    "arm_b_every_30_days": "every_30_days",
    "arm_c_drift_full_history": "drift_full_history",
    "arm_d_drift_recent_window": "drift_recent_window",
}
DATA_PATH = ROOT / "data" / "raw"
DETECTIONS_PATH = ROOT / "results" / "tables" / "detection" / "aemo_detection_changepoints.csv"
EVENTS_PATH = ROOT / "data" / "events.csv"
OUT = ROOT / "results" / "tables" / "adaptation"
FORECASTS_PATH = OUT / "four_arm_forecasts.csv"
RETRAIN_PATH = OUT / "four_arm_retraining_events.csv"
SUMMARY_PATH = OUT / "four_arm_summary.csv"
MANIFEST_PATH = OUT / "four_arm_manifest.json"
RUNS_PATH = ROOT / "results" / "tables" / "runs.csv"
GROUP = "adaptive_forecasting"
SPLIT_ID = "aemo_fixed_train2018_2019_test202003_sprint4_v1"
RECENT_WINDOW = "90D"


def _config_hash(model_name, detector, detector_config):
    payload = json.dumps({"model": model_name, "detector": detector,
                          "detector_config": detector_config,
                          "recent_window": RECENT_WINDOW}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _regime_mae(stream, event, regime):
    windows = build_aemo_event_windows(event, regime_days=7)
    start, end = windows[regime]
    timestamps = pd.DatetimeIndex(pd.to_datetime(stream["SETTLEMENTDATE"])).tz_localize(AEMO_TIMEZONE)
    rows = stream.loc[(timestamps >= start) & (timestamps < end)]
    return calculate_mae(rows.y_true, rows.y_pred) if len(rows) else np.nan


def _log_metric_once(**kwargs):
    """Make reruns idempotent after an interrupted experiment."""
    keys = ["group", "method", "dataset", "region", "config_hash",
            "split_id", "seed", "metric_name", "regime"]
    if RUNS_PATH.exists():
        existing = pd.read_csv(RUNS_PATH)
        mask = np.ones(len(existing), dtype=bool)
        for key in keys:
            mask &= existing[key].astype(str).eq(str(kwargs[key])).to_numpy()
        if mask.any():
            return
    log_run(**kwargs)


def run_experiment(*, model_name="gradient_boosted", detector="page_hinkley"):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    detections = pd.read_csv(DETECTIONS_PATH)
    required = {"region", "detector", "detected_index", "SETTLEMENTDATE"}
    if not required.issubset(detections.columns):
        raise ValueError(f"Task 120 detections missing: {sorted(required - set(detections.columns))}")
    if detector not in set(detections.detector):
        raise ValueError(f"No Task 120 detections found for detector {detector!r}")
    events = load_aemo_events(EVENTS_PATH)
    model_config = MODEL_REGISTRY[model_name]["config"]
    config_hash = _config_hash(model_name, detector, model_config)
    all_forecasts, all_retraining, summaries = [], [], []
    started_total = perf_counter()

    for region in REGIONS:
        _, train, calibration, test = load_and_split_aemo(region, DATA_PATH)
        region_cp = detections.loc[detections.region == region].copy()
        region_cp["SETTLEMENTDATE"] = pd.to_datetime(region_cp["SETTLEMENTDATE"])
        region_cp = region_cp.loc[region_cp.detector == detector].sort_values("SETTLEMENTDATE")
        for arm, strategy in ARMS.items():
            model = make_forecasters()[model_name]
            started = perf_counter()
            stream, metadata = adaptive_forecast_stream(
                region_cp, model, train, calibration, test,
                region=region, strategy=strategy, detector=detector,
                recent_window=RECENT_WINDOW,
            )
            elapsed = perf_counter() - started
            stream["arm"] = arm
            stream["region"] = region
            stream["model"] = model_name
            stream["detector"] = detector
            all_forecasts.append(stream)
            for event in events.loc[events.tier == 1].to_dict("records"):
                if event["region"] not in ("NEM", region):
                    continue
                for regime in ("pre-drift", "drift", "post-drift"):
                    value = _regime_mae(stream, event, regime)
                    if np.isnan(value):
                        continue
                    _log_metric_once(method=arm, dataset="AEMO_TOTALDEMAND", region=region,
                            metric_name=f"mae.{event['event_id']}", metric_value=value,
                            regime=regime, train_samples=metadata["train_samples_last"],
                            group=GROUP, seed=0, config_hash=config_hash,
                            split_id=f"{SPLIT_ID}|event={event['event_id']}",
                            n_retrains=metadata["n_retrains"], wall_clock_s=elapsed,
                            results_path=RUNS_PATH)
            summary = {
                "arm": arm, "strategy": strategy, "region": region,
                "model": model_name, "detector": detector,
                "n_observations": len(stream), "overall_mae": calculate_mae(stream.y_true, stream.y_pred),
                "n_retrains": metadata["n_retrains"],
                "train_samples_initial": metadata["train_samples_initial"],
                "train_samples_total": metadata["train_samples_total"],
                "train_samples_last": metadata["train_samples_last"],
                "wall_clock_s": elapsed,
            }
            summaries.append(summary)
            for event in metadata["retraining_events"]:
                all_retraining.append({"arm": arm, "model": model_name, **event})
            print(f"{arm}/{region}: MAE={summary['overall_mae']:.3f}, retrains={summary['n_retrains']}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(all_forecasts, ignore_index=True).to_csv(FORECASTS_PATH, index=False)
    pd.DataFrame(all_retraining).to_csv(RETRAIN_PATH, index=False)
    pd.DataFrame(summaries).to_csv(SUMMARY_PATH, index=False)
    manifest = {
        "group": GROUP, "split_id": SPLIT_ID, "regions": list(REGIONS),
        "arms": ARMS, "model": model_name, "model_config": model_config,
        "detector": detector, "recent_window": RECENT_WINDOW,
        "calibration_used_for_training": False,
        "same_changepoint_sequence_for_arms_c_d": True,
        "source_files": {"detections": str(DETECTIONS_PATH.relative_to(ROOT)),
                         "events": str(EVENTS_PATH.relative_to(ROOT))},
        "elapsed_s": perf_counter() - started_total,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gradient_boosted", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--detector", default="page_hinkley")
    args = parser.parse_args()
    run_experiment(model_name=args.model, detector=args.detector)
    print(f"Saved forecasts to {FORECASTS_PATH}")


if __name__ == "__main__":
    main()
