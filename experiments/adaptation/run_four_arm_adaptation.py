"""PGS242-128 revision: shared initial state and a four-window D sweep.

Supervisor scope: B remains every 30 days; D compares 30/60/120/180 days.
Detection calibration, seasonal adjustment, warm-up and refractory filtering
remain upstream responsibilities. This runner validates the supplied trigger
sequence, never retunes it, and never selects triggers using event matches.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drift_lab.adaptation import adaptive_forecast_stream, validate_adaptation_changepoints
from drift_lab.data_loader import load_and_split_aemo
from drift_lab.evaluation import calculate_mae
from drift_lab.forecasters import MODEL_REGISTRY
from drift_lab.run_logger import log_run

REGIONS = ("SA1", "NSW1")
WINDOW_DAYS = (30, 60, 120, 180)
DEFAULT_FORECAST_SEEDS = (1, 2, 3, 4, 5)
SUPPORTED_DETECTORS = ("adwin", "page_hinkley", "kswin")
DETECTOR_REFRACTORY_DAYS = 14
DETECTOR_WARMUP_SAMPLES = 2880
# Persisted identifiers retain their original values so existing run logs and
# manifests remain comparable. Python names describe their role in this runner.
ADAPTATION_RUN_GROUP = "adaptive_forecasting_v3_pr29"
ADAPTATION_SPLIT_ID = "aemo_train2018_2019_cal202001_202002_test202003_pr29"
EVENT_COLUMNS = [
    "event_id", "retrain_timestamp", "first_forecast_timestamp", "strategy",
    "region", "window_days", "trigger_type", "detector", "training_window_start",
    "training_window_end", "training_samples", "effective_training_samples",
    "wall_clock_s", "model_copy_s", "arm", "configuration", "seed", "run_id",
    "config_hash",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configurations(windows=WINDOW_DAYS):
    """Revision R2: keep B's frequency independent of D's history length."""
    if not windows or len(set(windows)) != len(windows) or any(
        isinstance(w, bool) or not isinstance(w, int) or w <= 0 for w in windows
    ):
        raise ValueError("D windows must be unique positive integer day counts")
    return [
        ("A", "never", None), ("B", "every_30_days", None),
        ("C", "drift_full_history", None),
    ] + [("D", "drift_recent_window", w) for w in windows]


def validate_experiment_decisions(decisions_path):
    """Validate the documented detection and four-arm experiment decisions.

    Revision R8: 128 consumes accepted upstream alarms and never reruns,
    retunes, deseasonalises, standardises, warms, or refractory-filters a
    detector. The decisions file is hashed in the resulting run manifest.
    """
    decisions_path = Path(decisions_path)
    if not decisions_path.is_file():
        raise FileNotFoundError(f"Missing shared decisions file: {decisions_path}")
    text = decisions_path.read_text(encoding="utf-8")
    required = (
        "**COVID-19 event:** 16 Mar 2020 – 17 May 2020.",
        "Five-Minute Settlement (1 Oct 2021):** removed from the target event catalogue",
        "Deseasonalisation parameters will be fitted using **TRAIN only**.",
        "Detector inputs will be **Z-score standardised using TRAIN only**.",
        "**five seeds: 1–5**",
        "Agreed refractory period: **14 days**.",
        "**Arm B — Periodic Retraining:** retrain at a fixed **30-day interval**.",
        "Evaluate training windows of **30, 60, 120, and 180 days**.",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError("Shared decisions file does not contain all PR #29 settings")
    # The document calls the Calibration warm-up proposal pending confirmation.
    # Preserve that qualification rather than silently promoting it to final.
    if "exact warm-up period remains to be confirmed" not in text:
        raise ValueError("The PR #29 warm-up qualification is missing from decisions")
    return decisions_path.resolve()


def validate_frozen_detector_changepoints(changepoints, test, *, region, detector):
    """Validate frozen detector provenance before it reaches Arms C and D.

    The checks deliberately reject old Task 120 output. They verify that the
    stored sequence already carries the agreed preprocessing and 14-day
    refractory metadata. This function never changes accepted detections.
    """
    if detector not in SUPPORTED_DETECTORS:
        raise ValueError(f"PR #29 detector must be one of {SUPPORTED_DETECTORS}")
    required = {
        "region", "detector", "detected_index", "SETTLEMENTDATE", "detector_config",
        "warmup_samples", "test_samples", "refractory_period", "refractory_days",
        "seasonal_profile", "seasonal_reference", "error_zscore_reference",
        "warmup_reference", "detector_input", "whitening_type",
    }
    missing = required - set(changepoints.columns)
    if missing:
        raise ValueError(f"Changepoints do not have PR #29 provenance fields: {sorted(missing)}")
    selected = validate_adaptation_changepoints(
        changepoints, test, region=region, detector=detector
    )
    if selected.empty:
        raise ValueError(f"PR #29 has no stored {detector}/{region} run")
    expected = {
        "warmup_samples": DETECTOR_WARMUP_SAMPLES,
        "test_samples": len(test),
        "refractory_period": DETECTOR_REFRACTORY_DAYS * 48,
        "refractory_days": DETECTOR_REFRACTORY_DAYS,
        "seasonal_reference": "complete_TRAIN_only",
        "warmup_reference": "processed CALIBRATION forecast errors",
        "detector_input": "TRAIN_zscored_absolute_forecast_error",
    }
    for column, value in expected.items():
        if not selected[column].eq(value).all():
            raise ValueError(f"Unexpected PR #29 metadata in {column} for {detector}/{region}")
    if not selected.seasonal_profile.str.contains("smoothed_DOY", regex=False).all():
        raise ValueError("PR #29 seasonal profile is missing its DOY component")
    if not selected.seasonal_profile.str.contains("DOWxhalf_hour", regex=False).all():
        raise ValueError("PR #29 seasonal profile is missing its DOW x half-hour component")
    if not selected.error_zscore_reference.str.contains("TRAIN", regex=False).all():
        raise ValueError("PR #29 error standardisation is not training-only")
    if selected.detector_config.map(json.loads).map(type).ne(dict).any():
        raise ValueError("PR #29 detector configuration is not JSON object data")
    if selected.detector_config.nunique() != 1:
        raise ValueError(f"Mixed detector configuration within {detector}/{region}")
    return selected


def run_regional_four_arm_comparison(
        changepoints, train, calibration, test, *, region, model_factory,
        detectors=SUPPORTED_DETECTORS, windows=WINDOW_DAYS, seed=0):
    """Compare all four adaptation arms for one region and forecasting seed.

    Revision R9: the accepted detector outputs are experimental scenarios, not
    candidates for retuning inside 128.  A and B are detector-independent, so
    they run once.  For each detector, C and every D window consume precisely
    the same frozen sequence validated from the PR #29 artifact.
    """
    configurations(windows)
    detectors = tuple(detectors)
    if not detectors or len(set(detectors)) != len(detectors) or any(
        detector not in SUPPORTED_DETECTORS for detector in detectors
    ):
        raise ValueError(f"detectors must be a unique non-empty subset of {SUPPORTED_DETECTORS}")

    selected = {
        detector: validate_frozen_detector_changepoints(
            changepoints, test, region=region, detector=detector
        )
        for detector in detectors
    }
    started = perf_counter()
    initial_model = model_factory().fit(train)
    initial_fit_s = perf_counter() - started
    forecasts, event_rows, summaries = [], [], []
    reference_predictions = None
    schedules = {}

    def execute(arm, strategy, *, detector=None, days=None):
        nonlocal reference_predictions
        configuration = (
            arm if detector is None else
            (f"{arm}_{detector}" if arm == "C" else f"D_{days}d_{detector}")
        )
        source = selected[detector] if detector else changepoints.iloc[0:0].copy()
        started_arm = perf_counter()
        stream, metadata = adaptive_forecast_stream(
            source, initial_model, train, calibration, test,
            region=region, strategy=strategy, detector=detector,
            recent_window=f"{days}D" if days else "90D",
            initial_model_fitted=True,
        )
        elapsed = perf_counter() - started_arm
        stream = stream.assign(
            arm=arm, configuration=configuration, window_days=days,
            region=region, detector=detector or "", seed=seed,
        )
        if arm == "A":
            reference_predictions = stream.y_pred.to_numpy()
        else:
            first = (metadata["adaptation_times"][0]
                     if metadata["adaptation_times"] else test.SETTLEMENTDATE.iloc[-1])
            before_first_update = pd.DatetimeIndex(stream.SETTLEMENTDATE) <= first
            np.testing.assert_allclose(
                stream.y_pred.to_numpy()[before_first_update],
                reference_predictions[before_first_update], rtol=0, atol=1e-10,
            )
        if arm == "C":
            schedules[detector] = metadata["adaptation_times"]
        if arm == "D" and metadata["adaptation_times"] != schedules[detector]:
            raise RuntimeError("C and D must use exactly the same PR #29 trigger sequence")
        for event in metadata["retraining_events"]:
            event_rows.append({
                **event, "arm": arm, "configuration": configuration,
                "seed": seed,
            })
        record = {
            "arm": arm, "configuration": configuration, "strategy": strategy,
            "region": region, "window_days": days, "seed": seed,
            "detector": detector or "", "n_observations": len(stream),
            "overall_mae": calculate_mae(stream.y_true, stream.y_pred),
            **{key: metadata[key] for key in (
                "n_retrains", "train_samples_initial", "train_samples_total",
                "train_samples_last", "effective_train_samples_total",
                "retraining_wall_clock_s")},
            "initial_fit_s": initial_fit_s, "arm_wall_clock_s": elapsed,
        }
        forecasts.append(stream)
        summaries.append(record)
        print(f"{region}/{configuration}: MAE={record['overall_mae']:.3f}, "
              f"retrains={record['n_retrains']}", flush=True)

    # Baselines are deliberately executed just once per region.
    execute("A", "never")
    execute("B", "every_30_days")
    for detector in detectors:
        execute("C", "drift_full_history", detector=detector)
        for days in windows:
            execute("D", "drift_recent_window", detector=detector, days=days)
    return (pd.concat(forecasts, ignore_index=True), pd.DataFrame(event_rows),
            pd.DataFrame(summaries))


def run_four_arm_adaptation_experiment(
        *, detections_path, decisions_path, run_id, detectors=SUPPORTED_DETECTORS,
        seed=0, windows=WINDOW_DAYS, data_path=ROOT / "data/raw",
        output_root=ROOT / "results/tables/adaptation/revised",
        shared_runs=ROOT / "results/tables/runs.csv"):
    """Run the four-arm experiment across regions for one forecasting seed.

    The function intentionally has no event-catalogue parameter: Arms C/D
    retrain from every accepted upstream detector alarm, including alarms that
    T2 later describes as unmatched contextual changes.  Event matching is not
    a selection mechanism for adaptation triggers.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("run_id must be a short filename-safe identifier")
    configurations(windows)
    detectors = tuple(detectors)
    if not detectors or len(set(detectors)) != len(detectors) or any(
        detector not in SUPPORTED_DETECTORS for detector in detectors
    ):
        raise ValueError(f"detectors must be a unique non-empty subset of {SUPPORTED_DETECTORS}")
    output = Path(output_root) / run_id
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    detections_path = Path(detections_path).resolve()
    decisions_path = validate_experiment_decisions(decisions_path)
    rows = pd.read_csv(detections_path)
    sources = [
        Path(__file__), ROOT / "src/drift_lab/adaptation.py",
        ROOT / "src/drift_lab/forecasters.py", ROOT / "src/drift_lab/data_loader.py",
        ROOT / "src/drift_lab/evaluation.py", ROOT / "src/drift_lab/run_logger.py",
        detections_path, decisions_path,
    ]
    for region in REGIONS:
        sources.extend(sorted((Path(data_path) / region).glob("*.csv")))
    hashes = {str(path.resolve()): sha256(path) for path in sources}
    spec = MODEL_REGISTRY["gradient_boosted"]
    model_config = {**spec["config"], "seed": seed}
    identity = {
        "source_sha256": hashes, "model_config": model_config,
        "detectors": list(detectors), "windows": list(windows),
        "split_id": ADAPTATION_SPLIT_ID, "upstream_detection_artifact": str(detections_path),
        "decisions_file": str(decisions_path), "execution_design":
            "A/B_once_per_region; C/D_per_detector; D_window_sensitivity",
    }
    config_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    prepared = []
    for region in REGIONS:
        _, train, calibration, test = load_and_split_aemo(region, data_path)
        # Validate every region/detector provenance before writing any result.
        for detector in detectors:
            validate_frozen_detector_changepoints(rows, test, region=region, detector=detector)
        prepared.append((region, train, calibration, test))
    split = f"{ADAPTATION_SPLIT_ID}|run={run_id}"
    if Path(shared_runs).exists():
        previous = pd.read_csv(shared_runs, usecols=["group", "split_id"])
        if (previous.group.eq(ADAPTATION_RUN_GROUP) & previous.split_id.eq(split)).any():
            raise ValueError("This PR #29 run ID already exists in the shared log")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        **identity, "run_id": run_id, "config_hash": config_hash, "status": "running",
        "model": "gradient_boosted", "seed": seed,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "log_regime": "full-test", "calibration_training": False,
        "window_boundary": "both_closed: [t - window, t]",
        "same_changepoints_for_c_and_all_d_within_detector": True,
        "upstream_processing_verified": {
            "train_only_deseasonalisation": True,
            "train_only_zscore": True,
            "warmup_samples": DETECTOR_WARMUP_SAMPLES,
            "refractory_days": DETECTOR_REFRACTORY_DAYS,
        },
        "review_note": (
            "The shared decisions file records the Calibration warm-up proposal "
            "as pending confirmation. This run records that qualification and "
            "does not represent it as a newly finalised 128 decision."
        ),
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = None
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = perf_counter()
    summaries, all_events = [], []
    try:
        for region, train, calibration, test in prepared:
            forecasts, events, summary = run_regional_four_arm_comparison(
                rows, train, calibration, test, region=region,
                model_factory=lambda: spec["class"](**model_config),
                detectors=detectors, windows=windows, seed=seed,
            )
            for table in (forecasts, events, summary):
                table["run_id"] = run_id
                table["config_hash"] = config_hash
            forecasts.to_csv(output / f"forecasts_{region}.csv.gz", index=False,
                             compression={"method": "gzip", "mtime": 0})
            summaries.append(summary)
            all_events.append(events)
        summary = pd.concat(summaries, ignore_index=True)
        events = pd.concat(all_events, ignore_index=True).reindex(columns=EVENT_COLUMNS)
        events.to_csv(output / "retraining_events.csv", index=False)
        summary.to_csv(output / "summary.csv", index=False)
        summary.loc[summary.arm.eq("D")].to_csv(output / "window_sensitivity.csv", index=False)
        summary.loc[summary.arm.isin(["C", "D"])].to_csv(
            output / "detector_comparison.csv", index=False)
        if any(sha256(path) != digest for path, digest in hashes.items()):
            raise RuntimeError("An input or source file changed during this run")
        for record in summary.to_dict("records"):
            common = dict(
                method=record["configuration"], dataset="AEMO_TOTALDEMAND",
                region=record["region"], group=ADAPTATION_RUN_GROUP, regime="full-test",
                seed=seed, config_hash=config_hash, split_id=split,
                train_samples=record["train_samples_total"],
                n_retrains=record["n_retrains"], wall_clock_s=record["arm_wall_clock_s"],
            )
            for metric in ("overall_mae", "n_retrains", "train_samples_total",
                           "retraining_wall_clock_s"):
                for log_path in (output / "runs.csv", Path(shared_runs)):
                    log_run(**common, metric_name=metric,
                            metric_value=record[metric], results_path=log_path)
        manifest.update(
            status="complete", elapsed_s=perf_counter() - started,
            outputs_sha256={
                path.name: sha256(path) for path in output.iterdir()
                if path.is_file() and path != manifest_path
            },
        )
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return summary

def run_multi_seed_adaptation_experiment(
    *,
    detections_path,
    decisions_path,
    run_id,
    seeds=DEFAULT_FORECAST_SEEDS,
    detectors=SUPPORTED_DETECTORS,
    windows=WINDOW_DAYS,
    data_path=ROOT / "data/raw",
    output_root=ROOT / "results/tables/adaptation/revised",
    shared_runs=ROOT / "results/tables/runs.csv",
):
    """
    Run the four-arm adaptation experiment once for each forecasting seed.

    Each seed gets its own output subdirectory:
    <output_root>/<run_id>/seed_<seed>/
    """
    seeds = tuple(seeds)

    if not seeds:
        raise ValueError("At least one forecasting seed is required.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Forecast seeds must be unique.")

    if any(
        isinstance(seed, bool)
        or not isinstance(seed, int)
        for seed in seeds
    ):
        raise ValueError(
            "Forecast seeds must be integer values."
        )

    parent_output = Path(output_root) / run_id

    if parent_output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run: {parent_output}"
        )

    parent_output.mkdir(parents=True, exist_ok=False)

    summaries = []

    try:
        for seed in seeds:
            seed_run_id = f"{run_id}_seed{seed}"

            summary = run_four_arm_adaptation_experiment(
                detections_path=detections_path,
                decisions_path=decisions_path,
                run_id=seed_run_id,
                detectors=detectors,
                seed=seed,
                windows=windows,
                data_path=data_path,
                output_root=parent_output,
                shared_runs=shared_runs,
            )

            seed_output = parent_output / seed_run_id
            canonical_output = parent_output / f"seed_{seed}"

            if not seed_output.is_dir():
                raise RuntimeError(
                    f"Expected seed output directory was not created: "
                    f"{seed_output}"
                )

            if canonical_output.exists():
                raise FileExistsError(
                    f"Refusing to overwrite seed directory: "
                    f"{canonical_output}"
                )

            seed_output.rename(canonical_output)

            summary = summary.copy()
            summary["seed"] = seed
            summaries.append(summary)

        combined = pd.concat(
            summaries,
            ignore_index=True,
        )

        combined.to_csv(
            parent_output / "summary_all_seeds.csv",
            index=False,
        )

        manifest = {
            "run_id": run_id,
            "seeds": list(seeds),
            "seed_run_ids": [
                f"{run_id}_seed{seed}"
                for seed in seeds
            ],
            "status": "complete",
        }

        (parent_output / "multi_seed_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        return combined

    except Exception:
        manifest = {
            "run_id": run_id,
            "seeds": list(seeds),
            "status": "failed",
        }

        (parent_output / "multi_seed_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--detectors", nargs="+", choices=SUPPORTED_DETECTORS,
                        default=list(SUPPORTED_DETECTORS),
                        help="Frozen PR #29 detector scenarios for C/D")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--decisions", type=Path, required=True,
                        help="Shared PR #29 decisions.md")
    parser.add_argument("--seeds",nargs="+", type=int, default=list(DEFAULT_FORECAST_SEEDS), help="Forecast model seeds. Default: 1 2 3 4 5.")
    parser.add_argument("--windows", nargs="+", type=int, default=list(WINDOW_DAYS))
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/tables/adaptation/revised")
    args = parser.parse_args()
    run_multi_seed_adaptation_experiment(
    detections_path=args.detections,
    decisions_path=args.decisions,
    run_id=args.run_id,
    detectors=tuple(args.detectors),
    seeds=tuple(args.seeds),
    windows=tuple(args.windows),
    data_path=args.data_path,
    output_root=args.output_root,
)


if __name__ == "__main__":
    main()
