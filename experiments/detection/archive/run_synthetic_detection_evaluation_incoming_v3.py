"""
V3 synthetic drift-detector evaluation using the daily residual stream.

This is an evaluation runner, not a tuning script.

Protocol
--------
- Synthetic scenarios: none, sudden, gradual, recurring
- Synthetic seeds: 1, 2, 3, 4, 5
- Synthetic length: 20,000 half-hour observations
- Noise SD: 50
- Detector input:
    1. Reconstruct expected Concept 1 demand
    2. Residual = TOTALDEMAND - expected Concept 1 demand
    3. Take absolute residual magnitude
    4. Aggregate 48 half-hour observations to daily mean absolute residual
- Ground-truth changepoints are converted to daily indices
- Metrics are logged through drift_lab.run_logger.log_run()
- Results are appended to results/runs.csv
"""

from pathlib import Path
import sys
import json
import hashlib

import numpy as np
import pandas as pd


# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drift_lab.synthetic_generator import (
    make_feature_series,
    DEFAULT_CONCEPT_1,
)
from drift_lab import detectors
from drift_lab import evaluation
from drift_lab.run_logger import log_run


# Frozen V3 protocol from the notebook
N_SYNTHETIC = 20_000
NOISE_SD = 50
SEEDS = (1, 2, 3, 4, 5)

DRIFT_TYPES = (
    "none",
    "sudden",
    "gradual",
    "recurring",
)

# Daily residual stream: 48 half-hour observations per detector observation.
OBS_PER_DAY = 48

GROUP = "detection"
REGION = "synthetic"
REGIME = "full_test"

SPLIT_ID = (
    "synthetic_20000_daily_abs_residual_concept1_v3"
)

RESULTS_PATH = PROJECT_ROOT / "results" / "runs.csv"


# Detector configurations selected in 04.1_Drift_Detection.ipynb
DETECTOR_CONFIGS = {
    "adwin": {
        "func": detectors.run_adwin,
        "config": {
            "delta": 0.002,
        },
    },

    "page_hinkley": {
        "func": detectors.run_page_hinkley,
        "config": {
            "min_instances": 30,
            "delta": 0.005,
            "threshold": 400,
            "alpha": 0.9999,
        },
    },

    "kswin": {
        "func": detectors.run_kswin,
        "config": {
            "alpha": 0.001,
            "window_size": 170,
            "stat_size": 40,
            "seed": 12,
        },
    },
}


# Helpers
def config_id(config):
    """Create a readable, reproducible configuration identifier."""
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:10]

    readable = ",".join(
        f"{key}={value}"
        for key, value in sorted(config.items())
    )

    return f"{readable}|sha256:{digest}"


def make_daily_residual_stream(data):
    """
    Reproduce the detector input from 04.1_Drift_Detection.ipynb.

    Residual = synthetic TOTALDEMAND - expected Concept 1 demand

    Absolute half-hour residual magnitudes are aggregated into daily
    mean absolute residuals using 48 observations per day.
    """
    expected_concept_1 = (
        DEFAULT_CONCEPT_1["base"]
        + DEFAULT_CONCEPT_1["daily_amplitude"]
        * np.sin(
            2 * np.pi
            * (data["hour"].to_numpy() - 7)
            / 24
        )
        + DEFAULT_CONCEPT_1["weekend_effect"]
        * data["is_weekend"].to_numpy()
    )

    residual = (
        data["TOTALDEMAND"].to_numpy(dtype=float)
        - expected_concept_1
    )

    daily_residual = (
        pd.Series(np.abs(residual))
        .groupby(
            np.arange(len(residual)) // OBS_PER_DAY
        )
        .mean()
        .to_numpy(dtype=float)
    )

    if len(daily_residual) == 0:
        raise RuntimeError(
            "Daily residual detector stream is empty."
        )

    if not np.isfinite(daily_residual).all():
        raise RuntimeError(
            "Daily residual detector stream contains NaN "
            "or infinite values."
        )

    return daily_residual


def to_daily_indices(full_changepoints):
    """Convert half-hour synthetic changepoints to daily detector indices."""
    return [
        int(cp // OBS_PER_DAY)
        for cp in full_changepoints
    ]


def ensure_no_duplicate_recorded_run():
    """
    Abort if this exact V3 protocol is already present in runs.csv.
    """
    path = Path(RESULTS_PATH)

    if not path.exists() or path.stat().st_size == 0:
        return

    existing = pd.read_csv(path)

    if existing.empty:
        return

    required = {
        "group",
        "seed",
        "split_id",
    }

    if not required.issubset(existing.columns):
        return

    duplicate = existing[
        (existing["group"] == GROUP)
        & (existing["split_id"] == SPLIT_ID)
        & (existing["seed"].isin(SEEDS))
    ]

    if not duplicate.empty:
        raise RuntimeError(
            "Matching V3 daily-residual detection results already "
            f"exist in {RESULTS_PATH}. Remove them deliberately "
            "before rerunning this exact protocol."
        )


def evaluate_detection(
    drift_type,
    true_changepoints,
    detections,
    n_observations,
):
    """
    Evaluate detections using the shared project evaluator.
    """

    return evaluation.evaluate_detections(
        detected_changepoints=detections,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=7,
    )
def log_metrics(
    detector_name,
    detector_config,
    drift_type,
    seed,
    metrics,
):
    """Write the three required detector metrics to runs.csv."""
    common = {
        "method": detector_name,
        "dataset": f"synthetic_{drift_type}",
        "region": REGION,
        "regime": REGIME,

        # This protocol does not fit a forecasting model.
        "train_samples": 0,

        "group": GROUP,
        "seed": seed,
        "config_hash": config_id(
            detector_config
        ),
        "split_id": SPLIT_ID,
        "n_retrains": 0,
        "wall_clock_s": 0.0,
        "results_path": str(RESULTS_PATH),
    }

    metric_values = {
        "detection_delay":
            metrics["detection_delay"],

        "false_alarms_per_10000":
            metrics["false_alarms_per_10000"],

        "missed_detections":
            metrics["missed_detections"],
    }

    for metric_name, metric_value in (
        metric_values.items()
    ):
        # Detection delay is not applicable to no drift.
        if (
            metric_name == "detection_delay"
            and np.isnan(metric_value)
        ):
            continue

        log_run(
            metric_name=metric_name,
            metric_value=metric_value,
            **common,
        )


def run_detector(stream, entry):
    """Run one detector with the notebook-selected configuration."""
    config = entry["config"].copy()

    detections = entry["func"](
        stream,
        **config,
    )

    return detections, config


# V3 experiment
def main():
    ensure_no_duplicate_recorded_run()

    rows = []

    print(
        "Running V3 synthetic detector evaluation "
        "on daily mean absolute residual stream"
    )
    print(f"Seeds: {SEEDS}")
    print(
        "Scenarios: none, sudden, gradual, recurring"
    )
    print(
        "Detector input: seasonality-adjusted daily "
        "mean absolute residual"
    )
    print(
        "Forecasting model: NONE "
        "(Concept 1 expectation is reconstructed directly)"
    )
    print(
        f"Half-hour observations per detector day: "
        f"{OBS_PER_DAY}"
    )
    print(f"Split ID: {SPLIT_ID}")
    print(f"Results file: {RESULTS_PATH}")
    print()

    for seed in SEEDS:
        for drift_type in DRIFT_TYPES:

            data, full_truth = (
                make_feature_series(
                    kind=drift_type,
                    n=N_SYNTHETIC,
                    noise=NOISE_SD,
                    seed=seed,
                )
            )

            stream = (
                make_daily_residual_stream(
                    data
                )
            )

            true_changepoints = (
                to_daily_indices(
                    full_truth
                )
            )

            for detector_name, entry in (
                DETECTOR_CONFIGS.items()
            ):
                detections, config = (
                    run_detector(
                        stream,
                        entry,
                    )
                )

                metrics = evaluate_detection(
                    drift_type=drift_type,
                    true_changepoints=
                        true_changepoints,
                    detections=detections,
                    n_observations=len(stream),
                )

                log_metrics(
                    detector_name=detector_name,
                    detector_config=config,
                    drift_type=drift_type,
                    seed=seed,
                    metrics=metrics,
                )

                rows.append({
                    "detector":
                        detector_name,

                    "drift_type":
                        drift_type,

                    "seed":
                        seed,

                    "true_changepoints":
                        true_changepoints,

                    "detections":
                        detections,

                    "detection_delay":
                        metrics[
                            "detection_delay"
                        ],

                    "false_alarms_per_10000":
                        metrics[
                            "false_alarms_per_10000"
                        ],

                    "missed_detections":
                        metrics[
                            "missed_detections"
                        ],

                    "config":
                        config_id(config),
                })

            print(
                f"completed seed={seed}, "
                f"scenario={drift_type}"
            )

    raw = pd.DataFrame(rows)

    grouped = (
        raw
        .groupby(
            [
                "detector",
                "drift_type",
                "config",
            ],
            as_index=False,
        )
        .agg(
            delay_mean=(
                "detection_delay",
                "mean",
            ),

            delay_sd=(
                "detection_delay",
                "std",
            ),

            false_alarms_per_10k_mean=(
                "false_alarms_per_10000",
                "mean",
            ),

            false_alarms_per_10k_sd=(
                "false_alarms_per_10000",
                "std",
            ),

            missed_mean=(
                "missed_detections",
                "mean",
            ),

            missed_sd=(
                "missed_detections",
                "std",
            ),

            seeds=(
                "seed",
                "nunique",
            ),
        )
    )

    print()
    print("=" * 120)
    print(
        "V3 T1-STYLE AGGREGATE SUMMARY "
        "— DAILY ABSOLUTE RESIDUAL STREAM"
    )
    print("=" * 120)

    print(
        grouped.to_string(
            index=False,
            float_format=
                lambda x: f"{x:.3f}",
        )
    )

    print()
    print(
        "Evaluation complete. V3 metric rows "
        f"were appended to {RESULTS_PATH}."
    )


if __name__ == "__main__":
    main()
