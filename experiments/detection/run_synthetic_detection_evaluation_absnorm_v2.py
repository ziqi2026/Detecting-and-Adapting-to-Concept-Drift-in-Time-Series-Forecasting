"""
Multi-seed synthetic drift-detector evaluation.

This runner executes the frozen detector design selected in the tuning
notebook. It is not a tuning script.

Protocol
--------
- Synthetic scenarios: none, sudden, gradual, recurring
- Synthetic seeds: 1, 2, 3, 4, 5
- Split: 4000 train / 1000 calibration / 15000 test observations
- Forecaster: frozen Gradient Boosting with calendar + lags (48, 336)
- Detector input: calibration-normalized absolute forecast error
- Evaluation tolerance: 336 observations
- Detectors/configurations: read from DETECTOR_REGISTRY in detectors.py
- Metrics are logged through the shared run logger to results/runs.csv

Update DETECTOR_REGISTRY in detectors.py with the frozen configurations
selected in the tuning notebook before running this file.
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

from drift_lab.synthetic_generator import make_feature_series
from drift_lab.detectors import DETECTOR_REGISTRY
from drift_lab.evaluation import evaluate_detections
from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)
from drift_lab.run_logger import log_run


# Frozen evaluation protocol
N_OBSERVATIONS = 20_000
TRAIN_END = 4_000
CALIBRATION_END = 5_000
TEST_START = 5_000

SEEDS = (1, 2, 3, 4, 5)
FORECASTER_SEED = 0
TOLERANCE = 48 * 7

DRIFT_TYPES = (
    "none",
    "sudden",
    "gradual",
    "recurring",
)

GROUP = "detection"
REGION = "synthetic"
REGIME = "full_test"

# New protocol ID: do not mix these rows with the earlier raw-error experiment.
SPLIT_ID = "synthetic_4000_1000_15000_tol336_absnorm_gb_lags_v2"
RESULTS_PATH = PROJECT_ROOT / "results" / "runs.csv"

# Helpers
def config_id(config):
    """Create a readable, reproducible detector configuration identifier."""
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


def to_test_relative(full_changepoints):
    """Convert full-series ground-truth indices to test-stream indices."""
    relative = []

    for cp in full_changepoints:
        if not TEST_START <= cp < N_OBSERVATIONS:
            raise ValueError(
                f"Ground-truth changepoint {cp} lies outside "
                "the detector test stream."
            )

        relative.append(int(cp - TEST_START))

    return relative


def build_normalized_absolute_error_stream(data):
    """
    Build the exact detector input selected in the tuning notebook.

    1. Fit the frozen Gradient Boosting forecaster on the training split.
    2. Predict the calibration split and estimate mean/std of absolute error.
    3. Predict the test split.
    4. Convert test residuals to absolute errors.
    5. Standardize test absolute errors using calibration-only statistics.

    Returns
    -------
    stream : np.ndarray
        Calibration-normalized absolute forecast-error test stream.
    train_samples : int
        Number of forecaster training observations.
    calibration_mean : float
        Mean absolute calibration error.
    calibration_std : float
        Standard deviation of absolute calibration error.
    """
    train = data.iloc[:TRAIN_END].copy()
    calibration = data.iloc[
        TRAIN_END:CALIBRATION_END
    ].copy()
    test = data.iloc[TEST_START:].copy()

    model = GradientBoostedForecaster(
        feature_mode="calendar+lags",
        lags=(48, 336),
        seed=FORECASTER_SEED,
    )
    model.fit(train)

    calibration_errors = frozen_error_stream(
        model,
        calibration,
        history_df=train,
    )

    calibration_abs = (
        calibration_errors["error"]
        .abs()
        .to_numpy(dtype=float)
    )

    calibration_mean = float(calibration_abs.mean())
    calibration_std = float(calibration_abs.std())

    if not np.isfinite(calibration_mean):
        raise RuntimeError(
            "Calibration absolute-error mean is not finite."
        )

    if not np.isfinite(calibration_std) or calibration_std <= 0:
        raise RuntimeError(
            "Calibration absolute-error standard deviation must be "
            "finite and greater than zero."
        )

    test_errors = frozen_error_stream(
        model,
        test,
        history_df=calibration,
    )

    test_abs = (
        test_errors["error"]
        .abs()
        .to_numpy(dtype=float)
    )

    stream = (
        test_abs - calibration_mean
    ) / calibration_std

    if len(stream) != len(test):
        raise RuntimeError(
            "Detector stream length does not match test length."
        )

    if not np.isfinite(stream).all():
        raise RuntimeError(
            "Detector stream contains NaN or infinite values."
        )

    return (
        stream,
        len(train),
        calibration_mean,
        calibration_std,
    )


def ensure_no_duplicate_recorded_run():
    """
    Abort if this exact final protocol already exists in runs.csv.

    This prevents accidental duplicate logging when the runner is executed
    more than once.
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
            "Matching final synthetic detection results already exist in "
            f"{RESULTS_PATH}. Remove them deliberately before rerunning "
            "this exact protocol."
        )


def log_metrics(
    detector_name,
    detector_config,
    drift_type,
    seed,
    metrics,
    train_samples,
):
    """Write the three required detection metrics using the shared logger."""
    common = {
        "method": detector_name,
        "dataset": f"synthetic_{drift_type}",
        "region": REGION,
        "regime": REGIME,
        "train_samples": train_samples,
        "group": GROUP,
        "seed": seed,
        "config_hash": config_id(detector_config),
        "split_id": SPLIT_ID,
        "n_retrains": 0,
        "wall_clock_s": 0.0,
        "results_path": str(RESULTS_PATH),
    }

    metric_values = {
        "detection_delay": metrics["detection_delay"],
        "false_alarms_per_10000": metrics[
            "false_alarms_per_10000"
        ],
        "missed_detections": metrics["missed_detections"],
    }

    for metric_name, metric_value in metric_values.items():
        # Detection delay is not applicable to the no-drift control.
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
    """Run one registered detector with its frozen registry configuration."""
    config = entry["config"].copy()

    changepoints = entry["func"](
        stream,
        **config,
    )

    return changepoints, config

# Run experiment
def main():
    ensure_no_duplicate_recorded_run()

    rows = []

    print("Running FINAL synthetic detector evaluation")
    print(f"Seeds: {SEEDS}")
    print("Scenarios: none, sudden, gradual, recurring")
    print(
        "Forecaster: frozen Gradient Boosting, "
        "calendar+lags (48, 336)"
    )
    print(
        "Detector input: calibration-normalized absolute forecast error"
    )
    print(f"Tolerance: {TOLERANCE} observations")
    print(f"Split ID: {SPLIT_ID}")
    print(f"Results file: {RESULTS_PATH}")
    print()

    for seed in SEEDS:
        for drift_type in DRIFT_TYPES:
            data, full_truth = make_feature_series(
                kind=drift_type,
                n=N_OBSERVATIONS,
                seed=seed,
            )

            truth = to_test_relative(full_truth)

            (
                stream,
                train_samples,
                calibration_mean,
                calibration_std,
            ) = build_normalized_absolute_error_stream(data)

            for detector_name, entry in DETECTOR_REGISTRY.items():
                detected, config = run_detector(
                    stream,
                    entry,
                )

                metrics = evaluate_detections(
                    detected_changepoints=detected,
                    true_changepoints=truth,
                    n_observations=len(stream),
                    drift_type=drift_type,
                    tolerance=TOLERANCE,
                )

                log_metrics(
                    detector_name=detector_name,
                    detector_config=config,
                    drift_type=drift_type,
                    seed=seed,
                    metrics=metrics,
                    train_samples=train_samples,
                )

                rows.append(
                    {
                        "detector": detector_name,
                        "drift_type": drift_type,
                        "seed": seed,
                        "detection_delay": metrics[
                            "detection_delay"
                        ],
                        "false_alarms_per_10000": metrics[
                            "false_alarms_per_10000"
                        ],
                        "missed_detections": metrics[
                            "missed_detections"
                        ],
                        "config": config_id(config),
                        "calibration_abs_error_mean": calibration_mean,
                        "calibration_abs_error_std": calibration_std,
                    }
                )

            print(
                f"completed seed={seed}, scenario={drift_type}"
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
    print("T1-STYLE FINAL AGGREGATE SUMMARY")
    print("=" * 120)
    print(
        grouped.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print(
        "Evaluation complete. Final metric rows were appended to "
        f"{RESULTS_PATH}."
    )


if __name__ == "__main__":
    main()
