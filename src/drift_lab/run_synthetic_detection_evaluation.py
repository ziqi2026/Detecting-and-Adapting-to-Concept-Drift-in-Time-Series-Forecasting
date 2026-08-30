"""
Multi-seed synthetic drift-detector evaluation.

Runs the integrated synthetic benchmark across four drift scenarios and
all registered detectors, records the required metrics through the shared
run logger, and prints a T1-style aggregate summary.

The seed tuple is defined in one place so it can be replaced if the team
confirms a different fixed set.
"""

from pathlib import Path
import sys
import json
import hashlib

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from synthetic_generator import make_feature_series
from drift_lab.detectors import DETECTOR_REGISTRY
from drift_lab.evaluation import evaluate_detections
from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)
from drift_lab.run_logger import log_run


N_OBSERVATIONS = 20000
TRAIN_END = 4000
CALIBRATION_END = 5000
TEST_START = 5000

# Fixed seeds used for the synthetic detector evaluation.
SEEDS = (0, 1, 2, 3, 4)

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
# Metrics are evaluated over the complete detector test stream.
REGIME = "full_test"
SPLIT_ID = "synthetic_4000_1000_15000_tol336_v1"
RESULTS_PATH = "results/runs.csv"


def config_id(config):
    """
    Create a readable, reproducible detector configuration identifier.
    """
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
    """
    Convert full-series ground-truth indices to detector-test indices.
    """
    relative = []

    for cp in full_changepoints:
        if not TEST_START <= cp < N_OBSERVATIONS:
            raise ValueError(
                f"Ground-truth changepoint {cp} lies outside "
                f"the detector test stream."
            )

        relative.append(
            int(cp - TEST_START)
        )

    return relative


def build_error_stream(data):
    """
    Fit the frozen calendar-only Gradient Boosting baseline and return
    the chronological test residual stream.
    """
    train = data.iloc[:TRAIN_END].copy()

    calibration = data.iloc[
        TRAIN_END:CALIBRATION_END
    ].copy()

    test = data.iloc[
        TEST_START:
    ].copy()

    model = GradientBoostedForecaster(
        feature_mode="calendar",
        seed=FORECASTER_SEED,
    )

    model.fit(train)

    errors = frozen_error_stream(
        model,
        test,
        history_df=calibration,
    )

    values = errors["error"].to_numpy(
        dtype=float
    )

    if len(values) != len(test):
        raise RuntimeError(
            "Error stream length does not match test length."
        )

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Error stream contains NaN or infinite values."
        )

    return errors, len(train)


def ensure_no_duplicate_recorded_run():
    """
    Abort if the same evaluation protocol and seed set already exist in
    runs.csv. This prevents accidental duplicate logging on reruns.
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
            "Matching synthetic detection results already exist in "
            "results/runs.csv. Remove the duplicate run deliberately "
            "or change SPLIT_ID before rerunning."
        )


def log_metrics(
    detector_name,
    detector_config,
    drift_type,
    seed,
    metrics,
    train_samples,
):
    """
    Write the three required detection metrics using the shared logger.
    """
    common = {
        "method": detector_name,
        "dataset": f"synthetic_{drift_type}",
        "region": REGION,
        "regime": REGIME,
        "train_samples": train_samples,
        "group": GROUP,
        "seed": seed,
        "config_hash": config_id(
            detector_config
        ),
        "split_id": SPLIT_ID,
        "n_retrains": 0,
        "wall_clock_s": 0.0,
        "results_path": RESULTS_PATH,
    }

    metric_values = {
        "detection_delay": metrics[
            "detection_delay"
        ],
        "false_alarms_per_10000": metrics[
            "false_alarms_per_10000"
        ],
        "missed_detections": metrics[
            "missed_detections"
        ],
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
    """
    Run one registered detector with its recorded configuration.
    """
    config = entry["config"].copy()

    changepoints = entry["func"](
        stream,
        **config,
    )

    return changepoints, config


def main():
    ensure_no_duplicate_recorded_run()

    rows = []

    print(
        "Running multi-seed synthetic detector evaluation"
    )
    print(
        f"Seeds: {SEEDS}"
    )
    print(
        "Scenarios: none, sudden, gradual, recurring"
    )
    print(
        "Forecaster: frozen Gradient Boosting, calendar-only"
    )
    print(
        f"Tolerance: {TOLERANCE} observations"
    )
    print(
        f"Results file: {RESULTS_PATH}"
    )

    for seed in SEEDS:
        for drift_type in DRIFT_TYPES:

            data, full_truth = make_feature_series(
                kind=drift_type,
                n=N_OBSERVATIONS,
                seed=seed,
            )

            truth = to_test_relative(
                full_truth
            )

            errors, train_samples = (
                build_error_stream(
                    data
                )
            )

            stream = errors[
                "error"
            ].to_numpy(
                dtype=float
            )

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
                        "config": config_id(
                            config
                        ),
                    }
                )

            print(
                f"completed seed={seed}, "
                f"scenario={drift_type}"
            )

    raw = pd.DataFrame(
        rows
    )

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
    print("T1-STYLE AGGREGATE SUMMARY")
    print("=" * 120)
    print(
        grouped.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print(
        "Evaluation complete. Metric rows were appended "
        "to results/runs.csv."
    )


if __name__ == "__main__":
    main()
