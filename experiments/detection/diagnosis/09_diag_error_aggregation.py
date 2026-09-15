"""
09_diag_error_aggregation.py

Diagnosis 09 — Can non-overlapping MAE aggregation reduce the
autocorrelation problem while preserving useful drift detection?
======================================================================

Background
----------
Diagnosis 08 established a controlled result:

    stationary process
    + no concept drift
    + strong serial dependence
            ->
    many ADWIN/KSWIN alarms

At approximately the same absolute-error lag-1 autocorrelation observed
on AEMO (~0.95), stationary AR(1) streams produced very large numbers
of false alarms.

Therefore strong serial dependence in the pointwise forecasting-error
stream is now a demonstrated contributor to over-detection.

Purpose
-------
Test a possible solution WITHOUT prewhitening:

    non-overlapping MAE aggregation.

Instead of sending every 30-minute absolute forecast error to the
detector:

    |e_1|, |e_2|, |e_3|, ...

we summarize consecutive NON-OVERLAPPING blocks:

    MAE_1 = mean(|e_1 ... e_B|)
    MAE_2 = mean(|e_(B+1) ... e_(2B)|)
    ...

This avoids the extreme overlap that a rolling MAE would create.

Candidate resolutions
---------------------
block_size = 1
    30-minute pointwise absolute error
    CURRENT representation

block_size = 6
    3-hour non-overlapping MAE

block_size = 12
    6-hour non-overlapping MAE

block_size = 24
    12-hour non-overlapping MAE

block_size = 48
    24-hour non-overlapping MAE


Three experiments
-----------------

A. REAL AEMO
    Rebuild current GB residual stream for SA1 and NSW1.

    For each block size:
        - aggregate calibration absolute errors
        - calculate calibration mean/std
        - aggregate test absolute errors
        - normalize using aggregated calibration statistics
        - calculate lag-1 ACF
        - run unchanged detectors
        - report detector counts and alarms/year


B. STATIONARY AR(1), NO DRIFT
    Generate phi=0.98 stationary AR(1) streams.

    This approximately reproduces the AEMO absolute-error
    autocorrelation level before aggregation.

    For each block size:
        - aggregate
        - normalize
        - calculate ACF
        - run unchanged detectors
        - every alarm is known to be false


C. CONTROLLED SYNTHETIC DRIFT
    Use existing project synthetic generator and frozen GB:

        4000 train
        1000 calibration
        15000 test

        seeds 1-5

        none
        sudden
        gradual
        recurring

        calendar + lags (48,336)

    For each block size:
        - aggregate calibration errors
        - aggregate test errors
        - normalize using calibration blocks
        - run unchanged detectors
        - map block detections back to original 30-minute indices
        - evaluate using original 336-observation tolerance

This lets detection delays remain interpretable in the ORIGINAL
half-hour observation scale.


Important warning
-----------------
Detector parameters are NOT rescaled in this diagnosis.

Therefore:

    KSWIN window_size=140

means:

    B=1:
        140 x 30 min = 70 hours

    B=12:
        140 x 6 h = 35 days

    B=48:
        140 x 24 h = 140 days

Consequently:

    FEWER alarms under aggregation are not sufficient evidence
    that aggregation is better.

The synthetic drift results are essential.

If coarse aggregation eliminates false alarms but also causes missed
drifts or extreme delay, it is too coarse under the current detector
configuration.

A later experiment may retune detector parameters for the selected
aggregation scale.

This script does NOT
--------------------
    - modify detectors.py
    - tune detectors
    - perform prewhitening
    - modify Gradient Boosting
    - modify Task 120
    - modify T2
    - write to runs.csv


Run
---
    python experiments/detection/09_diag_error_aggregation.py


Outputs
-------
results/tables/detection/diagnostics/

    09_aemo_aggregation_summary.csv
    09_stationary_aggregation_runs.csv
    09_stationary_aggregation_summary.csv
    09_synthetic_aggregation_runs.csv
    09_synthetic_aggregation_summary.csv
"""


from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd


# =====================================================================
# Project imports
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from drift_lab.data_loader import (
    load_and_split_aemo,
)

from drift_lab.synthetic_generator import (
    make_feature_series,
)

from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)

from drift_lab.detectors import (
    DETECTOR_REGISTRY,
)

from drift_lab.evaluation import (
    evaluate_detections,
)


# =====================================================================
# Configuration
# =====================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "diagnostics"
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


REGIONS = (
    "SA1",
    "NSW1",
)


# -------------------------------------------------------------
# Non-overlapping block sizes
# -------------------------------------------------------------

BLOCK_SIZES = (
    1,      # 30 minutes
    6,      # 3 hours
    12,     # 6 hours
    24,     # 12 hours
    48,     # 24 hours
)


BLOCK_LABELS = {
    1: "30min",
    6: "3h_MAE",
    12: "6h_MAE",
    24: "12h_MAE",
    48: "24h_MAE",
}


OBSERVATIONS_PER_DAY = 48
OBSERVATIONS_PER_YEAR = 48 * 365


# -------------------------------------------------------------
# Current GB
# -------------------------------------------------------------

FORECASTER_CONFIG = {
    "feature_mode": "calendar+lags",
    "lags": (48, 336),
    "seed": 0,
}


# -------------------------------------------------------------
# Synthetic benchmark
# -------------------------------------------------------------

SYN_N = 20_000
SYN_TRAIN_END = 4_000
SYN_CAL_END = 5_000
SYN_TEST_START = 5_000

SYN_SEEDS = (
    1,
    2,
    3,
    4,
    5,
)


DRIFT_TYPES = (
    "none",
    "sudden",
    "gradual",
    "recurring",
)


# Original half-hour tolerance.
TOLERANCE_OBSERVATIONS = 48 * 7


# -------------------------------------------------------------
# Stationary no-drift control
# -------------------------------------------------------------

AR_PHI = 0.98

AR_SEEDS = tuple(
    range(
        1,
        21,
    )
)

AR_CALIBRATION_SIZE = 2880
AR_TEST_SIZE = 15000
AR_BURN_IN = 2000


# =====================================================================
# Validation
# =====================================================================

def validate_stream(
    values,
    name,
):
    """
    Validate finite one-dimensional numeric stream.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return values


# =====================================================================
# Non-overlapping MAE aggregation
# =====================================================================

def aggregate_absolute_error(
    absolute_error,
    block_size,
):
    """
    Convert pointwise absolute errors to non-overlapping block MAE.

    Only complete blocks are retained.

    Returns
    -------
    aggregated : np.ndarray
        One MAE value per complete block.

    usable_observations : int
        Number of original observations represented.

    dropped_observations : int
        Incomplete trailing observations discarded.
    """

    values = validate_stream(
        absolute_error,
        "absolute-error stream",
    )

    block_size = int(
        block_size
    )

    if block_size <= 0:
        raise ValueError(
            "block_size must be positive."
        )

    n_complete = (
        len(values)
        // block_size
    )

    usable = (
        n_complete
        * block_size
    )

    dropped = (
        len(values)
        - usable
    )

    if n_complete < 2:
        raise RuntimeError(
            f"Not enough complete blocks for "
            f"block_size={block_size}."
        )

    trimmed = values[
        :usable
    ]

    aggregated = (
        trimmed
        .reshape(
            n_complete,
            block_size,
        )
        .mean(
            axis=1
        )
    )

    return (
        validate_stream(
            aggregated,
            "aggregated MAE",
        ),
        usable,
        dropped,
    )


# =====================================================================
# Calibration normalization after aggregation
# =====================================================================

def build_aggregated_detector_stream(
    calibration_absolute,
    test_absolute,
    block_size,
):
    """
    Aggregate calibration and test absolute errors independently.

    Then normalize the test block-MAE stream using statistics estimated
    ONLY from calibration block MAEs.
    """

    (
        calibration_blocks,
        calibration_usable,
        calibration_dropped,
    ) = aggregate_absolute_error(
        calibration_absolute,
        block_size,
    )

    (
        test_blocks,
        test_usable,
        test_dropped,
    ) = aggregate_absolute_error(
        test_absolute,
        block_size,
    )

    calibration_mean = float(
        calibration_blocks.mean()
    )

    calibration_std = float(
        calibration_blocks.std()
    )

    if (
        not np.isfinite(
            calibration_std
        )
        or calibration_std <= 0
    ):
        raise RuntimeError(
            f"Invalid calibration std for "
            f"block_size={block_size}."
        )

    normalized = (
        test_blocks
        - calibration_mean
    ) / calibration_std

    normalized = validate_stream(
        normalized,
        "aggregated normalized detector stream",
    )

    return {
        "stream":
            normalized,

        "calibration_blocks":
            calibration_blocks,

        "test_blocks":
            test_blocks,

        "calibration_mean":
            calibration_mean,

        "calibration_std":
            calibration_std,

        "calibration_usable":
            calibration_usable,

        "calibration_dropped":
            calibration_dropped,

        "test_usable":
            test_usable,

        "test_dropped":
            test_dropped,
    }


# =====================================================================
# ACF
# =====================================================================

def autocorrelation(
    values,
    lag=1,
):
    """
    Sample autocorrelation.
    """

    values = validate_stream(
        values,
        "ACF stream",
    )

    if lag >= len(values):
        return float(
            "nan"
        )

    return float(
        pd.Series(
            values,
            dtype=float,
        ).autocorr(
            lag=lag
        )
    )


# =====================================================================
# Detector runner
# =====================================================================

def run_detectors(
    stream,
):
    """
    Run all frozen detector configurations unchanged.
    """

    rows = []

    for (
        detector_name,
        entry,
    ) in DETECTOR_REGISTRY.items():

        config = entry[
            "config"
        ].copy()

        detections = entry[
            "func"
        ](
            stream,
            **config,
        )

        rows.append(
            {
                "detector":
                    detector_name,

                "detections":
                    list(
                        detections
                    ),

                "count":
                    len(
                        detections
                    ),

                "config":
                    json.dumps(
                        config,
                        sort_keys=True,
                    ),
            }
        )

    return rows


# =====================================================================
# AEMO residual construction
# =====================================================================

def build_aemo_absolute_errors(
    region,
):
    """
    Reproduce current Task 120 GB forecast errors.
    """

    (
        _,
        train,
        calibration,
        test,
    ) = load_and_split_aemo(
        region=region,
        data_path=DATA_PATH,
    )

    model = GradientBoostedForecaster(
        **FORECASTER_CONFIG
    )

    model.fit(
        train
    )

    calibration_errors = frozen_error_stream(
        model,
        calibration,
        history_df=train,
    )

    test_errors = frozen_error_stream(
        model,
        test,
        history_df=calibration,
    )

    calibration_absolute = np.abs(
        calibration_errors[
            "error"
        ].to_numpy(
            dtype=float
        )
    )

    test_absolute = np.abs(
        test_errors[
            "error"
        ].to_numpy(
            dtype=float
        )
    )

    return (
        calibration_absolute,
        test_absolute,
    )


# =====================================================================
# AEMO experiment
# =====================================================================

def run_aemo_experiment():
    """
    Compare aggregation resolutions on real AEMO errors.
    """

    rows = []

    print()
    print(
        "=" * 100
    )
    print(
        "A. REAL AEMO AGGREGATION"
    )
    print(
        "=" * 100
    )

    for region in REGIONS:

        print()
        print(
            f"Building {region} forecasting errors..."
        )

        (
            calibration_absolute,
            test_absolute,
        ) = build_aemo_absolute_errors(
            region
        )

        for block_size in BLOCK_SIZES:

            result = (
                build_aggregated_detector_stream(
                    calibration_absolute=calibration_absolute,
                    test_absolute=test_absolute,
                    block_size=block_size,
                )
            )

            stream = result[
                "stream"
            ]

            acf1 = autocorrelation(
                stream,
                lag=1,
            )

            detector_results = run_detectors(
                stream
            )

            represented_years = (
                result[
                    "test_usable"
                ]
                / OBSERVATIONS_PER_YEAR
            )

            for detector_result in detector_results:

                count = detector_result[
                    "count"
                ]

                alarms_per_year = (
                    count
                    / represented_years
                )

                rows.append(
                    {
                        "region":
                            region,

                        "block_size":
                            block_size,

                        "resolution":
                            BLOCK_LABELS[
                                block_size
                            ],

                        "detector":
                            detector_result[
                                "detector"
                            ],

                        "calibration_blocks":
                            len(
                                result[
                                    "calibration_blocks"
                                ]
                            ),

                        "test_blocks":
                            len(
                                stream
                            ),

                        "test_original_observations":
                            result[
                                "test_usable"
                            ],

                        "dropped_test_observations":
                            result[
                                "test_dropped"
                            ],

                        "acf_lag1":
                            acf1,

                        "detections":
                            count,

                        "alarms_per_year":
                            alarms_per_year,

                        "calibration_mean":
                            result[
                                "calibration_mean"
                            ],

                        "calibration_std":
                            result[
                                "calibration_std"
                            ],

                        "detector_config":
                            detector_result[
                                "config"
                            ],
                    }
                )

            print(
                f"{region:<4} "
                f"{BLOCK_LABELS[block_size]:<8} "
                f"blocks={len(stream):>6} "
                f"ACF1={acf1:>7.4f}"
            )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# Stationary AR(1)
# =====================================================================

def generate_stationary_ar1(
    seed,
):
    """
    Generate stationary AR(1) with phi=0.98 and unit marginal variance.

    No drift is introduced.
    """

    rng = np.random.default_rng(
        seed
    )

    total = (
        AR_BURN_IN
        + AR_CALIBRATION_SIZE
        + AR_TEST_SIZE
    )

    values = np.empty(
        total,
        dtype=float,
    )

    values[
        0
    ] = rng.normal()

    innovation_std = np.sqrt(
        1.0
        - AR_PHI ** 2
    )

    noise = rng.normal(
        loc=0.0,
        scale=innovation_std,
        size=total - 1,
    )

    for i in range(
        1,
        total,
    ):
        values[
            i
        ] = (
            AR_PHI
            * values[
                i - 1
            ]
            + noise[
                i - 1
            ]
        )

    usable = values[
        AR_BURN_IN:
    ]

    calibration = usable[
        :AR_CALIBRATION_SIZE
    ]

    test = usable[
        AR_CALIBRATION_SIZE:
    ]

    return (
        calibration,
        test,
    )


# =====================================================================
# Stationary aggregation experiment
# =====================================================================

def run_stationary_experiment():
    """
    Determine whether aggregation reduces false alarms when the
    underlying process is stationary but strongly autocorrelated.
    """

    rows = []

    print()
    print(
        "=" * 100
    )
    print(
        "B. STATIONARY AR(1) NO-DRIFT CONTROL"
    )
    print(
        "=" * 100
    )

    print(
        f"phi={AR_PHI}"
    )

    for seed in AR_SEEDS:

        (
            calibration,
            test,
        ) = generate_stationary_ar1(
            seed
        )

        calibration_absolute = np.abs(
            calibration
        )

        test_absolute = np.abs(
            test
        )

        for block_size in BLOCK_SIZES:

            result = (
                build_aggregated_detector_stream(
                    calibration_absolute,
                    test_absolute,
                    block_size,
                )
            )

            stream = result[
                "stream"
            ]

            acf1 = autocorrelation(
                stream,
                lag=1,
            )

            detector_results = run_detectors(
                stream
            )

            for detector_result in detector_results:

                count = detector_result[
                    "count"
                ]

                false_alarms_per_10k_original = (
                    count
                    / result[
                        "test_usable"
                    ]
                    * 10_000
                )

                rows.append(
                    {
                        "seed":
                            seed,

                        "phi":
                            AR_PHI,

                        "block_size":
                            block_size,

                        "resolution":
                            BLOCK_LABELS[
                                block_size
                            ],

                        "detector":
                            detector_result[
                                "detector"
                            ],

                        "acf_lag1":
                            acf1,

                        "detections":
                            count,

                        "false_alarms_per_10k_original":
                            false_alarms_per_10k_original,

                        "calibration_blocks":
                            len(
                                result[
                                    "calibration_blocks"
                                ]
                            ),

                        "test_blocks":
                            len(
                                stream
                            ),

                        "test_original_observations":
                            result[
                                "test_usable"
                            ],
                    }
                )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# Synthetic forecast-error construction
# =====================================================================

def build_synthetic_absolute_errors(
    drift_type,
    seed,
):
    """
    Build synthetic GB absolute forecast errors using the current
    synthetic protocol.

    Returns
    -------
    calibration_absolute
    test_absolute
    truth
        Ground-truth changepoints relative to original 30-minute
        test-stream coordinates.
    """

    data, full_truth = make_feature_series(
        kind=drift_type,
        n=SYN_N,
        seed=seed,
    )

    train = data.iloc[
        :SYN_TRAIN_END
    ].copy()

    calibration = data.iloc[
        SYN_TRAIN_END:
        SYN_CAL_END
    ].copy()

    test = data.iloc[
        SYN_TEST_START:
    ].copy()

    model = GradientBoostedForecaster(
        **FORECASTER_CONFIG
    )

    model.fit(
        train
    )

    calibration_errors = frozen_error_stream(
        model,
        calibration,
        history_df=train,
    )

    test_errors = frozen_error_stream(
        model,
        test,
        history_df=calibration,
    )

    calibration_absolute = np.abs(
        calibration_errors[
            "error"
        ].to_numpy(
            dtype=float
        )
    )

    test_absolute = np.abs(
        test_errors[
            "error"
        ].to_numpy(
            dtype=float
        )
    )

    truth = []

    for cp in full_truth:

        if not (
            SYN_TEST_START
            <= cp
            < SYN_N
        ):
            raise ValueError(
                f"Ground truth {cp} lies outside "
                "synthetic test period."
            )

        truth.append(
            int(
                cp
                - SYN_TEST_START
            )
        )

    return (
        calibration_absolute,
        test_absolute,
        truth,
    )


# =====================================================================
# Map aggregated detector positions back to original observations
# =====================================================================

def block_detections_to_original_indices(
    detected_blocks,
    block_size,
    usable_observations,
):
    """
    Map a block-level detector alarm to the END of that block.

    Example
    -------
    block_size = 12

    detector block 0 represents original observations:

        0 ... 11

    Therefore its alarm timestamp is represented as original index 11.

    This lets evaluate_detections operate in the original 30-minute
    observation scale.
    """

    mapped = []

    for block_index in detected_blocks:

        original_index = (
            (
                int(
                    block_index
                )
                + 1
            )
            * block_size
            - 1
        )

        if original_index >= usable_observations:
            raise RuntimeError(
                "Mapped block detection lies outside "
                "usable original test observations."
            )

        mapped.append(
            original_index
        )

    return mapped


# =====================================================================
# Synthetic experiment
# =====================================================================

def run_synthetic_experiment():
    """
    Test whether aggregation preserves detection of known synthetic
    drifts under the CURRENT frozen detector configurations.
    """

    rows = []

    print()
    print(
        "=" * 100
    )
    print(
        "C. CONTROLLED SYNTHETIC DRIFT"
    )
    print(
        "=" * 100
    )

    print(
        f"Seeds: {SYN_SEEDS}"
    )
    print(
        f"Drift types: {DRIFT_TYPES}"
    )

    for seed in SYN_SEEDS:

        for drift_type in DRIFT_TYPES:

            (
                calibration_absolute,
                test_absolute,
                truth,
            ) = build_synthetic_absolute_errors(
                drift_type=drift_type,
                seed=seed,
            )

            for block_size in BLOCK_SIZES:

                result = (
                    build_aggregated_detector_stream(
                        calibration_absolute,
                        test_absolute,
                        block_size,
                    )
                )

                stream = result[
                    "stream"
                ]

                usable_test = result[
                    "test_usable"
                ]

                # -------------------------------------------------
                # Ground truths must remain inside retained test data.
                # -------------------------------------------------

                for cp in truth:

                    if cp >= usable_test:
                        raise RuntimeError(
                            f"Ground truth {cp} was lost by "
                            f"block_size={block_size}. "
                            "Choose another block alignment."
                        )

                acf1 = autocorrelation(
                    stream,
                    lag=1,
                )

                detector_results = run_detectors(
                    stream
                )

                for detector_result in detector_results:

                    detected_original = (
                        block_detections_to_original_indices(
                            detected_blocks=detector_result[
                                "detections"
                            ],
                            block_size=block_size,
                            usable_observations=usable_test,
                        )
                    )

                    metrics = evaluate_detections(
                        detected_changepoints=(
                            detected_original
                        ),
                        true_changepoints=truth,
                        n_observations=usable_test,
                        drift_type=drift_type,
                        tolerance=(
                            TOLERANCE_OBSERVATIONS
                        ),
                    )

                    false_alarm_count = len(
                        metrics.get(
                            "false_alarm_indices",
                            [],
                        )
                    )

                    rows.append(
                        {
                            "seed":
                                seed,

                            "drift_type":
                                drift_type,

                            "block_size":
                                block_size,

                            "resolution":
                                BLOCK_LABELS[
                                    block_size
                                ],

                            "detector":
                                detector_result[
                                    "detector"
                                ],

                            "acf_lag1":
                                acf1,

                            "calibration_blocks":
                                len(
                                    result[
                                        "calibration_blocks"
                                    ]
                                ),

                            "test_blocks":
                                len(
                                    stream
                                ),

                            "usable_original_test_observations":
                                usable_test,

                            "detections":
                                len(
                                    detected_original
                                ),

                            # Delay remains in ORIGINAL 30-minute units.
                            "detection_delay_observations":
                                metrics[
                                    "detection_delay"
                                ],

                            "detection_delay_days":
                                (
                                    metrics[
                                        "detection_delay"
                                    ]
                                    / 48
                                    if np.isfinite(
                                        metrics[
                                            "detection_delay"
                                        ]
                                    )
                                    else np.nan
                                ),

                            "false_alarm_count":
                                false_alarm_count,

                            "false_alarms_per_10k_original":
                                (
                                    false_alarm_count
                                    / usable_test
                                    * 10_000
                                ),

                            "missed_detections":
                                metrics[
                                    "missed_detections"
                                ],

                            "detector_config":
                                detector_result[
                                    "config"
                                ],
                        }
                    )

            print(
                f"completed seed={seed}, "
                f"scenario={drift_type}"
            )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# Aggregate stationary results
# =====================================================================

def aggregate_stationary(
    raw,
):
    """
    Average stationary results across 20 seeds.
    """

    return (
        raw
        .groupby(
            [
                "block_size",
                "resolution",
                "detector",
            ],
            as_index=False,
        )
        .agg(
            acf_lag1_mean=(
                "acf_lag1",
                "mean",
            ),

            acf_lag1_sd=(
                "acf_lag1",
                "std",
            ),

            alarm_mean=(
                "detections",
                "mean",
            ),

            alarm_sd=(
                "detections",
                "std",
            ),

            alarm_min=(
                "detections",
                "min",
            ),

            alarm_max=(
                "detections",
                "max",
            ),

            false_alarms_per_10k_original_mean=(
                "false_alarms_per_10k_original",
                "mean",
            ),

            calibration_blocks=(
                "calibration_blocks",
                "first",
            ),

            test_blocks=(
                "test_blocks",
                "first",
            ),

            seeds=(
                "seed",
                "nunique",
            ),
        )
    )


# =====================================================================
# Aggregate synthetic results
# =====================================================================

def aggregate_synthetic(
    raw,
):
    """
    Aggregate controlled synthetic metrics across seeds.
    """

    return (
        raw
        .groupby(
            [
                "block_size",
                "resolution",
                "detector",
                "drift_type",
            ],
            as_index=False,
        )
        .agg(
            acf_lag1_mean=(
                "acf_lag1",
                "mean",
            ),

            detections_mean=(
                "detections",
                "mean",
            ),

            delay_obs_mean=(
                "detection_delay_observations",
                "mean",
            ),

            delay_obs_sd=(
                "detection_delay_observations",
                "std",
            ),

            delay_days_mean=(
                "detection_delay_days",
                "mean",
            ),

            false_alarm_count_mean=(
                "false_alarm_count",
                "mean",
            ),

            false_alarms_per_10k_original_mean=(
                "false_alarms_per_10k_original",
                "mean",
            ),

            missed_mean=(
                "missed_detections",
                "mean",
            ),

            missed_sd=(
                "missed_detections",
                "std",
            ),

            calibration_blocks=(
                "calibration_blocks",
                "first",
            ),

            test_blocks=(
                "test_blocks",
                "first",
            ),

            seeds=(
                "seed",
                "nunique",
            ),
        )
    )


# =====================================================================
# Console display
# =====================================================================

def print_aemo_summary(
    aemo,
):
    """
    Main AEMO table.
    """

    print()
    print(
        "=" * 120
    )
    print(
        "AEMO AGGREGATION SUMMARY"
    )
    print(
        "=" * 120
    )

    display = aemo[
        [
            "region",
            "resolution",
            "detector",
            "calibration_blocks",
            "test_blocks",
            "acf_lag1",
            "detections",
            "alarms_per_year",
        ]
    ]

    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def print_stationary_summary(
    summary,
):
    """
    Stationary no-drift table.
    """

    print()
    print(
        "=" * 120
    )
    print(
        "STATIONARY NO-DRIFT AGGREGATION SUMMARY"
    )
    print(
        "=" * 120
    )

    display = summary[
        [
            "resolution",
            "detector",
            "calibration_blocks",
            "test_blocks",
            "acf_lag1_mean",
            "alarm_mean",
            "alarm_sd",
            "false_alarms_per_10k_original_mean",
        ]
    ]

    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def print_synthetic_summary(
    summary,
):
    """
    Controlled drift preservation table.
    """

    print()
    print(
        "=" * 140
    )
    print(
        "SYNTHETIC DRIFT PRESERVATION SUMMARY"
    )
    print(
        "=" * 140
    )

    display = summary[
        [
            "resolution",
            "detector",
            "drift_type",
            "calibration_blocks",
            "test_blocks",
            "acf_lag1_mean",
            "delay_days_mean",
            "false_alarms_per_10k_original_mean",
            "missed_mean",
        ]
    ]

    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# =====================================================================
# Interpretation helper
# =====================================================================

def print_interpretation():
    """
    Explain what constitutes useful evidence.
    """

    print()
    print(
        "=" * 100
    )
    print(
        "HOW TO INTERPRET DIAGNOSIS 09"
    )
    print(
        "=" * 100
    )

    print(
        """
Do NOT select the resolution with the fewest AEMO alarms.

Instead look for the smallest aggregation level satisfying all three:

1. AEMO dependence falls materially.

       Example:
           30min ACF1 = 0.95
           6h ACF1    = 0.40

2. Stationary AR(1) false alarms fall materially.

       Because there is no drift in this experiment, every remaining
       alarm is known to be false.

3. Controlled synthetic drift remains detectable.

       sudden
       gradual
       recurring

       should not suddenly develop large missed-detection counts or
       unacceptable delays.


A likely pattern might be:

    30min:
        excessive dependence and false alarms

    3h:
        improved but still dependent

    6h:
        useful compromise

    12h:
        fewer false alarms but slower detection

    24h:
        very low false alarms but potentially too coarse


IMPORTANT KSWIN CAVEAT
----------------------
KSWIN's current window_size remains 140 detector samples.

Therefore aggregation changes its physical detection horizon.

For example:

    30min:
        140 samples = 70 hours

    6h:
        140 samples = 35 days

    24h:
        140 samples = 140 days

If aggregation appears promising but KSWIN becomes too slow, that does
NOT automatically invalidate aggregation.

It means the KSWIN configuration would need to be retuned specifically
for the chosen detector-input timescale.

Do not retune it inside Diagnosis 09.
"""
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print(
        "=" * 100
    )
    print(
        "DIAGNOSIS 09 — NON-OVERLAPPING ERROR AGGREGATION"
    )
    print(
        "=" * 100
    )

    print(
        "Question:"
    )

    print(
        "Can non-overlapping MAE summaries reduce the serial-dependence "
        "problem while preserving useful concept-drift detection?"
    )

    print()
    print(
        "Candidate resolutions:"
    )

    for block_size in BLOCK_SIZES:
        print(
            f"  {BLOCK_LABELS[block_size]:<8} "
            f"block_size={block_size}"
        )

    print()
    print(
        "Current detectors are frozen."
    )

    print(
        "No detector tuning is performed."
    )

    print(
        "No prewhitening is performed."
    )

    print(
        "runs.csv is not modified."
    )

    # -------------------------------------------------------------
    # AEMO
    # -------------------------------------------------------------

    aemo = run_aemo_experiment()

    # -------------------------------------------------------------
    # Stationary no-drift
    # -------------------------------------------------------------

    stationary_raw = (
        run_stationary_experiment()
    )

    stationary_summary = (
        aggregate_stationary(
            stationary_raw
        )
    )

    # -------------------------------------------------------------
    # Controlled synthetic drift
    # -------------------------------------------------------------

    synthetic_raw = (
        run_synthetic_experiment()
    )

    synthetic_summary = (
        aggregate_synthetic(
            synthetic_raw
        )
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    aemo_path = (
        OUTPUT_DIR
        / "09_aemo_aggregation_summary.csv"
    )

    stationary_raw_path = (
        OUTPUT_DIR
        / "09_stationary_aggregation_runs.csv"
    )

    stationary_summary_path = (
        OUTPUT_DIR
        / "09_stationary_aggregation_summary.csv"
    )

    synthetic_raw_path = (
        OUTPUT_DIR
        / "09_synthetic_aggregation_runs.csv"
    )

    synthetic_summary_path = (
        OUTPUT_DIR
        / "09_synthetic_aggregation_summary.csv"
    )

    aemo.to_csv(
        aemo_path,
        index=False,
    )

    stationary_raw.to_csv(
        stationary_raw_path,
        index=False,
    )

    stationary_summary.to_csv(
        stationary_summary_path,
        index=False,
    )

    synthetic_raw.to_csv(
        synthetic_raw_path,
        index=False,
    )

    synthetic_summary.to_csv(
        synthetic_summary_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Console
    # -------------------------------------------------------------

    print_aemo_summary(
        aemo
    )

    print_stationary_summary(
        stationary_summary
    )

    print_synthetic_summary(
        synthetic_summary
    )

    print_interpretation()

    print()
    print(
        "=" * 100
    )
    print(
        "Saved diagnostic outputs"
    )
    print(
        "=" * 100
    )

    print(
        aemo_path
    )

    print(
        stationary_raw_path
    )

    print(
        stationary_summary_path
    )

    print(
        synthetic_raw_path
    )

    print(
        synthetic_summary_path
    )

    print()
    print(
        "Do not modify Task 120 yet."
    )

    print(
        "First identify whether any aggregation resolution reduces "
        "dependence and stationary false alarms while still preserving "
        "the controlled synthetic drift signal."
    )


if __name__ == "__main__":
    main()