"""
12_validate_aggregated_detectors_on_aemo.py

Frozen external AEMO validation of all selected aggregated
drift-detector candidates.

======================================================================
IMPORTANT
======================================================================

This is NOT tuning.

The detector configurations were selected using synthetic experiments
before looking at these AEMO results.

All three detectors remain in the final comparison:

    ADWIN
    Page-Hinkley
    KSWIN

Two candidate detector-input resolutions are compared:

    3-hour non-overlapping MAE
    6-hour non-overlapping MAE


======================================================================
FROZEN CANDIDATES
======================================================================

3h MAE
------
ADWIN:
    delta = 0.001

Page-Hinkley:
    min_instances = 5
    delta         = 0.005
    threshold     = 50
    alpha         = 0.98
    mode          = "both"

KSWIN:
    alpha       = 5e-5
    window_size = 44
    stat_size   = 11
    seed        = 12


6h MAE
------
ADWIN:
    delta = 0.005

Page-Hinkley:
    min_instances = 3
    delta         = 0.005
    threshold     = 50
    alpha         = 0.98
    mode          = "both"

KSWIN:
    alpha       = 0.001
    window_size = 24
    stat_size   = 10
    seed        = 12


======================================================================
AEMO PROTOCOL
======================================================================

Regions:
    SA1
    NSW1

Forecaster:
    frozen Gradient Boosting
    calendar + lags (48, 336)

Forecast-error representation:

    signed forecast residual
        -> absolute error
        -> non-overlapping MAE aggregation
        -> calibration-only normalization
        -> detector

No detector parameter is adjusted using AEMO.

No T2 matching is done in this script.

The purpose is only to observe frozen detector behaviour on AEMO.


======================================================================
OUTPUTS
======================================================================

results/tables/detection/diagnostics/

    12_aemo_aggregated_all_changepoints.csv
    12_aemo_aggregated_all_summary.csv


======================================================================
IMPORTANT
======================================================================

This script does NOT:

    - overwrite aemo_detection_changepoints.csv
    - modify detectors.py
    - modify runs.csv
    - modify Task 120
    - modify T2
    - tune on AEMO
    - perform prewhitening
    - use rolling MAE


Run
---
python experiments/detection/12_validate_aggregated_detectors_on_aemo.py
"""


from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# =====================================================================
# Project imports
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from drift_lab.data_loader import (
    load_and_split_aemo,
)

from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)

from drift_lab.detectors import (
    DETECTOR_REGISTRY,
)


# =====================================================================
# Paths
# =====================================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "diagnostics"
)


CHANGEPOINT_OUTPUT = (
    OUTPUT_DIR
    / "12_aemo_aggregated_all_changepoints.csv"
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "12_aemo_aggregated_all_summary.csv"
)


# =====================================================================
# Regions
# =====================================================================

REGIONS = (
    "SA1",
    "NSW1",
)


# =====================================================================
# Forecaster
# =====================================================================

FORECASTER_CONFIG = {
    "feature_mode": "calendar+lags",
    "lags": (48, 336),
    "seed": 0,
}


# =====================================================================
# Original sampling
# =====================================================================

OBSERVATIONS_PER_DAY = 48

OBSERVATIONS_PER_YEAR = (
    48
    * 365
)


# =====================================================================
# Frozen candidate configurations
# =====================================================================

CANDIDATES = {

    # =================================================================
    # 3-hour non-overlapping MAE
    # =================================================================

    "3h_MAE": {

        "block_size": 6,

        "detectors": {

            "adwin": {
                "delta": 0.001,
            },

            "page_hinkley": {
                "min_instances": 5,
                "delta": 0.005,
                "threshold": 50,
                "alpha": 0.98,
                "mode": "both",
            },

            "kswin": {
                "alpha": 5e-5,
                "window_size": 44,
                "stat_size": 11,
                "seed": 12,
            },
        },
    },


    # =================================================================
    # 6-hour non-overlapping MAE
    # =================================================================

    "6h_MAE": {

        "block_size": 12,

        "detectors": {

            "adwin": {
                "delta": 0.005,
            },

            "page_hinkley": {
                "min_instances": 3,
                "delta": 0.005,
                "threshold": 50,
                "alpha": 0.98,
                "mode": "both",
            },

            "kswin": {
                "alpha": 0.001,
                "window_size": 24,
                "stat_size": 10,
                "seed": 12,
            },
        },
    },
}


# =====================================================================
# Helpers
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


    if lag >= len(
        values
    ):

        return np.nan


    return float(
        pd.Series(
            values,
            dtype=float,
        ).autocorr(
            lag=lag
        )
    )


# =====================================================================
# Build frozen AEMO forecast errors
# =====================================================================

def build_aemo_error_data(
    region,
):
    """
    Reproduce the same frozen GB forecasting-error stream used by the
    current detection pipeline.
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


    calibration_abs = np.abs(
        calibration_errors[
            "error"
        ].to_numpy(
            dtype=float
        )
    )


    test_abs = np.abs(
        test_errors[
            "error"
        ].to_numpy(
            dtype=float
        )
    )


    calibration_abs = validate_stream(
        calibration_abs,
        f"{region} calibration absolute errors",
    )


    test_abs = validate_stream(
        test_abs,
        f"{region} test absolute errors",
    )


    if len(
        test_abs
    ) != len(
        test_errors
    ):

        raise RuntimeError(
            f"{region}: test error length mismatch."
        )


    return {
        "train":
            train,

        "calibration":
            calibration,

        "test":
            test,

        "calibration_errors":
            calibration_errors.reset_index(
                drop=True
            ),

        "test_errors":
            test_errors.reset_index(
                drop=True
            ),

        "calibration_abs":
            calibration_abs,

        "test_abs":
            test_abs,
    }


# =====================================================================
# Non-overlapping MAE aggregation
# =====================================================================

def aggregate_mae(
    values,
    block_size,
):
    """
    Aggregate absolute errors using complete non-overlapping blocks.

    Example
    -------
    block_size = 6

        observations 0..5    -> MAE block 0
        observations 6..11   -> MAE block 1
        ...

    Trailing incomplete observations are discarded.
    """

    values = validate_stream(
        values,
        "absolute error",
    )


    block_size = int(
        block_size
    )


    if block_size <= 0:

        raise ValueError(
            "block_size must be positive."
        )


    n_blocks = (
        len(values)
        // block_size
    )


    usable = (
        n_blocks
        * block_size
    )


    dropped = (
        len(values)
        - usable
    )


    if n_blocks < 2:

        raise RuntimeError(
            f"Too few complete blocks for block_size={block_size}."
        )


    blocks = (
        values[
            :usable
        ]
        .reshape(
            n_blocks,
            block_size,
        )
        .mean(
            axis=1
        )
    )


    return (
        validate_stream(
            blocks,
            "aggregated MAE",
        ),
        usable,
        dropped,
    )


# =====================================================================
# Build normalized detector stream
# =====================================================================

def build_detector_stream(
    calibration_abs,
    test_abs,
    block_size,
):
    """
    Aggregate calibration and test errors independently.

    Then normalize test block-MAE values using calibration-block
    mean/std only.
    """

    (
        calibration_blocks,
        calibration_usable,
        calibration_dropped,
    ) = aggregate_mae(
        calibration_abs,
        block_size,
    )


    (
        test_blocks,
        test_usable,
        test_dropped,
    ) = aggregate_mae(
        test_abs,
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
            "Calibration block standard deviation "
            "must be finite and positive."
        )


    normalized = (
        test_blocks
        -
        calibration_mean
    ) / calibration_std


    normalized = validate_stream(
        normalized,
        "normalized aggregated detector stream",
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
# Run arbitrary frozen detector
# =====================================================================

def run_detector(
    detector_name,
    stream,
    config,
):
    """
    Use the project's existing detector wrapper/function.

    Only the supplied already-frozen configuration is changed relative
    to the registry baseline.
    """

    if detector_name not in DETECTOR_REGISTRY:

        raise KeyError(
            f"Unknown detector: {detector_name}"
        )


    entry = DETECTOR_REGISTRY[
        detector_name
    ]


    detections = entry[
        "func"
    ](
        stream,
        **config,
    )


    return [
        int(
            index
        )
        for index in detections
    ]


# =====================================================================
# Map block detection to original test coordinates
# =====================================================================

def map_block_detection(
    block_index,
    block_size,
    test_usable,
):
    """
    Map one aggregated detector observation to the original 30-minute
    test-stream coordinates.

    The alarm timestamp is the END of the aggregation block because
    the MAE value becomes available only after all samples in that
    block have been observed.
    """

    start_index = (
        int(
            block_index
        )
        * block_size
    )


    end_index = (
        (
            int(
                block_index
            )
            + 1
        )
        * block_size
        - 1
    )


    if not (
        0
        <= start_index
        <= end_index
        < test_usable
    ):

        raise RuntimeError(
            "Mapped detection lies outside usable test stream."
        )


    return (
        start_index,
        end_index,
    )


# =====================================================================
# One resolution / detector / region
# =====================================================================

def run_one(
    region,
    resolution,
    block_size,
    detector_name,
    detector_config,
    error_data,
):
    """
    Run one frozen detector configuration on one aggregated AEMO
    error stream.
    """

    aggregation = build_detector_stream(
        calibration_abs=error_data[
            "calibration_abs"
        ],
        test_abs=error_data[
            "test_abs"
        ],
        block_size=block_size,
    )


    stream = aggregation[
        "stream"
    ]


    detected_blocks = run_detector(
        detector_name=detector_name,
        stream=stream,
        config=detector_config,
    )


    test_errors = error_data[
        "test_errors"
    ]


    changepoint_rows = []


    for detection_number, block_index in enumerate(
        detected_blocks,
        start=1,
    ):

        (
            start_index,
            end_index,
        ) = map_block_detection(
            block_index=block_index,
            block_size=block_size,
            test_usable=aggregation[
                "test_usable"
            ],
        )


        start_row = test_errors.iloc[
            start_index
        ]


        end_row = test_errors.iloc[
            end_index
        ]


        block_abs_errors = error_data[
            "test_abs"
        ][
            start_index:
            end_index + 1
        ]


        block_mae = float(
            np.mean(
                block_abs_errors
            )
        )


        changepoint_rows.append(
            {
                "region":
                    region,

                "resolution":
                    resolution,

                "detector":
                    detector_name,

                "detection_number":
                    detection_number,

                "block_size":
                    block_size,

                "block_hours":
                    block_size
                    * 0.5,

                "detected_block_index":
                    int(
                        block_index
                    ),

                # Original half-hour test-stream coordinate.
                "detected_index":
                    int(
                        end_index
                    ),

                "block_start_index":
                    int(
                        start_index
                    ),

                "block_end_index":
                    int(
                        end_index
                    ),

                "block_start_time":
                    start_row[
                        "SETTLEMENTDATE"
                    ],

                "SETTLEMENTDATE":
                    end_row[
                        "SETTLEMENTDATE"
                    ],

                "block_mae":
                    block_mae,

                "normalized_block_mae":
                    float(
                        stream[
                            block_index
                        ]
                    ),

                "forecast_error_at_block_end":
                    float(
                        end_row[
                            "error"
                        ]
                    ),

                "calibration_block_mean":
                    aggregation[
                        "calibration_mean"
                    ],

                "calibration_block_std":
                    aggregation[
                        "calibration_std"
                    ],

                "detector_config":
                    json.dumps(
                        detector_config,
                        sort_keys=True,
                    ),
            }
        )


    # -----------------------------------------------------------------
    # Rate in original represented time
    # -----------------------------------------------------------------

    represented_years = (
        aggregation[
            "test_usable"
        ]
        /
        OBSERVATIONS_PER_YEAR
    )


    count = len(
        detected_blocks
    )


    summary = {
        "region":
            region,

        "resolution":
            resolution,

        "detector":
            detector_name,

        "block_size":
            block_size,

        "block_hours":
            block_size
            * 0.5,

        "calibration_blocks":
            len(
                aggregation[
                    "calibration_blocks"
                ]
            ),

        "test_blocks":
            len(
                stream
            ),

        "test_original_observations":
            aggregation[
                "test_usable"
            ],

        "dropped_test_observations":
            aggregation[
                "test_dropped"
            ],

        "acf_lag1":
            autocorrelation(
                stream,
                lag=1,
            ),

        "detections":
            count,

        "detections_per_year":
            (
                count
                /
                represented_years
            ),

        "calibration_mean":
            aggregation[
                "calibration_mean"
            ],

        "calibration_std":
            aggregation[
                "calibration_std"
            ],

        "detector_config":
            json.dumps(
                detector_config,
                sort_keys=True,
            ),
    }


    return (
        changepoint_rows,
        summary,
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print(
        "=" * 110
    )

    print(
        "VALIDATION 12 — ALL FROZEN AGGREGATED DETECTORS ON AEMO"
    )

    print(
        "=" * 110
    )


    print()
    print(
        "This is external validation, NOT tuning."
    )


    print(
        "All three detectors are evaluated."
    )


    print(
        "Both frozen 3h and 6h representations are evaluated."
    )


    print(
        "AEMO results will NOT be used to adjust parameters."
    )


    print(
        "Official Task 120 outputs are NOT overwritten."
    )


    print(
        "T2 matching is NOT performed in this script."
    )


    all_changepoints = []

    all_summaries = []


    # =================================================================
    # Regions
    # =================================================================

    for region in REGIONS:

        print()
        print(
            "#" * 110
        )

        print(
            region
        )

        print(
            "#" * 110
        )


        print(
            "Building frozen Gradient Boosting forecast-error stream..."
        )


        error_data = build_aemo_error_data(
            region
        )


        print(
            f"train={len(error_data['train']):,}, "
            f"calibration={len(error_data['calibration']):,}, "
            f"test={len(error_data['test']):,}"
        )


        # =============================================================
        # Resolutions
        # =============================================================

        for (
            resolution,
            candidate,
        ) in CANDIDATES.items():

            block_size = candidate[
                "block_size"
            ]


            print()
            print(
                "=" * 80
            )

            print(
                f"{region} / {resolution}"
            )

            print(
                "=" * 80
            )


            # =========================================================
            # Detectors
            # =========================================================

            for (
                detector_name,
                detector_config,
            ) in candidate[
                "detectors"
            ].items():

                (
                    changepoints,
                    summary,
                ) = run_one(
                    region=region,
                    resolution=resolution,
                    block_size=block_size,
                    detector_name=detector_name,
                    detector_config=detector_config,
                    error_data=error_data,
                )


                all_changepoints.extend(
                    changepoints
                )


                all_summaries.append(
                    summary
                )


                print()
                print(
                    f"{detector_name}"
                )

                print(
                    f"  config            = "
                    f"{json.dumps(detector_config, sort_keys=True)}"
                )

                print(
                    f"  ACF1              = "
                    f"{summary['acf_lag1']:.4f}"
                )

                print(
                    f"  detections        = "
                    f"{summary['detections']}"
                )

                print(
                    f"  detections/year   = "
                    f"{summary['detections_per_year']:.3f}"
                )

                print(
                    f"  calibration blocks= "
                    f"{summary['calibration_blocks']}"
                )

                print(
                    f"  test blocks       = "
                    f"{summary['test_blocks']}"
                )


    # =================================================================
    # Save
    # =================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    summaries = pd.DataFrame(
        all_summaries
    )


    changepoints = pd.DataFrame(
        all_changepoints
    )


    # -----------------------------------------------------------------
    # Stable schema even if no detector fires
    # -----------------------------------------------------------------

    if changepoints.empty:

        changepoints = pd.DataFrame(
            columns=[
                "region",
                "resolution",
                "detector",
                "detection_number",
                "block_size",
                "block_hours",
                "detected_block_index",
                "detected_index",
                "block_start_index",
                "block_end_index",
                "block_start_time",
                "SETTLEMENTDATE",
                "block_mae",
                "normalized_block_mae",
                "forecast_error_at_block_end",
                "calibration_block_mean",
                "calibration_block_std",
                "detector_config",
            ]
        )


    changepoints.to_csv(
        CHANGEPOINT_OUTPUT,
        index=False,
    )


    summaries.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )


    # =================================================================
    # Console summary
    # =================================================================

    print()
    print(
        "=" * 130
    )

    print(
        "AEMO EXTERNAL VALIDATION SUMMARY"
    )

    print(
        "=" * 130
    )


    summary_columns = [
        "region",
        "resolution",
        "detector",
        "acf_lag1",
        "detections",
        "detections_per_year",
        "calibration_blocks",
        "test_blocks",
    ]


    print(
        summaries[
            summary_columns
        ]
        .sort_values(
            [
                "region",
                "resolution",
                "detector",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # =================================================================
    # Compact comparison
    # =================================================================

    print()
    print(
        "=" * 130
    )

    print(
        "DETECTION COUNT COMPARISON"
    )

    print(
        "=" * 130
    )


    pivot = (
        summaries
        .pivot_table(
            index=[
                "region",
                "detector",
            ],
            columns="resolution",
            values="detections",
            aggfunc="first",
        )
        .reset_index()
    )


    print(
        pivot.to_string(
            index=False
        )
    )


    # =================================================================
    # Detection timestamps
    # =================================================================

    print()
    print(
        "=" * 130
    )

    print(
        "DETECTION TIMESTAMPS"
    )

    print(
        "=" * 130
    )


    if changepoints.empty:

        print(
            "No detections produced by any candidate."
        )


    else:

        timestamp_columns = [
            "region",
            "resolution",
            "detector",
            "detection_number",
            "detected_index",
            "SETTLEMENTDATE",
            "block_mae",
            "normalized_block_mae",
        ]


        print(
            changepoints[
                timestamp_columns
            ]
            .sort_values(
                [
                    "region",
                    "resolution",
                    "detector",
                    "SETTLEMENTDATE",
                ]
            )
            .to_string(
                index=False
            )
        )


    # =================================================================
    # Final notes
    # =================================================================

    print()
    print(
        "=" * 110
    )

    print(
        "Saved outputs"
    )

    print(
        "=" * 110
    )


    print(
        CHANGEPOINT_OUTPUT
    )

    print(
        SUMMARY_OUTPUT
    )


    print()
    print(
        "IMPORTANT"
    )


    print(
        "Do not tune detector parameters based on these AEMO results."
    )


    print(
        "Use this run only to compare the externally validated "
        "behaviour of the already-frozen 3h and 6h candidates."
    )


    print()
    print(
        "After selecting the final resolution, the next step is to "
        "apply the SAME T2 event-matching protocol to these detections."
    )


if __name__ == "__main__":
    main()