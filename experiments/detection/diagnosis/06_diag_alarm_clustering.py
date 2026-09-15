"""
06_diag_alarm_clustering.py

Diagnosis 06 — Are ADWIN/KSWIN repeatedly alarming within the same
persistent forecasting-error episodes?
======================================================================

Background
----------
The diagnosis sequence so far found:

Diagnosis 01
    Full random shuffling of the calibration error stream removed all
    detector alarms.

    Conclusion:
        Temporal structure matters.

Diagnosis 02
    Breaking short-range / within-day ordering reduced ADWIN/KSWIN
    alarms substantially.

    Conclusion:
        Local temporal persistence is important.

Diagnosis 03
    Strong dependence already exists in the SIGNED forecasting
    residuals.

    Conclusion:
        Taking absolute values is not the original cause.

Diagnosis 04
    Gradient Boosting and Seasonal Naive produce highly persistent
    residuals and many alarms, while SARIMAX produces much less
    persistence and fewer alarms.

Diagnosis 05
    Calibration mean/std normalization is NOT responsible for the
    excessive KSWIN alarms.

    KSWIN produced exactly the same changepoints under:
        raw absolute error
        calibration z-score
        centering
        scaling
        robust scaling
        log transformation

    Current scaling actually reduces ADWIN and Page-Hinkley alarm counts.

Therefore the next question is:

    Are ADWIN and KSWIN detecting hundreds of DISTINCT changes?

or:

    Are they repeatedly firing during the SAME sustained/local
    forecasting-error episodes?


Purpose
-------
This script analyzes the EXISTING Task 120 detections.

It does NOT rerun ADWIN, Page-Hinkley or KSWIN.

Input:

    results/tables/detection/aemo_detection_changepoints.csv

For every region/detector, it measures:

1. Inter-alarm gaps
       minimum
       median
       mean
       p25
       p75
       p90
       maximum

2. Fraction of consecutive alarms separated by <=:
       1 day
       3 days
       7 days

3. Alarm clusters using:
       1-day maximum inter-alarm gap
       3-day maximum inter-alarm gap
       7-day maximum inter-alarm gap

4. Cluster characteristics:
       number of clusters
       number of singleton clusters
       mean cluster size
       median cluster size
       maximum cluster size
       maximum cluster duration
       retrigger fraction

5. Monthly alarm counts.

6. ADWIN/KSWIN overlap:
       exact same timestamp
       nearest detection within 1 day
       nearest detection within 3 days
       nearest detection within 7 days

7. Local forecasting-error context around every alarm:
       current normalized absolute error
       mean/median error in 24 h before
       mean/median error in 24 h after
       mean error in 3 d before
       mean error in 3 d after
       mean error in 7 d before
       mean error in 7 d after

Why clustering matters
----------------------
Suppose KSWIN produces:

    297 detections

but with a 3-day separation rule these become:

    40 temporal clusters

Then the interpretation changes substantially.

It would suggest that KSWIN may be repeatedly reacting during
persistent local error regimes rather than identifying 297 independent
concept changes.

Conversely, if the 297 detections remain close to 297 clusters even
under a 3-day or 7-day rule, then repeated retriggering is not the main
explanation.


Cluster definition
------------------
Clusters use a simple consecutive-gap rule.

For a chosen gap G:

    if current_detection - previous_detection <= G:
        same cluster

    otherwise:
        start new cluster

This is a diagnostic grouping rule only.

It is NOT:

    - the T2 documented-event matching tolerance
    - a detector refractory period
    - a new definition of concept drift
    - a change to evaluation.py

The 1/3/7-day windows are used only to understand the temporal
concentration of the existing alarms.


Full error-stream reconstruction
--------------------------------
Task 120 stores only detected positions, not every forecast-error value.

To calculate the local forecasting-error context around each alarm,
this script reconstructs the SAME detector input used by Task 120:

    Gradient Boosting
        calendar + lags (48, 336)

    calibration residuals
        -> absolute error
        -> calibration mean/std

    test residuals
        -> absolute error
        -> frozen calibration normalization

The reconstructed stream is validated against the forecast_error and
normalized_abs_error values already stored at every Task 120 detection.

The detectors themselves are NEVER rerun.


This script does NOT
--------------------
    - modify detectors.py
    - rerun detectors
    - tune detector thresholds
    - change normalization
    - change Gradient Boosting
    - perform smoothing
    - perform prewhitening
    - modify T2 event matching
    - modify runs.csv
    - modify aemo_detection_changepoints.csv


Run
---
From repository root:

    python experiments/detection/06_diag_alarm_clustering.py


Outputs
-------
Saved under:

    results/tables/detection/diagnostics/

Files:

    06_alarm_gap_summary.csv
    06_alarm_cluster_summary.csv
    06_alarm_clusters.csv
    06_alarm_monthly_counts.csv
    06_detector_overlap.csv
    06_alarm_local_error_context.csv
"""


from pathlib import Path
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


# =====================================================================
# Paths / frozen settings
# =====================================================================

REGIONS = (
    "SA1",
    "NSW1",
)


DETECTORS = (
    "adwin",
    "page_hinkley",
    "kswin",
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


DETECTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "aemo_detection_changepoints.csv"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "diagnostics"
)


# Exact current Task 120 GB configuration.
FORECASTER_CONFIG = {
    "feature_mode": "calendar+lags",
    "lags": (48, 336),
    "seed": 0,
}


# AEMO standardized frequency.
OBSERVATIONS_PER_DAY = 48


# Diagnostic clustering gaps.
CLUSTER_WINDOWS_DAYS = (
    1,
    3,
    7,
)


# Local error-context windows.
LOCAL_WINDOWS = {
    "24h": 48,
    "3d": 48 * 3,
    "7d": 48 * 7,
}


# =====================================================================
# Validation helpers
# =====================================================================

def validate_numeric(
    values,
    name,
):
    """
    Return finite one-dimensional float array.
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
            f"{name} contains NaN or infinity."
        )

    return values


# =====================================================================
# Load existing Task 120 detections
# =====================================================================

def load_task120_detections():
    """
    Load EXISTING Task 120 changepoints.

    No detector is rerun.
    """

    if not DETECTION_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find Task 120 detections:\n"
            f"{DETECTION_PATH}"
        )

    detections = pd.read_csv(
        DETECTION_PATH
    )

    required = {
        "region",
        "detector",
        "detected_index",
        "SETTLEMENTDATE",
        "forecast_error",
        "normalized_abs_error",
    }

    missing = (
        required
        - set(
            detections.columns
        )
    )

    if missing:
        raise ValueError(
            "Task 120 file is missing columns: "
            f"{sorted(missing)}"
        )

    detections[
        "SETTLEMENTDATE"
    ] = pd.to_datetime(
        detections[
            "SETTLEMENTDATE"
        ]
    )

    if not detections[
        "region"
    ].isin(
        REGIONS
    ).all():
        raise ValueError(
            "Unexpected region in Task 120 detections."
        )

    if not detections[
        "detector"
    ].isin(
        DETECTORS
    ).all():
        raise ValueError(
            "Unexpected detector in Task 120 detections."
        )

    indices = pd.to_numeric(
        detections[
            "detected_index"
        ],
        errors="raise",
    )

    if (
        (indices < 0)
        |
        (indices % 1 != 0)
    ).any():
        raise ValueError(
            "detected_index must contain non-negative integers."
        )

    detections[
        "detected_index"
    ] = indices.astype(
        int
    )

    if detections.duplicated(
        [
            "region",
            "detector",
            "detected_index",
        ]
    ).any():
        raise ValueError(
            "Duplicate Task 120 detections found."
        )

    return detections


# =====================================================================
# Reconstruct exact Task 120 error stream
# =====================================================================

def reconstruct_task120_stream(
    region,
):
    """
    Reconstruct the full current Task 120 detector input.

    Detectors are NOT run.

    Returns
    -------
    pd.DataFrame

        detected_index
        SETTLEMENTDATE
        forecast_error
        absolute_error
        normalized_abs_error
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


    # -------------------------------------------------------------
    # Calibration errors
    # -------------------------------------------------------------

    calibration_errors = frozen_error_stream(
        model,
        calibration,
        history_df=train,
    )


    calibration_abs = (
        calibration_errors[
            "error"
        ]
        .abs()
        .to_numpy(
            dtype=float
        )
    )


    calibration_abs = validate_numeric(
        calibration_abs,
        "calibration absolute errors",
    )


    calibration_mean = float(
        calibration_abs.mean()
    )


    calibration_std = float(
        calibration_abs.std()
    )


    if (
        not np.isfinite(calibration_std)
        or calibration_std <= 0
    ):
        raise RuntimeError(
            "Invalid calibration absolute-error standard deviation."
        )


    # -------------------------------------------------------------
    # Test errors
    # -------------------------------------------------------------

    test_errors = frozen_error_stream(
        model,
        test,
        history_df=calibration,
    )


    forecast_error = validate_numeric(
        test_errors[
            "error"
        ],
        "test forecasting errors",
    )


    absolute_error = np.abs(
        forecast_error
    )


    normalized_abs_error = (
        absolute_error
        - calibration_mean
    ) / calibration_std


    validate_numeric(
        normalized_abs_error,
        "normalized absolute-error stream",
    )


    if len(
        test_errors
    ) != len(
        test
    ):
        raise RuntimeError(
            "Test stream length mismatch."
        )


    stream = pd.DataFrame(
        {
            "detected_index":
                np.arange(
                    len(test_errors),
                    dtype=int,
                ),

            "SETTLEMENTDATE":
                pd.to_datetime(
                    test_errors[
                        "SETTLEMENTDATE"
                    ]
                ).to_numpy(),

            "forecast_error":
                forecast_error,

            "absolute_error":
                absolute_error,

            "normalized_abs_error":
                normalized_abs_error,
        }
    )


    return (
        stream,
        calibration_mean,
        calibration_std,
    )


# =====================================================================
# Validate reconstruction against stored detections
# =====================================================================

def validate_reconstruction(
    region,
    stored,
    stream,
):
    """
    Verify that the rebuilt forecast-error stream reproduces Task 120
    at every stored detection point.
    """

    region_rows = stored[
        stored[
            "region"
        ] == region
    ]


    for row in region_rows.itertuples(
        index=False
    ):

        i = int(
            row.detected_index
        )


        if i >= len(
            stream
        ):
            raise RuntimeError(
                f"{region}: stored detection index "
                f"{i} exceeds reconstructed stream."
            )


        actual = stream.iloc[
            i
        ]


        stored_time = pd.Timestamp(
            row.SETTLEMENTDATE
        )


        actual_time = pd.Timestamp(
            actual[
                "SETTLEMENTDATE"
            ]
        )


        if stored_time != actual_time:
            raise RuntimeError(
                f"{region}: timestamp mismatch at "
                f"detection index {i}."
            )


        if not np.isclose(
            float(
                row.forecast_error
            ),
            float(
                actual[
                    "forecast_error"
                ]
            ),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise RuntimeError(
                f"{region}: forecast-error mismatch "
                f"at index {i}."
            )


        if not np.isclose(
            float(
                row.normalized_abs_error
            ),
            float(
                actual[
                    "normalized_abs_error"
                ]
            ),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise RuntimeError(
                f"{region}: normalized-error mismatch "
                f"at index {i}."
            )


# =====================================================================
# Inter-alarm gap diagnostics
# =====================================================================

def calculate_gap_summary(
    region,
    detector,
    rows,
):
    """
    Summarize elapsed time between consecutive existing alarms.
    """

    rows = (
        rows
        .sort_values(
            "SETTLEMENTDATE"
        )
        .reset_index(
            drop=True
        )
    )


    timestamps = pd.DatetimeIndex(
        rows[
            "SETTLEMENTDATE"
        ]
    )


    n_detections = len(
        timestamps
    )


    if n_detections <= 1:

        return {
            "region":
                region,

            "detector":
                detector,

            "detections":
                n_detections,

            "n_inter_alarm_gaps":
                0,

            "min_gap_days":
                float("nan"),

            "p25_gap_days":
                float("nan"),

            "median_gap_days":
                float("nan"),

            "mean_gap_days":
                float("nan"),

            "p75_gap_days":
                float("nan"),

            "p90_gap_days":
                float("nan"),

            "max_gap_days":
                float("nan"),

            "gaps_le_1d":
                0,

            "fraction_gaps_le_1d":
                float("nan"),

            "gaps_le_3d":
                0,

            "fraction_gaps_le_3d":
                float("nan"),

            "gaps_le_7d":
                0,

            "fraction_gaps_le_7d":
                float("nan"),
        }


    gaps_days = (
        timestamps
        .to_series()
        .diff()
        .dropna()
        .dt
        .total_seconds()
        .to_numpy()
        / 86400.0
    )


    result = {
        "region":
            region,

        "detector":
            detector,

        "detections":
            n_detections,

        "n_inter_alarm_gaps":
            len(
                gaps_days
            ),

        "min_gap_days":
            float(
                np.min(
                    gaps_days
                )
            ),

        "p25_gap_days":
            float(
                np.quantile(
                    gaps_days,
                    0.25,
                )
            ),

        "median_gap_days":
            float(
                np.median(
                    gaps_days
                )
            ),

        "mean_gap_days":
            float(
                np.mean(
                    gaps_days
                )
            ),

        "p75_gap_days":
            float(
                np.quantile(
                    gaps_days,
                    0.75,
                )
            ),

        "p90_gap_days":
            float(
                np.quantile(
                    gaps_days,
                    0.90,
                )
            ),

        "max_gap_days":
            float(
                np.max(
                    gaps_days
                )
            ),
    }


    for days in CLUSTER_WINDOWS_DAYS:

        count = int(
            np.sum(
                gaps_days
                <= days
            )
        )


        result[
            f"gaps_le_{days}d"
        ] = count


        result[
            f"fraction_gaps_le_{days}d"
        ] = float(
            count
            / len(
                gaps_days
            )
        )


    return result


# =====================================================================
# Temporal clustering
# =====================================================================

def assign_clusters(
    rows,
    gap_days,
):
    """
    Assign consecutive detections to temporal clusters.

    A new cluster begins only when:

        current timestamp - previous timestamp > gap_days
    """

    rows = (
        rows
        .sort_values(
            "SETTLEMENTDATE"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


    if rows.empty:

        rows[
            "cluster_id"
        ] = pd.Series(
            dtype=int
        )

        return rows


    cluster_ids = []

    cluster_id = 1

    previous_time = None


    for timestamp in rows[
        "SETTLEMENTDATE"
    ]:

        timestamp = pd.Timestamp(
            timestamp
        )


        if previous_time is None:

            cluster_ids.append(
                cluster_id
            )


        else:

            gap = (
                timestamp
                - previous_time
            ).total_seconds() / 86400.0


            if gap > gap_days:

                cluster_id += 1


            cluster_ids.append(
                cluster_id
            )


        previous_time = timestamp


    rows[
        "cluster_id"
    ] = cluster_ids


    return rows


def summarize_clusters(
    region,
    detector,
    rows,
    gap_days,
):
    """
    Produce cluster-level rows and one detector-level summary.
    """

    clustered = assign_clusters(
        rows,
        gap_days,
    )


    cluster_rows = []


    for (
        cluster_id,
        group,
    ) in clustered.groupby(
        "cluster_id",
        sort=True,
    ):

        start = pd.Timestamp(
            group[
                "SETTLEMENTDATE"
            ].iloc[0]
        )


        end = pd.Timestamp(
            group[
                "SETTLEMENTDATE"
            ].iloc[-1]
        )


        duration_days = (
            end
            - start
        ).total_seconds() / 86400.0


        cluster_rows.append(
            {
                "region":
                    region,

                "detector":
                    detector,

                "gap_rule_days":
                    gap_days,

                "cluster_id":
                    int(
                        cluster_id
                    ),

                "cluster_start":
                    start,

                "cluster_end":
                    end,

                "cluster_duration_days":
                    float(
                        duration_days
                    ),

                "cluster_size":
                    int(
                        len(
                            group
                        )
                    ),

                "first_detected_index":
                    int(
                        group[
                            "detected_index"
                        ].iloc[0]
                    ),

                "last_detected_index":
                    int(
                        group[
                            "detected_index"
                        ].iloc[-1]
                    ),
            }
        )


    cluster_frame = pd.DataFrame(
        cluster_rows
    )


    n_detections = len(
        clustered
    )


    n_clusters = len(
        cluster_frame
    )


    if n_clusters == 0:

        summary = {
            "region":
                region,

            "detector":
                detector,

            "gap_rule_days":
                gap_days,

            "detections":
                0,

            "clusters":
                0,

            "singleton_clusters":
                0,

            "mean_cluster_size":
                float("nan"),

            "median_cluster_size":
                float("nan"),

            "max_cluster_size":
                0,

            "mean_cluster_duration_days":
                float("nan"),

            "max_cluster_duration_days":
                float("nan"),

            "retrigger_fraction":
                float("nan"),
        }


        return (
            summary,
            cluster_frame,
        )


    sizes = cluster_frame[
        "cluster_size"
    ].to_numpy(
        dtype=float
    )


    durations = cluster_frame[
        "cluster_duration_days"
    ].to_numpy(
        dtype=float
    )


    # Fraction of detections beyond the first alarm in each cluster.
    #
    # Example:
    #   100 detections -> 30 clusters
    #
    #   70/100 = 70% of alarms are additional alarms occurring after
    #   an earlier alarm in their cluster.
    retrigger_fraction = (
        (
            n_detections
            - n_clusters
        )
        / n_detections
        if n_detections
        else float(
            "nan"
        )
    )


    summary = {
        "region":
            region,

        "detector":
            detector,

        "gap_rule_days":
            gap_days,

        "detections":
            n_detections,

        "clusters":
            n_clusters,

        "singleton_clusters":
            int(
                np.sum(
                    sizes == 1
                )
            ),

        "mean_cluster_size":
            float(
                np.mean(
                    sizes
                )
            ),

        "median_cluster_size":
            float(
                np.median(
                    sizes
                )
            ),

        "max_cluster_size":
            int(
                np.max(
                    sizes
                )
            ),

        "mean_cluster_duration_days":
            float(
                np.mean(
                    durations
                )
            ),

        "max_cluster_duration_days":
            float(
                np.max(
                    durations
                )
            ),

        "retrigger_fraction":
            float(
                retrigger_fraction
            ),
    }


    return (
        summary,
        cluster_frame,
    )


# =====================================================================
# Monthly counts
# =====================================================================

def calculate_monthly_counts(
    detections,
):
    """
    Count existing detector alarms by calendar month.
    """

    frame = detections.copy()


    frame[
        "month"
    ] = (
        frame[
            "SETTLEMENTDATE"
        ]
        .dt
        .to_period(
            "M"
        )
        .astype(str)
    )


    counts = (
        frame
        .groupby(
            [
                "region",
                "detector",
                "month",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "detections"
            }
        )
    )


    return counts


# =====================================================================
# Detector overlap
# =====================================================================

def nearest_distance_days(
    source_times,
    target_times,
):
    """
    For every source detection, find distance to nearest target
    detection.

    Returns distances in days.
    """

    source_times = pd.DatetimeIndex(
        source_times
    )


    target_times = pd.DatetimeIndex(
        target_times
    )


    if len(
        source_times
    ) == 0:

        return np.array(
            [],
            dtype=float,
        )


    if len(
        target_times
    ) == 0:

        return np.full(
            len(
                source_times
            ),
            np.inf,
            dtype=float,
        )


    source_ns = (
        source_times
        .asi8
    )


    target_ns = np.sort(
        target_times
        .asi8
    )


    positions = np.searchsorted(
        target_ns,
        source_ns,
    )


    distances = np.empty(
        len(
            source_ns
        ),
        dtype=float,
    )


    for i, (
        value,
        pos,
    ) in enumerate(
        zip(
            source_ns,
            positions,
        )
    ):

        candidates = []


        if pos < len(
            target_ns
        ):

            candidates.append(
                abs(
                    target_ns[
                        pos
                    ]
                    - value
                )
            )


        if pos > 0:

            candidates.append(
                abs(
                    target_ns[
                        pos - 1
                    ]
                    - value
                )
            )


        distances[
            i
        ] = min(
            candidates
        )


    # nanoseconds -> days
    return (
        distances
        / 1e9
        / 86400.0
    )


def detector_overlap(
    detections,
):
    """
    Compare ADWIN and KSWIN detection timing.

    Both directions are reported because:

        "fraction of ADWIN detections near KSWIN"

    is not necessarily equal to:

        "fraction of KSWIN detections near ADWIN"
    """

    rows = []


    for region in REGIONS:

        region_rows = detections[
            detections[
                "region"
            ] == region
        ]


        adwin_times = pd.DatetimeIndex(
            region_rows.loc[
                region_rows[
                    "detector"
                ] == "adwin",
                "SETTLEMENTDATE",
            ]
        )


        kswin_times = pd.DatetimeIndex(
            region_rows.loc[
                region_rows[
                    "detector"
                ] == "kswin",
                "SETTLEMENTDATE",
            ]
        )


        exact_overlap = len(
            set(
                adwin_times
            )
            &
            set(
                kswin_times
            )
        )


        for (
            source_name,
            source_times,
            target_name,
            target_times,
        ) in (
            (
                "adwin",
                adwin_times,
                "kswin",
                kswin_times,
            ),
            (
                "kswin",
                kswin_times,
                "adwin",
                adwin_times,
            ),
        ):

            distances = nearest_distance_days(
                source_times,
                target_times,
            )


            row = {
                "region":
                    region,

                "source_detector":
                    source_name,

                "target_detector":
                    target_name,

                "source_detections":
                    len(
                        source_times
                    ),

                "target_detections":
                    len(
                        target_times
                    ),

                "exact_same_timestamp_count":
                    exact_overlap,

                "nearest_gap_median_days":
                    float(
                        np.median(
                            distances
                        )
                    )
                    if len(
                        distances
                    )
                    else float(
                        "nan"
                    ),
            }


            for days in CLUSTER_WINDOWS_DAYS:

                count = int(
                    np.sum(
                        distances
                        <= days
                    )
                )


                row[
                    f"within_{days}d_count"
                ] = count


                row[
                    f"within_{days}d_fraction"
                ] = (
                    count
                    / len(
                        distances
                    )
                    if len(
                        distances
                    )
                    else float(
                        "nan"
                    )
                )


            rows.append(
                row
            )


    return pd.DataFrame(
        rows
    )


# =====================================================================
# Local error context
# =====================================================================

def safe_window_stats(
    values,
):
    """
    Mean / median / max for possibly empty local window.
    """

    values = np.asarray(
        values,
        dtype=float,
    )


    if len(
        values
    ) == 0:

        return {
            "mean":
                float("nan"),

            "median":
                float("nan"),

            "max":
                float("nan"),
        }


    return {
        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),
    }


def alarm_local_context(
    detections,
    streams,
):
    """
    For every stored alarm, inspect normalized absolute forecasting
    error immediately before and after it.
    """

    rows = []


    for detection in detections.itertuples(
        index=False
    ):

        region = detection.region

        detector = detection.detector

        i = int(
            detection.detected_index
        )


        stream = streams[
            region
        ]


        values = stream[
            "normalized_abs_error"
        ].to_numpy(
            dtype=float
        )


        if not (
            0
            <= i
            < len(
                values
            )
        ):

            raise RuntimeError(
                f"Invalid alarm index {i} "
                f"for {region}/{detector}."
            )


        row = {
            "region":
                region,

            "detector":
                detector,

            "detected_index":
                i,

            "SETTLEMENTDATE":
                detection.SETTLEMENTDATE,

            "normalized_abs_error_at_alarm":
                float(
                    values[
                        i
                    ]
                ),
        }


        for (
            label,
            window,
        ) in LOCAL_WINDOWS.items():

            before_start = max(
                0,
                i - window,
            )


            before = values[
                before_start:i
            ]


            after_end = min(
                len(
                    values
                ),
                i + 1 + window,
            )


            after = values[
                i + 1:
                after_end
            ]


            before_stats = safe_window_stats(
                before
            )


            after_stats = safe_window_stats(
                after
            )


            row[
                f"before_{label}_mean"
            ] = before_stats[
                "mean"
            ]


            row[
                f"after_{label}_mean"
            ] = after_stats[
                "mean"
            ]


            row[
                f"before_{label}_median"
            ] = before_stats[
                "median"
            ]


            row[
                f"after_{label}_median"
            ] = after_stats[
                "median"
            ]


            row[
                f"before_{label}_max"
            ] = before_stats[
                "max"
            ]


            row[
                f"after_{label}_max"
            ] = after_stats[
                "max"
            ]


            row[
                f"mean_change_{label}"
            ] = (
                after_stats[
                    "mean"
                ]
                -
                before_stats[
                    "mean"
                ]
            )


        rows.append(
            row
        )


    return pd.DataFrame(
        rows
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "DIAGNOSIS 06 — ALARM CLUSTERING / RETRIGGERING"
    )

    print(
        "=" * 80
    )


    print(
        "Question:"
    )


    print(
        "Are ADWIN and KSWIN producing many distinct changes, "
        "or repeatedly firing during the same persistent "
        "forecasting-error episodes?"
    )


    print()
    print(
        "Existing Task 120 detections will be analyzed."
    )


    print(
        "Detectors will NOT be rerun."
    )


    print(
        "Detector configurations will NOT be changed."
    )


    print(
        "Normalization will NOT be changed."
    )


    print(
        "T2 matching will NOT be changed."
    )


    # -----------------------------------------------------------------
    # Existing detections
    # -----------------------------------------------------------------

    detections = load_task120_detections()


    print()
    print(
        "Existing Task 120 counts:"
    )


    print(
        detections
        .groupby(
            [
                "region",
                "detector",
            ]
        )
        .size()
        .rename(
            "detections"
        )
        .to_string()
    )


    # -----------------------------------------------------------------
    # Rebuild stream for local context and validate Task 120
    # -----------------------------------------------------------------

    streams = {}


    print()
    print(
        "=" * 80
    )

    print(
        "VALIDATING TASK 120 ERROR STREAM"
    )

    print(
        "=" * 80
    )


    for region in REGIONS:

        (
            stream,
            calibration_mean,
            calibration_std,
        ) = reconstruct_task120_stream(
            region
        )


        validate_reconstruction(
            region=region,
            stored=detections,
            stream=stream,
        )


        streams[
            region
        ] = stream


        print(
            f"{region}: reproduced "
            f"{len(stream):,} test observations; "
            f"calibration mean={calibration_mean:.4f}, "
            f"std={calibration_std:.4f}"
        )


        print(
            f"{region}: all stored Task 120 alarm values reproduced."
        )


    # -----------------------------------------------------------------
    # Gap summaries
    # -----------------------------------------------------------------

    gap_rows = []


    for region in REGIONS:

        for detector in DETECTORS:

            rows = detections[
                (
                    detections[
                        "region"
                    ] == region
                )
                &
                (
                    detections[
                        "detector"
                    ] == detector
                )
            ]


            gap_rows.append(
                calculate_gap_summary(
                    region=region,
                    detector=detector,
                    rows=rows,
                )
            )


    gap_summary = pd.DataFrame(
        gap_rows
    )


    # -----------------------------------------------------------------
    # Cluster summaries
    # -----------------------------------------------------------------

    cluster_summary_rows = []

    cluster_frames = []


    for region in REGIONS:

        for detector in DETECTORS:

            rows = detections[
                (
                    detections[
                        "region"
                    ] == region
                )
                &
                (
                    detections[
                        "detector"
                    ] == detector
                )
            ]


            for gap_days in CLUSTER_WINDOWS_DAYS:

                (
                    cluster_summary,
                    cluster_frame,
                ) = summarize_clusters(
                    region=region,
                    detector=detector,
                    rows=rows,
                    gap_days=gap_days,
                )


                cluster_summary_rows.append(
                    cluster_summary
                )


                cluster_frames.append(
                    cluster_frame
                )


    cluster_summary = pd.DataFrame(
        cluster_summary_rows
    )


    clusters = pd.concat(
        cluster_frames,
        ignore_index=True,
    )


    # -----------------------------------------------------------------
    # Monthly counts
    # -----------------------------------------------------------------

    monthly = calculate_monthly_counts(
        detections
    )


    # -----------------------------------------------------------------
    # ADWIN/KSWIN overlap
    # -----------------------------------------------------------------

    overlap = detector_overlap(
        detections
    )


    # -----------------------------------------------------------------
    # Error context
    # -----------------------------------------------------------------

    local_context = alarm_local_context(
        detections=detections,
        streams=streams,
    )


    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    gap_path = (
        OUTPUT_DIR
        / "06_alarm_gap_summary.csv"
    )


    cluster_summary_path = (
        OUTPUT_DIR
        / "06_alarm_cluster_summary.csv"
    )


    clusters_path = (
        OUTPUT_DIR
        / "06_alarm_clusters.csv"
    )


    monthly_path = (
        OUTPUT_DIR
        / "06_alarm_monthly_counts.csv"
    )


    overlap_path = (
        OUTPUT_DIR
        / "06_detector_overlap.csv"
    )


    local_path = (
        OUTPUT_DIR
        / "06_alarm_local_error_context.csv"
    )


    gap_summary.to_csv(
        gap_path,
        index=False,
    )


    cluster_summary.to_csv(
        cluster_summary_path,
        index=False,
    )


    clusters.to_csv(
        clusters_path,
        index=False,
    )


    monthly.to_csv(
        monthly_path,
        index=False,
    )


    overlap.to_csv(
        overlap_path,
        index=False,
    )


    local_context.to_csv(
        local_path,
        index=False,
    )


    # -----------------------------------------------------------------
    # Console — inter-alarm gaps
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "INTER-ALARM GAP SUMMARY"
    )

    print(
        "=" * 80
    )


    gap_display = gap_summary[
        [
            "region",
            "detector",
            "detections",
            "min_gap_days",
            "median_gap_days",
            "mean_gap_days",
            "p90_gap_days",
            "fraction_gaps_le_1d",
            "fraction_gaps_le_3d",
            "fraction_gaps_le_7d",
        ]
    ]


    print(
        gap_display.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


    # -----------------------------------------------------------------
    # Console — clusters
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "TEMPORAL CLUSTER SUMMARY"
    )

    print(
        "=" * 80
    )


    cluster_display = cluster_summary[
        [
            "region",
            "detector",
            "gap_rule_days",
            "detections",
            "clusters",
            "singleton_clusters",
            "mean_cluster_size",
            "max_cluster_size",
            "max_cluster_duration_days",
            "retrigger_fraction",
        ]
    ]


    print(
        cluster_display.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


    # -----------------------------------------------------------------
    # Console — detector overlap
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "ADWIN / KSWIN TEMPORAL OVERLAP"
    )

    print(
        "=" * 80
    )


    print(
        overlap.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


    # -----------------------------------------------------------------
    # Monthly concentration
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "MOST ACTIVE MONTHS"
    )

    print(
        "=" * 80
    )


    for region in REGIONS:

        for detector in DETECTORS:

            subset = monthly[
                (
                    monthly[
                        "region"
                    ] == region
                )
                &
                (
                    monthly[
                        "detector"
                    ] == detector
                )
            ]


            top = (
                subset
                .sort_values(
                    "detections",
                    ascending=False,
                )
                .head(
                    5
                )
            )


            print()
            print(
                f"{region} / {detector}"
            )


            if top.empty:

                print(
                    "  no detections"
                )

            else:

                for row in top.itertuples(
                    index=False
                ):

                    print(
                        f"  {row.month}: "
                        f"{row.detections}"
                    )


    # -----------------------------------------------------------------
    # Final interpretation guide
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "HOW TO INTERPRET DIAGNOSIS 06"
    )

    print(
        "=" * 80
    )


    print(
        """
1. Look first at median inter-alarm gap.

   Very short median gaps indicate repeated detections occurring close
   together rather than independent isolated events.


2. Look at retrigger_fraction.

   Example:

       detections = 297
       3-day clusters = 60

       retrigger_fraction =
           (297 - 60) / 297
           = 79.8%

   This would mean that approximately 80% of alarms occur after another
   alarm in the same 3-day-linked temporal cluster.


3. Compare 1-day, 3-day and 7-day cluster counts.

   If:

       297 detections
       1-day clusters = 220
       3-day clusters = 80
       7-day clusters = 30

   alarms are heavily concentrated into prolonged episodes.


4. Compare ADWIN and KSWIN overlap.

   If many ADWIN detections lie near KSWIN detections, both detectors
   may be reacting to the same periods of changing forecast error.

   If overlap is low, their excessive alarm mechanisms may differ.


5. Inspect 06_alarm_local_error_context.csv.

   This tells us whether alarms occur around sustained increases or
   decreases in normalized absolute forecasting error.

   It can help determine whether detectors repeatedly trigger while
   the error process remains in one elevated/local regime.


IMPORTANT
---------
This diagnosis does not classify close alarms as false alarms.

Real AEMO events are not complete ground truth.

The purpose is only to determine whether the large raw detection count
represents many temporally distinct changes or repeated detector
responses during persistent forecasting-error episodes.
"""
    )


    print(
        "=" * 80
    )

    print(
        "Saved diagnostic outputs"
    )

    print(
        "=" * 80
    )


    print(
        gap_path
    )


    print(
        cluster_summary_path
    )


    print(
        clusters_path
    )


    print(
        monthly_path
    )


    print(
        overlap_path
    )


    print(
        local_path
    )


    print()
    print(
        "Do not change detector thresholds or introduce a refractory "
        "period based on this diagnosis alone."
    )


if __name__ == "__main__":
    main()