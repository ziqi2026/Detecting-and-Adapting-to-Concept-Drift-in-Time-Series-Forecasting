"""
05_diag_normalization_effect.py

Diagnosis 05 — Is calibration normalization causing the excessive alarms?
=========================================================================

Background
----------
The current AEMO Task 120 detector input is:

    frozen Gradient Boosting
        -> signed forecast residual
        -> absolute forecast error
        -> calibration mean/std normalization
        -> drift detector

Specifically:

    z_t = (|e_t| - mean_calibration) / std_calibration

Previous diagnostics found:

Diagnosis 01
    Temporal ordering strongly affects detector alarms.

Diagnosis 02
    Short-range/local temporal persistence is especially important.

Diagnosis 03
    The persistence already exists in the signed forecast residuals.

Diagnosis 04
    Gradient Boosting and Seasonal Naive retain very strong residual
    dependence, whereas SARIMAX residuals are substantially less
    dependent.

Before changing the detector input with smoothing, prewhitening, or
another temporal transformation, this diagnosis asks a separate and
simpler question:

    Is the NORMALIZATION step itself responsible for the large number
    of ADWIN/KSWIN detections?

Purpose
-------
This script isolates normalization from all other components.

The following are kept completely unchanged:

    - AEMO data
    - fixed train/calibration/test split
    - Gradient Boosting model
    - GB lags (48, 336)
    - forecast residuals
    - absolute-error representation
    - detector configurations
    - chronological ordering

Only the numerical transformation applied to the SAME absolute-error
stream is changed.

Representations
---------------
For the same test absolute errors |e_t|, compare:

1. raw_absolute

       |e_t|

   No normalization.


2. current_calibration_z

       (|e_t| - calibration_mean)
       / calibration_std

   This is the CURRENT Task 120 representation.


3. calibration_center_only

       |e_t| - calibration_mean

   Removes only the calibration location.


4. calibration_scale_only

       |e_t| / calibration_std

   Applies only the calibration scale.


5. robust_calibration_z

       (|e_t| - calibration_median)
       / robust_scale

   where:

       robust_scale = 1.4826 * MAD

   This checks whether the ordinary mean/std estimates are themselves
   unusually influenced by the calibration error distribution.


6. log_absolute

       log(1 + |e_t|)

   This is NOT proposed as a replacement.

   It is included only as an additional diagnostic transformation
   that compresses extreme magnitudes while preserving observation
   ordering.


Important interpretation
------------------------
The transformations above do NOT change the time ordering.

For affine transformations such as:

    raw absolute error
    current calibration z-score
    center-only
    scale-only
    robust z-score

the temporal ordering of values is identical.

Therefore:

    If KSWIN produces the same changepoints across these streams,
    that is strong evidence that the current mean/std normalization
    is NOT responsible for KSWIN's excessive alarms.

This is particularly important because KSWIN compares empirical
distributions rather than using the absolute physical scale directly.

For ADWIN, we also compare the resulting alarm counts and changepoints.

For Page-Hinkley, raw-scale comparisons must be interpreted carefully
because its fixed threshold has numerical units. A threshold selected
for a standardized stream is not directly equivalent on raw MW errors.

For that reason this script reports Page-Hinkley results but does NOT
use raw-vs-normalized PH counts alone to decide whether normalization
is good or bad.


Second question: is the calibration reference stable?
-----------------------------------------------------
This script also inspects the CURRENT normalized test stream over time.

For every calendar month, it reports:

    - mean normalized absolute error
    - standard deviation
    - median
    - 90th percentile
    - 95th percentile
    - 99th percentile

This helps distinguish:

A) normalization problem

from

B) genuine changes in the forecast-error process relative to the
   fixed Jan-Feb 2020 calibration reference.

For example, if normalized error has a persistent positive mean for
large portions of the test period, this does NOT necessarily mean the
normalization is wrong. It means later forecast errors differ from the
calibration reference.

That distinction is important in a concept-drift experiment.


Autocorrelation check
---------------------
The script calculates lag-1, lag-2, daily (48), and weekly (336)
autocorrelation for each representation.

A fixed linear normalization should NOT remove the strong temporal
dependence previously observed.

Therefore, if:

    raw_absolute lag-1 ACF       ~= 0.97
    current_calibration_z ACF    ~= 0.97

then normalization did not create the autocorrelation.


This script does NOT
--------------------
    - modify detectors.py
    - tune any detector
    - modify Gradient Boosting
    - change lags
    - perform prewhitening
    - perform smoothing
    - aggregate observations
    - change T2 matching
    - overwrite Task 120 detections
    - write to runs.csv

Run
---
From repository root:

    python experiments/detection/05_diag_normalization_effect.py


Outputs
-------
Saved to:

    results/tables/detection/diagnostics/

Files:

    05_normalization_summary.csv
    05_normalization_detector_counts.csv
    05_normalization_monthly_stability.csv
    05_normalization_acf.csv
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
# Fixed project configuration
# =====================================================================

REGIONS = (
    "SA1",
    "NSW1",
)


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


# EXACT Gradient Boosting configuration used by Task 120.
FORECASTER_CONFIG = {
    "feature_mode": "calendar+lags",
    "lags": (48, 336),
    "seed": 0,
}


REPRESENTATION_ORDER = (
    "raw_absolute",
    "current_calibration_z",
    "calibration_center_only",
    "calibration_scale_only",
    "robust_calibration_z",
    "log_absolute",
)


ACF_LAGS = (
    1,      # 30 minutes
    2,      # 1 hour
    48,     # 24 hours
    96,     # 48 hours
    336,    # 7 days
)


# =====================================================================
# Validation
# =====================================================================

def validate_stream(
    values,
    name,
):
    """
    Validate a finite one-dimensional numeric stream.
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
# Build exactly the Task 120 forecast errors
# =====================================================================

def build_error_streams(
    train,
    calibration,
    test,
):
    """
    Fit the current frozen Gradient Boosting forecaster using TRAIN only
    and generate calibration and test forecast-error streams.

    This reproduces the forecasting part of Task 120.

    Calibration:
        history_df=train

    Test:
        history_df=calibration

    Returns
    -------
    calibration_errors : pd.DataFrame

    test_errors : pd.DataFrame
    """

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


    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    if len(
        calibration_errors
    ) != len(
        calibration
    ):

        raise RuntimeError(
            "Calibration error-stream length mismatch."
        )


    if len(
        test_errors
    ) != len(
        test
    ):

        raise RuntimeError(
            "Test error-stream length mismatch."
        )


    validate_stream(
        calibration_errors[
            "error"
        ],
        "calibration forecast errors",
    )


    validate_stream(
        test_errors[
            "error"
        ],
        "test forecast errors",
    )


    return (
        calibration_errors,
        test_errors,
    )


# =====================================================================
# Calibration statistics
# =====================================================================

def calculate_calibration_statistics(
    calibration_absolute,
):
    """
    Calculate ordinary and robust calibration statistics.
    """

    values = validate_stream(
        calibration_absolute,
        "calibration absolute errors",
    )


    mean = float(
        values.mean()
    )


    std = float(
        values.std()
    )


    median = float(
        np.median(
            values
        )
    )


    mad = float(
        np.median(
            np.abs(
                values
                - median
            )
        )
    )


    robust_scale = float(
        1.4826
        * mad
    )


    if (
        not np.isfinite(std)
        or std <= 0
    ):

        raise RuntimeError(
            "Calibration standard deviation "
            "must be finite and > 0."
        )


    if (
        not np.isfinite(robust_scale)
        or robust_scale <= 0
    ):

        raise RuntimeError(
            "Calibration robust scale "
            "must be finite and > 0."
        )


    return {
        "mean":
            mean,

        "std":
            std,

        "median":
            median,

        "mad":
            mad,

        "robust_scale":
            robust_scale,
    }


# =====================================================================
# Construct normalization variants
# =====================================================================

def build_representations(
    test_absolute,
    calibration_stats,
):
    """
    Construct all diagnostic transformations from EXACTLY the same
    chronological test absolute-error values.

    Nothing about the forecasting model or temporal ordering changes.
    """

    values = validate_stream(
        test_absolute,
        "test absolute error",
    )


    mean = calibration_stats[
        "mean"
    ]


    std = calibration_stats[
        "std"
    ]


    median = calibration_stats[
        "median"
    ]


    robust_scale = calibration_stats[
        "robust_scale"
    ]


    representations = {

        # ---------------------------------------------------------
        # No normalization
        # ---------------------------------------------------------

        "raw_absolute":
            values.copy(),


        # ---------------------------------------------------------
        # CURRENT TASK 120
        # ---------------------------------------------------------

        "current_calibration_z":
            (
                values
                - mean
            ) / std,


        # ---------------------------------------------------------
        # Location only
        # ---------------------------------------------------------

        "calibration_center_only":
            (
                values
                - mean
            ),


        # ---------------------------------------------------------
        # Scale only
        # ---------------------------------------------------------

        "calibration_scale_only":
            (
                values
                / std
            ),


        # ---------------------------------------------------------
        # Robust affine normalization
        # ---------------------------------------------------------

        "robust_calibration_z":
            (
                values
                - median
            ) / robust_scale,


        # ---------------------------------------------------------
        # Diagnostic nonlinear magnitude compression
        # ---------------------------------------------------------

        "log_absolute":
            np.log1p(
                values
            ),
    }


    for (
        name,
        stream,
    ) in representations.items():

        validate_stream(
            stream,
            name,
        )


        if len(
            stream
        ) != len(
            values
        ):

            raise RuntimeError(
                f"{name}: stream length changed."
            )


    return representations


# =====================================================================
# Autocorrelation
# =====================================================================

def calculate_acf(
    region,
    representation,
    values,
):
    """
    Calculate descriptive sample autocorrelation.

    This explicitly checks whether normalization changes the temporal
    dependence found in Diagnoses 01-04.
    """

    values = validate_stream(
        values,
        representation,
    )


    series = pd.Series(
        values,
        dtype=float,
    )


    rows = []


    for lag in ACF_LAGS:

        if lag >= len(
            series
        ):

            value = float(
                "nan"
            )

        else:

            value = float(
                series.autocorr(
                    lag=lag
                )
            )


        rows.append(
            {
                "region":
                    region,

                "representation":
                    representation,

                "lag_observations":
                    lag,

                "lag_hours":
                    lag / 2,

                "autocorrelation":
                    value,
            }
        )


    return rows


# =====================================================================
# Detector runner
# =====================================================================

def run_detectors(
    region,
    representation,
    stream,
):
    """
    Run all CURRENT detector configurations on one diagnostic stream.

    IMPORTANT
    ---------
    Page-Hinkley uses a fixed numerical threshold, so comparisons
    involving raw MW-scale errors must be interpreted cautiously.

    KSWIN is particularly useful here because its distributional
    comparison should not depend on a simple positive affine rescaling.
    """

    stream = validate_stream(
        stream,
        "detector stream",
    )


    rows = []


    for (
        detector_name,
        entry,
    ) in DETECTOR_REGISTRY.items():

        config = entry[
            "config"
        ].copy()


        changepoints = entry[
            "func"
        ](
            stream,
            **config,
        )


        rows.append(
            {
                "region":
                    region,

                "representation":
                    representation,

                "detector":
                    detector_name,

                "detections":
                    len(
                        changepoints
                    ),

                "changepoints":
                    changepoints,

                "detector_config":
                    json.dumps(
                        config,
                        sort_keys=True,
                    ),
            }
        )


    return rows


# =====================================================================
# Changepoint comparison
# =====================================================================

def compare_to_current(
    detector_rows,
):
    """
    Compare every representation against CURRENT calibration z-score
    changepoints for the same region/detector.

    Returns serializable summary rows.
    """

    result = []


    by_key = {
        (
            row["region"],
            row["representation"],
            row["detector"],
        ):
        row
        for row
        in detector_rows
    }


    regions = sorted(
        {
            row["region"]
            for row
            in detector_rows
        }
    )


    detectors = list(
        DETECTOR_REGISTRY
    )


    for region in regions:

        for detector in detectors:

            current = by_key[
                (
                    region,
                    "current_calibration_z",
                    detector,
                )
            ]


            current_cps = set(
                current[
                    "changepoints"
                ]
            )


            for representation in REPRESENTATION_ORDER:

                row = by_key[
                    (
                        region,
                        representation,
                        detector,
                    )
                ]


                candidate_cps = set(
                    row[
                        "changepoints"
                    ]
                )


                intersection = (
                    current_cps
                    &
                    candidate_cps
                )


                union = (
                    current_cps
                    |
                    candidate_cps
                )


                if len(
                    union
                ) == 0:

                    jaccard = 1.0

                else:

                    jaccard = (
                        len(
                            intersection
                        )
                        /
                        len(
                            union
                        )
                    )


                exact_same = (
                    sorted(
                        current_cps
                    )
                    ==
                    sorted(
                        candidate_cps
                    )
                )


                result.append(
                    {
                        "region":
                            region,

                        "detector":
                            detector,

                        "representation":
                            representation,

                        "detections":
                            row[
                                "detections"
                            ],

                        "current_detections":
                            current[
                                "detections"
                            ],

                        "exact_same_changepoints_as_current":
                            exact_same,

                        "n_shared_changepoints":
                            len(
                                intersection
                            ),

                        "jaccard_vs_current":
                            jaccard,

                        "detector_config":
                            row[
                                "detector_config"
                            ],
                    }
                )


    return result


# =====================================================================
# Monthly stability of current normalized stream
# =====================================================================

def monthly_normalization_diagnostic(
    region,
    test_errors,
    current_stream,
):
    """
    Inspect how the CURRENT calibration-normalized absolute-error stream
    evolves month-by-month during the test period.

    This is not detector evaluation.

    It describes how different later forecast errors are from the fixed
    Jan-Feb 2020 calibration reference.
    """

    if len(
        test_errors
    ) != len(
        current_stream
    ):

        raise RuntimeError(
            "Monthly diagnostic length mismatch."
        )


    frame = pd.DataFrame(
        {
            "SETTLEMENTDATE":
                pd.to_datetime(
                    test_errors[
                        "SETTLEMENTDATE"
                    ]
                ),

            "normalized_abs_error":
                current_stream,
        }
    )


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


    rows = []


    for (
        month,
        group,
    ) in frame.groupby(
        "month",
        sort=True,
    ):

        values = (
            group[
                "normalized_abs_error"
            ]
            .to_numpy(
                dtype=float
            )
        )


        rows.append(
            {
                "region":
                    region,

                "month":
                    month,

                "n_observations":
                    len(
                        values
                    ),

                "mean":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "std":
                    float(
                        np.std(
                            values
                        )
                    ),

                "median":
                    float(
                        np.median(
                            values
                        )
                    ),

                "p90":
                    float(
                        np.quantile(
                            values,
                            0.90,
                        )
                    ),

                "p95":
                    float(
                        np.quantile(
                            values,
                            0.95,
                        )
                    ),

                "p99":
                    float(
                        np.quantile(
                            values,
                            0.99,
                        )
                    ),
            }
        )


    return rows


# =====================================================================
# One-region diagnosis
# =====================================================================

def diagnose_region(
    region,
):
    """
    Run Diagnosis 05 for one AEMO region.
    """

    print()
    print(
        "=" * 80
    )

    print(
        f"REGION: {region}"
    )

    print(
        "=" * 80
    )


    (
        _,
        train,
        calibration,
        test,
    ) = load_and_split_aemo(
        region=region,
        data_path=DATA_PATH,
    )


    print(
        f"Training observations    : "
        f"{len(train):,}"
    )


    print(
        f"Calibration observations : "
        f"{len(calibration):,}"
    )


    print(
        f"Test observations        : "
        f"{len(test):,}"
    )


    (
        calibration_errors,
        test_errors,
    ) = build_error_streams(
        train=train,
        calibration=calibration,
        test=test,
    )


    calibration_absolute = (
        calibration_errors[
            "error"
        ]
        .abs()
        .to_numpy(
            dtype=float
        )
    )


    test_absolute = (
        test_errors[
            "error"
        ]
        .abs()
        .to_numpy(
            dtype=float
        )
    )


    stats = (
        calculate_calibration_statistics(
            calibration_absolute
        )
    )


    print()
    print(
        "Calibration absolute-error statistics:"
    )


    print(
        f"  mean         : "
        f"{stats['mean']:.6f}"
    )


    print(
        f"  std          : "
        f"{stats['std']:.6f}"
    )


    print(
        f"  median       : "
        f"{stats['median']:.6f}"
    )


    print(
        f"  MAD          : "
        f"{stats['mad']:.6f}"
    )


    print(
        f"  robust scale : "
        f"{stats['robust_scale']:.6f}"
    )


    # -------------------------------------------------------------
    # Construct transformations
    # -------------------------------------------------------------

    representations = (
        build_representations(
            test_absolute,
            stats,
        )
    )


    # -------------------------------------------------------------
    # ACF
    # -------------------------------------------------------------

    acf_rows = []


    for (
        name,
        values,
    ) in representations.items():

        acf_rows.extend(
            calculate_acf(
                region=region,
                representation=name,
                values=values,
            )
        )


    # -------------------------------------------------------------
    # Detector runs
    # -------------------------------------------------------------

    detector_rows = []


    for (
        name,
        values,
    ) in representations.items():

        detector_rows.extend(
            run_detectors(
                region=region,
                representation=name,
                stream=values,
            )
        )


    # -------------------------------------------------------------
    # Monthly current-normalization stability
    # -------------------------------------------------------------

    monthly_rows = (
        monthly_normalization_diagnostic(
            region=region,
            test_errors=test_errors,
            current_stream=representations[
                "current_calibration_z"
            ],
        )
    )


    # -------------------------------------------------------------
    # Basic representation summary
    # -------------------------------------------------------------

    summary_rows = []


    for (
        name,
        values,
    ) in representations.items():

        values = np.asarray(
            values,
            dtype=float,
        )


        lag1 = float(
            pd.Series(
                values
            ).autocorr(
                lag=1
            )
        )


        summary_rows.append(
            {
                "region":
                    region,

                "representation":
                    name,

                "mean":
                    float(
                        values.mean()
                    ),

                "std":
                    float(
                        values.std()
                    ),

                "min":
                    float(
                        values.min()
                    ),

                "median":
                    float(
                        np.median(
                            values
                        )
                    ),

                "max":
                    float(
                        values.max()
                    ),

                "acf_lag1":
                    lag1,

                "calibration_abs_mean":
                    stats[
                        "mean"
                    ],

                "calibration_abs_std":
                    stats[
                        "std"
                    ],

                "calibration_abs_median":
                    stats[
                        "median"
                    ],

                "calibration_abs_mad":
                    stats[
                        "mad"
                    ],

                "calibration_robust_scale":
                    stats[
                        "robust_scale"
                    ],
            }
        )


    return (
        summary_rows,
        acf_rows,
        detector_rows,
        monthly_rows,
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "DIAGNOSIS 05 — NORMALIZATION EFFECT"
    )

    print(
        "=" * 80
    )


    print(
        "Question:"
    )


    print(
        "Is the current calibration mean/std normalization responsible "
        "for the excessive ADWIN and KSWIN alarms?"
    )


    print()
    print(
        "Gradient Boosting remains unchanged:"
    )


    print(
        "  calendar + lags (48, 336)"
    )


    print()
    print(
        "Absolute forecast errors remain unchanged."
    )


    print(
        "Chronological ordering remains unchanged."
    )


    print(
        "Detector configurations remain unchanged."
    )


    print(
        "No prewhitening or smoothing is performed."
    )


    print(
        "runs.csv will not be modified."
    )


    all_summary_rows = []
    all_acf_rows = []
    all_detector_rows = []
    all_monthly_rows = []


    # -------------------------------------------------------------
    # Regions
    # -------------------------------------------------------------

    for region in REGIONS:

        (
            summary_rows,
            acf_rows,
            detector_rows,
            monthly_rows,
        ) = diagnose_region(
            region
        )


        all_summary_rows.extend(
            summary_rows
        )


        all_acf_rows.extend(
            acf_rows
        )


        all_detector_rows.extend(
            detector_rows
        )


        all_monthly_rows.extend(
            monthly_rows
        )


    # -------------------------------------------------------------
    # DataFrames
    # -------------------------------------------------------------

    summary = pd.DataFrame(
        all_summary_rows
    )


    acf = pd.DataFrame(
        all_acf_rows
    )


    monthly = pd.DataFrame(
        all_monthly_rows
    )


    detector_comparison = pd.DataFrame(
        compare_to_current(
            all_detector_rows
        )
    )


    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    summary_path = (
        OUTPUT_DIR
        / "05_normalization_summary.csv"
    )


    detector_path = (
        OUTPUT_DIR
        / "05_normalization_detector_counts.csv"
    )


    monthly_path = (
        OUTPUT_DIR
        / "05_normalization_monthly_stability.csv"
    )


    acf_path = (
        OUTPUT_DIR
        / "05_normalization_acf.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    detector_comparison.to_csv(
        detector_path,
        index=False,
    )


    monthly.to_csv(
        monthly_path,
        index=False,
    )


    acf.to_csv(
        acf_path,
        index=False,
    )


    # -------------------------------------------------------------
    # Console: ACF comparison
    # -------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "LAG-1 AUTOCORRELATION BY REPRESENTATION"
    )

    print(
        "=" * 80
    )


    acf1 = (
        summary[
            [
                "region",
                "representation",
                "acf_lag1",
            ]
        ]
    )


    print(
        acf1.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # -------------------------------------------------------------
    # Console: detector comparison
    # -------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "DETECTOR COUNTS BY NORMALIZATION"
    )

    print(
        "=" * 80
    )


    display = detector_comparison[
        [
            "region",
            "detector",
            "representation",
            "detections",
            "current_detections",
            "exact_same_changepoints_as_current",
            "jaccard_vs_current",
        ]
    ]


    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # -------------------------------------------------------------
    # Focus specifically on KSWIN
    # -------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "KSWIN NORMALIZATION CHECK"
    )

    print(
        "=" * 80
    )


    kswin = detector_comparison[
        detector_comparison[
            "detector"
        ] == "kswin"
    ]


    print(
        kswin[
            [
                "region",
                "representation",
                "detections",
                "exact_same_changepoints_as_current",
                "jaccard_vs_current",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # -------------------------------------------------------------
    # Monthly stability summary
    # -------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "CURRENT NORMALIZED STREAM — MONTHLY RANGE"
    )

    print(
        "=" * 80
    )


    monthly_summary = (
        monthly
        .groupby(
            "region",
            as_index=False,
        )
        .agg(
            monthly_mean_min=(
                "mean",
                "min",
            ),
            monthly_mean_max=(
                "mean",
                "max",
            ),
            monthly_std_min=(
                "std",
                "min",
            ),
            monthly_std_max=(
                "std",
                "max",
            ),
            monthly_p95_min=(
                "p95",
                "min",
            ),
            monthly_p95_max=(
                "p95",
                "max",
            ),
        )
    )


    print(
        monthly_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # -------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "HOW TO INTERPRET DIAGNOSIS 05"
    )

    print(
        "=" * 80
    )


    print(
        """
1. Compare RAW ABSOLUTE vs CURRENT CALIBRATION Z.

   If KSWIN has the same detections/changepoints:
       The calibration z-normalization is NOT causing KSWIN's
       excessive alarm count.

2. Compare lag-1 ACF.

   If raw_absolute ACF and current_calibration_z ACF are identical:
       Normalization did NOT create the temporal dependence found
       in Diagnoses 01-04.

3. Compare robust_calibration_z with current_calibration_z.

   If results remain similar:
       Ordinary calibration mean/std estimation is unlikely to be
       the main problem.

4. Treat Page-Hinkley raw-scale results cautiously.

   Its fixed threshold is scale-dependent. A raw MW stream and a
   standardized stream do not give it the same effective threshold.

5. Inspect monthly stability.

   Large changes in monthly normalized-error statistics mean that the
   later forecast-error process differs from the Jan-Feb 2020 reference.

   That is not automatically a normalization error. It may be the
   non-stationarity / drift that the experiment is intended to study.
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
        summary_path
    )

    print(
        detector_path
    )

    print(
        monthly_path
    )

    print(
        acf_path
    )


    print()
    print(
        "Do not change normalization or detectors.py yet."
    )


    print(
        "Use this diagnosis first to establish whether normalization "
        "actually changes ADWIN/KSWIN behaviour."
    )


if __name__ == "__main__":
    main()