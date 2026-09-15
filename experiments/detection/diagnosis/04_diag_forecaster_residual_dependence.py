"""
04_diag_forecaster_residual_dependence.py

Diagnosis 04 — Is residual persistence specific to Gradient Boosting?
=====================================================================

Background
----------
The previous diagnostics found:

Diagnosis 01
    The chronological calibration forecast-error stream generated
    detector alarms, while full random shuffling of the exact same
    values reduced all detector alarms to zero.

Diagnosis 02
    Breaking short-range / within-day temporal ordering substantially
    reduced ADWIN and KSWIN alarms, while shuffling whole days or weeks
    did not remove the problem.

Diagnosis 03
    Very strong short-range dependence was already present in the
    SIGNED forecast residuals:

        SA1  signed-error lag-1 ACF ≈ 0.983
        NSW1 signed-error lag-1 ACF ≈ 0.985

    Therefore taking absolute values is not the original source of the
    dependence.

Purpose
-------
The next question is:

    Is this extreme residual persistence mainly a property of the
    selected Gradient Boosting forecaster?

or:

    Does similar residual persistence occur across the project's
    existing forecasting baselines?

This distinction matters before applying treatments such as:

    - prewhitening
    - additional short-term lag features
    - smoothing
    - aggregation
    - detector-input transformations

If all forecasters show very strong residual persistence, the problem
is more likely a general property of the electricity-demand forecasting
error process / evaluation setup.

If Gradient Boosting is dramatically worse than the other forecasters,
then the next diagnosis should focus on the Gradient Boosting model and
its feature specification.

Forecasters compared
--------------------
The project's existing required baselines are used unchanged:

    1. Seasonal Naive
    2. SARIMAX
    3. Gradient Boosting

No new model is introduced.

PatchTST is deliberately excluded because it is an optional additional
model and is not required to answer this diagnosis.

Protocol
--------
For each region:

    SA1
    NSW1

use the fixed project split:

    Training:
        2018-01-01 to 2019-12-31

    Calibration:
        2020-01-01 to 2020-02-29

Each forecaster is fitted using training data only.

Then calibration forecasts are generated causally.

IMPORTANT history handling
--------------------------
The existing forecasters use history_df differently.

Seasonal Naive
    Needs the immediately preceding training observations to provide
    the previous-day values for the beginning of calibration.

    Therefore:

        history_df=train

Gradient Boosting
    Uses lagged demand features (48 and 336). Training history is needed
    to construct lag features for the beginning of calibration.

    Therefore:

        history_df=train

SARIMAX
    After fit(train), the fitted state-space model already ends at the
    end of the training period.

    Passing train again as history_df would incorrectly attempt to
    advance the fitted state through the training period a second time.

    Therefore, for forecasting the immediately following calibration
    period:

        history_df=None

The existing SARIMAX predict() method then extends the frozen fitted
state chronologically through calibration without re-estimating model
parameters.

Diagnostics
-----------
For each forecaster and region calculate:

Forecast quality:
    - MAE
    - mean signed error
    - mean absolute error
    - signed-error standard deviation

Signed residual dependence:
    - ACF at lags:
        1, 2, 3, 4, 6, 12, 24, 48, 96, 336

Absolute-error dependence:
    - same ACF lags

Directional persistence:
    - fraction positive residuals
    - fraction negative residuals
    - fraction consecutive residual pairs with same sign
    - number of sign runs
    - mean sign-run length
    - median sign-run length
    - maximum sign-run length

Detector response:
    Each forecaster's absolute calibration errors are standardized using
    that forecaster's own calibration mean and standard deviation:

        z_t = (|e_t| - mean_calibration) / std_calibration

    The existing frozen ADWIN, Page-Hinkley, and KSWIN configurations
    are then applied.

Important
---------
Detector counts in this script are DIAGNOSTIC.

They are not a new benchmark and do not replace Task 120.

Different forecasters produce different forecast-error processes, so
the detector counts are used only to see whether high alarm rates track
high residual dependence.

This script does NOT:

    - change detectors.py
    - modify DETECTOR_REGISTRY
    - change any forecaster configuration
    - retrain during calibration
    - change Task 120
    - change T2
    - apply prewhitening
    - smooth the stream
    - aggregate the data
    - write to runs.csv

Run
---
From repository root:

    python experiments/detection/04_diag_forecaster_residual_dependence.py

Outputs
-------
Saved under:

    results/tables/detection/diagnostics/

Files:

    04_forecaster_residual_summary.csv
    04_forecaster_residual_acf.csv
    04_forecaster_detector_counts.csv
    04_forecaster_residual_streams.csv.gz
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

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
    SeasonalNaiveForecaster,
    SARIMAXForecaster,
    GradientBoostedForecaster,
    frozen_error_stream,
)

from drift_lab.detectors import (
    DETECTOR_REGISTRY,
)


# ---------------------------------------------------------------------
# Fixed diagnosis settings
# ---------------------------------------------------------------------

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


# Same existing project configurations.
FORECASTERS = {

    "seasonal_naive": {
        "display_name": "Seasonal Naive",
        "class": SeasonalNaiveForecaster,
        "config": {
            "season": 48,
        },
        "calibration_history": "train",
    },

    "sarimax": {
        "display_name": "SARIMAX",
        "class": SARIMAXForecaster,
        "config": {},
        "calibration_history": None,
    },

    "gradient_boosted": {
        "display_name": "Gradient Boosting",
        "class": GradientBoostedForecaster,
        "config": {
            "feature_mode": "calendar+lags",
            "lags": (48, 336),
            "seed": 0,
        },
        "calibration_history": "train",
    },
}


# AEMO is standardized to 30-minute observations.
ACF_LAGS = (
    1,      # 30 minutes
    2,      # 1 hour
    3,      # 1.5 hours
    4,      # 2 hours
    6,      # 3 hours
    12,     # 6 hours
    24,     # 12 hours
    48,     # 24 hours
    96,     # 48 hours
    336,    # 7 days
)


# ---------------------------------------------------------------------
# Stream validation
# ---------------------------------------------------------------------

def validate_numeric_stream(
    values,
    name,
):
    """
    Validate a one-dimensional finite numeric stream.
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


# ---------------------------------------------------------------------
# Forecast one calibration period
# ---------------------------------------------------------------------

def forecast_calibration(
    forecaster_name,
    train,
    calibration,
):
    """
    Fit one existing project forecaster on training data and generate
    causal forecasts over the immediately following calibration period.

    Returns
    -------
    errors : pd.DataFrame

        SETTLEMENTDATE
        y_true
        y_pred
        error

    model_config : dict
        Frozen forecaster configuration used.
    """

    spec = FORECASTERS[
        forecaster_name
    ]


    model_class = spec[
        "class"
    ]


    model_config = spec[
        "config"
    ].copy()


    model = model_class(
        **model_config
    )


    # -------------------------------------------------------------
    # Fit using training only
    # -------------------------------------------------------------

    model.fit(
        train
    )


    # -------------------------------------------------------------
    # Model-specific causal history handling
    # -------------------------------------------------------------

    history_mode = spec[
        "calibration_history"
    ]


    if history_mode == "train":

        history_df = train

    elif history_mode is None:

        history_df = None

    else:

        raise RuntimeError(
            f"Unsupported calibration history mode "
            f"for {forecaster_name}: {history_mode}"
        )


    errors = frozen_error_stream(
        model,
        calibration,
        history_df=history_df,
    )


    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    required = {
        "SETTLEMENTDATE",
        "y_true",
        "y_pred",
        "error",
    }


    if not required.issubset(
        errors.columns
    ):

        raise RuntimeError(
            f"{forecaster_name}: forecasting output "
            "is missing required columns."
        )


    if len(errors) != len(
        calibration
    ):

        raise RuntimeError(
            f"{forecaster_name}: calibration prediction "
            "length mismatch."
        )


    y_true = validate_numeric_stream(
        errors["y_true"],
        f"{forecaster_name} y_true",
    )


    y_pred = validate_numeric_stream(
        errors["y_pred"],
        f"{forecaster_name} y_pred",
    )


    residual = validate_numeric_stream(
        errors["error"],
        f"{forecaster_name} residual",
    )


    expected = (
        y_true
        - y_pred
    )


    if not np.allclose(
        residual,
        expected,
        rtol=1e-10,
        atol=1e-10,
    ):

        raise RuntimeError(
            f"{forecaster_name}: stored residual does "
            "not equal y_true - y_pred."
        )


    return (
        errors,
        model_config,
    )


# ---------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------

def calculate_acf(
    region,
    forecaster,
    representation,
    values,
):
    """
    Calculate descriptive sample autocorrelation at the configured lags.
    """

    values = validate_numeric_stream(
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

                "forecaster":
                    forecaster,

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


# ---------------------------------------------------------------------
# Sign-run diagnostics
# ---------------------------------------------------------------------

def calculate_sign_diagnostics(
    residual,
):
    """
    Describe persistence in forecast-error direction.

    Positive residual:
        observed > predicted
        model under-predicted

    Negative residual:
        observed < predicted
        model over-predicted
    """

    residual = validate_numeric_stream(
        residual,
        "signed residual",
    )


    signs = np.sign(
        residual
    )


    positive_fraction = float(
        np.mean(
            residual > 0
        )
    )


    negative_fraction = float(
        np.mean(
            residual < 0
        )
    )


    zero_fraction = float(
        np.mean(
            residual == 0
        )
    )


    # -------------------------------------------------------------
    # Consecutive sign agreement
    # -------------------------------------------------------------

    previous = signs[:-1]
    current = signs[1:]


    valid_pairs = (
        (previous != 0)
        &
        (current != 0)
    )


    if valid_pairs.any():

        same_sign_fraction = float(
            np.mean(
                previous[valid_pairs]
                ==
                current[valid_pairs]
            )
        )


        sign_change_fraction = float(
            np.mean(
                previous[valid_pairs]
                !=
                current[valid_pairs]
            )
        )

    else:

        same_sign_fraction = float(
            "nan"
        )


        sign_change_fraction = float(
            "nan"
        )


    # -------------------------------------------------------------
    # Sign-run lengths
    #
    # Zeros break runs rather than connecting two non-zero runs.
    # Exact zero forecast errors are expected to be rare.
    # -------------------------------------------------------------

    run_lengths = []

    current_sign = None
    current_length = 0


    for sign in signs:

        if sign == 0:

            if current_length > 0:

                run_lengths.append(
                    current_length
                )

            current_sign = None
            current_length = 0

            continue


        if current_sign is None:

            current_sign = sign
            current_length = 1


        elif sign == current_sign:

            current_length += 1


        else:

            run_lengths.append(
                current_length
            )

            current_sign = sign
            current_length = 1


    if current_length > 0:

        run_lengths.append(
            current_length
        )


    if len(run_lengths) > 0:

        n_sign_runs = int(
            len(
                run_lengths
            )
        )


        mean_run_length = float(
            np.mean(
                run_lengths
            )
        )


        median_run_length = float(
            np.median(
                run_lengths
            )
        )


        max_run_length = int(
            np.max(
                run_lengths
            )
        )

    else:

        n_sign_runs = 0

        mean_run_length = float(
            "nan"
        )

        median_run_length = float(
            "nan"
        )

        max_run_length = 0


    return {
        "positive_fraction":
            positive_fraction,

        "negative_fraction":
            negative_fraction,

        "zero_fraction":
            zero_fraction,

        "same_sign_fraction":
            same_sign_fraction,

        "sign_change_fraction":
            sign_change_fraction,

        "n_sign_runs":
            n_sign_runs,

        "mean_sign_run_length":
            mean_run_length,

        "median_sign_run_length":
            median_run_length,

        "max_sign_run_length":
            max_run_length,
    }


# ---------------------------------------------------------------------
# Normalize absolute error
# ---------------------------------------------------------------------

def normalize_absolute_error(
    residual,
):
    """
    Reproduce the Task-120 detector representation for one forecaster:

        residual
            -> absolute residual
            -> calibration mean/std normalization

    Each forecaster uses its own calibration absolute-error mean/std.
    """

    residual = validate_numeric_stream(
        residual,
        "residual",
    )


    absolute_error = np.abs(
        residual
    )


    mean = float(
        absolute_error.mean()
    )


    std = float(
        absolute_error.std()
    )


    if not np.isfinite(
        mean
    ):

        raise RuntimeError(
            "Absolute-error mean is invalid."
        )


    if (
        not np.isfinite(std)
        or std <= 0
    ):

        raise RuntimeError(
            "Absolute-error standard deviation "
            "must be finite and > 0."
        )


    normalized = (
        absolute_error
        - mean
    ) / std


    validate_numeric_stream(
        normalized,
        "normalized absolute-error stream",
    )


    return (
        normalized,
        mean,
        std,
    )


# ---------------------------------------------------------------------
# Run frozen detectors
# ---------------------------------------------------------------------

def run_detectors(
    region,
    forecaster,
    stream,
):
    """
    Run every existing detector configuration on one normalized
    absolute-error calibration stream.

    Counts are diagnostic only.
    """

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

                "forecaster":
                    forecaster,

                "detector":
                    detector_name,

                "detections":
                    len(
                        changepoints
                    ),

                "n_observations":
                    len(
                        stream
                    ),

                "detections_per_10000":
                    len(changepoints)
                    / len(stream)
                    * 10000,

                "detector_config":
                    json.dumps(
                        config,
                        sort_keys=True,
                    ),
            }
        )


    return rows


# ---------------------------------------------------------------------
# One forecaster / region diagnosis
# ---------------------------------------------------------------------

def diagnose_forecaster(
    region,
    forecaster_name,
    train,
    calibration,
):
    """
    Calculate residual dependence, forecast quality, sign persistence,
    and detector counts for one forecaster.
    """

    display_name = FORECASTERS[
        forecaster_name
    ][
        "display_name"
    ]


    print()
    print(
        "-" * 80
    )

    print(
        f"{region} / {display_name}"
    )

    print(
        "-" * 80
    )


    (
        errors,
        model_config,
    ) = forecast_calibration(
        forecaster_name=forecaster_name,
        train=train,
        calibration=calibration,
    )


    residual = validate_numeric_stream(
        errors["error"],
        "residual",
    )


    absolute_error = np.abs(
        residual
    )


    # -------------------------------------------------------------
    # Forecast-error quality
    # -------------------------------------------------------------

    mae = float(
        absolute_error.mean()
    )


    mean_signed_error = float(
        residual.mean()
    )


    residual_std = float(
        residual.std()
    )


    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )


    # -------------------------------------------------------------
    # ACF
    # -------------------------------------------------------------

    signed_acf_rows = calculate_acf(
        region=region,
        forecaster=forecaster_name,
        representation="signed_error",
        values=residual,
    )


    absolute_acf_rows = calculate_acf(
        region=region,
        forecaster=forecaster_name,
        representation="absolute_error",
        values=absolute_error,
    )


    acf_rows = (
        signed_acf_rows
        + absolute_acf_rows
    )


    signed_lookup = {
        row[
            "lag_observations"
        ]:
        row[
            "autocorrelation"
        ]
        for row
        in signed_acf_rows
    }


    absolute_lookup = {
        row[
            "lag_observations"
        ]:
        row[
            "autocorrelation"
        ]
        for row
        in absolute_acf_rows
    }


    # -------------------------------------------------------------
    # Sign persistence
    # -------------------------------------------------------------

    sign_metrics = calculate_sign_diagnostics(
        residual
    )


    # -------------------------------------------------------------
    # Detector input
    # -------------------------------------------------------------

    (
        normalized_abs,
        calibration_abs_mean,
        calibration_abs_std,
    ) = normalize_absolute_error(
        residual
    )


    detector_rows = run_detectors(
        region=region,
        forecaster=forecaster_name,
        stream=normalized_abs,
    )


    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    summary = {
        "region":
            region,

        "forecaster":
            forecaster_name,

        "display_name":
            display_name,

        "n_observations":
            len(residual),

        "mae":
            mae,

        "rmse":
            rmse,

        "mean_signed_error":
            mean_signed_error,

        "signed_error_std":
            residual_std,

        "calibration_abs_error_mean":
            calibration_abs_mean,

        "calibration_abs_error_std":
            calibration_abs_std,

        "signed_acf_lag1":
            signed_lookup[1],

        "signed_acf_lag2":
            signed_lookup[2],

        "signed_acf_lag6":
            signed_lookup[6],

        "signed_acf_lag12":
            signed_lookup[12],

        "signed_acf_lag24":
            signed_lookup[24],

        "signed_acf_lag48":
            signed_lookup[48],

        "signed_acf_lag96":
            signed_lookup[96],

        "signed_acf_lag336":
            signed_lookup[336],

        "absolute_acf_lag1":
            absolute_lookup[1],

        "absolute_acf_lag2":
            absolute_lookup[2],

        "absolute_acf_lag6":
            absolute_lookup[6],

        "absolute_acf_lag12":
            absolute_lookup[12],

        "absolute_acf_lag24":
            absolute_lookup[24],

        "absolute_acf_lag48":
            absolute_lookup[48],

        "absolute_acf_lag96":
            absolute_lookup[96],

        "absolute_acf_lag336":
            absolute_lookup[336],

        **sign_metrics,

        "forecaster_config":
            json.dumps(
                model_config,
                sort_keys=True,
            ),
    }


    # -------------------------------------------------------------
    # Stream rows for audit / future diagnosis
    # -------------------------------------------------------------

    stream_rows = pd.DataFrame(
        {
            "region":
                region,

            "forecaster":
                forecaster_name,

            "SETTLEMENTDATE":
                errors[
                    "SETTLEMENTDATE"
                ].values,

            "y_true":
                errors[
                    "y_true"
                ].to_numpy(
                    dtype=float
                ),

            "y_pred":
                errors[
                    "y_pred"
                ].to_numpy(
                    dtype=float
                ),

            "signed_error":
                residual,

            "absolute_error":
                absolute_error,

            "normalized_abs_error":
                normalized_abs,
        }
    )


    # -------------------------------------------------------------
    # Console
    # -------------------------------------------------------------

    print(
        f"MAE                     : "
        f"{mae:.4f}"
    )


    print(
        f"Mean signed error       : "
        f"{mean_signed_error:.4f}"
    )


    print(
        f"Signed-error lag-1 ACF  : "
        f"{signed_lookup[1]:.4f}"
    )


    print(
        f"Absolute-error lag-1 ACF: "
        f"{absolute_lookup[1]:.4f}"
    )


    print(
        f"Same consecutive sign   : "
        f"{sign_metrics['same_sign_fraction']:.4f}"
    )


    print(
        f"Mean sign-run length    : "
        f"{sign_metrics['mean_sign_run_length']:.2f}"
    )


    print(
        f"Maximum sign-run length : "
        f"{sign_metrics['max_sign_run_length']}"
    )


    print(
        "Detector counts          : "
        + ", ".join(
            f"{row['detector']}={row['detections']}"
            for row
            in detector_rows
        )
    )


    return (
        summary,
        acf_rows,
        detector_rows,
        stream_rows,
    )


# ---------------------------------------------------------------------
# Conservative interpretation
# ---------------------------------------------------------------------

def print_interpretation(
    summary,
):
    """
    Print comparison without prematurely deciding on a treatment.
    """

    print()
    print("=" * 80)
    print(
        "DIAGNOSTIC INTERPRETATION"
    )
    print("=" * 80)


    for region in REGIONS:

        region_data = (
            summary[
                summary[
                    "region"
                ] == region
            ]
            .set_index(
                "forecaster"
            )
        )


        print()
        print(
            region
        )


        for forecaster_name in FORECASTERS:

            row = region_data.loc[
                forecaster_name
            ]


            print(
                f"  {row['display_name']:<18} "
                f"signed lag1={row['signed_acf_lag1']:.4f}   "
                f"|error| lag1={row['absolute_acf_lag1']:.4f}   "
                f"same-sign={row['same_sign_fraction']:.4f}   "
                f"MAE={row['mae']:.2f}"
            )


        gb = region_data.loc[
            "gradient_boosted"
        ]


        others = region_data.loc[
            [
                "seasonal_naive",
                "sarimax",
            ]
        ]


        other_signed_max = float(
            others[
                "signed_acf_lag1"
            ].abs().max()
        )


        gb_signed = abs(
            float(
                gb[
                    "signed_acf_lag1"
                ]
            )
        )


        print()


        if (
            gb_signed >= 0.9
            and other_signed_max >= 0.9
        ):

            print(
                "  Pattern: extreme short-range residual persistence "
                "is present across multiple forecasters."
            )

            print(
                "  Implication: this is unlikely to be solely a "
                "Gradient Boosting-specific problem."
            )


        elif (
            gb_signed >= 0.9
            and other_signed_max < 0.7
        ):

            print(
                "  Pattern: Gradient Boosting residual dependence is "
                "much stronger than the alternative forecasters."
            )

            print(
                "  Implication: the next diagnosis should focus on "
                "the Gradient Boosting forecasting specification."
            )


        else:

            print(
                "  Pattern: forecaster dependence is mixed."
            )

            print(
                "  Implication: compare the complete ACF, sign-run "
                "statistics, and detector counts before selecting "
                "Diagnosis 05."
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print(
        "DIAGNOSIS 04 — FORECASTER RESIDUAL DEPENDENCE"
    )
    print("=" * 80)


    print(
        "Question:"
    )


    print(
        "Is the extreme short-range residual persistence specific "
        "to Gradient Boosting, or does it occur across the project's "
        "existing forecasting baselines?"
    )


    print()
    print(
        "Forecasters:"
    )

    for name, spec in FORECASTERS.items():

        print(
            f"  {name:<18} "
            f"{spec['display_name']}"
        )


    print()
    print(
        "No detector parameters will be changed."
    )

    print(
        "No forecaster configuration will be changed."
    )

    print(
        "No prewhitening or smoothing will be applied."
    )

    print(
        "runs.csv will not be modified."
    )


    all_summary_rows = []
    all_acf_rows = []
    all_detector_rows = []
    all_stream_frames = []


    # -----------------------------------------------------------------
    # Regions
    # -----------------------------------------------------------------

    for region in REGIONS:

        print()
        print("=" * 80)
        print(
            f"REGION: {region}"
        )
        print("=" * 80)


        (
            _,
            train,
            calibration,
            _,
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
            "Calibration period       : "
            f"{calibration['SETTLEMENTDATE'].iloc[0]}"
            " to "
            f"{calibration['SETTLEMENTDATE'].iloc[-1]}"
        )


        # -------------------------------------------------------------
        # Each existing baseline
        # -------------------------------------------------------------

        for forecaster_name in FORECASTERS:

            (
                summary_row,
                acf_rows,
                detector_rows,
                stream_rows,
            ) = diagnose_forecaster(
                region=region,
                forecaster_name=forecaster_name,
                train=train,
                calibration=calibration,
            )


            all_summary_rows.append(
                summary_row
            )


            all_acf_rows.extend(
                acf_rows
            )


            all_detector_rows.extend(
                detector_rows
            )


            all_stream_frames.append(
                stream_rows
            )


    # -----------------------------------------------------------------
    # DataFrames
    # -----------------------------------------------------------------

    summary = pd.DataFrame(
        all_summary_rows
    )


    acf = pd.DataFrame(
        all_acf_rows
    )


    detector_counts = pd.DataFrame(
        all_detector_rows
    )


    residual_streams = pd.concat(
        all_stream_frames,
        ignore_index=True,
    )


    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    summary_path = (
        OUTPUT_DIR
        / "04_forecaster_residual_summary.csv"
    )


    acf_path = (
        OUTPUT_DIR
        / "04_forecaster_residual_acf.csv"
    )


    detector_path = (
        OUTPUT_DIR
        / "04_forecaster_detector_counts.csv"
    )


    stream_path = (
        OUTPUT_DIR
        / "04_forecaster_residual_streams.csv.gz"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    acf.to_csv(
        acf_path,
        index=False,
    )


    detector_counts.to_csv(
        detector_path,
        index=False,
    )


    residual_streams.to_csv(
        stream_path,
        index=False,
        compression={
            "method": "gzip",
            "compresslevel": 6,
            "mtime": 0,
        },
    )


    # -----------------------------------------------------------------
    # Main comparison tables
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "FORECASTER RESIDUAL COMPARISON"
    )
    print("=" * 80)


    comparison_columns = [
        "region",
        "display_name",
        "mae",
        "mean_signed_error",
        "signed_acf_lag1",
        "signed_acf_lag2",
        "signed_acf_lag6",
        "signed_acf_lag12",
        "signed_acf_lag48",
        "absolute_acf_lag1",
        "same_sign_fraction",
        "mean_sign_run_length",
        "max_sign_run_length",
    ]


    print(
        summary[
            comparison_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    print()
    print("=" * 80)
    print(
        "DETECTOR COUNTS ON NORMALIZED ABSOLUTE CALIBRATION ERROR"
    )
    print("=" * 80)


    detector_pivot = (
        detector_counts
        .pivot(
            index=[
                "region",
                "forecaster",
            ],
            columns="detector",
            values="detections",
        )
        .reset_index()
    )


    print(
        detector_pivot.to_string(
            index=False
        )
    )


    # -----------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------

    print_interpretation(
        summary
    )


    # -----------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "Saved diagnostic outputs"
    )
    print("=" * 80)


    print(
        summary_path
    )


    print(
        acf_path
    )


    print(
        detector_path
    )


    print(
        stream_path
    )


    print()
    print(
        "Do not modify detectors.py yet."
    )


    print(
        "Diagnosis 05 should be selected only after determining whether "
        "the extreme residual persistence is specific to Gradient "
        "Boosting or shared across the forecasting baselines."
    )


if __name__ == "__main__":
    main()