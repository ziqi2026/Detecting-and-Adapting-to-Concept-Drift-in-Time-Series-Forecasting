"""
01_diag_temporal_structure.py

Diagnosis 01 — Temporal structure / shuffle control
====================================================

Problem
-------
The real AEMO detection results show very different behaviour:

    - ADWIN produces many detections.
    - KSWIN produces even more detections.
    - Page-Hinkley is much more selective.

Before changing detector thresholds, applying smoothing, using
prewhitening, or modifying the T2 matching protocol, we first need to
identify whether the excessive alarms are caused by temporal structure
in the forecast-error stream.

This script tests:

    Does keeping the chronological ordering of the forecast-error
    stream cause substantially more detector alarms than randomly
    shuffling the exact same values?

Why this is useful
------------------
Task 120 uses:

    Frozen Gradient Boosting
        -> forecasting residual
        -> absolute residual
        -> calibration normalization
        -> drift detector

The calibration period (Jan-Feb 2020) provides the reference
distribution used to standardize the real test stream.

If a detector already produces many alarms on the chronological
calibration stream, but very few after shuffling the same values,
then temporal ordering is important.

Shuffling preserves:

    - the same values
    - the same mean
    - the same standard deviation
    - the same marginal distribution
    - the same outliers

but destroys:

    - serial dependence
    - daily ordering
    - weekly ordering
    - local volatility clustering
    - other temporal structure

Therefore:

    many chronological alarms + few shuffled alarms
        -> temporal structure is a strong suspect

    few chronological alarms + few shuffled alarms
        -> calibration is stable; investigate changes after March 2020

    many chronological alarms + many shuffled alarms
        -> temporal structure alone does not explain the issue;
           investigate distribution/outliers/detector behaviour

Important
---------
This is a diagnostic experiment only.

It does NOT:

    - change detectors.py
    - retune any detector
    - modify DETECTOR_REGISTRY
    - change T2 event matching
    - modify runs.csv
    - modify Task 120 outputs
    - implement prewhitening
    - smooth or aggregate the stream

Run from repository root
------------------------

    python experiments/detection/01_diag_temporal_structure.py

Optional:

    python experiments/detection/01_diag_temporal_structure.py --shuffles 50

Outputs
-------
Saved under:

    results/tables/detection/diagnostics/

Files:

    01_temporal_structure_summary.csv
    01_temporal_structure_shuffle_runs.csv
    01_temporal_structure_autocorrelation.csv
    01_temporal_structure_changepoints.csv
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from drift_lab.data_loader import load_and_split_aemo
from drift_lab.detectors import DETECTOR_REGISTRY
from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)


# ---------------------------------------------------------------------
# Frozen project protocol
# ---------------------------------------------------------------------

REGIONS = ("SA1", "NSW1")

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


# Same forecasting configuration used by Task 120.
FORECASTER_CONFIG = {
    "feature_mode": "calendar+lags",
    "lags": (48, 336),
    "seed": 0,
}


# Diagnostic autocorrelation lags for 30-minute observations.
#
# 1   = 30 minutes
# 2   = 1 hour
# 48  = 1 day
# 96  = 2 days
# 336 = 1 week
AUTOCORRELATION_LAGS = (
    1,
    2,
    48,
    96,
    336,
)


DEFAULT_SHUFFLES = 20
DEFAULT_SHUFFLE_SEED = 12345


# ---------------------------------------------------------------------
# Build the reference detector stream
# ---------------------------------------------------------------------

def build_calibration_stream(
    train,
    calibration,
):
    """
    Construct the normalized absolute forecasting-error stream from
    the calibration period.

    This reproduces the representation used in Task 120:

        fit GB on train
        -> forecast calibration
        -> residual
        -> absolute residual
        -> normalize using calibration mean/std

    Returns
    -------
    stream : np.ndarray
        Normalized calibration absolute-error stream.

    calibration_errors : pd.DataFrame
        Timestamp, y_true, y_pred, and signed error.

    calibration_mean : float
        Mean absolute calibration forecasting error.

    calibration_std : float
        Standard deviation of absolute calibration forecasting error.
    """

    model = GradientBoostedForecaster(
        **FORECASTER_CONFIG
    )

    model.fit(train)

    calibration_errors = frozen_error_stream(
        model,
        calibration,
        history_df=train,
    )

    absolute_error = (
        calibration_errors["error"]
        .abs()
        .to_numpy(dtype=float)
    )

    if len(absolute_error) == 0:
        raise RuntimeError(
            "Calibration error stream is empty."
        )

    if not np.isfinite(absolute_error).all():
        raise RuntimeError(
            "Calibration error stream contains NaN or infinity."
        )

    calibration_mean = float(
        absolute_error.mean()
    )

    # Keep the same NumPy std convention used by Task 120.
    calibration_std = float(
        absolute_error.std()
    )

    if not np.isfinite(calibration_mean):
        raise RuntimeError(
            "Calibration absolute-error mean is invalid."
        )

    if (
        not np.isfinite(calibration_std)
        or calibration_std <= 0
    ):
        raise RuntimeError(
            "Calibration absolute-error std must be > 0."
        )

    stream = (
        absolute_error - calibration_mean
    ) / calibration_std

    if not np.isfinite(stream).all():
        raise RuntimeError(
            "Normalized stream contains NaN or infinity."
        )

    if len(stream) != len(calibration):
        raise RuntimeError(
            "Calibration stream length mismatch."
        )

    return (
        stream,
        calibration_errors,
        calibration_mean,
        calibration_std,
    )


# ---------------------------------------------------------------------
# Detector runner
# ---------------------------------------------------------------------

def run_detector(
    detector_name,
    stream,
):
    """
    Run one detector using its existing frozen registry configuration.

    A new detector instance is created by the registered function for
    every run, preventing state from leaking between experiments.
    """

    entry = DETECTOR_REGISTRY[
        detector_name
    ]

    config = entry[
        "config"
    ].copy()

    changepoints = entry[
        "func"
    ](
        stream,
        **config,
    )

    return changepoints, config


# ---------------------------------------------------------------------
# Autocorrelation diagnostic
# ---------------------------------------------------------------------

def calculate_autocorrelation(
    region,
    stream,
):
    """
    Calculate descriptive autocorrelation values.

    These values help identify whether dependence exists at short,
    daily, or weekly lags.

    They are diagnostic evidence only and do not prove causation.
    """

    series = pd.Series(
        stream,
        dtype=float,
    )

    rows = []

    for lag in AUTOCORRELATION_LAGS:

        if lag >= len(series):
            correlation = float("nan")

        else:
            correlation = float(
                series.autocorr(
                    lag=lag
                )
            )

        rows.append(
            {
                "region": region,
                "lag_observations": lag,
                "lag_hours": lag / 2,
                "autocorrelation": correlation,
            }
        )

    return rows


# ---------------------------------------------------------------------
# One-region diagnosis
# ---------------------------------------------------------------------

def diagnose_region(
    region,
    n_shuffles,
    rng,
):
    """
    Diagnose temporal-order sensitivity for one AEMO region.
    """

    print()
    print("=" * 75)
    print(
        f"REGION: {region}"
    )
    print("=" * 75)

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
        f"Training observations    : {len(train):,}"
    )

    print(
        f"Calibration observations : {len(calibration):,}"
    )

    print(
        "Calibration period       : "
        f"{calibration['SETTLEMENTDATE'].iloc[0]} "
        "to "
        f"{calibration['SETTLEMENTDATE'].iloc[-1]}"
    )

    (
        stream,
        calibration_errors,
        calibration_mean,
        calibration_std,
    ) = build_calibration_stream(
        train,
        calibration,
    )

    print()
    print(
        f"Calibration abs-error mean : "
        f"{calibration_mean:.4f}"
    )

    print(
        f"Calibration abs-error std  : "
        f"{calibration_std:.4f}"
    )

    print(
        f"Normalized stream mean     : "
        f"{stream.mean():.6f}"
    )

    print(
        f"Normalized stream std      : "
        f"{stream.std():.6f}"
    )


    # -----------------------------------------------------------------
    # Autocorrelation
    # -----------------------------------------------------------------

    autocorrelation_rows = (
        calculate_autocorrelation(
            region,
            stream,
        )
    )

    print()
    print(
        "Autocorrelation:"
    )

    for row in autocorrelation_rows:

        value = row[
            "autocorrelation"
        ]

        if np.isnan(value):
            shown = "N/A"
        else:
            shown = f"{value:.4f}"

        print(
            f"  lag {row['lag_observations']:>3} "
            f"({row['lag_hours']:>6.1f} h): "
            f"{shown}"
        )


    # -----------------------------------------------------------------
    # Original vs shuffled detector behaviour
    # -----------------------------------------------------------------

    summary_rows = []
    shuffle_rows = []
    changepoint_rows = []

    print()
    print(
        f"Running original stream and "
        f"{n_shuffles} shuffled controls..."
    )


    for detector_name in DETECTOR_REGISTRY:

        # -------------------------------------------------------------
        # Original chronological stream
        # -------------------------------------------------------------

        (
            chronological_cps,
            config,
        ) = run_detector(
            detector_name,
            stream,
        )

        chronological_count = len(
            chronological_cps
        )


        # Save chronological detection timestamps for later inspection.
        for cp in chronological_cps:

            if (
                cp < 0
                or cp >= len(calibration_errors)
            ):
                raise RuntimeError(
                    f"{detector_name} returned "
                    f"invalid changepoint {cp}."
                )

            error_row = (
                calibration_errors
                .iloc[cp]
            )

            changepoint_rows.append(
                {
                    "region": region,
                    "detector": detector_name,
                    "detected_index": int(cp),
                    "SETTLEMENTDATE": (
                        error_row["SETTLEMENTDATE"]
                    ),
                    "forecast_error": float(
                        error_row["error"]
                    ),
                    "absolute_forecast_error": float(
                        abs(
                            error_row["error"]
                        )
                    ),
                    "normalized_abs_error": float(
                        stream[cp]
                    ),
                    "detector_config": json.dumps(
                        config,
                        sort_keys=True,
                    ),
                }
            )


        shuffle_rows.append(
            {
                "region": region,
                "detector": detector_name,
                "scenario": "chronological",
                "shuffle_id": 0,
                "detections": chronological_count,
            }
        )


        # -------------------------------------------------------------
        # Shuffle controls
        # -------------------------------------------------------------

        shuffled_counts = []

        for shuffle_id in range(
            1,
            n_shuffles + 1,
        ):

            # Same values, different temporal order.
            shuffled_stream = (
                rng.permutation(
                    stream
                )
            )

            (
                shuffled_cps,
                shuffled_config,
            ) = run_detector(
                detector_name,
                shuffled_stream,
            )

            if shuffled_config != config:
                raise RuntimeError(
                    "Detector configuration changed "
                    "during diagnostic."
                )

            count = len(
                shuffled_cps
            )

            shuffled_counts.append(
                count
            )

            shuffle_rows.append(
                {
                    "region": region,
                    "detector": detector_name,
                    "scenario": "shuffled",
                    "shuffle_id": shuffle_id,
                    "detections": count,
                }
            )


        shuffled_counts = np.asarray(
            shuffled_counts,
            dtype=float,
        )

        shuffle_mean = float(
            shuffled_counts.mean()
        )

        shuffle_sd = float(
            shuffled_counts.std(
                ddof=1
            )
            if len(shuffled_counts) > 1
            else 0.0
        )

        shuffle_min = int(
            shuffled_counts.min()
        )

        shuffle_max = int(
            shuffled_counts.max()
        )


        if chronological_count > 0:

            reduction = (
                chronological_count
                - shuffle_mean
            ) / chronological_count

        else:
            reduction = float(
                "nan"
            )


        summary_rows.append(
            {
                "region": region,
                "detector": detector_name,

                "n_observations": len(stream),

                "chronological_detections":
                    chronological_count,

                "shuffle_mean_detections":
                    shuffle_mean,

                "shuffle_sd_detections":
                    shuffle_sd,

                "shuffle_min_detections":
                    shuffle_min,

                "shuffle_max_detections":
                    shuffle_max,

                "reduction_after_shuffle":
                    reduction,

                "calibration_abs_error_mean":
                    calibration_mean,

                "calibration_abs_error_std":
                    calibration_std,

                "detector_config":
                    json.dumps(
                        config,
                        sort_keys=True,
                    ),
            }
        )


        print()
        print(
            f"{detector_name}"
        )

        print(
            f"  chronological detections : "
            f"{chronological_count}"
        )

        print(
            f"  shuffled mean             : "
            f"{shuffle_mean:.2f} ± {shuffle_sd:.2f}"
        )

        print(
            f"  shuffled range            : "
            f"{shuffle_min} - {shuffle_max}"
        )

        if np.isfinite(reduction):

            print(
                f"  reduction after shuffle   : "
                f"{reduction:.2%}"
            )

        else:

            print(
                "  reduction after shuffle   : N/A"
            )


    return (
        summary_rows,
        shuffle_rows,
        autocorrelation_rows,
        changepoint_rows,
    )


# ---------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------

def print_interpretation(summary):
    """
    Print a simple descriptive interpretation.

    This deliberately avoids declaring a statistical causal conclusion.
    """

    print()
    print("=" * 75)
    print("HOW TO INTERPRET THIS DIAGNOSIS")
    print("=" * 75)

    for row in summary.to_dict(
        "records"
    ):

        detector = row[
            "detector"
        ]

        region = row[
            "region"
        ]

        original = row[
            "chronological_detections"
        ]

        shuffled = row[
            "shuffle_mean_detections"
        ]

        print()
        print(
            f"{region} / {detector}:"
        )

        if original == 0:

            print(
                "  No alarms occurred on the chronological "
                "calibration stream."
            )

            print(
                "  Temporal dependence is therefore unlikely "
                "to explain excessive calibration alarms."
            )

        elif shuffled <= original * 0.25:

            print(
                "  Alarm count dropped strongly after shuffling."
            )

            print(
                "  Temporal ordering/dependence is a strong "
                "candidate explanation."
            )

        elif shuffled <= original * 0.75:

            print(
                "  Alarm count dropped moderately after shuffling."
            )

            print(
                "  Temporal ordering probably contributes, but "
                "does not fully explain the detector behaviour."
            )

        else:

            print(
                "  Shuffling did not substantially reduce alarms."
            )

            print(
                "  Temporal ordering alone is unlikely to explain "
                "the detector behaviour."
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Diagnosis 01: test whether temporal ordering "
            "contributes to excessive AEMO drift detections."
        )
    )

    parser.add_argument(
        "--shuffles",
        type=int,
        default=DEFAULT_SHUFFLES,
        help=(
            "Number of shuffled controls per detector "
            f"(default={DEFAULT_SHUFFLES})."
        ),
    )

    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=(
            "Random seed used for reproducible shuffling "
            f"(default={DEFAULT_SHUFFLE_SEED})."
        ),
    )

    args = parser.parse_args()

    if args.shuffles < 1:
        raise ValueError(
            "--shuffles must be at least 1."
        )


    print("=" * 75)
    print(
        "DIAGNOSIS 01 — TEMPORAL STRUCTURE / SHUFFLE CONTROL"
    )
    print("=" * 75)

    print(
        "Question:"
    )

    print(
        "Are ADWIN, Page-Hinkley, or KSWIN reacting to temporal "
        "ordering in the normalized absolute forecasting errors?"
    )

    print()
    print(
        "No detector parameters will be changed."
    )

    print(
        "No T2 event matching will be performed."
    )

    print(
        "runs.csv will not be modified."
    )

    print()
    print(
        f"Regions       : {REGIONS}"
    )

    print(
        f"Shuffles      : {args.shuffles}"
    )

    print(
        f"Shuffle seed  : {args.shuffle_seed}"
    )

    print(
        "Forecaster    : Gradient Boosting "
        "calendar+lags (48, 336)"
    )


    rng = np.random.default_rng(
        args.shuffle_seed
    )


    all_summary = []
    all_shuffle_runs = []
    all_autocorrelation = []
    all_changepoints = []


    for region in REGIONS:

        (
            summary_rows,
            shuffle_rows,
            autocorrelation_rows,
            changepoint_rows,
        ) = diagnose_region(
            region=region,
            n_shuffles=args.shuffles,
            rng=rng,
        )

        all_summary.extend(
            summary_rows
        )

        all_shuffle_runs.extend(
            shuffle_rows
        )

        all_autocorrelation.extend(
            autocorrelation_rows
        )

        all_changepoints.extend(
            changepoint_rows
        )


    # -----------------------------------------------------------------
    # Convert to DataFrames
    # -----------------------------------------------------------------

    summary = pd.DataFrame(
        all_summary
    )

    shuffle_runs = pd.DataFrame(
        all_shuffle_runs
    )

    autocorrelation = pd.DataFrame(
        all_autocorrelation
    )

    changepoints = pd.DataFrame(
        all_changepoints
    )


    # -----------------------------------------------------------------
    # Save diagnostics separately from project result tables
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    summary_path = (
        OUTPUT_DIR
        / "01_temporal_structure_summary.csv"
    )

    shuffle_path = (
        OUTPUT_DIR
        / "01_temporal_structure_shuffle_runs.csv"
    )

    autocorrelation_path = (
        OUTPUT_DIR
        / "01_temporal_structure_autocorrelation.csv"
    )

    changepoint_path = (
        OUTPUT_DIR
        / "01_temporal_structure_changepoints.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )

    shuffle_runs.to_csv(
        shuffle_path,
        index=False,
    )

    autocorrelation.to_csv(
        autocorrelation_path,
        index=False,
    )

    if changepoints.empty:

        changepoints = pd.DataFrame(
            columns=[
                "region",
                "detector",
                "detected_index",
                "SETTLEMENTDATE",
                "forecast_error",
                "absolute_forecast_error",
                "normalized_abs_error",
                "detector_config",
            ]
        )

    changepoints.to_csv(
        changepoint_path,
        index=False,
    )


    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "DIAGNOSIS 01 SUMMARY"
    )
    print("=" * 75)

    display_columns = [
        "region",
        "detector",
        "chronological_detections",
        "shuffle_mean_detections",
        "shuffle_sd_detections",
        "shuffle_min_detections",
        "shuffle_max_detections",
        "reduction_after_shuffle",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    print_interpretation(
        summary
    )


    print()
    print("=" * 75)
    print(
        "Saved diagnostic outputs"
    )
    print("=" * 75)

    print(
        summary_path
    )

    print(
        shuffle_path
    )

    print(
        autocorrelation_path
    )

    print(
        changepoint_path
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Do not change detector thresholds based on this script alone."
    )

    print(
        "If chronological alarms collapse after shuffling, Diagnosis 02 "
        "should identify which temporal structure is responsible "
        "(short-lag, daily lag 48, weekly lag 336, or volatility clustering)."
    )


if __name__ == "__main__":
    main()


