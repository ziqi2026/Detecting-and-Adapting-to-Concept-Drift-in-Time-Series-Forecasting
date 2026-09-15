"""
02_diag_temporal_timescales.py

Diagnosis 02 — Which temporal timescale drives detector alarms?
================================================================

Background
----------
Diagnosis 01 found that:

    - The chronological calibration forecast-error stream produced alarms.
    - The exact same values, after full random shuffling, produced zero alarms.
    - Normalized absolute forecast errors showed very strong short-lag
      autocorrelation.

Therefore temporal ordering/dependence is strongly implicated.

However, Diagnosis 01 cannot tell us WHICH temporal structure is
responsible because full shuffling simultaneously destroys:

    - short-term serial dependence
    - within-day structure
    - day-to-day ordering
    - weekly ordering
    - longer-term local regimes
    - volatility/error clustering

Purpose
-------
This diagnostic separates several temporal scales using controlled
permutations of the SAME calibration normalized absolute-error values.

The detector configurations are NOT changed.

The following scenarios are compared:

1. chronological
   Original calibration stream.

2. within_day_shuffle
   Each 48-observation day keeps the same values, but their order inside
   that day is randomized.

   Preserves:
       - each day's error distribution
       - chronological order of days

   Disrupts:
       - 30-minute / hourly ordering
       - within-day serial dependence
       - precise intraday sequencing

3. day_block_shuffle
   Whole 48-observation days are randomly reordered.

   Preserves:
       - complete within-day sequences
       - short-range dependence inside each day

   Disrupts:
       - day-to-day chronological progression
       - longer local regimes across days

4. week_block_shuffle
   Whole 336-observation (7-day) blocks are randomly reordered.

   Preserves:
       - short-term structure
       - daily structure
       - most within-week structure

   Disrupts:
       - chronological progression between 7-day blocks
       - longer-than-week local regimes

   Any final remainder shorter than 336 observations is kept at the end
   rather than mixed with full blocks.

5. full_shuffle
   All observations are randomly permuted.

   Preserves:
       - exactly the same marginal values/distribution

   Destroys:
       - essentially all chronological structure

Interpretation
--------------
A strong reduction under within_day_shuffle suggests that short-range
or intraday ordering contributes to alarms.

A strong reduction under day_block_shuffle suggests that chronological
relationships between days contribute.

A strong reduction under week_block_shuffle suggests that longer-term
ordering across weeks contributes.

These controls are diagnostic, not an exact variance decomposition.
Block permutation creates artificial boundaries, so results should be
interpreted comparatively rather than as proof of one unique cause.

Important
---------
This script does NOT:

    - change detectors.py
    - retune detector parameters
    - modify DETECTOR_REGISTRY
    - change normalization
    - smooth the stream
    - prewhiten the stream
    - change T2 matching
    - write to runs.csv
    - overwrite Task 120 results

Data
----
As in Diagnosis 01, the experiment uses the calibration period only:

    2020-01-01 to 2020-02-29

The detector input is:

    frozen Gradient Boosting
        -> forecast residual
        -> absolute residual
        -> calibration normalization
        -> detector

The same Gradient Boosting configuration used by Task 120 is retained:

    feature_mode = calendar+lags
    lags = (48, 336)
    seed = 0

Outputs
-------
Saved under:

    results/tables/detection/diagnostics/

Files:

    02_temporal_timescales_summary.csv
    02_temporal_timescales_runs.csv
    02_temporal_timescales_dependence.csv

Run
---

    python experiments/detection/02_diag_temporal_timescales.py

Optional:

    python experiments/detection/02_diag_temporal_timescales.py --repeats 50
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
# Frozen project settings
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


FORECASTER_CONFIG = {
    "feature_mode": "calendar+lags",
    "lags": (48, 336),
    "seed": 0,
}


# 30-minute data
OBSERVATIONS_PER_DAY = 48
OBSERVATIONS_PER_WEEK = 48 * 7


AUTOCORRELATION_LAGS = (
    1,      # 30 minutes
    2,      # 1 hour
    48,     # 1 day
    96,     # 2 days
    336,    # 1 week
)


SCENARIO_ORDER = (
    "chronological",
    "within_day_shuffle",
    "day_block_shuffle",
    "week_block_shuffle",
    "full_shuffle",
)


DEFAULT_REPEATS = 20
DEFAULT_SEED = 23456


# ---------------------------------------------------------------------
# Build calibration detector stream
# ---------------------------------------------------------------------

def build_calibration_stream(
    train,
    calibration,
):
    """
    Reproduce the Task 120 detector representation on the calibration
    period.

    Steps
    -----
    1. Fit Gradient Boosting using training data only.
    2. Forecast the calibration period causally.
    3. Calculate forecasting residuals.
    4. Convert residuals to absolute errors.
    5. Normalize with calibration mean and standard deviation.

    Returns
    -------
    stream : np.ndarray
        Normalized absolute calibration forecast-error stream.

    calibration_mean : float

    calibration_std : float
    """

    model = GradientBoostedForecaster(
        **FORECASTER_CONFIG
    )

    model.fit(
        train
    )

    errors = frozen_error_stream(
        model,
        calibration,
        history_df=train,
    )

    absolute_error = (
        errors["error"]
        .abs()
        .to_numpy(dtype=float)
    )


    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    if len(absolute_error) == 0:

        raise RuntimeError(
            "Calibration error stream is empty."
        )


    if not np.isfinite(
        absolute_error
    ).all():

        raise RuntimeError(
            "Calibration errors contain NaN or infinity."
        )


    calibration_mean = float(
        absolute_error.mean()
    )

    # Same NumPy convention used by Task 120.
    calibration_std = float(
        absolute_error.std()
    )


    if not np.isfinite(
        calibration_mean
    ):

        raise RuntimeError(
            "Calibration mean is invalid."
        )


    if (
        not np.isfinite(calibration_std)
        or calibration_std <= 0
    ):

        raise RuntimeError(
            "Calibration standard deviation "
            "must be finite and > 0."
        )


    stream = (
        absolute_error
        - calibration_mean
    ) / calibration_std


    if not np.isfinite(
        stream
    ).all():

        raise RuntimeError(
            "Normalized stream contains NaN or infinity."
        )


    if len(stream) != len(
        calibration
    ):

        raise RuntimeError(
            "Calibration stream length mismatch."
        )


    return (
        stream,
        calibration_mean,
        calibration_std,
    )


# ---------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------

def shuffle_within_days(
    stream,
    rng,
):
    """
    Randomly reorder observations WITHIN every 48-observation day.

    Day order itself remains unchanged.

    The calibration split begins at midnight and should contain complete
    48-observation days. We validate that assumption explicitly.
    """

    values = np.asarray(
        stream,
        dtype=float,
    )


    if (
        len(values)
        % OBSERVATIONS_PER_DAY
        != 0
    ):

        raise ValueError(
            "within_day_shuffle requires complete "
            "48-observation days."
        )


    days = values.reshape(
        -1,
        OBSERVATIONS_PER_DAY,
    )


    shuffled_days = []

    for day in days:

        shuffled_days.append(
            rng.permutation(
                day
            )
        )


    result = np.concatenate(
        shuffled_days
    )


    return result


def shuffle_day_blocks(
    stream,
    rng,
):
    """
    Randomly reorder complete 48-observation day blocks.

    The ordering inside every day is unchanged.
    """

    values = np.asarray(
        stream,
        dtype=float,
    )


    if (
        len(values)
        % OBSERVATIONS_PER_DAY
        != 0
    ):

        raise ValueError(
            "day_block_shuffle requires complete "
            "48-observation days."
        )


    days = values.reshape(
        -1,
        OBSERVATIONS_PER_DAY,
    )


    order = rng.permutation(
        len(days)
    )


    result = (
        days[order]
        .reshape(-1)
    )


    return result


def shuffle_week_blocks(
    stream,
    rng,
):
    """
    Randomly reorder complete 336-observation / 7-day blocks.

    Any remainder shorter than seven days is preserved at the end.

    This avoids treating a shorter remainder block as if it were a full
    7-day block.
    """

    values = np.asarray(
        stream,
        dtype=float,
    )


    n_full_weeks = (
        len(values)
        // OBSERVATIONS_PER_WEEK
    )


    full_length = (
        n_full_weeks
        * OBSERVATIONS_PER_WEEK
    )


    full_part = values[
        :full_length
    ]


    remainder = values[
        full_length:
    ]


    if n_full_weeks < 2:

        raise ValueError(
            "Not enough complete 7-day blocks "
            "for week-block shuffling."
        )


    weeks = full_part.reshape(
        n_full_weeks,
        OBSERVATIONS_PER_WEEK,
    )


    order = rng.permutation(
        n_full_weeks
    )


    shuffled_full = (
        weeks[order]
        .reshape(-1)
    )


    result = np.concatenate(
        [
            shuffled_full,
            remainder,
        ]
    )


    return result


def full_shuffle(
    stream,
    rng,
):
    """
    Randomly permute every observation.
    """

    return rng.permutation(
        np.asarray(
            stream,
            dtype=float,
        )
    )


def build_transformed_streams(
    stream,
    rng,
):
    """
    Construct one realization of every permutation control.

    IMPORTANT:
    These transformed streams are created ONCE per repeat and are then
    given to every detector.

    Therefore ADWIN, Page-Hinkley, and KSWIN see exactly the same
    transformed stream in a given repeat.
    """

    return {
        "within_day_shuffle":
            shuffle_within_days(
                stream,
                rng,
            ),

        "day_block_shuffle":
            shuffle_day_blocks(
                stream,
                rng,
            ),

        "week_block_shuffle":
            shuffle_week_blocks(
                stream,
                rng,
            ),

        "full_shuffle":
            full_shuffle(
                stream,
                rng,
            ),
    }


# ---------------------------------------------------------------------
# Detector runner
# ---------------------------------------------------------------------

def run_detector(
    detector_name,
    stream,
):
    """
    Run one detector using the existing frozen DETECTOR_REGISTRY.
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


    return (
        changepoints,
        config,
    )


# ---------------------------------------------------------------------
# Dependence measurements
# ---------------------------------------------------------------------

def autocorrelation(
    stream,
    lag,
):
    """
    Calculate simple sample autocorrelation.
    """

    if lag >= len(stream):

        return float(
            "nan"
        )


    value = pd.Series(
        stream,
        dtype=float,
    ).autocorr(
        lag=lag
    )


    return float(
        value
    )


def dependence_rows(
    region,
    scenario,
    repeat,
    stream,
):
    """
    Record autocorrelation after each transformation.

    This lets us verify what temporal dependence the control actually
    removed rather than assuming that the transformation behaved as
    intended.
    """

    rows = []


    for lag in AUTOCORRELATION_LAGS:

        rows.append(
            {
                "region": region,
                "scenario": scenario,
                "repeat": repeat,
                "lag_observations": lag,
                "lag_hours": lag / 2,
                "autocorrelation":
                    autocorrelation(
                        stream,
                        lag,
                    ),
            }
        )


    return rows


# ---------------------------------------------------------------------
# One-region experiment
# ---------------------------------------------------------------------

def diagnose_region(
    region,
    repeats,
    seed,
):
    """
    Run all temporal-timescale controls for one region.
    """

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


    (
        stream,
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


    # -------------------------------------------------------------
    # Independent reproducible RNG per region
    # -------------------------------------------------------------

    region_offset = (
        0
        if region == "SA1"
        else 1_000_000
    )


    rng = np.random.default_rng(
        seed
        + region_offset
    )


    run_rows = []
    dep_rows = []


    # -------------------------------------------------------------
    # Original chronological stream
    # -------------------------------------------------------------

    dep_rows.extend(
        dependence_rows(
            region=region,
            scenario="chronological",
            repeat=0,
            stream=stream,
        )
    )


    original_counts = {}


    for detector_name in DETECTOR_REGISTRY:

        (
            changepoints,
            config,
        ) = run_detector(
            detector_name,
            stream,
        )


        count = len(
            changepoints
        )


        original_counts[
            detector_name
        ] = count


        run_rows.append(
            {
                "region": region,
                "detector": detector_name,
                "scenario": "chronological",
                "repeat": 0,
                "detections": count,
                "n_observations": len(stream),
                "detections_per_10000":
                    count
                    / len(stream)
                    * 10000,
                "detector_config":
                    json.dumps(
                        config,
                        sort_keys=True,
                    ),
            }
        )


    # -------------------------------------------------------------
    # Controlled temporal transformations
    # -------------------------------------------------------------

    for repeat in range(
        1,
        repeats + 1,
    ):

        transformed = (
            build_transformed_streams(
                stream,
                rng,
            )
        )


        # Verify every transformation preserves the exact values.
        original_sorted = np.sort(
            stream
        )


        for (
            scenario,
            transformed_stream,
        ) in transformed.items():

            if len(
                transformed_stream
            ) != len(stream):

                raise RuntimeError(
                    f"{scenario}: stream length changed."
                )


            if not np.array_equal(
                np.sort(transformed_stream),
                original_sorted,
            ):

                raise RuntimeError(
                    f"{scenario}: transformation changed "
                    "the stream values."
                )


            dep_rows.extend(
                dependence_rows(
                    region=region,
                    scenario=scenario,
                    repeat=repeat,
                    stream=transformed_stream,
                )
            )


            # Every detector receives the SAME transformed stream
            # for this scenario/repeat.
            for detector_name in DETECTOR_REGISTRY:

                (
                    changepoints,
                    config,
                ) = run_detector(
                    detector_name,
                    transformed_stream,
                )


                count = len(
                    changepoints
                )


                run_rows.append(
                    {
                        "region": region,
                        "detector": detector_name,
                        "scenario": scenario,
                        "repeat": repeat,
                        "detections": count,
                        "n_observations":
                            len(transformed_stream),
                        "detections_per_10000":
                            count
                            / len(transformed_stream)
                            * 10000,
                        "detector_config":
                            json.dumps(
                                config,
                                sort_keys=True,
                            ),
                    }
                )


    return (
        run_rows,
        dep_rows,
        original_counts,
    )


# ---------------------------------------------------------------------
# Summarize detection results
# ---------------------------------------------------------------------

def summarize_runs(
    runs,
):
    """
    Produce detector/scenario mean, SD, range and reduction relative
    to the chronological stream.
    """

    rows = []


    for (
        region,
        detector,
    ), detector_data in runs.groupby(
        [
            "region",
            "detector",
        ],
        sort=False,
    ):

        original_rows = detector_data[
            detector_data[
                "scenario"
            ] == "chronological"
        ]


        if len(original_rows) != 1:

            raise RuntimeError(
                f"Expected one chronological run for "
                f"{region}/{detector}."
            )


        original_count = int(
            original_rows[
                "detections"
            ].iloc[0]
        )


        for scenario in SCENARIO_ORDER:

            values = (
                detector_data.loc[
                    detector_data[
                        "scenario"
                    ] == scenario,
                    "detections",
                ]
                .astype(float)
            )


            if values.empty:

                raise RuntimeError(
                    f"Missing scenario {scenario} "
                    f"for {region}/{detector}."
                )


            mean = float(
                values.mean()
            )

            sd = float(
                values.std(
                    ddof=1
                )
                if len(values) > 1
                else 0.0
            )

            minimum = int(
                values.min()
            )

            maximum = int(
                values.max()
            )


            if original_count > 0:

                ratio_to_original = (
                    mean
                    / original_count
                )

                reduction_vs_original = (
                    1.0
                    - ratio_to_original
                )

            else:

                ratio_to_original = float(
                    "nan"
                )

                reduction_vs_original = float(
                    "nan"
                )


            rows.append(
                {
                    "region": region,
                    "detector": detector,
                    "scenario": scenario,

                    "chronological_detections":
                        original_count,

                    "mean_detections":
                        mean,

                    "sd_detections":
                        sd,

                    "min_detections":
                        minimum,

                    "max_detections":
                        maximum,

                    "ratio_to_chronological":
                        ratio_to_original,

                    "reduction_vs_chronological":
                        reduction_vs_original,
                }
            )


    summary = pd.DataFrame(
        rows
    )


    scenario_rank = {
        name: i
        for i, name
        in enumerate(
            SCENARIO_ORDER
        )
    }


    summary[
        "_scenario_order"
    ] = summary[
        "scenario"
    ].map(
        scenario_rank
    )


    summary = (
        summary
        .sort_values(
            [
                "region",
                "detector",
                "_scenario_order",
            ]
        )
        .drop(
            columns=[
                "_scenario_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    return summary


# ---------------------------------------------------------------------
# Summarize transformed autocorrelation
# ---------------------------------------------------------------------

def summarize_dependence(
    dependence,
):
    """
    Aggregate autocorrelation across repeated transformations.
    """

    summary = (
        dependence
        .groupby(
            [
                "region",
                "scenario",
                "lag_observations",
                "lag_hours",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            autocorrelation_mean=(
                "autocorrelation",
                "mean",
            ),
            autocorrelation_sd=(
                "autocorrelation",
                "std",
            ),
            autocorrelation_min=(
                "autocorrelation",
                "min",
            ),
            autocorrelation_max=(
                "autocorrelation",
                "max",
            ),
        )
    )


    # SD is undefined for the single original chronological run.
    summary[
        "autocorrelation_sd"
    ] = (
        summary[
            "autocorrelation_sd"
        ]
        .fillna(0.0)
    )


    scenario_rank = {
        name: i
        for i, name
        in enumerate(
            SCENARIO_ORDER
        )
    }


    summary[
        "_scenario_order"
    ] = summary[
        "scenario"
    ].map(
        scenario_rank
    )


    summary = (
        summary
        .sort_values(
            [
                "region",
                "_scenario_order",
                "lag_observations",
            ]
        )
        .drop(
            columns=[
                "_scenario_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    return summary


# ---------------------------------------------------------------------
# Console interpretation
# ---------------------------------------------------------------------

def print_detector_interpretation(
    summary,
):
    """
    Print descriptive comparisons without claiming a unique causal
    mechanism.
    """

    print()
    print("=" * 80)
    print(
        "DETECTION COMPARISON"
    )
    print("=" * 80)


    for (
        region,
        detector,
    ), group in summary.groupby(
        [
            "region",
            "detector",
        ],
        sort=False,
    ):

        print()
        print(
            f"{region} / {detector}"
        )


        original = int(
            group[
                "chronological_detections"
            ].iloc[0]
        )


        print(
            f"  chronological      : "
            f"{original}"
        )


        for scenario in SCENARIO_ORDER[1:]:

            row = group[
                group[
                    "scenario"
                ] == scenario
            ].iloc[0]


            print(
                f"  {scenario:<19}: "
                f"{row['mean_detections']:.2f} "
                f"± {row['sd_detections']:.2f} "
                f"[{int(row['min_detections'])}"
                f"-{int(row['max_detections'])}]"
            )


    print()
    print("=" * 80)
    print(
        "WHAT EACH REDUCTION WOULD SUGGEST"
    )
    print("=" * 80)

    print(
        """
within_day_shuffle drops strongly:
    Short-range / intraday ordering contributes to alarms.

day_block_shuffle drops strongly:
    Day-to-day chronological ordering contributes to alarms.

week_block_shuffle drops strongly:
    Longer-than-week chronological progression contributes to alarms.

full_shuffle drops strongly:
    Confirms the broad temporal-ordering result from Diagnosis 01.

Important:
These effects are not mutually exclusive. More than one temporal
timescale can contribute to the detector behaviour.
"""
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Diagnosis 02: isolate temporal timescales "
            "associated with AEMO detector alarms."
        )
    )


    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=(
            "Number of random transformations per control "
            f"(default={DEFAULT_REPEATS})."
        ),
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Seed used for reproducible temporal permutations "
            f"(default={DEFAULT_SEED})."
        ),
    )


    args = parser.parse_args()


    if args.repeats < 1:

        raise ValueError(
            "--repeats must be at least 1."
        )


    print("=" * 80)
    print(
        "DIAGNOSIS 02 — TEMPORAL TIMESCALES"
    )
    print("=" * 80)

    print(
        "Question:"
    )

    print(
        "Which temporal scale in the calibration normalized absolute "
        "forecast-error stream contributes to detector alarms?"
    )

    print()
    print(
        "No detector parameters will be changed."
    )

    print(
        "No smoothing or prewhitening will be applied."
    )

    print(
        "runs.csv will not be modified."
    )

    print()
    print(
        f"Regions  : {REGIONS}"
    )

    print(
        f"Repeats  : {args.repeats}"
    )

    print(
        f"Seed     : {args.seed}"
    )

    print(
        "Forecaster: frozen Gradient Boosting, "
        "calendar+lags (48, 336)"
    )


    all_run_rows = []
    all_dependence_rows = []


    for region in REGIONS:

        (
            run_rows,
            dependence_rows_region,
            _,
        ) = diagnose_region(
            region=region,
            repeats=args.repeats,
            seed=args.seed,
        )


        all_run_rows.extend(
            run_rows
        )


        all_dependence_rows.extend(
            dependence_rows_region
        )


    # -----------------------------------------------------------------
    # DataFrames
    # -----------------------------------------------------------------

    runs = pd.DataFrame(
        all_run_rows
    )


    dependence = pd.DataFrame(
        all_dependence_rows
    )


    summary = summarize_runs(
        runs
    )


    dependence_summary = (
        summarize_dependence(
            dependence
        )
    )


    # -----------------------------------------------------------------
    # Save diagnostics
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    summary_path = (
        OUTPUT_DIR
        / "02_temporal_timescales_summary.csv"
    )


    runs_path = (
        OUTPUT_DIR
        / "02_temporal_timescales_runs.csv"
    )


    dependence_path = (
        OUTPUT_DIR
        / "02_temporal_timescales_dependence.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    runs.to_csv(
        runs_path,
        index=False,
    )


    dependence_summary.to_csv(
        dependence_path,
        index=False,
    )


    # -----------------------------------------------------------------
    # Console result
    # -----------------------------------------------------------------

    print_detector_interpretation(
        summary
    )


    print()
    print("=" * 80)
    print(
        "AUTOCORRELATION AFTER EACH CONTROL"
    )
    print("=" * 80)


    display_dependence = (
        dependence_summary[
            dependence_summary[
                "lag_observations"
            ].isin(
                [
                    1,
                    48,
                    336,
                ]
            )
        ][
            [
                "region",
                "scenario",
                "lag_observations",
                "autocorrelation_mean",
                "autocorrelation_sd",
            ]
        ]
    )


    print(
        display_dependence.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


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
        runs_path
    )

    print(
        dependence_path
    )


    print()
    print(
        "Do not modify detectors.py yet."
    )

    print(
        "Use this result to determine which temporal structure should "
        "be examined in Diagnosis 03."
    )


if __name__ == "__main__":
    main()