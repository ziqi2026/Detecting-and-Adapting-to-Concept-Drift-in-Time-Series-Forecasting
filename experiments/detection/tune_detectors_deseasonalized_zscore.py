"""
Synthetic detector tuning for ADWIN, KSWIN, and Page-Hinkley.

The script evaluates three configurations for each detector on
synthetic none, sudden, gradual, and recurring drift streams
across seeds 1-5.

Synthetic demand is generated at 30-minute cadence and split into
2 years of TRAIN, 60 days of CALIBRATION, and 4 years of TEST.

A seasonal profile is fitted on complete TRAIN using:
    overall mean
    + smoothed day-of-year effect
    + day-of-week x half-hour effect

The frozen TRAIN seasonal profile is applied to TRAIN,
CALIBRATION, and TEST.

A temporary Gradient Boosting model is fitted on deseasonalised
TRAIN Year 1 and used to generate chronological TRAIN Year-2
forecast errors. A global absolute-error z-score is fitted on
these Year-2 errors.

A final Gradient Boosting model using calendar features and
lags (48, 336) is fitted on complete deseasonalised TRAIN.

Processed CALIBRATION errors warm each detector. Detections are
recorded on processed TEST errors using a 14-day refractory period.

Each configuration is evaluated using NONE detections, false alarms
per year, missed detections, and detection delay.

A configuration passes when NONE has zero TEST detections across
all five seeds and mean false alarms per year is at most 2 for
sudden, gradual, and recurring drift.

Among passing configurations, selection uses:
    fewest missed detections
    lowest worst drift-specific mean false alarms per year
    lowest overall mean false alarms per year
    lowest mean detection delay

Results are saved to:
results/tables/detection/
detector_tuning_results.csv
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

from river.drift import (
    ADWIN,
    KSWIN,
    PageHinkley,
)


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


# Project imports

from drift_lab.evaluation import (
    evaluate_detections,
)

from drift_lab.synthetic_generator import (
    make_feature_series,
)

from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)

from drift_lab.deseasonalise import (
    deseasonalise_demand_frame,
    ErrorZScorePreprocessor,
)

from drift_lab.detectors import (
    run_detector_with_protocol,
)


# Synthetic experiment

SEEDS = (
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

POSITIVE_DRIFT_TYPES = (
    "sudden",
    "gradual",
    "recurring",
)


# Time structure

OBSERVATIONS_PER_DAY = 48

OBSERVATIONS_PER_YEAR = (
    365
    * OBSERVATIONS_PER_DAY
)


# Chronological split

TRAIN_LENGTH = (
    2
    * OBSERVATIONS_PER_YEAR
)
# 35,040

CALIBRATION_LENGTH = (
    60
    * OBSERVATIONS_PER_DAY
)
# 2,880

TEST_LENGTH = (
    4
    * OBSERVATIONS_PER_YEAR
)
# 70,080

TRAIN_END = TRAIN_LENGTH

CALIBRATION_END = (
    TRAIN_END
    + CALIBRATION_LENGTH
)

TEST_START = CALIBRATION_END

N_OBSERVATIONS = (
    TRAIN_LENGTH
    + CALIBRATION_LENGTH
    + TEST_LENGTH
)
# 108,000


# Synthetic changepoints

SYNTHETIC_CHANGEPOINTS = {

    "none":
        None,

    "sudden": [
        TEST_START
        + int(
            0.50
            * TEST_LENGTH
        )
    ],

    "gradual": [
        TEST_START
        + int(
            0.45
            * TEST_LENGTH
        ),

        TEST_START
        + int(
            0.55
            * TEST_LENGTH
        ),
    ],

    "recurring": [
        TEST_START
        + int(
            0.35
            * TEST_LENGTH
        ),

        TEST_START
        + int(
            0.65
            * TEST_LENGTH
        ),
    ],
}


# Forecasting

FORECASTER_SEED = 0

FORECASTER_LAGS = (
    48,
    336,
)


# TRAIN Year 1 is used to fit the temporary model.
# TRAIN Year 2 provides the error-standardisation reference.

TRAIN_REFERENCE_FIT_END = (
    OBSERVATIONS_PER_YEAR
)


# Detection protocol

TOLERANCE_DAYS = 7

REFRACTORY_DAYS = 14

TOLERANCE = (
    TOLERANCE_DAYS
    * OBSERVATIONS_PER_DAY
)

REFRACTORY_PERIOD = (
    REFRACTORY_DAYS
    * OBSERVATIONS_PER_DAY
)

MAX_FALSE_ALARMS_PER_YEAR = 2.0

FA_PER_10K_TO_PER_YEAR = (
    OBSERVATIONS_PER_YEAR
    / 10_000.0
)


# Output

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "detector_tuning_results.csv"
)


# Candidate configurations
#
# Three configurations are evaluated for each detector.
# Total configurations = 9.


ADWIN_CONFIGS = [

    {
        "delta": 5e-6,
    },

    {
        "delta": 2e-6,
    },

    {
        "delta": 1e-6,
    },
]


KSWIN_CONFIGS = [

    {
        "alpha": 2e-7,
        "window_size": 336,
        "stat_size": 84,
        "seed": 12,
    },

    {
        "alpha": 2e-7,
        "window_size": 672,
        "stat_size": 168,
        "seed": 12,
    },

    {
        "alpha": 2e-7,
        "window_size": 1344,
        "stat_size": 336,
        "seed": 12,
    },
]


PAGE_HINKLEY_CONFIGS = [

    {
        "min_instances": 30,
        "delta": 0.005,
        "threshold": 350,
        "alpha": 0.999,
        "mode": "both",
    },

    {
        "min_instances": 30,
        "delta": 0.005,
        "threshold": 400,
        "alpha": 0.999,
        "mode": "both",
    },

    {
        "min_instances": 30,
        "delta": 0.005,
        "threshold": 450,
        "alpha": 0.999,
        "mode": "both",
    },
]


# Synthetic ground truth

def convert_and_validate_truth(
    full_truth,
    drift_type,
    seed,
):

    if full_truth is None:
        full_truth = []

    full_truth = [
        int(cp)
        for cp in full_truth
    ]

    if drift_type == "none":

        if full_truth:
            raise ValueError(
                "NONE unexpectedly contains changepoints: "
                f"seed={seed}, truth={full_truth}"
            )

        return []

    if not full_truth:
        raise ValueError(
            f"{drift_type}, seed={seed} "
            "contains no changepoints."
        )

    for cp in full_truth:

        if not (
            TEST_START
            <= cp
            < N_OBSERVATIONS
        ):
            raise ValueError(
                "Synthetic changepoint lies outside TEST: "
                f"drift_type={drift_type}, "
                f"seed={seed}, "
                f"cp={cp}"
            )

    return [
        int(
            cp - TEST_START
        )
        for cp in full_truth
    ]


# Detector-stream construction

def build_detector_stream(
    drift_type,
    seed,
):
    """
    Build the CALIBRATION and TEST detector streams.

    The seasonal profile is fitted on complete TRAIN and then
    applied unchanged to TRAIN, CALIBRATION, and TEST.

    The error z-score is fitted on chronological TRAIN Year-2
    absolute forecast errors.
    """

    data, full_truth = make_feature_series(
        kind=drift_type,
        n=N_OBSERVATIONS,
        seed=seed,
        changepoints=(
            SYNTHETIC_CHANGEPOINTS[
                drift_type
            ]
        ),
    )

    if len(data) != N_OBSERVATIONS:
        raise ValueError(
            "Synthetic generator returned "
            f"{len(data)} observations; "
            f"expected {N_OBSERVATIONS}."
        )


    # Chronological split

    train = (
        data.iloc[
            :TRAIN_END
        ]
        .copy()
        .reset_index(drop=True)
    )

    calibration = (
        data.iloc[
            TRAIN_END:
            CALIBRATION_END
        ]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        data.iloc[
            TEST_START:
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(train) != TRAIN_LENGTH:
        raise ValueError(
            f"Unexpected TRAIN length: "
            f"{len(train)}."
        )

    if len(calibration) != CALIBRATION_LENGTH:
        raise ValueError(
            "Unexpected CALIBRATION length: "
            f"{len(calibration)}."
        )

    if len(test) != TEST_LENGTH:
        raise ValueError(
            f"Unexpected TEST length: "
            f"{len(test)}."
        )


    # TRAIN-fitted demand deseasonalisation

    train_ds = (
        deseasonalise_demand_frame(
            data=train,
            reference=train,
        )
    )

    calibration_ds = (
        deseasonalise_demand_frame(
            data=calibration,
            reference=train,
        )
    )

    test_ds = (
        deseasonalise_demand_frame(
            data=test,
            reference=train,
        )
    )

    for name, frame in (
        ("TRAIN", train_ds),
        ("CALIBRATION", calibration_ds),
        ("TEST", test_ds),
    ):

        values = (
            frame[
                "TOTALDEMAND"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"{name} deseasonalised demand "
                "contains non-finite values."
            )


    # TRAIN Year 1 / Year 2 error reference

    reference_fit = (
        train_ds.iloc[
            :TRAIN_REFERENCE_FIT_END
        ]
        .copy()
    )

    reference_eval = (
        train_ds.iloc[
            TRAIN_REFERENCE_FIT_END:
        ]
        .copy()
    )

    if reference_fit.empty:
        raise ValueError(
            "TRAIN reference fit period is empty."
        )

    if reference_eval.empty:
        raise ValueError(
            "TRAIN reference evaluation period is empty."
        )


    # Temporary Gradient Boosting model

    reference_model = (
        GradientBoostedForecaster(
            feature_mode="calendar+lags",
            lags=FORECASTER_LAGS,
            seed=FORECASTER_SEED,
        )
    )

    reference_model.fit(
        reference_fit
    )


    # Chronological TRAIN Year-2 forecast errors

    reference_errors = (
        frozen_error_stream(
            reference_model,
            reference_eval,
            history_df=reference_fit,
        )
    )

    if reference_errors.empty:
        raise ValueError(
            "TRAIN reference forecast-error "
            "stream is empty."
        )

    reference_abs = (
        reference_errors[
            "error"
        ]
        .abs()
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        reference_abs
    ).all():
        raise ValueError(
            "TRAIN reference absolute-error "
            "stream contains non-finite values."
        )


    # TRAIN-only global forecast-error z-score

    error_scaler = (
        ErrorZScorePreprocessor()
    )

    error_scaler.fit(
        reference_abs
    )


    # Final Gradient Boosting model

    model = (
        GradientBoostedForecaster(
            feature_mode="calendar+lags",
            lags=FORECASTER_LAGS,
            seed=FORECASTER_SEED,
        )
    )

    model.fit(
        train_ds
    )


    # CALIBRATION detector stream

    calibration_errors = (
        frozen_error_stream(
            model,
            calibration_ds,
            history_df=train_ds,
        )
    )

    if calibration_errors.empty:
        raise ValueError(
            "Empty CALIBRATION forecast-error stream: "
            f"{drift_type}, seed={seed}"
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

    calibration_processed = (
        error_scaler
        .transform(
            calibration_abs
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        calibration_processed
    ).all():
        raise ValueError(
            "Processed CALIBRATION stream "
            "contains non-finite values."
        )


    # TEST detector stream

    test_errors = (
        frozen_error_stream(
            model,
            test_ds,
            history_df=calibration_ds,
        )
    )

    if test_errors.empty:
        raise ValueError(
            "Empty TEST forecast-error stream: "
            f"{drift_type}, seed={seed}"
        )

    test_abs = (
        test_errors[
            "error"
        ]
        .abs()
        .to_numpy(
            dtype=float
        )
    )

    test_processed = (
        error_scaler
        .transform(
            test_abs
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        test_processed
    ).all():
        raise ValueError(
            "Processed TEST stream "
            "contains non-finite values."
        )


    # Alignment

    if (
        len(calibration_processed)
        != len(calibration_ds)
    ):
        raise ValueError(
            "CALIBRATION detector stream "
            "is misaligned."
        )

    if (
        len(test_processed)
        != len(test_ds)
    ):
        raise ValueError(
            "TEST detector stream "
            "is misaligned."
        )


    # TEST-relative ground truth

    truth = (
        convert_and_validate_truth(
            full_truth=full_truth,
            drift_type=drift_type,
            seed=seed,
        )
    )

    return (
        calibration_processed,
        test_processed,
        truth,
    )


# Build synthetic streams once

def build_stream_cache():

    cache = {}

    print(
        "\nBuilding synthetic streams..."
    )

    print(
        "Seasonal profile: "
        "overall mean + smoothed DOY + "
        "(DOW x half-hour)"
    )

    print(
        "Seasonal reference: complete TRAIN"
    )

    print(
        "Forecast-error z-score reference: "
        "TRAIN Year-2 errors"
    )

    print(
        "Detector warm-up: CALIBRATION"
    )

    print(
        f"Refractory: "
        f"{REFRACTORY_PERIOD} observations "
        f"({REFRACTORY_DAYS} days)"
    )

    for drift_type in DRIFT_TYPES:

        for seed in SEEDS:

            print(
                f"  {drift_type:9s} "
                f"seed={seed}"
            )

            cache[
                (
                    drift_type,
                    seed,
                )
            ] = (
                build_detector_stream(
                    drift_type,
                    seed,
                )
            )

    return cache


# Detector runners

def run_adwin_candidate(
    test_stream,
    warmup_stream,
    *,
    delta,
):

    detector = ADWIN(
        delta=delta
    )

    return run_detector_with_protocol(
        test_stream=test_stream,
        detector=detector,
        warmup_stream=warmup_stream,
        refractory_period=REFRACTORY_PERIOD,
    )


def run_kswin_candidate(
    test_stream,
    warmup_stream,
    *,
    alpha,
    window_size,
    stat_size,
    seed,
):

    detector = KSWIN(
        alpha=alpha,
        window_size=window_size,
        stat_size=stat_size,
        seed=seed,
    )

    return run_detector_with_protocol(
        test_stream=test_stream,
        detector=detector,
        warmup_stream=warmup_stream,
        refractory_period=REFRACTORY_PERIOD,
    )


def run_page_hinkley_candidate(
    test_stream,
    warmup_stream,
    *,
    min_instances,
    delta,
    threshold,
    alpha,
    mode,
):

    detector = PageHinkley(
        min_instances=min_instances,
        delta=delta,
        threshold=threshold,
        alpha=alpha,
        mode=mode,
    )

    return run_detector_with_protocol(
        test_stream=test_stream,
        detector=detector,
        warmup_stream=warmup_stream,
        refractory_period=REFRACTORY_PERIOD,
    )


# Helpers

def config_id(
    config,
):

    return json.dumps(
        config,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def false_alarms_per_year(
    fa_per_10k,
):

    return (
        float(
            fa_per_10k
        )
        * FA_PER_10K_TO_PER_YEAR
    )


# NONE gate

def run_no_drift_gate(
    runner,
    config,
    stream_cache,
):

    counts = []

    for seed in SEEDS:

        (
            warmup_stream,
            test_stream,
            _,
        ) = stream_cache[
            (
                "none",
                seed,
            )
        ]

        detections = runner(
            test_stream,
            warmup_stream,
            **config,
        )

        counts.append(
            len(
                detections
            )
        )

    total = int(
        sum(
            counts
        )
    )

    return {
        "none_total_detections":
            total,

        "none_max_detections_single_seed":
            int(
                max(
                    counts
                )
            ),

        "passes_none_gate":
            total == 0,
    }


# Positive-drift evaluation

def evaluate_positive_drifts(
    detector_name,
    runner,
    config,
    stream_cache,
):

    rows = []

    for drift_type in POSITIVE_DRIFT_TYPES:

        for seed in SEEDS:

            (
                warmup_stream,
                test_stream,
                truth,
            ) = stream_cache[
                (
                    drift_type,
                    seed,
                )
            ]

            detections = runner(
                test_stream,
                warmup_stream,
                **config,
            )

            metrics = (
                evaluate_detections(
                    detected_changepoints=detections,
                    true_changepoints=truth,
                    n_observations=len(
                        test_stream
                    ),
                    drift_type=drift_type,
                    tolerance=TOLERANCE,
                )
            )

            fa_10k = float(
                metrics[
                    "false_alarms_per_10000"
                ]
            )

            rows.append(
                {
                    "detector":
                        detector_name,

                    "config":
                        config_id(
                            config
                        ),

                    "drift_type":
                        drift_type,

                    "seed":
                        seed,

                    "detection_delay":
                        metrics[
                            "detection_delay"
                        ],

                    "false_alarms_per_10k":
                        fa_10k,

                    "false_alarms_per_year":
                        false_alarms_per_year(
                            fa_10k
                        ),

                    "missed_detections":
                        int(
                            metrics[
                                "missed_detections"
                            ]
                        ),

                    "n_detections":
                        int(
                            len(
                                detections
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# Candidate summary

def summarize_candidate(
    detector_name,
    config,
    none_gate,
    positive_df,
):

    per_drift = (
        positive_df
        .groupby(
            "drift_type",
            as_index=False,
        )
        .agg(
            mean_delay=(
                "detection_delay",
                "mean",
            ),

            mean_fa_per_10k=(
                "false_alarms_per_10k",
                "mean",
            ),

            mean_fa_per_year=(
                "false_alarms_per_year",
                "mean",
            ),

            max_seed_fa_per_year=(
                "false_alarms_per_year",
                "max",
            ),

            missed_total=(
                "missed_detections",
                "sum",
            ),
        )
    )

    drift_metrics = {
        row[
            "drift_type"
        ]:
            row

        for _, row
        in per_drift.iterrows()
    }

    budget_pass = {
        drift_type:
            float(
                drift_metrics[
                    drift_type
                ][
                    "mean_fa_per_year"
                ]
            )
            <= MAX_FALSE_ALARMS_PER_YEAR

        for drift_type
        in POSITIVE_DRIFT_TYPES
    }

    passes_fa_budget = all(
        budget_pass.values()
    )

    delays = (
        positive_df[
            "detection_delay"
        ]
        .dropna()
    )

    summary = {
        "detector":
            detector_name,

        "config":
            config_id(
                config
            ),

        **config,

        **none_gate,

        "passes_fa_budget":
            passes_fa_budget,

        "passes_all_hard_constraints":
            (
                none_gate[
                    "passes_none_gate"
                ]
                and
                passes_fa_budget
            ),

        "missed_total_all_drifts":
            int(
                positive_df[
                    "missed_detections"
                ].sum()
            ),

        "worst_mean_fa_per_year":
            float(
                per_drift[
                    "mean_fa_per_year"
                ].max()
            ),

        "mean_fa_per_year_all_drifts":
            float(
                per_drift[
                    "mean_fa_per_year"
                ].mean()
            ),

        "worst_seed_fa_per_year":
            float(
                per_drift[
                    "max_seed_fa_per_year"
                ].max()
            ),

        "mean_detection_delay":
            (
                float(
                    delays.mean()
                )
                if not delays.empty
                else np.inf
            ),
    }

    for drift_type in POSITIVE_DRIFT_TYPES:

        row = (
            drift_metrics[
                drift_type
            ]
        )

        summary[
            f"{drift_type}_mean_delay"
        ] = row[
            "mean_delay"
        ]

        summary[
            f"{drift_type}_fa_per_10k"
        ] = row[
            "mean_fa_per_10k"
        ]

        summary[
            f"{drift_type}_fa_per_year"
        ] = row[
            "mean_fa_per_year"
        ]

        summary[
            f"{drift_type}_missed_total"
        ] = int(
            row[
                "missed_total"
            ]
        )

        summary[
            f"{drift_type}_passes_budget"
        ] = (
            budget_pass[
                drift_type
            ]
        )

    return summary


# Candidate that fails NONE

def failed_none_row(
    detector_name,
    config,
    none_gate,
):

    row = {
        "detector":
            detector_name,

        "config":
            config_id(
                config
            ),

        **config,

        **none_gate,

        "passes_fa_budget":
            False,

        "passes_all_hard_constraints":
            False,

        "missed_total_all_drifts":
            np.nan,

        "worst_mean_fa_per_year":
            np.nan,

        "mean_fa_per_year_all_drifts":
            np.nan,

        "worst_seed_fa_per_year":
            np.nan,

        "mean_detection_delay":
            np.nan,
    }

    for drift_type in POSITIVE_DRIFT_TYPES:

        row[
            f"{drift_type}_mean_delay"
        ] = np.nan

        row[
            f"{drift_type}_fa_per_10k"
        ] = np.nan

        row[
            f"{drift_type}_fa_per_year"
        ] = np.nan

        row[
            f"{drift_type}_missed_total"
        ] = np.nan

        row[
            f"{drift_type}_passes_budget"
        ] = False

    return row


# Tune one detector

def tune_detector(
    detector_name,
    runner,
    configs,
    stream_cache,
):

    results = []

    print(
        f"\nTuning {detector_name}: "
        f"{len(configs)} candidates"
    )

    for i, config in enumerate(
        configs,
        start=1,
    ):

        print(
            f"  candidate "
            f"{i}/{len(configs)}: "
            f"{config_id(config)}"
        )

        none_gate = (
            run_no_drift_gate(
                runner,
                config,
                stream_cache,
            )
        )

        if not none_gate[
            "passes_none_gate"
        ]:

            results.append(
                failed_none_row(
                    detector_name,
                    config,
                    none_gate,
                )
            )

            continue

        positive_df = (
            evaluate_positive_drifts(
                detector_name,
                runner,
                config,
                stream_cache,
            )
        )

        results.append(
            summarize_candidate(
                detector_name,
                config,
                none_gate,
                positive_df,
            )
        )

    return pd.DataFrame(
        results
    )


# Select the best passing configuration

def mark_selected(
    combined_df,
):

    combined_df = (
        combined_df.copy()
    )

    combined_df[
        "selected"
    ] = False

    for detector_name in (
        "adwin",
        "kswin",
        "page_hinkley",
    ):

        passing = (
            combined_df[
                (
                    combined_df[
                        "detector"
                    ]
                    == detector_name
                )
                &
                (
                    combined_df[
                        "passes_all_hard_constraints"
                    ]
                    == True
                )
            ]
            .copy()
        )

        if passing.empty:
            continue

        passing = (
            passing
            .sort_values(
                by=[
                    "missed_total_all_drifts",
                    "worst_mean_fa_per_year",
                    "mean_fa_per_year_all_drifts",
                    "mean_detection_delay",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    True,
                ],
            )
        )

        best_index = (
            passing.index[
                0
            ]
        )

        combined_df.loc[
            best_index,
            "selected",
        ] = True

    return combined_df


# Print selected configurations

def print_selected(
    combined_df,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SELECTED CONFIGURATIONS"
    )

    print(
        "=" * 80
    )

    for detector_name in (
        "adwin",
        "kswin",
        "page_hinkley",
    ):

        selected = (
            combined_df[
                (
                    combined_df[
                        "detector"
                    ]
                    == detector_name
                )
                &
                (
                    combined_df[
                        "selected"
                    ]
                    == True
                )
            ]
        )

        if selected.empty:

            print(
                f"\n{detector_name}: "
                "NO PASSING CONFIGURATION"
            )

            continue

        row = (
            selected.iloc[
                0
            ]
        )

        print(
            f"\n{detector_name}"
        )

        print(
            f"  config = "
            f"{row['config']}"
        )

        print(
            "  NONE detections = "
            f"{int(row['none_total_detections'])}"
        )

        print(
            "  sudden FA/year = "
            f"{row['sudden_fa_per_year']:.3f}"
        )

        print(
            "  gradual FA/year = "
            f"{row['gradual_fa_per_year']:.3f}"
        )

        print(
            "  recurring FA/year = "
            f"{row['recurring_fa_per_year']:.3f}"
        )

        print(
            "  total misses = "
            f"{int(row['missed_total_all_drifts'])}"
        )

        print(
            "  mean delay = "
            f"{row['mean_detection_delay']:.2f}"
        )


# Main

def main():

    print(
        "=" * 80
    )

    print(
        "SYNTHETIC DETECTOR TUNING"
    )

    print(
        "=" * 80
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"TRAIN: "
        f"{TRAIN_LENGTH:,}"
    )

    print(
        f"CALIBRATION: "
        f"{CALIBRATION_LENGTH:,}"
    )

    print(
        f"TEST: "
        f"{TEST_LENGTH:,}"
    )

    print(
        "Seasonal profile: "
        "overall mean + smoothed DOY + "
        "(DOW x half-hour)"
    )

    print(
        "Seasonal reference: "
        "complete TRAIN"
    )

    print(
        "Forecaster: Gradient Boosting "
        "calendar+lags (48, 336)"
    )

    print(
        "Error z-score reference: "
        "chronological TRAIN Year-2 errors"
    )

    print(
        "Detector warm-up: "
        "processed CALIBRATION errors"
    )

    print(
        f"Refractory: "
        f"{REFRACTORY_DAYS} days"
    )

    print(
        f"Matching tolerance: "
        f"{TOLERANCE_DAYS} days"
    )

    print(
        "\nCandidates:"
    )

    print(
        f"  ADWIN        = "
        f"{len(ADWIN_CONFIGS)}"
    )

    print(
        f"  KSWIN        = "
        f"{len(KSWIN_CONFIGS)}"
    )

    print(
        f"  Page-Hinkley = "
        f"{len(PAGE_HINKLEY_CONFIGS)}"
    )

    print(
        f"  Total        = "
        f"{(
            len(ADWIN_CONFIGS)
            + len(KSWIN_CONFIGS)
            + len(PAGE_HINKLEY_CONFIGS)
        )}"
    )


    # Build streams

    stream_cache = (
        build_stream_cache()
    )


    # Tune detectors

    adwin_df = (
        tune_detector(
            detector_name="adwin",
            runner=run_adwin_candidate,
            configs=ADWIN_CONFIGS,
            stream_cache=stream_cache,
        )
    )

    kswin_df = (
        tune_detector(
            detector_name="kswin",
            runner=run_kswin_candidate,
            configs=KSWIN_CONFIGS,
            stream_cache=stream_cache,
        )
    )

    ph_df = (
        tune_detector(
            detector_name="page_hinkley",
            runner=run_page_hinkley_candidate,
            configs=PAGE_HINKLEY_CONFIGS,
            stream_cache=stream_cache,
        )
    )


    # Combine and select

    combined_df = (
        pd.concat(
            [
                adwin_df,
                kswin_df,
                ph_df,
            ],
            ignore_index=True,
            sort=False,
        )
    )

    combined_df = (
        mark_selected(
            combined_df
        )
    )

    combined_df = (
        combined_df
        .sort_values(
            by=[
                "detector",
                "selected",
                "passes_all_hard_constraints",
                "missed_total_all_drifts",
                "worst_mean_fa_per_year",
                "mean_fa_per_year_all_drifts",
                "mean_detection_delay",
            ],
            ascending=[
                True,
                False,
                False,
                True,
                True,
                True,
                True,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


    # Save results

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


    # Summary

    print_selected(
        combined_df
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SYNTHETIC TUNING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()