"""
Re-evaluate KSWIN and Page-Hinkley on the synthetic concept-drift
benchmark under the project's explicit false-alarm operating requirement.

Purpose
-------
The original detector configurations were selected using the T1
synthetic benchmark, where performance was assessed using detection
delay, false alarms per 10,000 observations, missed detections, and the
no-drift control.

Before final application to the AEMO forecast-error streams, an
additional operational requirement was introduced: detector sensitivity
should be calibrated to approximately fewer than 2 false alarms per year.

This experiment applies that requirement to KSWIN and Page-Hinkley using
only the controlled synthetic benchmark with known changepoints.

KSWIN is reconsidered because its previously selected configuration
produced false-alarm rates materially above the new operating target.
Page-Hinkley is re-evaluated under the same protocol to confirm the
previously selected directionality and sensitivity configuration using a
consistent false-alarm-budget selection rule.

ADWIN is not included in further tuning because targeted searches over
delta and min_window_length did not identify a configuration satisfying
the false-alarm budget for every positive drift type while maintaining
zero missed detections. Its existing configuration is therefore retained
as the best observed sensitivity–false-alarm trade-off.

Protocol integrity
------------------
Parameter selection is performed entirely on synthetic data. AEMO
detections are not used to choose detector settings.

The experiment preserves the existing T1 protocol:

- seeds: 1, 2, 3, 4, 5
- drift types: none, sudden, gradual, recurring
- train/calibration/test boundaries unchanged
- frozen Gradient Boosting forecaster
- calendar features with lags 48 and 336
- detector input: calibration-normalized absolute forecast error
- matching tolerance: 336 half-hour observations (7 days)

Selection requirements
----------------------
A candidate must:

1. produce zero detections across all five no-drift seeds; and
2. remain below 2 mean false alarms/year separately for sudden,
   gradual, and recurring drift.

Passing candidates are ranked by:

1. fewest missed detections;
2. lowest worst drift-type mean false-alarm rate;
3. lowest mean false-alarm rate;
4. lowest mean detection delay.

The selected configurations are frozen from this synthetic evaluation
before final AEMO analysis.

Run:
    python experiments/detection/tune_detectors_false_alarm_budget.py

Output:
    results/tables/detection/tuning _detector_budget.csv
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from river.drift import PageHinkley


#  Project imports

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drift_lab import detectors
from drift_lab.evaluation import evaluate_detections
from drift_lab.synthetic_generator import make_feature_series
from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)


#  Shared synthetic protocol

SEEDS = [1, 2, 3, 4, 5]

DRIFT_TYPES = [
    "none",
    "sudden",
    "gradual",
    "recurring",
]

POSITIVE_DRIFT_TYPES = [
    "sudden",
    "gradual",
    "recurring",
]

N_OBSERVATIONS = 20_000

TRAIN_END = 4_000
CALIBRATION_END = 5_000
TEST_START = 5_000

FORECASTER_SEED = 0

# 7 days × 48 half-hour observations
TOLERANCE = 48 * 7

OBSERVATIONS_PER_YEAR = 48 * 365

MAX_FALSE_ALARMS_PER_YEAR = 2.0

FA_PER_10K_TO_PER_YEAR = (
    OBSERVATIONS_PER_YEAR / 10_000
)

MAX_FA_PER_10K = (
    MAX_FALSE_ALARMS_PER_YEAR
    / FA_PER_10K_TO_PER_YEAR
)


#  KSWIN candidate grid

KSWIN_CONFIGS = [
    # Original configuration
    {
        "alpha": 5e-5,
        "window_size": 140,
        "stat_size": 70,
        "seed": 12,
    },

    *[
        {
            "alpha": alpha,
            "window_size": 140,
            "stat_size": stat_size,
            "seed": 12,
        }
        for alpha in [5e-6, 1e-5, 2e-5]
        for stat_size in [30, 40]
    ],

    *[
        {
            "alpha": alpha,
            "window_size": 200,
            "stat_size": stat_size,
            "seed": 12,
        }
        for alpha in [5e-6, 1e-5, 2e-5]
        for stat_size in [40, 50]
    ],

    *[
        {
            "alpha": alpha,
            "window_size": 336,
            "stat_size": 48,
            "seed": 12,
        }
        for alpha in [5e-6, 1e-5, 2e-5]
    ],
]


#  Page-Hinkley candidate grid

PAGE_HINKLEY_CONFIGS = [
    {
        "min_instances": 30,
        "delta": 0.005,
        "threshold": threshold,
        "alpha": alpha,
        "mode": mode,
    }
    for mode in ["both", "up"]
    for alpha in [
        0.975,
        0.99,
        0.995,
        0.999,
    ]
    for threshold in [
        125,
        150,
        175,
        200,
        209,
    ]
]


#  Page-Hinkley local runner

def run_page_hinkley_candidate(
    stream,
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

    changepoints = []

    for index, value in enumerate(
        np.asarray(stream, dtype=float)
    ):
        detector.update(float(value))

        if detector.drift_detected:
            changepoints.append(index)

    return changepoints


#  Detector stream construction

def build_normalized_absolute_error_stream(
    drift_type,
    seed,
):
    data, full_truth = make_feature_series(
        kind=drift_type,
        n=N_OBSERVATIONS,
        seed=seed,
    )

    train = (
        data.iloc[:TRAIN_END]
        .copy()
    )

    calibration = (
        data.iloc[
            TRAIN_END:CALIBRATION_END
        ]
        .copy()
    )

    test = (
        data.iloc[TEST_START:]
        .copy()
    )

    model = GradientBoostedForecaster(
        feature_mode="calendar+lags",
        lags=(48, 336),
        seed=FORECASTER_SEED,
    )

    model.fit(train)

    calibration_errors = (
        frozen_error_stream(
            model,
            calibration,
            history_df=train,
        )
    )

    calibration_abs = (
        calibration_errors["error"]
        .abs()
        .to_numpy(dtype=float)
    )

    cal_mean = float(
        calibration_abs.mean()
    )

    cal_std = float(
        calibration_abs.std()
    )

    if (
        not np.isfinite(cal_std)
        or cal_std <= 0
    ):
        raise ValueError(
            f"Invalid calibration std: "
            f"{drift_type}, seed={seed}"
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
        test_abs - cal_mean
    ) / cal_std

    if not np.isfinite(stream).all():
        raise ValueError(
            f"Non-finite detector stream: "
            f"{drift_type}, seed={seed}"
        )

    truth = [
        int(cp - TEST_START)
        for cp in full_truth
        if cp >= TEST_START
    ]

    return stream, truth


def build_stream_cache():

    cache = {}

    print(
        "Building shared synthetic streams..."
    )

    for drift_type in DRIFT_TYPES:

        for seed in SEEDS:

            print(
                f"  {drift_type:9s} "
                f"seed={seed}"
            )

            cache[
                (drift_type, seed)
            ] = (
                build_normalized_absolute_error_stream(
                    drift_type,
                    seed,
                )
            )

    return cache


#  Helpers

def config_id(config):

    return json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    )


def false_alarms_per_year(
    fa_per_10k,
):

    return (
        float(fa_per_10k)
        * FA_PER_10K_TO_PER_YEAR
    )


#  No-drift gate

def run_no_drift_gate(
    runner,
    config,
    stream_cache,
):

    counts = []

    for seed in SEEDS:

        stream, _ = (
            stream_cache[
                ("none", seed)
            ]
        )

        detections = runner(
            stream,
            **config,
        )

        counts.append(
            len(detections)
        )

    total = int(
        sum(counts)
    )

    return {
        "none_total_detections": total,
        "none_max_detections_single_seed": int(
            max(counts)
        ),
        "passes_none_gate": (
            total == 0
        ),
    }


#  Positive drift evaluation

def evaluate_positive_drifts(
    detector_name,
    runner,
    config,
    stream_cache,
):

    rows = []

    for drift_type in POSITIVE_DRIFT_TYPES:

        for seed in SEEDS:

            stream, truth = (
                stream_cache[
                    (drift_type, seed)
                ]
            )

            detections = runner(
                stream,
                **config,
            )

            metrics = evaluate_detections(
                detected_changepoints=detections,
                true_changepoints=truth,
                n_observations=len(stream),
                drift_type=drift_type,
                tolerance=TOLERANCE,
            )

            fa_10k = float(
                metrics[
                    "false_alarms_per_10000"
                ]
            )

            rows.append(
                {
                    "detector": detector_name,
                    "config": config_id(
                        config
                    ),
                    "drift_type": drift_type,
                    "seed": seed,
                    "detection_delay": (
                        metrics[
                            "detection_delay"
                        ]
                    ),
                    "false_alarms_per_10k": (
                        fa_10k
                    ),
                    "false_alarms_per_year": (
                        false_alarms_per_year(
                            fa_10k
                        )
                    ),
                    "missed_detections": int(
                        metrics[
                            "missed_detections"
                        ]
                    ),
                    "n_detections": int(
                        len(detections)
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


#  Candidate summary

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
        row["drift_type"]: row
        for _, row
        in per_drift.iterrows()
    }

    budget_pass = {
        drift_type: (
            float(
                drift_metrics[
                    drift_type
                ][
                    "mean_fa_per_year"
                ]
            )
            < MAX_FALSE_ALARMS_PER_YEAR
        )
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
        "detector": detector_name,
        "config": config_id(
            config
        ),

        **config,
        **none_gate,

        "passes_fa_budget": (
            passes_fa_budget
        ),

        "passes_all_hard_constraints": (
            none_gate[
                "passes_none_gate"
            ]
            and passes_fa_budget
        ),

        "missed_total_all_drifts": int(
            positive_df[
                "missed_detections"
            ].sum()
        ),

        "worst_mean_fa_per_year": float(
            per_drift[
                "mean_fa_per_year"
            ].max()
        ),

        "mean_fa_per_year_all_drifts": float(
            per_drift[
                "mean_fa_per_year"
            ].mean()
        ),

        "worst_seed_fa_per_year": float(
            per_drift[
                "max_seed_fa_per_year"
            ].max()
        ),

        "mean_detection_delay": (
            float(
                delays.mean()
            )
            if not delays.empty
            else np.inf
        ),
    }

    for drift_type in (
        POSITIVE_DRIFT_TYPES
    ):

        row = drift_metrics[
            drift_type
        ]

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
        ] = budget_pass[
            drift_type
        ]

    return summary


#  No-drift failed row

def failed_none_row(
    detector_name,
    config,
    none_gate,
):

    row = {
        "detector": detector_name,
        "config": config_id(
            config
        ),

        **config,
        **none_gate,

        "passes_fa_budget": False,
        "passes_all_hard_constraints": False,

        "missed_total_all_drifts": np.nan,
        "worst_mean_fa_per_year": np.nan,
        "mean_fa_per_year_all_drifts": np.nan,
        "worst_seed_fa_per_year": np.nan,
        "mean_detection_delay": np.nan,
    }

    for drift_type in (
        POSITIVE_DRIFT_TYPES
    ):

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


# Generic detector tuner

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


# Select one best configuration per detector

def mark_selected(
    combined_df,
):

    combined_df = (
        combined_df.copy()
    )

    combined_df[
        "selected"
    ] = False

    for detector_name in [
        "kswin",
        "page_hinkley",
    ]:

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
            passing.sort_values(
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
            passing.index[0]
        )

        combined_df.loc[
            best_index,
            "selected",
        ] = True

    return combined_df


#  Console summary

def print_selected(
    combined_df,
):

    print(
        "\n" + "=" * 80
    )

    print(
        "SELECTED CONFIGURATIONS"
    )

    print(
        "=" * 80
    )

    for detector_name in [
        "kswin",
        "page_hinkley",
    ]:

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
            selected.iloc[0]
        )

        print(
            f"\n{detector_name}"
        )

        print(
            f"  config = "
            f"{row['config']}"
        )

        print(
            "  no-drift detections = "
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


#  Main

def main():

    print(
        "=" * 80
    )

    print(
        "Synthetic KSWIN + Page-Hinkley "
        "false-alarm-budget tuning"
    )

    print(
        "=" * 80
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Observations/year: "
        f"{OBSERVATIONS_PER_YEAR:,}"
    )

    print(
        f"Hard FA budget: "
        f"< {MAX_FALSE_ALARMS_PER_YEAR:g}"
        " / year for EACH positive drift"
    )

    print(
        "Equivalent: "
        f"< {MAX_FA_PER_10K:.5f} "
        "FA / 10k"
    )

    print(
        "No-drift gate: "
        "0 detections across all 5 seeds"
    )

    print(
        "Ranking: "
        "misses -> worst FA -> "
        "mean FA -> delay"
    )

    print(
        "AEMO is NOT used for tuning."
    )

    #  Build once, shared by both detectors
   
    stream_cache = (
        build_stream_cache()
    )

    # KSWIN
    
    kswin_df = tune_detector(
        detector_name="kswin",
        runner=detectors.run_kswin,
        configs=KSWIN_CONFIGS,
        stream_cache=stream_cache,
    )

    # Page-Hinkley
    
    ph_df = tune_detector(
        detector_name="page_hinkley",
        runner=run_page_hinkley_candidate,
        configs=PAGE_HINKLEY_CONFIGS,
        stream_cache=stream_cache,
    )

    # Combine
   
    combined_df = pd.concat(
        [
            kswin_df,
            ph_df,
        ],
        ignore_index=True,
        sort=False,
    )

    # Mark selected configuration per detector
   
    combined_df = mark_selected(
        combined_df
    )

    # Put selected/passing candidates first
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

    # ONE output file
   
    output_dir = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "detection"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "detector_budget_tuning.csv"
    )

    combined_df.to_csv(
        output_path,
        index=False,
    )

    # Print selected configs
    print_selected(
        combined_df
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "Combined tuning complete"
    )

    print(
        "=" * 80
    )

    print(
        f"Result file:\n"
        f"{output_path}"
    )

    print(
        "\nUse rows with "
        "selected=True as the final "
        "synthetically selected configurations."
    )


if __name__ == "__main__":
    main()