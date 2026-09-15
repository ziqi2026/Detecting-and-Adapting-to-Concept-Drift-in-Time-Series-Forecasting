"""
10_tune_detectors_for_aggregated_input.py

Focused tuning for 3-hour non-overlapping MAE detector input.
======================================================================

Purpose
-------
Diagnosis 08 showed that strong serial dependence alone can produce
large numbers of detector alarms even when there is no true drift.

Diagnosis 09 showed that 3-hour non-overlapping MAE substantially
reduces:

    - AEMO forecast-error autocorrelation
    - AEMO detection counts
    - stationary autocorrelation false alarms

while preserving more temporal resolution than coarser aggregation.

Therefore this script tunes detector parameters specifically for:

    3-hour non-overlapping MAE

We deliberately use a SMALL, interpretable candidate set rather than
a large Cartesian grid.


======================================================================
PROJECT RULES
======================================================================

RULE 1 — STRICT
---------------
Synthetic "none" must produce:

    EXACTLY 0 detections

across ALL official seeds:

    1, 2, 3, 4, 5

If a candidate produces even one detection in any "none" seed:

    REJECT IT IMMEDIATELY.

The positive drift scenarios and AR diagnostic are not run for that
candidate.


RULE 2 — SOFT TARGET
--------------------
For positive drift scenarios:

    sudden
    gradual
    recurring

the preferred false-alarm rate is approximately:

    <= 2 false alarms / year

But this is NOT a strict rejection threshold.

Examples:

    1.6/year    excellent
    2.0/year    target
    2.2/year    still very close
    2.8/year    may still be acceptable
    5.0/year    increasingly undesirable

Therefore we calculate:

    FA excess above target =
        max(0, worst_FA_per_year - 2)

A candidate is NOT rejected simply because it is slightly above 2.

Detection ability must also be considered.


======================================================================
SELECTION LOGIC
======================================================================

Step 1:
    Reject any configuration that fails Rule 1.

Step 2:
    Among surviving configurations, determine the best achievable
    synthetic drift coverage for each detector.

Step 3:
    Prefer candidates with that best coverage.

Step 4:
    Among similarly good coverage candidates, prefer false-alarm rates
    closest to or below 2/year.

Step 5:
    Then prefer lower detection delay.

Step 6:
    Use stationary AR(1) only as an ADDITIONAL robustness diagnostic.

AR(1) is NOT:
    - another official synthetic scenario
    - part of Rule 1
    - a hard parameter-selection gate


======================================================================
FALSE ALARM UNITS
======================================================================

Original AEMO/synthetic sampling:

    48 observations/day
    17,520 observations/year

Evaluation returns:

    false alarms / 10,000 original observations

Conversion:

    FA/year =
        FA_per_10k * 17,520 / 10,000


======================================================================
3-HOUR AGGREGATION
======================================================================

Each detector observation represents:

    6 x 30-minute errors
    = 3 hours

using non-overlapping MAE:

    block 1 = mean(|e1 ... e6|)
    block 2 = mean(|e7 ... e12|)
    ...


======================================================================
KSWIN CANDIDATES
======================================================================

Original configuration:

    window_size = 140
    stat_size   = 30
    alpha       = 5e-6

At 30-minute resolution:

    window horizon =
        140 * 0.5h
        = 70 hours

For 3-hour input, an approximately equivalent physical window is:

    70 / 3
    ~= 23 observations

Therefore we examine:

    window_size:
        20 = 60 h
        24 = 72 h
        28 = 84 h

stat_size stays around the original proportion:

    original:
        30 / 140
        ~= 21%

Candidates:

    window 20 -> stat 5
    window 24 -> stat 5
    window 28 -> stat 6

with three alpha levels:

    5e-6
    5e-5
    5e-4

Total:

    9 KSWIN candidates


======================================================================
PAGE-HINKLEY
======================================================================

Original:

    min_instances = 30 half-hours
                  = 15 hours

At 3-hour resolution:

    15 / 3
    = 5 observations

Therefore most candidates use:

    min_instances = 5

with a small number using:

    min_instances = 10

to test a more persistent requirement.


======================================================================
ADWIN
======================================================================

ADWIN has no fixed sample window that needs direct physical rescaling.

We therefore examine only five reasonable delta values around the
current:

    current = 1e-4


======================================================================
OUTPUTS
======================================================================

results/tables/detection/diagnostics/

    10_3h_tuning_runs.csv
    10_3h_tuning_summary.csv
    10_3h_tuning_ranked.csv
    10_3h_rejected_no_drift.csv


======================================================================
IMPORTANT
======================================================================

This script does NOT:

    - use AEMO for parameter tuning
    - modify detectors.py
    - modify Task 120
    - modify T2
    - modify runs.csv
    - use prewhitening
    - use rolling MAE
    - test 6h yet

Only if 3h cannot produce a reasonable trade-off should 6h be tested.


Run
---
python experiments/detection/10_tune_detectors_for_aggregated_input.py
"""


from pathlib import Path
import hashlib
import json
import sys

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
# Paths
# =====================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "diagnostics"
)


# =====================================================================
# Resolution
# =====================================================================

# 6 half-hour observations = 3 hours.
BLOCK_SIZE = 6

RESOLUTION = "3h_MAE"


# =====================================================================
# Original time units
# =====================================================================

OBSERVATIONS_PER_DAY = 48

OBSERVATIONS_PER_YEAR = (
    48
    * 365
)

HOURS_PER_BLOCK = 3.0


# =====================================================================
# Project FA target
# =====================================================================

# IMPORTANT:
#
# This is a TARGET, not a hard rejection limit.
TARGET_FALSE_ALARMS_PER_YEAR = 2.0


# Equivalent value in the metric returned by evaluation.py.
TARGET_FALSE_ALARMS_PER_10K = (
    TARGET_FALSE_ALARMS_PER_YEAR
    /
    OBSERVATIONS_PER_YEAR
    *
    10_000
)


# =====================================================================
# Synthetic protocol
# =====================================================================

N_OBSERVATIONS = 20_000

TRAIN_END = 4_000

CALIBRATION_END = 5_000

TEST_START = 5_000


SYNTHETIC_SEEDS = (
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


TOLERANCE = (
    48
    * 7
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
# Stationary AR(1) robustness diagnostic
# =====================================================================

# This is deliberately SECONDARY.
#
# Five seeds are sufficient here because Diagnosis 08 has already
# demonstrated the autocorrelation failure mode with 20 seeds.

AR_PHI = 0.98

AR_SEEDS = (
    1,
    2,
    3,
    4,
    5,
)

AR_BURN_IN = 2_000

AR_CALIBRATION_SIZE = 2_880

AR_TEST_SIZE = 15_000


# =====================================================================
# Small focused candidate sets
# =====================================================================

ADWIN_CANDIDATES = [
    {
        "delta": 1e-5,
    },
    {
        "delta": 5e-5,
    },
    {
        "delta": 1e-4,
    },
    {
        "delta": 5e-4,
    },
    {
        "delta": 1e-3,
    },
]


# ---------------------------------------------------------------------
# Page-Hinkley
# ---------------------------------------------------------------------

PAGE_HINKLEY_CANDIDATES = [

    # -------------------------------------------------------------
    # ~15-hour min_instances:
    #
    # 5 x 3h = 15 h
    # -------------------------------------------------------------

    {
        "min_instances": 5,
        "delta": 0.005,
        "threshold": 25,
        "alpha": 0.98,
        "mode": "up",
    },

    {
        "min_instances": 5,
        "delta": 0.005,
        "threshold": 25,
        "alpha": 0.99,
        "mode": "up",
    },

    {
        "min_instances": 5,
        "delta": 0.005,
        "threshold": 50,
        "alpha": 0.98,
        "mode": "up",
    },

    {
        "min_instances": 5,
        "delta": 0.005,
        "threshold": 50,
        "alpha": 0.99,
        "mode": "up",
    },

    {
        "min_instances": 5,
        "delta": 0.005,
        "threshold": 75,
        "alpha": 0.98,
        "mode": "up",
    },

    {
        "min_instances": 5,
        "delta": 0.005,
        "threshold": 75,
        "alpha": 0.99,
        "mode": "up",
    },

    {
        "min_instances": 5,
        "delta": 0.005,
        "threshold": 125,
        "alpha": 0.98,
        "mode": "up",
    },

    {
        "min_instances": 5,
        "delta": 0.005,
        "threshold": 125,
        "alpha": 0.99,
        "mode": "up",
    },


    # -------------------------------------------------------------
    # More persistent 30-hour requirement.
    # Only a few focused candidates.
    # -------------------------------------------------------------

    {
        "min_instances": 10,
        "delta": 0.005,
        "threshold": 50,
        "alpha": 0.98,
        "mode": "up",
    },

    {
        "min_instances": 10,
        "delta": 0.005,
        "threshold": 50,
        "alpha": 0.99,
        "mode": "up",
    },

    {
        "min_instances": 10,
        "delta": 0.005,
        "threshold": 75,
        "alpha": 0.98,
        "mode": "up",
    },

    {
        "min_instances": 10,
        "delta": 0.005,
        "threshold": 75,
        "alpha": 0.99,
        "mode": "up",
    },
]


# ---------------------------------------------------------------------
# KSWIN
# ---------------------------------------------------------------------
KSWIN_CANDIDATES = [
    # Moderate alpha, moderate statistical sample
    {
        "alpha": 0.005,
        "window_size": 32,
        "stat_size": 8,
        "seed": 12,
    },
    {
        "alpha": 0.001,
        "window_size": 32,
        "stat_size": 8,
        "seed": 12,
    },

    # Slightly larger statistical sample
    {
        "alpha": 0.001,
        "window_size": 40,
        "stat_size": 10,
        "seed": 12,
    },
    {
        "alpha": 0.0005,
        "window_size": 40,
        "stat_size": 10,
        "seed": 12,
    },

    # Allows very small alpha to actually be reachable
    {
        "alpha": 5e-5,
        "window_size": 44,
        "stat_size": 11,
        "seed": 12,
    },
    {
        "alpha": 5e-6,
        "window_size": 44,
        "stat_size": 11,
        "seed": 12,
    },

    {
        "alpha": 5e-5,
        "window_size": 48,
        "stat_size": 12,
        "seed": 12,
    },
    {
        "alpha": 5e-6,
        "window_size": 48,
        "stat_size": 12,
        "seed": 12,
    },
]

CANDIDATES = {
    "adwin":
        ADWIN_CANDIDATES,

    "page_hinkley":
        PAGE_HINKLEY_CANDIDATES,

    "kswin":
        KSWIN_CANDIDATES,
}


# =====================================================================
# Basic helpers
# =====================================================================

def validate_stream(
    values,
    name,
):

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


def config_json(
    config,
):

    return json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    )


def config_id(
    detector,
    config,
):

    payload = (
        detector
        + "|"
        + config_json(
            config
        )
    )


    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()[:10]


    return (
        f"{detector}_{digest}"
    )


# =====================================================================
# Aggregation
# =====================================================================

def aggregate_mae(
    absolute_error,
):

    values = validate_stream(
        absolute_error,
        "absolute error",
    )


    n_blocks = (
        len(values)
        // BLOCK_SIZE
    )


    usable = (
        n_blocks
        * BLOCK_SIZE
    )


    if n_blocks < 2:

        raise RuntimeError(
            "Too few complete 3-hour blocks."
        )


    blocks = (
        values[
            :usable
        ]
        .reshape(
            n_blocks,
            BLOCK_SIZE,
        )
        .mean(
            axis=1
        )
    )


    return (
        validate_stream(
            blocks,
            "3-hour MAE blocks",
        ),
        usable,
    )


def aggregate_and_normalize(
    calibration_absolute,
    test_absolute,
):

    (
        calibration_blocks,
        calibration_usable,
    ) = aggregate_mae(
        calibration_absolute
    )


    (
        test_blocks,
        test_usable,
    ) = aggregate_mae(
        test_absolute
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
        or
        calibration_std <= 0
    ):

        raise RuntimeError(
            "Invalid calibration block standard deviation."
        )


    detector_stream = (
        test_blocks
        -
        calibration_mean
    ) / calibration_std


    return {
        "stream":
            validate_stream(
                detector_stream,
                "normalized 3h detector stream",
            ),

        "calibration_blocks":
            len(
                calibration_blocks
            ),

        "test_blocks":
            len(
                test_blocks
            ),

        "calibration_usable":
            calibration_usable,

        "test_usable":
            test_usable,

        "calibration_mean":
            calibration_mean,

        "calibration_std":
            calibration_std,
    }


# =====================================================================
# Detection index mapping
# =====================================================================

def block_indices_to_original(
    detected_blocks,
    test_usable,
):

    mapped = []


    for block_index in detected_blocks:

        original_index = (
            (
                int(
                    block_index
                )
                + 1
            )
            * BLOCK_SIZE
            - 1
        )


        if not (
            0
            <= original_index
            < test_usable
        ):

            raise RuntimeError(
                "Mapped detector index outside test stream."
            )


        mapped.append(
            original_index
        )


    return mapped


# =====================================================================
# Synthetic forecast errors
# =====================================================================

def build_synthetic_absolute_errors(
    drift_type,
    seed,
):

    (
        data,
        full_truth,
    ) = make_feature_series(
        kind=drift_type,
        n=N_OBSERVATIONS,
        seed=seed,
    )


    train = data.iloc[
        :TRAIN_END
    ].copy()


    calibration = data.iloc[
        TRAIN_END:
        CALIBRATION_END
    ].copy()


    test = data.iloc[
        TEST_START:
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
            TEST_START
            <= cp
            < N_OBSERVATIONS
        ):

            raise ValueError(
                f"Ground-truth changepoint {cp} "
                "outside synthetic test period."
            )


        truth.append(
            int(
                cp
                - TEST_START
            )
        )


    return (
        calibration_absolute,
        test_absolute,
        truth,
    )


# =====================================================================
# Synthetic cache
# =====================================================================

def build_synthetic_cache():

    cache = {}


    print()
    print(
        "=" * 90
    )

    print(
        "PRECOMPUTING 3-HOUR SYNTHETIC STREAMS"
    )

    print(
        "=" * 90
    )


    for seed in SYNTHETIC_SEEDS:

        for drift_type in DRIFT_TYPES:

            (
                calibration_absolute,
                test_absolute,
                truth,
            ) = build_synthetic_absolute_errors(
                drift_type=drift_type,
                seed=seed,
            )


            result = aggregate_and_normalize(
                calibration_absolute,
                test_absolute,
            )


            for cp in truth:

                if cp >= result[
                    "test_usable"
                ]:

                    raise RuntimeError(
                        "Aggregation removed a true changepoint."
                    )


            cache[
                (
                    drift_type,
                    seed,
                )
            ] = {
                **result,
                "truth":
                    truth,
            }


            print(
                f"prepared seed={seed}, "
                f"scenario={drift_type}"
            )


    return cache


# =====================================================================
# Stationary AR(1) diagnostic
# =====================================================================

def generate_stationary_ar1(
    seed,
):

    rng = np.random.default_rng(
        seed
    )


    total = (
        AR_BURN_IN
        +
        AR_CALIBRATION_SIZE
        +
        AR_TEST_SIZE
    )


    values = np.empty(
        total,
        dtype=float,
    )


    values[
        0
    ] = rng.normal()


    innovation_std = np.sqrt(
        1
        -
        AR_PHI ** 2
    )


    noise = rng.normal(
        loc=0,
        scale=innovation_std,
        size=total - 1,
    )


    for index in range(
        1,
        total,
    ):

        values[
            index
        ] = (
            AR_PHI
            *
            values[
                index - 1
            ]
            +
            noise[
                index - 1
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


def build_ar_cache():

    cache = {}


    print()
    print(
        "=" * 90
    )

    print(
        "PRECOMPUTING AR(1) ROBUSTNESS STREAMS"
    )

    print(
        "=" * 90
    )


    for seed in AR_SEEDS:

        (
            calibration,
            test,
        ) = generate_stationary_ar1(
            seed
        )


        result = aggregate_and_normalize(
            np.abs(
                calibration
            ),
            np.abs(
                test
            ),
        )


        cache[
            seed
        ] = result


        print(
            f"prepared AR seed={seed}"
        )


    return cache


# =====================================================================
# Detector execution
# =====================================================================

def run_detector(
    detector,
    config,
    stream,
):

    entry = DETECTOR_REGISTRY[
        detector
    ]


    return entry[
        "func"
    ](
        stream,
        **config,
    )


# =====================================================================
# Rule 1
# =====================================================================

def evaluate_no_drift_gate(
    detector,
    config,
    synthetic_cache,
):
    """
    STRICT RULE 1.

    Every official synthetic none seed must produce zero detections.

    Stop immediately at the first violation.
    """

    rows = []


    for seed in SYNTHETIC_SEEDS:

        item = synthetic_cache[
            (
                "none",
                seed,
            )
        ]


        detected_blocks = run_detector(
            detector,
            config,
            item[
                "stream"
            ],
        )


        detected_original = (
            block_indices_to_original(
                detected_blocks,
                item[
                    "test_usable"
                ],
            )
        )


        count = len(
            detected_original
        )


        rows.append(
            {
                "source":
                    "synthetic",

                "drift_type":
                    "none",

                "seed":
                    seed,

                "detections":
                    count,

                "false_alarms_per_10000":
                    (
                        count
                        /
                        item[
                            "test_usable"
                        ]
                        *
                        10_000
                    ),
            }
        )


        # ---------------------------------------------------------
        # Strict early rejection
        # ---------------------------------------------------------

        if count != 0:

            return (
                False,
                rows,
            )


    return (
        True,
        rows,
    )


# =====================================================================
# Positive synthetic scenarios
# =====================================================================

def evaluate_positive_scenarios(
    detector,
    config,
    synthetic_cache,
):

    rows = []


    for drift_type in POSITIVE_DRIFT_TYPES:

        for seed in SYNTHETIC_SEEDS:

            item = synthetic_cache[
                (
                    drift_type,
                    seed,
                )
            ]


            detected_blocks = run_detector(
                detector,
                config,
                item[
                    "stream"
                ],
            )


            detected_original = (
                block_indices_to_original(
                    detected_blocks,
                    item[
                        "test_usable"
                    ],
                )
            )


            metrics = evaluate_detections(
                detected_changepoints=(
                    detected_original
                ),
                true_changepoints=item[
                    "truth"
                ],
                n_observations=item[
                    "test_usable"
                ],
                drift_type=drift_type,
                tolerance=TOLERANCE,
            )


            delay = metrics[
                "detection_delay"
            ]


            fa_per_10k = float(
                metrics[
                    "false_alarms_per_10000"
                ]
            )


            fa_per_year = (
                fa_per_10k
                *
                OBSERVATIONS_PER_YEAR
                /
                10_000
            )


            rows.append(
                {
                    "source":
                        "synthetic",

                    "drift_type":
                        drift_type,

                    "seed":
                        seed,

                    "detections":
                        len(
                            detected_original
                        ),

                    "missed_detections":
                        int(
                            metrics[
                                "missed_detections"
                            ]
                        ),

                    "detection_delay_observations":
                        delay,

                    "detection_delay_days":
                        (
                            float(
                                delay
                            )
                            /
                            OBSERVATIONS_PER_DAY
                            if np.isfinite(
                                delay
                            )
                            else np.nan
                        ),

                    "false_alarms_per_10000":
                        fa_per_10k,

                    "false_alarms_per_year":
                        fa_per_year,
                }
            )


    return rows


# =====================================================================
# AR robustness diagnostic
# =====================================================================

def evaluate_ar_robustness(
    detector,
    config,
    ar_cache,
):

    rows = []


    for seed in AR_SEEDS:

        item = ar_cache[
            seed
        ]


        detected = run_detector(
            detector,
            config,
            item[
                "stream"
            ],
        )


        count = len(
            detected
        )


        fa_per_year = (
            count
            /
            item[
                "test_usable"
            ]
            *
            OBSERVATIONS_PER_YEAR
        )


        rows.append(
            {
                "source":
                    "stationary_ar1",

                "drift_type":
                    "none_ar1",

                "seed":
                    seed,

                "detections":
                    count,

                "missed_detections":
                    0,

                "detection_delay_observations":
                    np.nan,

                "detection_delay_days":
                    np.nan,

                "false_alarms_per_10000":
                    (
                        count
                        /
                        item[
                            "test_usable"
                        ]
                        *
                        10_000
                    ),

                "false_alarms_per_year":
                    fa_per_year,
            }
        )


    return rows


# =====================================================================
# Candidate evaluation
# =====================================================================

def evaluate_candidate(
    detector,
    config,
    synthetic_cache,
    ar_cache,
):

    cid = config_id(
        detector,
        config,
    )


    # -------------------------------------------------------------
    # RULE 1 FIRST
    # -------------------------------------------------------------

    (
        passes_none,
        none_rows,
    ) = evaluate_no_drift_gate(
        detector,
        config,
        synthetic_cache,
    )


    if not passes_none:

        return {
            "config_id":
                cid,

            "detector":
                detector,

            "config":
                config_json(
                    config
                ),

            "passes_no_drift":
                False,

            "rejected_rows":
                none_rows,

            "run_rows":
                [],
        }


    # -------------------------------------------------------------
    # Positive scenarios
    # -------------------------------------------------------------

    positive_rows = (
        evaluate_positive_scenarios(
            detector,
            config,
            synthetic_cache,
        )
    )


    # -------------------------------------------------------------
    # AR stress test only AFTER official benchmark passes
    # -------------------------------------------------------------

    ar_rows = evaluate_ar_robustness(
        detector,
        config,
        ar_cache,
    )


    run_rows = (
        none_rows
        +
        positive_rows
        +
        ar_rows
    )


    for row in run_rows:

        row[
            "config_id"
        ] = cid

        row[
            "detector"
        ] = detector

        row[
            "config"
        ] = config_json(
            config
        )

        row[
            "resolution"
        ] = RESOLUTION

        row[
            "block_size"
        ] = BLOCK_SIZE


    return {
        "config_id":
            cid,

        "detector":
            detector,

        "config":
            config_json(
                config
            ),

        "passes_no_drift":
            True,

        "rejected_rows":
            [],

        "run_rows":
            run_rows,
    }


# =====================================================================
# Safe statistics
# =====================================================================

def safe_mean(
    series,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()


    if values.empty:

        return np.nan


    return float(
        values.mean()
    )


# =====================================================================
# Summarize surviving candidate
# =====================================================================

def summarize_candidate(
    group,
):

    first = group.iloc[
        0
    ]


    synthetic = group[
        group[
            "source"
        ] == "synthetic"
    ]


    positive = synthetic[
        synthetic[
            "drift_type"
        ].isin(
            POSITIVE_DRIFT_TYPES
        )
    ]


    ar = group[
        group[
            "source"
        ] == "stationary_ar1"
    ]


    row = {
        "resolution":
            RESOLUTION,

        "detector":
            first[
                "detector"
            ],

        "config_id":
            first[
                "config_id"
            ],

        "config":
            first[
                "config"
            ],

        "passes_no_drift":
            True,

        "positive_missed_total":
            int(
                positive[
                    "missed_detections"
                ].sum()
            ),

        "positive_missed_mean":
            float(
                positive[
                    "missed_detections"
                ].mean()
            ),

        "positive_delay_days_mean":
            safe_mean(
                positive[
                    "detection_delay_days"
                ]
            ),

        "positive_fa_per_year_mean":
            float(
                positive[
                    "false_alarms_per_year"
                ].mean()
            ),

        "ar1_fa_per_year_mean":
            float(
                ar[
                    "false_alarms_per_year"
                ].mean()
            ),

        "ar1_alarm_mean":
            float(
                ar[
                    "detections"
                ].mean()
            ),
    }


    # -------------------------------------------------------------
    # Scenario-specific results
    # -------------------------------------------------------------

    scenario_fa_values = []


    for drift_type in POSITIVE_DRIFT_TYPES:

        subset = positive[
            positive[
                "drift_type"
            ] == drift_type
        ]


        missed_mean = float(
            subset[
                "missed_detections"
            ].mean()
        )


        fa_per_year = float(
            subset[
                "false_alarms_per_year"
            ].mean()
        )


        delay_days = safe_mean(
            subset[
                "detection_delay_days"
            ]
        )


        row[
            f"{drift_type}_missed_mean"
        ] = missed_mean


        row[
            f"{drift_type}_fa_per_year"
        ] = fa_per_year


        row[
            f"{drift_type}_delay_days"
        ] = delay_days


        scenario_fa_values.append(
            fa_per_year
        )


    # -------------------------------------------------------------
    # Rule 2 representation
    # -------------------------------------------------------------

    worst_fa = float(
        max(
            scenario_fa_values
        )
    )


    row[
        "worst_positive_fa_per_year"
    ] = worst_fa


    # Zero penalty for anything <= target.
    #
    # Slightly above 2 gets only a slight penalty.
    row[
        "fa_excess_above_target"
    ] = max(
        0.0,
        worst_fa
        -
        TARGET_FALSE_ALARMS_PER_YEAR,
    )


    return row


# =====================================================================
# Build summary
# =====================================================================

def build_summary(
    runs,
):

    rows = []


    for _, group in runs.groupby(
        [
            "detector",
            "config_id",
        ],
        sort=False,
    ):

        rows.append(
            summarize_candidate(
                group
            )
        )


    summary = pd.DataFrame(
        rows
    )


    # -------------------------------------------------------------
    # Best achievable coverage per detector
    # -------------------------------------------------------------

    summary[
        "minimum_missed_for_detector"
    ] = (
        summary
        .groupby(
            "detector"
        )[
            "positive_missed_total"
        ]
        .transform(
            "min"
        )
    )


    summary[
        "best_coverage_tier"
    ] = (
        summary[
            "positive_missed_total"
        ]
        ==
        summary[
            "minimum_missed_for_detector"
        ]
    )


    return summary


# =====================================================================
# Ranking
# =====================================================================

def rank_candidates(
    summary,
):
    """
    Ranking philosophy:

    Rule 1 already passed for every row.

    Then:

        1. best achievable drift coverage
        2. lowest excess above ~2 FA/year target
        3. lower actual worst-case FA/year
        4. lower delay
        5. lower AR(1) diagnostic FA

    A candidate slightly above 2/year is NOT rejected.
    """

    ranked_frames = []


    for detector, group in summary.groupby(
        "detector",
        sort=True,
    ):

        ranked = (
            group
            .sort_values(
                [
                    "best_coverage_tier",
                    "fa_excess_above_target",
                    "worst_positive_fa_per_year",
                    "positive_delay_days_mean",
                    "ar1_fa_per_year_mean",
                    "positive_missed_total",
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            )
            .copy()
        )


        ranked[
            "rank"
        ] = np.arange(
            1,
            len(
                ranked
            )
            + 1
        )


        ranked_frames.append(
            ranked
        )


    return pd.concat(
        ranked_frames,
        ignore_index=True,
    )


# =====================================================================
# Console output
# =====================================================================

def print_search_space():

    print()
    print(
        "=" * 100
    )

    print(
        "FOCUSED SEARCH SPACE"
    )

    print(
        "=" * 100
    )


    total = 0


    for detector, candidates in CANDIDATES.items():

        count = len(
            candidates
        )


        total += count


        print(
            f"{detector:<15}: "
            f"{count} candidates"
        )


    print(
        "-" * 35
    )


    print(
        f"{'TOTAL':<15}: "
        f"{total} candidates"
    )


def print_ranked(
    ranked,
):

    print()
    print(
        "=" * 160
    )

    print(
        "RANKED SURVIVING CANDIDATES"
    )

    print(
        "=" * 160
    )


    columns = [
        "detector",
        "rank",
        "best_coverage_tier",
        "positive_missed_total",
        "worst_positive_fa_per_year",
        "fa_excess_above_target",
        "positive_delay_days_mean",
        "sudden_missed_mean",
        "sudden_fa_per_year",
        "sudden_delay_days",
        "gradual_missed_mean",
        "gradual_fa_per_year",
        "gradual_delay_days",
        "recurring_missed_mean",
        "recurring_fa_per_year",
        "recurring_delay_days",
        "ar1_fa_per_year_mean",
        "config",
    ]


    for detector in ranked[
        "detector"
    ].unique():

        print()
        print(
            "-" * 160
        )

        print(
            detector.upper()
        )

        print(
            "-" * 160
        )


        subset = ranked[
            ranked[
                "detector"
            ] == detector
        ]


        print(
            subset[
                columns
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )


# =====================================================================
# Main
# =====================================================================

def main():

    print(
        "=" * 100
    )

    print(
        "TUNING 10 — FOCUSED 3-HOUR DETECTOR TUNING"
    )

    print(
        "=" * 100
    )


    print()
    print(
        "Resolution:"
    )

    print(
        "  3-hour non-overlapping MAE"
    )


    print()
    print(
        "Rule 1:"
    )

    print(
        "  STRICT — synthetic none must have exactly "
        "0 detections across all 5 seeds."
    )


    print()
    print(
        "Rule 2:"
    )

    print(
        "  TARGET — approximately <= 2 false alarms/year "
        "is preferred."
    )

    print(
        "  Slightly above 2/year is NOT automatically rejected."
    )


    print()
    print(
        f"2 FA/year corresponds to approximately "
        f"{TARGET_FALSE_ALARMS_PER_10K:.4f} "
        f"FA/10k original observations."
    )


    print()
    print(
        "AR(1) is secondary robustness evidence only."
    )


    print_search_space()


    # -----------------------------------------------------------------
    # Precompute streams
    # -----------------------------------------------------------------

    synthetic_cache = (
        build_synthetic_cache()
    )


    ar_cache = (
        build_ar_cache()
    )


    # -----------------------------------------------------------------
    # Candidate search
    # -----------------------------------------------------------------

    all_run_rows = []

    rejected_rows = []


    print()
    print(
        "=" * 100
    )

    print(
        "RUNNING FOCUSED PARAMETER SEARCH"
    )

    print(
        "=" * 100
    )


    for detector, candidates in CANDIDATES.items():

        print()
        print(
            "#" * 100
        )

        print(
            f"{detector.upper()} — "
            f"{len(candidates)} candidates"
        )

        print(
            "#" * 100
        )


        for number, config in enumerate(
            candidates,
            start=1,
        ):

            result = evaluate_candidate(
                detector=detector,
                config=config,
                synthetic_cache=synthetic_cache,
                ar_cache=ar_cache,
            )


            if not result[
                "passes_no_drift"
            ]:

                first_failure = result[
                    "rejected_rows"
                ][-1]


                rejected_rows.append(
                    {
                        "detector":
                            detector,

                        "config_id":
                            result[
                                "config_id"
                            ],

                        "config":
                            result[
                                "config"
                            ],

                        "failed_seed":
                            first_failure[
                                "seed"
                            ],

                        "none_detections":
                            first_failure[
                                "detections"
                            ],
                    }
                )


                print(
                    f"  {number:>2}/{len(candidates)} "
                    f"REJECTED Rule 1 "
                    f"(none seed={first_failure['seed']}, "
                    f"detections={first_failure['detections']})"
                )


                continue


            all_run_rows.extend(
                result[
                    "run_rows"
                ]
            )


            print(
                f"  {number:>2}/{len(candidates)} "
                f"PASSED Rule 1"
            )


    # -----------------------------------------------------------------
    # Convert results
    # -----------------------------------------------------------------

    if not all_run_rows:

        print()
        print(
            "No configuration passed Rule 1."
        )

        print(
            "Do not apply any configuration to AEMO."
        )

        return


    runs = pd.DataFrame(
        all_run_rows
    )


    rejected = pd.DataFrame(
        rejected_rows
    )


    summary = build_summary(
        runs
    )


    ranked = rank_candidates(
        summary
    )


    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    runs_path = (
        OUTPUT_DIR
        / "10_3h_tuning_runs.csv"
    )


    summary_path = (
        OUTPUT_DIR
        / "10_3h_tuning_summary.csv"
    )


    ranked_path = (
        OUTPUT_DIR
        / "10_3h_tuning_ranked.csv"
    )


    rejected_path = (
        OUTPUT_DIR
        / "10_3h_rejected_no_drift.csv"
    )


    runs.to_csv(
        runs_path,
        index=False,
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    ranked.to_csv(
        ranked_path,
        index=False,
    )


    rejected.to_csv(
        rejected_path,
        index=False,
    )


    # -----------------------------------------------------------------
    # Console results
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "RULE 1 SUMMARY"
    )

    print(
        "=" * 100
    )


    for detector in CANDIDATES:

        tested = len(
            CANDIDATES[
                detector
            ]
        )


        rejected_count = (
            int(
                (
                    rejected[
                        "detector"
                    ]
                    == detector
                ).sum()
            )
            if not rejected.empty
            else 0
        )


        passed = (
            tested
            -
            rejected_count
        )


        print(
            f"{detector:<15}: "
            f"{passed}/{tested} passed "
            "zero-drift gate"
        )


    print_ranked(
        ranked
    )


    # -----------------------------------------------------------------
    # Best current candidate per detector
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 120
    )

    print(
        "TOP CURRENT CANDIDATE PER DETECTOR"
    )

    print(
        "=" * 120
    )


    top = (
        ranked[
            ranked[
                "rank"
            ] == 1
        ]
    )


    columns = [
        "detector",
        "positive_missed_total",
        "worst_positive_fa_per_year",
        "fa_excess_above_target",
        "positive_delay_days_mean",
        "ar1_fa_per_year_mean",
        "config",
    ]


    print(
        top[
            columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


    # -----------------------------------------------------------------
    # Explanation
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 100
    )


    print(
        """
Rule 1 is strict.

Any configuration appearing in the ranked table has already produced:

    0 detections

for every official synthetic none seed.


Rule 2 is intentionally NOT strict.

Look at:

    worst_positive_fa_per_year

and:

    fa_excess_above_target

Examples:

    worst FA/year = 1.7
        excess = 0
        comfortably meets target

    worst FA/year = 2.0
        excess = 0
        exactly around target

    worst FA/year = 2.3
        excess = 0.3
        still close to target

    worst FA/year = 3.0
        excess = 1.0
        less desirable, but not automatically rejected


Do NOT select a detector only because its FA rate is lowest.

A detector producing no alarms at all could obtain excellent FA numbers
while missing the actual drifts.

That is why the ranking first identifies the best achievable detection
coverage and then compares false-alarm behaviour and delay.


AR(1)
-----
The AR(1) column is a robustness diagnostic only.

It checks whether the configuration remains highly vulnerable to the
serial-dependence failure mode identified in Diagnosis 08.

It is NOT part of the official zero-drift rule.


Next
----
If good 3h configurations exist:

    freeze the selected 3h representation + detector configurations
    and validate them once more on synthetic data.

Only after that should the frozen pipeline be applied to AEMO.

If no reasonable 3h configuration exists:

    then repeat this small search for 6h.

Do not expand immediately to hundreds of candidates.
"""
    )


    print()
    print(
        "=" * 100
    )

    print(
        "Saved outputs"
    )

    print(
        "=" * 100
    )


    print(
        runs_path
    )

    print(
        summary_path
    )

    print(
        ranked_path
    )

    print(
        rejected_path
    )


if __name__ == "__main__":
    main()