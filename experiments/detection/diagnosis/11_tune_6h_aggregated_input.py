"""
11_tune_6h_aggregated_input.py

Focused tuning for 6-hour non-overlapping MAE detector input.
======================================================================

Purpose
-------
Diagnosis 09 showed that 6-hour MAE substantially reduces the serial
dependence responsible for excessive detector alarms.

Diagnosis 10 tuned 3-hour MAE.

Best 3h reference result:
    Page-Hinkley
        synthetic none: 0 detections
        positive missed total: 0
        worst FA/year: ~2.34
        mean delay: ~5.29 days
        stationary AR(1) FA/year: ~4.67

The purpose of this experiment is to determine whether 6-hour MAE can:

    - preserve zero alarms under synthetic no-drift,
    - preserve detection of sudden/gradual/recurring drift,
    - remain around the project's ~2 FA/year target,
    - improve robustness to strong serial dependence.


======================================================================
PROJECT RULES
======================================================================

RULE 1 — STRICT
---------------
Synthetic "none":

    EXACTLY 0 detections

across ALL five official seeds.

Any candidate producing a detection in synthetic "none" is rejected
immediately.


RULE 2 — SOFT TARGET
--------------------
For each positive drift type:

    sudden
    gradual
    recurring

we prefer approximately:

    <= 2 false alarms / year

but this is NOT an absolute cutoff.

Slightly above 2/year may be accepted if detection coverage and delay
are clearly better.


======================================================================
RESOLUTION
======================================================================

6-hour non-overlapping MAE:

    block_size = 12

because:

    12 x 30 minutes = 6 hours


======================================================================
AR(1) ROBUSTNESS
======================================================================

Stationary AR(1), phi=0.98, remains a SECONDARY diagnostic.

It is not an official synthetic scenario.

It does not replace the project's synthetic "none" scenario.

Its purpose is only to check the specific autocorrelation failure mode
identified in Diagnosis 08.


======================================================================
KSWIN
======================================================================

Original KSWIN at 30-minute input:

    window_size = 140
    stat_size   = 30

Physical window:

    140 x 0.5h = 70 hours

At 6-hour resolution, exactly preserving 70 hours would require only
~12 detector observations.

That is too small for the statistical sample sizes needed by KSWIN,
especially with very small alpha.

Therefore this experiment does NOT force exact physical equivalence.

Instead it tests compact but statistically usable windows:

    16 samples = 96 hours
    20 samples = 120 hours
    24 samples = 144 hours
    28 samples = 168 hours

with stat_size:

    8, 10, 12

depending on window size.

The goal is a practical statistical trade-off, not exact preservation
of the old 70-hour window.


======================================================================
OUTPUTS
======================================================================

results/tables/detection/diagnostics/

    11_6h_tuning_runs.csv
    11_6h_tuning_summary.csv
    11_6h_tuning_ranked.csv
    11_6h_rejected_no_drift.csv


Run
---
python experiments/detection/11_tune_6h_aggregated_input.py
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

# 12 half-hour observations = 6 hours.
BLOCK_SIZE = 12

RESOLUTION = "6h_MAE"


# =====================================================================
# Original time units
# =====================================================================

OBSERVATIONS_PER_DAY = 48

OBSERVATIONS_PER_YEAR = (
    48
    * 365
)


# =====================================================================
# False-alarm target
# =====================================================================

TARGET_FALSE_ALARMS_PER_YEAR = 2.0

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


# 7 days in original half-hour observations.
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

AR_PHI = 0.98

# Five seeds only because Diagnosis 08 already established the effect
# using a larger experiment.
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
# Focused candidate sets
# =====================================================================

# ---------------------------------------------------------------------
# ADWIN
# ---------------------------------------------------------------------

ADWIN_CANDIDATES = [
    {
        "delta": 1e-4,
    },
    {
        "delta": 5e-4,
    },
    {
        "delta": 1e-3,
    },
    {
        "delta": 2e-3,
    },
    {
        "delta": 5e-3,
    },
]


# ---------------------------------------------------------------------
# Page-Hinkley
# ---------------------------------------------------------------------
#
# Original:
#     min_instances = 30 x 30 min
#                   = 15 hours
#
# At 6h:
#     3 observations = 18 hours
#
# Therefore min_instances=3 is the closest practical value.
#
# min_instances=5 gives a more persistent 30-hour requirement.
# ---------------------------------------------------------------------

PAGE_HINKLEY_CANDIDATES = [

    # ~18-hour minimum
    {
        "min_instances": 3,
        "delta": 0.005,
        "threshold": 25,
        "alpha": 0.98,
        "mode": "up",
    },

    {
        "min_instances": 3,
        "delta": 0.005,
        "threshold": 50,
        "alpha": 0.98,
        "mode": "up",
    },

    {
        "min_instances": 3,
        "delta": 0.005,
        "threshold": 75,
        "alpha": 0.98,
        "mode": "up",
    },

    {
        "min_instances": 3,
        "delta": 0.005,
        "threshold": 100,
        "alpha": 0.98,
        "mode": "up",
    },

    {
        "min_instances": 3,
        "delta": 0.005,
        "threshold": 125,
        "alpha": 0.98,
        "mode": "up",
    },


    # ~30-hour minimum
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
        "threshold": 50,
        "alpha": 0.98,
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
        "threshold": 100,
        "alpha": 0.98,
        "mode": "up",
    },

    {
        "min_instances": 5,
        "delta": 0.005,
        "threshold": 125,
        "alpha": 0.98,
        "mode": "up",
    },
]


# ---------------------------------------------------------------------
# KSWIN
# ---------------------------------------------------------------------
#
# We deliberately do NOT use stat_size=3-6.
#
# Diagnosis 10 showed that very small stat_size can make tiny alpha
# values effectively incapable of triggering.
#
# These candidates use statistically more meaningful recent samples.
# ---------------------------------------------------------------------

KSWIN_CANDIDATES = [

    # =============================================================
    # 16 x 6h = 96-hour window
    # =============================================================

    {
        "alpha": 0.005,
        "window_size": 16,
        "stat_size": 8,
        "seed": 12,
    },

    {
        "alpha": 0.001,
        "window_size": 16,
        "stat_size": 8,
        "seed": 12,
    },


    # =============================================================
    # 20 x 6h = 120-hour window
    # =============================================================

    {
        "alpha": 0.005,
        "window_size": 20,
        "stat_size": 8,
        "seed": 12,
    },

    {
        "alpha": 0.001,
        "window_size": 20,
        "stat_size": 8,
        "seed": 12,
    },


    # =============================================================
    # 24 x 6h = 144-hour window
    # =============================================================

    {
        "alpha": 0.001,
        "window_size": 24,
        "stat_size": 10,
        "seed": 12,
    },

    {
        "alpha": 0.0005,
        "window_size": 24,
        "stat_size": 10,
        "seed": 12,
    },


    # =============================================================
    # 28 x 6h = 168-hour window
    # =============================================================

    {
        "alpha": 5e-5,
        "window_size": 28,
        "stat_size": 12,
        "seed": 12,
    },

    {
        "alpha": 5e-6,
        "window_size": 28,
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
# Helpers
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
# 6-hour MAE aggregation
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
            "Too few complete 6-hour MAE blocks."
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
            "6-hour MAE blocks",
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
        or calibration_std <= 0
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
                "normalized 6h detector stream",
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
# Map detector block index back to original half-hour index
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
        "PRECOMPUTING 6-HOUR SYNTHETIC STREAMS"
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
# Stationary AR(1)
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
        loc=0.0,
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
        "PRECOMPUTING 6-HOUR AR(1) ROBUSTNESS STREAMS"
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

    return DETECTOR_REGISTRY[
        detector
    ][
        "func"
    ](
        stream,
        **config,
    )


# =====================================================================
# RULE 1 — synthetic none
# =====================================================================

def evaluate_no_drift_gate(
    detector,
    config,
    synthetic_cache,
):

    rows = []

    for seed in SYNTHETIC_SEEDS:

        item = synthetic_cache[
            (
                "none",
                seed,
            )
        ]

        detections = run_detector(
            detector,
            config,
            item[
                "stream"
            ],
        )

        mapped = block_indices_to_original(
            detections,
            item[
                "test_usable"
            ],
        )

        count = len(
            mapped
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

                "false_alarms_per_year":
                    (
                        count
                        /
                        item[
                            "test_usable"
                        ]
                        *
                        OBSERVATIONS_PER_YEAR
                    ),

                "missed_detections":
                    0,

                "detection_delay_observations":
                    np.nan,

                "detection_delay_days":
                    np.nan,
            }
        )

        # Strict Rule 1.
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

            detections = run_detector(
                detector,
                config,
                item[
                    "stream"
                ],
            )

            mapped = block_indices_to_original(
                detections,
                item[
                    "test_usable"
                ],
            )

            metrics = evaluate_detections(
                detected_changepoints=mapped,
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
                            mapped
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
# AR(1) secondary robustness check
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

        detections = run_detector(
            detector,
            config,
            item[
                "stream"
            ],
        )

        count = len(
            detections
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
                    (
                        count
                        /
                        item[
                            "test_usable"
                        ]
                        *
                        OBSERVATIONS_PER_YEAR
                    ),
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

    positive_rows = evaluate_positive_scenarios(
        detector,
        config,
        synthetic_cache,
    )

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
# Statistics
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
# Summarize candidate
# =====================================================================

def summarize_candidate(
    group,
):

    first = group.iloc[
        0
    ]

    positive = group[
        (
            group[
                "source"
            ] == "synthetic"
        )
        &
        (
            group[
                "drift_type"
            ].isin(
                POSITIVE_DRIFT_TYPES
            )
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

    worst_fa = float(
        max(
            scenario_fa_values
        )
    )

    row[
        "worst_positive_fa_per_year"
    ] = worst_fa

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

    frames = []

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

        frames.append(
            ranked
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# =====================================================================
# Console
# =====================================================================

def print_search_space():

    print()
    print(
        "=" * 90
    )
    print(
        "FOCUSED 6-HOUR SEARCH SPACE"
    )
    print(
        "=" * 90
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
        "=" * 170
    )
    print(
        "RANKED SURVIVING 6-HOUR CANDIDATES"
    )
    print(
        "=" * 170
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
            "-" * 170
        )
        print(
            detector.upper()
        )
        print(
            "-" * 170
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
        "TUNING 11 — FOCUSED 6-HOUR DETECTOR TUNING"
    )
    print(
        "=" * 100
    )

    print()
    print(
        "Resolution:"
    )
    print(
        "  6-hour non-overlapping MAE"
    )

    print()
    print(
        "Rule 1:"
    )
    print(
        "  STRICT — synthetic none must produce exactly "
        "0 detections across all five seeds."
    )

    print()
    print(
        "Rule 2:"
    )
    print(
        "  TARGET — approximately <=2 false alarms/year preferred."
    )
    print(
        "  Slightly above 2/year is not automatically rejected."
    )

    print()
    print(
        "AR(1): secondary robustness diagnostic only."
    )

    print()
    print(
        f"Target 2 FA/year = "
        f"{TARGET_FALSE_ALARMS_PER_10K:.4f} "
        "FA/10k original observations."
    )

    print_search_space()

    # -----------------------------------------------------------------
    # Precompute
    # -----------------------------------------------------------------

    synthetic_cache = build_synthetic_cache()

    ar_cache = build_ar_cache()

    # -----------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------

    all_run_rows = []

    rejected_rows = []

    print()
    print(
        "=" * 100
    )
    print(
        "RUNNING FOCUSED 6-HOUR SEARCH"
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

                failure = result[
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
                            failure[
                                "seed"
                            ],

                        "none_detections":
                            failure[
                                "detections"
                            ],
                    }
                )

                print(
                    f"  {number:>2}/{len(candidates)} "
                    f"REJECTED Rule 1 "
                    f"(seed={failure['seed']}, "
                    f"detections={failure['detections']})"
                )

                continue

            all_run_rows.extend(
                result[
                    "run_rows"
                ]
            )

            print(
                f"  {number:>2}/{len(candidates)} "
                "PASSED Rule 1"
            )

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    if not all_run_rows:

        print()
        print(
            "No 6-hour candidate passed Rule 1."
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
        / "11_6h_tuning_runs.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "11_6h_tuning_summary.csv"
    )

    ranked_path = (
        OUTPUT_DIR
        / "11_6h_tuning_ranked.csv"
    )

    rejected_path = (
        OUTPUT_DIR
        / "11_6h_rejected_no_drift.csv"
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
    # Rule 1 summary
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
            f"{passed}/{tested} passed"
        )

    # -----------------------------------------------------------------
    # Ranked table
    # -----------------------------------------------------------------

    print_ranked(
        ranked
    )

    # -----------------------------------------------------------------
    # Top candidate
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 130
    )
    print(
        "TOP 6-HOUR CANDIDATE PER DETECTOR"
    )
    print(
        "=" * 130
    )

    top = ranked[
        ranked[
            "rank"
        ] == 1
    ]

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
    # Comparison reminder
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 100
    )
    print(
        "DECISION RULE AFTER THIS RUN"
    )
    print(
        "=" * 100
    )

    print(
        """
Compare the best 6h result against the 3h benchmark.

The strongest 3h candidate so far was Page-Hinkley:

    none detections          = 0
    positive missed total    = 0
    worst FA/year            ~= 2.34
    mean delay               ~= 5.29 days
    AR(1) FA/year            ~= 4.67

A 6h configuration is particularly attractive if it can:

    1. maintain zero synthetic-none alarms;
    2. maintain zero or very few missed drifts;
    3. remain around 2 FA/year;
    4. materially reduce AR(1) false alarms;
    5. avoid excessive detection delay.

Do not select 6h simply because it produces fewer alarms.

It must preserve useful drift sensitivity.
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