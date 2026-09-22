"""
Synthetic Drift-Detection Evaluation

Evaluates the frozen ADWIN, KSWIN, and Page-Hinkley configurations
from DETECTOR_REGISTRY on synthetic none, sudden, gradual, and
recurring drift scenarios across five seeds.

Data and preprocessing:
    • 30-minute synthetic demand
    • TRAIN: 2 years (35,040 observations)
    • CALIBRATION: 60 days (2,880 observations)
    • TEST: 1 year (17,520 observations)
    • TRAIN-fitted seasonal profile:
      overall mean + smoothed DOY + (DOW × half-hour)
    • Gradient Boosting with calendar features and lags (48, 336)
    • Absolute forecast-error stream
    • Global error z-score fitted on chronological TRAIN Year-2 errors
    • AR(1) fitted on TRAIN Year-2 processed errors only
    • ADWIN/KSWIN: partial AR(1) whitening, lambda = 0.80
    • Page-Hinkley: no whitening

Detection protocol:
    • Processed CALIBRATION errors are used for detector warm-up
    • TEST errors are used for recorded detections
    • Refractory period: 14 days
    • Matching tolerance: 7 days
    • Frozen detector configurations are loaded from DETECTOR_REGISTRY
    • Evaluation is repeated across seeds 1–5
    • Hard constraints:
      zero detections on the no-drift control
      mean false alarms <= 2 per year for each positive drift type

Synthetic drift scenarios:
    • None: no changepoint
    • Sudden: one changepoint at 50% of TEST
    • Gradual: 30-day transition centred at 50% of TEST
    • Recurring: changepoints at 35% and 65% of TEST

Outputs:
    results/tables/detection/synthetic_detection_raw.csv
    results/tables/detection/synthetic_detection_summary.csv
    results/tables/runs.csv
"""

from pathlib import Path
import sys
import json
import hashlib

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

from drift_lab.synthetic_generator import (
    make_feature_series,
)

from drift_lab.detectors import (
    DETECTOR_REGISTRY,
    run_detector_with_protocol,
)

from drift_lab.evaluation import (
    evaluate_detections,
)

from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)

from drift_lab.deseasonalise import (
    deseasonalise_demand_frame,
    ErrorZScorePreprocessor,
)

from drift_lab.run_logger import (
    log_run,
)


# Experiment settings

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
    1
    * OBSERVATIONS_PER_YEAR
)
# 17,520

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
# 55,440


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
            0.50
            * TEST_LENGTH
        )
        - (
            15
            * OBSERVATIONS_PER_DAY
        ),

        TEST_START
        + int(
            0.50
            * TEST_LENGTH
        )
        + (
            15
            * OBSERVATIONS_PER_DAY
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

TRAIN_REFERENCE_FIT_END = (
    OBSERVATIONS_PER_YEAR
)


# Detection

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

FA_PER_10K_TO_PER_YEAR = (
    OBSERVATIONS_PER_YEAR
    / 10_000.0
)

MAX_FALSE_ALARMS_PER_YEAR = 2.0


# Partial AR(1) configuration

PARTIAL_AR1_LAMBDA = 0.80

WHITENED_DETECTORS = (
    "adwin",
    "kswin",
)


# Registered detectors

REQUIRED_DETECTORS = (
    "adwin",
    "kswin",
    "page_hinkley",
)


# Run metadata

GROUP = "detection"

REGION = "synthetic"

REGIME = "full_test"

SPLIT_ID = (
    "synthetic_35040_2880_17520_"
    "demand_deseasonalised_"
    "trainerror_zscore_"
    "warmup_calibration_"
    "refractory672_tol336_"
    "registry_V2"
)


# Output paths

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "runs.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
)

RAW_RESULTS_PATH = (
    OUTPUT_DIR
    / "synthetic_detection_raw.csv"
)

SUMMARY_RESULTS_PATH = (
    OUTPUT_DIR
    / "synthetic_detection_summary.csv"
)


# Helpers

def config_id(
    config,
):
    """Create a reproducible detector configuration ID."""

    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()[:10]

    readable = ",".join(
        f"{key}={value}"
        for key, value
        in sorted(
            config.items()
        )
    )

    return (
        f"{readable}"
        f"|sha256:{digest}"
    )


def false_alarms_per_year(
    false_alarms_per_10k,
):
    """Convert false alarms per 10k observations to FA/year."""

    return (
        float(
            false_alarms_per_10k
        )
        * FA_PER_10K_TO_PER_YEAR
    )


# Partial AR(1) whitening

class PartialAR1Whitener:
    """
    Partial AR(1) preprocessing.

    Fit on TRAIN Year-2 standardised absolute forecast errors:

        z_t = intercept + phi * z_(t-1) + e_t

    Partial whitening:

        r_t =
            z_t
            - lambda * (
                intercept
                + phi * z_(t-1)
            )

    This transformation is used only for ADWIN and KSWIN.
    """

    def __init__(self):

        self.intercept_ = None
        self.phi_ = None

    def fit(
        self,
        series,
    ):
        """
        Fit AR(1) using ordinary least squares.
        """

        series = np.asarray(
            series,
            dtype=float,
        )

        if len(series) < 20:

            raise ValueError(
                "TRAIN reference stream is too short "
                "for AR(1) fitting."
            )

        if not np.isfinite(
            series
        ).all():

            raise ValueError(
                "TRAIN AR(1) reference contains "
                "NaN or infinite values."
            )

        previous = (
            series[:-1]
        )

        current = (
            series[1:]
        )

        design = np.column_stack(
            [
                np.ones(
                    len(previous)
                ),
                previous,
            ]
        )

        coefficients, _, _, _ = (
            np.linalg.lstsq(
                design,
                current,
                rcond=None,
            )
        )

        self.intercept_ = float(
            coefficients[0]
        )

        self.phi_ = float(
            coefficients[1]
        )

        return self

    def transform(
        self,
        series,
        strength,
        history_tail,
    ):
        """
        Apply causal partial AR(1) whitening.
        """

        if (
            self.intercept_ is None
            or self.phi_ is None
        ):

            raise RuntimeError(
                "AR(1) whitener must be fitted first."
            )

        series = np.asarray(
            series,
            dtype=float,
        )

        history_tail = np.asarray(
            history_tail,
            dtype=float,
        )

        strength = float(
            strength
        )

        if not (
            0.0
            <= strength
            <= 1.0
        ):

            raise ValueError(
                "Partial-whitening lambda must be "
                "between 0 and 1."
            )

        if len(series) == 0:

            raise ValueError(
                "Cannot whiten an empty stream."
            )

        if len(history_tail) == 0:

            raise ValueError(
                "AR history_tail cannot be empty."
            )

        previous = np.concatenate(
            [
                history_tail[-1:],
                series[:-1],
            ]
        )

        predicted_component = (
            self.intercept_
            + self.phi_
            * previous
        )

        transformed = (
            series
            - strength
            * predicted_component
        )

        if not np.isfinite(
            transformed
        ).all():

            raise RuntimeError(
                "Partially whitened stream contains "
                "NaN or infinite values."
            )

        if len(transformed) != len(series):

            raise RuntimeError(
                "Partial whitening changed stream length."
            )

        return transformed


# Registry validation

def validate_detector_registry():
    """Check that all required detector configurations are available."""

    missing = [
        name
        for name in REQUIRED_DETECTORS
        if name not in DETECTOR_REGISTRY
    ]

    if missing:

        raise RuntimeError(
            "DETECTOR_REGISTRY is missing required detectors: "
            f"{missing}"
        )

    for detector_name in REQUIRED_DETECTORS:

        entry = (
            DETECTOR_REGISTRY[
                detector_name
            ]
        )

        if "config" not in entry:

            raise RuntimeError(
                f"DETECTOR_REGISTRY['{detector_name}'] "
                "does not contain a config."
            )

        if not isinstance(
            entry["config"],
            dict,
        ):

            raise RuntimeError(
                f"DETECTOR_REGISTRY['{detector_name}']['config'] "
                "must be a dictionary."
            )


# Ground truth

def to_test_relative(
    full_changepoints,
    drift_type,
    seed,
):
    """Validate changepoints and convert them to TEST-relative indices."""

    if full_changepoints is None:
        full_changepoints = []

    full_changepoints = [
        int(cp)
        for cp in full_changepoints
    ]

    if drift_type == "none":

        if full_changepoints:

            raise ValueError(
                f"NONE seed={seed} unexpectedly "
                f"contains changepoints: "
                f"{full_changepoints}"
            )

        return []

    if not full_changepoints:

        raise ValueError(
            f"{drift_type} seed={seed} "
            "contains no changepoints."
        )

    relative = []

    for cp in full_changepoints:

        if not (
            TEST_START
            <= cp
            < N_OBSERVATIONS
        ):

            raise ValueError(
                f"Ground-truth changepoint "
                f"{cp} for {drift_type}, "
                f"seed={seed} lies outside TEST."
            )

        relative.append(
            int(
                cp
                - TEST_START
            )
        )

    return relative


# Detector-stream construction

def build_detector_streams(
    data,
):
    """
    Build processed CALIBRATION and TEST forecast-error streams.

    Demand deseasonalisation is fitted on complete TRAIN.
    Error standardisation is fitted on chronological TRAIN Year-2
    absolute forecast errors.

    Partial AR(1) parameters are fitted on the processed
    chronological TRAIN Year-2 error stream.

    ADWIN and KSWIN use partially whitened CALIBRATION and TEST
    streams. Page-Hinkley uses the unwhitened globally
    standardised streams.
    """

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

        raise RuntimeError(
            f"Unexpected TRAIN length: "
            f"{len(train)}"
        )

    if len(calibration) != CALIBRATION_LENGTH:

        raise RuntimeError(
            f"Unexpected CALIBRATION length: "
            f"{len(calibration)}"
        )

    if len(test) != TEST_LENGTH:

        raise RuntimeError(
            f"Unexpected TEST length: "
            f"{len(test)}"
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

            raise RuntimeError(
                f"{name} deseasonalised demand "
                "contains NaN or infinite values."
            )


    # TRAIN Year 1 / Year 2 reference

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

        raise RuntimeError(
            "TRAIN reference fit period is empty."
        )

    if reference_eval.empty:

        raise RuntimeError(
            "TRAIN reference evaluation period is empty."
        )


    # Temporary TRAIN model

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


    # TRAIN Year-2 forecast errors

    reference_errors = (
        frozen_error_stream(
            reference_model,
            reference_eval,
            history_df=reference_fit,
        )
    )

    if reference_errors.empty:

        raise RuntimeError(
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

        raise RuntimeError(
            "TRAIN reference absolute-error "
            "stream contains NaN or infinite values."
        )


    # TRAIN-only error standardisation

    error_scaler = (
        ErrorZScorePreprocessor()
    )

    error_scaler.fit(
        reference_abs
    )

    reference_processed = (
        error_scaler
        .transform(
            reference_abs
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        reference_processed
    ).all():

        raise RuntimeError(
            "Processed TRAIN Year-2 error reference "
            "contains NaN or infinite values."
        )


    # TRAIN-only AR(1) fitting

    whitener = (
        PartialAR1Whitener()
    )

    whitener.fit(
        reference_processed
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


    # CALIBRATION forecast errors

    calibration_errors = (
        frozen_error_stream(
            model,
            calibration_ds,
            history_df=train_ds,
        )
    )

    if calibration_errors.empty:

        raise RuntimeError(
            "Calibration forecast-error stream is empty."
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


    # TEST forecast errors

    test_errors = (
        frozen_error_stream(
            model,
            test_ds,
            history_df=calibration_ds,
        )
    )

    if test_errors.empty:

        raise RuntimeError(
            "Test forecast-error stream is empty."
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


    # Partial-whiten CALIBRATION

    calibration_whitened = (
        whitener.transform(
            series=calibration_processed,
            strength=PARTIAL_AR1_LAMBDA,
            history_tail=reference_processed,
        )
    )


    # Partial-whiten TEST

    test_whitened = (
        whitener.transform(
            series=test_processed,
            strength=PARTIAL_AR1_LAMBDA,
            history_tail=calibration_processed,
        )
    )


    # Stream validation

    for name, values in (
        (
            "processed CALIBRATION",
            calibration_processed,
        ),
        (
            "processed TEST",
            test_processed,
        ),
        (
            "whitened CALIBRATION",
            calibration_whitened,
        ),
        (
            "whitened TEST",
            test_whitened,
        ),
    ):

        if not np.isfinite(
            values
        ).all():

            raise RuntimeError(
                f"{name} stream "
                "contains NaN or infinite values."
            )

    if (
        len(calibration_processed)
        != len(calibration_ds)
    ):

        raise RuntimeError(
            "CALIBRATION detector stream is misaligned."
        )

    if (
        len(test_processed)
        != len(test_ds)
    ):

        raise RuntimeError(
            "TEST detector stream is misaligned."
        )

    if (
        len(calibration_whitened)
        != len(calibration_ds)
    ):

        raise RuntimeError(
            "Whitened CALIBRATION detector stream "
            "is misaligned."
        )

    if (
        len(test_whitened)
        != len(test_ds)
    ):

        raise RuntimeError(
            "Whitened TEST detector stream "
            "is misaligned."
        )

    return {

        "calibration_processed":
            calibration_processed,

        "test_processed":
            test_processed,

        "calibration_whitened":
            calibration_whitened,

        "test_whitened":
            test_whitened,

        "ar1_intercept":
            whitener.intercept_,

        "ar1_phi":
            whitener.phi_,

        "train_samples":
            len(train),
    }


# Detector construction

def make_registered_detector(
    detector_name,
):
    """Create a River detector from its DETECTOR_REGISTRY config."""

    entry = (
        DETECTOR_REGISTRY[
            detector_name
        ]
    )

    config = (
        entry[
            "config"
        ]
        .copy()
    )

    if detector_name == "adwin":

        detector = (
            ADWIN(
                **config
            )
        )

    elif detector_name == "kswin":

        detector = (
            KSWIN(
                **config
            )
        )

    elif detector_name == "page_hinkley":

        detector = (
            PageHinkley(
                **config
            )
        )

    else:

        raise ValueError(
            "Unsupported registered detector: "
            f"{detector_name}"
        )

    return (
        detector,
        config,
    )


# Detector execution

def run_registered_detector(
    detector_name,
    test_stream,
    warmup_stream,
):
    """
    Warm a registered detector on CALIBRATION and record its
    TEST detections using the 14-day refractory period.
    """

    (
        detector,
        config,
    ) = (
        make_registered_detector(
            detector_name
        )
    )

    changepoints = (
        run_detector_with_protocol(
            test_stream=test_stream,
            detector=detector,
            warmup_stream=warmup_stream,
            refractory_period=REFRACTORY_PERIOD,
        )
    )

    return (
        changepoints,
        config,
    )


# Duplicate-run check

def ensure_no_duplicate_recorded_run():
    """Stop if this experiment is already recorded in runs.csv."""

    path = Path(
        RESULTS_PATH
    )

    if (
        not path.exists()
        or path.stat().st_size == 0
    ):
        return

    existing = pd.read_csv(
        path
    )

    if existing.empty:
        return

    required = {
        "group",
        "seed",
        "split_id",
    }

    if not required.issubset(
        existing.columns
    ):
        return

    duplicate = existing[
        (
            existing[
                "group"
            ]
            == GROUP
        )
        &
        (
            existing[
                "split_id"
            ]
            == SPLIT_ID
        )
        &
        (
            existing[
                "seed"
            ]
            .isin(
                SEEDS
            )
        )
    ]

    if not duplicate.empty:

        raise RuntimeError(
            "Matching synthetic detection results "
            "already exist in "
            f"{RESULTS_PATH}. "
            "Remove those rows before rerunning."
        )


# Shared metric logging

def log_metrics(
    detector_name,
    detector_config,
    drift_type,
    seed,
    metrics,
    train_samples,
):
    """Log detection metrics to the shared runs.csv file."""

    common = {

        "method":
            detector_name,

        "dataset":
            f"synthetic_{drift_type}",

        "region":
            REGION,

        "regime":
            REGIME,

        "train_samples":
            train_samples,

        "group":
            GROUP,

        "seed":
            seed,

        "config_hash":
            config_id(
                detector_config
            ),

        "split_id":
            SPLIT_ID,

        "n_retrains":
            0,

        "wall_clock_s":
            0.0,

        "results_path":
            str(
                RESULTS_PATH
            ),
    }

    metric_values = {

        "detection_delay":
            metrics[
                "detection_delay"
            ],

        "false_alarms_per_10000":
            metrics[
                "false_alarms_per_10000"
            ],

        "missed_detections":
            metrics[
                "missed_detections"
            ],
    }

    for (
        metric_name,
        metric_value,
    ) in metric_values.items():

        if (
            metric_name
            == "detection_delay"
            and np.isnan(
                metric_value
            )
        ):
            continue

        log_run(
            metric_name=metric_name,
            metric_value=metric_value,
            **common,
        )


# Experiment

def main():

    validate_detector_registry()

    ensure_no_duplicate_recorded_run()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print(
        "=" * 80
    )

    print(
        "SYNTHETIC DRIFT-DETECTION EVALUATION"
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
        "Forecaster: Gradient Boosting "
        "calendar+lags (48, 336)"
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
        "Error z-score reference: "
        "chronological TRAIN Year-2 errors"
    )

    print(
        "AR(1) reference: "
        "chronological TRAIN Year-2 "
        "processed errors"
    )

    print(
        "ADWIN/KSWIN whitening: "
        f"partial AR(1), lambda="
        f"{PARTIAL_AR1_LAMBDA:.2f}"
    )

    print(
        "Page-Hinkley whitening: none"
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
        f"Tolerance: "
        f"{TOLERANCE_DAYS} days"
    )

    print(
        "\nFrozen detector configurations:"
    )

    for detector_name in REQUIRED_DETECTORS:

        config = (
            DETECTOR_REGISTRY[
                detector_name
            ][
                "config"
            ]
        )

        print(
            f"  {detector_name}: "
            f"{config}"
        )


    # Run each seed and drift scenario

    for seed in SEEDS:

        for drift_type in DRIFT_TYPES:

            print(
                f"\nRunning "
                f"seed={seed}, "
                f"scenario={drift_type}"
            )

            (
                data,
                full_truth,
            ) = (
                make_feature_series(
                    kind=drift_type,
                    n=N_OBSERVATIONS,
                    seed=seed,
                    changepoints=(
                        SYNTHETIC_CHANGEPOINTS[
                            drift_type
                        ]
                    ),
                )
            )

            if len(data) != N_OBSERVATIONS:

                raise RuntimeError(
                    "Synthetic generator returned "
                    f"{len(data)} observations; "
                    f"expected {N_OBSERVATIONS}."
                )

            truth = (
                to_test_relative(
                    full_changepoints=full_truth,
                    drift_type=drift_type,
                    seed=seed,
                )
            )

            streams = (
                build_detector_streams(
                    data
                )
            )

            train_samples = (
                streams[
                    "train_samples"
                ]
            )

            for detector_name in REQUIRED_DETECTORS:

                if (
                    detector_name
                    in WHITENED_DETECTORS
                ):

                    warmup_stream = (
                        streams[
                            "calibration_whitened"
                        ]
                    )

                    test_stream = (
                        streams[
                            "test_whitened"
                        ]
                    )

                    whitening_type = (
                        "partial_AR1"
                    )

                    whitening_lambda = (
                        PARTIAL_AR1_LAMBDA
                    )

                else:

                    warmup_stream = (
                        streams[
                            "calibration_processed"
                        ]
                    )

                    test_stream = (
                        streams[
                            "test_processed"
                        ]
                    )

                    whitening_type = (
                        "none"
                    )

                    whitening_lambda = 0.0

                (
                    detected,
                    config,
                ) = (
                    run_registered_detector(
                        detector_name=detector_name,
                        test_stream=test_stream,
                        warmup_stream=warmup_stream,
                    )
                )

                metrics = (
                    evaluate_detections(
                        detected_changepoints=detected,
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

                fa_year = (
                    false_alarms_per_year(
                        fa_10k
                    )
                )

                log_metrics(
                    detector_name=detector_name,
                    detector_config=config,
                    drift_type=drift_type,
                    seed=seed,
                    metrics=metrics,
                    train_samples=train_samples,
                )

                rows.append(
                    {

                        "detector":
                            detector_name,

                        "drift_type":
                            drift_type,

                        "seed":
                            seed,

                        "n_detections":
                            len(
                                detected
                            ),

                        "detection_delay":
                            metrics[
                                "detection_delay"
                            ],

                        "false_alarms_per_10000":
                            fa_10k,

                        "false_alarms_per_year":
                            fa_year,

                        "missed_detections":
                            metrics[
                                "missed_detections"
                            ],

                        "config":
                            config_id(
                                config
                            ),

                        "train_samples":
                            train_samples,

                        "warmup_samples":
                            len(
                                warmup_stream
                            ),

                        "test_samples":
                            len(
                                test_stream
                            ),

                        "refractory_period":
                            REFRACTORY_PERIOD,

                        "refractory_days":
                            REFRACTORY_DAYS,

                        "tolerance":
                            TOLERANCE,

                        "tolerance_days":
                            TOLERANCE_DAYS,

                        "seasonal_profile":
                            (
                                "overall_mean+"
                                "smoothed_DOY+"
                                "(DOWxhalf_hour)"
                            ),

                        "seasonal_reference":
                            "complete_TRAIN_only",

                        "detector_input":
                            (
                                "TRAIN_zscored_"
                                "absolute_forecast_error"
                            ),

                        "whitening_type":
                            whitening_type,

                        "whitening_lambda":
                            whitening_lambda,

                        "ar1_intercept":
                            float(
                                streams[
                                    "ar1_intercept"
                                ]
                            ),

                        "ar1_phi":
                            float(
                                streams[
                                    "ar1_phi"
                                ]
                            ),

                        "split_id":
                            SPLIT_ID,
                    }
                )

            print(
                f"Completed "
                f"seed={seed}, "
                f"scenario={drift_type}"
            )


    # Save raw results

    raw = pd.DataFrame(
        rows
    )

    raw.to_csv(
        RAW_RESULTS_PATH,
        index=False,
    )


    # Build and save summary

    summary = (
        raw
        .groupby(
            [
                "detector",
                "drift_type",
                "config",
            ],
            as_index=False,
        )
        .agg(

            detections_mean=(
                "n_detections",
                "mean",
            ),

            detections_sd=(
                "n_detections",
                "std",
            ),

            delay_mean=(
                "detection_delay",
                "mean",
            ),

            delay_sd=(
                "detection_delay",
                "std",
            ),

            false_alarms_per_10k_mean=(
                "false_alarms_per_10000",
                "mean",
            ),

            false_alarms_per_10k_sd=(
                "false_alarms_per_10000",
                "std",
            ),

            false_alarms_per_year_mean=(
                "false_alarms_per_year",
                "mean",
            ),

            false_alarms_per_year_sd=(
                "false_alarms_per_year",
                "std",
            ),

            missed_mean=(
                "missed_detections",
                "mean",
            ),

            missed_sd=(
                "missed_detections",
                "std",
            ),

            seeds=(
                "seed",
                "nunique",
            ),
        )
    )

    summary.to_csv(
        SUMMARY_RESULTS_PATH,
        index=False,
    )


    # Print summary

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SYNTHETIC DETECTION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )


    # No-drift check

    none_summary = (
        raw[
            raw[
                "drift_type"
            ]
            == "none"
        ]
        .groupby(
            "detector",
            as_index=False,
        )
        .agg(

            total_detections=(
                "n_detections",
                "sum",
            ),

            max_single_seed=(
                "n_detections",
                "max",
            ),
        )
    )

    print(
        "\nNO-DRIFT CHECK"
    )

    print(
        none_summary.to_string(
            index=False
        )
    )


    # Positive-drift summary

    positive_summary = (
        raw[
            raw[
                "drift_type"
            ]
            .isin(
                POSITIVE_DRIFT_TYPES
            )
        ]
        .groupby(
            [
                "detector",
                "drift_type",
            ],
            as_index=False,
        )
        .agg(

            fa_per_year_mean=(
                "false_alarms_per_year",
                "mean",
            ),

            fa_per_year_sd=(
                "false_alarms_per_year",
                "std",
            ),

            missed_total=(
                "missed_detections",
                "sum",
            ),

            delay_mean=(
                "detection_delay",
                "mean",
            ),
        )
    )

    print(
        "\nPOSITIVE-DRIFT SUMMARY"
    )

    print(
        positive_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )


    # Hard-constraint check

    print(
        "\nFROZEN CONFIGURATION VALIDATION"
    )

    for detector_name in REQUIRED_DETECTORS:

        detector_rows = (
            raw[
                raw[
                    "detector"
                ]
                == detector_name
            ]
        )

        none_rows = (
            detector_rows[
                detector_rows[
                    "drift_type"
                ]
                == "none"
            ]
        )

        none_total = int(
            none_rows[
                "n_detections"
            ]
            .sum()
        )

        passes_none = (
            none_total
            == 0
        )

        passes_fa = True

        print(
            f"\n{detector_name}"
        )

        print(
            f"  NONE detections = "
            f"{none_total}"
        )

        for drift_type in POSITIVE_DRIFT_TYPES:

            drift_rows = (
                detector_rows[
                    detector_rows[
                        "drift_type"
                    ]
                    == drift_type
                ]
            )

            mean_fa = float(
                drift_rows[
                    "false_alarms_per_year"
                ]
                .mean()
            )

            missed_total = int(
                drift_rows[
                    "missed_detections"
                ]
                .sum()
            )

            drift_passes = (
                mean_fa
                <= MAX_FALSE_ALARMS_PER_YEAR
            )

            if not drift_passes:
                passes_fa = False

            print(
                f"  {drift_type:9s}: "
                f"FA/year={mean_fa:.3f}, "
                f"missed={missed_total}, "
                f"budget="
                f"{'PASS' if drift_passes else 'FAIL'}"
            )

        passes_all = (
            passes_none
            and passes_fa
        )

        print(
            "  Hard constraints = "
            f"{'PASS' if passes_all else 'FAIL'}"
        )


    print(
        "\nSynthetic drift-detection evaluation complete."
    )

    print(
        f"Raw results: "
        f"{RAW_RESULTS_PATH}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_RESULTS_PATH}"
    )

    print(
        f"Shared runs: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()