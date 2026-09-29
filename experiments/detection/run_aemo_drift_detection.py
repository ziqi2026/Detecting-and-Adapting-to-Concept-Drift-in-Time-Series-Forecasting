"""
PGS242-120 — AEMO Drift Detection

Applies the frozen synthetic-calibrated ADWIN, KSWIN, and
Page-Hinkley configurations from DETECTOR_REGISTRY to the
SA1 and NSW1 AEMO forecasting-error streams.

Data and preprocessing:
    • 30-minute AEMO demand
    • Fixed TRAIN / CALIBRATION / TEST split
    • TRAIN-fitted seasonal profile:
      overall mean + smoothed DOY + (DOW × half-hour)
    • Gradient Boosting with calendar features and lags (48, 336)
    • Global error z-score fitted on chronological TRAIN Year-2 errors

Detector-specific preprocessing:
    • ADWIN: partial AR(1) whitening, lambda = 0.80
    • KSWIN: partial AR(1) whitening, lambda = 0.80
    • Page-Hinkley: no whitening
    • AR(1) parameters are fitted on TRAIN Year-2 processed errors

Detection:
    • Processed CALIBRATION errors are used for detector warm-up
    • TEST errors are used for recorded detections
    • Refractory period: 14 days
    • Detector configurations: DETECTOR_REGISTRY

Output:
    results/tables/detection/aemo_detection_changepoints.csv
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

from river.drift import ADWIN, KSWIN, PageHinkley

# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Project imports

from drift_lab.data_loader import load_and_split_aemo
from drift_lab.detector_diagnostics import (
    run_detector_diagnostic,
)
from drift_lab.detectors import (
    DETECTOR_REGISTRY,
    run_detector_with_protocol,
)

from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)

from drift_lab.deseasonalise import (
    deseasonalise_demand_frame,
    ErrorZScorePreprocessor,
)



# Experiment settings

REGIONS = (
    "SA1",
    "NSW1",
)

OBSERVATIONS_PER_DAY = 48
OBSERVATIONS_PER_YEAR = 365 * OBSERVATIONS_PER_DAY

REFRACTORY_DAYS = 14
REFRACTORY_PERIOD = REFRACTORY_DAYS * OBSERVATIONS_PER_DAY

FORECASTER_SEED = 0

FORECASTER_LAGS = (
    48,
    336,
)

TRAIN_REFERENCE_FIT_END = OBSERVATIONS_PER_YEAR

REQUIRED_DETECTORS = (
    "adwin",
    "kswin",
    "page_hinkley",
)


# Partial AR(1)

PARTIAL_AR1_LAMBDA = 0.80

WHITENED_DETECTORS = (
    "adwin",
    "kswin",
)


# Metadata

SEASONAL_PROFILE = (
    "overall_mean+"
    "smoothed_DOY+"
    "(DOWxhalf_hour)"
)

SEASONAL_REFERENCE = "complete_TRAIN_only"

ERROR_ZSCORE_REFERENCE = (
    "chronological_"
    "TRAIN_year2_errors"
)

ERROR_ZSCORE_TYPE = "global"

WARMUP_REFERENCE = (
    "processed CALIBRATION forecast errors"
)

DETECTOR_INPUT = (
    "TRAIN_zscored_"
    "absolute_forecast_error"
)


# Paths

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "aemo_detection_changepoints.csv"
)

DIAGNOSTIC_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "aemo_detection_diagnostics.csv.gz"
)


def validate_detector_registry():
    """Validate the required frozen detector configurations."""

    missing = [
        detector_name
        for detector_name in REQUIRED_DETECTORS
        if detector_name not in DETECTOR_REGISTRY
    ]

    if missing:
        raise RuntimeError(
            "DETECTOR_REGISTRY is missing required detectors: "
            f"{missing}"
        )

    for detector_name in REQUIRED_DETECTORS:

        entry = DETECTOR_REGISTRY[detector_name]

        if "config" not in entry:
            raise RuntimeError(
                f"DETECTOR_REGISTRY['{detector_name}'] "
                "does not contain 'config'."
            )

        if not isinstance(entry["config"], dict):
            raise RuntimeError(
                f"DETECTOR_REGISTRY['{detector_name}']['config'] "
                "must be a dictionary."
            )


def make_registered_detector(detector_name):
    """Create a fresh detector from its frozen registry configuration."""

    config = (
        DETECTOR_REGISTRY[
            detector_name
        ]["config"]
        .copy()
    )

    if detector_name == "adwin":
        detector = ADWIN(**config)

    elif detector_name == "kswin":
        detector = KSWIN(**config)

    elif detector_name == "page_hinkley":
        detector = PageHinkley(**config)

    else:
        raise ValueError(
            "Unsupported registered detector: "
            f"{detector_name}"
        )

    return detector, config


class PartialAR1Whitener:
    """
    Partial AR(1) preprocessing fitted on TRAIN Year-2
    standardised absolute forecast errors.

    Model:
        z_t = intercept + phi * z_(t-1) + e_t

    Transformation:
        r_t = z_t - lambda * (intercept + phi * z_(t-1))
    """

    def __init__(self):
        self.intercept_ = None
        self.phi_ = None

    def fit(self, series):
        """Fit the AR(1) model using ordinary least squares."""

        series = np.asarray(
            series,
            dtype=float,
        )

        if len(series) < 20:
            raise ValueError(
                "TRAIN reference stream is too short "
                "for AR(1) fitting."
            )

        if not np.isfinite(series).all():
            raise ValueError(
                "TRAIN AR(1) reference contains "
                "NaN or infinite values."
            )

        previous = series[:-1]
        current = series[1:]

        design = np.column_stack(
            [
                np.ones(len(previous)),
                previous,
            ]
        )

        coefficients, _, _, _ = np.linalg.lstsq(
            design,
            current,
            rcond=None,
        )

        self.intercept_ = float(coefficients[0])
        self.phi_ = float(coefficients[1])

        return self

    def transform(
        self,
        series,
        strength,
        history_tail,
    ):
        """
        Apply causal partial AR(1) whitening while preserving
        stream length and alignment.
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

        strength = float(strength)

        if not 0.0 <= strength <= 1.0:
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
            + self.phi_ * previous
        )

        transformed = (
            series
            - strength * predicted_component
        )

        if not np.isfinite(transformed).all():
            raise RuntimeError(
                "Partially whitened stream contains "
                "NaN or infinite values."
            )

        if len(transformed) != len(series):
            raise RuntimeError(
                "Partial whitening changed stream length."
            )

        return transformed


def build_detector_streams(
    train,
    calibration,
    test,
):
    """
    Build the AEMO detector streams.

    Demand deseasonalisation is fitted on complete TRAIN.
    Global error standardisation and AR(1) parameters are fitted
    on chronological TRAIN Year-2 forecast errors.
    """

    if train.empty:
        raise RuntimeError(
            "AEMO TRAIN split is empty."
        )

    if calibration.empty:
        raise RuntimeError(
            "AEMO CALIBRATION split is empty."
        )

    if test.empty:
        raise RuntimeError(
            "AEMO TEST split is empty."
        )

    if len(train) <= TRAIN_REFERENCE_FIT_END:
        raise RuntimeError(
            "AEMO TRAIN does not contain enough observations "
            "for the chronological Year-1 / Year-2 protocol."
        )

    # TRAIN-fitted demand deseasonalisation

    train_ds = deseasonalise_demand_frame(
        data=train,
        reference=train,
    )

    calibration_ds = deseasonalise_demand_frame(
        data=calibration,
        reference=train,
    )

    test_ds = deseasonalise_demand_frame(
        data=test,
        reference=train,
    )

    for name, frame in (
        ("TRAIN", train_ds),
        ("CALIBRATION", calibration_ds),
        ("TEST", test_ds),
    ):

        values = (
            frame["TOTALDEMAND"]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"{name} deseasonalised TOTALDEMAND "
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
            "TRAIN Year-1 reference-fit period is empty."
        )

    if reference_eval.empty:
        raise RuntimeError(
            "TRAIN Year-2 reference-evaluation period is empty."
        )

    # Temporary model fitted on TRAIN Year 1

    reference_model = GradientBoostedForecaster(
        feature_mode="calendar+lags",
        lags=FORECASTER_LAGS,
        seed=FORECASTER_SEED,
    )

    reference_model.fit(reference_fit)

    reference_errors = frozen_error_stream(
        reference_model,
        reference_eval,
        history_df=reference_fit,
    )

    if reference_errors.empty:
        raise RuntimeError(
            "TRAIN Year-2 forecast-error stream is empty."
        )

    reference_abs = (
        reference_errors["error"]
        .abs()
        .to_numpy(dtype=float)
    )

    if not np.isfinite(reference_abs).all():
        raise RuntimeError(
            "TRAIN Year-2 absolute forecast errors contain "
            "NaN or infinite values."
        )

    # TRAIN-only global error z-score

    error_scaler = ErrorZScorePreprocessor()
    error_scaler.fit(reference_abs)

    reference_processed = (
        error_scaler
        .transform(reference_abs)
        .to_numpy(dtype=float)
    )

    if not np.isfinite(reference_processed).all():
        raise RuntimeError(
            "Processed TRAIN Year-2 error reference contains "
            "NaN or infinite values."
        )

    # TRAIN-only AR(1)

    whitener = PartialAR1Whitener()
    whitener.fit(reference_processed)

    # Final model fitted on complete TRAIN

    model = GradientBoostedForecaster(
        feature_mode="calendar+lags",
        lags=FORECASTER_LAGS,
        seed=FORECASTER_SEED,
    )

    model.fit(train_ds)

    # CALIBRATION forecast errors

    calibration_errors = frozen_error_stream(
        model,
        calibration_ds,
        history_df=train_ds,
    )

    if calibration_errors.empty:
        raise RuntimeError(
            "AEMO Calibration forecast-error stream is empty."
        )

    calibration_abs = (
        calibration_errors["error"]
        .abs()
        .to_numpy(dtype=float)
    )

    calibration_processed = (
        error_scaler
        .transform(calibration_abs)
        .to_numpy(dtype=float)
    )

    # TEST forecast errors

    test_errors = frozen_error_stream(
        model,
        test_ds,
        history_df=calibration_ds,
    )

    if test_errors.empty:
        raise RuntimeError(
            "AEMO Test forecast-error stream is empty."
        )

    test_abs = (
        test_errors["error"]
        .abs()
        .to_numpy(dtype=float)
    )

    test_processed = (
        error_scaler
        .transform(test_abs)
        .to_numpy(dtype=float)
    )

    # Validate processed streams

    if not np.isfinite(calibration_processed).all():
        raise RuntimeError(
            "Processed Calibration stream contains "
            "NaN or infinite values."
        )

    if not np.isfinite(test_processed).all():
        raise RuntimeError(
            "Processed Test stream contains "
            "NaN or infinite values."
        )

    if len(calibration_processed) != len(calibration):
        raise RuntimeError(
            "Processed Calibration stream is misaligned "
            "with the AEMO Calibration split."
        )

    if len(test_processed) != len(test):
        raise RuntimeError(
            "Processed Test stream is misaligned "
            "with the AEMO Test split."
        )

    # Partial AR(1) whitening

    calibration_whitened = whitener.transform(
        series=calibration_processed,
        strength=PARTIAL_AR1_LAMBDA,
        history_tail=reference_processed,
    )

    test_whitened = whitener.transform(
        series=test_processed,
        strength=PARTIAL_AR1_LAMBDA,
        history_tail=calibration_processed,
    )

    if (
        len(calibration_whitened)
        != len(calibration_processed)
    ):
        raise RuntimeError(
            "Whitened Calibration stream is misaligned."
        )

    if len(test_whitened) != len(test_processed):
        raise RuntimeError(
            "Whitened Test stream is misaligned."
        )

    if not np.isfinite(calibration_whitened).all():
        raise RuntimeError(
            "Whitened Calibration stream contains "
            "NaN or infinite values."
        )

    if not np.isfinite(test_whitened).all():
        raise RuntimeError(
            "Whitened Test stream contains "
            "NaN or infinite values."
        )

    return {
        "reference_processed":
            reference_processed,

        "calibration_processed":
            calibration_processed,

        "test_processed":
            test_processed,

        "calibration_whitened":
            calibration_whitened,

        "test_whitened":
            test_whitened,

        "calibration_errors":
            calibration_errors,

        "test_errors":
            test_errors,

        "ar1_intercept":
            whitener.intercept_,

        "ar1_phi":
            whitener.phi_,
    }

def run_region(region):
    """Run the frozen detectors for one AEMO region."""

    print(
        f"\nLoading {region}..."
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
        f"{region}: "
        f"TRAIN={len(train):,}, "
        f"CALIBRATION={len(calibration):,}, "
        f"TEST={len(test):,}"
    )

    print(
        f"{region}: "
        f"TEST start={test['SETTLEMENTDATE'].iloc[0]}, "
        f"TEST end={test['SETTLEMENTDATE'].iloc[-1]}"
    )

    streams = build_detector_streams(
        train=train,
        calibration=calibration,
        test=test,
    )

    print(
        f"{region}: "
        f"warm-up={len(streams['calibration_processed']):,}, "
        f"detector-test={len(streams['test_processed']):,}"
    )

    print(
        f"{region}: "
        f"TRAIN AR(1) "
        f"intercept={streams['ar1_intercept']:.6f}, "
        f"phi={streams['ar1_phi']:.6f}"
    )

    rows = []
    diagnostic_rows = []

    for detector_name in REQUIRED_DETECTORS:

        detector, config = make_registered_detector(
            detector_name
        )

        if detector_name in WHITENED_DETECTORS:

            warmup_stream = (
                streams["calibration_whitened"]
            )

            test_stream = (
                streams["test_whitened"]
            )

            whitening_type = "partial_AR1"
            whitening_lambda = PARTIAL_AR1_LAMBDA

        else:

            warmup_stream = (
                streams["calibration_processed"]
            )

            test_stream = (
                streams["test_processed"]
            )

            whitening_type = "none"
            whitening_lambda = 0.0

        changepoints = run_detector_with_protocol(
            test_stream=test_stream,
            detector=detector,
            warmup_stream=warmup_stream,
            refractory_period=REFRACTORY_PERIOD,
        )
        diagnostic_result = run_detector_diagnostic(
            stream=test_stream,
            detector_name=detector_name,
            config=config,
            warmup_stream=warmup_stream,
        )

        diagnostic_trace = diagnostic_result[
            "trace"
        ]

        if len(diagnostic_trace) != len(test_stream):
            raise RuntimeError(
                f"{region} / {detector_name}: "
                "diagnostic trace length does not match "
                "the detector TEST stream."
            )


        preprocessing = (
            f"partial_AR1_lambda_{PARTIAL_AR1_LAMBDA:.2f}"
            if detector_name in WHITENED_DETECTORS
            else "none"
        )

        print(
            f"{region} / {detector_name}: "
            f"{len(changepoints)} detections "
            f"[{preprocessing}]"
        )

        # F2 diagnostic-only replay rows.
        # raw_drift is the replay detector signal and is not an official
        # refractory-filtered project changepoint.

        for diagnostic in diagnostic_trace:

            index = diagnostic["stream_index"]

            if (
                index < 0
                or index >= len(streams["test_errors"])
            ):
                raise RuntimeError(
                    f"{region} / {detector_name}: "
                    f"out-of-range diagnostic index {index}."
                )

            error_row = (
                streams["test_errors"]
                .iloc[index]
            )

            diagnostic_rows.append(
                {
                    "region":
                        region,

                    "detector":
                        detector_name,

                    "stream_index":
                        int(index),

                    "SETTLEMENTDATE":
                        error_row["SETTLEMENTDATE"],

                    "forecast_error":
                        float(error_row["error"]),

                    "absolute_forecast_error":
                        float(abs(error_row["error"])),

                    "detector_input_value":
                        float(
                            diagnostic[
                                "stream_value"
                            ]
                        ),

                    "statistic":
                        diagnostic["statistic"],

                    "threshold":
                        diagnostic["threshold"],

                    "statistic_name":
                        diagnostic["statistic_name"],

                    "threshold_name":
                        diagnostic["threshold_name"],

                    "raw_drift":
                        bool(
                            diagnostic["raw_drift"]
                        ),

                    "detector_config":
                        json.dumps(
                            config,
                            sort_keys=True,
                        ),

                    "warmup_samples":
                        int(len(warmup_stream)),

                    "test_samples":
                        int(len(test_stream)),

                    "whitening_type":
                        whitening_type,

                    "whitening_lambda":
                        float(whitening_lambda),
                }
            )

        # Official changepoint rows

        for index in changepoints:

            if (
                index < 0
                or index >= len(streams["test_errors"])
            ):
                raise RuntimeError(
                    f"{region} / {detector_name}: "
                    f"out-of-range detection index {index}."
                )

            error_row = (
                streams["test_errors"]
                .iloc[index]
            )

            rows.append(
                {
                    "region":
                        region,

                    "detector":
                        detector_name,

                    "detected_index":
                        int(index),

                    "SETTLEMENTDATE":
                        error_row["SETTLEMENTDATE"],

                    "forecast_error":
                        float(error_row["error"]),

                    "absolute_forecast_error":
                        float(abs(error_row["error"])),

                    "normalized_abs_error":
                        float(test_stream[index]),

                    "detector_input_value":
                        float(test_stream[index]),

                    "detector_config":
                        json.dumps(
                            config,
                            sort_keys=True,
                        ),
                    "warmup_samples":
                        int(len(warmup_stream)),

                    "test_samples":
                        int(len(test_stream)),

                    "refractory_period":
                        REFRACTORY_PERIOD,

                    "refractory_days":
                        REFRACTORY_DAYS,

                    "seasonal_profile":
                        SEASONAL_PROFILE,

                    "seasonal_reference":
                        SEASONAL_REFERENCE,

                    "error_zscore_reference":
                        ERROR_ZSCORE_REFERENCE,

                    "error_zscore_type":
                        ERROR_ZSCORE_TYPE,

                    "warmup_reference":
                        WARMUP_REFERENCE,

                    "detector_input":
                        DETECTOR_INPUT,

                    "whitening_type":
                        whitening_type,

                    "whitening_lambda":
                        float(whitening_lambda),

                    "ar1_intercept":
                        float(
                            streams["ar1_intercept"]
                        ),

                    "ar1_phi":
                        float(
                            streams["ar1_phi"]
                        ),
                }
            )

    return rows, diagnostic_rows


OUTPUT_COLUMNS = [
    "region",
    "detector",
    "detected_index",
    "SETTLEMENTDATE",
    "forecast_error",
    "absolute_forecast_error",
    "normalized_abs_error",
    "detector_input_value",
    "detector_config",
    "warmup_samples",
    "test_samples",
    "refractory_period",
    "refractory_days",
    "seasonal_profile",
    "seasonal_reference",
    "error_zscore_reference",
    "error_zscore_type",
    "warmup_reference",
    "detector_input",
    "whitening_type",
    "whitening_lambda",
    "ar1_intercept",
    "ar1_phi",
]


DIAGNOSTIC_COLUMNS = [
    "region",
    "detector",
    "stream_index",
    "SETTLEMENTDATE",
    "forecast_error",
    "absolute_forecast_error",
    "detector_input_value",
    "statistic",
    "threshold",
    "statistic_name",
    "threshold_name",
    "raw_drift",
    "detector_config",
    "warmup_samples",
    "test_samples",
    "whitening_type",
    "whitening_lambda",
]


def main():

    validate_detector_registry()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("PGS242-120: FINAL AEMO DRIFT DETECTION")
    print("=" * 90)

    print(f"Regions: {REGIONS}")
    print("Cadence: standardised 30-minute AEMO")

    print(
        "Forecaster: Gradient Boosting "
        "calendar+lags (48, 336)"
    )

    print(
        "Demand preprocessing: "
        "TRAIN-fitted deseasonalisation"
    )

    print(
        "Seasonal profile: "
        "overall mean + smoothed DOY + "
        "(DOW x half-hour)"
    )

    print(
        "Error standardisation: "
        "global z-score fitted on chronological "
        "TRAIN Year-2 forecast errors"
    )

    print(
        "Detector warm-up: "
        "processed CALIBRATION errors"
    )

    print(
        f"ADWIN whitening: partial AR(1), "
        f"lambda={PARTIAL_AR1_LAMBDA:.2f}"
    )

    print(
        f"KSWIN whitening: partial AR(1), "
        f"lambda={PARTIAL_AR1_LAMBDA:.2f}"
    )

    print("Page-Hinkley whitening: none")

    print(
        f"Refractory: "
        f"{REFRACTORY_PERIOD} observations "
        f"({REFRACTORY_DAYS} days)"
    )

    print(
        "Detector parameters: "
        "frozen DETECTOR_REGISTRY"
    )

    print("\nFrozen detector registry:")

    for detector_name in REQUIRED_DETECTORS:
        print(
            f"  {detector_name}: "
            f"{DETECTOR_REGISTRY[detector_name]['config']}"
        )

    all_rows = []
    all_diagnostic_rows = []

    for region in REGIONS:

        (
            region_rows,
            region_diagnostic_rows,
        ) = run_region(region)

        all_rows.extend(
            region_rows
        )

        all_diagnostic_rows.extend(
            region_diagnostic_rows
        )

    output = pd.DataFrame(
        all_rows,
        columns=OUTPUT_COLUMNS,
    )

    if not output.empty:
        output = (
            output
            .sort_values(
                [
                    "region",
                    "detector",
                    "detected_index",
                ]
            )
            .reset_index(drop=True)
        )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    diagnostic_output = pd.DataFrame(
        all_diagnostic_rows,
        columns=DIAGNOSTIC_COLUMNS,
    )

    if not diagnostic_output.empty:
        diagnostic_output = (
            diagnostic_output
            .sort_values(
                [
                    "region",
                    "detector",
                    "stream_index",
                ]
            )
            .reset_index(drop=True)
        )

    diagnostic_output.to_csv(
        DIAGNOSTIC_OUTPUT_PATH,
        index=False,
        compression="gzip",
    )

    print()
    print("=" * 90)
    print("AEMO DETECTION SUMMARY")
    print("=" * 90)

    if output.empty:

        print("No AEMO detections.")

    else:

        summary = (
            output
            .groupby(
                [
                    "region",
                    "detector",
                ]
            )
            .size()
            .rename("detections")
            .reset_index()
        )

        print(
            summary.to_string(index=False)
        )

    print()
    print("AEMO detection complete.")

    print(
        f"Saved changepoints to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved F2 diagnostics to:\n"
        f"{DIAGNOSTIC_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()