"""
Apply the T1-validated drift detectors to the real AEMO
SA1 and NSW1 forecasting-error streams.

PGS242-120

Protocol
--------
- Regions: SA1, NSW1
- AEMO frequency: standardised 30-minute observations
- Fixed split from data_loader.py
- Forecaster: frozen Gradient Boosting with calendar + lags (48, 336)
- Detector input: calibration-normalized absolute forecast error
- Detector configurations: frozen DETECTOR_REGISTRY
- Output: detected test-stream indices mapped to SETTLEMENTDATE

This runner performs detection only.
It does NOT perform documented-event matching or T2 evaluation.
"""

from pathlib import Path
import sys
import json

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
from drift_lab.detectors import run_all_detectors
from drift_lab.detector_diagnostics import run_detector_diagnostic
from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)


# ---------------------------------------------------------------------
# Frozen PGS242-120 protocol
# ---------------------------------------------------------------------

REGIONS = ("SA1", "NSW1")

DATA_PATH = PROJECT_ROOT / "data" / "raw"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    /"tables"
    /"detection"
    / "aemo_detection_changepoints.csv"
)

DIAGNOSTIC_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "aemo_detection_diagnostics.csv.gz"
)

FORECASTER_CONFIG = {
    "feature_mode": "calendar+lags",
    "lags": (48, 336),
    "seed": 0,
}


# ---------------------------------------------------------------------
# Detector-input construction
# ---------------------------------------------------------------------

def build_normalized_absolute_error_stream(
    train,
    calibration,
    test,
):
    """
    Reproduce the detector-input protocol validated in T1.

    1. Fit the frozen Gradient Boosting forecaster on training data.
    2. Generate calibration forecasting errors.
    3. Estimate mean/std of absolute calibration error.
    4. Generate test forecasting errors.
    5. Convert test residuals to absolute errors.
    6. Standardize them using calibration-only statistics.
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

    calibration_abs = (
        calibration_errors["error"]
        .abs()
        .to_numpy(dtype=float)
    )

    calibration_mean = float(
        calibration_abs.mean()
    )

    calibration_std = float(
        calibration_abs.std()
    )

    if not np.isfinite(calibration_mean):
        raise RuntimeError(
            "Calibration absolute-error mean is not finite."
        )

    if (
        not np.isfinite(calibration_std)
        or calibration_std <= 0
    ):
        raise RuntimeError(
            "Calibration absolute-error standard deviation "
            "must be finite and greater than zero."
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
        test_abs - calibration_mean
    ) / calibration_std

    if len(stream) != len(test):
        raise RuntimeError(
            "Detector stream length does not match test length."
        )

    if not np.isfinite(stream).all():
        raise RuntimeError(
            "Detector stream contains NaN or infinite values."
        )

    return (
        stream,
        test_errors,
        calibration_mean,
        calibration_std,
    )


# ---------------------------------------------------------------------
# Region runner
# ---------------------------------------------------------------------

def run_region(region):
    """
    Run all frozen T1 detector configurations on one AEMO region.
    """

    print(f"Loading {region}...")

    _, train, calibration, test = load_and_split_aemo(
        region=region,
        data_path=DATA_PATH,
    )

    print(
        f"{region}: "
        f"train={len(train)}, "
        f"calibration={len(calibration)}, "
        f"test={len(test)}"
    )

    (
        stream,
        test_errors,
        calibration_mean,
        calibration_std,
    ) = build_normalized_absolute_error_stream(
        train=train,
        calibration=calibration,
        test=test,
    )

    detector_results = run_all_detectors(
        stream
    )

    rows = []
    diagnostic_rows = []

    for detector_name, result in detector_results.items():

        config = result["config"]

        changepoints = result["changepoints"]
        diagnostic_result = run_detector_diagnostic(
            stream,
            detector_name,
        )

        if (
            diagnostic_result["changepoints"]
            != changepoints
        ):
            raise RuntimeError(
                f"{region} / {detector_name}: "
                "diagnostic changepoints differ from "
                "the frozen detector-interface results."
            )

        print(
            f"{region} / {detector_name}: "
            f"{len(changepoints)} detections"
        )

        for trace_row in diagnostic_result["trace"]:
            trace_index = trace_row["detected_index"]

            if (
                trace_index < 0
                or trace_index >= len(test_errors)
            ):
                raise RuntimeError(
                    f"{detector_name} diagnostic returned "
                    f"out-of-range index {trace_index}."
                )

            error_row = test_errors.iloc[trace_index]

            diagnostic_rows.append(
                {
                    "region": region,
                    "detector": detector_name,
                    "detected_index": trace_index,
                    "SETTLEMENTDATE": (
                        error_row["SETTLEMENTDATE"]
                    ),
                    "forecast_error": float(
                        error_row["error"]
                    ),
                    "normalized_abs_error": (
                        trace_row["stream_value"]
                    ),
                    "statistic": trace_row["statistic"],
                    "threshold": trace_row["threshold"],
                    "statistic_name": (
                        trace_row["statistic_name"]
                    ),
                    "threshold_name": (
                        trace_row["threshold_name"]
                    ),
                    "is_drift": trace_row["is_drift"],
                    "detector_config": json.dumps(
                        config,
                        sort_keys=True,
                    ),
                }
            )
        for index in changepoints:

            if index < 0 or index >= len(test_errors):
                raise RuntimeError(
                    f"Detector {detector_name} returned "
                    f"out-of-range index {index}."
                )

            error_row = test_errors.iloc[index]

            rows.append(
                {
                    "region": region,
                    "detector": detector_name,
                    "detected_index": int(index),
                    "SETTLEMENTDATE": (
                        error_row["SETTLEMENTDATE"]
                    ),
                    "forecast_error": float(
                        error_row["error"]
                    ),
                    "normalized_abs_error": float(
                        stream[index]
                    ),
                    "calibration_abs_error_mean": (
                        calibration_mean
                    ),
                    "calibration_abs_error_std": (
                        calibration_std
                    ),
                    "detector_config": json.dumps(
                        config,
                        sort_keys=True,
                    ),
                }
            )

    return rows, diagnostic_rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 72)
    print("PGS242-120: AEMO drift detection")
    print("=" * 72)

    print(
        "Forecaster: frozen Gradient Boosting, "
        "calendar+lags (48, 336)"
    )

    print(
        "Detector input: "
        "calibration-normalized absolute forecast error"
    )

    print(
        f"Raw AEMO path: {DATA_PATH}"
    )

    print()

    rows = []
    diagnostic_rows = []

    for region in REGIONS:
        (
            region_rows,
            region_diagnostic_rows,
        ) = run_region(region)

        rows.extend(region_rows)
        diagnostic_rows.extend(
            region_diagnostic_rows
        )
        print()

    output = pd.DataFrame(rows)
    diagnostic_output = pd.DataFrame(
        diagnostic_rows
    )

    if output.empty:
        output = pd.DataFrame(
            columns=[
                "region",
                "detector",
                "detected_index",
                "SETTLEMENTDATE",
                "forecast_error",
                "normalized_abs_error",
                "calibration_abs_error_mean",
                "calibration_abs_error_std",
                "detector_config",
            ]
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )
    diagnostic_output.to_csv(
        DIAGNOSTIC_OUTPUT_PATH,
        index=False,
        compression={
            "method": "gzip",
            "compresslevel": 6,
            "mtime": 0,
        },
    )
    print("=" * 72)
    print("Detection complete")
    print("=" * 72)

    print(
        output
        .groupby(
            ["region", "detector"]
        )
        .size()
        .rename("detections")
        .to_string()
        if not output.empty
        else "No changepoints detected."
    )

    print()

    print(
        f"Saved detection output to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
