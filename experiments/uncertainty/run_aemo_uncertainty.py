"""Run the initial 90% prediction-interval baseline on AEMO SA1 and NSW1.

PGS242-141

Protocol
--------
- Regions: SA1, NSW1
- Fixed chronological split from data_loader.py
- Forecaster: frozen Gradient Boosting with calendar + lags (48, 336)
- UQ calibration input: calibration-period forecast residuals only
- Interval: static symmetric 90% split-conformal-style interval
- Evaluation: empirical coverage and mean interval width
- Output: timestamp-level point forecast / lower 90% / upper 90%
- T3 handoff: lower_90 and upper_90 are the nominal 0.05/0.95
  interval endpoints required for downstream pinball-loss evaluation

This runner intentionally does NOT implement adaptive conformal updates,
escalation sweeps, F4/F5, or T5/T6.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from drift_lab.data_loader import load_and_split_aemo
from drift_lab.forecasters import GradientBoostedForecaster, frozen_error_stream
from drift_lab.evaluation import (
    calculate_empirical_coverage,
    calculate_mean_interval_width,
)
from drift_lab.uncertainty import (
    DEFAULT_ALPHA,
    conformal_radius,
    prediction_intervals,
    uq,
)


REGIONS = ("SA1", "NSW1")
DATA_PATH = PROJECT_ROOT / "data" / "raw"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "uncertainty"
    / "aemo_prediction_intervals_90.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "uncertainty"
    / "aemo_prediction_intervals_90_summary.csv"
)

FORECASTER_CONFIG = {
    "feature_mode": "calendar+lags",
    "lags": (48, 336),
    "seed": 0,
}


def run_region(region):
    """Generate and evaluate static 90% prediction intervals for one region."""
    print(f"Loading {region}...")

    _, train, calibration, test = load_and_split_aemo(
        region=region,
        data_path=DATA_PATH,
    )

    print(
        f"{region}: train={len(train)}, "
        f"calibration={len(calibration)}, "
        f"test={len(test)}"
    )

    model = GradientBoostedForecaster(**FORECASTER_CONFIG)
    model.fit(train)

    calibration_stream = frozen_error_stream(
        model,
        calibration,
        history_df=train,
    )

    calibration_residuals = (
        calibration_stream["error"].to_numpy(dtype=float)
    )

    radius = conformal_radius(
        calibration_residuals,
        alpha=DEFAULT_ALPHA,
    )

    test_stream = frozen_error_stream(
        model,
        test,
        history_df=calibration,
    )

    point_forecasts = test_stream["y_pred"].to_numpy(dtype=float)

    lower, upper, escalate = prediction_intervals(
        point_forecasts,
        calibration_residuals,
        alpha=DEFAULT_ALPHA,
    )

    # Contract-compatibility smoke test: scalar uq() must match the batch helper.
    if len(point_forecasts) > 0:
        scalar_lower, scalar_upper, scalar_escalate = uq(
            point_forecasts[0],
            calibration_residuals,
        )
        np.testing.assert_allclose(scalar_lower, lower[0])
        np.testing.assert_allclose(scalar_upper, upper[0])
        if bool(scalar_escalate) != bool(escalate[0]):
            raise RuntimeError("Scalar uq() and batch helper disagree.")

    y_true = test_stream["y_true"].to_numpy(dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower

    coverage = calculate_empirical_coverage(
        y_true,
        lower,
        upper,
    )
    mean_width = calculate_mean_interval_width(
        lower,
        upper,
    )

    output = pd.DataFrame(
        {
            "region": region,
            "SETTLEMENTDATE": test_stream["SETTLEMENTDATE"].values,
            "y_true": y_true,
            "point_forecast": point_forecasts,
            "lower_90": lower,
            "upper_90": upper,
            "covered_90": covered,
            "interval_width": widths,
            "escalate_flag": escalate,
        }
    )

    summary = {
        "region": region,
        "alpha": DEFAULT_ALPHA,
        "target_coverage": 1.0 - DEFAULT_ALPHA,
        "calibration_size": len(calibration_residuals),
        "conformal_radius": radius,
        "test_size": len(test_stream),
        "empirical_coverage": coverage,
        "mean_interval_width": mean_width,
    }

    print(
        f"{region}: q_hat={radius:.3f}, "
        f"coverage={coverage:.4f}, "
        f"mean_width={mean_width:.3f}"
    )

    return output, summary


def main():
    print("=" * 72)
    print("PGS242-141: Initial 90% AEMO prediction intervals")
    print("=" * 72)
    print(
        "Forecaster: frozen Gradient Boosting, "
        "calendar+lags (48, 336)"
    )
    print(
        "UQ: static calibration-residual interval; "
        "no test residuals used for interval construction"
    )
    print(f"Raw AEMO path: {DATA_PATH}")
    print()

    outputs = []
    summaries = []

    for region in REGIONS:
        output, summary = run_region(region)
        outputs.append(output)
        summaries.append(summary)
        print()

    all_output = pd.concat(outputs, ignore_index=True)
    summary_output = pd.DataFrame(summaries)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_output.to_csv(OUTPUT_PATH, index=False)
    summary_output.to_csv(SUMMARY_PATH, index=False)

    print("=" * 72)
    print("Uncertainty baseline complete")
    print("=" * 72)
    print(summary_output.to_string(index=False))
    print()
    print(f"Saved timestamp-level output to:\n{OUTPUT_PATH}")
    print(f"Saved summary output to:\n{SUMMARY_PATH}")


if __name__ == "__main__":
    main()
