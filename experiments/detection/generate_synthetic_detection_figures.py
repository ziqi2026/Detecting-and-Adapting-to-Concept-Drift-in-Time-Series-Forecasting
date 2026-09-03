"""
Generate synthetic drift-detection evaluation figures matching the
04.1_Synthetic_Drift_Detector_Tuning_and_Evaluation notebook.

Each output figure contains four panels (none, sudden, gradual, recurring)
for one detector and one representative seed.

The plotted detector input is the calibration-normalized absolute
forecast-error stream. Ground truth, matched detections, unmatched
detections, detection delay, false alarms / 10k, and missed detections
are shown using the shared evaluation logic.

No experiment metrics are written to results/runs.csv.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from drift_lab import detectors
from drift_lab.evaluation import evaluate_detections
from drift_lab.synthetic_generator import make_feature_series
from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)

# Frozen synthetic evaluation settings
N_OBSERVATIONS = 20_000
TRAIN_END = 4_000
CALIBRATION_END = 5_000
TEST_START = 5_000

REPRESENTATIVE_SEED = 1
FORECASTER_SEED = 0

DRIFT_TYPES = [
    "none",
    "sudden",
    "gradual",
    "recurring",
]

# 7 days × 48 half-hour observations
TOLERANCE = 48 * 7

OUTPUT_DIR = Path("results/figures/detection")


# Final detector configurations used in the notebook

ADWIN_CONFIG = {
    "delta": 0.0001,
}

PAGE_HINKLEY_CONFIG = {
    "threshold": 209,
    "delta": 0.005,
    "alpha": 0.999,
    "min_instances": 30,
}

KSWIN_CONFIG = {
    "alpha": 0.00005,
    "window_size": 140,
    "stat_size": 70,
    "seed": 12,
}


def build_normalized_absolute_error_stream(drift_type, seed):
    """
    Build the calibration-normalized absolute forecast-error stream
    used as the common detector input.

    Returns
    -------
    stream : np.ndarray
        Normalized absolute forecast-error stream for the test period.

    truth : list[int]
        Ground-truth changepoints relative to the test stream.

    cal_mean : float
        Mean absolute calibration error.

    cal_std : float
        Standard deviation of absolute calibration error.
    """

    data, full_truth = make_feature_series(
        kind=drift_type,
        n=N_OBSERVATIONS,
        seed=seed,
    )

    train = data.iloc[:TRAIN_END].copy()

    calibration = data.iloc[
        TRAIN_END:CALIBRATION_END
    ].copy()

    test = data.iloc[
        TEST_START:
    ].copy()

    model = GradientBoostedForecaster(
        feature_mode="calendar+lags",
        lags=(48, 336),
        seed=FORECASTER_SEED,
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

    cal_mean = calibration_abs.mean()
    cal_std = calibration_abs.std()

    if not np.isfinite(cal_mean):
        raise RuntimeError(
            "Calibration absolute-error mean is not finite."
        )

    if not np.isfinite(cal_std) or cal_std <= 0:
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
        test_abs - cal_mean
    ) / cal_std

    if not np.isfinite(stream).all():
        raise RuntimeError(
            "Normalized error stream contains NaN or infinite values."
        )

    truth = [
        int(cp - TEST_START)
        for cp in full_truth
    ]

    return stream, truth, cal_mean, cal_std


def plot_detector_evaluation(
    detector_name,
    detector_func,
    detector_config,
    seed=REPRESENTATIVE_SEED,
):
    """
    Reproduce the notebook's four-panel detector evaluation figure.
    """

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 14),
        sharex=False,
    )

    for ax, drift_type in zip(
        axes,
        DRIFT_TYPES,
    ):
        
        # Build detector input
        stream, truth, _, _ = (
            build_normalized_absolute_error_stream(
                drift_type=drift_type,
                seed=seed,
            )
        )

      
        # Run detector
        detections = detector_func(
            stream,
            **detector_config,
        )

     
        # Shared evaluation
        metrics = evaluate_detections(
            detected_changepoints=detections,
            true_changepoints=truth,
            n_observations=len(stream),
            drift_type=drift_type,
            tolerance=TOLERANCE,
        )

        matched_pairs = metrics[
            "matched_pairs"
        ]

        false_alarm_indices = metrics[
            "false_alarm_indices"
        ]

        # Detector input stream
        ax.plot(
            stream,
            color="steelblue",
            linewidth=0.7,
            alpha=0.8,
            label="Normalized absolute forecast error",
        )


        # Ground truth
        if drift_type == "gradual":
            drift_start = truth[0]
            drift_end = truth[1]

            ax.axvspan(
                drift_start,
                drift_end,
                color="red",
                alpha=0.12,
                label="True gradual drift window",
            )

            ax.axvline(
                drift_start,
                color="red",
                linestyle="--",
                linewidth=1.5,
            )

            ax.axvline(
                drift_end,
                color="red",
                linestyle="--",
                linewidth=1.5,
            )

        else:
            for i, true_cp in enumerate(truth):
                ax.axvline(
                    true_cp,
                    color="red",
                    linestyle="--",
                    linewidth=1.8,
                    label=(
                        "True changepoint"
                        if i == 0
                        else None
                    ),
                )

        # Matched detections
        for i, match in enumerate(matched_pairs):
            detected_cp = int(
                match["detected"]
            )

            if drift_type == "gradual":
                true_reference = int(
                    match["true_start"]
                )
            else:
                true_reference = int(
                    match["true"]
                )

            delay = (
                detected_cp
                - true_reference
            )

            ax.axvline(
                detected_cp,
                color="green",
                linewidth=2,
                label=(
                    "Matched detection"
                    if i == 0
                    else None
                ),
            )

            y_min, y_max = ax.get_ylim()

            y_position = (
                y_min
                + 0.82 * (y_max - y_min)
            )

            ax.text(
                detected_cp,
                y_position,
                f"delay={delay}",
                color="green",
                rotation=90,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
            )

   
        # Unmatched detections
        for i, detected_cp in enumerate(
            false_alarm_indices
        ):
            ax.axvline(
                int(detected_cp),
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label=(
                    "Unmatched detection"
                    if i == 0
                    else None
                ),
            )


        # Evaluation summary
        detection_delay = metrics[
            "detection_delay"
        ]

        if np.isnan(detection_delay):
            delay_text = "—"
        else:
            delay_text = (
                f"{detection_delay:.1f}"
            )

        false_alarm_rate = metrics[
            "false_alarms_per_10000"
        ]

        missed = metrics[
            "missed_detections"
        ]

        ax.set_title(
            f"{drift_type.capitalize()} drift"
            f"  |  delay = {delay_text}"
            f"  |  false alarms / 10k = "
            f"{false_alarm_rate:.2f}"
            f"  |  missed = {missed}"
        )

        ax.set_ylabel(
            "Normalized\nabsolute error"
        )

        ax.legend(
            loc="upper right",
            fontsize=8,
        )

        ax.grid(
            alpha=0.15
        )

    axes[-1].set_xlabel(
        "Test observation index"
    )

    fig.suptitle(
        f"{detector_name} — Synthetic Drift Detection Evaluation "
        f"(Seed {seed})",
        fontsize=16,
        y=1.01,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = (
        detector_name.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    output_path = (
        OUTPUT_DIR
        / f"{safe_name}_synthetic_evaluation_seed{seed}.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def main():
    print(
        "Generating notebook-style synthetic detection figures"
    )

    print(
        f"Representative seed: {REPRESENTATIVE_SEED}"
    )

    print(
        "No metrics will be written to runs.csv."
    )

    plot_detector_evaluation(
        detector_name="ADWIN",
        detector_func=detectors.run_adwin,
        detector_config=ADWIN_CONFIG,
    )

    plot_detector_evaluation(
        detector_name="Page-Hinkley",
        detector_func=detectors.run_page_hinkley,
        detector_config=PAGE_HINKLEY_CONFIG,
    )

    plot_detector_evaluation(
        detector_name="KSWIN",
        detector_func=detectors.run_kswin,
        detector_config=KSWIN_CONFIG,
    )

    print(
        "Figure generation complete."
    )


if __name__ == "__main__":
    main()
