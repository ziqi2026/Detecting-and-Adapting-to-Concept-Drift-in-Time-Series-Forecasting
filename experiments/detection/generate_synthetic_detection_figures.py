"""
Generate representative Seed-1 synthetic drift-detection figures
for the final T1 experiment.

Each output figure contains four panels:

    none
    sudden
    gradual
    recurring

for one detector and representative seed 1.

The experiment configuration, preprocessing, detector registry,
warm-up protocol, refractory period, changepoints, and evaluation
tolerance are imported directly from run_synthetic_drift_detection.py.

This ensures that the figures reproduce the same final synthetic
pipeline used for T1.

No experiment metrics are written to runs.csv.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_synthetic_drift_detection as synthetic_runner


REPRESENTATIVE_SEED = 1

DRIFT_TYPES = (
    "none",
    "sudden",
    "gradual",
    "recurring",
)

OUTPUT_DIR = Path(
    "results/figures/detection"
)


def get_detector_streams(
    detector_name,
    streams,
):
    """
    Select the same detector input used by the final T1 runner.

    ADWIN and KSWIN use the partially whitened stream.
    Page-Hinkley uses the unwhitened processed stream.
    """

    if (
        detector_name
        in synthetic_runner.WHITENED_DETECTORS
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

    return (
        warmup_stream,
        test_stream,
    )


def build_seed_scenario(
    drift_type,
    seed,
):
    """
    Build one synthetic scenario using the exact final T1 pipeline.
    """

    (
        data,
        full_truth,
    ) = (
        synthetic_runner.make_feature_series(
            kind=drift_type,
            n=synthetic_runner.N_OBSERVATIONS,
            seed=seed,
            changepoints=(
                synthetic_runner
                .SYNTHETIC_CHANGEPOINTS[
                    drift_type
                ]
            ),
        )
    )

    if (
        len(data)
        != synthetic_runner.N_OBSERVATIONS
    ):
        raise RuntimeError(
            "Synthetic generator returned "
            f"{len(data)} observations; "
            f"expected "
            f"{synthetic_runner.N_OBSERVATIONS}."
        )

    truth = (
        synthetic_runner.to_test_relative(
            full_changepoints=full_truth,
            drift_type=drift_type,
            seed=seed,
        )
    )

    streams = (
        synthetic_runner
        .build_detector_streams(
            data
        )
    )

    return (
        streams,
        truth,
    )


def plot_detector_evaluation(
    detector_name,
    detector_title,
    seed=REPRESENTATIVE_SEED,
):
    """
    Plot the final T1 Seed-1 result for one detector.
    """

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(
            15,
            14,
        ),
        sharex=False,
    )

    for ax, drift_type in zip(
        axes,
        DRIFT_TYPES,
    ):

        (
            streams,
            truth,
        ) = build_seed_scenario(
            drift_type=drift_type,
            seed=seed,
        )

        (
            warmup_stream,
            test_stream,
        ) = get_detector_streams(
            detector_name,
            streams,
        )

        (
            detections,
            detector_config,
        ) = (
            synthetic_runner
            .run_registered_detector(
                detector_name=detector_name,
                test_stream=test_stream,
                warmup_stream=warmup_stream,
            )
        )

        metrics = (
            synthetic_runner
            .evaluate_detections(
                detected_changepoints=detections,
                true_changepoints=truth,
                n_observations=len(
                    test_stream
                ),
                drift_type=drift_type,
                tolerance=(
                    synthetic_runner
                    .TOLERANCE
                ),
            )
        )

        matched_pairs = (
            metrics[
                "matched_pairs"
            ]
        )

        false_alarm_indices = (
            metrics[
                "false_alarm_indices"
            ]
        )

        ax.plot(
            test_stream,
            color="steelblue",
            linewidth=0.7,
            alpha=0.8,
            label="Detector input",
        )

        if drift_type == "gradual":

            drift_start = truth[0]
            drift_end = truth[1]

            ax.axvspan(
                drift_start,
                drift_end,
                color="red",
                alpha=0.12,
                label=(
                    "True gradual drift window"
                ),
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

            for i, true_cp in enumerate(
                truth
            ):

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

        for i, match in enumerate(
            matched_pairs
        ):

            detected_cp = int(
                match[
                    "detected"
                ]
            )

            if drift_type == "gradual":

                true_reference = int(
                    match[
                        "true_start"
                    ]
                )

            else:

                true_reference = int(
                    match[
                        "true"
                    ]
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

            y_min, y_max = (
                ax.get_ylim()
            )

            y_position = (
                y_min
                + 0.82
                * (
                    y_max
                    - y_min
                )
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

        for i, detected_cp in enumerate(
            false_alarm_indices
        ):

            ax.axvline(
                int(
                    detected_cp
                ),
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label=(
                    "Unmatched detection"
                    if i == 0
                    else None
                ),
            )

        detection_delay = (
            metrics[
                "detection_delay"
            ]
        )

        if np.isnan(
            detection_delay
        ):
            delay_text = "—"

        else:
            delay_text = (
                f"{detection_delay:.1f}"
            )

        false_alarm_rate = (
            metrics[
                "false_alarms_per_10000"
            ]
        )

        missed = (
            metrics[
                "missed_detections"
            ]
        )

        ax.set_title(
            f"{drift_type.capitalize()} drift"
            f"  |  delay = {delay_text}"
            f"  |  false alarms / 10k = "
            f"{false_alarm_rate:.2f}"
            f"  |  missed = {missed}"
        )

        if (
            detector_name
            in synthetic_runner
            .WHITENED_DETECTORS
        ):

            ylabel = (
                "Partially whitened\n"
                "standardised error"
            )

        else:

            ylabel = (
                "Standardised\n"
                "absolute error"
            )

        ax.set_ylabel(
            ylabel
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

    config = (
        synthetic_runner
        .DETECTOR_REGISTRY[
            detector_name
        ][
            "config"
        ]
    )

    fig.suptitle(
        (
            f"{detector_title} — "
            f"Synthetic Drift Detection Evaluation "
            f"(Seed {seed})"
        ),
        fontsize=16,
        y=1.01,
    )

    fig.text(
        0.5,
        0.005,
        (
            f"Final T1 configuration: "
            f"{config}"
        ),
        ha="center",
        fontsize=8,
    )

    plt.tight_layout(
        rect=[
            0,
            0.025,
            1,
            1,
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = (
        detector_name
        .lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"{safe_name}_"
            f"synthetic_evaluation_"
            f"seed{seed}.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: "
        f"{output_path}"
    )


def main():
    """
    Generate the three final T1 representative Seed-1 figures.
    """

    synthetic_runner.validate_detector_registry()

    print(
        "Generating final T1 synthetic detection figures"
    )

    print(
        f"Representative seed: "
        f"{REPRESENTATIVE_SEED}"
    )

    print(
        f"TRAIN: "
        f"{synthetic_runner.TRAIN_LENGTH:,}"
    )

    print(
        f"CALIBRATION: "
        f"{synthetic_runner.CALIBRATION_LENGTH:,}"
    )

    print(
        f"TEST: "
        f"{synthetic_runner.TEST_LENGTH:,}"
    )

    print(
        f"Refractory: "
        f"{synthetic_runner.REFRACTORY_PERIOD} "
        f"observations"
    )

    print(
        f"Tolerance: "
        f"{synthetic_runner.TOLERANCE} "
        f"observations"
    )

    print(
        "No metrics will be written to runs.csv."
    )

    plot_detector_evaluation(
        detector_name="adwin",
        detector_title="ADWIN",
    )

    plot_detector_evaluation(
        detector_name="page_hinkley",
        detector_title="Page-Hinkley",
    )

    plot_detector_evaluation(
        detector_name="kswin",
        detector_title="KSWIN",
    )

    print(
        "Figure generation complete."
    )


if __name__ == "__main__":
    main()