"""
Generate synthetic drift-detection review figures.

The figures use one representative recorded seed (seed=0) and the same
synthetic split, forecaster, detector configurations, and evaluation
tolerance as the multi-seed evaluation.

Output
------
results/figures/detection_none_seed0.png
results/figures/detection_sudden_seed0.png
results/figures/detection_gradual_seed0.png
results/figures/detection_recurring_seed0.png

Each figure is a detector timeline:
    - true changepoints are shown as vertical reference lines;
    - gradual drift is shown as a shaded transition window;
    - detector alarms are shown as event ticks on separate detector rows.

No experiment metrics are written to results/runs.csv.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from synthetic_generator import make_feature_series
from drift_lab.detectors import run_all_detectors
from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)


N_OBSERVATIONS = 20000
TRAIN_END = 4000
CALIBRATION_END = 5000
TEST_START = 5000

SEED = 0
FORECASTER_SEED = 0

DRIFT_TYPES = (
    "none",
    "sudden",
    "gradual",
    "recurring",
)

OUTPUT_DIR = Path("results/figures")


def to_test_relative(full_changepoints):
    return [
        int(cp - TEST_START)
        for cp in full_changepoints
    ]


def build_error_stream(data):
    train = data.iloc[:TRAIN_END].copy()
    calibration = data.iloc[
        TRAIN_END:CALIBRATION_END
    ].copy()
    test = data.iloc[
        TEST_START:
    ].copy()

    model = GradientBoostedForecaster(
        feature_mode="calendar",
        seed=FORECASTER_SEED,
    )
    model.fit(train)

    errors = frozen_error_stream(
        model,
        test,
        history_df=calibration,
    )

    values = errors["error"].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Error stream contains NaN or infinite values."
        )

    return errors


def draw_truth(ax, drift_type, truth):
    """
    Add the synthetic ground truth to the detector timeline.
    """
    if drift_type == "none":
        return

    if drift_type == "gradual":
        drift_start, drift_end = truth
        ax.axvspan(
            drift_start,
            drift_end,
            alpha=0.15,
            label="True gradual drift window",
        )
        ax.axvline(
            drift_start,
            linestyle="--",
            linewidth=1.5,
        )
        ax.axvline(
            drift_end,
            linestyle="--",
            linewidth=1.5,
        )
        return

    for index, cp in enumerate(truth):
        ax.axvline(
            cp,
            linestyle="--",
            linewidth=1.5,
            label=(
                "True changepoint"
                if index == 0
                else None
            ),
        )


def make_figure(drift_type):
    data, full_truth = make_feature_series(
        kind=drift_type,
        n=N_OBSERVATIONS,
        seed=SEED,
    )

    truth = to_test_relative(
        full_truth
    )

    errors = build_error_stream(
        data
    )

    detector_results = run_all_detectors(
        errors["error"].to_numpy(
            dtype=float
        )
    )

    detector_names = [
        "adwin",
        "kswin",
        "page_hinkley",
    ]

    fig, ax = plt.subplots(
        figsize=(14, 4.8)
    )

    y_positions = {
        "adwin": 3,
        "kswin": 2,
        "page_hinkley": 1,
    }

    for name in detector_names:
        detections = detector_results[
            name
        ]["changepoints"]

        if detections:
            ax.eventplot(
                detections,
                lineoffsets=y_positions[name],
                linelengths=0.65,
                linewidths=1.0,
            )

    draw_truth(
        ax,
        drift_type,
        truth,
    )

    ax.set_xlim(
        0,
        len(errors),
    )

    ax.set_ylim(
        0.4,
        3.6,
    )

    ax.set_yticks(
        [
            y_positions["page_hinkley"],
            y_positions["kswin"],
            y_positions["adwin"],
        ]
    )

    ax.set_yticklabels(
        [
            "Page-Hinkley",
            "KSWIN",
            "ADWIN",
        ]
    )

    ax.set_xlabel(
        "Detector-test observation index"
    )

    ax.set_title(
        f"Synthetic detection timeline: "
        f"{drift_type} drift (seed={SEED})"
    )

    if drift_type != "none":
        ax.legend(
            loc="upper right"
        )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"detection_{drift_type}_seed{SEED}.png"
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
        f"Saved: {output_path}"
    )


def main():
    print(
        "Generating synthetic detection figures"
    )
    print(
        f"Representative seed: {SEED}"
    )
    print(
        "No metrics will be written to runs.csv."
    )

    for drift_type in DRIFT_TYPES:
        make_figure(
            drift_type
        )

    print(
        "Figure generation complete."
    )


if __name__ == "__main__":
    main()
