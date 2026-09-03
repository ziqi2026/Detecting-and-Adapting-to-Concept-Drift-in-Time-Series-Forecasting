"""
PatchTST synthetic forecasting-degradation experiment.

PURPOSE
-------
Evaluate whether the frozen PatchTST forecasting model experiences
forecast-error degradation when exposed to controlled synthetic concept
drift.

This experiment is PatchTST-specific and is separate from the F1 AEMO
forecasting-degradation experiment.

It answers the controlled question:

    Does a frozen PatchTST forecaster exhibit measurable forecasting
    degradation when exposed to known, controlled synthetic concept
    changes?

Motivation
----------
PatchTST is also evaluated on the real AEMO test stream. However,
documented AEMO events are contextual reference points rather than
known ground-truth concept-drift changepoints, and the observed changes
may not be sufficiently strong to produce clear forecasting
degradation.

Because PatchTST uses recent causally available observations as its
forecasting context, it may remain relatively stable under some
real-world changes even though its learned model parameters remain
frozen.

This synthetic experiment therefore provides a controlled stress test
with known programmed concept changes. It examines whether PatchTST
forecasting error increases when the underlying concept is deliberately
changed, while keeping the fitted model parameters frozen.

EXPERIMENT PROTOCOL
-------------------
Synthetic stream:

    observations 0:4000
        training

    observations 4000:5000
        calibration/history

    observations 5000:20000
        chronological test stream

PatchTST is:

    - fitted on the training split only
    - not retrained during the test stream
    - supplied only causally available historical observations
    - evaluated through the shared frozen_error_stream() interface

The experiment covers:

    - no drift
    - sudden drift
    - gradual drift
    - recurring drift


SHARED EVALUATION
-----------------
Forecast metrics are calculated using drift_lab.evaluation.

    calculate_mae()
        Calculates MAE from y_true and y_pred.

    calculate_rolling_mae()
        Calculates the chronological 7-day rolling MAE.

The rolling and local evaluation window is:

    48 half-hourly observations/day × 7 days = 336 observations.

Metric implementations are therefore shared with the rest of the
forecasting pipeline rather than reimplemented inside this experiment.


DEGRADATION WINDOWS
-------------------
Sudden and recurring point changes:

    pre
        one week immediately before the true changepoint

    drift
        first week immediately after the changepoint

    post
        second week after the changepoint

The second post-change week is retained because PatchTST uses recent
observed demand as causal context. Its context can therefore gradually
fill with observations from the new regime while model parameters
remain frozen.


Gradual drift:

    pre
        one week before the gradual transition begins

    drift
        complete programmed transition window

    post
        first week after the transition ends


No-drift control:

    compares an early stable week with a late stable week.

This provides a negative control for checking whether forecasting error
changes substantially in the absence of programmed concept drift.


SEEDS
-----
The quantitative experiment is run using seeds:

    1, 2, 3, 4, 5

Seed-level results are retained in the output CSV.

Mean and standard deviation across seeds are printed at the end of the
experiment.


REPRESENTATIVE FIGURE
---------------------
Only one figure is saved.

Representative seed:

    seed = 1

The figure contains four panels:

    - no drift
    - sudden drift
    - gradual drift
    - recurring drift

The representative seed is used for visualisation only. Quantitative
results remain based on all five seeds.

The representative seed-1 forecast streams are retained during the
main experiment, so PatchTST is not trained again solely for plotting.


OUTPUTS
-------
1. Seed-level quantitative results:

    results/tables/degradation/
        patchtst_synthetic_degradation_results.csv

2. One representative PatchTST figure:

    results/figures/degradation/
        PatchTST_synthetic_degradation_seed1.png

This experiment does not write to the shared runs.csv because it is a
controlled PatchTST-specific diagnostic experiment rather than one of
the required AEMO deliverable tables.
"""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

# Project imports
from drift_lab.synthetic_generator import make_feature_series

from drift_lab.forecasters import (
    MODEL_REGISTRY,
    frozen_error_stream,
)

from drift_lab.evaluation import (
    calculate_mae,
    calculate_rolling_mae,
)

# Experiment configuration
MODEL_NAME = "patchtst"

N_OBSERVATIONS = 20_000

TRAIN_END = 4_000
CALIBRATION_END = 5_000
TEST_START = 5_000

OBSERVATIONS_PER_DAY = 48

EVALUATION_WINDOW = (
    7 * OBSERVATIONS_PER_DAY
)

ROLLING_WINDOW = (
    7 * OBSERVATIONS_PER_DAY
)

DRIFT_TYPES = (
    "none",
    "sudden",
    "gradual",
    "recurring",
)

SEEDS = (
    1,
    2,
    3,
    4,
    5,
)

REPRESENTATIVE_SEED = 1

# Output paths
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "degradation"
    / "patchtst_synthetic_degradation_results.csv"
)

FIGURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "degradation"
    / f"PatchTST_synthetic_degradation_seed{REPRESENTATIVE_SEED}.png"
)

# Model construction
def make_patchtst():
    """
    Construct one fresh PatchTST forecaster from MODEL_REGISTRY.
    """

    if MODEL_NAME not in MODEL_REGISTRY:
        raise ValueError(
            f"{MODEL_NAME!r} is not registered. "
            f"Available models: {list(MODEL_REGISTRY)}"
        )

    spec = MODEL_REGISTRY[
        MODEL_NAME
    ]

    model_class = spec[
        "class"
    ]

    config = dict(
        spec["config"]
    )

    return model_class(
        **config
    )

# Synthetic-data utilities
def split_synthetic_data(
    data,
):
    """
    Apply the fixed synthetic train/calibration/test split.
    """

    train = data.iloc[
        :TRAIN_END
    ].copy()

    calibration = data.iloc[
        TRAIN_END:CALIBRATION_END
    ].copy()

    test = data.iloc[
        TEST_START:
    ].copy()

    return (
        train,
        calibration,
        test,
    )


def to_test_relative(
    full_changepoints,
):
    """
    Convert full-series changepoints to test-stream-relative indices.
    """

    relative = []

    for cp in full_changepoints:

        if cp < TEST_START:
            continue

        relative.append(
            int(
                cp - TEST_START
            )
        )

    return relative


def safe_stream_slice(
    stream,
    start,
    end,
):
    """
    Return a safe positional slice of a forecast stream.
    """

    start = max(
        0,
        int(start),
    )

    end = min(
        len(stream),
        int(end),
    )

    if start >= end:
        return stream.iloc[
            0:0
        ].copy()

    return stream.iloc[
        start:end
    ].copy()


def window_mae(
    stream,
):
    """
    Calculate MAE for one forecast-stream window using the shared
    project evaluation function.

    Returns NaN for an empty window.
    """

    if stream.empty:
        return np.nan

    return calculate_mae(
        stream["y_true"],
        stream["y_pred"],
    )

# Degradation calculations
def degradation_values(
    reference_mae,
    changed_mae,
):
    """
    Return absolute and percentage change in MAE.

    Positive:
        forecasting error increased.

    Zero:
        no change.

    Negative:
        forecasting error decreased.
    """

    if (
        not np.isfinite(reference_mae)
        or not np.isfinite(changed_mae)
    ):
        return (
            np.nan,
            np.nan,
        )

    difference = (
        changed_mae
        - reference_mae
    )

    if reference_mae == 0:
        percentage = np.nan

    else:
        percentage = (
            100.0
            * difference
            / reference_mae
        )

    return (
        float(difference),
        float(percentage),
    )


def evaluate_point_changepoint(
    stream,
    cp,
    event_name,
):
    """
    Evaluate sudden or recurring point drift.

    pre:
        one week immediately before the changepoint

    drift:
        first week immediately after the changepoint

    post:
        second week after the changepoint
    """

    pre = safe_stream_slice(
        stream,
        cp - EVALUATION_WINDOW,
        cp,
    )

    immediate_post = safe_stream_slice(
        stream,
        cp,
        cp + EVALUATION_WINDOW,
    )

    settled_post = safe_stream_slice(
        stream,
        cp + EVALUATION_WINDOW,
        cp + 2 * EVALUATION_WINDOW,
    )

    pre_mae = window_mae(
        pre
    )

    immediate_mae = window_mae(
        immediate_post
    )

    settled_mae = window_mae(
        settled_post
    )

    immediate_delta, immediate_pct = (
        degradation_values(
            pre_mae,
            immediate_mae,
        )
    )

    settled_delta, settled_pct = (
        degradation_values(
            pre_mae,
            settled_mae,
        )
    )

    return {
        "event": event_name,
        "pre_mae": pre_mae,
        "drift_mae": immediate_mae,
        "post_mae": settled_mae,
        "degradation_absolute": immediate_delta,
        "degradation_percent": immediate_pct,
        "settled_degradation_absolute": settled_delta,
        "settled_degradation_percent": settled_pct,
    }


def evaluate_gradual_drift(
    stream,
    start,
    end,
):
    """
    Evaluate one programmed gradual transition.

    pre:
        one week before transition start

    drift:
        complete gradual-transition window

    post:
        first week after transition end
    """

    pre = safe_stream_slice(
        stream,
        start - EVALUATION_WINDOW,
        start,
    )

    drift = safe_stream_slice(
        stream,
        start,
        end,
    )

    post = safe_stream_slice(
        stream,
        end,
        end + EVALUATION_WINDOW,
    )

    pre_mae = window_mae(
        pre
    )

    drift_mae = window_mae(
        drift
    )

    post_mae = window_mae(
        post
    )

    drift_delta, drift_pct = (
        degradation_values(
            pre_mae,
            drift_mae,
        )
    )

    post_delta, post_pct = (
        degradation_values(
            pre_mae,
            post_mae,
        )
    )

    return {
        "event": "gradual_transition",
        "pre_mae": pre_mae,
        "drift_mae": drift_mae,
        "post_mae": post_mae,
        "degradation_absolute": drift_delta,
        "degradation_percent": drift_pct,
        "settled_degradation_absolute": post_delta,
        "settled_degradation_percent": post_pct,
    }


def evaluate_no_drift(
    stream,
):
    """
    Evaluate the no-drift negative control.

    An early stable week is compared with a late stable week.
    """

    early_start = (
        EVALUATION_WINDOW
    )

    early = safe_stream_slice(
        stream,
        early_start,
        early_start + EVALUATION_WINDOW,
    )

    late = safe_stream_slice(
        stream,
        len(stream) - EVALUATION_WINDOW,
        len(stream),
    )

    early_mae = window_mae(
        early
    )

    late_mae = window_mae(
        late
    )

    delta, pct = degradation_values(
        early_mae,
        late_mae,
    )

    return {
        "event": "no_drift_control",
        "pre_mae": early_mae,
        "drift_mae": late_mae,
        "post_mae": np.nan,
        "degradation_absolute": delta,
        "degradation_percent": pct,
        "settled_degradation_absolute": np.nan,
        "settled_degradation_percent": np.nan,
    }

# Evaluate one synthetic drift scenario
def evaluate_scenario(
    drift_type,
    seed,
):
    """
    Run one PatchTST synthetic degradation experiment.

    Returns
    -------
    rows : list[dict]
        Seed-level quantitative results.

    representative : dict or None
        Forecast stream and changepoints retained only when this run
        corresponds to REPRESENTATIVE_SEED.
    """

    print()

    print(
        "=" * 90
    )

    print(
        f"Model={MODEL_NAME} | "
        f"drift={drift_type} | "
        f"seed={seed}"
    )

    # Generate controlled synthetic stream.
    data, full_truth = (
        make_feature_series(
            kind=drift_type,
            n=N_OBSERVATIONS,
            seed=seed,
        )
    )

    (
        train,
        calibration,
        test,
    ) = split_synthetic_data(
        data
    )

    truth = to_test_relative(
        full_truth
    )

    # Construct and fit a fresh PatchTST.
    model = make_patchtst()

    start_time = (
        time.perf_counter()
    )

    model.fit(
        train
    )

    # Chronological frozen forecasting.
    stream = frozen_error_stream(
        model,
        test,
        history_df=calibration,
    ).copy()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if not np.isfinite(
        stream["y_pred"].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "PatchTST produced non-finite forecasts."
        )

    if not np.isfinite(
        stream["error"].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Forecast error stream contains "
            "non-finite values."
        )

    # Shared rolling-MAE implementation.
    stream["rolling_mae"] = (
        calculate_rolling_mae(
            stream["y_true"],
            stream["y_pred"],
            window=ROLLING_WINDOW,
        )
        .to_numpy()
    )

    # Evaluate known synthetic drift.
    results = []

    if drift_type == "none":

        result = evaluate_no_drift(
            stream
        )

        results.append(
            result
        )

    elif drift_type == "sudden":

        if len(truth) != 1:
            raise RuntimeError(
                "Sudden drift should contain "
                "exactly one changepoint."
            )

        result = evaluate_point_changepoint(
            stream,
            truth[0],
            "sudden_change",
        )

        results.append(
            result
        )

    elif drift_type == "gradual":

        if len(truth) != 2:
            raise RuntimeError(
                "Gradual drift should return "
                "[start, end]."
            )

        result = evaluate_gradual_drift(
            stream,
            truth[0],
            truth[1],
        )

        results.append(
            result
        )

    elif drift_type == "recurring":

        if len(truth) != 2:
            raise RuntimeError(
                "Recurring drift should contain "
                "two changepoints."
            )

        first = evaluate_point_changepoint(
            stream,
            truth[0],
            "concept_1_to_2",
        )

        second = evaluate_point_changepoint(
            stream,
            truth[1],
            "concept_2_to_1",
        )

        results.extend(
            [
                first,
                second,
            ]
        )

    else:
        raise ValueError(
            f"Unknown drift type: {drift_type}"
        )

    # Add experiment metadata.
    rows = []

    for result in results:

        row = {
            "model": MODEL_NAME,
            "drift_type": drift_type,
            "seed": seed,
            "event": result["event"],
            "pre_mae": result["pre_mae"],
            "drift_mae": result["drift_mae"],
            "post_mae": result["post_mae"],
            "degradation_absolute": (
                result[
                    "degradation_absolute"
                ]
            ),
            "degradation_percent": (
                result[
                    "degradation_percent"
                ]
            ),
            "settled_degradation_absolute": (
                result[
                    "settled_degradation_absolute"
                ]
            ),
            "settled_degradation_percent": (
                result[
                    "settled_degradation_percent"
                ]
            ),
            "train_samples": len(
                train
            ),
            "test_samples": len(
                test
            ),
            "wall_clock_s": elapsed,
        }

        rows.append(
            row
        )

        print(
            f"{result['event']}: "
            f"pre={result['pre_mae']:.2f} | "
            f"drift={result['drift_mae']:.2f} | "
            f"post={result['post_mae']:.2f} | "
            f"Δ={result['degradation_absolute']:.2f} | "
            f"Δ%={result['degradation_percent']:.2f}%"
        )

    # Retain seed 1 for plotting without retraining.
    representative = None

    if seed == REPRESENTATIVE_SEED:

        representative = {
            "drift_type": drift_type,
            "stream": stream.copy(),
            "truth": list(truth),
        }

    return (
        rows,
        representative,
    )

# One representative PatchTST figure
def make_patchtst_figure(
    representative_streams,
):
    """
    Generate one PatchTST synthetic-degradation figure containing
    all four drift scenarios for REPRESENTATIVE_SEED.
    """

    missing = (
        set(DRIFT_TYPES)
        - set(representative_streams)
    )

    if missing:
        raise ValueError(
            "Representative figure is missing drift scenarios: "
            f"{sorted(missing)}"
        )

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
    )

    axes = axes.flatten()

    for ax, drift_type in zip(
        axes,
        DRIFT_TYPES,
    ):

        payload = representative_streams[
            drift_type
        ]

        stream = payload[
            "stream"
        ]

        truth = payload[
            "truth"
        ]

        x = np.arange(
            len(stream)
        )

        ax.plot(
            x,
            stream["rolling_mae"],
            linewidth=1.4,
            label="7-day rolling MAE",
        )

        if drift_type == "gradual":

            start, end = truth

            ax.axvspan(
                start,
                end,
                alpha=0.15,
                label="True gradual drift window",
            )

            ax.axvline(
                start,
                linestyle="--",
                linewidth=1.5,
            )

            ax.axvline(
                end,
                linestyle="--",
                linewidth=1.5,
            )

        else:

            for i, cp in enumerate(
                truth
            ):

                ax.axvline(
                    cp,
                    linestyle="--",
                    linewidth=1.5,
                    label=(
                        "True changepoint"
                        if i == 0
                        else None
                    ),
                )

        ax.set_title(
            drift_type.capitalize()
        )

        ax.set_xlabel(
            "Test-stream observation"
        )

        ax.set_ylabel(
            "7-day rolling MAE"
        )

        ax.grid(
            alpha=0.2
        )

        ax.legend()

    fig.suptitle(
        "Frozen PatchTST — Synthetic Forecasting Degradation "
        f"(Representative Seed {REPRESENTATIVE_SEED})",
        fontsize=16,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    fig.savefig(
        FIGURE_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()

    print(
        f"Saved representative PatchTST figure: "
        f"{FIGURE_PATH}"
    )

# Aggregate five-seed results
def aggregate_results(
    results,
):
    """
    Print mean ± standard deviation across the five synthetic seeds.
    """

    print()

    print(
        "=" * 100
    )

    print(
        "PATCHTST SYNTHETIC DEGRADATION SUMMARY"
    )

    print(
        "=" * 100
    )

    summary = (
        results
        .groupby(
            [
                "model",
                "drift_type",
                "event",
            ],
            as_index=False,
        )
        .agg(
            pre_mae_mean=(
                "pre_mae",
                "mean",
            ),
            pre_mae_sd=(
                "pre_mae",
                "std",
            ),
            drift_mae_mean=(
                "drift_mae",
                "mean",
            ),
            drift_mae_sd=(
                "drift_mae",
                "std",
            ),
            post_mae_mean=(
                "post_mae",
                "mean",
            ),
            post_mae_sd=(
                "post_mae",
                "std",
            ),
            degradation_percent_mean=(
                "degradation_percent",
                "mean",
            ),
            degradation_percent_sd=(
                "degradation_percent",
                "std",
            ),
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return summary

# Main
def main():

    print(
        "=" * 100
    )

    print(
        "PATCHTST SYNTHETIC FORECASTING-DEGRADATION EXPERIMENT"
    )

    print(
        "=" * 100
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Drift types: {DRIFT_TYPES}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Representative figure seed: "
        f"{REPRESENTATIVE_SEED}"
    )

    print(
        "Protocol: fit on training only, freeze parameters, "
        "forecast chronologically"
    )

    print(
        "Metrics: shared drift_lab.evaluation MAE and rolling MAE"
    )

    all_rows = []

    representative_streams = {}

    for drift_type in DRIFT_TYPES:

        for seed in SEEDS:

            (
                rows,
                representative,
            ) = evaluate_scenario(
                drift_type=drift_type,
                seed=seed,
            )

            all_rows.extend(
                rows
            )

            if representative is not None:

                representative_streams[
                    drift_type
                ] = representative

    results = pd.DataFrame(
        all_rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()

    print(
        f"Saved PatchTST seed-level results: "
        f"{OUTPUT_PATH}"
    )

    aggregate_results(
        results
    )

    make_patchtst_figure(
        representative_streams
    )

    print()

    print(
        "=" * 100
    )

    print(
        "PATCHTST SYNTHETIC DEGRADATION EXPERIMENT COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()