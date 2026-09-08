"""
Shared drift detectors for the concept-drift forecasting pipeline.

Project contract:
    detector(stream) -> changepoint indices

Detectors are applied to a one-dimensional chronological stream,
typically the frozen forecasting error stream produced by
forecasters.frozen_error_stream().

Implemented detectors:
    - ADWIN
    - Page-Hinkley
    - KSWIN

Original file is located at
    https://colab.research.google.com/drive/1pTYC62TCSpIFG8yt-Ks4y2BJia5ReB0_

"""

import numpy as np

from river.drift import ADWIN, PageHinkley, KSWIN

# Stream validation
def _validate_stream(stream):
    """
    Validate and convert a detector input stream.

    Parameters
    ----------
    stream : array-like
        One-dimensional numeric values in chronological order.

    Returns
    -------
    np.ndarray
        Validated numeric stream.
    """
    values = np.asarray(stream, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            "stream must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            "stream must not be empty."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "stream must not contain NaN or infinite values."
        )

    return values

# Shared detector runner
def run_detector(stream, detector):
    """
    Feed a detector one observation at a time
    and collect detected changepoint indices.
    """
    values = _validate_stream(stream)

    changepoints = []

    for i, value in enumerate(values):
        detector.update(value)

        if detector.drift_detected:
            changepoints.append(int(i))

    return changepoints

# ADWIN
def run_adwin(
    stream,
    delta=0.002,
):
    """
    Run ADWIN over a chronological numeric stream.

    Default configuration was retained after synthetic
    benchmark validation.
    """
    detector = ADWIN(
        delta=delta,
    )

    return run_detector(
        stream,
        detector,
    )

# Page-Hinkley
def run_page_hinkley(
    stream,
    min_instances=30,
    delta=0.005,
    threshold=400.0,
    alpha=0.9999,
    mode="both",
):
    """
    Run Page-Hinkley over a chronological numeric stream.

    The default threshold was selected from synthetic benchmark
    experiments to balance detection delay, false alarms,
    missed detections, and no-drift stability.
    """
    detector = PageHinkley(
        min_instances=min_instances,
        delta=delta,
        threshold=threshold,
        alpha=alpha,
        mode=mode,
    )

    return run_detector(
        stream,
        detector,
    )

# KSWIN
def run_kswin(
    stream,
    alpha=0.001,
    window_size=170,
    stat_size=40,
    seed=12,
):
    """
    Run KSWIN over a chronological numeric stream.

    The default configuration was selected from synthetic benchmark
    experiments to prioritise zero missed detections and strong
    false-alarm control while maintaining reasonable detection delay.

    The KSWIN internal random seed is fixed separately from any
    synthetic-data generation seed.
    """
    detector = KSWIN(
        alpha=alpha,
        window_size=window_size,
        stat_size=stat_size,
        seed=seed,
    )

    return run_detector(
        stream,
        detector,
    )


# Detector registry
DETECTOR_REGISTRY = {
    "adwin": {
        "func": run_adwin,
        "config": {
            "delta": 0.0001,
        },
    },

    "page_hinkley": {
        "func": run_page_hinkley,
        "config": {
            "min_instances": 30,
            "delta": 0.005,
            "threshold": 125,
            "alpha": 0.99,
            "mode":"up",
        },
    },

    "kswin": {
        "func": run_kswin,
        "config": {
            "alpha": 5e-6,
            "window_size": 140,
            "stat_size": 30,
            "seed": 12,
        },
    },
}

# Run all detectors
def run_all_detectors(stream):
    """
    Run every registered detector on the same validated stream.

    Returns
    -------
    dict
        Detector name mapped to:
            - changepoints
            - configuration used
    """
    values = _validate_stream(stream)

    results = {}

    for name, entry in DETECTOR_REGISTRY.items():

        config = entry["config"].copy()

        changepoints = entry["func"](
            values,
            **config,
        )

        results[name] = {
            "changepoints": changepoints,
            "config": config,
        }

    return results
