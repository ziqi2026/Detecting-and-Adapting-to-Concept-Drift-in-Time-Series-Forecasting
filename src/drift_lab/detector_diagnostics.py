"""
Non-invasive detector diagnostics for the F2 Detection Panel.

This module preserves the frozen project contract:

    detector(stream) -> changepoint indices

It uses the same frozen configurations from DETECTOR_REGISTRY, but runs fresh
detector instances to collect diagnostic values alongside changepoints.
"""

import numpy as np

from river.drift import ADWIN, KSWIN, PageHinkley

from drift_lab.detectors import DETECTOR_REGISTRY, _validate_stream


def _build_detector(detector_name, config):
    """Create a fresh River detector using a frozen registry configuration."""
    if detector_name == "adwin":
        return ADWIN(**config)

    if detector_name == "page_hinkley":
        return PageHinkley(**config)

    if detector_name == "kswin":
        return KSWIN(**config)

    raise ValueError(
        f"Unsupported detector: {detector_name}"
    )

def _read_diagnostic(
    detector_name,
    detector,
    config,
    kswin_test_available=False,
):
    """
    Read a diagnostic value after one detector update.

    Page-Hinkley exposes its cumulative increase statistic and uses the frozen
    threshold directly.

    KSWIN exposes its p-value. The p-value is converted to -log10(p), so drift
    evidence increases upward and the corresponding line is -log10(alpha).
    Values are unavailable until the first valid KS test.

    River's ADWIN implementation does not expose its internal cut-point
    statistic or dynamic bound. Therefore its diagnostic is explicitly a
    binary decision trace, with 0.5 used only as a plotting boundary.
    """
    if detector_name == "page_hinkley":
        statistic = float(
            detector._sum_increase
            - detector._min_increase
        )
        threshold = float(config["threshold"])
        statistic_name = "page_hinkley_cumulative_increase"
        threshold_name = "frozen_threshold"

    elif detector_name == "kswin":
        if kswin_test_available:
            p_value = max(
                float(detector.p_value),
                np.finfo(float).tiny,
            )
            statistic = float(-np.log10(p_value))
        else:
            statistic = np.nan

        threshold = float(
            -np.log10(config["alpha"])
        )
        statistic_name = "negative_log10_p_value"
        threshold_name = "negative_log10_alpha"

    elif detector_name == "adwin":
        statistic = float(detector.drift_detected)
        threshold = 0.5
        statistic_name = "binary_drift_decision"
        threshold_name = "plotting_boundary_not_native_adwin_bound"

    else:
        raise ValueError(
            f"Unsupported detector: {detector_name}"
        )

    return {
        "statistic": statistic,
        "threshold": threshold,
        "statistic_name": statistic_name,
        "threshold_name": threshold_name,
    }

def run_detector_diagnostic(stream, detector_name):
    """
    Run one fresh detector and collect its diagnostic trace.

    Returns the same changepoint-index concept as the frozen detector
    interface, plus one diagnostic row for every stream observation.
    """
    values = _validate_stream(stream)

    if detector_name not in DETECTOR_REGISTRY:
        raise ValueError(
            f"Unsupported detector: {detector_name}"
        )

    config = (
        DETECTOR_REGISTRY[detector_name]["config"]
        .copy()
    )
    detector = _build_detector(
        detector_name,
        config,
    )

    changepoints = []
    trace = []

    previous_drift = False
    kswin_observations_since_reset = 0

    for index, value in enumerate(values):
        if detector_name == "kswin":
            if previous_drift:
                kswin_observations_since_reset = 0

            kswin_observations_since_reset += 1

        detector.update(value)

        is_drift = bool(
            detector.drift_detected
        )

        kswin_test_available = (
            detector_name == "kswin"
            and kswin_observations_since_reset
            >= config["window_size"]
        )

        diagnostic = _read_diagnostic(
            detector_name,
            detector,
            config,
            kswin_test_available=kswin_test_available,
        )

        trace.append(
            {
                "detected_index": int(index),
                "stream_value": float(value),
                "statistic": diagnostic["statistic"],
                "threshold": diagnostic["threshold"],
                "statistic_name": diagnostic["statistic_name"],
                "threshold_name": diagnostic["threshold_name"],
                "is_drift": is_drift,
            }
        )

        if is_drift:
            changepoints.append(int(index))

        previous_drift = is_drift

    return {
        "changepoints": changepoints,
        "config": config,
        "trace": trace,
    }