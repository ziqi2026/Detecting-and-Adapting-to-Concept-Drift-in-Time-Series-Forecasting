"""
Non-invasive detector diagnostics for the F2 Detection Panel.

This module performs a diagnostic-only replay using the same frozen detector
configuration and processed detector-input streams as the official detection
pipeline.

It does not modify the shared detector interface, apply preprocessing, perform
event matching, apply refractory filtering, or determine official
changepoints.

Official changepoints remain the outputs of the shared detection pipeline.
The diagnostic replay is used only to expose detector state or statistics
required for F2 visualisation.

For ADWIN, River does not expose its internal cut-point statistic or dynamic
decision bound. The diagnostic therefore records the adaptive window width
maintained by the River ADWIN instance after each update. No ADWIN decision
threshold is reconstructed.

Detector-specific preprocessing, such as partial AR(1) whitening, must be
applied upstream before calling the diagnostic replay.
"""

import numpy as np

from river.drift import ADWIN, KSWIN, PageHinkley

from drift_lab.detectors import (
    DETECTOR_REGISTRY,
    _validate_stream,
)

def _build_detector(
    detector_name,
    config,
):
    """Create a fresh detector for diagnostic replay."""

    if detector_name == "adwin":
        return ADWIN(
            **config
        )

    if detector_name == "page_hinkley":
        return PageHinkley(
            **config
        )

    if detector_name == "kswin":
        return KSWIN(
            **config
        )

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

    For ADWIN, the diagnostic records the adaptive window width exposed by River.
    This is detector state rather than an explicit decision statistic, so no
    threshold is reported.
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
            p_value = float(detector.p_value)
            if p_value <= 0.0:
                statistic = float(-np.log10(config["alpha"])+1.0)
            else:
                statistic = float(-np.log10(p_value))
        else:
            statistic=np.nan

        threshold = float(
            -np.log10(config["alpha"])
        )
        statistic_name = "negative_log10_p_value"
        threshold_name = "negative_log10_alpha"

    elif detector_name == "adwin":
        statistic = float(detector.width)
        threshold = np.nan

        statistic_name = "adaptive_window_width"
        threshold_name = "not_applicable"
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

def run_detector_diagnostic(stream, detector_name, config,warmup_stream=None,):
    """
    Run one fresh detector and collect its diagnostic trace.

    The diagnostic replay uses the same configuration and warm-up stream
    as the official detector pipeline. It records one diagnostic row for
    every TEST stream observation but does not determine official
    changepoints.

    """
    values = _validate_stream(stream)

    if detector_name not in DETECTOR_REGISTRY:
        raise ValueError(
            f"Unsupported detector: {detector_name}"
        )

    config = config.copy()

    detector = _build_detector(
        detector_name,
        config,
    )

    if warmup_stream is not None:
        warmup_values = _validate_stream(
            warmup_stream
        )

        for value in warmup_values:
            detector.update(value)
    else:
        warmup_values = np.array(
            [],
            dtype=float,
        )
    trace = []

    previous_drift = False
    kswin_observations_since_reset = (
        len(warmup_values)
        if detector_name == "kswin"
        else 0
    )

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
                "stream_index": int(index),
                "stream_value": float(value),
                "statistic": diagnostic["statistic"],
                "threshold": diagnostic["threshold"],
                "statistic_name": diagnostic["statistic_name"],
                "threshold_name": diagnostic["threshold_name"],
                "raw_drift": is_drift,
            }
        )

        previous_drift = is_drift

    return {
        "config": config,
        "trace": trace,
    }