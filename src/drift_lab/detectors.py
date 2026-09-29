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

# Shared warm-up and refractory protocol
def run_detector_with_protocol(
    test_stream,
    detector,
    warmup_stream=None,
    refractory_period=0,
):
    """
    Run a detector using the shared operational protocol.

    The detector is optionally warmed on a processed calibration
    forecast-error stream before detections are recorded on the test
    stream.

    Forecast-error drift detection requires the warm-up stream to
    represent the type of errors that the detector will encounter
    during evaluation. The calibration period provides an appropriate
    chronological, out-of-sample-like error stream because the
    forecasting model has already been fitted on earlier training data,
    while the calibration observations occur strictly after that
    fitting period.

    Using calibration forecast errors for warm-up therefore gives the
    detector realistic pre-test error history without using test data.
    The calibration period also immediately precedes the test period,
    preserving the chronological detector state at the transition into
    evaluation.

    Any drift alarms raised during warm-up are ignored. The same
    detector instance, including its internal state, is then continued
    into the test stream, where detections are recorded.

    An optional refractory period can suppress repeated alarms that
    occur too soon after an accepted detection. The refractory period
    is expressed in observations and must be applied consistently
    across synthetic calibration and AEMO evaluation.

    The same protocol is used for synthetic data: processed calibration
    forecast errors warm the detector, while only detections from the
    synthetic test stream are evaluated.

    Parameters
    ----------
    test_stream : array-like
        Processed forecast-error stream used for evaluation.

    detector
        Initialised River drift detector.

    warmup_stream : array-like or None
        Processed calibration forecast-error stream used to initialise
        detector state before evaluation. Warm-up alarms are ignored.

    refractory_period : int, default=0
        Minimum number of observations between accepted detections.
        A value of 0 disables refractory suppression.

    Returns
    -------
    list[int]
        Accepted changepoint indices relative to the test stream.
    """

    test_values = _validate_stream(
        test_stream
    )

    if refractory_period < 0:
        raise ValueError(
            "refractory_period must be >= 0."
        )

    #  Warm-up:
    #  Feed the chronological calibration error stream to the detector
    # before evaluation begins. Signals raised here are deliberately
    # ignored; only the detector's internal state is carried forward.

    if warmup_stream is not None:

        warmup_values = _validate_stream(
            warmup_stream
        )

        for value in warmup_values:
            detector.update(
                float(value)
            )

    #  Test-period detection
    changepoints = []
    last_accepted_detection = None

    for i, value in enumerate(
        test_values
    ):

        detector.update(
            float(value)
        )

        if not detector.drift_detected:
            continue

        # First test-period detection is always accepted.
        if last_accepted_detection is None:

            changepoints.append(
                int(i)
            )

            last_accepted_detection = i

            continue

        # If refractory suppression is disabled, accept every
        # detector alarm.
        if refractory_period == 0:

            changepoints.append(
                int(i)
            )

            last_accepted_detection = i

            continue

        # Otherwise accept the alarm only when the required
        # refractory interval has elapsed since the previous
        # accepted detection.
        if (
            i - last_accepted_detection
            >= refractory_period
        ):

            changepoints.append(
                int(i)
            )

            last_accepted_detection = i

    return changepoints


# ADWIN
def run_adwin(
    stream,
    delta=1e-7,
):
    """
    Run ADWIN over a chronological numeric stream.

    The default configuration was selected through synthetic tuning
    under the shared no-drift and false-alarm constraints.
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
    delta=0.001,
    threshold=700,
    alpha=0.999,
    mode="both",
):
    """
    Run Page-Hinkley over a chronological numeric stream.

    The default configuration was selected through synthetic tuning
    under the shared no-drift and false-alarm constraints, with
    selection based on missed detections, false alarms, and
    detection delay.
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
    alpha=5e-8,
    window_size=480,
    stat_size=120,
    seed=12,
):

    """
    Run KSWIN over a chronological numeric stream.

    The default configuration was selected from synthetic tuning
    under the shared no-drift and false-alarm constraints, with
    selection based on missed detections, false alarms, and
    detection delay.

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
            "delta": 1e-7,
        },
    },

    "page_hinkley": {
        "func": run_page_hinkley,
        "config": {
            "min_instances": 30,
            "delta": 0.001,
            "threshold": 700,
            "alpha": 0.999,
            "mode": "both",
        },
    },

    "kswin": {
        "func": run_kswin,
        "config": {
            "alpha": 5e-8,
            "window_size": 480,
            "stat_size": 120,
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
