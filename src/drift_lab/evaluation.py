"""
Shared evaluation metrics for the concept-drift forecasting pipeline.

This module provides the common evaluation functions used across the
project so that all models and components are assessed consistently.

CURRENT METRICS
---------------
Forecast evaluation:

    calculate_mae()
        Calculates Mean Absolute Error (MAE) between observed and
        predicted demand.

    calculate_rolling_mae()
        Calculates rolling MAE over the forecast stream. The default
        window is 336 observations, corresponding to 7 days of
        half-hourly AEMO data.

The rolling MAE is used to evaluate degradation of the frozen baseline
forecasters over the test stream for F1.

INPUT VALIDATION
----------------
The implemented forecast metrics verify that:

    - y_true and y_pred have equal lengths.
    - y_true and y_pred contain no missing values.

PLANNED METRICS
---------------
Detection metrics will be added in Step 4, including:

    - detection delay
    - false alarms per 10,000 observations
    - missed detections
    - precision for matching AEMO detections to documented events

Uncertainty metrics will be added in Step 6, including:

    - empirical coverage
    - mean interval width
    - normalised interval width
    - worst 24-hour coverage

This module calculates evaluation metrics only. Experiment results are
recorded separately through the shared runs.csv logging component.
"""


import pandas as pd

# Forecast evaluation metrics

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) using the shared
    project evaluation logic.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    y_pred : array-like
        Forecast values.

    Returns
    -------
    float
        Mean absolute error.
    """

    y_true = pd.Series(y_true, dtype=float)
    y_pred = pd.Series(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError(
            "y_true and y_pred must not contain missing values."
        )

    return float(
        (y_true - y_pred).abs().mean()
    )


def calculate_rolling_mae(
    y_true,
    y_pred,
    window=48 * 7
):
    """
    Calculate rolling Mean Absolute Error.

    Default:
        48 half-hour intervals × 7 days = 336 observations.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    y_pred : array-like
        Forecast values.

    window : int
        Number of observations in the rolling window.

    Returns
    -------
    pd.Series
        Rolling MAE.
    """

    y_true = pd.Series(y_true, dtype=float)
    y_pred = pd.Series(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError(
            "y_true and y_pred must not contain missing values."
        )

    absolute_error = (
        y_true - y_pred
    ).abs()

    return absolute_error.rolling(
        window=window
    ).mean()

# Detection evaluation metrics

def _validate_changepoints(changepoints, n_observations, name):
    """
    Validate changepoint indices and return them as sorted unique integers.
    """
    if n_observations <= 0:
        raise ValueError("n_observations must be greater than 0.")

    values = []

    for cp in changepoints:
        if isinstance(cp, bool):
            raise ValueError(f"{name} must contain integer indices.")

        cp_int = int(cp)

        if cp_int != cp:
            raise ValueError(f"{name} must contain integer indices.")

        if not 0 <= cp_int < n_observations:
            raise ValueError(
                f"{name} contains index {cp_int}, which is outside "
                f"0..{n_observations - 1}."
            )

        values.append(cp_int)

    return sorted(set(values))


def _match_detections(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=48 * 7,
    gradual_grace=None,
):
    """
    Match detected changepoints to synthetic ground truth.

    Matching rules
    --------------
    none
        There are no true drift events. Every detection is a false alarm.

    sudden / recurring
        Each true point changepoint is matched to the first unused
        detection in [true_cp, true_cp + tolerance].

    gradual
        The generator returns [drift_start, drift_end] for one gradual
        transition window, not two separate drift events. The first unused
        detection in [drift_start, drift_end + gradual_grace] is matched
        to that single drift event. Delay is measured from drift_start.

    A detection can match at most one true drift event.

    Parameters
    ----------
    detected_changepoints : array-like
        Detector output indices, relative to the detector input stream.

    true_changepoints : array-like
        Ground-truth indices, expressed in the same index system as
        detected_changepoints.

    n_observations : int
        Number of observations in the detector input stream.

    drift_type : str
        One of: "none", "sudden", "gradual", "recurring".

    tolerance : int
        Maximum allowed post-changepoint delay for sudden and recurring
        point events. Default is 336 observations (7 days of half-hourly
        data). The tolerance is configurable.

    gradual_grace : int, optional
        Additional post-window allowance for gradual drift. If None,
        the same value as tolerance is used.

    Returns
    -------
    dict
        Matching details containing:
            matched_pairs
            delays
            false_alarm_indices
            missed_true_events
            n_true_events
    """
    drift_type = str(drift_type).lower()

    if drift_type not in {
        "none",
        "sudden",
        "gradual",
        "recurring",
    }:
        raise ValueError(
            "drift_type must be 'none', 'sudden', "
            "'gradual', or 'recurring'."
        )

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    if gradual_grace is None:
        gradual_grace = tolerance

    if gradual_grace < 0:
        raise ValueError("gradual_grace must be non-negative.")

    detected = _validate_changepoints(
        detected_changepoints,
        n_observations,
        "detected_changepoints",
    )

    truth = _validate_changepoints(
        true_changepoints,
        n_observations,
        "true_changepoints",
    )

    if drift_type == "none":
        if truth:
            raise ValueError(
                "No-drift evaluation requires true_changepoints=[]"
            )

        return {
            "matched_pairs": [],
            "delays": [],
            "false_alarm_indices": detected,
            "missed_true_events": 0,
            "n_true_events": 0,
        }

    if drift_type == "sudden" and len(truth) != 1:
        raise ValueError(
            "Sudden drift requires exactly one true changepoint."
        )

    if drift_type == "gradual" and len(truth) != 2:
        raise ValueError(
            "Gradual drift requires [drift_start, drift_end]."
        )

    if drift_type == "recurring" and len(truth) != 2:
        raise ValueError(
            "Recurring drift requires exactly two true changepoints."
        )

    unused = set(detected)
    matched_pairs = []
    delays = []
    missed = 0

    if drift_type == "gradual":
        drift_start, drift_end = truth
        match_end = min(
            n_observations - 1,
            drift_end + gradual_grace,
        )

        candidates = [
            cp for cp in detected
            if cp in unused and drift_start <= cp <= match_end
        ]

        if candidates:
            detected_cp = candidates[0]
            unused.remove(detected_cp)

            matched_pairs.append(
                {
                    "true_start": drift_start,
                    "true_end": drift_end,
                    "detected": detected_cp,
                }
            )
            delays.append(detected_cp - drift_start)
        else:
            missed = 1

        n_true_events = 1

    else:
        # sudden and recurring are point-event matching cases
        for true_cp in truth:
            match_end = min(
                n_observations - 1,
                true_cp + tolerance,
            )

            candidates = [
                cp for cp in detected
                if cp in unused and true_cp <= cp <= match_end
            ]

            if candidates:
                detected_cp = candidates[0]
                unused.remove(detected_cp)

                matched_pairs.append(
                    {
                        "true": true_cp,
                        "detected": detected_cp,
                    }
                )
                delays.append(detected_cp - true_cp)
            else:
                missed += 1

        n_true_events = len(truth)

    false_alarm_indices = [
        cp for cp in detected
        if cp in unused
    ]

    return {
        "matched_pairs": matched_pairs,
        "delays": delays,
        "false_alarm_indices": false_alarm_indices,
        "missed_true_events": missed,
        "n_true_events": n_true_events,
    }


def calculate_detection_delay(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=48 * 7,
    gradual_grace=None,
):
    """
    Calculate mean detection delay in observations.

    Delay is calculated only for successfully matched true drift events.
    Returns NaN when no drift event is matched, including the no-drift
    control.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    if not result["delays"]:
        return float("nan")

    return float(
        pd.Series(
            result["delays"],
            dtype=float,
        ).mean()
    )


def calculate_false_alarms_per_10000(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=48 * 7,
    gradual_grace=None,
):
    """
    Calculate false alarms normalised per 10,000 observations.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    n_false_alarms = len(
        result["false_alarm_indices"]
    )

    return float(
        n_false_alarms
        / n_observations
        * 10000
    )


def calculate_missed_detections(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=48 * 7,
    gradual_grace=None,
):
    """
    Calculate the number of true drift events that were not detected.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    return int(
        result["missed_true_events"]
    )


def evaluate_detections(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=48 * 7,
    gradual_grace=None,
):
    """
    Evaluate detector output against synthetic ground truth.

    Returns the shared detection metrics plus matching details
    used for diagnostics and plotting.

    Notes
    -----
    Ground-truth indices and detected indices must already use the same
    coordinate system. If synthetic data is split before detection,
    convert full-series ground-truth indices to detector-stream-relative
    indices before calling this function.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    if result["delays"]:
        detection_delay = float(
            pd.Series(
                result["delays"],
                dtype=float,
            ).mean()
        )
    else:
        detection_delay = float("nan")

    false_alarms_per_10000 = float(
        len(result["false_alarm_indices"])
        / n_observations
        * 10000
    )

    return {
        "detection_delay": detection_delay,
        "false_alarms_per_10000": false_alarms_per_10000,
        "missed_detections": int(
            result["missed_true_events"]
        ),
        "matched_pairs": result["matched_pairs"],
        "false_alarm_indices": result["false_alarm_indices"],
        "delays": result["delays"],
    }


# Uncertainty evaluation metrics
# To be implemented in Step 6 (Uncertainty)
# Empirical coverage:
# Fraction of observed values that fall inside the
# prediction interval.

# Mean interval width:
# Average width of the prediction intervals.

# Normalised interval width:
# Interval width scaled to make uncertainty widths
# comparable across settings.

# Worst 24-hour coverage:
# Lowest empirical coverage observed over a rolling
# 24-hour window.