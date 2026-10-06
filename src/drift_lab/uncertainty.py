"""Initial uncertainty-quantification utilities for the forecasting pipeline.

PGS242-141

This module implements the Sprint 4 uncertainty foundation while preserving
Sprint 1's frozen interface:

    uq(point_forecast, calibration_residuals)
        -> lower_bound, upper_bound, escalate_flag

The baseline method is a static, symmetric split-conformal-style prediction
interval calibrated only on historical calibration residuals. No test-period
future information is used to construct the interval.

The current Sprint 4 implementation intentionally leaves escalation inactive.
Adaptive conformal updates and escalation policies belong to Sprint 5.
Shared uncertainty evaluation metrics live in drift_lab.evaluation.
"""

import math

import numpy as np


DEFAULT_ALPHA = 0.10
DEFAULT_GAMMA = 0.005


def _validate_calibration_residuals(calibration_residuals):
    """Validate and convert calibration residuals to a 1-D float array."""
    values = np.asarray(calibration_residuals, dtype=float)

    if values.ndim != 1:
        raise ValueError("calibration_residuals must be one-dimensional.")

    if len(values) == 0:
        raise ValueError("calibration_residuals must not be empty.")

    if not np.isfinite(values).all():
        raise ValueError(
            "calibration_residuals must not contain NaN or infinite values."
        )

    return values


def _validate_alpha(alpha):
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1.")
    return alpha


def conformal_radius(calibration_residuals, alpha=DEFAULT_ALPHA):
    """Compute a finite-sample conformal radius from calibration residuals.

    Scores are absolute calibration residuals. For n calibration scores, the
    selected 1-indexed order-statistic rank is

        ceil((n + 1) * (1 - alpha))

    clipped to n for completeness.
    """
    residuals = _validate_calibration_residuals(calibration_residuals)
    alpha = _validate_alpha(alpha)

    scores = np.abs(residuals)
    n = len(scores)
    rank = min(math.ceil((n + 1) * (1.0 - alpha)), n)

    # np.partition uses zero-based indexing and avoids sorting the whole array.
    radius = float(np.partition(scores, rank - 1)[rank - 1])

    if not np.isfinite(radius) or radius < 0:
        raise RuntimeError("Computed conformal radius is invalid.")

    return radius


def uq(point_forecast, calibration_residuals):
    """Return a static 90% prediction interval for one point forecast.

    This function preserves the frozen Sprint 1 contract exactly.

    Returns
    -------
    lower_bound : float
    upper_bound : float
    escalate_flag : bool
        Always False in the Sprint 4 baseline. Escalation policy is deferred
        to Sprint 5.
    """
    point_forecast = float(point_forecast)
    if not np.isfinite(point_forecast):
        raise ValueError("point_forecast must be finite.")

    radius = conformal_radius(
        calibration_residuals,
        alpha=DEFAULT_ALPHA,
    )

    lower_bound = float(point_forecast - radius)
    upper_bound = float(point_forecast + radius)
    escalate_flag = False

    return lower_bound, upper_bound, escalate_flag


def prediction_intervals(
    point_forecasts,
    calibration_residuals,
    alpha=DEFAULT_ALPHA,
):
    """Vectorised companion utility for experiment runners.

    The public frozen interface remains `uq(...)`; this helper avoids
    recomputing the identical calibration radius for every test timestamp.
    """
    forecasts = np.asarray(point_forecasts, dtype=float)

    if forecasts.ndim != 1:
        raise ValueError("point_forecasts must be one-dimensional.")

    if not np.isfinite(forecasts).all():
        raise ValueError("point_forecasts must not contain NaN or infinite values.")

    radius = conformal_radius(calibration_residuals, alpha=alpha)

    lower = forecasts - radius
    upper = forecasts + radius
    escalate = np.zeros(len(forecasts), dtype=bool)

    return lower, upper, escalate


def _validate_gamma(gamma):
    """Validate the adaptive conformal learning rate."""
    gamma = float(gamma)

    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and greater than zero.")

    return gamma


def adaptive_prediction_intervals(
    point_forecasts,
    observed_values,
    calibration_residuals,
    alpha=DEFAULT_ALPHA,
    gamma=DEFAULT_GAMMA,
):
    """Generate causal adaptive conformal prediction intervals.

    At timestamp t, the interval is constructed before observed_values[t]
    is used. The realised coverage error at t updates the adaptive alpha
    only for timestamp t + 1.

    The Sprint 4 static functions remain unchanged.

    Returns
    -------
    lower : np.ndarray
    upper : np.ndarray
    escalate : np.ndarray
        False for all timestamps; escalation is handled separately.
    alpha_path : np.ndarray
        Adaptive alpha used to construct each timestamp's interval.
    radius_path : np.ndarray
        Conformal radius used at each timestamp.
    """
    forecasts = np.asarray(point_forecasts, dtype=float)
    observations = np.asarray(observed_values, dtype=float)

    if forecasts.ndim != 1:
        raise ValueError("point_forecasts must be one-dimensional.")

    if observations.ndim != 1:
        raise ValueError("observed_values must be one-dimensional.")

    if len(forecasts) != len(observations):
        raise ValueError(
            "point_forecasts and observed_values must have equal length."
        )

    if not np.isfinite(forecasts).all():
        raise ValueError(
            "point_forecasts must not contain NaN or infinite values."
        )

    if not np.isfinite(observations).all():
        raise ValueError(
            "observed_values must not contain NaN or infinite values."
        )

    residuals = _validate_calibration_residuals(
        calibration_residuals
    )
    target_alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)

    sorted_scores = np.sort(np.abs(residuals))
    n_scores = len(sorted_scores)

    # Finite-sample order statistics can represent ranks 1 through n.
    min_alpha = 1.0 / (n_scores + 1.0)
    max_alpha = n_scores / (n_scores + 1.0)

    current_alpha = float(
        np.clip(target_alpha, min_alpha, max_alpha)
    )

    lower = np.empty(len(forecasts), dtype=float)
    upper = np.empty(len(forecasts), dtype=float)
    alpha_path = np.empty(len(forecasts), dtype=float)
    radius_path = np.empty(len(forecasts), dtype=float)
    escalate = np.zeros(len(forecasts), dtype=bool)

    for index in range(len(forecasts)):
        # Construct C_t before consulting y_t.
        alpha_path[index] = current_alpha

        rank = min(
            math.ceil(
                (n_scores + 1)
                * (1.0 - current_alpha)
            ),
            n_scores,
        )

        radius = float(sorted_scores[rank - 1])
        radius_path[index] = radius

        lower[index] = forecasts[index] - radius
        upper[index] = forecasts[index] + radius

        # Only after C_t exists do we observe y_t and update alpha_(t+1).
        missed = (
            observations[index] < lower[index]
            or observations[index] > upper[index]
        )

        error = 1.0 if missed else 0.0

        current_alpha = current_alpha + gamma * (
            target_alpha - error
        )

        current_alpha = float(
            np.clip(
                current_alpha,
                min_alpha,
                max_alpha,
            )
        )

    return (
        lower,
        upper,
        escalate,
        alpha_path,
        radius_path,
    )
