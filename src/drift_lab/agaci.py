"""Aggregated Adaptive Conformal Inference (AgACI).

Experimental implementation for Team 42.

Uses the same calibration residuals and forecast streams
as the existing static conformal and ACI experiments.

This module is separate from uncertainty.py to preserve
the team's original implementation.
"""

import numpy as np

from drift_lab.uncertainty import (
    _validate_alpha,
    _validate_calibration_residuals,
)
from drift_lab.evaluation import calculate_pinball_loss

# Candidate ACI learning rates for AgACI
AGACI_GAMMAS = np.array([
    0.001,
    0.002,
    0.004,
    0.008,
    0.016,
    0.032,
    0.064,
    0.128,
])

def initialise_experts(calibration_residuals, alpha=0.10):
    """Initialise independent ACI experts for AgACI."""

    residuals = _validate_calibration_residuals(
        calibration_residuals
    )
    alpha = _validate_alpha(alpha)

    scores = np.sort(np.abs(residuals))
    n = len(scores)

    min_alpha = 1.0 / (n + 1.0)
    max_alpha = n / (n + 1.0)

    expert_alphas = np.full(
        len(AGACI_GAMMAS),
        np.clip(alpha, min_alpha, max_alpha),
        dtype=float,
    )

    return scores, expert_alphas
import math


def expert_intervals(forecast, scores, expert_alphas):
    """Generate prediction intervals for all ACI experts."""

    n = len(scores)

    ranks = np.ceil(
        (n + 1) * (1.0 - expert_alphas)
    ).astype(int)

    ranks = np.clip(ranks, 1, n)

    radii = scores[ranks - 1]

    lower = forecast - radii
    upper = forecast + radii

    return lower, upper, radii

def update_experts(
    observation,
    lower,
    upper,
    expert_alphas,
    target_alpha=0.10,
    min_alpha=0.0001,
    max_alpha=0.9999,
):
    """Update ACI experts using the observed outcome."""

    errors = (
        (observation < lower) |
        (observation > upper)
    ).astype(float)

    updated_alphas = expert_alphas + AGACI_GAMMAS * (
        target_alpha - errors
    )

    updated_alphas = np.clip(
        updated_alphas,
        min_alpha,
        max_alpha,
    )

    return updated_alphas

def initialise_boa_state():
    """Initialise BOA state following opera's implementation."""
    k = len(AGACI_GAMMAS)

    return {
        "w0": np.full(k, 1.0 / k),
        "eta": np.ones(k),
        "R": np.zeros(k),
        "R_reg": np.zeros(k),
        "V": np.zeros(k),
        "B": np.full(k, 2.0 ** -20),
    }

def calculate_boa_weights(state):
    """Calculate BOA expert weights following opera's implementation."""
    eta = state["eta"]
    w0 = state["w0"]
    R_reg = state["R_reg"]

    log_weights = np.log(eta) + np.log(w0) + eta * R_reg

    log_weights -= np.max(log_weights)

    weights = np.exp(log_weights)
    return weights / weights.sum()

def update_boa_state(state, observation, expert_bounds, tau):
    """Update BOA state using opera's gradient-based algorithm."""
    weights = calculate_boa_weights(state)
    prediction = float(np.dot(weights, expert_bounds))

    # Pinball gradient evaluated at the aggregated prediction
    gradient = pinball_gradient(prediction, observation, tau)

    # Instantaneous regret for each expert
    r = gradient * (prediction - np.asarray(expert_bounds))

    # Update adaptive learning rates
    state["B"] = np.maximum(state["B"], np.abs(r))
    B2 = 2.0 ** np.ceil(np.log2(state["B"]))
    state["V"] += r ** 2

    with np.errstate(divide="ignore"):
        eta = np.minimum(
            1.0 / B2,
            np.sqrt(np.log(1.0 / state["w0"]) / state["V"])
        )

    # Update regularised regret
    r_reg = 0.5 * (
        r - eta * r ** 2 + B2 * (eta * r > 0.5)
    )

    state["R"] += r
    state["R_reg"] += r_reg
    state["eta"] = eta

    return state

def initialise_aggregation_weights():
    """Start all ACI experts with equal aggregation weights."""

    n_experts = len(AGACI_GAMMAS)

    lower_weights = np.full(
        n_experts, 1.0 / n_experts, dtype=float
    )

    upper_weights = np.full(
        n_experts, 1.0 / n_experts, dtype=float
    )

    return lower_weights, upper_weights

def pinball_gradient(prediction, observation, tau):
    """Calculate the pinball gradient for BOA."""
    return float(observation < prediction) - tau

def update_aggregation_weights(
    observation,
    lower,
    upper,
    lower_weights,
    upper_weights,
    alpha=0.10,
    eta=0.01,
):
    """Update expert weights using observed pinball losses."""

    lower_losses = np.array([
        calculate_pinball_loss(
            [observation], [bound], alpha / 2.0
        )
        for bound in lower
    ])

    upper_losses = np.array([
        calculate_pinball_loss(
            [observation], [bound], 1.0 - alpha / 2.0
        )
        for bound in upper
    ])

    def update(weights, losses):
        log_weights = (
            np.log(weights)
            - eta * losses
        )
        log_weights -= np.max(log_weights)

        updated = np.exp(log_weights)
        return updated / updated.sum()

    return (
        update(lower_weights, lower_losses),
        update(upper_weights, upper_losses),
    )

def aggregate_intervals(
    lower,
    upper,
    lower_weights,
    upper_weights,
):
    """Combine ACI expert bounds using their current weights."""

    aggregated_lower = float(
        np.dot(lower_weights, lower)
    )

    aggregated_upper = float(
        np.dot(upper_weights, upper)
    )

    return aggregated_lower, aggregated_upper

def agaci_prediction_intervals(
    point_forecasts,
    observed_values,
    calibration_residuals,
    alpha=0.10,
    eta=0.01,
):
    """Generate causal, experimentally aggregated ACI intervals."""

    forecasts = np.asarray(point_forecasts, dtype=float)
    observations = np.asarray(observed_values, dtype=float)

    if forecasts.ndim != 1 or observations.ndim != 1:
        raise ValueError("Forecasts and observations must be 1-D.")

    if len(forecasts) != len(observations):
        raise ValueError("Forecasts and observations must have equal lengths.")

    if not np.isfinite(forecasts).all() or not np.isfinite(observations).all():
        raise ValueError("Inputs must contain finite values.")

    scores, expert_alphas = initialise_experts(
        calibration_residuals, alpha=alpha
    )

    lower_state = initialise_boa_state()
    upper_state = initialise_boa_state()

    n = len(forecasts)
    lower_path = np.empty(n)
    upper_path = np.empty(n)

    min_alpha = 1.0 / (len(scores) + 1.0)
    max_alpha = len(scores) / (len(scores) + 1.0)

    for t in range(n):
        # Step 1: All experts predict before observing y[t].
        expert_lower, expert_upper, _ = expert_intervals(
            forecasts[t], scores, expert_alphas
        )

        # Step 2: Calculate BOA weights before observing y[t].
        lower_weights = calculate_boa_weights(lower_state)
        upper_weights = calculate_boa_weights(upper_state)

        lower_path[t], upper_path[t] = aggregate_intervals(
            expert_lower,
            expert_upper,
            lower_weights,
            upper_weights,
        )

        # Step 3: Update BOA after observing y[t].
        lower_state = update_boa_state(
            lower_state,
            observations[t],
            expert_lower,
            tau=alpha / 2.0,
        )

        upper_state = update_boa_state(
            upper_state,
            observations[t],
            expert_upper,
            tau=1.0 - alpha / 2.0,
        )

        # Step 4: Update each ACI expert for the next prediction.
        expert_alphas = update_experts(
            observations[t],
            expert_lower,
            expert_upper,
            expert_alphas,
            target_alpha=alpha,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
        )

    return lower_path, upper_path
