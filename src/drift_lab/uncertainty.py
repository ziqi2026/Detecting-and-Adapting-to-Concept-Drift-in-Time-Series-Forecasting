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

PGS242-163 adds pure escalation-policy helpers at the end of this module.
The frozen uq() contract is unchanged: its escalate_flag stays False, and
escalation is evaluated separately as a threshold sweep.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


DEFAULT_ALPHA = 0.10
DEFAULT_GAMMA = 0.005

# Calibration-quantile tau grid: p0, p5, ..., p95, p99, p100.
ESCALATION_QUANTILES = tuple(
    [round(0.05 * step, 2) for step in range(20)] + [0.99, 1.0]
)


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


# Escalation policy (PGS242-163)
#
# escalate(t) = score(t) > tau, with a strict inequality: a step whose score
# equals tau is retained. The intended score is the interval width, which is
# fixed when the interval is issued and so is known before y_t is observed.
# The current-step nonconformity |y_t - yhat_t| must not be used as a score.

SWEEP_METRIC_COLUMNS = (
    "n_total",
    "n_escalated",
    "n_retained",
    "review_rate",
    "retained_coverage",
    "escalated_coverage",
    "retained_mean_width",
)


def _as_1d_float(values, name):
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    return array


def _validate_bounds(lower, upper):
    lower = _as_1d_float(lower, "lower")
    upper = _as_1d_float(upper, "upper")

    if len(lower) != len(upper):
        raise ValueError("lower and upper must have equal length.")

    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("lower and upper must contain only finite values.")

    if (lower > upper).any():
        raise ValueError("lower must not exceed upper.")

    return lower, upper


def _validate_scores(scores):
    scores = _as_1d_float(scores, "scores")

    if not np.isfinite(scores).all():
        raise ValueError("scores must not contain NaN or infinite values.")

    return scores


def _validate_tau(tau):
    if isinstance(tau, (bool, np.bool_)) or not np.isscalar(tau):
        raise ValueError("tau must be a real scalar.")

    tau = float(tau)

    if np.isnan(tau):
        raise ValueError(
            "tau must not be NaN; use -inf or +inf for the endpoints."
        )

    return tau


def _validate_flags(flags):
    flags = np.asarray(flags)

    if flags.ndim != 1:
        raise ValueError("flags must be one-dimensional.")

    if flags.dtype != bool:
        raise ValueError("flags must be a boolean array.")

    return flags


def _mean_where(values, mask):
    """Mean of values[mask], or NaN when the mask selects nothing."""
    if not mask.any():
        return float("nan")

    return float(values[mask].mean())


def interval_width(lower: ArrayLike, upper: ArrayLike) -> np.ndarray:
    """Return the width ``upper - lower`` of each prediction interval.

    The width is fixed when the interval is issued, before y_t is observed,
    so it is a valid ex-ante escalation score.

    Raises
    ------
    ValueError
        If the bounds differ in length, are non-finite, or lower > upper.
    """
    lower, upper = _validate_bounds(lower, upper)
    return upper - lower


def escalation_flags(scores: ArrayLike, tau: float) -> np.ndarray:
    """Return ``scores > tau`` as a boolean escalation array.

    The comparison is strict: a step whose score equals tau is retained.
    tau = -inf escalates every step; tau = +inf escalates none.

    Raises
    ------
    ValueError
        If scores contain NaN/inf or tau is NaN or not a real scalar.
    """
    scores = _validate_scores(scores)
    tau = _validate_tau(tau)
    return scores > tau


def review_rate(flags: ArrayLike) -> float:
    """Return the fraction of steps escalated: ``n_escalated / n_total``.

    Raises
    ------
    ValueError
        If flags are empty, not one-dimensional or not boolean.
    """
    flags = _validate_flags(flags)

    if len(flags) == 0:
        raise ValueError("flags must not be empty.")

    return float(flags.mean())


def retained_coverage(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    flags: ArrayLike,
) -> float:
    """Return coverage ``mean(lower <= y_true <= upper)`` on retained steps.

    Retained steps are those with ``flags == False``. Bounds are inclusive,
    matching drift_lab.evaluation.calculate_empirical_coverage. If every step
    is escalated (n_retained == 0) the coverage is undefined and NaN is
    returned, never 0 or 1.

    Raises
    ------
    ValueError
        If lengths differ, y_true is non-finite, bounds are invalid, or
        flags are not boolean.
    """
    y_true = _as_1d_float(y_true, "y_true")
    lower, upper = _validate_bounds(lower, upper)
    flags = _validate_flags(flags)

    if not (len(y_true) == len(lower) == len(flags)):
        raise ValueError(
            "y_true, lower, upper and flags must have equal length."
        )

    if not np.isfinite(y_true).all():
        raise ValueError("y_true must contain only finite values.")

    covered = (y_true >= lower) & (y_true <= upper)
    return _mean_where(covered.astype(float), ~flags)


def _with_endpoints(taus, labels, extra=None):
    """Add -inf/+inf endpoints and order the grid strict -> loose."""
    extra = extra or {}

    grid = pd.DataFrame(
        {
            "tau": np.r_[-np.inf, taus, np.inf],
            "tau_quantile": ["-inf", *labels, "+inf"],
            **{
                name: np.r_[-np.inf, values, np.inf]
                for name, values in extra.items()
            },
        }
    )

    return grid.sort_values(
        "tau", kind="mergesort"
    ).reset_index(drop=True)


def calibration_tau_grid(
    cal_scores: ArrayLike,
    quantiles: Sequence[float] = ESCALATION_QUANTILES,
) -> pd.DataFrame:
    """Build a tau grid from quantiles of calibration-window scores.

    Use calibration-window scores only, never the test stream. Quantiles use
    numpy's default linear interpolation. The grid always includes
    tau = -inf (escalate everything) and tau = +inf (escalate nothing).

    Returns
    -------
    pd.DataFrame
        Columns ``tau`` and ``tau_quantile`` (labels "-inf", "p0", "p5",
        ..., "p100", "+inf"), ordered strict -> loose (tau ascending).

    Raises
    ------
    ValueError
        If cal_scores are empty or non-finite, or quantiles are not unique
        values in [0, 1].
    """
    scores = _validate_scores(cal_scores)

    if len(scores) == 0:
        raise ValueError("cal_scores must not be empty.")

    quantiles = _as_1d_float(quantiles, "quantiles")

    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty.")

    if not np.isfinite(quantiles).all() or (
        (quantiles < 0.0) | (quantiles > 1.0)
    ).any():
        raise ValueError("quantiles must lie in [0, 1].")

    if len(np.unique(quantiles)) != len(quantiles):
        raise ValueError("quantiles must be unique.")

    quantiles = np.sort(quantiles)
    labels = [f"p{round(100.0 * q, 6):g}" for q in quantiles]

    return _with_endpoints(np.quantile(scores, quantiles), labels)


def calibration_width_tau_grid(
    calibration_width: float,
    multipliers: Sequence[float],
) -> pd.DataFrame:
    """Build a tau grid as multiples of a calibration-window interval width.

    tau = k * calibration_width for each multiplier k, plus the -inf and
    +inf endpoints. Use this when calibration-window intervals exist only as
    one reference width (for example 2 * static conformal radius).

    Returns
    -------
    pd.DataFrame
        Columns ``tau``, ``tau_quantile`` (labels "-inf", "k0.5", ...,
        "+inf") and ``tau_multiplier`` (k, with -inf/+inf at the endpoints),
        ordered strict -> loose (tau ascending).

    Raises
    ------
    ValueError
        If calibration_width is not a finite positive scalar, or multipliers
        are empty, non-finite, non-positive or not unique.
    """
    if isinstance(calibration_width, (bool, np.bool_)) or not np.isscalar(
        calibration_width
    ):
        raise ValueError("calibration_width must be a real scalar.")

    calibration_width = float(calibration_width)

    if not np.isfinite(calibration_width) or calibration_width <= 0.0:
        raise ValueError("calibration_width must be finite and positive.")

    multipliers = _as_1d_float(multipliers, "multipliers")

    if len(multipliers) == 0:
        raise ValueError("multipliers must not be empty.")

    if not np.isfinite(multipliers).all() or (multipliers <= 0.0).any():
        raise ValueError("multipliers must be finite and positive.")

    if len(np.unique(multipliers)) != len(multipliers):
        raise ValueError("multipliers must be unique.")

    multipliers = np.sort(multipliers)
    labels = [f"k{k:g}" for k in multipliers]

    return _with_endpoints(
        multipliers * calibration_width,
        labels,
        extra={"tau_multiplier": multipliers},
    )


def escalation_sweep(
    df: pd.DataFrame,
    taus: pd.DataFrame | Sequence[float],
    group_cols: Sequence[str],
    *,
    score_col: str = "score",
    y_true_col: str = "y_true",
    lower_col: str = "lower",
    upper_col: str = "upper",
) -> pd.DataFrame:
    """Evaluate the escalation rule for every group and every tau.

    Parameters
    ----------
    df : pd.DataFrame
        One row per forecast step with score, y_true, lower and upper
        columns plus the grouping columns.
    taus : pd.DataFrame or sequence of float
        Tau values. A DataFrame must have a ``tau`` column; its other
        columns (for example ``tau_quantile``) are copied to the output.
    group_cols : sequence of str
        Columns that define independent groups (may be empty).

    Returns
    -------
    pd.DataFrame
        One row per (group, tau): group columns, tau columns, then
        n_total, n_escalated, n_retained, review_rate, retained_coverage,
        escalated_coverage and retained_mean_width. Groups are sorted and
        taus run strict -> loose. Coverage/width on an empty subset is NaN.

    Raises
    ------
    ValueError
        On missing columns, NaN group keys, NaN taus or invalid inputs.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    group_cols = list(group_cols)
    value_cols = [score_col, y_true_col, lower_col, upper_col]
    missing = set(group_cols + value_cols) - set(df.columns)

    if missing:
        raise ValueError(f"df is missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("df must not be empty.")

    if group_cols and df[group_cols].isna().any().any():
        raise ValueError("Group columns must not contain missing values.")

    grid = (
        taus.copy()
        if isinstance(taus, pd.DataFrame)
        else pd.DataFrame({"tau": _as_1d_float(taus, "taus")})
    )

    if "tau" not in grid.columns or grid.empty:
        raise ValueError("taus must provide at least one tau value.")

    tau_values = [_validate_tau(tau) for tau in grid["tau"].tolist()]
    grid["tau"] = tau_values
    grid = grid.sort_values("tau", kind="mergesort").reset_index(drop=True)

    scores = _validate_scores(df[score_col])
    lower, upper = _validate_bounds(df[lower_col], df[upper_col])
    y_true = _as_1d_float(df[y_true_col], "y_true")

    if not np.isfinite(y_true).all():
        raise ValueError("y_true must contain only finite values.")

    covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
    width = upper - lower

    if group_cols:
        groups = df.groupby(group_cols, sort=True, observed=True).indices
        group_items = sorted(
            groups.items(),
            key=lambda item: item[0],
        )
    else:
        group_items = [((), np.arange(len(df)))]

    grid_records = grid.to_dict("records")
    rows = []

    for key, positions in group_items:
        key = key if isinstance(key, tuple) else (key,)
        group_values = dict(zip(group_cols, key))
        group_scores = scores[positions]
        group_covered = covered[positions]
        group_width = width[positions]

        for tau_row in grid_records:
            flags = group_scores > tau_row["tau"]
            retained = ~flags
            n_escalated = int(flags.sum())

            rows.append(
                {
                    **group_values,
                    **tau_row,
                    "n_total": len(flags),
                    "n_escalated": n_escalated,
                    "n_retained": len(flags) - n_escalated,
                    "review_rate": review_rate(flags),
                    "retained_coverage": _mean_where(
                        group_covered, retained
                    ),
                    "escalated_coverage": _mean_where(
                        group_covered, flags
                    ),
                    "retained_mean_width": _mean_where(
                        group_width, retained
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[*group_cols, *grid.columns, *SWEEP_METRIC_COLUMNS],
    )
