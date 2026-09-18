import numpy as np
import pytest

from drift_lab.evaluation import (
    calculate_empirical_coverage,
    calculate_mean_interval_width,
)
from drift_lab.uncertainty import (
    conformal_radius,
    prediction_intervals,
    uq,
)


def test_conformal_radius_uses_finite_sample_rank():
    residuals = np.arange(1.0, 11.0)

    radius = conformal_radius(
        residuals,
        alpha=0.1,
    )

    # n = 10
    # k = ceil((n + 1) * 0.9) = 10
    assert radius == pytest.approx(10.0)


def test_uq_returns_frozen_scalar_contract():
    residuals = np.array([
        -1.0,
        2.0,
        -3.0,
        4.0,
        -5.0,
        6.0,
        -7.0,
        8.0,
        -9.0,
        10.0,
    ])

    lower, upper, escalate = uq(
        100.0,
        residuals,
    )

    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert isinstance(escalate, bool)

    assert lower == pytest.approx(90.0)
    assert upper == pytest.approx(110.0)
    assert escalate is False


def test_batch_intervals_match_scalar_uq():
    residuals = np.arange(1.0, 11.0)
    forecasts = np.array([
        100.0,
        200.0,
        300.0,
    ])

    lower, upper, escalate = prediction_intervals(
        forecasts,
        residuals,
    )

    for i, forecast in enumerate(forecasts):
        scalar_lower, scalar_upper, scalar_escalate = uq(
            forecast,
            residuals,
        )

        assert lower[i] == pytest.approx(scalar_lower)
        assert upper[i] == pytest.approx(scalar_upper)
        assert bool(escalate[i]) is scalar_escalate


def test_empirical_coverage():
    y_true = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    lower = np.array([
        0.0,
        1.0,
        3.5,
        3.0,
    ])

    upper = np.array([
        2.0,
        3.0,
        4.0,
        5.0,
    ])

    coverage = calculate_empirical_coverage(
        y_true,
        lower,
        upper,
    )

    assert coverage == pytest.approx(0.75)


def test_mean_interval_width():
    lower = np.array([
        0.0,
        1.0,
        2.0,
    ])

    upper = np.array([
        2.0,
        4.0,
        6.0,
    ])

    width = calculate_mean_interval_width(
        lower,
        upper,
    )

    assert width == pytest.approx(3.0)


def test_empty_calibration_residuals_are_rejected():
    with pytest.raises(ValueError):
        conformal_radius(
            [],
            alpha=0.1,
        )


def test_nonfinite_calibration_residuals_are_rejected():
    with pytest.raises(ValueError):
        conformal_radius(
            [1.0, np.nan, 2.0],
            alpha=0.1,
        )


def test_invalid_interval_bounds_are_rejected():
    with pytest.raises(ValueError):
        calculate_empirical_coverage(
            [1.0],
            [2.0],
            [0.0],
        )

    with pytest.raises(ValueError):
        calculate_mean_interval_width(
            [2.0],
            [0.0],
        )
