import numpy as np
import pytest

from drift_lab.evaluation import (
    calculate_empirical_coverage,
    calculate_mean_interval_width,
)
from drift_lab.uncertainty import (
    adaptive_prediction_intervals,
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



def test_adaptive_initial_interval_matches_static_baseline():
    residuals = np.arange(1.0, 101.0)
    forecasts = np.array([1000.0, 1000.0])
    observations = np.array([1000.0, 1000.0])

    static_lower, static_upper, _ = prediction_intervals(
        forecasts,
        residuals,
        alpha=0.1,
    )

    (
        adaptive_lower,
        adaptive_upper,
        escalate,
        alpha_path,
        radius_path,
    ) = adaptive_prediction_intervals(
        forecasts,
        observations,
        residuals,
        alpha=0.1,
        gamma=0.1,
    )

    assert adaptive_lower[0] == pytest.approx(static_lower[0])
    assert adaptive_upper[0] == pytest.approx(static_upper[0])
    assert alpha_path[0] == pytest.approx(0.1)
    assert radius_path[0] == pytest.approx(
        conformal_radius(residuals, alpha=0.1)
    )
    assert bool(escalate[0]) is False


def test_current_observation_only_affects_future_interval():
    residuals = np.arange(1.0, 101.0)
    forecasts = np.array([1000.0, 1000.0])

    covered_observations = np.array([
        1000.0,
        1000.0,
    ])

    missed_observations = np.array([
        2000.0,
        1000.0,
    ])

    covered = adaptive_prediction_intervals(
        forecasts,
        covered_observations,
        residuals,
        alpha=0.1,
        gamma=0.1,
    )

    missed = adaptive_prediction_intervals(
        forecasts,
        missed_observations,
        residuals,
        alpha=0.1,
        gamma=0.1,
    )

    covered_lower, covered_upper = covered[0], covered[1]
    missed_lower, missed_upper = missed[0], missed[1]

    # y_0 is unavailable when C_0 is constructed.
    assert covered_lower[0] == pytest.approx(missed_lower[0])
    assert covered_upper[0] == pytest.approx(missed_upper[0])

    # The different y_0 outcome is allowed to affect C_1.
    assert covered_lower[1] != pytest.approx(missed_lower[1])
    assert covered_upper[1] != pytest.approx(missed_upper[1])


def test_future_observations_do_not_change_past_intervals():
    residuals = np.arange(1.0, 101.0)
    forecasts = np.full(6, 1000.0)

    original_observations = np.array([
        1000.0,
        2000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
    ])

    changed_future = original_observations.copy()
    changed_future[3:] = np.array([
        -5000.0,
        9000.0,
        -9000.0,
    ])

    original = adaptive_prediction_intervals(
        forecasts,
        original_observations,
        residuals,
        alpha=0.1,
        gamma=0.1,
    )

    changed = adaptive_prediction_intervals(
        forecasts,
        changed_future,
        residuals,
        alpha=0.1,
        gamma=0.1,
    )

    # Changing y_3 onward cannot change C_0, C_1, C_2, or C_3.
    np.testing.assert_allclose(
        original[0][:4],
        changed[0][:4],
    )
    np.testing.assert_allclose(
        original[1][:4],
        changed[1][:4],
    )
    np.testing.assert_allclose(
        original[3][:4],
        changed[3][:4],
    )
    np.testing.assert_allclose(
        original[4][:4],
        changed[4][:4],
    )


def test_adaptive_alpha_updates_chronologically():
    residuals = np.arange(1.0, 101.0)
    forecasts = np.array([
        1000.0,
        1000.0,
        1000.0,
    ])
    observations = np.array([
        2000.0,
        1000.0,
        1000.0,
    ])

    (
        _,
        _,
        _,
        alpha_path,
        radius_path,
    ) = adaptive_prediction_intervals(
        forecasts,
        observations,
        residuals,
        alpha=0.1,
        gamma=0.1,
    )

    # C_0 uses the nominal alpha.
    assert alpha_path[0] == pytest.approx(0.1)

    # Miss at t=0 lowers alpha for C_1, making it more conservative.
    assert alpha_path[1] < alpha_path[0]
    assert radius_path[1] >= radius_path[0]

    # Coverage at t=1 raises alpha for C_2.
    assert alpha_path[2] > alpha_path[1]
    assert radius_path[2] <= radius_path[1]


def test_adaptive_invalid_inputs_are_rejected():
    residuals = np.arange(1.0, 11.0)

    with pytest.raises(ValueError):
        adaptive_prediction_intervals(
            [1.0, 2.0],
            [1.0],
            residuals,
        )

    with pytest.raises(ValueError):
        adaptive_prediction_intervals(
            [1.0],
            [np.nan],
            residuals,
        )

    with pytest.raises(ValueError):
        adaptive_prediction_intervals(
            [1.0],
            [1.0],
            residuals,
            gamma=0.0,
        )

    with pytest.raises(ValueError):
        adaptive_prediction_intervals(
            [1.0],
            [1.0],
            residuals,
            gamma=np.inf,
        )
