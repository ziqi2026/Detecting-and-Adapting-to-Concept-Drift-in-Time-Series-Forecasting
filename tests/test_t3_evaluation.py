import pandas as pd
import pytest

from drift_lab.evaluation import (
    assign_forecast_regimes,
    build_forecast_event_windows,
    calculate_interval_pinball_loss,
    calculate_pinball_loss,
)


def test_pinball_loss_zero_for_perfect_quantile_forecast():
    result = calculate_pinball_loss(
        [10.0, 20.0],
        [10.0, 20.0],
        quantile=0.5,
    )

    assert result == 0.0


def test_interval_pinball_loss_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="must not exceed"):
        calculate_interval_pinball_loss(
            [10.0],
            [11.0],
            [9.0],
            alpha=0.10,
        )


def test_regime_assignment_uses_drift_before_pre_or_post():
    events = pd.DataFrame(
        {
            "event_id": ["EVT-TEST"],
            "event_name": ["Test event"],
            "start_date": ["2021-01-10"],
            "end_date": ["2021-01-10"],
            "date_precision": ["day"],
            "region": ["SA1"],
            "tier": [2],
        }
    )

    windows = build_forecast_event_windows(
        events,
        region="SA1",
        regime_days=7,
    )

    assignments = assign_forecast_regimes(
        pd.to_datetime(
            [
                "2021-01-03 00:00:00",
                "2021-01-10 00:00:00",
                "2021-01-11 00:00:00",
                "2021-02-01 00:00:00",
            ]
        ),
        windows,
    )

    assert assignments["regime"].tolist() == [
        "pre-drift",
        "drift",
        "post-drift",
        "unassigned",
    ]
