from pathlib import Path

import pandas as pd
import pytest

from drift_lab.evaluation import (
    FULL_STREAM_BUFFER_MODES,
    assign_forecast_regimes,
    assign_full_stream_forecast_regimes,
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

    # Keep the frozen Tier 1 catalogue required by production validation.
    catalogue = pd.read_csv(Path(__file__).resolve().parents[1] / "data/events.csv")
    events = pd.concat([catalogue[catalogue.tier == 1], events], ignore_index=True)

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



# --- Full-stream buffer modes (PGS242-173) --------------------------------
#
# Toy timeline, one event on 2021-03-10, 7-day buffer:
#
#   2021-03-01 ... 03-02 | 03-03 ... 03-09 | 03-10 (event) | 03-11 ... 03-17 | 03-18 ...
#   symmetric:  pre-drift |     drift (pre buffer)  |  drift  | drift (post buffer) | post-drift
#   post_only:  pre-drift        pre-drift          |  drift  | drift (post buffer) | post-drift

def _toy_windows(events):
    """Hand-built windows with the columns build_forecast_event_windows emits."""
    rows = []
    for event_id, start, end in events:
        drift_start = pd.Timestamp(start, tz="Etc/GMT-10")
        drift_end = pd.Timestamp(end, tz="Etc/GMT-10") + pd.Timedelta(days=1)
        rows.append(
            {
                "event_id": event_id,
                "drift_start": drift_start,
                "drift_end": drift_end,
                "pre_drift_start": drift_start - pd.Timedelta(days=7),
                "post_drift_end": drift_end + pd.Timedelta(days=7),
            }
        )
    return pd.DataFrame(rows)


def _labels(timestamps, windows, **kwargs):
    out = assign_full_stream_forecast_regimes(
        pd.to_datetime(timestamps), windows, **kwargs
    )
    return out["regime"].tolist()


TOY_TIMES = [
    "2021-03-01 00:00",  # well before the event
    "2021-03-02 23:30",  # last half-hour before the symmetric buffer starts
    "2021-03-03 00:00",  # first half-hour of the symmetric pre-buffer
    "2021-03-09 23:30",  # last half-hour before the event
    "2021-03-10 00:00",  # event start
    "2021-03-10 23:30",  # last half-hour of the event day
    "2021-03-11 00:00",  # first half-hour of the post buffer
    "2021-03-17 23:30",  # last half-hour of the post buffer
    "2021-03-18 00:00",  # first half-hour after the buffer (right-open)
    "2021-03-25 00:00",
]


def test_post_only_buffers_after_the_event_only():
    windows = _toy_windows([("EVT-TOY", "2021-03-10", "2021-03-10")])

    assert _labels(TOY_TIMES, windows, buffer_mode="post_only") == [
        "pre-drift",
        "pre-drift",
        "pre-drift",
        "pre-drift",  # 03-09 23:30 is still pre-drift: no buffer before
        "drift",
        "drift",
        "drift",
        "drift",
        "post-drift",
        "post-drift",
    ]


def test_symmetric_buffers_before_and_after_the_event():
    windows = _toy_windows([("EVT-TOY", "2021-03-10", "2021-03-10")])

    assert _labels(TOY_TIMES, windows, buffer_mode="symmetric") == [
        "pre-drift",
        "pre-drift",
        "drift",  # 03-03 00:00 starts the 7-day pre buffer
        "drift",  # 03-09 23:30 is inside the pre buffer
        "drift",
        "drift",
        "drift",
        "drift",
        "post-drift",
        "post-drift",
    ]


def test_default_buffer_mode_is_post_only():
    windows = _toy_windows([("EVT-TOY", "2021-03-10", "2021-03-10")])

    assert _labels(TOY_TIMES, windows) == _labels(
        TOY_TIMES, windows, buffer_mode="post_only"
    )
    assert _labels(TOY_TIMES, windows) != _labels(
        TOY_TIMES, windows, buffer_mode="symmetric"
    )


def test_modes_differ_only_inside_the_pre_buffer():
    windows = _toy_windows([("EVT-TOY", "2021-03-10", "2021-03-10")])
    times = pd.date_range("2021-03-01", "2021-03-25", freq="30min")

    post_only = pd.Series(_labels(times, windows, buffer_mode="post_only"))
    symmetric = pd.Series(_labels(times, windows, buffer_mode="symmetric"))

    differs = post_only != symmetric
    assert differs.sum() == 7 * 48  # exactly the seven pre-buffer days
    assert set(post_only[differs]) == {"pre-drift"}
    assert set(symmetric[differs]) == {"drift"}
    assert (differs.to_numpy().nonzero()[0][[0, -1]] == [2 * 48, 9 * 48 - 1]).all()


def test_overlapping_windows_form_a_union_in_both_modes():
    # The second event starts inside the first event's post buffer, so no
    # post-drift row may appear between them.
    windows = _toy_windows(
        [
            ("EVT-A", "2021-03-10", "2021-03-10"),
            ("EVT-B", "2021-03-15", "2021-03-15"),
        ]
    )
    times = ["2021-03-14 00:00", "2021-03-20 00:00", "2021-03-23 00:00"]

    for mode in FULL_STREAM_BUFFER_MODES:
        assert _labels(times, windows, buffer_mode=mode) == [
            "drift",
            "drift",
            "post-drift",
        ]


def test_buffer_mode_is_validated_and_columns_are_required():
    windows = _toy_windows([("EVT-TOY", "2021-03-10", "2021-03-10")])

    with pytest.raises(ValueError, match="buffer_mode"):
        _labels(TOY_TIMES, windows, buffer_mode="pre_only")

    with pytest.raises(ValueError, match="missing columns"):
        _labels(
            TOY_TIMES,
            windows.drop(columns=["pre_drift_start"]),
            buffer_mode="symmetric",
        )

    # post_only does not need the pre-buffer column.
    assert _labels(
        TOY_TIMES,
        windows.drop(columns=["pre_drift_start"]),
        buffer_mode="post_only",
    )


def test_buffer_modes_with_catalogue_built_windows():
    events = pd.DataFrame(
        {
            "event_id": ["EVT-TEST"],
            "event_name": ["Test event"],
            "start_date": ["2021-03-10"],
            "end_date": ["2021-03-10"],
            "date_precision": ["day"],
            "region": ["SA1"],
            "tier": [2],
        }
    )
    catalogue = pd.read_csv(Path(__file__).resolve().parents[1] / "data/events.csv")
    events = pd.concat([catalogue[catalogue.tier == 1], events], ignore_index=True)

    windows = build_forecast_event_windows(events, region="SA1", regime_days=7)
    times = ["2021-03-05 00:00", "2021-03-12 00:00", "2021-03-20 00:00"]

    assert _labels(times, windows, buffer_mode="post_only") == [
        "pre-drift",
        "drift",
        "post-drift",
    ]
    assert _labels(times, windows, buffer_mode="symmetric") == [
        "drift",
        "drift",
        "post-drift",
    ]
