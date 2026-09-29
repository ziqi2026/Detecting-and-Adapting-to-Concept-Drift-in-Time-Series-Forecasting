"""Adaptation policies for concept drift in time-series forecasting.

This module provides reusable logic for the four adaptation strategies:
- never retrain,
- periodic retraining every 30 days,
- drift-triggered retraining using full available history,
- drift-triggered retraining using a recent data window.

It determines adaptation timestamps, selects causal retraining data,
and updates the forecasting model while preventing future-data leakage.

The module preserves the project interface:
    adapter(changepoints, model, data) -> updated model

Original file is located at
    https://colab.research.google.com/drive/1Ba6zxWsxUQNqg8-mGlDM7w8OlnU3GQD3
"""

from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd

#cp stands for "changepoints"


def _get_adaptation_times(
    changepoints: pd.DataFrame,
    data: pd.DataFrame,
    *,
    region: str,
    strategy: str,
    detector: str | None = None,
):
    """Return the timestamps at which the selected strategy adapts."""

    if strategy == "never":
        return []

    if strategy == "every_30_days":
        return pd.date_range(
            start=(
                data["SETTLEMENTDATE"].iloc[0]
                + pd.Timedelta(days=30)
            ),
            end=data["SETTLEMENTDATE"].iloc[-1],
            freq="30D",
        ).tolist()

    if strategy in {
        "drift_full_history",
        "drift_recent_window",
    }:
        selected = changepoints.loc[
            changepoints["region"] == region
        ].copy()

        if detector is not None:
            selected = selected.loc[
                selected["detector"] == detector
            ].copy()

        return (
            selected["SETTLEMENTDATE"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

    raise ValueError(
        "strategy must be one of: never, every_30_days, "
        "drift_full_history, drift_recent_window."
    )

def _get_training_data(
    data: pd.DataFrame,
    adaptation_time,
    *,
    strategy: str,
    recent_window: str = "90D",
):
    """Select the causal training data available at an adaptation time."""

    adaptation_time = pd.Timestamp(adaptation_time)
    data_start = data["SETTLEMENTDATE"].iloc[0]

    if adaptation_time < data_start:
        raise ValueError(
            f"Adaptation time is before available data: "
            f"{adaptation_time}"
        )

    if strategy == "drift_recent_window":
        train_start = adaptation_time - pd.Timedelta(recent_window)
    else:
        train_start = data_start
    
    # Check if any point is a leakage
    training_data = data.loc[
        (data["SETTLEMENTDATE"] >= train_start)
        & (data["SETTLEMENTDATE"] <= adaptation_time)
    ].copy()

    if training_data.empty:
        raise ValueError(
            f"No training rows available at {adaptation_time}."
        )

    # Anti-leakage check: no future observation is used.
    assert training_data["SETTLEMENTDATE"].max() <= adaptation_time

    return training_data

def adapter(
    changepoints: pd.DataFrame,
    model,
    data: pd.DataFrame,
    *,
    region: str,
    strategy: str = "never",
    detector: str | None = None,
    recent_window: str = "90D",
):
    """
    Step 5 interface.

    Parameters

    changepoints
        Unchanged changepoint DataFrame.
    model
        Existing forecaster object with a .fit(data) method.
    data
        Chronologically ordered AEMO DataFrame with SETTLEMENTDATE.
        The caller is responsible for providing the appropriate
        retraining history for the selected strategy.
    region
        AEMO region to adapt, for example "SA1"
    strategy
        "never", "every_30_days", "drift_full_history",
        or "drift_recent_window".
    detector
        Optional detector name used to select changepoints.
    recent_window
        Training window for "drift_recent_window", for example "90D".

    Returns

    Updated forecaster model. No forecasts or experiment results
    are produced here.
    """
    required_cp_columns = {
        "region",
        "detector",
        "detected_index",
        "SETTLEMENTDATE",
    }

    missing_cp_columns = required_cp_columns - set(changepoints.columns)

    if missing_cp_columns:
        raise ValueError(
            f"Missing changepoint columns: "
            f"{sorted(missing_cp_columns)}"
        )

    if "SETTLEMENTDATE" not in data.columns:
        raise ValueError("data must contain SETTLEMENTDATE.")

    if not hasattr(model, "fit"):
        raise TypeError("model must have a fit(training_data) method.")

    data = data.copy()
    data["SETTLEMENTDATE"] = pd.to_datetime(
        data["SETTLEMENTDATE"],
        errors="raise",
    )

    if not data["SETTLEMENTDATE"].is_monotonic_increasing:
        raise ValueError(
            "data must be sorted ascending by SETTLEMENTDATE."
        )

    changepoints = changepoints.copy()
    changepoints["SETTLEMENTDATE"] = pd.to_datetime(
        changepoints["SETTLEMENTDATE"],
        errors="raise",
    )

    adaptation_times = _get_adaptation_times(
        changepoints,
        data,
        region=region,
        strategy=strategy,
        detector=detector,
    )

    if strategy == "never":
        return model

    updated_model = deepcopy(model)

    for adaptation_time in adaptation_times:
        training_data = _get_training_data(
            data,
            adaptation_time,
            strategy=strategy,
            recent_window=recent_window,
        )

        updated_model.fit(training_data)

    return updated_model


def _validate_adaptation_frame(frame, name, *, allow_empty=False):
    """Validate observations before scheduling any fitting or prediction."""
    required = {"SETTLEMENTDATE", "TOTALDEMAND"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{name} needs {sorted(required)}")
    result = frame.copy().reset_index(drop=True)
    result["SETTLEMENTDATE"] = pd.to_datetime(result.SETTLEMENTDATE, errors="raise")
    times = result.SETTLEMENTDATE
    if (not allow_empty and result.empty) or times.isna().any():
        raise ValueError(f"{name} must contain valid observations")
    if times.dt.tz is not None:
        raise ValueError(f"{name} must use naive fixed-AEST market timestamps")
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError(f"{name} timestamps must be unique and chronological")
    if not np.isfinite(result.TOTALDEMAND.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} demand must be finite")
    return result


def validate_adaptation_changepoints(changepoints, test, *, region, detector):
    """Validate the selected upstream sequence without filtering its alarms."""
    required = {"region", "detector", "detected_index", "SETTLEMENTDATE"}
    if not required.issubset(changepoints.columns):
        raise ValueError(f"Changepoints need {sorted(required)}")
    if not detector:
        raise ValueError("An explicit detector is required for drift-triggered arms")
    rows = changepoints.loc[
        changepoints.region.eq(region) & changepoints.detector.eq(detector)
    ].copy()
    rows["SETTLEMENTDATE"] = pd.to_datetime(rows.SETTLEMENTDATE, errors="raise")
    indices = pd.to_numeric(rows.detected_index, errors="raise")
    if (indices.isna().any() or not np.isfinite(indices).all()
            or ((indices % 1 != 0) | (indices < 0) | (indices >= len(test))).any()):
        raise ValueError("Changepoint indices must be integers inside the test stream")
    if indices.duplicated().any() or not indices.is_monotonic_increasing:
        raise ValueError("Changepoints must be unique and chronological")
    expected = pd.DatetimeIndex(test.SETTLEMENTDATE.iloc[indices.astype(int)])
    if not pd.DatetimeIndex(rows.SETTLEMENTDATE).equals(expected):
        raise ValueError("Changepoint index/timestamp mapping does not match the test stream")
    rows["detected_index"] = indices.astype(int)
    return rows.reset_index(drop=True)


def adaptive_forecast_stream(
    changepoints, model, train, calibration, test, *, region, strategy,
    detector=None, recent_window="90D", initial_model_fitted=False,
):
    """Forecast, observe, then refit; preserve the PGS242-127 policy helpers.

    Revision R1 (same initial state): the revised runner fits once and passes
    initial_model_fitted=True. Each arm deep-copies that fitted state.
    The False default retains compatibility with older direct callers only.
    Revision R2 (window sweep): recent_window is an independent D parameter;
    B's 30-day schedule still comes from _get_adaptation_times.
    Revision R3 (causality): calibration is prediction context only, never
    part of the retraining frame. Every refit follows the forecast at t.
    """
    strategies = {"never", "every_30_days", "drift_full_history", "drift_recent_window"}
    if strategy not in strategies:
        raise ValueError("Unknown adaptation strategy")
    if region not in {"SA1", "NSW1"}:
        raise ValueError("Unsupported region")
    if strategy == "drift_recent_window":
        span = pd.Timedelta(recent_window)
        if pd.isna(span) or span <= pd.Timedelta(0):
            raise ValueError("Recent window must be positive")
    train = _validate_adaptation_frame(train, "train")
    calibration = _validate_adaptation_frame(calibration, "calibration", allow_empty=True)
    test = _validate_adaptation_frame(test, "test")
    if train.SETTLEMENTDATE.iloc[-1] >= test.SETTLEMENTDATE.iloc[0]:
        raise ValueError("Training must precede test")
    if not calibration.empty and not (
        train.SETTLEMENTDATE.iloc[-1] < calibration.SETTLEMENTDATE.iloc[0]
        and calibration.SETTLEMENTDATE.iloc[-1] < test.SETTLEMENTDATE.iloc[0]
    ):
        raise ValueError("Calibration must lie strictly between training and test")
    if not test.SETTLEMENTDATE.diff().dropna().eq(pd.Timedelta(minutes=30)).all():
        raise ValueError("The AEMO test stream must have continuous half-hour rows")
    drift = strategy.startswith("drift_")
    selected = validate_adaptation_changepoints(
        changepoints, test, region=region, detector=detector
    ) if drift else changepoints
    times = _get_adaptation_times(
        selected, test, region=region, strategy=strategy, detector=detector
    )
    from drift_lab.forecasters import frozen_error_stream

    current_model = deepcopy(model)
    if not initial_model_fitted:
        current_model.fit(train)
    initial_effective = getattr(current_model, "n_training_samples_", None)
    all_history = pd.concat([train, calibration, test], ignore_index=True)
    training_pool = pd.concat([train, test], ignore_index=True)
    test_times = pd.DatetimeIndex(test.SETTLEMENTDATE)
    events, chunks = [], []
    offset, last_event = 0, None
    trigger = "drift" if drift else ("scheduled" if strategy == "every_30_days" else "none")
    window_days = pd.Timedelta(recent_window).total_seconds() / 86400 if strategy == "drift_recent_window" else None

    for timestamp in list(times) + [None]:
        stop = len(test) if timestamp is None else int(test_times.searchsorted(timestamp, side="right"))
        segment = test.iloc[offset:stop]
        if len(segment):
            history = all_history.loc[all_history.SETTLEMENTDATE < segment.SETTLEMENTDATE.iloc[0]]
            stream = frozen_error_stream(current_model, segment, history_df=history)
            if not np.isfinite(stream[["y_true", "y_pred", "error"]].to_numpy()).all():
                raise ValueError("A forecaster returned non-finite predictions")
            stream["model_version"] = len(events)
            stream["retrain_event_id"] = last_event["event_id"] if last_event else ""
            stream["retrain_timestamp"] = last_event["retrain_timestamp"] if last_event else pd.NaT
            stream["retrain_indicator"] = False
            if last_event:
                stream.loc[stream.index[0], "retrain_indicator"] = True
            stream["retrain_strategy"] = strategy
            stream["retrain_trigger"] = trigger
            stream["detector"] = detector if drift else ""
            chunks.append(stream)
        offset = stop
        if timestamp is None:
            break

        # Revision R3: the existing helper enforces [t - window, t] for D.
        # Selecting from train + test excludes calibration targets AND lag sources.
        training = _get_training_data(
            training_pool, timestamp, strategy=strategy, recent_window=recent_window
        )
        started = perf_counter()
        updated_model = deepcopy(current_model)
        copy_s = perf_counter() - started
        started = perf_counter()
        updated_model.fit(training)
        fit_s = perf_counter() - started
        current_model = updated_model
        # Revision R4: record actual refits, including a final-row refit with
        # no subsequent forecast (first_forecast_timestamp is then missing).
        event_id = f"{region}:{strategy}:{window_days}:{len(events) + 1}"
        last_event = {
            "event_id": event_id, "retrain_timestamp": pd.Timestamp(timestamp),
            "first_forecast_timestamp": test_times[stop] if stop < len(test) else pd.NaT,
            "strategy": strategy, "region": region, "window_days": window_days,
            "trigger_type": trigger, "detector": detector if drift else "",
            "training_window_start": training.SETTLEMENTDATE.iloc[0],
            "training_window_end": training.SETTLEMENTDATE.iloc[-1],
            "training_samples": len(training),
            "effective_training_samples": getattr(current_model, "n_training_samples_", None),
            "wall_clock_s": fit_s, "model_copy_s": copy_s,
        }
        events.append(last_event)

    result = pd.concat(chunks, ignore_index=True)
    if not pd.DatetimeIndex(result.SETTLEMENTDATE).equals(test_times):
        raise RuntimeError("Forecast alignment changed")
    effective = [initial_effective] + [e["effective_training_samples"] for e in events]
    metadata = {
        "adaptation_times": list(times), "n_retrains": len(events),
        "train_samples_initial": len(train),
        "train_samples_total": len(train) + sum(e["training_samples"] for e in events),
        "train_samples_last": events[-1]["training_samples"] if events else len(train),
        "effective_train_samples_total": sum(effective) if all(x is not None for x in effective) else None,
        "retraining_wall_clock_s": sum(e["wall_clock_s"] for e in events),
        "retraining_events": events,
    }
    return result, metadata
