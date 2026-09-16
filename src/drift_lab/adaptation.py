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


def adaptive_forecast_stream(
    changepoints: pd.DataFrame,
    model,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    region: str,
    strategy: str,
    detector: str | None = None,
    recent_window: str = "90D",
):
    """Produce a causal forecast stream for one adaptation arm.

    The initial model is fitted on ``train``.  Test rows are forecast in
    chronological chunks.  A scheduled adaptation is applied only after
    the row at its timestamp has been forecast, so the target at that
    timestamp cannot leak into its own prediction.  Retraining uses only
    observations through that timestamp.

    Returns a tuple ``(stream, metadata)``.  The stream has the same
    columns as :func:`drift_lab.forecasters.frozen_error_stream`; metadata
    records adaptation timestamps, retraining count, and training sizes.
    """
    required = {"SETTLEMENTDATE", "TOTALDEMAND"}
    for name, frame in (("train", train), ("calibration", calibration), ("test", test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    if strategy not in {
        "never", "every_30_days", "drift_full_history", "drift_recent_window",
    }:
        raise ValueError("Unknown adaptation strategy.")

    train = train.copy()
    calibration = calibration.copy()
    test = test.copy()
    for frame in (train, calibration, test):
        frame["SETTLEMENTDATE"] = pd.to_datetime(frame["SETTLEMENTDATE"], errors="raise")
    if not test["SETTLEMENTDATE"].is_monotonic_increasing:
        raise ValueError("test must be sorted ascending by SETTLEMENTDATE.")

    # The helper intentionally receives test as its schedule domain: this
    # prevents periodic retraining during the historical training period.
    adaptation_times = _get_adaptation_times(
        changepoints,
        test,
        region=region,
        strategy=strategy,
        detector=detector,
    )
    adaptation_times = [pd.Timestamp(t) for t in adaptation_times]
    adaptation_times = sorted(set(t for t in adaptation_times if t <= test["SETTLEMENTDATE"].iloc[-1]))

    from drift_lab.forecasters import frozen_error_stream

    current_model = deepcopy(model).fit(train)
    # Calibration is forecasting context only.  It is deliberately not
    # included in any adaptation training window (the Sprint 4 protocol
    # keeps the calibration split excluded from retraining).
    observed_history = pd.concat([calibration], ignore_index=True)
    observed_test = test.iloc[0:0].copy()
    output_frames = []
    retrain_rows = [len(train)]
    previous_cutoff = None
    retraining_events = []

    for adaptation_time in adaptation_times + [None]:
        if adaptation_time is None:
            segment = test.loc[
                test["SETTLEMENTDATE"] > (previous_cutoff if previous_cutoff is not None else test["SETTLEMENTDATE"].iloc[0] - pd.Timedelta(nanoseconds=1))
            ]
        else:
            lower = previous_cutoff if previous_cutoff is not None else test["SETTLEMENTDATE"].iloc[0] - pd.Timedelta(nanoseconds=1)
            segment = test.loc[
                (test["SETTLEMENTDATE"] > lower)
                & (test["SETTLEMENTDATE"] <= adaptation_time)
            ]

        if not segment.empty:
            forecasted = frozen_error_stream(current_model, segment, history_df=observed_history)
            forecasted["retrain_indicator"] = False
            forecasted["retrain_timestamp"] = pd.NaT
            forecasted["retrain_strategy"] = strategy
            forecasted["retrain_trigger"] = detector or "scheduled"
            if previous_cutoff is not None:
                first = forecasted.index[0]
                forecasted.loc[first, "retrain_indicator"] = True
                forecasted.loc[first, "retrain_timestamp"] = previous_cutoff
            output_frames.append(forecasted)
            observed_history = pd.concat([observed_history, segment], ignore_index=True)
            observed_test = pd.concat([observed_test, segment], ignore_index=True)

        if adaptation_time is None:
            break

        # The target at adaptation_time is now observed and may be used for
        # the next forecast.  Training data selection remains centralized in
        # _get_training_data and enforces the no-future-row invariant.
        available = pd.concat([train, observed_test], ignore_index=True)
        available["SETTLEMENTDATE"] = pd.to_datetime(available["SETTLEMENTDATE"], errors="raise")
        training_data = _get_training_data(
            available,
            adaptation_time,
            strategy=strategy,
            recent_window=recent_window,
        )
        retrain_started = perf_counter()
        current_model = deepcopy(current_model).fit(training_data)
        retrain_elapsed = perf_counter() - retrain_started
        retrain_rows.append(len(training_data))
        retraining_events.append({
            "retrain_timestamp": adaptation_time,
            "strategy": strategy,
            "region": region,
            "detector": detector or "scheduled",
            "training_window_start": training_data["SETTLEMENTDATE"].min(),
            "training_window_end": training_data["SETTLEMENTDATE"].max(),
            "training_samples": len(training_data),
            "wall_clock_s": retrain_elapsed,
        })
        previous_cutoff = adaptation_time

    stream = pd.concat(output_frames, ignore_index=True) if output_frames else pd.DataFrame(
        columns=["SETTLEMENTDATE", "y_true", "y_pred", "error"]
    )
    stream["SETTLEMENTDATE"] = pd.to_datetime(stream["SETTLEMENTDATE"])
    metadata = {
        "adaptation_times": adaptation_times,
        "n_retrains": len(adaptation_times),
        "train_samples_initial": len(train),
        "train_samples_total": int(np.sum(retrain_rows)),
        "train_samples_last": int(retrain_rows[-1]),
        "retraining_events": retraining_events,
    }
    return stream, metadata
