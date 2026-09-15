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