"""
Baseline degradation experiment for F1.

This experiment reproduces the forecasting and rolling-MAE logic from
03_F1_Baseline_Degradation.ipynb.

Outputs
-------
1. results/F1_rolling_mae.csv
   One row per timestamp, region, and baseline model.
   This file stores the complete 7-day rolling-MAE series required
   to reproduce F1.

2. results/runs.csv
   Long-format experiment results.
   Logs event-specific MAE for:
       pre-drift
       drift
       post-drift

The baseline models are fitted on the fixed training split only and
remain frozen while forecasting chronologically through the test stream.

This script does not generate F1.
Figure generation is handled separately by:
    experiments/degradation/generate_f1_degradation_figure.py
"""

from pathlib import Path
import sys
import json
import hashlib
import time

import numpy as np
import pandas as pd


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_PATH = PROJECT_ROOT / "data" / "raw"
RUNS_PATH = PROJECT_ROOT / "results" / "runs.csv"
ROLLING_OUTPUT_PATH = PROJECT_ROOT / "results" / "F1_rolling_mae.csv"


# Project imports

from drift_lab.data_loader import load_and_split_aemo

from drift_lab.forecasters import (
    SeasonalNaiveForecaster,
    SARIMAXForecaster,
    GradientBoostedForecaster,
    frozen_error_stream,
)

from drift_lab.evaluation import (
    calculate_mae,
    calculate_rolling_mae,
)

from drift_lab.run_logger import (
    ensure_runs_file,
    log_run,
)


# Experiment protocol

REGIONS = (
    "SA1",
    "NSW1",
)

GROUP = "degradation"

DATASET = "AEMO_TOTALDEMAND"

SPLIT_ID = (
    "aemo_sa1_nsw1_fixed_split_2018_2023_frozen_baselines_v1"
)

SEED = 0

ROLLING_WINDOW_DAYS = 7
OBSERVATIONS_PER_DAY = 48
ROLLING_WINDOW = OBSERVATIONS_PER_DAY * ROLLING_WINDOW_DAYS


# Baseline model configurations

MODEL_CONFIGS = {
    "seasonal_naive": {
        "display_name": "Seasonal Naive",
        "config": {
            "season": 48,
        },
    },

    "sarimax": {
        "display_name": "SARIMAX",
        "config": {},
    },

    "gradient_boosted": {
        "display_name": "Gradient Boosted",
        "config": {
            "feature_mode": "calendar+lags",
            "lags": [48, 336],
            "seed": 0,
        },
    },
}


# Documented AEMO events

EVENTS = {
    "SA1": [
        {
            "event_id": "covid_2020",
            "label": "COVID-19 disruption period",
            "start": "2020-03-01",
            "end": "2020-04-30",
        },
        {
            "event_id": "minimum_operational_demand_2021",
            "label": "SA minimum operational demand intervention",
            "start": "2021-03-14",
            "end": "2021-03-14",
        },
        {
            "event_id": "winter_maximum_demand_2021",
            "label": "SA record-high winter maximum demand",
            "start": "2021-07-22",
            "end": "2021-07-22",
        },
        {
            "event_id": "five_minute_settlement_2021",
            "label": "Five-Minute Settlement",
            "start": "2021-10-01",
            "end": "2021-10-01",
        },
        {
            "event_id": "record_minimum_demand_2021",
            "label": "SA record minimum operational demand",
            "start": "2021-11-21",
            "end": "2021-11-21",
        },
        {
            "event_id": "nem_market_suspension_2022",
            "label": "NEM Market Suspension",
            "start": "2022-06-15",
            "end": "2022-06-24",
        },
        {
            "event_id": "sa_security_directions_2023",
            "label": "SA system-security directions",
            "start": "2023-01-07",
            "end": "2023-02-25",
        },
    ],

    "NSW1": [
        {
            "event_id": "covid_2020",
            "label": "COVID-19 demand impact",
            "start": "2020-03-01",
            "end": "2020-04-30",
        },
        {
            "event_id": "five_minute_settlement_2021",
            "label": "Five-Minute Settlement",
            "start": "2021-10-01",
            "end": "2021-10-01",
        },
        {
            "event_id": "nem_market_suspension_2022",
            "label": "NEM Market Suspension",
            "start": "2022-06-15",
            "end": "2022-06-24",
        },
        {
            "event_id": "nsw_generator_direction_2023",
            "label": "Direction to NSW Generator",
            "start": "2023-03-16",
            "end": "2023-03-16",
        },
        {
            "event_id": "nsw_market_suspension_2023",
            "label": "NSW Market Suspension",
            "start": "2023-03-17",
            "end": "2023-03-17",
        },
        {
            "event_id": "security_directions_2023",
            "label": "SA & NSW system-security directions",
            "start": "2023-10-27",
            "end": "2023-11-25",
        },
    ],
}


# Helper functions

def config_id(config):
    """
    Build a readable and reproducible model configuration identifier.
    """

    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:10]

    readable_parts = []

    for key, value in sorted(config.items()):
        readable_parts.append(
            f"{key}={value}"
        )

    readable = (
        ",".join(readable_parts)
        if readable_parts
        else "default"
    )

    return f"{readable}|sha256:{digest}"


def make_models():
    """
    Construct fresh baseline model instances using the frozen
    notebook configurations.
    """

    return {
        "seasonal_naive": SeasonalNaiveForecaster(
            season=48
        ),

        "sarimax": SARIMAXForecaster(),

        "gradient_boosted": GradientBoostedForecaster(
            feature_mode="calendar+lags",
            lags=(48, 336),
            seed=0,
        ),
    }


def build_event_windows(event):
    """
    Construct the evaluation windows for one documented event.

    pre-drift:
        7 days immediately before the documented event.

    drift:
        The documented event duration.

    post-drift:
        7 days immediately after the documented event.

    The intervals use [start, end), so the documented final event
    date is converted to an exclusive end timestamp by adding one day.
    """

    event_start = pd.Timestamp(
        event["start"]
    )

    event_end = pd.Timestamp(
        event["end"]
    )

    event_end_exclusive = (
        event_end
        + pd.Timedelta(days=1)
    )

    return {
        "pre-drift": (
            event_start - pd.Timedelta(days=7),
            event_start,
        ),

        "drift": (
            event_start,
            event_end_exclusive,
        ),

        "post-drift": (
            event_end_exclusive,
            event_end_exclusive + pd.Timedelta(days=7),
        ),
    }


def calculate_window_mae(
    stream,
    start,
    end,
):
    """
    Calculate MAE over the interval [start, end).

    Returns NaN if no forecasts occur inside the requested window.
    """

    timestamps = pd.to_datetime(
        stream["SETTLEMENTDATE"]
    )

    mask = (
        (timestamps >= start)
        & (timestamps < end)
    )

    subset = stream.loc[mask]

    if subset.empty:
        return np.nan

    return calculate_mae(
        subset["y_true"],
        subset["y_pred"],
    )


def log_metric(
    method,
    region,
    config_hash,
    metric_name,
    metric_value,
    regime,
    split_id,
    train_samples,
    wall_clock_s,
):
    """
    Append one event-regime metric to results/runs.csv.
    """

    if pd.isna(metric_value):
        return

    if regime not in {
        "pre-drift",
        "drift",
        "post-drift",
    }:
        raise ValueError(
            f"Invalid regime: {regime}"
        )

    log_run(
        method=method,
        dataset=DATASET,
        region=region,
        metric_name=metric_name,
        metric_value=float(metric_value),
        regime=regime,
        train_samples=int(train_samples),
        group=GROUP,
        seed=SEED,
        config_hash=config_hash,
        split_id=split_id,
        n_retrains=0,
        wall_clock_s=float(wall_clock_s),
        results_path=RUNS_PATH,
    )


def run_region_baselines(region):
    """
    Fit the three baseline models on the fixed training split,
    freeze them, and forecast through the test stream.

    Returns one frozen error stream for each baseline model.
    """

    _, train, calibration, test = load_and_split_aemo(
        region=region,
        data_path=DATA_PATH,
    )

    models = make_models()

    streams = {}

    for method, model in models.items():

        print(
            f"\nRunning {region} | {method}"
        )

        start_time = time.perf_counter()

        model.fit(train)

        stream = frozen_error_stream(
            model,
            test,
            history_df=calibration,
        ).copy()

        stream["SETTLEMENTDATE"] = pd.to_datetime(
            stream["SETTLEMENTDATE"]
        )

        stream["rolling_mae"] = calculate_rolling_mae(
            stream["y_true"],
            stream["y_pred"],
            window=ROLLING_WINDOW,
        ).to_numpy()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        stream["region"] = region
        stream["method"] = method

        overall_mae = calculate_mae(
            stream["y_true"],
            stream["y_pred"],
        )

        streams[method] = stream

        print(
            f"{region} | {method}: "
            f"overall MAE={overall_mae:.2f} MW | "
            f"rows={len(stream)} | "
            f"rolling window={ROLLING_WINDOW} observations | "
            f"rolling NaNs={stream['rolling_mae'].isna().sum()} | "
            f"time={elapsed:.2f}s"
        )

        model_config = MODEL_CONFIGS[
            method
        ]["config"]

        model_config_hash = config_id(
            model_config
        )

        for event in EVENTS[region]:

            windows = build_event_windows(
                event
            )

            event_split_id = (
                f"{SPLIT_ID}"
                f"|event={event['event_id']}"
            )

            for regime, (
                window_start,
                window_end,
            ) in windows.items():

                regime_mae = calculate_window_mae(
                    stream,
                    window_start,
                    window_end,
                )

                log_metric(
                    method=method,
                    region=region,
                    config_hash=model_config_hash,
                    metric_name="mae",
                    metric_value=regime_mae,
                    regime=regime,
                    split_id=event_split_id,
                    train_samples=len(train),
                    wall_clock_s=elapsed,
                )

                if not pd.isna(regime_mae):
                    print(
                        f"  {event['event_id']} | "
                        f"{regime} | "
                        f"MAE={regime_mae:.2f} MW"
                    )

    return streams


def combine_streams(
    region_results,
):
    """
    Combine all region and model streams into the source dataset
    used to reproduce F1.
    """

    frames = []

    for region, streams in region_results.items():

        for method, stream in streams.items():

            temp = stream[
                [
                    "SETTLEMENTDATE",
                    "region",
                    "method",
                    "y_true",
                    "y_pred",
                    "error",
                    "rolling_mae",
                ]
            ].copy()

            frames.append(temp)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "region",
            "method",
            "SETTLEMENTDATE",
        ]
    ).reset_index(drop=True)

    return combined


def validate_predictions(
    all_predictions,
):
    """
    Validate completeness of every region and model forecast stream.
    """

    validation = (
        all_predictions
        .groupby(
            [
                "region",
                "method",
            ]
        )
        .agg(
            total_forecasts=(
                "y_pred",
                "size",
            ),
            null_forecasts=(
                "y_pred",
                lambda x: x.isna().sum(),
            ),
            null_actuals=(
                "y_true",
                lambda x: x.isna().sum(),
            ),
            null_errors=(
                "error",
                lambda x: x.isna().sum(),
            ),
            null_rolling_mae=(
                "rolling_mae",
                lambda x: x.isna().sum(),
            ),
        )
        .reset_index()
    )

    print()
    print("=" * 90)
    print("FORECAST STREAM VALIDATION")
    print("=" * 90)

    print(
        validation.to_string(
            index=False
        )
    )

    for (
        region,
        method,
    ), stream in all_predictions.groupby(
        [
            "region",
            "method",
        ]
    ):

        timestamps = pd.to_datetime(
            stream["SETTLEMENTDATE"]
        ).sort_values()

        print()
        print(
            f"{region} | {method}"
        )

        print(
            "Forecasts:",
            len(stream),
        )

        print(
            "Unique timestamps:",
            timestamps.nunique(),
        )

        print(
            "Duplicate timestamps:",
            timestamps.duplicated().sum(),
        )

        print(
            "Missing predictions:",
            stream["y_pred"].isna().sum(),
        )

        expected_initial_nans = (
            ROLLING_WINDOW - 1
        )

        actual_initial_nans = (
            stream["rolling_mae"]
            .isna()
            .sum()
        )

        if actual_initial_nans != expected_initial_nans:
            raise ValueError(
                f"{region} | {method}: expected "
                f"{expected_initial_nans} rolling-MAE NaNs, "
                f"found {actual_initial_nans}."
            )


def save_rolling_mae(
    all_predictions,
):
    """
    Save every timestamped rolling-MAE observation required
    to regenerate F1.

    SETTLEMENTDATE is retained here because the shared runs.csv
    timestamp records experiment logging time rather than the
    timestamp of an AEMO observation.
    """

    ROLLING_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_predictions.to_csv(
        ROLLING_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Saved full F1 rolling-MAE source:"
    )

    print(
        ROLLING_OUTPUT_PATH
    )

    print(
        f"Rows: {len(all_predictions):,}"
    )


def main():

    print("=" * 100)
    print(
        "F1 BASELINE DEGRADATION EXPERIMENT"
    )
    print("=" * 100)

    print(
        f"Regions: {REGIONS}"
    )

    print(
        "Models: seasonal_naive, "
        "sarimax, gradient_boosted"
    )

    print(
        "Forecasting protocol: "
        "fit on training data only, then freeze"
    )

    print(
        f"Rolling MAE window: "
        f"{ROLLING_WINDOW_DAYS} days "
        f"({ROLLING_WINDOW} half-hourly observations)"
    )

    print(
        "Logged regimes: "
        "pre-drift, drift, post-drift"
    )

    print(
        f"runs.csv: {RUNS_PATH}"
    )

    print(
        f"Rolling source file: "
        f"{ROLLING_OUTPUT_PATH}"
    )

    print()

    ensure_runs_file(
        results_path=RUNS_PATH
    )

    region_results = {}

    for region in REGIONS:

        region_results[
            region
        ] = run_region_baselines(
            region
        )

    all_predictions = combine_streams(
        region_results
    )

    validate_predictions(
        all_predictions
    )

    save_rolling_mae(
        all_predictions
    )

    print()
    print("=" * 100)
    print(
        "BASELINE DEGRADATION EXPERIMENT COMPLETE"
    )
    print("=" * 100)

    print(
        "Event-specific pre-drift, drift, and "
        "post-drift MAE results were logged to runs.csv."
    )

    print(
        "The complete 7-day rolling-MAE series was saved "
        "to F1_rolling_mae.csv."
    )

    print(
        "No figure was generated by this experiment."
    )

    print(
        "Run generate_f1_degradation_figure.py "
        "to generate F1 from the saved results."
    )


if __name__ == "__main__":
    main()