"""
Shared experiment-results logger for the concept-drift forecasting pipeline.

This module creates and writes the machine-readable results/runs.csv file
used to record experiment metrics across the project.

RESULTS SCHEMA
--------------
Each row stores one metric value for an experiment configuration using
the required fields:

    group
    method
    dataset
    region
    seed
    config_hash
    split_id
    metric_name
    metric_value
    regime
    n_retrains
    train_samples
    wall_clock_s
    timestamp

FUNCTIONS
---------
ensure_runs_file()
    Creates results/runs.csv with the required schema if the file does
    not already exist.

log_run()
    Appends one metric result as a new row while preserving the shared
    long-format schema.

The logger does not calculate evaluation metrics. Metric calculations
are handled separately by drift_lab.evaluation.

The regime field is intended to identify results as pre-drift, drift,
or post-drift once the corresponding evaluation windows have been
agreed and defined.

This shared results file provides the machine-readable record from
which project tables and reported results can be reproduced.
"""

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


RUNS_COLUMNS = [
    "group",
    "method",
    "dataset",
    "region",
    "seed",
    "config_hash",
    "split_id",
    "metric_name",
    "metric_value",
    "regime",
    "n_retrains",
    "train_samples",
    "wall_clock_s",
    "timestamp",
]

def ensure_runs_file(results_path="results/runs.csv"):
    """
    Create the runs.csv file with the required schema
    if it does not already exist.

    Parameters
    ----------
    results_path : str or Path
        Path to the shared experiment results file.

    Returns
    -------
    Path
        Path to the runs.csv file.
    """

    results_path = Path(results_path)

    # Create the results directory if needed
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Create an empty runs.csv only if it does not exist
    if not results_path.exists():
        empty_results = pd.DataFrame(
            columns=RUNS_COLUMNS
        )

        empty_results.to_csv(
            results_path,
            index=False
        )

    return results_path

def log_run(
    method,
    dataset,
    region,
    metric_name,
    metric_value,
    regime,
    train_samples,
    *,
    group,
    seed=0,
    config_hash="default",
    split_id="fixed_v1",
    n_retrains=0,
    wall_clock_s=0.0,
    results_path="results/runs.csv",
):
    """
    Append one experiment metric to the shared runs.csv file.
    """

    results_path = ensure_runs_file(results_path)

    row = {
        "group": group,
        "method": method,
        "dataset": dataset,
        "region": region,
        "seed": seed,
        "config_hash": config_hash,
        "split_id": split_id,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "regime": regime,
        "n_retrains": n_retrains,
        "train_samples": train_samples,
        "wall_clock_s": float(wall_clock_s),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    row_df = pd.DataFrame(
        [row],
        columns=RUNS_COLUMNS
    )

    row_df.to_csv(
        results_path,
        mode="a",
        header=False,
        index=False
    )

    return row
