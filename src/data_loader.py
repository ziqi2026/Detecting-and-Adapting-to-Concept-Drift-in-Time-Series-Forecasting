"""
Shared AEMO data loader and preprocessing utilities for the
concept-drift forecasting pipeline.

This module provides the common data-loading and chronological splitting
logic used by all downstream project components.

DATA
----
The project uses AEMO regional TOTALDEMAND data for:

    - SA1  : South Australia
    - NSW1 : New South Wales

Monthly CSV files for each region are loaded, combined, sorted
chronologically, and checked for the required columns:

    SETTLEMENTDATE
    TOTALDEMAND

FREQUENCY STANDARDISATION
-------------------------
The AEMO data changes from 30-minute to 5-minute observations from
2021-10-01.

To maintain a consistent forecasting frequency:

    - Existing 30-minute observations before the transition are retained.
    - From 2021-10-01 00:05 onward, six consecutive 5-minute
      TOTALDEMAND observations are averaged into one 30-minute value.
    - Each resulting timestamp represents the end of its 30-minute
      interval.
    - Incomplete 30-minute groups in the 5-minute portion raise an error.

FIXED PROJECT SPLIT
-------------------
The standardised data are divided chronologically using the fixed
project split:

    Training:
        2018-01-01 to 2019-12-31

    Calibration:
        2020-01-01 to 2020-02-29

    Test stream:
        2020-03-01 onward

The test data are retained in chronological order for sequential
forecasting, drift detection, adaptation and uncertainty evaluation.

FUNCTIONS
---------
load_aemo_region()
    Loads and combines the monthly CSV files for one AEMO region.

standardize_to_30min()
    Converts the mixed-frequency AEMO series to a consistent
    30-minute demand series.

split_aemo_data()
    Applies the fixed chronological train/calibration/test split.

load_and_split_aemo()
    Runs the complete loading, standardisation and splitting pipeline
    and returns the full data, training data, calibration data and
    chronological test stream.
"""

from pathlib import Path
import pandas as pd



# FIXED PROJECT SPLIT

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2020-01-01")

CALIBRATION_START = pd.Timestamp("2020-01-01")
CALIBRATION_END = pd.Timestamp("2020-03-01")

TEST_START = pd.Timestamp("2020-03-01")

# First known 5-minute AEMO observation
FIVE_MINUTE_START = pd.Timestamp("2021-10-01 00:05:00")


def load_aemo_region(region, data_path="data"):
    """
    Load and combine all monthly AEMO CSV files for one region.

    Parameters
    ----------
    region : str
        "SA1" or "NSW1"

    data_path : str
        Path to the main data folder.

    Returns
    -------
    pd.DataFrame
        Combined raw regional dataset sorted chronologically.
    """

    region = region.upper()

    if region not in {"SA1", "NSW1"}:
        raise ValueError(
            "Region must be 'SA1' or 'NSW1'."
        )

    region_path = Path(data_path) / region

    csv_files = sorted(
        region_path.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {region_path}"
        )

    frames = [
        pd.read_csv(file)
        for file in csv_files
    ]

    data = pd.concat(
        frames,
        ignore_index=True
    )

    required_columns = {
        "SETTLEMENTDATE",
        "TOTALDEMAND"
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    data["SETTLEMENTDATE"] = pd.to_datetime(
        data["SETTLEMENTDATE"]
    )

    data = (
        data
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    return data


def standardize_to_30min(data):
    """
    Convert the AEMO demand series to a consistent
    30-minute frequency.

    Before 2021-10-01 00:05:
        original 30-minute observations are retained.

    From 2021-10-01 00:05 onward:
        six 5-minute TOTALDEMAND observations are
        averaged into one 30-minute observation.

    The resulting timestamp represents the end of
    the 30-minute interval.
    """

    data = data.copy()

    data["SETTLEMENTDATE"] = pd.to_datetime(
        data["SETTLEMENTDATE"]
    )

    data = (
        data
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    before = data[
        data["SETTLEMENTDATE"] < FIVE_MINUTE_START
    ].copy()

    after = data[
        data["SETTLEMENTDATE"] >= FIVE_MINUTE_START
    ].copy()

    after = after.set_index(
        "SETTLEMENTDATE"
    )

    counts = (
        after["TOTALDEMAND"]
        .resample(
            "30min",
            closed="right",
            label="right"
        )
        .count()
    )

    incomplete = counts[
        counts != 6
    ]

    if len(incomplete) > 0:
        raise ValueError(
            "Incomplete 30-minute intervals found "
            "in the 5-minute portion of the data."
        )

    after_30min = (
        after["TOTALDEMAND"]
        .resample(
            "30min",
            closed="right",
            label="right"
        )
        .mean()
        .to_frame()
        .reset_index()
    )

    result = pd.concat(
        [
            before[
                ["SETTLEMENTDATE", "TOTALDEMAND"]
            ],
            after_30min
        ],
        ignore_index=True
    )

    result = (
        result
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    return result


def split_aemo_data(data):
    """
    Apply the permanent chronological split.

    Train:
        2018-01-01 to 2019-12-31

    Calibration:
        2020-01-01 to 2020-02-29

    Test:
        2020-03-01 onward
    """

    train = data[
        (data["SETTLEMENTDATE"] >= TRAIN_START)
        &
        (data["SETTLEMENTDATE"] < TRAIN_END)
    ].copy()

    calibration = data[
        (data["SETTLEMENTDATE"] >= CALIBRATION_START)
        &
        (data["SETTLEMENTDATE"] < CALIBRATION_END)
    ].copy()

    test = data[
        data["SETTLEMENTDATE"] >= TEST_START
    ].copy()

    return train, calibration, test


def load_and_split_aemo(region, data_path="data"):
    """
    Load one AEMO region, standardise it to 30-minute
    resolution, and apply the fixed project split.

    Returns
    -------
    data : pd.DataFrame
        Full standardized 30-minute dataset.

    train : pd.DataFrame
        Training period.

    calibration : pd.DataFrame
        Calibration period.

    test : pd.DataFrame
        Chronological test stream.
    """

    raw_data = load_aemo_region(
        region=region,
        data_path=data_path
    )

    data = standardize_to_30min(
        raw_data
    )

    train, calibration, test = (
        split_aemo_data(data)
    )

    return data, train, calibration, test