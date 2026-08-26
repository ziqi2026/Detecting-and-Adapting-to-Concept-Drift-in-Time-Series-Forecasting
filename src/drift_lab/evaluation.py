"""
Shared evaluation metrics for the concept-drift forecasting pipeline.

This module provides the common evaluation functions used across the
project so that all models and components are assessed consistently.

CURRENT METRICS
---------------
Forecast evaluation:

    calculate_mae()
        Calculates Mean Absolute Error (MAE) between observed and
        predicted demand.

    calculate_rolling_mae()
        Calculates rolling MAE over the forecast stream. The default
        window is 336 observations, corresponding to 7 days of
        half-hourly AEMO data.

The rolling MAE is used to evaluate degradation of the frozen baseline
forecasters over the test stream for F1.

INPUT VALIDATION
----------------
The implemented forecast metrics verify that:

    - y_true and y_pred have equal lengths.
    - y_true and y_pred contain no missing values.

PLANNED METRICS
---------------
Detection metrics will be added in Step 4, including:

    - detection delay
    - false alarms per 10,000 observations
    - missed detections
    - precision for matching AEMO detections to documented events

Uncertainty metrics will be added in Step 6, including:

    - empirical coverage
    - mean interval width
    - normalised interval width
    - worst 24-hour coverage

This module calculates evaluation metrics only. Experiment results are
recorded separately through the shared runs.csv logging component.
"""


import pandas as pd

# Forecast evaluation metrics

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) using the shared
    project evaluation logic.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    y_pred : array-like
        Forecast values.

    Returns
    -------
    float
        Mean absolute error.
    """

    y_true = pd.Series(y_true, dtype=float)
    y_pred = pd.Series(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError(
            "y_true and y_pred must not contain missing values."
        )

    return float(
        (y_true - y_pred).abs().mean()
    )


def calculate_rolling_mae(
    y_true,
    y_pred,
    window=48 * 7
):
    """
    Calculate rolling Mean Absolute Error.

    Default:
        48 half-hour intervals × 7 days = 336 observations.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    y_pred : array-like
        Forecast values.

    window : int
        Number of observations in the rolling window.

    Returns
    -------
    pd.Series
        Rolling MAE.
    """

    y_true = pd.Series(y_true, dtype=float)
    y_pred = pd.Series(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError(
            "y_true and y_pred must not contain missing values."
        )

    absolute_error = (
        y_true - y_pred
    ).abs()

    return absolute_error.rolling(
        window=window
    ).mean()




# Detection evaluation metrics
# To be implemented in Step 4 (Detection)
# Detection delay:
# Number of time steps/days between a true changepoint
# and the corresponding detected changepoint.

# False alarms per 10,000 steps:
# Number of detections that do not correspond to a true
# changepoint, normalised per 10,000 observations.

# Missed detections:
# Number of true changepoints that the detector fails to detect.

# Precision:
# Proportion of detections that match documented events.
# Used for AEMO evaluation, where events are documented
# historical events rather than ground-truth changepoints.



# Uncertainty evaluation metrics
# To be implemented in Step 6 (Uncertainty)
# Empirical coverage:
# Fraction of observed values that fall inside the
# prediction interval.

# Mean interval width:
# Average width of the prediction intervals.

# Normalised interval width:
# Interval width scaled to make uncertainty widths
# comparable across settings.

# Worst 24-hour coverage:
# Lowest empirical coverage observed over a rolling
# 24-hour window.