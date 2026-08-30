"""
Frozen component interfaces for the
Concept Drift in Time Series Forecasting project.

These interfaces are fixed during Sprint 1.

The internal algorithms may change during the project,
but the inputs and outputs of these interfaces should
remain consistent.

Frozen contracts:
1. detector(stream) -> changepoint indices

2. adapter(changepoints, model, data)
   -> updated model

3. uq(point_forecast, calibration_residuals)
   -> lower_bound, upper_bound, escalate_flag

Important:
All components must operate chronologically and must
not use information from the future.
"""


def detector(stream):
    """
    Detect concept drift in a chronological numeric stream.

    Parameters
    ----------
    stream : array-like
        One-dimensional sequence of values arriving in time order.

        In this project, the stream will typically be the
        forecasting error produced by a frozen prediction model.

    Project convention
    ------------------
    For the main AEMO forecasting experiments, `stream` will be the
    chronological prediction-error stream produced by the forecasting
    model.
    Alternative experiments may monitor another numeric stream, such
    as raw demand, but this must be explicitly documented and must not
    be mixed with prediction-error results in the same comparison.

    
    Returns
    -------
    changepoints : list[int]
        Indices where the detector raises a drift alarm.

    Rules
    -----
    - The stream must be processed in chronological order.
    - No future observations may be used.
    - The returned changepoints must be integer positions.
    - Different detector algorithms may be used internally,
      such as ADWIN, Page-Hinkley, or KSWIN, but the input
      and output format must remain unchanged.
    """

    raise NotImplementedError(
        "A detector implementation must be provided."
    )

def adapter(changepoints, model, data):
    """
    Update or adapt a forecasting model after detected concept drift.

    Parameters
    ----------
    changepoints : list[int]
        Chronological indices where concept drift was detected.

    model : object
        Current forecasting model before adaptation.

    data : object
        Historical data available up to the current point in time
        that may be used for model adaptation.
        pandas.DataFrame
        Chronologically sorted data available up to the current point.
        It must contain SETTLEMENTDATE and the variables required by
        the forecasting model.

    Returns
    -------
    updated_model : object
        Forecasting model after the adaptation strategy has been
        applied.

    Rules
    -----
    - Adaptation must only use information available up to the
      current point in time.
    - Different adaptation strategies may be implemented internally.
    - The interface must always return an updated forecasting model.
    """

    raise NotImplementedError(
        "An adaptation implementation must be provided."
    )

def uq(point_forecast, calibration_residuals):
    """
    Quantify forecast uncertainty and determine whether escalation
    is required.

    Parameters
    ----------
    point_forecast : float
        Forecast produced by the forecasting model for the current
        observation.

    calibration_residuals : array-like
        Historical forecast residuals from the calibration period
        used to estimate forecast uncertainty.

    Returns
    -------
    lower_bound : float
        Lower bound of the prediction interval.

    upper_bound : float
        Upper bound of the prediction interval.

    escalate_flag : bool
        Indicates whether the uncertainty is sufficiently high to
        trigger escalation.

    Rules
    -----
    - Only calibration information available up to the current
      point may be used.
    - The internal uncertainty method may change later.
    - The function must always return:
        lower_bound, upper_bound, escalate_flag
    """

    raise NotImplementedError(
        "An uncertainty quantification implementation "
        "must be provided."
    )