"""
Shared evaluation metrics for the concept-drift forecasting pipeline.

This module provides common evaluation functions used across the
project so that forecasting, drift detection, adaptation, and
uncertainty components are assessed consistently.


CURRENT METRICS
---------------

Forecast evaluation:

    calculate_mae()
        Calculates Mean Absolute Error (MAE) between observed and
        predicted demand.

    calculate_rolling_mae()
        Calculates rolling MAE over the forecast stream.

        The default window is 336 observations:

            48 half-hourly observations/day × 7 days = 336 observations.

            Rolling MAE is used to track forecasting performance over time,
            including degradation of frozen baseline models and comparison
            of forecasting performance across the adaptation arms.


Synthetic drift-detection evaluation:

    calculate_detection_delay()
        Calculates mean delay between known synthetic drift events and
        successfully matched detector outputs.

    calculate_false_alarms_per_10000()
        Calculates unmatched detector outputs normalised per 10,000
        observations.

    calculate_missed_detections()
        Counts known synthetic drift events for which no detector output
        is successfully matched.

    evaluate_detections()
        Applies the shared synthetic event-matching protocol and returns
        the required detection metrics together with diagnostic matching
        information.


SYNTHETIC DETECTION EVALUATION
------------------------------
Synthetic data provide known ground-truth changepoints and therefore
support objective calculation of:

    - detection delay
    - false alarms per 10,000 observations
    - missed detections

The evaluation supports:

    - no drift
    - sudden drift
    - gradual drift
    - recurring drift

These metrics are used to validate detector behaviour on controlled
data before detectors are applied to real AEMO streams.


EVENT-MATCHING PROTOCOL
-----------------------
Detector outputs must be associated with known synthetic drift events
before delay, false alarms, and missed detections can be calculated.

For sudden and recurring drift, detections are matched using a
configurable post-changepoint tolerance.

The default tolerance is 336 observations:

    48 half-hourly observations/day × 7 days = 336 observations.

A detection is considered matched when it occurs at or after the true
changepoint and within the configured tolerance.

The first unused detection inside the matching window is assigned to
that drift event.

The 336-observation tolerance is an evaluation design choice used to
provide a fixed and reproducible event-association rule across
detectors and experiments.

It is not a detector threshold.

The tolerance determines whether a detection can be associated with an
event; it does not replace or redefine the measured detection delay.

For a successfully matched point event:

    detection_delay = detected_index - true_changepoint_index

If no detection occurs within the matching window, the true event is
counted as a missed detection.


GRADUAL DRIFT
-------------
Gradual drift is represented as one transition interval:

    [drift_start, drift_end]

rather than as two independent drift events.

A detection occurring from drift_start through drift_end can therefore
be associated with the same gradual-drift event.

A configurable grace period is also allowed after drift_end. If no
separate grace period is supplied, it defaults to the same value as the
matching tolerance.

Detection delay for a matched gradual event is measured from
drift_start.


RECURRING DRIFT
---------------
Recurring drift contains two true transition points, for example:

    A -> B -> A

Each transition is evaluated as a separate drift event.

A detector output can be matched to at most one true event.


NO-DRIFT CONTROL
----------------
The no-drift synthetic stream contains no true drift events.

Therefore:

    - every detector output is treated as a false alarm
    - detection delay is undefined
    - missed detections are zero

The no-drift control is used to verify that detectors do not raise
unnecessary alarms on a stable stream.


FALSE ALARMS
------------
Any synthetic detector output that is not matched to a known true drift
event is treated as a false alarm.

False alarms are normalised by stream length as:

    false_alarms_per_10000 =
        number_of_false_alarms
        / number_of_observations
        * 10000


REAL-DATA DETECTION EVALUATION
------------------------------
Real-data detection evaluation for AEMO SA1 and NSW1 is implemented by
evaluate_aemo_detections(). At the user's request, T2 reuses T1's
seven-day tolerance and first-unused detection rule. See
docs/aemo_matching_protocol.md and the explicit protocol argument.

Unlike the synthetic benchmark, AEMO data do not provide ground-truth
changepoints.

Detected changepoints will therefore be compared with historically
documented events as contextual reference points rather than treated as
known true drift events.

Planned real-data detection outputs include:

    - delay from a documented event to a matched detection
    - documented events with no matched detection
    - detections unmatched to documented events

Unmatched AEMO detections will not automatically be classified as false
alarms because documented events are not exhaustive ground truth.

The AEMO documented-event protocol is supplied explicitly. Frozen tiers
are preserved. The revised protocol reuses T1 matching, with Tier 1
allocation completed before Tier 2. Versions remain separate in the log.


INPUT VALIDATION
----------------
The implemented evaluation functions verify that:

    - y_true and y_pred have equal lengths
    - y_true and y_pred contain no missing values
    - changepoint indices are valid integer indices
    - changepoint indices fall within the evaluated stream
    - detected and true changepoints use the same coordinate system
    - synthetic drift types contain the expected number of true events


PIPELINE INTEGRATION
--------------------
This module provides shared metric implementations for the full
experimental pipeline.

Different stages use different subsets of these metrics:

    - Baseline degradation evaluation uses calculate_mae() and
      calculate_rolling_mae().

    - Synthetic drift-detection evaluation uses
      calculate_detection_delay(),
      calculate_false_alarms_per_10000(), and
      calculate_missed_detections() through the shared
      evaluate_detections() workflow.

    - Real AEMO detection evaluation will extend this module with
      documented-event matching because real data do not provide
      ground-truth changepoints.

    - Adaptation evaluation will reuse calculate_mae() and
      calculate_rolling_mae() from this module to compare forecasting
      performance across the four adaptation arms. Retraining count,
      training samples, and wall-clock cost are recorded by the
      experiment workflow rather than calculated as forecast-error
      metrics here.

    - Uncertainty evaluation uses calculate_empirical_coverage() and
      calculate_mean_interval_width(). Normalised interval width and
      worst 24-hour coverage remain planned for later Step 6 work.

Per-run evaluation metrics are calculated here. Aggregation across
seeds, including mean ± standard deviation where required, is performed
by the experiment workflow.

Experiment results are recorded separately through the shared
runs.csv logging component.

PLANNED METRICS
---------------
Real-data detection evaluation:

    - documented-event matching
    - delay from documented event to matched detection
    - unmatched detections
    - documented events with no matched detection


Uncertainty evaluation:

    - empirical coverage
    - mean interval width
    - normalised interval width
    - worst 24-hour coverage
"""

import numpy as np
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


# Uncertainty evaluation metrics

def _validate_interval_inputs(
    lower_bound,
    upper_bound,
):
    """
    Validate prediction-interval bounds and return float Series.
    """
    lower_bound = pd.Series(lower_bound, dtype=float)
    upper_bound = pd.Series(upper_bound, dtype=float)

    if len(lower_bound) != len(upper_bound):
        raise ValueError(
            "lower_bound and upper_bound must have the same length."
        )

    if len(lower_bound) == 0:
        raise ValueError(
            "Uncertainty evaluation inputs must not be empty."
        )

    if not (
        np.isfinite(lower_bound.to_numpy()).all()
        and np.isfinite(upper_bound.to_numpy()).all()
    ):
        raise ValueError(
            "Uncertainty evaluation inputs must contain only finite values."
        )

    if (lower_bound > upper_bound).any():
        raise ValueError(
            "lower_bound must not exceed upper_bound."
        )

    return lower_bound, upper_bound


def calculate_empirical_coverage(
    y_true,
    lower_bound,
    upper_bound,
):
    """
    Calculate empirical prediction-interval coverage.

    Coverage is the proportion of observed values lying inside the
    supplied prediction interval, including both interval boundaries.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    lower_bound : array-like
        Lower prediction-interval bounds.

    upper_bound : array-like
        Upper prediction-interval bounds.

    Returns
    -------
    float
        Fraction of observations covered by the interval.
    """
    y_true = pd.Series(y_true, dtype=float)
    lower_bound, upper_bound = _validate_interval_inputs(
        lower_bound,
        upper_bound,
    )

    if len(y_true) != len(lower_bound):
        raise ValueError(
            "y_true, lower_bound, and upper_bound "
            "must have the same length."
        )

    if not np.isfinite(y_true.to_numpy()).all():
        raise ValueError(
            "y_true must contain only finite values."
        )

    covered = (
        (y_true >= lower_bound)
        &
        (y_true <= upper_bound)
    )

    return float(covered.mean())


def calculate_mean_interval_width(
    lower_bound,
    upper_bound,
):
    """
    Calculate mean prediction-interval width.

    Parameters
    ----------
    lower_bound : array-like
        Lower prediction-interval bounds.

    upper_bound : array-like
        Upper prediction-interval bounds.

    Returns
    -------
    float
        Mean interval width.
    """
    lower_bound, upper_bound = _validate_interval_inputs(
        lower_bound,
        upper_bound,
    )

    return float(
        (upper_bound - lower_bound).mean()
    )



# Detection evaluation metrics

DEFAULT_DETECTION_TOLERANCE = 48 * 7


def _first_unused_in_window(detected, unused, start, end):
    """Shared T1/T2 primitive: earliest unused detection in inclusive [start, end]."""
    return next((cp for cp in detected if cp in unused and start <= cp <= end), None)

def _validate_changepoints(changepoints, n_observations, name):
    """
    Validate changepoint indices and return them as sorted unique integers.
    """
    if n_observations <= 0:
        raise ValueError("n_observations must be greater than 0.")

    values = []

    for cp in changepoints:
        if isinstance(cp, bool):
            raise ValueError(f"{name} must contain integer indices.")

        cp_int = int(cp)

        if cp_int != cp:
            raise ValueError(f"{name} must contain integer indices.")

        if not 0 <= cp_int < n_observations:
            raise ValueError(
                f"{name} contains index {cp_int}, which is outside "
                f"0..{n_observations - 1}."
            )

        values.append(cp_int)

    return sorted(set(values))


def _match_detections(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=DEFAULT_DETECTION_TOLERANCE,
    gradual_grace=None,
):
    """
    Match detected changepoints to synthetic ground truth.

    Matching rules
    --------------
    none
        There are no true drift events. Every detection is a false alarm.

    sudden / recurring
        Each true point changepoint is matched to the first unused
        detection in [true_cp, true_cp + tolerance].

    gradual
        The generator returns [drift_start, drift_end] for one gradual
        transition window, not two separate drift events. The first unused
        detection in [drift_start, drift_end + gradual_grace] is matched
        to that single drift event. Delay is measured from drift_start.

    A detection can match at most one true drift event.

    Parameters
    ----------
    detected_changepoints : array-like
        Detector output indices, relative to the detector input stream.

    true_changepoints : array-like
        Ground-truth indices, expressed in the same index system as
        detected_changepoints.

    n_observations : int
        Number of observations in the detector input stream.

    drift_type : str
        One of: "none", "sudden", "gradual", "recurring".

    tolerance : int
        Maximum allowed post-changepoint delay for sudden and recurring
        point events. Default is 336 observations (7 days of half-hourly
        data). The tolerance is configurable.

    gradual_grace : int, optional
        Additional post-window allowance for gradual drift. If None,
        the same value as tolerance is used.

    Returns
    -------
    dict
        Matching details containing:
            matched_pairs
            delays
            false_alarm_indices
            missed_true_events
            n_true_events
    """
    drift_type = str(drift_type).lower()

    if drift_type not in {
        "none",
        "sudden",
        "gradual",
        "recurring",
    }:
        raise ValueError(
            "drift_type must be 'none', 'sudden', "
            "'gradual', or 'recurring'."
        )

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    if gradual_grace is None:
        gradual_grace = tolerance

    if gradual_grace < 0:
        raise ValueError("gradual_grace must be non-negative.")

    detected = _validate_changepoints(
        detected_changepoints,
        n_observations,
        "detected_changepoints",
    )

    truth = _validate_changepoints(
        true_changepoints,
        n_observations,
        "true_changepoints",
    )

    if drift_type == "none":
        if truth:
            raise ValueError(
                "No-drift evaluation requires true_changepoints=[]"
            )

        return {
            "matched_pairs": [],
            "delays": [],
            "false_alarm_indices": detected,
            "missed_true_events": 0,
            "n_true_events": 0,
        }

    if drift_type == "sudden" and len(truth) != 1:
        raise ValueError(
            "Sudden drift requires exactly one true changepoint."
        )

    if drift_type == "gradual" and len(truth) != 2:
        raise ValueError(
            "Gradual drift requires [drift_start, drift_end]."
        )

    if drift_type == "recurring" and len(truth) != 2:
        raise ValueError(
            "Recurring drift requires exactly two true changepoints."
        )

    unused = set(detected)
    matched_pairs = []
    delays = []
    missed = 0

    if drift_type == "gradual":
        drift_start, drift_end = truth
        match_end = min(
            n_observations - 1,
            drift_end + gradual_grace,
        )

        detected_cp = _first_unused_in_window(detected, unused, drift_start, match_end)

        if detected_cp is not None:
            unused.remove(detected_cp)

            matched_pairs.append(
                {
                    "true_start": drift_start,
                    "true_end": drift_end,
                    "detected": detected_cp,
                }
            )
            delays.append(detected_cp - drift_start)
        else:
            missed = 1

        n_true_events = 1

    else:
        # sudden and recurring are point-event matching cases
        for true_cp in truth:
            match_end = min(
                n_observations - 1,
                true_cp + tolerance,
            )

            detected_cp = _first_unused_in_window(detected, unused, true_cp, match_end)

            if detected_cp is not None:
                unused.remove(detected_cp)

                matched_pairs.append(
                    {
                        "true": true_cp,
                        "detected": detected_cp,
                    }
                )
                delays.append(detected_cp - true_cp)
            else:
                missed += 1

        n_true_events = len(truth)

    false_alarm_indices = [
        cp for cp in detected
        if cp in unused
    ]

    return {
        "matched_pairs": matched_pairs,
        "delays": delays,
        "false_alarm_indices": false_alarm_indices,
        "missed_true_events": missed,
        "n_true_events": n_true_events,
    }


def calculate_detection_delay(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=DEFAULT_DETECTION_TOLERANCE,
    gradual_grace=None,
):
    """
    Calculate mean detection delay in observations.

    Delay is calculated only for successfully matched true drift events.
    Returns NaN when no drift event is matched, including the no-drift
    control.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    if not result["delays"]:
        return float("nan")

    return float(
        pd.Series(
            result["delays"],
            dtype=float,
        ).mean()
    )


def calculate_false_alarms_per_10000(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=DEFAULT_DETECTION_TOLERANCE,
    gradual_grace=None,
):
    """
    Calculate false alarms normalised per 10,000 observations.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    n_false_alarms = len(
        result["false_alarm_indices"]
    )

    return float(
        n_false_alarms
        / n_observations
        * 10000
    )


def calculate_missed_detections(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=DEFAULT_DETECTION_TOLERANCE,
    gradual_grace=None,
):
    """
    Calculate the number of true drift events that were not detected.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    return int(
        result["missed_true_events"]
    )


def evaluate_detections(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=DEFAULT_DETECTION_TOLERANCE,
    gradual_grace=None,
):
    """
    Evaluate detector output against synthetic ground truth.

    Returns the shared detection metrics plus matching details
    used for diagnostics and plotting.

    Notes
    -----
    Ground-truth indices and detected indices must already use the same
    coordinate system. If synthetic data is split before detection,
    convert full-series ground-truth indices to detector-stream-relative
    indices before calling this function.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    if result["delays"]:
        detection_delay = float(
            pd.Series(
                result["delays"],
                dtype=float,
            ).mean()
        )
    else:
        detection_delay = float("nan")

    false_alarms_per_10000 = float(
        len(result["false_alarm_indices"])
        / n_observations
        * 10000
    )

    return {
        "detection_delay": detection_delay,
        "false_alarms_per_10000": false_alarms_per_10000,
        "missed_detections": int(
            result["missed_true_events"]
        ),
        "matched_pairs": result["matched_pairs"],
        "false_alarm_indices": result["false_alarm_indices"],
        "delays": result["delays"],
    }


# AEMO contextual event evaluation. Kept separate from synthetic ground truth.

AEMO_TIER1_COLUMNS = {
    "EVT-2020-02": "COVID delay",
    "EVT-2020-07": "solar/mild-weather delay",
    "EVT-2021-10": "5-minute settlement delay",
    "EVT-2022-03": "2022 suspension delay",
    "EVT-2023-14": "security-directions delay",
}
AEMO_REGIMES = ("pre-drift", "drift", "post-drift")
AEMO_TIMEZONE = "Etc/GMT-10"  # fixed UTC+10, not UTC and not local DST


def aemo_timestamp(value):
    """Convert a naive market-time or timezone-aware value to fixed AEST."""
    value = pd.Timestamp(value)
    if pd.isna(value):
        raise ValueError("AEMO timestamps must not be missing.")
    if value.tzinfo is None:
        return value.tz_localize(AEMO_TIMEZONE)
    return value.tz_convert(AEMO_TIMEZONE)


def validate_aemo_protocol(protocol):
    """Require the user-requested reuse of T1's seven-day matching rule."""
    for key in ("point_window_days", "interval_grace_days", "regime_days"):
        value = protocol.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be an explicit non-negative integer.")
    if protocol["point_window_days"] == 0 or protocol["regime_days"] == 0:
        raise ValueError("Point and regime windows must be positive.")
    for key in ("point_window_days", "interval_grace_days"):
        if protocol[key] * 48 != DEFAULT_DETECTION_TOLERANCE:
            raise ValueError(f"{key} must reuse T1's 336-observation tolerance.")
    expected = {
        "timezone": AEMO_TIMEZONE,
        "assignment": "tier_then_start_then_event_id_first_unused",
        "multiple_detections_per_event": False,
        "boundary": "both_closed",
    }
    if not protocol.get("protocol_id"):
        raise ValueError("An explicit protocol_id is required.")
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Unsupported AEMO protocol setting: {key}.")


def load_aemo_events(source):
    """Validate the frozen catalogue; return a copy with AEST date columns."""
    events = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    required = {"event_id", "event_name", "start_date", "end_date",
                "date_precision", "region", "tier"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing event fields: {sorted(required - set(events.columns))}")
    if events[list(required)].isna().any().any():
        raise ValueError("Required event fields must not be missing.")
    if events.event_id.duplicated().any():
        raise ValueError("Event IDs must be unique.")
    tiers = pd.to_numeric(events.tier, errors="raise")
    if not tiers.isin([1, 2]).all():
        raise ValueError("Frozen tiers must be exactly 1 or 2.")
    events["tier"] = tiers.astype(int)
    if set(events.loc[events.tier == 1, "event_id"]) != set(AEMO_TIER1_COLUMNS):
        raise ValueError("The fixed five Tier 1 event IDs must be preserved exactly.")
    if not events.date_precision.isin(["day", "month", "date_range"]).all():
        raise ValueError("Unsupported event date_precision.")
    if not events.region.isin(["NEM", "SA1", "NSW1"]).all():
        raise ValueError("Unsupported event region.")
    for name in ("start_date", "end_date"):
        events[name] = events[name].map(aemo_timestamp)
        if not (events[name] == events[name].dt.normalize()).all():
            raise ValueError("The frozen catalogue uses dates, not intraday timestamps.")
    if (events.end_date < events.start_date).any():
        raise ValueError("Event end_date precedes start_date.")
    return events.sort_values(["tier", "start_date", "event_id"]).reset_index(drop=True)


def build_aemo_event_windows(event, *, regime_days):
    """F1-compatible event-relative windows, all left-closed/right-open."""
    start = aemo_timestamp(event["start_date"])
    end = aemo_timestamp(event["end_date"]) + pd.Timedelta(days=1)
    span = pd.Timedelta(days=regime_days)
    return {"pre-drift": (start - span, start), "drift": (start, end),
            "post-drift": (end, end + span)}


def _aemo_match_end(event, protocol):
    if event["date_precision"] == "day":
        return event["start_date"] + pd.Timedelta(days=protocol["point_window_days"])
    # As in T1 gradual matching: use the recorded end anchor + grace,
    # with an inclusive endpoint. Do not add an extra day from F1 regimes.
    return event["end_date"] + pd.Timedelta(days=protocol["interval_grace_days"])


def _aemo_global_regime(timestamp, events, protocol):
    """Disjoint full-stream accounting; never sum overlapping event windows."""
    windows = [build_aemo_event_windows(e, regime_days=protocol["regime_days"])
               for e in events.to_dict("records")]
    for regime in ("drift", "pre-drift"):
        if any(w[regime][0] <= timestamp < w[regime][1] for w in windows):
            return regime
    if any(timestamp >= start for start in events.start_date):
        return "post-drift"
    return "pre-drift"


def _aemo_counts(rows):
    """Detection-based numerator, including both tiers; undefined if empty."""
    n = len(rows)
    tier1 = int((rows.match_tier == 1).sum())
    tier2 = int((rows.match_tier == 2).sum())
    return {"detections": n, "tier1_matched": tier1, "tier2_matched": tier2,
            "unmatched": n - tier1 - tier2,
            "precision": (tier1 + tier2) / n if n else float("nan")}


def evaluate_aemo_detections(detections, events, region, *, protocol,
                             stream_start, stream_end_exclusive):
    """Assign Task 120 detections to contextual events without fitting models.

    Return per-detection audit rows, all Tier 1 statuses, and disjoint global
    metrics. Match every Tier 1 event before processing Tier 2. For each
    event select the first unused detection within the inclusive window,
    using the same primitive as T1. Every event and detection can be used
    at most once. No result here is synthetic ground-truth precision.
    """
    validate_aemo_protocol(protocol)
    if region not in ("SA1", "NSW1"):
        raise ValueError("T2 supports SA1 and NSW1.")
    events = load_aemo_events(events)
    start, end = aemo_timestamp(stream_start), aemo_timestamp(stream_end_exclusive)
    if end <= start:
        raise ValueError("Invalid stream coverage.")
    applicable = events[events.region.isin(["NEM", region])]
    rows = detections.copy()
    if not {"detected_index", "SETTLEMENTDATE"}.issubset(rows.columns):
        raise ValueError("Detections need detected_index and SETTLEMENTDATE.")
    rows["SETTLEMENTDATE"] = pd.Series(
        [aemo_timestamp(t) for t in rows.SETTLEMENTDATE], index=rows.index,
        dtype=f"datetime64[ns, {AEMO_TIMEZONE}]")
    indices = pd.to_numeric(rows.detected_index, errors="raise")
    if indices.isna().any() or (indices < 0).any() or (indices % 1 != 0).any():
        raise ValueError("Detection indices must be non-negative integers.")
    if indices.duplicated().any() or rows.SETTLEMENTDATE.duplicated().any():
        raise ValueError("Duplicate detections must be resolved in the input.")
    if not rows.SETTLEMENTDATE.is_monotonic_increasing or not indices.is_monotonic_increasing:
        raise ValueError("Task 120 detections must be chronological.")
    if ((rows.SETTLEMENTDATE < start) | (rows.SETTLEMENTDATE >= end)).any():
        raise ValueError("Detection outside the supplied test coverage.")
    timestamps = rows.SETTLEMENTDATE.tolist()
    unused = set(timestamps)
    assignments = {}
    # Catalogue order is tier -> start -> ID. Complete all Tier 1 allocations
    # before Tier 2 can consume any remaining detection (T1 event-first order).
    for event in applicable.to_dict("records"):
        timestamp = _first_unused_in_window(
            timestamps, unused, event["start_date"], _aemo_match_end(event, protocol))
        if timestamp is not None:
            unused.remove(timestamp)
            assignments[timestamp] = event
    audits = []
    for row in rows.to_dict("records"):
        timestamp = row["SETTLEMENTDATE"]
        candidates = [e for e in applicable.to_dict("records")
                      if e["start_date"] <= timestamp <= _aemo_match_end(e, protocol)]
        event = assignments.get(timestamp)
        row.update({
            "match_status": f"tier{event['tier']}" if event else "unmatched",
            "match_tier": event["tier"] if event else 0,
            "matched_event_id": event["event_id"] if event else "",
            "candidate_event_ids": "|".join(e["event_id"] for e in candidates),
            "delay_days": (timestamp - event["start_date"]).total_seconds() / 86400
                          if event else float("nan"),
            "global_regime": _aemo_global_regime(timestamp, applicable, protocol),
        })
        audits.append(row)
    audit = pd.DataFrame(audits, columns=list(rows.columns) + [
        "match_status", "match_tier", "matched_event_id", "candidate_event_ids",
        "delay_days", "global_regime"])
    event_results = []
    for event in events[events.tier == 1].to_dict("records"):
        applies = event["region"] in ("NEM", region)
        if applies and (event["start_date"] < start or _aemo_match_end(event, protocol) >= end):
            raise ValueError(f"Incomplete Tier 1 coverage for {event['event_id']}.")
        matches = audit[audit.matched_event_id == event["event_id"]]
        first = matches.iloc[0] if len(matches) else None
        windows = build_aemo_event_windows(event, regime_days=protocol["regime_days"])
        delay_regime = "drift"
        if first is not None and first["SETTLEMENTDATE"] >= windows["drift"][1]:
            delay_regime = "post-drift"
        event_results.append({
            "event_id": event["event_id"], "event_name": event["event_name"],
            "applicable": int(applies), "detected": int(first is not None),
            "status": "not applicable" if not applies else
                      ("detected" if first is not None else "not detected"),
            "delay_days": first["delay_days"] if first is not None else float("nan"),
            "first_detection": first["SETTLEMENTDATE"] if first is not None else pd.NaT,
            "delay_regime": delay_regime, "assigned_detections": len(matches),
        })
    global_metrics = {regime: _aemo_counts(audit[audit.global_regime == regime])
                      for regime in AEMO_REGIMES}
    return {"audit": audit, "tier1_events": pd.DataFrame(event_results),
            "global_metrics": global_metrics, "totals": _aemo_counts(audit)}


# Planned uncertainty metrics for later Step 6 work:
# - normalised interval width
# - worst 24-hour coverage
