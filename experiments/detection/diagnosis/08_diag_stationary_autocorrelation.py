"""
08_diag_stationary_autocorrelation.py

Diagnosis 08 — Can stationary autocorrelation alone cause repeated
drift-detector alarms?
======================================================================

Background
----------
Previous diagnostics established:

Diagnosis 01
    Destroying temporal ordering removed detector alarms.

Diagnosis 02
    Short-range/local temporal dependence was especially important.

Diagnosis 03
    Extreme persistence was already present in the signed Gradient
    Boosting residuals.

Diagnosis 04
    Highly persistent forecasting residuals were associated with many
    ADWIN/KSWIN alarms.

Diagnosis 05
    Calibration normalization was NOT the source of the KSWIN problem.
    KSWIN produced exactly the same changepoints under several
    monotonic/affine transformations.

Diagnosis 06
    KSWIN alarms occurred approximately every few days.

Diagnosis 07
    KSWIN's minimum alarm spacing was exactly its 140-observation
    window size, but alarms were NOT concentrated exactly at that
    boundary.

The remaining key question is:

    Is strong temporal dependence BY ITSELF sufficient to make the
    frozen detectors produce repeated alarms?

This diagnosis answers that question using controlled stationary data.


Core idea
---------
Generate stationary Gaussian AR(1) processes:

    x_t = phi * x_(t-1) + epsilon_t

with:

    epsilon_t ~ N(0, 1 - phi^2)

Therefore, for every phi:

    E[x_t] = 0
    Var[x_t] ~= 1

The marginal distribution remains approximately N(0, 1).

Only temporal dependence changes.


NO DRIFT IS INTRODUCED
----------------------
There are:

    - no changepoints
    - no mean shifts
    - no variance shifts
    - no gradual transitions
    - no recurring regimes

Therefore every detector alarm is a false alarm by construction.


Detector representation
-----------------------
To remain close to the CURRENT AEMO Task 120 pipeline:

    stationary signed AR(1)
        -> absolute value
        -> calibration mean/std normalization
        -> detector

For each generated series:

    calibration = first 1,000 observations
    test        = next 15,000 observations

The calibration statistics are frozen:

    calibration_mean = mean(|x_cal|)
    calibration_std  = std(|x_cal|)

Then:

    detector_input =
        (|x_test| - calibration_mean)
        / calibration_std


Why 1,000 + 15,000?
--------------------
This matches the calibration/test lengths used in the project's
synthetic detector benchmark:

    calibration = 1,000
    test        = 15,000

Using the same test length also allows false alarms per 10,000
observations to be compared directly with the synthetic benchmark.


AR(1) levels
------------
The following dependence strengths are tested:

    phi = 0.00
    phi = 0.50
    phi = 0.80
    phi = 0.90
    phi = 0.95
    phi = 0.98
    phi = 0.99

Important:

    phi is the dependence of the SIGNED AR(1) process.

After taking absolute values, lag-1 ACF will be lower.

Therefore we record both:

    signed ACF1
    absolute-error ACF1
    normalized detector-input ACF1

The important comparison with AEMO is the realized ACF of the
ABSOLUTE detector input, because that is what the real detectors see.


Seeds
-----
20 independent data seeds are used:

    1 ... 20

This is a diagnostic experiment only.

It does not change the project's official benchmark seed protocol.


Hypotheses
----------
If autocorrelation alone is NOT sufficient:

    phi=0.00    ~0 alarms
    phi=0.50    ~0 alarms
    phi=0.80    ~0 alarms
    phi=0.90    ~0 alarms
    phi=0.95    ~0 alarms
    phi=0.98    ~0 alarms

Then something beyond stationary autocorrelation must be responsible
for the real AEMO alarms, such as:

    - local non-stationarity
    - variance changes
    - seasonal residual structure
    - nonlinear dependence
    - genuine repeated distribution changes


If autocorrelation IS sufficient:

    alarms should increase as dependence increases,

for example:

    phi=0.00    KSWIN ~0
    phi=0.80    KSWIN small
    phi=0.95    KSWIN larger
    phi=0.98    KSWIN many

That would provide controlled evidence that a detector calibrated on
weakly dependent synthetic errors may behave very differently on
strongly dependent forecasting-error streams.


Important experimental control
------------------------------
The innovation standard deviation is:

    sqrt(1 - phi^2)

rather than a constant 1.

This ensures all AR(1) processes have approximately the SAME stationary
marginal variance.

Without this correction, increasing phi would increase both:

    temporal dependence
    AND
    marginal variance

which would confound the diagnosis.


Detectors
---------
The CURRENT DETECTOR_REGISTRY is used unchanged.

No detector parameter is tuned.

This script does NOT:
    - change alpha
    - change delta
    - change threshold
    - change KSWIN window_size
    - change KSWIN stat_size
    - perform smoothing
    - perform prewhitening
    - modify Task 120
    - modify T2
    - write to runs.csv


Outputs
-------
Saved under:

    results/tables/detection/diagnostics/

Files:

    08_stationary_autocorrelation_runs.csv
    08_stationary_autocorrelation_summary.csv


Run
---
From repository root:

    python experiments/detection/08_diag_stationary_autocorrelation.py
"""


from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd


# =====================================================================
# Project imports
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from drift_lab.detectors import (
    DETECTOR_REGISTRY,
)

from drift_lab.evaluation import (
    evaluate_detections,
)


# =====================================================================
# Diagnostic configuration
# =====================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "diagnostics"
)


# Match the project's synthetic detector evaluation layout.
CALIBRATION_SIZE = 1000
TEST_SIZE = 15000


# Burn-in is deliberately generous.
#
# The process is initialized from its stationary marginal distribution
# anyway, but burn-in makes the diagnostic insensitive to initialization.
BURN_IN = 2000


# Fixed diagnostic data seeds.
SEEDS = tuple(
    range(
        1,
        21,
    )
)


# Increasing levels of serial dependence.
PHI_VALUES = (
    0.00,
    0.50,
    0.80,
    0.90,
    0.95,
    0.98,
    0.99,
)


# Same tolerance used elsewhere in the synthetic evaluation.
#
# There are no true changepoints here, so this does not affect the
# false-alarm calculation. It is supplied for API consistency.
TOLERANCE = 48 * 7


# ACF lags recorded for descriptive diagnostics.
ACF_LAGS = (
    1,
    2,
    6,
    12,
    24,
    48,
    96,
    336,
)


# =====================================================================
# Validation
# =====================================================================

def validate_stream(
    values,
    name,
):
    """
    Validate a finite one-dimensional numeric stream.
    """

    values = np.asarray(
        values,
        dtype=float,
    )


    if values.ndim != 1:

        raise ValueError(
            f"{name} must be one-dimensional."
        )


    if len(values) == 0:

        raise ValueError(
            f"{name} must not be empty."
        )


    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            f"{name} contains NaN or infinite values."
        )


    return values


# =====================================================================
# Stationary AR(1) generator
# =====================================================================

def generate_stationary_ar1(
    phi,
    seed,
):
    """
    Generate a stationary zero-mean Gaussian AR(1) process.

    Model
    -----
        x_t = phi * x_(t-1) + epsilon_t

    where:

        epsilon_t ~ N(
            0,
            1 - phi^2
        )

    Therefore the stationary marginal variance is:

        Var(x_t) = 1

    for every tested phi.

    This is essential because the diagnosis is intended to change
    TEMPORAL DEPENDENCE only, not marginal variance.

    Returns
    -------
    calibration : np.ndarray

    test : np.ndarray
    """

    phi = float(
        phi
    )


    if not (
        0 <= phi < 1
    ):

        raise ValueError(
            "phi must satisfy 0 <= phi < 1."
        )


    rng = np.random.default_rng(
        seed
    )


    total_size = (
        BURN_IN
        +
        CALIBRATION_SIZE
        +
        TEST_SIZE
    )


    values = np.empty(
        total_size,
        dtype=float,
    )


    # -------------------------------------------------------------
    # Start from stationary distribution
    # -------------------------------------------------------------

    values[
        0
    ] = rng.normal(
        loc=0.0,
        scale=1.0,
    )


    # -------------------------------------------------------------
    # Choose innovation variance so stationary variance remains 1
    # -------------------------------------------------------------

    innovation_std = float(
        np.sqrt(
            1.0
            -
            phi ** 2
        )
    )


    innovations = rng.normal(
        loc=0.0,
        scale=innovation_std,
        size=total_size - 1,
    )


    for t in range(
        1,
        total_size,
    ):

        values[
            t
        ] = (
            phi
            *
            values[
                t - 1
            ]
            +
            innovations[
                t - 1
            ]
        )


    # -------------------------------------------------------------
    # Remove burn-in
    # -------------------------------------------------------------

    usable = values[
        BURN_IN:
    ]


    expected = (
        CALIBRATION_SIZE
        +
        TEST_SIZE
    )


    if len(
        usable
    ) != expected:

        raise RuntimeError(
            "Generated AR(1) stream has incorrect length."
        )


    calibration = usable[
        :CALIBRATION_SIZE
    ].copy()


    test = usable[
        CALIBRATION_SIZE:
    ].copy()


    validate_stream(
        calibration,
        "AR1 calibration",
    )


    validate_stream(
        test,
        "AR1 test",
    )


    return (
        calibration,
        test,
    )


# =====================================================================
# Current-like detector representation
# =====================================================================

def build_detector_input(
    calibration,
    test,
):
    """
    Reproduce the current detector-input transformation:

        signed residual
            -> absolute value
            -> calibration mean/std normalization

    Calibration statistics are estimated from calibration only.
    """

    calibration = validate_stream(
        calibration,
        "calibration signed stream",
    )


    test = validate_stream(
        test,
        "test signed stream",
    )


    calibration_abs = np.abs(
        calibration
    )


    test_abs = np.abs(
        test
    )


    calibration_mean = float(
        calibration_abs.mean()
    )


    # NumPy population std, matching the current Task 120 logic.
    calibration_std = float(
        calibration_abs.std()
    )


    if (
        not np.isfinite(
            calibration_std
        )
        or calibration_std <= 0
    ):

        raise RuntimeError(
            "Calibration absolute-error std "
            "must be finite and > 0."
        )


    normalized = (
        test_abs
        -
        calibration_mean
    ) / calibration_std


    normalized = validate_stream(
        normalized,
        "normalized detector input",
    )


    return (
        normalized,
        calibration_mean,
        calibration_std,
    )


# =====================================================================
# ACF
# =====================================================================

def autocorrelation(
    values,
    lag,
):
    """
    Sample autocorrelation at one lag.
    """

    values = validate_stream(
        values,
        "ACF input",
    )


    if lag >= len(
        values
    ):

        return float(
            "nan"
        )


    return float(
        pd.Series(
            values,
            dtype=float,
        ).autocorr(
            lag=lag
        )
    )


# =====================================================================
# Detector runner
# =====================================================================

def run_detector(
    stream,
    entry,
):
    """
    Run one CURRENT registered detector unchanged.
    """

    config = entry[
        "config"
    ].copy()


    detected = entry[
        "func"
    ](
        stream,
        **config,
    )


    return (
        detected,
        config,
    )


# =====================================================================
# One phi / seed
# =====================================================================

def run_one_stream(
    phi,
    seed,
):
    """
    Generate one stationary AR(1) stream and run every frozen detector.

    Returns one result row per detector.
    """

    (
        calibration,
        test,
    ) = generate_stationary_ar1(
        phi=phi,
        seed=seed,
    )


    (
        detector_input,
        calibration_mean,
        calibration_std,
    ) = build_detector_input(
        calibration=calibration,
        test=test,
    )


    # -------------------------------------------------------------
    # Descriptive dependence diagnostics
    # -------------------------------------------------------------

    signed_acf = {
        lag:
            autocorrelation(
                test,
                lag,
            )
        for lag in ACF_LAGS
    }


    absolute_test = np.abs(
        test
    )


    absolute_acf = {
        lag:
            autocorrelation(
                absolute_test,
                lag,
            )
        for lag in ACF_LAGS
    }


    normalized_acf = {
        lag:
            autocorrelation(
                detector_input,
                lag,
            )
        for lag in ACF_LAGS
    }


    rows = []


    for (
        detector_name,
        entry,
    ) in DETECTOR_REGISTRY.items():

        (
            detected,
            config,
        ) = run_detector(
            stream=detector_input,
            entry=entry,
        )


        # ---------------------------------------------------------
        # There are NO true changepoints.
        #
        # Therefore every detector alarm is a false alarm.
        # ---------------------------------------------------------

        metrics = evaluate_detections(
            detected_changepoints=detected,
            true_changepoints=[],
            n_observations=len(
                detector_input
            ),
            drift_type="none",
            tolerance=TOLERANCE,
        )


        row = {
            "phi":
                float(
                    phi
                ),

            "seed":
                int(
                    seed
                ),

            "detector":
                detector_name,

            "detections":
                int(
                    len(
                        detected
                    )
                ),

            "false_alarms_per_10000":
                float(
                    metrics[
                        "false_alarms_per_10000"
                    ]
                ),

            "any_alarm":
                int(
                    len(
                        detected
                    ) > 0
                ),

            "calibration_abs_mean":
                calibration_mean,

            "calibration_abs_std":
                calibration_std,

            "test_signed_mean":
                float(
                    np.mean(
                        test
                    )
                ),

            "test_signed_std":
                float(
                    np.std(
                        test
                    )
                ),

            "test_absolute_mean":
                float(
                    np.mean(
                        absolute_test
                    )
                ),

            "detector_input_mean":
                float(
                    np.mean(
                        detector_input
                    )
                ),

            "detector_input_std":
                float(
                    np.std(
                        detector_input
                    )
                ),

            "detector_config":
                json.dumps(
                    config,
                    sort_keys=True,
                ),
        }


        # ---------------------------------------------------------
        # Attach ACF values
        # ---------------------------------------------------------

        for lag in ACF_LAGS:

            row[
                f"signed_acf_lag{lag}"
            ] = signed_acf[
                lag
            ]


            row[
                f"absolute_acf_lag{lag}"
            ] = absolute_acf[
                lag
            ]


            row[
                f"normalized_acf_lag{lag}"
            ] = normalized_acf[
                lag
            ]


        rows.append(
            row
        )


    return rows


# =====================================================================
# Aggregate results
# =====================================================================

def aggregate_results(
    raw,
):
    """
    Aggregate across diagnostic seeds.

    One row is produced per:

        phi x detector
    """

    summary = (
        raw
        .groupby(
            [
                "phi",
                "detector",
            ],
            as_index=False,
        )
        .agg(

            # -----------------------------------------------------
            # Realized temporal dependence
            # -----------------------------------------------------

            signed_acf1_mean=(
                "signed_acf_lag1",
                "mean",
            ),

            signed_acf1_sd=(
                "signed_acf_lag1",
                "std",
            ),

            absolute_acf1_mean=(
                "absolute_acf_lag1",
                "mean",
            ),

            absolute_acf1_sd=(
                "absolute_acf_lag1",
                "std",
            ),

            normalized_acf1_mean=(
                "normalized_acf_lag1",
                "mean",
            ),

            normalized_acf1_sd=(
                "normalized_acf_lag1",
                "std",
            ),


            # -----------------------------------------------------
            # Alarm count
            # -----------------------------------------------------

            alarm_mean=(
                "detections",
                "mean",
            ),

            alarm_sd=(
                "detections",
                "std",
            ),

            alarm_median=(
                "detections",
                "median",
            ),

            alarm_min=(
                "detections",
                "min",
            ),

            alarm_max=(
                "detections",
                "max",
            ),


            # -----------------------------------------------------
            # False alarms / 10k
            # -----------------------------------------------------

            false_alarms_per_10k_mean=(
                "false_alarms_per_10000",
                "mean",
            ),

            false_alarms_per_10k_sd=(
                "false_alarms_per_10000",
                "std",
            ),


            # -----------------------------------------------------
            # How often does the detector fire at all?
            # -----------------------------------------------------

            seeds_with_alarm=(
                "any_alarm",
                "sum",
            ),

            fraction_seeds_with_alarm=(
                "any_alarm",
                "mean",
            ),

            seeds=(
                "seed",
                "nunique",
            ),
        )
    )


    return summary


# =====================================================================
# Trend diagnostic
# =====================================================================

def print_detector_trend(
    summary,
    detector,
):
    """
    Print one detector's alarm behaviour as autocorrelation increases.
    """

    subset = (
        summary[
            summary[
                "detector"
            ] == detector
        ]
        .sort_values(
            "phi"
        )
    )


    print()
    print(
        detector.upper()
    )


    columns = [
        "phi",
        "signed_acf1_mean",
        "absolute_acf1_mean",
        "alarm_mean",
        "alarm_sd",
        "false_alarms_per_10k_mean",
        "fraction_seeds_with_alarm",
    ]


    print(
        subset[
            columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# =====================================================================
# Conservative interpretation
# =====================================================================

def print_interpretation(
    summary,
):
    """
    Interpret only whether stationary dependence is sufficient to
    increase false alarms.

    This does NOT recommend a treatment.
    """

    print()
    print(
        "=" * 90
    )

    print(
        "DIAGNOSTIC INTERPRETATION"
    )

    print(
        "=" * 90
    )


    for detector in DETECTOR_REGISTRY:

        subset = (
            summary[
                summary[
                    "detector"
                ] == detector
            ]
            .sort_values(
                "phi"
            )
        )


        independent = subset[
            subset[
                "phi"
            ] == 0.0
        ].iloc[
            0
        ]


        high = subset[
            subset[
                "phi"
            ] == 0.98
        ].iloc[
            0
        ]


        print()
        print(
            detector.upper()
        )


        print(
            f"  phi=0.00:"
        )


        print(
            f"    realized |x| ACF1       = "
            f"{independent['absolute_acf1_mean']:.4f}"
        )


        print(
            f"    mean alarms             = "
            f"{independent['alarm_mean']:.3f}"
        )


        print(
            f"    false alarms / 10k      = "
            f"{independent['false_alarms_per_10k_mean']:.3f}"
        )


        print(
            f"    seeds with >=1 alarm    = "
            f"{independent['fraction_seeds_with_alarm']:.1%}"
        )


        print()


        print(
            f"  phi=0.98:"
        )


        print(
            f"    realized |x| ACF1       = "
            f"{high['absolute_acf1_mean']:.4f}"
        )


        print(
            f"    mean alarms             = "
            f"{high['alarm_mean']:.3f}"
        )


        print(
            f"    false alarms / 10k      = "
            f"{high['false_alarms_per_10k_mean']:.3f}"
        )


        print(
            f"    seeds with >=1 alarm    = "
            f"{high['fraction_seeds_with_alarm']:.1%}"
        )


        print()


        # ---------------------------------------------------------
        # Conservative classification
        # ---------------------------------------------------------

        low_alarm = float(
            independent[
                "alarm_mean"
            ]
        )


        high_alarm = float(
            high[
                "alarm_mean"
            ]
        )


        if (
            high_alarm
            >= low_alarm + 5
        ):

            print(
                "  Pattern:"
            )


            print(
                "    The same stationary marginal process produces "
                "substantially more alarms when strong serial "
                "dependence is introduced."
            )


            print(
                "  Conclusion:"
            )


            print(
                "    Stationary autocorrelation alone is sufficient "
                "to materially increase this detector's false-alarm "
                "rate under the current configuration."
            )


        elif (
            high_alarm
            > low_alarm
        ):

            print(
                "  Pattern:"
            )


            print(
                "    False alarms increase somewhat with dependence."
            )


            print(
                "  Conclusion:"
            )


            print(
                "    Autocorrelation contributes, but does not by "
                "itself reproduce a strong over-detection problem."
            )


        else:

            print(
                "  Pattern:"
            )


            print(
                "    Strong stationary autocorrelation does not "
                "materially increase alarms."
            )


            print(
                "  Conclusion:"
            )


            print(
                "    Autocorrelation alone is insufficient. "
                "The next diagnosis should examine changing local "
                "distributions/variance in the real AEMO error stream."
            )


    print()
    print(
        "Important:"
    )


    print(
        "A detector alarm in this experiment cannot represent a true "
        "concept drift because the data-generating process is stationary "
        "for the complete run."
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print(
        "=" * 90
    )


    print(
        "DIAGNOSIS 08 — STATIONARY AUTOCORRELATION"
    )


    print(
        "=" * 90
    )


    print(
        "Question:"
    )


    print(
        "Can strong serial dependence alone produce repeated detector "
        "alarms even when there is no concept drift?"
    )


    print()
    print(
        f"Calibration observations : {CALIBRATION_SIZE:,}"
    )


    print(
        f"Test observations        : {TEST_SIZE:,}"
    )


    print(
        f"Burn-in observations     : {BURN_IN:,}"
    )


    print(
        f"Seeds                    : {SEEDS}"
    )


    print(
        f"AR(1) phi values         : {PHI_VALUES}"
    )


    print()
    print(
        "All AR(1) processes have stationary marginal variance ~= 1."
    )


    print(
        "Only temporal dependence is intentionally changed."
    )


    print(
        "There are NO true changepoints."
    )


    print(
        "Detector configurations remain unchanged."
    )


    print(
        "No results will be written to runs.csv."
    )


    all_rows = []


    # -----------------------------------------------------------------
    # Run experiment
    # -----------------------------------------------------------------

    for phi in PHI_VALUES:

        print()
        print(
            "-" * 90
        )


        print(
            f"phi = {phi:.2f}"
        )


        print(
            "-" * 90
        )


        for seed in SEEDS:

            rows = run_one_stream(
                phi=phi,
                seed=seed,
            )


            all_rows.extend(
                rows
            )


            if (
                seed == SEEDS[0]
                or
                seed == SEEDS[-1]
                or
                seed % 5 == 0
            ):

                print(
                    f"  completed seed={seed}"
                )


    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    raw = pd.DataFrame(
        all_rows
    )


    summary = aggregate_results(
        raw
    )


    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    raw_path = (
        OUTPUT_DIR
        / "08_stationary_autocorrelation_runs.csv"
    )


    summary_path = (
        OUTPUT_DIR
        / "08_stationary_autocorrelation_summary.csv"
    )


    raw.to_csv(
        raw_path,
        index=False,
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    # -----------------------------------------------------------------
    # Main summary
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 120
    )


    print(
        "STATIONARY AUTOCORRELATION SUMMARY"
    )


    print(
        "=" * 120
    )


    columns = [
        "phi",
        "detector",
        "signed_acf1_mean",
        "absolute_acf1_mean",
        "alarm_mean",
        "alarm_sd",
        "alarm_min",
        "alarm_max",
        "false_alarms_per_10k_mean",
        "fraction_seeds_with_alarm",
    ]


    print(
        summary[
            columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # -----------------------------------------------------------------
    # Detector-specific views
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 120
    )


    print(
        "ALARM TREND AS AUTOCORRELATION INCREASES"
    )


    print(
        "=" * 120
    )


    for detector in DETECTOR_REGISTRY:

        print_detector_trend(
            summary=summary,
            detector=detector,
        )


    # -----------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------

    print_interpretation(
        summary
    )


    # -----------------------------------------------------------------
    # Files
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 90
    )


    print(
        "Saved diagnostic outputs"
    )


    print(
        "=" * 90
    )


    print(
        raw_path
    )


    print(
        summary_path
    )


    print()
    print(
        "Do not tune the detectors based on this experiment."
    )


    print(
        "First determine whether stationary temporal dependence alone "
        "is sufficient to reproduce the excessive-alarm behaviour."
    )


if __name__ == "__main__":
    main()