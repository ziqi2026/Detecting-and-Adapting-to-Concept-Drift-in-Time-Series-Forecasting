"""
03_diag_error_persistence_source.py

Diagnosis 03 — Where does short-range error persistence come from?
==================================================================

Background
----------
Diagnosis 01 showed that detector alarms disappear when the calibration
forecast-error values are fully shuffled.

Diagnosis 02 then showed that:

    - preserving short-range / within-day ordering retains many alarms
    - destroying within-day ordering substantially reduces alarms
    - day-block and week-block shuffling do not remove the problem
    - lag-1 autocorrelation of normalized absolute errors is extremely high

Therefore the next question is:

    Is this strong short-range persistence already present in the
    SIGNED forecast residuals?

or:

    Does it mainly appear after converting the residuals into
    ABSOLUTE ERROR / ERROR MAGNITUDE?

Why this matters
----------------
Task 120 currently feeds the detectors:

    residual e_t
        -> |e_t|
        -> calibration normalization
        -> detector

If signed residuals themselves are strongly autocorrelated, then the
forecast-error process has persistent directional structure.

For example:

    e_t > 0 for many consecutive observations
    or
    e_t < 0 for many consecutive observations

This would suggest the forecasting residual process itself is not close
to temporally independent.

However, another possibility is:

    signed residual autocorrelation       = relatively low
    absolute-error autocorrelation        = very high
    squared-error autocorrelation         = very high

That pattern would be more consistent with ERROR-MAGNITUDE or
VOLATILITY CLUSTERING:

    periods of small errors
        followed by
    periods of large errors

even if the error direction changes.

This distinction is important before considering any treatment such as
prewhitening.

Diagnostic representations
--------------------------
This script examines three representations of the SAME calibration
forecast residuals:

1. signed_error

       e_t = y_t - y_hat_t

2. absolute_error

       |e_t|

   This is the representation currently used by Task 120.

3. squared_error

       e_t^2

   This provides another diagnostic view of error magnitude.

For detector-count comparison only, each representation is independently
standardized using its own calibration mean and standard deviation.

The detector configurations are unchanged.

Additional directional diagnostics
----------------------------------
The script also measures:

    - fraction of positive residuals
    - fraction of consecutive residuals with the same sign
    - number of sign changes
    - mean sign-run length
    - maximum sign-run length

These help determine whether high signed-residual autocorrelation is
associated with long periods of systematic over- or under-prediction.

Important limitations
---------------------
This is a diagnostic experiment.

Detector counts across signed, absolute, and squared errors have
different semantic meanings and should NOT be treated as a detector
performance benchmark.

For example:

    - on signed residuals, an upward change means increasingly positive
      residuals
    - on absolute residuals, an upward change means increasing error
      magnitude

The main evidence from this diagnosis is therefore the dependence
structure, especially autocorrelation and sign persistence.

This script does NOT:

    - change detectors.py
    - tune detector thresholds
    - implement prewhitening
    - smooth the stream
    - aggregate to daily data
    - change Task 120
    - change T2 matching
    - write to runs.csv

Run
---
From the repository root:

    python experiments/detection/03_diag_error_persistence_source.py

Outputs
-------
Saved under:

    results/tables/detection/diagnostics/

Files:

    03_error_persistence_summary.csv
    03_error_persistence_acf.csv
    03_error_persistence_detector_counts.csv
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

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


from drift_lab.data_loader import load_and_split_aemo

from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)

from drift_lab.detectors import (
    DETECTOR_REGISTRY,
)


# ---------------------------------------------------------------------
# Frozen project settings
# ---------------------------------------------------------------------

REGIONS = (
    "SA1",
    "NSW1",
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "diagnostics"
)


# Same frozen forecaster used by Task 120.
FORECASTER_CONFIG = {
    "feature_mode": "calendar+lags",
    "lags": (48, 336),
    "seed": 0,
}


# Focus on short-range dependence while retaining
# daily and weekly reference points.
#
# AEMO stream is standardized to 30-minute intervals.
ACF_LAGS = (
    1,      # 30 minutes
    2,      # 1 hour
    3,      # 1.5 hours
    4,      # 2 hours
    6,      # 3 hours
    12,     # 6 hours
    24,     # 12 hours
    48,     # 24 hours
    96,     # 48 hours
    336,    # 7 days
)


REPRESENTATIONS = (
    "signed_error",
    "absolute_error",
    "squared_error",
)


# ---------------------------------------------------------------------
# Forecast calibration errors
# ---------------------------------------------------------------------

def build_calibration_errors(
    train,
    calibration,
):
    """
    Fit the frozen Task-120 Gradient Boosting forecaster on training
    data and produce chronological calibration residuals.

    The current target is never used in its own prediction.

    Returns
    -------
    pd.DataFrame

        Columns:

            SETTLEMENTDATE
            y_true
            y_pred
            error
    """

    model = GradientBoostedForecaster(
        **FORECASTER_CONFIG
    )

    model.fit(
        train
    )

    errors = frozen_error_stream(
        model,
        calibration,
        history_df=train,
    )


    required = {
        "SETTLEMENTDATE",
        "y_true",
        "y_pred",
        "error",
    }


    if not required.issubset(
        errors.columns
    ):

        raise RuntimeError(
            "Calibration error stream is missing "
            "required columns."
        )


    residual = (
        errors["error"]
        .to_numpy(dtype=float)
    )


    if len(residual) == 0:

        raise RuntimeError(
            "Calibration residual stream is empty."
        )


    if not np.isfinite(
        residual
    ).all():

        raise RuntimeError(
            "Calibration residual stream contains "
            "NaN or infinite values."
        )


    if len(errors) != len(
        calibration
    ):

        raise RuntimeError(
            "Calibration prediction length mismatch."
        )


    return errors


# ---------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------

def make_representations(
    signed_error,
):
    """
    Construct the three representations used in Diagnosis 03.

    All are created from exactly the same signed residual sequence.
    """

    signed_error = np.asarray(
        signed_error,
        dtype=float,
    )


    representations = {
        "signed_error":
            signed_error.copy(),

        "absolute_error":
            np.abs(
                signed_error
            ),

        "squared_error":
            np.square(
                signed_error
            ),
    }


    for (
        name,
        values,
    ) in representations.items():

        if len(values) != len(
            signed_error
        ):

            raise RuntimeError(
                f"{name}: representation length changed."
            )


        if not np.isfinite(
            values
        ).all():

            raise RuntimeError(
                f"{name}: representation contains "
                "non-finite values."
            )


    return representations


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def standardize(
    values,
):
    """
    Standardize one calibration representation using its own
    calibration mean and population standard deviation.

    NumPy's default ddof=0 is retained to match Task 120.
    """

    values = np.asarray(
        values,
        dtype=float,
    )


    mean = float(
        values.mean()
    )


    std = float(
        values.std()
    )


    if not np.isfinite(
        mean
    ):

        raise RuntimeError(
            "Representation mean is invalid."
        )


    if (
        not np.isfinite(std)
        or std <= 0
    ):

        raise RuntimeError(
            "Representation standard deviation "
            "must be finite and > 0."
        )


    normalized = (
        values
        - mean
    ) / std


    if not np.isfinite(
        normalized
    ).all():

        raise RuntimeError(
            "Standardized representation contains "
            "non-finite values."
        )


    return (
        normalized,
        mean,
        std,
    )


# ---------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------

def calculate_acf(
    region,
    representation,
    values,
):
    """
    Calculate sample autocorrelation at predefined lags.

    Scaling does not affect ordinary autocorrelation, so ACF is
    calculated directly on the natural representation.
    """

    series = pd.Series(
        values,
        dtype=float,
    )


    rows = []


    for lag in ACF_LAGS:

        if lag >= len(series):

            value = float(
                "nan"
            )

        else:

            value = float(
                series.autocorr(
                    lag=lag
                )
            )


        rows.append(
            {
                "region": region,
                "representation": representation,
                "lag_observations": lag,
                "lag_hours": lag / 2,
                "autocorrelation": value,
            }
        )


    return rows


# ---------------------------------------------------------------------
# Sign persistence
# ---------------------------------------------------------------------

def calculate_sign_diagnostics(
    signed_error,
):
    """
    Measure persistence in the direction of forecasting residuals.

    Positive residual:
        observed demand > forecast
        model under-predicted

    Negative residual:
        observed demand < forecast
        model over-predicted
    """

    values = np.asarray(
        signed_error,
        dtype=float,
    )


    signs = np.sign(
        values
    )


    nonzero = signs != 0


    positive_fraction = float(
        np.mean(
            values > 0
        )
    )


    negative_fraction = float(
        np.mean(
            values < 0
        )
    )


    zero_fraction = float(
        np.mean(
            values == 0
        )
    )


    # -------------------------------------------------------------
    # Consecutive same-sign fraction
    # -------------------------------------------------------------

    previous = signs[:-1]
    current = signs[1:]


    valid_pairs = (
        (previous != 0)
        &
        (current != 0)
    )


    if valid_pairs.any():

        same_sign_fraction = float(
            np.mean(
                previous[valid_pairs]
                ==
                current[valid_pairs]
            )
        )


        sign_change_fraction = float(
            np.mean(
                previous[valid_pairs]
                !=
                current[valid_pairs]
            )
        )

    else:

        same_sign_fraction = float(
            "nan"
        )

        sign_change_fraction = float(
            "nan"
        )


    # -------------------------------------------------------------
    # Sign run lengths
    # -------------------------------------------------------------

    nonzero_signs = signs[
        nonzero
    ]


    run_lengths = []


    if len(nonzero_signs) > 0:

        current_sign = (
            nonzero_signs[0]
        )

        current_length = 1


        for sign in nonzero_signs[1:]:

            if sign == current_sign:

                current_length += 1

            else:

                run_lengths.append(
                    current_length
                )

                current_sign = sign

                current_length = 1


        run_lengths.append(
            current_length
        )


    if run_lengths:

        mean_run_length = float(
            np.mean(
                run_lengths
            )
        )

        median_run_length = float(
            np.median(
                run_lengths
            )
        )

        max_run_length = int(
            np.max(
                run_lengths
            )
        )

        n_runs = int(
            len(
                run_lengths
            )
        )

    else:

        mean_run_length = float(
            "nan"
        )

        median_run_length = float(
            "nan"
        )

        max_run_length = 0

        n_runs = 0


    return {
        "positive_fraction":
            positive_fraction,

        "negative_fraction":
            negative_fraction,

        "zero_fraction":
            zero_fraction,

        "same_sign_fraction":
            same_sign_fraction,

        "sign_change_fraction":
            sign_change_fraction,

        "n_sign_runs":
            n_runs,

        "mean_sign_run_length":
            mean_run_length,

        "median_sign_run_length":
            median_run_length,

        "max_sign_run_length":
            max_run_length,
    }


# ---------------------------------------------------------------------
# Detector diagnostic
# ---------------------------------------------------------------------

def run_detectors(
    region,
    representation,
    normalized_stream,
):
    """
    Run frozen detector configurations on one standardized representation.

    IMPORTANT
    ---------
    These counts are diagnostic only.

    signed_error, absolute_error and squared_error have different
    meanings, so this is NOT a detector-performance comparison.
    """

    rows = []


    for (
        detector_name,
        entry,
    ) in DETECTOR_REGISTRY.items():

        config = entry[
            "config"
        ].copy()


        changepoints = entry[
            "func"
        ](
            normalized_stream,
            **config,
        )


        rows.append(
            {
                "region": region,
                "representation": representation,
                "detector": detector_name,
                "detections": len(
                    changepoints
                ),
                "n_observations": len(
                    normalized_stream
                ),
                "detections_per_10000":
                    len(changepoints)
                    / len(normalized_stream)
                    * 10000,
                "detector_config":
                    json.dumps(
                        config,
                        sort_keys=True,
                    ),
            }
        )


    return rows


# ---------------------------------------------------------------------
# Region diagnosis
# ---------------------------------------------------------------------

def diagnose_region(
    region,
):
    """
    Run Diagnosis 03 for one AEMO region.
    """

    print()
    print("=" * 80)
    print(
        f"REGION: {region}"
    )
    print("=" * 80)


    (
        _,
        train,
        calibration,
        _,
    ) = load_and_split_aemo(
        region=region,
        data_path=DATA_PATH,
    )


    print(
        f"Training observations    : "
        f"{len(train):,}"
    )


    print(
        f"Calibration observations : "
        f"{len(calibration):,}"
    )


    print(
        "Calibration period       : "
        f"{calibration['SETTLEMENTDATE'].iloc[0]}"
        " to "
        f"{calibration['SETTLEMENTDATE'].iloc[-1]}"
    )


    errors = build_calibration_errors(
        train=train,
        calibration=calibration,
    )


    signed_error = (
        errors["error"]
        .to_numpy(dtype=float)
    )


    representations = (
        make_representations(
            signed_error
        )
    )


    # -------------------------------------------------------------
    # Direction / sign persistence
    # -------------------------------------------------------------

    sign_metrics = (
        calculate_sign_diagnostics(
            signed_error
        )
    )


    print()
    print(
        "Residual-direction diagnostics:"
    )


    print(
        f"  positive residual fraction : "
        f"{sign_metrics['positive_fraction']:.4f}"
    )

    print(
        f"  negative residual fraction : "
        f"{sign_metrics['negative_fraction']:.4f}"
    )

    print(
        f"  same consecutive sign      : "
        f"{sign_metrics['same_sign_fraction']:.4f}"
    )

    print(
        f"  mean sign-run length       : "
        f"{sign_metrics['mean_sign_run_length']:.2f}"
    )

    print(
        f"  max sign-run length        : "
        f"{sign_metrics['max_sign_run_length']}"
    )


    summary_rows = []
    acf_rows = []
    detector_rows = []


    # -------------------------------------------------------------
    # Representation diagnostics
    # -------------------------------------------------------------

    for representation in REPRESENTATIONS:

        values = representations[
            representation
        ]


        (
            normalized,
            mean,
            std,
        ) = standardize(
            values
        )


        representation_acf = calculate_acf(
            region=region,
            representation=representation,
            values=values,
        )


        acf_rows.extend(
            representation_acf
        )


        acf_lookup = {
            row["lag_observations"]:
                row["autocorrelation"]
            for row in representation_acf
        }


        summary_row = {
            "region":
                region,

            "representation":
                representation,

            "mean":
                mean,

            "std":
                std,

            "min":
                float(
                    values.min()
                ),

            "median":
                float(
                    np.median(
                        values
                    )
                ),

            "max":
                float(
                    values.max()
                ),

            "acf_lag1":
                acf_lookup[1],

            "acf_lag2":
                acf_lookup[2],

            "acf_lag6":
                acf_lookup[6],

            "acf_lag12":
                acf_lookup[12],

            "acf_lag24":
                acf_lookup[24],

            "acf_lag48":
                acf_lookup[48],

            "acf_lag336":
                acf_lookup[336],
        }


        # Sign diagnostics only have a direct interpretation
        # for the signed residual representation.
        if representation == "signed_error":

            summary_row.update(
                sign_metrics
            )


        summary_rows.append(
            summary_row
        )


        detector_rows.extend(
            run_detectors(
                region=region,
                representation=representation,
                normalized_stream=normalized,
            )
        )


    # -------------------------------------------------------------
    # Console ACF summary
    # -------------------------------------------------------------

    print()
    print(
        "Autocorrelation by representation:"
    )


    print(
        f"{'representation':<18}"
        f"{'lag1':>10}"
        f"{'lag2':>10}"
        f"{'lag6':>10}"
        f"{'lag12':>10}"
        f"{'lag24':>10}"
        f"{'lag48':>10}"
        f"{'lag336':>10}"
    )


    for row in summary_rows:

        print(
            f"{row['representation']:<18}"
            f"{row['acf_lag1']:>10.4f}"
            f"{row['acf_lag2']:>10.4f}"
            f"{row['acf_lag6']:>10.4f}"
            f"{row['acf_lag12']:>10.4f}"
            f"{row['acf_lag24']:>10.4f}"
            f"{row['acf_lag48']:>10.4f}"
            f"{row['acf_lag336']:>10.4f}"
        )


    return (
        summary_rows,
        acf_rows,
        detector_rows,
    )


# ---------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------

def print_interpretation(
    summary,
):
    """
    Provide a conservative descriptive interpretation.

    This does not claim a formal causal result.
    """

    print()
    print("=" * 80)
    print(
        "DIAGNOSTIC INTERPRETATION"
    )
    print("=" * 80)


    for region in REGIONS:

        data = summary[
            summary[
                "region"
            ] == region
        ]


        signed = data[
            data[
                "representation"
            ] == "signed_error"
        ].iloc[0]


        absolute = data[
            data[
                "representation"
            ] == "absolute_error"
        ].iloc[0]


        squared = data[
            data[
                "representation"
            ] == "squared_error"
        ].iloc[0]


        print()
        print(
            f"{region}"
        )


        print(
            f"  signed lag-1 ACF   : "
            f"{signed['acf_lag1']:.4f}"
        )


        print(
            f"  absolute lag-1 ACF : "
            f"{absolute['acf_lag1']:.4f}"
        )


        print(
            f"  squared lag-1 ACF  : "
            f"{squared['acf_lag1']:.4f}"
        )


        signed_lag1 = abs(
            signed[
                "acf_lag1"
            ]
        )


        magnitude_lag1 = max(
            abs(
                absolute[
                    "acf_lag1"
                ]
            ),
            abs(
                squared[
                    "acf_lag1"
                ]
            ),
        )


        # These labels are heuristic diagnostic descriptions only.
        if (
            signed_lag1 >= 0.7
            and magnitude_lag1 >= 0.7
        ):

            print(
                "  Pattern: strong persistence exists in both "
                "residual direction and magnitude."
            )

            print(
                "  Implication: the dependence is already present "
                "in the forecast residual process; taking absolute "
                "values is not its sole source."
            )


        elif (
            signed_lag1 < 0.3
            and magnitude_lag1 >= 0.7
        ):

            print(
                "  Pattern: signed residual persistence is relatively "
                "weak but magnitude persistence is strong."
            )

            print(
                "  Implication: error-magnitude / volatility clustering "
                "is the stronger candidate."
            )


        elif magnitude_lag1 > signed_lag1 + 0.25:

            print(
                "  Pattern: magnitude dependence is substantially "
                "stronger than signed-residual dependence."
            )

            print(
                "  Implication: converting residuals to error magnitude "
                "appears to amplify temporal persistence."
            )


        else:

            print(
                "  Pattern: dependence is mixed."
            )

            print(
                "  Implication: inspect the complete ACF and sign-run "
                "diagnostics before selecting the next test."
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print(
        "DIAGNOSIS 03 — ERROR PERSISTENCE SOURCE"
    )
    print("=" * 80)


    print(
        "Question:"
    )


    print(
        "Is the short-range dependence already present in signed "
        "forecast residuals, or mainly in error magnitude?"
    )


    print()
    print(
        "Representations:"
    )

    print(
        "  signed_error    = e_t"
    )

    print(
        "  absolute_error  = |e_t|"
    )

    print(
        "  squared_error   = e_t^2"
    )


    print()
    print(
        "No detector parameters will be changed."
    )

    print(
        "No smoothing or prewhitening will be applied."
    )

    print(
        "runs.csv will not be modified."
    )


    all_summary_rows = []
    all_acf_rows = []
    all_detector_rows = []


    for region in REGIONS:

        (
            summary_rows,
            acf_rows,
            detector_rows,
        ) = diagnose_region(
            region
        )


        all_summary_rows.extend(
            summary_rows
        )


        all_acf_rows.extend(
            acf_rows
        )


        all_detector_rows.extend(
            detector_rows
        )


    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    summary = pd.DataFrame(
        all_summary_rows
    )


    acf = pd.DataFrame(
        all_acf_rows
    )


    detector_counts = pd.DataFrame(
        all_detector_rows
    )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    summary_path = (
        OUTPUT_DIR
        / "03_error_persistence_summary.csv"
    )


    acf_path = (
        OUTPUT_DIR
        / "03_error_persistence_acf.csv"
    )


    detector_path = (
        OUTPUT_DIR
        / "03_error_persistence_detector_counts.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    acf.to_csv(
        acf_path,
        index=False,
    )


    detector_counts.to_csv(
        detector_path,
        index=False,
    )


    # -----------------------------------------------------------------
    # Detector counts
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "DETECTOR COUNTS BY ERROR REPRESENTATION"
    )
    print("=" * 80)


    pivot = detector_counts.pivot(
        index=[
            "region",
            "detector",
        ],
        columns="representation",
        values="detections",
    ).reset_index()


    print(
        pivot.to_string(
            index=False
        )
    )


    # -----------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------

    print_interpretation(
        summary
    )


    print()
    print("=" * 80)
    print(
        "Saved diagnostic outputs"
    )
    print("=" * 80)


    print(
        summary_path
    )

    print(
        acf_path
    )

    print(
        detector_path
    )


    print()
    print(
        "Do not modify detectors.py yet."
    )


    print(
        "Use the signed-vs-magnitude dependence pattern to determine "
        "the appropriate Diagnosis 04."
    )


if __name__ == "__main__":
    main()