"""
07_diag_kswin_window_cycle.py

Diagnosis 07 — Are repeated KSWIN alarms associated with its
140-observation window cycle?
======================================================================

Background
----------
Diagnosis 06 found:

SA1 KSWIN
    detections        = 297
    minimum gap       = 2.917 days
    median gap        = 4.042 days

NSW1 KSWIN
    detections        = 292
    minimum gap       = 2.917 days
    median gap        = 4.125 days

At 48 half-hour observations per day:

    2.9167 days * 48 ≈ 140 observations

The current KSWIN configuration has:

    window_size = 140
    stat_size   = 30

This numerical match is interesting, but Diagnosis 06 did NOT establish
that the KSWIN window mechanism causes the high detection count.

Purpose
-------
This diagnosis asks:

    Is KSWIN simply unable to fire again before approximately
    window_size observations have accumulated?

or:

    Do a large proportion of its alarms actually occur very soon after
    that window-size boundary?

These are different conclusions.

Example A
---------
Suppose:

    minimum gap = 140

but:

    only 2% of gaps occur between 140 and 170

Then 140 is probably just a lower implementation/algorithmic boundary.

It would NOT explain why approximately 300 detections occur.


Example B
---------
Suppose:

    minimum gap = 140

and:

    60% of gaps occur between 140 and 170

Then alarms are strongly concentrated immediately after the detector
has accumulated approximately another window of observations.

That would support a KSWIN-specific repeated-window mechanism.


Approach
--------
The script analyzes existing Task 120 detections from:

    results/tables/detection/aemo_detection_changepoints.csv

For each detector and region it calculates consecutive detection gaps
directly in OBSERVATION INDICES:

    gap_samples =
        current_detected_index
        - previous_detected_index

This avoids timestamp-unit ambiguity.

For KSWIN it then compares these gaps with the configured window_size.

Metrics
-------
General gap statistics:

    minimum
    p10
    p25
    median
    mean
    p75
    p90
    maximum

KSWIN-specific:

    fraction gap < window_size
    fraction gap == window_size

    fraction gap <= window_size + 10
    fraction gap <= window_size + 30
    fraction gap <= window_size + 60

    fraction gap <= 2 * window_size
    fraction gap <= 3 * window_size

    excess above window:
        gap_samples - window_size

    ratio:
        gap_samples / window_size

Gap bands:

    < W
    = W
    W+1 ... W+10
    W+11 ... W+30
    W+31 ... W+60
    W+61 ... 2W
    2W+1 ... 3W
    >3W

The script also prints the most common exact KSWIN gaps.

Timestamp validation
--------------------
The AEMO detector test stream is standardized to 30-minute observations.

Therefore:

    gap_days_from_index = gap_samples / 48

should agree with:

    timestamp_difference_in_days

This script validates that relationship.

This also avoids the datetime-unit issue seen in the overlap section of
Diagnosis 06.


Interpretation
--------------
Strong evidence for window-cycle concentration would require more than:

    minimum_gap == window_size

The important evidence is whether many gaps are concentrated close to W.

For example:

    exactly W              15%
    W ... W+10             30%
    W ... W+30             55%

would be meaningful.

But:

    exactly W               0.5%
    W ... W+30              5%
    median gap             400 samples

would mean the 140-observation minimum is merely a lower boundary and
does not explain the overall alarm frequency.


Important
---------
This script does NOT:

    - rerun KSWIN
    - rerun ADWIN
    - rerun Page-Hinkley
    - change KSWIN window_size
    - change KSWIN alpha
    - tune any detector
    - change normalization
    - alter Task 120
    - alter T2
    - perform smoothing
    - perform prewhitening
    - write to runs.csv


Run
---
From repository root:

    python experiments/detection/07_diag_kswin_window_cycle.py


Outputs
-------
Saved under:

    results/tables/detection/diagnostics/

Files:

    07_alarm_gap_samples.csv
    07_alarm_gap_sample_summary.csv
    07_kswin_window_bands.csv
    07_kswin_window_summary.csv
    07_kswin_common_gaps.csv
"""


from pathlib import Path
import sys

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


# =====================================================================
# Paths
# =====================================================================

DETECTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "aemo_detection_changepoints.csv"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "diagnostics"
)


REGIONS = (
    "SA1",
    "NSW1",
)


DETECTORS = (
    "adwin",
    "page_hinkley",
    "kswin",
)


# Standardized AEMO half-hour frequency.
OBSERVATIONS_PER_DAY = 48


# =====================================================================
# Recover CURRENT KSWIN configuration
# =====================================================================

def get_kswin_configuration():
    """
    Recover KSWIN settings directly from the project's existing
    DETECTOR_REGISTRY.

    Nothing is hard-coded except the expected existence of the
    window_size parameter.
    """

    if "kswin" not in DETECTOR_REGISTRY:
        raise RuntimeError(
            "KSWIN is not present in DETECTOR_REGISTRY."
        )


    entry = DETECTOR_REGISTRY[
        "kswin"
    ]


    if "config" not in entry:
        raise RuntimeError(
            "KSWIN registry entry has no config."
        )


    config = entry[
        "config"
    ].copy()


    if "window_size" not in config:
        raise RuntimeError(
            "KSWIN configuration has no window_size."
        )


    window_size = int(
        config[
            "window_size"
        ]
    )


    if window_size <= 0:
        raise RuntimeError(
            "KSWIN window_size must be positive."
        )


    return (
        config,
        window_size,
    )


# =====================================================================
# Load stored Task 120 detections
# =====================================================================

def load_detections():
    """
    Load existing Task 120 detections.

    Detectors are NOT rerun.
    """

    if not DETECTION_PATH.exists():
        raise FileNotFoundError(
            f"Task 120 detection file not found:\n"
            f"{DETECTION_PATH}"
        )


    detections = pd.read_csv(
        DETECTION_PATH
    )


    required = {
        "region",
        "detector",
        "detected_index",
        "SETTLEMENTDATE",
    }


    missing = (
        required
        -
        set(
            detections.columns
        )
    )


    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )


    detections[
        "SETTLEMENTDATE"
    ] = pd.to_datetime(
        detections[
            "SETTLEMENTDATE"
        ],
        errors="raise",
    )


    indices = pd.to_numeric(
        detections[
            "detected_index"
        ],
        errors="raise",
    )


    if (
        (indices < 0)
        |
        (indices % 1 != 0)
    ).any():

        raise ValueError(
            "detected_index must contain "
            "non-negative integer values."
        )


    detections[
        "detected_index"
    ] = indices.astype(
        int
    )


    if detections.duplicated(
        [
            "region",
            "detector",
            "detected_index",
        ]
    ).any():

        raise RuntimeError(
            "Duplicate stored detections found."
        )


    return detections


# =====================================================================
# Build consecutive alarm gaps
# =====================================================================

def build_gap_rows(
    detections,
):
    """
    Calculate consecutive detection gaps using detector-test indices.

    This is the primary Diagnosis 07 representation.
    """

    all_rows = []


    for region in REGIONS:

        for detector in DETECTORS:

            subset = (
                detections[
                    (
                        detections[
                            "region"
                        ] == region
                    )
                    &
                    (
                        detections[
                            "detector"
                        ] == detector
                    )
                ]
                .sort_values(
                    "detected_index"
                )
                .reset_index(
                    drop=True
                )
            )


            if len(
                subset
            ) <= 1:
                continue


            for i in range(
                1,
                len(
                    subset
                ),
            ):

                previous = subset.iloc[
                    i - 1
                ]


                current = subset.iloc[
                    i
                ]


                previous_index = int(
                    previous[
                        "detected_index"
                    ]
                )


                current_index = int(
                    current[
                        "detected_index"
                    ]
                )


                gap_samples = (
                    current_index
                    - previous_index
                )


                if gap_samples <= 0:
                    raise RuntimeError(
                        f"Non-positive gap for "
                        f"{region}/{detector}."
                    )


                previous_time = pd.Timestamp(
                    previous[
                        "SETTLEMENTDATE"
                    ]
                )


                current_time = pd.Timestamp(
                    current[
                        "SETTLEMENTDATE"
                    ]
                )


                timestamp_gap_days = (
                    current_time
                    - previous_time
                ).total_seconds() / 86400.0


                index_gap_days = (
                    gap_samples
                    / OBSERVATIONS_PER_DAY
                )


                all_rows.append(
                    {
                        "region":
                            region,

                        "detector":
                            detector,

                        "previous_detection_number":
                            i,

                        "current_detection_number":
                            i + 1,

                        "previous_index":
                            previous_index,

                        "current_index":
                            current_index,

                        "previous_time":
                            previous_time,

                        "current_time":
                            current_time,

                        "gap_samples":
                            gap_samples,

                        "gap_days_from_index":
                            index_gap_days,

                        "gap_days_from_timestamp":
                            timestamp_gap_days,

                        "timestamp_index_difference_days":
                            (
                                timestamp_gap_days
                                - index_gap_days
                            ),
                    }
                )


    return pd.DataFrame(
        all_rows
    )


# =====================================================================
# Validate sample-gap / timestamp relationship
# =====================================================================

def validate_time_relationship(
    gaps,
):
    """
    Confirm that detector indices correspond to the standardized
    30-minute AEMO timeline.
    """

    differences = gaps[
        "timestamp_index_difference_days"
    ].to_numpy(
        dtype=float
    )


    max_difference = float(
        np.max(
            np.abs(
                differences
            )
        )
    )


    # Very small numerical tolerance.
    tolerance = 1e-9


    if max_difference > tolerance:

        problematic = gaps[
            np.abs(
                gaps[
                    "timestamp_index_difference_days"
                ]
            ) > tolerance
        ]


        print()
        print(
            "WARNING:"
        )


        print(
            "Some timestamp gaps do not exactly equal "
            "gap_samples / 48."
        )


        print(
            f"Maximum difference = "
            f"{max_difference:.9f} days"
        )


        print()
        print(
            problematic.head(
                10
            ).to_string(
                index=False
            )
        )


    else:

        print(
            "Index/timestamp validation passed:"
        )


        print(
            "  gap_samples / 48 exactly reproduces "
            "timestamp gaps."
        )


    return max_difference


# =====================================================================
# General gap summary
# =====================================================================

def summarize_gaps(
    gaps,
):
    """
    Summarize alarm spacing for every region/detector.
    """

    rows = []


    for (
        region,
        detector,
    ), group in gaps.groupby(
        [
            "region",
            "detector",
        ],
        sort=True,
    ):

        values = group[
            "gap_samples"
        ].to_numpy(
            dtype=float
        )


        rows.append(
            {
                "region":
                    region,

                "detector":
                    detector,

                "n_gaps":
                    len(
                        values
                    ),

                "min_gap_samples":
                    int(
                        np.min(
                            values
                        )
                    ),

                "p10_gap_samples":
                    float(
                        np.quantile(
                            values,
                            0.10,
                        )
                    ),

                "p25_gap_samples":
                    float(
                        np.quantile(
                            values,
                            0.25,
                        )
                    ),

                "median_gap_samples":
                    float(
                        np.median(
                            values
                        )
                    ),

                "mean_gap_samples":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "p75_gap_samples":
                    float(
                        np.quantile(
                            values,
                            0.75,
                        )
                    ),

                "p90_gap_samples":
                    float(
                        np.quantile(
                            values,
                            0.90,
                        )
                    ),

                "max_gap_samples":
                    int(
                        np.max(
                            values
                        )
                    ),

                "median_gap_days":
                    float(
                        np.median(
                            values
                        )
                        / OBSERVATIONS_PER_DAY
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# =====================================================================
# KSWIN window bands
# =====================================================================

def classify_window_band(
    gap,
    window_size,
):
    """
    Place one KSWIN inter-alarm gap into a window-relative band.
    """

    W = window_size


    if gap < W:
        return "<W"

    if gap == W:
        return "=W"

    if gap <= W + 10:
        return "W+1_to_W+10"

    if gap <= W + 30:
        return "W+11_to_W+30"

    if gap <= W + 60:
        return "W+31_to_W+60"

    if gap <= 2 * W:
        return "W+61_to_2W"

    if gap <= 3 * W:
        return "2W+1_to_3W"

    return ">3W"


BAND_ORDER = (
    "<W",
    "=W",
    "W+1_to_W+10",
    "W+11_to_W+30",
    "W+31_to_W+60",
    "W+61_to_2W",
    "2W+1_to_3W",
    ">3W",
)


def build_kswin_window_analysis(
    gaps,
    window_size,
):
    """
    Add KSWIN window-relative quantities to every KSWIN inter-alarm gap.
    """

    kswin = gaps[
        gaps[
            "detector"
        ] == "kswin"
    ].copy()


    if kswin.empty:
        raise RuntimeError(
            "No KSWIN gaps found."
        )


    kswin[
        "window_size"
    ] = window_size


    kswin[
        "gap_over_window"
    ] = (
        kswin[
            "gap_samples"
        ]
        / window_size
    )


    kswin[
        "excess_over_window"
    ] = (
        kswin[
            "gap_samples"
        ]
        - window_size
    )


    kswin[
        "window_band"
    ] = [
        classify_window_band(
            int(
                gap
            ),
            window_size,
        )
        for gap
        in kswin[
            "gap_samples"
        ]
    ]


    return kswin


# =====================================================================
# KSWIN band summary
# =====================================================================

def summarize_kswin_bands(
    kswin,
):
    """
    Count KSWIN gaps within each predefined window-relative band.
    """

    rows = []


    for region in REGIONS:

        subset = kswin[
            kswin[
                "region"
            ] == region
        ]


        total = len(
            subset
        )


        for band in BAND_ORDER:

            count = int(
                np.sum(
                    subset[
                        "window_band"
                    ] == band
                )
            )


            rows.append(
                {
                    "region":
                        region,

                    "window_band":
                        band,

                    "count":
                        count,

                    "fraction":
                        (
                            count
                            / total
                            if total
                            else float(
                                "nan"
                            )
                        ),
                }
            )


    return pd.DataFrame(
        rows
    )


# =====================================================================
# KSWIN cumulative window summary
# =====================================================================

def summarize_kswin_window_relationship(
    kswin,
    window_size,
):
    """
    Calculate cumulative proportions near the KSWIN window boundary.
    """

    rows = []


    W = window_size


    for region in REGIONS:

        subset = kswin[
            kswin[
                "region"
            ] == region
        ]


        gaps = subset[
            "gap_samples"
        ].to_numpy(
            dtype=int
        )


        if len(
            gaps
        ) == 0:
            continue


        def fraction(
            condition,
        ):

            return float(
                np.mean(
                    condition
                )
            )


        rows.append(
            {
                "region":
                    region,

                "window_size":
                    W,

                "n_gaps":
                    len(
                        gaps
                    ),

                "minimum_gap":
                    int(
                        np.min(
                            gaps
                        )
                    ),

                "median_gap":
                    float(
                        np.median(
                            gaps
                        )
                    ),

                "median_gap_over_window":
                    float(
                        np.median(
                            gaps
                            / W
                        )
                    ),

                "fraction_gap_lt_W":
                    fraction(
                        gaps < W
                    ),

                "fraction_gap_eq_W":
                    fraction(
                        gaps == W
                    ),

                "fraction_gap_le_W_plus_10":
                    fraction(
                        gaps
                        <= W + 10
                    ),

                "fraction_gap_le_W_plus_30":
                    fraction(
                        gaps
                        <= W + 30
                    ),

                "fraction_gap_le_W_plus_60":
                    fraction(
                        gaps
                        <= W + 60
                    ),

                "fraction_gap_le_2W":
                    fraction(
                        gaps
                        <= 2 * W
                    ),

                "fraction_gap_le_3W":
                    fraction(
                        gaps
                        <= 3 * W
                    ),

                "median_excess_over_W":
                    float(
                        np.median(
                            gaps
                            - W
                        )
                    ),

                "p25_excess_over_W":
                    float(
                        np.quantile(
                            gaps - W,
                            0.25,
                        )
                    ),

                "p75_excess_over_W":
                    float(
                        np.quantile(
                            gaps - W,
                            0.75,
                        )
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# =====================================================================
# Most common exact KSWIN gaps
# =====================================================================

def common_kswin_gaps(
    kswin,
    top_n=20,
):
    """
    Find most frequently occurring exact inter-alarm gaps.
    """

    rows = []


    for region in REGIONS:

        subset = kswin[
            kswin[
                "region"
            ] == region
        ]


        counts = (
            subset[
                "gap_samples"
            ]
            .value_counts()
            .sort_values(
                ascending=False
            )
            .head(
                top_n
            )
        )


        total = len(
            subset
        )


        rank = 1


        for (
            gap,
            count,
        ) in counts.items():

            rows.append(
                {
                    "region":
                        region,

                    "rank":
                        rank,

                    "gap_samples":
                        int(
                            gap
                        ),

                    "gap_days":
                        float(
                            gap
                            / OBSERVATIONS_PER_DAY
                        ),

                    "count":
                        int(
                            count
                        ),

                    "fraction_of_all_gaps":
                        float(
                            count
                            / total
                        ),

                    "gap_over_window":
                        float(
                            gap
                            /
                            kswin[
                                "window_size"
                            ].iloc[0]
                        ),
                }
            )


            rank += 1


    return pd.DataFrame(
        rows
    )


# =====================================================================
# Optional multiple-of-window diagnostic
# =====================================================================

def nearest_window_multiple_analysis(
    kswin,
    window_size,
):
    """
    Determine whether gaps also tend to concentrate near integer
    multiples of the KSWIN window size.

    This is descriptive only.

    Example:

        W
        2W
        3W

    A tolerance of +/-10 observations is used simply as a diagnostic.
    """

    rows = []


    tolerance = 10


    for region in REGIONS:

        subset = kswin[
            kswin[
                "region"
            ] == region
        ]


        gaps = subset[
            "gap_samples"
        ].to_numpy(
            dtype=float
        )


        if len(
            gaps
        ) == 0:
            continue


        ratios = (
            gaps
            / window_size
        )


        nearest_multiple = np.maximum(
            1,
            np.rint(
                ratios
            ),
        )


        nearest_multiple_gap = (
            nearest_multiple
            * window_size
        )


        distance = np.abs(
            gaps
            - nearest_multiple_gap
        )


        near_multiple = (
            distance
            <= tolerance
        )


        rows.append(
            {
                "region":
                    region,

                "n_gaps":
                    len(
                        gaps
                    ),

                "fraction_within_10_samples_of_window_multiple":
                    float(
                        np.mean(
                            near_multiple
                        )
                    ),

                "median_distance_to_nearest_window_multiple":
                    float(
                        np.median(
                            distance
                        )
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# =====================================================================
# Console interpretation
# =====================================================================

def print_interpretation(
    window_summary,
):
    """
    Print cautious interpretation.

    Minimum == W alone is NOT considered evidence that the window
    explains the high alarm frequency.
    """

    print()
    print(
        "=" * 80
    )

    print(
        "DIAGNOSTIC INTERPRETATION"
    )

    print(
        "=" * 80
    )


    for row in window_summary.itertuples(
        index=False
    ):

        print()
        print(
            row.region
        )


        print(
            f"  window size                     : "
            f"{row.window_size}"
        )


        print(
            f"  minimum gap                     : "
            f"{row.minimum_gap}"
        )


        print(
            f"  median gap                      : "
            f"{row.median_gap:.1f}"
        )


        print(
            f"  median gap / window             : "
            f"{row.median_gap_over_window:.3f}"
        )


        print(
            f"  exactly W                       : "
            f"{row.fraction_gap_eq_W:.2%}"
        )


        print(
            f"  <= W + 10                       : "
            f"{row.fraction_gap_le_W_plus_10:.2%}"
        )


        print(
            f"  <= W + 30                       : "
            f"{row.fraction_gap_le_W_plus_30:.2%}"
        )


        print(
            f"  <= W + 60                       : "
            f"{row.fraction_gap_le_W_plus_60:.2%}"
        )


        print(
            f"  <= 2W                           : "
            f"{row.fraction_gap_le_2W:.2%}"
        )


        print()


        # ---------------------------------------------------------
        # Deliberately conservative classification
        # ---------------------------------------------------------

        if (
            row.fraction_gap_le_W_plus_30
            >= 0.50
        ):

            print(
                "  Pattern:"
            )

            print(
                "    At least half of consecutive KSWIN alarms "
                "occur within 30 observations after one window length."
            )


            print(
                "  Interpretation:"
            )

            print(
                "    There is strong temporal concentration near the "
                "KSWIN window-size boundary."
            )


            print(
                "    The detector's window behaviour should therefore "
                "be investigated before threshold tuning."
            )


        elif (
            row.fraction_gap_le_W_plus_60
            >= 0.50
        ):

            print(
                "  Pattern:"
            )

            print(
                "    Many alarms occur relatively soon after one "
                "window length, but not immediately at the boundary."
            )


            print(
                "  Interpretation:"
            )

            print(
                "    Window-scale behaviour may contribute, but the "
                "window size alone does not fully explain the alarms."
            )


        else:

            print(
                "  Pattern:"
            )

            print(
                "    Most KSWIN alarm gaps are substantially larger "
                "than one window length."
            )


            print(
                "  Interpretation:"
            )

            print(
                "    The 140-observation minimum is likely only a "
                "lower spacing constraint."
            )


            print(
                "    It is not sufficient to explain the approximately "
                "300 AEMO detections."
            )


    print()
    print(
        "IMPORTANT:"
    )


    print(
        "Do not infer causality from minimum_gap == window_size."
    )


    print(
        "The useful evidence is the complete distribution of "
        "gap_samples relative to window_size."
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "DIAGNOSIS 07 — KSWIN WINDOW-CYCLE CHECK"
    )

    print(
        "=" * 80
    )


    print(
        "Question:"
    )


    print(
        "Are KSWIN's repeated AEMO alarms concentrated close to its "
        "configured window_size, or is 140 only a minimum spacing?"
    )


    print()
    print(
        "Stored Task 120 detections only."
    )


    print(
        "No detector will be rerun."
    )


    print(
        "No detector parameter will be changed."
    )


    # -----------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------

    (
        kswin_config,
        window_size,
    ) = get_kswin_configuration()


    print()
    print(
        "Current KSWIN configuration:"
    )


    for (
        key,
        value,
    ) in sorted(
        kswin_config.items()
    ):

        print(
            f"  {key:<15}: {value}"
        )


    print()
    print(
        f"Window size = {window_size} observations"
    )


    print(
        f"At 48 observations/day = "
        f"{window_size / OBSERVATIONS_PER_DAY:.4f} days"
    )


    # -----------------------------------------------------------------
    # Existing detections
    # -----------------------------------------------------------------

    detections = load_detections()


    print()
    print(
        "Stored detection counts:"
    )


    print(
        detections
        .groupby(
            [
                "region",
                "detector",
            ]
        )
        .size()
        .rename(
            "detections"
        )
        .to_string()
    )


    # -----------------------------------------------------------------
    # Gap construction
    # -----------------------------------------------------------------

    gaps = build_gap_rows(
        detections
    )


    print()
    print(
        "=" * 80
    )

    print(
        "INDEX / TIMESTAMP VALIDATION"
    )

    print(
        "=" * 80
    )


    validate_time_relationship(
        gaps
    )


    # -----------------------------------------------------------------
    # General summaries
    # -----------------------------------------------------------------

    gap_summary = summarize_gaps(
        gaps
    )


    kswin = build_kswin_window_analysis(
        gaps=gaps,
        window_size=window_size,
    )


    bands = summarize_kswin_bands(
        kswin
    )


    window_summary = (
        summarize_kswin_window_relationship(
            kswin=kswin,
            window_size=window_size,
        )
    )


    common_gaps = common_kswin_gaps(
        kswin,
        top_n=20,
    )


    multiple_summary = (
        nearest_window_multiple_analysis(
            kswin=kswin,
            window_size=window_size,
        )
    )


    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    gaps_path = (
        OUTPUT_DIR
        / "07_alarm_gap_samples.csv"
    )


    summary_path = (
        OUTPUT_DIR
        / "07_alarm_gap_sample_summary.csv"
    )


    bands_path = (
        OUTPUT_DIR
        / "07_kswin_window_bands.csv"
    )


    window_path = (
        OUTPUT_DIR
        / "07_kswin_window_summary.csv"
    )


    common_path = (
        OUTPUT_DIR
        / "07_kswin_common_gaps.csv"
    )


    gaps.to_csv(
        gaps_path,
        index=False,
    )


    gap_summary.to_csv(
        summary_path,
        index=False,
    )


    bands.to_csv(
        bands_path,
        index=False,
    )


    window_summary.to_csv(
        window_path,
        index=False,
    )


    common_gaps.to_csv(
        common_path,
        index=False,
    )


    # -----------------------------------------------------------------
    # General gap summary
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "ALARM GAPS IN OBSERVATIONS"
    )

    print(
        "=" * 80
    )


    print(
        gap_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


    # -----------------------------------------------------------------
    # KSWIN cumulative relationship
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "KSWIN GAP RELATION TO WINDOW SIZE"
    )

    print(
        "=" * 80
    )


    columns = [
        "region",
        "window_size",
        "n_gaps",
        "minimum_gap",
        "median_gap",
        "median_gap_over_window",
        "fraction_gap_lt_W",
        "fraction_gap_eq_W",
        "fraction_gap_le_W_plus_10",
        "fraction_gap_le_W_plus_30",
        "fraction_gap_le_W_plus_60",
        "fraction_gap_le_2W",
        "fraction_gap_le_3W",
    ]


    print(
        window_summary[
            columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # -----------------------------------------------------------------
    # Window bands
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "KSWIN WINDOW-RELATIVE GAP BANDS"
    )

    print(
        "=" * 80
    )


    print(
        bands.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # -----------------------------------------------------------------
    # Most common exact gaps
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "MOST COMMON EXACT KSWIN GAPS"
    )

    print(
        "=" * 80
    )


    for region in REGIONS:

        print()
        print(
            region
        )


        subset = common_gaps[
            common_gaps[
                "region"
            ] == region
        ]


        print(
            subset[
                [
                    "rank",
                    "gap_samples",
                    "gap_days",
                    "count",
                    "fraction_of_all_gaps",
                    "gap_over_window",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )


    # -----------------------------------------------------------------
    # Window multiples
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "DISTANCE TO NEAREST KSWIN WINDOW MULTIPLE"
    )

    print(
        "=" * 80
    )


    print(
        multiple_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # -----------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------

    print_interpretation(
        window_summary
    )


    # -----------------------------------------------------------------
    # Files
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "Saved diagnostic outputs"
    )

    print(
        "=" * 80
    )


    print(
        gaps_path
    )


    print(
        summary_path
    )


    print(
        bands_path
    )


    print(
        window_path
    )


    print(
        common_path
    )


    print()
    print(
        "Do not change KSWIN window_size or alpha based only on "
        "the minimum gap."
    )


    print(
        "First determine whether the entire gap distribution is "
        "actually concentrated near the 140-observation window."
    )


if __name__ == "__main__":
    main()