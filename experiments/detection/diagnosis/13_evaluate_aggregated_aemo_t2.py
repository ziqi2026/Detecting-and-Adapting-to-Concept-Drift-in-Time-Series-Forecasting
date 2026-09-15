"""
13_evaluate_aggregated_aemo_t2.py

Diagnostic T2-style evaluation of the frozen aggregated AEMO detections.

IMPORTANT
---------
This script DOES NOT:
- rerun forecasting
- rerun detectors
- modify detectors.py
- modify runs.csv
- overwrite official Task 120 outputs
- overwrite official T2 outputs

It evaluates the stored diagnostic detections from Validation 12 using
the SAME AEMO event-matching implementation and protocol as the official
T2 script.

Inputs
------
results/tables/detection/diagnostics/
    12_aemo_aggregated_all_changepoints.csv
    12_aemo_aggregated_all_summary.csv

Outputs
-------
results/tables/detection/diagnostics/
    13_T2_aggregated_aemo_detection.csv
    13_T2_aggregated_aemo_detection.md
    13_T2_aggregated_detection_matches.csv
    13_T2_aggregated_tier1_event_details.csv
    13_T2_aggregated_detection_summary.csv

The table contains:
    3h_MAE
    6h_MAE

for:
    ADWIN
    Page-Hinkley
    KSWIN

and:
    SA1
    NSW1
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# =====================================================================
# Project imports
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from drift_lab.data_loader import (
    load_and_split_aemo,
)

from drift_lab.evaluation import (
    AEMO_TIER1_COLUMNS,
    AEMO_TIMEZONE,
    aemo_timestamp,
    evaluate_aemo_detections,
    load_aemo_events,
    validate_aemo_protocol,
)


# =====================================================================
# Paths
# =====================================================================

DIAG_DIR = (
    ROOT
    / "results"
    / "tables"
    / "detection"
    / "diagnostics"
)

DETECTIONS_INPUT = (
    DIAG_DIR
    / "12_aemo_aggregated_all_changepoints.csv"
)

SUMMARY_INPUT = (
    DIAG_DIR
    / "12_aemo_aggregated_all_summary.csv"
)

EVENTS = (
    ROOT
    / "data"
    / "events.csv"
)

PROTOCOL = (
    ROOT
    / "docs"
    / "aemo_matching_protocol.json"
)

TABLE_CSV = (
    DIAG_DIR
    / "13_T2_aggregated_aemo_detection.csv"
)

TABLE_MD = (
    DIAG_DIR
    / "13_T2_aggregated_aemo_detection.md"
)

MATCHES_OUTPUT = (
    DIAG_DIR
    / "13_T2_aggregated_detection_matches.csv"
)

EVENT_DETAILS_OUTPUT = (
    DIAG_DIR
    / "13_T2_aggregated_tier1_event_details.csv"
)

SUMMARY_OUTPUT = (
    DIAG_DIR
    / "13_T2_aggregated_detection_summary.csv"
)


# =====================================================================
# Expected runs
# =====================================================================

REGIONS = (
    "SA1",
    "NSW1",
)

RESOLUTIONS = (
    "3h_MAE",
    "6h_MAE",
)

DETECTORS = (
    "adwin",
    "page_hinkley",
    "kswin",
)


# =====================================================================
# Helpers
# =====================================================================

def markdown_table(frame):
    """Convert a DataFrame into a simple Markdown table."""

    display = (
        frame
        .fillna("N/A")
        .astype(str)
    )

    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(
            ["---"] * len(display.columns)
        ) + " |",
    ]

    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in display.to_numpy()
    )

    return "\n".join(lines)


def make_empty_detection_frame():
    """
    Empty detection frame for runs with zero alarms.
    """

    return pd.DataFrame(
        columns=[
            "region",
            "detector",
            "resolution",
            "detected_index",
            "SETTLEMENTDATE",
        ]
    )


def move_metadata_to_front(
    frame,
    resolution,
    detector,
    region,
):
    """
    Safely assign metadata columns.

    Some evaluator outputs may already contain one or more of these
    columns, so assignment is used instead of DataFrame.insert().
    """

    frame = frame.copy()

    frame["resolution"] = resolution
    frame["detector"] = detector
    frame["region"] = region

    front = [
        "resolution",
        "detector",
        "region",
    ]

    remaining = [
        column
        for column in frame.columns
        if column not in front
    ]

    return frame[
        front + remaining
    ]


# =====================================================================
# Validate stored diagnostic inputs
# =====================================================================

def prepare_runs():
    """
    Build all 12 frozen diagnostic runs.

    The summary CSV acts as the run roster so that zero-detection runs
    are included.
    """

    detections = pd.read_csv(
        DETECTIONS_INPUT
    )

    summaries = pd.read_csv(
        SUMMARY_INPUT
    )

    required_detection_columns = {
        "region",
        "resolution",
        "detector",
        "detected_index",
        "SETTLEMENTDATE",
        "detector_config",
    }

    required_summary_columns = {
        "region",
        "resolution",
        "detector",
        "detections",
        "block_size",
        "detector_config",
    }

    if not required_detection_columns.issubset(
        detections.columns
    ):
        missing = (
            required_detection_columns
            - set(detections.columns)
        )

        raise ValueError(
            f"Missing detection columns: {sorted(missing)}"
        )

    if not required_summary_columns.issubset(
        summaries.columns
    ):
        missing = (
            required_summary_columns
            - set(summaries.columns)
        )

        raise ValueError(
            f"Missing summary columns: {sorted(missing)}"
        )

    # -----------------------------------------------------------------
    # Expected roster
    # -----------------------------------------------------------------

    expected = {
        (
            region,
            resolution,
            detector,
        )
        for region in REGIONS
        for resolution in RESOLUTIONS
        for detector in DETECTORS
    }

    actual = set(
        summaries[
            [
                "region",
                "resolution",
                "detector",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    if actual != expected:
        missing = expected - actual
        extra = actual - expected

        raise ValueError(
            "Diagnostic run roster mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra: {sorted(extra)}"
        )

    if summaries.duplicated(
        [
            "region",
            "resolution",
            "detector",
        ]
    ).any():
        raise ValueError(
            "Duplicate rows in diagnostic summary."
        )

    if detections.duplicated(
        [
            "region",
            "resolution",
            "detector",
            "detected_index",
        ]
    ).any():
        raise ValueError(
            "Duplicate diagnostic changepoints."
        )

    prepared = []

    # =================================================================
    # Validate against original AEMO test timeline
    # =================================================================

    for region in REGIONS:

        (
            _,
            _train,
            _calibration,
            test,
        ) = load_and_split_aemo(
            region,
            ROOT / "data/raw",
        )

        test_times = pd.DatetimeIndex(
            test.SETTLEMENTDATE
        ).tz_localize(
            AEMO_TIMEZONE
        )

        for resolution in RESOLUTIONS:

            for detector in DETECTORS:

                summary_match = summaries[
                    (summaries.region == region)
                    & (summaries.resolution == resolution)
                    & (summaries.detector == detector)
                ]

                if len(summary_match) != 1:
                    raise ValueError(
                        f"Expected one summary row for "
                        f"{region}/{resolution}/{detector}; "
                        f"found {len(summary_match)}."
                    )

                summary_row = summary_match.iloc[0]

                rows = detections[
                    (detections.region == region)
                    & (detections.resolution == resolution)
                    & (detections.detector == detector)
                ].copy()

                expected_count = int(
                    summary_row["detections"]
                )

                if len(rows) != expected_count:
                    raise ValueError(
                        f"{region}/{resolution}/{detector}: "
                        f"summary says {expected_count} detections "
                        f"but changepoint file contains {len(rows)}."
                    )

                # -----------------------------------------------------
                # Zero detection run
                # -----------------------------------------------------

                if rows.empty:

                    rows = make_empty_detection_frame()

                else:

                    indices = pd.to_numeric(
                        rows["detected_index"],
                        errors="raise",
                    )

                    invalid = (
                        (indices % 1 != 0)
                        | (indices < 0)
                        | (indices >= len(test_times))
                    )

                    if invalid.any():
                        raise ValueError(
                            f"{region}/{resolution}/{detector}: "
                            "detected_index outside original "
                            "AEMO test stream."
                        )

                    indices = indices.astype(int)

                    actual_times = pd.DatetimeIndex(
                        [
                            aemo_timestamp(timestamp)
                            for timestamp
                            in rows["SETTLEMENTDATE"]
                        ]
                    )

                    expected_times = test_times[
                        indices
                    ]

                    if not (
                        actual_times
                        == expected_times
                    ).all():
                        raise ValueError(
                            f"{region}/{resolution}/{detector}: "
                            "detected_index / SETTLEMENTDATE "
                            "mapping does not reproduce."
                        )

                    if not indices.is_monotonic_increasing:
                        raise ValueError(
                            f"{region}/{resolution}/{detector}: "
                            "detections are not chronological."
                        )

                prepared.append(
                    {
                        "region":
                            region,

                        "resolution":
                            resolution,

                        "detector":
                            detector,

                        "rows":
                            rows,

                        "times":
                            test_times,

                        "expected_count":
                            expected_count,

                        "block_size":
                            int(
                                summary_row[
                                    "block_size"
                                ]
                            ),

                        "detector_config":
                            summary_row[
                                "detector_config"
                            ],
                    }
                )

    return prepared


# =====================================================================
# Build one T2-style row
# =====================================================================

def build_table_row(
    result,
    detector,
    region,
    resolution,
):
    """
    Build same core table fields as official T2, with resolution added.
    """

    row = {
        "resolution": resolution,
        "detector": detector,
        "region": region,
    }

    # =================================================================
    # Tier 1 event delays
    # =================================================================

    event_details = result[
        "tier1_events"
    ]

    for (
        event_id,
        column_name,
    ) in AEMO_TIER1_COLUMNS.items():

        event_rows = event_details[
            event_details.event_id
            == event_id
        ]

        if len(event_rows) != 1:
            raise ValueError(
                f"Expected exactly one Tier 1 event row "
                f"for {event_id}; found {len(event_rows)}."
            )

        event = event_rows.iloc[0]

        applicable = bool(
            event["applicable"]
        )

        detected = bool(
            event["detected"]
        )

        if not applicable:

            row[
                column_name
            ] = "not applicable"

        elif not detected:

            row[
                column_name
            ] = "not detected"

        else:

            delay = float(
                event["delay_days"]
            )

            if (
                not np.isfinite(delay)
                or delay < 0
            ):
                raise ValueError(
                    f"Invalid delay for event {event_id}."
                )

            row[
                column_name
            ] = round(
                delay,
                3,
            )

    # =================================================================
    # Detection accounting
    # =================================================================

    totals = result[
        "totals"
    ]

    required_totals = {
        "detections",
        "tier1_matched",
        "tier2_matched",
        "unmatched",
    }

    if not required_totals.issubset(
        totals.keys()
    ):
        raise ValueError(
            "Unexpected totals returned by evaluator. "
            f"Available: {sorted(totals.keys())}"
        )

    detections = int(
        totals["detections"]
    )

    tier1_matched = int(
        totals["tier1_matched"]
    )

    tier2_matched = int(
        totals["tier2_matched"]
    )

    unmatched = int(
        totals["unmatched"]
    )

    if (
        detections
        != tier1_matched
        + tier2_matched
        + unmatched
    ):
        raise ValueError(
            f"{region}/{resolution}/{detector}: "
            "detection accounting does not balance."
        )

    # Same definitions as official T2
    row["tier1_unmatched"] = (
        detections
        - tier1_matched
    )

    row["tier2_contextual"] = (
        tier2_matched
    )

    row["unmatched"] = (
        unmatched
    )

    row["precision"] = (
        (
            tier1_matched
            + tier2_matched
        )
        / detections
        if detections
        else float("nan")
    )

    return row


# =====================================================================
# Write table
# =====================================================================

def write_table(
    table,
):
    """
    Save CSV and Markdown.
    """

    table.to_csv(
        TABLE_CSV,
        index=False,
        na_rep="N/A",
    )

    shown = table.copy()

    shown[
        "precision"
    ] = shown[
        "precision"
    ].map(
        lambda x:
            f"{x:.2%}"
            if pd.notna(x)
            else "N/A"
    )

    text = (
        "# Diagnostic T2 — Aggregated AEMO detections vs documented events\n\n"
        + markdown_table(shown)
        + "\n\n"
        + "This diagnostic table applies the same AEMO event-matching "
          "implementation and protocol as the official T2 evaluation, "
          "but to the frozen 3-hour and 6-hour aggregated detector outputs. "
          "Delays are fractional days from the frozen reference date, "
          "rounded to 3 decimals. "
          "Tier 1 unmatched counts detections that did not match a primary "
          "Tier 1 event. Tier 2 contextual reports how many of those "
          "detections were subsequently matched to Tier 2 events. "
          "Unmatched counts detections matched to neither tier. "
          "Precision is the proportion of detections matched to either "
          "Tier 1 or Tier 2. Historical events are contextual references; "
          "unmatched detections are not automatically false positives.\n"
    )

    TABLE_MD.write_text(
        text,
        encoding="utf-8",
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print(
        "=" * 120
    )

    print(
        "DIAGNOSTIC 13 — T2 EVENT MATCHING FOR AGGREGATED AEMO DETECTIONS"
    )

    print(
        "=" * 120
    )

    print()
    print(
        "Stored detections only — no forecaster or detector reruns."
    )

    print(
        "Using the existing AEMO matching protocol."
    )

    print(
        "Official T2 outputs and runs.csv will NOT be modified."
    )

    # =================================================================
    # Load protocol and events
    # =================================================================

    protocol = json.loads(
        PROTOCOL.read_text(
            encoding="utf-8"
        )
    )

    validate_aemo_protocol(
        protocol
    )

    events = load_aemo_events(
        EVENTS
    )

    # =================================================================
    # Prepare diagnostic runs
    # =================================================================

    prepared = prepare_runs()

    table_rows = []
    audit_frames = []
    event_frames = []
    summary_rows = []

    # =================================================================
    # Evaluate all 12 runs
    # =================================================================

    for run in prepared:

        region = run[
            "region"
        ]

        resolution = run[
            "resolution"
        ]

        detector = run[
            "detector"
        ]

        rows = run[
            "rows"
        ]

        times = run[
            "times"
        ]

        result = evaluate_aemo_detections(
            rows,
            events,
            region,
            protocol=protocol,
            stream_start=times[0],
            stream_end_exclusive=(
                times[-1]
                + pd.Timedelta(
                    minutes=30
                )
            ),
        )

        # -------------------------------------------------------------
        # Verify count
        # -------------------------------------------------------------

        evaluated_count = int(
            result[
                "totals"
            ][
                "detections"
            ]
        )

        if (
            evaluated_count
            != run[
                "expected_count"
            ]
        ):
            raise RuntimeError(
                f"{region}/{resolution}/{detector}: "
                f"expected {run['expected_count']} detections "
                f"but evaluator returned {evaluated_count}."
            )

        # -------------------------------------------------------------
        # Main table
        # -------------------------------------------------------------

        table_rows.append(
            build_table_row(
                result=result,
                detector=detector,
                region=region,
                resolution=resolution,
            )
        )

        # -------------------------------------------------------------
        # Detection-level audit
        #
        # FIX:
        # Do not use DataFrame.insert() because some of these columns
        # may already exist in the evaluator output.
        # -------------------------------------------------------------

        audit = move_metadata_to_front(
            frame=result[
                "audit"
            ],
            resolution=resolution,
            detector=detector,
            region=region,
        )

        audit_frames.append(
            audit
        )

        # -------------------------------------------------------------
        # Tier-1 event details
        #
        # Same safe metadata handling.
        # -------------------------------------------------------------

        event_details = move_metadata_to_front(
            frame=result[
                "tier1_events"
            ],
            resolution=resolution,
            detector=detector,
            region=region,
        )

        event_frames.append(
            event_details
        )

        # -------------------------------------------------------------
        # Compact summary
        # -------------------------------------------------------------

        summary_rows.append(
            {
                "resolution":
                    resolution,

                "detector":
                    detector,

                "region":
                    region,

                "block_size":
                    run[
                        "block_size"
                    ],

                "detector_config":
                    run[
                        "detector_config"
                    ],

                **result[
                    "totals"
                ],
            }
        )

        print(
            f"{resolution:6s} | "
            f"{detector:13s} | "
            f"{region:4s} | "
            f"{result['totals']}"
        )

    # =================================================================
    # Save
    # =================================================================

    DIAG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pd.DataFrame(
        table_rows
    )

    table = (
        table
        .sort_values(
            [
                "resolution",
                "region",
                "detector",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    write_table(
        table
    )

    # =================================================================
    # Detection-level matching audit
    # =================================================================

    nonempty_audits = [
        frame
        for frame in audit_frames
        if not frame.empty
    ]

    if nonempty_audits:

        pd.concat(
            nonempty_audits,
            ignore_index=True,
        ).to_csv(
            MATCHES_OUTPUT,
            index=False,
        )

    else:

        pd.DataFrame(
            columns=[
                "resolution",
                "detector",
                "region",
            ]
        ).to_csv(
            MATCHES_OUTPUT,
            index=False,
        )

    # =================================================================
    # Tier-1 event details
    # =================================================================

    if event_frames:

        pd.concat(
            event_frames,
            ignore_index=True,
        ).to_csv(
            EVENT_DETAILS_OUTPUT,
            index=False,
        )

    else:

        pd.DataFrame().to_csv(
            EVENT_DETAILS_OUTPUT,
            index=False,
        )

    # =================================================================
    # Compact summary
    # =================================================================

    pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "resolution",
            "region",
            "detector",
        ]
    ).to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # =================================================================
    # Console output
    # =================================================================

    shown = table.copy()

    shown[
        "precision"
    ] = shown[
        "precision"
    ].map(
        lambda x:
            f"{x:.2%}"
            if pd.notna(x)
            else "N/A"
    )

    print()
    print(
        "=" * 180
    )

    print(
        "DIAGNOSTIC T2 — AGGREGATED AEMO VS DOCUMENTED EVENTS"
    )

    print(
        "=" * 180
    )

    print(
        shown.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "SAVED"
    )

    print(
        "=" * 120
    )

    print(
        TABLE_CSV
    )

    print(
        TABLE_MD
    )

    print(
        MATCHES_OUTPUT
    )

    print(
        EVENT_DETAILS_OUTPUT
    )

    print(
        SUMMARY_OUTPUT
    )

    print()
    print(
        "These are diagnostic outputs only."
    )

    print(
        "Official Task 120, official T2, detectors.py "
        "and runs.csv were not changed."
    )


if __name__ == "__main__":
    main()