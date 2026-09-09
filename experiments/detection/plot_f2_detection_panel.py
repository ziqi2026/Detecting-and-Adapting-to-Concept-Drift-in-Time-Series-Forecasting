"""
Generate the F2 AEMO Detection Panel from saved pipeline outputs.

Inputs
------
- Task 120 detector diagnostic trace
- T2 detection-event matching results
- Fixed documented-event catalogue

The plotting script does not train forecasters, rerun detectors, or rematch
events.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DIAGNOSTIC_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "aemo_detection_diagnostics.csv.gz"
)

MATCHES_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "detection"
    / "T2_detection_matches.csv"
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "events.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "detection"
)

REGIONS = ("SA1", "NSW1")

DETECTORS = (
    ("adwin", "ADWIN"),
    ("page_hinkley", "Page-Hinkley"),
    ("kswin", "KSWIN"),
)


def load_inputs():
    """Load and validate the frozen upstream outputs used by F2."""
    diagnostics = pd.read_csv(
        DIAGNOSTIC_PATH,
        parse_dates=["SETTLEMENTDATE"],
    )
    matches = pd.read_csv(
        MATCHES_PATH,
        parse_dates=["SETTLEMENTDATE"],
    )
    events = pd.read_csv(EVENTS_PATH)

    events["start_date"] = pd.to_datetime(
        events["start_date"]
    )
    events["end_date"] = pd.to_datetime(
        events["end_date"]
    )

    tier1_events = events.loc[
        events["tier"].astype(int).eq(1)
    ].copy()

    return diagnostics, matches, tier1_events

def validate_inputs(
    diagnostics,
    matches,
    tier1_events,
):
    """Ensure F2 inputs remain aligned with Task 120 and T2."""
    key_columns = [
        "region",
        "detector",
        "detected_index",
    ]

    required_diagnostic_columns = {
        *key_columns,
        "SETTLEMENTDATE",
        "normalized_abs_error",
        "statistic",
        "threshold",
        "statistic_name",
        "threshold_name",
        "is_drift",
    }
    required_match_columns = {
        *key_columns,
        "SETTLEMENTDATE",
        "match_status",
        "match_tier",
    }

    missing_diagnostic = (
        required_diagnostic_columns
        - set(diagnostics.columns)
    )
    missing_matches = (
        required_match_columns
        - set(matches.columns)
    )

    if missing_diagnostic:
        raise RuntimeError(
            "Diagnostic input is missing columns: "
            f"{sorted(missing_diagnostic)}"
        )

    if missing_matches:
        raise RuntimeError(
            "T2 match input is missing columns: "
            f"{sorted(missing_matches)}"
        )

    valid_statuses = {
        "tier1",
        "tier2",
        "unmatched",
    }
    observed_statuses = set(
        matches["match_status"].dropna().unique()
    )

    if not observed_statuses.issubset(
        valid_statuses
    ):
        raise RuntimeError(
            "Unexpected T2 match status: "
            f"{sorted(observed_statuses)}"
        )

    diagnostic_detections = (
        diagnostics.loc[
            diagnostics["is_drift"].astype(bool),
            key_columns,
        ]
        .sort_values(key_columns)
        .reset_index(drop=True)
    )

    matched_detections = (
        matches[key_columns]
        .sort_values(key_columns)
        .reset_index(drop=True)
    )

    if not diagnostic_detections.equals(
        matched_detections
    ):
        raise RuntimeError(
            "Task 120 diagnostic detections do not "
            "match the T2 detection rows."
        )

    if len(tier1_events) != 5:
        raise RuntimeError(
            "Expected exactly five fixed Tier 1 events, "
            f"found {len(tier1_events)}."
        )

def plot_region(
    region,
    diagnostics,
    matches,
    tier1_events,
):
    """Create one six-panel F2 figure for an AEMO region."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 8),
        sharex="col",
        gridspec_kw={
            "height_ratios": [2.2, 1.2],
            "hspace": 0.10,
            "wspace": 0.18,
        },
    )

    for column, (
        detector_name,
        detector_title,
    ) in enumerate(DETECTORS):
        ax_stream = axes[0, column]
        ax_statistic = axes[1, column]

        trace = (
            diagnostics.loc[
                diagnostics["region"].eq(region)
                & diagnostics["detector"].eq(
                    detector_name
                )
            ]
            .sort_values("detected_index")
            .copy()
        )

        detections = (
            matches.loc[
                matches["region"].eq(region)
                & matches["detector"].eq(
                    detector_name
                )
            ]
            .sort_values("detected_index")
            .copy()
        )

        matched = detections.loc[
            detections["match_status"].isin(
                ["tier1", "tier2"]
            )
        ]
        unmatched = detections.loc[
            detections["match_status"].eq(
                "unmatched"
            )
        ]
        ax_stream.plot(
            trace["SETTLEMENTDATE"],
            trace["normalized_abs_error"],
            color="#4C78A8",
            linewidth=0.45,
            alpha=0.65,
            rasterized=True,
        )

        ax_stream.scatter(
            matched["SETTLEMENTDATE"],
            matched["normalized_abs_error"],
            s=15,
            color="#2CA02C",
            edgecolor="white",
            linewidth=0.25,
            zorder=4,
        )

        ax_stream.scatter(
            unmatched["SETTLEMENTDATE"],
            unmatched["normalized_abs_error"],
            s=12,
            color="#D62728",
            marker="x",
            linewidth=0.65,
            zorder=3,
        )

        if detector_name == "adwin":
            ax_statistic.step(
                trace["SETTLEMENTDATE"],
                trace["statistic"],
                where="post",
                color="#6F4E7C",
                linewidth=0.7,
            )
        else:
            ax_statistic.plot(
                trace["SETTLEMENTDATE"],
                trace["statistic"],
                color="#6F4E7C",
                linewidth=0.55,
                rasterized=True,
            )

        ax_statistic.plot(
            trace["SETTLEMENTDATE"],
            trace["threshold"],
            color="#E45756",
            linestyle="--",
            linewidth=1.0,
        )
        for _, event in tier1_events.iterrows():
            start = event["start_date"]
            end = event["end_date"]

            if end <= start:
                end = start + pd.Timedelta(
                    days=1
                )

            for axis in (
                ax_stream,
                ax_statistic,
            ):
                axis.axvspan(
                    start,
                    end,
                    color="#F2CF5B",
                    alpha=0.18,
                    linewidth=0,
                    zorder=0,
                )

            ax_stream.text(
                start,
                0.98,
                event["event_id"],
                transform=ax_stream.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="left",
                fontsize=6.5,
                color="#6B5B00",
            )

        ax_stream.set_title(
            detector_title,
            fontsize=12,
            fontweight="bold",
        )
        ax_stream.grid(
            alpha=0.18,
            linewidth=0.5,
        )
        ax_statistic.grid(
            alpha=0.18,
            linewidth=0.5,
        )
        ax_statistic.set_xlabel(
            "Settlement date"
        )

        if column == 0:
            ax_stream.set_ylabel(
                "Calibration-normalized\n"
                "absolute forecast error"
            )
            ax_statistic.set_ylabel(
                "Detector diagnostic"
            )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#4C78A8",
            linewidth=1.0,
            label="Monitored error stream",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2CA02C",
            markeredgecolor="#2CA02C",
            markersize=6,
            label="Matched detection",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#D62728",
            markersize=6,
            linestyle="none",
            label="Unmatched detection",
        ),
        Line2D(
            [0],
            [0],
            color="#7A5195",
            linewidth=1.0,
            label="Detector diagnostic",
        ),
        Line2D(
            [0],
            [0],
            color="#F58518",
            linewidth=1.0,
            linestyle="--",
            label="Threshold / plotting boundary",
        ),
        Patch(
            facecolor="#F2CF5B",
            alpha=0.18,
            label="Tier 1 event period",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"F2 Detection Panel — {region}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        (
            "ADWIN uses a binary drift-decision trace with a 0.5 plotting "
            "boundary because River does not expose its internal dynamic bound."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(
        top=0.88,
        bottom=0.09,
        left=0.07,
        right=0.99,
        hspace=0.12,
        wspace=0.18,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    png_path = OUTPUT_DIR / f"F2_detection_panel_{region}.png"
    pdf_path = OUTPUT_DIR / f"F2_detection_panel_{region}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(fig)

    return png_path, pdf_path
def main():
    """Validate saved inputs and generate both regional F2 panels."""
    print("Loading saved F2 pipeline outputs...")

    (
        diagnostics,
        matches,
        tier1_events,
    ) = load_inputs()

    validate_inputs(
        diagnostics,
        matches,
        tier1_events,
    )

    print(
        "Input validation passed: "
        f"{len(diagnostics)} diagnostic rows, "
        f"{len(matches)} T2 detection rows, "
        f"{len(tier1_events)} Tier 1 events."
    )

    for region in REGIONS:
        png_path, pdf_path = plot_region(
            region,
            diagnostics,
            matches,
            tier1_events,
        )

        print(f"Saved {region} PNG: {png_path}")
        print(f"Saved {region} PDF: {pdf_path}")

    print("F2 Detection Panel generation complete.")


if __name__ == "__main__":
    main()