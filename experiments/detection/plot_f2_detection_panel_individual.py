"""
F2 individual AEMO drift-detection figures.

Creates six figures:
    SA1 x ADWIN
    SA1 x Page-Hinkley
    SA1 x KSWIN
    NSW1 x ADWIN
    NSW1 x Page-Hinkley
    NSW1 x KSWIN

Each figure shows:
    observed AEMO demand
    15-day rolling mean for visual context
    documented Tier-1 event periods
    official matched and unmatched detections
    detector statistic and threshold

Tier-1 AEMO events are documented historical references,
not known ground-truth changepoints.

For ADWIN, River's adaptive window width is shown as the
diagnostic state. River does not expose the internal ADWIN
decision statistic or dynamic cut threshold, so no threshold
is plotted for ADWIN. Official detections remain the
refractory-filtered outputs of the shared detection pipeline.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from drift_lab.data_loader import (
    load_and_split_aemo,
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

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
    / "F2_individual"
)


# Figure settings

REGIONS = (
    "SA1",
    "NSW1",
)

DETECTORS = (
    (
        "adwin",
        "ADWIN",
    ),
    (
        "page_hinkley",
        "Page-Hinkley",
    ),
    (
        "kswin",
        "KSWIN",
    ),
)


# Colour palette

OBSERVED = "#B8D3DD"
SMOOTH = "#247B98"

EVENT = "#263D4A"

MATCHED = "#4D8F70"
UNMATCHED = "#B64B4B"

WINDOW = "#F4E3CE"

STATISTIC = "#247B98"
THRESHOLD = "#B64B4B"

GRID = "#D8E0E4"
TEXT = "#243746"


def load_inputs():
    """Load saved F2 diagnostics, T2 matches, and Tier-1 events."""

    diagnostics = pd.read_csv(
        DIAGNOSTIC_PATH,
        parse_dates=[
            "SETTLEMENTDATE",
        ],
    )

    matches = pd.read_csv(
        MATCHES_PATH,
        parse_dates=[
            "SETTLEMENTDATE",
        ],
    )

    events = pd.read_csv(
        EVENTS_PATH,
    )

    events[
        "start_date"
    ] = pd.to_datetime(
        events[
            "start_date"
        ]
    )

    events[
        "end_date"
    ] = pd.to_datetime(
        events[
            "end_date"
        ]
    )

    tier1_events = (
        events.loc[
            events[
                "tier"
            ]
            .astype(int)
            .eq(1)
        ]
        .copy()
    )

    return (
        diagnostics,
        matches,
        tier1_events,
    )


def load_test_demand(
    region,
):
    """Load the official 30-minute TEST demand stream."""

    _, _, _, test = (
        load_and_split_aemo(
            region,
            data_path=DATA_PATH,
        )
    )

    demand = (
        test[
            [
                "SETTLEMENTDATE",
                "TOTALDEMAND",
            ]
        ]
        .copy()
        .sort_values(
            "SETTLEMENTDATE"
        )
        .reset_index(
            drop=True
        )
    )

    demand[
        "SETTLEMENTDATE"
    ] = pd.to_datetime(
        demand[
            "SETTLEMENTDATE"
        ]
    )

    demand[
        "mean_15d"
    ] = (
        demand[
            "TOTALDEMAND"
        ]
        .rolling(
            window=15 * 48,
            min_periods=48,
            center=True,
        )
        .mean()
    )

    return demand


def validate_inputs(
    diagnostics,
    matches,
):
    """Check that saved outputs contain the required F2 fields."""

    required_diagnostic = {
        "region",
        "detector",
        "stream_index",
        "SETTLEMENTDATE",
        "statistic",
        "threshold",
    }

    required_matches = {
        "region",
        "detector",
        "detected_index",
        "SETTLEMENTDATE",
        "match_status",
        "match_tier",
    }

    missing_diagnostic = (
        required_diagnostic
        - set(
            diagnostics.columns
        )
    )

    missing_matches = (
        required_matches
        - set(
            matches.columns
        )
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


def event_applies(
    event,
    region,
):
    """Return whether an event applies to the selected region."""

    event_region = str(
        event.get(
            "region",
            "",
        )
    ).strip()

    valid_regions = {
        value.strip()
        for value
        in event_region.split("|")
    }

    return (
        event_region
        in {
            "",
            "NEM",
            region,
        }
        or region
        in valid_regions
    )


def event_short_name(
    event,
):
    """Return a compact event label."""

    event_id = str(
        event.get(
            "event_id",
            "",
        )
    ).strip()

    name = str(
        event.get(
            "event_name",
            event_id,
        )
    ).strip()

    if len(name) > 28:
        return event_id

    return name


def detection_label(
    row,
):
    """Return a compact match label."""

    status = row[
        "match_status"
    ]

    if status == "tier1":
        return "MATCHED / T1"

    if status == "tier2":
        return "MATCHED / T2"

    return "UNMATCHED"


def statistic_axis_label(
    detector_name,
):
    """Return the appropriate statistic-axis label."""

    if detector_name == "adwin":
        return (
            "Adaptive window width"
        )

    if detector_name == "page_hinkley":
        return (
            "Cumulative increase"
        )

    if detector_name == "kswin":
        return (
            r"$-\log_{10}(p)$"
        )

    return "Detector statistic"


def add_event_references(
    ax_top,
    ax_bottom,
    tier1_events,
    region,
):
    """Add Tier-1 event periods and event-start references."""

    applicable = (
        tier1_events.loc[
            tier1_events.apply(
                lambda row:
                event_applies(
                    row,
                    region,
                ),
                axis=1,
            )
        ]
        .copy()
    )

    for _, event in (
        applicable.iterrows()
    ):

        start = event[
            "start_date"
        ]

        end = event[
            "end_date"
        ]

        for ax in (
            ax_top,
            ax_bottom,
        ):

            ax.axvspan(
                start,
                end,
                facecolor=WINDOW,
                edgecolor="none",
                alpha=0.70,
                zorder=0,
            )

        ax_top.axvline(
            start,
            color=EVENT,
            linewidth=1.15,
            zorder=4,
        )

        ax_top.text(
            start,
            0.02,
            event_short_name(
                event
            ),
            transform=(
                ax_top
                .get_xaxis_transform()
            ),
            color=EVENT,
            fontsize=6.2,
            fontweight="bold",
            ha="right",
            va="bottom",
            rotation=90,
            clip_on=True,
            zorder=7,
        )


def add_detection_lines(
    ax,
    detections,
):
    """Add official detection timestamps to the demand panel."""

    for _, row in (
        detections.iterrows()
    ):

        matched = (
            row[
                "match_status"
            ]
            in {
                "tier1",
                "tier2",
            }
        )

        colour = (
            MATCHED
            if matched
            else UNMATCHED
        )

        ax.axvline(
            row[
                "SETTLEMENTDATE"
            ],
            color=colour,
            linestyle=(
                0,
                (
                    4,
                    3,
                ),
            ),
            linewidth=1.15,
            alpha=0.90,
            zorder=5,
        )


def add_detection_labels(
    ax,
    detections,
):
    """Add compact detection labels fully inside the plot."""

    levels = (
        0.82,
        0.74,
        0.66,
    )

    for i, (_, row) in enumerate(
        detections.iterrows()
    ):

        matched = (
            row["match_status"]
            in {
                "tier1",
                "tier2",
            }
        )

        colour = (
            MATCHED
            if matched
            else UNMATCHED
        )

        y = levels[
            i % len(levels)
        ]

        ax.text(
            row["SETTLEMENTDATE"],
            y,
            detection_label(row),
            transform=ax.get_xaxis_transform(),
            color=colour,
            fontsize=6.2,
            fontweight="bold",
            ha="left",
            va="top",
            rotation=90,
            rotation_mode="anchor",
            clip_on=True,
            zorder=8,
        )

def draw_threshold(
    ax,
    detector_name,
    trace,
):
    """Draw the detector threshold or diagnostic cut."""

    if detector_name == "adwin":
        return

    threshold = pd.to_numeric(
        trace[
            "threshold"
        ],
        errors="coerce",
    )

    finite = threshold[
        np.isfinite(
            threshold
        )
    ]

    if finite.empty:
        return

    if np.allclose(
        finite.to_numpy(),
        finite.iloc[0],
        rtol=1e-10,
        atol=1e-12,
    ):

        ax.axhline(
            y=float(
                finite.iloc[0]
            ),
            color=THRESHOLD,
            linestyle=(
                0,
                (
                    5,
                    4,
                ),
            ),
            linewidth=1.15,
            alpha=0.90,
            zorder=3,
        )

        return

    ax.plot(
        trace[
            "SETTLEMENTDATE"
        ],
        threshold,
        color=THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        alpha=0.90,
        zorder=3,
    )


def add_detection_markers(
    ax,
    trace,
    detections,
):
    """Mark official detections on the statistic panel."""

    statistic_lookup = (
        trace[
            [
                "stream_index",
                "statistic",
            ]
        ]
        .drop_duplicates(
            "stream_index"
        )
        .set_index(
            "stream_index"
        )[
            "statistic"
        ]
    )

    for _, row in (
        detections.iterrows()
    ):

        index = int(
            row[
                "detected_index"
            ]
        )

        if (
            index
            not in
            statistic_lookup.index
        ):
            continue

        value = (
            statistic_lookup.loc[
                index
            ]
        )

        if not np.isfinite(
            value
        ):
            continue

        matched = (
            row[
                "match_status"
            ]
            in {
                "tier1",
                "tier2",
            }
        )

        colour = (
            MATCHED
            if matched
            else UNMATCHED
        )

        ax.scatter(
            row[
                "SETTLEMENTDATE"
            ],
            value,
            marker="v",
            s=62,
            color=colour,
            edgecolor="white",
            linewidth=0.65,
            zorder=7,
        )


def format_axes(
    ax_top,
    ax_bottom,
):
    """Apply consistent academic figure formatting."""

    for ax in (
        ax_top,
        ax_bottom,
    ):

        ax.set_facecolor(
            "white"
        )

        ax.grid(
            axis="y",
            color=GRID,
            linewidth=0.7,
            alpha=0.85,
        )

        ax.grid(
            axis="x",
            visible=False,
        )

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

        ax.spines[
            "left"
        ].set_color(
            "#AAB7BE"
        )

        ax.spines[
            "bottom"
        ].set_color(
            "#AAB7BE"
        )

        ax.tick_params(
            colors="#58666E",
            labelsize=8,
        )

    ax_bottom.xaxis.set_major_locator(
        mdates.MonthLocator(
            interval=6,
        )
    )

    ax_bottom.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%b\n%Y"
        )
    )


def add_legend(
    fig,
):
    """Add the common figure legend."""

    legend_handles = [

        Line2D(
            [0],
            [0],
            color=OBSERVED,
            linewidth=1.2,
            label="Observed demand",
        ),

        Line2D(
            [0],
            [0],
            color=SMOOTH,
            linewidth=1.8,
            label="15-day mean",
        ),

        Line2D(
            [0],
            [0],
            color=EVENT,
            linewidth=1.2,
            label=(
                "Documented Tier-1 event"
            ),
        ),

        Line2D(
            [0],
            [0],
            color=MATCHED,
            linewidth=1.3,
            linestyle="--",
            label=(
                "Detected — matched"
            ),
        ),

        Line2D(
            [0],
            [0],
            color=UNMATCHED,
            linewidth=1.3,
            linestyle="--",
            label=(
                "Detected — unmatched"
            ),
        ),

        Patch(
            facecolor=WINDOW,
            edgecolor="none",
            alpha=0.70,
            label=(
                "Documented event period"
            ),
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.025,
        ),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=2.8,
        columnspacing=1.8,
    )


def add_footer(
    fig,
    detector_name,
):
    """Add the relevant interpretation note."""

    if detector_name == "adwin":

        footer = (
            "ADWIN adaptive window width is shown as detector state; "
            "River does not expose its internal decision threshold. "
            "Markers indicate official project detections."
        )

    else:

        footer = (
            "Tier-1 periods are documented historical "
            "references, not synthetic ground-truth changepoints."
        )

    fig.text(
        0.5,
        0.092,
        footer,
        ha="center",
        va="center",
        fontsize=7.2,
        color="#6B747A",
    )


def plot_detector_region(
    region,
    detector_name,
    detector_title,
    diagnostics,
    matches,
    tier1_events,
):
    """Create one F2 detector-region figure."""

    demand = (
        load_test_demand(
            region
        )
    )

    trace = (
        diagnostics.loc[
            diagnostics[
                "region"
            ].eq(
                region
            )
            &
            diagnostics[
                "detector"
            ].eq(
                detector_name
            )
        ]
        .sort_values(
            "stream_index"
        )
        .copy()
    )

    detections = (
        matches.loc[
            matches[
                "region"
            ].eq(
                region
            )
            &
            matches[
                "detector"
            ].eq(
                detector_name
            )
        ]
        .sort_values(
            "detected_index"
        )
        .copy()
    )

    if trace.empty:

        raise RuntimeError(
            f"No diagnostics for "
            f"{region}/{detector_name}."
        )

    start = (
        trace[
            "SETTLEMENTDATE"
        ].min()
    )

    end = (
        trace[
            "SETTLEMENTDATE"
        ].max()
    )

    demand = (
        demand.loc[
            demand[
                "SETTLEMENTDATE"
            ].between(
                start,
                end,
            )
        ]
        .copy()
    )

    fig, (
        ax_top,
        ax_bottom,
    ) = plt.subplots(
        2,
        1,
        figsize=(
            14.0,
            8.2,
        ),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                2.35,
                1.0,
            ],
            "hspace": 0.07,
        },
    )

    fig.patch.set_facecolor(
        "white"
    )

    ax_top.plot(
        demand[
            "SETTLEMENTDATE"
        ],
        demand[
            "TOTALDEMAND"
        ],
        color=OBSERVED,
        linewidth=0.42,
        alpha=0.70,
        rasterized=True,
        zorder=1,
    )

    ax_top.plot(
        demand[
            "SETTLEMENTDATE"
        ],
        demand[
            "mean_15d"
        ],
        color=SMOOTH,
        linewidth=1.7,
        zorder=3,
    )

    add_event_references(
        ax_top,
        ax_bottom,
        tier1_events,
        region,
    )

    add_detection_lines(
        ax_top,
        detections,
    )

    raw_min = float(
        demand[
            "TOTALDEMAND"
        ].quantile(
            0.005
        )
    )

    raw_max = float(
        demand[
            "TOTALDEMAND"
        ].quantile(
            0.995
        )
    )

    raw_span = (
        raw_max
        - raw_min
    )

    ax_top.set_ylim(
        raw_min
        - 0.05
        * raw_span,
        raw_max
        + 0.48
        * raw_span,
    )

    add_detection_labels(
        ax_top,
        detections,
    )

    ax_top.set_ylabel(
        "Demand (MW)",
        fontsize=10,
        color=TEXT,
    )

    ax_top.set_title(
        (
            f"F2 — {detector_title} "
            f"drift detection on "
            f"{region} demand"
        ),
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        pad=24,
    )

    statistic = pd.to_numeric(
        trace[
            "statistic"
        ],
        errors="coerce",
    )

    ax_bottom.plot(
        trace[
            "SETTLEMENTDATE"
        ],
        statistic,
        color=STATISTIC,
        linewidth=0.70,
        alpha=0.90,
        rasterized=True,
        zorder=2,
    )

    draw_threshold(
        ax_bottom,
        detector_name,
        trace,
    )

    add_detection_markers(
        ax_bottom,
        trace,
        detections,
    )

    ax_bottom.set_ylabel(
        statistic_axis_label(
            detector_name
        ),
        fontsize=10,
        color=TEXT,
    )

    ax_bottom.set_xlabel(
        "Settlement date",
        fontsize=10,
        color=TEXT,
    )

    format_axes(
        ax_top,
        ax_bottom,
    )

    add_legend(
        fig,
    )

    add_footer(
        fig,
        detector_name,
    )

    fig.subplots_adjust(
        top=0.88,
        bottom=0.18,
        left=0.085,
        right=0.985,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"F2_{region}_"
        f"{detector_name}_"
        ".png"
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    return output_path


def main():
    """Generate all six F2 figures."""

    print(
        "Loading frozen F2 outputs..."
    )

    (
        diagnostics,
        matches,
        tier1_events,
    ) = load_inputs()

    validate_inputs(
        diagnostics,
        matches,
    )

    for region in REGIONS:

        for (
            detector_name,
            detector_title,
        ) in DETECTORS:

            output_path = (
                plot_detector_region(
                    region,
                    detector_name,
                    detector_title,
                    diagnostics,
                    matches,
                    tier1_events,
                )
            )

            print(
                f"Saved: "
                f"{output_path}"
            )

    print(
        "\nGenerated 6 F2 figures."
    )


if __name__ == "__main__":
    main()