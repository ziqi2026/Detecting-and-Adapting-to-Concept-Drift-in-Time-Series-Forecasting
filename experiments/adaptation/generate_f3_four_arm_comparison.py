"""Generate F3 four-arm rolling-MAE and retraining comparison figures.

The four displayed configurations share the same forecasting model and initial
state.  Arms C and D must use the same explicitly selected detector; Arm D's
recent-history window is also explicit to avoid post-hoc cherry-picking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drift_lab.evaluation import calculate_rolling_mae


REGIONS = ("SA1", "NSW1")
ARMS = ("A", "B", "C", "D")
COLORS = {
    "A": "#8C8C8C",
    "B": "#B8B8B8",
    "C": "#0072B2",
    "D": "#D55E00",
}
LINESTYLES = {"A": "--", "B": "-.", "C": "-", "D": "-"}


def selected_configurations(detector: str, window_days: int) -> dict[str, str]:
    return {
        "A": "A",
        "B": "B",
        "C": f"C_{detector}",
        "D": f"D_{window_days}d_{detector}",
    }


def load_inputs(input_dir: Path, region: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecasts = pd.read_csv(
        input_dir / f"forecasts_{region}.csv.gz",
        parse_dates=["SETTLEMENTDATE"],
        low_memory=False,
    )
    events = pd.read_csv(
        input_dir / "retraining_events.csv",
        parse_dates=["retrain_timestamp"],
        low_memory=False,
    )
    summary = pd.read_csv(input_dir / "summary.csv", low_memory=False)
    return forecasts, events, summary


def validate_selection(
    forecasts: pd.DataFrame,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    region: str,
    configurations: dict[str, str],
) -> None:
    available = set(forecasts.loc[forecasts["region"].eq(region), "configuration"])
    missing = set(configurations.values()) - available
    if missing:
        raise ValueError(f"Missing forecast configurations for {region}: {sorted(missing)}")

    chosen = summary[
        summary["region"].eq(region)
        & summary["configuration"].isin(configurations.values())
    ]
    if len(chosen) != 4 or set(chosen["arm"]) != set(ARMS):
        raise ValueError(f"Expected exactly one summary row per arm for {region}")
    if chosen[["n_observations", "overall_mae", "n_retrains"]].isna().any().any():
        raise ValueError(f"Incomplete summary values for {region}")

    counts = (
        events[
            events["region"].eq(region)
            & events["configuration"].isin(configurations.values())
        ]
        .groupby("configuration")
        .size()
        .to_dict()
    )
    for row in chosen.itertuples(index=False):
        observed = int(counts.get(row.configuration, 0))
        if observed != int(row.n_retrains):
            raise ValueError(
                f"Retrain-count mismatch for {region}/{row.configuration}: "
                f"events={observed}, summary={row.n_retrains}"
            )


def prepare_rolling_mae(
    forecasts: pd.DataFrame,
    *,
    region: str,
    configurations: dict[str, str],
    window: int,
) -> dict[str, pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    reference_timestamps = None
    reference_truth = None
    for arm, configuration in configurations.items():
        frame = (
            forecasts[
                forecasts["region"].eq(region)
                & forecasts["configuration"].eq(configuration)
            ][["SETTLEMENTDATE", "y_true", "y_pred"]]
            .sort_values("SETTLEMENTDATE")
            .reset_index(drop=True)
        )
        if frame.empty:
            raise ValueError(f"No forecasts for {region}/{configuration}")
        if frame[["y_true", "y_pred"]].isna().any().any():
            raise ValueError(f"Missing y_true/y_pred for {region}/{configuration}")
        if reference_timestamps is None:
            reference_timestamps = frame["SETTLEMENTDATE"].to_numpy()
            reference_truth = frame["y_true"].to_numpy()
        else:
            if not np.array_equal(reference_timestamps, frame["SETTLEMENTDATE"].to_numpy()):
                raise ValueError(f"Timestamp mismatch for {region}/{configuration}")
            np.testing.assert_allclose(reference_truth, frame["y_true"].to_numpy())
        frame["rolling_mae"] = calculate_rolling_mae(
            frame["y_true"], frame["y_pred"], window=window
        )
        prepared[arm] = frame
    return prepared


def plot_region(
    *,
    region: str,
    forecasts: pd.DataFrame,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    detector: str,
    window_days: int,
    rolling_window: int,
    output_dir: Path,
) -> Path:
    configurations = selected_configurations(detector, window_days)
    validate_selection(
        forecasts, events, summary,
        region=region, configurations=configurations,
    )
    series = prepare_rolling_mae(
        forecasts,
        region=region,
        configurations=configurations,
        window=rolling_window,
    )
    chosen_summary = (
        summary[
            summary["region"].eq(region)
            & summary["configuration"].isin(configurations.values())
        ]
        .set_index("configuration")
    )
    chosen_events = events[
        events["region"].eq(region)
        & events["configuration"].isin(configurations.values())
    ].copy()

    drift_events = chosen_events[
        chosen_events["configuration"].eq(configurations["C"])
    ].sort_values("retrain_timestamp")
    first_drift = (
        drift_events["retrain_timestamp"].iloc[0]
        if not drift_events.empty else series["A"]["SETTLEMENTDATE"].iloc[-1]
    )
    arm_a = series["A"]
    pre_drift = arm_a[arm_a["SETTLEMENTDATE"] < first_drift]
    pre_drift_mae = (pre_drift["y_true"] - pre_drift["y_pred"]).abs().mean()

    fig = plt.figure(figsize=(15, 8.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[5.4, 1.35], hspace=0.04)
    ax = fig.add_subplot(grid[0])
    rug = fig.add_subplot(grid[1], sharex=ax)

    descriptions = {
        "A": "Never retrain",
        "B": "Every 30 days",
        "C": f"Drift-full ({detector.upper()})",
        "D": f"Drift-recent {window_days}d ({detector.upper()})",
    }
    for arm in ARMS:
        configuration = configurations[arm]
        count = int(chosen_summary.loc[configuration, "n_retrains"])
        frame = series[arm]
        ax.plot(
            frame["SETTLEMENTDATE"], frame["rolling_mae"],
            color=COLORS[arm], linestyle=LINESTYLES[arm],
            linewidth=1.35 if arm in ("A", "B") else 1.65,
            alpha=0.84 if arm in ("A", "B") else 0.95,
            label=f"Arm {arm}: {descriptions[arm]} (retrain n={count})",
        )

    ax.axhline(
        pre_drift_mae, color="#333333", linewidth=1.1, linestyle=":",
        label=f"Arm A pre-drift MAE reference ({pre_drift_mae:.1f} MW)",
    )
    ax.axvline(first_drift, color="#333333", linewidth=0.9, alpha=0.45)
    ax.annotate(
        "First accepted drift trigger",
        xy=(first_drift, ax.get_ylim()[1]), xytext=(5, -8),
        textcoords="offset points", ha="left", va="top", fontsize=8.5,
        color="#333333",
    )
    ax.set_ylabel(f"{rolling_window}-observation rolling MAE (MW)")
    ax.set_title(
        f"F3 — {region} Four-Arm Forecasting and Retraining Comparison\n"
        f"Gradient Boosting forecaster | {rolling_window} half-hour observations = "
        f"{rolling_window / 48:.0f} days"
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=8.5)
    ax.tick_params(axis="x", labelbottom=False)

    y_positions = {"A": 3, "B": 2, "C": 1, "D": 0}
    for arm in ARMS:
        configuration = configurations[arm]
        arm_events = chosen_events[
            chosen_events["configuration"].eq(configuration)
        ]["retrain_timestamp"]
        if not arm_events.empty:
            rug.vlines(
                arm_events, y_positions[arm] - 0.30, y_positions[arm] + 0.30,
                color=COLORS[arm], linewidth=1.05, alpha=0.9,
            )
    rug.set_yticks([3, 2, 1, 0], ["Arm A", "Arm B", "Arm C", "Arm D"])
    rug.set_ylim(-0.65, 3.65)
    rug.set_ylabel("Retrain\nevents", rotation=0, labelpad=38, va="center")
    rug.set_xlabel("Test-stream date")
    rug.grid(axis="x", alpha=0.18)
    rug.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    rug.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in rug.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    for spine in ("top", "right"):
        rug.spines[spine].set_visible(False)

    fig.text(
        0.01, 0.008,
        "Controls A/B are muted grey. Arms C/D use the same frozen detector alarms. "
        "The dotted reference is Arm A MAE before the first accepted drift trigger.",
        fontsize=8.5, color="#444444",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"F3_{region}_four_arm_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/figures/adaptation")
    parser.add_argument("--detector", choices=("adwin", "page_hinkley", "kswin"),
                        required=True)
    parser.add_argument("--window-days", type=int, choices=(30, 60, 120, 180),
                        required=True)
    parser.add_argument("--rolling-window", type=int, default=48 * 7)
    args = parser.parse_args()

    for region in REGIONS:
        forecasts, events, summary = load_inputs(args.input_dir, region)
        output = plot_region(
            region=region, forecasts=forecasts, events=events, summary=summary,
            detector=args.detector, window_days=args.window_days,
            rolling_window=args.rolling_window, output_dir=args.output_dir,
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
