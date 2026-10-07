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
SEEDS = (1, 2, 3, 4, 5)
DETECTORS = ("adwin", "page_hinkley", "kswin")
ARMS = ("A", "B", "C", "D")
COLORS = {
    "A": "#666666",   
    "B": "#B0B0B0",   
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


def load_inputs(input_dir: Path, region: str,  seed: int,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_dir = input_dir / f"seed_{seed}"

    forecasts = pd.read_csv(
        seed_dir  / f"forecasts_{region}.csv.gz",
        parse_dates=["SETTLEMENTDATE"],
        low_memory=False,
    )
    events = pd.read_csv(
        seed_dir  / "retraining_events.csv",
        parse_dates=["retrain_timestamp"],
        low_memory=False,
    )
    summary = pd.read_csv(seed_dir  / "summary.csv", low_memory=False)
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
    seed_inputs: dict[int, dict[str, pd.DataFrame]],
    detector: str,
    window_days: int,
    rolling_window: int,
    output_dir: Path,
) -> Path:
    configurations = selected_configurations(detector, window_days)
    seed_series = {}

    for seed in SEEDS:
        forecasts = seed_inputs[seed]["forecasts"]
        events = seed_inputs[seed]["events"]
        summary = seed_inputs[seed]["summary"]

        validate_selection(
            forecasts,
            events,
            summary,
            region=region,
            configurations=configurations,
        )

        seed_series[seed] = prepare_rolling_mae(
            forecasts,
            region=region,
            configurations=configurations,
            window=rolling_window,
        )
    
    mean_series = {}

    for arm in ARMS:
        seed_frames = []

        for seed in SEEDS:
            frame = seed_series[seed][arm][
                ["SETTLEMENTDATE", "rolling_mae"]
            ].copy()

            frame = frame.rename(
                columns={"rolling_mae": f"rolling_mae_seed_{seed}"}
            )

            seed_frames.append(frame)

        combined = seed_frames[0]

        for frame in seed_frames[1:]:
            combined = combined.merge(
                frame,
                on="SETTLEMENTDATE",
                how="inner",
                validate="one_to_one",
            )

        mae_columns = [
            f"rolling_mae_seed_{seed}"
            for seed in SEEDS
        ]

        combined["rolling_mae_mean"] = combined[mae_columns].mean(axis=1)
        combined["rolling_mae_sd"] = combined[mae_columns].std(
            axis=1,
            ddof=1,
        )

        mean_series[arm] = combined
    
    summary_frames = []
    for seed in SEEDS:
        seed_summary = seed_inputs[seed]["summary"].copy()
        seed_summary["seed"] = seed
        summary_frames.append(seed_summary)

    all_summaries = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    chosen_summary = all_summaries[
        all_summaries["region"].eq(region)
        & all_summaries["configuration"].isin(configurations.values())
    ].copy()

    event_frames = []

    for seed in SEEDS:
        seed_events = seed_inputs[seed]["events"].copy()
        seed_events["seed"] = seed
        event_frames.append(seed_events)

    all_events = pd.concat(
        event_frames,
        ignore_index=True,
    )

    chosen_events = all_events[
        all_events["region"].eq(region)
        & all_events["configuration"].isin(configurations.values())
    ].copy()

    drift_events = chosen_events[
        chosen_events["configuration"].eq(configurations["C"])
    ].sort_values("retrain_timestamp")
    first_drift = (
        drift_events["retrain_timestamp"].iloc[0]
        if not drift_events.empty else mean_series["A"]["SETTLEMENTDATE"].iloc[-1]
    )
    pre_drift_maes = []

    for seed in SEEDS:
        arm_a = seed_series[seed]["A"]

        pre_drift = arm_a[
            arm_a["SETTLEMENTDATE"] < first_drift
        ]

        pre_drift_mae = (
            pre_drift["y_true"] - pre_drift["y_pred"]
        ).abs().mean()

        pre_drift_maes.append(pre_drift_mae)

    pre_drift_mae = float(np.mean(pre_drift_maes))
    pre_drift_mae_sd = float(np.std(pre_drift_maes, ddof=1))


    fig = plt.figure(figsize=(19, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=[5.4, 1.35], hspace=0.04)
    fig.subplots_adjust(right=0.72, bottom=0.14, top=0.88,)
    ax = fig.add_subplot(grid[0])
    rug = fig.add_subplot(grid[1])
    rug.set_xlim(
    mean_series["A"]["SETTLEMENTDATE"].min(),
    mean_series["A"]["SETTLEMENTDATE"].max(),
    )

    descriptions = {
        "A": "Never retrain",
        "B": "Every 30 days",
        "C": f"Drift-full ({detector.upper()})",
        "D": f"Drift-recent {window_days}d ({detector.upper()})",
    }
    for arm in ARMS:
        configuration = configurations[arm]
        counts = chosen_summary.loc[
            chosen_summary["configuration"].eq(configuration),
            "n_retrains",
        ]
        count_mean = counts.mean()
        count_sd = counts.std(ddof=1)
        frame = mean_series[arm]
        ax.plot(
            frame["SETTLEMENTDATE"],
            frame["rolling_mae_mean"],
            color=COLORS[arm],
            linestyle=LINESTYLES[arm],
            linewidth=1.8 if arm in ("A", "B") else 2.0,
            alpha=0.95,
            label=(
                f"Arm {arm}: {descriptions[arm]} "
                f"(retrain n={count_mean:.1f} ± {count_sd:.1f})"
            ),
        )

        ax.fill_between(
            frame["SETTLEMENTDATE"],
            frame["rolling_mae_mean"] - frame["rolling_mae_sd"],
            frame["rolling_mae_mean"] + frame["rolling_mae_sd"],
            color=COLORS[arm],
            alpha=0.12,
            linewidth=0,
        )

    ax.axhline(
        pre_drift_mae, color="#333333", linewidth=1.1, linestyle=":",
        label=(f"Arm A pre-drift MAE reference "
               f"({pre_drift_mae:.1f} ± {pre_drift_mae_sd:.1f} MW)"),
    )
    ax.axvline(first_drift, color="#333333", linewidth=0.9, alpha=0.45)
    ax.annotate(
        "First accepted drift trigger",
        xy=(first_drift, ax.get_ylim()[1]), xytext=(5, -10),
        textcoords="offset points", ha="left", va="top", fontsize=11,
        color="#333333",fontweight="bold",
    )
    ax.set_ylabel(f"{rolling_window}-observation rolling MAE (MW)", fontsize=11, labelpad=10,)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title(
        f"F3 — {region} Four-Arm Forecasting and Retraining Comparison\n"
        f"Gradient Boosting forecaster | {rolling_window} half-hour observations = "
        f"{rolling_window / 48:.0f} days", fontsize=15, fontweight="bold", pad=14,
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=10, bbox_to_anchor=(1.01, 1.0), borderaxespad=0, borderpad=0.8,labelspacing=0.6,)
    ax.tick_params(axis="x", labelbottom=False)

    # Validate that retraining schedules are identical across seeds
    for arm in ARMS:
        configuration = configurations[arm]
        reference_events = None

        for seed in SEEDS:
            seed_events = (
                chosen_events[
                    chosen_events["seed"].eq(seed)
                    & chosen_events["configuration"].eq(configuration)
                ]["retrain_timestamp"]
                .sort_values()
                .reset_index(drop=True)
            )

            if reference_events is None:
                reference_events = seed_events
            elif not reference_events.equals(seed_events):
                raise ValueError(
                    f"Retraining timestamps differ across seeds for "
                    f"{region}/{configuration}"
                )

    y_positions = {"A": 3, "B": 2, "C": 1, "D": 0}
    for arm in ARMS:
        configuration = configurations[arm]
        arm_events = ( 
            chosen_events[
                chosen_events["configuration"].eq(configuration)
            ]["retrain_timestamp"]
            .drop_duplicates()
            .sort_values()
        )
        if not arm_events.empty:
            rug.vlines(
                arm_events,
                y_positions[arm] - 0.22,
                y_positions[arm] + 0.22,
                color=COLORS[arm],
                linewidth=2.2,
                alpha=0.95,
                zorder=3,
            )
    rug.set_yticks([3, 2, 1, 0], ["Arm A", "Arm B", "Arm C", "Arm D"])
    rug.tick_params(axis="y", labelsize=10)
    rug.tick_params(axis="x", labelsize=10)
    rug.set_ylim(-0.65, 3.65)
    rug.set_ylabel("Retrain\nevents", rotation=0, labelpad=45, va="center", fontsize=11,)
    rug.set_xlabel("Test-stream date", fontsize=11, labelpad=12,)
    rug.grid(axis="x", alpha=0.18)
    rug.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    rug.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in rug.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    for spine in ("top", "right"):
        rug.spines[spine].set_visible(False)

    fig.text(
        0.73,
        0.67,
        "Figure notes: \n\n"
        "• Lines show mean rolling MAE across seeds 1–5\n"
        "• Controls A/B are muted grey\n"
        "• Arms C/D use the same frozen detector alarms\n"
        "• Dotted line = Arm A mean MAE before first trigger",
        fontsize=11,
        color="#333333",
        ha="left",
        va="top",
        linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=0.9",
            facecolor="#F4F6F7",
            edgecolor="#D5D8DC",
            linewidth=1.0,
        ),
    )



    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"F3_{region}_four_arm_comparison_{detector}_{window_days}d.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/figures/adaptation")
    #parser.add_argument("--detector", choices=("adwin", "page_hinkley", "kswin"),required=True)
    parser.add_argument("--window-days", type=int, choices=(30, 60, 120, 180),
                        required=True)
    parser.add_argument("--rolling-window", type=int, default=48 * 7)

    args = parser.parse_args()

    for region in REGIONS:
        seed_inputs = {}
        
        for seed in SEEDS:
            forecasts, events, summary = load_inputs(
                args.input_dir,
                region,
                seed,
            )

            seed_inputs[seed] = {
                "forecasts": forecasts,
                "events": events,
                "summary": summary,
            }  
        for detector in DETECTORS:
            output = plot_region(
                region=region, seed_inputs=seed_inputs, 
                detector=detector, window_days=args.window_days,
                rolling_window=args.rolling_window, output_dir=args.output_dir,
            )
            print(f"Wrote {output}")


if __name__ == "__main__":
    main()
