"""
Generate T3 — Forecast Accuracy by Regime.

This script:
- loads timestamp-level PGS242-128 forecast streams;
- uses shared event-aware regime utilities from drift_lab.evaluation;
- computes MAE by pre-drift, drift, and post-drift regime;
- computes a 336-observation rolling MAE;
- computes 90% interval pinball loss with shared functions;
- aggregates seed-level results as mean ± sample standard deviation.

T3 is generated from experiment outputs, never manually entered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drift_lab.evaluation import (
    assign_forecast_regimes,
    build_forecast_event_windows,
    calculate_interval_pinball_loss,
    calculate_mae,
    calculate_rolling_mae,
)


REGIONS = ("SA1", "NSW1")
ROLLING_WINDOW = 48 * 7
REGIME_ORDER = ("pre-drift", "drift", "post-drift")

PRIMARY_CONFIGURATIONS = {
    "A": "A",
    "B": "B",
    "C": "C_kswin",
    "D": "D_180d_page_hinkley",
}

DISPLAY_ARM_LABELS = {
    "A": "A · never retrain",
    "B": "B · periodic 30d",
    "C": "C · drift + full history (KSWIN)",
    "D": "D · drift + 180d window (Page-Hinkley)",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Parent directory containing seed_<n> outputs.",
    )

    parser.add_argument(
        "--events",
        type=Path,
        default=ROOT / "data" / "events.csv",
        help="Frozen documented AEMO events catalogue.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/t3",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Expected forecast seeds.",
    )

    parser.add_argument(
        "--regime-days",
        type=int,
        default=7,
        help="Days before/after each documented event.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Prediction interval alpha; 0.10 means 90%% intervals.",
    )

    return parser.parse_args()


def load_seed_forecasts(run_dir, seed):
    """Load SA1 and NSW1 timestamp-level forecasts for one seed."""
    seed_dir = run_dir / f"seed_{seed}"

    if not seed_dir.is_dir():
        raise FileNotFoundError(
            f"Missing seed output directory: {seed_dir}"
        )

    frames = []

    for region in REGIONS:
        path = seed_dir / f"forecasts_{region}.csv.gz"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing forecast stream: {path}"
            )

        frame = pd.read_csv(path)

        required = {
            "SETTLEMENTDATE",
            "y_true",
            "y_pred",
            "arm",
            "configuration",
            "region",
            "seed",
        }

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        if set(frame["region"].dropna().unique()) != {region}:
            raise ValueError(
                f"{path} does not contain only region {region}."
            )

        if set(frame["seed"].dropna().unique()) != {seed}:
            raise ValueError(
                f"{path} does not contain only seed {seed}."
            )

        frames.append(frame)

    forecasts = pd.concat(frames, ignore_index=True)

    forecasts["SETTLEMENTDATE"] = pd.to_datetime(
    forecasts["SETTLEMENTDATE"],
    errors="raise",
)

if forecasts["SETTLEMENTDATE"].dt.tz is None:
    forecasts["SETTLEMENTDATE"] = (
        forecasts["SETTLEMENTDATE"]
        .dt.tz_localize("Etc/GMT-10")
    )
else:
    forecasts["SETTLEMENTDATE"] = (
        forecasts["SETTLEMENTDATE"]
        .dt.tz_convert("Etc/GMT-10")
    )

    forecasts["y_true"] = pd.to_numeric(
        forecasts["y_true"],
        errors="raise",
    )

    forecasts["y_pred"] = pd.to_numeric(
        forecasts["y_pred"],
        errors="raise",
    )

    if forecasts[["y_true", "y_pred"]].isna().any().any():
        raise ValueError("Forecast values contain missing values.")

    keys = [
        "region",
        "configuration",
        "seed",
        "SETTLEMENTDATE",
    ]

    if forecasts.duplicated(keys).any():
        raise ValueError(
            "Duplicate region/configuration/seed/timestamp keys found."
        )

    return forecasts.sort_values(
        [
            "region",
            "configuration",
            "seed",
            "SETTLEMENTDATE",
        ]
    ).reset_index(drop=True)


def add_regimes(forecasts, events, regime_days):
    """Assign event-aware regimes separately for SA1 and NSW1."""
    outputs = []

    for region, region_frame in forecasts.groupby(
        "region",
        sort=False,
    ):
        windows = build_forecast_event_windows(
            events,
            region=region,
            regime_days=regime_days,
        )

        timestamps = region_frame["SETTLEMENTDATE"].drop_duplicates()

        assignments = assign_forecast_regimes(
            timestamps,
            windows,
        )

        output = region_frame.merge(
            assignments,
            on="SETTLEMENTDATE",
            how="left",
            validate="many_to_one",
        )

        if output["regime"].isna().any():
            raise RuntimeError(
                f"Missing regime assignment for {region}."
            )

        outputs.append(output)

    return pd.concat(outputs, ignore_index=True)


def make_intervals(frame, radii, alpha):
    """
    Attach fixed-radius intervals.

    The radius must be loaded from the seed-specific PGS242-141
    calibration output in the final workflow.
    """
    output = frame.copy()

    output["conformal_radius"] = output.apply(
        lambda row: radii[(row["region"], int(row["seed"]))],
        axis=1,
    )

    output["lower_90"] = (
        output["y_pred"] - output["conformal_radius"]
    )

    output["upper_90"] = (
        output["y_pred"] + output["conformal_radius"]
    )

    output["alpha"] = alpha

    return output


def load_seed_radii(run_dir, seeds):
    """
    Load seed-specific PGS242-141 radii.

    Required file for each seed:
    <run-dir>/seed_<seed>/uncertainty/aemo_prediction_intervals_90_summary.csv
    """
    radii = {}

    for seed in seeds:
        path = (
            run_dir
            / f"seed_{seed}"
            / "uncertainty"
            / "aemo_prediction_intervals_90_summary.csv"
        )

        if not path.is_file():
            raise FileNotFoundError(
                "Missing uncertainty summary for seed "
                f"{seed}: {path}"
            )

        summary = pd.read_csv(path)

        required = {"region", "conformal_radius"}

        missing = required - set(summary.columns)

        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        for row in summary.itertuples(index=False):
            radii[(row.region, seed)] = float(
                row.conformal_radius
            )

    return radii


def calculate_seed_metrics(frame, alpha):
    """Calculate per-seed MAE and interval pinball loss."""
    rows = []

    group_columns = [
        "arm",
        "configuration",
        "detector",
        "window_days",
        "region",
        "seed",
    ]

    for key, group in frame.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        row = dict(zip(group_columns, key))

        for regime in REGIME_ORDER:
            subset = group[group["regime"] == regime]

            row[f"mae_{regime}"] = (
                calculate_mae(
                    subset["y_true"],
                    subset["y_pred"],
                )
                if not subset.empty
                else np.nan
            )

            row[f"n_{regime}"] = int(len(subset))

        row["pinball_loss"] = calculate_interval_pinball_loss(
            group["y_true"],
            group["lower_90"],
            group["upper_90"],
            alpha=alpha,
        )

        rows.append(row)

    return pd.DataFrame(rows)


def add_rolling_mae(frame):
    """Calculate 7-day rolling MAE using the shared function."""
    outputs = []

    group_columns = [
        "region",
        "configuration",
        "seed",
    ]

    for _, group in frame.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        group = group.sort_values(
            "SETTLEMENTDATE"
        ).copy()

        group["rolling_mae_336"] = calculate_rolling_mae(
            group["y_true"],
            group["y_pred"],
            window=ROLLING_WINDOW,
        ).to_numpy()

        outputs.append(group)

    return pd.concat(outputs, ignore_index=True)


def format_mean_sd(series):
    """Format a metric as mean ± sample SD."""
    series = pd.Series(series, dtype=float).dropna()

    if len(series) == 0:
        return ""

    if len(series) == 1:
        return f"{series.iloc[0]:.1f}"

    return (
        f"{series.mean():.1f} ± "
        f"{series.std(ddof=1):.1f}"
    )


def aggregate_t3(seed_metrics, expected_seeds):
    """Aggregate seed-level metrics into display-ready T3 rows."""
    selected = seed_metrics[
        seed_metrics["configuration"].isin(
            PRIMARY_CONFIGURATIONS.values()
        )
    ].copy()

    selected["arm_label"] = selected["arm"].map(
        DISPLAY_ARM_LABELS
    )

    group_columns = [
        "arm",
        "arm_label",
        "region",
        "configuration",
    ]

    rows = []

    for key, group in selected.groupby(
        group_columns,
        sort=False,
    ):
        row = dict(zip(group_columns, key))

        for regime in REGIME_ORDER:
            row[f"MAE {regime}"] = format_mean_sd(
                group[f"mae_{regime}"]
            )

        row["pinball loss"] = format_mean_sd(
            group["pinball_loss"]
        )

        actual_seeds = sorted(
            group["seed"].astype(int).unique().tolist()
        )

        if actual_seeds != sorted(expected_seeds):
            raise ValueError(
                "T3 aggregation does not contain exactly the "
                f"requested seeds. Found {actual_seeds}; "
                f"expected {sorted(expected_seeds)}."
            )

        row["seeds"] = len(actual_seeds)

        rows.append(row)

    result = pd.DataFrame(rows)

    arm_order = {"A": 0, "B": 1, "C": 2, "D": 3}

    return result.assign(
        _arm_order=result["arm"].map(arm_order)
    ).sort_values(
        ["region", "_arm_order"]
    ).drop(
        columns=["_arm_order", "configuration"]
    )


def main():
    args = parse_args()

    if args.regime_days <= 0:
        raise ValueError("--regime-days must be positive.")

    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be strictly between 0 and 1.")

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique.")

    output_dir = args.output_dir or (args.run_dir / "t3")
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events)
    radii = load_seed_radii(args.run_dir, args.seeds)

    all_forecasts = []

    for seed in args.seeds:
        forecasts = load_seed_forecasts(args.run_dir, seed)
        forecasts = add_regimes(
            forecasts,
            events,
            regime_days=args.regime_days,
        )
        all_forecasts.append(forecasts)

    all_forecasts = pd.concat(
        all_forecasts,
        ignore_index=True,
    )

    all_forecasts = make_intervals(
        all_forecasts,
        radii=radii,
        alpha=args.alpha,
    )

    seed_metrics = calculate_seed_metrics(
        all_forecasts,
        alpha=args.alpha,
    )

    rolling = add_rolling_mae(all_forecasts)

    t3 = aggregate_t3(
        seed_metrics,
        expected_seeds=args.seeds,
    )

    seed_metrics.to_csv(
        output_dir / "t3_seed_level_metrics.csv",
        index=False,
    )

    t3.to_csv(
        output_dir / "t3_forecast_accuracy_by_regime.csv",
        index=False,
    )

    rolling[
        [
            "SETTLEMENTDATE",
            "region",
            "arm",
            "configuration",
            "detector",
            "window_days",
            "seed",
            "regime",
            "event_id",
            "rolling_mae_336",
        ]
    ].to_csv(
        output_dir / "rolling_mae_336.csv",
        index=False,
    )

    regime_audit = all_forecasts[
        [
            "SETTLEMENTDATE",
            "region",
            "regime",
            "event_id",
        ]
    ].drop_duplicates().sort_values(
        [
            "region",
            "SETTLEMENTDATE",
        ]
    )

    regime_audit.to_csv(
        output_dir / "forecast_regime_assignment_audit.csv",
        index=False,
    )

    manifest = {
        "events_path": str(args.events.resolve()),
        "seeds": args.seeds,
        "regime_days": args.regime_days,
        "alpha": args.alpha,
        "rolling_window": ROLLING_WINDOW,
        "pinball_method": (
            "mean of lower q=alpha/2 and upper "
            "q=1-alpha/2 pinball losses"
        ),
        "interval_method": (
            "seed-specific fixed conformal radius around "
            "each adaptation point forecast"
        ),
        "primary_configurations": PRIMARY_CONFIGURATIONS,
    }

    (output_dir / "t3_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(t3.to_string(index=False))


if __name__ == "__main__":
    main()
