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
    assign_full_stream_forecast_regimes,
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

    parser.add_argument(
        "--write-rolling-mae",
        action="store_true",
        help=(
            "Write timestamp-level 336-observation rolling MAE. "
            "This file is large and is disabled by default."
        ),
    )

    return parser.parse_args()


def load_seed_forecasts(run_dir, seed):
    """Load SA1 and NSW1 timestamp-level forecasts for one seed."""
    seed_dir = run_dir / f"seed_{seed}"

    if not seed_dir.is_dir():
        raise FileNotFoundError(
            f"Missing seed output directory: {seed_dir}"
        )

    required_columns = [
        "SETTLEMENTDATE",
        "y_true",
        "y_pred",
        "arm",
        "configuration",
        "detector",
        "window_days",
        "region",
        "seed",
    ]

    frames = []

    for region in REGIONS:
        path = seed_dir / f"forecasts_{region}.csv.gz"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing forecast stream: {path}"
            )

        frame = pd.read_csv(
            path,
            usecols=lambda column: column in required_columns,
        )

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
                f"{path} is missing required columns: "
                f"{sorted(missing)}"
            )

        if "detector" not in frame.columns:
            frame["detector"] = pd.NA

        if "window_days" not in frame.columns:
            frame["window_days"] = np.nan

        if set(frame["region"].dropna().unique()) != {region}:
            raise ValueError(
                f"{path} does not contain only region {region}."
            )

        seed_values = set(
            pd.to_numeric(
                frame["seed"],
                errors="raise",
            ).dropna().astype(int).unique()
        )

        if seed_values != {seed}:
            raise ValueError(
                f"{path} does not contain only seed {seed}. "
                f"Found {sorted(seed_values)}."
            )

        frames.append(frame)

    forecasts = pd.concat(
        frames,
        ignore_index=True,
    )

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

    forecasts["seed"] = pd.to_numeric(
        forecasts["seed"],
        errors="raise",
    ).astype(int)

    if forecasts[["y_true", "y_pred"]].isna().any().any():
        raise ValueError(
            "Forecast values contain missing values."
        )

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


def add_regimes(forecasts, events, regime_days, scope="full"):
    """Attach either full-stream or event-local regime labels."""
    if scope not in {"full", "local"}:
        raise ValueError("scope must be 'full' or 'local'.")

    outputs = []

    for region, region_frame in forecasts.groupby("region", sort=False):
        windows = build_forecast_event_windows(
            events,
            region=region,
            regime_days=regime_days,
        )

        timestamps = (
            region_frame["SETTLEMENTDATE"]
            .drop_duplicates()
            .sort_values()
        )

        if scope == "full":
            assignments = assign_full_stream_forecast_regimes(
                timestamps,
                windows,
            )
        else:
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
            raise RuntimeError(f"Missing regime assignment for {region}.")

        if scope == "full" and output["regime"].eq("unassigned").any():
            raise RuntimeError("Full-stream evaluation must assign every row.")

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
        return "N/A"

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

    if args.regime_days != 7:
        raise ValueError("The agreed T3 protocol uses seven-day buffers.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be strictly between 0 and 1.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique.")

    output_dir = args.output_dir or (args.run_dir / "t3_full_and_local")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {output_dir}. "
            "Use a new --output-dir to preserve earlier results."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events)
    radii = load_seed_radii(args.run_dir, args.seeds)

    metrics_by_scope = {"full": [], "local": []}
    audits = {}
    counts = []

    for seed in args.seeds:
        print(f"Processing seed {seed}...", flush=True)

        forecasts = load_seed_forecasts(args.run_dir, seed)
        forecasts = make_intervals(
            forecasts,
            radii=radii,
            alpha=args.alpha,
        )

        for scope in ("full", "local"):
            labelled = add_regimes(
                forecasts,
                events,
                regime_days=args.regime_days,
                scope=scope,
            )

            # Preserve all local assignments in the audit, including
            # unassigned rows. Metrics use only included local rows.
            audit = labelled[
                ["SETTLEMENTDATE", "region", "regime", "event_id"]
            ].drop_duplicates().sort_values(
                ["region", "SETTLEMENTDATE"]
            ).reset_index(drop=True)

            if scope not in audits:
                audits[scope] = audit
                audit.to_csv(
                    output_dir / f"regime_assignment_{scope}.csv",
                    index=False,
                )
            else:
                pd.testing.assert_frame_equal(
                    audits[scope],
                    audit,
                    check_dtype=False,
                )

            selected = (
                labelled
                if scope == "full"
                else labelled[
                    labelled["regime"].isin(REGIME_ORDER)
                ]
            )

            group_keys = ["region", "configuration", "seed"]
            total_counts = labelled.groupby(
                group_keys
            ).size()

            for key, group in selected.groupby(group_keys):
                row = dict(zip(group_keys, key))
                row["scope"] = scope
                row["n_total"] = int(total_counts.loc[key])
                row["n_included"] = len(group)

                for regime in REGIME_ORDER:
                    row[f"n_{regime}"] = int(
                        group["regime"].eq(regime).sum()
                    )

                if scope == "full":
                    if row["n_included"] != row["n_total"]:
                        raise RuntimeError("Full-stream rows were excluded.")
                    if sum(
                        row[f"n_{regime}"] for regime in REGIME_ORDER
                    ) != row["n_total"]:
                        raise RuntimeError("Full-stream regimes are incomplete.")

                counts.append(row)

            metrics_by_scope[scope].append(
                calculate_seed_metrics(selected, alpha=args.alpha)
            )
            del selected, labelled

        del forecasts

    for scope in ("full", "local"):
        seed_metrics = pd.concat(
            metrics_by_scope[scope],
            ignore_index=True,
        )
        table = aggregate_t3(
            seed_metrics,
            expected_seeds=args.seeds,
        )

        # Match the required seven-column T3 schema.
        table["arm"] = table["arm_label"]
        table = table[
            [
                "arm",
                "region",
                "MAE pre-drift",
                "MAE drift",
                "MAE post-drift",
                "pinball loss",
                "seeds",
            ]
        ]

        table.to_csv(
            output_dir / f"t3_{scope}.csv",
            index=False,
        )
        seed_metrics.to_csv(
            output_dir / f"t3_seed_metrics_{scope}.csv",
            index=False,
        )

        headers = list(table.columns)
        lines = [
            f"# T3 — {scope} forecast accuracy by regime",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for values in table.itertuples(index=False, name=None):
            lines.append(
                "| " + " | ".join(map(str, values)) + " |"
            )

        (output_dir / f"t3_{scope}.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        print(f"\nT3 — {scope}")
        print(table.to_string(index=False))

    pd.DataFrame(counts).to_csv(
        output_dir / "regime_observation_counts.csv",
        index=False,
    )

    import hashlib

    manifest = {
        "seeds": args.seeds,
        "regime_days": args.regime_days,
        "timezone": "Etc/GMT-10",
        "boundary": "left_closed_right_open",
        "full_rule": (
            "pre before first buffered window; drift inside union "
            "of buffered windows; post elsewhere after first window"
        ),
        "event_selection": (
            "NEM plus region-specific events; full-stream buffered "
            "windows must overlap evaluated stream"
        ),
        "local_rule": (
            "7-day pre; documented event drift; 7-day post; "
            "outside excluded; priority drift > pre > post"
        ),
        "pinball_scope": {
            "full": "all test timestamps",
            "local": "union of included local timestamps",
        },
        "interval_method": (
            "seed-specific fixed regional conformal radius "
            "transferred to each adaptation point forecast"
        ),
        "alpha": args.alpha,
        "standard_deviation": "sample, ddof=1",
        "primary_configurations": PRIMARY_CONFIGURATIONS,
        "selection_note": (
            "Primary configurations retained from previous analysis; "
            "not newly selected using these regime results."
        ),
        "events_sha256": hashlib.sha256(
            args.events.read_bytes()
        ).hexdigest(),
        "generator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "rolling_mae": (
            "Not regenerated; rolling errors do not change when "
            "only regime labels change."
        ),
    }

    (output_dir / "t3_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nBoth T3 tables saved to: {output_dir}")


if __name__ == "__main__":
    main()