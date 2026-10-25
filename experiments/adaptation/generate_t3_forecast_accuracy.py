"""
Generate T3 — Forecast Accuracy by Regime.

This script:
- loads timestamp-level PGS242-128 forecast streams;
- uses shared event-aware regime utilities from drift_lab.evaluation;
- labels the full-stream regimes with a selectable buffer convention
  (--buffer-mode post_only | symmetric, see below);
- computes MAE by pre-drift, drift, and post-drift regime;
- computes a 336-observation rolling MAE;
- computes 90% interval pinball loss with shared functions;
- aggregates seed-level results as mean ± sample standard deviation.

Buffer modes (full-stream scope; buffer length --regime-days = 7):

    post_only (default, supervisor 20 Oct)
        drift = [event_start, event_end + 1 day + 7 days)
    symmetric (earlier +/- 7-day convention, kept reproducible)
        drift = [event_start - 7 days, event_end + 1 day + 7 days)

    pre-drift = everything before the first drift window; post-drift =
    everything after it that is outside the union of drift windows. The
    event-local scope has its own fixed 7-day pre/post windows and is not
    affected by --buffer-mode.

Forecasters (PGS242-169): --forecaster selects which saved run to score.
The default, gradient_boosted, reads the original five-seed run and writes to
t3/<buffer-mode>/ with the original schema (unchanged). Any other forecaster
reads its own run directory, writes to t3/<forecaster>/<buffer-mode>/, and
adds a leading `forecaster` column to its tables.

T3 is generated from experiment outputs, never manually entered.

Run from the repository root, once per mode:
    python experiments/adaptation/generate_t3_forecast_accuracy.py \
        --buffer-mode post_only --scopes full
    python experiments/adaptation/generate_t3_forecast_accuracy.py \
        --buffer-mode symmetric --scopes full
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
    FULL_STREAM_BUFFER_MODES,
    assign_full_stream_forecast_regimes,
    assign_forecast_regimes,
    build_forecast_event_windows,
    calculate_interval_pinball_loss,
    calculate_mae,
    calculate_rolling_mae,
)


DEFAULT_RUN_DIR = (
    ROOT / "results/tables/adaptation/revised/pgs242_128_five_seed_final"
)
DEFAULT_OUTPUT_ROOT = ROOT / "results/tables/adaptation/t3"
DEFAULT_BUFFER_MODE = "post_only"
DEFAULT_FORECASTER = "gradient_boosted"
FORECASTER_RUN_DIRS = {
    DEFAULT_FORECASTER: DEFAULT_RUN_DIR,
    "seasonal_naive": (
        ROOT / "results/tables/adaptation/revised/seasonal_naive_five_seed"
    ),
}

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


BUFFER_MODE_NOTES = {
    "post_only": (
        "Drift window = documented event + 7 days after it "
        "(`[event_start, event_end + 1 day + 7 days)`); pre-drift = "
        "everything before the first drift window; post-drift = "
        "everything after it outside the union of drift windows."
    ),
    "symmetric": (
        "Drift window = 7 days before + documented event + 7 days after "
        "(`[event_start - 7 days, event_end + 1 day + 7 days)`); "
        "pre-drift = everything before the first drift window; "
        "post-drift = everything after it outside the union of drift "
        "windows."
    ),
}

LOCAL_NOTE = (
    "Event-local windows: 7 days pre, documented event as drift, 7 days "
    "post; timestamps outside every window are excluded. Independent of "
    "the full-stream buffer mode."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--forecaster",
        choices=sorted(FORECASTER_RUN_DIRS),
        default=DEFAULT_FORECASTER,
        help=(
            "Forecaster whose saved four-arm run is scored. Non-default "
            "forecasters get a `forecaster` column and their own folder."
        ),
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory containing seed_<n> outputs. "
            "Default: the run saved for --forecaster."
        ),
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
        help=(
            "Default: results/tables/adaptation/t3/<buffer-mode>, or "
            "t3/<forecaster>/<buffer-mode> for a non-default forecaster."
        ),
    )

    parser.add_argument(
        "--buffer-mode",
        choices=FULL_STREAM_BUFFER_MODES,
        default=DEFAULT_BUFFER_MODE,
        help=(
            "Full-stream regime convention: post_only = buffer after the "
            "event only (default); symmetric = buffer before and after."
        ),
    )

    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=("full", "local"),
        default=["full", "local"],
        help=(
            "Which T3 scopes to write. The local scope does not depend on "
            "--buffer-mode."
        ),
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
        help="Buffer length in days (fixed at the agreed 7).",
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


def add_regimes(
    forecasts,
    events,
    regime_days,
    scope="full",
    buffer_mode=DEFAULT_BUFFER_MODE,
):
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
                buffer_mode=buffer_mode,
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


def seed_invariant(seed_metrics):
    """True when every metric is identical across seeds (deterministic model)."""
    metrics = [
        column for column in seed_metrics.columns
        if column.startswith("mae_") or column == "pinball_loss"
    ]
    spread = seed_metrics.groupby(
        ["configuration", "region"]
    )[metrics].std(ddof=1)
    return bool((spread.fillna(0.0).abs() < 1e-9).all().all())


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

    scopes = tuple(dict.fromkeys(args.scopes))
    buffer_mode = args.buffer_mode
    forecaster = args.forecaster
    legacy = forecaster == DEFAULT_FORECASTER
    run_dir = args.run_dir or FORECASTER_RUN_DIRS[forecaster]

    def tag(scope):
        """File-name tag: full-stream files carry the buffer mode."""
        return f"{scope}_{buffer_mode}" if scope == "full" else scope

    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / buffer_mode
        if legacy
        else DEFAULT_OUTPUT_ROOT / forecaster / buffer_mode
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {output_dir}. "
            "Use a new --output-dir to preserve earlier results."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events)
    radii = load_seed_radii(run_dir, args.seeds)

    def label(frame):
        """Non-default forecasters carry their name as the first column."""
        if legacy:
            return frame
        frame = frame.copy()
        frame.insert(0, "forecaster", forecaster)
        return frame

    metrics_by_scope = {scope: [] for scope in scopes}
    audits = {}
    counts = []

    for seed in args.seeds:
        print(f"Processing seed {seed}...", flush=True)

        forecasts = load_seed_forecasts(run_dir, seed)
        forecasts = make_intervals(
            forecasts,
            radii=radii,
            alpha=args.alpha,
        )

        for scope in scopes:
            labelled = add_regimes(
                forecasts,
                events,
                regime_days=args.regime_days,
                scope=scope,
                buffer_mode=buffer_mode,
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
                    output_dir / f"regime_assignment_{tag(scope)}.csv",
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

    for scope in scopes:
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

        table = label(table)
        table.to_csv(
            output_dir / f"t3_{tag(scope)}.csv",
            index=False,
        )
        label(seed_metrics).to_csv(
            output_dir / f"t3_seed_metrics_{tag(scope)}.csv",
            index=False,
        )

        headers = list(table.columns)
        title = (
            f"T3 — full-stream forecast accuracy by regime "
            f"(buffer mode: {buffer_mode})"
            if scope == "full"
            else "T3 — event-local forecast accuracy by regime"
        )
        notes = [BUFFER_MODE_NOTES[buffer_mode] if scope == "full" else LOCAL_NOTE]
        if not legacy:
            notes.append(f"Forecaster: `{forecaster}`.")
            if seed_invariant(seed_metrics):
                notes.append(
                    "Deterministic forecaster: all five seeds give identical "
                    "results, so sd = 0.0 reflects no seed variation, not "
                    "independent replicates."
                )
        lines = [
            f"# {title}",
            "",
            *[line for note in notes for line in (note, "")],
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for values in table.itertuples(index=False, name=None):
            lines.append(
                "| " + " | ".join(map(str, values)) + " |"
            )

        (output_dir / f"t3_{tag(scope)}.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        print(f"\nT3 — {tag(scope)}")
        print(table.to_string(index=False))

    counts_name = (
        f"regime_observation_counts_{buffer_mode}.csv"
        if "full" in scopes
        else "regime_observation_counts.csv"
    )
    label(pd.DataFrame(counts)).to_csv(
        output_dir / counts_name,
        index=False,
    )

    import hashlib

    manifest = {
        "seeds": args.seeds,
        "regime_days": args.regime_days,
        "buffer_mode": buffer_mode,
        "scopes": list(scopes),
        "timezone": "Etc/GMT-10",
        "boundary": "left_closed_right_open",
        "full_rule": (
            "pre before first drift window; drift inside union of "
            "drift windows; post elsewhere after first window"
        ),
        "full_drift_window": {
            "post_only": "[event_start, event_end + 1 day + 7 days)",
            "symmetric": (
                "[event_start - 7 days, event_end + 1 day + 7 days)"
            ),
        }[buffer_mode],
        "event_selection": (
            "NEM plus region-specific events; full-stream buffered "
            "windows must overlap evaluated stream"
        ),
        "local_rule": (
            "7-day pre; documented event drift; 7-day post; "
            "outside excluded; priority drift > pre > post "
            "(independent of buffer_mode)"
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
    if not legacy:
        manifest["forecaster"] = forecaster
        manifest["run_dir"] = (
            str(run_dir.resolve().relative_to(ROOT))
            if run_dir.resolve().is_relative_to(ROOT)
            else str(run_dir)
        )

    (output_dir / f"t3_manifest_{buffer_mode}.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nT3 ({buffer_mode}) saved to: {output_dir}")


if __name__ == "__main__":
    main()