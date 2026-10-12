"""Run static and adaptive conformal UQ on final AEMO forecasts.

PGS242-160

Protocol
--------
- Regions: SA1, NSW1
- Forecast seeds: 1, 2, 3, 4, 5
- Point forecasts: final PGS242-128 adaptation outputs
- Calibration residuals: seed-specific, calibration-period only
- Static UQ: frozen Sprint 4 split-conformal baseline
- Adaptive UQ: causal adaptive conformal alpha update
- Adaptive state is reset independently for every
  seed x region x forecasting configuration
- TEST observation y_t is used only after interval C_t is constructed
- Metrics: empirical coverage and mean interval width
- Shared results: results/tables/runs.csv

This runner does not regenerate TEST forecasts.
"""

from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from drift_lab.data_loader import load_and_split_aemo
from drift_lab.forecasters import (
    GradientBoostedForecaster,
    frozen_error_stream,
)
from drift_lab.evaluation import (
    calculate_empirical_coverage,
    calculate_mean_interval_width,
)
from drift_lab.run_logger import log_run
from drift_lab.uncertainty import (
    DEFAULT_ALPHA,
    DEFAULT_GAMMA,
    adaptive_prediction_intervals,
    conformal_radius,
    prediction_intervals,
)


REGIONS = ("SA1", "NSW1")
DEFAULT_SEEDS = (1, 2, 3, 4, 5)

DATA_PATH = PROJECT_ROOT / "data" / "raw"
SHARED_RUNS = PROJECT_ROOT / "results" / "tables" / "runs.csv"

GROUP = "uncertainty"
DATASET = "AEMO_TOTALDEMAND"
REGIME = "full-test"
PROTOCOL_ID = "pgs242-160-adaptive-conformal-v1"


def uq_config_hash(
    upstream_config_hash,
    configuration,
    method,
    alpha,
    gamma,
):
    """Create a reproducible UQ-specific configuration hash."""
    payload = {
        "protocol": PROTOCOL_ID,
        "upstream_config_hash": upstream_config_hash,
        "forecast_configuration": configuration,
        "method": method,
        "alpha": float(alpha),
        "gamma": (
            float(gamma)
            if method == "adaptive_conformal"
            else None
        ),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def load_manifest(seed_dir):
    path = seed_dir / "manifest.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing upstream manifest: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_adaptation_summary(seed_dir, seed):
    path = seed_dir / "summary.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing upstream summary: {path}"
        )

    summary = pd.read_csv(path)

    required = {
        "configuration",
        "region",
        "seed",
        "n_retrains",
        "train_samples_total",
        "config_hash",
    }

    missing = required.difference(summary.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: "
            f"{sorted(missing)}"
        )

    actual_seeds = set(
        summary["seed"].dropna().astype(int).unique()
    )

    if actual_seeds != {seed}:
        raise ValueError(
            f"{path} does not contain only seed {seed}: "
            f"{sorted(actual_seeds)}"
        )

    duplicates = summary.duplicated(
        subset=["region", "configuration"]
    )

    if duplicates.any():
        raise ValueError(
            f"{path} contains duplicate "
            "region/configuration rows."
        )

    return summary


def build_calibration_residuals(region, seed):
    """Build calibration-only residuals for one region and seed."""
    _, train, calibration, test = load_and_split_aemo(
        region=region,
        data_path=DATA_PATH,
    )

    model = GradientBoostedForecaster(
        feature_mode="calendar+lags",
        lags=(48, 336),
        seed=seed,
    )
    model.fit(train)

    calibration_stream = frozen_error_stream(
        model,
        calibration,
        history_df=train,
    )

    residuals = (
        calibration_stream["error"]
        .to_numpy(dtype=float)
    )

    if len(residuals) != len(calibration):
        raise RuntimeError(
            "Calibration residual length mismatch."
        )

    if not np.isfinite(residuals).all():
        raise RuntimeError(
            "Calibration residuals contain non-finite values."
        )

    test_timestamps = pd.to_datetime(
        test["SETTLEMENTDATE"]
    ).reset_index(drop=True)

    return (
        residuals,
        len(train),
        test_timestamps,
    )


def load_forecasts(seed_dir, seed, region):
    path = seed_dir / f"forecasts_{region}.csv.gz"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing upstream forecast file: {path}"
        )

    usecols = [
        "SETTLEMENTDATE",
        "y_true",
        "y_pred",
        "configuration",
        "region",
        "seed",
    ]

    frame = pd.read_csv(
        path,
        usecols=usecols,
        low_memory=False,
    )

    if frame.empty:
        raise ValueError(
            f"Upstream forecast file is empty: {path}"
        )

    actual_seeds = set(
        frame["seed"].dropna().astype(int).unique()
    )

    if actual_seeds != {seed}:
        raise ValueError(
            f"{path} does not contain only seed {seed}: "
            f"{sorted(actual_seeds)}"
        )

    actual_regions = set(
        frame["region"].dropna().astype(str).unique()
    )

    if actual_regions != {region}:
        raise ValueError(
            f"{path} does not contain only region {region}: "
            f"{sorted(actual_regions)}"
        )

    frame["SETTLEMENTDATE"] = pd.to_datetime(
        frame["SETTLEMENTDATE"]
    )

    return frame


def validate_configuration_stream(
    config_frame,
    expected_timestamps,
    seed,
    region,
    configuration,
):
    """Validate one chronological forecasting stream."""
    frame = (
        config_frame
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    if frame["SETTLEMENTDATE"].duplicated().any():
        raise ValueError(
            f"Duplicate timestamps for seed={seed}, "
            f"region={region}, configuration={configuration}."
        )

    if len(frame) != len(expected_timestamps):
        raise ValueError(
            f"Unexpected test length for seed={seed}, "
            f"region={region}, configuration={configuration}: "
            f"{len(frame)} vs {len(expected_timestamps)}."
        )

    timestamps = frame["SETTLEMENTDATE"].reset_index(
        drop=True
    )

    if not timestamps.equals(expected_timestamps):
        raise ValueError(
            f"Timestamp mismatch for seed={seed}, "
            f"region={region}, configuration={configuration}."
        )

    for column in ("y_true", "y_pred"):
        values = frame[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite {column} values for "
                f"seed={seed}, region={region}, "
                f"configuration={configuration}."
            )

    return frame


def evaluate_method(y_true, lower, upper):
    coverage = calculate_empirical_coverage(
        y_true,
        lower,
        upper,
    )

    mean_width = calculate_mean_interval_width(
        lower,
        upper,
    )

    return coverage, mean_width


def log_metrics(
    *,
    configuration,
    method,
    region,
    seed,
    coverage,
    mean_width,
    upstream_meta,
    split_id,
    uq_hash,
    wall_clock_s,
    results_path,
):
    """Write PGS242-160 metrics to the shared long-format harness."""
    common = {
        "method": f"{configuration}|{method}",
        "dataset": DATASET,
        "region": region,
        "regime": REGIME,
        "train_samples": int(
            upstream_meta["train_samples_total"]
        ),
        "group": GROUP,
        "seed": int(seed),
        "config_hash": uq_hash,
        "split_id": split_id,
        "n_retrains": int(
            upstream_meta["n_retrains"]
        ),
        "wall_clock_s": wall_clock_s,
        "results_path": str(results_path),
    }

    for metric_name, metric_value in (
        ("empirical_coverage", coverage),
        ("mean_interval_width", mean_width),
    ):
        log_run(
            metric_name=metric_name,
            metric_value=metric_value,
            **common,
        )


def ensure_no_duplicate_logs(
    results_path,
    seeds,
    methods,
    split_id,
):
    """Prevent accidental duplicate final UQ logging."""
    path = Path(results_path)

    if not path.exists() or path.stat().st_size == 0:
        return

    existing = pd.read_csv(path)

    if existing.empty:
        return

    required = {
        "group",
        "method",
        "seed",
        "split_id",
    }

    if not required.issubset(existing.columns):
        return

    duplicate = existing[
        (existing["group"] == GROUP)
        & (existing["split_id"] == split_id)
        & (existing["seed"].isin(seeds))
        & (existing["method"].isin(methods))
    ]

    if not duplicate.empty:
        raise RuntimeError(
            "Matching PGS242-160 uncertainty rows already exist in "
            f"{results_path}. Refusing to duplicate shared logs."
        )


def process_region(
    *,
    seed,
    region,
    seed_dir,
    adaptation_summary,
    split_id,
    upstream_config_hash,
    alpha,
    gamma,
    write_shared_log,
):
    """Run static and adaptive UQ for one seed and region."""
    (
        calibration_residuals,
        train_size,
        expected_timestamps,
    ) = build_calibration_residuals(
        region=region,
        seed=seed,
    )

    static_radius = conformal_radius(
        calibration_residuals,
        alpha=alpha,
    )

    forecasts = load_forecasts(
        seed_dir=seed_dir,
        seed=seed,
        region=region,
    )

    configurations = sorted(
        forecasts["configuration"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    region_outputs = []
    summary_rows = []

    for configuration in configurations:
        config_frame = forecasts[
            forecasts["configuration"] == configuration
        ].copy()

        config_frame = validate_configuration_stream(
            config_frame=config_frame,
            expected_timestamps=expected_timestamps,
            seed=seed,
            region=region,
            configuration=configuration,
        )

        meta_rows = adaptation_summary[
            (adaptation_summary["region"] == region)
            & (
                adaptation_summary["configuration"]
                == configuration
            )
        ]

        if len(meta_rows) != 1:
            raise ValueError(
                f"Expected exactly one summary row for "
                f"{region}/{configuration}; found {len(meta_rows)}."
            )

        upstream_meta = meta_rows.iloc[0]

        point_forecasts = config_frame[
            "y_pred"
        ].to_numpy(dtype=float)

        y_true = config_frame[
            "y_true"
        ].to_numpy(dtype=float)

        static_started = perf_counter()

        (
            static_lower,
            static_upper,
            static_escalate,
        ) = prediction_intervals(
            point_forecasts,
            calibration_residuals,
            alpha=alpha,
        )

        static_elapsed = (
            perf_counter() - static_started
        )

        (
            static_coverage,
            static_mean_width,
        ) = evaluate_method(
            y_true,
            static_lower,
            static_upper,
        )

        adaptive_started = perf_counter()

        (
            adaptive_lower,
            adaptive_upper,
            adaptive_escalate,
            alpha_path,
            radius_path,
        ) = adaptive_prediction_intervals(
            point_forecasts,
            y_true,
            calibration_residuals,
            alpha=alpha,
            gamma=gamma,
        )

        adaptive_elapsed = (
            perf_counter() - adaptive_started
        )

        (
            adaptive_coverage,
            adaptive_mean_width,
        ) = evaluate_method(
            y_true,
            adaptive_lower,
            adaptive_upper,
        )

        static_hash = uq_config_hash(
            upstream_config_hash,
            configuration,
            "static_conformal",
            alpha,
            gamma,
        )

        adaptive_hash = uq_config_hash(
            upstream_config_hash,
            configuration,
            "adaptive_conformal",
            alpha,
            gamma,
        )

        if write_shared_log:
            log_metrics(
                configuration=configuration,
                method="static_conformal",
                region=region,
                seed=seed,
                coverage=static_coverage,
                mean_width=static_mean_width,
                upstream_meta=upstream_meta,
                split_id=split_id,
                uq_hash=static_hash,
                wall_clock_s=static_elapsed,
                results_path=SHARED_RUNS,
            )

            log_metrics(
                configuration=configuration,
                method="adaptive_conformal",
                region=region,
                seed=seed,
                coverage=adaptive_coverage,
                mean_width=adaptive_mean_width,
                upstream_meta=upstream_meta,
                split_id=split_id,
                uq_hash=adaptive_hash,
                wall_clock_s=adaptive_elapsed,
                results_path=SHARED_RUNS,
            )

        output = pd.DataFrame(
            {
                "SETTLEMENTDATE": (
                    config_frame["SETTLEMENTDATE"].values
                ),
                "region": region,
                "seed": seed,
                "configuration": configuration,
                "y_true": y_true,
                "point_forecast": point_forecasts,
                "static_lower_90": static_lower,
                "static_upper_90": static_upper,
                "static_covered_90": (
                    (y_true >= static_lower)
                    & (y_true <= static_upper)
                ),
                "static_interval_width": (
                    static_upper - static_lower
                ),
                "static_escalate_flag": static_escalate,
                "adaptive_lower_90": adaptive_lower,
                "adaptive_upper_90": adaptive_upper,
                "adaptive_covered_90": (
                    (y_true >= adaptive_lower)
                    & (y_true <= adaptive_upper)
                ),
                "adaptive_interval_width": (
                    adaptive_upper - adaptive_lower
                ),
                "adaptive_alpha": alpha_path,
                "adaptive_radius": radius_path,
                "adaptive_escalate_flag": (
                    adaptive_escalate
                ),
            }
        )

        region_outputs.append(output)

        summary_rows.extend(
            [
                {
                    "region": region,
                    "seed": seed,
                    "configuration": configuration,
                    "method": "static_conformal",
                    "alpha": alpha,
                    "gamma": np.nan,
                    "calibration_size": (
                        len(calibration_residuals)
                    ),
                    "conformal_radius": static_radius,
                    "test_size": len(y_true),
                    "empirical_coverage": static_coverage,
                    "mean_interval_width": (
                        static_mean_width
                    ),
                    "n_retrains": int(
                        upstream_meta["n_retrains"]
                    ),
                    "train_samples_total": int(
                        upstream_meta[
                            "train_samples_total"
                        ]
                    ),
                    "wall_clock_s": static_elapsed,
                    "uq_config_hash": static_hash,
                },
                {
                    "region": region,
                    "seed": seed,
                    "configuration": configuration,
                    "method": "adaptive_conformal",
                    "alpha": alpha,
                    "gamma": gamma,
                    "calibration_size": (
                        len(calibration_residuals)
                    ),
                    "conformal_radius": np.nan,
                    "test_size": len(y_true),
                    "empirical_coverage": (
                        adaptive_coverage
                    ),
                    "mean_interval_width": (
                        adaptive_mean_width
                    ),
                    "n_retrains": int(
                        upstream_meta["n_retrains"]
                    ),
                    "train_samples_total": int(
                        upstream_meta[
                            "train_samples_total"
                        ]
                    ),
                    "wall_clock_s": adaptive_elapsed,
                    "uq_config_hash": adaptive_hash,
                },
            ]
        )

        print(
            f"seed={seed} region={region} "
            f"configuration={configuration}: "
            f"static coverage={static_coverage:.4f}, "
            f"adaptive coverage={adaptive_coverage:.4f}"
        )

    region_output = pd.concat(
        region_outputs,
        ignore_index=True,
    )

    region_summary = pd.DataFrame(summary_rows)

    # Compatibility summary for the existing Sprint 4/T3 handoff.
    baseline = region_summary[
        (region_summary["configuration"] == "A")
        & (
            region_summary["method"]
            == "static_conformal"
        )
    ]

    if len(baseline) != 1:
        raise ValueError(
            f"Expected exactly one Arm A static row for {region}."
        )

    baseline = baseline.iloc[0]

    legacy_summary = {
        "region": region,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "calibration_size": len(
            calibration_residuals
        ),
        "conformal_radius": static_radius,
        "test_size": int(baseline["test_size"]),
        "empirical_coverage": float(
            baseline["empirical_coverage"]
        ),
        "mean_interval_width": float(
            baseline["mean_interval_width"]
        ),
    }

    if train_size != int(
        adaptation_summary[
            adaptation_summary["region"] == region
        ]["train_samples_initial"].iloc[0]
    ):
        raise RuntimeError(
            f"Initial training-size mismatch for {region}."
        )

    return (
        region_output,
        region_summary,
        legacy_summary,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Parent PGS242-128 run directory containing "
            "seed_<n> subdirectories."
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Forecast seeds. Default: 1 2 3 4 5.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
    )

    parser.add_argument(
        "--skip-shared-log",
        action="store_true",
        help=(
            "Run and write uncertainty artifacts without "
            "appending to results/tables/runs.csv."
        ),
    )

    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    seeds = tuple(args.seeds)

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}"
        )

    if not seeds:
        raise ValueError(
            "At least one seed is required."
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "--seeds must contain unique values."
        )

    all_seed_summaries = []

    # Read one manifest first so duplicate-log checking uses the exact
    # upstream split identifier.
    first_seed_dir = run_dir / f"seed_{seeds[0]}"
    first_manifest = load_manifest(first_seed_dir)
    split_id = first_manifest["split_id"]

    expected_methods = []

    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        summary = load_adaptation_summary(
            seed_dir,
            seed,
        )

        expected_methods.extend(
            f"{configuration}|{method}"
            for configuration in sorted(
                summary["configuration"]
                .astype(str)
                .unique()
                .tolist()
            )
            for method in (
                "static_conformal",
                "adaptive_conformal",
            )
        )

    if not args.skip_shared_log:
        ensure_no_duplicate_logs(
            results_path=SHARED_RUNS,
            seeds=seeds,
            methods=set(expected_methods),
            split_id=split_id,
        )

    for seed in seeds:
        print("=" * 72)
        print(f"PGS242-160 adaptive conformal: seed {seed}")
        print("=" * 72)

        seed_dir = run_dir / f"seed_{seed}"
        manifest = load_manifest(seed_dir)

        if int(manifest["seed"]) != seed:
            raise ValueError(
                f"Manifest seed mismatch in {seed_dir}."
            )

        if manifest["split_id"] != split_id:
            raise ValueError(
                "Seed manifests do not share the same split_id."
            )

        upstream_config_hash = manifest["config_hash"]

        adaptation_summary = load_adaptation_summary(
            seed_dir,
            seed,
        )

        seed_output_dir = seed_dir / "uncertainty"
        seed_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        seed_summaries = []
        legacy_summaries = []

        for region in REGIONS:
            (
                region_output,
                region_summary,
                legacy_summary,
            ) = process_region(
                seed=seed,
                region=region,
                seed_dir=seed_dir,
                adaptation_summary=adaptation_summary,
                split_id=split_id,
                upstream_config_hash=(
                    upstream_config_hash
                ),
                alpha=args.alpha,
                gamma=args.gamma,
                write_shared_log=(
                    not args.skip_shared_log
                ),
            )

            region_output.to_csv(
                seed_output_dir
                / (
                    "aemo_conformal_intervals_"
                    f"{region}.csv.gz"
                ),
                index=False,
                compression={
                    "method": "gzip",
                    "mtime": 0,
                },
            )

            seed_summaries.append(region_summary)
            legacy_summaries.append(
                legacy_summary
            )

        seed_summary = pd.concat(
            seed_summaries,
            ignore_index=True,
        )

        seed_summary.to_csv(
            seed_output_dir
            / "aemo_conformal_comparison_summary.csv",
            index=False,
        )

        pd.DataFrame(
            legacy_summaries
        ).to_csv(
            seed_output_dir
            / "aemo_prediction_intervals_90_summary.csv",
            index=False,
        )

        uq_manifest = {
            "task": "PGS242-160",
            "protocol": PROTOCOL_ID,
            "upstream_run_id": manifest["run_id"],
            "seed": seed,
            "split_id": split_id,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "regions": list(REGIONS),
            "adaptive_update": (
                "alpha_(t+1) = alpha_t + "
                "gamma * (target_alpha - error_t)"
            ),
            "causal_order": (
                "construct interval_t before observing y_t"
            ),
            "shared_log_written": (
                not args.skip_shared_log
            ),
        }

        (
            seed_output_dir
            / "adaptive_conformal_manifest.json"
        ).write_text(
            json.dumps(
                uq_manifest,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        all_seed_summaries.append(
            seed_summary
        )

    combined = pd.concat(
        all_seed_summaries,
        ignore_index=True,
    )

    combined.to_csv(
        run_dir
        / "aemo_conformal_comparison_all_seeds.csv",
        index=False,
    )

    print()
    print("=" * 72)
    print("PGS242-160 uncertainty experiment complete")
    print("=" * 72)
    print(
        f"Seeds: {list(seeds)} | "
        f"alpha={args.alpha} | gamma={args.gamma}"
    )
    print(
        f"Shared log written: "
        f"{not args.skip_shared_log}"
    )


if __name__ == "__main__":
    main()
