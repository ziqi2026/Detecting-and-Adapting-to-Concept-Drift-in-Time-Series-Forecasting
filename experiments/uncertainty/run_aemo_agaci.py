"""Experimental AEMO conformal comparison: Static vs ACI vs AgACI (BOA).

Uses the final PGS242-128 forecast streams without regenerating test forecasts.
Calibration is seed-specific and calibration-period-only (FROZEN across TEST).
Resets all online states per seed x region x forecast configuration.
Test observation y_t updates ACI/BOA only AFTER interval t is constructed.
Outputs are independent of the team's PGS242-160 files and runs.csv.
This experiment is not the original rolling-retrained AgACI paper protocol.
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
from drift_lab.forecasters import GradientBoostedForecaster, frozen_error_stream
from drift_lab.evaluation import calculate_empirical_coverage, calculate_mean_interval_width
from drift_lab.uncertainty import (
    DEFAULT_ALPHA,
    DEFAULT_GAMMA,
    adaptive_prediction_intervals,
    prediction_intervals,
)
from drift_lab.agaci import AGACI_GAMMAS, agaci_prediction_intervals

REGIONS = ("SA1", "NSW1")
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
DATA_PATH = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "experimental" / "agaci_boa_fixed_calibration"
PROTOCOL_ID = "team42-experimental-agaci-boa-fixed-calibration-v1"


def experiment_hash(upstream_hash, configuration, method, alpha, gamma):
    payload = dict(protocol=PROTOCOL_ID, upstream_config_hash=upstream_hash,
                   configuration=configuration, method=method,
                   alpha=float(alpha), gamma=float(gamma) if method == "adaptive_conformal" else None,
                   agaci_gammas=AGACI_GAMMAS.tolist() if method == "agaci_boa" else None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("Non-finite conformal interval bounds")
    if np.any(lower > upper):
        raise ValueError("Inverted conformal interval bounds")
    return (
        float(calculate_empirical_coverage(y_true, lower, upper)),
        float(calculate_mean_interval_width(lower, upper)),
    )


def process_region(seed, region, seed_dir, adaptation_summary, split_id,
                   upstream_config_hash, alpha, gamma):
    calibration_residuals, train_size, expected_timestamps = build_calibration_residuals(
        region=region, seed=seed
    )
    forecasts = load_forecasts(seed_dir=seed_dir, seed=seed, region=region)
    configurations = sorted(forecasts["configuration"].dropna().astype(str).unique())
    prediction_frames = []
    summary_rows = []

    for configuration in configurations:
        config_frame = validate_configuration_stream(
            config_frame=forecasts[forecasts["configuration"] == configuration].copy(),
            expected_timestamps=expected_timestamps,
            seed=seed, region=region, configuration=configuration,
        )
        meta_rows = adaptation_summary.loc[
            (adaptation_summary["region"] == region)
            & (adaptation_summary["configuration"] == configuration)
        ]
        if len(meta_rows) != 1:
            raise ValueError(f"Expected one metadata row for {region}/{configuration}, got {len(meta_rows)}")
        meta = meta_rows.iloc[0]
        if train_size != int(meta["train_samples_initial"]):
            raise RuntimeError(f"Initial training size mismatch for {region}/{configuration}")
        point_forecasts = config_frame["y_pred"].to_numpy(dtype=float)
        y_true = config_frame["y_true"].to_numpy(dtype=float)

        start = perf_counter()
        static_lower, static_upper, _ = prediction_intervals(
            point_forecasts, calibration_residuals, alpha=alpha
        )
        static_elapsed = perf_counter() - start

        start = perf_counter()
        aci_lower, aci_upper, _, _, _ = adaptive_prediction_intervals(
            point_forecasts, y_true, calibration_residuals, alpha=alpha, gamma=gamma
        )
        aci_elapsed = perf_counter() - start

        start = perf_counter()
        agaci_lower, agaci_upper = agaci_prediction_intervals(
            point_forecasts, y_true, calibration_residuals, alpha=alpha
        )
        agaci_elapsed = perf_counter() - start

        prediction_frames.append(pd.DataFrame({
            "SETTLEMENTDATE": config_frame["SETTLEMENTDATE"].values,
            "region": region,
            "seed": seed,
            "configuration": configuration,
            "y_true": y_true,
            "point_forecast": point_forecasts,
            "static_lower": static_lower,
            "static_upper": static_upper,
            "static_covered": (y_true >= static_lower) & (y_true <= static_upper),
            "static_width": static_upper - static_lower,
            "aci_lower": aci_lower,
            "aci_upper": aci_upper,
            "aci_covered": (y_true >= aci_lower) & (y_true <= aci_upper),
            "aci_width": aci_upper - aci_lower,
            "agaci_lower": agaci_lower,
            "agaci_upper": agaci_upper,
            "agaci_covered": (y_true >= agaci_lower) & (y_true <= agaci_upper),
            "agaci_width": agaci_upper - agaci_lower,
        }))
        for method, lower, upper, elapsed in (
            ("static_conformal", static_lower, static_upper, static_elapsed),
            ("adaptive_conformal", aci_lower, aci_upper, aci_elapsed),
            ("agaci_boa", agaci_lower, agaci_upper, agaci_elapsed),
        ):
            coverage, width = evaluate_method(y_true, lower, upper)
            summary_rows.append({
                "protocol": PROTOCOL_ID,
                "split_id": split_id,
                "seed": seed,
                "region": region,
                "configuration": configuration,
                "method": method,
                "alpha": alpha,
                "gamma": gamma if method == "adaptive_conformal" else np.nan,
                "calibration_size": len(calibration_residuals),
                "test_size": len(y_true),
                "empirical_coverage": coverage,
                "mean_interval_width": width,
                "n_retrains": int(meta["n_retrains"]),
                "train_samples_total": int(meta["train_samples_total"]),
                "wall_clock_s": elapsed,
                "config_hash": experiment_hash(upstream_config_hash, configuration, method, alpha, gamma),
            })
            print(f"seed={seed} {region} {configuration} {method}: "
                  f"coverage={coverage:.4f}, mean_width={width:.2f}", flush=True)

    if not prediction_frames:
        raise ValueError(f"No forecast configurations found for {seed}/{region}")
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Input directory with seed_<n>/manifest.json, summary.csv and forecast CSVs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Independent experimental output directory (never inside input run-dir)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help="ACI comparator only; AgACI uses AGACI_GAMMAS")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    seeds = tuple(args.seeds)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing forecast run directory: {run_dir}")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 1 for s in seeds):
        raise ValueError("--seeds must be a nonempty list of unique positive integers")
    if not (0 < args.alpha < 1):
        raise ValueError("--alpha must be between 0 and 1")
    if not (args.gamma > 0):
        raise ValueError("--gamma must be positive")
    # Never allow an output path inside the input run directory or the team's original log path.
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("--output-dir must be outside --run-dir to protect original results")
    if output_dir == PROJECT_ROOT or output_dir == PROJECT_ROOT / "results":
        raise ValueError("Choose a dedicated experimental --output-dir")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Experimental output directory is not empty: {output_dir}. "
                              "Choose a fresh --output-dir for each run.")

    first_manifest = load_manifest(run_dir / f"seed_{seeds[0]}")
    split_id = first_manifest["split_id"]
    # Validate all metadata before creating output files.
    manifests = {}
    summaries = {}
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        manifest = load_manifest(seed_dir)
        if int(manifest["seed"]) != seed or manifest["split_id"] != split_id:
            raise ValueError(f"Invalid seed or split in {seed_dir}/manifest.json")
        manifests[seed] = manifest
        summaries[seed] = load_adaptation_summary(seed_dir, seed)
        for region in REGIONS:
            if not (seed_dir / f"forecasts_{region}.csv.gz").is_file():
                raise FileNotFoundError(f"Missing forecasts for {region} in {seed_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_output_dir = output_dir / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=False)
        for region in REGIONS:
            intervals, summary = process_region(
                seed=seed,
                region=region,
                seed_dir=seed_dir,
                adaptation_summary=summaries[seed],
                split_id=split_id,
                upstream_config_hash=manifests[seed]["config_hash"],
                alpha=args.alpha,
                gamma=args.gamma,
            )
            intervals.to_csv(seed_output_dir / f"agaci_comparison_intervals_{region}.csv.gz",
                             index=False, compression="gzip")
            summary.to_csv(seed_output_dir / f"agaci_comparison_summary_{region}.csv", index=False)
            all_summaries.append(summary)
        (seed_output_dir / "agaci_manifest.json").write_text(
            json.dumps({
                "protocol": PROTOCOL_ID,
                "mode": "BOA aggregation of ACI experts, fixed calibration residuals",
                "calibration": "seed-specific calibration period only (frozen in TEST)",
                "upstream_run_id": manifests[seed].get("run_id"),
                "upstream_config_hash": manifests[seed]["config_hash"],
                "seed": seed,
                "split_id": split_id,
                "alpha": args.alpha,
                "aci_gamma": args.gamma,
                "agaci_gammas": AGACI_GAMMAS.tolist(),
                "regions": list(REGIONS),
                "causality": "predict t before using observation t",
                "shared_runs_csv_modified": False,
            }, indent=2) + "\n", encoding="utf-8"
        )
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(output_dir / "agaci_comparison_all_seeds.csv", index=False)
    print(f"\nExperimental AgACI comparison complete. Results: {output_dir}")
    print("Original team uncertainty files and results/tables/runs.csv were not modified.")


if __name__ == "__main__":
    main()
