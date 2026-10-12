"""
PGS242-148 - Regenerate T4 (Adaptation cost) in the supervisor's schema.

Schema (requiredoutputs_Drift.pdf, "Tables T3 and T4 · Step 5"):

    arm | retrains | cumulative train samples | wall clock (s)
        | MAE gain vs arm A | gain per retrain

plus `region` and `seeds`, matching the final T3 table.

Metric definition, aligned with PG-S2-41 through the SHARED
drift_lab.evaluation.calculate_adaptation_gain():

    MAE gain vs arm A = MAE_drift(arm) - MAE_drift(arm A)   negative = better
    gain per retrain  = MAE gain vs arm A / n_retrains       NaN when 0 retrains

  * Drift-regime MAE, i.e. the same number as T3's "MAE drift" column
    (not overall test MAE, and not relative to Arm B as in the previous
    generate_t4_cost_summary.py).
  * Each arm is paired with Arm A on (region, seed) before differencing.
    Gain and gain per retrain are computed per seed, then reported as
    mean +/- sd across seeds.
  * Sign check against the supervisor's worked example: arm D with drift
    MAE 54.3, 2 retrains, gain -34.0 -> gain per retrain -17.0.

Inputs (final five-seed outputs, no re-run needed):
  <run_dir>/t3/t3_seed_level_metrics.csv  -> mae_drift per configuration/region/seed
  <run_dir>/summary_all_seeds.csv         -> n_retrains, train_samples_total,
                                             retraining_wall_clock_s

Outputs (<run_dir>/t4/):
  t4_seed_level_metrics.csv          every configuration x region x seed
  t4_all_configurations.csv          mean +/- sd for all 17 configurations
  t4_adaptation_cost.csv             headline table, same arms as final T3

Run from the repository root:
    python experiments/adaptation/generate_t4_adaptation_cost.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drift_lab.evaluation import calculate_adaptation_gain  # noqa: E402

DEFAULT_RUN_DIR = ROOT / Path(
    "results/tables/adaptation/revised/pgs242_128_five_seed_final"
)

# Same headline configurations as the final T3 table.
HEADLINE = {
    "A": "A · never retrain",
    "B": "B · periodic 30d",
    "C_kswin": "C · drift + full history (KSWIN)",
    "D_180d_page_hinkley": "D · drift + 180d window (Page-Hinkley)",
}
REFERENCE = "A"
KEYS = ["configuration", "region", "seed"]


def load_seed_level(run_dir: Path) -> pd.DataFrame:
    t3 = pd.read_csv(run_dir / "t3" / "t3_seed_level_metrics.csv")
    cost = pd.read_csv(run_dir / "summary_all_seeds.csv")

    t3 = t3[["arm", *KEYS, "detector", "window_days", "mae_drift"]]
    cost = cost[[*KEYS, "n_retrains", "train_samples_total", "retraining_wall_clock_s"]]

    merged = t3.merge(cost, on=KEYS, how="left", validate="one_to_one")
    missing = merged["n_retrains"].isna()
    if missing.any():
        raise ValueError(
            "No cost row in summary_all_seeds.csv for:\n"
            + merged.loc[missing, KEYS].to_string(index=False)
        )

    ref = (
        merged[merged["configuration"] == REFERENCE][["region", "seed", "mae_drift"]]
        .rename(columns={"mae_drift": "mae_drift_arm_a"})
    )
    merged = merged.merge(ref, on=["region", "seed"], how="left", validate="many_to_one")
    if merged["mae_drift_arm_a"].isna().any():
        raise ValueError("Some (region, seed) pairs have no Arm A row.")

    gains = merged.apply(
        lambda r: calculate_adaptation_gain(
            r["mae_drift"], r["mae_drift_arm_a"], int(r["n_retrains"])
        ),
        axis=1,
        result_type="expand",
    )
    merged["mae_gain_vs_arm_a"] = gains["drift_mae_difference_vs_arm_a"]
    merged["gain_per_retrain"] = gains["drift_mae_difference_per_retrain"]
    return merged.sort_values(["region", "configuration", "seed"]).reset_index(drop=True)


def aggregate(seed_level: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "n_retrains",
        "train_samples_total",
        "retraining_wall_clock_s",
        "mae_drift",
        "mae_gain_vs_arm_a",
        "gain_per_retrain",
    ]
    grouped = seed_level.groupby(["region", "arm", "configuration"], sort=True)
    stats = grouped[cols].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    stats["seeds"] = grouped["seed"].nunique()
    return stats.reset_index()


def _pm(mean, sd, decimals=1):
    if pd.isna(mean):
        return "—"
    sd = 0.0 if pd.isna(sd) else sd
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def _count(mean, sd):
    # retrains and train samples are deterministic given the frozen
    # changepoints; show sd only if seeds disagree.
    if pd.isna(sd) or sd == 0:
        return f"{mean:,.0f}"
    return f"{mean:,.0f} ± {sd:,.0f}"


def format_table(stats: pd.DataFrame, labels: dict | None) -> pd.DataFrame:
    df = stats.copy()
    if labels is not None:
        df = df[df["configuration"].isin(labels)].copy()
        df["arm_label"] = df["configuration"].map(labels)
        order = {c: i for i, c in enumerate(labels)}
        df["_order"] = df["configuration"].map(order)
        df = df.sort_values(["region", "_order"])
    else:
        df["arm_label"] = df["configuration"]

    return pd.DataFrame(
        {
            "arm": df["arm"],
            "arm_label": df["arm_label"],
            "region": df["region"],
            "retrains": [_count(m, s) for m, s in zip(df.n_retrains_mean, df.n_retrains_std)],
            "cumulative train samples": [
                _count(m, s) for m, s in zip(df.train_samples_total_mean, df.train_samples_total_std)
            ],
            "wall clock (s)": [
                _pm(m, s) for m, s in zip(df.retraining_wall_clock_s_mean, df.retraining_wall_clock_s_std)
            ],
            "MAE gain vs arm A": [
                _pm(m, s) for m, s in zip(df.mae_gain_vs_arm_a_mean, df.mae_gain_vs_arm_a_std)
            ],
            "gain per retrain": [
                _pm(m, s, 2) for m, s in zip(df.gain_per_retrain_mean, df.gain_per_retrain_std)
            ],
            "seeds": df["seeds"],
        }
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Generate T4 adaptation cost table")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    seed_level = load_seed_level(args.run_dir)
    stats = aggregate(seed_level)

    out_dir = args.run_dir / "t4"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_level.to_csv(out_dir / "t4_seed_level_metrics.csv", index=False)
    format_table(stats, None).to_csv(out_dir / "t4_all_configurations.csv", index=False)
    headline = format_table(stats, HEADLINE)
    headline.to_csv(out_dir / "t4_adaptation_cost.csv", index=False)

    print(headline.to_string(index=False))
    print(f"\nWrote T4 outputs to {out_dir}")


if __name__ == "__main__":
    main()
