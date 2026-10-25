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

Buffer mode (PGS242-169): the drift-regime MAE comes from T3, so T4 is
generated per T3 regime convention. --buffer-mode post_only (default, event +
7 days after) or symmetric (7 days before and after) selects which T3 seed
metrics are read and where the outputs go. Retrain counts, training samples
and wall-clock do not depend on the regime labels; only the MAE gain and the
gain per retrain do.

Inputs (final five-seed outputs, no re-run needed):
  results/tables/adaptation/t3/<mode>/t3_seed_metrics_full_<mode>.csv
                                          -> full-stream mae_drift
  <run_dir>/summary_all_seeds.csv         -> n_retrains, train_samples_total,
                                             retraining_wall_clock_s

Outputs (results/tables/adaptation/t4/<mode>/), CSV + markdown:
  t4_seed_level_metrics_<mode>.csv        every configuration x region x seed
  t4_all_configurations_<mode>.*          mean +/- sd, all 17 configurations
  t4_representative_per_arm_<mode>.*      one configuration per arm (A-D)

Forecasters (PGS242-169): --forecaster selects which forecaster's T3 seed
metrics and four-arm run are read. The default, gradient_boosted, keeps the
original paths and schema. Any other forecaster reads t3/<forecaster>/<mode>/,
writes t4/<forecaster>/<mode>/, and adds a leading `forecaster` column.

Run from the repository root:
    python experiments/adaptation/generate_t4_adaptation_cost.py
    python experiments/adaptation/generate_t4_adaptation_cost.py \
        --buffer-mode symmetric
    python experiments/adaptation/generate_t4_adaptation_cost.py \
        --forecaster seasonal_naive
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

BUFFER_MODES = ("post_only", "symmetric")
DEFAULT_BUFFER_MODE = "post_only"
DEFAULT_RUN_DIR = ROOT / "results/tables/adaptation/revised/pgs242_128_five_seed_final"
DEFAULT_FORECASTER = "gradient_boosted"
FORECASTER_RUN_DIRS = {
    DEFAULT_FORECASTER: DEFAULT_RUN_DIR,
    "seasonal_naive": ROOT / "results/tables/adaptation/revised/seasonal_naive_five_seed",
}
T3_ROOT = ROOT / "results/tables/adaptation/t3"
T4_ROOT = ROOT / "results/tables/adaptation/t4"


def _forecaster_dir(root, forecaster, buffer_mode):
    """Legacy layout <root>/<mode> for the default; <root>/<forecaster>/<mode> else."""
    if forecaster == DEFAULT_FORECASTER:
        return root / buffer_mode
    return root / forecaster / buffer_mode


def default_t3_file(buffer_mode, forecaster=DEFAULT_FORECASTER):
    return (_forecaster_dir(T3_ROOT, forecaster, buffer_mode)
            / f"t3_seed_metrics_full_{buffer_mode}.csv")


def default_output_dir(buffer_mode, forecaster=DEFAULT_FORECASTER):
    return _forecaster_dir(T4_ROOT, forecaster, buffer_mode)


FORECASTER_NOTES = {
    "seasonal_naive": (
        "Seasonal-naive has no fitted state: `fit()` is a no-op and `predict()` "
        "reads the observation 48 half-hours (one day) earlier. A retrain "
        "therefore changes no forecast. `retrains` and `cumulative train samples` "
        "are the schedule's bookkeeping (the refit data selection is unchanged) "
        "and `wall clock` is the cost of an empty `fit()`."
    ),
}


def seed_invariant(seed_level: pd.DataFrame) -> bool:
    """True when every seed gives the same drift MAE (deterministic model)."""
    spread = seed_level.groupby(["configuration", "region"])["mae_drift"].std(ddof=1)
    return bool((spread.fillna(0.0).abs() < 1e-9).all())


# Same headline configurations as the final T3 table.
HEADLINE = {
    "A": "A · never retrain",
    "B": "B · periodic 30d",
    "C_kswin": "C · drift + full history (KSWIN)",
    "D_180d_page_hinkley": "D · drift + 180d window (Page-Hinkley)",
}
BUFFER_NOTES = {
    "post_only": "drift = event + 7 days after it",
    "symmetric": "drift = 7 days before + event + 7 days after",
}
REFERENCE = "A"
KEYS = ["configuration", "region", "seed"]


def load_seed_level(t3_file: Path, cost_file: Path) -> pd.DataFrame:
    if not t3_file.is_file():
        raise FileNotFoundError(f"Updated full T3 metrics not found: {t3_file}")
    if not cost_file.is_file():
        raise FileNotFoundError(f"Original adaptation cost summary not found: {cost_file}")
    t3 = pd.read_csv(t3_file)
    cost = pd.read_csv(cost_file)
    expected_t3 = {"arm", *KEYS, "detector", "window_days", "mae_drift"}
    expected_cost = {*KEYS, "n_retrains", "train_samples_total", "retraining_wall_clock_s"}
    for name, df, expected in (("T3", t3, expected_t3), ("cost", cost, expected_cost)):
        missing = expected.difference(df.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
    if t3.duplicated(KEYS).any() or cost.duplicated(KEYS).any():
        raise ValueError("Duplicate configuration/region/seed combinations in T3 or cost inputs")
    if t3["mae_drift"].isna().any():
        raise ValueError("T3 contains missing drift-regime MAE values")

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
        # A, B, C_<detector>, D_30d..D_180d (numeric window), then detector.
        window = df["configuration"].str.extract(r"^D_(\d+)d_")[0].astype(float)
        df["_arm"] = df["arm"].map({"A": 0, "B": 1, "C": 2, "D": 3})
        df["_window"] = window.fillna(0)
        df = df.sort_values(["region", "_arm", "_window", "configuration"])

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


def to_markdown(table: pd.DataFrame, title: str, note: str) -> str:
    headers = [c for c in table.columns if c != "arm"]
    lines = [f"# {title}", "", note, "",
             "| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in table[headers].iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate T4 adaptation cost table")
    parser.add_argument("--buffer-mode", choices=BUFFER_MODES, default=DEFAULT_BUFFER_MODE,
                        help="T3 regime convention the drift MAE comes from "
                             "(post_only = default, symmetric = +/- 7 days)")
    parser.add_argument("--forecaster", choices=sorted(FORECASTER_RUN_DIRS),
                        default=DEFAULT_FORECASTER,
                        help="Forecaster to report. Non-default forecasters get a "
                             "`forecaster` column and their own folder.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Five-seed adaptation results containing summary_all_seeds.csv. "
                             "Default: the run saved for --forecaster")
    parser.add_argument("--t3-file", type=Path, default=None,
                        help="Full-stream T3 seed-level metrics (NOT the local T3 file). "
                             "Default: t3/<buffer-mode>/t3_seed_metrics_full_<buffer-mode>.csv")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: t4/<buffer-mode>, or t4/<forecaster>/<buffer-mode>")
    args = parser.parse_args()

    mode = args.buffer_mode
    forecaster = args.forecaster
    legacy = forecaster == DEFAULT_FORECASTER
    t3_file = args.t3_file or default_t3_file(mode, forecaster)
    out_dir = args.output_dir or default_output_dir(mode, forecaster)
    run_dir = args.run_dir or FORECASTER_RUN_DIRS[forecaster]
    cost_file = run_dir / "summary_all_seeds.csv"
    print(f"Forecaster: {forecaster}")
    print(f"Buffer mode: {mode}")
    print(f"Using full-stream T3 metrics: {t3_file}")
    print(f"Using unchanged retraining costs: {cost_file}")
    seed_level = load_seed_level(t3_file, cost_file)
    stats = aggregate(seed_level)

    def label(frame):
        """Non-default forecasters carry their name as the first column."""
        if legacy:
            return frame
        frame = frame.copy()
        frame.insert(0, "forecaster", forecaster)
        return frame

    out_dir.mkdir(parents=True, exist_ok=True)
    label(seed_level).to_csv(out_dir / f"t4_seed_level_metrics_{mode}.csv", index=False)

    note = (
        f"Regime convention: `{mode}` ({BUFFER_NOTES[mode]}). Mean ± sample SD over "
        "5 seeds; MAE gain vs arm A = MAE_drift(arm) − MAE_drift(A), paired on "
        "region and seed (negative = better); gain per retrain = gain / retrains "
        "(computed per seed, then averaged; — when there are no retrains)."
    )
    if not legacy:
        note += f" Forecaster: `{forecaster}`."
        if seed_invariant(seed_level):
            note += (" Deterministic forecaster: all five seeds give identical results, "
                     "so sd = 0.0 reflects no seed variation, not independent replicates.")
        note += " " + FORECASTER_NOTES.get(forecaster, "")
    full = format_table(stats, None)
    headline = format_table(stats, HEADLINE)
    for name, table, title in (
        ("t4_all_configurations", full, "T4 — adaptation cost, all 17 configurations"),
        ("t4_representative_per_arm", headline, "T4 — adaptation cost, representative configuration per arm"),
    ):
        table = label(table)
        table.to_csv(out_dir / f"{name}_{mode}.csv", index=False)
        (out_dir / f"{name}_{mode}.md").write_text(
            to_markdown(table, f"{title} ({mode})", note), encoding="utf-8")

    print(headline.to_string(index=False))
    print(f"\nWrote T4 outputs to {out_dir}")


if __name__ == "__main__":
    main()
