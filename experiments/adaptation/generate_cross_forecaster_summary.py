"""
PGS242-169 - Cross-forecaster summary of the four adaptation arms.

One row per forecaster x region x arm (the headline configuration of each arm,
as in T3/T4), combining

    MAE pre-drift / drift / post-drift   (T3 seed metrics, full stream)
    retrains                             (T4 seed-level metrics)
    MAE gain vs arm A                    (T4 seed-level metrics; drift regime)

for one buffer mode (default post_only). Every number is mean +/- sample sd
over seeds, formatted exactly as in T3/T4. Nothing is re-run: the inputs are
the seed-level T3/T4 files of each forecaster.

Seasonal-naive is deterministic, so its five seeds are identical and its sd is
0.0 by construction; the table caption says so whenever that is the case.

Inputs (per forecaster, <mode> = --buffer-mode):
    gradient_boosted  t3/<mode>/t3_seed_metrics_full_<mode>.csv
                      t4/<mode>/t4_seed_level_metrics_<mode>.csv
    other forecasters t3/<forecaster>/<mode>/..., t4/<forecaster>/<mode>/...

Outputs: results/tables/adaptation/cross_forecaster/
    cross_forecaster_summary_<mode>.{csv,md}

Run from the repository root:
    python experiments/adaptation/generate_cross_forecaster_summary.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BUFFER_MODES = ("post_only", "symmetric")
DEFAULT_FORECASTERS = ("gradient_boosted", "seasonal_naive")
DEFAULT_OUTPUT_DIR = ROOT / "results/tables/adaptation/cross_forecaster"
KEYS = ["configuration", "region", "seed"]
REGIMES = ("pre-drift", "drift", "post-drift")


def _load_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t3 = _load_module("generate_t3_forecast_accuracy")
t4 = _load_module("generate_t4_adaptation_cost")


def seed_level(forecaster, mode):
    """Join T3 regime MAEs with T4 retrains/gain for one forecaster."""
    t3_file = t4.default_t3_file(mode, forecaster)
    t4_file = (t4.default_output_dir(mode, forecaster)
               / f"t4_seed_level_metrics_{mode}.csv")
    for path in (t3_file, t4_file):
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed-level input for {forecaster}: {path}")
    left = pd.read_csv(t3_file)
    right = pd.read_csv(t4_file)[[*KEYS, "n_retrains", "mae_gain_vs_arm_a"]]
    merged = left.merge(right, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(f"T3 and T4 seed rows do not match for {forecaster}")
    return merged


def summarise(forecaster, mode, seeds):
    data = seed_level(forecaster, mode)
    data = data[data["configuration"].isin(t3.PRIMARY_CONFIGURATIONS.values())]
    rows = []
    for (region, arm), group in data.groupby(["region", "arm"], sort=False):
        if sorted(group["seed"].unique()) != sorted(seeds):
            raise ValueError(f"{forecaster} {region} {arm}: seeds are not {sorted(seeds)}")
        retrains = group["n_retrains"]
        if retrains.nunique() != 1:
            raise ValueError(f"{forecaster} {region} {arm}: retrains differ across seeds")
        row = {
            "forecaster": forecaster,
            "region": region,
            "arm": t3.DISPLAY_ARM_LABELS[arm],
        }
        for regime in REGIMES:
            row[f"MAE {regime}"] = t3.format_mean_sd(group[f"mae_{regime}"])
        row["retrains"] = f"{int(retrains.iloc[0])}"
        row["MAE gain vs arm A"] = t4._pm(
            group["mae_gain_vs_arm_a"].mean(), group["mae_gain_vs_arm_a"].std(ddof=1))
        row["seeds"] = group["seed"].nunique()
        row["_deterministic"] = t3.seed_invariant(group)
        row["_zero_gain"] = bool((group["mae_gain_vs_arm_a"].abs() < 1e-9).all())
        row["_order"] = "ABCD".index(arm)
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(table, title, note):
    headers = list(table.columns)
    lines = [f"# {title}", "", note, "",
             "| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for values in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--buffer-mode", choices=BUFFER_MODES, default="post_only")
    parser.add_argument("--forecasters", nargs="+", default=list(DEFAULT_FORECASTERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    mode = args.buffer_mode
    parts = [summarise(f, mode, args.seeds) for f in args.forecasters]
    full = pd.concat(parts, ignore_index=True)
    deterministic = sorted(full.loc[full["_deterministic"], "forecaster"].unique())
    no_effect = sorted(
        f for f, g in full.groupby("forecaster") if g["_zero_gain"].all()
    )
    full["_forecaster_order"] = full["forecaster"].map(
        {f: i for i, f in enumerate(args.forecasters)})
    table = full.sort_values(["region", "_forecaster_order", "_order"]).drop(
        columns=["_deterministic", "_zero_gain", "_order", "_forecaster_order"]
    ).reset_index(drop=True)

    note = (
        f"Regime convention: `{mode}` ({t3.BUFFER_MODE_NOTES[mode].rstrip('.')}). Headline "
        "configuration per arm as in T3/T4 (C = KSWIN, D = 180d window with "
        "Page-Hinkley). Mean ± sample SD over 5 seeds; MAE in MW; MAE gain vs "
        "arm A is the drift-regime MAE difference to arm A (negative = better)."
    )
    if deterministic:
        note += (
            " " + ", ".join(f"`{f}`" for f in deterministic)
            + " is deterministic: its five seeds are identical, so sd = 0.0 "
            "carries no information about variability."
        )
    if no_effect:
        note += (
            " " + ", ".join(f"`{f}`" for f in no_effect)
            + ": arms B-D have exactly the same forecasts as arm A (gain 0.0)."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / f"cross_forecaster_summary_{mode}.csv", index=False)
    (args.output_dir / f"cross_forecaster_summary_{mode}.md").write_text(
        to_markdown(table, f"Cross-forecaster adaptation summary ({mode})", note),
        encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nWrote cross-forecaster summary to {args.output_dir}")


if __name__ == "__main__":
    main()
