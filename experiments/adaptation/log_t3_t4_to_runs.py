"""
PGS242-169 - Record the seed-level T3 / T4 numbers in the shared results log.

results/runs.csv is the project's machine-readable results file (schema in
drift_lab.run_logger). This script appends the *seed-level* numbers behind the
published T3 and T4 tables, one group per table and regime convention:

    t3_forecast_accuracy_<mode>   mae (regime pre-drift | drift | post-drift)
                                  interval_pinball_loss (regime full-test)
    t4_adaptation_cost_<mode>     n_retrains, train_samples_total,
                                  retraining_wall_clock_s (regime full-test)
                                  mae_gain_vs_arm_a, gain_per_retrain
                                  (regime drift)

with <mode> in {post_only, symmetric}. After writing, the tables are
recomputed from the log and compared with the published CSVs, so a mismatch
between runs.csv and the tables fails loudly.

Nothing is re-run: inputs are the committed seed-level T3/T4 files and the
five-seed summary (for config_hash).

Run from the repository root (after generate_t3 / generate_t4 for each mode):
    python experiments/adaptation/log_t3_t4_to_runs.py
Re-running requires --replace, which rewrites only these four groups.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drift_lab.run_logger import RUNS_COLUMNS  # noqa: E402

HERE = Path(__file__).resolve().parent
BUFFER_MODES = ("post_only", "symmetric")
DATASET = "AEMO_TOTALDEMAND"
SPLIT_ID = "aemo_train2018_2019_cal202001_202002_test202003_pr29"
DEFAULT_RUNS = ROOT / "results/runs.csv"
DEFAULT_RUN_DIR = ROOT / "results/tables/adaptation/revised/pgs242_128_five_seed_final"
T3_ROOT = ROOT / "results/tables/adaptation/t3"
T4_ROOT = ROOT / "results/tables/adaptation/t4"
REGIMES = ("pre-drift", "drift", "post-drift")


def _load_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def groups_for(mode):
    return f"t3_forecast_accuracy_{mode}", f"t4_adaptation_cost_{mode}"


def build_rows(mode, run_dir):
    """Long-format rows (RUNS_COLUMNS) for one buffer mode."""
    t3 = pd.read_csv(T3_ROOT / mode / f"t3_seed_metrics_full_{mode}.csv")
    t4 = pd.read_csv(T4_ROOT / mode / f"t4_seed_level_metrics_{mode}.csv")
    summary = pd.read_csv(run_dir / "summary_all_seeds.csv")
    key = ["configuration", "region", "seed"]
    meta = summary[[*key, "config_hash", "n_retrains", "train_samples_total",
                    "arm_wall_clock_s"]]
    t3_group, t4_group = groups_for(mode)
    split_id = f"{SPLIT_ID}|buffer_mode={mode}"
    stamp = datetime.now(timezone.utc).isoformat()

    def frame(data, group, metric, regime, column):
        data = data.merge(meta, on=key, how="left", validate="one_to_one",
                          suffixes=("", "_summary"))
        if data["config_hash"].isna().any():
            raise ValueError("Missing config_hash for some configurations")
        data = data[data[column].notna()]
        return pd.DataFrame({
            "group": group, "method": data["configuration"], "dataset": DATASET,
            "region": data["region"], "seed": data["seed"].astype(int),
            "config_hash": data["config_hash"], "split_id": split_id,
            "metric_name": metric, "metric_value": data[column].astype(float),
            "regime": regime, "n_retrains": data["n_retrains"].astype(int),
            "train_samples": data["train_samples_total"].astype(int),
            "wall_clock_s": data["arm_wall_clock_s"].astype(float),
            "timestamp": stamp,
        })

    parts = [frame(t3, t3_group, "mae", regime, f"mae_{regime}") for regime in REGIMES]
    parts.append(frame(t3, t3_group, "interval_pinball_loss", "full-test", "pinball_loss"))
    for metric in ("n_retrains", "train_samples_total", "retraining_wall_clock_s"):
        parts.append(frame(t4, t4_group, metric, "full-test", metric))
    parts.append(frame(t4, t4_group, "mae_gain_vs_arm_a", "drift", "mae_gain_vs_arm_a"))
    parts.append(frame(t4, t4_group, "gain_per_retrain", "drift", "gain_per_retrain"))
    return pd.concat(parts, ignore_index=True)[RUNS_COLUMNS]


def verify(runs_path, mode):
    """Recompute the published tables from the log and compare."""
    t3_mod = _load_module("generate_t3_forecast_accuracy")
    t4_mod = _load_module("generate_t4_adaptation_cost")
    log = pd.read_csv(runs_path)
    t3_group, t4_group = groups_for(mode)
    problems = []

    # T3 (representative arms)
    t3_log = log[log.group == t3_group]
    published = pd.read_csv(T3_ROOT / mode / f"t3_full_{mode}.csv")
    for row in published.itertuples(index=False):
        config = {v: k for k, v in t3_mod.DISPLAY_ARM_LABELS.items()}[row.arm]
        config = t3_mod.PRIMARY_CONFIGURATIONS[config]
        sel = t3_log[(t3_log.method == config) & (t3_log.region == row.region)]
        expected = {
            "pre-drift": row[2], "drift": row[3], "post-drift": row[4],
        }
        for regime, text in expected.items():
            values = sel[(sel.metric_name == "mae") & (sel.regime == regime)].metric_value
            if t3_mod.format_mean_sd(values) != text or len(values) != 5:
                problems.append((t3_group, config, row.region, regime, text))
        pin = sel[sel.metric_name == "interval_pinball_loss"].metric_value
        if t3_mod.format_mean_sd(pin) != row[5]:
            problems.append((t3_group, config, row.region, "pinball", row[5]))

    # T4 (all 17 configurations)
    t4_log = log[log.group == t4_group]
    published = pd.read_csv(T4_ROOT / mode / f"t4_all_configurations_{mode}.csv")
    for row in published.itertuples(index=False):
        sel = t4_log[(t4_log.method == row.arm_label) & (t4_log.region == row.region)]
        gain = sel[sel.metric_name == "mae_gain_vs_arm_a"].metric_value
        per = sel[sel.metric_name == "gain_per_retrain"].metric_value
        gain_txt = t4_mod._pm(gain.mean(), gain.std(ddof=1))
        per_txt = t4_mod._pm(per.mean() if len(per) else np.nan,
                             per.std(ddof=1) if len(per) else np.nan, 2)
        if gain_txt != row[6] or per_txt != row[7] or len(gain) != 5:
            problems.append((t4_group, row.arm_label, row.region, gain_txt, per_txt))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--buffer-modes", nargs="+", choices=BUFFER_MODES,
                        default=list(BUFFER_MODES))
    parser.add_argument("--replace", action="store_true",
                        help="Drop existing T3/T4 groups of the selected modes first")
    args = parser.parse_args()

    groups = [g for mode in args.buffer_modes for g in groups_for(mode)]
    existing = pd.read_csv(args.runs) if args.runs.exists() else pd.DataFrame(columns=RUNS_COLUMNS)
    present = sorted(set(existing.group) & set(groups))
    if present and not args.replace:
        raise SystemExit(f"Groups already in {args.runs}: {present}. Use --replace to rewrite them.")
    if present:
        existing = existing[~existing.group.isin(groups)]

    new = pd.concat([build_rows(mode, args.run_dir) for mode in args.buffer_modes],
                    ignore_index=True)
    # Existing rows are written back untouched except for the removed groups;
    # appending keeps the file append-only in the normal (no --replace) case.
    if present:
        pd.concat([existing, new], ignore_index=True)[RUNS_COLUMNS].to_csv(args.runs, index=False)
    else:
        new.to_csv(args.runs, mode="a", header=not args.runs.exists(), index=False)
    print(f"Appended {len(new):,} rows to {args.runs} ({', '.join(groups)})")

    problems = [p for mode in args.buffer_modes for p in verify(args.runs, mode)]
    if problems:
        raise SystemExit(f"runs.csv does not reproduce the tables: {problems[:5]}")
    print("Verified: T3 and T4 recomputed from runs.csv match the published tables.")


if __name__ == "__main__":
    main()
