"""
PGS242-130 - Generate T4: adaptation cost summary + cost-vs-accuracy view.

Reads the already-completed run outputs (no re-run needed):
  results/tables/adaptation/revised/final_updated_changepoints

Writes:
  results/tables/adaptation/T4_adaptation_cost_summary.csv
  results/figures/adaptation/T4_cost_vs_accuracy.png

Run from repo root, on branch pgs242-128:
    python experiments/adaptation/generate_t4_cost_summary.py

Terminology (confirmed against run_four_arm_adaptation.py):
  - initial_fit_s: measured once per region, before execute(); identical across
    all 17 configs in a region. A fixed shared offset, NOT an adaptation cost.
  - arm_wall_clock_s: measured only around adaptive_forecast_stream(...) with
    initial_model_fitted=True, so it EXCLUDES initial_fit_s. This equals
    runs.csv's wall_clock_s. Use this for "total per-arm cost".
  - retraining_wall_clock_s: pure retraining time. Use this for "adaptation cost".
  - overhead_s = arm_wall_clock_s - retraining_wall_clock_s: scales roughly
    linearly with n_retrains (~0.01s per retrain across both regions), so it
    looks like per-retrain bookkeeping rather than pure forecast-loop time.
    Reported for transparency, not claimed to be fully explained.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RUN_DIR = Path("results/tables/adaptation/revised/final_updated_changepoints")
OUT_TABLE_DIR = Path("results/tables/adaptation")
OUT_FIG_DIR = Path("results/figures/adaptation")
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(RUN_DIR / "summary.csv")
assert len(summary) == 34, f"expected 34 rows (17 configs x 2 regions), got {len(summary)}"

def label_row(row):
    if row["arm"] == "A":
        return "A: never"
    if row["arm"] == "B":
        return "B: every_30_days"
    if row["arm"] == "C":
        return f"C: drift_full_history [{row['detector']}]"
    if row["arm"] == "D":
        wd = int(row["window_days"])
        return f"D: drift_recent_window {wd}d [{row['detector']}]"
    return row["configuration"]

summary["config_label"] = summary.apply(label_row, axis=1)

# --- sanity check (corrected) ---
# arm_wall_clock_s excludes the shared initial_fit_s, so the only valid
# relationship is arm_wall_clock_s >= retraining_wall_clock_s.
summary["overhead_s"] = summary["arm_wall_clock_s"] - summary["retraining_wall_clock_s"]
bad = summary[summary["overhead_s"] < -1e-6]
print(f"[check] rows where arm_wall_clock_s < retraining_wall_clock_s: {len(bad)} / {len(summary)} "
      f"(should be 0)")
if len(bad):
    print(bad[["region", "config_label", "arm_wall_clock_s", "retraining_wall_clock_s"]].to_string(index=False))
print(f"[info] overhead_s range: {summary['overhead_s'].min():.3f}-{summary['overhead_s'].max():.3f}s "
      f"(forecast-loop overhead outside retraining, not the shared initial_fit_s)")

cost_cols = [
    "region", "arm", "config_label", "detector", "window_days",
    "n_retrains",
    "train_samples_initial", "train_samples_total", "train_samples_last",
    "effective_train_samples_total",
    "initial_fit_s", "retraining_wall_clock_s", "arm_wall_clock_s", "overhead_s",
    "overall_mae",
]
t4 = summary[cost_cols].copy()
t4["wall_clock_per_retrain_s"] = (
    t4["retraining_wall_clock_s"] / t4["n_retrains"].replace(0, pd.NA)
)

# --- cost relative to Arm B (always-retrain), within the same region ---
b_ref = (
    t4[t4["arm"] == "B"]
    .set_index("region")[["retraining_wall_clock_s", "train_samples_total", "overall_mae"]]
    .rename(columns={
        "retraining_wall_clock_s": "b_retraining_wall_clock_s",
        "train_samples_total": "b_train_samples_total",
        "overall_mae": "b_overall_mae",
    })
)
t4 = t4.join(b_ref, on="region")
t4["retrain_time_pct_of_B"] = 100 * t4["retraining_wall_clock_s"] / t4["b_retraining_wall_clock_s"]
t4["train_samples_pct_of_B"] = 100 * t4["train_samples_total"] / t4["b_train_samples_total"]
t4["mae_pct_of_B"] = 100 * t4["overall_mae"] / t4["b_overall_mae"]
t4 = t4.drop(columns=["b_retraining_wall_clock_s", "b_train_samples_total", "b_overall_mae"])

t4 = t4.sort_values(["region", "arm", "detector", "window_days"], na_position="first")

out_path = OUT_TABLE_DIR / "T4_adaptation_cost_summary.csv"
t4.to_csv(out_path, index=False)
print(f"\nWrote {len(t4)} rows -> {out_path}")
print(t4.to_string(index=False))

# --- cost vs accuracy figure ---
DETECTOR_COLOR = {"adwin": "tab:blue", "kswin": "tab:orange", "page_hinkley": "tab:green"}
ARM_MARKER = {"A": "*", "B": "s", "C": "o", "D": "^"}

def marker_size(row):
    if row["arm"] == "D":
        return 40 + row["window_days"] * (120 / 180)  # bigger marker = longer window
    return {"A": 140, "B": 100}.get(row["arm"], 90)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=False)
for ax, region in zip(axes, sorted(t4["region"].unique())):
    sub = t4[t4["region"] == region]
    for _, r in sub.iterrows():
        color = DETECTOR_COLOR.get(r["detector"], "tab:gray")
        ax.scatter(r["retraining_wall_clock_s"], r["overall_mae"],
                   c=color, marker=ARM_MARKER[r["arm"]], s=marker_size(r),
                   edgecolors="black", linewidths=0.6, alpha=0.9, zorder=3)
    ax.set_xlabel("Retraining wall-clock cost (s) -- single run, seed 0")
    ax.set_ylabel("Overall MAE")
    ax.set_title(region)
    ax.grid(alpha=0.3)

from matplotlib.lines import Line2D
arm_handles = [
    Line2D([0], [0], marker=ARM_MARKER[a], color="w", markerfacecolor="tab:gray",
           markeredgecolor="black", markersize=10, label=lbl)
    for a, lbl in [("A", "A: never"), ("B", "B: every_30_days"),
                   ("C", "C: drift_full_history"), ("D", "D: drift_recent_window\n(bigger = longer window)")]
]
detector_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black",
           markersize=9, label=d)
    for d, c in DETECTOR_COLOR.items()
]
fig.legend(handles=arm_handles + detector_handles, loc="lower center",
           ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("T4: Adaptation cost vs. accuracy\n(single-run timings -- see train_samples_total "
              "for a deterministic cost proxy)")
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig_path = OUT_FIG_DIR / "T4_cost_vs_accuracy.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Wrote figure -> {fig_path}")
