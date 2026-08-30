
from pathlib import Path
import pandas as pd

RUNS_PATH = Path("results/runs.csv")
OUTPUT_PATH = Path("results/T1_detection_on_synthetic.csv")
GROUP = "detection"
SPLIT_ID = "synthetic_4000_1000_15000_tol336_v1"

DRIFT_ORDER = {"none": 0, "sudden": 1, "gradual": 2, "recurring": 3}
DETECTOR_ORDER = {"adwin": 0, "kswin": 1, "page_hinkley": 2}


def mean_sd_text(values, decimals=2):
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return "—"
    mean = float(series.mean())
    sd = 0.0 if len(series) == 1 else float(series.std(ddof=1))
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def configuration_text(config_hash):
    return str(config_hash).split("|sha256:", maxsplit=1)[0]


def main():
    if not RUNS_PATH.exists():
        raise FileNotFoundError(f"Cannot find {RUNS_PATH}")

    runs = pd.read_csv(RUNS_PATH)

    required = {
        "group", "method", "dataset", "seed",
        "config_hash", "split_id", "metric_name", "metric_value",
    }
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"runs.csv is missing columns: {sorted(missing)}")

    data = runs[
        (runs["group"] == GROUP)
        & (runs["split_id"] == SPLIT_ID)
    ].copy()

    if data.empty:
        raise ValueError(
            "No matching synthetic detection results were found in runs.csv."
        )

    data["drift_type"] = data["dataset"].str.replace(
        "synthetic_", "", regex=False
    )

    rows = []

    for (method, drift_type, config_hash), group in data.groupby(
        ["method", "drift_type", "config_hash"], sort=False
    ):
        def metric_values(name):
            return group.loc[
                group["metric_name"] == name, "metric_value"
            ].astype(float)

        seeds = sorted(
            int(seed) for seed in group["seed"].dropna().unique()
        )

        rows.append(
            {
                "detector": method,
                "drift_type": drift_type,
                "detection_delay_mean_sd": mean_sd_text(
                    metric_values("detection_delay")
                ),
                "false_alarms_per_10k_mean_sd": mean_sd_text(
                    metric_values("false_alarms_per_10000")
                ),
                "missed_detections_mean_sd": mean_sd_text(
                    metric_values("missed_detections")
                ),
                "configuration": configuration_text(config_hash),
                "seeds": ",".join(map(str, seeds)),
            }
        )

    table = pd.DataFrame(rows)
    table["_detector_order"] = table["detector"].map(DETECTOR_ORDER).fillna(999)
    table["_drift_order"] = table["drift_type"].map(DRIFT_ORDER).fillna(999)

    table = (
        table.sort_values(["_detector_order", "_drift_order"])
        .drop(columns=["_detector_order", "_drift_order"])
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)

    print("T1 - Detection on Synthetic Data")
    print("=" * 110)
    print(table.to_string(index=False))
    print()
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(table)}")


if __name__ == "__main__":
    main()
