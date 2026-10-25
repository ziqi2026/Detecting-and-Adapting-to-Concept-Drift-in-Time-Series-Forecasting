import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_t4_adaptation_cost",
    ROOT / "experiments/adaptation/generate_t4_adaptation_cost.py",
)
t4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t4)


def _inputs(tmp_path):
    t3 = pd.DataFrame(
        [
            # region, seed, configuration, arm, mae_drift
            ("SA1", 1, "A", "A", 100.0),
            ("SA1", 1, "B", "B", 90.0),
            ("SA1", 1, "C_kswin", "C", 95.0),
            ("SA1", 2, "A", "A", 200.0),
            ("SA1", 2, "B", "B", 170.0),
            ("SA1", 2, "C_kswin", "C", 185.0),
            ("NSW1", 1, "A", "A", 300.0),
            ("NSW1", 1, "B", "B", 280.0),
            ("NSW1", 1, "C_kswin", "C", 290.0),
            ("NSW1", 2, "A", "A", 310.0),
            ("NSW1", 2, "B", "B", 280.0),
            ("NSW1", 2, "C_kswin", "C", 300.0),
        ],
        columns=["region", "seed", "configuration", "arm", "mae_drift"],
    )
    t3["detector"] = ""
    t3["window_days"] = float("nan")
    retrains = {"A": 0, "B": 10, "C_kswin": 5}
    cost = t3[["configuration", "region", "seed"]].copy()
    cost["n_retrains"] = cost["configuration"].map(retrains)
    cost["train_samples_total"] = 1000 + 100 * cost["n_retrains"]
    cost["retraining_wall_clock_s"] = 2.0 * cost["n_retrains"]
    t3_path, cost_path = tmp_path / "t3.csv", tmp_path / "summary_all_seeds.csv"
    t3.to_csv(t3_path, index=False)
    cost.to_csv(cost_path, index=False)
    return t3_path, cost_path


def test_gain_is_paired_with_arm_a_on_region_and_seed(tmp_path):
    t3_path, cost_path = _inputs(tmp_path)
    out = t4.load_seed_level(t3_path, cost_path).set_index(
        ["region", "configuration", "seed"]
    )

    # Seed 2 of SA1 must use SA1 seed 2's Arm A (200), not seed 1's (100).
    assert out.loc[("SA1", "B", 1), "mae_gain_vs_arm_a"] == pytest.approx(-10.0)
    assert out.loc[("SA1", "B", 2), "mae_gain_vs_arm_a"] == pytest.approx(-30.0)
    # NSW1 is paired with NSW1's Arm A, not SA1's.
    assert out.loc[("NSW1", "B", 1), "mae_gain_vs_arm_a"] == pytest.approx(-20.0)
    assert out.loc[("NSW1", "C_kswin", 2), "mae_gain_vs_arm_a"] == pytest.approx(-10.0)
    # gain per retrain = gain / retrains, per seed.
    assert out.loc[("SA1", "B", 2), "gain_per_retrain"] == pytest.approx(-3.0)
    assert out.loc[("SA1", "C_kswin", 2), "gain_per_retrain"] == pytest.approx(-3.0)
    # Arm A: zero gain, undefined gain per retrain.
    assert out.loc[("SA1", "A", 1), "mae_gain_vs_arm_a"] == 0.0
    assert math.isnan(out.loc[("SA1", "A", 1), "gain_per_retrain"])


def test_aggregate_reports_mean_and_sample_std_across_seeds(tmp_path):
    t3_path, cost_path = _inputs(tmp_path)
    stats = t4.aggregate(t4.load_seed_level(t3_path, cost_path))
    row = stats[(stats.region == "SA1") & (stats.configuration == "B")].iloc[0]

    assert row["mae_gain_vs_arm_a_mean"] == pytest.approx(-20.0)
    assert row["mae_gain_vs_arm_a_std"] == pytest.approx(math.sqrt(200.0))  # ddof=1
    assert row["gain_per_retrain_mean"] == pytest.approx(-2.0)
    assert row["seeds"] == 2

    table = t4.format_table(stats, {"A": "A", "B": "B"})
    sa1_b = table[(table.region == "SA1") & (table.arm == "B")].iloc[0]
    assert sa1_b["MAE gain vs arm A"] == "-20.0 ± 14.1"
    assert sa1_b["gain per retrain"] == "-2.00 ± 1.41"
    sa1_a = table[(table.region == "SA1") & (table.arm == "A")].iloc[0]
    assert sa1_a["gain per retrain"] == "—"
