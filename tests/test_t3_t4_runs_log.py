import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/runs.csv"
SPEC = importlib.util.spec_from_file_location(
    "log_t3_t4_to_runs", ROOT / "experiments/adaptation/log_t3_t4_to_runs.py"
)
logger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(logger)


def _has_groups(mode):
    if not RUNS.exists():
        return False
    groups = set(pd.read_csv(RUNS, usecols=["group"])["group"])
    return set(logger.groups_for(mode)) <= groups


@pytest.mark.parametrize("mode", logger.BUFFER_MODES)
def test_runs_csv_reproduces_published_t3_and_t4(mode):
    if not _has_groups(mode):
        pytest.skip(f"T3/T4 {mode} rows not in results/runs.csv")
    assert logger.verify(RUNS, mode) == []


def test_verification_detects_a_changed_value(tmp_path):
    if not _has_groups("post_only"):
        pytest.skip("T3/T4 post_only rows not in results/runs.csv")
    log = pd.read_csv(RUNS)
    row = log[
        (log.group == "t3_forecast_accuracy_post_only")
        & (log.metric_name == "mae")
        & (log.regime == "drift")
    ].index[0]
    log.loc[row, "metric_value"] += 10.0
    corrupted = tmp_path / "runs.csv"
    log.to_csv(corrupted, index=False)

    assert logger.verify(corrupted, "post_only") != []
