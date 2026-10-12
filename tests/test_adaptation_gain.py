import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drift_lab.evaluation import calculate_adaptation_gain  # noqa: E402


def test_supervisor_worked_example():
    # requiredoutputs_Drift.pdf, T4: arm D, 2 retrains, gain -34.0, per retrain -17.0
    result = calculate_adaptation_gain(54.3, 88.3, 2)
    assert math.isclose(result["drift_mae_difference_vs_arm_a"], -34.0, abs_tol=1e-9)
    assert math.isclose(result["drift_mae_difference_per_retrain"], -17.0, abs_tol=1e-9)


def test_zero_retrains_gives_nan_per_retrain():
    result = calculate_adaptation_gain(165.5, 165.5, 0)
    assert result["drift_mae_difference_vs_arm_a"] == 0.0
    assert math.isnan(result["drift_mae_difference_per_retrain"])
