"""Synthetic detection evaluation: seven independently collected cases."""
import math
import pytest
from drift_lab.evaluation import evaluate_detections

@pytest.mark.parametrize("detections,truth,kind,delay,fa,missed", [
    ([], [], "none", None, 0.0, 0),
    ([500, 15000], [], "none", None, 1.0, 0),
    ([10042], [10000], "sudden", 42.0, 0.0, 0),
    ([500, 10042, 15000], [10000], "sudden", 42.0, 1.0, 0),
    ([10400], [10000], "sudden", None, 0.5, 1),
    ([9050], [9000, 11000], "gradual", 50.0, 0.0, 0),
    ([7020, 13050], [7000, 13000], "recurring", 35.0, 0.0, 0),
], ids=["no-alarms", "false-alarms", "sudden", "sudden-with-false-alarms",
        "outside-tolerance", "gradual", "recurring"])
def test_detection_evaluation(detections, truth, kind, delay, fa, missed):
    result = evaluate_detections(detections, truth, 20000, kind)
    if delay is None:
        assert math.isnan(result["detection_delay"])
    else:
        assert result["detection_delay"] == delay
    assert result["false_alarms_per_10000"] == fa
    assert result["missed_detections"] == missed
