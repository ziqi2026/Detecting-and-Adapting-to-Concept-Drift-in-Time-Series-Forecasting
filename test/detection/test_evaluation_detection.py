import math

from evaluation import evaluate_detections


def check(name, result, expected_delay, expected_fa, expected_missed):
    print(f"\n{name}")
    print(result)

    if expected_delay is None:
        assert math.isnan(result["detection_delay"])
    else:
        assert result["detection_delay"] == expected_delay

    assert result["false_alarms_per_10000"] == expected_fa
    assert result["missed_detections"] == expected_missed
    print("PASS")


result = evaluate_detections([], [], 20000, "none")
check("No drift - no alarms", result, None, 0.0, 0)

result = evaluate_detections([500, 15000], [], 20000, "none")
check("No drift - two false alarms", result, None, 1.0, 0)

result = evaluate_detections([10042], [10000], 20000, "sudden")
check("Sudden drift - correct detection", result, 42.0, 0.0, 0)

result = evaluate_detections([500, 10042, 15000], [10000], 20000, "sudden")
check("Sudden drift - detection plus false alarms", result, 42.0, 1.0, 0)

result = evaluate_detections([10400], [10000], 20000, "sudden")
check("Sudden drift - detection outside tolerance", result, None, 0.5, 1)

result = evaluate_detections([9050], [9000, 11000], 20000, "gradual")
check("Gradual drift - detected inside drift window", result, 50.0, 0.0, 0)

result = evaluate_detections([7020, 13050], [7000, 13000], 20000, "recurring")
check("Recurring drift - both events detected", result, 35.0, 0.0, 0)

print("\n====================================")
print("ALL DETECTION EVALUATION TESTS PASS")
print("====================================")
