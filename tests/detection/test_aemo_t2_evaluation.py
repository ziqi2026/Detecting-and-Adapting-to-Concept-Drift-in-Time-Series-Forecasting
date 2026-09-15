"""Protocol edge cases use artificial detections, never tune on Task 120."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from drift_lab.evaluation import (
    AEMO_REGIMES, AEMO_TIER1_COLUMNS, aemo_timestamp,
    build_aemo_event_windows, evaluate_aemo_detections, evaluate_detections, load_aemo_events,
)

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "t2_runner", ROOT / "experiments/detection/run_aemo_t2_evaluation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestAemoMatching(unittest.TestCase):
    def setUp(self):
        self.events = load_aemo_events(ROOT / "data/events.csv")
        self.protocol = json.loads((ROOT / "docs/aemo_matching_protocol.json").read_text())

    def evaluate(self, timestamps, region="SA1", events=None):
        rows = pd.DataFrame({"detected_index": range(len(timestamps)), "SETTLEMENTDATE": timestamps})
        return evaluate_aemo_detections(rows, self.events if events is None else events, region,
                                       protocol=self.protocol, stream_start="2020-03-01",
                                       stream_end_exclusive="2024-01-01 00:30")

    def test_five_events_empty_detections_and_undefined_precision(self):
        result = self.evaluate([])
        self.assertEqual(set(result["tier1_events"].event_id), set(AEMO_TIER1_COLUMNS))
        self.assertTrue((result["tier1_events"].status == "not detected").all())
        self.assertEqual(result["totals"]["unmatched"], 0)
        self.assertTrue(math.isnan(result["totals"]["precision"]))

    def test_point_boundary_and_duration_context(self):
        for timestamp, expected in [("2022-06-15", 1), ("2022-06-22", 1),
                                    ("2022-06-22 00:30", 0), ("2022-06-24", 0)]:
            with self.subTest(timestamp=timestamp):
                result = self.evaluate([timestamp])
                self.assertEqual(result["audit"].match_tier.iloc[0], expected)
        result = self.evaluate(["2022-06-15", "2022-06-22"])
        self.assertEqual(result["audit"].match_tier.tolist(), [1, 0])

    def test_range_reuses_recorded_end_plus_grace_inclusively(self):
        for timestamp, expected in [("2020-03-31 23:30", 0), ("2020-05-17 23:30", 1),
                                    ("2020-05-24", 1), ("2020-05-24 00:30", 0)]:
            with self.subTest(timestamp=timestamp):
                self.assertEqual(self.evaluate([timestamp])["audit"].match_tier.iloc[0], expected)
        result = self.evaluate(["2020-05-17 23:30", "2020-05-24"])
        self.assertEqual(result["audit"].match_tier.tolist(), [1, 0])
        delay = result["tier1_events"].set_index("event_id").loc["EVT-2020-02", "delay_days"]
        self.assertAlmostEqual(delay, 46 + 47 / 48)

    def test_tier1_priority_repeated_detections_and_tier2_fallback(self):
        result = self.evaluate(["2023-03-16 12:00", "2023-07-01", "2023-10-29 12:00", "2023-10-30"], "NSW1")
        self.assertEqual(result["audit"].match_tier.tolist(), [2, 0, 1, 2])
        self.assertEqual(result["totals"]["precision"], .75)
        self.assertIn("EVT-2023-05", result["audit"].candidate_event_ids.iloc[2])
        security = result["tier1_events"].set_index("event_id").loc["EVT-2023-14"]
        self.assertEqual(security.assigned_detections, 1)
        self.assertEqual(security.delay_days, 2.5)
        self.assertEqual(result["audit"].matched_event_id.iloc[3], "EVT-2023-05")

    def test_additional_detections_are_unmatched_when_both_events_used(self):
        result = self.evaluate(["2023-10-29 12:00", "2023-10-30", "2023-10-31"], "NSW1")
        self.assertEqual(result["audit"].match_tier.tolist(), [1, 2, 0])
        self.assertEqual(result["totals"]["unmatched"], 1)
        self.assertAlmostEqual(result["totals"]["precision"], 2 / 3)
        self.assertTrue(result["audit"].candidate_event_ids.iloc[2])

    def test_t2_matches_t1_sudden_and_gradual_rules(self):
        # Compare actual T1 output on integer steps with T2 on mapped dates.
        # Only the selected event can match in these artificial test windows.
        for kind, truth, anchor, event_id in [
            ("sudden", [1000], pd.Timestamp("2021-10-01"), "EVT-2021-10"),
            ("gradual", [1000, 1096], pd.Timestamp("2020-04-01"), "EVT-2020-02"),
        ]:
            events = self.events.copy()
            if kind == "gradual":
                events.loc[events.event_id == event_id, "end_date"] = aemo_timestamp("2020-04-03")
            endpoint = truth[-1] + 336
            for indices in ([], [999], [1000], [endpoint], [endpoint + 1], [1000, 1001, endpoint]):
                with self.subTest(kind=kind, indices=indices):
                    synthetic = evaluate_detections(indices, truth, 2000, kind)
                    timestamps = [anchor + pd.Timedelta(minutes=(i - 1000) * 30) for i in indices]
                    aemo = self.evaluate(timestamps, "NSW1", events)
                    matched = aemo["audit"][aemo["audit"].match_tier > 0]
                    self.assertEqual(len(matched), len(synthetic["matched_pairs"]))
                    self.assertEqual(aemo["totals"]["unmatched"], len(synthetic["false_alarm_indices"]))
                    if len(matched):
                        self.assertEqual(matched.matched_event_id.iloc[0], event_id)
                        self.assertEqual(matched.delay_days.iloc[0], synthetic["detection_delay"] / 48)

    def test_region_filter(self):
        self.assertEqual(self.evaluate(["2023-03-16 12:00"], "SA1")["totals"]["unmatched"], 1)
        self.assertEqual(self.evaluate(["2023-03-16 12:00"], "NSW1")["totals"]["tier2_matched"], 1)

    def test_same_tier_overlap_is_deterministic(self):
        events = self.events.copy()
        for event_id in ("EVT-2020-04", "EVT-2020-06"):
            mask = events.event_id == event_id
            events.loc[mask, ["start_date", "end_date"]] = aemo_timestamp("2020-08-29")
            events.loc[mask, "region"] = "SA1"
        result = self.evaluate(["2020-08-29 12:00"], events=events.iloc[::-1])
        self.assertEqual(result["audit"].matched_event_id.iloc[0], "EVT-2020-04")

    def test_naive_market_time_and_explicit_utc_are_equivalent(self):
        a = self.evaluate(["2021-10-01 00:30"])
        b = self.evaluate(["2021-09-30T14:30:00Z"])
        self.assertEqual(a["audit"].delay_days.iloc[0], b["audit"].delay_days.iloc[0])

    def test_regime_edges_follow_f1_full_event_duration(self):
        event = self.events.set_index("event_id").loc["EVT-2022-03"]
        w = build_aemo_event_windows(event, regime_days=7)
        self.assertEqual(w["drift"], (aemo_timestamp("2022-06-15"), aemo_timestamp("2022-06-25")))
        self.assertEqual(w["pre-drift"][1], w["drift"][0])
        self.assertEqual(w["post-drift"][0], w["drift"][1])

    def test_bad_inputs_and_changed_tiers_fail(self):
        for timestamps in (["2021-01-01", "2021-01-01"], ["2022-01-01", "2021-01-01"], [None]):
            with self.assertRaises(ValueError):
                self.evaluate(timestamps)
        for tier in (2, 1.5):
            events = self.events.copy()
            events["tier"] = events.tier.astype(float)
            events.loc[events.event_id == "EVT-2020-02", "tier"] = tier
            with self.assertRaises(ValueError):
                self.evaluate([], events=events)

    def test_incomplete_coverage_fails_instead_of_not_detected(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            evaluate_aemo_detections(pd.DataFrame(columns=["detected_index", "SETTLEMENTDATE"]),
                                     self.events, "SA1", protocol=self.protocol,
                                     stream_start="2021-01-01", stream_end_exclusive="2024-01-01")

    def test_inclusive_matching_endpoint_requires_coverage(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            evaluate_aemo_detections(pd.DataFrame(columns=["detected_index", "SETTLEMENTDATE"]),
                                     self.events, "SA1", protocol=self.protocol,
                                     stream_start="2020-03-01", stream_end_exclusive="2023-12-02")

    def test_real_logger_round_trip_and_no_duplicate_reruns(self):
        result = self.evaluate(["2020-04-02 12:00", "2021-03-14 12:00", "2023-07-01"])
        times = pd.date_range(aemo_timestamp("2020-03-01"), aemo_timestamp("2024-01-01"), freq="30min")
        records = runner.metric_records(result, self.events, times, "SA1", self.protocol)
        common = {"method": "adwin", "region": "SA1", "group": runner.GROUP,
                  "dataset": "AEMO_TOTALDEMAND", "split_id": runner.SPLIT,
                  "seed": 0, "config_hash": "test_identity", "train_samples": 35039,
                  "n_retrains": 0, "wall_clock_s": .1}
        original = runner.RUNS
        try:
            with tempfile.TemporaryDirectory() as temp:
                runner.RUNS = Path(temp) / "runs.csv"
                self.assertTrue(runner.append_metrics(records, common))
                before = runner.RUNS.read_bytes()
                self.assertFalse(runner.append_metrics(records, common))
                self.assertEqual(before, runner.RUNS.read_bytes())
                runs = pd.read_csv(runner.RUNS)
                self.assertEqual(set(runs.regime), set(AEMO_REGIMES))
                table = runner.build_t2_from_runs(runs, [{"detector": "adwin", "region": "SA1", "seed": 0, "config_hash": "test_identity"}])
                self.assertEqual(table.iloc[0]["COVID delay"], 1.5)
                self.assertEqual(table.iloc[0]["5-minute settlement delay"], "not detected")
                self.assertEqual(table.iloc[0].unmatched, 1)
                self.assertAlmostEqual(table.iloc[0].precision, 2 / 3)
        finally:
            runner.RUNS = original


if __name__ == "__main__":
    unittest.main()
