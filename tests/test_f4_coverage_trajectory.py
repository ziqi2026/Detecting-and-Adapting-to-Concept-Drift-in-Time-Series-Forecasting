"""Focused F4 tests: trailing windows, boundary coverage, and seed alignment."""
import importlib.util
from pathlib import Path
import unittest
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("f4", ROOT / "experiments/uncertainty/generate_f4_coverage_trajectory.py")
f4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f4)


def stream(actual):
    frame = pd.DataFrame({"SETTLEMENTDATE": pd.date_range("2020-03-01", periods=len(actual), freq="30min"),
                          "y_true": actual})
    for method in f4.METHODS:
        frame[f"{method}_lower"] = 0.
        frame[f"{method}_upper"] = 1.
        frame[f"{method}_width"] = 1.
        frame[f"{method}_covered"] = frame.y_true.between(0, 1)
    return frame


class TestF4(unittest.TestCase):
    def test_trailing_boundaries_and_future_change(self):
        frame = stream([0., 1., 2., 0., 2., 1.])
        _, rolling, metrics = f4.check_stream(frame, 3, 48)
        np.testing.assert_allclose(rolling.static_coverage, [np.nan, np.nan, 2/3, 2/3, 1/3, 2/3], equal_nan=True)
        self.assertAlmostEqual(metrics["static"]["empirical_coverage"], 4/6)
        changed = frame.copy()
        changed.loc[5, "y_true"] = 100.
        for method in f4.METHODS:
            changed.loc[5, f"{method}_covered"] = False
        _, after, _ = f4.check_stream(changed, 3, 48)
        pd.testing.assert_series_equal(rolling.static_coverage.iloc[:5], after.static_coverage.iloc[:5])

    def test_short_stream_and_irregular_spacing(self):
        with self.assertRaises(ValueError):
            f4.check_stream(stream([0, 1]), 3, 48)
        frame = stream([0, 1, 0, 1])
        frame.loc[3, "SETTLEMENTDATE"] += pd.Timedelta(minutes=30)
        with self.assertRaises(ValueError):
            f4.check_stream(frame, 3, 48)

    def test_bad_flag_is_rejected(self):
        frame = stream([0, 1, 2])
        frame.loc[0, "static_covered"] = False
        with self.assertRaises(ValueError):
            f4.check_stream(frame, 3, 48)

    def test_stored_width_mismatch_is_rejected(self):
        for method in f4.METHODS:
            with self.subTest(method=method):
                frame = stream([0., 1., 2.])
                frame.loc[0, f"{method}_width"] = 2.
                with self.assertRaisesRegex(AssertionError, "Stored width mismatch"):
                    f4.check_stream(frame, 3, 48)

    def test_sample_sd_and_seed_alignment(self):
        _, one, _ = f4.check_stream(stream([0, 0, 0, 0]), 3, 48)
        _, two, _ = f4.check_stream(stream([2, 2, 2, 2]), 3, 48)
        result = f4.aggregate_trajectories([one, two], [1, 2])
        self.assertAlmostEqual(result.static_mean.iloc[-1], .5)
        self.assertAlmostEqual(result.static_sd.iloc[-1], np.sqrt(.5))
        two.loc[0, "SETTLEMENTDATE"] += pd.Timedelta(minutes=30)
        with self.assertRaises(ValueError):
            f4.aggregate_trajectories([one, two], [1, 2])


class TestT5Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ.get("F4_T5_REVIEW_DIR", ROOT / "T5_PR38_Review"))
        if not root.exists():
            raise unittest.SkipTest("Extract T5_PR38_Review.zip to run the T5 integration tests.")
        cls.ev, cls.protocol, _, cls.stored, cls.audit, cls.base = f4.load_t5_reference(root)

    def test_full_stream_assigns_every_timestamp_and_keeps_buffer(self):
        timestamps = pd.date_range("2020-03-01", periods=400, freq="30min")
        windows = pd.DataFrame({"event_id": ["test"], "pre_drift_start": [timestamps[100]],
                                "post_drift_end": [timestamps[200]]})
        labels = self.ev.assign_full_stream_forecast_regimes(timestamps, windows)
        self.assertEqual(labels.regime.iloc[99], "pre-drift")
        self.assertEqual(labels.regime.iloc[100], "drift")
        self.assertEqual(labels.regime.iloc[199], "drift")
        self.assertEqual(labels.regime.iloc[200], "post-drift")
        self.assertNotIn("unassigned", labels.regime.tolist())

    def test_24h_boundary_support_and_future_invariance(self):
        frame = stream([0.] * 120)
        labels = pd.DataFrame({"regime": ["pre-drift"] * 60 + ["drift"] * 60})
        protocol = dict(self.protocol, minimum_complete_24h_windows=10,
                        normalisation={"SA1": {"value": 2.}})
        rows = f4.t5_regime_metrics(frame, labels, "static", self.ev, protocol, "SA1")
        self.assertEqual([r["n_eligible_24h_windows"] for r in rows], [13, 13, 0])
        self.assertEqual(rows[0]["normalised_interval_width"], .5)
        self.assertEqual(rows[0]["worst_24h_coverage"], 1.)
        self.assertTrue(np.isnan(rows[2]["worst_24h_coverage"]))
        protocol["minimum_complete_24h_windows"] = 14
        self.assertTrue(np.isnan(f4.t5_regime_metrics(frame, labels, "static", self.ev, protocol, "SA1")[0]["worst_24h_coverage"]))
        before = self.ev.calculate_rolling_24h_coverage(frame.SETTLEMENTDATE, frame.y_true,
            frame.static_lower, frame.static_upper, cadence="30min", regimes=labels.regime,
            boundary_policy="within_regime")
        changed = frame.y_true.copy()
        changed.iloc[-1] = 99.
        after = self.ev.calculate_rolling_24h_coverage(frame.SETTLEMENTDATE, changed,
            frame.static_lower, frame.static_upper, cadence="30min", regimes=labels.regime,
            boundary_policy="within_regime")
        np.testing.assert_allclose(before.iloc[:-1], after.iloc[:-1], equal_nan=True)
        self.assertTrue(np.isnan(before.iloc[106]))
        self.assertEqual(before.iloc[107], 1.)

    def test_t5_metric_and_support_mismatch_rejected(self):
        selected = self.audit[(self.audit.region == "SA1") & (self.audit.seed == 1) &
                              (self.audit.method == "A|static_conformal")]
        rows = selected.to_dict("records")
        records = f4.check_t5_rows(rows, self.stored, self.audit, "SA1", 1, "A", "static")
        self.assertEqual(len(records), 12)
        rows[0]["empirical_coverage"] += .01
        with self.assertRaises(AssertionError):
            f4.check_t5_rows(rows, self.stored, self.audit, "SA1", 1, "A", "static")
        rows = selected.to_dict("records")
        rows[0]["n_eligible_24h_windows"] += 1
        with self.assertRaises(ValueError):
            f4.check_t5_rows(rows, self.stored, self.audit, "SA1", 1, "A", "static")

    def test_t5_displayed_cells_use_sample_sd_and_all_five_seeds(self):
        frame = self.audit.copy()
        frame[["configuration", "method"]] = frame.method.str.split("|", expand=True)
        records = f4.check_t5_tables(frame, self.base, [1, 2, 3, 4, 5])
        self.assertEqual(len(records), 816)
        selected = frame[(frame.region == "SA1") & (frame.configuration == "A")].copy()
        selected.loc[selected.index[0], "empirical_coverage"] += .01
        with self.assertRaises(ValueError):
            f4.check_t5_tables(selected, self.base, [1, 2, 3, 4, 5])



if __name__ == "__main__":
    unittest.main()
