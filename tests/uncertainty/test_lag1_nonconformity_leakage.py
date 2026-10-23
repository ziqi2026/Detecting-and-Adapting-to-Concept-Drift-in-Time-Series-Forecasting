"""Leakage audit of the ACTUAL score expression in run_escalation_lag1_nonconformity.py.

Run from repository root:
  PYTHONPATH=src python3 -m pytest tests/uncertainty/test_lag1_nonconformity_leakage.py -v

This tests decision-score causality, not availability of real-time AEMO observations
or post-test selection of the best operating threshold.

The additional tests explicitly audit threshold-selection causality and document
that the current ~5/10/20% operating points are RETROSPECTIVE test summaries.
Passing these tests does not certify an operational, test-free threshold policy.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "uncertainty"
    / "run_escalation_lag1_nonconformity.py"
)
DECIMALS = 6


def actual_score_expression():
    """Compile the 'score = ...' expression INSIDE sweep_stream, without mocking it."""
    source = RUNNER.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(RUNNER))
    sweep = next((node for node in module.body if isinstance(node, ast.FunctionDef)
                  and node.name == "sweep_stream"), None)
    if sweep is None:
        raise AssertionError("sweep_stream function not found")
    assignments = [node for node in ast.walk(sweep) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "score"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise AssertionError(f"Expected exactly one score assignment, found {len(assignments)}")
    expr = ast.Expression(body=assignments[0].value)
    ast.fix_missing_locations(expr)
    return compile(expr, str(RUNNER), "eval")


SCORE_CODE = None


def actual_score(truth, forecast):
    global SCORE_CODE
    if SCORE_CODE is None:
        SCORE_CODE = actual_score_expression()
    truth = np.asarray(truth, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if len(truth) != len(forecast):
        raise ValueError("truth and forecast lengths differ")
    # Evaluate the code actually found in the user's runner.
    score = eval(SCORE_CODE, {"__builtins__": {}}, {
        "np": np, "truth": truth, "forecast": forecast,
        "WIDTH_DECIMALS": DECIMALS,
    })
    return np.asarray(score, dtype=float)


def actual_operating_point_selector():
    """Load the runner's unmodified operating-point selector, without its imports."""
    module = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    fn = next((node for node in module.body
               if isinstance(node, ast.FunctionDef) and node.name == "select_operating_points"), None)
    if fn is None:
        raise AssertionError("Runner must have select_operating_points")
    fn_module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(fn_module)
    namespace = {"np": np, "REVIEW_TARGETS": (0.05, 0.10, 0.20),
                 "COVERAGE_TARGET": 0.90,
                 "target_label": lambda x: f"{round(100 * x):02d}pct"}
    exec(compile(fn_module, str(RUNNER), "exec"), namespace)
    return namespace["select_operating_points"]


def synthetic_test_sweep():
    """Small completed TEST sweep with two candidate thresholds and both endpoints."""
    return pd.DataFrame({
        "tau": [-np.inf, 10.0, 20.0, np.inf],
        "review_rate": [1.00, 0.20, 0.05, 0.00],
        "n_total": [100] * 4,
        "retained_coverage": [np.nan, 0.96, 0.88, 0.85],
        "retained_mean_width": [np.nan, 100.0, 100.0, 100.0],
        "retained_mae": [np.nan, 20.0, 30.0, 35.0],
        "escalated_mae": [35.0, 90.0, 80.0, np.nan],
        "tau_quantile": ["-inf", "0.2", "0.05", "+inf"],
    })


class TestLag1NonconformityLeakage(unittest.TestCase):
    def setUp(self):
        self.forecast = np.array([100, 110, 120, 130, 140, 150, 160, 170], dtype=float)
        self.truth = np.array([105, 125, 113, 141, 135, 169, 152, 192], dtype=float)
        self.tau = 10.0

    def test_exact_previous_half_hour_error(self):
        scores = actual_score(self.truth, self.forecast)
        expected = np.array([0, 5, 15, 7, 11, 5, 19, 8], dtype=float)
        np.testing.assert_array_equal(scores, expected)
        self.assertEqual(scores[0], 0.0, "First TEST observation must have no unknown prior TEST error")

    def test_current_and_future_truth_cannot_change_past_or_current_scores(self):
        baseline = actual_score(self.truth, self.forecast)
        baseline_flags = baseline > self.tau
        for t in range(len(self.truth)):
            changed = self.truth.copy()
            changed[t:] += 1_000_000  # corrupt y[t] and all future observed demands
            scores = actual_score(changed, self.forecast)
            np.testing.assert_array_equal(scores[:t + 1], baseline[:t + 1],
                err_msg=f"LEAKAGE: current/future actual demand affected score at/before t={t}")
            np.testing.assert_array_equal((scores > self.tau)[:t + 1], baseline_flags[:t + 1],
                err_msg=f"LEAKAGE: current/future actual demand affected decisions at/before t={t}")

    def test_truth_at_t_affects_next_score_not_same_time(self):
        baseline = actual_score(self.truth, self.forecast)
        for t in range(len(self.truth) - 1):
            changed = self.truth.copy()
            changed[t] += 1000
            scores = actual_score(changed, self.forecast)
            np.testing.assert_array_equal(scores[:t + 1], baseline[:t + 1])
            self.assertNotEqual(scores[t + 1], baseline[t + 1],
                f"y[{t}] should become available for score[{t + 1}]")
            np.testing.assert_array_equal(scores[t + 2:], baseline[t + 2:],
                err_msg="Single lag-one observation unexpectedly influences later scores")

    def test_no_configuration_boundary_carryover(self):
        # The runner iterates 'for configuration, group in frame.groupby(...)'.
        # Treat every configuration independently, as the actual runner does.
        a = actual_score([100, 150, 90], [100, 100, 100])
        b = actual_score([9999, 100, 101], [100, 100, 100])
        self.assertEqual(a[0], 0.0)
        self.assertEqual(b[0], 0.0)
        self.assertEqual(b[1], 9899.0)

    def test_threshold_does_not_use_current_truth(self):
        baseline = actual_score(self.truth, self.forecast) > self.tau
        changed_truth = self.truth.copy()
        changed_truth[-1] = -1_000_000
        flags = actual_score(changed_truth, self.forecast) > self.tau
        np.testing.assert_array_equal(flags, baseline,
            err_msg="Changing final actual demand must not change any reviewed prediction")


class TestThresholdSelectionAudit(unittest.TestCase):
    """Audit the ACTUAL runner's threshold selection and its limitations."""

    def test_candidate_grid_is_calibration_anchored_not_test_truth(self):
        """Candidate tau values are derived from saved calibration width."""
        source = RUNNER.read_text(encoding="utf-8")
        module = ast.parse(source)
        sweep = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                     and n.name == "sweep_stream")
        calls = [n for n in ast.walk(sweep) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "calibration_width_tau_grid"]
        self.assertEqual(len(calls), 1, "Expected exactly one threshold-grid construction")
        grid = calls[0]
        self.assertIsInstance(grid.args[0], ast.Name)
        self.assertEqual(grid.args[0].id, "score_reference",
                         "Threshold grid must use the calibration reference, not TEST scores")
        self.assertIsInstance(grid.args[1], ast.Name)
        self.assertEqual(grid.args[1].id, "WIDTH_MULTIPLIERS")
        self.assertIn("width_cal / 2.0", source)

    def test_operating_point_selector_uses_completed_test_review_rates(self):
        """Flag that ~10% operating points are selected from completed TEST results."""
        selector = actual_operating_point_selector()
        group = synthetic_test_sweep()
        selected = selector(group)
        self.assertEqual(selected["review_rate_at_review_10pct"], 0.05)
        self.assertEqual(selected["review_rate_for_retained_coverage_90pct"], 0.20)
        # Changing the TEST review rates changes which precomputed threshold is picked.
        changed = group.copy()
        changed.loc[changed.tau == 10.0, "review_rate"] = 0.095
        selected_changed = selector(changed)
        self.assertEqual(selected_changed["review_rate_at_review_10pct"], 0.095)
        self.assertNotEqual(selected["tau_at_review_10pct"],
                            selected_changed["tau_at_review_10pct"])

    def test_coverage_target_selector_uses_completed_test_outcomes(self):
        """The 'minimum review for 90% coverage' threshold is also retrospective."""
        selector = actual_operating_point_selector()
        group = synthetic_test_sweep()
        initial = selector(group)
        self.assertEqual(initial["review_rate_for_retained_coverage_90pct"], 0.20)
        changed = group.copy()
        changed.loc[changed.tau == 20.0, "retained_coverage"] = 0.95
        revised = selector(changed)
        self.assertEqual(revised["review_rate_for_retained_coverage_90pct"], 0.05)
        self.assertNotEqual(initial["tau_for_retained_coverage_90pct"],
                            revised["tau_for_retained_coverage_90pct"])

    def test_fixed_threshold_decisions_ignore_future_truth(self):
        """Frozen tau keeps the causal score->review decision independent of future truth."""
        baseline = actual_score([10, 30, 80, 20], [10, 10, 10, 10]) > 15
        changed = actual_score([10, 30, 80, 900000], [10, 10, 10, 10]) > 15
        np.testing.assert_array_equal(baseline, changed)


if __name__ == "__main__":
    print(f"Auditing runner: {RUNNER}")
    unittest.main(verbosity=2)
