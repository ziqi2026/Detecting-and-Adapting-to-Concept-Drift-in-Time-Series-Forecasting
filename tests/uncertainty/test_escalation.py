"""PGS242-163 escalation policy: pure functions and synthetic runner tests."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_lab.evaluation import calculate_empirical_coverage
from drift_lab.uncertainty import (
    ESCALATION_QUANTILES,
    adaptive_prediction_intervals,
    calibration_tau_grid,
    calibration_width_tau_grid,
    conformal_radius,
    escalation_flags,
    escalation_sweep,
    interval_width,
    prediction_intervals,
    retained_coverage,
    retained_mae,
    review_rate,
)

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "escalation_runner", ROOT / "experiments/uncertainty/run_escalation_sweep.py")
escalation_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(escalation_runner)


def sweep_frame(scores, y_true, lower, upper, group="g"):
    return pd.DataFrame({"group": group, "score": scores, "y_true": y_true,
                         "lower": lower, "upper": upper})


# Threshold semantics and hand-computed metrics

def test_strict_threshold_retains_score_equal_to_tau():
    flags = escalation_flags([1.0, 2.0, 3.0], 2.0)
    np.testing.assert_array_equal(flags, [False, False, True])


def test_review_rate_hand_computed():
    flags = escalation_flags([5.0, 1.0, 3.0, 7.0, 2.0], 3.0)
    np.testing.assert_array_equal(flags, [True, False, False, True, False])
    assert review_rate(flags) == pytest.approx(2 / 5)


def test_retained_coverage_hand_computed():
    y_true = [1.0, 5.0, 2.0, 9.0, 3.0]
    lower = [0.0, 0.0, 0.0, 0.0, 0.0]
    upper = [2.0, 4.0, 4.0, 10.0, 2.0]
    # covered: T, F, T, T, F; step 2 is escalated.
    flags = np.array([False, False, True, False, False])
    assert retained_coverage(y_true, lower, upper, flags) == pytest.approx(2 / 4)


def test_retained_coverage_uses_inclusive_bounds():
    flags = np.zeros(2, dtype=bool)
    assert retained_coverage([0.0, 1.0], [0.0, 0.0], [1.0, 1.0], flags) == 1.0


def test_interval_width():
    np.testing.assert_allclose(interval_width([0.0, -3.0], [2.0, 1.0]), [2.0, 4.0])


# Evaluation-only MAE

def mae_example():
    """Large point errors coincide with the widest intervals."""
    y_true = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    y_pred = np.array([10.0, 11.0, 9.0, 20.0, 30.0])  # |errors| 0, 1, 1, 10, 20
    half_width = np.array([0.5, 0.5, 1.0, 4.0, 6.0])   # widths 1, 1, 2, 8, 12
    return y_true, y_pred, y_pred - half_width, y_pred + half_width


def test_escalated_mae_exceeds_retained_mae_when_errors_track_width():
    y_true, y_pred, lower, upper = mae_example()
    flags = escalation_flags(interval_width(lower, upper), 2.0)
    np.testing.assert_array_equal(flags, [False, False, False, True, True])
    assert retained_mae(y_true, y_pred, flags) == pytest.approx(2 / 3)
    assert retained_mae(y_true, y_pred, ~flags) == pytest.approx(15.0)

    frame = sweep_frame(interval_width(lower, upper), y_true, lower, upper).assign(y_pred=y_pred)
    sweep = escalation_sweep(frame, [2.0], ["group"], y_pred_col="y_pred")
    row = sweep.iloc[0]
    assert row.escalated_mae > row.retained_mae
    assert row.retained_mae == pytest.approx(2 / 3)
    assert row.escalated_mae == pytest.approx(15.0)


def test_mae_is_nan_on_empty_sets_and_matches_overall_mae():
    y_true, y_pred, lower, upper = mae_example()
    frame = sweep_frame(interval_width(lower, upper), y_true, lower, upper).assign(y_pred=y_pred)
    sweep = escalation_sweep(frame, [-np.inf, np.inf], ["group"], y_pred_col="y_pred")
    strict, loose = sweep.iloc[0], sweep.iloc[1]
    assert np.isnan(strict.retained_mae) and strict.escalated_mae == pytest.approx(6.4)
    assert np.isnan(loose.escalated_mae) and loose.retained_mae == pytest.approx(6.4)
    assert np.isnan(retained_mae(y_true, y_pred, np.ones(5, dtype=bool)))


def test_point_forecast_does_not_change_escalation():
    y_true, y_pred, lower, upper = mae_example()
    scores = interval_width(lower, upper)
    grid = calibration_tau_grid(scores)
    sweeps = [escalation_sweep(sweep_frame(scores, y_true, lower, upper).assign(y_pred=pred),
                               grid, ["group"], y_pred_col="y_pred")
              for pred in (y_pred, y_pred + 100.0)]
    decision_columns = ["tau", "n_escalated", "review_rate", "retained_coverage"]
    pd.testing.assert_frame_equal(sweeps[0][decision_columns], sweeps[1][decision_columns])
    assert not np.allclose(sweeps[0].retained_mae.dropna(), sweeps[1].retained_mae.dropna())


def test_sweep_omits_mae_columns_without_point_forecast():
    y_true, _, lower, upper = mae_example()
    sweep = escalation_sweep(sweep_frame(interval_width(lower, upper), y_true, lower, upper),
                             [1.0], ["group"])
    assert "retained_mae" not in sweep and "escalated_mae" not in sweep


# Monotonicity and extremes

def test_review_rate_non_increasing_as_tau_increases():
    rng = np.random.default_rng(163)
    scores = rng.gamma(2.0, 50.0, size=500)
    taus = np.r_[-np.inf, np.sort(rng.uniform(0.0, 400.0, size=60)), np.inf]
    rates = [review_rate(escalation_flags(scores, tau)) for tau in taus]
    assert np.all(np.diff(rates) <= 0.0)
    assert rates[0] == 1.0 and rates[-1] == 0.0


def test_sweep_review_rate_non_increasing_per_group():
    rng = np.random.default_rng(7)
    n = 300
    scores = rng.gamma(2.0, 5.0, size=n)
    frame = sweep_frame(scores, rng.normal(size=n), -scores / 2, scores / 2,
                        group=np.repeat(["a", "b", "c"], n // 3))
    sweep = escalation_sweep(frame, calibration_tau_grid(scores[:50]), ["group"])
    for _, group in sweep.groupby("group"):
        assert np.all(np.diff(group.review_rate.to_numpy()) <= 0.0)
        assert np.all(np.diff(group.tau.to_numpy()) >= 0.0)


def test_tau_extremes():
    y_true = np.array([1.0, 5.0, 2.0, 9.0, 3.0])
    lower = np.zeros(5)
    upper = np.array([2.0, 4.0, 4.0, 10.0, 2.0])
    scores = interval_width(lower, upper)

    loose = escalation_flags(scores, np.inf)
    assert review_rate(loose) == 0.0
    assert retained_coverage(y_true, lower, upper, loose) == pytest.approx(
        calculate_empirical_coverage(y_true, lower, upper))

    strict = escalation_flags(scores, -np.inf)
    assert review_rate(strict) == 1.0
    assert np.isnan(retained_coverage(y_true, lower, upper, strict))

    sweep = escalation_sweep(sweep_frame(scores, y_true, lower, upper),
                             [np.inf, -np.inf], ["group"])
    first, last = sweep.iloc[0], sweep.iloc[-1]
    assert first.tau == -np.inf and first.n_retained == 0
    assert np.isnan(first.retained_coverage) and np.isnan(first.retained_mean_width)
    assert first.escalated_coverage == pytest.approx(0.6)
    assert last.tau == np.inf and last.n_escalated == 0
    assert np.isnan(last.escalated_coverage)
    assert last.retained_coverage == pytest.approx(0.6)
    assert last.retained_mean_width == pytest.approx(scores.mean())


def test_sweep_matches_primitive_functions():
    rng = np.random.default_rng(11)
    n = 120
    lower = rng.normal(size=n)
    upper = lower + rng.gamma(2.0, 1.0, size=n)
    y_true = rng.normal(size=n)
    y_pred = (lower + upper) / 2
    scores = interval_width(lower, upper)
    grid = calibration_tau_grid(scores[:40])
    frame = sweep_frame(scores, y_true, lower, upper).assign(y_pred=y_pred)
    sweep = escalation_sweep(frame, grid, ["group"], y_pred_col="y_pred")
    assert list(sweep.tau_quantile) == list(grid.tau_quantile)
    for row in sweep.itertuples():
        flags = escalation_flags(scores, row.tau)
        assert row.review_rate == review_rate(flags)
        assert row.n_escalated + row.n_retained == row.n_total == n
        np.testing.assert_equal(row.retained_coverage,
                                retained_coverage(y_true, lower, upper, flags))
        np.testing.assert_equal(row.escalated_coverage,
                                retained_coverage(y_true, lower, upper, ~flags))
        np.testing.assert_equal(row.retained_mae, retained_mae(y_true, y_pred, flags))
        np.testing.assert_equal(row.escalated_mae, retained_mae(y_true, y_pred, ~flags))


# Tau grids

def test_calibration_tau_grid_labels_order_and_endpoints():
    grid = calibration_tau_grid(np.arange(1.0, 101.0))
    expected = ["-inf", *[f"p{5 * i}" for i in range(20)], "p99", "p100", "+inf"]
    assert list(grid.tau_quantile) == expected
    assert len(ESCALATION_QUANTILES) == 22
    assert grid.tau.iloc[0] == -np.inf and grid.tau.iloc[-1] == np.inf
    assert grid.tau.iloc[1] == 1.0 and grid.tau.iloc[-2] == 100.0
    assert grid.tau.iloc[11] == pytest.approx(np.quantile(np.arange(1.0, 101.0), 0.5))
    assert grid.tau.is_monotonic_increasing


def test_calibration_width_tau_grid():
    grid = calibration_width_tau_grid(10.0, [3.0, 0.5, 1.0])
    assert list(grid.tau_quantile) == ["-inf", "k0.5", "k1", "k3", "+inf"]
    np.testing.assert_array_equal(grid.tau, [-np.inf, 5.0, 10.0, 30.0, np.inf])
    np.testing.assert_array_equal(grid.tau_multiplier, [-np.inf, 0.5, 1.0, 3.0, np.inf])


def test_runner_grid_has_51_multipliers_and_53_taus():
    grid = calibration_width_tau_grid(100.0, escalation_runner.WIDTH_MULTIPLIERS)
    assert len(escalation_runner.WIDTH_MULTIPLIERS) == 51
    assert len(grid) == 53
    assert grid.tau_multiplier.iloc[1] == 0.5 and grid.tau_multiplier.iloc[-2] == 3.0
    assert 1.0 in set(grid.tau_multiplier)


# Static conformal collapses to {0, 1}

def test_static_constant_width_review_rate_only_zero_or_one():
    rng = np.random.default_rng(3)
    residuals = rng.normal(0.0, 40.0, size=200)
    forecasts = rng.uniform(300.0, 9000.0, size=1000)
    lower, upper, _ = prediction_intervals(forecasts, residuals)
    decimals = escalation_runner.WIDTH_DECIMALS
    scores = np.round(interval_width(lower, upper), decimals)
    width_cal = round(2.0 * conformal_radius(residuals), decimals)
    assert np.all(scores == width_cal)

    y_true = forecasts + rng.normal(0.0, 40.0, size=1000)
    frame = sweep_frame(scores, y_true, lower, upper)
    for grid in (calibration_width_tau_grid(width_cal, escalation_runner.WIDTH_MULTIPLIERS),
                 calibration_tau_grid(np.full(50, width_cal))):
        sweep = escalation_sweep(frame, grid, ["group"])
        assert set(sweep.review_rate) == {0.0, 1.0}
        # tau == width is retained under the strict rule.
        assert sweep.loc[sweep.tau == width_cal, "review_rate"].eq(0.0).all()


# No future-data leakage

def test_flags_before_t_unchanged_when_future_truth_perturbed():
    rng = np.random.default_rng(21)
    n, t = 400, 250
    residuals = rng.normal(0.0, 10.0, size=300)
    forecasts = 1000.0 + rng.normal(0.0, 5.0, size=n)
    y_true = forecasts + rng.normal(0.0, 10.0, size=n)
    perturbed = y_true.copy()
    perturbed[t:] += rng.normal(0.0, 60.0, size=n - t)

    widths = []
    for truth in (y_true, perturbed):
        lower, upper, *_ = adaptive_prediction_intervals(forecasts, truth, residuals)
        widths.append(interval_width(lower, upper))
    assert not np.array_equal(widths[0][t + 1:], widths[1][t + 1:])

    for tau in calibration_tau_grid(2.0 * np.abs(residuals)).tau:
        original = escalation_flags(widths[0], tau)
        changed = escalation_flags(widths[1], tau)
        # The interval at t is built before y_t, so flags through t agree.
        np.testing.assert_array_equal(original[: t + 1], changed[: t + 1])


def test_calibration_tau_grid_ignores_test_truth():
    rng = np.random.default_rng(5)
    residuals = rng.normal(0.0, 10.0, size=300)
    forecasts = 500.0 + rng.normal(0.0, 5.0, size=200)
    grids = []
    for shift in (0.0, 250.0):
        y_true = forecasts + rng.normal(0.0, 10.0, size=200) + shift
        lower, upper, *_ = adaptive_prediction_intervals(forecasts, y_true, residuals)
        assert len(lower) == len(y_true)
        grids.append(calibration_tau_grid(2.0 * np.abs(residuals)))
    pd.testing.assert_frame_equal(grids[0], grids[1])


# Input validation

@pytest.mark.parametrize("call", [
    lambda: interval_width([0.0, 1.0], [1.0]),
    lambda: interval_width([2.0], [1.0]),
    lambda: interval_width([np.nan], [1.0]),
    lambda: escalation_flags([1.0, np.nan], 0.5),
    lambda: escalation_flags([1.0], np.nan),
    lambda: escalation_flags([1.0], True),
    lambda: escalation_flags([[1.0]], 0.5),
    lambda: review_rate([1, 0]),
    lambda: review_rate(np.array([], dtype=bool)),
    lambda: retained_coverage([1.0], [0.0, 0.0], [1.0, 1.0], np.array([False, False])),
    lambda: retained_coverage([1.0, 2.0], [0.0, 0.0], [1.0, 1.0], np.array([False])),
    lambda: retained_coverage([np.nan], [0.0], [1.0], np.array([False])),
    lambda: retained_coverage([1.0], [0.0], [1.0], np.array([0])),
    lambda: retained_mae([1.0, 2.0], [1.0], np.array([False, False])),
    lambda: retained_mae([1.0], [np.nan], np.array([False])),
    lambda: retained_mae([1.0], [1.0], np.array([1])),
    lambda: escalation_sweep(sweep_frame([1.0], [1.0], [0.0], [2.0]), [1.0], ["group"],
                             y_pred_col="y_pred"),
    lambda: escalation_sweep(sweep_frame([1.0], [1.0], [0.0], [2.0]).assign(y_pred=np.nan),
                             [1.0], ["group"], y_pred_col="y_pred"),
    lambda: calibration_tau_grid([]),
    lambda: calibration_tau_grid([1.0, np.nan]),
    lambda: calibration_tau_grid([1.0], [0.5, 1.5]),
    lambda: calibration_tau_grid([1.0], [0.5, 0.5]),
    lambda: calibration_width_tau_grid(0.0, [1.0]),
    lambda: calibration_width_tau_grid(1.0, [0.0]),
    lambda: calibration_width_tau_grid(1.0, [1.0, 1.0]),
    lambda: escalation_sweep(pd.DataFrame({"score": [1.0]}), [1.0], []),
    lambda: escalation_sweep(sweep_frame([1.0], [1.0], [0.0], [2.0]), [np.nan], ["group"]),
    lambda: escalation_sweep(sweep_frame([np.nan], [1.0], [0.0], [2.0]), [1.0], ["group"]),
    lambda: escalation_sweep(sweep_frame([1.0], [1.0], [3.0], [2.0]), [1.0], ["group"]),
])
def test_invalid_inputs_are_rejected(call):
    with pytest.raises(ValueError):
        call()


# Runner on synthetic PGS242-160 fixtures

# Keep the required March 1 TEST origin; extend through COVID onset on March 16.
TEST_START = "2020-03-01"
TEST_SIZE = 48 * 28


def write_fixture(run_dir, *, seed=1, region="SA1", truth_shift=0.0):
    """Write a consistent synthetic PGS242-160 seed directory."""
    rng = np.random.default_rng(seed)
    base = run_dir / f"seed_{seed}"
    (base / "uncertainty").mkdir(parents=True, exist_ok=True)
    split_id = "synthetic_split"
    (base / "manifest.json").write_text(json.dumps(
        {"seed": seed, "run_id": f"synthetic_seed{seed}", "split_id": split_id,
         "config_hash": "upstream"}))
    (base / "uncertainty/adaptive_conformal_manifest.json").write_text(json.dumps(
        {"task": "PGS242-160", "seed": seed, "alpha": 0.1, "gamma": 0.005,
         "upstream_run_id": f"synthetic_seed{seed}", "split_id": split_id}))

    residuals = rng.normal(0.0, 30.0, size=480)
    timestamps = pd.date_range(TEST_START, periods=TEST_SIZE, freq="30min")
    y_true = 1300.0 + 200.0 * np.sin(np.arange(TEST_SIZE) / 48 * 2 * np.pi)
    y_true = y_true + rng.normal(0.0, 30.0, size=TEST_SIZE)
    y_true[TEST_SIZE // 2:] += truth_shift
    radius = conformal_radius(residuals)
    frames, rows = [], []
    for configuration, bias in (("A", 0.0), ("B", 15.0)):
        forecast = y_true - truth_shift * (np.arange(TEST_SIZE) >= TEST_SIZE // 2)
        forecast = forecast + bias + rng.normal(0.0, 25.0, size=TEST_SIZE)
        static_lower, static_upper, _ = prediction_intervals(forecast, residuals)
        adaptive_lower, adaptive_upper, *_ = adaptive_prediction_intervals(
            forecast, y_true, residuals)
        frames.append(pd.DataFrame({
            "SETTLEMENTDATE": timestamps, "region": region, "seed": seed,
            "configuration": configuration, "y_true": y_true, "point_forecast": forecast,
            "static_lower_90": static_lower, "static_upper_90": static_upper,
            "adaptive_lower_90": adaptive_lower, "adaptive_upper_90": adaptive_upper}))
        for method, lower, upper in (("static_conformal", static_lower, static_upper),
                                     ("adaptive_conformal", adaptive_lower, adaptive_upper)):
            rows.append({
                "region": region, "seed": seed, "configuration": configuration,
                "method": method, "alpha": 0.1, "gamma": 0.005,
                "calibration_size": len(residuals),
                "conformal_radius": radius if method == "static_conformal" else np.nan,
                "test_size": TEST_SIZE,
                "empirical_coverage": calculate_empirical_coverage(y_true, lower, upper),
                "mean_interval_width": float(np.mean(upper - lower)),
                "n_retrains": 0, "train_samples_total": 1000, "wall_clock_s": 0.0,
                "uq_config_hash": f"{configuration}-{method}"})
    pd.concat(frames).to_csv(
        base / f"uncertainty/aemo_conformal_intervals_{region}.csv.gz", index=False,
        compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(rows).to_csv(base / "uncertainty/aemo_conformal_comparison_summary.csv",
                              index=False)
    return radius


def run_fixture(tmp_path, run_dir, name, **kwargs):
    options = dict(run_dir=run_dir, events_path=ROOT / "data/events.csv",
                   output_dir=tmp_path / name, runs_path=tmp_path / "runs.csv",
                   t5_manifest=tmp_path / "no_t5_manifest.json", seeds=(1,))
    options.update(kwargs)
    return escalation_runner.run_sweep(**options)


def test_runner_fixture_crosses_event_start_without_pre_event_buffer():
    timestamps = pd.date_range(TEST_START, periods=TEST_SIZE, freq="30min")
    labels = escalation_runner.assign_regimes(timestamps, ROOT / "data/events.csv", "SA1")
    assert list(labels[:48 * 15]) == ["pre-drift"] * (48 * 15)
    assert timestamps[48 * 15] == pd.Timestamp("2020-03-16")
    assert list(labels[48 * 15:]) == ["drift"] * (48 * 13)


def test_runner_end_to_end_schema_and_static_collapse(tmp_path):
    run_dir = tmp_path / "run"
    radius = write_fixture(run_dir)
    sweep, by_regime, skipped = run_fixture(tmp_path, run_dir, "out")

    assert list(sweep.columns) == list(escalation_runner.SWEEP_COLUMNS)
    assert set(sweep.regime) == {"pre-drift", "drift", "all"}
    assert set(sweep.configuration) == {"A", "B"}
    assert len(sweep) == 2 * 2 * 3 * 53
    assert [(s["seed"], s["region"]) for s in skipped] == [(1, "NSW1")]
    assert skipped[0]["reason"].startswith("missing ")
    assert skipped[0]["reason"].endswith("aemo_conformal_intervals_NSW1.csv.gz")
    assert sweep.calibration_width.eq(round(2 * radius, 6)).all()
    assert set(sweep[sweep.uq_method == "static_conformal"].review_rate) == {0.0, 1.0}
    all_rows = sweep[sweep.regime == "all"].groupby(["configuration", "uq_method"]).n_total
    assert all_rows.first().eq(TEST_SIZE).all()

    runs = pd.read_csv(tmp_path / "runs.csv")
    assert len(runs) == 2 * 2 * 3 * 6
    assert set(runs.metric_name) == set(escalation_runner.metric_names())
    assert {"retained_mae_at_review_10pct", "escalated_mae_at_review_10pct"} <= set(runs.metric_name)
    assert set(runs.group) == {"escalation_sweep"}
    assert set(runs.method) == {"A|static_conformal", "A|adaptive_conformal",
                                "B|static_conformal", "B|adaptive_conformal"}
    assert set(runs.config_hash) <= set(sweep.config_hash)
    manifest = json.loads((tmp_path / "out/escalation_manifest.json").read_text())
    assert manifest["runs"]["n_rows"] == len(runs)
    assert manifest["schema_version"] == 2
    assert manifest["evaluation_only"]["affects_escalation"] is False
    summary = (tmp_path / "out/escalation_summary.md").read_text()
    assert "Distinct static review rates over all rows: [0.0, 1.0]." in summary
    assert "NSW1: missing" in summary
    assert "Escalated/retained MAE" in summary

    # +inf keeps every step, so retained MAE is the stream's overall MAE.
    frame = pd.read_csv(run_dir / "seed_1/uncertainty/aemo_conformal_intervals_SA1.csv.gz")
    loose = sweep[np.isposinf(sweep.tau) & (sweep.regime == "all")]
    for row in loose.itertuples():
        stream = frame[frame.configuration == row.configuration]
        assert row.retained_mae == pytest.approx(
            np.abs(stream.y_true - stream.point_forecast).mean())
        assert np.isnan(row.escalated_mae)


def test_runner_operating_point_selection(tmp_path):
    run_dir = tmp_path / "run"
    write_fixture(run_dir)
    sweep, by_regime, _ = run_fixture(tmp_path, run_dir, "out", write_runs=False)
    keys = list(escalation_runner.GROUP_KEYS)
    for record in by_regime.to_dict("records"):
        group = sweep.merge(pd.DataFrame([{k: record[k] for k in keys}]), on=keys)
        distance = (group.review_rate - 0.10).abs()
        assert abs(record["review_rate_at_review_10pct"] - 0.10) == pytest.approx(distance.min())
        reached = group[group.retained_coverage >= 0.90]
        expected = reached.review_rate.min() if len(reached) else np.nan
        np.testing.assert_equal(record["review_rate_for_retained_coverage_90pct"], expected)
        static = record["uq_method"] == "static_conformal"
        assert record["n_distinct_review_rates"] == 2 or not static


def test_runner_is_deterministic_and_runs_log_idempotent(tmp_path):
    run_dir = tmp_path / "run"
    write_fixture(run_dir)
    run_fixture(tmp_path, run_dir, "first")
    first_log = (tmp_path / "runs.csv").read_bytes()
    run_fixture(tmp_path, run_dir, "second")
    assert (tmp_path / "runs.csv").read_bytes() == first_log
    for name in ("escalation_sweep.csv", "escalation_by_regime.csv",
                 "escalation_summary.md"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    first = json.loads((tmp_path / "first/escalation_manifest.json").read_text())
    second = json.loads((tmp_path / "second/escalation_manifest.json").read_text())
    assert first == second


def test_runner_appends_only_missing_metrics_on_schema_rerun(tmp_path):
    run_dir = tmp_path / "run"
    write_fixture(run_dir)
    run_fixture(tmp_path, run_dir, "first", regions=("SA1",))
    runs_path = tmp_path / "runs.csv"
    full = pd.read_csv(runs_path)
    new_metrics = {"retained_mae_at_review_10pct", "escalated_mae_at_review_10pct"}
    # Simulate a log written before the MAE metrics existed.
    lines = runs_path.read_text().splitlines(keepends=True)
    keep = [lines[0]] + [line for line, metric in zip(lines[1:], full.metric_name)
                         if metric not in new_metrics]
    runs_path.write_text("".join(keep))
    older = runs_path.read_bytes()

    run_fixture(tmp_path, run_dir, "second", regions=("SA1",))
    rerun = pd.read_csv(runs_path)
    assert runs_path.read_bytes().startswith(older)
    assert len(rerun) == len(full)
    keys = ["method", "region", "seed", "config_hash", "metric_name", "regime"]
    assert not rerun.duplicated(keys).any()
    added = rerun.iloc[len(keep) - 1:]
    assert set(added.metric_name) == new_metrics


def test_runner_preserves_existing_runs_rows(tmp_path):
    run_dir = tmp_path / "run"
    write_fixture(run_dir)
    from drift_lab.run_logger import log_run
    log_run("A|static_conformal", "AEMO_TOTALDEMAND", "SA1", "empirical_coverage", 0.9,
            "pre-drift", 1000, group="t5_uncertainty", seed=1,
            results_path=tmp_path / "runs.csv")
    history = (tmp_path / "runs.csv").read_bytes()
    run_fixture(tmp_path, run_dir, "out")
    assert (tmp_path / "runs.csv").read_bytes().startswith(history)


def test_runner_tau_grid_unchanged_when_test_truth_perturbed(tmp_path):
    outputs = []
    for name, shift in (("base", 0.0), ("shifted", 400.0)):
        run_dir = tmp_path / name
        write_fixture(run_dir, truth_shift=shift)
        sweep, _, _ = run_fixture(tmp_path, run_dir, f"out_{name}", write_runs=False)
        outputs.append(sweep)
    tau_columns = ["region", "configuration", "regime", "uq_method", "seed",
                   "tau", "tau_quantile", "tau_multiplier", "calibration_width"]
    pd.testing.assert_frame_equal(outputs[0][tau_columns], outputs[1][tau_columns])
    assert not np.allclose(outputs[0].review_rate, outputs[1].review_rate)


def test_runner_skips_streams_with_unexpected_sha256(tmp_path):
    run_dir = tmp_path / "run"
    write_fixture(run_dir)
    manifest = tmp_path / "t5_manifest.json"
    manifest.write_text(json.dumps({"sources": [
        {"seed": 1, "region": "SA1", "sha256": "0" * 64}]}))
    with pytest.raises(SystemExit, match="sha256 differs from the T5 manifest"):
        run_fixture(tmp_path, run_dir, "out", t5_manifest=manifest)
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "runs.csv").exists()


def test_runner_verifies_matching_sha256(tmp_path):
    run_dir = tmp_path / "run"
    write_fixture(run_dir)
    path = run_dir / "seed_1/uncertainty/aemo_conformal_intervals_SA1.csv.gz"
    manifest = tmp_path / "t5_manifest.json"
    manifest.write_text(json.dumps({"sources": [
        {"seed": 1, "region": "SA1",
         "sha256": escalation_runner.calculate_file_sha256(path)}]}))
    run_fixture(tmp_path, run_dir, "out", t5_manifest=manifest, regions=("SA1",),
                write_runs=False)
    written = json.loads((tmp_path / "out/escalation_manifest.json").read_text())
    assert written["inputs"][0]["verified_against_t5_manifest"] is True
    assert written["skipped"] == []


def test_runner_rejects_summary_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    write_fixture(run_dir)
    path = run_dir / "seed_1/uncertainty/aemo_conformal_comparison_summary.csv"
    summary = pd.read_csv(path)
    summary.loc[0, "empirical_coverage"] += 0.01
    summary.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagrees with the PGS242-160 summary"):
        run_fixture(tmp_path, run_dir, "out")
    assert not (tmp_path / "runs.csv").exists()
