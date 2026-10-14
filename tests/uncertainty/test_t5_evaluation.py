"""T5 arithmetic, temporal edge cases and reproducible handoff tests."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_lab.evaluation import (
    calculate_empirical_coverage, calculate_mean_interval_width,
    calculate_normalised_interval_width, calculate_rolling_24h_coverage,
    calculate_worst_24h_coverage,
)

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "t5", ROOT / "experiments/uncertainty/generate_t5_uncertainty_table.py")
t5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t5)


def rolling(index, actual=None, regimes=None, policy="ending_regime"):
    n = len(index)
    return calculate_rolling_24h_coverage(
        index, np.zeros(n) if actual is None else actual, np.zeros(n), np.ones(n),
        cadence="30min", regimes=regimes, boundary_policy=policy)


def test_metrics_use_positions_and_inclusive_bounds():
    assert calculate_empirical_coverage(pd.Series([0, 1], index=[7, 8]),
        pd.Series([0, 0], index=[3, 4]), pd.Series([1, 1], index=[1, 2])) == 1
    assert calculate_mean_interval_width([0, -10], [2, -4]) == 4
    assert calculate_normalised_interval_width([0, -10], [2, -4], scale=8) == .5


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, True, [1]])
def test_bad_normalisation(scale):
    with pytest.raises(ValueError):
        calculate_normalised_interval_width([0], [1], scale=scale)


def test_exact_24h_boundary_and_independent_worst():
    index = pd.date_range("2020-03-01", periods=96, freq="30min")
    actual = np.r_[np.zeros(48), np.full(48, 2)]
    result = rolling(index, actual)
    assert result.iloc[:47].isna().all()
    assert result.iloc[47] == 1
    assert result.iloc[48] == 47 / 48  # left endpoint is excluded
    expected = [sum(0 <= x <= 1 for x in actual[k-47:k+1]) / 48
                for k in range(47, len(actual))]
    np.testing.assert_allclose(result.iloc[47:], expected)
    assert calculate_worst_24h_coverage(index, actual, np.zeros(96), np.ones(96),
        cadence="30min", boundary_policy="ending_regime") == 0


def test_incomplete_gap_and_recovery():
    index = pd.date_range("2020-03-01", periods=100, freq="30min").delete(49)
    result = rolling(index)
    assert result.iloc[48] == 1
    assert result.iloc[49:96].isna().all()
    assert result.iloc[96] == 1
    assert rolling(index[:47]).isna().all()
    assert np.isnan(calculate_worst_24h_coverage(index[:47], [0]*47, [0]*47, [1]*47,
        cadence="30min", boundary_policy="ending_regime"))


def test_irregular_grid_invalidates_windows():
    index = pd.date_range("2020-03-01", periods=100, freq="30min").to_list()
    index[49] += pd.Timedelta(minutes=1)
    result = rolling(index)
    assert result.iloc[49:97].isna().all()
    assert result.iloc[97] == 1


def test_regime_boundaries_and_disjoint_segments():
    index = pd.date_range("2020-03-01", periods=150, freq="30min")
    labels = ["pre-drift"]*50 + ["drift"]*50 + ["pre-drift"]*50
    within = rolling(index, regimes=labels, policy="within_regime")
    assert within.iloc[50:97].isna().all()
    assert within.iloc[100:147].isna().all()
    assert within.iloc[147] == 1
    assert rolling(index, regimes=labels).iloc[50] == 1


@pytest.mark.parametrize("kind", ["duplicate", "unsorted", "missing"])
def test_bad_timestamps(kind):
    index = pd.date_range("2020-03-01", periods=50, freq="30min").to_list()
    if kind == "duplicate":
        index[1] = index[0]
    elif kind == "unsorted":
        index.reverse()
    else:
        index[1] = pd.NaT
    with pytest.raises(ValueError):
        rolling(index)


def test_aware_time_converts_without_local_dst():
    index = pd.date_range("2020-10-03", periods=96, freq="30min", tz="Australia/Adelaide")
    result = rolling(index)
    assert result.index.tz.zone == "Etc/GMT-10"
    assert result.iloc[47:].eq(1).all()


def protocol(index):
    return dict(protocol_id="synthetic-test-only", regime_scope="full", regime_days=7,
        rolling_boundary="within_regime", rolling_closed="right", incomplete_windows="exclude",
        cadence="30min", granularity="region_configuration", configurations=["A"],
        test_start=str(index[0]), test_end=str(index[-1]),
        normalisation={r: dict(value=10, statistic="synthetic fixed scale",
                              source="synthetic fixture", source_sha256="0"*64)
                       for r in t5.REGIONS})


def make_inputs(tmp_path):
    index = pd.date_range("2020-03-01", "2020-06-01", freq="30min")
    run_dir = tmp_path / "inputs"
    for seed in t5.SEEDS:
        base = run_dir / f"seed_{seed}"
        uq_dir = base / "uncertainty"
        uq_dir.mkdir(parents=True)
        manifest = dict(seed=seed, run_id=f"test{seed}", split_id="synthetic")
        (base / "manifest.json").write_text(json.dumps(manifest))
        (uq_dir / "adaptive_conformal_manifest.json").write_text(json.dumps(
            dict(task="PGS242-160", seed=seed, upstream_run_id=f"test{seed}",
                 split_id="synthetic", alpha=.1)))
        summaries = []
        for region in t5.REGIONS:
            frame = pd.DataFrame(dict(SETTLEMENTDATE=index, region=region, seed=seed,
                configuration="A", y_true=0, static_lower_90=-seed, static_upper_90=seed,
                adaptive_lower_90=-2*seed, adaptive_upper_90=2*seed))
            frame.to_csv(uq_dir / f"aemo_conformal_intervals_{region}.csv.gz", index=False)
            for method in t5.METHODS:
                summaries.append(dict(region=region, seed=seed, configuration="A",
                    method=method, alpha=.1, test_size=len(index), empirical_coverage=1,
                    mean_interval_width=(2 if method=="static_conformal" else 4)*seed,
                    uq_config_hash=f"{method}{seed}", n_retrains=0, train_samples_total=123))
        pd.DataFrame(summaries).to_csv(uq_dir / "aemo_conformal_comparison_summary.csv", index=False)
    return run_dir, protocol(index)


def test_generator_end_to_end_traceability_and_sample_sd(tmp_path):
    run_dir, p = make_inputs(tmp_path)
    runs = tmp_path / "results/runs.csv"
    out = tmp_path / "t5"
    audit = t5.generate(run_dir, ROOT / "data/events.csv", out, runs, p)
    table = pd.read_csv(out / "T5_SA1_A.csv")
    assert list(table) == list(t5.T5_COLUMNS)
    row = table[(table["UQ method"] == "static_conformal") & (table.regime == "drift")].iloc[0]
    assert row["mean width"] == "6.00000000 ± 3.16227766"
    assert row.coverage == "1.00000000 ± 0.00000000"
    assert row["worst 24h coverage"] == "1.00000000 ± 0.00000000"
    assert len(pd.read_csv(runs)) == 5 * 2 * 2 * 3 * 4
    assert len(audit) == 5 * 2 * 2 * 3
    original = runs.read_bytes()
    t5.generate(run_dir, ROOT / "data/events.csv", tmp_path / "repeat", runs, p)
    assert runs.read_bytes() == original
    snapshot = pd.read_csv(out / "t5_seed_metrics.csv")
    pd.testing.assert_frame_equal(snapshot, pd.read_csv(runs))


def test_missing_stream_fails_before_writing(tmp_path):
    run_dir, p = make_inputs(tmp_path)
    (run_dir / "seed_5/uncertainty/aemo_conformal_intervals_NSW1.csv.gz").unlink()
    with pytest.raises(FileNotFoundError, match="seed_5"):
        t5.generate(run_dir, ROOT / "data/events.csv", tmp_path/"out", tmp_path/"runs.csv", p)
    assert not (tmp_path/"out").exists()
    assert not (tmp_path/"runs.csv").exists()


def test_missing_protocol_is_not_invented():
    with pytest.raises(ValueError, match="Missing explicit"):
        t5.validate_protocol({})


@pytest.mark.parametrize("fault", ["missing_timestamp", "changed_truth", "bad_summary", "wrong_seed"])
def test_corrupt_upstream_fails_without_writes(tmp_path, fault):
    run_dir, p = make_inputs(tmp_path)
    path = run_dir / "seed_5/uncertainty/aemo_conformal_intervals_SA1.csv.gz"
    frame = pd.read_csv(path)
    if fault == "missing_timestamp":
        frame = frame.drop(50)
    elif fault == "changed_truth":
        frame["y_true"] = frame["y_true"].astype(float)
        frame.loc[0, "y_true"] = .5
    elif fault == "wrong_seed":
        frame.loc[0, "seed"] = 4
    else:
        frame["static_upper_90"] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        t5.generate(run_dir, ROOT / "data/events.csv", tmp_path/"out", tmp_path/"runs.csv", p)
    assert not (tmp_path/"out").exists()
    assert not (tmp_path/"runs.csv").exists()


def test_log_preserves_history_and_rejects_conflicting_rerun(tmp_path):
    runs = tmp_path / "runs.csv"
    historical = dict(group="history", method="old", dataset="synthetic", region="SA1",
        seed=1, config_hash="old", split_id="old", metric_name="mae", metric_value=4.,
        regime="drift", n_retrains=0, train_samples=1, wall_clock_s=0.)
    t5.log_run(results_path=runs, **historical)
    original = runs.read_bytes()
    current = {**historical, "group": t5.GROUP, "metric_value": .5}
    t5.append_logs_safely([current], runs)
    assert runs.read_bytes().startswith(original)
    before = runs.read_bytes()
    with pytest.raises(ValueError, match="conflicts"):
        t5.append_logs_safely([{**current, "metric_value": .6}], runs)
    assert runs.read_bytes() == before


def test_full_regime_union_and_half_open_boundaries():
    from drift_lab.evaluation import assign_full_stream_forecast_regimes
    windows = pd.DataFrame(dict(event_id=["x", "y"],
        pre_drift_start=["2020-03-02", "2020-03-03"],
        post_drift_end=["2020-03-04", "2020-03-05"]))
    stamps = ["2020-03-01", "2020-03-02", "2020-03-04", "2020-03-05"]
    assigned = assign_full_stream_forecast_regimes(stamps, windows)
    assert assigned.regime.to_list() == ["pre-drift", "drift", "drift", "post-drift"]


def test_aggregation_does_not_silently_drop_missing_seed():
    frame = pd.DataFrame([dict(method="A|static_conformal", regime="drift",
        seed=seed, metric_name=metric, metric_value=.9)
        for seed in range(1,5) for metric in t5.METRICS])
    with pytest.raises(ValueError, match="seeds"):
        t5.aggregate(frame)


def test_training_mean_excludes_calibration_and_test(tmp_path):
    source = tmp_path / "SA1.csv"
    pd.DataFrame({
        "SETTLEMENTDATE": ["2018-01-01", "2019-12-31 23:30", "2020-01-01", "2020-03-01"],
        "TOTALDEMAND": [100., 300., 9000., 9999.],
    }).to_csv(source, index=False)
    value, count = t5.training_demand_mean(source, "2018-01-01", "2020-01-01")
    assert value == 200. and count == 2


def test_aemo_requires_window_support_threshold():
    p = protocol(pd.date_range("2020-03-01", periods=60, freq="30min"))
    p["protocol_id"] = "aemo-t5-training-mean-v1"
    with pytest.raises(ValueError, match="minimum_complete"):
        t5.validate_protocol(p)


def test_too_few_complete_windows_masks_only_worst_metric():
    index = pd.date_range("2020-03-01", periods=49, freq="30min")
    p = protocol(index)
    p["minimum_complete_24h_windows"] = 3
    frame = pd.DataFrame(dict(SETTLEMENTDATE=index, y_true=0., lower=-1., upper=1.))
    row = t5.evaluate_stream(frame, ["drift"] * len(index), scale=10, protocol=p)[1]
    assert row["n_eligible_24h_windows"] == 2
    assert row["worst_window_status"] == "insufficient_windows"
    assert np.isnan(row["worst_24h_coverage"])
    assert row["empirical_coverage"] == 1
    p["minimum_complete_24h_windows"] = 2
    row = t5.evaluate_stream(frame, ["drift"] * len(index), scale=10, protocol=p)[1]
    assert row["worst_24h_coverage"] == 1
    assert row["worst_window_status"] == "sufficient"


def test_aemo_scales_must_match_regional_training_means(tmp_path):
    p = protocol(pd.date_range("2020-03-01", periods=60, freq="30min"))
    p.update(protocol_id="aemo-t5-training-mean-v1", minimum_complete_24h_windows=5,
             training_window={"start": "2018-01-01", "end_exclusive": "2020-01-01"})
    for region, value in [("SA1", 100.), ("NSW1", 500.)]:
        source = tmp_path / f"{region}.csv"
        pd.DataFrame(dict(SETTLEMENTDATE=["2018-01-01", "2020-01-01"],
                          TOTALDEMAND=[value, 99999.])).to_csv(source, index=False)
        p["normalisation"][region] = dict(value=value, statistic="training_mean_actual_demand",
            source=str(source), source_sha256=t5.digest(source))
    t5.verify_aemo_normalisation(p)
    p["normalisation"]["NSW1"]["value"] = 100.
    with pytest.raises(ValueError, match="TRAIN mean"):
        t5.verify_aemo_normalisation(p)


def test_caption_and_window_counts_are_published(tmp_path):
    inputs, p = make_inputs(tmp_path)
    t5.generate(inputs, ROOT / "data/events.csv", tmp_path / "out", tmp_path / "runs.csv", p)
    report = (tmp_path / "out/T5_SA1_A.md").read_text()
    assert "denominator: 10" in report
    assert "Complete 24-hour windows" in report
    assert "| drift | 1 |" in report
    audit = pd.read_csv(tmp_path / "out/t5_window_audit.csv")
    assert set(audit.worst_window_status) == {"sufficient"}


def test_aemo_denominators_are_derived_without_manual_values(tmp_path):
    sources = {}
    for region, demand in [("SA1", 100.), ("NSW1", 500.)]:
        path = tmp_path / f"{region}.csv"
        pd.DataFrame(dict(SETTLEMENTDATE=["2018-01-01", "2019-01-01", "2020-01-01"],
                          TOTALDEMAND=[demand, demand * 3, 99999.])).to_csv(path, index=False)
        sources[region] = {"source": str(path), "value": -999}
    p = dict(protocol_id="aemo-t5-training-mean-v1", normalisation=sources)
    resolved = t5.resolve_aemo_normalisation(p)
    assert resolved["normalisation"]["SA1"]["value"] == 200.
    assert resolved["normalisation"]["NSW1"]["value"] == 1000.
    assert resolved["normalisation"]["SA1"]["n_training_observations"] == 2
    assert p["normalisation"]["SA1"]["value"] == -999  # caller's input stays untouched
    t5.verify_aemo_normalisation(resolved)


def test_monthly_training_sources_and_hash_are_cross_platform(tmp_path):
    for year in [2018, 2019]:
        for month in range(1, 13):
            path = tmp_path / f"PRICE_AND_DEMAND_{year}{month:02d}_SA1.csv"
            path.write_text(f"SETTLEMENTDATE,TOTALDEMAND\n{year}-{month:02d}-01,100\n")
    value, count = t5.training_demand_mean(tmp_path, "2018-01-01", "2020-01-01")
    assert value == 100. and count == 24
    before = t5.training_source_digest(tmp_path)
    for path in tmp_path.glob("*.csv"):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert t5.training_source_digest(tmp_path) == before
    next(tmp_path.glob("*.csv")).unlink()
    with pytest.raises(ValueError, match="24 monthly"):
        t5.training_source_files(tmp_path)
