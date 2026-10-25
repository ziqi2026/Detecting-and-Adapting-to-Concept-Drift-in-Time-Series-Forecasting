"""PGS242-169: forecaster argument, seasonal-naive arms and the forecaster column."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_lab.adaptation import adaptive_forecast_stream
from drift_lab.forecasters import SeasonalNaiveForecaster
from drift_lab.uncertainty import conformal_radius

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTATION = ROOT / "experiments/adaptation"
fixtures = load("four_arm_fixtures", ROOT / "tests/adaptation/test_four_arm_adaptation.py")
runner = fixtures.runner
t3 = load("t3_generator", ADAPTATION / "generate_t3_forecast_accuracy.py")
t4 = load("t4_generator", ADAPTATION / "generate_t4_adaptation_cost.py")
cross = load("cross_forecaster", ADAPTATION / "generate_cross_forecaster_summary.py")


def wavy(start, periods, offset=0.0):
    """A varying demand series, so a lag copy cannot be mistaken for a constant."""
    times = pd.date_range(start, periods=periods, freq="30min")
    demand = 1000 + 100 * np.sin(np.arange(periods) / 5.0) + offset
    return pd.DataFrame({"SETTLEMENTDATE": times, "TOTALDEMAND": demand})


# ---------------------------------------------------------------- the runner

def test_forecaster_spec_passes_the_seed_only_to_seeded_forecasters():
    spec, config = runner.forecaster_spec("gradient_boosted", 3)
    assert config == {**spec["config"], "seed": 3}  # unchanged from before PGS242-169
    spec, config = runner.forecaster_spec("seasonal_naive", 3)
    assert config == {"season": 48}  # no seed option: the seed only labels the run
    with pytest.raises(ValueError, match="forecaster must be one of"):
        runner.forecaster_spec("not_a_model", 1)


def test_forecaster_argument_defaults_to_gradient_boosted(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "run_multi_seed_adaptation_experiment",
                        lambda **kwargs: calls.append(kwargs))
    base = ["run", "--detections", "d.csv", "--decisions", "m.md", "--run-id", "x"]
    monkeypatch.setattr(sys, "argv", base)
    runner.main()
    monkeypatch.setattr(sys, "argv", base + ["--forecaster", "seasonal_naive"])
    runner.main()
    assert [call["forecaster"] for call in calls] == ["gradient_boosted", "seasonal_naive"]
    monkeypatch.setattr(sys, "argv", base + ["--forecaster", "not_a_model"])
    with pytest.raises(SystemExit):
        runner.main()


def test_seasonal_naive_fit_is_a_no_op_and_predict_reads_only_the_lag():
    train, other = wavy("2020-01-01", 20), wavy("2020-01-01", 20, offset=500)
    test = wavy("2020-02-01", 6)
    history = wavy("2020-01-31", 4)
    first = SeasonalNaiveForecaster(season=4).fit(train).predict(test, history_df=history)
    second = SeasonalNaiveForecaster(season=4).fit(other).predict(test, history_df=history)
    np.testing.assert_array_equal(first, second)  # what it was fitted on is irrelevant
    combined = pd.concat([history, test]).TOTALDEMAND.to_numpy()
    np.testing.assert_array_equal(first, combined[len(history) - 4: len(combined) - 4])


@pytest.mark.parametrize("strategy,recent", [
    ("every_30_days", "90D"), ("drift_full_history", "90D"),
    ("drift_recent_window", "7D"), ("drift_recent_window", "180D"),
])
def test_seasonal_naive_arms_reproduce_arm_a_exactly(strategy, recent):
    train = wavy("2019-12-01", 48 * 20)
    calibration = wavy(train.SETTLEMENTDATE.iloc[-1] + pd.Timedelta(minutes=30), 48 * 5)
    test = wavy(calibration.SETTLEMENTDATE.iloc[-1] + pd.Timedelta(minutes=30), 48 * 70, 7.0)
    cp = pd.DataFrame({
        "region": "SA1", "detector": "kswin", "detected_index": [500, 2000],
        "SETTLEMENTDATE": test.SETTLEMENTDATE.iloc[[500, 2000]].to_numpy(),
    })
    model = SeasonalNaiveForecaster(season=48).fit(train)
    common = dict(region="SA1", detector="kswin", initial_model_fitted=True)
    reference, _ = adaptive_forecast_stream(
        cp, model, train, calibration, test, strategy="never", **common)
    stream, meta = adaptive_forecast_stream(
        cp, model, train, calibration, test, strategy=strategy,
        recent_window=recent, **common)
    assert meta["n_retrains"] >= 2  # the schedule really ran ...
    assert meta["retraining_wall_clock_s"] < 0.1  # ... on an empty fit()
    np.testing.assert_array_equal(stream.y_pred, reference.y_pred)  # ... and changed nothing
    # The forecast is the observation one day earlier, never anything newer.
    history = pd.concat([train, calibration, test], ignore_index=True).TOTALDEMAND
    start = len(train) + len(calibration)
    np.testing.assert_array_equal(
        stream.y_pred, history.iloc[start - 48: start + len(test) - 48].to_numpy())


def test_runner_end_to_end_for_seasonal_naive(tmp_path, monkeypatch):
    train = wavy("2019-12-01", 48 * 10)
    calibration = wavy("2020-02-20", 48 * 5)
    test = wavy("2020-03-01", 48 * 8)
    changepoints = pd.concat([
        fixtures.make_frozen_detector_changepoints(test, region=region, detector=detector)
        for region in runner.REGIONS for detector in runner.SUPPORTED_DETECTORS
    ], ignore_index=True)
    detections = tmp_path / "changepoints.csv"
    changepoints.to_csv(detections, index=False)
    decisions = tmp_path / "decisions.md"
    fixtures.write_experiment_decisions(decisions)
    monkeypatch.setattr(runner, "load_and_split_aemo",
                        lambda *args: (None, train, calibration, test))
    monkeypatch.setitem(runner.MODEL_REGISTRY, "seasonal_naive", {
        "class": SeasonalNaiveForecaster, "config": {"season": 48}})
    shared = tmp_path / "runs.csv"
    summary = runner.run_multi_seed_adaptation_experiment(
        detections_path=detections, decisions_path=decisions, run_id="sn",
        seeds=(1, 2), data_path=tmp_path, output_root=tmp_path / "out",
        shared_runs=shared, forecaster="seasonal_naive")
    out = tmp_path / "out/sn"

    # Deterministic: the seeds are identical and every arm equals arm A.
    by_seed = summary.pivot_table(index=["region", "configuration"],
                                  columns="seed", values="overall_mae")
    assert by_seed[1].equals(by_seed[2])
    assert (summary.groupby("region").overall_mae.nunique() == 1).all()
    assert (summary.groupby("region").n_retrains.max() > 0).all()

    manifest = json.loads((out / "seed_1/manifest.json").read_text())
    assert manifest["model"] == "seasonal_naive"
    assert manifest["forecaster"] == "seasonal_naive"
    assert manifest["seed_affects_model"] is False
    assert manifest["model_config"] == {"season": 48}
    assert json.loads((out / "multi_seed_manifest.json").read_text())["forecaster"] == "seasonal_naive"

    # Pinball needs a radius: calibrated on the calibration frame only.
    radii = pd.read_csv(out / "seed_1/uncertainty/aemo_prediction_intervals_90_summary.csv")
    model = SeasonalNaiveForecaster(season=48).fit(train)
    expected = conformal_radius(
        calibration.TOTALDEMAND.to_numpy() - model.predict(calibration, history_df=train))
    assert set(radii.region) == set(runner.REGIONS)
    np.testing.assert_allclose(radii.conformal_radius, expected)
    assert {"alpha", "calibration_size", "empirical_coverage"} <= set(radii.columns)

    # Logged under its own group, so the gradient_boosted rows are untouched.
    assert set(pd.read_csv(shared).group) == {"adaptive_forecasting_v3_pr29_seasonal_naive"}


# --------------------------------------------------------- T3 / T4 / summary

def write_t3_run(run_dir, seeds=(1, 2), jitter=0.0):
    """Tiny saved four-arm run: 4 headline configurations, both regions."""
    times = pd.date_range("2020-03-01", periods=1600, freq="30min")
    truth = 1000 + 50 * np.sin(np.arange(len(times)) / 7.0)
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        (seed_dir / "uncertainty").mkdir(parents=True)
        for region in t3.REGIONS:
            frames = []
            for arm, configuration in t3.PRIMARY_CONFIGURATIONS.items():
                frames.append(pd.DataFrame({
                    "SETTLEMENTDATE": times, "y_true": truth,
                    "y_pred": truth + 5.0 + jitter * seed, "arm": arm,
                    "configuration": configuration, "detector": "",
                    "window_days": np.nan, "region": region, "seed": seed,
                }))
            pd.concat(frames).to_csv(seed_dir / f"forecasts_{region}.csv.gz", index=False)
        pd.DataFrame({"region": list(t3.REGIONS), "conformal_radius": [20.0, 30.0]}).to_csv(
            seed_dir / "uncertainty/aemo_prediction_intervals_90_summary.csv", index=False)


def run_t3(monkeypatch, run_dir, out_dir, *extra):
    monkeypatch.setattr(sys, "argv", [
        "t3", "--run-dir", str(run_dir), "--output-dir", str(out_dir),
        "--scopes", "full", "--seeds", "1", "2", *extra])
    t3.main()


def test_t3_adds_a_forecaster_column_only_for_non_default_forecasters(tmp_path, monkeypatch):
    write_t3_run(tmp_path / "run")
    run_t3(monkeypatch, tmp_path / "run", tmp_path / "sn", "--forecaster", "seasonal_naive")
    run_t3(monkeypatch, tmp_path / "run", tmp_path / "gb")  # default forecaster

    table = pd.read_csv(tmp_path / "sn/t3_full_post_only.csv")
    assert table.columns[0] == "forecaster"
    assert set(table.forecaster) == {"seasonal_naive"}
    seed_metrics = pd.read_csv(tmp_path / "sn/t3_seed_metrics_full_post_only.csv")
    assert set(seed_metrics.forecaster) == {"seasonal_naive"}
    assert pd.read_csv(tmp_path / "sn/regime_observation_counts_post_only.csv").columns[0] == "forecaster"
    markdown = (tmp_path / "sn/t3_full_post_only.md").read_text()
    assert "| forecaster |" in markdown
    assert "Deterministic forecaster" in markdown and "sd = 0.0" in markdown
    assert json.loads((tmp_path / "sn/t3_manifest_post_only.json").read_text())["forecaster"] == "seasonal_naive"

    legacy = pd.read_csv(tmp_path / "gb/t3_full_post_only.csv")
    assert "forecaster" not in legacy.columns
    assert list(legacy.columns) == [c for c in table.columns if c != "forecaster"]
    assert "forecaster" not in pd.read_csv(tmp_path / "gb/t3_seed_metrics_full_post_only.csv").columns
    assert "forecaster" not in json.loads((tmp_path / "gb/t3_manifest_post_only.json").read_text())
    # Same numbers either way: the column is a label, not a change of metric.
    pd.testing.assert_frame_equal(table.drop(columns="forecaster"), legacy)


def test_t3_does_not_call_seed_dependent_results_deterministic(tmp_path, monkeypatch):
    write_t3_run(tmp_path / "run", jitter=1.0)  # predictions differ by seed
    run_t3(monkeypatch, tmp_path / "run", tmp_path / "sn", "--forecaster", "seasonal_naive")
    assert "Deterministic forecaster" not in (tmp_path / "sn/t3_full_post_only.md").read_text()


def test_default_paths_keep_the_legacy_layout_and_separate_other_forecasters():
    assert t3.FORECASTER_RUN_DIRS["gradient_boosted"] == t3.DEFAULT_RUN_DIR
    assert t3.FORECASTER_RUN_DIRS["seasonal_naive"].name == "seasonal_naive_five_seed"
    legacy = t4.default_t3_file("post_only")
    assert legacy == t4.T3_ROOT / "post_only/t3_seed_metrics_full_post_only.csv"
    assert t4.default_t3_file("post_only", "seasonal_naive") == (
        t4.T3_ROOT / "seasonal_naive/post_only/t3_seed_metrics_full_post_only.csv")
    assert t4.default_output_dir("symmetric") == t4.T4_ROOT / "symmetric"
    assert t4.default_output_dir("symmetric", "seasonal_naive") == (
        t4.T4_ROOT / "seasonal_naive/symmetric")


def write_t4_inputs(tmp_path, *, varying):
    rows, cost = [], []
    for region in ("SA1", "NSW1"):
        for seed in (1, 2):
            for arm, configuration, retrains in (("A", "A", 0), ("B", "B", 4)):
                mae = 100.0 + (seed * 2.0 if varying else 0.0) - (5.0 if varying and arm == "B" else 0.0)
                rows.append(dict(arm=arm, configuration=configuration, region=region,
                                 seed=seed, detector="", window_days=np.nan, mae_drift=mae))
                cost.append(dict(configuration=configuration, region=region, seed=seed,
                                 n_retrains=retrains, train_samples_total=1000 + 10 * retrains,
                                 retraining_wall_clock_s=0.0))
    pd.DataFrame(rows).to_csv(tmp_path / "t3.csv", index=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(cost).to_csv(run_dir / "summary_all_seeds.csv", index=False)
    return tmp_path / "t3.csv", run_dir


def run_t4(monkeypatch, t3_file, run_dir, out_dir, *extra):
    monkeypatch.setattr(sys, "argv", [
        "t4", "--t3-file", str(t3_file), "--run-dir", str(run_dir),
        "--output-dir", str(out_dir), *extra])
    t4.main()


def test_t4_forecaster_column_and_caption(tmp_path, monkeypatch):
    t3_file, run_dir = write_t4_inputs(tmp_path, varying=False)
    run_t4(monkeypatch, t3_file, run_dir, tmp_path / "sn", "--forecaster", "seasonal_naive")
    run_t4(monkeypatch, t3_file, run_dir, tmp_path / "gb")

    for name in ("t4_all_configurations", "t4_representative_per_arm", "t4_seed_level_metrics"):
        frame = pd.read_csv(tmp_path / f"sn/{name}_post_only.csv")
        assert frame.columns[0] == "forecaster"
        assert set(frame.forecaster) == {"seasonal_naive"}
        assert "forecaster" not in pd.read_csv(tmp_path / f"gb/{name}_post_only.csv").columns
    markdown = (tmp_path / "sn/t4_all_configurations_post_only.md").read_text()
    assert "Deterministic forecaster" in markdown and "sd = 0.0" in markdown
    assert "`fit()` is a no-op" in markdown
    assert "Deterministic forecaster" not in (tmp_path / "gb/t4_all_configurations_post_only.md").read_text()
    row = pd.read_csv(tmp_path / "sn/t4_all_configurations_post_only.csv").query("arm == 'B'").iloc[0]
    assert row["MAE gain vs arm A"] == "0.0 ± 0.0" and row["retrains"] == 4


def test_t4_does_not_claim_determinism_when_seeds_differ(tmp_path, monkeypatch):
    t3_file, run_dir = write_t4_inputs(tmp_path, varying=True)
    run_t4(monkeypatch, t3_file, run_dir, tmp_path / "sn", "--forecaster", "seasonal_naive")
    assert "Deterministic forecaster" not in (tmp_path / "sn/t4_all_configurations_post_only.md").read_text()


def test_cross_forecaster_summary_combines_t3_t4_and_flags_determinism(tmp_path, monkeypatch):
    monkeypatch.setattr(cross.t4, "T3_ROOT", tmp_path / "t3")
    monkeypatch.setattr(cross.t4, "T4_ROOT", tmp_path / "t4")
    for forecaster, deterministic in (("gradient_boosted", False), ("seasonal_naive", True)):
        t3_rows, t4_rows = [], []
        for region in ("SA1", "NSW1"):
            for seed in (1, 2, 3, 4, 5):
                for arm, configuration in t3.PRIMARY_CONFIGURATIONS.items():
                    base = 200.0 + (0.0 if deterministic else seed)
                    effect = 0.0 if deterministic or arm == "A" else -10.0
                    key = {"arm": arm, "configuration": configuration,
                           "region": region, "seed": seed}
                    t3_rows.append({
                        **key, "mae_pre-drift": base, "mae_drift": base + effect,
                        "mae_post-drift": base + 1, "pinball_loss": 5.0})
                    t4_rows.append({
                        **key, "n_retrains": {"A": 0, "B": 46, "C": 17, "D": 5}[arm],
                        "mae_gain_vs_arm_a": effect})
        subdir = "" if forecaster == "gradient_boosted" else forecaster
        for root, rows, name in (
                ("t3", t3_rows, "t3_seed_metrics_full_post_only.csv"),
                ("t4", t4_rows, "t4_seed_level_metrics_post_only.csv")):
            target = tmp_path / root / subdir / "post_only"
            target.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(target / name, index=False)
    table = cross.summarise("seasonal_naive", "post_only", [1, 2, 3, 4, 5])
    assert len(table) == 8 and set(table.arm.str[0]) == {"A", "B", "C", "D"}
    assert table["_deterministic"].all() and table["_zero_gain"].all()
    row = table[(table.region == "SA1") & table.arm.str.startswith("D")].iloc[0]
    assert row["MAE drift"] == "200.0 ± 0.0" and row["retrains"] == "5"
    assert row["MAE gain vs arm A"] == "0.0 ± 0.0"
    assert not cross.summarise("gradient_boosted", "post_only", [1, 2, 3, 4, 5])["_deterministic"].any()
    with pytest.raises(ValueError, match="seeds are not"):
        cross.summarise("seasonal_naive", "post_only", [1, 2, 3])
    with pytest.raises(FileNotFoundError):
        cross.summarise("seasonal_naive", "symmetric", [1, 2, 3, 4, 5])
