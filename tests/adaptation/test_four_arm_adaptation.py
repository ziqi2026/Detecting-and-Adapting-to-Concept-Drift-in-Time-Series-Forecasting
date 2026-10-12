import numpy as np
import pandas as pd
import pytest
import importlib.util
import json
from pathlib import Path

from drift_lab.adaptation import adaptive_forecast_stream
from drift_lab.forecasters import make_features, GradientBoostedForecaster

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "four_arm_runner", ROOT / "experiments/adaptation/run_four_arm_adaptation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MeanForecaster:
    def __init__(self):
        self.fit_windows = []
        self.mean = 0.0

    def fit(self, train):
        self.fit_windows.append(train[["SETTLEMENTDATE"]].copy())
        self.mean = float(train["TOTALDEMAND"].mean())
        return self

    def predict(self, data, history_df=None):
        return np.repeat(self.mean, len(data))


def frame(start, periods, value):
    return pd.DataFrame({
        "SETTLEMENTDATE": pd.date_range(start, periods=periods, freq="30min"),
        "TOTALDEMAND": np.repeat(value, periods),
    })


def test_retraining_affects_only_forecasts_after_trigger_and_excludes_calibration():
    train = frame("2020-01-01", 4, 10)
    calibration = frame("2020-01-02", 2, 999)
    test = frame("2020-01-03", 8, 20)
    changepoints = pd.DataFrame({
        "region": ["SA1"], "detector": ["page_hinkley"],
        "detected_index": [3],
        "SETTLEMENTDATE": [test.SETTLEMENTDATE.iloc[3]],
    })

    stream, metadata = adaptive_forecast_stream(
        changepoints, MeanForecaster(), train, calibration, test,
        region="SA1", strategy="drift_full_history", detector="page_hinkley",
    )

    assert metadata["n_retrains"] == 1
    assert not bool(stream.loc[3, "retrain_indicator"])
    assert bool(stream.loc[4, "retrain_indicator"])
    # The post-trigger fit contains original train + observed test, not calibration.
    retrain_event = metadata["retraining_events"][0]
    assert retrain_event["training_samples"] == len(train) + 4
    assert retrain_event["training_window_start"] == train.SETTLEMENTDATE.iloc[0]
    assert retrain_event["training_window_end"] == test.SETTLEMENTDATE.iloc[3]
    # REVISION R3: assert actual predictions, not only the event indicators.
    np.testing.assert_array_equal(stream.y_pred.iloc[:4], np.repeat(10.0, 4))
    np.testing.assert_array_equal(stream.y_pred.iloc[4:], np.repeat(15.0, 4))
    assert retrain_event["first_forecast_timestamp"] == test.SETTLEMENTDATE.iloc[4]


def test_drift_arms_use_the_same_changepoint_schedule():
    train = frame("2020-01-01", 2, 10)
    calibration = frame("2020-01-02", 1, 10)
    test = frame("2020-01-03", 10, 20)
    times = [test.SETTLEMENTDATE.iloc[2], test.SETTLEMENTDATE.iloc[7]]
    changepoints = pd.DataFrame({
        "region": ["NSW1", "NSW1"], "detector": ["adwin", "adwin"],
        "detected_index": [2, 7], "SETTLEMENTDATE": times,
    })
    common = dict(changepoints=changepoints, train=train, calibration=calibration,
                  test=test, region="NSW1", detector="adwin")
    _, full = adaptive_forecast_stream(model=MeanForecaster(), strategy="drift_full_history", **common)
    _, recent = adaptive_forecast_stream(model=MeanForecaster(), strategy="drift_recent_window", **common)
    assert full["adaptation_times"] == recent["adaptation_times"]
    assert [e["retrain_timestamp"] for e in full["retraining_events"]] == [e["retrain_timestamp"] for e in recent["retraining_events"]]


def example_inputs():
    train = frame("2019-12-01", 10, 10)
    calibration = frame("2020-02-28", 10, 999)
    test = frame("2020-03-01", 12, 20)
    cp = pd.DataFrame({"region": ["SA1"], "detector": ["adwin"],
                       "detected_index": [3], "SETTLEMENTDATE": [test.SETTLEMENTDATE.iloc[3]]})
    return train, calibration, test, cp


def test_shared_initial_state_and_all_d_windows():
    train, calibration, test, _ = example_inputs()
    cp = make_frozen_detector_changepoints(test, region="SA1", detector="adwin")

    class FitSpy(MeanForecaster):
        calls = []

        def fit(self, data):
            type(self).calls.append(data.SETTLEMENTDATE.tolist())
            self.n_training_samples_ = len(data)
            return super().fit(data)

    forecasts, events, summary = runner.run_regional_four_arm_comparison(
        cp, train, calibration, test, region="SA1", model_factory=FitSpy,
        detectors=("adwin",))
    initial_calls = [call for call in FitSpy.calls if call == train.SETTLEMENTDATE.tolist()]
    assert len(initial_calls) == 1
    assert set(summary.configuration) == {
        "A", "B", "C_adwin", "D_30d_adwin", "D_60d_adwin",
        "D_120d_adwin", "D_180d_adwin",
    }
    assert len(forecasts) == 7 * len(test)
    assert len(events) == 5
    assert events.retrain_timestamp.nunique() == 1
    for _, rows in forecasts.groupby("configuration"):
        np.testing.assert_array_equal(rows.y_true, test.TOTALDEMAND)
        np.testing.assert_array_equal(rows.y_pred.iloc[:4], np.repeat(10.0, 4))
    calibration_times = set(calibration.SETTLEMENTDATE)
    assert all(not calibration_times.intersection(call) for call in FitSpy.calls)


def test_initial_model_is_not_mutated():
    train, calibration, test, cp = example_inputs()
    model = MeanForecaster().fit(train)
    adaptive_forecast_stream(cp, model, train, calibration, test, region="SA1",
                             strategy="drift_full_history", detector="adwin", initial_model_fitted=True)
    assert model.mean == 10
    assert len(model.fit_windows) == 1


def test_future_targets_cannot_change_past_predictions():
    train, calibration, test, cp = example_inputs()
    args = dict(region="SA1", strategy="drift_full_history", detector="adwin")
    original, _ = adaptive_forecast_stream(cp, MeanForecaster(), train, calibration, test, **args)
    changed = test.copy()
    changed.loc[8:, "TOTALDEMAND"] = 1e6
    revised, _ = adaptive_forecast_stream(cp, MeanForecaster(), train, calibration, changed, **args)
    np.testing.assert_array_equal(original.y_pred.iloc[:8], revised.y_pred.iloc[:8])


def test_b_schedule_and_trigger_metadata():
    train, calibration, _, cp = example_inputs()
    test = frame("2020-03-01", 48 * 61 + 1, 20)
    _, meta = adaptive_forecast_stream(cp, MeanForecaster(), train, calibration, test,
                                      region="SA1", strategy="every_30_days", detector="adwin")
    assert meta["adaptation_times"] == [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-04-30")]
    assert all(e["trigger_type"] == "scheduled" and e["detector"] == "" for e in meta["retraining_events"])


@pytest.mark.parametrize("days", [30, 60, 120, 180])
def test_recent_window_boundaries_and_calibration_exclusion(days):
    train = frame("2019-01-01", 48 * 365, 10)
    calibration = frame("2020-01-01", 48 * 60, 999)
    test = frame("2020-03-01", 48 * 200 + 1, 20)
    cp = pd.DataFrame({"region": ["SA1"], "detector": ["adwin"],
                       "detected_index": [len(test) - 2], "SETTLEMENTDATE": [test.SETTLEMENTDATE.iloc[-2]]})
    _, meta = adaptive_forecast_stream(cp, MeanForecaster(), train, calibration, test,
                                     region="SA1", strategy="drift_recent_window", detector="adwin",
                                     recent_window=f"{days}D")
    event = meta["retraining_events"][0]
    assert event["training_window_start"] == event["retrain_timestamp"] - pd.Timedelta(days=days)
    assert event["training_samples"] == days * 48 + 1


@pytest.mark.parametrize("problem", ["negative", "fraction", "outside", "mismatch", "duplicate"])
def test_invalid_changepoints_rejected(problem):
    train, calibration, test, cp = example_inputs()
    if problem == "negative":
        cp["detected_index"] = -1
    elif problem == "fraction":
        cp["detected_index"] = 1.5
    elif problem == "outside":
        cp["detected_index"] = len(test)
    elif problem == "mismatch":
        cp["SETTLEMENTDATE"] = test.SETTLEMENTDATE.iloc[4]
    else:
        cp = pd.concat([cp, cp], ignore_index=True)
    with pytest.raises(ValueError):
        adaptive_forecast_stream(cp, MeanForecaster(), train, calibration, test,
                                 region="SA1", strategy="drift_full_history", detector="adwin")


def test_empty_test_rejected_and_zero_alarm_stream_supported():
    train, calibration, test, cp = example_inputs()
    with pytest.raises(ValueError):
        adaptive_forecast_stream(cp, MeanForecaster(), train, calibration, test.iloc[:0],
                                 region="SA1", strategy="never")
    stream, meta = adaptive_forecast_stream(cp.iloc[:0], MeanForecaster(), train, calibration, test,
                                           region="SA1", strategy="drift_full_history", detector="adwin")
    assert meta["n_retrains"] == 0
    assert len(stream) == len(test)


def test_final_timestamp_refit_has_no_affected_forecast():
    train, calibration, test, cp = example_inputs()
    cp["detected_index"] = len(test) - 1
    cp["SETTLEMENTDATE"] = test.SETTLEMENTDATE.iloc[-1]
    stream, meta = adaptive_forecast_stream(cp, MeanForecaster(), train, calibration, test,
                                           region="SA1", strategy="drift_full_history", detector="adwin")
    assert meta["n_retrains"] == 1
    assert pd.isna(meta["retraining_events"][0]["first_forecast_timestamp"])
    assert not stream.retrain_indicator.any()


def test_timestamp_lags_do_not_bridge_calibration_gap():
    data = pd.concat([frame("2019-12-31", 48, 10), frame("2020-03-01", 100, 20)], ignore_index=True)
    features = make_features(data, lags=(48,))
    assert features.lag_48.iloc[48:96].isna().all()
    assert features.lag_48.iloc[96:].eq(20).all()
    contiguous = frame("2020-03-01", 200, 5)
    pd.testing.assert_series_equal(make_features(contiguous, (48,)).lag_48,
                                   contiguous.TOTALDEMAND.shift(48), check_names=False)


def test_effective_training_samples_and_real_model_causality():
    data = frame("2019-12-01", 400, 10)
    data["TOTALDEMAND"] = 10 + np.sin(np.arange(len(data)) / 10)
    model = GradientBoostedForecaster(feature_mode="calendar+lags", lags=(48,), max_iter=3).fit(data)
    assert model.n_training_samples_ == 352
    test = frame(data.SETTLEMENTDATE.iloc[-1] + pd.Timedelta(minutes=30), 100, 12)
    reference = model.predict(test, history_df=data)
    changed = test.copy()
    changed.loc[20:, "TOTALDEMAND"] = 9999
    actual = model.predict(changed, history_df=data)
    np.testing.assert_array_equal(reference[:21], actual[:21])


def make_frozen_detector_changepoints(test, *, region, detector, index=3):
    """Build a minimal frozen detector fixture with complete provenance."""
    return pd.DataFrame({
        "region": [region], "detector": [detector], "detected_index": [index],
        "SETTLEMENTDATE": [test.SETTLEMENTDATE.iloc[index]],
        "detector_config": [json.dumps({"detector": detector, "fixture": True})],
        "warmup_samples": [2880], "test_samples": [len(test)],
        "refractory_period": [672], "refractory_days": [14],
        "seasonal_profile": ["overall_mean+smoothed_DOY+(DOWxhalf_hour)"],
        "seasonal_reference": ["complete_TRAIN_only"],
        "error_zscore_reference": ["TRAIN absolute forecast error"],
        "warmup_reference": ["processed CALIBRATION forecast errors"],
        "detector_input": ["TRAIN_zscored_absolute_forecast_error"],
        "whitening_type": ["partial_AR1"],
    })


def write_experiment_decisions(path):
    """Write only the shared-decision statements required by the runner."""
    path.write_text("\n".join([
        "**COVID-19 event:** 16 Mar 2020 – 17 May 2020.",
        "Five-Minute Settlement (1 Oct 2021):** removed from the target event catalogue",
        "Deseasonalisation parameters will be fitted using **TRAIN only**.",
        "Detector inputs will be **Z-score standardised using TRAIN only**.",
        "**five seeds: 1–5**",
        "Agreed refractory period: **14 days**.",
        "**Arm B — Periodic Retraining:** retrain at a fixed **30-day interval**.",
        "Evaluate training windows of **30, 60, 120, and 180 days**.",
        "The exact warm-up period remains to be confirmed.",
    ]))


def test_frozen_changepoint_contract_rejects_rows_without_provenance():
    _, _, test, incomplete_rows = example_inputs()
    with pytest.raises(ValueError, match="provenance"):
        runner.validate_frozen_detector_changepoints(
            incomplete_rows, test, region="SA1", detector="adwin")
    accepted = make_frozen_detector_changepoints(test, region="SA1", detector="adwin")
    selected = runner.validate_frozen_detector_changepoints(
        accepted, test, region="SA1", detector="adwin")
    assert selected.detected_index.tolist() == [3]


def test_four_arm_comparison_runs_a_b_once_and_reuses_each_detector_schedule(tmp_path, monkeypatch):
    train, calibration, test, _ = example_inputs()
    changepoints = pd.concat([
        make_frozen_detector_changepoints(test, region=region, detector=detector)
        for region in runner.REGIONS for detector in runner.SUPPORTED_DETECTORS
    ], ignore_index=True)
    detections = tmp_path / "frozen_detector_changepoints.csv"
    changepoints.to_csv(detections, index=False)
    decisions = tmp_path / "decisions.md"
    write_experiment_decisions(decisions)
    monkeypatch.setattr(runner, "load_and_split_aemo", lambda *args: (None, train, calibration, test))
    monkeypatch.setitem(runner.MODEL_REGISTRY, "gradient_boosted",
                        {"class": lambda **kw: MeanForecaster(), "config": {}})
    run_args = dict(
        detections_path=detections, decisions_path=decisions, run_id="four_arm_fixture",
        data_path=tmp_path, output_root=tmp_path / "out",
        shared_runs=tmp_path / "runs.csv")
    summary = runner.run_four_arm_adaptation_experiment(**run_args)
    out = tmp_path / "out/four_arm_fixture"
    assert len(summary) == 34  # A/B once plus C and four D windows for each detector.
    for region, records in summary.groupby("region"):
        assert records.loc[records.arm.isin(["A", "B"])].configuration.tolist() == ["A", "B"]
        assert len(records.loc[records.arm.eq("C")]) == 3
        assert len(records.loc[records.arm.eq("D")]) == 12
        assert set(records.loc[records.arm.isin(["C", "D"]), "detector"]) == set(runner.SUPPORTED_DETECTORS)
    events = pd.read_csv(out / "retraining_events.csv")
    assert len(events) == 30
    for (region, detector), rows in events.groupby(["region", "detector"]):
        schedules = rows.groupby("configuration").retrain_timestamp.apply(list)
        assert len(schedules) == 5
        assert all(schedule == schedules.iloc[0] for schedule in schedules)
    assert len(pd.read_csv(out / "runs.csv")) == 136
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["same_changepoints_for_c_and_all_d_within_detector"] is True
    assert set(manifest["outputs_sha256"]) == {
        "forecasts_SA1.csv.gz", "forecasts_NSW1.csv.gz", "summary.csv", "runs.csv",
        "window_sensitivity.csv", "detector_comparison.csv", "retraining_events.csv",
    }
    with pytest.raises(FileExistsError):
        runner.run_four_arm_adaptation_experiment(**run_args)
