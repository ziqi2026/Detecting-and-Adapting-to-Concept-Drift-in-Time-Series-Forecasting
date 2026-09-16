import numpy as np
import pandas as pd

from drift_lab.adaptation import adaptive_forecast_stream


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
