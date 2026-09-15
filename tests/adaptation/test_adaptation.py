import pandas as pd

from drift_lab.adaptation import (
    adapter,
    _get_adaptation_times,
    _get_training_data,
)
class DummyModel:
    """Minimal model used to check whether fit() is called."""

    def __init__(self):
        self.fit_calls = 0

    def fit(self, data):
        self.fit_calls += 1
        return self


def test_never_strategy_does_not_retrain():
    """Arm A should return the original model without retraining."""

    model = DummyModel()

    data = pd.DataFrame({
        "SETTLEMENTDATE": pd.date_range(
            "2020-03-01",
            periods=10,
            freq="30min",
        )
    })

    changepoints = pd.DataFrame({
        "region": ["SA1"],
        "detector": ["adwin"],
        "detected_index": [5],
        "SETTLEMENTDATE": [pd.Timestamp("2020-03-01 02:30:00")],
    })

    result = adapter(
        changepoints,
        model,
        data,
        region="SA1",
        strategy="never",
    )

    assert result is model
    assert model.fit_calls == 0

  
def test_every_30_days_schedule():
    """Arm B should generate retraining timestamps every 30 days."""

    data = pd.DataFrame({
        "SETTLEMENTDATE": pd.date_range(
            "2020-03-01",
            "2020-06-15",
            freq="1D",
        )
    })

    changepoints = pd.DataFrame()

    adaptation_times = _get_adaptation_times(
        changepoints,
        data,
        region="SA1",
        strategy="every_30_days",
    )

    expected = [
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-04-30"),
        pd.Timestamp("2020-05-30"),
    ]

    assert adaptation_times == expected

def test_drift_strategies_use_filtered_changepoints():
    """Arms C and D should use the same filtered changepoint sequence."""

    data = pd.DataFrame({
        "SETTLEMENTDATE": pd.date_range(
            "2020-03-01",
            periods=10,
            freq="1D",
        )
    })

    changepoints = pd.DataFrame({
        "region": ["SA1", "SA1", "NSW1", "SA1"],
        "detector": ["adwin", "adwin", "adwin", "page_hinkley"],
        "detected_index": [1, 3, 5, 7],
        "SETTLEMENTDATE": pd.to_datetime([
            "2020-03-02",
            "2020-03-04",
            "2020-03-06",
            "2020-03-08",
        ]),
    })

    arm_c_times = _get_adaptation_times(
        changepoints,
        data,
        region="SA1",
        strategy="drift_full_history",
        detector="adwin",
    )

    arm_d_times = _get_adaptation_times(
        changepoints,
        data,
        region="SA1",
        strategy="drift_recent_window",
        detector="adwin",
    )

    expected = [
        pd.Timestamp("2020-03-02"),
        pd.Timestamp("2020-03-04"),
    ]

    assert arm_c_times == expected
    assert arm_d_times == expected

def test_full_history_vs_recent_window():
    """Arms C and D should use different training windows."""

    data = pd.DataFrame({
        "SETTLEMENTDATE": pd.date_range(
            "2020-01-01",
            "2020-04-30",
            freq="1D",
        )
    })

    adaptation_time = pd.Timestamp("2020-04-30")

    full_history = _get_training_data(
        data,
        adaptation_time,
        strategy="drift_full_history",
    )

    recent_window = _get_training_data(
        data,
        adaptation_time,
        strategy="drift_recent_window",
        recent_window="30D",
    )

    assert full_history["SETTLEMENTDATE"].min() == pd.Timestamp("2020-01-01")
    assert full_history["SETTLEMENTDATE"].max() == adaptation_time

    assert recent_window["SETTLEMENTDATE"].min() == pd.Timestamp("2020-03-31")
    assert recent_window["SETTLEMENTDATE"].max() == adaptation_time

    assert len(recent_window) < len(full_history)

def test_training_data_does_not_include_future_rows():
    """Training data must not contain observations after adaptation time."""

    data = pd.DataFrame({
        "SETTLEMENTDATE": pd.date_range(
            "2020-03-01",
            "2020-03-10",
            freq="1D",
        )
    })

    adaptation_time = pd.Timestamp("2020-03-05")

    training_data = _get_training_data(
        data,
        adaptation_time,
        strategy="drift_full_history",
    )

    assert training_data["SETTLEMENTDATE"].max() == adaptation_time
    assert (
        training_data["SETTLEMENTDATE"] <= adaptation_time
    ).all()