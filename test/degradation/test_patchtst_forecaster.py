import copy

import numpy as np
import pandas as pd
import pytest
import torch

from drift_lab.forecasters import PatchTSTForecaster


#  Helpers

def make_halfhourly_frame(
    n=1200,
    start="2019-01-01 00:30:00",
    seed=123,
):
    """
    Small deterministic synthetic demand series for unit tests.

    Contains:
        - daily seasonality
        - weekly seasonality
        - weak trend
        - small noise
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start=start,
        periods=n,
        freq="30min",
    )

    t = np.arange(n)

    daily = 100 * np.sin(
        2 * np.pi * t / 48
    )

    weekly = 40 * np.sin(
        2 * np.pi * t / 336
    )

    trend = 0.03 * t

    noise = rng.normal(
        loc=0.0,
        scale=5.0,
        size=n,
    )

    demand = (
        1500
        + daily
        + weekly
        + trend
        + noise
    )

    return pd.DataFrame(
        {
            "SETTLEMENTDATE": timestamps,
            "TOTALDEMAND": demand,
        }
    )


def make_small_model(seed=0):
    """
    Small PatchTST configuration so unit tests run quickly.

    This is deliberately smaller than the experiment model.
    """
    return PatchTSTForecaster(
        context_length=96,
        patch_length=16,
        patch_stride=8,
        d_model=32,
        n_heads=4,
        n_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        epochs=2,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=2,
        validation_fraction=0.1,
        seed=seed,
        device="cpu",
    )


def split_test_data():
    """
    Return:
        train
        calibration
        test
    """
    df = make_halfhourly_frame(
        n=1400,
        seed=123,
    )

    train = df.iloc[:1000].copy()
    calibration = df.iloc[1000:1150].copy()
    test = df.iloc[1150:1250].copy()

    return train, calibration, test


def snapshot_state_dict(model):
    """
    Deep copy all network parameters.
    """
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor
        in model.model.state_dict().items()
    }


def assert_state_dict_equal(before, after):
    """
    Assert exact equality of two PyTorch model states.
    """
    assert before.keys() == after.keys()

    for name in before:
        torch.testing.assert_close(
            before[name],
            after[name],
            rtol=0,
            atol=0,
        )


#  1. FIT / BASIC CONTRACT

def test_patchtst_fit_returns_self():
    train, _, _ = split_test_data()

    model = make_small_model()

    result = model.fit(train)

    assert result is model
    assert model.model is not None


def test_patchtst_predict_returns_one_prediction_per_row():
    train, calibration, test = split_test_data()

    model = make_small_model()
    model.fit(train)

    pred = model.predict(
        test,
        history_df=calibration,
    )

    assert isinstance(pred, np.ndarray)
    assert pred.shape == (len(test),)


def test_patchtst_predictions_are_finite():
    train, calibration, test = split_test_data()

    model = make_small_model()
    model.fit(train)

    pred = model.predict(
        test,
        history_df=calibration,
    )

    assert np.isfinite(pred).all()


# 2. FROZEN MODEL TEST

def test_patchtst_weights_do_not_change_during_prediction():
    """
    Core project requirement:

    predict() must not retrain or update PatchTST parameters.
    """

    train, calibration, test = split_test_data()

    model = make_small_model()
    model.fit(train)

    before = snapshot_state_dict(model)

    _ = model.predict(
        test,
        history_df=calibration,
    )

    after = snapshot_state_dict(model)

    assert_state_dict_equal(
        before,
        after,
    )


# 3. SAME-TIME LEAKAGE TEST

def test_patchtst_same_time_target_does_not_affect_its_prediction():
    """
    Change y_t.

    Prediction for the same timestamp t must remain unchanged.

    If it changes, the model is peeking at the target it is
    supposed to predict.
    """

    train, calibration, test = split_test_data()

    model = make_small_model()
    model.fit(train)

    baseline_pred = model.predict(
        test,
        history_df=calibration,
    )

    modified_test = test.copy()

    t = 25

    modified_test.loc[
        modified_test.index[t],
        "TOTALDEMAND",
    ] += 10000.0

    modified_pred = model.predict(
        modified_test,
        history_df=calibration,
    )

    np.testing.assert_allclose(
        baseline_pred[t],
        modified_pred[t],
        rtol=0,
        atol=1e-6,
    )


#  4. FUTURE LEAKAGE TEST

def test_patchtst_future_observation_cannot_change_earlier_predictions():
    """
    Change an observation in the future.

    Every prediction before that observation must remain unchanged.
    """

    train, calibration, test = split_test_data()

    model = make_small_model()
    model.fit(train)

    baseline_pred = model.predict(
        test,
        history_df=calibration,
    )

    modified_test = test.copy()

    future_index = 60

    modified_test.loc[
        modified_test.index[future_index],
        "TOTALDEMAND",
    ] += 10000.0

    modified_pred = model.predict(
        modified_test,
        history_df=calibration,
    )

    np.testing.assert_allclose(
        baseline_pred[: future_index + 1],
        modified_pred[: future_index + 1],
        rtol=0,
        atol=1e-6,
    )


# 5. CAUSAL STATE / HISTORY TEST

def test_patchtst_observed_value_can_affect_later_predictions():
    """
    This is the causal counterpart to the leakage tests.

    Changing y_t SHOULD be allowed to affect predictions after t,
    because once y_t has been observed it becomes historical context.

    We do not require every later prediction to change, only that
    at least one later prediction within the context horizon changes.
    """

    train, calibration, test = split_test_data()

    model = make_small_model()
    model.fit(train)

    baseline_pred = model.predict(
        test,
        history_df=calibration,
    )

    modified_test = test.copy()

    t = 25

    modified_test.loc[
        modified_test.index[t],
        "TOTALDEMAND",
    ] += 10000.0

    modified_pred = model.predict(
        modified_test,
        history_df=calibration,
    )

    later_difference = np.abs(
        baseline_pred[t + 1:]
        - modified_pred[t + 1:]
    )

    assert np.any(
        later_difference > 1e-6
    )


#  6. HISTORY BOUNDARY TEST

def test_patchtst_first_prediction_uses_history_only():
    """
    The first test prediction must be generated from history_df,
    not from the first test target.
    """

    train, calibration, test = split_test_data()

    model = make_small_model()
    model.fit(train)

    baseline_pred = model.predict(
        test,
        history_df=calibration,
    )

    modified_test = test.copy()

    modified_test.loc[
        modified_test.index[0],
        "TOTALDEMAND",
    ] += 50000.0

    modified_pred = model.predict(
        modified_test,
        history_df=calibration,
    )

    np.testing.assert_allclose(
        baseline_pred[0],
        modified_pred[0],
        rtol=0,
        atol=1e-6,
    )


#  7. HISTORY REALLY MATTERS

def test_patchtst_history_changes_first_prediction():
    """
    Change the immediately preceding historical context.

    The first test prediction should normally change because
    PatchTST is allowed to use historical observations.
    """

    train, calibration, test = split_test_data()

    model = make_small_model()
    model.fit(train)

    baseline_pred = model.predict(
        test,
        history_df=calibration,
    )

    modified_history = calibration.copy()

    modified_history.loc[
        modified_history.index[-1],
        "TOTALDEMAND",
    ] += 10000.0

    modified_pred = model.predict(
        test,
        history_df=modified_history,
    )

    assert not np.isclose(
        baseline_pred[0],
        modified_pred[0],
        rtol=0,
        atol=1e-6,
    )


# 8. REPRODUCIBILITY

def test_patchtst_same_seed_gives_same_predictions():
    """
    Two fresh models with identical data/config/seed should produce
    the same predictions.
    """

    train, calibration, test = split_test_data()

    model_a = make_small_model(
        seed=42
    )

    model_b = make_small_model(
        seed=42
    )

    model_a.fit(train)
    model_b.fit(train)

    pred_a = model_a.predict(
        test,
        history_df=calibration,
    )

    pred_b = model_b.predict(
        test,
        history_df=calibration,
    )

    np.testing.assert_allclose(
        pred_a,
        pred_b,
        rtol=1e-5,
        atol=1e-5,
    )


# 9. TRAINING-ONLY SCALING

def test_patchtst_scaling_comes_only_from_training_data():
    """
    train_mean_ and train_std_ must equal statistics calculated
    only from the training frame.
    """

    train, _, _ = split_test_data()

    model = make_small_model()
    model.fit(train)

    expected_mean = float(
        train["TOTALDEMAND"].astype(float).mean()
    )

    expected_std = float(
        train["TOTALDEMAND"]
        .astype(float)
        .to_numpy()
        .std()
    )

    assert np.isclose(
        model.train_mean_,
        expected_mean,
    )

    assert np.isclose(
        model.train_std_,
        expected_std,
    )


# 10. PREDICT DOES NOT MODIFY TRAINING STATISTICS

def test_patchtst_prediction_does_not_change_training_scaler():
    train, calibration, test = split_test_data()

    model = make_small_model()
    model.fit(train)

    mean_before = model.train_mean_
    std_before = model.train_std_

    _ = model.predict(
        test,
        history_df=calibration,
    )

    assert model.train_mean_ == mean_before
    assert model.train_std_ == std_before


# 11. INPUT VALIDATION

def test_patchtst_requires_fit_before_predict():
    _, calibration, test = split_test_data()

    model = make_small_model()

    with pytest.raises(
        RuntimeError,
        match="Call fit",
    ):
        model.predict(
            test,
            history_df=calibration,
        )


def test_patchtst_rejects_empty_training_frame():
    model = make_small_model()

    empty = pd.DataFrame(
        columns=[
            "SETTLEMENTDATE",
            "TOTALDEMAND",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Training frame is empty",
    ):
        model.fit(empty)


def test_patchtst_rejects_nonfinite_demand():
    train, _, _ = split_test_data()

    train = train.copy()

    train.loc[
        train.index[10],
        "TOTALDEMAND",
    ] = np.nan

    model = make_small_model()

    with pytest.raises(
        ValueError,
        match="missing or non-finite",
    ):
        model.fit(train)


def test_patchtst_requires_sufficient_history():
    train, calibration, test = split_test_data()

    model = make_small_model()
    model.fit(train)

    too_short_history = calibration.iloc[:20].copy()

    with pytest.raises(
        ValueError,
        match="requires at least",
    ):
        model.predict(
            test,
            history_df=too_short_history,
        )

# 12. EMPTY TEST STREAM

def test_patchtst_empty_test_returns_empty_array():
    train, calibration, test = split_test_data()

    model = make_small_model()
    model.fit(train)

    empty_test = test.iloc[:0].copy()

    pred = model.predict(
        empty_test,
        history_df=calibration,
    )

    assert isinstance(pred, np.ndarray)
    assert len(pred) == 0


    #poetry run pytest test/degradation/test_patchtst_forecaster.py -v