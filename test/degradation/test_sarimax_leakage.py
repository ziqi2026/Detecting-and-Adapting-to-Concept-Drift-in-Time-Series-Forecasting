"""
SARIMAX leakage and causal-state tests.

Purpose
-------
These tests investigate whether the SARIMAX baseline uses unavailable
test-period information when generating forecasts.

The project requires baseline models to be frozen and evaluated
chronologically without peeking.

We investigate three questions:

1. FUTURE LEAKAGE
   Can changing an observation at time t alter predictions made before t?

2. SAME-TIMESTAMP TARGET LEAKAGE
   Can changing y[t] alter the prediction y_hat[t]?

3. CAUSAL STATE UPDATE
   Once y[t] has been observed, does it affect later predictions?

Expected causal behaviour:

    predictions before t   -> unchanged
    prediction at t        -> unchanged
    predictions after t    -> may change

This distinguishes legitimate causal state updating from target leakage.
"""

import numpy as np
import pandas as pd

from drift_lab.forecasters import SARIMAXForecaster


N_TRAIN = 48 * 30
N_CAL = 48 * 7
N_TEST = 48 * 7

TARGET_POSITION = 100
PERTURBATION = 10000.0


def make_test_data():
    """
    Create deterministic half-hourly demand-like data.

    Includes:
    daily seasonality,
    weekly seasonality,
    random noise.
    """

    rng = np.random.default_rng(42)

    n_total = N_TRAIN + N_CAL + N_TEST

    index = pd.date_range(
        start="2020-01-01",
        periods=n_total,
        freq="30min",
    )

    t = np.arange(n_total)

    demand = (
        1000
        + 150 * np.sin(2 * np.pi * t / 48)
        + 50 * np.sin(2 * np.pi * t / 336)
        + rng.normal(0, 10, n_total)
    )

    df = pd.DataFrame(
        {
            "SETTLEMENTDATE": index,
            "TOTALDEMAND": demand,
        }
    )

    train = df.iloc[:N_TRAIN].copy()

    calibration = df.iloc[
        N_TRAIN:N_TRAIN + N_CAL
    ].copy()

    test = df.iloc[
        N_TRAIN + N_CAL:
    ].copy()

    return train, calibration, test


def run_original_and_perturbed():
    """
    Run SARIMAX twice.

    Run 1:
        original test stream

    Run 2:
        identical test stream except y[t] is increased by 10,000

    Returns both prediction arrays and the modified test stream.
    """

    train, calibration, test = make_test_data()

    model_original = SARIMAXForecaster()
    model_original.fit(train)

    pred_original = np.asarray(
        model_original.predict(
            test,
            history_df=calibration,
        )
    )

    test_changed = test.copy()

    target_col = test_changed.columns.get_loc(
        "TOTALDEMAND"
    )

    original_target = test_changed.iloc[
        TARGET_POSITION
    ]["TOTALDEMAND"]

    test_changed.iloc[
        TARGET_POSITION,
        target_col,
    ] = original_target + PERTURBATION

    model_changed = SARIMAXForecaster()
    model_changed.fit(train)

    pred_changed = np.asarray(
        model_changed.predict(
            test_changed,
            history_df=calibration,
        )
    )

    return (
        test,
        test_changed,
        pred_original,
        pred_changed,
    )


def test_01_no_future_leakage():
    """
    TEST 1: Future leakage.

    Changing y[t] must not affect any prediction produced
    before timestamp t.
    """

    (
        test,
        test_changed,
        pred_original,
        pred_changed,
    ) = run_original_and_perturbed()

    differences = np.abs(
        pred_original - pred_changed
    )

    before_diff = differences[:TARGET_POSITION]

    max_before = (
        before_diff.max()
        if len(before_diff)
        else 0.0
    )

    print()
    print("=" * 65)
    print("TEST 1: FUTURE LEAKAGE")
    print("=" * 65)
    print(f"Changed position: {TARGET_POSITION}")
    print(f"Perturbation: +{PERTURBATION:.0f} MW")
    print(
        "Maximum prediction difference before t:",
        max_before,
    )

    if np.allclose(
        pred_original[:TARGET_POSITION],
        pred_changed[:TARGET_POSITION],
        rtol=1e-10,
        atol=1e-10,
    ):
        print(
            "RESULT: PASS"
        )
        print(
            "Changing y[t] did NOT affect earlier forecasts."
        )
        print(
            "No evidence of future-to-past leakage."
        )

    assert np.allclose(
        pred_original[:TARGET_POSITION],
        pred_changed[:TARGET_POSITION],
        rtol=1e-10,
        atol=1e-10,
    ), (
        "FAIL: FUTURE LEAKAGE DETECTED. "
        "Changing y[t] altered predictions before t."
    )


def test_02_no_same_timestamp_target_leakage():
    """
    TEST 2: Same-timestamp target leakage.

    Prediction y_hat[t] must be generated without using
    the actual target y[t].
    """

    (
        test,
        test_changed,
        pred_original,
        pred_changed,
    ) = run_original_and_perturbed()

    original_y = test.iloc[
        TARGET_POSITION
    ]["TOTALDEMAND"]

    changed_y = test_changed.iloc[
        TARGET_POSITION
    ]["TOTALDEMAND"]

    original_pred = pred_original[
        TARGET_POSITION
    ]

    changed_pred = pred_changed[
        TARGET_POSITION
    ]

    difference = abs(
        original_pred - changed_pred
    )

    print()
    print("=" * 65)
    print("TEST 2: SAME-TIMESTAMP TARGET LEAKAGE")
    print("=" * 65)

    print(
        f"Original y[{TARGET_POSITION}]: "
        f"{original_y:.6f}"
    )

    print(
        f"Changed y[{TARGET_POSITION}]: "
        f"{changed_y:.6f}"
    )

    print(
        f"Original prediction[{TARGET_POSITION}]: "
        f"{original_pred:.6f}"
    )

    print(
        f"Changed prediction[{TARGET_POSITION}]: "
        f"{changed_pred:.6f}"
    )

    print(
        "Same-timestamp prediction difference:",
        difference,
    )

    if np.isclose(
        original_pred,
        changed_pred,
        rtol=1e-10,
        atol=1e-10,
    ):
        print("RESULT: PASS")
        print(
            "Changing y[t] did NOT affect prediction[t]."
        )
        print(
            "No evidence of same-timestamp target leakage."
        )

    assert np.isclose(
        original_pred,
        changed_pred,
        rtol=1e-10,
        atol=1e-10,
    ), (
        "FAIL: SAME-TIMESTAMP TARGET LEAKAGE DETECTED. "
        "Changing y[t] altered prediction[t]."
    )


def test_03_causal_state_update_after_observation():
    """
    TEST 3: Causal state update.

    Once y[t] has been observed, it is historical information.

    Therefore changing y[t] is expected to affect at least
    some forecasts after t if SARIMAX updates its latent state.

    This is causal state updating, not same-timestamp leakage.
    """

    (
        test,
        test_changed,
        pred_original,
        pred_changed,
    ) = run_original_and_perturbed()

    differences = np.abs(
        pred_original - pred_changed
    )

    after_diff = differences[
        TARGET_POSITION + 1:
    ]

    max_after = (
        after_diff.max()
        if len(after_diff)
        else 0.0
    )

    number_changed = np.sum(
        after_diff > 1e-8
    )

    print()
    print("=" * 65)
    print("TEST 3: CAUSAL STATE UPDATE")
    print("=" * 65)

    print(
        "Maximum prediction difference after t:",
        max_after,
    )

    print(
        "Number of later predictions affected:",
        number_changed,
    )

    if np.any(after_diff > 1e-8):
        print("RESULT: PASS")
        print(
            "Observed y[t] affects later forecasts only after "
            "it becomes available."
        )
        print(
            "This is consistent with causal SARIMAX state updating."
        )

    assert np.any(
        after_diff > 1e-8
    ), (
        "FAIL: y[t] did not affect any later prediction. "
        "SARIMAX may not be updating its state with newly "
        "observed demand."
    )

    # pytest test/detection/test_sarimax_leakage.py -v -s
    
    '''
    SARIMAX stability investigation: Leakage tests passed,
    but revealed causal test-period state updates. 
    Removing those updates in a blind multi-step diagnostic
    exposed substantially greater degradation, 
    indicating that state updating (not leakage ) 
    was responsible for the unusually stable original curve.
    SARIMAX was not retrained, but its internal state was
    updated after each observed test value,
    allowing it to correct subsequent forecasts 
    more directly than the other frozen baselines.
    '''