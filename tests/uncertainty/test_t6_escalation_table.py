import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("t6", ROOT / "experiments/uncertainty/generate_t6_escalation_table.py")
t6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t6)


def curve(rates, coverage):
    return pd.DataFrame({"review_rate": rates, "retained_coverage": coverage,
                         "n_retained": [0 if x == 1 else 100 for x in rates],
                         "tau": list(range(len(rates)))})


def test_exact_target():
    assert t6.select_target(curve([0,.1,.2],[.85,.90,.95]), .9).review_rate == .1


def test_between_thresholds_uses_observed_point_not_interpolation():
    assert t6.select_target(curve([0,.1,.2],[.85,.89,.92]), .9).review_rate == .2


def test_unreachable():
    assert t6.select_target(curve([0,.2],[.8,.89]), .95) is None


@pytest.mark.parametrize("budget, expected",[(.05,.04),(.10,.09),(.20,.19)])
def test_budget_caps(budget, expected):
    g = curve([0,.04,.06,.09,.11,.19,.21],[.8,.81,.82,.83,.84,.85,.86])
    assert t6.select_budget(g,budget).review_rate == expected


def test_nearest_threshold_cannot_exceed_budget():
    assert t6.select_budget(curve([0,.06],[.8,.95]),.05).review_rate == 0


def test_nonmonotone_coverage_target_searches_all_points():
    assert t6.select_target(curve([0,.1,.2,.3],[.88,.91,.87,.96]),.9).review_rate == .1


def test_more_review_may_be_worse():
    assert t6.select_budget(curve([0,.1,.2],[.94,.91,.89]),.2).review_rate == 0


def test_empty_retained_not_success():
    assert t6.select_target(curve([1],[np.nan]),.9) is None


def test_budget_zero_and_one():
    g=curve([0,.1,1],[.8,.95,np.nan])
    assert t6.select_budget(g,0).review_rate == 0
    assert t6.select_budget(g,1).review_rate == .1


def test_exact_budget_boundary():
    assert t6.select_budget(curve([0,.05],[.8,.9]),.05).review_rate == .05


def test_static_collapse():
    g=curve([0,1],[.93,np.nan])
    assert t6.select_target(g,.9).review_rate == 0
    assert t6.select_target(g,.95) is None
    assert t6.select_budget(g,.2).review_rate == 0


def test_tie_smallest_tau():
    g=curve([.1,.1],[.9,.9]); g['tau']=[3,2]
    assert t6.select_target(g,.9).tau == 2
    assert t6.select_budget(g,.1).tau == 2


def test_empty_curve():
    g=curve([],[])
    assert t6.select_target(g,.9) is None
    assert t6.select_budget(g,.1) is None


@pytest.mark.parametrize("value",[-.1,1.1])
def test_invalid_requests(value):
    with pytest.raises(ValueError):
        t6.select_target(curve([0],[.9]),value)
    with pytest.raises(ValueError):
        t6.select_budget(curve([0],[.9]),value)


def test_rounded_ninety_percent_is_not_exact_target():
    assert t6.select_target(curve([0],[.89999]),.90) is None


def test_exact_zero_and_one_targets():
    g=curve([0,.2],[0,1])
    assert t6.select_target(g,0).review_rate == 0
    assert t6.select_target(g,1).review_rate == .2
