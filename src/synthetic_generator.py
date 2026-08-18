import numpy as np
import pandas as pd

import numpy as np
import pandas as pd


# Default parameters for the two synthetic demand concepts
DEFAULT_CONCEPT_1 = {
    "base": 1500,
    "daily_amplitude": 300,
    "weekend_effect": -150,
}

DEFAULT_CONCEPT_2 = {
    "base": 1650,
    "daily_amplitude": 180,
    "weekend_effect": -80,
}

def _make_concept(data, concept):
    """
    Calculate synthetic electricity demand for one concept.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing the time features.

    concept : dict
        Parameters defining the demand concept.

    Returns
    -------
    np.ndarray
        Expected demand values before random noise.
    """

    hour = data["hour"].to_numpy()
    weekend = data["is_weekend"].to_numpy()

    daily_pattern = (
        concept["daily_amplitude"]
        * np.sin(
            2 * np.pi * (hour - 7) / 24
        )
    )

    weekend_pattern = (
        concept["weekend_effect"] * weekend
    )

    demand = (
        concept["base"]
        + daily_pattern
        + weekend_pattern
    )

    return demand

def make_time_features(
    n=20000,
    start="2018-01-01 00:30:00"
):
    """
    Create synthetic 30-minute timestamps and
    time-derived features.

    Parameters
    ----------
    n : int
        Number of half-hour observations.

    start : str
        Starting timestamp.

    Returns
    -------
    pd.DataFrame
        DataFrame containing timestamps and
        time-derived features.
    """

    timestamps = pd.date_range(
        start=start,
        periods=n,
        freq="30min"
    )

    data = pd.DataFrame({
        "SETTLEMENTDATE": timestamps
    })

    # Hour as a numeric value
    # Example: 18:30 -> 18.5
    data["hour"] = (
        data["SETTLEMENTDATE"].dt.hour
        + data["SETTLEMENTDATE"].dt.minute / 60
    )

    # Monday = 0, ..., Sunday = 6
    data["day_of_week"] = (
        data["SETTLEMENTDATE"].dt.dayofweek
    )

    # January = 1, ..., December = 12
    data["month"] = (
        data["SETTLEMENTDATE"].dt.month
    )

    # Weekend indicator
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    return data

def add_synthetic_demand(
    data,
    noise=50,
    seed=42,
    concept=None
):
    """
    Generate synthetic electricity demand with no concept drift.

    The same feature-demand relationship is maintained
    throughout the complete stream.
    """

    data = data.copy()

    # Use the default stable concept unless another is supplied
    if concept is None:
        concept = DEFAULT_CONCEPT_1

    rng = np.random.default_rng(seed)

    # Generate demand from the selected concept
    demand = _make_concept(
        data,
        concept
    )

    # Add controllable random noise
    demand += rng.normal(
        loc=0,
        scale=noise,
        size=len(data)
    )

    data["TOTALDEMAND"] = demand

    return data

def make_sudden_drift(
    data,
    noise=50,
    seed=42,
    changepoint=None,
    concept_1=None,
    concept_2=None
):
    """
    Generate synthetic electricity demand with sudden concept drift.

    The stream switches abruptly from Concept 1 to Concept 2.

    Parameters
    ----------
    data : pd.DataFrame
        Synthetic time-feature dataset.

    noise : float
        Standard deviation of random noise.

    seed : int
        Random seed for reproducibility.

    changepoint : int, optional
        Index where the sudden drift occurs.
        If None, the midpoint of the stream is used.

    concept_1 : dict, optional
        Parameters defining the initial concept.

    concept_2 : dict, optional
        Parameters defining the new concept.

    Returns
    -------
    data : pd.DataFrame
        Dataset containing synthetic TOTALDEMAND.

    changepoint : int
        True changepoint index.
    """

    data = data.copy()

    n = len(data)

    # Use default concepts unless alternatives are supplied
    if concept_1 is None:
        concept_1 = DEFAULT_CONCEPT_1

    if concept_2 is None:
        concept_2 = DEFAULT_CONCEPT_2

    # Default sudden drift occurs halfway through the stream
    if changepoint is None:
        changepoint = n // 2

    # Validate changepoint
    if not 0 < changepoint < n:
        raise ValueError(
            "changepoint must be between 1 and n - 1"
        )

    rng = np.random.default_rng(seed)

    # Generate both concepts
    demand_1 = _make_concept(
        data,
        concept_1
    )

    demand_2 = _make_concept(
        data,
        concept_2
    )

    # Start with Concept 1
    demand = demand_1.copy()

    # Abruptly switch to Concept 2
    demand[changepoint:] = demand_2[changepoint:]

    # Add controllable random noise
    demand += rng.normal(
        loc=0,
        scale=noise,
        size=n
    )

    data["TOTALDEMAND"] = demand

    return data, changepoint
def make_gradual_drift(
    data,
    noise=50,
    seed=42,
    drift_start=None,
    drift_end=None,
    concept_1=None,
    concept_2=None
):
    """
    Generate synthetic electricity demand with gradual concept drift.

    Concept 1 gradually transitions into Concept 2
    over a specified drift window.

    Parameters
    ----------
    data : pd.DataFrame
        Synthetic time-feature dataset.

    noise : float
        Standard deviation of random noise.

    seed : int
        Random seed for reproducibility.

    drift_start : int, optional
        Index where the gradual transition begins.
        If None, 45% of the stream length is used.

    drift_end : int, optional
        Index where the gradual transition ends.
        If None, 55% of the stream length is used.

    concept_1 : dict, optional
        Parameters defining the initial concept.

    concept_2 : dict, optional
        Parameters defining the new concept.

    Returns
    -------
    data : pd.DataFrame
        Dataset containing synthetic TOTALDEMAND.

    drift_start : int
        Start index of the gradual drift window.

    drift_end : int
        End index of the gradual drift window.
    """

    data = data.copy()

    n = len(data)

    # Use default concepts unless alternatives are supplied
    if concept_1 is None:
        concept_1 = DEFAULT_CONCEPT_1

    if concept_2 is None:
        concept_2 = DEFAULT_CONCEPT_2

    # Default gradual drift window
    if drift_start is None:
        drift_start = int(0.45 * n)

    if drift_end is None:
        drift_end = int(0.55 * n)

    # Validate drift window
    if not 0 < drift_start < drift_end < n:
        raise ValueError(
            "drift_start and drift_end must satisfy "
            "0 < drift_start < drift_end < n"
        )

    rng = np.random.default_rng(seed)

    # Generate both concepts
    demand_1 = _make_concept(
        data,
        concept_1
    )

    demand_2 = _make_concept(
        data,
        concept_2
    )

    # Transition weight
    weight = np.zeros(n)

    weight[drift_end:] = 1.0

    weight[drift_start:drift_end] = np.linspace(
        0,
        1,
        drift_end - drift_start
    )

    # Gradually blend Concept 1 into Concept 2
    demand = (
        (1 - weight) * demand_1
        + weight * demand_2
    )

    # Add controllable random noise
    demand += rng.normal(
        loc=0,
        scale=noise,
        size=n
    )

    data["TOTALDEMAND"] = demand

    return data, drift_start, drift_end

def make_recurring_drift(
    data,
    noise=50,
    seed=42,
    cp1=None,
    cp2=None,
    concept_1=None,
    concept_2=None
):
    """
    Generate synthetic electricity demand with recurring concept drift.

    The stream begins with Concept 1, switches to Concept 2,
    then returns to Concept 1.

    Parameters
    ----------
    data : pd.DataFrame
        Synthetic time-feature dataset.

    noise : float
        Standard deviation of random noise.

    seed : int
        Random seed for reproducibility.

    cp1 : int, optional
        Index where Concept 1 switches to Concept 2.
        If None, 35% of the stream length is used.

    cp2 : int, optional
        Index where Concept 1 returns.
        If None, 65% of the stream length is used.

    concept_1 : dict, optional
        Parameters defining the initial concept.

    concept_2 : dict, optional
        Parameters defining the temporary concept.

    Returns
    -------
    data : pd.DataFrame
        Dataset containing synthetic TOTALDEMAND.

    changepoints : list[int]
        True recurring changepoint indices.
    """

    data = data.copy()

    n = len(data)

    # Use default concepts unless alternatives are supplied
    if concept_1 is None:
        concept_1 = DEFAULT_CONCEPT_1

    if concept_2 is None:
        concept_2 = DEFAULT_CONCEPT_2

    # Default recurring changepoints
    if cp1 is None:
        cp1 = int(0.35 * n)

    if cp2 is None:
        cp2 = int(0.65 * n)

    # Validate changepoints
    if not 0 < cp1 < cp2 < n:
        raise ValueError(
            "cp1 and cp2 must satisfy "
            "0 < cp1 < cp2 < n"
        )

    rng = np.random.default_rng(seed)

    # Generate both concepts
    demand_1 = _make_concept(
        data,
        concept_1
    )

    demand_2 = _make_concept(
        data,
        concept_2
    )

    # Start with Concept 1
    demand = demand_1.copy()

    # Temporary switch to Concept 2
    demand[cp1:cp2] = demand_2[cp1:cp2]

    # After cp2, Concept 1 returns automatically

    # Add controllable random noise
    demand += rng.normal(
        loc=0,
        scale=noise,
        size=n
    )

    data["TOTALDEMAND"] = demand

    changepoints = [cp1, cp2]

    return data, changepoints

def make_feature_series(
    kind,
    n=20000,
    noise=50,
    seed=42,
    changepoints=None,
    concept_1=None,
    concept_2=None
):
    """
    Create one feature-based synthetic electricity-demand stream.

    Parameters
    ----------
    kind : str
        One of:
        "none", "sudden", "gradual", "recurring"

    n : int
        Number of half-hour observations.

    noise : float
        Standard deviation of random noise.

    seed : int
        Random seed for reproducibility.

    changepoints : list[int], optional
        Custom changepoint locations.
        If None, default locations based on stream length are used.

    concept_1 : dict, optional
        Parameters defining the initial concept.

    concept_2 : dict, optional
        Parameters defining the changed concept.

    Returns
    -------
    data : pd.DataFrame
        Synthetic feature-based electricity-demand dataset.

    changepoints : list[int]
        True changepoint indices.
    """

    # Create common time features
    X = make_time_features(n=n)

    kind = kind.lower()


    # No drift
    if kind == "none":

        data = add_synthetic_demand(
            X,
            noise=noise,
            seed=seed,
            concept=concept_1
        )

        changepoints = []

   
    # Sudden drift
    elif kind == "sudden":

        cp = (
            changepoints[0]
            if changepoints is not None
            else None
        )

        data, cp = make_sudden_drift(
            X,
            noise=noise,
            seed=seed,
            changepoint=cp,
            concept_1=concept_1,
            concept_2=concept_2
        )

        changepoints = [cp]


    # Gradual drift
    elif kind == "gradual":

        if changepoints is None:
            drift_start = None
            drift_end = None
        else:
            drift_start, drift_end = changepoints

        data, drift_start, drift_end = make_gradual_drift(
            X,
            noise=noise,
            seed=seed,
            drift_start=drift_start,
            drift_end=drift_end,
            concept_1=concept_1,
            concept_2=concept_2
        )

        changepoints = [
            drift_start,
            drift_end
        ]


    # Recurring drift
    elif kind == "recurring":

        if changepoints is None:
            cp1 = None
            cp2 = None
        else:
            cp1, cp2 = changepoints

        data, changepoints = make_recurring_drift(
            X,
            noise=noise,
            seed=seed,
            cp1=cp1,
            cp2=cp2,
            concept_1=concept_1,
            concept_2=concept_2
        )

    else:
        raise ValueError(
            "kind must be 'none', 'sudden', "
            "'gradual', or 'recurring'"
        )

    return data, changepoints
