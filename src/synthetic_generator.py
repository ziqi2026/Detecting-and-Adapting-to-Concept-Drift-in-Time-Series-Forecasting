import numpy as np
import pandas as pd


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
    seed=42
):
    """
    Generate synthetic electricity demand with no concept drift.

    The relationship between the time features and TOTALDEMAND
    remains fixed throughout the complete stream.

    Parameters
    ----------
    data : pd.DataFrame
        Synthetic time-feature dataset.

    noise : float
        Standard deviation of random noise.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Dataset with synthetic TOTALDEMAND added.
    """

    data = data.copy()

    rng = np.random.default_rng(seed)

    hour = data["hour"].to_numpy()
    weekend = data["is_weekend"].to_numpy()

    # Daily demand cycle
    daily_pattern = (
        300 * np.sin(
            2 * np.pi * (hour - 7) / 24
        )
    )

    # Lower demand during weekends
    weekend_effect = -150 * weekend

    # Random variation
    random_noise = rng.normal(
        loc=0,
        scale=noise,
        size=len(data)
    )

    # Stable synthetic demand
    data["TOTALDEMAND"] = (
        1500
        + daily_pattern
        + weekend_effect
        + random_noise
    )

    return data

def make_sudden_drift(
    data,
    noise=50,
    seed=42
):
    """
    Generate synthetic electricity demand with sudden concept drift.

    The relationship between the time features and TOTALDEMAND
    changes abruptly at the midpoint of the stream.

    Parameters
    ----------
    data : pd.DataFrame
        Synthetic time-feature dataset.

    noise : float
        Standard deviation of random noise.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : pd.DataFrame
        Dataset containing synthetic TOTALDEMAND.

    changepoint : int
        True changepoint index.
    """

    data = data.copy()

    rng = np.random.default_rng(seed)

    hour = data["hour"].to_numpy()
    weekend = data["is_weekend"].to_numpy()

    n = len(data)

    # True changepoint
    changepoint = n // 2


    # Concept 1: before drift
    concept_1 = (
        1500
        + 300 * np.sin(
            2 * np.pi * (hour - 7) / 24
        )
        - 150 * weekend
    )


    # Concept 2: after drift
    concept_2 = (
        1650
        + 180 * np.sin(
            2 * np.pi * (hour - 7) / 24
        )
        - 80 * weekend
    )

    # Begin with Concept 1
    demand = concept_1.copy()

    # Sudden switch to Concept 2
    demand[changepoint:] = (
        concept_2[changepoint:]
    )

    # Add random noise
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
    drift_start=9000,
    drift_end=11000
):
    """
    Generate synthetic electricity demand with gradual concept drift.

    Concept 1 gradually changes into Concept 2 between
    drift_start and drift_end.

    Parameters
    ----------
    data : pd.DataFrame
        Synthetic time-feature dataset.

    noise : float
        Standard deviation of random noise.

    seed : int
        Random seed for reproducibility.

    drift_start : int
        Index where the gradual transition begins.

    drift_end : int
        Index where the gradual transition ends.

    Returns
    -------
    data : pd.DataFrame
        Dataset containing synthetic TOTALDEMAND.

    drift_start : int
        Start index of the true gradual drift window.

    drift_end : int
        End index of the true gradual drift window.
    """

    data = data.copy()

    rng = np.random.default_rng(seed)

    hour = data["hour"].to_numpy()
    weekend = data["is_weekend"].to_numpy()

    n = len(data)


    # Concept 1
    concept_1 = (
        1500
        + 300 * np.sin(
            2 * np.pi * (hour - 7) / 24
        )
        - 150 * weekend
    )


    # Concept 2
    concept_2 = (
        1650
        + 180 * np.sin(
            2 * np.pi * (hour - 7) / 24
        )
        - 80 * weekend
    )


    # Gradual transition weight
    weight = np.zeros(n)

    weight[drift_end:] = 1.0

    weight[drift_start:drift_end] = np.linspace(
        0,
        1,
        drift_end - drift_start
    )

    # Blend Concept 1 into Concept 2
    demand = (
        (1 - weight) * concept_1
        + weight * concept_2
    )

    # Add random noise
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
    cp1=7000,
    cp2=13000
):
    """
    Generate synthetic electricity demand with recurring concept drift.

    The stream begins with Concept 1, switches to Concept 2
    at cp1, and returns to Concept 1 at cp2.

    Parameters
    ----------
    data : pd.DataFrame
        Synthetic time-feature dataset.

    noise : float
        Standard deviation of random noise.

    seed : int
        Random seed for reproducibility.

    cp1 : int
        First true changepoint.

    cp2 : int
        Second true changepoint.

    Returns
    -------
    data : pd.DataFrame
        Dataset containing synthetic TOTALDEMAND.

    changepoints : list[int]
        True recurring changepoint indices.
    """

    data = data.copy()

    rng = np.random.default_rng(seed)

    hour = data["hour"].to_numpy()
    weekend = data["is_weekend"].to_numpy()

    n = len(data)


    # Concept 1
    concept_1 = (
        1500
        + 300 * np.sin(
            2 * np.pi * (hour - 7) / 24
        )
        - 150 * weekend
    )


    # Concept 2
    concept_2 = (
        1650
        + 180 * np.sin(
            2 * np.pi * (hour - 7) / 24
        )
        - 80 * weekend
    )

    # Start with Concept 1
    demand = concept_1.copy()

    # Switch temporarily to Concept 2
    demand[cp1:cp2] = concept_2[cp1:cp2]

    # After cp2, Concept 1 returns automatically

    # Add random noise
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
    seed=42
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

    Returns
    -------
    data : pd.DataFrame
        Synthetic feature-based electricity-demand dataset.

    changepoints : list[int]
        True changepoint indices.

        For gradual drift, the two indices represent
        the start and end of the gradual transition window.
    """

    # Create common time-based features
    X = make_time_features(
        n=n
    )

    kind = kind.lower()

    # No drift
    if kind == "none":

        data = add_synthetic_demand(
            X,
            noise=noise,
            seed=seed
        )

        changepoints = []

    # Sudden drift
    elif kind == "sudden":

        data, cp = make_sudden_drift(
            X,
            noise=noise,
            seed=seed
        )

        changepoints = [cp]

    # Gradual drift
    elif kind == "gradual":

        data, drift_start, drift_end = make_gradual_drift(
            X,
            noise=noise,
            seed=seed
        )

        changepoints = [
            drift_start,
            drift_end
        ]

    # Recurring drift
    elif kind == "recurring":

        data, changepoints = make_recurring_drift(
            X,
            noise=noise,
            seed=seed
        )

    else:
        raise ValueError(
            "kind must be 'none', 'sudden', "
            "'gradual', or 'recurring'"
        )

    return data, changepoints