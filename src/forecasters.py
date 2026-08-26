"""
Gradient-boosted baseline forecaster.

Owner: Sukarna Paul (Scrum Master / Data Analyst)
Sprint 2 deliverable: one of the three frozen baselines feeding F1.

PROPOSED FORECASTER CONTRACT  -- not yet frozen in interfaces.py
---------------------------------------------------------------
    fit(train_df)                    -> self
    predict(df, history_df=None)     -> np.ndarray of point forecasts
    feature_names_                   -> list[str], OPTIONAL

    history_df : rows immediately preceding df, used only to fill
    lag features at the start of a stream.

    feature_names_ is optional because it has no meaning for a
    model with no features, e.g. seasonal-naive.

All three Sprint 2 baselines should expose this surface so that
adapter(changepoints, model, data) can treat them interchangeably.
Raised for the 20/08 team meeting -- per Sahar's review comments.

Input frames are the output of data_loader.load_and_split_aemo:
columns SETTLEMENTDATE (period-ending, 30-min) and TOTALDEMAND.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

"""
CALENDAR_FEATURES has been put here as part of documentation.
This has never been used in the code.
"""
CALENDAR_FEATURES = [
    "halfhour",      # 0-47 position within the day
    "dayofweek",     # 0=Mon
    "is_weekend",
    "month",
    "dayofyear",
]


def make_features(data, lags=(), target="TOTALDEMAND"):
    """
    Build the design matrix.

    lags : tuple[int]
        Lags in 30-minute steps. () gives calendar-only features.
        48 = same time yesterday. 336 = same time last week.

    No time-index / trend feature is included: tree ensembles cannot
    extrapolate beyond the training range, so a trend column would be
    constant-clipped across the whole test stream.
    """
    out = pd.DataFrame(index=data.index)    # index will remain the same as data table.
    ts = data["SETTLEMENTDATE"]             # shortcut for SETTLEMENTDATE column

    out["halfhour"] = ts.dt.hour * 2 + (ts.dt.minute >= 30).astype(int)
    out["dayofweek"] = ts.dt.dayofweek
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["month"] = ts.dt.month
    out["dayofyear"] = ts.dt.dayofyear

    for lag in lags:
        out[f"lag_{lag}"] = data[target].shift(lag)

    return out


class GradientBoostedForecaster:
    """
    Frozen gradient-boosted baseline.

    feature_mode:
        "calendar"       calendar features only
        "calendar+lags"  calendar plus `lags`

    See the module note in the project write-up: the choice is not
    cosmetqic. A frozen model with short lags tracks the level through
    its own inputs and absorbs level drift, which flattens both the
    degradation curve and the error stream the detector reads.
    """

    def __init__(self, feature_mode="calendar", lags=(48, 336), seed=0, **kwargs):
        if feature_mode not in {"calendar", "calendar+lags"}:
            raise ValueError("feature_mode must be 'calendar' or 'calendar+lags'")
        self.feature_mode = feature_mode
        self.lags = tuple(lags) if feature_mode == "calendar+lags" else ()
        self.seed = seed
        self.model = HistGradientBoostingRegressor(random_state=seed, **kwargs)
        self.feature_names_ = None

    def fit(self, train):
        if len(train) == 0:
            raise ValueError(
                "Training frame is empty. Check that the data on disk covers "
                "TRAIN_START..TRAIN_END in data_loader.py."
            )
        X = make_features(train, self.lags)
        y = train["TOTALDEMAND"]
        keep = X.notna().all(axis=1)
        self.feature_names_ = list(X.columns)
        self.model.fit(X.loc[keep], y.loc[keep])    # strip out NaN rows before passing them to scikit-learn's .fit() method.
        return self

    def predict(self, data, history_df=None):
        """
        history_df : DataFrame, optional
            Rows immediately preceding `data`, used only to fill the lag
            features of the first max(lags) rows. Without it those rows
            get NaN lags and visibly inflated error -- an artifact at the
            head of the error stream, which is where the detector starts
            reading. Ignored when feature_mode is "calendar".

            Only past rows are used; no future information enters.
        """
        if history_df is not None and self.lags:
            n = max(self.lags)
            tail = history_df.iloc[-n:]
            combined = pd.concat([tail, data], ignore_index=True)
            X = make_features(combined, self.lags).iloc[len(tail):]
        else:
            X = make_features(data, self.lags)
        return self.model.predict(X[self.feature_names_])


def frozen_error_stream(model, test, history_df=None):
    """
    Roll a frozen model forward through the test stream.

    Returns SETTLEMENTDATE, y_true, y_pred, error.
    `error` is the stream consumed by interfaces.detector().

    Pass `history_df` (typically the calibration frame) whenever the
    model uses lags, so the head of the stream is not corrupted.

    The model is never refitted here -- that is the adapter's job.
    """
    pred = model.predict(test, history_df=history_df)
    return pd.DataFrame({
        "SETTLEMENTDATE": test["SETTLEMENTDATE"].values,
        "y_true": test["TOTALDEMAND"].values,
        "y_pred": pred,
        "error": test["TOTALDEMAND"].values - pred,
    })
