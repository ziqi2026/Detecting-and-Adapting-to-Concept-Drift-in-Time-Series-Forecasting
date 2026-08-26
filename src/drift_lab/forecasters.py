"""
Integrated frozen baseline forecasters for the concept-drift forecasting pipeline.

This module contains the three Sprint 2 baseline models used to produce
the F1 degradation curve:

    - SeasonalNaiveForecaster
    - SARIMAXForecaster
    - GradientBoostedForecaster

All three models expose a shared forecasting interface:

    fit(train_df) -> self
    predict(data, history_df=None) -> np.ndarray

The shared interface allows downstream components to work with each
forecaster consistently.

history_df
----------
Optional observations immediately preceding the forecast data.

Its role depends on the forecaster:

    - Seasonal Naive uses the previous 48 observations so the first
      test prediction can reference the same half-hour one day earlier.

    - Gradient Boosting uses previous observations when lag features
      are enabled, supplying up to max(lags) historical values.

    - SARIMAX uses previous observations to advance the fitted model
      state through the calibration/history period before entering the
      test stream. Model parameters are not re-estimated.

Only historical observations are used; no future observations are used
to construct forecasts.

FROZEN BASELINE BEHAVIOUR
-------------------------
All models are fitted using the training period only.

    - Seasonal Naive has no trainable parameters.
    - SARIMAX parameters are estimated during fit() and remain fixed
      during history/test state updates.
    - Gradient Boosting is fitted once and is not retrained while
      producing the frozen test-stream forecasts.

INPUT
-----
Input data frames contain:

    SETTLEMENTDATE
        Period-ending timestamps at 30-minute intervals.

    TOTALDEMAND
        Observed regional electricity demand.

FEATURES
--------
Gradient Boosting can use:

    - calendar features only, or
    - calendar features plus lagged demand values.

SARIMAX uses deterministic daily, weekly and annual Fourier/calendar
features generated from the timestamps.

Seasonal Naive uses no explicit feature matrix.

OUTPUT
------
predict() returns one point forecast for each row of the supplied data.

frozen_error_stream() provides a common downstream representation with:

    SETTLEMENTDATE
    y_true
    y_pred
    error

where:

    error = y_true - y_pred

This error stream can then be used by the drift-detection component.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

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


class SeasonalNaiveForecaster:
    """
    Frozen seasonal-naive baseline.

    Predicts each half-hour using the observed demand from the
    same half-hour one day earlier.

    For 30-minute AEMO data:
        season = 48 observations = 24 hours.

    The model has no trainable parameters and is never retrained.
    """

    def __init__(self, season=48):
        self.season = season

    def fit(self, train):
        """
        Seasonal naive has no parameters to estimate.

        The fit method is included so that this model follows
        the same interface as the other forecasters.
        """
        if len(train) == 0:
            raise ValueError("Training frame is empty.")

        return self

    def predict(self, data, history_df=None):
        """
        Generate seasonal-naive forecasts.

        Parameters
        ----------
        data : pd.DataFrame
            Chronological data to forecast.

        history_df : pd.DataFrame, optional
            Observations immediately preceding `data`.
            Used to provide the previous-day values required
            for the first 48 predictions.

        Returns
        -------
        np.ndarray
            Point forecasts in the same order as `data`.
        """

        if history_df is not None:
            history = history_df.iloc[-self.season:]

            combined = pd.concat(
                [history, data],
                ignore_index=True
            )

            predictions = (
                combined["TOTALDEMAND"]
                .shift(self.season)
                .iloc[len(history):]
                .to_numpy()
            )

        else:
            predictions = (
                data["TOTALDEMAND"]
                .shift(self.season)
                .to_numpy()
            )

        return predictions
    
class SARIMAXForecaster:
    """
    Frozen SARIMAX baseline.

    Uses:
        order = (1, 0, 0)
        seasonal_order = (0, 0, 0, 0)
        trend = "c"

    Demand scaling is estimated from the training set only.
    Calendar/Fourier features are deterministic functions of time.

    After fitting, model parameters remain frozen.
    """

    def __init__(
        self,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend

        self.model = None
        self.train_mean_ = None
        self.train_std_ = None
        self.frozen_params_ = None
        self.feature_names_ = None

    @staticmethod
    def _make_calendar_features(index):
        """
        Create the same deterministic calendar/Fourier features
        used in the original SARIMAX notebook.
        """
        index = pd.DatetimeIndex(index)

        points_per_day = 48
        points_per_week = 7 * points_per_day

        half_hour_slot = (
            index.hour * 2
            + (index.minute // 30)
        )

        day_of_week = index.dayofweek

        week_slot = (
            day_of_week * points_per_day
            + half_hour_slot
        )

        day_of_year = (
            index.dayofyear
            + half_hour_slot / points_per_day
        )

        features = pd.DataFrame(index=index)

        # Daily Fourier terms
        for harmonic in (1, 2, 3):
            angle = (
                2
                * np.pi
                * harmonic
                * half_hour_slot
                / points_per_day
            )

            features[f"daily_sin_{harmonic}"] = np.sin(angle)
            features[f"daily_cos_{harmonic}"] = np.cos(angle)

        # Weekly Fourier terms
        for harmonic in (1, 2):
            angle = (
                2
                * np.pi
                * harmonic
                * week_slot
                / points_per_week
            )

            features[f"weekly_sin_{harmonic}"] = np.sin(angle)
            features[f"weekly_cos_{harmonic}"] = np.cos(angle)

        # Annual Fourier terms
        annual_angle = (
            2
            * np.pi
            * day_of_year
            / 365.25
        )

        features["annual_sin_1"] = np.sin(annual_angle)
        features["annual_cos_1"] = np.cos(annual_angle)

        features["is_weekend"] = (
            day_of_week >= 5
        ).astype(int)

        return features.astype(float)

    @staticmethod
    def _to_demand_series(data):
        """
        Convert the input frame to a regular 30-minute demand series.
        """
        series = (
            data[
                ["SETTLEMENTDATE", "TOTALDEMAND"]
            ]
            .set_index("SETTLEMENTDATE")["TOTALDEMAND"]
            .astype(float)
            .asfreq("30min")
        )

        if series.isna().any():
            raise ValueError(
                "Demand series contains missing half-hour rows."
            )

        return series

    def fit(self, train):
        """
        Fit SARIMAX using the training period only.
        """

        if len(train) == 0:
            raise ValueError("Training frame is empty.")

        train_series = self._to_demand_series(train)

        # Training-only scaling
        self.train_mean_ = float(train_series.mean())
        self.train_std_ = float(train_series.std())

        if (
            not np.isfinite(self.train_std_)
            or self.train_std_ <= 0
        ):
            raise ValueError(
                "Invalid training standard deviation."
            )

        train_scaled = (
            train_series - self.train_mean_
        ) / self.train_std_

        exog = self._make_calendar_features(
            train_series.index
        )

        self.feature_names_ = list(exog.columns)

        model = SARIMAX(
            endog=train_scaled,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=True,
            concentrate_scale=False,
        )

        fitted = model.fit(
            method="lbfgs",
            disp=False,
            maxiter=200,
            cov_type="none",
        )

        converged = bool(
            fitted.mle_retvals.get(
                "converged",
                False
            )
        )

        # Same fallback used in the original notebook
        if not converged:
            fitted = model.fit(
                start_params=np.asarray(
                    fitted.params
                ),
                method="powell",
                disp=False,
                maxiter=200,
                cov_type="none",
            )

            converged = bool(
                fitted.mle_retvals.get(
                    "converged",
                    False
                )
            )

        if not converged:
            raise RuntimeError(
                "SARIMAX did not converge."
            )

        self.model = fitted

        # Snapshot parameters so we can verify
        # that prediction does not retrain the model.
        self.frozen_params_ = np.asarray(
            fitted.params
        ).copy()

        return self

    def predict(self, data, history_df=None):
        """
        Roll the frozen SARIMAX model forward.

        history_df is normally the calibration period.
        It updates the model state using observed historical values,
        but does not re-estimate model parameters.
        """

        if self.model is None:
            raise RuntimeError(
                "Call fit() before predict()."
            )

        test_raw = self._to_demand_series(data)

        test_scaled = (
            test_raw - self.train_mean_
        ) / self.train_std_

        current_state = self.model

        # Move through calibration/history first.
        if history_df is not None:
            history_raw = self._to_demand_series(
                history_df
            )

            history_scaled = (
                history_raw - self.train_mean_
            ) / self.train_std_

            current_state = current_state.extend(
                endog=history_scaled,
                exog=self._make_calendar_features(
                    history_raw.index
                ),
            )

        # Roll frozen model through test stream.
        test_state = current_state.extend(
            endog=test_scaled,
            exog=self._make_calendar_features(
                test_raw.index
            ),
        )

        # Verify that no parameter retraining occurred.
        np.testing.assert_allclose(
            self.frozen_params_,
            np.asarray(test_state.params),
            rtol=1e-10,
            atol=1e-10,
        )

        predictions = (
            np.asarray(test_state.fittedvalues)
            * self.train_std_
            + self.train_mean_
        )

        return predictions

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