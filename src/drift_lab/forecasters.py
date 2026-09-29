"""
Integrated frozen forecasters for the concept-drift forecasting pipeline.

This module contains the forecasting models used by the shared pipeline
to produce point forecasts, forecast-error streams, and the F1
degradation comparison.

The required project baselines are:

    - Seasonal Naive
    - SARIMAX
    - Gradient Boosting

PatchTST is included as an additional Transformer-based forecasting
model and does not replace the required baselines. PyTorch is required
only for PatchTST and is therefore treated as an optional dependency.
When PyTorch is unavailable, the required baseline forecasters remain
fully operational and PatchTST is omitted from MODEL_REGISTRY.


SHARED INTERFACE
----------------
All registered forecasters expose:

    fit(train) -> self
    predict(data, history_df=None) -> np.ndarray

This shared interface allows forecasting, degradation analysis,
drift detection, and later pipeline components to consume model
outputs consistently.

MODEL_REGISTRY provides the central configuration used to construct
the forecasting models that are available in the current environment.


TRAINING AND FROZEN-MODEL PROTOCOL
----------------------------------
All trainable models are fitted using the fixed training data only.

After fit():

    - Seasonal Naive has no trainable parameters.
    - SARIMAX parameters remain fixed.
    - Gradient Boosting is not refitted during frozen evaluation.
    - PatchTST network parameters remain fixed when PatchTST is enabled.

The frozen forecasting stage does not retrain model parameters while
producing predictions over the test stream.

This implements the project's frozen-model requirement while allowing
causally available historical observations to be used where required
by a model's forecasting mechanism.


CAUSAL FORECASTING
------------------
Forecasts are generated chronologically.

For a prediction at time t, only information available before t and
deterministic information associated with t may be used.

After the target at time t has been observed, that observation may
become historical information for forecasting later time points.

The exact use of historical observations depends on the forecaster.


HISTORY_DF
----------
history_df contains observations immediately preceding the supplied
forecast data.

Its role depends on the model:

    Seasonal Naive
        Uses previous observations to provide the seasonal reference
        required for the beginning of the forecast stream. With the
        default season of 48, the prediction uses demand from the same
        half-hour one day earlier.

    SARIMAX
        Uses historical observations to advance the fitted state-space
        model through the supplied history period before evaluation.
        The fitted parameter vector is not re-estimated.

    Gradient Boosting
        When lag features are enabled, history_df supplies up to
        max(lags) observations preceding the forecast stream so that
        the first lagged feature rows are constructed from valid past
        observations.

        With calendar-only features, history_df is not required.

    PatchTST
        When enabled, uses the most recent context_length observations
        preceding the forecast stream as the initial Transformer
        context.

        During chronological prediction, an observed target may enter
        the context only for forecasts of later observations.

If history_df is not supplied, individual models use the fallback
behaviour implemented by their forecasting class.


FORECASTER-SPECIFIC BEHAVIOUR
-----------------------------

Seasonal Naive
    Uses a default seasonal period of 48 half-hourly observations.
    Each prediction therefore uses the observed demand from the same
    half-hour on the previous day.

SARIMAX
    Uses:

        order = (1, 0, 0)
        seasonal_order = (0, 0, 0, 0)
        trend = "c"

    Demand scaling is estimated from training data only.

    Daily, weekly, and annual Fourier/calendar features are
    deterministic functions of timestamps.

    The fitted parameter vector is stored after training and checked
    during prediction to verify that no parameter re-estimation has
    occurred.

Gradient Boosting
    Uses HistGradientBoostingRegressor.

    It supports:

        - calendar features only
        - calendar features plus lagged demand

    The registered configuration currently uses:

        feature_mode = "calendar+lags"
        lags = (48, 336)

    corresponding to previous-day and previous-week demand.

    The estimator is fitted once on training data and is not retrained
    during frozen evaluation.

PatchTST
    Implements a compact univariate PatchTST-style Transformer and is
    enabled only when the optional PyTorch dependency is available.

    The registered configuration uses a context of 336 observations.
    The context is divided into overlapping patches and processed by a
    Transformer encoder to produce one-step forecasts.

    Scaling statistics are estimated from training data only.

    A chronological validation subset is taken from the end of the
    training data for checkpoint selection and early stopping.

    After training, network parameters are frozen and explicitly
    checked after prediction.

    Each prediction uses only the preceding context_length observed
    demand values; the current target is never included in its own
    forecasting context.


FEATURES
--------
The shared make_features() function provides Gradient Boosting with
calendar features:

    - half-hour position
    - day of week
    - weekend indicator
    - month
    - day of year

When configured, lagged TOTALDEMAND values are added.

SARIMAX constructs its own deterministic daily, weekly, and annual
Fourier/calendar features.

Seasonal Naive does not use an explicit feature matrix.

When enabled, PatchTST operates directly on windows of historical
TOTALDEMAND rather than the shared calendar-feature matrix.


OUTPUT AND ERROR STREAM
-----------------------
predict() returns one point forecast for each row of the supplied
forecast data.

frozen_error_stream() converts model output into the common downstream
representation:

    SETTLEMENTDATE
    y_true
    y_pred
    error

where:

    error = y_true - y_pred

This provides the common forecast-error stream consumed by the
error-based drift-detection pipeline.

frozen_error_stream() does not refit the forecasting model. Model
adaptation and retraining are handled separately by the adaptation
component.


MODEL REGISTRY AND OPTIONAL DEPENDENCIES
----------------------------------------
MODEL_REGISTRY is the central forecasting-model registry.

Each registered model defines:

    - display name
    - forecasting class
    - model configuration

The required baseline models are always registered. PatchTST is
registered only when PyTorch is available. If PyTorch is unavailable,
the module remains importable and make_forecasters() constructs only
the required baseline models.

Attempting to instantiate PatchTST without PyTorch raises an explicit
ImportError rather than preventing the other forecasting components
from being used.

This allows later pipeline stages that require the shared forecasting
module but not PatchTST to run without installing the comparatively
large PyTorch dependency.

New forecasting models should therefore be integrated by:

    1. implementing the shared fit()/predict() interface
    2. adding the model and its configuration to MODEL_REGISTRY
    3. using conditional registration when the model depends on an
       optional package

This keeps downstream forecasting, degradation, and drift-detection
workflows independent of model-specific construction and optional
dependency logic.
"""

import random

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True

except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

    TORCH_AVAILABLE = False

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

    # Look up the actual half-hour timestamp for every lag. This preserves
    # continuous-stream behavior and leaves missing history as NaN. fit()
    # drops incomplete feature rows; no calibration values are added.
    if lags:
        times = pd.DatetimeIndex(ts)
        if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError("Lag features require unique chronological timestamps")
        values = pd.Series(data[target].to_numpy(), index=times)
        for lag in lags:
            if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag <= 0:
                raise ValueError("Lags must be positive integer half-hour counts")
            out[f"lag_{lag}"] = values.reindex(
                times - pd.Timedelta(minutes=30 * int(lag))
            ).to_numpy()

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
        # Report actual fitted rows separately from raw input rows.
        self.n_training_samples_ = int(keep.sum())
        if not self.n_training_samples_:
            raise ValueError("No complete training features remain after timestamp lag construction")
        self.model.fit(X.loc[keep], y.loc[keep])
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


if TORCH_AVAILABLE:
 class _PatchTSTNetwork(nn.Module):
    """
    Compact univariate PatchTST-style network for one-step forecasting.

    The input context is split into overlapping patches. Each patch is
    projected to a token embedding, processed by a Transformer encoder,
    and the flattened encoded representation is mapped to one forecast.

    For this project the target is univariate TOTALDEMAND, so PatchTST's
    channel-independent treatment reduces naturally to a single channel.
    """

    def __init__(
        self,
        context_length=336,
        patch_length=16,
        patch_stride=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):

        super().__init__()

        if context_length < patch_length:
            raise ValueError("context_length must be >= patch_length.")
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.context_length = int(context_length)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.n_patches = (
            (self.context_length - self.patch_length) // self.patch_stride
        ) + 1

        self.patch_projection = nn.Linear(self.patch_length, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.n_patches, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.n_patches * d_model, 1)

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        # x: (batch, context_length)
        patches = x.unfold(
            dimension=1,
            size=self.patch_length,
            step=self.patch_stride,
        )
        tokens = self.patch_projection(patches)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        encoded = encoded.reshape(encoded.shape[0], -1)
        return self.head(encoded).squeeze(-1)


class PatchTSTForecaster:
    """
    Frozen PatchTST-style Transformer forecaster.

    Protocol
    --------
    - fit() uses the fixed training split only.
    - Network parameters are frozen after fit().
    - predict() performs inference only; it never updates parameters.
    - Forecast at time t uses only observations available before t.
    - Once y_t has been observed, it may become historical context for t+1.

    This mirrors the project's no-retraining, no-peeking protocol while
    allowing the input history to advance causally through the test stream.
    """

    def __init__(
        self,
        context_length=336,
        patch_length=16,
        patch_stride=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        epochs=20,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=3,
        validation_fraction=0.1,
        seed=0,
        device=None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PatchTST requires PyTorch. "
                "Install torch to use PatchTSTForecaster."
            )
        self.context_length = int(context_length)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.validation_fraction = float(validation_fraction)
        self.seed = int(seed)

        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.model = None
        self.train_mean_ = None
        self.train_std_ = None
        self.train_tail_ = None
        self.frozen_params_ = None

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _extract_demand(data):
        if "TOTALDEMAND" not in data.columns:
            raise ValueError("Input frame must contain TOTALDEMAND.")

        values = data["TOTALDEMAND"].astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("TOTALDEMAND contains missing or non-finite values.")
        return values.astype(np.float32)

    def _make_windows(self, values):
        n_samples = len(values) - self.context_length
        if n_samples <= 0:
            raise ValueError(
                "Not enough observations for the configured context_length."
            )

        # Sliding-window view avoids a slow Python loop and is memory efficient.
        windows = np.lib.stride_tricks.sliding_window_view(
            values,
            self.context_length + 1,
        )
        X = np.asarray(windows[:, :-1], dtype=np.float32)
        y = np.asarray(windows[:, -1], dtype=np.float32)
        return X, y

    def fit(self, train):
        """Fit the network using the training split only."""
        if len(train) == 0:
            raise ValueError("Training frame is empty.")

        self._set_seed()
        train_values = self._extract_demand(train)

        if len(train_values) <= self.context_length:
            raise ValueError(
                "Training data must contain more observations than context_length."
            )

        # Training-only scaling.
        self.train_mean_ = float(train_values.mean())
        self.train_std_ = float(train_values.std())
        if not np.isfinite(self.train_std_) or self.train_std_ <= 0:
            raise ValueError("Invalid training standard deviation.")

        scaled = (
            (train_values - self.train_mean_) / self.train_std_
        ).astype(np.float32)

        self.train_tail_ = train_values[-self.context_length:].copy()
        X, y = self._make_windows(scaled)

        # Chronological validation: final fraction of TRAINING only.
        n_val = max(1, int(len(X) * self.validation_fraction))
        n_train = len(X) - n_val
        if n_train <= 0:
            raise ValueError("Training set is too small after validation split.")

        train_dataset = TensorDataset(
            torch.from_numpy(X[:n_train].copy()),
            torch.from_numpy(y[:n_train].copy()),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X[n_train:].copy()),
            torch.from_numpy(y[n_train:].copy()),
        )

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model = _PatchTSTNetwork(
            context_length=self.context_length,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

        optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        best_loss = np.inf
        best_state = None
        epochs_without_improvement = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimiser.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimiser.step()

            self.model.eval()
            total_loss = 0.0
            total_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb)
                    loss = criterion(pred, yb)
                    batch_n = len(xb)
                    total_loss += float(loss.item()) * batch_n
                    total_n += batch_n

            val_loss = total_loss / total_n
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is None:
            raise RuntimeError("PatchTST training did not produce a valid checkpoint.")

        self.model.load_state_dict(best_state)
        self.model.eval()

        # Freeze network parameters after training.
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.frozen_params_ = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }
        return self

    def _verify_frozen(self):
        current = self.model.state_dict()
        for name, frozen_value in self.frozen_params_.items():
            torch.testing.assert_close(
                current[name].detach().cpu(),
                frozen_value,
                rtol=0,
                atol=0,
            )

    def predict(self, data, history_df=None):
        """
        Produce causal one-step-ahead forecasts for the supplied stream.

        The prediction for row i uses the previous context_length observed
        demand values. The current target is never part of its own context.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        if len(data) == 0:
            return np.array([], dtype=float)

        test_values = self._extract_demand(data)

        if history_df is not None:
            history_values = self._extract_demand(history_df)
        else:
            history_values = self.train_tail_.copy()

        if len(history_values) < self.context_length:
            raise ValueError(
                f"PatchTST requires at least {self.context_length} historical "
                "observations before the forecast stream."
            )

        history_values = history_values[-self.context_length:]
        combined = np.concatenate([history_values, test_values])

        # Sliding windows are aligned so context i ends immediately before y_i.
        all_windows = np.lib.stride_tricks.sliding_window_view(
            combined,
            self.context_length,
        )
        contexts = np.asarray(
            all_windows[: len(test_values)],
            dtype=np.float32,
        )

        contexts = (
            (contexts - self.train_mean_) / self.train_std_
        ).astype(np.float32)

        predictions_scaled = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(contexts), self.batch_size):
                xb = torch.from_numpy(
                    contexts[start:start + self.batch_size].copy()
                ).to(self.device)
                pred = self.model(xb)
                predictions_scaled.append(pred.detach().cpu().numpy())

        predictions_scaled = np.concatenate(predictions_scaled)
        predictions = (
            predictions_scaled * self.train_std_ + self.train_mean_
        )

        self._verify_frozen()
        return predictions.astype(float)

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



# Central forecasting-model registry:
# Add a new forecasting model in ONE place only:
#   1. implement a class with fit(train) and predict(data, history_df=None)
#   2. add one entry below
#
# The degradation experiment and F1 plotting script read this registry, so
# they do not need model-specific edits.
MODEL_REGISTRY = {
    "seasonal_naive": {
        "display_name": "Seasonal Naive",
        "class": SeasonalNaiveForecaster,
        "config": {
            "season": 48,
        },
    },
    "sarimax": {
        "display_name": "SARIMAX",
        "class": SARIMAXForecaster,
        "config": {},
    },
    "gradient_boosted": {
        "display_name": "Gradient Boosted",
        "class": GradientBoostedForecaster,
        "config": {
            "feature_mode": "calendar+lags",
            "lags": (48, 336),
            "seed": 0,
        },
    },
}


# PatchTST is registered only when PyTorch is available.
if TORCH_AVAILABLE:
    MODEL_REGISTRY["patchtst"] = {
        "display_name": "PatchTST",
        "class": PatchTSTForecaster,
        "config": {
            "context_length": 336,
            "patch_length": 16,
            "patch_stride": 8,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "epochs": 20,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "patience": 3,
            "validation_fraction": 0.1,
            "seed": 0,
        },
    }

def make_forecasters():
    """Construct fresh instances of all models in MODEL_REGISTRY."""
    return {
        name: spec["class"](**spec["config"])
        for name, spec in MODEL_REGISTRY.items()
    }
