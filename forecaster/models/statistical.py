"""
statistical.py
Classic statistical forecasting models, from the statsmodels library.

1. Exponential smoothing (Holt-Winters)
   Keeps a running estimate of three things, updated every period:
     level  - "where are sales right now?"
     trend  - "how fast are they growing?" (damped, so it does not grow forever)
     season - "how much higher or lower is this weekday / month than usual?"
   Recent periods count more than old ones ("exponentially" less weight further back).

2. ARIMA
   Predicts the next value from a mix of recent values and recent forecast errors,
   after removing the overall trend. We use a fixed, common setup, (1,1,1) plus one
   seasonal layer, instead of searching many setups, to keep training fast.

Seasons used (one season = one repeating cycle):
   daily data   -> 7   (the weekly pattern)
   weekly data  -> 52  (the yearly pattern; exponential smoothing only, ARIMA would be too slow)
   monthly data -> 12  (the yearly pattern)
Neither model here understands festivals whose date moves every year (like Diwali).
"""

import warnings

import numpy as np

from forecaster.models.base import ForecastModel

SEASON_LENGTH = {"daily": 7, "weekly": 52, "monthly": 12}


class ExponentialSmoothingModel(ForecastModel):
    name = "exponential_smoothing"
    family = "statistical"

    def _fit(self, history):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        season = SEASON_LENGTH[self.frequency]
        # A season can only be estimated from at least 2 full cycles of it.
        use_season = len(history) >= 2 * season
        model = ExponentialSmoothing(
            history.to_numpy(dtype=float),
            trend="add",
            damped_trend=True,
            seasonal="add" if use_season else None,
            seasonal_periods=season if use_season else None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # statsmodels prints many harmless fitting notes
            self.fitted = model.fit(optimized=True)

    def _predict(self, n_steps):
        return np.asarray(self.fitted.forecast(n_steps))


class ArimaModel(ForecastModel):
    name = "arima"
    family = "statistical"

    def _fit(self, history):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        season = SEASON_LENGTH[self.frequency]
        seasonal_order = (0, 0, 0, 0)
        if self.frequency != "weekly" and len(history) >= 2 * season:
            seasonal_order = (1, 0, 1, season)
        model = SARIMAX(history.to_numpy(dtype=float), order=(1, 1, 1),
                        seasonal_order=seasonal_order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted = model.fit(disp=False)

    def _predict(self, n_steps):
        return np.asarray(self.fitted.forecast(n_steps))


def statistical_models(frequency: str) -> list:
    return [ExponentialSmoothingModel(frequency), ArimaModel(frequency)]
