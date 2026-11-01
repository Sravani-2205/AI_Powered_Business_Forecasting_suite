"""
prophet_model.py
Prophet (made by Meta), the first model in the pipeline that knows WHEN festivals happen.

Prophet adds up simple parts:
    sales = trend + weekly pattern + yearly pattern + festival effects

Festivals:
  Each selected festival is given a window from 7 days before to 1 day after the date,
  so Prophet can learn the shopping build-up (not just the day itself). Because the
  dates come from the calendar, Prophet knows that Diwali moves every year, which the
  earlier models could not know.
  Festivals are only used with DAILY data: in weekly or monthly totals a single day is
  hidden inside the period.

Yearly pattern:
  Prophet silently switches its yearly pattern OFF when it has less than 2 years of data
  (we were caught by this before). Here it is switched ON whenever there is at least one
  year of training data, and OFF otherwise.
"""

import logging

import numpy as np
import pandas as pd

from forecaster.calendar import DAYS_AFTER, DAYS_BEFORE
from forecaster.models.base import ForecastModel

# Prophet's helper engine (cmdstanpy) prints progress lines on every fit. Silence it,
# but keep Prophet's own messages, which warn about things like switched-off patterns.
logging.getLogger("cmdstanpy").disabled = True


class ProphetModel(ForecastModel):
    name = "prophet"
    family = "prophet"

    def __init__(self, frequency: str, calendar: pd.DataFrame | None = None):
        super().__init__(frequency)
        self.calendar = calendar if frequency == "daily" else None
        if self.calendar is not None and len(self.calendar):
            self.name = "prophet_with_festivals"

    def _fit(self, history):
        from prophet import Prophet

        years_of_history = (history.index[-1] - history.index[0]).days / 365.25
        holidays = None
        if self.calendar is not None and len(self.calendar):
            holidays = pd.DataFrame({
                "holiday": self.calendar["name"].to_numpy(),
                "ds": self.calendar["date"].to_numpy(),
                "lower_window": -DAYS_BEFORE,
                "upper_window": DAYS_AFTER,
            })
        self.model = Prophet(
            yearly_seasonality=years_of_history >= 1,
            weekly_seasonality=self.frequency == "daily",
            daily_seasonality=False,
            holidays=holidays,
        )
        self.model.fit(pd.DataFrame({"ds": history.index, "y": history.to_numpy(dtype=float)}))

    def _predict(self, n_steps):
        future = pd.DataFrame({"ds": self.future_dates(n_steps)})
        prediction = self.model.predict(future)["yhat"].to_numpy()
        return np.clip(prediction, 0, None)       # sales cannot be negative


def prophet_models(frequency: str, calendar: pd.DataFrame | None = None) -> list:
    return [ProphetModel(frequency, calendar)]
