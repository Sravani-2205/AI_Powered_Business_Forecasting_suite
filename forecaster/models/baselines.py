"""
baselines.py
Simple "common sense" forecasts. Every real model must beat these to be worth using.
They work for daily, weekly and monthly data.
"""

import numpy as np

from forecaster.models.base import PERIODS_PER_YEAR, ForecastModel


class LastValue(ForecastModel):
    """Tomorrow will be like today: repeat the last known value."""
    name = "last_value"
    family = "baselines"

    def _predict(self, n_steps):
        return np.full(n_steps, self.history.iloc[-1])


class LastWeekRepeated(ForecastModel):
    """Daily data only: repeat the last 7 days, so the weekly pattern is kept."""
    name = "last_week_repeated"
    family = "baselines"

    def _predict(self, n_steps):
        last_week = self.history.iloc[-7:].to_numpy()
        return np.resize(last_week, n_steps)   # repeats the 7 values until n_steps long


class SameTimeLastYear(ForecastModel):
    """Copy the value from exactly one year earlier (52 weeks for daily data,
    so a Monday is compared with a Monday)."""
    name = "same_time_last_year"
    family = "baselines"

    def _predict(self, n_steps):
        season = PERIODS_PER_YEAR[self.frequency]
        if len(self.history) < season:
            raise ValueError("Needs at least one year of history.")
        values = self.history.to_numpy()
        # Step i of the forecast copies the value `season` periods before it.
        # If the forecast is longer than a year, it keeps repeating the last year.
        last_year = values[-season:]
        return np.resize(last_year, n_steps)


class RecentAverage(ForecastModel):
    """The average of the most recent stretch: last 4 weeks (daily),
    last 8 weeks (weekly) or last 6 months (monthly)."""
    name = "recent_average"
    family = "baselines"
    WINDOW = {"daily": 28, "weekly": 8, "monthly": 6}

    def _predict(self, n_steps):
        return np.full(n_steps, self.history.iloc[-self.WINDOW[self.frequency]:].mean())


def baseline_models(frequency: str) -> list:
    models = [LastValue, RecentAverage, SameTimeLastYear]
    if frequency == "daily":
        models.append(LastWeekRepeated)
    return [m(frequency) for m in models]
