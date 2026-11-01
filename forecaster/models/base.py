"""
base.py
The common "shape" every model in the pipeline must have.

Every model, from the simplest baseline to the LSTM, has exactly two jobs:
    fit(history)        - learn from a series of past sales
    predict(n_steps)    - forecast the next n_steps periods after the history

Because all models look the same from the outside, the competition code can treat them
all identically: it never needs to know how a model works inside. Adding a new model
later just means writing one new class with these two methods.
"""

import pandas as pd

# How to step forward in time for each frequency ("MS" = month start).
PANDAS_FREQ = {"daily": "D", "weekly": "7D", "monthly": "MS"}

# How many periods make one week / one year at each frequency.
PERIODS_PER_YEAR = {"daily": 364, "weekly": 52, "monthly": 12}   # 364 days = 52 whole weeks


class ForecastModel:
    """Parent class. Real models fill in _fit and _predict."""

    name = "unnamed"
    family = "unknown"

    def __init__(self, frequency: str):
        self.frequency = frequency
        self.history = None

    def fit(self, history: pd.Series) -> "ForecastModel":
        self.history = history
        self._fit(history)
        return self

    def predict(self, n_steps: int) -> pd.Series:
        if self.history is None:
            raise RuntimeError(f"{self.name}: call fit() before predict().")
        values = self._predict(n_steps)
        dates = self.future_dates(n_steps)
        dates.name = "date"
        return pd.Series(values, index=dates, name=self.name)

    def future_dates(self, n_steps: int) -> pd.DatetimeIndex:
        """The n_steps dates that come right after the end of the history."""
        dates = pd.date_range(self.history.index[-1], periods=n_steps + 1,
                              freq=PANDAS_FREQ[self.frequency])
        return dates[1:]

    # Children replace these two:
    def _fit(self, history: pd.Series) -> None:
        pass

    def _predict(self, n_steps: int):
        raise NotImplementedError
