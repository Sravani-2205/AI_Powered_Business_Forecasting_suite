"""
machine_learning.py
Machine-learning models that learn from a table of "facts about each day".

How it works:
  Every period becomes one row of a table. The columns ("features") are facts that are
  KNOWN IN ADVANCE, so they are available for every day we have to forecast:
    - calendar facts: weekday, day of month, week of year, month
    - time since the start of the data (lets a model notice growth)
    - recent sales level: the average of the 4 weeks (or 8 weeks / 6 months) that ended
      one full horizon earlier. Because it looks back at least a whole horizon, it is
      known on the day the forecast is made, for every day being forecast.
    - festival facts (daily data only), for each selected festival:
        days until it   (counts down from 30 to 0)
        days since it   (counts up from 0 to 10)
      This lets a model learn "sales climb in the 10 days before Diwali".
  The model then learns how sales depend on those facts.

  It never uses yesterday's sales, because in a 90-day forecast yesterday's sales are
  unknown for most of the period (that would be data leakage).

Three models, from simple to powerful:
  linear_regression  - adds up a weight for each fact. Fast and easy to explain, but it
                       cannot learn "only in the last 3 days before Diwali" style shapes.
  random_forest      - hundreds of decision trees, each trained on a random sample, averaged.
  gradient_boosting  - hundreds of small trees, each one correcting the previous ones' mistakes.
"""

import numpy as np
import pandas as pd

from forecaster.data_check import HORIZON
from forecaster.models.base import ForecastModel

LEVEL_WINDOW = {"daily": 28, "weekly": 8, "monthly": 6}
DAYS_UNTIL_CAP = 30
DAYS_SINCE_CAP = 10


def festival_features(dates: pd.DatetimeIndex, calendar: pd.DataFrame) -> pd.DataFrame:
    """For each festival: days until the next one, and days since the last one (capped)."""
    out = {}
    for name, group in calendar.groupby("name"):
        fest = np.sort(group["date"].to_numpy(dtype="datetime64[D]"))
        d = dates.to_numpy(dtype="datetime64[D]")
        nxt = np.searchsorted(fest, d, side="left")          # first festival on or after each day
        until = np.full(len(d), DAYS_UNTIL_CAP)
        has_next = nxt < len(fest)
        until[has_next] = np.minimum((fest[nxt[has_next]] - d[has_next]).astype(int), DAYS_UNTIL_CAP)
        prev = np.searchsorted(fest, d, side="right") - 1    # last festival on or before each day
        since = np.full(len(d), DAYS_SINCE_CAP)
        has_prev = prev >= 0
        since[has_prev] = np.minimum((d[has_prev] - fest[prev[has_prev]]).astype(int), DAYS_SINCE_CAP)
        key = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
        out[f"until_{key}"] = until
        out[f"since_{key}"] = since
    return pd.DataFrame(out, index=dates)


class FeatureModel(ForecastModel):
    """Shared feature-building for all machine-learning models."""
    family = "machine_learning"

    def __init__(self, frequency: str, calendar: pd.DataFrame | None = None):
        super().__init__(frequency)
        self.calendar = calendar if (frequency == "daily" and calendar is not None and len(calendar)) else None
        self.horizon = HORIZON[frequency]

    def make_features(self, values: pd.Series) -> pd.DataFrame:
        """Features for every date in `values` (future dates may hold NaN sales)."""
        idx = values.index
        X = pd.DataFrame(index=idx)
        X["month"] = idx.month
        if self.frequency != "monthly":
            X["week_of_year"] = idx.isocalendar().week.astype(int).to_numpy()
        if self.frequency == "daily":
            X["weekday"] = idx.day_of_week
            X["day_of_month"] = idx.day
        X["time_index"] = np.arange(len(idx))
        window = LEVEL_WINDOW[self.frequency]
        # Average of the window that ended one full horizon earlier: known at forecast time.
        X["recent_level"] = values.rolling(window, min_periods=window // 2).mean().shift(self.horizon)
        if self.calendar is not None:
            X = X.join(festival_features(idx, self.calendar))
        return X

    def _fit(self, history):
        X = self.make_features(history)
        usable = X["recent_level"].notna()          # the first few weeks have no "recent level" yet
        self.columns = list(X.columns)
        self.estimator = self.build_estimator()
        self.estimator.fit(self.prepare(X[usable]), history[usable].to_numpy(dtype=float))

    def _predict(self, n_steps):
        if n_steps > self.horizon:
            raise ValueError(f"This model can forecast at most {self.horizon} periods ahead.")
        future_index = self.future_dates(n_steps)
        values = pd.concat([self.history, pd.Series(np.nan, index=future_index)])
        X = self.make_features(values).loc[future_index, self.columns]
        return np.clip(self.estimator.predict(self.prepare(X)), 0, None)

    # Children choose the estimator; linear models also need the table prepared differently.
    def build_estimator(self):
        raise NotImplementedError

    def prepare(self, X: pd.DataFrame):
        return X


class GradientBoostingModel(FeatureModel):
    name = "gradient_boosting"

    def build_estimator(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             loss="absolute_error", random_state=42)


class RandomForestModel(FeatureModel):
    name = "random_forest"

    def build_estimator(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=42)


class LinearRegressionModel(FeatureModel):
    name = "linear_regression"

    def build_estimator(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        # Ridge = linear regression that keeps weights small, so it does not overreact.
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    def prepare(self, X: pd.DataFrame):
        # A linear model cannot treat "weekday 6" as a category on its own, so each
        # weekday and month gets its own yes/no column ("one-hot encoding").
        X = X.copy()
        for col, values in [("weekday", range(7)), ("month", range(1, 13))]:
            if col in X:
                for v in values:
                    X[f"{col}_{v}"] = (X[col] == v).astype(int)
                X = X.drop(columns=col)
        return X


def machine_learning_models(frequency: str, calendar: pd.DataFrame | None = None) -> list:
    return [GradientBoostingModel(frequency, calendar), RandomForestModel(frequency, calendar),
            LinearRegressionModel(frequency, calendar)]
