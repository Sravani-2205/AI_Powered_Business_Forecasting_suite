"""
competition.py
Pipeline steps 4-7: backtest every model over a full year, pick the winner, score it
on a final unseen test period, and forecast the future.

WHY BACKTESTING?
  With only ONE validation window, a model can look good just because that window was a
  quiet period (we saw this: the sweet shop was judged on July-September, then had to
  forecast Diwali). So instead, every model is tested on SEVERAL windows that together
  cover about a year, so every season and festival is included:

    fold 1: train ───────────────┤ forecast ├
    fold 2: train ──────────────────────────┤ forecast ├
    fold 3: train ─────────────────────────────────────┤ forecast ├
    fold 4: train ────────────────────────────────────────────────┤ forecast ├
    final:  train ───────────────────────────────────────────────────────────┤ TEST ├

  Each fold only learns from data BEFORE its window (no peeking at the future).
  This is called "rolling-origin backtesting" or "time-series cross-validation".

  Step 4  Hold back the last horizon as the final TEST period. Nothing touches it yet.
  Step 5  Backtest every allowed model on the folds before it. Its score is calculated over
          all fold windows together. The lowest backtest MAE wins.
  Step 6  The winner learns from everything before the test period and forecasts it ONCE.
  Step 7  The winner learns from ALL the data and forecasts the real future.

If a model crashes, it is shown as FAILED with the reason, never given a fake score.
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecaster.data_check import DataReport
from forecaster.models import models_for
from forecaster.models.base import PERIODS_PER_YEAR


def score(actual: pd.Series, predicted: pd.Series) -> dict:
    """Error measures, lower is better.
      MAE  - average size of the error per period, in the same units as sales
      RMSE - like MAE, but big misses count extra
      WAPE - total error as a % of total sales ("off by about X%")
    """
    a = actual.to_numpy(dtype=float)
    p = predicted.to_numpy(dtype=float)
    if len(a) != len(p) or np.isnan(p).any():
        raise ValueError("Forecast has the wrong length or missing values.")
    errors = a - p
    total = np.sum(np.abs(a))
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "WAPE_%": float(np.sum(np.abs(errors)) / total * 100) if total > 0 else float("nan"),
    }


def plan_folds(n_points: int, horizon: int, frequency: str) -> list[int]:
    """Decide where each backtest window starts (as positions in the series).

    Aim: enough windows to cover about one year. Every fold must keep at least one
    year of training data before its window, so with little data there are fewer folds.
    """
    year = PERIODS_PER_YEAR[frequency]
    wanted = max(1, round(year / horizon))          # daily: 4 x 90 days, weekly: 4 x 13, monthly: 1 x 12
    test_start = n_points - horizon
    starts = []
    for k in range(wanted, 0, -1):
        start = test_start - k * horizon
        if start >= year:                           # at least one year of training before it
            starts.append(start)
    if not starts:
        raise ValueError("Too little data for even one backtest window plus a test period.")
    return starts


@dataclass
class CompetitionResult:
    leaderboard: pd.DataFrame          # backtest scores of every model
    winner: str
    winner_fold_scores: pd.DataFrame   # the winner's score in each backtest window
    backtest_forecast: pd.Series       # the winner's forecasts in all backtest windows
    test_scores: dict                  # the winner's score on the final test period
    test_forecast: pd.Series
    future_forecast: pd.Series
    test: pd.Series
    n_folds: int
    covers_full_year: bool
    not_built: list


def backtest(model, series: pd.Series, fold_starts: list[int], horizon: int):
    """Run one model through every fold. Returns (forecasts of all windows, per-fold scores)."""
    forecasts, fold_rows = [], []
    for i, start in enumerate(fold_starts, 1):
        train = series.iloc[:start]
        actual = series.iloc[start:start + horizon]
        forecast = model.fit(train).predict(horizon)
        forecast.index = actual.index
        forecasts.append(forecast)
        fold_rows.append({"fold": i, "from": actual.index[0].date(), "to": actual.index[-1].date(),
                          **score(actual, forecast)})
    return pd.concat(forecasts), pd.DataFrame(fold_rows).set_index("fold")


def run_competition(series: pd.Series, report: DataReport, calendar=None,
                    families: list | None = None, on_progress=None) -> CompetitionResult:
    """calendar:    the selected festival dates (or None). Only festival-aware models use it.
    families:    which model families to run (default: all the tier allows).
    on_progress: optional function(message, fraction_done), e.g. to move a progress bar."""
    if report.tier == "too_small":
        raise ValueError(report.message)

    h = report.horizon
    fold_starts = plan_folds(len(series), h, report.frequency)
    test = series.iloc[-h:]
    models, not_built = models_for(families or report.allowed_families, report.frequency, calendar)

    # Step 5: backtest every model.
    rows, results = [], {}
    for i, model in enumerate(models):
        if on_progress:
            on_progress(f"Backtesting {model.name} ({i + 1} of {len(models)})", i / len(models))
        started = time.perf_counter()
        try:
            forecasts, folds = backtest(model, series, fold_starts, h)
            rows.append({"model": model.name, "family": model.family, "status": "ok",
                         **score(series.loc[forecasts.index], forecasts),
                         "seconds": round(time.perf_counter() - started, 1)})
            results[model.name] = (model, forecasts, folds)
        except Exception as error:
            rows.append({"model": model.name, "family": model.family,
                         "status": f"FAILED: {type(error).__name__}: {error}",
                         "seconds": round(time.perf_counter() - started, 1)})
    leaderboard = pd.DataFrame(rows).set_index("model")
    ok = leaderboard[leaderboard["status"] == "ok"]
    if ok.empty:
        raise RuntimeError("Every model failed, so no forecast can be made.")
    leaderboard = pd.concat([ok.sort_values("MAE"), leaderboard[leaderboard["status"] != "ok"]])
    winner_name = ok["MAE"].idxmin()
    winner, backtest_forecast, winner_folds = results[winner_name]

    if on_progress:
        on_progress(f"Final test and future forecast with {winner_name}", 1.0)

    # Step 6: the winner forecasts the final test period ONCE.
    test_forecast = winner.fit(series.iloc[:-h]).predict(h)
    test_scores = score(test, test_forecast)

    # Step 7: the winner learns from everything and forecasts the real future.
    future_forecast = winner.fit(series).predict(h)

    covered = len(fold_starts) * h + h          # backtest windows + test period
    return CompetitionResult(
        leaderboard=leaderboard,
        winner=winner_name,
        winner_fold_scores=winner_folds,
        backtest_forecast=backtest_forecast,
        test_scores=test_scores,
        test_forecast=test_forecast,
        future_forecast=future_forecast,
        test=test,
        n_folds=len(fold_starts),
        covers_full_year=covered >= PERIODS_PER_YEAR[report.frequency],
        not_built=not_built,
    )
