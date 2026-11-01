"""
plots.py
Charts shared by the command-line scripts and the app.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from forecaster.competition import CompetitionResult


def forecast_chart(series: pd.Series, result: CompetitionResult, title: str):
    """Everything the winner was judged on (backtest windows + final test), plus the future."""
    fig, ax = plt.subplots(figsize=(13, 5))
    shown = series.loc[result.backtest_forecast.index[0]:]
    ax.plot(shown.index, shown.values, color="black", linewidth=1, label="actual sales")
    ax.plot(result.backtest_forecast.index, result.backtest_forecast.values, color="tab:orange",
            linestyle=":", label="backtest forecasts")
    ax.plot(result.test_forecast.index, result.test_forecast.values, color="tab:blue",
            linestyle="--", label="final test forecast")
    ax.plot(result.future_forecast.index, result.future_forecast.values, color="tab:green",
            label="future forecast")
    ax.axvspan(result.test.index[0], result.test.index[-1], color="tab:blue", alpha=0.07)
    ax.set_title(title)
    ax.set_ylabel("Sales")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def holidays_chart(series: pd.Series, calendar: pd.DataFrame, names: list, title: str):
    """The sales history with the chosen holidays drawn as coloured lines."""
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(series.index, series.values, color="lightgray", linewidth=0.6, label="daily sales")
    ax.plot(series.index, series.rolling(7, center=True).mean(), color="black", linewidth=1,
            label="7-day average")
    for i, name in enumerate(names[:10]):
        days = calendar.loc[calendar["name"] == name, "date"]
        days = days[(days >= series.index.min()) & (days <= series.index.max())]
        for j, day in enumerate(days):
            ax.axvline(day, color=colors(i % 10), linewidth=1, alpha=0.8, label=name if j == 0 else None)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def history_chart(series: pd.Series, title: str):
    """A compact chart of the whole sales history (daily values plus a smoothed line)."""
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(series.index, series.values, color="lightgray", linewidth=0.6, label="sales")
    if len(series) > 60:          # long series: add a smoothed line so the trend is visible
        ax.plot(series.index, series.rolling(28, center=True, min_periods=1).mean(),
                color="black", linewidth=1, label="28-period average")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig
