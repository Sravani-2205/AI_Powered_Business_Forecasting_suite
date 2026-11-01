"""
charts.py
Interactive Plotly charts for the app (hover for exact values, drag to zoom,
double-click to zoom back out). Colours match the app's dark theme.
"""

import pandas as pd
import plotly.graph_objects as go

# The app's palette
TEXT = "#E6E9EF"
MUTED = "#8A94A7"
GRID = "#263042"
ACTUAL = "#AEB6C6"      # past sales: soft grey
SMOOTH = "#E6E9EF"
FUTURE = "#F4A63A"      # marigold: the forecast people care about (the one warm colour)
TEST = "#5B9BF0"        # the app's blue: the final test
BACKTEST = "#7C869B"    # muted slate: backtest windows
HOLIDAY_COLORS = ["#F4A63A", "#5B9BF0", "#E0789A", "#59C3A6", "#B79CE8",
                  "#E8C46A", "#7FB8E6", "#D98E6A", "#9AD1C9", "#C7CEDB"]


def _style(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=TEXT), x=0, xanchor="left"),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=14, color=MUTED),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickformat=",.0f")
    return fig


def history_figure(series: pd.Series, title: str, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="sales",
                             line=dict(color=ACTUAL, width=0.8), opacity=0.45))
    if len(series) > 60:
        smooth = series.rolling(28, center=True, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=smooth.index, y=smooth.values, name="28-period average",
                                 line=dict(color=SMOOTH, width=2)))
    fig = _style(fig, title, height)
    if not title:                                   # small preview: no title, tighter margins
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def holidays_figure(series: pd.Series, calendar: pd.DataFrame, names: list, title: str) -> go.Figure:
    fig = go.Figure()
    smooth = series.rolling(7, center=True, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="daily sales",
                             line=dict(color=ACTUAL, width=0.8), opacity=0.35))
    fig.add_trace(go.Scatter(x=smooth.index, y=smooth.values, name="7-day average",
                             line=dict(color=SMOOTH, width=1.6)))
    top = float(series.max())
    for i, name in enumerate(names[:10]):
        color = HOLIDAY_COLORS[i % len(HOLIDAY_COLORS)]
        days = calendar.loc[calendar["name"] == name, "date"]
        days = days[(days >= series.index.min()) & (days <= series.index.max())]
        # One vertical line per year, drawn as a single trace so it has one legend entry.
        xs, ys = [], []
        for day in days:
            xs += [day, day, None]
            ys += [0, top, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name,
                                 line=dict(color=color, width=1.4), hoverinfo="skip"))
    return _style(fig, title, 380)


def forecast_figure(series: pd.Series, result, title: str) -> go.Figure:
    fig = go.Figure()
    shown = series.loc[result.backtest_forecast.index[0]:]
    fig.add_vrect(x0=result.test.index[0], x1=result.test.index[-1],
                  fillcolor=TEST, opacity=0.08, line_width=0,
                  annotation_text="final test", annotation_position="top left",
                  annotation_font_color=TEST)
    fig.add_trace(go.Scatter(x=shown.index, y=shown.values, name="actual sales",
                             line=dict(color=ACTUAL, width=1.2)))
    fig.add_trace(go.Scatter(x=result.backtest_forecast.index, y=result.backtest_forecast.values,
                             name="backtest forecasts", line=dict(color=BACKTEST, width=1.4, dash="dot")))
    fig.add_trace(go.Scatter(x=result.test_forecast.index, y=result.test_forecast.values,
                             name="final test forecast", line=dict(color=TEST, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=result.future_forecast.index, y=result.future_forecast.values,
                             name="future forecast", line=dict(color=FUTURE, width=2.6)))
    return _style(fig, title, 460)


# ---------------------------------------------------------------- business story and plan charts
POSITIVE = "#59C3A6"
NEGATIVE = "#E0697A"
ACCENT = "#5B9BF0"


def percent_bars(values: pd.Series, title: str, height: int = 300) -> go.Figure:
    """Bars above / below zero, e.g. '% vs an average day' for each weekday."""
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in values.to_numpy()]
    fig = go.Figure(go.Bar(x=list(values.index), y=values.to_numpy(), marker_color=colors,
                           text=[f"{v:+.0f}%" for v in values.to_numpy()], textposition="outside",
                           hovertemplate="%{x}: %{y:+.1f}%<extra></extra>"))
    fig = _style(fig, title, height)
    fig.update_yaxes(ticksuffix="%", tickformat="+.0f", zeroline=True, zerolinecolor=MUTED)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def money_bars(labels, values, title: str, color: str = ACCENT, height: int = 300,
               symbol: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(x=list(labels), y=list(values), marker_color=color,
                           hovertemplate="%{x}: " + symbol + "%{y:,.0f}<extra></extra>"))
    fig = _style(fig, title, height)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_yaxes(tickprefix=symbol)
    return fig


def real_vs_money_figure(table: pd.DataFrame, title: str, symbol: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table.index, y=table["as_recorded"], name="as recorded",
                             line=dict(color=ACTUAL, width=1.6)))
    fig.add_trace(go.Scatter(x=table.index, y=table["in_todays_prices"], name="in today's prices",
                             line=dict(color=FUTURE, width=2)))
    fig = _style(fig, title, 340)
    fig.update_yaxes(tickprefix=symbol)
    return fig


def weekly_plan_figure(plan: pd.DataFrame, title: str, symbol: str = "") -> go.Figure:
    """One bar per week, coloured by how busy it is. All bars are ONE trace, so they all get the
    same width (separate traces with a single bar each come out as hair-thin lines)."""
    colors = {"Busy": FUTURE, "Normal": ACCENT, "Quiet": "#7C869B"}
    fig = go.Figure(go.Bar(
        x=plan["week_starting"], y=plan["expected"], marker_color=[colors[b] for b in plan["how_busy"]],
        customdata=plan["how_busy"], showlegend=False,
        error_y=dict(type="data", symmetric=False, array=plan["high"] - plan["expected"],
                     arrayminus=plan["expected"] - plan["low"], color=MUTED, thickness=1),
        hovertemplate="Week of %{x|%d %b}: " + symbol + "%{y:,.0f}<extra>%{customdata}</extra>"))
    for level, color in colors.items():               # legend entries only
        if (plan["how_busy"] == level).any():
            fig.add_trace(go.Bar(x=[None], y=[None], name=level, marker_color=color))
    fig = _style(fig, title, 340)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(tickprefix=symbol)
    return fig


def cash_figure(cash: pd.DataFrame, title: str, symbol: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=cash["month"], y=cash["expected"], marker_color=ACCENT,
        error_y=dict(type="data", symmetric=False, array=cash["high"] - cash["expected"],
                     arrayminus=cash["expected"] - cash["low"], color=MUTED, thickness=1.2),
        hovertemplate="%{x}: " + symbol + "%{y:,.0f}<extra></extra>"))
    fig = _style(fig, title, 300)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_yaxes(tickprefix=symbol)
    return fig


def scenario_figure(base: pd.Series, adjusted: pd.Series, title: str, symbol: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base.index, y=base.values, name="forecast as it is",
                             line=dict(color=ACTUAL, width=1.4)))
    fig.add_trace(go.Scatter(x=adjusted.index, y=adjusted.values, name="with your assumptions",
                             line=dict(color=FUTURE, width=2.2)))
    fig = _style(fig, title, 380)
    fig.update_yaxes(tickprefix=symbol)
    return fig


def tracking_figure(table: pd.DataFrame, title: str, symbol: str = "") -> go.Figure:
    fig = go.Figure()
    x = table["period_starting"]
    fig.add_trace(go.Scatter(x=x, y=table["high"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=table["low"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(91,155,240,0.15)", name="expected range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=table["expected"], name="forecast", line=dict(color=ACCENT, width=1.6, dash="dash")))
    colors = [NEGATIVE if s == "Below range" else (POSITIVE if s == "Above range" else TEXT) for s in table["status"]]
    fig.add_trace(go.Scatter(x=x, y=table["actual"], name="actual", mode="lines+markers",
                             line=dict(color=TEXT, width=2), marker=dict(color=colors, size=9)))
    fig = _style(fig, title, 380)
    fig.update_yaxes(tickprefix=symbol)
    return fig
