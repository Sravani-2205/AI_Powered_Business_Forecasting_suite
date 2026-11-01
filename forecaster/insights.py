"""
insights.py
Turns a business's sales history and its forecast into plain-language findings and plans.

Everything here is CALCULATED from the business's own data; nothing is invented.
Suggestions are simple rules of thumb tied to a number, not expert business advice.

Sections:
  growth_summary      - how the business is doing (last 12 months vs the 12 before, yearly totals)
  shocks              - long closures (e.g. a lockdown): estimated lost sales and recovery time
  weekday_pattern     - which weekdays sell more or less than average (daily data)
  month_pattern       - which months sell more or less than average
  festival_value      - extra sales each festival brings, in money, per year
  outlook             - next period's expected total vs the same period last year
  forecast_bias       - whether the model tended to forecast too high or too low in the backtest
  weekly_plan         - busy/normal/quiet weeks ahead, with stock to prepare
  monthly_cash        - expected money in, month by month, with a low/high range
  suggestions         - rules of thumb, each backed by one of the numbers above
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecaster.calendar import DAYS_AFTER, DAYS_BEFORE
from forecaster.models.base import PERIODS_PER_YEAR

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------- money formatting
def money(x: float, symbol: str = "₹") -> str:
    """Readable money. Rupees use lakh and crore (1 lakh = 100,000; 1 crore = 100 lakh)."""
    sign = "-" if x < 0 else ""
    x = abs(float(x))
    if symbol == "₹":
        if x >= 1e7:
            return f"{sign}₹{x / 1e7:.2f} crore"
        if x >= 1e5:
            return f"{sign}₹{x / 1e5:.1f} lakh"
        return f"{sign}₹{x:,.0f}"
    if x >= 1e6:
        return f"{sign}{symbol}{x / 1e6:.2f}M"
    return f"{sign}{symbol}{x:,.0f}"


# ---------------------------------------------------------------- 1. how the business is doing
@dataclass
class Growth:
    last_12: float
    previous_12: float
    change_pct: float
    yearly: pd.Series          # total per calendar year (complete years only)


def growth_summary(series: pd.Series, frequency: str) -> Growth | None:
    year = {"daily": 365, "weekly": 52, "monthly": 12}[frequency]
    if len(series) < 2 * year:
        return None
    last = float(series.iloc[-year:].sum())
    prev = float(series.iloc[-2 * year:-year].sum())
    per_year = series.groupby(series.index.year).agg(["sum", "count"])
    full = per_year[per_year["count"] >= year * 0.97]["sum"]          # complete years only
    return Growth(last, prev, (last / prev - 1) * 100 if prev else float("nan"), full)


@dataclass
class Shock:
    start: pd.Timestamp
    end: pd.Timestamp
    days_closed: int
    yearly: bool                  # True = a closure that happens every year (e.g. a Christmas shutdown)
    lost_sales: float | None      # only estimated for one-off shocks
    recovery_days: int | None     # days until sales were back to 90% of normal (None = not within a year)


def _closed_runs(series: pd.Series, min_days: int) -> list[tuple[int, int]]:
    closed = (series <= 0).to_numpy()
    runs, i = [], 0
    while i < len(closed):
        if closed[i]:
            j = i
            while j < len(closed) and closed[j]:
                j += 1
            if j - i >= min_days:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def shocks(series: pd.Series, frequency: str, min_days: int = 7) -> list[Shock]:
    """Runs of at least `min_days` closed days in a row (daily data only).
    A closure that also happens around the same dates in another year is a planned YEARLY
    closure, not a shock, so no "lost sales" are estimated for it."""
    if frequency != "daily":
        return []
    runs = _closed_runs(series, min_days)
    found = []
    for i, j in runs:
        days = series.index[i:j]
        yearly = False
        for k, l in runs:
            if (k, l) == (i, j):
                continue
            other = series.index[k:l]
            # same time of year (within 5 days) in a different year?
            if any(abs((d.replace(year=2000) - o.replace(year=2000)).days) <= 5
                   for d in days[:3] for o in other[:3] if not (d.month == 2 and d.day == 29)
                   and not (o.month == 2 and o.day == 29)):
                yearly = True
                break
        lost, recovery = None, None
        if not yearly and i >= 28:
            before = series.iloc[i - 28:i]
            normal = float(before[before > 0].median())
            after = series.iloc[j:j + 365].rolling(28, min_periods=28).mean()
            back = after[after >= 0.9 * normal]
            if len(back):
                recovery = int((back.index[0] - series.index[j]).days)
            reopen = series.iloc[j:j + (recovery or 0)]
            lost = normal * (j - i) + float((normal - reopen).clip(lower=0).sum())
        found.append(Shock(series.index[i], series.index[j - 1], j - i, yearly, lost, recovery))
    return found


# ---------------------------------------------------------------- 2. when customers buy
def weekday_pattern(series: pd.Series, frequency: str) -> pd.Series | None:
    """% above or below the average open day, for each weekday (daily data only)."""
    if frequency != "daily":
        return None
    recent = series.iloc[-365:]
    open_days = recent[recent > 0]
    by_day = open_days.groupby(open_days.index.day_of_week).mean()
    pattern = (by_day / open_days.mean() - 1) * 100
    pattern.index = [WEEKDAYS[d] for d in pattern.index]
    return pattern


def month_pattern(series: pd.Series) -> pd.Series:
    """% above or below an average month, typical across the complete years in the data."""
    monthly = series.resample("MS").sum()
    counts = series.resample("MS").count()
    expected = monthly.index.days_in_month if len(series) > 400 else counts
    monthly = monthly[counts >= np.asarray(expected) * 0.9]           # complete months only
    rows = []
    for year, months in monthly.groupby(monthly.index.year):
        if len(months) == 12 and (months > 0).all():
            rows.append(months.to_numpy() / months.mean() - 1)
    if not rows:
        return pd.Series(dtype=float)
    typical = np.median(np.array(rows), axis=0) * 100                 # median: not thrown by one odd year
    return pd.Series(typical, index=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])


# ---------------------------------------------------------------- 3. what drives the peaks
def festival_value(series: pd.Series, calendar: pd.DataFrame, names: list) -> pd.DataFrame:
    """For each festival: extra sales (above a normal day) in the week around it, per year,
    averaged over its last 3 appearances so it reflects the business's current size."""
    rows = []
    for name in names:
        extras = []
        for day in sorted(calendar.loc[calendar["name"] == name, "date"]):
            window = series.loc[day - pd.Timedelta(days=DAYS_BEFORE): day + pd.Timedelta(days=DAYS_AFTER)]
            normal = pd.concat([
                series.loc[day - pd.Timedelta(days=35): day - pd.Timedelta(days=DAYS_BEFORE + 1)],
                series.loc[day + pd.Timedelta(days=DAYS_AFTER + 1): day + pd.Timedelta(days=29)]])
            normal = normal[normal > 0]
            if len(window) < DAYS_BEFORE + DAYS_AFTER + 1 or len(normal) < 20 or (window == 0).mean() > 0.5:
                continue
            extras.append((float((window - normal.median()).sum()), float(window.max() / normal.median())))
        if extras:
            recent = extras[-3:]
            rows.append({"festival": name,
                         "extra_sales_per_year": float(np.mean([e for e, _ in recent])),
                         "peak_day_vs_normal": float(np.mean([p for _, p in recent]))})
    if not rows:
        return pd.DataFrame(columns=["festival", "extra_sales_per_year", "peak_day_vs_normal"])
    return pd.DataFrame(rows).sort_values("extra_sales_per_year", ascending=False).reset_index(drop=True)


def festival_share(series: pd.Series, calendar: pd.DataFrame, names: list) -> float | None:
    """Share of the last 12 months' sales that fell inside the selected festival windows."""
    last = series.iloc[-365:]
    if not names or len(last) < 365:
        return None
    inside = pd.Series(False, index=last.index)
    for day in calendar.loc[calendar["name"].isin(names), "date"]:
        inside |= (last.index >= day - pd.Timedelta(days=DAYS_BEFORE)) & \
                  (last.index <= day + pd.Timedelta(days=DAYS_AFTER))
    total = last.sum()
    return float(last[inside].sum() / total * 100) if total > 0 else None


# ---------------------------------------------------------------- 4. what's coming
@dataclass
class Outlook:
    total: float
    low: float
    high: float
    same_period_last_year: float | None
    change_pct: float | None
    busiest: pd.Series            # the busiest forecast periods, biggest first


def outlook(series: pd.Series, future: pd.Series, frequency: str, wape_pct: float) -> Outlook:
    total = float(future.sum())
    band = wape_pct / 100
    lag = pd.Timedelta(days=364) if frequency == "daily" else None
    if frequency == "daily":
        past_index = future.index - lag
    else:
        past_index = future.index - pd.DateOffset(years=1)
        past_index = series.index[series.index.get_indexer(past_index, method="nearest")]
    past = series.reindex(past_index)
    last_year = float(past.sum()) if past.notna().all() else None
    change = (total / last_year - 1) * 100 if last_year else None
    return Outlook(total, total * (1 - band), total * (1 + band), last_year, change,
                   future.sort_values(ascending=False).head(5))


def forecast_bias(series: pd.Series, backtest_forecast: pd.Series) -> float:
    """How much the winning model over- (+) or under- (-) forecast in the backtest, in % of sales.
    A model can be accurate day to day yet still run a little low or high overall."""
    actual = series.loc[backtest_forecast.index]
    return float((backtest_forecast.sum() - actual.sum()) / actual.sum() * 100)


def weekly_plan(future: pd.Series, frequency: str, wape_pct: float,
                value_per_unit: float | None = None, buffer_pct: float | None = None) -> pd.DataFrame | None:
    """One row per week ahead: expected sales, a low-high range, how busy it is, and
    (if the average value of one unit is given) how many units to prepare."""
    if frequency == "monthly":
        return None
    weeks = future.resample("W-SUN").sum() if frequency == "daily" else future
    if frequency == "daily":
        days = future.resample("W-SUN").count()
        weeks = weeks[days == 7]                                 # full weeks only: a 3-day stub is not "quiet"
    typical = weeks.median()
    band = wape_pct / 100
    plan = pd.DataFrame({
        "week_starting": (weeks.index - pd.Timedelta(days=6)) if frequency == "daily" else weeks.index,
        "expected": weeks.to_numpy(),
        "low": weeks.to_numpy() * (1 - band),
        "high": weeks.to_numpy() * (1 + band),
    })
    ratio = plan["expected"] / typical
    plan["how_busy"] = np.select([ratio >= 1.2, ratio <= 0.9], ["Busy", "Quiet"], "Normal")
    if value_per_unit:
        buffer = (buffer_pct if buffer_pct is not None else wape_pct / 2) / 100
        plan["units_to_prepare"] = np.ceil(plan["expected"] / value_per_unit * (1 + buffer))
    return plan


def monthly_cash(future: pd.Series, wape_pct: float) -> pd.DataFrame:
    months = future.resample("MS").agg(["sum", "count"])
    band = wape_pct / 100
    return pd.DataFrame({"month": months.index.strftime("%b %Y"), "days_covered": months["count"].to_numpy(),
                         "expected": months["sum"].to_numpy(),
                         "low": months["sum"].to_numpy() * (1 - band),
                         "high": months["sum"].to_numpy() * (1 + band)})


# ---------------------------------------------------------------- suggestions
def suggestions(series, future, frequency, calendar, names, wape_pct, symbol="₹",
                real_growth_pct: float | None = None, money_growth_pct: float | None = None) -> list[str]:
    """Rules of thumb. Each one names the number it comes from."""
    tips = []
    typical_day = float(future.median()) if len(future) else 0

    # Festivals inside the forecast period: when to start preparing
    if names and calendar is not None and frequency == "daily":
        upcoming = calendar[calendar["name"].isin(names) &
                            (calendar["date"] >= future.index[0]) & (calendar["date"] <= future.index[-1])]
        for _, row in upcoming.sort_values("date").iterrows():
            window = future.loc[row["date"] - pd.Timedelta(days=DAYS_BEFORE): row["date"] + pd.Timedelta(days=DAYS_AFTER)]
            if len(window) and typical_day > 0:
                start = (row["date"] - pd.Timedelta(days=10)).strftime("%d %b")
                tips.append(f"**{row['name']}** falls on {row['date'].strftime('%d %b %Y')}. The week around it "
                            f"is forecast at about **{money(window.sum(), symbol)}**, with its busiest day near "
                            f"**{window.max() / typical_day:.1f}×** a normal day. Start building stock and "
                            f"booking extra hands from around **{start}**.")

    # Weekday pattern: staffing and promotions
    pattern = weekday_pattern(series, frequency)
    if pattern is not None and len(pattern) >= 2:
        best, worst = pattern.idxmax(), pattern.idxmin()
        if pattern[best] > 10:
            tips.append(f"**{best}** sells **{pattern[best]:+.0f}%** compared with an average open day. "
                        f"That is the day to have your most experienced staff and fullest shelves.")
        if pattern[worst] < -10:
            tips.append(f"**{worst}** sells **{pattern[worst]:+.0f}%** compared with an average day. A small "
                        f"{worst} offer or a regular-customer reward could lift it; it is also the natural day "
                        f"for restocking and maintenance.")

    # Quiet weeks ahead: promotion windows
    plan = weekly_plan(future, frequency, wape_pct)
    if plan is not None:
        quiet = plan[plan["how_busy"] == "Quiet"]
        if len(quiet):
            dates = ", ".join(quiet["week_starting"].dt.strftime("%d %b").head(3))
            tips.append(f"The quietest weeks ahead start on **{dates}**. Offers work best in quiet weeks; "
                        f"during busy weeks customers come anyway, so discounts mostly give away margin.")

    # Seasonal dip: planning leave
    months = month_pattern(series)
    if len(months) and months.min() < -10:
        low = months[months < -10].index.tolist()
        tips.append(f"Sales are usually lowest in **{', '.join(low)}** (down to **{months.min():+.0f}%** "
                    f"against an average month). Plan staff leave, repairs and stock clear-outs for then.")

    # Inflation: is growth real?
    if real_growth_pct is not None and money_growth_pct is not None:
        if money_growth_pct > 0 and real_growth_pct <= 0:
            tips.append(f"Sales grew **{money_growth_pct:+.1f}%** in money, but **{real_growth_pct:+.1f}%** after "
                        f"inflation. The growth comes from higher prices, not more business. Watch customer "
                        f"numbers and quantities sold, not just takings.")
        elif real_growth_pct > 0:
            tips.append(f"After removing inflation, the business still grew **{real_growth_pct:+.1f}%** "
                        f"in the last year: real growth, not just higher prices.")
    return tips


# ---------------------------------------------------------------- what-if scenarios
def scenario(future: pd.Series, frequency: str, price_pct: float = 0.0, quantity_pct: float = 0.0,
             outlet_pct: float = 0.0, outlet_start: pd.Timestamp | None = None,
             promo_pct: float = 0.0, wape_pct: float = 0.0) -> pd.Series:
    """The forecast with the user's OWN assumptions applied. Nothing here is learned from data:
      price_pct     - prices change by this %
      quantity_pct  - customers buy this % more (+) or less (-) as a result
      outlet_pct    - a new outlet or channel adds this % of current sales, from outlet_start
      promo_pct     - offers during the quiet weeks lift those weeks by this %
    """
    adjusted = future * (1 + price_pct / 100) * (1 + quantity_pct / 100)
    if outlet_pct and outlet_start is not None:
        adjusted = adjusted.where(adjusted.index < outlet_start, adjusted * (1 + outlet_pct / 100))
    if promo_pct:
        plan = weekly_plan(future, frequency, wape_pct)
        if plan is not None:
            for start in plan.loc[plan["how_busy"] == "Quiet", "week_starting"]:
                end = start + (pd.Timedelta(days=6) if frequency == "daily" else pd.Timedelta(days=0))
                inside = (adjusted.index >= start) & (adjusted.index <= end)
                adjusted[inside] = adjusted[inside] * (1 + promo_pct / 100)
    return adjusted.rename("scenario")


# ---------------------------------------------------------------- tracking actual sales
def track(forecast: pd.Series, actual: pd.Series, frequency: str, band_pct: float) -> pd.DataFrame:
    """Compare real sales with the forecast, period by period (weekly for daily data).
    'Below' = under the forecast's low end, i.e. worse than the model's normal error explains."""
    common = forecast.index.intersection(actual.index)
    if len(common) == 0:
        return pd.DataFrame()
    f, a = forecast.loc[common], actual.loc[common]
    if frequency == "daily":
        days = f.resample("W-SUN").count()
        f, a = f.resample("W-SUN").sum()[days == 7], a.resample("W-SUN").sum()[days == 7]
        start = f.index - pd.Timedelta(days=6)
    else:
        start = f.index
    band = band_pct / 100
    table = pd.DataFrame({"period_starting": start, "expected": f.to_numpy(), "low": f.to_numpy() * (1 - band),
                          "high": f.to_numpy() * (1 + band), "actual": a.to_numpy()})
    table["difference_%"] = (table["actual"] / table["expected"] - 1) * 100
    table["status"] = np.select([table["actual"] < table["low"], table["actual"] > table["high"]],
                                ["Below range", "Above range"], "Within range")
    return table


def tracking_alerts(table: pd.DataFrame, period_word: str = "week") -> list[tuple[str, str]]:
    """(kind, message) pairs: early warnings when sales keep falling short of the forecast."""
    if table.empty:
        return []
    alerts = []
    below = (table["status"] == "Below range").to_numpy()
    run, worst_run, run_start = 0, 0, None
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run > worst_run:
            worst_run, run_start = run, table["period_starting"].iloc[i - run + 1]
    total_diff = (table["actual"].sum() / table["expected"].sum() - 1) * 100
    if worst_run >= 2:
        alerts.append(("error", f"Sales were below the expected range for **{worst_run} {period_word}s in a row**, "
                                f"starting {run_start.strftime('%d %b %Y')}. That is more than normal forecast "
                                f"error explains: worth checking what changed (stock, staff, a competitor, prices)."))
    elif below.any():
        alerts.append(("warn", f"**{int(below.sum())} {period_word}{'s' if below.sum() != 1 else ''}** fell below "
                               f"the expected range, but not in a row. Keep an eye on it."))
    above = int((table["status"] == "Above range").sum())
    if above:
        alerts.append(("ok", f"**{above} {period_word}{'s' if above != 1 else ''}** beat the expected range."))
    kind = "ok" if abs(total_diff) < 5 else ("warn" if total_diff < 0 else "ok")
    alerts.insert(0, (kind, f"So far, actual sales are **{total_diff:+.1f}%** compared with the forecast, over "
                            f"{len(table)} {period_word}{'s' if len(table) != 1 else ''}."))
    return alerts
