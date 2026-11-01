"""
calendar.py
Festival and holiday dates for the pipeline, and a measurement of how much each one
actually matters for THIS business.

Three layers of dates:
  1. Country holidays from the open-source `holidays` library (public + optional lists).
     This is a community-maintained package, not an official government list.
  2. A default shortlist of big commercial festivals (India only for now). The user can
     select or unselect any holiday from the full list.
  3. The business's own special dates from a small CSV (columns: date, name), e.g.
     Dhanteras, an anniversary sale, a local fair.

Holiday impact:
  For every holiday, we look at each year it happened inside the sales history and compare
  sales AROUND it (a week before to a day after, because shoppers buy in advance)
  with normal sales a few weeks before and after. Averaged over the years, that gives
  "sales are usually X% higher around this holiday". The user can then keep the ones
  that matter and drop the rest.

  Two traps the table protects against:
    - Years where the shop was mostly closed around the holiday (e.g. a lockdown) are
      skipped, otherwise a closure would look like "this holiday kills sales".
    - Holidays a few days from a BIGGER one borrow its effect (Bhai Duj sits a few days
      after Diwali). The `near` column names the bigger neighbour, so the user knows the
      effect may not be its own. Telling such neighbours apart is genuinely hard.
  This works for daily data only; weekly or monthly totals are too coarse to see a
  single day's effect.
"""

import numpy as np
import pandas as pd

# The biggest commercial festivals in India. Names exactly as the holidays library writes them.
DEFAULT_SHORTLIST = {
    "IN": [
        "Diwali (Deepavali)", "Dussehra", "Holi", "Raksha Bandhan",
        "Ganesh Chaturthi / Vinayak Chaturthi", "Id-ul-Fitr", "Christmas",
        "Makar Sankranti", "Pongal", "Onam", "Karaka Chaturthi (Karwa Chouth)", "Bhai Duj",
    ],
}

DAYS_BEFORE = 7   # the window starts a week before the holiday (people buy in advance)
DAYS_AFTER = 1    # and ends the day after


def country_holidays(country: str, years: range) -> pd.DataFrame:
    """Every holiday the library knows for a country, one row per (date, name)."""
    import holidays

    try:   # India and some other countries also have an "optional" list
        found = holidays.country_holidays(country, years=list(years), categories=("public", "optional"))
    except Exception:
        found = holidays.country_holidays(country, years=list(years))
    rows = []
    for day, names in found.items():
        for name in names.split("; "):            # one date can hold several holidays
            rows.append({"date": pd.Timestamp(day), "name": name.strip(), "source": "library"})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def custom_dates(path) -> pd.DataFrame:
    """The business's own special dates from a CSV with columns: date, name."""
    df = pd.read_csv(path)
    if not {"date", "name"} <= set(df.columns):
        raise ValueError("The special-dates file needs two columns: date, name")
    return pd.DataFrame({"date": pd.to_datetime(df["date"]), "name": df["name"].astype(str),
                         "source": "custom"})


def build_calendar(country: str, years: range, selected: list | None = None,
                   custom_path=None) -> pd.DataFrame:
    """The final list of dates the models will use.
    selected=None means: use the default shortlist (or all holidays if there is no shortlist)."""
    calendar = country_holidays(country, years)
    if selected is None:
        # Default shortlist: keep whichever of its names the library has for these years.
        selected = DEFAULT_SHORTLIST.get(country, sorted(calendar["name"].unique()))
    else:
        # The user picked these by hand, so a typo should be reported, not silently ignored.
        unknown = set(selected) - set(calendar["name"])
        if unknown:
            import difflib
            hints = []
            for name in sorted(unknown):
                close = [c for c in calendar["name"].unique() if name.lower() in c.lower()]
                close = close or difflib.get_close_matches(name, calendar["name"].unique(), n=3)
                hints.append(f"'{name}' (did you mean: {', '.join(repr(c) for c in close)}?)"
                             if close else f"'{name}'")
            raise ValueError(f"Not in the {country} holiday list: {'; '.join(hints)}")
    calendar = calendar[calendar["name"].isin(selected)]
    if custom_path:
        calendar = pd.concat([calendar, custom_dates(custom_path)])
    return calendar.sort_values("date").reset_index(drop=True)


def holiday_impact(series: pd.Series, calendar: pd.DataFrame) -> pd.DataFrame:
    """How much higher (or lower) sales are around each holiday, compared with normal days."""
    rows = []
    for name, group in calendar.groupby("name"):
        uplifts, peaks = [], []
        for day in group["date"]:
            window = series.loc[day - pd.Timedelta(days=DAYS_BEFORE): day + pd.Timedelta(days=DAYS_AFTER)]
            before = series.loc[day - pd.Timedelta(days=35): day - pd.Timedelta(days=DAYS_BEFORE + 1)]
            after = series.loc[day + pd.Timedelta(days=DAYS_AFTER + 1): day + pd.Timedelta(days=29)]
            normal = pd.concat([before, after])
            normal = normal[normal > 0]               # ignore closed days
            if len(window) < DAYS_BEFORE + DAYS_AFTER + 1 or len(normal) < 20:
                continue                               # holiday too close to the start/end of the data
            if (window == 0).mean() > 0.5:
                continue                               # shop mostly closed that year (e.g. lockdown)
            baseline = normal.median()                 # median: not pulled up by other spikes
            uplifts.append(window.mean() / baseline - 1)
            peaks.append(window.max() / baseline - 1)
        if uplifts:
            ups = np.array(uplifts)
            rows.append({
                "holiday": name,
                "times_seen": len(ups),
                "avg_uplift_%": 100 * float(ups.mean()),
                "peak_day_uplift_%": 100 * float(np.mean(peaks)),
                # Same direction every year AND seen at least 3 times (once proves nothing)
                "consistent": bool(len(ups) >= 3 and (np.all(ups > 0.05) or np.all(ups < -0.05))),
            })
    if not rows:
        return pd.DataFrame(columns=["holiday", "times_seen", "avg_uplift_%", "peak_day_uplift_%",
                                     "consistent", "near"])
    table = pd.DataFrame(rows).set_index("holiday")
    table["near"] = [bigger_neighbour(name, table, calendar) for name in table.index]
    return table.reindex(table["avg_uplift_%"].abs().sort_values(ascending=False).index)


def bigger_neighbour(name: str, table: pd.DataFrame, calendar: pd.DataFrame) -> str:
    """A holiday with a bigger (or equal) effect that falls within a few days of this one
    in at least half the years, or "" if there is none. Holidays on the very same dates
    (e.g. Diwali and Naraka Chaturdashi) cannot be told apart at all, so they are marked
    "same days as ..."."""
    span = DAYS_BEFORE + DAYS_AFTER
    my_dates = set(calendar.loc[calendar["name"] == name, "date"])
    best, best_effect = "", abs(table.loc[name, "avg_uplift_%"])
    for other in table.index:
        effect = abs(table.loc[other, "avg_uplift_%"])
        if other == name or effect < best_effect or (best and effect == best_effect):
            continue                                   # smaller, or a tie with one already found
        other_dates = set(calendar.loc[calendar["name"] == other, "date"])
        close = sum(any(abs((d - o).days) <= span for o in other_dates) for d in my_dates)
        if close >= len(my_dates) / 2:
            same = len(my_dates & other_dates) >= len(my_dates) / 2
            best, best_effect = (f"same days as {other}" if same else other), effect
    return best


def holiday_groups(table: pd.DataFrame, calendar: pd.DataFrame, preferred: list | None = None) -> pd.Series:
    """Put holidays that fall within a few days of each other (in at least half the years)
    into one group. Sales data cannot tell the members of a group apart, so the user should
    normally pick ONE from each group.

    Each group is named after its LEADER: a well-known name from `preferred` (e.g. the
    country's shortlist, so "Holi" leads rather than "Holika Dahan", the evening before it),
    or else the member with the strongest effect."""
    span = DAYS_BEFORE + DAYS_AFTER
    names = list(table.index)
    dates = {n: set(calendar.loc[calendar["name"] == n, "date"]) for n in names}
    parent = {n: n for n in names}

    def root(n):
        while parent[n] != n:
            n = parent[n]
        return n

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            close = sum(any(abs((d - o).days) <= span for o in dates[b]) for d in dates[a])
            if close >= min(len(dates[a]), len(dates[b])) / 2:
                parent[root(b)] = root(a)

    members = {}
    for n in names:
        members.setdefault(root(n), []).append(n)
    label = {}
    preferred = set(preferred or [])
    for group in members.values():
        candidates = [n for n in group if n in preferred] or group
        leader = max(candidates, key=lambda n: abs(table.loc[n, "avg_uplift_%"]))
        for n in group:
            label[n] = leader
    return pd.Series(label, name="group")


def default_selection(table: pd.DataFrame, groups: pd.Series, min_uplift: float = 10.0) -> list:
    """A sensible starting choice: from each group, its leader, if the leader's effect is
    consistent every year and at least `min_uplift` % in size."""
    chosen = []
    for leader in groups.unique():
        row = table.loc[leader]
        if row["consistent"] and abs(row["avg_uplift_%"]) >= min_uplift:
            chosen.append(leader)
    return chosen
