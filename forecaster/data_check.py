"""
data_check.py
Pipeline steps 1-3: load ANY business's sales file, check it, and decide its size tier.

The user only has to say which column holds the dates and which holds the sales.
Everything else is worked out here:
  - how often the data is recorded (daily, weekly or monthly)
  - duplicate dates (e.g. one row per transaction) are added together
  - missing dates are found and filled with 0, with a warning
  - how much history there is, and which tier that puts the business in

The tiers decide which models are allowed to compete later:
  too_small : less than 1 year            -> no forecast
  small     : 1 to 2 years                -> baselines + simple statistical models
  medium    : 2+ years                    -> + Prophet and machine-learning models
  large     : 2+ years AND 1,000+ points  -> + deep learning (LSTM)
These thresholds are rules of thumb chosen for this project, not universal laws.
"""

from dataclasses import dataclass, field

import pandas as pd

# How far ahead we forecast, and so how long the validation and test periods are.
HORIZON = {"daily": 90, "weekly": 13, "monthly": 12}

MODEL_FAMILIES = {
    "small": ["baselines", "statistical"],
    "medium": ["baselines", "statistical", "prophet", "machine_learning"],
    "large": ["baselines", "statistical", "prophet", "machine_learning", "deep_learning"],
}

TIER_MESSAGES = {
    "too_small": "Less than 1 year of history. That is not enough to forecast reliably, "
                 "so no forecast will be made.",
    "small": "Between 1 and 2 years of history. Yearly patterns (like festival seasons) "
             "cannot be learned reliably yet, so only simple models will be used. "
             "For this amount of data, simple models are usually enough.",
    "medium": "At least 2 years of history: enough for seasonal and machine-learning models.",
    "large": "At least 2 years and 1,000+ data points: enough to also try deep learning.",
}


@dataclass
class DataReport:
    """Everything the data check found out, in one place."""
    frequency: str
    n_points: int
    start: pd.Timestamp
    end: pd.Timestamp
    years: float
    tier: str
    horizon: int
    allowed_families: list
    message: str
    warnings: list = field(default_factory=list)

    def print(self) -> None:
        line = "-" * 60
        print(line)
        print(f"Frequency:      {self.frequency}")
        print(f"History:        {self.start.date()} to {self.end.date()} "
              f"({self.years:.1f} years, {self.n_points:,} data points)")
        print(f"Tier:           {self.tier.upper()}")
        print(f"Models allowed: {', '.join(self.allowed_families) or 'none'}")
        print(f"Forecast horizon: {self.horizon} {self.frequency_unit()}")
        print(line)
        print(self.message)
        if self.warnings:
            print(line)
            print("Warnings:")
            for w in self.warnings:
                print(f"  - {w}")
        print(line)

    def frequency_unit(self) -> str:
        return {"daily": "days", "weekly": "weeks", "monthly": "months"}[self.frequency]


def detect_frequency(dates: pd.Series) -> str:
    """Look at the typical gap between dates: 1 day, 7 days or about a month."""
    gaps = dates.drop_duplicates().sort_values().diff().dt.days.dropna()
    if gaps.empty:
        raise ValueError("Need at least two different dates to detect the frequency.")
    typical = gaps.median()
    if typical == 1:
        return "daily"
    if typical == 7:
        return "weekly"
    if 28 <= typical <= 31:
        return "monthly"
    raise ValueError(f"Could not recognise the frequency: the typical gap is {typical:.0f} days. "
                     "Daily, weekly and monthly data are supported.")


def load_series(path, date_col: str, value_col: str) -> tuple[pd.Series, DataReport]:
    """Load a CSV, clean it into one regular series, and return it with a DataReport."""
    raw = pd.read_csv(path)
    for col in (date_col, value_col):
        if col not in raw.columns:
            raise ValueError(f"Column '{col}' not found. Columns in the file: {list(raw.columns)}")

    warnings = []
    df = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], errors="coerce"),
        "value": pd.to_numeric(raw[value_col], errors="coerce"),
    })
    bad = df.isna().any(axis=1)
    if bad.any():
        warnings.append(f"{bad.sum()} rows had an unreadable date or sales value and were skipped.")
        df = df[~bad]

    frequency = detect_frequency(df["date"])

    # Monthly data is stored on the first day of each month, so dates line up.
    if frequency == "monthly":
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    if df["date"].duplicated().any():
        n_dupes = df["date"].duplicated().sum()
        warnings.append(f"{n_dupes} rows shared a date with another row; values on the same "
                        "date were added together (e.g. several transactions in one day).")
    series = df.groupby("date")["value"].sum().sort_index()

    # Fill missing dates so the series is regular (models expect evenly spaced data).
    step = {"daily": "D", "weekly": pd.offsets.Week(weekday=series.index[0].day_of_week), "monthly": "MS"}
    full_range = pd.date_range(series.index.min(), series.index.max(), freq=step[frequency])
    missing = full_range.difference(series.index)
    if len(missing) > 0:
        warnings.append(f"{len(missing)} dates were missing and were filled with 0. If the business "
                        "was open on those dates, the forecast will be too low; please check.")
    series = series.reindex(full_range, fill_value=0.0)
    series.index.name = "date"

    zeros = int((series == 0).sum())
    if zeros:
        reason = "closed days, plus the missing dates above" if len(missing) else "probably closed days"
        warnings.append(f"{zeros} periods have zero sales ({reason}).")
    if (series < 0).any():
        warnings.append(f"{int((series < 0).sum())} periods have negative sales (e.g. more returns than sales).")

    report = build_report(series, frequency, warnings)
    return series, report


def build_report(series: pd.Series, frequency: str, warnings: list) -> DataReport:
    start, end = series.index.min(), series.index.max()
    years = (end - start).days / 365.25
    n_points = len(series)

    if years < 1:
        tier = "too_small"
    elif years < 2:
        tier = "small"
    elif n_points < 1000:
        tier = "medium"
    else:
        tier = "large"

    return DataReport(
        frequency=frequency,
        n_points=n_points,
        start=start,
        end=end,
        years=years,
        tier=tier,
        horizon=HORIZON[frequency],
        allowed_families=MODEL_FAMILIES.get(tier, []),
        message=TIER_MESSAGES[tier],
        warnings=warnings,
    )
