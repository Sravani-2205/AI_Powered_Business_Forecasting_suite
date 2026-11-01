"""
inflation.py
Is the business really growing, or are prices just going up?

If prices rose 6% and sales rose 8%, the business sold only about 2% MORE STUFF; the rest
is higher prices. Removing inflation like this gives "real" growth.

Two sources of inflation:
  1. Official Indian consumer prices (CPI), from a file `data_external/cpi_india.csv` with
     columns: month (YYYY-MM), cpi. One continuous index, monthly. (India switched to a
     new CPI series with base year 2024 in February 2026; the file must join the old and
     new series into one. A helper for that comes with the file.)
  2. A yearly rate the user types in. Default 4%: the RBI's inflation target, which the
     government kept at 4% (2-6% band) until March 2031. A business can also use its OWN
     price rises here, which may differ from national inflation.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CPI_FILE = Path(__file__).resolve().parent.parent / "data_external" / "cpi_india.csv"
DEFAULT_RATE = 4.0


def load_cpi(path: Path = CPI_FILE) -> pd.Series | None:
    """The official monthly price index, or None if the file is not there."""
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    months = pd.DatetimeIndex(pd.to_datetime(df["month"])).to_period("M").to_timestamp()
    cpi = pd.Series(df["cpi"].to_numpy(dtype=float), index=months)
    return cpi.sort_index()


def index_from_rate(start: pd.Timestamp, end: pd.Timestamp, yearly_rate_pct: float) -> pd.Series:
    """A made-up price index that rises by the same % every year (used when there is no CPI file)."""
    months = pd.date_range(start.to_period("M").to_timestamp(), end, freq="MS")
    monthly = (1 + yearly_rate_pct / 100) ** (1 / 12)
    return pd.Series(100 * monthly ** pd.RangeIndex(len(months)).to_numpy(), index=months)


@dataclass
class RealGrowth:
    money_growth_pct: float       # growth in plain money terms
    inflation_pct: float          # how much prices rose over the same 12 months
    real_growth_pct: float        # growth after removing inflation
    source: str                   # where the inflation figure came from


def real_growth(series: pd.Series, cpi: pd.Series, source: str) -> RealGrowth | None:
    """Last 12 months vs the 12 before, with and without inflation."""
    monthly = series.resample("MS").sum()
    counts = series.resample("MS").count()
    full = monthly[counts >= counts.max() * 0.9] if len(series) > 60 else monthly
    if len(full) < 24:
        return None
    last, prev = full.iloc[-12:], full.iloc[-24:-12]
    prices = cpi.reindex(full.index[-24:])
    if prices.isna().any():
        return None                                   # the CPI file does not cover these months
    money = last.sum() / prev.sum() - 1
    inflation = prices.iloc[-12:].mean() / prices.iloc[:12].mean() - 1
    real = (1 + money) / (1 + inflation) - 1
    return RealGrowth(money * 100, inflation * 100, real * 100, source)


def in_todays_prices(series: pd.Series, cpi: pd.Series) -> pd.DataFrame | None:
    """Monthly sales as recorded, and the same sales expressed in the latest month's prices."""
    monthly = series.resample("MS").sum()
    prices = cpi.reindex(monthly.index)
    if prices.isna().any():
        return None
    return pd.DataFrame({"as_recorded": monthly, "in_todays_prices": monthly * prices.iloc[-1] / prices})
