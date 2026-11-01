"""
build_cpi.py
Builds data_external/cpi_india.csv: ONE continuous monthly price index for India,
used by the app to separate real growth from price rises.

Why this is needed:
  India's Consumer Price Index (CPI) switched to a new series (base year 2024) from
  January 2026. The old series (base 2012) stops in December 2025. To join them we:
    1. keep the old series' index values up to December 2025, as published, and
    2. continue each 2026 month from the same month a year earlier, using the official
       year-on-year inflation rate of the new series:
           index(2026 month) = index(same month in 2025) x (1 + inflation / 100)

Inputs:
  data_external/cpi_2085.xlsx  - downloaded from MoSPI's eSankhyiki portal: All India,
                                 Combined, General Index, base 2012, Jan 2013 to Dec 2025
  NEW_SERIES_INFLATION below  - 2026 year-on-year rates (All India, Combined, General),
                                 base 2024, as first published in MoSPI's monthly releases.
                                 Later revisions of a month by MoSPI may differ by ~0.1 point.

Run it from the project folder whenever new months are added:
    python build_cpi.py
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RAW = PROJECT_DIR / "data_external" / "cpi_2085.xlsx"
OUT = PROJECT_DIR / "data_external" / "cpi_india.csv"

# Year-on-year CPI inflation, new series (base 2024 = 100), % . Add new months at the end.
NEW_SERIES_INFLATION = {
    "2026-01": 2.75, "2026-02": 3.21, "2026-03": 3.40, "2026-04": 3.48, "2026-05": 3.93,
    "2026-06": 4.38, "2026-07": 4.45, "2026-08": 4.82, "2026-09": 5.54,
}


def main() -> None:
    raw = pd.read_excel(RAW)
    raw = raw[(raw["state"] == "All India") & (raw["sector"] == "Combined") &
              (raw["subgroup"] == "General-Overall")]
    old = pd.Series(raw["index"].to_numpy(dtype=float),
                    index=pd.to_datetime(dict(year=raw["year"], month=raw["month_code"], day=1))).sort_index()

    # Sanity check: the published yearly inflation must match the index itself
    check = (old / old.shift(12) - 1) * 100
    published = pd.Series(raw["inflation"].to_numpy(dtype=float), index=old.index.sort_values()[::-1]).sort_index()
    diff = (check - published.reindex(check.index)).abs().max()
    print(f"Old series: {old.index.min():%b %Y} to {old.index.max():%b %Y}, {len(old)} months "
          f"(largest gap between published and recomputed inflation: {diff:.2f} points)")

    index = old.copy()
    for month, rate in NEW_SERIES_INFLATION.items():
        when = pd.Timestamp(month + "-01")
        index[when] = index[when - pd.DateOffset(years=1)] * (1 + rate / 100)
    index = index.sort_index()

    out = pd.DataFrame({"month": index.index.strftime("%Y-%m"), "cpi": index.round(2).to_numpy()})
    out.to_csv(OUT, index=False)
    print(f"Saved {OUT.name}: {out['month'].iloc[0]} to {out['month'].iloc[-1]} ({len(out)} months)")
    print(out.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
