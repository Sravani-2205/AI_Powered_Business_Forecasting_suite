"""
synthetic_data.py
Step 0 of the pipeline: generate SIMULATED daily sales for a business, with patterns we
choose ourselves. Because we know the true patterns, we can later check whether the
forecasting models really find them ("testing against a known ground truth").

IMPORTANT: this data is simulated. It is inspired by how such a business behaves in
India, but it is NOT real company data and must always be labelled as simulated.

How a day's sales are built (each part is a multiplier, 1.0 = "normal"):
    sales = base level
            x growth trend        (the business slowly grows every year)
            x weekly pattern      (e.g. Sundays busier, Mondays quieter)
            x yearly season       (e.g. summer dip, winter wedding season)
            x festival effect     (e.g. Diwali spike; the date moves every year)
            x shock               (e.g. a COVID-style lockdown in 2020)
            x random noise        (no two days are the same)

Outputs (in data_synthetic/):
    <name>_7y.csv, <name>_2.5y.csv, <name>_1.5y.csv  - date + sales, in three sizes
    <name>_truth.csv                                  - every hidden part, day by day
    <name>.png                                        - a chart of the 7-year series

Run it from the project folder:
    python synthetic_data.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "data_synthetic"

START = pd.Timestamp("2019-10-01")
END = pd.Timestamp("2026-09-30")
SIZES = {"7y": START, "2.5y": pd.Timestamp("2024-04-01"), "1.5y": pd.Timestamp("2025-04-01")}

# Festival dates move every year (they follow the lunar calendar).
# NOTE: typed in from general knowledge; a date may be off by a day in some years.
# That does not matter for simulated data, but do not reuse this table as a reference.
FESTIVALS = {
    "diwali": ["2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12",
               "2024-10-31", "2025-10-20", "2026-11-08"],
    "raksha_bandhan": ["2019-08-15", "2020-08-03", "2021-08-22", "2022-08-11", "2023-08-30",
                       "2024-08-19", "2025-08-09", "2026-08-28"],
    "holi": ["2020-03-10", "2021-03-29", "2022-03-18", "2023-03-08", "2024-03-25",
             "2025-03-14", "2026-03-04"],
}

# The sweet shop's "personality". Other businesses will get their own version of this.
SWEET_SHOP = {
    "name": "sweet_shop",
    "base_level": 25_000,            # rupees on a normal day at the start
    "growth_per_year": 0.08,         # grows 8% a year
    # Monday ... Sunday
    "weekly": [0.85, 0.90, 0.90, 0.95, 1.00, 1.15, 1.25],
    # Yearly season: (peak day of year, size, width in days). Negative size = a dip.
    "seasons": [(150, -0.15, 45),    # summer dip around late May (heat, fewer sweets)
                (15, 0.10, 50)],     # winter wedding season around mid January
    # Festival: (days before it starts building, peak multiplier on the day before)
    "festivals": {"diwali": (10, 4.0), "raksha_bandhan": (4, 2.5), "holi": (3, 1.8)},
    "noise": 0.12,                   # about +/-12% random day-to-day variation
    "closed_days_per_year": 3,       # random closures (family events, repairs)
    "covid": True,
}


def growth_trend(dates, cfg):
    years = (dates - START).days / 365.25
    return (1 + cfg["growth_per_year"]) ** years


def weekly_pattern(dates, cfg):
    return np.array(cfg["weekly"])[dates.day_of_week]


def yearly_season(dates, cfg):
    day = dates.day_of_year.to_numpy()
    effect = np.zeros(len(dates))
    for peak, size, width in cfg["seasons"]:
        distance = np.minimum(np.abs(day - peak), 365 - np.abs(day - peak))  # wraps around New Year
        effect += size * np.exp(-0.5 * (distance / width) ** 2)              # smooth bell shape
    return 1 + effect


def festival_effect(dates, cfg):
    effect = np.ones(len(dates))
    for fest, (build_days, peak) in cfg["festivals"].items():
        for d in pd.to_datetime(FESTIVALS[fest]):
            days_before = (d - dates).days.to_numpy()
            # Build up over `build_days`, highest on the day before the festival
            building = (days_before >= 1) & (days_before <= build_days)
            ramp = 1 + (peak - 1) * (1 - (days_before - 1) / build_days)
            effect = np.where(building, np.maximum(effect, ramp), effect)
            # On the festival day itself, about half the peak; the next 2 days are quiet
            effect = np.where(days_before == 0, np.maximum(effect, 1 + (peak - 1) / 2), effect)
            effect = np.where((days_before == -1) | (days_before == -2), effect * 0.7, effect)
    return effect


def covid_shock(dates, cfg):
    effect = np.ones(len(dates))
    if not cfg["covid"]:
        return effect
    lockdown = (dates >= "2020-03-25") & (dates <= "2020-05-03")
    reopening = (dates > "2020-05-03") & (dates <= "2020-08-31")
    effect[lockdown] = 0.0
    # Recover in a straight line from 40% to 100% between May and the end of August
    progress = (dates[reopening] - pd.Timestamp("2020-05-04")).days / 119
    effect[reopening] = 0.4 + 0.6 * np.asarray(progress)
    return effect


def simulate(cfg, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    parts = pd.DataFrame(index=dates)
    parts["trend"] = growth_trend(dates, cfg)
    parts["weekly"] = weekly_pattern(dates, cfg)
    parts["yearly"] = yearly_season(dates, cfg)
    parts["festival"] = festival_effect(dates, cfg)
    parts["shock"] = covid_shock(dates, cfg)

    # "expected" = what sales would be with no luck involved. Models should get close to this.
    parts["expected_sales"] = (cfg["base_level"] * parts["trend"] * parts["weekly"]
                               * parts["yearly"] * parts["festival"] * parts["shock"])
    parts["noise"] = rng.lognormal(mean=0, sigma=cfg["noise"], size=len(dates))

    closed = rng.random(len(dates)) < cfg["closed_days_per_year"] / 365
    parts["closed"] = closed
    parts["sales"] = (parts["expected_sales"] * parts["noise"]).where(~closed, 0).round(0)
    parts.index.name = "date"
    return parts


def main():
    cfg = SWEET_SHOP
    OUT_DIR.mkdir(exist_ok=True)
    parts = simulate(cfg)

    parts.round(4).to_csv(OUT_DIR / f"{cfg['name']}_truth.csv")
    for label, start in SIZES.items():
        sales = parts.loc[start:, ["sales"]]
        sales.to_csv(OUT_DIR / f"{cfg['name']}_{label}.csv")
        print(f"{cfg['name']}_{label}.csv: {len(sales)} days, "
              f"{sales.index.min().date()} to {sales.index.max().date()}")

    yearly_totals = parts["sales"].groupby(parts.index.year).sum()
    print("-" * 60)
    print("Total simulated sales per year (rupees):")
    print(yearly_totals.map("{:,.0f}".format).to_string())
    biggest = parts["sales"].nlargest(3)
    print("Three biggest days (should be just before Diwali):")
    for day, value in biggest.items():
        print(f"  {day.date()}  {value:,.0f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(parts.index, parts["sales"], color="tab:orange", linewidth=0.6, label="simulated daily sales")
    weekly_avg = parts["sales"].rolling(28, center=True).mean()
    ax.plot(parts.index, weekly_avg, color="black", linewidth=1.5, label="28-day average")
    ax.set_title("SIMULATED sweet shop: daily sales (not real data)")
    ax.set_ylabel("Sales per day (INR)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{cfg['name']}.png", dpi=120)
    print(f"Saved files in {OUT_DIR.name}/")


if __name__ == "__main__":
    main()
