"""
holiday_report.py
See which holidays fall inside a business's sales history and how much each one matters.

Usage (from the project folder):
    python holiday_report.py <file> --date <date column> --value <sales column> [options]

Options:
    --country IN             country code for the holiday list (default: IN = India)
    --all                    measure EVERY holiday in the list, not just the default shortlist
    --custom <file.csv>      add the business's own special dates (columns: date, name)

Example:
    python holiday_report.py data_synthetic/sweet_shop_7y.csv --date date --value sales --all

Saves results_pipeline/<file name>/holiday_impact.csv and holidays_on_sales.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forecaster.calendar import build_calendar, country_holidays, holiday_impact
from forecaster.data_check import load_series

parser = argparse.ArgumentParser(description="Measure how much each holiday matters for one business.")
parser.add_argument("file")
parser.add_argument("--date", required=True)
parser.add_argument("--value", required=True)
parser.add_argument("--country", default="IN")
parser.add_argument("--all", action="store_true", help="measure every holiday, not just the shortlist")
parser.add_argument("--custom", help="CSV of the business's own special dates (date, name)")
args = parser.parse_args()

series, report = load_series(args.file, args.date, args.value)
if report.frequency != "daily":
    raise SystemExit("Holiday impact needs daily data; weekly/monthly totals hide single days.")

years = range(series.index.min().year, series.index.max().year + 1)
selected = sorted(country_holidays(args.country, years)["name"].unique()) if args.all else None
calendar = build_calendar(args.country, years, selected=selected, custom_path=args.custom)
impact = holiday_impact(series, calendar)

line = "-" * 72
print(f"File: {args.file}   ({report.start.date()} to {report.end.date()})")
print(f"Holidays measured: {'all in the list' if args.all else 'default shortlist'}"
      f"{' + custom dates' if args.custom else ''}  (country: {args.country})")
print(line)
print("Sales around each holiday (a week before to the day after) vs normal days:")
print(impact.to_string(float_format="{:+.0f}".format))
print(line)
print("How to read it:")
print("  avg_uplift_%       how much higher sales are, on average, across the whole window")
print("  peak_day_uplift_%  how much higher the single busiest day in the window is")
print("  consistent         True if the effect points the same way every year (seen 3+ times)")
print("  near               a BIGGER holiday a few days away: this one may only be borrowing its effect")
print("Holidays with a big, consistent uplift are worth giving to the models.")

out_dir = Path("results_pipeline") / Path(args.file).stem
out_dir.mkdir(parents=True, exist_ok=True)
impact.to_csv(out_dir / "holiday_impact.csv")

# Chart: the sales history with the top holidays marked, so you can SEE where they fall.
top = impact.head(5).index.tolist()
colors = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(series.index, series.values, color="lightgray", linewidth=0.6, label="daily sales")
ax.plot(series.index, series.rolling(7, center=True).mean(), color="black", linewidth=1, label="7-day average")
for i, name in enumerate(top):
    days = calendar.loc[calendar["name"] == name, "date"]
    days = days[(days >= series.index.min()) & (days <= series.index.max())]
    for j, day in enumerate(days):
        ax.axvline(day, color=colors(i), linewidth=1, alpha=0.8, label=name if j == 0 else None)
ax.set_title(f"{Path(args.file).stem}: where the 5 most important holidays fall")
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / "holidays_on_sales.png", dpi=120)
print(f"Saved: {out_dir}/holiday_impact.csv and holidays_on_sales.png")
