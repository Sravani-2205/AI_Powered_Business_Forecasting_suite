"""
run_forecast.py
Run the whole pipeline on any CSV file:
data check -> backtest competition -> final test -> future forecast.

Usage (from the project folder):
    python run_forecast.py <file> --date <date column> --value <sales column>

Festival options (daily data only):
    --country IN         country for the holiday list (default IN = India)
    --holidays "A,B,C"   exact holiday names to use (pick them with holiday_report.py);
                         without this, the default shortlist for the country is used
    --custom file.csv    the business's own special dates (columns: date, name)
    --no-holidays        use no festival dates at all

Example:
    python run_forecast.py data_synthetic/sweet_shop_7y.csv --date date --value sales ^
        --holidays "Diwali (Deepavali),Raksha Bandhan,Holi"

Results are saved in results_pipeline/<file name>/:
    leaderboard.csv, winner_folds.csv, future_forecast.csv, forecast.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forecaster.calendar import build_calendar
from forecaster.competition import run_competition
from forecaster.data_check import load_series

parser = argparse.ArgumentParser(description="Personalised forecast for one business's sales file.")
parser.add_argument("file", help="path to a CSV file")
parser.add_argument("--date", required=True, help="name of the date column")
parser.add_argument("--value", required=True, help="name of the sales column")
parser.add_argument("--country", default="IN", help="country code for holidays (default IN)")
parser.add_argument("--holidays", help='comma-separated holiday names, e.g. "Holi,Raksha Bandhan"')
parser.add_argument("--custom", help="CSV of the business's own special dates (date, name)")
parser.add_argument("--no-holidays", action="store_true", help="do not use festival dates")
args = parser.parse_args()

series, report = load_series(args.file, args.date, args.value)
print(f"File: {args.file}")
report.print()
if report.tier == "too_small":
    raise SystemExit("Stopping: not enough data.")

calendar = None
if report.frequency != "daily":
    print("Festivals: not used (weekly/monthly totals hide single days).")
elif args.no_holidays:
    print("Festivals: switched off (--no-holidays).")
else:
    # Years from the start of the data to one year past the end, so future festivals are known.
    years = range(series.index.min().year, series.index.max().year + 2)
    selected = [h.strip() for h in args.holidays.split(",")] if args.holidays else None
    calendar = build_calendar(args.country, years, selected=selected, custom_path=args.custom)
    names = sorted(calendar["name"].unique())
    print(f"Festivals given to festival-aware models ({len(names)}): {', '.join(names)}")

print("Backtesting every model (this can take a few minutes)...")
result = run_competition(series, report, calendar)
unit = report.frequency_unit()
line = "-" * 60

windows = "window" if result.n_folds == 1 else "windows"
print(f"Backtest: {result.n_folds} {windows} of {report.horizon} {unit}, then a final test of "
      f"{report.horizon} {unit}.")
if not result.covers_full_year:
    print("NOTE: with this much data the windows do not cover a full year, so some seasons "
          "were never tested. Treat the accuracy below with caution.")
print(line)
print("Leaderboard, scored over ALL backtest windows (lower is better):")
cols = [c for c in ["family", "MAE", "RMSE", "WAPE_%", "status"] if c in result.leaderboard.columns]
print(result.leaderboard[cols].to_string(float_format="{:,.1f}".format))
if result.not_built:
    print(f"(Allowed for this data but not built yet: {', '.join(result.not_built)})")
print(line)
print(f"Winner: {result.winner}")
print("Its score in each backtest window (shows which seasons it handles badly):")
print(result.winner_fold_scores[["from", "to", "WAPE_%"]].to_string(float_format="{:,.1f}".format))
print(line)
backtest_wape = result.leaderboard.loc[result.winner, "WAPE_%"]
print(f"Expected accuracy (backtest, across the seasons covered): off by about {backtest_wape:.0f}%")
print(f"Final check on the unseen test period: WAPE {result.test_scores['WAPE_%']:.1f}%, "
      f"MAE {result.test_scores['MAE']:,.0f}")
print(line)

out_dir = Path("results_pipeline") / Path(args.file).stem
out_dir.mkdir(parents=True, exist_ok=True)
result.leaderboard.to_csv(out_dir / "leaderboard.csv")
result.winner_fold_scores.to_csv(out_dir / "winner_folds.csv")
result.future_forecast.rename("forecast").to_csv(out_dir / "future_forecast.csv")

# Chart: everything the winner was judged on, plus the future.
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
ax.set_title(f"{Path(args.file).stem}: winner = {result.winner}, "
             f"backtest WAPE {backtest_wape:.1f}%, test WAPE {result.test_scores['WAPE_%']:.1f}%")
ax.set_ylabel("Sales")
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(out_dir / "forecast.png", dpi=120)
print(f"Saved: {out_dir}/leaderboard.csv, winner_folds.csv, future_forecast.csv, forecast.png")
