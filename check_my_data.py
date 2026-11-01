"""
check_my_data.py
Try the pipeline's data check on any CSV file.

Usage (from the project folder):
    python check_my_data.py <file> --date <date column> --value <sales column>

Examples:
    python check_my_data.py data_synthetic/sweet_shop_7y.csv --date date --value sales
    python check_my_data.py data/daily_sales.csv --date date --value net_revenue
"""

import argparse

from forecaster.data_check import load_series

parser = argparse.ArgumentParser(description="Check a sales file and find its data-size tier.")
parser.add_argument("file", help="path to a CSV file")
parser.add_argument("--date", required=True, help="name of the date column")
parser.add_argument("--value", required=True, help="name of the sales column")
args = parser.parse_args()

series, report = load_series(args.file, args.date, args.value)
print(f"File: {args.file}")
report.print()
