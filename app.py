"""
app.py
Personalised Sales Forecasting: the web app.

A business uploads its own sales history and is guided through seven steps
(shown in the left sidebar):
  1. Your data      - upload a CSV or pick a demo, choose the date and sales columns
  2. Data check     - frequency, history length, warnings, and the size tier
  3. Festivals      - which holidays matter for THIS business; tick the ones to use
  4. Run            - every allowed model is backtested over a year
  5. Results        - the winner, an honest accuracy, the chart and the forecast
  6. Business story - what the data says: growth, inflation, patterns, festival value
  7. Plan ahead     - the forecast turned into stock, staffing and cash-flow plans
Tools:
  What if           - the forecast with the user's own assumptions (prices, a new outlet, offers)
  Track actuals     - real sales compared with the forecast, with early warnings

Run it from the project folder:
    streamlit run app.py
"""

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from forecaster.calendar import (DEFAULT_SHORTLIST, country_holidays, custom_dates,
                                 default_selection, holiday_groups, holiday_impact)
from forecaster import inflation as infl
from forecaster import insights as ins
from forecaster.charts import (cash_figure, forecast_figure, history_figure, holidays_figure,
                               money_bars, percent_bars, real_vs_money_figure, scenario_figure,
                               tracking_figure, weekly_plan_figure)
from forecaster.competition import plan_folds, run_competition
from forecaster.data_check import series_from_frame

PROJECT_DIR = Path(__file__).resolve().parent
DEMOS = {
    "Simulated sweet shop, 7 years of daily sales": "data_synthetic/sweet_shop_7y.csv",
    "Simulated sweet shop, 1.5 years of daily sales": "data_synthetic/sweet_shop_1.5y.csv",
}
STEPS = ["Your data", "Data check", "Festivals", "Run", "Results", "Business story", "Plan ahead",
         "What if", "Track actuals"]
FIRST_TOOL = 7                        # steps from here on are optional tools, not a sequence
CURRENCIES = ["₹", "$", "£", "€", "(none)"]
FESTIVAL_FAMILIES = {"prophet", "machine_learning", "deep_learning"}
FAMILY_NAMES = {"baselines": "simple baselines", "statistical": "statistical models",
                "prophet": "Prophet", "machine_learning": "machine learning",
                "deep_learning": "deep learning (LSTM)"}

st.set_page_config(page_title="Personalised Sales Forecasting", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, p, li, label, input, textarea, button, h1, h2, h3, h4,
[data-testid="stMarkdownContainer"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
/* Page: remove the empty strip Streamlit keeps for its own header */
[data-testid="stHeader"] { background: transparent; height: 2rem; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1160px; }
[data-testid="stSidebarHeader"] { padding: 0.5rem 1rem 0; height: auto; min-height: 0; }
[data-testid="stSidebarUserContent"] { padding-top: 0.6rem; }
h1 { font-weight: 600; font-size: 2.1rem; letter-spacing: -0.015em; }
h2, h3 { font-weight: 600; letter-spacing: -0.005em; }
.lede { color: #8A94A7; font-size: 1.05rem; max-width: 64ch; margin-top: -0.5rem; margin-bottom: 1.8rem; }

/* Sidebar: same colour as the page, separated by a hairline */
[data-testid="stSidebar"] { background: #11161F; border-right: 1px solid #263042; }
[data-testid="stSidebar"] .stButton button {
    background: transparent; border: none; border-left: 2px solid transparent; border-radius: 0;
    color: #8A94A7; justify-content: flex-start; text-align: left;
    padding: 0.55rem 0.9rem; min-height: 0; box-shadow: none;
}
[data-testid="stSidebar"] .stButton button:hover:not(:disabled) { color: #E6E9EF; background: #171D29; }
[data-testid="stSidebar"] .stButton button[kind="primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    color: #E6E9EF; background: #171D29; border-left: 2px solid #5B9BF0; font-weight: 600;
}
[data-testid="stSidebar"] .stButton button:disabled { color: #4A5468; background: transparent; }
[data-testid="stSidebar"] .stButton { margin-bottom: -0.6rem; }
[data-testid="stSidebar"] .stButton button > div,
[data-testid="stSidebar"] .stButton button p { width: 100%; justify-content: flex-start; text-align: left; }
.brand { font-size: 1.05rem; font-weight: 600; color: #E6E9EF; margin: 0; display: flex; align-items: center; gap: 0.55rem; }
.section { font-size: 1.3rem; font-weight: 600; color: #E6E9EF; margin: 2.2rem 0 0.2rem;
           padding-top: 1.2rem; border-top: 1px solid #263042; }
.finding { font-size: 1.04rem; color: #C9CFDB; line-height: 1.6; max-width: 72ch; margin: 0.3rem 0 0.8rem; }
.finding b { color: #E6E9EF; }
.tip { border-left: 2px solid #5B9BF0; padding: 0.15rem 0 0.15rem 0.9rem; margin: 0 0 0.9rem;
       color: #C9CFDB; line-height: 1.6; max-width: 78ch; }
.tip b { color: #E6E9EF; }
.sidebar-note { color: #8A94A7; font-size: 0.9rem; line-height: 1.5; }
.dataset { border-top: 1px solid #263042; padding-top: 1rem; margin-top: 1.4rem; }

/* Headline numbers: equal tiles */
[data-testid="stMetric"] {
    background: #171D29; border: 1px solid #263042; border-radius: 8px; padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] p { color: #8A94A7; font-size: 0.92rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; font-variant-numeric: tabular-nums; }
[data-testid="stMetricValue"] > div, [data-testid="stMetricValue"] p {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height: 1.25; }

/* Quiet notes instead of Streamlit's bright alert boxes */
.note { background: #171D29; border: 1px solid #263042; border-left: 3px solid var(--tone);
        border-radius: 6px; padding: 0.75rem 1rem; margin: 0.6rem 0 1rem; color: #C9CFDB;
        font-size: 0.98rem; line-height: 1.55; }
.note b { color: #E6E9EF; }
.tag { display: inline-block; font-size: 0.8rem; color: #8A94A7; border: 1px solid #263042;
       border-radius: 999px; padding: 0.1rem 0.6rem; margin-left: 0.4rem; vertical-align: middle; }
.panel-title { font-size: 0.95rem; color: #8A94A7; margin: 0 0 0.4rem; }

/* The one-sentence verdict on the results page */
.verdict { font-size: 1.15rem; line-height: 1.6; color: #C9CFDB; max-width: 70ch; margin: 1.4rem 0 0.6rem; }
.verdict b { color: #E6E9EF; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- state helpers
state = st.session_state
state.setdefault("step", 0)
state.setdefault("results", {})       # remembered competition results, by settings
state.setdefault("selections", {})    # remembered holiday ticks, by data + country


def go_to(step: int) -> None:
    state.step = step
    st.rerun()


def run_key() -> tuple:
    return (state.get("data_name"), state.get("date_col"), state.get("value_col"),
            state.get("country"), tuple(sorted(state.get("selected", []))),
            bool(state.get("include_dl")))


def reachable(step: int) -> bool:
    if step == 0:
        return True
    if "series" not in state:
        return False
    if step >= 4:                                   # Results, story, plan and the tools
        return run_key() in state.results
    return state.report.tier != "too_small" or step == 1


def done(step: int) -> bool:
    if step == 0:
        return "series" in state
    if step in (1, 2):
        return "series" in state and state.report.tier != "too_small" and state.step > step
    if step == 3:
        return run_key() in state.results
    return False


# ---------------------------------------------------------------- sidebar navigation
with st.sidebar:
    st.markdown('<p class="brand">'
                '<svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">'
                '<rect x="2" y="12" width="4" height="8" rx="1" fill="#7C869B"/>'
                '<rect x="9" y="8" width="4" height="12" rx="1" fill="#5B9BF0"/>'
                '<rect x="16" y="3" width="4" height="17" rx="1" fill="#F4A63A"/></svg>'
                'Sales forecasting</p>'
                '<p class="sidebar-note">Your data, every suitable model, one honest answer.</p>',
                unsafe_allow_html=True)
    st.write("")
    for i, name in enumerate(STEPS):
        if i == FIRST_TOOL:
            st.markdown('<p class="sidebar-note" style="margin:1.1rem 0 0.2rem 0.9rem">Tools</p>',
                        unsafe_allow_html=True)
        mark = "✓" if done(i) else (f"{i + 1}" if i < FIRST_TOOL else "•")
        if st.button(f"{mark}\u2003{name}", key=f"nav_{i}", disabled=not reachable(i),
                     type="primary" if i == state.step else "secondary", width="stretch"):
            go_to(i)
    if "series" in state:
        r = state.report
        st.markdown(f'<div class="dataset"><p class="brand">{state.data_name}</p>'
                    f'<p class="sidebar-note">{r.years:.1f} years of {r.frequency} data<br>'
                    f'{r.tier.replace("_", " ").capitalize()} tier, {r.n_points:,} data points</p></div>',
                    unsafe_allow_html=True)


def page_title(title: str, lede: str) -> None:
    st.title(title)
    st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)


TONES = {"info": "#5B9BF0", "warn": "#D9A84E", "error": "#E0697A", "ok": "#59C3A6"}


def note(text: str, kind: str = "info") -> None:
    """A quiet message box: dark background, a thin coloured edge that shows the kind."""
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text).replace("\n\n", "<br>")
    st.markdown(f'<div class="note" style="--tone:{TONES[kind]}">{html}</div>', unsafe_allow_html=True)


def guess_column(columns: list, hints: list) -> int:
    for i, col in enumerate(columns):
        if any(h in str(col).lower() for h in hints):
            return i
    return 0


@st.cache_data(show_spinner=False)
def cached_holidays(country: str, first_year: int, last_year: int) -> pd.DataFrame:
    return country_holidays(country, range(first_year, last_year + 1))


@st.cache_data(show_spinner="Measuring each holiday's effect on your sales...")
def cached_impact(series: pd.Series, calendar: pd.DataFrame, country: str):
    impact = holiday_impact(series, calendar)
    if impact.empty:
        return impact, pd.Series(dtype=str)
    return impact, holiday_groups(impact, calendar, preferred=DEFAULT_SHORTLIST.get(country))


def supported_countries() -> list:
    try:
        import holidays
        return sorted(holidays.list_supported_countries())
    except Exception:
        return ["IN", "GB", "US"]


# ---------------------------------------------------------------- 1. your data
def page_data() -> None:
    page_title("Your sales data",
               "Upload a CSV with one column of dates and one column of sales, "
               "or try a demo business first.")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        choice = st.radio("Data source", ["Upload my own CSV", *DEMOS])
        if choice == "Upload my own CSV":
            uploaded = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
            if uploaded is None:
                with right:
                    note("Choose a CSV file on the left, or pick one of the demo businesses. "
                         "The file needs a date column and a sales column; any other columns are ignored.")
                return
            raw, name = pd.read_csv(uploaded), Path(uploaded.name).stem
        else:
            raw, name = pd.read_csv(PROJECT_DIR / DEMOS[choice]), Path(DEMOS[choice]).stem
            st.caption("Demo data is simulated by synthetic_data.py, not taken from a real business.")
        columns = list(raw.columns)
        date_col = st.selectbox("Date column", columns, index=guess_column(columns, ["date", "day", "time"]))
        value_col = st.selectbox("Sales column", columns,
                                 index=guess_column(columns, ["sales", "revenue", "amount", "value", "qty"]))
        check = st.button("Check this data", type="primary")

    with right:
        try:
            preview, preview_report = series_from_frame(raw, date_col, value_col)
        except Exception as error:
            preview = None
            note(f"These columns cannot be read as dates and sales yet: {error}", "warn")
        tag = '<span class="tag">simulated</span>' if choice != "Upload my own CSV" else ""
        st.markdown(f'<p class="panel-title">Preview of <b>{name}</b>{tag}</p>', unsafe_allow_html=True)
        if preview is not None:
            a, b, c = st.columns(3)
            a.metric("Rows", f"{len(raw):,}")
            b.metric("From", preview.index.min().strftime("%b %Y"))
            c.metric("To", preview.index.max().strftime("%b %Y"))
            st.plotly_chart(history_figure(preview, "", height=230), width="stretch")
        st.dataframe(raw.head(5), hide_index=True, width="stretch")

    if check:
        try:
            series, report = series_from_frame(raw, date_col, value_col)
        except Exception as error:
            note(f"The data could not be read: {error}", "error")
            return
        if name != state.get("data_name"):
            state.pop("selected", None)
        state.update(series=series, report=report, data_name=name, date_col=date_col, value_col=value_col)
        go_to(1)


# ---------------------------------------------------------------- 2. data check
def page_check() -> None:
    r = state.report
    page_title("Data check", "What the data looks like, and which models it can support.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recorded", r.frequency.capitalize())
    m2.metric("History", f"{r.years:.1f} years")
    m3.metric("Data points", f"{r.n_points:,}")
    m4.metric("Size tier", r.tier.replace("_", " ").capitalize())
    st.write("")
    if r.tier == "too_small":
        note(r.message + " Upload a longer history to forecast.", "error")
        return
    allowed = ", ".join(FAMILY_NAMES[f] for f in r.allowed_families)
    note(f"{r.message}\n\nModels that will compete: **{allowed}**. "
            f"Forecasts look **{r.horizon} {r.frequency_unit()}** ahead.")
    for warning in r.warnings:
        note(warning, "warn")
    st.plotly_chart(history_figure(state.series, "Sales history"), width="stretch")
    if st.button("Continue to festivals", type="primary"):
        go_to(2)


# ---------------------------------------------------------------- 3. festivals
def page_festivals() -> None:
    r, series = state.report, state.series
    page_title("Festivals and special days",
               "Which holidays move your sales? Each one is measured on your own history. "
               "Tick the ones the forecasting models should know about.")
    if r.frequency != "daily":
        note("Festivals are only used with daily data, because weekly or monthly totals hide single days.")
        state.selected, state.country = [], None
    elif not FESTIVAL_FAMILIES & set(r.allowed_families):
        note("The models that use festivals need at least 2 years of history, so they see each "
                "festival more than once. With this much data only simple models run, so festivals are skipped.")
        state.selected, state.country = [], None
    else:
        countries = supported_countries()
        left, right = st.columns([1, 2])
        current = state.get("country") or "IN"
        country = left.selectbox("Country", countries,
                                 index=countries.index(current) if current in countries else 0)
        custom_file = right.file_uploader("Your own special dates (optional CSV with columns: date, name)",
                                          type="csv")
        calendar = cached_holidays(country, series.index.min().year, series.index.max().year + 1)
        if custom_file is not None:
            try:
                calendar = pd.concat([calendar, custom_dates(custom_file)]).sort_values("date")
            except Exception as error:
                note(f"The special-dates file could not be read: {error}", "error")
        state.calendar_full = calendar
        impact, groups = cached_impact(series, calendar, country)
        if impact.empty:
            note("None of these holidays fall inside your data, so no festivals will be used.")
            state.selected, state.country = [], country
        else:
            memory_key = (state.data_name, country)
            ticked = state.selections.get(memory_key, default_selection(impact, groups))
            table = impact.join(groups).reset_index()
            table.insert(1, "use", table["holiday"].isin(ticked))
            table = table.sort_values(["use", "avg_uplift_%"], ascending=[False, False],
                                      key=lambda col: col.abs() if col.name == "avg_uplift_%" else col)
            table["consistent"] = table["consistent"].map({True: "Yes", False: "No"})
            table = table[["use", "holiday", "avg_uplift_%", "peak_day_uplift_%", "times_seen",
                           "consistent", "group"]]
            st.caption("Uplift compares sales from a week before each holiday to the day after with normal "
                       "days. Holidays in the same group fall within days of each other, so their effects "
                       "cannot be told apart: tick one per group. Consistent means the effect pointed the "
                       "same way every year it appeared, at least 3 times.")
            edited = st.data_editor(
                table,
                column_config={
                    "use": st.column_config.CheckboxColumn("Use", width="small"),
                    "holiday": st.column_config.TextColumn("Holiday", width="medium"),
                    "avg_uplift_%": st.column_config.NumberColumn("Average uplift", format="%+.0f%%"),
                    "peak_day_uplift_%": st.column_config.NumberColumn("Busiest day", format="%+.0f%%"),
                    "times_seen": st.column_config.NumberColumn("Years seen"),
                    "consistent": st.column_config.TextColumn("Consistent", width="small"),
                    "group": st.column_config.TextColumn("Group", width="medium"),
                },
                disabled=["holiday", "avg_uplift_%", "peak_day_uplift_%", "times_seen", "consistent", "group"],
                hide_index=True, width="stretch", height=420,
                key=f"editor_{state.data_name}_{country}",
            )
            selected = edited.loc[edited["use"], "holiday"].tolist()
            state.selections[memory_key] = selected
            state.selected, state.country = selected, country
            if selected:
                st.plotly_chart(holidays_figure(series, calendar, selected, "Your ticked holidays on the sales history"),
                                width="stretch")
            else:
                note("No holidays ticked. The forecasting models will run without festival dates.")
    if st.button("Continue to run", type="primary"):
        go_to(3)


# ---------------------------------------------------------------- 4. run
def page_run() -> None:
    r, series = state.report, state.series
    page_title("Run the model competition",
               "Every model is tested the same way: it forecasts several past periods it has "
               "never seen, covering about a year. The most accurate one wins.")
    families = [f for f in r.allowed_families if f != "deep_learning"]
    if "deep_learning" in r.allowed_families:
        state.include_dl = st.toggle("Include deep learning (LSTM). Usually adds several minutes.",
                                     value=bool(state.get("include_dl")))
        if state.include_dl:
            families.append("deep_learning")
    else:
        state.include_dl = False

    n_windows = len(plan_folds(len(series), r.horizon, r.frequency))
    festivals = state.get("selected") or []
    st.markdown(
        f"- **Competing:** {', '.join(FAMILY_NAMES[f] for f in families)}\n"
        f"- **Backtest:** {n_windows} window{'s' if n_windows != 1 else ''} of {r.horizon} "
        f"{r.frequency_unit()}, then one final test on data no model has seen\n"
        f"- **Festivals given to the models:** {', '.join(festivals) if festivals else 'none'}")

    key = run_key()
    if key in state.results:
        note("These settings have already been run. The results are ready.", "ok")
        if st.button("See results", type="primary"):
            go_to(4)
        return
    if st.button("Run forecast", type="primary"):
        calendar = None
        if festivals:
            full = state.calendar_full
            calendar = full[full["name"].isin(festivals)]
        bar = st.progress(0.0, text="Starting...")
        try:
            result = run_competition(series, r, calendar, families=families,
                                     on_progress=lambda msg, f: bar.progress(min(f, 1.0), text=msg))
        except Exception as error:
            note(f"The competition could not run: {error}", "error")
            return
        state.results[key] = result
        go_to(4)


# ---------------------------------------------------------------- 5. results
def page_results() -> None:
    r, series = state.report, state.series
    result = state.results[run_key()]
    board = result.leaderboard
    backtest_wape = board.loc[result.winner, "WAPE_%"]
    page_title("Results", f"The best model for {state.data_name}, and how much to trust it.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Best model", result.winner.replace("_", " "))
    m2.metric("Expected error", f"{backtest_wape:.1f}%", help="Total error as a % of total sales, "
              "averaged over all backtest windows.")
    m3.metric("Final test error", f"{result.test_scores['WAPE_%']:.1f}%",
              help="Error on the last period, which no model saw while being chosen.")
    st.markdown(f'<p class="verdict">For this business, <b>{result.winner.replace("_", " ")}</b> '
                f'forecast best. Expect its forecasts to be off by about <b>{backtest_wape:.0f}%</b> '
                f'in total. On the final period, which no model saw while being chosen, it was off by '
                f'{result.test_scores["WAPE_%"]:.0f}%.</p>', unsafe_allow_html=True)
    if not result.covers_full_year:
        note("With this much data the backtest does not cover a full year, so some seasons "
             "were never tested. Treat the accuracy with caution.", "warn")

    st.plotly_chart(forecast_figure(series, result, "Forecast"), width="stretch")

    st.subheader("Leaderboard")
    table = board.reset_index()
    table["model"] = table["model"].str.replace("_", " ")
    table["family"] = table["family"].map(FAMILY_NAMES).fillna(table["family"])
    cols = [c for c in ["model", "family", "WAPE_%", "MAE", "RMSE", "seconds", "status"] if c in table]
    st.dataframe(
        table[cols].round({"MAE": 0, "RMSE": 0}), hide_index=True, width="stretch",
        column_config={
            "model": "Model", "family": "Family",
            "WAPE_%": st.column_config.NumberColumn("Error (WAPE)", format="%.1f%%"),
            "MAE": st.column_config.NumberColumn("Avg error per period", format="localized"),
            "RMSE": st.column_config.NumberColumn("RMSE", format="localized"),
            "seconds": st.column_config.NumberColumn("Time", format="%.0f s"),
            "status": "Status",
        })

    left, right = st.columns(2)
    with left:
        st.subheader("Winner, window by window")
        folds = result.winner_fold_scores[["from", "to", "WAPE_%"]].reset_index(drop=True)
        st.dataframe(folds, hide_index=True, width="stretch",
                     column_config={"from": "From", "to": "To",
                                    "WAPE_%": st.column_config.NumberColumn("Error", format="%.1f%%")})
        st.caption("A window much worse than the others points to a season the model handles badly.")
    with right:
        st.subheader("Future forecast")
        future = result.future_forecast.rename("forecast").round(0).to_frame()
        future.index = future.index.strftime("%Y-%m-%d")
        future.index.name = "date"
        st.dataframe(future, width="stretch", height=280,
                     column_config={"forecast": st.column_config.NumberColumn("Forecast", format="localized")})
        st.download_button("Download forecast (CSV)", future.to_csv().encode("utf-8"),
                           file_name=f"{state.data_name}_forecast.csv", mime="text/csv")
    st.write("")
    if st.button("Continue to your business story", type="primary"):
        go_to(5)


# ---------------------------------------------------------------- shared helpers for pages 6-7
def section(title: str) -> None:
    st.markdown(f'<p class="section">{title}</p>', unsafe_allow_html=True)


def finding(text: str) -> None:
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    st.markdown(f'<p class="finding">{html}</p>', unsafe_allow_html=True)


def currency() -> str:
    """The currency symbol chosen by the user (rupees by default for India)."""
    default = "₹" if state.get("country") in (None, "IN") else "(none)"
    symbol = state.get("currency", default)
    return "" if symbol == "(none)" else symbol


def selected_calendar():
    names = state.get("selected") or []
    full = state.get("calendar_full")
    if not names or full is None:
        return None, []
    return full, names


# ---------------------------------------------------------------- 6. business story
def page_story() -> None:
    r, series = state.report, state.series
    page_title("Your business story", f"What {state.data_name}'s own sales history says: how the business "
               "is doing, when customers buy, and what drives the peaks.")
    top_left, top_right = st.columns([3, 1])
    with top_right:
        options = CURRENCIES
        state.currency = st.selectbox("Currency", options, index=options.index(state.get("currency", "₹")))
    sym = currency()
    calendar, names = selected_calendar()

    # --- inflation source
    cpi = infl.load_cpi()
    with top_left:
        with st.expander("Inflation settings"):
            choices = (["Official Indian consumer prices (CPI)"] if cpi is not None else []) + ["A yearly rate I choose"]
            source = st.radio("Measure inflation with", choices, horizontal=True)
            if source.startswith("Official"):
                price_index, label = cpi, "official Indian consumer price index"
            else:
                rate = st.number_input("Yearly price rise (%)", 0.0, 30.0, float(state.get("rate", infl.DEFAULT_RATE)), 0.5,
                                       help="4% is the RBI's inflation target. Use your own price rises if you know them.")
                state.rate = rate
                price_index = infl.index_from_rate(series.index.min(), series.index.max(), rate)
                label = f"{rate:g}% a year"
            if cpi is None:
                st.caption("Official CPI data is not loaded yet (data_external/cpi_india.csv), so a yearly rate is used.")
    state.price_index = price_index

    # --- 1. how the business is doing
    section("How the business is doing")
    growth = ins.growth_summary(series, r.frequency)
    real = infl.real_growth(series, price_index, label)
    if growth is None:
        finding("There is less than two years of history, so year-on-year growth cannot be measured yet.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Sales, last 12 months", ins.money(growth.last_12, sym))
        m2.metric("Change vs the 12 before", f"{growth.change_pct:+.1f}%")
        if real is not None:
            m3.metric("Real growth after inflation", f"{real.real_growth_pct:+.1f}%",
                      help=f"Inflation measured with: {real.source}")
        if real is None:
            note("The inflation data does not cover your whole last two years (the official Indian CPI file "
                 "runs from Jan 2013 to its latest month). Choose a yearly rate in Inflation settings instead.",
                 "warn")
        words = "grew" if growth.change_pct >= 0 else "fell"
        text = (f"Over the last 12 months the business took **{ins.money(growth.last_12, sym)}**, which "
                f"{words} **{abs(growth.change_pct):.1f}%** compared with the year before.")
        if real is not None:
            text += (f" Prices rose about **{real.inflation_pct:.1f}%** over the same time ({real.source}), so "
                     f"in real terms the business {'grew' if real.real_growth_pct >= 0 else 'shrank'} "
                     f"**{abs(real.real_growth_pct):.1f}%**: that is the part that comes from selling more, "
                     f"not from higher prices.")
        finding(text)
        if len(growth.yearly) >= 2:
            st.plotly_chart(money_bars([str(y) for y in growth.yearly.index], growth.yearly.to_numpy(),
                                       "Sales per full calendar year", symbol=sym), width="stretch")
    table = infl.in_todays_prices(series, price_index)
    if table is not None and len(table) >= 24:
        with st.expander("Monthly sales in today's prices"):
            st.plotly_chart(real_vs_money_figure(table, "Monthly sales, as recorded and in today's prices", sym),
                            width="stretch")
            st.caption("Older months are scaled up by how much prices have risen since, so all months "
                       "compare like with like.")
    for shock in ins.shocks(series, r.frequency):
        if shock.yearly:
            finding(f"The business closes for about **{shock.days_closed} days** around "
                    f"{shock.start.strftime('%d %b')} most years; this is treated as a planned closure.")
        else:
            rec = (f"Sales took about **{shock.recovery_days} days** to get back to 90% of normal."
                   if shock.recovery_days is not None else "Sales had not recovered to normal within a year.")
            finding(f"From **{shock.start.strftime('%d %b %Y')}** the business was closed for "
                    f"**{shock.days_closed} days** in a row. Counting the slow reopening, that cost about "
                    f"**{ins.money(shock.lost_sales, sym)}** in sales. {rec}")

    # --- 2. when customers buy
    section("When customers buy")
    weekdays = ins.weekday_pattern(series, r.frequency)
    months = ins.month_pattern(series)
    left, right = st.columns(2)
    if weekdays is not None and len(weekdays) >= 2:
        with left:
            st.plotly_chart(percent_bars(weekdays, "Each weekday vs an average day (last 12 months)"), width="stretch")
            finding(f"**{weekdays.idxmax()}** is the strongest day (**{weekdays.max():+.0f}%**) and "
                    f"**{weekdays.idxmin()}** the weakest (**{weekdays.min():+.0f}%**).")
    if len(months):
        with right if weekdays is not None else left:
            st.plotly_chart(percent_bars(months, "Each month vs an average month (typical year)"), width="stretch")
            finding(f"The strongest month is usually **{months.idxmax()}** (**{months.max():+.0f}%**), the "
                    f"weakest **{months.idxmin()}** (**{months.min():+.0f}%**).")
    else:
        finding("A full calendar year of data is needed to show the monthly pattern.")

    # --- 3. what drives the peaks
    section("What drives the peaks")
    if r.frequency != "daily":
        finding("Festival effects need daily data: weekly or monthly totals hide single days.")
    elif not FESTIVAL_FAMILIES & set(r.allowed_families):
        finding("Festival effects need at least 2 years of history, so each festival is seen more than once.")
    elif calendar is None:
        finding("No festivals were ticked on the Festivals step, so there is nothing to measure here.")
    else:
        value = ins.festival_value(series, calendar, names)
        share = ins.festival_share(series, calendar, names)
        if len(value):
            st.plotly_chart(money_bars(value["festival"], value["extra_sales_per_year"],
                                       "Extra sales each festival brings in a year (recent years)",
                                       color="#F4A63A", symbol=sym), width="stretch")
            top = value.iloc[0]
            text = (f"**{top['festival']}** is worth about **{ins.money(top['extra_sales_per_year'], sym)}** of "
                    f"extra sales each year, with its busiest day around **{top['peak_day_vs_normal']:.1f}×** "
                    f"a normal day.")
            if share is not None:
                text += (f" Over the last 12 months, **{share:.0f}%** of all sales happened in the weeks "
                         f"around your ticked festivals.")
            finding(text)
    st.write("")
    if st.button("Continue to plan ahead", type="primary"):
        go_to(6)


# ---------------------------------------------------------------- 7. plan ahead
def page_plan() -> None:
    r, series = state.report, state.series
    result = state.results[run_key()]
    wape = float(result.leaderboard.loc[result.winner, "WAPE_%"])
    future = result.future_forecast
    sym = currency()
    calendar, names = selected_calendar()
    page_title("Plan ahead", f"The next {r.horizon} {r.frequency_unit()} turned into plans for stock, "
               "staff and cash. Ranges come from how accurate the model was in testing.")

    # --- outlook
    look = ins.outlook(series, future, r.frequency, wape)
    bias = ins.forecast_bias(series, result.backtest_forecast)
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Expected sales, next {r.horizon} {r.frequency_unit()}", ins.money(look.total, sym))
    m2.metric("Likely range", f"{ins.money(look.low, sym)} to {ins.money(look.high, sym)}")
    if look.change_pct is not None:
        m3.metric("Vs the same period last year", f"{look.change_pct:+.1f}%")
    if r.tier == "small":
        note("With less than 2 years of history the models cannot learn yearly peaks such as festival "
             "seasons. If last year's same period had a festival rush, this forecast will not show it.", "warn")
    if abs(bias) >= 3:
        direction = "below" if bias < 0 else "above"
        note(f"In testing, this model's totals came out about **{abs(bias):.0f}% {direction}** what really "
             f"happened. Keep that in mind when reading the totals.", "warn")
    if r.frequency == "daily":
        busiest = ", ".join(f"**{d.strftime('%a %d %b')}** ({ins.money(v, sym)})" for d, v in look.busiest.items())
        finding(f"Busiest days ahead: {busiest}.")

    # --- weekly plan
    section("Week by week: stock and staff")
    c1, c2, c3 = st.columns(3)
    unit_value = c1.number_input(f"Average value of one unit ({sym or 'money'})", min_value=0.0, value=0.0, step=10.0,
                                 help="For example the average price of one box or one kg. Leave at 0 to skip units.")
    unit_name = c2.text_input("Unit name", value="units")
    buffer = c3.number_input("Safety buffer (%)", 0.0, 100.0, round(wape / 2, 0), 1.0,
                             help="Extra stock on top of the forecast. The default is half the model's tested error.")
    plan = ins.weekly_plan(future, r.frequency, wape, unit_value or None, buffer)
    if plan is None:
        finding("Weekly plans need daily or weekly data.")
    else:
        st.plotly_chart(weekly_plan_figure(plan, "Expected sales per week (lines show the likely range)", sym),
                        width="stretch")
        show = plan.copy()
        show[["expected", "low", "high"]] = show[["expected", "low", "high"]].round(0)
        show["week_starting"] = show["week_starting"].dt.strftime("%d %b %Y")
        config = {"week_starting": "Week starting", "how_busy": "How busy",
                  "expected": st.column_config.NumberColumn(f"Expected ({sym})", format="localized"),
                  "low": st.column_config.NumberColumn("Low", format="localized"),
                  "high": st.column_config.NumberColumn("High", format="localized")}
        if "units_to_prepare" in show:
            show = show.rename(columns={"units_to_prepare": unit_name or "units"})
            config[unit_name or "units"] = st.column_config.NumberColumn(f"{unit_name} to prepare", format="localized")
        st.dataframe(show, hide_index=True, width="stretch", column_config=config)
        busy = plan[plan["how_busy"] == "Busy"]
        if len(busy):
            finding(f"**{len(busy)} busy week{'s' if len(busy) != 1 else ''}** ahead (at least 20% above a "
                    f"typical week): plan extra staff and stock for "
                    f"{', '.join(busy['week_starting'].dt.strftime('%d %b'))}.")

    # --- cash
    section("Money coming in, month by month")
    cash = ins.monthly_cash(future, wape)
    st.plotly_chart(cash_figure(cash, "Expected sales per month (lines show the likely range)", sym), width="stretch")
    partial = cash[cash["days_covered"] < 28] if r.frequency == "daily" else cash.iloc[0:0]
    if len(partial):
        st.caption(f"{', '.join(partial['month'])} is only partly inside the forecast period.")

    # --- suggestions
    section("Rules of thumb")
    price_index = state.get("price_index")             # chosen on the Business story page
    if price_index is None:
        price_index = infl.index_from_rate(series.index.min(), series.index.max(), infl.DEFAULT_RATE)
    real = infl.real_growth(series, price_index, "")
    tips = ins.suggestions(series, future, r.frequency, calendar, names, wape, sym or "",
                           real.real_growth_pct if real else None, real.money_growth_pct if real else None)
    st.caption("Simple rules based on the numbers above, not expert business advice. You know your business best.")
    for tip in tips:
        html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", tip)
        st.markdown(f'<p class="tip">{html}</p>', unsafe_allow_html=True)
    if plan is not None:
        st.download_button("Download the weekly plan (CSV)", plan.to_csv(index=False).encode("utf-8"),
                           file_name=f"{state.data_name}_weekly_plan.csv", mime="text/csv")
    st.write("")
    c1, c2 = st.columns(2)
    if c1.button("Try your own assumptions (What if)", width="stretch"):
        go_to(7)
    if c2.button("Compare with actual sales (Track actuals)", width="stretch"):
        go_to(8)


# ---------------------------------------------------------------- tool: what if
def page_whatif() -> None:
    r = state.report
    result = state.results[run_key()]
    wape = float(result.leaderboard.loc[result.winner, "WAPE_%"])
    future = result.future_forecast
    sym = currency()
    page_title("What if", "Try your own assumptions on top of the forecast. These numbers are your guesses, "
               "not something the model learned, so the result is only as good as the assumptions.")
    left, right = st.columns(2, gap="large")
    with left:
        price = st.slider("Change prices by (%)", -30, 30, 0, 1)
        quantity = st.slider("Customers then buy this much more or less (%)", -50, 50, 0, 1,
                             help="Your guess of how customers react. Higher prices usually mean somewhat "
                                  "fewer items sold; lower prices, somewhat more.")
    with right:
        outlet = st.slider("A new outlet or online channel adds (% of current sales)", 0, 100, 0, 5)
        outlet_start = st.date_input("Starting from", value=future.index[len(future) // 3].date(),
                                     min_value=future.index[0].date(), max_value=future.index[-1].date())
        promo = st.slider("Offers in the quiet weeks lift those weeks by (%)", 0, 50, 0, 5,
                          help="Only applied to weeks marked Quiet on the Plan ahead page.")
    adjusted = ins.scenario(future, r.frequency, price, quantity, outlet, pd.Timestamp(outlet_start), promo, wape)
    base_total, new_total = float(future.sum()), float(adjusted.sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast as it is", ins.money(base_total, sym))
    m2.metric("With your assumptions", ins.money(new_total, sym))
    change = (new_total / base_total - 1) * 100 if base_total else 0.0
    m3.metric("Difference", ins.money(new_total - base_total, sym),
              f"{change:+.1f}%" if abs(change) >= 0.05 else None)
    if price and not quantity:
        note("You changed prices but not how many items customers buy. In practice, customers usually react "
             "to price changes, so set the second slider to your best guess.", "warn")
    st.plotly_chart(scenario_figure(future, adjusted, "Forecast as it is vs with your assumptions", sym),
                    width="stretch")
    months = pd.DataFrame({"forecast": future.resample("MS").sum(), "with_assumptions": adjusted.resample("MS").sum()})
    months.index = months.index.strftime("%b %Y")
    months.index.name = "month"
    months["difference"] = months["with_assumptions"] - months["forecast"]
    months = months.round(0)
    st.dataframe(months, width="stretch", column_config={
        "forecast": st.column_config.NumberColumn("Forecast", format="localized"),
        "with_assumptions": st.column_config.NumberColumn("With your assumptions", format="localized"),
        "difference": st.column_config.NumberColumn("Difference", format="localized")})


# ---------------------------------------------------------------- tool: track actuals
def page_track() -> None:
    r = state.report
    result = state.results[run_key()]
    wape = float(result.leaderboard.loc[result.winner, "WAPE_%"])
    sym = currency()
    page_title("Track actuals", "As real sales come in, compare them with the forecast. Weeks that fall "
               "below the expected range point to something worth checking early.")
    left, right = st.columns(2, gap="large")
    with left:
        sources = ["Upload my actual sales (CSV)"]
        if state.data_name.startswith("sweet_shop"):
            sources.append("Simulated actual sales for the demo shop")
        source = st.radio("Actual sales", sources)
    actual = None
    with right:
        if source.startswith("Upload"):
            uploaded = st.file_uploader("CSV with a date column and a sales column, covering days after "
                                        f"{state.series.index.max().strftime('%d %b %Y')}", type="csv")
            if uploaded is not None:
                raw = pd.read_csv(uploaded)
                cols = list(raw.columns)
                c1, c2 = st.columns(2)
                dcol = c1.selectbox("Date column", cols, index=guess_column(cols, ["date", "day"]), key="t_date")
                vcol = c2.selectbox("Sales column", cols,
                                    index=guess_column(cols, ["sales", "revenue", "amount"]), key="t_val")
                try:
                    actual, _ = series_from_frame(raw, dcol, vcol)
                except Exception as error:
                    note(f"These columns cannot be read as dates and sales: {error}", "error")
        else:
            drop = st.toggle("Add a problem: sales fall 40% from mid-November", value=False,
                             help="Shows what an early warning looks like.")
            import synthetic_data
            sim = synthetic_data.simulate(synthetic_data.SWEET_SHOP, end=result.future_forecast.index[-1])["sales"]
            actual = sim.loc[sim.index > state.series.index.max()]
            if drop:
                actual = actual.where(actual.index < pd.Timestamp(actual.index[0].year, 11, 15), actual * 0.6)
            st.caption("Simulated with the same rules as the demo shop's history. Not real sales.")
    if actual is None:
        note("Add actual sales to compare them with the forecast.")
        return
    period = "week" if r.frequency in ("daily", "weekly") else "month"
    table = ins.track(result.future_forecast, actual, r.frequency, wape)
    if table.empty:
        note("None of these dates fall inside the forecast period "
             f"({result.future_forecast.index[0].strftime('%d %b %Y')} to "
             f"{result.future_forecast.index[-1].strftime('%d %b %Y')}).", "warn")
        return
    for kind, message in ins.tracking_alerts(table, period):
        note(message, kind)
    st.plotly_chart(tracking_figure(table, f"Actual sales vs the forecast range, per {period}", sym), width="stretch")
    show = table.copy()
    show[["expected", "low", "high", "actual"]] = show[["expected", "low", "high", "actual"]].round(0)
    show["period_starting"] = show["period_starting"].dt.strftime("%d %b %Y")
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "period_starting": f"{period.capitalize()} starting",
        "expected": st.column_config.NumberColumn("Expected", format="localized"),
        "low": st.column_config.NumberColumn("Low", format="localized"),
        "high": st.column_config.NumberColumn("High", format="localized"),
        "actual": st.column_config.NumberColumn("Actual", format="localized"),
        "difference_%": st.column_config.NumberColumn("Difference", format="%+.1f%%"),
        "status": "Status"})


# ---------------------------------------------------------------- show the current step
if not reachable(state.step):
    state.step = 0
[page_data, page_check, page_festivals, page_run, page_results, page_story, page_plan,
 page_whatif, page_track][state.step]()
