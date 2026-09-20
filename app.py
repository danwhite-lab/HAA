from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from haa.constants import ASSETS, DEFAULT_TAX_RATE
from haa.data import combine_replacements, common_monthly_period, date_ranges, download_yahoo_prices, read_uploaded_csv, to_month_end
from haa.engine import run_backtest
from haa.metrics import annual_returns, performance_metrics
from haa.strategies import HAASimple

st.set_page_config(page_title="HAA-Simple Backtest", layout="wide")
st.title("HAA-Simple — transparent monthly backtest")
st.caption("Canonical rules only. Decisions are made at month-end and executed for the following month; no optimization or synthetic history.")

@st.cache_data(ttl=3600, show_spinner="Downloading Yahoo Finance price history...")
def load_data():
    return download_yahoo_prices()

try:
    downloaded = load_data()
except Exception as exc:
    st.error(f"Data download failed: {exc}")
    st.stop()

with st.sidebar:
    st.header("Controls")
    uploads = {asset: st.file_uploader(f"Replace {asset} CSV", type="csv", key=asset) for asset in ASSETS}
    initial = st.number_input("Initial investment", min_value=1.0, value=100_000.0, step=1_000.0)
    cost_pct = st.number_input("Transaction cost per entry/change (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.01) / 100
    tax_enabled = st.toggle("Israeli capital-gains tax", value=False)
    tax_rate = st.number_input("Tax rate (%)", min_value=0.0, max_value=100.0, value=DEFAULT_TAX_RATE * 100, step=0.1, disabled=not tax_enabled) / 100

replacements = {}
for asset, upload in uploads.items():
    if upload is not None:
        try:
            replacements[asset] = read_uploaded_csv(upload.getvalue(), asset)
        except ValueError as exc:
            st.sidebar.error(str(exc))
prices = combine_replacements(downloaded, replacements)
monthly = to_month_end(prices)
ranges = date_ranges(prices)
common_start, common_end = common_monthly_period(monthly)

st.subheader("Data coverage")
st.dataframe(ranges, use_container_width=True)
if common_start is None:
    st.error("The four assets have no common month-end observations. Upload compatible CSV histories.")
    st.stop()
st.info(f"Actual common monthly data period: {common_start.date()} through {common_end.date()}.")
with st.sidebar:
    start = st.date_input("Backtest start (holding-period end)", value=common_start.date(), min_value=common_start.date(), max_value=common_end.date())
    end = st.date_input("Backtest end (holding-period end)", value=common_end.date(), min_value=common_start.date(), max_value=common_end.date())

strategy = HAASimple()
decisions = strategy.decisions(monthly)
if decisions.empty:
    st.error("Insufficient common history: HAA-Simple needs 13 complete month-end observations before its first signal.")
    st.stop()
first_signal = decisions.index.min()
try:
    result = run_backtest(decisions, monthly, initial, cost_pct, tax_enabled, tax_rate, pd.Timestamp(start), pd.Timestamp(end))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

backtest_tab, validation_tab = st.tabs(["Backtest", "Validation"])
with backtest_tab:
    st.caption(f"Holding periods: {result.monthly.index.min().date()} through {result.monthly.index.max().date()}. SPY benchmark uses these same monthly periods.")
    comparison = {"HAA-Simple pre-tax": performance_metrics(result.monthly["pre_tax_value"], initial), "SPY buy-and-hold": performance_metrics(result.monthly["benchmark_value"], initial)}
    if tax_enabled:
        comparison["HAA-Simple after-tax"] = performance_metrics(result.monthly["after_tax_value"], initial)
    summary = pd.DataFrame(comparison)
    changes = int(result.monthly["allocation_change"].sum())
    years = len(result.monthly) / 12
    summary.loc["Allocation changes", "HAA-Simple pre-tax"] = changes
    summary.loc["Average changes/year", "HAA-Simple pre-tax"] = changes / years if years else 0
    summary.loc["Annual turnover", "HAA-Simple pre-tax"] = changes / years if years else 0
    st.subheader("Results")
    st.dataframe(summary.style.format("{:.2%}", subset=[x for x in summary.index if x not in {"Final value", "Allocation changes", "Average changes/year", "Annual turnover"}]).format("{:.2f}", subset=["Final value", "Allocation changes", "Average changes/year", "Annual turnover"]), use_container_width=True)
    curves = result.monthly[["pre_tax_value", "benchmark_value"]].rename(columns={"pre_tax_value": "HAA-Simple pre-tax", "benchmark_value": "SPY buy-and-hold"})
    if tax_enabled:
        curves["HAA-Simple after-tax"] = result.monthly["after_tax_value"]
    st.plotly_chart(px.line(curves, title="Equity curve"), use_container_width=True)
    drawdowns = curves.div(curves.cummax()).sub(1)
    st.plotly_chart(px.line(drawdowns, title="Drawdown"), use_container_width=True)
    st.subheader("Annual returns")
    annual = pd.concat({"HAA-Simple pre-tax": annual_returns(result.monthly["pre_tax_monthly_return"]), "SPY buy-and-hold": annual_returns(result.monthly["benchmark_monthly_return"])}, axis=1)
    if tax_enabled:
        annual["HAA-Simple after-tax"] = annual_returns(result.monthly["after_tax_monthly_return"])
    st.dataframe(annual.style.format("{:.2%}"), use_container_width=True)
    st.subheader("Monthly returns")
    monthly_returns = pd.DataFrame({"HAA-Simple pre-tax": result.monthly["pre_tax_monthly_return"], "SPY buy-and-hold": result.monthly["benchmark_monthly_return"]})
    if tax_enabled:
        monthly_returns["HAA-Simple after-tax"] = result.monthly["after_tax_monthly_return"]
    st.dataframe(monthly_returns.style.format("{:.2%}"), use_container_width=True)

with validation_tab:
    st.subheader("Rules and calculation")
    st.markdown("""**HAA-Simple:** at each month-end calculate 13612W momentum for SPY and TIP. If both are strictly positive, select SPY. Otherwise select IEF when IEF momentum is greater than BIL momentum; select BIL on a tie or when BIL is greater. The selection earns the *following* month’s return only.

**13612W:** `(12×1-month return + 4×3-month return + 2×6-month return + 1×12-month return) / 4`. Each return is `price at signal date / price at its historical month-end - 1`. This implementation therefore requires 12 earlier complete month-end observations and uses no later prices.

**Data:** Yahoo Finance download, using `Adj Close` when Yahoo supplies it and `Close` only as a visible fallback. Uploaded CSV data replaces an asset’s entire history. No missing ETF history is fabricated.

**Tax:** applies only when an existing position is sold due to an allocation change. It tracks cost basis and loss carryforward, never taxes the final unrealized position, and is independent of the strategy module.""")
    st.write(f"First valid signal date: **{first_signal.date()}**")
    missing = monthly[monthly.isna().any(axis=1)]
    st.write(f"Months with at least one missing canonical price: **{len(missing)}**")
    st.subheader("Monthly audit table")
    audit_columns = [f"{asset}_price" for asset in ASSETS] + [f"{asset}_13612w" for asset in ASSETS] + ["regime", "selected_asset", "previous_asset", "trade", "holding_end", "holding_period_return"]
    audit = result.audit[audit_columns]
    st.dataframe(audit.style.format("{:.6f}", subset=[c for c in audit.columns if c.endswith("13612w") or c.endswith("return")]), use_container_width=True)
    st.download_button("Download audit CSV", audit.to_csv().encode("utf-8"), "haa_simple_monthly_audit.csv", "text/csv")
    st.subheader("Raw and monthly data used")
    st.dataframe(prices, use_container_width=True)
    st.dataframe(monthly, use_container_width=True)
    st.caption("Automated validation: run `pytest` locally; tests cover strategy selection, timing, benchmark dates, tax realization, and a hand-calculated 13612W example.")
