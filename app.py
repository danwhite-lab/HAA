from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from haa.constants import ASSETS, DEFAULT_TAX_RATE
from haa.data import combine_replacements, common_monthly_period, date_ranges, default_ticker_map, download_yahoo_prices, parse_ticker_map, read_uploaded_csv, to_month_end, upload_asset_from_filename
from haa.engine import run_backtest
from haa.metrics import annual_returns, performance_metrics
from haa.signals import first_trading_day_after, latest_actionable_signal
from haa.strategies import HAASimple, HAASimpleLeveraged2x

MODEL_OPTIONS = {"HAA-Simple": HAASimple, "HAA-Simple Leveraged 2x (SSO)": HAASimpleLeveraged2x}

st.set_page_config(page_title="HAA Backtest", layout="wide")


def sync_model_from_sidebar() -> None:
    st.session_state["signals_model_name"] = st.session_state["model_name"]


def sync_model_from_signals() -> None:
    st.session_state["model_name"] = st.session_state["signals_model_name"]


if "model_name" not in st.session_state:
    st.session_state["model_name"] = next(iter(MODEL_OPTIONS))
if "signals_model_name" not in st.session_state:
    st.session_state["signals_model_name"] = st.session_state["model_name"]

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("HAA model", tuple(MODEL_OPTIONS), key="model_name", on_change=sync_model_from_sidebar)
    strategy = MODEL_OPTIONS[model_name]()
    data_assets = getattr(strategy, "data_assets", ASSETS)
    ticker_text = st.text_area("Yahoo Finance ticker sources", value="\n".join(f"{role}={ticker}" for role, ticker in default_ticker_map(data_assets).items()), help="One asset role per line. Example: SPY=SPY. The leveraged model also downloads SSO, but uses SPY—not SSO—for risk-on signals.")
    uploads = st.file_uploader("Upload replacement CSV files", type="csv", accept_multiple_files=True, help=f"Upload one or more files named with one valid asset: {', '.join(data_assets)}.")
    initial = st.number_input("Initial investment", min_value=1.0, value=100_000.0, step=1_000.0)
    cost_pct = st.number_input("Transaction cost per entry/change (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.01) / 100
    tax_enabled = st.toggle("Israeli capital-gains tax", value=False)
    tax_rate = st.number_input("Tax rate (%)", min_value=0.0, max_value=100.0, value=DEFAULT_TAX_RATE * 100, step=0.1, disabled=not tax_enabled) / 100

try:
    ticker_map = parse_ticker_map(ticker_text, data_assets)
except ValueError as exc:
    st.sidebar.error(str(exc))
    st.stop()

@st.cache_data(ttl=3600, show_spinner="Downloading Yahoo Finance price history...")
def load_data(source_items: tuple[tuple[str, str], ...]):
    return download_yahoo_prices(dict(source_items))

try:
    downloaded = load_data(tuple(ticker_map.items()))
except Exception as exc:
    st.error(f"Yahoo Finance download failed: {exc}")
    st.stop()

replacements = {}
for upload in uploads or []:
    try:
        asset = upload_asset_from_filename(upload.name, data_assets)
        if asset in replacements:
            raise ValueError(f"More than one upload targets {asset}; upload only one replacement file per canonical role.")
        replacements[asset] = read_uploaded_csv(upload.getvalue(), asset)
    except ValueError as exc:
        st.sidebar.error(str(exc))
prices = combine_replacements(downloaded, replacements, data_assets)
monthly = to_month_end(prices)
ranges = date_ranges(prices)
common_start, common_end = common_monthly_period(monthly)

st.subheader("Data coverage")
st.dataframe(ranges, use_container_width=True)
if common_start is None:
    st.error("The selected model assets have no common month-end observations. Check the Yahoo ticker mappings or upload compatible CSV histories.")
    st.stop()
st.info(f"Actual common monthly data period: {common_start.date()} through {common_end.date()}.")
with st.sidebar:
    start = st.date_input("Backtest start (holding-period end)", value=common_start.date(), min_value=common_start.date(), max_value=common_end.date())
    end = st.date_input("Backtest end (holding-period end)", value=common_end.date(), min_value=common_start.date(), max_value=common_end.date())

decisions = strategy.decisions(monthly)
if decisions.empty:
    st.error(f"Insufficient history for {strategy.name}.")
    st.stop()
first_signal = decisions.index.min()
try:
    result = run_backtest(decisions, monthly, initial, cost_pct, tax_enabled, tax_rate, pd.Timestamp(start), pd.Timestamp(end))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

backtest_tab, signals_tab, validation_tab = st.tabs(["Backtest", "Signals", "Validation"])
with backtest_tab:
    st.title(f"{strategy.name} — transparent monthly backtest")
    st.caption("Signals are evaluated at month-end and execute for the following holding period; no optimization or synthetic history.")
    if hasattr(strategy, "risk_warning"):
        st.warning(strategy.risk_warning)
    st.caption(f"Holding periods: {result.monthly.index.min().date()} through {result.monthly.index.max().date()}. SPY benchmark uses these same monthly periods.")
    pre_tax_label = f"{strategy.name} pre-tax"
    after_tax_label = f"{strategy.name} after-tax"
    comparison = {pre_tax_label: performance_metrics(result.monthly["pre_tax_value"], initial), "SPY buy-and-hold": performance_metrics(result.monthly["benchmark_value"], initial)}
    if tax_enabled:
        comparison[after_tax_label] = performance_metrics(result.monthly["after_tax_value"], initial)
    summary = pd.DataFrame(comparison)
    changes = int(result.monthly["allocation_change"].sum())
    years = len(result.monthly) / 12
    summary.loc["Allocation changes", pre_tax_label] = changes
    summary.loc["Average changes/year", pre_tax_label] = changes / years if years else 0
    summary.loc["Annual turnover", pre_tax_label] = changes / years if years else 0
    st.subheader("Results")
    percentage_rows = ["CAGR", "Total return", "Maximum drawdown", "Annualized volatility", "Best month", "Worst month", "Annual turnover"]
    ratio_rows = ["Sharpe", "Sortino", "Calmar"]
    numeric_rows = ["Final value", "Allocation changes", "Average changes/year"]
    styled_summary = summary.style.format("{:.2%}", subset=pd.IndexSlice[percentage_rows, :]).format("{:.2f}", subset=pd.IndexSlice[ratio_rows + numeric_rows, :])
    st.dataframe(styled_summary, use_container_width=True)
    curves = result.monthly[["pre_tax_value", "benchmark_value"]].rename(columns={"pre_tax_value": pre_tax_label, "benchmark_value": "SPY buy-and-hold"})
    if tax_enabled:
        curves[after_tax_label] = result.monthly["after_tax_value"]
    st.plotly_chart(px.line(curves, title="Equity curve"), use_container_width=True)
    drawdowns = curves.div(curves.cummax()).sub(1)
    st.plotly_chart(px.line(drawdowns, title="Drawdown"), use_container_width=True)
    st.subheader("Annual returns")
    annual = pd.concat({pre_tax_label: annual_returns(result.monthly["pre_tax_monthly_return"]), "SPY buy-and-hold": annual_returns(result.monthly["benchmark_monthly_return"])}, axis=1)
    if tax_enabled:
        annual[after_tax_label] = annual_returns(result.monthly["after_tax_monthly_return"])
    st.dataframe(annual.style.format("{:.2%}"), use_container_width=True)
    st.subheader("Monthly returns")
    monthly_returns = pd.DataFrame({pre_tax_label: result.monthly["pre_tax_monthly_return"], "SPY buy-and-hold": result.monthly["benchmark_monthly_return"]})
    if tax_enabled:
        monthly_returns[after_tax_label] = result.monthly["after_tax_monthly_return"]
    st.dataframe(monthly_returns.style.format("{:.2%}"), use_container_width=True)

with signals_tab:
    st.title(f"{strategy.name} — current monthly signal")
    st.selectbox("Model", tuple(MODEL_OPTIONS), key="signals_model_name", on_change=sync_model_from_signals, help="Changing the model refreshes the data and all tabs using that model.")
    st.caption("This view uses the same completed-month decisions as the backtest. It does not calculate a separate live strategy.")
    signal_status = latest_actionable_signal(decisions, monthly, data_assets)
    raw_ranges = date_ranges(prices)
    st.subheader("Data status")
    st.dataframe(raw_ranges[["last_available"]], use_container_width=True)
    st.caption(f"Latest eligible completed month: {signal_status.completed_through.date()}. A partial current month is never presented as a final signal.")
    if signal_status.decision is None:
        st.error(signal_status.reason)
    else:
        signal = signal_status.decision
        signal_date = pd.Timestamp(signal.name)
        effective_start = first_trading_day_after(prices, signal_date)
        previous = signal["previous_asset"] if pd.notna(signal["previous_asset"]) else "No prior allocation"
        action = "Hold" if not bool(signal["trade"]) else (f"Buy {signal['selected_asset']}" if previous == "No prior allocation" else f"Switch {previous} → {signal['selected_asset']}")
        st.success(f"**Target allocation: 100% {signal['selected_asset']}**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Signal date", signal_date.date().isoformat())
        col2.metric("Regime", str(signal["regime"]).replace("-", " ").title())
        col3.metric("Instruction", action)
        col4.metric("Previous allocation", previous)
        if effective_start is not None:
            st.info(f"Effective holding period: **{effective_start.date()}** until the next month-end decision, subject to your own execution timing.")
        else:
            st.warning("No later trading observation is available yet, so an effective start date cannot be shown.")
        st.subheader("Why this allocation")
        if signal["regime"] == "risk-on":
            holding = "SSO" if isinstance(strategy, HAASimpleLeveraged2x) else "SPY"
            st.write(f"SPY and TIP 13612U momentum are both strictly positive, so the model selects {holding}.")
        else:
            st.write(f"At least one of SPY or TIP 13612U momentum is not positive, so the model selects the higher-momentum defensive asset: {signal['selected_asset']}.")
        price_columns = [f"{asset}_price" for asset in data_assets]
        momentum_columns = [f"{asset}_13612u" for asset in ASSETS]
        inputs = pd.DataFrame({
            "month-end price": {asset: signal[f"{asset}_price"] for asset in data_assets},
            "13612U momentum": {asset: signal[f"{asset}_13612u"] for asset in ASSETS},
        }).T
        st.dataframe(inputs.style.format("{:.6f}"), use_container_width=True)
        st.caption("13612U = (1-month return + 3-month return + 6-month return + 12-month return) / 4. The leveraged model uses SPY and TIP—not SSO momentum—to determine its gate.")
    st.subheader("Signal history")
    history_columns = ["regime", "selected_asset", "previous_asset", "trade"]
    history = decisions.loc[:, history_columns].copy()
    history["effective_start"] = [first_trading_day_after(prices, date) for date in history.index]
    history = history.rename_axis("signal_date")
    st.dataframe(history.sort_index(ascending=False), use_container_width=True)
    st.download_button("Download signal history CSV", history.to_csv().encode("utf-8"), f"{strategy.name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_signal_history.csv", "text/csv")
    st.caption("Rules-based informational signal only; not investment advice. You are responsible for any trading decision and execution.")

with validation_tab:
    st.subheader("Rules and calculation")
    if isinstance(strategy, HAASimpleLeveraged2x):
        st.markdown("""**HAA-Simple Leveraged 2x (SSO):** calculate equal-weighted 13612U using unleveraged SPY and TIP. If both are strictly positive, hold 100% SSO. Otherwise select the available defensive asset with the higher 13612U momentum: IEF or BIL. SSO momentum never controls the gate; using SPY avoids de-risking the leveraged sleeve solely because of SSO's amplified drawdown. IEF/BIL remain unleveraged.

**Risk:** high-drawdown satellite, not a core holding. A monthly signal cannot prevent losses from a fast intramonth crash.""")
    else:
        st.markdown("""**HAA-Simple:** at each month-end calculate equal-weighted 13612U momentum for SPY and TIP. If both are strictly positive, select SPY. Otherwise select the available defensive asset with the higher momentum: IEF or BIL. The selection earns the *following* month’s return only. SPY/TIP history is sufficient for a risk-on decision; early defensive months use IEF when BIL has not yet accumulated sufficient history—no BIL proxy is created.""")

    st.markdown("""**13612U:** `(1-month return + 3-month return + 6-month return + 12-month return) / 4`. Each return is `price at signal date / price at its historical month-end - 1`. This implementation therefore requires 12 earlier observations of each asset it actually needs and uses no later prices.

**Data:** Enter an `ASSET=YAHOO_TICKER` mapping in the sidebar to download Yahoo Finance data automatically. The leveraged model additionally requires `SSO=SSO`. Uploaded CSV data replaces an asset’s entire history; use one uploader and name files with their target role, e.g. `SSO.csv`. `Adj Close` is used when Yahoo supplies it; `Close` is the visible fallback. No missing ETF history is fabricated.

**Tax:** applies only when an existing position is sold due to an allocation change. It tracks cost basis and loss carryforward, never taxes the final unrealized position, and is independent of the strategy module.""")
    st.write(f"First valid signal date: **{first_signal.date()}**")
    missing = monthly[monthly.isna().any(axis=1)]
    st.write(f"Months with at least one missing canonical price: **{len(missing)}**")
    st.subheader("Monthly audit table")
    audit_columns = [f"{asset}_price" for asset in data_assets] + [f"{asset}_13612u" for asset in ASSETS] + ["regime", "selected_asset", "previous_asset", "trade", "holding_end", "holding_period_return"]
    audit = result.audit[audit_columns]
    st.dataframe(audit.style.format("{:.6f}", subset=[c for c in audit.columns if c.endswith("13612u") or c.endswith("return")]), use_container_width=True)
    st.download_button("Download audit CSV", audit.to_csv().encode("utf-8"), f"{strategy.name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_monthly_audit.csv", "text/csv")
    st.subheader("Raw and monthly data used")
    st.dataframe(prices, use_container_width=True)
    st.dataframe(monthly, use_container_width=True)
    st.caption("Automated validation: run `pytest` locally; tests cover strategy selection, timing, benchmark dates, tax realization, conditional early risk-on execution, and a hand-calculated 13612U example.")
