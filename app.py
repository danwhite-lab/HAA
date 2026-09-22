from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from haa.constants import ASSETS, DEFAULT_TAX_RATE
# Comparison logic stays outside the UI so it can enforce a shared period.
from haa.comparison import ModelInput, compare_models
from haa.data import combine_replacements, common_monthly_period, date_ranges, default_ticker_map, download_yahoo_prices, parse_ticker_map, read_uploaded_csv, to_month_end, upload_asset_from_filename
from haa.engine import run_backtest
from haa.metrics import annual_returns, performance_metrics
from haa.signals import first_trading_day_after, latest_actionable_signal
from haa.strategies import HAAClassicLeveragedNoQQQ, HAAClassicNoQQQ, HAASimple, HAASimpleIsrael, HAASimpleLeveraged2x

MODEL_OPTIONS = {
    "HAA-Simple": HAASimple,
    "HAA-Simple Leveraged 2x (SSO)": HAASimpleLeveraged2x,
    "HAA-Simple Israel": HAASimpleIsrael,
    "HAA Classic (No QQQ)": HAAClassicNoQQQ,
    "HAA Classic Leveraged 2x (No QQQ)": HAAClassicLeveragedNoQQQ,
}
MODEL_RULES = {
    "HAA-Simple": """**HAA-Simple:** At each month-end, calculate equal-weighted 13612U momentum for SPY and TIP. If both are strictly positive, hold 100% SPY. Otherwise, compare IEF and BIL 13612U momentum and hold 100% of the higher-momentum asset. The decision earns the following month's return only.""",
    "HAA-Simple Leveraged 2x (SSO)": """**HAA-Simple Leveraged 2x (SSO):** Calculate equal-weighted 13612U using unleveraged SPY and TIP. If both are strictly positive, hold 100% SSO. Otherwise, compare IEF and BIL momentum and hold the higher-momentum defensive asset. SSO momentum never controls the gate; IEF and BIL remain unleveraged.

**Risk:** high-drawdown satellite, not a core holding. A monthly signal cannot prevent losses from a fast intramonth crash.""",
    "HAA-Simple Israel": """**HAA-Simple Israel:** TIP is a U.S. signal-only canary. Use equal-weighted 13612U momentum for TIP, TASE-listed CSPX (1159250), iShares $ Treasury Bond 7–10yr UCITS (1159268), and Ayalon Kaspit (5136866). If TIP and CSPX are strictly positive, hold 100% CSPX. Otherwise compare 1159268 and Ayalon Kaspit and hold the higher-momentum asset; an exact tie selects 1159268. All instruments require real 12-month history. This ILS local-investability variant includes USD/ILS exposure; MAKAM 800 is not a holding series.""",
    "HAA Classic (No QQQ)": """**HAA Classic (No QQQ):** TIP is the sole canary. If TIP's equal-weighted 13612U momentum is strictly positive, rank IEF, SPY, IWM, PDBC, TLT, VEA, VNQ, and VWO by 13612U and hold the top four at 25% each. If TIP is zero or negative, compare IEF and BIL momentum and hold the higher-momentum asset. QQQ is intentionally excluded; no leverage is used.""",
    "HAA Classic Leveraged 2x (No QQQ)": """**HAA Classic Leveraged 2x (No QQQ):** TIP is the sole canary and all momentum scores use unleveraged ETFs. If TIP's equal-weighted 13612U momentum is strictly positive, rank IEF, SPY, IWM, PDBC, TLT, VEA, VNQ, and VWO, select the top four, and allocate 25% to each mapped holding: IEF→UST, SPY→SSO, IWM→UWM, PDBC→PDBC, TLT→UBT, VEA→EFO, VNQ→URE, and VWO→EET. If TIP is zero or negative, compare 1× IEF and BIL momentum; hold UST if IEF wins or BIL otherwise. QQQ is excluded.

**Risk:** high-drawdown leveraged satellite, not a core holding. A monthly signal cannot prevent losses from a fast intramonth crash.""",
}
ALL_MODEL_ASSETS = tuple(dict.fromkeys(asset for model_class in MODEL_OPTIONS.values() for asset in getattr(model_class, "data_assets", ASSETS)))

st.set_page_config(page_title="HAA Backtest", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container { padding-top: 0.8rem !important; }
.st-key-header-row { padding-top: 2.35rem !important; }
.st-key-header-row [data-testid="stHorizontalBlock"] {
  flex-wrap: nowrap !important;
  align-items: flex-start !important;
}
.st-key-header-row [data-testid="column"]:last-child {
  flex: 0 0 3rem !important;
  width: 3rem !important;
  min-width: 3rem !important;
}
@media (max-width: 640px) {
  .block-container { padding: 0.6rem 0.75rem 1.5rem !important; }
  [data-testid="stDataFrame"] { max-width: 100%; overflow-x: auto; }
}
</style>
""", unsafe_allow_html=True)

DEFAULT_TICKERS = "\n".join(f"{role}={ticker}" for role, ticker in default_ticker_map(ALL_MODEL_ASSETS).items())
DEFAULT_MODEL = "HAA-Simple"


def append_missing_default_tickers(text: str) -> str:
    """Retain custom sources while migrating saved sessions to new model assets."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    roles = {line.split("=", 1)[0].strip().upper() for line in lines if "=" in line}
    defaults = default_ticker_map(ALL_MODEL_ASSETS)
    lines.extend(f"{asset}={defaults[asset]}" for asset in ALL_MODEL_ASSETS if asset not in roles)
    return "\n".join(lines)


for key, value in {
    "model_name": DEFAULT_MODEL,
    "signals_model_name": DEFAULT_MODEL,
    "page": "Signals",
    "ticker_text": DEFAULT_TICKERS,
    "initial": 100_000.0,
    "cost_pct": 0.0,
    "tax_enabled": False,
    "tax_rate": DEFAULT_TAX_RATE,
    "uploaded_replacements": {},
}.items():
    st.session_state.setdefault(key, value)
if st.session_state["page"] == "Validation":
    st.session_state["page"] = "Rules"
st.session_state["ticker_text"] = append_missing_default_tickers(st.session_state["ticker_text"])

# Keep the primary signal uncluttered. The compact menu holds navigation and,
# on Signals, the model chooser; the sidebar remains Backtest-only.
with st.container(key="header-row"):
    title_column, menu_column = st.columns([12, 1])
    with menu_column:
        with st.popover("⋮", help="Navigation and signal model"):
            page = st.radio("View", ("Signals", "Backtest", "Compare Models", "Rules"), key="page")
            if page == "Signals":
                st.selectbox("Signal model", tuple(MODEL_OPTIONS), key="signals_model_name", help="This selector controls the Signals page only; it does not change the Backtest configuration.")

model_name = st.session_state["model_name"]
ticker_text = st.session_state["ticker_text"]
initial = st.session_state["initial"]
cost_pct = st.session_state["cost_pct"]
tax_enabled = st.session_state["tax_enabled"]
tax_rate = st.session_state["tax_rate"]
uploads = []

if page == "Backtest":
    with st.sidebar:
        st.header("Backtest configuration")
        model_name = st.selectbox("Backtest model", tuple(MODEL_OPTIONS), key="model_name")
        ticker_text = st.text_area("Yahoo Finance ticker sources", value=ticker_text, help="One asset role per line. All available model assets are downloaded so selected models can be compared on the same source data.")
        uploads = st.file_uploader("Upload replacement CSV files", type="csv", accept_multiple_files=True, help=f"Upload one or more files named with one valid asset: {', '.join(ALL_MODEL_ASSETS)}.")
        initial = st.number_input("Initial investment", min_value=1.0, value=initial, step=1_000.0)
        cost_pct = st.number_input("Transaction cost per entry/change (%)", min_value=0.0, max_value=10.0, value=cost_pct * 100, step=0.01) / 100
        tax_enabled = st.toggle("Israeli capital-gains tax", value=tax_enabled)
        tax_rate = st.number_input("Tax rate (%)", min_value=0.0, max_value=100.0, value=tax_rate * 100, step=0.1, disabled=not tax_enabled) / 100
    # The model selectbox already owns ``model_name`` in session state. Writing
    # it again after instantiation raises StreamlitWidgetAlreadyInstantiatedError.
    st.session_state.update({"ticker_text": ticker_text, "initial": initial, "cost_pct": cost_pct, "tax_enabled": tax_enabled, "tax_rate": tax_rate})

strategy = MODEL_OPTIONS[model_name]()
data_assets = getattr(strategy, "data_assets", ASSETS)
benchmark_asset = getattr(strategy, "benchmark_asset", "SPY")
benchmark_label = f"{benchmark_asset} buy-and-hold"

try:
    ticker_map = parse_ticker_map(ticker_text, ALL_MODEL_ASSETS)
except (ValueError, TypeError) as exc:
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

replacements = st.session_state["uploaded_replacements"].copy()
if page == "Backtest":
    replacements = {}
    for upload in uploads or []:
        try:
            asset = upload_asset_from_filename(upload.name, ALL_MODEL_ASSETS)
            if asset in replacements:
                raise ValueError(f"More than one upload targets {asset}; upload only one replacement file per canonical role.")
            replacements[asset] = read_uploaded_csv(upload.getvalue(), asset)
        except ValueError as exc:
            st.sidebar.error(str(exc))
    st.session_state["uploaded_replacements"] = replacements
all_prices = combine_replacements(downloaded, replacements, ALL_MODEL_ASSETS)
prices = all_prices.loc[:, data_assets]
monthly = to_month_end(prices)
all_monthly = to_month_end(all_prices)
ranges = date_ranges(prices)
common_start, common_end = common_monthly_period(monthly)

if common_start is None:
    st.error("The selected model assets have no common month-end observations. Check the Yahoo ticker mappings or upload compatible CSV histories.")
    st.stop()
start = st.session_state.get("start", common_start.date())
execution_end_limit = prices.index.max().date()
end = st.session_state.get("end", execution_end_limit)
# A model can have a shorter history than the previously configured model.
# Keep saved backtest dates valid when returning to its configuration page.
start = min(max(start, common_start.date()), common_end.date())
end = min(max(end, common_start.date()), execution_end_limit)
if start > end:
    start = common_start.date()
if page == "Backtest":
    with st.sidebar:
        st.caption(f"Common monthly data: {common_start.date()} to {common_end.date()}")
        with st.expander("Data & validation"):
            st.caption("Available adjusted-price history for the assets required by the selected model.")
            st.dataframe(ranges, use_container_width=True, hide_index=True)
            st.caption(f"Actual common monthly data period: {common_start.date()} through {common_end.date()}.")
        start = st.date_input("Backtest start (holding-period end)", value=start, min_value=common_start.date(), max_value=common_end.date())
        end = st.date_input("Backtest end (holding-period execution date)", value=end, min_value=common_start.date(), max_value=execution_end_limit)
    st.session_state.update({"start": start, "end": end})

decisions = strategy.decisions(monthly)
if decisions.empty:
    st.error(f"Insufficient history for {strategy.name}.")
    st.stop()
first_signal = decisions.index.min()
try:
    result = run_backtest(decisions, monthly, initial, cost_pct, tax_enabled, tax_rate, pd.Timestamp(start), pd.Timestamp(end), daily_prices=prices, benchmark_asset=benchmark_asset)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# Shared result-table formatting used by both Backtest and Compare Models.
percentage_rows = ["CAGR", "Total return", "Maximum drawdown", "Annualized volatility", "Best month", "Worst month", "Annual turnover"]
ratio_rows = ["Sharpe", "Sortino", "Calmar"]
numeric_rows = ["Final value", "Allocation changes", "Average changes/year"]

if page == "Backtest":
    title_column.title(strategy.name)
    st.caption("The sidebar configures this backtest only.")
    st.caption("Signals are evaluated at month-end and execute for the following holding period; no optimization or synthetic history.")
    if hasattr(strategy, "risk_warning"):
        st.warning(strategy.risk_warning)
    st.caption(f"Holding periods: {result.monthly.index.min().date()} through {result.monthly.index.max().date()}. SPY benchmark uses these same monthly periods.")
    pre_tax_label = f"{strategy.name} pre-tax"
    after_tax_label = f"{strategy.name} after-tax"
    comparison = {pre_tax_label: performance_metrics(result.monthly["pre_tax_value"], initial)}
    if tax_enabled:
        comparison[after_tax_label] = performance_metrics(result.monthly["after_tax_value"], initial)
    # Keep the benchmark at the far right; the after-tax strategy result sits
    # beside its pre-tax counterpart for direct capital-gains comparison.
    comparison[benchmark_label] = performance_metrics(result.monthly["benchmark_value"], initial)
    summary = pd.DataFrame(comparison)
    changes = int(result.monthly["allocation_change"].sum())
    years = len(result.monthly) / 12
    annual_turnover = result.monthly["turnover"].sum() / years if "turnover" in result.monthly and years else changes / years if years else 0
    summary.loc["Allocation changes", pre_tax_label] = changes
    summary.loc["Average changes/year", pre_tax_label] = changes / years if years else 0
    summary.loc["Annual turnover", pre_tax_label] = annual_turnover
    st.subheader("Results")
    styled_summary = summary.style.format("{:.2%}", subset=pd.IndexSlice[percentage_rows, :]).format("{:.2f}", subset=pd.IndexSlice[ratio_rows + numeric_rows, :])
    st.dataframe(styled_summary, use_container_width=True)
    curves = pd.DataFrame({pre_tax_label: result.monthly["pre_tax_value"]})
    if tax_enabled:
        curves[after_tax_label] = result.monthly["after_tax_value"]
    curves[benchmark_label] = result.monthly["benchmark_value"]
    st.plotly_chart(px.line(curves, title="Equity curve"), use_container_width=True)
    drawdowns = curves.div(curves.cummax()).sub(1)
    st.plotly_chart(px.line(drawdowns, title="Drawdown"), use_container_width=True)
    st.subheader("Annual returns")
    annual = pd.DataFrame({pre_tax_label: annual_returns(result.monthly["pre_tax_monthly_return"])})
    if tax_enabled:
        annual[after_tax_label] = annual_returns(result.monthly["after_tax_monthly_return"])
    annual[benchmark_label] = annual_returns(result.monthly["benchmark_monthly_return"])
    st.dataframe(annual.style.format("{:.2%}"), use_container_width=True)
    st.subheader("Monthly returns")
    monthly_returns = pd.DataFrame({pre_tax_label: result.monthly["pre_tax_monthly_return"]})
    if tax_enabled:
        monthly_returns[after_tax_label] = result.monthly["after_tax_monthly_return"]
    monthly_returns[benchmark_label] = result.monthly["benchmark_monthly_return"]
    st.dataframe(monthly_returns.style.format("{:.2%}"), use_container_width=True)

if page == "Compare Models":
    title_column.title("Compare Models")
    st.caption("Each selected model is independently backtested, then restarted over the exact shared completed holding periods. This is informational only and does not recommend one model.")
    default_comparison = [model_name, next(name for name in MODEL_OPTIONS if name != model_name)]
    selected_models = st.multiselect("Models", tuple(MODEL_OPTIONS), default=default_comparison, key="compare_models")
    if len(selected_models) < 2:
        st.info("Select at least two models to compare.")
    else:
        try:
            comparison_inputs = {}
            for selected_name in selected_models:
                selected_strategy = MODEL_OPTIONS[selected_name]()
                selected_assets = getattr(selected_strategy, "data_assets", ASSETS)
                selected_daily = all_prices.loc[:, selected_assets]
                selected_monthly = to_month_end(selected_daily)
                comparison_inputs[selected_name] = ModelInput(selected_name, selected_strategy.decisions(selected_monthly), selected_monthly, selected_daily, getattr(selected_strategy, "benchmark_asset", "SPY"))
            if len({model.benchmark_asset for model in comparison_inputs.values()}) != 1:
                raise ValueError("Compare only models with the same buy-and-hold benchmark. HAA-Simple Israel uses CSPX_IL; the other current models use SPY.")
            model_comparison = compare_models(
                comparison_inputs,
                initial,
                cost_pct,
                tax_enabled,
                tax_rate,
                pd.Timestamp(start),
                pd.Timestamp(end),
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.subheader("Comparable period")
            st.dataframe(model_comparison.available_periods, use_container_width=True)
            st.success(f"All results below use **{model_comparison.common_index.min().date()} through {model_comparison.common_index.max().date()}** ({len(model_comparison.common_index)} complete monthly holding periods).")
            comparison_pre = {name: performance_metrics(backtest.monthly["pre_tax_value"], initial) for name, backtest in model_comparison.results.items()}
            first_comparison = next(iter(model_comparison.results.values()))
            comparison_benchmark_label = f"{next(iter(comparison_inputs.values())).benchmark_asset} buy-and-hold"
            comparison_pre[comparison_benchmark_label] = performance_metrics(first_comparison.monthly["benchmark_value"], initial)
            comparison_summary = pd.DataFrame(comparison_pre)
            comparison_years = len(model_comparison.common_index) / 12
            for name, backtest in model_comparison.results.items():
                changes = int(backtest.monthly["allocation_change"].sum())
                turnover = backtest.monthly["turnover"].sum() / comparison_years if "turnover" in backtest.monthly and comparison_years else changes / comparison_years if comparison_years else 0
                comparison_summary.loc["Allocation changes", name] = changes
                comparison_summary.loc["Average changes/year", name] = changes / comparison_years if comparison_years else 0
                comparison_summary.loc["Annual turnover", name] = turnover
            st.subheader("Pre-tax results")
            st.dataframe(comparison_summary.style.format("{:.2%}", subset=pd.IndexSlice[percentage_rows, :]).format("{:.2f}", subset=pd.IndexSlice[ratio_rows + numeric_rows, :]), use_container_width=True)

            pre_curves = pd.DataFrame({name: backtest.monthly["pre_tax_value"] for name, backtest in model_comparison.results.items()})
            pre_curves[comparison_benchmark_label] = first_comparison.monthly["benchmark_value"]
            st.plotly_chart(px.line(pre_curves, title="Pre-tax equity curves"), use_container_width=True)
            st.plotly_chart(px.line(pre_curves.div(pre_curves.cummax()).sub(1), title="Monthly drawdown"), use_container_width=True)
            annual_comparison = pd.DataFrame({name: annual_returns(backtest.monthly["pre_tax_monthly_return"]) for name, backtest in model_comparison.results.items()})
            annual_comparison[comparison_benchmark_label] = annual_returns(first_comparison.monthly["benchmark_monthly_return"])
            st.subheader("Annual returns")
            st.dataframe(annual_comparison.style.format("{:.2%}"), use_container_width=True)
            monthly_comparison = pd.DataFrame({name: backtest.monthly["pre_tax_monthly_return"] for name, backtest in model_comparison.results.items()})
            monthly_comparison[comparison_benchmark_label] = first_comparison.monthly["benchmark_monthly_return"]
            st.subheader("Monthly returns")
            st.dataframe(monthly_comparison.style.format("{:.2%}"), use_container_width=True)
            st.download_button("Download common-period monthly returns CSV", monthly_comparison.to_csv().encode("utf-8"), "haa_model_comparison_monthly_returns.csv", "text/csv", key="comparison_monthly_download")

            if tax_enabled:
                after_summary = pd.DataFrame({name: performance_metrics(backtest.monthly["after_tax_value"], initial) for name, backtest in model_comparison.results.items()})
                st.subheader("After-tax results")
                st.dataframe(after_summary.style.format("{:.2%}", subset=pd.IndexSlice[["CAGR", "Total return", "Maximum drawdown", "Annualized volatility", "Best month", "Worst month"], :]).format("{:.2f}", subset=pd.IndexSlice[ratio_rows + ["Final value"], :]), use_container_width=True)
                after_curves = pd.DataFrame({name: backtest.monthly["after_tax_value"] for name, backtest in model_comparison.results.items()})
                st.plotly_chart(px.line(after_curves, title="After-tax equity curves"), use_container_width=True)

            st.subheader("Latest allocation in the shared period")
            allocation_rows = []
            for name, backtest in model_comparison.results.items():
                latest = backtest.monthly.iloc[-1]
                weights = latest.get("target_weights")
                allocation = ", ".join(f"{asset} {weight:.0%}" for asset, weight in weights.items()) if isinstance(weights, dict) else latest["selected_asset"]
                allocation_rows.append({"model": name, "signal_date": latest["signal_date"], "regime": latest["regime"], "target allocation": allocation})
            st.dataframe(pd.DataFrame(allocation_rows).set_index("model"), use_container_width=True)
            st.subheader("Model audit downloads")
            for name, backtest in model_comparison.results.items():
                st.download_button(f"Download {name} common-period audit CSV", backtest.audit.to_csv().encode("utf-8"), f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_comparison_audit.csv", "text/csv", key=f"comparison_audit_{name}")

if page == "Signals":
    signal_model_name = st.session_state["signals_model_name"]
    signal_strategy = MODEL_OPTIONS[signal_model_name]()
    signal_data_assets = getattr(signal_strategy, "data_assets", ASSETS)
    signal_momentum_assets = getattr(signal_strategy, "signal_assets", signal_data_assets)
    signal_prices = all_prices.loc[:, signal_data_assets]
    signal_monthly = to_month_end(signal_prices)
    signal_decisions = signal_strategy.decisions(signal_monthly)
    signal_status = latest_actionable_signal(signal_decisions, signal_monthly, signal_data_assets)
    if signal_status.decision is None:
        title_column.title("Signal")
        title_column.caption(f"Model: {signal_model_name} · Completed month-end signal")
        st.error(signal_status.reason)
    else:
        signal = signal_status.decision
        signal_date = pd.Timestamp(signal.name)
        effective_start = first_trading_day_after(signal_prices, signal_date, (signal["selected_asset"], getattr(signal_strategy, "benchmark_asset", "SPY")))
        previous = signal["previous_asset"] if pd.notna(signal["previous_asset"]) else "No prior allocation"
        weights = signal.get("target_weights", {signal["selected_asset"]: 1.0})
        if getattr(signal_strategy, "is_multi_asset", False):
            action = "Hold allocation" if not bool(signal["trade"]) else ("Establish allocation" if previous == "No prior allocation" else "Rebalance allocation")
            target_allocation = ", ".join(f"{asset} {weight:.0%}" for asset, weight in weights.items())
        else:
            action = "Hold" if not bool(signal["trade"]) else (f"Buy {signal['selected_asset']}" if previous == "No prior allocation" else f"Switch {previous} → {signal['selected_asset']}")
            target_allocation = f"100% {signal['selected_asset']}"
        title_column.title(f"Signal - {target_allocation}")
        title_column.caption(f"Model: {signal_model_name} · Completed month-end signal")
        signal_summary = pd.DataFrame([{
            "signal date": signal_date.date().isoformat(),
            "regime": str(signal["regime"]).replace("-", " ").title(),
            "instruction": action,
            "target allocation": target_allocation,
        }])
        st.dataframe(signal_summary.style.set_properties(**{"background-color": "#fff3cd"}), use_container_width=True, hide_index=True)
        if effective_start is not None:
            st.info(f"Effective holding period: **{effective_start.date()}** until the next month-end decision.")
        else:
            st.warning("No later trading observation is available yet, so an effective start date cannot be shown.")
        st.subheader("Why this allocation")
        if isinstance(signal_strategy, (HAAClassicNoQQQ, HAAClassicLeveragedNoQQQ)) and signal["regime"] == "risk-on":
            if isinstance(signal_strategy, HAAClassicLeveragedNoQQQ):
                st.write(f"TIP 13612U momentum is strictly positive, so the model selects the four highest-momentum 1x underlyings ({signal['selected_underlying_assets']}) and holds their mapped 2x ETFs ({signal['mapped_holding_assets']}).")
            else:
                st.write(f"TIP 13612U momentum is strictly positive, so the model selects the four highest-momentum offensive assets: {signal['selected_assets']}.")
        elif isinstance(signal_strategy, (HAAClassicNoQQQ, HAAClassicLeveragedNoQQQ)):
            if isinstance(signal_strategy, HAAClassicLeveragedNoQQQ):
                st.write(f"TIP 13612U momentum is not positive, so the model compares 1x IEF and BIL momentum and holds the mapped defensive asset: {signal['selected_asset']}.")
            else:
                st.write(f"TIP 13612U momentum is not positive, so the model selects the higher-momentum defensive asset: {signal['selected_asset']}.")
        elif isinstance(signal_strategy, HAASimpleIsrael) and signal["regime"] == "risk-on":
            st.write("TIP and CSPX_IL 13612U momentum are both strictly positive, so the model selects CSPX_IL.")
        elif isinstance(signal_strategy, HAASimpleIsrael):
            st.write(f"At least one of TIP or CSPX_IL 13612U momentum is not positive, so the model selects the higher-momentum Israeli defensive asset: {signal['selected_asset']}.")
        elif signal["regime"] == "risk-on":
            holding = "SSO" if isinstance(signal_strategy, HAASimpleLeveraged2x) else "SPY"
            st.write(f"SPY and TIP 13612U momentum are both strictly positive, so the model selects {holding}.")
        else:
            st.write(f"At least one of SPY or TIP 13612U momentum is not positive, so the model selects the higher-momentum defensive asset: {signal['selected_asset']}.")
        price_columns = [f"{asset}_price" for asset in signal_momentum_assets if f"{asset}_price" in signal.index]
        momentum_columns = [f"{asset}_13612u" for asset in signal_momentum_assets if f"{asset}_13612u" in signal.index]
        inputs = pd.DataFrame({
            "month-end price": {column.removesuffix("_price"): signal[column] for column in price_columns},
            "13612U momentum": {column.removesuffix("_13612u"): signal[column] for column in momentum_columns},
        }).T
        st.dataframe(inputs.style.format("{:.6f}"), use_container_width=True)
        st.caption("13612U = (1-month return + 3-month return + 6-month return + 12-month return) / 4. The leveraged model uses SPY and TIP—not SSO momentum—to determine its gate.")
    history_columns = ["regime", "selected_asset", "previous_asset", "trade"]
    if "target_weights" in signal_decisions:
        history_columns.insert(2, "target_weights")
    if signal_decisions.empty:
        history = pd.DataFrame(columns=[*history_columns, "effective_start"])
        history.index.name = "signal_date"
    else:
        history = signal_decisions.loc[:, history_columns].copy()
        if "target_weights" in history:
            history["target_weights"] = history["target_weights"].map(lambda weights: ", ".join(f"{asset} {weight:.0%}" for asset, weight in weights.items()))
        history["effective_start"] = [first_trading_day_after(signal_prices, date, (history.loc[date, "selected_asset"], getattr(signal_strategy, "benchmark_asset", "SPY"))) for date in history.index]
        history.index = pd.to_datetime(history.index).strftime("%Y-%m-%d")
        history = history.rename_axis("signal_date")
    with st.expander("Signal history"):
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
        st.download_button("Download signal history CSV", history.to_csv().encode("utf-8"), f"{signal_strategy.name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_signal_history.csv", "text/csv")
    with st.expander("Data status"):
        st.caption("This signal uses completed month-end data only. Backtest settings do not affect it.")
        raw_ranges = date_ranges(signal_prices)
        st.dataframe(raw_ranges[["last_available"]], use_container_width=True)
        st.caption(f"Latest eligible completed month: {signal_status.completed_through.date()}. A partial current month is never presented as a final signal.")
    st.caption("Rules-based informational signal only; not investment advice. You are responsible for any trading decision and execution.")

if page == "Rules":
    title_column.title("Rules")
    st.subheader("Model rules")
    for rule_name, rule in MODEL_RULES.items():
        with st.expander(rule_name, expanded=rule_name == model_name):
            st.markdown(rule)

    st.markdown("""**13612U:** `(1-month return + 3-month return + 6-month return + 12-month return) / 4`. Each return is `price at signal date / price at its historical month-end - 1`. This implementation therefore requires 12 earlier observations of each asset it actually needs and uses no later prices.

**Data:** Enter an `ASSET=YAHOO_TICKER` mapping in the sidebar to download Yahoo Finance data automatically. Leveraged models additionally require their mapped holding tickers (such as SSO, UST, UWM, UBT, EFO, URE, and EET). Uploaded CSV data replaces an asset’s entire history; use one uploader and name files with their target role, e.g. `SSO.csv`. `Adj Close` is used when Yahoo supplies it; `Close` is the visible fallback. No missing ETF history is fabricated.

**Tax:** applies only when an existing position is sold due to an allocation change. It tracks cost basis and loss carryforward, never taxes the final unrealized position, and is independent of the strategy module.""")
    st.write(f"First valid signal date: **{first_signal.date()}**")
    missing = monthly[monthly.isna().any(axis=1)]
    st.write(f"Months with at least one missing canonical price: **{len(missing)}**")
    audit_momentum_assets = getattr(strategy, "signal_assets", data_assets)
    audit_columns = [f"{asset}_price" for asset in data_assets] + [f"{asset}_13612u" for asset in audit_momentum_assets]
    if isinstance(strategy, (HAAClassicNoQQQ, HAAClassicLeveragedNoQQQ)):
        audit_columns += [f"{asset}_rank" for asset in strategy.offensive_assets] + ["selected_assets", "target_weights", "previous_weights"]
        if isinstance(strategy, HAAClassicLeveragedNoQQQ):
            audit_columns += ["selected_underlying_assets", "mapped_holding_assets"]
    audit_columns += ["regime", "selected_asset", "previous_asset", "trade", "execution_date", "holding_end", "holding_period_return"]
    audit = result.audit[audit_columns]
    with st.expander("Monthly audit table"):
        st.dataframe(audit.style.format("{:.6f}", subset=[c for c in audit.columns if c.endswith("13612u") or c.endswith("return")]), use_container_width=True)
        st.download_button("Download audit CSV", audit.to_csv().encode("utf-8"), f"{strategy.name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_monthly_audit.csv", "text/csv")
    st.subheader("Raw and monthly data used")
    st.dataframe(prices, use_container_width=True)
    st.dataframe(monthly, use_container_width=True)
    st.caption("Automated validation: run `pytest` locally; tests cover strategy selection, timing, benchmark dates, tax realization, conditional early risk-on execution, and a hand-calculated 13612U example.")
