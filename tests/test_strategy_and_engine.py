import pandas as pd
import pytest

from haa.engine import run_backtest
from haa.strategies import HAASimple


def make_prices():
    idx = pd.date_range("2020-01-31", periods=16, freq="ME")
    return pd.DataFrame({"SPY": range(100, 116), "TIP": range(100, 116), "IEF": range(100, 116), "BIL": range(100, 116)}, index=idx, dtype=float)


def make_decision(spy, tip, ief, bil, selected):
    return pd.DataFrame([{"SPY_13612w": spy, "TIP_13612w": tip, "IEF_13612w": ief, "BIL_13612w": bil, "selected_asset": selected, "previous_asset": None, "trade": True}], index=pd.DatetimeIndex([pd.Timestamp("2021-01-31")], name="signal_date"))


@pytest.mark.parametrize(("spy", "tip", "ief", "bil", "expected"), [
    (.1, .1, -.1, .1, "SPY"),  # both positive: SPY
    (0, .1, .2, .1, "IEF"),  # SPY <= 0: defensive
    (.1, 0, .1, .2, "BIL"),  # TIP <= 0: defensive
    (-.1, .1, .2, .1, "IEF"),  # IEF wins defensive
    (-.1, .1, .1, .2, "BIL"),  # BIL wins defensive
])
def test_selection_rules(spy, tip, ief, bil, expected):
    strategy = HAASimple()
    _, actual = strategy.select_asset(spy, tip, ief, bil)
    assert actual == expected


def test_execution_starts_after_signal_and_benchmark_dates_match():
    prices = make_prices()
    decisions = make_decision(.1, .1, .1, .0, "SPY")
    result = run_backtest(decisions, prices, 100)
    assert result.monthly.index[0] == pd.Timestamp("2021-02-28")
    assert result.audit.loc[pd.Timestamp("2021-01-31"), "holding_end"] == pd.Timestamp("2021-02-28")
    assert result.monthly["benchmark_value"].index.equals(result.monthly["pre_tax_value"].index)


def test_no_allocation_change_when_asset_is_unchanged():
    prices = make_prices()
    dates = pd.DatetimeIndex([pd.Timestamp("2021-01-31"), pd.Timestamp("2021-02-28")], name="signal_date")
    decisions = pd.DataFrame({"selected_asset": ["SPY", "SPY"]}, index=dates)
    result = run_backtest(decisions, prices, 100)
    assert result.monthly["allocation_change"].sum() == 0
