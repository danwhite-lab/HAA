import pandas as pd

from haa.engine import run_backtest
from haa.strategies import HAASimpleLeveraged2x


def leveraged_prices() -> pd.DataFrame:
    index = pd.date_range("2020-01-31", periods=16, freq="ME")
    return pd.DataFrame({
        "SPY": range(100, 116),
        "TIP": range(100, 116),
        "IEF": range(100, 116),
        "BIL": range(100, 116),
        # SSO falls over the final year: its momentum must not affect the gate.
        "SSO": list(range(100, 104)) + list(range(103, 91, -1)),
    }, index=index, dtype=float)


def test_risk_on_uses_spy_signal_but_holds_sso():
    decisions = HAASimpleLeveraged2x().decisions(leveraged_prices())
    assert decisions.iloc[0]["SPY_13612u"] > 0
    assert decisions.iloc[0]["TIP_13612u"] > 0
    assert decisions.iloc[0]["selected_asset"] == "SSO"
    assert "SSO_13612u" not in decisions.columns


def test_negative_spy_or_tip_selects_unleveraged_defense():
    prices = leveraged_prices()
    prices.loc[prices.index[-1], "SPY"] = 50
    decisions = HAASimpleLeveraged2x().decisions(prices)
    assert decisions.iloc[-1]["regime"] == "risk-off"
    assert decisions.iloc[-1]["selected_asset"] in {"IEF", "BIL"}


def test_sso_execution_begins_after_its_signal():
    prices = leveraged_prices()
    decisions = HAASimpleLeveraged2x().decisions(prices)
    result = run_backtest(decisions, prices, 100)
    first_signal = decisions.index[0]
    assert result.audit.loc[first_signal, "selected_asset"] == "SSO"
    assert result.monthly.index[0] > first_signal
