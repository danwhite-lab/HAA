import pandas as pd

from haa.signals import latest_actionable_signal


def signal_inputs():
    index = pd.to_datetime(["2026-07-31", "2026-08-31", "2026-09-30"])
    prices = pd.DataFrame({asset: [100.0, 101.0, 102.0] for asset in ("SPY", "TIP", "IEF", "BIL")}, index=index)
    decisions = pd.DataFrame(
        {"selected_asset": ["SPY", "SPY", "IEF"], "regime": ["risk-on", "risk-on", "risk-off"]}, index=index
    )
    return decisions, prices


def test_signal_uses_latest_completed_month_not_partial_current_month():
    decisions, prices = signal_inputs()
    status = latest_actionable_signal(decisions, prices, ("SPY", "TIP", "IEF", "BIL"), as_of=pd.Timestamp("2026-09-21"))
    assert status.reason is None
    assert status.decision is not None
    assert status.decision.name == pd.Timestamp("2026-08-31")
    assert status.decision["selected_asset"] == "SPY"


def test_missing_completed_month_data_blocks_signal():
    decisions, prices = signal_inputs()
    prices.loc[pd.Timestamp("2026-08-31"), "TIP"] = None
    status = latest_actionable_signal(decisions, prices, ("SPY", "TIP", "IEF", "BIL"), as_of=pd.Timestamp("2026-09-21"))
    assert status.decision is None
    assert "missing completed-month" in status.reason


def test_signal_is_the_strategy_decision_without_reimplemented_rules():
    decisions, prices = signal_inputs()
    status = latest_actionable_signal(decisions, prices, ("SPY", "TIP", "IEF", "BIL"), as_of=pd.Timestamp("2026-09-21"))
    assert status.decision is not None
    pd.testing.assert_series_equal(status.decision, decisions.loc[pd.Timestamp("2026-08-31")])
