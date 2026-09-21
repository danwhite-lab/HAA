import pandas as pd

from haa.engine import run_backtest
from haa.strategies import HAAClassicNoQQQ
from haa.tax import IsraeliTaxState


CLASSIC = ("TIP", "IEF", "BIL", "SPY", "IWM", "PDBC", "TLT", "VEA", "VNQ", "VWO")


def prices_with_terminal_scores(scores):
    index = pd.date_range("2020-01-31", periods=13, freq="ME")
    prices = pd.DataFrame({asset: [100.0] * 13 for asset in CLASSIC}, index=index)
    for asset, score in scores.items():
        prices.loc[index[-1], asset] = 100 * (1 + score)
    return prices


def test_positive_tip_selects_top_four_without_qqq():
    prices = prices_with_terminal_scores({"TIP": .1, "SPY": .7, "IWM": .6, "PDBC": .5, "TLT": .4, "VEA": .3, "VNQ": .2, "VWO": .1})
    decision = HAAClassicNoQQQ().decisions(prices).iloc[-1]
    assert decision["regime"] == "risk-on"
    assert decision["target_weights"] == {"SPY": .25, "IWM": .25, "PDBC": .25, "TLT": .25}
    assert "QQQ" not in decision["target_weights"]


def test_nonpositive_tip_selects_best_defensive_asset():
    prices = prices_with_terminal_scores({"TIP": 0, "IEF": .1, "BIL": .2})
    decision = HAAClassicNoQQQ().decisions(prices).iloc[-1]
    assert decision["regime"] == "risk-off"
    assert decision["target_weights"] == {"BIL": 1.0}


def test_offensive_ties_use_alphabetical_ticker_order():
    prices = prices_with_terminal_scores({"TIP": .1, "SPY": .2, "IWM": .2, "PDBC": .2, "TLT": .2, "VEA": .2, "VNQ": .2, "VWO": .2})
    decision = HAAClassicNoQQQ().decisions(prices).iloc[-1]
    assert tuple(decision["target_weights"]) == ("IWM", "PDBC", "SPY", "TLT")


def test_weighted_engine_uses_next_period_and_identical_spy_dates():
    index = pd.to_datetime(["2021-01-31", "2021-02-28"])
    prices = pd.DataFrame({"SPY": [100, 110], "IEF": [100, 90]}, index=index, dtype=float)
    decisions = pd.DataFrame({"target_weights": [{"SPY": .5, "IEF": .5}]}, index=pd.DatetimeIndex([index[0]], name="signal_date"))
    result = run_backtest(decisions, prices, 100)
    assert result.monthly.index[0] == index[1]
    assert result.monthly["pre_tax_value"].iloc[0] == 100
    assert result.monthly["benchmark_value"].index.equals(result.monthly["pre_tax_value"].index)


def test_partial_sale_taxes_only_the_partial_realized_gain():
    tax = IsraeliTaxState(tax_rate=.25)
    tax.buy("SPY", 100)
    sale = tax.sell(60, asset="SPY", cost_basis_sold=50)
    assert sale["realized_gain"] == 10
    assert sale["tax_paid"] == 2.5
    assert tax.cost_bases["SPY"] == 50


def test_weighted_engine_taxes_only_an_asset_that_is_reduced_or_sold():
    index = pd.to_datetime(["2021-01-31", "2021-02-28", "2021-03-31"])
    prices = pd.DataFrame({"SPY": [100, 120, 120], "IEF": [100, 100, 100]}, index=index, dtype=float)
    decisions = pd.DataFrame(
        {"target_weights": [{"SPY": 1.0}, {"IEF": 1.0}]},
        index=pd.DatetimeIndex(index[:2], name="signal_date"),
    )
    result = run_backtest(decisions, prices, 100, tax_enabled=True)
    assert result.tax_events.iloc[0]["sold_asset"] == "SPY"
    assert result.tax_events.iloc[0]["realized_gain"] == 20
    assert result.tax_events.iloc[0]["tax_paid"] == 5
