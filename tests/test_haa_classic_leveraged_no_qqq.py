import pandas as pd

from haa.engine import run_backtest
from haa.strategies import HAAClassicLeveragedNoQQQ


STRATEGY = HAAClassicLeveragedNoQQQ()


def leveraged_classic_prices(scores: dict[str, float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-31", periods=14, freq="ME")
    prices = pd.DataFrame({asset: [100.0] * len(index) for asset in STRATEGY.data_assets}, index=index)
    # The final two observations provide a valid 12-month 13612U score and a
    # following monthly holding period for engine timing verification.
    for asset, score in scores.items():
        prices.loc[index[-2], asset] = 100 * (1 + score)
    return prices


def test_risk_on_ranks_1x_assets_and_maps_to_2x_holdings():
    prices = leveraged_classic_prices({
        "TIP": .1, "IEF": .8, "SPY": .7, "IWM": .6, "PDBC": .5,
        "TLT": .4, "VEA": .3, "VNQ": .2, "VWO": .1,
        # Deliberately adverse leveraged prices: they must not affect ranking.
        "UST": -.9, "SSO": -.9, "UWM": -.9, "UBT": -.9,
    })
    decision = STRATEGY.decisions(prices).iloc[-2]
    assert decision["regime"] == "risk-on"
    assert decision["selected_underlying_assets"] == "IEF, SPY, IWM, PDBC"
    assert decision["target_weights"] == {"UST": .25, "SSO": .25, "UWM": .25, "PDBC": .25}
    assert "SSO_13612u" not in decision.index


def test_risk_off_maps_ief_winner_to_ust_and_bil_winner_to_bil():
    ief_prices = leveraged_classic_prices({"TIP": 0, "IEF": .2, "BIL": .1})
    bil_prices = leveraged_classic_prices({"TIP": 0, "IEF": .1, "BIL": .2})
    assert STRATEGY.decisions(ief_prices).iloc[-2]["target_weights"] == {"UST": 1.0}
    assert STRATEGY.decisions(bil_prices).iloc[-2]["target_weights"] == {"BIL": 1.0}


def test_pdbc_stays_unleveraged_and_unchanged_weights_do_not_trade():
    prices = leveraged_classic_prices({"TIP": .1, "PDBC": .9, "SPY": .8, "IWM": .7, "IEF": .6})
    decisions = STRATEGY.decisions(prices)
    decision = decisions.iloc[-2]
    assert "PDBC" in decision["target_weights"]

    # Stable monthly 1x trends preserve the same mapped basket across signals.
    index = pd.date_range("2020-01-31", periods=15, freq="ME")
    stable = pd.DataFrame({asset: [100.0] * len(index) for asset in STRATEGY.data_assets}, index=index)
    growth = {"TIP": .01, "IEF": .08, "SPY": .07, "IWM": .06, "PDBC": .05, "TLT": .04, "VEA": .03, "VNQ": .02, "VWO": .01}
    for asset, monthly_growth in growth.items():
        stable[asset] = [100 * (1 + monthly_growth) ** step for step in range(len(index))]
    stable_decisions = STRATEGY.decisions(stable)
    assert stable_decisions.iloc[-1]["target_weights"] == stable_decisions.iloc[-2]["target_weights"]
    assert not bool(stable_decisions.iloc[-1]["trade"])


def test_mapped_holdings_execute_only_after_the_signal():
    prices = leveraged_classic_prices({"TIP": .1, "SPY": .9, "IWM": .8, "PDBC": .7, "TLT": .6})
    decisions = STRATEGY.decisions(prices)
    result = run_backtest(decisions, prices, 100)
    first_signal = decisions.index[0]
    assert result.monthly.index[0] > first_signal
    assert result.audit.loc[first_signal, "target_weights"] == decisions.loc[first_signal, "target_weights"]
    assert result.monthly.index.equals(result.monthly["benchmark_value"].index)
