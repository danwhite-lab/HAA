import pandas as pd

from haa.strategies import HAA4, HAA4Leveraged2x


SIGNAL_ASSETS = ("TIP", "BIL", "SPY", "VEA", "VNQ", "IEF")
ALL_ASSETS = (*SIGNAL_ASSETS, "SSO", "EFO", "URE", "UST")


def prices_with_terminal_scores(scores, unavailable=()):
    index = pd.date_range("2020-01-31", periods=13, freq="ME")
    prices = pd.DataFrame({asset: [100.0] * 13 for asset in ALL_ASSETS}, index=index)
    for asset, score in scores.items():
        prices.loc[index[-1], asset] = 100 * (1 + score)
    for asset in unavailable:
        prices.loc[index[-1], asset] = float("nan")
    return prices


def test_bad_tip_maps_ief_defensive_winner_to_ust():
    decision = HAA4Leveraged2x().decisions(prices_with_terminal_scores({"TIP": 0, "IEF": .2, "BIL": .1})).iloc[-1]
    assert decision["defensive_winner"] == "IEF"
    assert decision["target_weights"] == {"UST": 1.0}


def test_bad_tip_keeps_bil_unleveraged():
    decision = HAA4Leveraged2x().decisions(prices_with_terminal_scores({"TIP": 0, "IEF": .1, "BIL": .2})).iloc[-1]
    assert decision["target_weights"] == {"BIL": 1.0}


def test_positive_tip_maps_two_positive_selected_sleeves():
    decision = HAA4Leveraged2x().decisions(prices_with_terminal_scores({"TIP": .1, "SPY": .7, "VEA": .6, "VNQ": .2, "IEF": .1})).iloc[-1]
    assert decision["selected_underlying_assets"] == "SPY, VEA"
    assert decision["target_weights"] == {"SSO": .5, "EFO": .5}


def test_nonpositive_selected_sleeve_is_mapped_to_defensive_winner():
    decision = HAA4Leveraged2x().decisions(prices_with_terminal_scores({"TIP": .1, "SPY": .3, "VEA": -.1, "VNQ": -.2, "IEF": -.3, "BIL": .1})).iloc[-1]
    assert decision["replaced_offensive_assets"] == "VEA"
    assert decision["target_weights"] == {"SSO": .5, "BIL": .5}


def test_duplicate_ust_sleeves_consolidate_to_one_holding():
    decision = HAA4Leveraged2x().decisions(prices_with_terminal_scores({"TIP": .1, "IEF": .4, "SPY": -.1, "VEA": -.2, "VNQ": -.3, "BIL": .1})).iloc[-1]
    assert decision["defensive_winner"] == "IEF"
    assert decision["target_weights"] == {"UST": 1.0}


def test_underlying_decision_matches_unleveraged_haa4_before_mapping():
    prices = prices_with_terminal_scores({"TIP": .1, "SPY": .3, "VEA": -.1, "VNQ": -.2, "IEF": -.3, "BIL": .1})
    plain = HAA4().decisions(prices.loc[:, SIGNAL_ASSETS]).iloc[-1]
    leveraged = HAA4Leveraged2x().decisions(prices).iloc[-1]
    assert leveraged["selected_underlying_assets"] == plain["selected_offensive_assets"]
    assert leveraged["replaced_offensive_assets"] == plain["replaced_offensive_assets"]
    assert leveraged["defensive_winner"] == plain["defensive_winner"]


def test_decision_waits_for_all_required_execution_histories():
    prices = prices_with_terminal_scores({"TIP": 0, "IEF": .2, "BIL": .1}, unavailable=("UST",))
    assert HAA4Leveraged2x().decisions(prices).empty

    # Even an execution ETF not selected in this month establishes the model's
    # shared valid history; the app must not stitch in a selective earlier run.
    prices = prices_with_terminal_scores({"TIP": 0, "IEF": .2, "BIL": .1}, unavailable=("EFO",))
    assert HAA4Leveraged2x().decisions(prices).empty
