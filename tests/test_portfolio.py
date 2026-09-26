import pytest

from haa.portfolio import aggregate_holdings, total_weight


def test_aggregate_holdings_combines_duplicate_and_multi_asset_sleeves():
    sleeves = [
        {"name": "HAA", "weight": 60, "target_weights": {"SPY": 1.0}},
        {"name": "Compass", "weight": 40, "target_weights": {"SPY": .5, "IEF": .5}},
    ]
    result = aggregate_holdings(sleeves)
    assert result["SPY"] == {"weight": .8, "sleeves": ["HAA", "Compass"]}
    assert result["IEF"] == {"weight": .2, "sleeves": ["Compass"]}
    assert total_weight(sleeves) == 100


def test_total_weight_allows_drafts_to_be_flagged_by_the_ui():
    assert total_weight([{"weight": 45}, {"weight": 50}]) == 95
