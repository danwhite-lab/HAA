import numpy as np
import pandas as pd

from haa.data import parse_ticker_map, upload_asset_from_filename
from haa.momentum import momentum_13612u


def test_13612u_matches_manual_example():
    # Month 12 price is 120 from historical prices: 100, then 110/105/102 at 1/3/6 months ago.
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 102, 108, 110, 112, 110, 120], index=pd.date_range("2020-01-31", periods=13, freq="ME"))
    expected = ((120 / 110 - 1) + (120 / 110 - 1) + (120 / 106 - 1) + (120 / 100 - 1)) / 4
    assert momentum_13612u(prices).iloc[-1] == pytest_approx(expected)


def test_momentum_never_uses_future_prices():
    prices = pd.Series(np.arange(100, 116), index=pd.date_range("2020-01-31", periods=16, freq="ME"), dtype=float)
    baseline = momentum_13612u(prices).iloc[12]
    prices.iloc[13:] = 1_000_000
    assert momentum_13612u(prices).iloc[12] == baseline


def pytest_approx(value):
    import pytest
    return pytest.approx(value)


def test_yahoo_ticker_mapping_and_single_upload_file_names_are_explicit():
    assert parse_ticker_map("SPY=VOO\nTIP=TIP\nIEF=IEF\nBIL=BIL")["SPY"] == "VOO"
    assert upload_asset_from_filename("TIP_validation.csv") == "TIP"
