import numpy as np
import pandas as pd

from haa.strategies import InflationCompassPlus, InflationCompassPlusDriftBands


def prices(days=420, current=2.5, comparison=2.4):
    idx = pd.bdate_range("2023-01-02", periods=days)
    assets = ("SPY", "XLE", "XLK", "XLU", "XLP", "IEF", "XLI", "XLF", "XLB", "XLV")
    frame = pd.DataFrame({asset: 100 * (1.0005 ** np.arange(days)) for asset in assets}, index=idx)
    fred = pd.Series(2.0, index=idx, name="T5YIE")
    fred.iloc[-2] = current
    fred.iloc[-65] = comparison
    return frame.assign(T5YIE=fred)


def test_plus_exposes_three_state_audit_and_rising_allocation():
    result = InflationCompassPlus().decisions(prices())
    assert not result.empty
    latest = result.iloc[-1]
    assert latest["inflation_state"] == "rising"
    assert latest["target_weights"] == {"XLE": 1.0}
    assert {"raw_direction", "persisted_direction", "rebalance_reason"}.issubset(result.columns)


def test_plus_drift_variant_is_distinct():
    assert InflationCompassPlusDriftBands.name.endswith("Drift Bands")
    assert InflationCompassPlusDriftBands.drift_bands is True
