import pandas as pd
import pytest

from haa.comparison import ModelInput, compare_models


def prices(index):
    return pd.DataFrame({"SPY": range(100, 100 + len(index)), "IEF": range(100, 100 + len(index))}, index=index, dtype=float)


def decisions(index, asset="SPY"):
    return pd.DataFrame({"selected_asset": [asset] * len(index)}, index=pd.DatetimeIndex(index, name="signal_date"))


def test_comparison_requires_two_models():
    index = pd.date_range("2020-01-31", periods=3, freq="ME")
    one = {"one": ModelInput("one", decisions(index[:2]), prices(index))}
    with pytest.raises(ValueError, match="at least two"):
        compare_models(one, 100, completed_through=index[-1])


def test_comparison_rebases_models_to_identical_common_dates_and_benchmark():
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    shared_prices = prices(index)
    models = {
        "early": ModelInput("early", decisions(index[:5]), shared_prices),
        "late": ModelInput("late", decisions(index[2:5], "IEF"), shared_prices),
    }
    comparison = compare_models(models, 100, completed_through=index[-1])
    assert comparison.common_index.equals(pd.DatetimeIndex([index[3], index[4], index[5]]))
    early = comparison.results["early"].monthly
    late = comparison.results["late"].monthly
    assert early.index.equals(late.index)
    assert early["benchmark_monthly_return"].equals(late["benchmark_monthly_return"])


def test_comparison_excludes_incomplete_current_month():
    index = pd.date_range("2020-01-31", periods=5, freq="ME")
    shared_prices = prices(index)
    models = {
        "one": ModelInput("one", decisions(index[:4]), shared_prices),
        "two": ModelInput("two", decisions(index[:4], "IEF"), shared_prices),
    }
    comparison = compare_models(models, 100, completed_through=index[-2])
    assert comparison.common_index.max() == index[-2]


def test_comparison_tax_states_are_independent():
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    asset_prices = pd.DataFrame({"SPY": [100, 120, 120, 120], "IEF": [100, 100, 100, 100]}, index=index, dtype=float)
    switch = pd.DataFrame({"selected_asset": ["SPY", "IEF", "IEF"]}, index=pd.DatetimeIndex(index[:3], name="signal_date"))
    models = {name: ModelInput(name, switch, asset_prices) for name in ("a", "b")}
    comparison = compare_models(models, 100, tax_enabled=True, completed_through=index[-1])
    assert comparison.results["a"].monthly["after_tax_value"].equals(comparison.results["b"].monthly["after_tax_value"])
    assert comparison.results["a"].monthly["after_tax_value"].iloc[-1] < comparison.results["a"].monthly["pre_tax_value"].iloc[-1]
    assert not comparison.results["a"].tax_events.empty
