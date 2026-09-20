"""Monthly execution engine: decisions execute after their month-end signal."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .tax import IsraeliTaxState


@dataclass
class BacktestResult:
    monthly: pd.DataFrame
    audit: pd.DataFrame
    tax_events: pd.DataFrame


def run_backtest(
    decisions: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    initial_investment: float,
    transaction_cost: float = 0.0,
    tax_enabled: bool = False,
    tax_rate: float = 0.25,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> BacktestResult:
    """Execute one allocation for the month following each signal.

    Each decision at t uses prices through t. Its selected asset earns only the
    t-to-next-month-end return. The last decision has no future holding period
    and is excluded, preventing a same-date look-ahead trade.
    """
    if decisions.empty:
        raise ValueError("No valid signals: at least 13 complete month-end observations are required.")
    prices = monthly_prices.sort_index()
    rows = []
    for signal_date, decision in decisions.iterrows():
        loc = prices.index.get_indexer([signal_date])[0]
        if loc < 0 or loc + 1 >= len(prices.index):
            continue
        holding_end = prices.index[loc + 1]
        if start is not None and holding_end < pd.Timestamp(start):
            continue
        if end is not None and holding_end > pd.Timestamp(end):
            continue
        asset = decision["selected_asset"]
        asset_return = prices.loc[holding_end, asset] / prices.loc[signal_date, asset] - 1
        spy_return = prices.loc[holding_end, "SPY"] / prices.loc[signal_date, "SPY"] - 1
        rows.append({**decision.to_dict(), "signal_date": signal_date, "holding_end": holding_end, "holding_period_return": asset_return, "spy_return": spy_return})
    execution = pd.DataFrame(rows)
    if execution.empty:
        raise ValueError("No complete holding periods within the selected date range.")

    pre_value = initial_investment
    after_value = initial_investment
    benchmark_value = initial_investment
    tax_state = IsraeliTaxState(tax_rate=tax_rate)
    previous_asset: str | None = None
    tax_events: list[dict] = []
    output = []
    for _, row in execution.iterrows():
        asset = row.selected_asset
        changing = previous_asset is not None and asset != previous_asset
        entering = previous_asset is None or changing
        pre_tax_cost = transaction_cost if entering else 0.0
        prior_pre_value, prior_after_value, prior_benchmark_value = pre_value, after_value, benchmark_value
        pre_value *= 1 - pre_tax_cost
        if tax_enabled and changing:
            event = tax_state.sell(after_value)
            after_value -= event["tax_paid"]
            tax_events.append({"date": row.signal_date, "sold_asset": previous_asset, **event})
        if tax_enabled and entering:
            after_value *= 1 - transaction_cost
            tax_state.buy(asset, after_value)
        elif not tax_enabled:
            after_value *= 1 - pre_tax_cost
        pre_value *= 1 + row.holding_period_return
        after_value *= 1 + row.holding_period_return
        benchmark_value *= 1 + row.spy_return
        output.append({**row.to_dict(), "pre_tax_value": pre_value, "after_tax_value": after_value, "benchmark_value": benchmark_value, "pre_tax_monthly_return": pre_value / prior_pre_value - 1, "after_tax_monthly_return": after_value / prior_after_value - 1, "benchmark_monthly_return": benchmark_value / prior_benchmark_value - 1, "allocation_change": changing})
        previous_asset = asset
    result = pd.DataFrame(output).set_index("holding_end")
    audit = execution.set_index("signal_date")
    return BacktestResult(result, audit, pd.DataFrame(tax_events))
