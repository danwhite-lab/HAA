"""Momentum calculations used by HAA strategies."""
from __future__ import annotations

import pandas as pd


def momentum_13612w(monthly_prices: pd.Series) -> pd.Series:
    """Return the HAA 13612W momentum series without using future prices.

    For a month-end t, this is:
    (12 * (P[t]/P[t-1]-1) + 4 * (P[t]/P[t-3]-1)
     + 2 * (P[t]/P[t-6]-1) + (P[t]/P[t-12]-1)) / 4.

    A value at t therefore requires t and the prior 12 month-end observations;
    the first 12 rows are intentionally NaN.
    """
    prices = pd.to_numeric(monthly_prices, errors="coerce")
    r1 = prices.div(prices.shift(1)).sub(1)
    r3 = prices.div(prices.shift(3)).sub(1)
    r6 = prices.div(prices.shift(6)).sub(1)
    r12 = prices.div(prices.shift(12)).sub(1)
    return (12 * r1 + 4 * r3 + 2 * r6 + r12) / 4
