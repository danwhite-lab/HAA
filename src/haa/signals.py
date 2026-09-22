"""Safely surface the latest completed-month strategy decision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SignalStatus:
    """The latest decision that is safe to present as an actionable signal."""

    decision: pd.Series | None
    reason: str | None
    completed_through: pd.Timestamp


def last_completed_month_end(as_of: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the calendar month-end before ``as_of``'s current month."""
    timestamp = pd.Timestamp.now(tz="UTC").tz_localize(None) if as_of is None else pd.Timestamp(as_of).tz_localize(None)
    return (timestamp.to_period("M") - 1).to_timestamp("M")


def latest_actionable_signal(
    decisions: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    required_assets: Iterable[str],
    as_of: pd.Timestamp | None = None,
) -> SignalStatus:
    """Return only a fully priced decision from the latest completed month.

    A current partial calendar month is never actionable, even when Yahoo has
    already returned one or more daily observations for it. A stale or missing
    completed month also blocks a signal instead of falling back to an older
    allocation silently.
    """
    completed_month_end = last_completed_month_end(as_of)
    completed_month = completed_month_end.to_period("M")
    assets = tuple(required_assets)
    if decisions.empty:
        return SignalStatus(None, "No strategy decisions exist yet; at least 12 prior monthly observations are required.", completed_month_end)
    if any(asset not in monthly_prices.columns for asset in assets):
        return SignalStatus(None, "Required asset data is unavailable.", completed_month_end)
    completed_dates = monthly_prices.index[monthly_prices.index.to_period("M") == completed_month]
    if len(completed_dates) != 1:
        return SignalStatus(None, f"No complete monthly price row is available for {completed_month_end.date()}.", completed_month_end)
    completed_through = completed_dates[0]
    missing = monthly_prices.loc[completed_through, list(assets)].isna()
    if missing.any():
        return SignalStatus(None, f"No actionable signal: missing completed-month price data for {', '.join(missing[missing].index)}.", completed_through)
    if completed_through not in decisions.index:
        return SignalStatus(None, f"No actionable signal: insufficient valid history to calculate {completed_through.date()}'s decision.", completed_through)
    return SignalStatus(decisions.loc[completed_through].copy(), None, completed_through)


def first_trading_day_after(daily_prices: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Timestamp | None:
    """Find the first available market observation after a signal month-end."""
    future = daily_prices.index[daily_prices.index > pd.Timestamp(signal_date)]
    return future.min() if len(future) else None
