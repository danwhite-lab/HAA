"""Inflation Compass Plus: three-state T5YIE regime allocation."""
# Deployment marker: force Streamlit Cloud to rebuild the Plus strategy tree.
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import INFLATION_COMPASS_DATA_ASSETS, INFLATION_COMPASS_MARKET_ASSETS


class InflationCompassPlus:
    """Inflation Compass with persistent three-state inflation direction."""

    name = "Inflation Compass Plus"
    data_assets = INFLATION_COMPASS_DATA_ASSETS
    market_data_assets = INFLATION_COMPASS_MARKET_ASSETS
    signal_assets = INFLATION_COMPASS_DATA_ASSETS
    benchmark_asset = "SPY"
    is_multi_asset = True
    uses_daily_signals = True
    risk_warning = "Concentrated sector allocation. T5YIE is market-implied and is read with a one-trading-day lag."
    momentum_window = 80
    sma_window = 200
    drift_bands = False

    positive_weights = {"XLE": 0.5, "XLI": 1 / 6, "XLF": 1 / 6, "XLB": 1 / 6}
    negative_weights = {"XLU": 1 / 3, "XLV": 1 / 3, "XLP": 1 / 3}

    def _weights(self, growth_up: bool, inflation_state: str) -> tuple[dict[str, float], str]:
        if growth_up and inflation_state == "rising":
            return {"XLE": 1.0}, "inflationary-expansion-rising"
        if growth_up and inflation_state in {"falling", "neutral"}:
            return {"XLE": 0.5, "XLK": 0.5}, "inflationary-expansion-transition"
        if growth_up:
            return {"XLK": 1.0}, "disinflationary-expansion"
        if inflation_state == "rising":
            return {"XLU": 1.0}, "stagflation-rising"
        if inflation_state in {"falling", "neutral"}:
            return {"XLU": 0.5, "XLP": 0.25, "IEF": 0.25}, "stagflation-transition"
        return {"XLP": 0.5, "IEF": 0.5}, "disinflationary-slowdown"

    def decisions(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.data_assets) - set(daily_prices.columns)
        if missing:
            raise ValueError(f"{self.name} is missing assets: {sorted(missing)}")
        prices = daily_prices.loc[:, self.data_assets].sort_index()
        market = prices.loc[:, list(self.market_data_assets)].dropna(how="any")
        market_returns = market.pct_change(fill_method=None)
        positive_returns = sum(market_returns[a] * w for a, w in self.positive_weights.items())
        negative_returns = sum(market_returns[a] * w for a, w in self.negative_weights.items())
        positive_growth = (1 + positive_returns).cumprod()
        negative_growth = (1 + negative_returns).cumprod()
        indicator = positive_growth / negative_growth
        spy_sma = market["SPY"].rolling(self.sma_window, min_periods=self.sma_window).mean()
        fred = prices["T5YIE"].dropna()
        completed_period = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M") - 1
        market_dates = market.index[market.index.to_period("M") <= completed_period]
        decision_dates = market_dates.to_series().groupby(market_dates.to_period("M")).tail(1)
        rows: list[dict] = []
        previous_weights: dict[str, float] = {}
        previous_direction: str | None = None
        previous_signal_date: pd.Timestamp | None = None
        for date in pd.DatetimeIndex(decision_dates):
            available = fred.loc[fred.index < date]
            prior_period = date.to_period("M") - 3
            prior_month_end = market.index[market.index.to_period("M") == prior_period]
            if len(available) == 0 or len(prior_month_end) == 0:
                continue
            comparison_cutoff = prior_month_end[-1]
            comparison_available = fred.loc[fred.index < comparison_cutoff]
            window = indicator.loc[:date].dropna().tail(self.momentum_window)
            if len(comparison_available) == 0 or len(window) < self.momentum_window or pd.isna(spy_sma.loc[date]):
                continue
            current_date = available.index[-1]
            current = float(available.iloc[-1])
            comparison_date = comparison_available.index[-1]
            comparison = float(comparison_available.iloc[-1])
            level = "above-2" if current > 2.0 else "at-or-below-2"
            raw_direction = "rising" if current > comparison else "falling" if current < comparison else "equal"
            if level == "at-or-below-2":
                inflation_state = "at-or-below-2"
            elif raw_direction == "equal":
                inflation_state = previous_direction or "neutral"
            else:
                inflation_state = raw_direction
            stored_direction = previous_direction
            if level == "above-2" and raw_direction in {"rising", "falling"}:
                stored_direction = raw_direction
            slope = float(np.polyfit(np.arange(self.momentum_window), window.to_numpy(), 1)[0])
            growth_up = bool(market.loc[date, "SPY"] > spy_sma.loc[date])
            weights, regime = self._weights(growth_up, inflation_state)
            rebalance_required = weights != previous_weights
            reason = "target-change" if rebalance_required else "hold"
            if self.drift_bands and previous_signal_date is not None and weights == previous_weights and len(weights) > 1:
                prior_prices = market.loc[previous_signal_date, list(weights)]
                current_prices = market.loc[date, list(weights)]
                values = {asset: weights[asset] * current_prices[asset] / prior_prices[asset] for asset in weights}
                total = sum(values.values())
                drift = {asset: values[asset] / total for asset in weights}
                if any(abs(drift[a] - weights[a]) > 0.10 for a in weights):
                    rebalance_required = True
                    reason = "drift-band"
            rows.append({"signal_date": date, **{f"{asset}_price": prices.loc[date, asset] for asset in self.data_assets}, "SPY_200d_sma": float(spy_sma.loc[date]), "t5yie_lag_date": current_date, "t5yie_lagged": current, "t5yie_three_month_date": comparison_date, "t5yie_three_month": comparison, "t5yie_80d_date": comparison_date, "t5yie_80d": comparison, "inflation_level": level, "raw_direction": raw_direction, "persisted_direction": stored_direction, "inflation_state": inflation_state, "growth_up": growth_up, "positive_basket_growth": float(positive_growth.loc[date]), "negative_basket_growth": float(negative_growth.loc[date]), "inflation_indicator": float(indicator.loc[date]), "indicator_80d_slope": slope, "breakeven_momentum": raw_direction == "rising", "asset_momentum": slope > 0, "inflation_on": inflation_state in {"rising", "falling", "neutral"}, "regime": regime, "selected_asset": ", ".join(weights), "selected_assets": ", ".join(weights), "target_weights": weights, "previous_weights": previous_weights.copy(), "previous_asset": ", ".join(previous_weights) if previous_weights else None, "rebalance_required": rebalance_required, "rebalance_reason": reason, "trade": rebalance_required})
            previous_weights = weights
            previous_direction = stored_direction
            previous_signal_date = date
        return pd.DataFrame(rows).set_index("signal_date") if rows else pd.DataFrame()


class InflationCompassPlusDriftBands(InflationCompassPlus):
    """Inflation Compass Plus with 10-percentage-point drift bands."""

    name = "Inflation Compass Plus Drift Bands"
    drift_bands = True
