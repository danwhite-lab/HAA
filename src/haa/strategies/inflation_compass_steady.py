"""Inflation Compass Steady: an 80-trading-day market-regime model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import INFLATION_COMPASS_DATA_ASSETS, INFLATION_COMPASS_MARKET_ASSETS


class InflationCompassSteady:
    """Monthly implementation of the post-2003 T5YIE Inflation Compass rules."""

    name = "Inflation Compass Steady (80-day)"
    data_assets = INFLATION_COMPASS_DATA_ASSETS
    market_data_assets = INFLATION_COMPASS_MARKET_ASSETS
    signal_assets = INFLATION_COMPASS_DATA_ASSETS
    benchmark_asset = "SPY"
    is_multi_asset = True
    uses_daily_signals = True
    risk_warning = "Concentrated sector allocation. T5YIE is a market-implied inflation measure and is read with a one-trading-day lag."
    momentum_window = 80
    sma_window = 200

    positive_weights = {"XLE": 0.5, "XLI": 1 / 6, "XLF": 1 / 6, "XLB": 1 / 6}
    negative_weights = {"XLU": 1 / 3, "XLV": 1 / 3, "XLP": 1 / 3}

    def decisions(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.data_assets) - set(daily_prices.columns)
        if missing:
            raise ValueError(f"Inflation Compass Steady is missing assets: {sorted(missing)}")
        prices = daily_prices.loc[:, self.data_assets].sort_index()
        # FRED has observations on some Federal business days when NYSE ETFs
        # do not trade. Build every market indicator on the shared ETF trading
        # calendar, rather than letting those macro-only rows break a 200-day
        # moving average or a daily sector return.
        market = prices.loc[:, list(self.market_data_assets)].dropna(how="any")
        market_returns = market.pct_change(fill_method=None)
        positive_returns = sum(market_returns[asset] * weight for asset, weight in self.positive_weights.items())
        negative_returns = sum(market_returns[asset] * weight for asset, weight in self.negative_weights.items())
        # The published indicator compounds daily rebalanced basket returns;
        # raw ETF price levels are never combined.
        positive_growth = (1 + positive_returns).cumprod()
        negative_growth = (1 + negative_returns).cumprod()
        inflation_indicator = positive_growth / negative_growth
        spy_sma = market["SPY"].rolling(self.sma_window, min_periods=self.sma_window).mean()
        fred = prices["T5YIE"].dropna()

        as_of = pd.Timestamp.now(tz="UTC").tz_localize(None)
        completed_period = as_of.to_period("M") - 1
        market_dates = market.index[market.index.to_period("M") <= completed_period]
        decision_dates = market_dates.to_series().groupby(market_dates.to_period("M")).tail(1)
        rows: list[dict] = []
        previous_weights: dict[str, float] = {}

        for date in pd.DatetimeIndex(decision_dates):
            # Strictly earlier means a live month-end decision never relies on
            # that day's potentially unpublished FRED observation.
            available_fred = fred.loc[fred.index < date]
            indicator_window = inflation_indicator.loc[:date].dropna().tail(self.momentum_window)
            if len(available_fred) < self.momentum_window + 1 or len(indicator_window) < self.momentum_window or pd.isna(spy_sma.loc[date]):
                continue
            t5yie_date = available_fred.index[-1]
            t5yie_value = float(available_fred.iloc[-1])
            t5yie_80_date = available_fred.index[-(self.momentum_window + 1)]
            t5yie_80_value = float(available_fred.iloc[-(self.momentum_window + 1)])
            slope = float(np.polyfit(np.arange(self.momentum_window), indicator_window.to_numpy(), 1)[0])
            growth_up = bool(market.loc[date, "SPY"] > spy_sma.loc[date])
            inflation_level = t5yie_value > 2.0
            breakeven_momentum = t5yie_value > t5yie_80_value
            asset_momentum = slope > 0
            inflation_on = inflation_level and (breakeven_momentum or asset_momentum)

            if growth_up and inflation_on:
                weights, regime = {"XLE": 1.0}, "inflationary-expansion"
            elif growth_up:
                weights, regime = {"XLK": 1.0}, "disinflationary-expansion"
            elif inflation_on:
                weights, regime = {"XLU": 1.0}, "stagflation"
            else:
                weights, regime = {"XLP": 0.5, "IEF": 0.5}, "disinflationary-slowdown"

            rows.append({
                "signal_date": date,
                **{f"{asset}_price": prices.loc[date, asset] for asset in self.data_assets},
                "SPY_200d_sma": float(spy_sma.loc[date]),
                "t5yie_lag_date": t5yie_date,
                "t5yie_lagged": t5yie_value,
                "t5yie_80d_date": t5yie_80_date,
                "t5yie_80d": t5yie_80_value,
                "positive_basket_growth": float(positive_growth.loc[date]),
                "negative_basket_growth": float(negative_growth.loc[date]),
                "inflation_indicator": float(inflation_indicator.loc[date]),
                "indicator_80d_slope": slope,
                "growth_up": growth_up,
                "inflation_level": inflation_level,
                "breakeven_momentum": breakeven_momentum,
                "asset_momentum": asset_momentum,
                "inflation_on": inflation_on,
                "regime": regime,
                "selected_asset": ", ".join(weights),
                "selected_assets": ", ".join(weights),
                "target_weights": weights,
                "previous_weights": previous_weights.copy(),
                "previous_asset": ", ".join(previous_weights) if previous_weights else None,
                "trade": weights != previous_weights,
            })
            previous_weights = weights
        return pd.DataFrame(rows).set_index("signal_date") if rows else pd.DataFrame()


class InflationCompassPlus(InflationCompassSteady):
    """Three-state T5YIE Inflation Compass Plus implementation."""

    name = "Inflation Compass Plus"
    drift_bands = False

    def _weights(self, growth_up: bool, state: str):
        if growth_up and state == "rising":
            return {"XLE": 1.0}, "inflationary-expansion-rising"
        if growth_up and state in {"falling", "neutral"}:
            return {"XLE": 0.5, "XLK": 0.5}, "inflationary-expansion-transition"
        if growth_up:
            return {"XLK": 1.0}, "disinflationary-expansion"
        if state == "rising":
            return {"XLU": 1.0}, "stagflation-rising"
        if state in {"falling", "neutral"}:
            return {"XLU": 0.5, "XLP": 0.25, "IEF": 0.25}, "stagflation-transition"
        return {"XLP": 0.5, "IEF": 0.5}, "disinflationary-slowdown"

    def decisions(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.data_assets) - set(daily_prices.columns)
        if missing:
            raise ValueError(f"{self.name} is missing assets: {sorted(missing)}")
        prices = daily_prices.loc[:, self.data_assets].sort_index()
        market = prices.loc[:, list(self.market_data_assets)].dropna(how="any")
        returns = market.pct_change(fill_method=None)
        pos = sum(returns[a] * w for a, w in self.positive_weights.items())
        neg = sum(returns[a] * w for a, w in self.negative_weights.items())
        pos_growth, neg_growth = (1 + pos).cumprod(), (1 + neg).cumprod()
        indicator = pos_growth / neg_growth
        sma = market["SPY"].rolling(self.sma_window, min_periods=self.sma_window).mean()
        fred = prices["T5YIE"].dropna()
        completed = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M") - 1
        dates = market.index[market.index.to_period("M") <= completed]
        decision_dates = dates.to_series().groupby(dates.to_period("M")).tail(1)
        rows, previous_weights, previous_direction = [], {}, None
        previous_signal_date = None
        for date in pd.DatetimeIndex(decision_dates):
            available = fred.loc[fred.index < date]
            prior_dates = market.index[market.index.to_period("M") == date.to_period("M") - 3]
            window = indicator.loc[:date].dropna().tail(self.momentum_window)
            if not len(available) or not len(prior_dates) or len(window) < self.momentum_window or pd.isna(sma.loc[date]):
                continue
            compare = fred.loc[fred.index < prior_dates[-1]]
            if not len(compare):
                continue
            current, prior = float(available.iloc[-1]), float(compare.iloc[-1])
            raw = "rising" if current > prior else "falling" if current < prior else "equal"
            level = "above-2" if current > 2.0 else "at-or-below-2"
            if level == "at-or-below-2":
                state = level
            elif raw == "equal":
                state = previous_direction or "neutral"
            else:
                state = raw
            stored = raw if level == "above-2" and raw in {"rising", "falling"} else previous_direction
            growth_up = bool(market.loc[date, "SPY"] > sma.loc[date])
            weights, regime = self._weights(growth_up, state)
            rebalance = weights != previous_weights
            reason = "target-change" if rebalance else "hold"
            if self.drift_bands and previous_signal_date is not None and not rebalance and len(weights) > 1:
                values = {a: weights[a] * market.loc[date, a] / market.loc[previous_signal_date, a] for a in weights}
                total = sum(values.values())
                if any(abs(values[a] / total - weights[a]) > 0.10 for a in weights):
                    rebalance, reason = True, "drift-band"
            rows.append({"signal_date": date, **{f"{a}_price": prices.loc[date, a] for a in self.data_assets},
                "SPY_200d_sma": float(sma.loc[date]), "t5yie_lag_date": available.index[-1], "t5yie_lagged": current,
                "t5yie_three_month_date": compare.index[-1], "t5yie_three_month": prior, "t5yie_80d_date": compare.index[-1], "t5yie_80d": prior,
                "inflation_level": level, "raw_direction": raw, "persisted_direction": stored, "inflation_state": state,
                "growth_up": growth_up, "positive_basket_growth": float(pos_growth.loc[date]), "negative_basket_growth": float(neg_growth.loc[date]),
                "inflation_indicator": float(indicator.loc[date]), "indicator_80d_slope": float(np.polyfit(np.arange(self.momentum_window), window.to_numpy(), 1)[0]),
                "breakeven_momentum": raw == "rising", "asset_momentum": True, "inflation_on": state in {"rising", "falling", "neutral"},
                "regime": regime, "selected_asset": ", ".join(weights), "selected_assets": ", ".join(weights), "target_weights": weights,
                "previous_weights": previous_weights.copy(), "previous_asset": ", ".join(previous_weights) if previous_weights else None,
                "rebalance_required": rebalance, "rebalance_reason": reason, "trade": rebalance})
            previous_weights, previous_direction, previous_signal_date = weights, stored, date
        return pd.DataFrame(rows).set_index("signal_date") if rows else pd.DataFrame()


class InflationCompassPlusDriftBands(InflationCompassPlus):
    name = "Inflation Compass Plus Drift Bands"
    drift_bands = True
