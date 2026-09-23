"""HAA-4 decisions with actual 2x ETF execution holdings."""
from __future__ import annotations

import pandas as pd

from ..constants import (
    HAA4_DATA_ASSETS,
    HAA4_DEFENSIVE_ASSETS,
    HAA4_LEVERAGED_DATA_ASSETS,
    HAA4_LEVERAGED_SUBSTITUTIONS,
    HAA4_OFFENSIVE_ASSETS,
)
from ..momentum import momentum_13612u


class HAA4Leveraged2x:
    """Published HAA-4 decisions, executed through real daily-reset 2x ETFs.

    Signal inputs deliberately remain unleveraged.  The substitutions are made
    only after the TIP gate, Top-2 selection, and sleeve-level absolute-
    momentum replacement have been determined.
    """

    name = "HAA 4 Leveraged 2x"
    data_assets = HAA4_LEVERAGED_DATA_ASSETS
    signal_assets = HAA4_DATA_ASSETS
    offensive_assets = HAA4_OFFENSIVE_ASSETS
    defensive_assets = HAA4_DEFENSIVE_ASSETS
    substitutions = HAA4_LEVERAGED_SUBSTITUTIONS
    is_multi_asset = True
    risk_warning = (
        "High-drawdown leveraged satellite, not a core holding. Signals use "
        "unleveraged HAA-4 assets; 2x ETFs target daily, not monthly, returns."
    )

    def decisions(self, monthly_prices: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.data_assets) - set(monthly_prices.columns)
        if missing:
            raise ValueError(f"HAA 4 Leveraged 2x is missing assets: {sorted(missing)}")
        prices = monthly_prices.loc[:, self.data_assets]
        signal_prices = prices.loc[:, self.signal_assets]
        momenta = signal_prices.apply(momentum_13612u)
        rows: list[dict] = []
        previous_weights: dict[str, float] = {}

        for date, scores in momenta.iterrows():
            # A valid HAA-4 decision needs every unleveraged score. Actual
            # holding availability is checked after mapping below.
            if scores.isna().any():
                continue
            defensive = scores.loc[list(self.defensive_assets)].sort_values(ascending=False, kind="stable")
            defensive_winner = defensive.index[0]
            mapped_defensive = self.substitutions[defensive_winner]
            ranks = {asset: pd.NA for asset in self.offensive_assets}
            selected_underlyings: tuple[str, ...] = ()
            replaced_underlyings: tuple[str, ...] = ()

            if scores["TIP"] <= 0:
                weights = {mapped_defensive: 1.0}
                regime = "risk-off"
            else:
                candidates = scores.loc[list(self.offensive_assets)].reindex(sorted(self.offensive_assets))
                candidates = candidates.sort_values(ascending=False, kind="stable")
                selected_underlyings = tuple(candidates.index[:2])
                ranks = {asset: int(candidates.index.get_loc(asset)) + 1 for asset in self.offensive_assets}
                weights: dict[str, float] = {}
                replaced: list[str] = []
                for underlying in selected_underlyings:
                    holding = self.substitutions[underlying] if scores[underlying] > 0 else mapped_defensive
                    if scores[underlying] <= 0:
                        replaced.append(underlying)
                    weights[holding] = weights.get(holding, 0.0) + 0.5
                replaced_underlyings = tuple(replaced)
                regime = "risk-on" if not replaced_underlyings else "risk-on-defensive-replacement"

            # A model period begins only when every source and execution
            # series is real. This avoids selectively stitching an earlier
            # history based on the holding chosen in a particular month.
            if prices.loc[date, list(self.data_assets)].isna().any():
                continue
            mapped_holdings = tuple(weights)
            rows.append({
                "signal_date": date,
                **{f"{asset}_price": prices.loc[date, asset] for asset in self.data_assets},
                **{f"{asset}_13612u": scores[asset] for asset in self.signal_assets},
                **{f"{asset}_rank": ranks[asset] for asset in self.offensive_assets},
                "defensive_winner": defensive_winner,
                "selected_offensive_assets": ", ".join(selected_underlyings),
                "selected_underlying_assets": ", ".join(selected_underlyings),
                "replaced_offensive_assets": ", ".join(replaced_underlyings),
                "mapped_holding_assets": ", ".join(mapped_holdings),
                "regime": regime,
                "selected_asset": ", ".join(mapped_holdings),
                "selected_assets": ", ".join(selected_underlyings),
                "target_weights": weights,
                "previous_weights": previous_weights.copy(),
                "previous_asset": ", ".join(previous_weights) if previous_weights else None,
                "trade": weights != previous_weights,
            })
            previous_weights = weights
        return pd.DataFrame(rows).set_index("signal_date") if rows else pd.DataFrame()
