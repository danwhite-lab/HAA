"""Canonical HAA-Simple decision rules only; no portfolio accounting lives here."""
from __future__ import annotations

import pandas as pd

from ..constants import ASSETS, STRATEGY_NAME
from ..momentum import momentum_13612w


class HAASimple:
    name = STRATEGY_NAME

    @staticmethod
    def select_asset(spy_momentum: float, tip_momentum: float, ief_momentum: float, bil_momentum: float) -> tuple[str, str]:
        """Return (regime, selected asset) for one fully known month-end signal."""
        if spy_momentum > 0 and tip_momentum > 0:
            return "risk-on", "SPY"
        return "risk-off", "IEF" if ief_momentum > bil_momentum else "BIL"

    def decisions(self, monthly_prices: pd.DataFrame) -> pd.DataFrame:
        """Make month-end decisions, leaving execution to the next holding period."""
        missing = set(ASSETS) - set(monthly_prices.columns)
        if missing:
            raise ValueError(f"Missing canonical assets: {sorted(missing)}")
        prices = monthly_prices.loc[:, ASSETS].dropna(how="any")
        momenta = prices.apply(momentum_13612w)
        valid = momenta.dropna(how="any")
        rows: list[dict] = []
        previous: str | None = None
        for date, values in valid.iterrows():
            regime, selected = self.select_asset(values["SPY"], values["TIP"], values["IEF"], values["BIL"])
            rows.append({
                "signal_date": date,
                **{f"{asset}_price": prices.loc[date, asset] for asset in ASSETS},
                **{f"{asset}_13612w": values[asset] for asset in ASSETS},
                "regime": regime,
                "selected_asset": selected,
                "previous_asset": previous,
                "trade": previous is None or selected != previous,
            })
            previous = selected
        return pd.DataFrame(rows).set_index("signal_date") if rows else pd.DataFrame()
