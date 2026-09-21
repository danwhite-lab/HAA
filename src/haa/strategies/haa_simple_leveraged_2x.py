"""HAA-Simple 2x: SPY signals with SSO exposure in risk-on periods."""
from __future__ import annotations

import pandas as pd

from ..constants import LEVERAGED_ASSETS
from .haa_simple import HAASimple


class HAASimpleLeveraged2x(HAASimple):
    """Use unleveraged SPY for the gate and SSO only as the risk-on holding."""

    name = "HAA-Simple Leveraged 2x (SSO)"
    risk_on_asset = "SSO"
    data_assets = ("SPY", "TIP", "IEF", "BIL", *LEVERAGED_ASSETS)
    risk_warning = "High-drawdown satellite, not a core holding."

    def decisions(self, monthly_prices: pd.DataFrame) -> pd.DataFrame:
        if "SSO" not in monthly_prices.columns:
            raise ValueError("HAA-Simple Leveraged 2x requires SSO price history.")
        decisions = super().decisions(monthly_prices)
        if decisions.empty:
            return decisions
        decisions["SSO_price"] = monthly_prices.loc[decisions.index, "SSO"]
        risk_on = decisions["regime"].eq("risk-on")
        # SSO price availability is required for a risk-on trade, but SSO
        # momentum is deliberately never used to determine the regime.
        decisions = decisions.loc[~(risk_on & decisions["SSO_price"].isna())].copy()
        decisions.loc[decisions["regime"].eq("risk-on"), "selected_asset"] = "SSO"
        decisions["previous_asset"] = decisions["selected_asset"].shift()
        decisions["trade"] = decisions["previous_asset"].isna() | decisions["selected_asset"].ne(decisions["previous_asset"])
        return decisions
