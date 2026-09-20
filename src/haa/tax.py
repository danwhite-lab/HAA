"""Realized-gain-only Israeli capital-gains tax accounting, independent of strategy rules."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IsraeliTaxState:
    tax_rate: float = 0.25
    cost_basis: float | None = None
    asset: str | None = None
    loss_carryforward: float = 0.0
    total_tax_paid: float = 0.0

    def sell(self, proceeds: float) -> dict[str, float]:
        """Realize a sale; losses offset future realized gains and unrealized gains are untouched."""
        if self.cost_basis is None:
            return {"realized_gain": 0.0, "taxable_gain": 0.0, "tax_paid": 0.0, "loss_carryforward": self.loss_carryforward}
        realized_gain = proceeds - self.cost_basis
        if realized_gain >= 0:
            taxable_gain = max(0.0, realized_gain - self.loss_carryforward)
            self.loss_carryforward = max(0.0, self.loss_carryforward - realized_gain)
        else:
            taxable_gain = 0.0
            self.loss_carryforward += -realized_gain
        tax_paid = taxable_gain * self.tax_rate
        self.total_tax_paid += tax_paid
        self.cost_basis = None
        self.asset = None
        return {"realized_gain": realized_gain, "taxable_gain": taxable_gain, "tax_paid": tax_paid, "loss_carryforward": self.loss_carryforward}

    def buy(self, asset: str, amount: float) -> None:
        self.asset = asset
        self.cost_basis = amount
