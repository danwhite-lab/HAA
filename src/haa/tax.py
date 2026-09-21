"""Realized-gain-only Israeli capital-gains tax accounting, independent of strategy rules."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IsraeliTaxState:
    tax_rate: float = 0.25
    cost_basis: float | None = None
    asset: str | None = None
    loss_carryforward: float = 0.0
    total_tax_paid: float = 0.0
    cost_bases: dict[str, float] = field(default_factory=dict)

    def sell(self, proceeds: float, asset: str | None = None, cost_basis_sold: float | None = None) -> dict[str, float]:
        """Realize a sale; losses offset future realized gains and unrealized gains are untouched."""
        sold_asset = asset or self.asset
        available_basis = self.cost_bases.get(sold_asset, self.cost_basis if sold_asset == self.asset else None)
        if available_basis is None:
            return {"realized_gain": 0.0, "taxable_gain": 0.0, "tax_paid": 0.0, "loss_carryforward": self.loss_carryforward}
        basis = available_basis if cost_basis_sold is None else cost_basis_sold
        if basis < 0 or basis > available_basis + 1e-9:
            raise ValueError("Sold cost basis must be between zero and the held asset's cost basis.")
        realized_gain = proceeds - basis
        if realized_gain >= 0:
            taxable_gain = max(0.0, realized_gain - self.loss_carryforward)
            self.loss_carryforward = max(0.0, self.loss_carryforward - realized_gain)
        else:
            taxable_gain = 0.0
            self.loss_carryforward += -realized_gain
        tax_paid = taxable_gain * self.tax_rate
        self.total_tax_paid += tax_paid
        remaining_basis = max(0.0, available_basis - basis)
        if sold_asset is not None:
            if remaining_basis:
                self.cost_bases[sold_asset] = remaining_basis
            else:
                self.cost_bases.pop(sold_asset, None)
        if sold_asset == self.asset:
            self.cost_basis = remaining_basis or None
            if not remaining_basis:
                self.asset = None
        return {"realized_gain": realized_gain, "taxable_gain": taxable_gain, "tax_paid": tax_paid, "loss_carryforward": self.loss_carryforward}

    def buy(self, asset: str, amount: float) -> None:
        self.asset = asset
        self.cost_bases[asset] = self.cost_bases.get(asset, 0.0) + amount
        self.cost_basis = self.cost_bases[asset]
