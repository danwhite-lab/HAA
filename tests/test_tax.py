from haa.tax import IsraeliTaxState


def test_tax_only_applies_when_gain_is_realized():
    tax = IsraeliTaxState(tax_rate=0.25)
    tax.buy("SPY", 100)
    # An unrealized 20 gain does not itself create tax; sell does.
    assert tax.total_tax_paid == 0
    sale = tax.sell(120)
    assert sale["realized_gain"] == 20
    assert sale["tax_paid"] == 5


def test_realized_loss_offsets_later_realized_gain():
    tax = IsraeliTaxState(tax_rate=0.25)
    tax.buy("SPY", 100)
    loss = tax.sell(80)
    assert loss["tax_paid"] == 0
    tax.buy("IEF", 80)
    gain = tax.sell(110)
    assert gain["taxable_gain"] == 10  # 30 gain less 20 prior realized loss
    assert gain["tax_paid"] == 2.5
