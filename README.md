# HAA-Simple

A deliberately small, auditable Streamlit backtester for the canonical **HAA-Simple** Hybrid Asset Allocation rules. It implements no leverage, optimization, parameter sweeps, synthetic data, or non-Simple HAA variants.

## What it does

At each calendar month-end it uses Yahoo Finance adjusted close data (or a user replacement CSV) for SPY, TIP, IEF, and BIL. It calculates HAA's equal-weighted **13612U** composite:

```
(1-month return + 3-month return + 6-month return + 12-month return) / 4
```

Each return is `price at the signal date / price at the corresponding earlier month-end - 1`. This needs 12 prior observations for each asset involved in the decision. If SPY and TIP momentum are strictly positive, HAA-Simple selects SPY; otherwise it selects IEF if its momentum is higher than BIL's, or BIL otherwise. Valid risk-on signals do not wait for BIL history; IEF/BIL history is required only for a defensive selection. The selected asset earns the **next** month's return, so the month-end signal cannot affect the same period it observes.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
streamlit run app.py
```

The app downloads Yahoo Finance history automatically. In the sidebar, enter the source ticker for each canonical role as `ROLE=TICKER` (for example, `SPY=SPY` or `SPY=VOO` for a validation replacement). The strategy's roles and rules remain fixed. For independent validation, use the single multi-file upload control and name each CSV with its target role, for example `SPY.csv` or `TIP_validation.csv`. A CSV needs `Date` and `Adj Close` (preferred) or `Close`; the upload replaces that asset's full history rather than silently filling gaps.

## Test

```bash
pytest
```

Tests cover the decision branches, a manually calculated 13612U example, future-data isolation, next-period execution, unchanged-allocation trade handling, matching benchmark dates, conditional early risk-on execution, and realized-only tax accounting.

## Tax treatment

Tax is optional and defaults to 25%. The independent tax module taxes only realized positive gains when a position is sold on an allocation change. It records cost basis and uses prior realized losses to offset later realized gains. It does not liquidate or tax the final, unrealized position.

## Deployment

This is a standard Streamlit project. Create an app at [Streamlit Community Cloud](https://share.streamlit.io/) and point it to `app.py` on the `main` branch. Community Cloud installs the root `requirements.txt`; local development can use `pip install -e '.[dev]'`.

## Independent checks built into the UI

The Validation tab shows data coverage, common backtest period, first valid signal, the raw and month-end price data, and a downloadable audit row per decision. The audit shows prices, all four momenta, regime, asset decision, prior asset, trade flag, and the following holding-period return.
