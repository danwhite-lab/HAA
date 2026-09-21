# HAA-Simple

A deliberately small, auditable Streamlit backtester for defined HAA variants. It implements no optimization, parameter sweeps, or synthetic data.

## Available models

- **HAA-Simple:** uses SPY and TIP 13612U signals; holds SPY only when both are positive, otherwise the stronger of IEF/BIL.
- **HAA-Simple Leveraged 2x (SSO):** uses the *same unleveraged SPY and TIP signals* but holds SSO in risk-on periods. It always de-risks to unleveraged IEF/BIL. SSO momentum never controls the gate. This is a high-drawdown satellite, not a core holding.
- **HAA Classic (No QQQ):** TIP is the sole canary. When TIP 13612U is positive, it holds the top four assets at 25% each from IEF, SPY, IWM, PDBC, TLT, VEA, VNQ, and VWO. IEF is eligible in both the risk-on ranking and the IEF/BIL defensive choice; BIL is defensive-only. QQQ and leverage are intentionally excluded.

## What it does

At each calendar month-end it uses Yahoo Finance adjusted close data (or a user replacement CSV) for SPY, TIP, IEF, and BIL. It calculates HAA's equal-weighted **13612U** composite:

```
(1-month return + 3-month return + 6-month return + 12-month return) / 4
```

Each return is `price at the signal date / price at the corresponding earlier month-end - 1`. This needs 12 prior observations for each asset involved in the decision. If SPY and TIP momentum are strictly positive, HAA-Simple selects SPY; otherwise it selects the available defensive asset with the higher momentum. Before BIL has accumulated enough history, a defensive allocation uses IEF when its valid history is available; no proxy data is created. The selected asset earns the **next** month's return, so the month-end signal cannot affect the same period it observes.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
streamlit run app.py
```

The app downloads Yahoo Finance history automatically. In the sidebar, enter the source ticker for each model asset as `ROLE=TICKER` (for example, `SPY=SPY` or `SPY=VOO` for a validation replacement). The leveraged model also requests `SSO=SSO`; Classic requests IWM, PDBC, TLT, VEA, VNQ, and VWO. The strategy's roles and rules remain fixed. For independent validation, use the single multi-file upload control and name each CSV with its target role, for example `SPY.csv`, `SSO.csv`, or `TIP_validation.csv`. A CSV needs `Date` and `Adj Close` (preferred) or `Close`; the upload replaces that asset's full history rather than silently filling gaps.

## Test

```bash
pytest
```

Tests cover the decision branches, a manually calculated 13612U example, future-data isolation, next-period execution, unchanged-allocation trade handling, matching benchmark dates, Classic HAA top-four selection and tie-breaking, weighted portfolio execution, and realized-only tax accounting including partial sales.

## Compare models

The **Compare Models** tab runs two or more existing models independently over the intersection of their valid, completed monthly holding periods. It restarts every selected model at the first shared period, so all curves, returns, and the SPY benchmark use identical dates and initial value. Classic HAA can shorten the comparison period because its PDBC history begins later. The comparison is informational only and does not recommend one model.

## Tax treatment

Tax is optional and defaults to 25%. The independent tax module taxes only realized positive gains when a position is sold on an allocation change. It records cost basis and uses prior realized losses to offset later realized gains. It does not liquidate or tax the final, unrealized position.

## Deployment

This is a standard Streamlit project. Create an app at [Streamlit Community Cloud](https://share.streamlit.io/) and point it to `app.py` on the `main` branch. Community Cloud installs the root `requirements.txt`; local development can use `pip install -e '.[dev]'`.

## Independent checks built into the UI

The Validation tab shows data coverage, common backtest period, first valid signal, raw and month-end price data, and a downloadable audit row per decision. Classic HAA audit rows additionally show all asset scores, offensive ranks, selected basket, current/previous target weights, trade flag, and the following holding-period return. For weighted portfolios, annual turnover is the one-way fraction of portfolio value purchased at each rebalance, annualized.
