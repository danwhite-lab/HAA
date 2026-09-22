# HAA-Simple

A deliberately small, auditable Streamlit backtester for defined HAA variants. It implements no optimization, parameter sweeps, or synthetic data.

## Available models

- **HAA-Simple:** uses SPY and TIP 13612U signals; holds SPY only when both are positive, otherwise the stronger of IEF/BIL.
- **HAA-Simple Leveraged 2x (SSO):** uses the *same unleveraged SPY and TIP signals* but holds SSO in risk-on periods. It always de-risks to unleveraged IEF/BIL. SSO momentum never controls the gate. This is a high-drawdown satellite, not a core holding.
- **HAA-Simple Israel:** retains TIP as a U.S. signal-only canary, uses TASE-listed CSPX (1159250) for the equity signal/holding, and compares TASE-listed iShares $ Treasury Bond 7–10yr UCITS (1159268) with Ayalon Kaspit (5136866) in risk-off periods. It is an ILS local-investability variant with its own CSPX benchmark; it does not use MAKAM 800 as a holding series.
- **HAA Classic (No QQQ):** TIP is the sole canary. When TIP 13612U is positive, it holds the top four assets at 25% each from IEF, SPY, IWM, PDBC, TLT, VEA, VNQ, and VWO. IEF is eligible in both the risk-on ranking and the IEF/BIL defensive choice; BIL is defensive-only. QQQ and leverage are intentionally excluded.
- **HAA Classic Leveraged 2x (No QQQ):** calculates the same TIP gate and top-four ranking on the 1× Classic no-QQQ universe, then holds 2× substitutes: IEF→UST, SPY→SSO, IWM→UWM, TLT→UBT, VEA→EFO, VNQ→URE, and VWO→EET. PDBC and BIL remain unleveraged. Risk-off compares 1× IEF/BIL momentum and holds UST or BIL. This is a high-drawdown satellite, not a core holding.

## What it does

At each completed month it uses Yahoo Finance adjusted close data (or a user replacement CSV) for the selected model's assets. It calculates HAA's equal-weighted **13612U** composite:

```
(1-month return + 3-month return + 6-month return + 12-month return) / 4
```

Each return is `price at the signal date / price at the corresponding earlier month-end - 1`. This needs 12 prior observations for every asset involved in the decision; no pre-inception cash proxy is created. The signal is calculated after the final complete trading-day close, entered on the next valid execution date, and exited on the next monthly execution date. The current incomplete month is excluded.

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
