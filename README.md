# AI Trading Laboratory — Mobile Market Scanner

A mobile-friendly Streamlit web app for screening US stocks and ETFs on daily and 5-minute data.

## Important limitations

- This is a research/screening prototype, not a brokerage app. It does not connect to Robinhood or place orders.
- It uses Yahoo Finance through `yfinance`, an unofficial data interface. Quotes can be delayed, coverage can be incomplete, and bulk requests can be throttled.
- The initial app defaults to a 25-symbol starter watchlist. The full US listing universe can contain thousands of tickers and may take a long time or fail under Yahoo Finance rate limits.
- The “liquid large-cap” choice currently uses the starter list; it is a placeholder until a proper market-data provider is connected.
- The dashboard remains hosted online, but this first version scans only when you press **Run market scan**. It does not yet run scheduled scans in the background after you close the browser. That requires a separate scheduled worker and persistent hosting.

## Easiest phone-first deployment

You can do the setup from your Android phone using Chrome, GitHub, and Streamlit Community Cloud.

1. Create/sign in to a GitHub account.
2. Create a new repository named `ai-trading-laboratory`.
3. Upload `app.py`, `scanner.py`, `requirements.txt`, and `README.md` from this folder to the repository root.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Choose the repository, branch `main`, and main file `app.py`; deploy.
6. Open the generated app URL on your phone. In Chrome, use **Add to Home screen** for an app-like shortcut.

## Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Next engineering stage

Before relying on this for live trading decisions, replace the Yahoo Finance data adapter with a licensed market-data API that supports reliable intraday bars, then add persistent storage, scheduled background scans, alert delivery, and historical backtesting. API keys should be stored as deployment secrets, never committed to GitHub.


## Stage 2 — Long/short and charting

The current version scores bullish and bearish setups for both daily swing and 5-minute intraday horizons. The detail view includes candlestick charts with EMA overlays and an illustrative ATR-based risk framework. Scores are ranking heuristics, not win probabilities. Short flags do not establish that a security is borrowable or shortable.

After replacing `app.py` and `scanner.py` in GitHub, Streamlit Community Cloud should detect the commit and redeploy. `requirements.txt` now includes Plotly for charts.


## Stage 3 — Historical backtesting

`backtest.py` adds a rule-based historical simulator. It evaluates signals at bar close, enters at the next bar open, and applies ATR-based stop/target exits. If stop and target both fall within the same bar, the stop is assumed first (conservative intrabar ordering). Slippage and per-side fees are configurable.

Important limitations: Yahoo Finance intraday history is short (typically about five trading days for 5-minute bars); daily history is longer. Results are single-symbol/small-watchlist research tests, not a full-market walk-forward study. The model does not include borrow fees/availability, market impact, taxes, corporate-action execution details, or survivorship-bias correction. Do not interpret simulated returns as expected future performance.


## Stage 3.1 — Portfolio-aware backtesting

The backtest UI now supports daily, intraday, or both horizons; risk-budget sizing; max simultaneous positions; and position exposure caps. It reports strategy-sliced results and a chronological 70/30 research split. The portfolio engine processes overlapping candidate positions and reserves gross exposure while positions remain open.

Important caveat: portfolio equity is updated when trades exit, not continuously marked to market. The resulting drawdown can understate interim drawdown. The 70/30 split is a basic holdout, not a fully nested walk-forward test. A buy-and-hold benchmark and borrow-cost model are not included in this revision. Yahoo intraday history remains short.


## Stage 3.2 — Strategy diagnostics

- Fixed the UI mapping so “Both (daily + intraday)” explicitly runs both Swing and Day trading histories. Previously the UI label was passed literally, which could cause the backtester to treat the selection as intraday only.
- Strategy diagnostics now always show four distinct buckets: Swing LONG, Swing SHORT, Day trading LONG, Day trading SHORT, including zero-trade buckets.
- Prevents overlapping open positions in the same ticker across horizons/directions.
- Adds per-symbol, per-horizon price-only buy-and-hold reference returns and drawdowns.
- Adds in-sample/out-of-sample metrics split by each of the four strategy buckets.

Limitations: buy-and-hold figures exclude dividends and are reference series, not a portfolio-equivalent benchmark. Portfolio drawdown is still based on realized exits rather than continuous mark-to-market of open positions. The 70/30 split remains a basic chronological holdout, not a nested walk-forward test. Intraday history remains limited by Yahoo Finance.


## Stage 3.5 — Individual ticker diagnostics

Adds a strategy selector for Swing LONG, Swing SHORT, Day trading LONG, and Day trading SHORT. For each selected strategy, the dashboard reports per-ticker candidate trade count, win rate, profit factor, mean trade return, compounded sequence return, average win/loss, gross profit/loss, accepted portfolio trade count, and allocated portfolio P&L. It also independently splits each ticker's selected strategy trades chronologically into in-sample/out-of-sample subsets and reports both samples side by side.

Important interpretation: raw candidate trade-return statistics are not portfolio returns. Compounded trade return assumes sequential reinvestment for that ticker and is diagnostic only. Portfolio P&L includes only trades accepted by portfolio constraints. This stage does not modify trading rules, place orders, connect to a broker, or activate the user's $50 allocation controls.


## Stage 3.6 — Market regime analysis

Adds SPY-based daily market-condition labels to historical signal candidates. Trend is classified as Bullish, Bearish, or Sideways / mixed using SPY close relative to EMA50 and EMA200. Volatility is classified as Low, Normal, or High using SPY ATR(14) percentage against trailing 252-session 33rd/67th percentile thresholds. The dashboard compares the selected strategy's candidate-trade performance by combined trend/volatility regime and by ticker within each regime.

Regimes are joined using the latest available SPY daily observation on or before each signal date. Intraday candidates inherit the latest daily SPY regime; this is not an intraday volatility classification. The feature is diagnostic only and does not alter entry signals, position sizing, or order execution. Results depend on the availability and quality of Yahoo Finance daily SPY data. Small sample sizes must not be treated as evidence of a durable edge.


## Stage 3.7 — Strategy parameter research

Adds a bounded 27-configuration grid for a selected strategy: minimum signal scores 50/60/70, ATR stop distances 1.0/1.5/2.0, and reward targets 1.5R/2.0R/3.0R. Configurations are ranked using chronological in-sample trade results only, subject to a minimum in-sample trade count. The selected configuration's later chronological holdout metrics are reported separately and are not used for ranking. The feature does not modify the scanner's live signals or execute trades.

This is a single-split research aid, not nested walk-forward validation. Repeatedly changing the grid after viewing holdout results invalidates the holdout as untouched data. Profit factor can be unstable with small samples; configurations with infinite in-sample profit factor are not eligible for selection. Costs are modeled using the selected slippage and fee settings. Yahoo intraday history remains short, so Day trading optimization is preliminary.


## Stage 3.8 — Walk-forward validation

Adds rolling chronological validation for Swing LONG and Swing SHORT. For each fold, the engine evaluates the predefined 27-configuration grid using only trades exiting in the prior training window, selects the best eligible configuration using training profit factor and mean return, and evaluates that configuration on trades entering in the next forward test window. The dashboard reports fold-level candidate metrics, per-fold portfolio simulations, aggregate consistency statistics, selected parameters, and downloadable CSVs.

Each fold starts a fresh portfolio with the configured starting capital; fold returns are not compounded into one continuous account. The rolling test windows can overlap when the advance is shorter than the test window. Day trading is exposed in the selector but intentionally blocked because the current Yahoo intraday history is only a few recent trading days and cannot support multi-month validation. A longer-history intraday provider is required before day-trading walk-forward results can be meaningful.

This is research tooling, not live execution or proof of profitability. The 27-configuration grid is still a limited search, and repeated model development can contaminate later periods viewed by the researcher. Small samples, survivorship bias, market impact, short borrow availability/costs, taxes, and continuous intrabar mark-to-market are not fully modeled.


## Stage 3.9 — Signal quality and feature research

Adds a descriptive feature-analysis panel for one selected strategy at a time. It reruns historical candidate trades with a zero minimum score threshold (while retaining the scanner's underlying eligibility rules), applies the latest-known SPY market regime as of each signal date, and groups outcomes by signal-score bucket, RSI(14) bucket, relative-volume bucket, SPY market regime, ticker, signal type, and exit reason. Groups below the user-selected minimum sample count are omitted. Reports include trade count, win rate, profit factor, mean/median return, and average win/loss, with CSV downloads for the grouped report and underlying candidate trades.

The diagnostic run uses a fixed 1.5 ATR stop and 2.0R target and does not reuse or automatically adopt Stage 3.7/3.8 selected parameters. Findings are descriptive associations, not causal evidence or predictive filters. Any proposed filter must be evaluated on later untouched periods using walk-forward validation. Yahoo intraday history remains too short for robust multi-regime day-trading research. This stage does not modify live scanner signals or execute orders.
