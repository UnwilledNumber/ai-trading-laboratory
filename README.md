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
