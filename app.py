from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from scanner import (
    fetch_universe, scan_market, load_watchlist, save_watchlist,
    DEFAULT_WATCHLIST, DATA_NOTE
)

st.set_page_config(
    page_title="AI Trading Laboratory",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px;}
  [data-testid="stMetric"] {background: #111d2b; border: 1px solid #26384d; padding: 12px; border-radius: 12px;}
  div.stButton > button {width: 100%; min-height: 2.8rem; border-radius: 10px; font-weight: 650;}
  .small-note {font-size: .82rem; opacity: .78;}
  h1 {letter-spacing: -.04em;}
  @media (max-width: 640px) {
    .block-container {padding-left: .8rem; padding-right: .8rem;}
    h1 {font-size: 1.8rem !important;}
  }
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "last_scan" not in st.session_state:
    st.session_state.last_scan = None
if "symbols" not in st.session_state:
    st.session_state.symbols = DEFAULT_WATCHLIST.copy()

st.title("AI Trading Laboratory")
st.caption("Mobile market scanner · Daily swing + intraday momentum")
st.warning(DATA_NOTE, icon="⚠️")

with st.expander("Scanner settings", expanded=True):
    mode = st.radio("Scan type", ["Both", "Daily swing", "Intraday"], horizontal=True)
    universe_choice = st.selectbox(
        "Stocks to scan",
        ["Starter watchlist (25)", "Full US listing universe"],
        index=0,
    )
    min_price = st.number_input("Minimum share price ($)", min_value=0.50, max_value=1000.0, value=2.0, step=0.50)
    min_dollar_volume = st.number_input("Minimum average daily dollar volume ($)", min_value=0, max_value=10_000_000_000, value=5_000_000, step=1_000_000)
    min_score = st.slider("Minimum signal score", 0, 100, 50, 5)
    run_scan = st.button("Run market scan", type="primary")

if run_scan:
    progress = st.progress(0, text="Preparing symbol universe…")
    status = st.empty()
    try:
        if universe_choice == "Starter watchlist (25)":
            symbols = DEFAULT_WATCHLIST.copy()
        else:
            status.info("Loading US exchange listings…")
            symbols = fetch_universe(include_etfs=True)
            symbols = symbols[:]
        st.session_state.symbols = symbols
        status.info(f"Scanning {len(symbols):,} symbols. This may take several minutes.")
        results = scan_market(
            symbols=symbols,
            mode={"Both": "both", "Daily swing": "daily", "Intraday": "intraday"}[mode],
            min_price=float(min_price),
            min_avg_dollar_volume=float(min_dollar_volume),
            progress_callback=lambda done, total: progress.progress(
                min(done / max(total, 1), 1.0), text=f"Scanned {done:,} / {total:,} symbols"
            ),
        )
        if not results.empty:
            results = results[results["score"] >= min_score].copy()
        st.session_state.results = results
        st.session_state.last_scan = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        progress.progress(1.0, text="Scan complete")
        status.success(f"Scan complete. {len(results):,} qualifying signals.")
    except Exception as exc:
        status.error(f"Scan failed: {exc}")
        st.exception(exc)

results = st.session_state.results
if st.session_state.last_scan:
    st.caption(f"Last scan: {st.session_state.last_scan}")

if results.empty:
    st.info("No scan results yet. Start with the 25-symbol watchlist, then expand coverage after confirming the data feed works.")
else:
    daily_count = int((results["timeframe"] == "1D").sum()) if "timeframe" in results else 0
    intra_count = int((results["timeframe"] == "5m").sum()) if "timeframe" in results else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Qualifying signals", f"{len(results):,}")
    c2.metric("Daily setups", f"{daily_count:,}")
    c3.metric("Intraday setups", f"{intra_count:,}")

    tabs = st.tabs(["Ranked opportunities", "Charts & details", "Export"])
    with tabs[0]:
        timeframe_filter = st.multiselect(
            "Timeframe", options=sorted(results["timeframe"].dropna().unique().tolist()),
            default=sorted(results["timeframe"].dropna().unique().tolist())
        )
        signal_filter = st.multiselect(
            "Signal type", options=sorted(results["signal"].dropna().unique().tolist()),
            default=sorted(results["signal"].dropna().unique().tolist())
        )
        view = results[
            results["timeframe"].isin(timeframe_filter) &
            results["signal"].isin(signal_filter)
        ].sort_values(["score", "rvol"], ascending=[False, False])
        display_cols = [c for c in [
            "symbol", "timeframe", "signal", "score", "price", "rvol",
            "rsi14", "vwap", "atr_pct", "avg_dollar_volume_20d"
        ] if c in view.columns]
        st.dataframe(
            view[display_cols].round(2),
            use_container_width=True,
            hide_index=True,
            column_config={
                "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "avg_dollar_volume_20d": st.column_config.NumberColumn("Avg $ volume (20d)", format="$%.0f"),
            },
        )
    with tabs[1]:
        selected = st.selectbox("Select a ticker", sorted(results["symbol"].unique().tolist()))
        row = results[results["symbol"] == selected].sort_values("score", ascending=False).iloc[0]
        a, b, c = st.columns(3)
        a.metric("Last price", f"${row['price']:.2f}")
        b.metric("Signal score", f"{int(row['score'])}/100")
        c.metric("Signal", str(row["signal"]))
        st.json({k: (None if pd.isna(v) else (float(v) if hasattr(v, "item") else v))
                 for k, v in row.to_dict().items()})
        st.caption("Indicator values are screening inputs, not trade instructions.")
    with tabs[2]:
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download scan results (CSV)", csv, "market_scan_results.csv", "text/csv", use_container_width=True)

st.divider()
st.subheader("Watchlist")
watchlist = load_watchlist()
new_symbols = st.text_input("Add ticker symbols (comma-separated)", placeholder="AAPL, MSFT, NVDA")
if st.button("Save watchlist"):
    additions = [s.strip().upper().replace(".", "-") for s in new_symbols.split(",") if s.strip()]
    watchlist = sorted(set(watchlist + additions))
    save_watchlist(watchlist)
    st.success(f"Saved {len(watchlist)} tickers.")
st.caption("This first build is a screening prototype. It does not connect to Robinhood, place orders, or provide guaranteed real-time quotes.")
