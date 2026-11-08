from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scanner import (
    fetch_universe, scan_market, load_watchlist, save_watchlist,
    fetch_chart_data, DEFAULT_WATCHLIST, DATA_NOTE
)

st.set_page_config(page_title="AI Trading Laboratory", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1200px}
[data-testid="stMetric"]{background:#111d2b;border:1px solid #26384d;padding:12px;border-radius:12px}
div.stButton>button{width:100%;min-height:2.8rem;border-radius:10px;font-weight:650}
@media(max-width:640px){.block-container{padding-left:.8rem;padding-right:.8rem}h1{font-size:1.8rem!important}}
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state: st.session_state.results = pd.DataFrame()
if "last_scan" not in st.session_state: st.session_state.last_scan = None

st.title("AI Trading Laboratory")
st.caption("Advanced mobile market scanner · Day trading + swing trading · Long + short")
st.warning(DATA_NOTE, icon="⚠️")

with st.expander("Scanner settings", expanded=True):
    mode_label = st.radio("Trading horizon", ["Both", "Day trading", "Swing trading"], horizontal=True)
    universe_choice = st.selectbox("Stocks to scan", ["Starter watchlist (25)", "Full US listing universe"], index=0)
    min_price = st.number_input("Minimum share price ($)", min_value=0.50, max_value=1000.0, value=2.0, step=0.50)
    min_dv = st.number_input("Minimum average daily dollar volume ($)", min_value=0, max_value=10_000_000_000, value=5_000_000, step=1_000_000)
    min_score = st.slider("Minimum signal score", 0, 100, 50, 5)
    direction_filter = st.multiselect("Position direction", ["LONG", "SHORT"], default=["LONG","SHORT"])
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
        status.info(f"Scanning {len(symbols):,} symbols for bullish and bearish setups…")
        mode = {"Both":"both","Day trading":"intraday","Swing trading":"daily"}[mode_label]
        results = scan_market(
            symbols, mode=mode, min_price=float(min_price), min_avg_dollar_volume=float(min_dv),
            progress_callback=lambda done,total: progress.progress(min(done/max(total,1),1.0), text=f"Processed {done:,} / {total:,} timeframe-symbol checks")
        )
        if not results.empty:
            results = results[(results["score"] >= min_score) & (results["direction"].isin(direction_filter))].copy()
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
    st.info("No results loaded yet, or no signals passed your filters. Try lowering the minimum score or scanning the starter watchlist.")
else:
    long_count = int((results["direction"] == "LONG").sum())
    short_count = int((results["direction"] == "SHORT").sum())
    day_count = int((results["horizon"] == "Day trade").sum())
    swing_count = int((results["horizon"] == "Swing").sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Signals", len(results))
    c2.metric("Long setups", long_count)
    c3.metric("Short setups", short_count)
    c4.metric("Day / swing", f"{day_count} / {swing_count}")

    tab1,tab2,tab3 = st.tabs(["Ranked opportunities","Charts & risk model","Export"])
    with tab1:
        directions = st.multiselect("Direction", ["LONG","SHORT"], default=["LONG","SHORT"], key="result_direction")
        horizons = st.multiselect("Horizon", sorted(results["horizon"].unique()), default=sorted(results["horizon"].unique()))
        signals = st.multiselect("Signal type", sorted(results["signal"].unique()), default=sorted(results["signal"].unique()))
        view = results[results["direction"].isin(directions) & results["horizon"].isin(horizons) & results["signal"].isin(signals)]
        cols = ["symbol","direction","horizon","timeframe","signal","score","price","rsi14","rvol","vwap","atr_pct"]
        cols = [c for c in cols if c in view.columns]
        st.dataframe(view[cols].round(2), use_container_width=True, hide_index=True,
            column_config={
                "score":st.column_config.ProgressColumn("Score",min_value=0,max_value=100,format="%d"),
                "price":st.column_config.NumberColumn("Price",format="$%.2f"),
                "rvol":st.column_config.NumberColumn("RVOL",format="%.2fx"),
                "vwap":st.column_config.NumberColumn("VWAP",format="$%.2f"),
            })
        st.caption("Scores rank indicator alignment; they are not probabilities of profit.")
    with tab2:
        symbols = sorted(results["symbol"].unique().tolist())
        selected = st.selectbox("Ticker", symbols)
        selected_rows = results[results["symbol"] == selected].sort_values("score",ascending=False)
        chosen = st.selectbox("Setup", selected_rows.index.tolist(), format_func=lambda i: f"{results.loc[i,'direction']} · {results.loc[i,'horizon']} · {results.loc[i,'signal']} · score {results.loc[i,'score']}")
        row = results.loc[chosen]
        timeframe = row["timeframe"]
        chart = fetch_chart_data(selected, timeframe)
        if not chart.empty:
            fig = go.Figure(data=[go.Candlestick(x=chart.index,open=chart["Open"],high=chart["High"],low=chart["Low"],close=chart["Close"],name=selected)])
            for col,label in [("EMA9","EMA 9"),("EMA21","EMA 21"),("EMA50","EMA 50")]:
                if col in chart: fig.add_trace(go.Scatter(x=chart.index,y=chart[col],mode="lines",name=label))
            if timeframe != "1D" and "VWAP" in chart:
                fig.add_trace(go.Scatter(x=chart.index,y=chart["VWAP"],mode="lines",name="VWAP"))
            fig.update_layout(height=430,margin=dict(l=5,r=5,t=25,b=5),xaxis_rangeslider_visible=False,legend_orientation="h")
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("Chart data is temporarily unavailable from the current data provider.")
        price = float(row["price"])
        atr = float(row["atr"]) if pd.notna(row["atr"]) else np.nan
        st.subheader("Illustrative ATR risk framework")
        st.caption("Educational reference only. These levels are mechanical examples, not personalized trade recommendations.")
        if np.isfinite(atr) and atr > 0:
            if row["direction"] == "LONG":
                stop = price - 1.5*atr
                target = price + 3.0*atr
            else:
                stop = price + 1.5*atr
                target = price - 3.0*atr
            a,b,c = st.columns(3)
            a.metric("Reference price",f"${price:.2f}")
            b.metric("Illustrative stop",f"${stop:.2f}")
            c.metric("Illustrative 2R target",f"${target:.2f}")
            st.write(f"Direction: **{row['direction']}** · ATR(14): **${atr:.2f}** · Risk distance: **${1.5*atr:.2f}**")
        else:
            st.info("ATR is unavailable for this row; no example levels calculated.")
        st.write("**Why it qualified**")
        st.write(f"{row['signal']} · {row['direction']} · {row['horizon']} · score {int(row['score'])}/100")
        st.write(f"RSI: {row['rsi14']:.1f} · Relative volume: {row['rvol']:.2f}x")
    with tab3:
        st.download_button("Download scan results (CSV)",results.to_csv(index=False).encode("utf-8"),"market_scan_results.csv","text/csv",use_container_width=True)

st.divider()
st.subheader("Watchlist")
watchlist = load_watchlist()
new_symbols = st.text_input("Add ticker symbols (comma-separated)",placeholder="AAPL, MSFT, NVDA")
if st.button("Save watchlist"):
    additions=[x.strip().upper().replace(".","-") for x in new_symbols.split(",") if x.strip()]
    watchlist=sorted(set(watchlist+additions))
    save_watchlist(watchlist)
    st.success(f"Saved {len(watchlist)} tickers.")
st.caption("Research scanner only. It does not connect to Robinhood, route orders, confirm borrow availability, or guarantee real-time quotes.")
