from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import run_backtest, simulate_portfolio, summarize_portfolio, summarize_by_strategy, make_walk_forward_split

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
st.header("Historical Backtesting")
st.caption("Evaluate the scanner's rules on historical bars. Backtests are simulations, not forecasts.")
with st.expander("Backtest settings", expanded=False):
    bt_symbols_text = st.text_input(
        "Ticker symbols to test (comma-separated)",
        value="AAPL,MSFT,NVDA,AMZN,META",
        key="bt_symbols",
    )
    bt_horizon = st.selectbox("Historical timeframe", ["Both (daily + intraday)", "Swing", "Day trading"], key="bt_horizon")
    bt_direction = st.selectbox("Direction", ["Both", "LONG", "SHORT"], key="bt_direction")
    bt_min_score = st.slider("Minimum signal score", 0, 100, 60, 5, key="bt_min_score")
    bt_stop_atr = st.selectbox("Stop distance (ATR)", [1.0, 1.5, 2.0], index=1, key="bt_stop")
    bt_target_r = st.selectbox("Profit target (R multiple)", [1.0, 1.5, 2.0, 3.0], index=2, key="bt_target")
    bt_slippage = st.selectbox("Slippage per side (basis points)", [0.0, 5.0, 10.0, 20.0], index=1, key="bt_slippage")
    bt_fees = st.selectbox("Fees per side (basis points)", [0.0, 0.5, 1.0, 2.0], index=0, key="bt_fees")
    bt_capital = st.number_input("Starting portfolio capital ($)", min_value=1000.0, max_value=1000000.0, value=10000.0, step=1000.0)
    bt_risk = st.selectbox("Risk budget per trade (% equity)", [0.25, 0.5, 0.75, 1.0], index=1)
    bt_max_positions = st.selectbox("Maximum simultaneous positions", [2, 3, 5, 8], index=2)
    bt_position_cap = st.selectbox("Maximum position exposure (% equity)", [10, 15, 20, 25], index=3)
    run_bt = st.button("Run historical backtest", type="primary", key="run_backtest")

if run_bt:
    bt_symbols = [s.strip().upper() for s in bt_symbols_text.split(",") if s.strip()]
    if not bt_symbols:
        st.error("Enter at least one ticker symbol.")
    else:
        with st.spinner("Downloading historical bars and simulating trades. Intraday history is limited to approximately five trading days by the current provider."):
            bt_trades = run_backtest(
                bt_symbols, horizon=bt_horizon, direction=bt_direction,
                min_score=bt_min_score, min_price=2.0,
                min_dollar_volume=5_000_000, stop_atr=float(bt_stop_atr),
                target_r=float(bt_target_r), slippage_bps=float(bt_slippage),
                fee_bps_per_side=float(bt_fees),
            )
        allocated, equity = simulate_portfolio(
            bt_trades, starting_capital=float(bt_capital),
            risk_per_trade_pct=float(bt_risk), max_positions=int(bt_max_positions),
            max_position_exposure_pct=float(bt_position_cap),
        )
        st.session_state["bt_trades"] = bt_trades
        st.session_state["bt_allocated"] = allocated
        st.session_state["bt_summary"], st.session_state["bt_equity"] = summarize_portfolio(
            allocated, equity, starting_capital=float(bt_capital)
        )
        st.session_state["bt_strategy"] = summarize_by_strategy(allocated)
        train_trades, test_trades = make_walk_forward_split(
            allocated[allocated["portfolio_status"] == "TAKEN"] if not allocated.empty else allocated
        )
        st.session_state["bt_train"] = train_trades
        st.session_state["bt_test"] = test_trades

if "bt_trades" in st.session_state:
    bt_trades = st.session_state["bt_trades"]
    allocated = st.session_state.get("bt_allocated", pd.DataFrame())
    bt_summary = st.session_state.get("bt_summary", {})
    if bt_trades.empty or allocated.empty or not bt_summary:
        st.warning("No portfolio trades were generated. Try a lower signal score, another symbol, or swing timeframe.")
    else:
        st.subheader("Portfolio backtest results")
        st.caption("Capital-aware simulation with risk-based sizing, concurrent-position limits, and gross-exposure caps.")
        m1,m2,m3 = st.columns(3)
        m1.metric("Trades taken", bt_summary["trades"])
        m2.metric("Win rate", f'{bt_summary["win_rate_pct"]:.1f}%')
        m3.metric("Profit factor", f'{bt_summary["profit_factor"]:.2f}' if np.isfinite(bt_summary["profit_factor"]) else "∞")
        m4,m5,m6 = st.columns(3)
        m4.metric("Portfolio return", f'{bt_summary["net_return_pct_compounded"]:.2f}%')
        m5.metric("Expectancy / trade", f'${bt_summary["expectancy_dollars"]:.2f}')
        m6.metric("Maximum drawdown", f'{bt_summary["max_drawdown_pct"]:.2f}%')
        st.caption(
            f'Starting capital ${bt_summary["starting_capital"]:,.0f} → '
            f'ending equity ${bt_summary["ending_equity"]:,.2f} · '
            f'{bt_summary["skipped_trades"]} candidate trades skipped by portfolio constraints.'
        )
        equity = st.session_state.get("bt_equity", pd.DataFrame())
        if not equity.empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=equity["exit_time"], y=equity["equity"], mode="lines", name="Portfolio equity"))
            fig_eq.update_layout(height=300, margin=dict(l=5,r=5,t=25,b=5), yaxis_title="Portfolio value ($)")
            st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader("Performance by strategy")
        strategy = st.session_state.get("bt_strategy", pd.DataFrame())
        if not strategy.empty:
            st.dataframe(strategy.round(3), use_container_width=True, hide_index=True)

        st.subheader("Chronological 70/30 research split")
        train = st.session_state.get("bt_train", pd.DataFrame())
        test = st.session_state.get("bt_test", pd.DataFrame())
        if not train.empty and not test.empty:
            c1,c2 = st.columns(2)
            c1.metric("In-sample trades (70%)", len(train))
            c2.metric("Out-of-sample trades (30%)", len(test))
            st.caption("The final 30% is held out chronologically for a basic out-of-sample check. Do not tune rules against this holdout and then treat it as independent.")
            st.dataframe(pd.DataFrame([
                {"sample":"In-sample","trades":len(train),"win_rate_pct":(train["net_return_pct"]>0).mean()*100,
                 "mean_trade_pct":train["net_return_pct"].mean()},
                {"sample":"Out-of-sample","trades":len(test),"win_rate_pct":(test["net_return_pct"]>0).mean()*100,
                 "mean_trade_pct":test["net_return_pct"].mean()},
            ]).round(3), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough distinct trade dates to form both in-sample and out-of-sample periods.")

        st.write("**Trade log (portfolio allocations)**")
        display_cols = ["symbol","horizon","direction","signal_time","entry_time","exit_time",
                        "entry_price","exit_price","exit_reason","signal_score","net_return_pct",
                        "portfolio_status","position_notional","portfolio_pnl"]
        display_cols = [c for c in display_cols if c in allocated.columns]
        st.dataframe(allocated[display_cols].round(3), use_container_width=True, hide_index=True)
        st.download_button("Download portfolio backtest (CSV)", allocated.to_csv(index=False).encode("utf-8"),
                           "portfolio_backtest.csv", "text/csv", use_container_width=True)
        st.caption(
            "Methodology and limitations: signals are evaluated at bar close and entered at next bar open. "
            "Stops/targets use ATR at signal time; if both levels trade within one bar, stop is assumed first. "
            "Position size targets the selected equity-risk budget, subject to per-position and total gross-exposure caps. "
            "Equity is updated on exits; open positions are not continuously marked to market, so interim drawdown can be understated. "
            "The 70/30 split is chronological but this is not a fully nested walk-forward test. "
            "Benchmark comparison, borrow fees/availability, market impact, taxes, survivorship bias, and corporate-action execution "
            "are not modeled. Yahoo 5-minute history is limited, so intraday sample size is short."
        )

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
