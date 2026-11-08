from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import (run_backtest, simulate_portfolio, summarize_portfolio, summarize_by_strategy, make_walk_forward_split, build_benchmark_report, summarize_sample_by_strategy, audit_history_coverage, strategy_split_diagnostics, summarize_by_ticker, summarize_ticker_samples, apply_market_regimes, summarize_by_regime, optimize_strategy_parameters, run_walk_forward_validation, analyze_signal_features)

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
            normalized_horizon = (
                "Both" if bt_horizon == "Both (daily + intraday)" else bt_horizon
            )
            bt_trades = run_backtest(
                bt_symbols, horizon=normalized_horizon, direction=bt_direction,
                min_score=bt_min_score, min_price=2.0,
                min_dollar_volume=5_000_000, stop_atr=float(bt_stop_atr),
                target_r=float(bt_target_r), slippage_bps=float(bt_slippage),
                fee_bps_per_side=float(bt_fees),
            )
            bt_trades = apply_market_regimes(bt_trades)
        allocated, equity = simulate_portfolio(
            bt_trades, starting_capital=float(bt_capital),
            risk_per_trade_pct=float(bt_risk), max_positions=int(bt_max_positions),
            max_position_exposure_pct=float(bt_position_cap),
        )
        st.session_state["bt_trades"] = bt_trades
        st.session_state["bt_coverage"] = audit_history_coverage(
            bt_symbols, ["Swing", "Day trading"] if normalized_horizon == "Both" else
            (["Day trading"] if str(normalized_horizon).lower() in {"day trading", "intraday"} else ["Swing"])
        )
        st.session_state["bt_allocated"] = allocated
        st.session_state["bt_summary"], st.session_state["bt_equity"] = summarize_portfolio(
            allocated, equity, starting_capital=float(bt_capital)
        )
        st.session_state["bt_strategy"] = summarize_by_strategy(allocated)
        benchmark_horizons = (
            ["Swing", "Day trading"]
            if normalized_horizon == "Both" else [normalized_horizon]
        )
        st.session_state["bt_benchmark"] = build_benchmark_report(
            bt_symbols, benchmark_horizons
        )
        # Validate signal candidates before portfolio allocation so exposure caps do not
        # distort the independent performance of each strategy.
        train_trades, test_trades = make_walk_forward_split(bt_trades)
        st.session_state["bt_split_diagnostics"] = strategy_split_diagnostics(bt_trades)
        st.session_state["bt_train"] = train_trades
        st.session_state["bt_test"] = test_trades

if "bt_trades" in st.session_state:
    bt_trades = st.session_state["bt_trades"]
    allocated = st.session_state.get("bt_allocated", pd.DataFrame())
    bt_summary = st.session_state.get("bt_summary", {})
    st.subheader("Historical data coverage audit")
    coverage = st.session_state.get("bt_coverage", pd.DataFrame())
    if not coverage.empty:
        st.dataframe(coverage, use_container_width=True, hide_index=True)
        if (coverage["coverage_status"] == "LIMITED HISTORY").any():
            st.warning("Intraday results use a very short provider history window. Treat day-trading statistics as preliminary, not robust out-of-sample evidence.")
    if bt_trades.empty or allocated.empty or not bt_summary:
        st.warning("No portfolio trades were generated. The coverage audit above still shows whether usable historical bars were available.")
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
            if int(strategy["trades"].sum()) == 0 and not allocated.empty:
                st.warning(
                    "Strategy buckets found no matching horizon labels. "
                                       "Inspect the trade log's horizon column; do not interpret these zeros as no signals."
                )

        st.subheader("Buy-and-hold reference benchmark")
        benchmark = st.session_state.get("bt_benchmark", pd.DataFrame())
        if not benchmark.empty:
            st.dataframe(benchmark.round(3), use_container_width=True, hide_index=True)
            st.caption("Price-only reference by symbol and timeframe. Excludes dividends, fees, portfolio sizing, and cash allocation; it is not a portfolio-equivalent benchmark.")
        else:
            st.info("Benchmark data was unavailable for the selected symbols.")

        st.subheader("Independent strategy validation — chronological 70/30")
        st.caption("Each horizon/direction strategy is split separately by whole trading sessions. These are raw signal candidates before portfolio allocation, so capital limits do not bias the strategy comparison.")
        split_diag = st.session_state.get("bt_split_diagnostics", pd.DataFrame())
        if not split_diag.empty:
            st.dataframe(split_diag, use_container_width=True, hide_index=True)
        train = st.session_state.get("bt_train", pd.DataFrame())
        test = st.session_state.get("bt_test", pd.DataFrame())
        if not train.empty or not test.empty:
            split_report = pd.concat([
                summarize_sample_by_strategy(train, "In-sample"),
                summarize_sample_by_strategy(test, "Out-of-sample"),
            ], ignore_index=True)
            st.dataframe(split_report.round(3), use_container_width=True, hide_index=True)
            st.caption("Do not tune parameters against the out-of-sample portion and then treat it as independent.")
        else:
            st.info("No strategy has at least three distinct trading sessions available for a chronological split. The coverage audit explains the data limitation.")


        st.subheader("Stage 3.5 — Individual ticker diagnostics")
        st.caption("Select one strategy to identify which symbols contributed to gains or losses. Signal-return metrics use raw candidates; portfolio P&L includes only trades accepted by the portfolio constraints.")
        strategy_options = [
            "Swing LONG", "Swing SHORT", "Day trading LONG", "Day trading SHORT"
        ]
        selected_ticker_strategy = st.selectbox(
            "Strategy to diagnose", strategy_options, index=0, key="ticker_diagnostic_strategy"
        )
        selected_horizon, selected_direction = selected_ticker_strategy.rsplit(" ", 1)
        ticker_subset = bt_trades[
            (bt_trades["horizon"].astype(str).str.lower() == selected_horizon.lower()) &
            (bt_trades["direction"].astype(str).str.upper() == selected_direction)
        ].copy()
        allocated_subset = allocated[
            (allocated["horizon"].astype(str).str.lower() == selected_horizon.lower()) &
            (allocated["direction"].astype(str).str.upper() == selected_direction)
        ].copy() if not allocated.empty else pd.DataFrame()
        if ticker_subset.empty:
            st.info("No candidate trades exist for this strategy in the selected backtest.")
        else:
            ticker_report = summarize_by_ticker(ticker_subset, allocated_subset)
            st.dataframe(ticker_report.round(3), use_container_width=True, hide_index=True)
            st.caption("Profit factor is gross winning trade returns divided by absolute gross losing trade returns. Compounded trade return assumes sequential reinvestment of each trade return per ticker; it is a diagnostic statistic, not portfolio performance.")
            selected_train, selected_test = make_walk_forward_split(ticker_subset)
            ticker_sample_report = summarize_ticker_samples(selected_train, selected_test)
            st.markdown("**Ticker-by-ticker in-sample versus out-of-sample**")
            if not ticker_sample_report.empty:
                st.dataframe(ticker_sample_report.round(3), use_container_width=True, hide_index=True)
            else:
                st.info("Insufficient distinct sessions to create an independent ticker-level split.")
            st.download_button(
                "Download ticker diagnostics (CSV)",
                ticker_report.to_csv(index=False).encode("utf-8"),
                "ticker_diagnostics.csv", "text/csv", use_container_width=True
            )

        st.subheader("Stage 3.6 — Market regime analysis")
        st.caption("SPY daily trend and volatility conditions are attached using information available on each signal date. These are raw candidate-trade statistics, not portfolio returns.")
        regime_strategy = st.selectbox(
            "Strategy for regime analysis",
            ["Swing LONG", "Swing SHORT", "Day trading LONG", "Day trading SHORT"],
            key="regime_strategy",
        )
        regime_horizon, regime_direction = regime_strategy.rsplit(" ", 1)
        regime_subset = bt_trades[
            (bt_trades["horizon"].astype(str).str.lower() == regime_horizon.lower()) &
            (bt_trades["direction"].astype(str).str.upper() == regime_direction)
        ].copy()
        if regime_subset.empty:
            st.info("No candidate trades exist for this strategy in the selected backtest.")
        elif "market_regime" not in regime_subset.columns or regime_subset["market_regime"].eq("Unavailable").all():
            st.warning("Market regime history could not be attached. Rerun the backtest after the SPY data provider is available.")
        else:
            trend_report = summarize_by_regime(regime_subset)
            st.markdown("**Performance by market trend and volatility**")
            st.dataframe(trend_report.round(3), use_container_width=True, hide_index=True)
            ticker_regime = summarize_by_regime(regime_subset, group_by_ticker=True)
            st.markdown("**Ticker performance inside each market regime**")
            st.dataframe(ticker_regime.round(3), use_container_width=True, hide_index=True)
            st.download_button(
                "Download regime analysis (CSV)",
                ticker_regime.to_csv(index=False).encode("utf-8"),
                "market_regime_diagnostics.csv", "text/csv", use_container_width=True
            )
            st.caption("Trend regime uses SPY close versus EMA50 and EMA200. Volatility regime compares SPY ATR(14) percentage with trailing 252-session 33rd/67th percentiles. Regime labels are descriptive filters, not predictive signals. Intraday trades inherit the latest available daily SPY regime.")


        st.subheader("Stage 3.7 — Strategy parameter research")
        st.caption(
            "Tests a small, predefined grid of minimum signal scores, ATR stop distances, "
            "and reward targets. Configurations are ranked using the chronological training portion only; "
            "the selected configuration's later holdout is reported separately and is not used for ranking."
        )
        with st.expander("Stage 3.7 settings", expanded=False):
            opt_strategy = st.selectbox(
                "Strategy to optimize",
                ["Swing LONG", "Swing SHORT", "Day trading LONG", "Day trading SHORT"],
                key="opt_strategy",
            )
            opt_horizon, opt_direction = opt_strategy.rsplit(" ", 1)
            opt_min_trades = st.selectbox(
                "Minimum in-sample trades per configuration", [10, 20, 30, 40],
                index=1, key="opt_min_trades"
            )
            opt_slippage = st.selectbox(
                "Optimization slippage per side (bps)", [0.0, 5.0, 10.0, 20.0],
                index=1, key="opt_slippage"
            )
            opt_fees = st.selectbox(
                "Optimization fees per side (bps)", [0.0, 0.5, 1.0, 2.0],
                index=0, key="opt_fees"
            )
            run_optimizer = st.button(
                "Run Stage 3.7 parameter research", type="primary", key="run_optimizer"
            )
        if run_optimizer:
            opt_symbols = [s.strip().upper() for s in bt_symbols_text.split(",") if s.strip()]
            if not opt_symbols:
                st.error("Enter ticker symbols in the backtest settings before running optimization.")
            else:
                try:
                    with st.spinner("Testing 27 parameter combinations. This may take several minutes on the first run…"):
                        leaderboard, holdout, opt_train, opt_test = optimize_strategy_parameters(
                            symbols=opt_symbols, horizon=opt_horizon,
                            direction=opt_direction, score_values=[50, 60, 70],
                            stop_values=[1.0, 1.5, 2.0],
                            target_values=[1.5, 2.0, 3.0],
                            min_train_trades=int(opt_min_trades),
                            slippage_bps=float(opt_slippage),
                            fee_bps_per_side=float(opt_fees),
                        )
                    st.session_state["opt_leaderboard"] = leaderboard
                    st.session_state["opt_holdout"] = holdout
                    st.session_state["opt_train"] = opt_train
                    st.session_state["opt_test"] = opt_test
                    st.session_state["opt_strategy_label"] = opt_strategy
                except Exception as exc:
                    st.error(f"Parameter research failed: {exc}")
                    st.exception(exc)

        if "opt_holdout" in st.session_state:
            opt_holdout = st.session_state["opt_holdout"]
            opt_leaderboard = st.session_state.get("opt_leaderboard", pd.DataFrame())
            st.markdown(f"**Latest research: {st.session_state.get('opt_strategy_label', '')}**")
            if opt_leaderboard.empty:
                st.warning("No parameter combinations generated enough usable trades.")
            else:
                st.markdown("**In-sample configuration leaderboard**")
                st.dataframe(opt_leaderboard.round(3), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download in-sample optimization leaderboard (CSV)",
                    opt_leaderboard.to_csv(index=False).encode("utf-8"),
                    "stage3_7_training_leaderboard.csv", "text/csv", use_container_width=True
                )
            if opt_holdout.get("status", "").startswith("SELECTED"):
                st.markdown("**Selected configuration — untouched chronological holdout**")
                a, b, c = st.columns(3)
                a.metric("Holdout trades", opt_holdout.get("trades", 0))
                b.metric(
                    "Holdout profit factor",
                    f"{opt_holdout.get('profit_factor', np.nan):.2f}"
                    if np.isfinite(opt_holdout.get("profit_factor", np.nan)) else "∞"
                )
                c.metric(
                    "Holdout mean trade",
                    f"{opt_holdout.get('mean_trade_pct', np.nan):.3f}%"
                    if np.isfinite(opt_holdout.get("mean_trade_pct", np.nan)) else "N/A"
                )
                st.write(
                    f"Selected settings: minimum score **{opt_holdout['selected_min_score']}**, "
                    f"stop **{opt_holdout['selected_stop_atr']} ATR**, "
                    f"target **{opt_holdout['selected_target_r']}R**."
                )
                st.write(
                    f"Holdout win rate: **{opt_holdout.get('win_rate_pct', np.nan):.1f}%** · "
                    f"Compounded candidate-trade return: **{opt_holdout.get('compounded_return_pct', np.nan):.2f}%** · "
                    f"Max drawdown: **{opt_holdout.get('max_drawdown_pct', np.nan):.2f}%**"
                )
                st.caption(
                    "Important: the holdout is a single historical split, not a live-performance guarantee. "
                    "Do not repeatedly change the grid after viewing holdout results and continue calling that same period untouched. "
                    "The ranking uses in-sample profit factor with mean return and drawdown as tie-breakers; "
                    "this is a research ranking, not an automatically deployable strategy."
                )
            else:
                st.warning(opt_holdout.get("status", "No configuration passed the minimum training sample rule."))


        st.subheader("Stage 3.8 — Walk-forward validation")
        st.caption(
            "Repeatedly selects parameters using only the preceding training window, then evaluates "
            "the selected configuration on the next forward test window. Each fold starts a separate "
            "portfolio simulation; fold returns are not compounded into one continuous account."
        )
        with st.expander("Stage 3.8 validation settings", expanded=False):
            wf_strategy = st.selectbox(
                "Strategy to validate",
                ["Swing LONG", "Swing SHORT", "Day trading LONG", "Day trading SHORT"],
                key="wf_strategy",
            )
            wf_horizon, wf_direction = wf_strategy.rsplit(" ", 1)
            wf_train_months = st.selectbox(
                "Training window (months)", [6, 9, 12, 18, 24],
                index=2, key="wf_train_months"
            )
            wf_test_months = st.selectbox(
                "Forward test window (months)", [1, 2, 3, 6],
                index=2, key="wf_test_months"
            )
            wf_step_months = st.selectbox(
                "Advance between folds (months)", [1, 2, 3, 6],
                index=2, key="wf_step_months"
            )
            wf_min_trades = st.selectbox(
                "Minimum training trades per configuration", [10, 20, 30, 40],
                index=1, key="wf_min_trades"
            )
            wf_run = st.button(
                "Run Stage 3.8 walk-forward validation", type="primary", key="wf_run"
            )
        if wf_run:
            wf_symbols = [s.strip().upper() for s in bt_symbols_text.split(",") if s.strip()]
            if not wf_symbols:
                st.error("Enter ticker symbols in the backtest settings before running validation.")
            elif wf_step_months > wf_test_months:
                st.error("Advance between folds cannot exceed the forward test window; otherwise gaps appear.")
            elif wf_horizon == "Day trading":
                st.warning(
                    "The current Yahoo Finance intraday source provides only a few recent trading days. "
                    "It cannot support multi-month walk-forward validation. Run Swing strategies for now; "
                    "a longer-history intraday data provider is required for valid Day trading folds."
                )
            else:
                try:
                    with st.spinner("Running 27 parameter configurations across rolling historical windows. This can take several minutes…"):
                        wf_folds, wf_summary, wf_selected = run_walk_forward_validation(
                            symbols=wf_symbols, horizon=wf_horizon, direction=wf_direction,
                            score_values=[50, 60, 70], stop_values=[1.0, 1.5, 2.0],
                            target_values=[1.5, 2.0, 3.0],
                            train_months=int(wf_train_months),
                            test_months=int(wf_test_months),
                            step_months=int(wf_step_months),
                            min_train_trades=int(wf_min_trades),
                            slippage_bps=float(bt_slippage),
                            fee_bps_per_side=float(bt_fees),
                            starting_capital=float(bt_capital),
                            risk_per_trade_pct=float(bt_risk),
                            max_positions=int(bt_max_positions),
                            max_position_exposure_pct=float(bt_position_cap),
                        )
                    st.session_state["wf_folds"] = wf_folds
                    st.session_state["wf_summary"] = wf_summary
                    st.session_state["wf_selected"] = wf_selected
                    st.session_state["wf_strategy_label"] = wf_strategy
                except Exception as exc:
                    st.error(f"Walk-forward validation failed: {exc}")
                    st.exception(exc)

        if "wf_folds" in st.session_state:
            wf_folds = st.session_state["wf_folds"]
            wf_summary = st.session_state.get("wf_summary", pd.DataFrame())
            st.markdown(f"**Latest validation: {st.session_state.get('wf_strategy_label', '')}**")
            if wf_folds.empty:
                st.warning(
                    "No walk-forward folds could be formed from the available historical data. "
                    "The selected training window may be too long for the provider's history."
                )
            else:
                st.markdown("**Fold-by-fold results**")
                st.dataframe(wf_folds.round(3), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download walk-forward fold results (CSV)",
                    wf_folds.to_csv(index=False).encode("utf-8"),
                    "stage3_8_walk_forward_folds.csv", "text/csv", use_container_width=True
                )
            if not wf_summary.empty:
                st.markdown("**Aggregate consistency summary**")
                st.dataframe(wf_summary.round(3), use_container_width=True, hide_index=True)
            wf_selected = st.session_state.get("wf_selected", pd.DataFrame())
            if not wf_selected.empty:
                st.markdown("**Forward-test portfolio trades**")
                wf_taken = wf_selected[wf_selected["portfolio_status"].eq("TAKEN")].copy()
                st.dataframe(
                    wf_taken[[c for c in [
                        "walk_forward_fold", "symbol", "entry_time", "exit_time",
                        "selected_min_score", "selected_stop_atr", "selected_target_r",
                        "net_return_pct", "portfolio_pnl"
                    ] if c in wf_taken.columns]].round(3),
                    use_container_width=True, hide_index=True
                )
                st.download_button(
                    "Download forward-test portfolio trades (CSV)",
                    wf_selected.to_csv(index=False).encode("utf-8"),
                    "stage3_8_forward_test_trades.csv", "text/csv", use_container_width=True
                )
            st.caption(
                "Each fold chooses parameters only from its prior training window. Test results are not "
                "used to rank that fold's configuration. The current engine uses overlapping rolling "
                "windows and resets starting capital for each fold; summed fold returns are descriptive, "
                "not a continuous compounded account curve. Small samples, provider history limits, "
                "survivorship bias, market impact, borrow costs and taxes remain limitations."
            )

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


st.divider()
st.header("Stage 3.9 — Signal quality & feature research")
st.caption(
    "Studies which observed signal characteristics coincide with stronger or weaker historical trade outcomes. "
    "These are descriptive associations, not proof that a feature causes returns or will predict future performance."
)
with st.expander("Stage 3.9 research settings", expanded=True):
    f_strategy = st.selectbox(
        "Strategy to analyze",
        ["Swing LONG", "Swing SHORT", "Swing BOTH", "Day trading LONG", "Day trading SHORT", "Day trading BOTH"],
        index=0, key="feature_strategy",
    )
    f_min_group = st.selectbox(
        "Minimum trades required per feature group", [3, 5, 10, 15, 20],
        index=1, key="feature_min_group",
    )
    f_run = st.button("Run Stage 3.9 signal quality research", type="primary", key="feature_run")

if f_run:
    if f_strategy.startswith("Day trading "):
        f_horizon = "Day trading"
        f_direction = f_strategy.replace("Day trading ", "", 1)
    else:
        f_horizon, f_direction = f_strategy.split(" ", 1)
    if f_direction == "BOTH":
        f_direction = "Both"
    f_symbols = [x.strip().upper() for x in bt_symbols_text.split(",") if x.strip()]
    if not f_symbols:
        st.error("Enter ticker symbols in the Historical Backtesting settings first.")
    elif f_horizon == "Day trading":
        st.warning(
            "Intraday history is currently too short for reliable feature research across market conditions. "
            "The report can be generated for available data, but treat it as exploratory only."
        )
    try:
        if f_symbols:
            with st.spinner("Collecting historical trades and measuring signal features…"):
                f_trades = run_backtest(
                    symbols=f_symbols, horizon=f_horizon, direction=f_direction,
                    min_score=0, min_price=2.0, min_dollar_volume=5_000_000,
                    stop_atr=1.5, target_r=2.0,
                    slippage_bps=float(bt_slippage), fee_bps_per_side=float(bt_fees),
                )
                f_trades = apply_market_regimes(f_trades)
                f_report, f_ticker = analyze_signal_features(
                    f_trades, min_group_trades=int(f_min_group)
                )
            st.session_state["feature_trades"] = f_trades
            st.session_state["feature_report"] = f_report
            st.session_state["feature_ticker"] = f_ticker
            st.session_state["feature_strategy_label"] = f_strategy
    except Exception as exc:
        st.error(f"Signal feature research failed: {exc}")
        st.exception(exc)

if "feature_report" in st.session_state:
    f_report = st.session_state["feature_report"]
    f_ticker = st.session_state.get("feature_ticker", pd.DataFrame())
    f_trades = st.session_state.get("feature_trades", pd.DataFrame())
    st.markdown(f"**Latest research: {st.session_state.get('feature_strategy_label', '')}**")
    if f_trades.empty:
        st.warning("No candidate trades were generated for these symbols and settings.")
    else:
        st.metric("Historical candidate trades analyzed", len(f_trades))
        st.caption(
            "The research reruns the selected strategy with a zero minimum score to expose score buckets. "
            "It uses a fixed 1.5 ATR stop and 2.0R target for this diagnostic run; these are not optimized settings."
        )
    if f_report.empty:
        st.info("No feature groups met the minimum trade-count requirement.")
    else:
        st.markdown("**Performance by signal feature**")
        st.dataframe(f_report.round(3), use_container_width=True, hide_index=True)
        st.download_button(
            "Download feature research report (CSV)",
            f_report.to_csv(index=False).encode("utf-8"),
            "stage3_9_feature_research.csv", "text/csv", use_container_width=True
        )
    if not f_ticker.empty:
        st.markdown("**Ticker-level feature results**")
        st.dataframe(f_ticker.round(3), use_container_width=True, hide_index=True)
    if not f_trades.empty:
        st.markdown("**Underlying candidate-trade dataset**")
        st.dataframe(
            f_trades[[c for c in ["symbol","signal_time","direction","signal","signal_score","rsi","rvol",
                                   "market_regime","exit_reason","net_return_pct"] if c in f_trades]].round(3),
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "Download underlying feature dataset (CSV)",
            f_trades.to_csv(index=False).encode("utf-8"),
            "stage3_9_candidate_trades.csv", "text/csv", use_container_width=True
        )
    st.caption(
        "Important limitations: groups are descriptive and may have small samples; overlapping signals, "
        "ticker selection, market-regime classification, and fixed exit assumptions affect results. "
        "Feature buckets must be evaluated on later untouched periods before being used as filters. "
        "This stage does not alter scanner signals, choose a live strategy, or execute orders."
    )
