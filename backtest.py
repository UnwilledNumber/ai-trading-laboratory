from __future__ import annotations

from functools import lru_cache


import numpy as np
import pandas as pd
import yfinance as yf

from scanner import add_indicators, _score_daily, _score_intraday

@lru_cache(maxsize=64)
def _download_history(symbol: str, horizon: str) -> pd.DataFrame:
    if horizon == "Swing":
        raw = yf.download(symbol, period="5y", interval="1d", auto_adjust=True,
                          progress=False, threads=False)
        intraday = False
    else:
        # Yahoo Finance generally limits 5-minute history to approximately 5 trading days.
        raw = yf.download(symbol, period="5d", interval="5m", auto_adjust=True,
                          progress=False, threads=False)
        intraday = True
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in raw.columns for c in needed):
        return pd.DataFrame()
    frame = raw[needed].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if frame.empty:
        return frame
    return add_indicators(frame, intraday=intraday)



@lru_cache(maxsize=2)
def _download_market_regime_history() -> pd.DataFrame:
    """Download SPY daily bars and classify market conditions without look-ahead."""
    raw = yf.download("SPY", period="5y", interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    required = {"High", "Low", "Close"}
    if not required.issubset(raw.columns):
        return pd.DataFrame()
    market = raw[["High", "Low", "Close"]].copy().dropna()
    close = pd.to_numeric(market["Close"], errors="coerce")
    market["EMA50"] = close.ewm(span=50, adjust=False).mean()
    market["EMA200"] = close.ewm(span=200, adjust=False).mean()
    previous = close.shift(1)
    tr = pd.concat([market["High"] - market["Low"],
                    (market["High"] - previous).abs(),
                    (market["Low"] - previous).abs()], axis=1).max(axis=1)
    market["ATR14_PCT"] = 100 * tr.ewm(alpha=1/14, adjust=False).mean() / close
    q_low = market["ATR14_PCT"].rolling(252, min_periods=60).quantile(0.33)
    q_high = market["ATR14_PCT"].rolling(252, min_periods=60).quantile(0.67)
    market["volatility_regime"] = np.select(
        [market["ATR14_PCT"] <= q_low, market["ATR14_PCT"] >= q_high],
        ["Low volatility", "High volatility"], default="Normal volatility"
    )
    market["trend_regime"] = np.select(
        [(close > market["EMA50"]) & (market["EMA50"] > market["EMA200"]),
         (close < market["EMA50"]) & (market["EMA50"] < market["EMA200"])],
        ["Bullish", "Bearish"], default="Sideways / mixed"
    )
    market["market_regime"] = market["trend_regime"] + " · " + market["volatility_regime"]
    market.index = pd.to_datetime(market.index, errors="coerce", utc=True).tz_convert(None).normalize()
    return market[["trend_regime", "volatility_regime", "market_regime", "ATR14_PCT"]].loc[lambda x: ~x.index.isna()]


def apply_market_regimes(trades: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest known SPY regime on or before each trade signal date."""
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    market = _download_market_regime_history()
    if market.empty or "signal_time" not in out:
        out["trend_regime"] = "Unavailable"
        out["volatility_regime"] = "Unavailable"
        out["market_regime"] = "Unavailable"
        out["market_atr_pct"] = np.nan
        return out
    signal_dates = pd.to_datetime(out["signal_time"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    left = pd.DataFrame({"_row": np.arange(len(out)), "_date": signal_dates})
    right = market.reset_index()
    right = right.rename(columns={right.columns[0]: "_date"})
    merged = pd.merge_asof(left.sort_values("_date"), right.sort_values("_date"),
                           on="_date", direction="backward")
    merged = merged.sort_values("_row")
    out["trend_regime"] = merged["trend_regime"].fillna("Unavailable").to_numpy()
    out["volatility_regime"] = merged["volatility_regime"].fillna("Unavailable").to_numpy()
    out["market_regime"] = merged["market_regime"].fillna("Unavailable").to_numpy()
    out["market_atr_pct"] = pd.to_numeric(merged["ATR14_PCT"], errors="coerce").to_numpy()
    return out


def summarize_by_regime(trades: pd.DataFrame, group_by_ticker: bool = False) -> pd.DataFrame:
    """Summarize raw trade-candidate performance by SPY market regime."""
    columns = ["market_regime", "symbol", "trades", "win_rate_pct", "profit_factor",
               "mean_trade_pct", "average_win_pct", "average_loss_pct"]
    if trades.empty or "market_regime" not in trades.columns:
        return pd.DataFrame(columns=columns)
    keys = ["market_regime"] + (["symbol"] if group_by_ticker else [])
    rows = []
    for key, group in trades.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        returns = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna()
        if returns.empty:
            continue
        wins, losses = returns[returns > 0], returns[returns < 0]
        gp, gl = float(wins.sum()), float(abs(losses.sum()))
        pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
        row = {"market_regime": key[0], "symbol": key[1] if group_by_ticker else "ALL",
               "trades": int(len(returns)), "win_rate_pct": float((returns > 0).mean()*100),
               "profit_factor": pf, "mean_trade_pct": float(returns.mean()),
               "average_win_pct": float(wins.mean()) if len(wins) else np.nan,
               "average_loss_pct": float(losses.mean()) if len(losses) else np.nan}
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["market_regime", "symbol"]).reset_index(drop=True)


def audit_history_coverage(symbols: list[str], horizons: list[str] | None = None) -> pd.DataFrame:
    """Report actual historical coverage and warmup-eligible bars per symbol/horizon."""
    horizons = horizons or ["Swing", "Day trading"]
    rows = []
    unique_symbols = list(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))
    for horizon in horizons:
        intraday = str(horizon).strip().lower() in {"day trading", "intraday", "day-trading"}
        interval = "5m" if intraday else "1d"
        requested_period = "5d" if intraday else "5y"
        warmup = 35 if intraday else 210
        for symbol in unique_symbols:
            try:
                bars = _download_history(symbol, "Day trading" if intraday else "Swing")
                count = int(len(bars))
                sessions = (int(pd.Index(pd.to_datetime(bars.index).date).nunique()) if count else 0)
                eligible = max(0, count - warmup)
                start = str(bars.index.min()) if count else ""
                end = str(bars.index.max()) if count else ""
                if not count:
                    status = "NO DATA"
                    note = "Provider returned no usable bars."
                elif eligible < 3:
                    status = "INSUFFICIENT BARS"
                    note = f"{count} bars; {warmup} required for indicator warmup."
                elif intraday:
                    status = "LIMITED HISTORY"
                    note = "Yahoo 5-minute feed is requested at 5d; too short for robust multi-regime validation."
                else:
                    status = "USABLE"
                    note = "Daily history available for swing research; check start/end dates for gaps."
                rows.append({
                    "symbol": symbol, "horizon": "Day trading" if intraday else "Swing",
                    "interval": interval, "requested_period": requested_period,
                    "bars": count, "sessions": sessions, "warmup_bars": warmup,
                    "eligible_bars": eligible, "first_bar": start, "last_bar": end,
                    "coverage_status": status, "coverage_note": note,
                })
            except Exception as exc:
                rows.append({
                    "symbol": symbol, "horizon": "Day trading" if intraday else "Swing",
                    "interval": interval, "requested_period": requested_period,
                    "bars": 0, "sessions": 0, "warmup_bars": warmup,
                    "eligible_bars": 0, "first_bar": "", "last_bar": "",
                    "coverage_status": "DOWNLOAD ERROR", "coverage_note": str(exc)[:180],
                })
    return pd.DataFrame(rows)


def _get_signal(history: pd.DataFrame, i: int, horizon: str, min_price: float,
                min_dollar_volume: float) -> dict | None:
    sample = history.iloc[:i+1]
    if horizon == "Swing":
        return _score_daily(sample, "", min_price, min_dollar_volume)
    return _score_intraday(sample, "", min_price)

def backtest_symbol(
    symbol: str,
    horizon: str = "Swing",
    direction: str = "Both",
    min_score: int = 60,
    min_price: float = 2.0,
    min_dollar_volume: float = 5_000_000,
    stop_atr: float = 1.5,
    target_r: float = 2.0,
    max_holding_bars: int | None = None,
    slippage_bps: float = 5.0,
    fee_bps_per_side: float = 0.0,
) -> pd.DataFrame:
    bars = _download_history(symbol, horizon)
    warmup = 210 if horizon == "Swing" else 35
    if len(bars) <= warmup + 3:
        return pd.DataFrame()
    if max_holding_bars is None:
        max_holding_bars = 10 if horizon == "Swing" else 24

    trades = []
    i = warmup
    while i < len(bars) - 2:
        sig = _get_signal(bars, i, horizon, min_price, min_dollar_volume)
        if not sig or int(sig.get("score", 0)) < min_score:
            i += 1
            continue
        side = sig["direction"]
        if direction != "Both" and side != direction.upper():
            i += 1
            continue

        entry_i = i + 1
        entry_bar = bars.iloc[entry_i]
        raw_entry = float(entry_bar["Open"])
        atr = float(bars.iloc[i]["ATR14"])
        if not np.isfinite(raw_entry) or not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        risk = stop_atr * atr
        if side == "LONG":
            stop = raw_entry - risk
            target = raw_entry + target_r * risk
            entry = raw_entry * (1 + slippage_bps / 10000)
        else:
            stop = raw_entry + risk
            target = raw_entry - target_r * risk
            entry = raw_entry * (1 - slippage_bps / 10000)

        exit_i = min(entry_i + max_holding_bars, len(bars) - 1)
        exit_price = float(bars.iloc[exit_i]["Close"])
        exit_reason = "TIME EXIT"
        actual_exit_i = exit_i

        for j in range(entry_i, exit_i + 1):
            bar = bars.iloc[j]
            high, low = float(bar["High"]), float(bar["Low"])
            if side == "LONG":
                stop_hit, target_hit = low <= stop, high >= target
                # Conservative assumption: if both levels trade in one bar, stop fills first.
                if stop_hit:
                    exit_price, exit_reason, actual_exit_i = stop, "STOP", j
                    break
                if target_hit:
                    exit_price, exit_reason, actual_exit_i = target, "TARGET", j
                    break
            else:
                stop_hit, target_hit = high >= stop, low <= target
                if stop_hit:
                    exit_price, exit_reason, actual_exit_i = stop, "STOP", j
                    break
                if target_hit:
                    exit_price, exit_reason, actual_exit_i = target, "TARGET", j
                    break

        if side == "LONG":
            exit_fill = exit_price * (1 - slippage_bps / 10000)
            gross_return = (exit_fill - entry) / entry
        else:
            exit_fill = exit_price * (1 + slippage_bps / 10000)
            gross_return = (entry - exit_fill) / entry

        total_fee_rate = 2 * fee_bps_per_side / 10000
        net_return = gross_return - total_fee_rate
        trades.append({
            "symbol": symbol.upper(),
            "horizon": horizon,
            "direction": side,
            "signal_time": bars.index[i],
            "entry_time": bars.index[entry_i],
            "exit_time": bars.index[actual_exit_i],
            "entry_price": entry,
            "exit_price": exit_fill,
            "stop_price": stop,
            "target_price": target,
            "signal": sig.get("signal", ""),
            "signal_score": int(sig.get("score", 0)),
            "rsi": float(sig.get("rsi14", np.nan)),
            "rvol": float(sig.get("rvol", np.nan)),
            "exit_reason": exit_reason,
            "gross_return_pct": gross_return * 100,
            "net_return_pct": net_return * 100,
        })
        # One position at a time per symbol; no overlapping trades.
        i = actual_exit_i + 1

    return pd.DataFrame(trades)

def summarize_trades(trades: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if trades.empty:
        return {}, pd.DataFrame()
    t = trades.sort_values("exit_time").copy()
    returns = t["net_return_pct"].astype(float) / 100
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    equity = (1 + returns).cumprod()
    peaks = equity.cummax()
    drawdown = equity / peaks - 1
    summary = {
        "trades": int(len(t)),
        "win_rate_pct": float((returns > 0).mean() * 100),
        "profit_factor": float(profit_factor),
        "net_return_pct_compounded": float((equity.iloc[-1] - 1) * 100),
        "average_trade_pct": float(returns.mean() * 100),
        "average_win_pct": float(wins.mean() * 100) if len(wins) else 0.0,
        "average_loss_pct": float(losses.mean() * 100) if len(losses) else 0.0,
        "expectancy_pct": float(returns.mean() * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "gross_profit_pct": gross_profit * 100,
        "gross_loss_pct": gross_loss * 100,
    }
    t["equity_index"] = equity.values
    t["drawdown_pct"] = drawdown.values * 100
    return summary, t


def run_backtest(
    symbols: list[str],
    horizon: str = "Swing",
    direction: str = "Both",
    min_score: int = 60,
    min_price: float = 2.0,
    min_dollar_volume: float = 5_000_000,
    stop_atr: float = 1.5,
    target_r: float = 2.0,
    slippage_bps: float = 5.0,
    fee_bps_per_side: float = 0.0,
) -> pd.DataFrame:
    horizon_key = str(horizon).strip().lower()
    if horizon_key in {"both", "both (daily + intraday)", "both (daily and intraday)"}:
        horizons = ["Swing", "Day trading"]
    elif horizon_key in {"swing", "swing trading", "daily"}:
        horizons = ["Swing"]
    elif horizon_key in {"day trading", "intraday", "day-trading"}:
        horizons = ["Day trading"]
    else:
        raise ValueError(
            f"Unsupported backtest horizon: {horizon!r}. "
            "Use Both, Swing, or Day trading."
        )
    frames = []
    for selected_horizon in horizons:
        for symbol in list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip())):
            try:
                result = backtest_symbol(
                    symbol, horizon=selected_horizon, direction=direction,
                    min_score=min_score, min_price=min_price,
                    min_dollar_volume=min_dollar_volume,
                    stop_atr=stop_atr, target_r=target_r,
                    slippage_bps=slippage_bps, fee_bps_per_side=fee_bps_per_side,
                )
                if not result.empty:
                    result = result.copy()
                    # Daily bars may be timezone-naive while intraday bars carry an
                    # exchange timezone. Canonicalize before concatenating/sorting.
                    for time_col in ("signal_time", "entry_time", "exit_time"):
                        if time_col in result.columns:
                            result[time_col] = (
                                pd.to_datetime(result[time_col], errors="coerce", utc=True)
                                .dt.tz_convert(None)
                            )
                    frames.append(result)
            except Exception as exc:
                print(f"Backtest skipped {symbol} ({selected_horizon}): {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def simulate_portfolio(
    trades: pd.DataFrame,
    starting_capital: float = 10_000.0,
    risk_per_trade_pct: float = 0.5,
    max_positions: int = 5,
    max_position_exposure_pct: float = 25.0,
    max_gross_exposure_pct: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event-driven portfolio allocation using realized equity and exposure caps.

    Candidate trades are entered at their recorded next-bar open. Existing open
    positions reserve gross exposure until their recorded exit. P&L is realized
    at exit; this is not intrabar mark-to-market portfolio accounting.
    """
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    candidates = trades.sort_values(["entry_time", "symbol"]).copy()
    active = []
    pending_exits = []
    realized_equity = float(starting_capital)
    accepted = []
    skipped = []
    events = []

    for _, trade in candidates.iterrows():
        entry_time = pd.Timestamp(trade["entry_time"])
        # Realize completed trades before sizing the next entry.
        still_pending = []
        for exit_time, pnl, pos in pending_exits:
            if exit_time <= entry_time:
                realized_equity += float(pnl)
            else:
                still_pending.append((exit_time, pnl, pos))
        pending_exits = still_pending
        active = [pos for pos in active if pos["exit_time"] > entry_time]
        if any(pos["symbol"] == str(trade["symbol"]) for pos in active):
            skipped.append({**trade.to_dict(), "portfolio_status": "SKIPPED_DUPLICATE_SYMBOL"})
            continue
        if len(active) >= int(max_positions):
            skipped.append({**trade.to_dict(), "portfolio_status": "SKIPPED_MAX_POSITIONS"})
            continue

        active_notional = sum(pos["notional"] for pos in active)
        gross_cap = realized_equity * max_gross_exposure_pct / 100.0
        remaining_gross = max(0.0, gross_cap - active_notional)
        entry = float(trade["entry_price"])
        stop = float(trade["stop_price"])
        stop_fraction = abs(entry - stop) / entry if entry else np.nan
        if not np.isfinite(stop_fraction) or stop_fraction <= 0:
            skipped.append({**trade.to_dict(), "portfolio_status": "SKIPPED_INVALID_RISK"})
            continue

        risk_budget = realized_equity * risk_per_trade_pct / 100.0
        risk_sized_notional = risk_budget / stop_fraction
        position_cap = realized_equity * max_position_exposure_pct / 100.0
        notional = min(risk_sized_notional, position_cap, remaining_gross)
        if notional <= 0:
            skipped.append({**trade.to_dict(), "portfolio_status": "SKIPPED_EXPOSURE_CAP"})
            continue

        item = trade.to_dict()
        item.update({
            "portfolio_status": "TAKEN",
            "portfolio_equity_at_entry": realized_equity,
            "risk_budget": risk_budget,
            "position_notional": notional,
            "position_weight_pct": notional / realized_equity * 100.0,
            "allocated_risk_pct_equity": notional * stop_fraction / realized_equity * 100.0,
            "portfolio_pnl": notional * float(trade["net_return_pct"]) / 100.0,
        })
        accepted.append(item)
        active.append({
            "symbol": str(trade["symbol"]),
            "exit_time": pd.Timestamp(trade["exit_time"]),
            "notional": notional,
        })
        exit_event = (pd.Timestamp(trade["exit_time"]), item["portfolio_pnl"], item)
        events.append(exit_event)
        pending_exits.append(exit_event)

    if not accepted:
        return pd.DataFrame(skipped), pd.DataFrame()

    events.sort(key=lambda x: x[0])
    equity = float(starting_capital)
    peak = equity
    curve = []
    for exit_time, pnl, item in events:
        equity += float(pnl)
        peak = max(peak, equity)
        curve.append({
            "exit_time": exit_time,
            "equity": equity,
            "equity_index": equity / starting_capital,
            "drawdown_pct": (equity / peak - 1.0) * 100.0,
            "symbol": item["symbol"],
            "trade_pnl": pnl,
        })

    accepted_df = pd.DataFrame(accepted).sort_values("entry_time").reset_index(drop=True)
    if skipped:
        skipped_df = pd.DataFrame(skipped)
        accepted_df = pd.concat([accepted_df, skipped_df], ignore_index=True, sort=False)
    return accepted_df, pd.DataFrame(curve)


def _trade_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
                "average_trade_pct": np.nan}
    r = frame["net_return_pct"].astype(float)
    wins, losses = r[r > 0], r[r < 0]
    gp, gl = float(wins.sum()), float(abs(losses.sum()))
    return {
        "trades": int(len(frame)),
        "win_rate_pct": float((r > 0).mean() * 100),
        "profit_factor": gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0),
        "average_trade_pct": float(r.mean()),
        "average_win_pct": float(wins.mean()) if len(wins) else np.nan,
        "average_loss_pct": float(losses.mean()) if len(losses) else np.nan,
    }


def summarize_portfolio(
    allocated_trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    starting_capital: float = 10_000.0,
) -> tuple[dict, pd.DataFrame]:
    if allocated_trades.empty or equity_curve.empty:
        return {}, pd.DataFrame()
    taken = allocated_trades[allocated_trades["portfolio_status"] == "TAKEN"].copy()
    pnl = taken["portfolio_pnl"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit, gross_loss = float(wins.sum()), float(abs(losses.sum()))
    final_equity = float(equity_curve["equity"].iloc[-1])
    summary = {
        "trades": int(len(taken)),
        "skipped_trades": int((allocated_trades["portfolio_status"] != "TAKEN").sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "net_return_pct_compounded": (final_equity / starting_capital - 1) * 100,
        "expectancy_dollars": float(pnl.mean()),
        "expectancy_pct_equity": float((pnl / taken["portfolio_equity_at_entry"]).mean() * 100),
        "average_win_dollars": float(wins.mean()) if len(wins) else 0.0,
        "average_loss_dollars": float(losses.mean()) if len(losses) else 0.0,
        "max_drawdown_pct": float(equity_curve["drawdown_pct"].min()),
        "starting_capital": float(starting_capital),
        "ending_equity": final_equity,
    }
    return summary, equity_curve



def summarize_by_strategy(allocated_trades: pd.DataFrame) -> pd.DataFrame:
    """Report the four distinct horizon/direction combinations, including zero-trade rows."""
    columns = ["strategy", "trades", "win_rate_pct", "profit_factor",
               "average_trade_pct", "portfolio_pnl", "average_pnl_per_trade"]
    if allocated_trades.empty or "portfolio_status" not in allocated_trades:
        return pd.DataFrame(columns=columns)
    taken = allocated_trades[allocated_trades["portfolio_status"] == "TAKEN"].copy()
    expected = [
        ("Swing", "LONG"), ("Swing", "SHORT"),
        ("Day trading", "LONG"), ("Day trading", "SHORT"),
    ]
    rows = []
    for horizon, direction in expected:
        horizon_values = taken["horizon"].astype(str).str.strip().str.lower()
        direction_values = taken["direction"].astype(str).str.strip().str.upper()
        horizon_aliases = {"swing": {"swing", "swing trading", "daily"},
                           "day trading": {"day trading", "intraday", "day-trading"}}
        group = taken[
            horizon_values.isin(horizon_aliases[horizon.lower()]) &
            (direction_values == direction)
        ]
        if group.empty:
            rows.append({
                "strategy": f"{horizon} {direction}", "trades": 0,
                "win_rate_pct": np.nan, "profit_factor": np.nan,
                "average_trade_pct": np.nan, "portfolio_pnl": 0.0,
                "average_pnl_per_trade": np.nan,
            })
            continue
        r = group["net_return_pct"].astype(float)
        wins, losses = r[r > 0], r[r < 0]
        gp, gl = float(wins.sum()), float(abs(losses.sum()))
        pnl = group["portfolio_pnl"].astype(float)
        rows.append({
            "strategy": f"{horizon} {direction}",
            "trades": int(len(group)),
            "win_rate_pct": float((r > 0).mean() * 100),
            "profit_factor": gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0),
            "average_trade_pct": float(r.mean()),
            "portfolio_pnl": float(pnl.sum()),
            "average_pnl_per_trade": float(pnl.mean()),
        })
    return pd.DataFrame(rows, columns=columns)


def build_benchmark_report(symbols: list[str], horizons: list[str]) -> pd.DataFrame:
    """Reference buy-and-hold returns for each symbol and data horizon.

    This is a simple price-only benchmark, not a portfolio-equivalent comparison:
    it excludes dividends, trading costs, position sizing, and cash allocation.
    """
    rows = []
    for horizon in horizons:
        for symbol in list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip())):
            try:
                bars = _download_history(symbol, horizon)
                if bars.empty or len(bars) < 2:
                    continue
                close = pd.to_numeric(bars["Close"], errors="coerce").dropna()
                if len(close) < 2 or float(close.iloc[0]) <= 0:
                    continue
                equity = close / float(close.iloc[0])
                drawdown = equity / equity.cummax() - 1.0
                rows.append({
                    "symbol": symbol,
                    "horizon": horizon,
                    "bars": int(len(close)),
                    "start": str(close.index[0]),
                    "end": str(close.index[-1]),
                    "buy_hold_return_pct": float((equity.iloc[-1] - 1) * 100),
                    "buy_hold_max_drawdown_pct": float(drawdown.min() * 100),
                })
            except Exception as exc:
                print(f"Benchmark skipped {symbol} ({horizon}): {exc}")
    return pd.DataFrame(rows)


def summarize_sample_by_strategy(trades: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    """Trade-level validation for all four horizon/direction buckets, including zeros."""
    cols = ["sample", "strategy", "trades", "sessions", "win_rate_pct",
            "profit_factor", "mean_trade_pct", "expectancy_pct"]
    rows = []
    for horizon, direction in [
        ("Swing", "LONG"), ("Swing", "SHORT"),
        ("Day trading", "LONG"), ("Day trading", "SHORT"),
    ]:
        group = trades[
            (trades.get("horizon", pd.Series(index=trades.index, dtype=str)).astype(str) == horizon) &
            (trades.get("direction", pd.Series(index=trades.index, dtype=str)).astype(str).str.upper() == direction)
        ] if not trades.empty else pd.DataFrame()
        if group.empty:
            rows.append({"sample": sample_name, "strategy": f"{horizon} {direction}",
                         "trades": 0, "sessions": 0, "win_rate_pct": np.nan,
                         "profit_factor": np.nan, "mean_trade_pct": np.nan, "expectancy_pct": np.nan})
            continue
        r = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna()
        wins, losses = r[r > 0], r[r < 0]
        gp, gl = float(wins.sum()), float(abs(losses.sum()))
        dates = pd.to_datetime(group["entry_time"], errors="coerce", utc=True).dt.tz_convert(None).dt.date
        rows.append({
            "sample": sample_name, "strategy": f"{horizon} {direction}",
            "trades": int(len(r)), "sessions": int(dates.nunique()),
            "win_rate_pct": float((r > 0).mean() * 100),
            "profit_factor": gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0),
            "mean_trade_pct": float(r.mean()), "expectancy_pct": float(r.mean()),
        })
    return pd.DataFrame(rows, columns=cols)


def strategy_split_diagnostics(trades: pd.DataFrame, train_fraction: float = 0.70) -> pd.DataFrame:
    """Show whether each of the four strategies has enough distinct sessions to split."""
    rows = []
    for horizon, direction in [
        ("Swing", "LONG"), ("Swing", "SHORT"),
        ("Day trading", "LONG"), ("Day trading", "SHORT"),
    ]:
        group = trades[(trades["horizon"].astype(str) == horizon) &
                       (trades["direction"].astype(str).str.upper() == direction)] if not trades.empty else pd.DataFrame()
        if group.empty:
            rows.append({"strategy": f"{horizon} {direction}", "candidate_trades": 0,
                         "distinct_sessions": 0, "split_cutoff_session": "",
                         "validation_status": "NO TRADES"})
            continue
        dates = pd.to_datetime(group["entry_time"], errors="coerce", utc=True).dt.tz_convert(None).dt.date
        sessions = sorted(d for d in dates.dropna().unique())
        if len(sessions) < 3:
            status, cutoff = "INSUFFICIENT SESSIONS (<3)", ""
        else:
            idx = max(1, min(len(sessions)-1, int(len(sessions)*train_fraction)))
            cutoff = str(sessions[idx])
            status = "SPLITTABLE"
        rows.append({"strategy": f"{horizon} {direction}",
                     "candidate_trades": int(len(group)), "distinct_sessions": len(sessions),
                     "split_cutoff_session": cutoff, "validation_status": status})
    return pd.DataFrame(rows)



def summarize_by_ticker(
    trades: pd.DataFrame,
    allocated_trades: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ticker-level signal diagnostics plus realized portfolio contribution.

    Trade-return statistics use raw candidate trades, before portfolio constraints.
    Portfolio P&L uses only rows marked TAKEN in allocated_trades.
    """
    columns = [
        "symbol", "candidate_trades", "win_rate_pct", "profit_factor",
        "mean_trade_pct", "compounded_trade_return_pct",
        "average_win_pct", "average_loss_pct", "gross_profit_pct",
        "gross_loss_pct", "taken_trades", "portfolio_pnl",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    taken_by_symbol = {}
    if allocated_trades is not None and not allocated_trades.empty:
        if {"symbol", "portfolio_status", "portfolio_pnl"}.issubset(allocated_trades.columns):
            accepted = allocated_trades[allocated_trades["portfolio_status"] == "TAKEN"]
            for symbol, group in accepted.groupby(accepted["symbol"].astype(str).str.upper()):
                taken_by_symbol[symbol] = {
                    "taken_trades": int(len(group)),
                    "portfolio_pnl": float(pd.to_numeric(group["portfolio_pnl"], errors="coerce").fillna(0).sum()),
                }

    rows = []
    for symbol, group in trades.groupby(trades["symbol"].astype(str).str.upper()):
        returns = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna()
        wins, losses = returns[returns > 0], returns[returns < 0]
        gross_profit = float(wins.sum())
        gross_loss = float(abs(losses.sum()))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        )
        compounded = float(((1 + returns / 100).prod() - 1) * 100) if len(returns) else 0.0
        portfolio = taken_by_symbol.get(symbol, {"taken_trades": 0, "portfolio_pnl": 0.0})
        rows.append({
            "symbol": symbol,
            "candidate_trades": int(len(returns)),
            "win_rate_pct": float((returns > 0).mean() * 100) if len(returns) else np.nan,
            "profit_factor": float(profit_factor),
            "mean_trade_pct": float(returns.mean()) if len(returns) else np.nan,
            "compounded_trade_return_pct": compounded,
            "average_win_pct": float(wins.mean()) if len(wins) else np.nan,
            "average_loss_pct": float(losses.mean()) if len(losses) else np.nan,
            "gross_profit_pct": gross_profit,
            "gross_loss_pct": gross_loss,
            "taken_trades": int(portfolio["taken_trades"]),
            "portfolio_pnl": float(portfolio["portfolio_pnl"]),
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["portfolio_pnl", "mean_trade_pct"], ascending=[False, False]
    ).reset_index(drop=True)


def summarize_ticker_samples(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Per-ticker in-sample and out-of-sample diagnostics for a chosen strategy."""
    columns = [
        "symbol", "sample", "trades", "win_rate_pct", "profit_factor",
        "mean_trade_pct", "average_win_pct", "average_loss_pct",
        "compounded_trade_return_pct",
    ]
    rows = []
    for sample_name, frame in (("In-sample", train), ("Out-of-sample", test)):
        if frame.empty:
            continue
        for symbol, group in frame.groupby(frame["symbol"].astype(str).str.upper()):
            returns = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna()
            if returns.empty:
                continue
            wins, losses = returns[returns > 0], returns[returns < 0]
            gp, gl = float(wins.sum()), float(abs(losses.sum()))
            pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
            rows.append({
                "symbol": symbol,
                "sample": sample_name,
                "trades": int(len(returns)),
                "win_rate_pct": float((returns > 0).mean() * 100),
                "profit_factor": float(pf),
                "mean_trade_pct": float(returns.mean()),
                "average_win_pct": float(wins.mean()) if len(wins) else np.nan,
                "average_loss_pct": float(losses.mean()) if len(losses) else np.nan,
                "compounded_trade_return_pct": float(((1 + returns / 100).prod() - 1) * 100),
            })
    return pd.DataFrame(rows, columns=columns)

def make_walk_forward_split(trades: pd.DataFrame, train_fraction: float = 0.70):
    """Split each strategy independently by whole trading sessions, not mixed horizons."""
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    ordered = trades.copy()
    normalized = pd.to_datetime(ordered["entry_time"], errors="coerce", utc=True).dt.tz_convert(None)
    ordered = ordered.loc[normalized.notna()].copy()
    ordered["_session_date"] = normalized.loc[normalized.index].dt.date
    train_parts, test_parts = [], []
    for horizon, direction in [("Swing", "LONG"), ("Swing", "SHORT"),
                               ("Day trading", "LONG"), ("Day trading", "SHORT")]:
        group = ordered[(ordered["horizon"].astype(str) == horizon) &
                        (ordered["direction"].astype(str).str.upper() == direction)].copy()
        sessions = sorted(group["_session_date"].dropna().unique())
        # Fewer than three sessions cannot support a minimally meaningful chronological split.
        if len(sessions) < 3:
            continue
        split_index = max(1, min(len(sessions)-1, int(len(sessions)*train_fraction)))
        cutoff = sessions[split_index]
        train_parts.append(group[group["_session_date"] < cutoff])
        test_parts.append(group[group["_session_date"] >= cutoff])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    for frame in (train, test):
        if not frame.empty and "_session_date" in frame.columns:
            frame.drop(columns=["_session_date"], inplace=True)
    return train, test


def summarize_trades(trades: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Backward-compatible trade-level summary; portfolio results use summarize_portfolio."""
    if trades.empty:
        return {}, pd.DataFrame()
    t = trades.sort_values("exit_time").copy()
    returns = t["net_return_pct"].astype(float) / 100
    wins, losses = returns[returns > 0], returns[returns < 0]
    gross_profit, gross_loss = float(wins.sum()), float(abs(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    equity = (1 + returns).cumprod()
    peaks = equity.cummax()
    drawdown = equity / peaks - 1
    summary = {
        "trades": int(len(t)),
        "win_rate_pct": float((returns > 0).mean() * 100),
        "profit_factor": float(profit_factor),
        "net_return_pct_compounded": float((equity.iloc[-1] - 1) * 100),
        "average_trade_pct": float(returns.mean() * 100),
        "average_win_pct": float(wins.mean() * 100) if len(wins) else 0.0,
        "average_loss_pct": float(losses.mean() * 100) if len(losses) else 0.0,
        "expectancy_pct": float(returns.mean() * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
    }
    t["equity_index"] = equity.values
    t["drawdown_pct"] = drawdown.values * 100
    return summary, t


def _optimization_metrics(trades: pd.DataFrame) -> dict:
    """Compact trade-level metrics for parameter research."""
    if trades.empty:
        return {"trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
                "mean_trade_pct": np.nan, "compounded_return_pct": np.nan,
                "max_drawdown_pct": np.nan}
    ordered = trades.sort_values("exit_time").copy()
    r = pd.to_numeric(ordered["net_return_pct"], errors="coerce").dropna()
    if r.empty:
        return {"trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
                "mean_trade_pct": np.nan, "compounded_return_pct": np.nan,
                "max_drawdown_pct": np.nan}
    wins, losses = r[r > 0], r[r < 0]
    gp, gl = float(wins.sum()), float(abs(losses.sum()))
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    curve = (1 + r / 100).cumprod()
    dd = curve / curve.cummax() - 1
    return {
        "trades": int(len(r)),
        "win_rate_pct": float((r > 0).mean() * 100),
        "profit_factor": float(pf),
        "mean_trade_pct": float(r.mean()),
        "compounded_return_pct": float((curve.iloc[-1] - 1) * 100),
        "max_drawdown_pct": float(dd.min() * 100),
    }


def optimize_strategy_parameters(
    symbols: list[str],
    horizon: str = "Swing",
    direction: str = "LONG",
    score_values: list[int] | None = None,
    stop_values: list[float] | None = None,
    target_values: list[float] | None = None,
    min_train_trades: int = 20,
    slippage_bps: float = 5.0,
    fee_bps_per_side: float = 0.0,
    train_fraction: float = 0.70,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Grid-search parameters on chronological in-sample trades only.

    Returns an in-sample-only leaderboard, the selected configuration's untouched
    chronological holdout metrics, and its train/test trades. The holdout is not
    used to rank configurations.
    """
    score_values = score_values or [50, 60, 70]
    stop_values = stop_values or [1.0, 1.5, 2.0]
    target_values = target_values or [1.5, 2.0, 3.0]
    candidates = []
    for score in score_values:
        for stop in stop_values:
            for target in target_values:
                trades = run_backtest(
                    symbols=symbols, horizon=horizon, direction=direction,
                    min_score=int(score), min_price=2.0,
                    min_dollar_volume=5_000_000, stop_atr=float(stop),
                    target_r=float(target), slippage_bps=float(slippage_bps),
                    fee_bps_per_side=float(fee_bps_per_side),
                )
                if trades.empty:
                    continue
                train, test = make_walk_forward_split(trades, train_fraction=train_fraction)
                train_metrics = _optimization_metrics(train)
                test_metrics = _optimization_metrics(test)
                candidates.append({
                    "min_score": int(score), "stop_atr": float(stop),
                    "target_r": float(target), "train_trades": train_metrics["trades"],
                    "train_win_rate_pct": train_metrics["win_rate_pct"],
                    "train_profit_factor": train_metrics["profit_factor"],
                    "train_mean_trade_pct": train_metrics["mean_trade_pct"],
                    "train_compounded_return_pct": train_metrics["compounded_return_pct"],
                    "train_max_drawdown_pct": train_metrics["max_drawdown_pct"],
                    "_train": train, "_test": test,
                    "_test_metrics": test_metrics,
                })
    if not candidates:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    eligible = [c for c in candidates if c["train_trades"] >= int(min_train_trades)
                and np.isfinite(c["train_profit_factor"])]
    if not eligible:
        # Do not silently select a tiny or infinite-PF sample.
        leaderboard = pd.DataFrame([{k: v for k, v in c.items()
                                      if not k.startswith("_")} for c in candidates])
        leaderboard = leaderboard.sort_values(
            ["train_trades", "train_mean_trade_pct"], ascending=[False, False]
        ).reset_index(drop=True)
        return leaderboard, {"status": "NO CONFIGURATION MET MINIMUM TRAINING SAMPLE",
                             "configurations_tested": len(candidates)}, pd.DataFrame(), pd.DataFrame()

    # Ranking uses in-sample PF, with mean return and drawdown as tie-breakers.
    eligible.sort(key=lambda c: (
        c["train_profit_factor"], c["train_mean_trade_pct"],
        c["train_max_drawdown_pct"]
    ), reverse=True)
    best = eligible[0]
    leaderboard = pd.DataFrame([{k: v for k, v in c.items()
                                  if not k.startswith("_")} for c in eligible])
    leaderboard = leaderboard.sort_values(
        ["train_profit_factor", "train_mean_trade_pct"], ascending=[False, False]
    ).reset_index(drop=True)
    holdout = dict(best["_test_metrics"])
    holdout.update({
        "status": "SELECTED ON IN-SAMPLE ONLY; HOLDOUT NOT USED FOR RANKING",
        "selected_min_score": best["min_score"],
        "selected_stop_atr": best["stop_atr"],
        "selected_target_r": best["target_r"],
        "configurations_tested": len(candidates),
        "eligible_configurations": len(eligible),
    })
    return leaderboard, holdout, best["_train"].copy(), best["_test"].copy()


def run_walk_forward_validation(
    symbols: list[str],
    horizon: str = "Swing",
    direction: str = "LONG",
    score_values: list[int] | None = None,
    stop_values: list[float] | None = None,
    target_values: list[float] | None = None,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    min_train_trades: int = 20,
    slippage_bps: float = 5.0,
    fee_bps_per_side: float = 0.0,
    starting_capital: float = 10_000.0,
    risk_per_trade_pct: float = 0.5,
    max_positions: int = 5,
    max_position_exposure_pct: float = 25.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeated rolling walk-forward parameter selection and forward evaluation.

    Each fold selects a parameter set using only trades whose exits fall in the
    preceding training window. The selected set is then evaluated on trades
    entering in the immediately following test window. No test-window metric is
    used to choose parameters. Returns fold report, aggregate summary and
    concatenated selected test trades (including portfolio allocation fields).
    """
    score_values = score_values or [50, 60, 70]
    stop_values = stop_values or [1.0, 1.5, 2.0]
    target_values = target_values or [1.5, 2.0, 3.0]
    configs = []
    for score in score_values:
        for stop in stop_values:
            for target in target_values:
                candidate = run_backtest(
                    symbols=symbols, horizon=horizon, direction=direction,
                    min_score=int(score), min_price=2.0,
                    min_dollar_volume=5_000_000, stop_atr=float(stop),
                    target_r=float(target), slippage_bps=float(slippage_bps),
                    fee_bps_per_side=float(fee_bps_per_side),
                )
                if not candidate.empty:
                    for col in ("signal_time", "entry_time", "exit_time"):
                        candidate[col] = pd.to_datetime(candidate[col], errors="coerce", utc=True).dt.tz_convert(None)
                    configs.append({
                        "min_score": int(score), "stop_atr": float(stop),
                        "target_r": float(target), "trades": candidate,
                    })
    if not configs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    valid_dates = pd.concat([
        pd.to_datetime(c["trades"]["entry_time"], errors="coerce") for c in configs
    ]).dropna()
    if valid_dates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    first_date, last_date = valid_dates.min().normalize(), valid_dates.max().normalize()
    test_start = first_date + pd.DateOffset(months=int(train_months))
    folds, all_test_trades = [], []
    fold_number = 0
    while test_start <= last_date:
        test_end = test_start + pd.DateOffset(months=int(test_months))
        train_start = test_start - pd.DateOffset(months=int(train_months))
        ranked = []
        for config in configs:
            trades = config["trades"]
            train = trades[
                (pd.to_datetime(trades["exit_time"]) >= train_start) &
                (pd.to_datetime(trades["exit_time"]) < test_start)
            ]
            metrics = _optimization_metrics(train)
            if metrics["trades"] >= int(min_train_trades) and np.isfinite(metrics["profit_factor"]):
                ranked.append((metrics["profit_factor"], metrics["mean_trade_pct"],
                               config, metrics))
        fold_number += 1
        if not ranked:
            folds.append({
                "fold": fold_number, "train_start": train_start.date(),
                "train_end": (test_start - pd.Timedelta(days=1)).date(),
                "test_start": test_start.date(),
                "test_end": (test_end - pd.Timedelta(days=1)).date(),
                "status": "NO ELIGIBLE TRAINING CONFIGURATION",
                "selected_min_score": np.nan, "selected_stop_atr": np.nan,
                "selected_target_r": np.nan, "train_trades": 0,
                "train_profit_factor": np.nan, "test_trades": 0,
                "test_profit_factor": np.nan, "test_mean_trade_pct": np.nan,
                "test_win_rate_pct": np.nan, "test_compounded_return_pct": np.nan,
                "test_max_drawdown_pct": np.nan, "portfolio_trades": 0,
                "portfolio_return_pct": np.nan, "portfolio_max_drawdown_pct": np.nan,
            })
            test_start += pd.DateOffset(months=int(step_months))
            continue

        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, _, selected, train_metrics = ranked[0]
        selected_trades = selected["trades"]
        test = selected_trades[
            (pd.to_datetime(selected_trades["entry_time"]) >= test_start) &
            (pd.to_datetime(selected_trades["entry_time"]) < test_end)
        ].copy()
        test_metrics = _optimization_metrics(test)
        allocated, equity_curve = simulate_portfolio(
            test, starting_capital=float(starting_capital),
            risk_per_trade_pct=float(risk_per_trade_pct),
            max_positions=int(max_positions),
            max_position_exposure_pct=float(max_position_exposure_pct),
        )
        taken = allocated[allocated.get("portfolio_status", pd.Series(index=allocated.index, dtype=str)).eq("TAKEN")].copy() if not allocated.empty else pd.DataFrame()
        if not taken.empty:
            end_equity = float(starting_capital) + float(taken["portfolio_pnl"].sum())
            portfolio_return = (end_equity / float(starting_capital) - 1) * 100
            portfolio_dd = float(equity_curve["drawdown_pct"].min()) if not equity_curve.empty else 0.0
            taken["walk_forward_fold"] = fold_number
            taken["selected_min_score"] = selected["min_score"]
            taken["selected_stop_atr"] = selected["stop_atr"]
            taken["selected_target_r"] = selected["target_r"]
            all_test_trades.append(taken)
        else:
            end_equity, portfolio_return, portfolio_dd = float(starting_capital), 0.0, 0.0
        folds.append({
            "fold": fold_number, "train_start": train_start.date(),
            "train_end": (test_start - pd.Timedelta(days=1)).date(),
            "test_start": test_start.date(),
            "test_end": (test_end - pd.Timedelta(days=1)).date(),
            "status": "EVALUATED" if not test.empty else "NO TEST TRADES",
            "selected_min_score": selected["min_score"],
            "selected_stop_atr": selected["stop_atr"],
            "selected_target_r": selected["target_r"],
            "train_trades": train_metrics["trades"],
            "train_profit_factor": train_metrics["profit_factor"],
            "test_trades": test_metrics["trades"],
            "test_profit_factor": test_metrics["profit_factor"],
            "test_mean_trade_pct": test_metrics["mean_trade_pct"],
            "test_win_rate_pct": test_metrics["win_rate_pct"],
            "test_compounded_return_pct": test_metrics["compounded_return_pct"],
            "test_max_drawdown_pct": test_metrics["max_drawdown_pct"],
            "portfolio_trades": int(len(taken)),
            "portfolio_return_pct": portfolio_return,
            "portfolio_max_drawdown_pct": portfolio_dd,
        })
        test_start += pd.DateOffset(months=int(step_months))

    fold_report = pd.DataFrame(folds)
    selected_test = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame()
    evaluated = fold_report[fold_report["status"].isin(["EVALUATED", "NO TEST TRADES"])] if not fold_report.empty else pd.DataFrame()
    if not evaluated.empty:
        valid_pf = pd.to_numeric(evaluated["test_profit_factor"], errors="coerce").dropna()
        valid_mean = pd.to_numeric(evaluated["test_mean_trade_pct"], errors="coerce").dropna()
        valid_port = pd.to_numeric(evaluated["portfolio_return_pct"], errors="coerce").dropna()
        aggregate = pd.DataFrame([{
            "folds_total": int(len(fold_report)),
            "folds_with_test_trades": int((evaluated["test_trades"] > 0).sum()),
            "positive_test_folds": int((pd.to_numeric(evaluated["test_mean_trade_pct"], errors="coerce") > 0).sum()),
            "median_test_profit_factor": float(valid_pf.median()) if len(valid_pf) else np.nan,
            "average_test_mean_trade_pct": float(valid_mean.mean()) if len(valid_mean) else np.nan,
            "combined_portfolio_trades": int(evaluated["portfolio_trades"].sum()),
            "sum_of_fold_portfolio_returns_pct": float(valid_port.sum()) if len(valid_port) else np.nan,
            "median_fold_portfolio_return_pct": float(valid_port.median()) if len(valid_port) else np.nan,
            "worst_fold_portfolio_drawdown_pct": float(pd.to_numeric(evaluated["portfolio_max_drawdown_pct"], errors="coerce").min()),
            "configurations_tested_per_fold": int(len(score_values) * len(stop_values) * len(target_values)),
        }])
    else:
        aggregate = pd.DataFrame()
    return fold_report, aggregate, selected_test


def analyze_signal_features(trades: pd.DataFrame, min_group_trades: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe historical trade outcomes by signal features; association only, not causation."""
    columns = ["feature", "bucket", "trades", "win_rate_pct", "profit_factor",
               "mean_trade_pct", "median_trade_pct", "average_win_pct", "average_loss_pct"]
    if trades is None or trades.empty or "net_return_pct" not in trades.columns:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    data = trades.copy()
    data["net_return_pct"] = pd.to_numeric(data["net_return_pct"], errors="coerce")
    data = data.dropna(subset=["net_return_pct"])

    def _bucket_numeric(series, bins, labels):
        values = pd.to_numeric(series, errors="coerce")
        return pd.cut(values, bins=bins, labels=labels, include_lowest=True).astype("object").where(values.notna(), "Unavailable")

    feature_specs = []
    if "signal_score" in data:
        data["_score_bucket"] = _bucket_numeric(data["signal_score"], [-np.inf, 49, 59, 69, 79, np.inf],
                                                 ["0-49", "50-59", "60-69", "70-79", "80+"])
        feature_specs.append(("Signal score", "_score_bucket"))
    if "rsi" in data:
        data["_rsi_bucket"] = _bucket_numeric(data["rsi"], [-np.inf, 30, 45, 55, 70, np.inf],
                                               ["<=30", "30-45", "45-55", "55-70", ">70"])
        feature_specs.append(("RSI(14)", "_rsi_bucket"))
    if "rvol" in data:
        data["_rvol_bucket"] = _bucket_numeric(data["rvol"], [-np.inf, 0.8, 1.2, 2.0, np.inf],
                                                ["<0.8x", "0.8-1.2x", "1.2-2.0x", ">2.0x"])
        feature_specs.append(("Relative volume", "_rvol_bucket"))
    for col, label in [("market_regime", "SPY market regime"), ("symbol", "Ticker"),
                       ("signal", "Signal type"), ("exit_reason", "Exit reason")]:
        if col in data:
            data[col] = data[col].fillna("Unavailable").astype(str)
            feature_specs.append((label, col))

    rows = []
    for label, col in feature_specs:
        for bucket, group in data.groupby(col, dropna=False, observed=False):
            returns = group["net_return_pct"].dropna()
            if len(returns) < int(min_group_trades):
                continue
            wins, losses = returns[returns > 0], returns[returns < 0]
            gross_profit, gross_loss = float(wins.sum()), float(abs(losses.sum()))
            pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
            rows.append({
                "feature": label, "bucket": str(bucket), "trades": int(len(returns)),
                "win_rate_pct": float((returns > 0).mean() * 100),
                "profit_factor": float(pf), "mean_trade_pct": float(returns.mean()),
                "median_trade_pct": float(returns.median()),
                "average_win_pct": float(wins.mean()) if len(wins) else np.nan,
                "average_loss_pct": float(losses.mean()) if len(losses) else np.nan,
            })
    report = pd.DataFrame(rows, columns=columns)
    ticker = report[report["feature"].eq("Ticker")].copy() if not report.empty else pd.DataFrame(columns=columns)
    return report, ticker


def run_portfolio_robustness(
    trades: pd.DataFrame,
    starting_capital: float = 10000.0,
    risk_per_trade_pct: float = 0.5,
    max_positions: int = 5,
    max_position_exposure_pct: float = 25.0,
    max_gross_exposure_pct: float = 100.0,
    extra_round_trip_cost_bps: float = 0.0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Stage 3.10: chronological, realized-equity portfolio and robustness diagnostics.

    Uses only trades with entry and exit timestamps before today's date, removes
    duplicate symbol/horizon/direction/entry records, and sizes positions using
    realized equity. Drawdown is exit-to-exit (not intrabar mark-to-market).
    """
    empty = pd.DataFrame()
    if trades is None or trades.empty:
        return {}, empty, empty, empty, {}
    data = trades.copy()
    for col in ("signal_time", "entry_time", "exit_time"):
        data[col] = pd.to_datetime(data[col], errors="coerce", utc=True).dt.tz_convert(None)
    today = pd.Timestamp.now().normalize()
    raw_count = len(data)
    data = data.dropna(subset=["entry_time", "exit_time", "net_return_pct"])
    data = data[(data["entry_time"] < today) & (data["exit_time"] < today)].copy()
    complete_count = len(data)
    keys = [c for c in ("symbol", "horizon", "direction", "entry_time") if c in data.columns]
    if keys:
        data = data.sort_values("entry_time").drop_duplicates(keys, keep="first")
    duplicate_count = complete_count - len(data)
    if data.empty:
        return {"raw_trades": raw_count, "completed_trades": 0,
                "excluded_incomplete": raw_count, "duplicates_removed": duplicate_count}, empty, empty, empty, {}
    data["net_return_pct"] = pd.to_numeric(data["net_return_pct"], errors="coerce")
    data = data.dropna(subset=["net_return_pct"]).sort_values(["entry_time", "symbol"])
    active, pending, accepted, skipped = [], [], [], []
    equity = float(starting_capital)
    for _, row in data.iterrows():
        entry_time = row["entry_time"]
        still = []
        for exit_time, pnl in pending:
            if exit_time <= entry_time:
                equity += pnl
            else:
                still.append((exit_time, pnl))
        pending = still
        active = [p for p in active if p["exit_time"] > entry_time]
        status = None
        if any(p["symbol"] == str(row["symbol"]) for p in active):
            status = "SKIPPED_DUPLICATE_SYMBOL"
        elif len(active) >= int(max_positions):
            status = "SKIPPED_MAX_POSITIONS"
        entry, stop = float(row["entry_price"]), float(row["stop_price"])
        stop_fraction = abs(entry-stop)/entry if entry else np.nan
        if status is None and (not np.isfinite(stop_fraction) or stop_fraction <= 0):
            status = "SKIPPED_INVALID_RISK"
        active_notional = sum(p["notional"] for p in active)
        remaining = max(0.0, equity*max_gross_exposure_pct/100-active_notional)
        risk_budget = equity*risk_per_trade_pct/100
        if status is None:
            notional = min(risk_budget/stop_fraction, equity*max_position_exposure_pct/100, remaining)
            if notional <= 0:
                status = "SKIPPED_EXPOSURE_CAP"
        if status:
            item = row.to_dict()
            item["portfolio_status"] = status
            skipped.append(item)
            continue
        pnl = notional*float(row["net_return_pct"])/100
        item = row.to_dict()
        item.update({"portfolio_status":"TAKEN", "portfolio_equity_at_entry":equity,
                     "risk_budget":risk_budget, "position_notional":notional,
                     "allocated_risk_pct_equity":notional*stop_fraction/equity*100,
                     "portfolio_pnl":pnl, "position_weight_pct":notional/equity*100})
        accepted.append(item)
        active.append({"symbol":str(row["symbol"]), "exit_time":row["exit_time"], "notional":notional})
        pending.append((row["exit_time"], pnl))
    taken = pd.DataFrame(accepted)
    skipped_df = pd.DataFrame(skipped)
    allocation = pd.concat([taken, skipped_df], ignore_index=True, sort=False) if not skipped_df.empty else taken.copy()
    if taken.empty:
        return {"raw_trades":raw_count,"completed_trades":len(data),"excluded_incomplete":raw_count-complete_count,
                "duplicates_removed":duplicate_count,"accepted_trades":0}, allocation, empty, empty, {}
    # Exit-event account curve, sorted chronologically.
    events = taken.sort_values(["exit_time","symbol"])
    acct, peak, curve_rows = float(starting_capital), float(starting_capital), []
    for _, t in events.iterrows():
        acct += float(t["portfolio_pnl"])
        peak = max(peak, acct)
        curve_rows.append({"time":t["exit_time"],"equity":acct,
                           "drawdown_pct":(acct/peak-1)*100,
                           "symbol":t["symbol"],"pnl":t["portfolio_pnl"]})
    curve = pd.DataFrame(curve_rows)
    pnl = taken["portfolio_pnl"].astype(float)
    wins, losses = pnl[pnl>0], pnl[pnl<0]
    gross_profit, gross_loss = float(wins.sum()), float(abs(losses.sum()))
    pf = gross_profit/gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    peak_to_valley = curve["drawdown_pct"].astype(float)
    loss_flags = (pnl < 0).tolist()
    max_streak = streak = 0
    for is_loss in loss_flags:
        streak = streak+1 if is_loss else 0
        max_streak = max(max_streak, streak)
    # Recovery measured in exit events from first drawdown trough to prior equity peak.
    eq = curve["equity"].astype(float)
    running_peak = eq.cummax()
    under = eq < running_peak
    recovery_events = 0
    if under.any():
        longest, current = 0, 0
        for is_under in under:
            current = current+1 if is_under else 0
            longest = max(longest,current)
        recovery_events = int(longest)
    symbol_pnl = taken.groupby("symbol")["portfolio_pnl"].sum().sort_values(ascending=False)
    concentration = (float(symbol_pnl.head(3).sum()/pnl.sum()*100)
                     if pnl.sum() > 0 else np.nan)
    summary = {
        "raw_trades":raw_count, "completed_trades":len(data),
        "excluded_incomplete":raw_count-complete_count, "duplicates_removed":duplicate_count,
        "accepted_trades":len(taken), "skipped_trades":len(data)-len(taken),
        "ending_equity":float(curve["equity"].iloc[-1]),
        "portfolio_return_pct":float((curve["equity"].iloc[-1]/starting_capital-1)*100),
        "profit_factor":float(pf), "win_rate_pct":float((pnl>0).mean()*100),
        "max_exit_to_exit_drawdown_pct":float(peak_to_valley.min()),
        "max_consecutive_losses":int(max_streak),
        "longest_underwater_exit_events":recovery_events,
        "top3_ticker_pnl_share_pct":concentration,
        "max_position_weight_pct":float(taken["position_weight_pct"].max()),
        "max_allocated_risk_pct_equity":float(taken["allocated_risk_pct_equity"].max()),
    }
    stress_rows=[]
    for extra_bps in (0, 5, 10, 20, 40):
        stressed = taken.copy()
        stressed["stress_return_pct"] = stressed["net_return_pct"].astype(float)-2*extra_bps/100
        stressed_pnl = stressed["position_notional"]*stressed["stress_return_pct"]/100
        final = float(starting_capital+stressed_pnl.sum())
        stress_rows.append({"additional_slippage_bps_per_side":extra_bps,
                            "ending_equity":final,
                            "portfolio_return_pct":(final/starting_capital-1)*100,
                            "total_pnl":float(stressed_pnl.sum())})
    stress = pd.DataFrame(stress_rows)
    benchmark = {}
    try:
        start = data["entry_time"].min().date().isoformat()
        end = (data["exit_time"].max()+pd.Timedelta(days=2)).date().isoformat()
        spy = yf.download("SPY", start=start, end=end, interval="1d",
                          auto_adjust=True, progress=False, threads=False)
        if isinstance(spy.columns,pd.MultiIndex):
            spy.columns=spy.columns.get_level_values(0)
        if not spy.empty and "Close" in spy:
            prices = pd.to_numeric(spy["Close"],errors="coerce").dropna()
            if len(prices)>=2:
                benchmark={"spy_start":float(prices.iloc[0]),"spy_end":float(prices.iloc[-1]),
                           "spy_buy_hold_return_pct":float((prices.iloc[-1]/prices.iloc[0]-1)*100),
                           "spy_start_date":str(prices.index[0]),"spy_end_date":str(prices.index[-1])}
    except Exception as exc:
        benchmark={"spy_status":f"Unavailable: {str(exc)[:160]}"}
    return summary, allocation, curve, stress, benchmark
