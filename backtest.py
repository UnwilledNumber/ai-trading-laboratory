from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from scanner import add_indicators, _score_daily, _score_intraday

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
    frames = []
    for symbol in list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip())):
        try:
            result = backtest_symbol(
                symbol, horizon=horizon, direction=direction, min_score=min_score,
                min_price=min_price, min_dollar_volume=min_dollar_volume,
                stop_atr=stop_atr, target_r=target_r,
                slippage_bps=slippage_bps, fee_bps_per_side=fee_bps_per_side,
            )
            if not result.empty:
                frames.append(result)
        except Exception as exc:
            print(f"Backtest skipped {symbol}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
