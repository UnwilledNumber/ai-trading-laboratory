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
