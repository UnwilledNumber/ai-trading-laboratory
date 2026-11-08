#!/usr/bin/env python3
"""
AI Trading Laboratory — Stage 1 Market Scanner
Research prototype. No brokerage connection or order execution.

Scans US-listed common stocks and ETFs using:
- Daily trend / swing signals
- Intraday momentum / VWAP / relative-volume signals

Data source: Yahoo Finance via yfinance (unofficial; subject to delays,
rate limits, symbol coverage gaps, and provider changes).

Examples:
  python market_scanner.py --mode both --limit 100
  python market_scanner.py --mode daily
  python market_scanner.py --mode intraday --refresh 300 --limit 250
  python market_scanner.py --symbols AAPL,MSFT,NVDA,SPY --mode both
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"
CACHE_DIR = APP_DIR / "cache"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Initial research filters. These are configurable hypotheses, not proven edges.
MIN_PRICE = 2.00
MIN_AVG_DOLLAR_VOLUME = 5_000_000
MIN_RELATIVE_VOLUME = 1.50
MAX_SYMBOLS_DEFAULT = 0  # 0 means no cap; use --limit for testing.
BATCH_SIZE = 40

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("market_scanner")


def clean_symbol(symbol: str) -> str:
    """Convert exchange symbols to Yahoo Finance format."""
    return str(symbol).strip().replace("$", "").replace(".", "-")


def fetch_universe(include_etfs: bool = True) -> list[str]:
    """
    Download Nasdaq Trader symbol directories and combine Nasdaq + other
    US exchange listings. Removes test issues and obvious non-common-stock
    instruments. ETF inclusion is configurable.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketScanner/1.0)"}
    frames = []

    for url, kind in ((NASDAQ_TRADED_URL, "nasdaq"), (NASDAQ_LISTED_URL, "other")):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text), sep="|")
            df.columns = [str(c).strip() for c in df.columns]
            frames.append((df, kind))
        except Exception as exc:
            log.warning("Could not load %s symbol directory: %s", kind, exc)

    symbols: set[str] = set()
    for df, kind in frames:
        if kind == "nasdaq":
            sym_col = "Nasdaq Traded"
            test_col = "Test Issue"
            etf_col = "ETF"
            name_col = "Security Name"
        else:
            sym_col = "ACT Symbol"
            test_col = "Test Issue"
            etf_col = "ETF"
            name_col = "Security Name"

        if sym_col not in df.columns:
            continue

        subset = df.copy()
        if test_col in subset.columns:
            subset = subset[subset[test_col].astype(str).str.upper().eq("N")]
        if not include_etfs and etf_col in subset.columns:
            subset = subset[~subset[etf_col].astype(str).str.upper().eq("Y")]

        # Filter obvious non-operating-company instruments; retain ETFs when requested.
        if name_col in subset.columns:
            names = subset[name_col].astype(str).str.lower()
            excluded = r"\b(warrant|right|unit|depositary|debenture|notes?|preferred|when issued)\b"
            subset = subset[~names.str.contains(excluded, regex=True, na=False)]

        for raw in subset[sym_col].dropna().astype(str):
            raw = raw.strip()
            if not raw or raw in {"File Creation Time", "Symbol"}:
                continue
            if any(ch in raw for ch in ("^", "/", " ", "+")):
                continue
            symbol = clean_symbol(raw)
            if symbol and len(symbol) <= 12:
                symbols.add(symbol)

    if not symbols:
        raise RuntimeError(
            "Could not build the US listing universe. Check internet access or "
            "use --symbols AAPL,MSFT,NVDA to supply a manual universe."
        )
    return sorted(symbols)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    """Calculate common technical indicators from OHLCV data."""
    if df.empty:
        return df
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"].fillna(0)

    out["EMA9"] = close.ewm(span=9, adjust=False).mean()
    out["EMA21"] = close.ewm(span=21, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()
    out["EMA200"] = close.ewm(span=200, adjust=False).mean()
    out["RSI14"] = rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["ATR_PCT"] = 100 * out["ATR14"] / close.replace(0, np.nan)

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["BB_MID"] = mid
    out["BB_UPPER"] = mid + 2 * std
    out["BB_LOWER"] = mid - 2 * std
    out["BB_WIDTH_PCT"] = 100 * (out["BB_UPPER"] - out["BB_LOWER"]) / mid.replace(0, np.nan)

    # Relative volume: intraday compares the latest bar to the median of prior bars;
    # daily compares today's volume with the prior 20-session average.
    if intraday:
        out["RVOL"] = volume / volume.shift(1).rolling(20).median().replace(0, np.nan)
        # Session VWAP. Intraday yfinance bars are generally timestamped in exchange time.
        typical = (high + low + close) / 3
        dates = pd.Series(out.index.date, index=out.index)
        pv = typical * volume
        out["VWAP"] = pv.groupby(dates).cumsum() / volume.groupby(dates).cumsum().replace(0, np.nan)
    else:
        out["RVOL"] = volume / volume.shift(1).rolling(20).mean().replace(0, np.nan)
        out["VWAP"] = np.nan

    out["DOLLAR_VOLUME"] = close * volume
    return out


def last_valid(row: pd.Series, key: str, default=np.nan):
    value = row.get(key, default)
    try:
        return float(value) if pd.notna(value) else default
    except (TypeError, ValueError):
        return default


def score_daily(df: pd.DataFrame, symbol: str) -> dict | None:
    if len(df) < 60:
        return None
    row = df.iloc[-1]
    price = last_valid(row, "Close")
    avg_dv = last_valid(df["DOLLAR_VOLUME"].tail(20).mean(), "x") if False else float(df["DOLLAR_VOLUME"].tail(20).mean())
    if not np.isfinite(price) or price < MIN_PRICE or avg_dv < MIN_AVG_DOLLAR_VOLUME:
        return None

    rsi_v = last_valid(row, "RSI14")
    rvol = last_valid(row, "RVOL")
    ema9, ema21 = last_valid(row, "EMA9"), last_valid(row, "EMA21")
    ema50, ema200 = last_valid(row, "EMA50"), last_valid(row, "EMA200")
    macd, macds = last_valid(row, "MACD"), last_valid(row, "MACD_SIGNAL")
    high20 = float(df["High"].iloc[-21:-1].max()) if len(df) >= 22 else np.nan

    trend = price > ema50 and ema50 > ema200
    short_trend = ema9 > ema21
    macd_bull = macd > macds
    breakout = np.isfinite(high20) and price > high20
    momentum = trend and short_trend and macd_bull and 45 <= rsi_v <= 75
    reversal = rsi_v < 35 and price > ema9 and macd > macds

    score = 0
    score += 20 if trend else 0
    score += 15 if short_trend else 0
    score += 15 if macd_bull else 0
    score += 15 if breakout else 0
    score += 10 if rvol >= MIN_RELATIVE_VOLUME else 0
    score += 10 if 50 <= rsi_v <= 70 else 0
    score += 5 if last_valid(row, "ATR_PCT") >= 1 else 0
    score += 10 if reversal else 0

    if breakout:
        signal = "BREAKOUT"
    elif reversal:
        signal = "REVERSAL WATCH"
    elif momentum:
        signal = "MOMENTUM"
    else:
        signal = "WATCH"

    return {
        "symbol": symbol, "timeframe": "1D", "signal": signal, "score": score,
        "price": price, "rsi14": rsi_v, "rvol": rvol,
        "ema9": ema9, "ema21": ema21, "ema50": ema50, "ema200": ema200,
        "macd_hist": last_valid(row, "MACD_HIST"),
        "atr_pct": last_valid(row, "ATR_PCT"),
        "avg_dollar_volume_20d": avg_dv,
        "close_vs_20d_high_pct": (price / high20 - 1) * 100 if high20 else np.nan,
    }


def score_intraday(df: pd.DataFrame, symbol: str) -> dict | None:
    if len(df) < 30:
        return None
    row = df.iloc[-1]
    price = last_valid(row, "Close")
    rsi_v = last_valid(row, "RSI14")
    rvol = last_valid(row, "RVOL")
    vwap = last_valid(row, "VWAP")
    ema9, ema21 = last_valid(row, "EMA9"), last_valid(row, "EMA21")
    macd, macds = last_valid(row, "MACD"), last_valid(row, "MACD_SIGNAL")
    if not np.isfinite(price) or price < MIN_PRICE:
        return None

    above_vwap = np.isfinite(vwap) and price > vwap
    momentum = above_vwap and ema9 > ema21 and macd > macds and 45 <= rsi_v <= 78
    breakout_level = float(df["High"].iloc[-13:-1].max()) if len(df) >= 14 else np.nan
    breakout = np.isfinite(breakout_level) and price > breakout_level and rvol >= MIN_RELATIVE_VOLUME

    score = 0
    score += 20 if above_vwap else 0
    score += 15 if ema9 > ema21 else 0
    score += 15 if macd > macds else 0
    score += 20 if rvol >= MIN_RELATIVE_VOLUME else 0
    score += 10 if 50 <= rsi_v <= 72 else 0
    score += 20 if breakout else 0

    signal = "INTRADAY BREAKOUT" if breakout else ("INTRADAY MOMENTUM" if momentum else "WATCH")
    return {
        "symbol": symbol, "timeframe": "5m", "signal": signal, "score": score,
        "price": price, "rsi14": rsi_v, "rvol": rvol, "vwap": vwap,
        "above_vwap": bool(above_vwap), "ema9": ema9, "ema21": ema21,
        "macd_hist": last_valid(row, "MACD_HIST"),
        "atr_pct": last_valid(row, "ATR_PCT"),
    }


def download_batch(symbols: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    """Download a batch and normalize yfinance's single-/multi-symbol output."""
    if not symbols:
        return {}
    try:
        raw = yf.download(
            tickers=symbols,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
            prepost=False,
        )
    except Exception as exc:
        log.warning("Batch download failed (%s): %s", interval, exc)
        return {}

    result = {}
    for symbol in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if symbol not in raw.columns.get_level_values(0):
                    continue
                frame = raw[symbol].copy()
            else:
                if len(symbols) != 1:
                    continue
                frame = raw.copy()
            required = ["Open", "High", "Low", "Close", "Volume"]
            if not all(c in frame.columns for c in required):
                continue
            frame = frame[required].dropna(subset=["Close"])
            if not frame.empty:
                result[symbol] = frame
        except Exception:
            continue
    return result


def scan(symbols: list[str], mode: str, limit: int = 0) -> pd.DataFrame:
    if limit > 0:
        symbols = symbols[:limit]
    results: list[dict] = []

    if mode in ("daily", "both"):
        log.info("Starting daily scan for %d symbols", len(symbols))
        for start in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[start:start + BATCH_SIZE]
            data = download_batch(batch, period="1y", interval="1d")
            for symbol, frame in data.items():
                try:
                    ind = add_indicators(frame, intraday=False)
                    result = score_daily(ind, symbol)
                    if result:
                        results.append(result)
                except Exception as exc:
                    log.debug("Daily calculation failed for %s: %s", symbol, exc)
            log.info("Daily progress: %d/%d", min(start + BATCH_SIZE, len(symbols)), len(symbols))
            time.sleep(0.5)

    if mode in ("intraday", "both"):
        log.info("Starting intraday scan for %d symbols", len(symbols))
        for start in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[start:start + BATCH_SIZE]
            data = download_batch(batch, period="5d", interval="5m")
            for symbol, frame in data.items():
                try:
                    ind = add_indicators(frame, intraday=True)
                    result = score_intraday(ind, symbol)
                    if result:
                        results.append(result)
                except Exception as exc:
                    log.debug("Intraday calculation failed for %s: %s", symbol, exc)
            log.info("Intraday progress: %d/%d", min(start + BATCH_SIZE, len(symbols)), len(symbols))
            time.sleep(0.5)

    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    return out.sort_values(["timeframe", "score", "rvol"], ascending=[True, False, False])


def save_results(results: pd.DataFrame) -> Path | None:
    if results.empty:
        log.warning("No qualifying results in this scan.")
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"scan_results_{stamp}.csv"
    results.to_csv(path, index=False)
    latest = OUTPUT_DIR / "latest_scan.csv"
    results.to_csv(latest, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="US market intraday + daily scanner")
    parser.add_argument("--mode", choices=["daily", "intraday", "both"], default="both")
    parser.add_argument("--limit", type=int, default=MAX_SYMBOLS_DEFAULT,
                        help="Limit symbols for a test run; 0 scans the full available universe")
    parser.add_argument("--refresh", type=int, default=0,
                        help="Repeat intraday scan every N seconds; 0 runs once")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated manual symbol list, e.g. AAPL,MSFT,NVDA")
    parser.add_argument("--exclude-etfs", action="store_true")
    args = parser.parse_args()

    if args.symbols.strip():
        symbols = [clean_symbol(s) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = fetch_universe(include_etfs=not args.exclude_etfs)

    log.info("Universe loaded: %d symbols", len(symbols))
    if args.limit > 0:
        log.info("Test limit active: scanning first %d symbols", args.limit)

    while True:
        results = scan(symbols, args.mode, args.limit)
        saved = save_results(results)
        if not results.empty:
            print("\nTOP 25 OPPORTUNITIES")
            print(results.head(25).to_string(index=False))
            print(f"\nSaved results: {saved}")
        if args.refresh <= 0 or args.mode == "daily":
            break
        log.info("Next scan in %d seconds", args.refresh)
        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
