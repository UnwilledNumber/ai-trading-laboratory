from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
WATCHLIST_FILE = APP_DIR / "watchlist.txt"

DEFAULT_WATCHLIST = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","AVGO","NFLX",
    "JPM","XOM","LLY","WMT","COST","ORCL","CRM","PLTR","COIN","HOOD",
    "SPY","QQQ","IWM","DIA","SMH"
]
DATA_NOTE = (
    "Data source: Yahoo Finance through yfinance (unofficial). Quotes may be delayed, "
    "intraday history is limited, and bulk requests can be throttled. Do not use this "
    "prototype as an execution-grade real-time feed."
)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("market_scanner")

NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

def clean_symbol(symbol: str) -> str:
    return str(symbol).strip().replace("$", "").replace(".", "-")

def load_watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        items = [clean_symbol(x) for x in WATCHLIST_FILE.read_text().splitlines() if x.strip()]
        return items or DEFAULT_WATCHLIST.copy()
    return DEFAULT_WATCHLIST.copy()

def save_watchlist(symbols: list[str]) -> None:
    WATCHLIST_FILE.write_text("\n".join(sorted(set(map(clean_symbol, symbols)))) + "\n")

def fetch_universe(include_etfs: bool = True) -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    frames = []
    for url, kind in ((NASDAQ_TRADED_URL, "nasdaq"), (NASDAQ_LISTED_URL, "other")):
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), sep="|")
        df.columns = [str(c).strip() for c in df.columns]
        frames.append((df, kind))
    symbols = set()
    for df, kind in frames:
        sym_col = "Nasdaq Traded" if kind == "nasdaq" else "ACT Symbol"
        if sym_col not in df:
            continue
        subset = df.copy()
        if "Test Issue" in subset:
            subset = subset[subset["Test Issue"].astype(str).str.upper().eq("N")]
        if not include_etfs and "ETF" in subset:
            subset = subset[~subset["ETF"].astype(str).str.upper().eq("Y")]
        if "Security Name" in subset:
            names = subset["Security Name"].astype(str).str.lower()
            subset = subset[~names.str.contains(
                r"\b(warrant|right|unit|depositary|debenture|notes?|preferred|when issued)\b",
                regex=True, na=False
            )]
        for raw in subset[sym_col].dropna().astype(str):
            raw = raw.strip()
            if raw and not any(ch in raw for ch in ("^", "/", " ", "+")):
                symbols.add(clean_symbol(raw))
    if not symbols:
        raise RuntimeError("Unable to retrieve US listing universe.")
    return sorted(symbols)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def add_indicators(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]
    volume = out["Volume"].fillna(0)
    out["EMA9"] = close.ewm(span=9, adjust=False).mean()
    out["EMA21"] = close.ewm(span=21, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()
    out["EMA200"] = close.ewm(span=200, adjust=False).mean()
    out["RSI14"] = rsi(close)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["MACD"] = macd
    out["MACD_SIGNAL"] = macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    out["ATR_PCT"] = 100 * out["ATR14"] / close.replace(0, np.nan)
    if intraday:
        out["RVOL"] = volume / volume.shift(1).rolling(20).median().replace(0, np.nan)
        typical = (high + low + close) / 3
        dates = pd.Series(out.index.date, index=out.index)
        out["VWAP"] = (typical * volume).groupby(dates).cumsum() / volume.groupby(dates).cumsum().replace(0, np.nan)
    else:
        out["RVOL"] = volume / volume.shift(1).rolling(20).mean().replace(0, np.nan)
        out["VWAP"] = np.nan
    out["DOLLAR_VOLUME"] = close * volume
    return out

def _score_daily(df: pd.DataFrame, symbol: str, min_price: float, min_dv: float):
    if len(df) < 60:
        return None
    r = df.iloc[-1]
    price = float(r["Close"])
    avg_dv = float(df["DOLLAR_VOLUME"].tail(20).mean())
    if not np.isfinite(price) or price < min_price or avg_dv < min_dv:
        return None
    rv = float(r["RSI14"]) if pd.notna(r["RSI14"]) else 0
    rvol = float(r["RVOL"]) if pd.notna(r["RVOL"]) else 0
    trend = r["Close"] > r["EMA50"] > r["EMA200"]
    short = r["EMA9"] > r["EMA21"]
    macd_bull = r["MACD"] > r["MACD_SIGNAL"]
    prior_high = float(df["High"].iloc[-21:-1].max())
    breakout = price > prior_high
    reversal = rv < 35 and price > r["EMA9"] and macd_bull
    score = (20 if trend else 0) + (15 if short else 0) + (15 if macd_bull else 0)
    score += (15 if breakout else 0) + (10 if rvol >= 1.5 else 0)
    score += (10 if 50 <= rv <= 70 else 0) + (5 if r["ATR_PCT"] >= 1 else 0) + (10 if reversal else 0)
    signal = "BREAKOUT" if breakout else "REVERSAL WATCH" if reversal else "MOMENTUM" if trend and short and macd_bull and 45 <= rv <= 75 else "WATCH"
    return {"symbol":symbol,"timeframe":"1D","signal":signal,"score":int(score),"price":price,
            "rsi14":rv,"rvol":rvol,"ema9":float(r["EMA9"]),"ema21":float(r["EMA21"]),
            "ema50":float(r["EMA50"]),"ema200":float(r["EMA200"]),
            "macd_hist":float(r["MACD"]-r["MACD_SIGNAL"]),"atr_pct":float(r["ATR_PCT"]),
            "avg_dollar_volume_20d":avg_dv,"close_vs_20d_high_pct":(price/prior_high-1)*100 if prior_high else np.nan}

def _score_intraday(df: pd.DataFrame, symbol: str, min_price: float):
    if len(df) < 30:
        return None
    r = df.iloc[-1]
    price = float(r["Close"])
    if not np.isfinite(price) or price < min_price:
        return None
    rv = float(r["RSI14"]) if pd.notna(r["RSI14"]) else 0
    rvol = float(r["RVOL"]) if pd.notna(r["RVOL"]) else 0
    vwap = float(r["VWAP"]) if pd.notna(r["VWAP"]) else np.nan
    above = np.isfinite(vwap) and price > vwap
    short = r["EMA9"] > r["EMA21"]
    macd_bull = r["MACD"] > r["MACD_SIGNAL"]
    prior_high = float(df["High"].iloc[-13:-1].max())
    breakout = price > prior_high and rvol >= 1.5
    momentum = above and short and macd_bull and 45 <= rv <= 78
    score = (20 if above else 0) + (15 if short else 0) + (15 if macd_bull else 0)
    score += (20 if rvol >= 1.5 else 0) + (10 if 50 <= rv <= 72 else 0) + (20 if breakout else 0)
    signal = "INTRADAY BREAKOUT" if breakout else "INTRADAY MOMENTUM" if momentum else "WATCH"
    return {"symbol":symbol,"timeframe":"5m","signal":signal,"score":int(score),"price":price,
            "rsi14":rv,"rvol":rvol,"vwap":vwap,"above_vwap":bool(above),
            "ema9":float(r["EMA9"]),"ema21":float(r["EMA21"]),
            "macd_hist":float(r["MACD"]-r["MACD_SIGNAL"]),"atr_pct":float(r["ATR_PCT"])}

def _download(symbols: list[str], period: str, interval: str) -> dict[str,pd.DataFrame]:
    if not symbols:
        return {}
    raw = yf.download(tickers=symbols, period=period, interval=interval, group_by="ticker",
                      auto_adjust=True, progress=False, threads=True, prepost=False, timeout=25)
    found = {}
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
            cols = ["Open","High","Low","Close","Volume"]
            if all(c in frame.columns for c in cols):
                frame = frame[cols].dropna(subset=["Close"])
                if not frame.empty:
                    found[symbol] = frame
        except Exception:
            continue
    return found

def scan_market(symbols: list[str], mode: str = "both", min_price: float = 2.0,
                min_avg_dollar_volume: float = 5_000_000,
                progress_callback: Callable[[int,int],None] | None = None) -> pd.DataFrame:
    results = []
    symbols = list(dict.fromkeys(clean_symbol(s) for s in symbols if s))
    batch_size = 25
    total_tasks = len(symbols) * (2 if mode == "both" else 1)
    completed = 0
    for timeframe in (["daily","intraday"] if mode == "both" else [mode]):
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start:start+batch_size]
            if timeframe == "daily":
                data = _download(batch, "1y", "1d")
                for sym, frame in data.items():
                    try:
                        result = _score_daily(add_indicators(frame), sym, min_price, min_avg_dollar_volume)
                        if result: results.append(result)
                    except Exception as exc:
                        log.warning("Daily scan failed for %s: %s", sym, exc)
            else:
                data = _download(batch, "5d", "5m")
                for sym, frame in data.items():
                    try:
                        result = _score_intraday(add_indicators(frame, intraday=True), sym, min_price)
                        if result: results.append(result)
                    except Exception as exc:
                        log.warning("Intraday scan failed for %s: %s", sym, exc)
            completed += len(batch)
            if progress_callback:
                progress_callback(min(completed, total_tasks), total_tasks)
            time.sleep(0.25)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(["score","rvol"], ascending=[False,False]).reset_index(drop=True)
