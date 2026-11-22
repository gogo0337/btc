"""Yahoo Finance 공개 차트 API에서 일별 종가 수집 (키 불필요, 비공식 API라 형식이 바뀔 수 있음)."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
SYMBOLS = {"btc": "BTC-USD", "nasdaq": "^IXIC", "gold": "GC=F", "dxy": "DX-Y.NYB",
           "us10y": "^TNX", "eth": "ETH-USD"}


def fetch_yahoo(symbol: str, start: str = "2014-01-01") -> pd.Series:
    params = {"period1": int(pd.Timestamp(start).timestamp()),
              "period2": int(datetime.now(timezone.utc).timestamp()), "interval": "1d"}
    r = requests.get(YAHOO_URL.format(symbol=symbol), params=params, timeout=60,
                     headers={"User-Agent": "Mozilla/5.0 (research script)"})
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    s = pd.Series(res["indicators"]["quote"][0]["close"],
                  index=pd.to_datetime(res["timestamp"], unit="s").floor("D"), dtype=float)
    return s[~s.index.duplicated(keep="last")].dropna()


def fetch_all(cache_dir: Path | None = None) -> pd.DataFrame:
    """일별 종가 표(달력일 기준, 휴장일은 직전 값). 마감되지 않은 오늘 값은 뺀다."""
    cache = (cache_dir / "macro.csv") if cache_dir else None
    today = datetime.now(timezone.utc).date()
    if cache and cache.exists() and \
            datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc).date() == today:
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    cols = {}
    for name, sym in SYMBOLS.items():
        cols[name] = fetch_yahoo(sym)
        time.sleep(0.5)
    df = pd.DataFrame(cols).sort_index()
    df = df[df.index < pd.Timestamp(today)]
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="D"))
    raw_btc = df["btc"].copy()
    df = df.ffill()
    df["btc"] = raw_btc.ffill(limit=2)
    df.index.name = "date"
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
    return df
