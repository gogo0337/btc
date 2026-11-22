"""Binance 공개 REST API에서 일봉 OHLCV 수집 (API 키 불필요)."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List

import pandas as pd
import requests

# api.binance.com 은 일부 지역(예: 미국 소재 GitHub Actions 러너)에서 차단될 수 있어
# 공개 시세 전용 미러를 예비로 둔다.
BINANCE_HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]
DAY_MS = 86_400_000


def _get_klines(session: requests.Session, params: dict) -> list:
    last_err: Exception | None = None
    for host in BINANCE_HOSTS:
        try:
            r = session.get(f"{host}/api/v3/klines", params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # 다음 호스트로 재시도
            last_err = e
    raise RuntimeError(f"Binance 시세 조회 실패: {last_err}")


def fetch_daily(symbol: str = "BTCUSDT", years: float = 8) -> pd.DataFrame:
    """마감된 일봉만 반환한다. index=date(UTC, tz 없음), 컬럼=open/high/low/close/volume."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cursor = now_ms - int(years * 365.25 * DAY_MS)
    rows: List[list] = []
    session = requests.Session()
    while cursor < now_ms:
        batch = _get_klines(session, {"symbol": symbol, "interval": "1d",
                                      "startTime": cursor, "endTime": now_ms, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + DAY_MS
        if nxt <= cursor or len(batch) < 1000:
            break
        cursor = nxt
        time.sleep(0.15)
    if not rows:
        raise RuntimeError("Binance가 캔들 데이터를 반환하지 않았습니다.")

    df = pd.DataFrame(rows).iloc[:, :7]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df = df[df["close_time"].astype("int64") <= now_ms]  # 진행 중인 오늘 캔들 제외
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    cols = ["open", "high", "low", "close", "volume"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df.set_index("date")[cols].dropna()
