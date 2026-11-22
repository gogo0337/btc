"""가격 밖의 정보(파생상품·심리·온체인) 수집과 피처 생성. 모두 무료 공개 API, 키 불필요.

출처
- 펀딩비: Binance 선물 (2019-09~)
- 공포·탐욕 지수: alternative.me (2018-02~)
- 온체인: Coin Metrics 커뮤니티 API — MVRV, 활성 주소, 해시레이트, 거래소 입출금, 수수료

미래 정보 차단
- 모든 피처는 해당 일자 종가 시점에 이미 알 수 있는 값만 쓴다.
- 온체인 값은 다음 날 새벽에 확정되므로 하루 늦춰서 쓴다.
- 미국 현물 ETF 순유입은 무료 API가 없어 넣지 못했다(거래소 순유입이 가장 가까운 대체 지표).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FNG_URL = "https://api.alternative.me/fng/"
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_METRICS = {"CapMVRVCur": "mvrv", "AdrActCnt": "addr", "HashRate": "hash",
              "FlowInExNtv": "flow_in", "FlowOutExNtv": "flow_out", "FeeTotNtv": "fee"}

PRICE_FEATURES = ["ret_7", "ret_30", "ret_90", "vol_30", "vol_ratio", "dist_ma50",
                  "dist_ma200", "drawdown_365", "volume_trend"]
EXTERNAL_FEATURES = ["funding_7", "funding_30", "fng", "fng_chg_7", "mvrv", "netflow_7",
                     "addr_trend", "hash_trend", "fee_trend"]
FEATURE_LABELS = {
    "ret_7": "7일 수익률", "ret_30": "30일 수익률", "ret_90": "90일 수익률",
    "vol_30": "30일 변동성", "vol_ratio": "단기/장기 변동성 비", "dist_ma50": "50일선 대비 위치",
    "dist_ma200": "200일선 대비 위치", "drawdown_365": "1년 고점 대비 하락폭", "volume_trend": "거래량 추세",
    "funding_7": "펀딩비(7일)", "funding_30": "펀딩비(30일)", "fng": "공포·탐욕 지수",
    "fng_chg_7": "공포·탐욕 7일 변화", "mvrv": "MVRV", "netflow_7": "거래소 순유입(7일)",
    "addr_trend": "활성 주소 추세", "hash_trend": "해시레이트 추세", "fee_trend": "수수료 추세",
}


def _get(url: str, params: dict) -> dict | list:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_funding() -> pd.Series:
    rows, cursor = [], int(pd.Timestamp("2019-09-01").timestamp() * 1000)
    while True:
        batch = _get(FUNDING_URL, {"symbol": "BTCUSDT", "startTime": cursor, "limit": 1000})
        rows += batch
        if len(batch) < 1000:
            break
        cursor = int(batch[-1]["fundingTime"]) + 1
        time.sleep(0.2)
    s = pd.Series([float(r["fundingRate"]) for r in rows],
                  index=pd.to_datetime([r["fundingTime"] for r in rows], unit="ms").floor("D"))
    return s.groupby(level=0).mean().rename("funding")


def fetch_fng() -> pd.Series:
    data = _get(FNG_URL, {"limit": 0, "format": "json"})["data"]
    s = pd.Series([float(d["value"]) for d in data],
                  index=pd.to_datetime([int(d["timestamp"]) for d in data], unit="s").floor("D"))
    return s[~s.index.duplicated()].sort_index().rename("fng")


def fetch_onchain() -> pd.DataFrame:
    params = {"assets": "btc", "metrics": ",".join(CM_METRICS), "frequency": "1d",
              "start_time": "2017-01-01", "page_size": 10000}
    rows, url = [], CM_URL
    while url:
        page = _get(url, params)
        rows += page["data"]
        url, params = page.get("next_page_url"), {}
        time.sleep(0.7)
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.floor("D")
    return df[list(CM_METRICS)].apply(pd.to_numeric, errors="coerce").rename(columns=CM_METRICS)


def fetch_external(cache_dir: Path | None = None) -> pd.DataFrame:
    """일자별 외부 데이터. 같은 날(UTC) 다시 실행하면 캐시 파일을 쓴다."""
    cache = (cache_dir / "external.csv") if cache_dir else None
    today = datetime.now(timezone.utc).date()
    if cache and cache.exists() and \
            datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc).date() == today:
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    ext = pd.concat([fetch_funding(), fetch_fng(), fetch_onchain()], axis=1).sort_index()
    ext.index.name = "date"
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        ext.to_csv(cache)
    return ext


def build_features(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """df와 같은 날짜 index의 피처 표. 각 행은 그날 종가 시점에 알 수 있는 값만 담는다."""
    c, lr = df["close"], np.log(df["close"]).diff()
    f = pd.DataFrame(index=df.index)
    for d in (7, 30, 90):
        f[f"ret_{d}"] = np.log(c / c.shift(d))
    f["vol_30"] = lr.rolling(30).std()
    f["vol_ratio"] = lr.rolling(7).std() / lr.rolling(90).std()
    f["dist_ma50"] = np.log(c / c.rolling(50).mean())
    f["dist_ma200"] = np.log(c / c.rolling(200).mean())
    f["drawdown_365"] = np.log(c / c.rolling(365).max())
    f["volume_trend"] = np.log(df["volume"].rolling(7).mean() / df["volume"].rolling(90).mean())

    e = ext.reindex(df.index).ffill(limit=3)
    f["funding_7"] = e["funding"].rolling(7).mean() * 1e4   # bp 단위
    f["funding_30"] = e["funding"].rolling(30).mean() * 1e4
    f["fng"] = e["fng"] / 100.0
    f["fng_chg_7"] = f["fng"].diff(7)

    oc = e[list(CM_METRICS.values())].shift(1)  # 온체인은 하루 늦게 확정
    f["mvrv"] = np.log(oc["mvrv"])
    f["netflow_7"] = (oc["flow_in"] - oc["flow_out"]).rolling(7).sum() / \
        (oc["flow_in"] + oc["flow_out"]).rolling(7).sum()
    f["addr_trend"] = np.log(oc["addr"].rolling(7).mean() / oc["addr"].rolling(90).mean())
    f["hash_trend"] = np.log(oc["hash"].rolling(30).mean() / oc["hash"].rolling(60).mean())
    f["fee_trend"] = np.log(oc["fee"].rolling(7).mean() / oc["fee"].rolling(90).mean())
    return f.replace([np.inf, -np.inf], np.nan)
