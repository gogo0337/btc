#!/usr/bin/env python3
"""전체 기간 캔들 차트: 이동평균선(모델 미적용)과 모델 적용 결과를 함께 표시.

  python chart.py        ->  data/chart.html

- 가격: 2014-09 ~ Binance 상장 전은 Yahoo Finance(BTC-USD), 이후는 Binance(BTCUSDT)
- 이동평균선: 20/50/200일 단순 이동평균 (모델과 무관)
- 모델 적용: 각 날짜의 값은 그 날짜보다 7일(또는 30일) 전에, 그때까지의 데이터만으로 만든 예측이다.
  앙상블 가중치도 30일마다 그 시점까지 결과가 확정된 표본만으로 다시 맞춘다.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import backtest as bt
import rates as rates_mod
from data import fetch_daily
from macro import YAHOO_URL, fetch_yahoo
from models import BASELINE, MIN_HISTORY, MODEL_NAMES, predict_all

HORIZONS = (7, 30)
REFIT_DAYS, MIN_FIT_ROWS, MIN_BAND_ROWS = 30, 200, 250


def fetch_full_history() -> pd.DataFrame:
    """Yahoo(2014-09~)와 Binance(2017-08~)를 이어 붙인 일봉. 거래량은 BTC 단위로 맞춘다."""
    binance = fetch_daily(years=15)
    try:  # Yahoo는 비공식 API라 서버(예: GitHub Actions)에서 막힐 때가 있다 → Binance 구간만으로 계속
        r = requests.get(YAHOO_URL.format(symbol="BTC-USD"), timeout=60,
                         params={"period1": int(pd.Timestamp("2014-01-01").timestamp()),
                                 "period2": int(binance.index[0].timestamp()), "interval": "1d"},
                         headers={"User-Agent": "Mozilla/5.0 (research script)"})
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
    except Exception as e:
        print(f"  (Yahoo 가격을 받지 못해 {binance.index[0].date()} 이후 Binance 일봉만 씁니다: {e})")
        binance.index.name = "date"
        return binance
    q = res["indicators"]["quote"][0]
    old = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                       index=pd.to_datetime(res["timestamp"], unit="s").floor("D")).dropna()
    old = old[~old.index.duplicated(keep="last")]
    old = old[old.index < binance.index[0]]
    old["volume"] = old["volume"] / old["close"]  # 달러 거래대금 → BTC 수량
    df = pd.concat([old, binance])
    df.index.name = "date"
    return df


def walk_forward_ensemble(df: pd.DataFrame, h: int) -> tuple[pd.Series, dict]:
    """목표일 기준 앙상블 예측가(미래 h일 포함)와 전 구간 성적."""
    tbl = bt.prediction_table(df, h, stride=1)
    origin = tbl["origin"].to_numpy()
    logr = np.log(tbl[MODEL_NAMES].to_numpy(dtype=float) / tbl[["base"]].to_numpy(dtype=float))
    k = len(MODEL_NAMES)
    w = np.full(k, 1.0 / k)
    ens = np.empty(len(tbl))
    for i, o in enumerate(origin):
        if (o - origin[0]) % REFIT_DAYS == 0:
            known = tbl[origin + h < o]  # o 시점에 결과가 확정된 표본만
            if len(known) >= MIN_FIT_ROWS:
                w = bt.fit_weights(known)
        ens[i] = tbl["base"].iloc[i] * np.exp(logr[i] @ w)
    w = bt.fit_weights(tbl)
    dates = list(df.index[origin + h])
    values = list(ens)
    for o in range(len(df) - h, len(df)):  # 아직 결과가 없는 최근 시점 → 미래 날짜의 예측
        p = predict_all(df, h, end=o)
        base = float(df["close"].iloc[o])
        values.append(base * float(np.exp(np.array([np.log(p[m] / base) for m in MODEL_NAMES]) @ w)))
        dates.append(df.index[o] + pd.Timedelta(days=h))
    actual = tbl["actual"].to_numpy()
    mape = float(np.mean(np.abs((actual - ens) / actual)) * 100)
    rw = float(np.mean(np.abs((actual - tbl[BASELINE].to_numpy()) / actual)) * 100)
    stats = {"n": int(len(tbl)), "ensemble_mape": round(mape, 2), "rw_mape": round(rw, 2),
             "vs_rw_pct": round((1 - mape / rw) * 100, 1)}
    return pd.Series(values, index=pd.DatetimeIndex(dates)), stats


def walk_forward_band(df: pd.DataFrame, h: int) -> tuple[pd.DataFrame, dict]:
    """목표일 기준 80% 구간(미래 h일 포함)과 전 구간 적중률."""
    close = df["close"].to_numpy(dtype=float)
    logc, n = np.log(close), len(df)
    sigma = pd.Series(np.diff(logc, prepend=np.nan)).ewm(alpha=0.06, min_periods=30).std().to_numpy() * np.sqrt(h)
    ret = np.full(n, np.nan)
    ret[: n - h] = logc[h:] - logc[: n - h]
    z = ret / sigma
    rows, hit = [], []
    for t in range(n):
        past = z[: max(t - h, 0)]
        past = past[np.isfinite(past)]
        if len(past) < MIN_BAND_ROWS or not np.isfinite(sigma[t]):
            continue
        lo, hi = np.quantile(past, [0.1, 0.9]) * sigma[t]
        rows.append((df.index[t] + pd.Timedelta(days=h), close[t] * np.exp(lo), close[t] * np.exp(hi)))
        if t < n - h:
            hit.append(lo <= ret[t] <= hi)
    band = pd.DataFrame(rows, columns=["date", "lo", "hi"]).set_index("date")
    return band, {"n": len(hit), "coverage": round(float(np.mean(hit)) * 100, 1)}


GENESIS = pd.Timestamp("2009-01-03")
PAST_HALVINGS = {"2012 반감기": "2012-11-28", "2016 반감기": "2016-07-09", "2020 반감기": "2020-05-11",
                 "2024 반감기": "2024-04-20"}
HALVING_BLOCKS, BLOCK_MINUTES = 210_000, 10
ETF_DATE = pd.Timestamp("2024-01-10")
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"


def fetch_old_close(before: pd.Timestamp) -> pd.Series:
    """차트 데이터 이전(2010-07~)의 일별 종가 (Coin Metrics). 받지 못하면 빈 Series."""
    try:
        r = requests.get(CM_URL, timeout=60, params={"assets": "btc", "metrics": "PriceUSD", "frequency": "1d",
                                                     "start_time": "2010-07-01", "page_size": 10000})
        r.raise_for_status()
        rows = r.json()["data"]
        s = pd.Series([float(x["PriceUSD"]) for x in rows],
                      index=pd.to_datetime([x["time"] for x in rows]).tz_localize(None).floor("D"))
        s = s[(s > 0) & (s.index < before)]
        return s.reindex(pd.date_range(s.index[0], before - pd.Timedelta(days=1))).ffill()
    except Exception as e:
        print(f"  (Coin Metrics 가격을 받지 못했습니다: {e})")
        return pd.Series(dtype=float)


def pseudo_candles(close: pd.Series) -> pd.DataFrame:
    """종가만 있는 구간을 캔들 모양으로: 시가=전일 종가, 고가·저가=시가와 종가 중 큰 값·작은 값."""
    o = close.shift(1).fillna(close)
    return pd.DataFrame({"open": o, "high": np.maximum(o, close), "low": np.minimum(o, close),
                         "close": close, "volume": np.nan})


def future_halvings() -> dict:
    """다음 두 번의 반감기 예상일. 현재 블록 높이에서 남은 블록 × 10분으로 계산한다."""
    cycle = pd.Timedelta(minutes=HALVING_BLOCKS * BLOCK_MINUTES)
    try:
        height = int(requests.get("https://blockchain.info/q/getblockcount", timeout=30).text)
        nxt = (height // HALVING_BLOCKS + 1) * HALVING_BLOCKS
        d1 = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) + pd.Timedelta(minutes=(nxt - height) * BLOCK_MINUTES)
        print(f"  현재 블록 {height:,}, 다음 반감기 블록 {nxt:,}")
    except Exception as e:
        print(f"  (블록 높이를 받지 못해 4년 주기로 추정: {e})")
        d1 = pd.Timestamp(PAST_HALVINGS["2024 반감기"]) + cycle
    d1 = d1.floor("D")
    d2 = (d1 + cycle).floor("D")
    return {f"{d1.year} 반감기(예상)": d1, f"{d2.year} 반감기(예상)": d2}


def rainbow_fit(price: pd.Series, axis_first: pd.Timestamp) -> dict:
    """레인보우 방식: log10(가격) = a + b·log10(제네시스 이후 일수) 회귀선과 그 위아래 9개 밴드.

    곡선은 전체 기간 데이터로 사후에 맞춘 것이다(과거 구간도 지금의 곡선으로 그린다).
    널리 알려진 레인보우 차트와 같은 방식이지만 계수는 여기서 직접 구한 값이라 선 위치가 조금 다르다.
    """
    t = np.log10((price.index - GENESIS).days.to_numpy(dtype=float))
    y = np.log10(price.to_numpy(dtype=float))
    b, a = np.polyfit(t, y, 1)
    resid = y - (a + b * t)
    offsets = np.linspace(np.quantile(resid, 0.002), np.quantile(resid, 0.998), 10)
    return {"a": round(float(a), 6), "b": round(float(b), 6), "offsets": [round(float(o), 4) for o in offsets],
            "first_day": int((axis_first - GENESIS).days), "fit_from": price.index[0].strftime("%Y-%m-%d"),
            "current_band": int(np.clip(np.searchsorted(offsets, float(resid[-1])), 1, 9))}


def halving_results(close: pd.Series, model_df: pd.DataFrame, ens7: pd.Series, band7: pd.DataFrame,
                    future: dict, axis_first: pd.Timestamp, span: int = 500) -> tuple[list, list, dict]:
    """반감기 ±500일 구간의 가격 흐름과 그 구간 안에서의 모델(7일 예측) 성적. close는 전체 기간 종가."""
    first, last = close.index[0], close.index[-1]
    day = lambda d: int((pd.Timestamp(d) - axis_first).days)
    rnd = lambda v, nd=1: None if v is None or not np.isfinite(v) else round(float(v), nd)
    mc = model_df["close"]
    joined = pd.DataFrame({"close": mc, "ens": ens7, "lo": band7["lo"], "hi": band7["hi"],
                           "rw": mc.shift(7)}).dropna(subset=["close"])
    rows, events = [], []
    for name, d in PAST_HALVINGS.items():
        h = pd.Timestamp(d)
        a, b = h - pd.Timedelta(days=span), h + pd.Timedelta(days=span)
        events.append({"kind": "halving", "label": name, "i": day(h), "i0": day(a), "i1": day(b)})
        if h < first:
            continue
        p0, after = float(close[h]), close[h:min(b, last)]
        w = joined[a:b]
        wb, we = w.dropna(subset=["lo", "hi"]), w.dropna(subset=["ens", "rw"])
        mape = lambda col: float(np.mean(np.abs((we["close"] - we[col]) / we["close"])) * 100) if len(we) > 30 else np.nan
        rows.append({"name": name, "date": d, "halving_price": rnd(p0, 2 if p0 < 100 else 0),
                     "pre_pct": rnd((p0 / float(close[a]) - 1) * 100) if a >= first else None,
                     "post_pct": rnd((float(after.iloc[-1]) / p0 - 1) * 100),
                     "peak_mult": rnd(float(after.max()) / p0, 2), "peak_day": int((after.idxmax() - h).days),
                     "low_mult": rnd(float(after.min()) / p0, 2),
                     "band_coverage": rnd(((wb["close"] >= wb["lo"]) & (wb["close"] <= wb["hi"])).mean() * 100) if len(wb) > 30 else None,
                     "band_n": int(len(wb)), "ens_mape": rnd(mape("ens"), 2), "rw_mape": rnd(mape("rw"), 2)})
    for name, h in future.items():
        events.append({"kind": "halving", "label": name, "i": day(h), "i0": day(h - pd.Timedelta(days=span)),
                       "i1": day(h + pd.Timedelta(days=span))})
        rows.append({"name": name, "date": h.strftime("%Y-%m-%d"), "days_until": int((h - last).days),
                     "window_start": (h - pd.Timedelta(days=span)).strftime("%Y-%m-%d")})
    events.append({"kind": "etf", "label": "ETF 승인", "i": day(ETF_DATE),
                   "i0": day(ETF_DATE - pd.Timedelta(days=100)), "i1": day(ETF_DATE + pd.Timedelta(days=100))})
    # 이전 사이클의 ±500일 경로를 2024 반감기 가격에 맞춰 옮긴 선
    cur = pd.Timestamp(PAST_HALVINGS["2024 반감기"])
    paths = {}
    for name, d in list(PAST_HALVINGS.items())[:-1]:
        h = pd.Timestamp(d)
        if h < first:
            continue
        seg = close[max(h - pd.Timedelta(days=span), first):h + pd.Timedelta(days=span)] / float(close[h])
        paths[name.split()[0] + " 사이클 경로"] = pd.Series(seg.to_numpy() * float(close[cur]), index=cur + (seg.index - h))
    return rows, events, paths


NYFED_URL = "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json"


def fetch_us_rates(index: pd.DatetimeIndex) -> dict:
    """미국 금리(%)를 차트 날짜에 맞춘다. 휴일은 직전 값.

    - 기준금리: 연준 목표 범위의 상단 (뉴욕 연은 공개 API). 받지 못하면 생략.
    - 10년 국채금리: Yahoo Finance ^TNX.
    """
    out = {}
    try:
        r = requests.get(NYFED_URL, timeout=90, params={"startDate": index[0].strftime("%Y-%m-%d"),
                                                        "endDate": index[-1].strftime("%Y-%m-%d")})
        r.raise_for_status()
        rows = [x for x in r.json()["refRates"] if x.get("targetRateTo") is not None]
        s = pd.Series([float(x["targetRateTo"]) for x in rows],
                      index=pd.to_datetime([x["effectiveDate"] for x in rows])).sort_index()
        out["policy"] = s[~s.index.duplicated()].reindex(index, method="ffill")
    except Exception as e:
        print(f"  (기준금리를 받지 못했습니다: {e})")
    try:
        out["us10y"] = fetch_yahoo("^TNX", start="2010-01-01").reindex(index, method="ffill")
    except Exception as e:
        print(f"  (10년 국채금리를 받지 못했습니다: {e})")
    return out


def main() -> None:
    out = Path("data")
    out.mkdir(exist_ok=True)
    print("전체 기간 가격 수집 중...")
    df = fetch_full_history()                     # 모델에 쓰는 실제 일봉 (2014-09~)
    old = fetch_old_close(df.index[0])            # 그 이전은 종가만 있다 (2010-07~)
    plot = pd.concat([pseudo_candles(old), df]) if len(old) else df
    print(f"캔들 {len(df):,}일 ({df.index[0].date()} ~ {df.index[-1].date()}), 종가만 있는 이전 구간 {len(old):,}일")

    future = future_halvings()
    tomorrow = df.index[-1] + pd.Timedelta(days=1)
    axis = plot.index.append(pd.date_range(tomorrow, max(future.values()) + pd.Timedelta(days=500)))
    near = plot.index.append(pd.date_range(tomorrow, periods=max(HORIZONS)))  # 모델 값이 있는 범위
    rnd = lambda s: [None if not np.isfinite(v) else round(float(v), 1 if abs(v) >= 100 else 4) for v in s]
    payload = {"updated_at": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST"),
               "first_date": plot.index[0].strftime("%Y-%m-%d"), "last_date": df.index[-1].strftime("%Y-%m-%d"),
               "real_from": df.index[0].strftime("%Y-%m-%d"), "last_close": round(float(df["close"].iloc[-1]), 1),
               "dates": [d.strftime("%Y-%m-%d") for d in axis], "n_candles": len(plot),
               **{k: rnd(plot[k]) for k in ("open", "high", "low", "close")},
               "ma": {str(w): rnd(plot["close"].rolling(w).mean()) for w in (20, 50, 200)},
               "model": {}}
    keep = {}
    for h in HORIZONS:
        print(f"{h}일 예측을 과거 전 구간에 적용 중...")
        ens, es = walk_forward_ensemble(df, h)
        band, bs = walk_forward_band(df, h)
        keep[h] = (ens, band)
        payload["model"][str(h)] = {"ensemble": rnd(ens.reindex(near)), "lo": rnd(band["lo"].reindex(near)),
                                    "hi": rnd(band["hi"].reindex(near)), "ensemble_stats": es, "band_stats": bs}
        print(f"  앙상블 MAPE {es['ensemble_mape']}% vs Random Walk {es['rw_mape']}% ({es['vs_rw_pct']:+}%), n={es['n']}")
        print(f"  80% 구간 적중률 {bs['coverage']}% (n={bs['n']})")

    print("레인보우 밴드와 반감기 구간 계산 중...")
    try:  # 미국 금리와 '금리 ↔ BTC' 가설 검증
        rt = rates_mod.fetch_rates(cache_dir=out / "cache")
        aligned = rt.reindex(plot.index)
        payload["rates"] = {k: [None if not np.isfinite(v) else round(float(v), 3) for v in aligned[c]]
                            for k, c in (("policy", "policy_upper"), ("us2y", "us2y"), ("us10y", "us10y"))}
        day = lambda d: int((d - axis[0]).days)
        payload["rate_marks"] = [{"i": day(d), "change": round(float(c), 2)} for d, c in rates_mod.policy_moves(aligned).items()]
        payload["rate_regimes"] = rates_mod.segments(rates_mod.expectation_regime(aligned), axis[0], skip=("neutral",))
        payload["rate_stance"] = rates_mod.segments(rates_mod.stance_regime(rt).reindex(plot.index), axis[0])
        payload["rate_study"] = rates_mod.study(plot["close"], rt)
        print(f"  금리: 현재 {payload['rate_study']['now']}")
    except Exception as e:
        print(f"  (금리 자료를 받지 못해 금리 표시를 건너뜁니다: {e})")
    payload["rainbow"] = rainbow_fit(plot["close"], axis[0])
    rows, events, paths = halving_results(plot["close"], df, *keep[7], future, axis[0])
    payload["halvings"], payload["events"] = rows, events
    payload["cycle_paths"] = {k: rnd(v.reindex(near)) for k, v in paths.items()}
    print(f"  레인보우: {payload['rainbow']['fit_from']}부터 적합, 현재 9단계 중 {payload['rainbow']['current_band']}단계")
    print(pd.DataFrame(rows).to_string(index=False))

    html = (Path(__file__).parent / "chart_template.html").read_text(encoding="utf-8")
    (out / "chart.html").write_text(html.replace("__DATA__", json.dumps(payload)), encoding="utf-8")
    print(f"저장: {(out / 'chart.html').resolve()}")


if __name__ == "__main__":
    main()
