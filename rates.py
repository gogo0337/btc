"""미국 금리 수집과 '금리 ↔ 비트코인' 가설 검증.

가설: 금리가 오르면 BTC가 내리고, 금리 인하 기대가 생기면 BTC가 오른다.

자료 (모두 무료, 키 불필요)
- 기준금리: 연준 목표 범위 (뉴욕 연은 공개 API)
- 2년·10년 국채금리: 미국 재무부 일별 수익률 곡선

'금리 인하 기대'의 정의
- 2년 국채금리는 시장이 앞으로 2년간 예상하는 평균 정책금리에 가깝다.
- 2년 금리가 기준금리(목표 범위 중간값)보다 0.25%p 넘게 낮으면 '인하 기대',
  0.25%p 넘게 높으면 '인상 기대', 그 사이는 '중립'으로 본다.
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

NYFED_URL = "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json"
TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
                "{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
UA = {"User-Agent": "Mozilla/5.0 (research script)"}
EXPECT_GAP = 0.25  # %p
REGIME_LABEL = {"cut": "인하 기대", "neutral": "중립", "hike": "인상 기대"}
HIGH_RATE = 2.0     # 기준금리(목표 상단)가 이 값 이상이면 '고금리'
TREND_DAYS = 180    # 이 기간 동안의 기준금리 변화로 '기조'를 정한다
STANCE_LABEL = {"low_down": "저금리 유지·하락", "low_up": "저금리 상승",
                "high_up": "고금리 유지·상승", "high_down": "고금리 하락"}


def fetch_rates(start: str = "2010-07-01", cache_dir: Path | None = None) -> pd.DataFrame:
    """일별(달력일, 휴일은 직전 값) 금리 표: policy_upper, policy_mid, us2y, us10y (%)."""
    cache = (cache_dir / "rates.csv") if cache_dir else None
    today = datetime.now(timezone.utc).date()
    if cache and cache.exists() and datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc).date() == today:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    r = requests.get(NYFED_URL, timeout=90, params={"startDate": start, "endDate": today.strftime("%Y-%m-%d")}, headers=UA)
    r.raise_for_status()
    rows = [x for x in r.json()["refRates"] if x.get("targetRateTo") is not None]
    idx = pd.to_datetime([x["effectiveDate"] for x in rows])
    pol = pd.DataFrame({"policy_upper": [float(x["targetRateTo"]) for x in rows],
                        "policy_lower": [float(x.get("targetRateFrom") or 0.0) for x in rows]}, index=idx).sort_index()
    pol = pol[~pol.index.duplicated()]

    frames = []
    for y in range(pd.Timestamp(start).year, today.year + 1):
        t = requests.get(TREASURY_URL.format(y=y), timeout=60, headers=UA)
        t.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(t.text)))
        time.sleep(0.3)
    tr = pd.concat(frames)
    tr.index = pd.to_datetime(tr["Date"], format="%m/%d/%Y")
    tr = tr.sort_index()[["2 Yr", "10 Yr"]].rename(columns={"2 Yr": "us2y", "10 Yr": "us10y"}).apply(pd.to_numeric, errors="coerce")

    days = pd.date_range(start, pd.Timestamp(today) - pd.Timedelta(days=1))
    out = pd.concat([pol.reindex(days, method="ffill"), tr.reindex(days, method="ffill")], axis=1)
    out["policy_mid"] = (out["policy_upper"] + out["policy_lower"]) / 2
    out = out.drop(columns="policy_lower")
    out.index.name = "date"
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache)
    return out


def expectation_regime(rates: pd.DataFrame) -> pd.Series:
    gap = rates["us2y"] - rates["policy_mid"]
    return pd.Series(np.where(gap < -EXPECT_GAP, "cut", np.where(gap > EXPECT_GAP, "hike", "neutral")),
                     index=rates.index).where(gap.notna())


def stance_regime(rates: pd.DataFrame) -> pd.Series:
    """금리 수준(고·저)과 기조(최근 180일 변화)로 나눈 네 가지 국면."""
    pol = rates["policy_upper"]
    chg = pol - pol.shift(TREND_DAYS)
    high = pol >= HIGH_RATE
    k = np.where(high & (chg >= 0), "high_up", np.where(high, "high_down",
                 np.where(chg > 0, "low_up", "low_down")))
    return pd.Series(k, index=rates.index).where(chg.notna())


def segments(regime: pd.Series, axis_first: pd.Timestamp, skip: tuple = ()) -> list:
    """연속된 같은 국면을 차트용 구간 [{i0, i1, k}]으로 묶는다."""
    out, start, prev = [], None, None
    day = lambda d: int((d - axis_first).days)
    for d, k in regime.dropna().items():
        if start is None or k != start[1] or (d - prev).days > 1:
            if start is not None and start[1] not in skip:
                out.append({"i0": day(start[0]), "i1": day(prev) + 1, "k": start[1]})
            start = (d, k)
        prev = d
    if start is not None and start[1] not in skip:
        out.append({"i0": day(start[0]), "i1": day(prev) + 1, "k": start[1]})
    return out


def policy_moves(rates: pd.DataFrame) -> pd.Series:
    """기준금리가 바뀐 날과 변화폭(%p)."""
    d = rates["policy_upper"].diff()
    return d[d.abs() > 1e-9]


def _r(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def study(close: pd.Series, rates: pd.DataFrame) -> dict:
    """가설 검증. close: BTC 일별 종가(달력일)."""
    df = pd.DataFrame({"btc": close}).join(rates, how="inner").dropna(subset=["btc", "us2y", "policy_mid"])
    out: dict = {"from": df.index[0].strftime("%Y-%m-%d"), "to": df.index[-1].strftime("%Y-%m-%d")}

    # ---- 1) 주간: 2년 금리 변화와 BTC 수익률
    w = df.resample("W-FRI").last()
    wk = pd.DataFrame({"d2": w["us2y"].diff(), "ret": np.log(w["btc"]).diff()})
    wk["next4"] = np.log(w["btc"].shift(-4) / w["btc"])
    wk = wk.dropna(subset=["d2", "ret"])
    rows = []
    for label, sub in (("전체 기간", wk), ("2018년 이후", wk["2018":]), ("ETF 승인 이후", wk["2024-01-10":])):
        up, dn = sub[sub["d2"] > 0.02], sub[sub["d2"] < -0.02]
        big_up, big_dn = sub[sub["d2"] > 0.10], sub[sub["d2"] < -0.10]
        fwd = sub.dropna(subset=["next4"])
        fu, fd = fwd[fwd["d2"] > 0.10], fwd[fwd["d2"] < -0.10]
        rows.append({"period": label, "weeks": int(len(sub)), "base_up": _r((sub["ret"] > 0).mean() * 100),
                     "corr_same_week": _r(sub["d2"].corr(sub["ret"]), 2),
                     "down_when_rate_up": _r((up["ret"] < 0).mean() * 100), "n_rate_up": int(len(up)),
                     "up_when_rate_down": _r((dn["ret"] > 0).mean() * 100), "n_rate_down": int(len(dn)),
                     "down_when_big_up": _r((big_up["ret"] < 0).mean() * 100), "n_big_up": int(len(big_up)),
                     "up_when_big_down": _r((big_dn["ret"] > 0).mean() * 100), "n_big_down": int(len(big_dn)),
                     "next4_up_after_big_up": _r((fu["next4"] > 0).mean() * 100), "n_fu": int(len(fu)),
                     "next4_up_after_big_down": _r((fd["next4"] > 0).mean() * 100), "n_fd": int(len(fd)),
                     "next4_base_up": _r((fwd["next4"] > 0).mean() * 100)})
    out["weekly"] = rows

    # ---- 2) 기대 국면별 이후 수익률
    reg = expectation_regime(df)
    fwd = {h: np.log(df["btc"].shift(-h) / df["btc"]) for h in (30, 90)}
    rows = []
    for label, mask in (("전체 기간", slice(None)), ("2018년 이후", slice("2018", None))):
        rg = reg[mask]
        episodes = (rg != rg.shift()).cumsum()
        for k in ("cut", "neutral", "hike"):
            m = rg == k
            row = {"period": label, "regime": REGIME_LABEL[k], "days": int(m.sum()),
                   "share": _r(m.mean() * 100), "episodes": int(episodes[m].nunique())}
            for h in (30, 90):
                f = fwd[h][mask][m].dropna()
                row |= {f"up_{h}": _r((f > 0).mean() * 100) if len(f) else None,
                        f"median_{h}": _r((np.exp(f.median()) - 1) * 100) if len(f) else None,
                        f"n_indep_{h}": int(len(f) // h)}
            rows.append(row)
        f = fwd[90][mask].dropna()
        rows.append({"period": label, "regime": "기준(모든 날)", "days": int(len(rg)), "share": 100.0, "episodes": None,
                     "up_30": _r((fwd[30][mask].dropna() > 0).mean() * 100),
                     "median_30": _r((np.exp(fwd[30][mask].dropna().median()) - 1) * 100),
                     "n_indep_30": int(len(fwd[30][mask].dropna()) // 30),
                     "up_90": _r((f > 0).mean() * 100), "median_90": _r((np.exp(f.median()) - 1) * 100),
                     "n_indep_90": int(len(f) // 90)})
    out["regimes"] = rows

    # ---- 3) 실제 인상·인하 이후
    moves = policy_moves(df)
    ev = []
    for d, chg in moves.items():
        p0 = df["btc"][d]
        row = {"date": d.strftime("%Y-%m-%d"), "change": _r(chg, 2), "rate_after": _r(df["policy_upper"][d], 2)}
        for h in (30, 90):
            t = d + pd.Timedelta(days=h)
            row[f"ret_{h}"] = _r((df["btc"][t] / p0 - 1) * 100) if t in df.index else None
        ev.append(row)
    out["moves"] = ev
    summ = []
    for label, sel in (("인상", [e for e in ev if e["change"] > 0]), ("인하", [e for e in ev if e["change"] < 0])):
        for h in (30, 90):
            vals = [e[f"ret_{h}"] for e in sel if e[f"ret_{h}"] is not None]
            hit = [v < 0 for v in vals] if label == "인상" else [v > 0 for v in vals]
            summ.append({"kind": label, "horizon": h, "events": len(vals),
                         "hit_pct": _r(np.mean(hit) * 100) if vals else None,
                         "median_ret": _r(float(np.median(vals))) if vals else None})
    out["moves_summary"] = summ
    # ---- 4) 금리 수준·기조 국면별 BTC
    st = stance_regime(df).dropna()
    monthly = np.log(df["btc"].resample("ME").last()).diff().dropna()
    mst = st.resample("ME").last().reindex(monthly.index)
    ep_id = (st != st.shift()).cumsum()
    rows = []
    for k, label in STANCE_LABEL.items():
        m = st == k
        mm = monthly[mst == k]
        eps = []
        for _, g in df.loc[st.index][m].groupby(ep_id[m]):
            if len(g) >= 30:
                eps.append({"from": g.index[0].strftime("%Y-%m-%d"), "to": g.index[-1].strftime("%Y-%m-%d"),
                            "ret": _r((g["btc"].iloc[-1] / g["btc"].iloc[0] - 1) * 100, 0)})
        rows.append({"key": k, "regime": label, "days": int(m.sum()), "months": int(len(mm)),
                     "up_months": _r((mm > 0).mean() * 100) if len(mm) else None,
                     "median_month": _r((np.exp(mm.median()) - 1) * 100) if len(mm) else None,
                     "episodes": eps, "episodes_up": int(sum(e["ret"] > 0 for e in eps))})
    out["stance"] = {"rows": rows, "base_up_months": _r((monthly[mst.notna()] > 0).mean() * 100),
                     "high_rate": HIGH_RATE, "trend_days": TREND_DAYS,
                     "now": STANCE_LABEL.get(st.iloc[-1])}
    out["now"] = {"policy_upper": _r(df["policy_upper"].iloc[-1], 2), "us2y": _r(df["us2y"].iloc[-1], 2),
                  "us10y": _r(df["us10y"].iloc[-1], 2), "gap": _r(df["us2y"].iloc[-1] - df["policy_mid"].iloc[-1], 2),
                  "regime": REGIME_LABEL.get(reg.iloc[-1])}
    return out


if __name__ == "__main__":
    from chart import fetch_full_history, fetch_old_close, pseudo_candles
    df = fetch_full_history()
    close = pd.concat([fetch_old_close(df.index[0]), df["close"]])
    rates = fetch_rates(cache_dir=Path("data/cache"))
    res = study(close, rates)
    pd.set_option("display.width", 250)
    print(res["from"], "~", res["to"], "| 현재:", res["now"])
    print("\n[주간: 2년 금리 변화와 BTC]\n", pd.DataFrame(res["weekly"]).T.to_string())
    print("\n[기대 국면별 이후 수익률]\n", pd.DataFrame(res["regimes"]).to_string(index=False))
    print("\n[실제 인상·인하 이후]\n", pd.DataFrame(res["moves_summary"]).to_string(index=False))
    print(pd.DataFrame(res["moves"]).to_string(index=False))
