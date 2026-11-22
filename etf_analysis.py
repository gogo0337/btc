#!/usr/bin/env python3
"""ETF 승인 전후 비교, 'ETF가 없었다면'의 가격 추정, 반감기 사이클 겹쳐 보기.

  python etf_analysis.py        ->  data/etf_report.html, data/etf_analysis.json

방법과 한계
- 'ETF가 없었다면'은 실제로 일어나지 않은 일이라 정답이 없다. 여기서는 ETF 이전 기간에
  BTC 주간 수익률과 거시 지표(나스닥·금·달러·미국 10년 금리)의 관계를 맞춘 뒤 그 관계를
  ETF 이후로 연장한다. 실제 가격과의 차이를 ETF 효과 추정치로 본다.
- 이 방법이 얼마나 틀리는지는 '가짜 시점 검증'으로 잰다. ETF가 없던 과거 시점을 승인일인 척하고
  같은 계산을 했을 때 생기는 차이가, 이 방법의 기본 오차다.
- ETF 일별 순유입 데이터는 무료 공개 API가 없어 쓰지 못했다.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import etf_flows
from macro import fetch_all

ETF_DATE = pd.Timestamp("2024-01-10")      # 미국 현물 ETF 승인
FILING_DATE = pd.Timestamp("2023-06-15")   # 블랙록 신청 (기대감이 가격에 반영되기 시작한 시점)
HALVINGS = {"2016년 반감기": "2016-07-09", "2020년 반감기": "2020-05-11", "2024년 반감기(현재)": "2024-04-20"}
MACRO = ["nasdaq", "gold", "dxy", "us10y"]
PLACEBO_DATES = ["2018-01-05", "2019-01-04", "2020-01-03", "2021-01-01"]
RNG = np.random.default_rng(0)


def r2(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ------------------------------------------------------------ 1. 전후 비교
def weekly_returns(px: pd.DataFrame) -> pd.DataFrame:
    w = px.resample("W-FRI").last()
    r = np.log(w).diff()
    r["us10y"] = w["us10y"].diff()  # 금리는 수준 차이(%p)
    return r


def period_stats(px: pd.DataFrame, wr: pd.DataFrame, start, end, label: str) -> dict:
    p = px["btc"][start:end].dropna()
    lr = np.log(p).diff().dropna()
    wk = lr[lr.index.day_of_week < 5].std()
    we = lr[lr.index.day_of_week >= 5].std()
    w = wr[start:end].dropna(subset=["btc"] + MACRO)
    return {"label": label, "start": p.index[0].strftime("%Y-%m-%d"), "end": p.index[-1].strftime("%Y-%m-%d"),
            "vol_ann": r2(lr.std() * np.sqrt(365) * 100, 1), "mean_abs": r2(lr.abs().mean() * 100),
            "big_days_pct": r2((lr.abs() > 0.05).mean() * 100, 1),
            "max_drawdown": r2((p / p.cummax() - 1).min() * 100, 0),
            "weekend_vs_weekday": r2(we / wk, 2),
            **{f"corr_{m}": r2(w["btc"].corr(w[m]), 2) for m in MACRO}}


# ------------------------------------------------------------ 2. ETF 없는 가격 추정
def counterfactual(px: pd.DataFrame, wr: pd.DataFrame, t0: pd.Timestamp, factors: list[str],
                   use_alpha: bool, horizon: int | None = None, sims: int = 1000) -> dict | None:
    """t0 이전 관계를 t0 이후로 연장. use_alpha=False면 BTC 고유의 평균 상승률(절편)을 0으로 둔다."""
    data = wr.dropna(subset=["btc"] + factors)
    train, post = data[data.index < t0], data[data.index >= t0]
    if horizon:
        post = post.iloc[:horizon]
    if len(train) < 100 or len(post) < 10:
        return None
    X = np.column_stack([np.ones(len(train)), train[factors].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, train["btc"].to_numpy(), rcond=None)
    resid = train["btc"].to_numpy() - X @ coef
    pred = post[factors].to_numpy() @ coef[1:] + (coef[0] if use_alpha else 0.0)

    # BTC 고유 움직임(잔차)을 4주 묶음으로 다시 뽑아 80% 범위를 만든다
    n, block = len(post), 4
    starts = RNG.integers(0, len(resid) - block, size=(sims, n // block + 1))
    noise = np.concatenate([resid[s:s + block] for s in starts.ravel()]).reshape(sims, -1)[:, :n]
    noise -= resid.mean()
    paths = np.cumsum(pred) + np.cumsum(noise, axis=1)
    base = float(px["btc"][:train.index[-1]].iloc[-1])
    actual = base * np.exp(np.cumsum(post["btc"].to_numpy()))
    lo, mid, hi = (base * np.exp(np.quantile(paths, q, axis=0)) for q in (0.1, 0.5, 0.9))
    ratio = actual / mid
    return {"t0": t0.strftime("%Y-%m-%d"), "factors": factors, "use_alpha": use_alpha,
            "weeks": n, "train_weeks": len(train), "r_squared": r2(1 - resid.var() / train["btc"].var(), 3),
            "alpha_annual_pct": r2((np.exp(coef[0] * 52) - 1) * 100, 0),
            "betas": {f: r2(b, 2) for f, b in zip(factors, coef[1:])},
            "actual_end": r2(actual[-1], 0), "cf_end": r2(mid[-1], 0),
            "cf_low": r2(lo[-1], 0), "cf_high": r2(hi[-1], 0),
            "ratio_end": r2(ratio[-1], 2), "ratio_min": r2(ratio.min(), 2), "ratio_max": r2(ratio.max(), 2),
            "dates": [d.strftime("%Y-%m-%d") for d in post.index],
            "actual": [r2(v, 0) for v in actual], "cf": [r2(v, 0) for v in mid],
            "lo": [r2(v, 0) for v in lo], "hi": [r2(v, 0) for v in hi]}


# ------------------------------------------------------------ 3. 반감기 사이클
def cycles(px: pd.DataFrame, span: int = 1300) -> dict:
    btc = px["btc"].dropna()
    last = btc.index[-1]
    day_now = (last - pd.Timestamp(HALVINGS["2024년 반감기(현재)"])).days
    out = {"day_now": day_now, "series": {}, "rows": []}
    for name, d in HALVINGS.items():
        h = pd.Timestamp(d)
        s = btc[h:h + pd.Timedelta(days=span)]
        if s.empty:
            continue
        mult = s / s.iloc[0]
        out["series"][name] = {"days": [(i - h).days for i in mult.index[::3]],
                               "multiple": [r2(v, 3) for v in mult.iloc[::3]]}
        at = mult.get(h + pd.Timedelta(days=day_now))
        row = {"cycle": name, "halving_price": r2(s.iloc[0], 0), "multiple_now": r2(at, 2) if at is not None else None}
        for fwd in (30, 90, 180):
            later = mult.get(h + pd.Timedelta(days=day_now + fwd))
            row[f"fwd_{fwd}"] = r2((later / at - 1) * 100, 1) if at is not None and later is not None else None
        out["rows"].append(row)
    cur = float(btc[pd.Timestamp(HALVINGS["2024년 반감기(현재)"])])
    for row in out["rows"]:  # 이전 사이클과 같은 배수였다면 지금 가격
        row["implied_price_now"] = r2(cur * row["multiple_now"], 0) if row["multiple_now"] else None
    return out


def main() -> None:
    out = Path("data")
    px = fetch_all(out / "cache")
    wr = weekly_returns(px)
    last = px["btc"].dropna().index[-1]
    same_len = last - ETF_DATE

    stats = [period_stats(px, wr, px.index[0], ETF_DATE - pd.Timedelta(days=1), "ETF 이전 전체"),
             period_stats(px, wr, "2020-01-10", pd.Timestamp("2020-01-10") + same_len, "이전 사이클의 같은 위치"),
             period_stats(px, wr, FILING_DATE, ETF_DATE - pd.Timedelta(days=1), "신청~승인 사이 (기대감 구간)"),
             period_stats(px, wr, ETF_DATE, last, "ETF 이후")]

    real = []
    for label, t0, factors, alpha in [
            ("기본: 승인일 기준, BTC 고유 상승률 유지", ETF_DATE, MACRO, True),
            ("승인일 기준, 고유 상승률 0 가정", ETF_DATE, MACRO, False),
            ("블랙록 신청일 기준, 고유 상승률 유지", FILING_DATE, MACRO, True),
            ("블랙록 신청일 기준, 고유 상승률 0 가정", FILING_DATE, MACRO, False),
            ("승인일 기준, 이더리움 추가 (참고)", ETF_DATE, MACRO + ["eth"], True)]:
        cf = counterfactual(px, wr, t0, factors, alpha)
        if cf:
            real.append({"label": label, **cf})

    horizon = real[0]["weeks"]
    placebo = []
    for d in PLACEBO_DATES:
        t0 = pd.Timestamp(d)
        h = min(horizon, int((ETF_DATE - t0).days // 7))  # ETF 승인 이후 구간은 섞지 않는다
        for alpha in (True, False):
            cf = counterfactual(px, wr, t0, MACRO, alpha, horizon=h)
            if cf:
                placebo.append({k: v for k, v in cf.items() if k not in ("dates", "actual", "cf", "lo", "hi")}
                               | {"label": f"{d} ({'상승률 유지' if alpha else '상승률 0'})"})

    pl = np.abs(np.log([p["ratio_end"] for p in placebo]))
    re = np.abs(np.log([v["ratio_end"] for v in real[:4]]))
    verdict = {"placebo_typical_factor": r2(np.exp(np.median(pl)), 2), "placebo_worst_factor": r2(np.exp(pl.max()), 2),
               "real_min_ratio": min(v["ratio_end"] for v in real[:4]), "real_max_ratio": max(v["ratio_end"] for v in real[:4]),
               "distinguishable": bool(re.min() > pl.max())}

    payload = {"updated_at": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST"),
               "last_date": last.strftime("%Y-%m-%d"), "current_price": r2(px["btc"].dropna().iloc[-1], 0),
               "etf_date": ETF_DATE.strftime("%Y-%m-%d"), "filing_date": FILING_DATE.strftime("%Y-%m-%d"),
               "stats": stats, "counterfactual": real, "placebo": placebo, "verdict": verdict, "cycles": cycles(px)}

    csv = etf_flows.find_csv()  # Farside 표를 저장한 파일이 있을 때만 순유입 분석을 한다
    if csv:
        from data import fetch_daily
        from features import build_features, fetch_external
        df = fetch_daily(years=8)
        payload["flows"] = etf_flows.study(df, build_features(df, fetch_external(out / "cache")),
                                           etf_flows.load_flows(csv))
    (out / "etf_analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    html = (Path(__file__).parent / "etf_report_template.html").read_text(encoding="utf-8")
    (out / "etf_report.html").write_text(
        html.replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")

    pd.set_option("display.width", 220)
    print(f"기준일 {payload['last_date']}, BTC ${payload['current_price']:,.0f}\n\n=== 1. ETF 전후 비교 ===")
    print(pd.DataFrame(stats).to_string(index=False))
    print("\n=== 2. 'ETF가 없었다면' 추정 (실제/추정 배수: 1보다 크면 실제가 더 높음) ===")
    cols = ["label", "weeks", "r_squared", "alpha_annual_pct", "actual_end", "cf_end", "cf_low", "cf_high", "ratio_end"]
    print(pd.DataFrame(real)[cols].to_string(index=False))
    print("\n=== 가짜 시점 검증 (ETF가 없던 때에 같은 계산을 한 결과) ===")
    print(pd.DataFrame(placebo)[cols].to_string(index=False))
    print("\n판정:", verdict)
    print("\n=== 3. 반감기 사이클 (반감기 후", payload["cycles"]["day_now"], "일째) ===")
    print(pd.DataFrame(payload["cycles"]["rows"]).to_string(index=False))
    if "flows" in payload:
        f = payload["flows"]
        print(f"\n=== 4. ETF 순유입 ({f['first_date']} ~ {f['last_date']}, 누적 ${f['cum_usd_bn']}bn) ===")
        print("주간 상관:", f["weekly"])
        print(pd.DataFrame(f["quintiles"]).to_string(index=False))
        print(pd.DataFrame(f["predict"]["rows"]).to_string(index=False))
    print(f"\n저장: {(out / 'etf_report.html').resolve()}")


if __name__ == "__main__":
    main()
