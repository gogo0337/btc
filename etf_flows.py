"""미국 현물 BTC ETF 일별 순유입 (Farside 표를 복사해 저장한 파일) 읽기와 검증.

파일 형식: 한 칸이 한 줄. 날짜 줄 다음에 ETF별 값과 Total이 이어진다. (123.4)는 음수, '-'는 값 없음.
시점 처리: 미국 거래일 D의 순유입은 D일 밤(UTC로는 다음 날 새벽)에 공개되므로,
UTC 기준 D일 종가 시점에는 아직 모른다. 그래서 피처는 하루 늦춰서 쓴다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import signals
from features import EXTERNAL_FEATURES, PRICE_FEATURES

FLOW_FEATURES = ["flow_1", "flow_5", "flow_20", "flow_streak"]
FLOW_LABELS = {"flow_1": "ETF 순유입(전일)", "flow_5": "ETF 순유입(5일)", "flow_20": "ETF 순유입(20일)",
               "flow_streak": "ETF 연속 유입/유출 일수"}
_DATE = re.compile(r"^\d{1,2} [A-Za-z]{3} \d{4}$")


def find_csv() -> Path | None:
    here = Path(__file__).parent
    for p in (here / "data" / "Farside_alldata.csv", here / "Farside_alldata.csv", here.parent / "Farside_alldata.csv"):
        if p.exists():
            return p
    return None


def _num(cell: str) -> float:
    cell = cell.strip().replace(",", "")
    if cell in ("", "-"):
        return np.nan
    neg = cell.startswith("(") and cell.endswith(")")
    return -float(cell.strip("()")) if neg else float(cell)


def load_flows(path: Path) -> pd.DataFrame:
    """index=미국 거래일, 컬럼=ETF별 순유입과 Total (백만 달러)."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines()]
    start = lines.index("Date")
    end = next(i for i in range(start + 1, len(lines)) if _DATE.match(lines[i]))
    cols = [c for c in lines[start + 1:end] if c]
    rows, i = {}, end
    while i < len(lines):
        if _DATE.match(lines[i]):
            rows[pd.to_datetime(lines[i], format="%d %b %Y")] = [_num(c) for c in lines[i + 1:i + 1 + len(cols)]]
            i += 1 + len(cols)
        else:
            i += 1
    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols).sort_index()
    df.index.name = "date"
    return df


def flow_features(flows: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """price(일별 BTC 종가)의 날짜에 맞춘 피처. 단위는 천 BTC, 하루 늦춘 값."""
    kbtc = (flows["Total"] * 1e6 / price.reindex(flows.index)).dropna() / 1e3
    sign = np.sign(kbtc)
    streak = sign.groupby((sign != sign.shift()).cumsum()).cumsum()
    biz = pd.DataFrame({"flow_1": kbtc, "flow_5": kbtc.rolling(5).sum(), "flow_20": kbtc.rolling(20).sum(),
                        "flow_streak": streak})
    return biz.reindex(price.index).ffill(limit=4).shift(1)


def study(df: pd.DataFrame, feats: pd.DataFrame, flows: pd.DataFrame) -> dict:
    """df: Binance 일봉, feats: features.build_features 결과."""
    r = lambda x, nd=2: None if x is None or not np.isfinite(x) else round(float(x), nd)
    price = df["close"]
    total = flows["Total"].dropna()
    out = {"first_date": total.index[0].strftime("%Y-%m-%d"), "last_date": total.index[-1].strftime("%Y-%m-%d"),
           "days": int(len(total)), "cum_usd_bn": r(total.sum() / 1e3, 1),
           "last_row_incomplete": bool(flows.drop(columns="Total").iloc[-1].isna().any()),
           "by_year": [{"year": int(y), "net_usd_bn": r(s.sum() / 1e3, 1), "inflow_days_pct": r((s > 0).mean() * 100, 0)}
                       for y, s in total.groupby(total.index.year)]}

    # 1) 무엇이 먼저인가: 주 단위로 순유입과 가격 수익률의 앞뒤 관계
    wk_flow = total.resample("W-FRI").sum()
    wk_ret = np.log(price.resample("W-FRI").last()).diff().reindex(wk_flow.index)
    w = pd.DataFrame({"flow": wk_flow, "ret": wk_ret}).dropna()
    out["weekly"] = {"n": int(len(w)),
                     "same_week": r(w["flow"].corr(w["ret"])),
                     "flow_then_price": r(w["flow"].corr(w["ret"].shift(-1))),   # 이번 주 유입 → 다음 주 가격
                     "price_then_flow": r(w["ret"].corr(w["flow"].shift(-1)))}   # 이번 주 가격 → 다음 주 유입
    q = pd.qcut(w["flow"], 5, labels=["유출 최대", "유출", "중간", "유입", "유입 최대"])
    nxt = w["ret"].shift(-1)
    out["quintiles"] = [{"group": str(g), "weeks": int((q == g).sum()),
                         "avg_flow_usd_bn": r(w["flow"][q == g].mean() / 1e3, 2),
                         "same_week_ret": r(w["ret"][q == g].mean() * 100, 1),
                         "next_week_ret": r(nxt[q == g].mean() * 100, 1),
                         "next_week_up_pct": r((nxt[q == g].dropna() > 0).mean() * 100, 0)} for g in q.cat.categories]

    # 2) 예측 피처로서의 가치: ETF 시대 안에서만, 같은 표본으로 피처 묶음을 비교
    ff = flow_features(flows, price)
    allf = feats.join(ff)
    allf[~allf[PRICE_FEATURES + EXTERNAL_FEATURES + FLOW_FEATURES].notna().all(axis=1)] = np.nan
    era = allf.index >= allf.dropna().index[0]
    d2, f2 = df[era], allf[era]
    split = int(len(d2) * 0.6)
    sets = {"가격만": PRICE_FEATURES, "가격 + ETF 순유입": PRICE_FEATURES + FLOW_FEATURES,
            "가격 + 펀딩비·심리·온체인": PRICE_FEATURES + EXTERNAL_FEATURES,
            "전체": PRICE_FEATURES + EXTERNAL_FEATURES + FLOW_FEATURES}
    signals.FEATURE_LABELS.update(FLOW_LABELS)
    out["predict"] = {"era_start": d2.index[0].strftime("%Y-%m-%d"), "era_days": int(len(d2)),
                      "test_start": d2.index[split].strftime("%Y-%m-%d"), "rows": []}
    for h in (7, 30):
        for name, cols in sets.items():
            res = signals.direction(d2, f2, h, cols, split)
            out["predict"]["rows"].append({"horizon": h, "features": name, **{k: res.get(k) for k in (
                "n", "n_independent", "accuracy", "base_accuracy", "brier_skill_pct", "auc", "p_up", "base_p_up")}})
    return out
