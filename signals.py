"""가격 점 예측 대신 쓰는 두 가지 신호.

1) 방향 확률: horizon일 뒤 종가가 지금보다 높을 확률 (로지스틱 회귀, 강한 규제)
2) 변동 구간: horizon일 뒤 종가가 80% 확률로 들어갈 범위 (현재 변동성에 맞춰 폭 조절)

검증 원칙
- 테스트 구간(뒤쪽 25%)에서는 30일마다 모델을 다시 맞추되, 그 시점에 결과가 이미 확정된
  과거 표본만 학습에 쓴다(목표일이 재학습 시점보다 앞선 것만).
- 규제 강도 등 설정값은 미리 고정했고 테스트 결과를 보고 고르지 않았다.
- 테스트 표본은 하루 간격이라 목표 구간이 겹친다. 독립 표본 수는 대략 n / horizon 이다.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_LABELS

REFIT_DAYS = 30
LOGIT_C = 0.1          # 작을수록 규제가 강하다
MIN_TRAIN = 250
LO, HI = 0.10, 0.90    # 80% 구간


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=LOGIT_C, max_iter=1000))


def _nan(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ---------------------------------------------------------------- 방향 확률
def direction(df: pd.DataFrame, feats: pd.DataFrame, h: int, cols: List[str], split: int) -> Dict:
    """테스트 구간 성능과 현재 시점의 상승 확률."""
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    X = feats[cols].to_numpy(dtype=float)
    valid = np.isfinite(X).all(axis=1)
    y = np.full(n, np.nan)
    y[: n - h] = (close[h:] > close[: n - h]).astype(float)
    idx = np.arange(n)

    p, base = np.full(n, np.nan), np.full(n, np.nan)
    for start in range(split, n - h, REFIT_DAYS):
        train = idx[valid & (idx + h < start)]
        if len(train) < MIN_TRAIN or len(np.unique(y[train])) < 2:
            continue
        block = idx[start:min(start + REFIT_DAYS, n - h)]
        block = block[valid[block]]
        if len(block) == 0:
            continue
        p[block] = _model().fit(X[train], y[train]).predict_proba(X[block])[:, 1]
        base[block] = y[train].mean()  # 비교 기준: 과거 상승 비율을 그대로 확률로 쓰는 경우

    m = np.isfinite(p)
    out: Dict = {"n": int(m.sum()), "n_independent": int(m.sum() // h)}
    if m.sum() >= 20:
        yt, pt, bt = y[m], p[m], base[m]
        brier, brier_base = float(np.mean((pt - yt) ** 2)), float(np.mean((bt - yt) ** 2))
        out.update({
            "accuracy": _nan(np.mean((pt > 0.5) == yt) * 100, 1),
            "base_accuracy": _nan(np.mean((bt > 0.5) == yt) * 100, 1),
            "up_rate": _nan(yt.mean() * 100, 1),
            "brier": _nan(brier, 4), "brier_base": _nan(brier_base, 4),
            "brier_skill_pct": _nan((1 - brier / brier_base) * 100, 1),
            "auc": _nan(roc_auc_score(yt, pt), 3) if len(np.unique(yt)) == 2 else None,
        })

    # 현재 시점 예측: 결과가 확정된 모든 표본으로 학습
    train = idx[valid & (idx + h < n)]
    if valid[-1] and len(train) >= MIN_TRAIN and len(np.unique(y[train])) == 2:
        model = _model().fit(X[train], y[train])
        out["p_up"] = _nan(model.predict_proba(X[-1:])[0, 1] * 100, 1)
        out["base_p_up"] = _nan(y[train].mean() * 100, 1)
        z = model[0].transform(X[-1:])[0]
        contrib = model[1].coef_[0] * z  # 표준화 값 × 계수 = 로그 오즈 기여
        top = np.argsort(-np.abs(contrib))[:3]
        out["drivers"] = [{"feature": FEATURE_LABELS.get(cols[i], cols[i]),
                           "effect": "상승 쪽" if contrib[i] > 0 else "하락 쪽",
                           "strength": _nan(abs(contrib[i]), 2)} for i in top]
    return out


# ---------------------------------------------------------------- 변동 구간
def _winkler(y, lo, hi, alpha):
    return (hi - lo) + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)


def interval(df: pd.DataFrame, h: int, split: int) -> Dict:
    """80% 구간. 방식 A: 현재 변동성으로 폭을 조절 / 기준 B: 과거 수익률 분포를 그대로 사용."""
    logc = np.log(df["close"].to_numpy(dtype=float))
    n = len(df)
    lr = pd.Series(np.diff(logc, prepend=np.nan))
    sigma = lr.ewm(alpha=0.06, min_periods=30).std().to_numpy() * np.sqrt(h)
    r = np.full(n, np.nan)
    r[: n - h] = logc[h:] - logc[: n - h]
    z = r / sigma
    qs = [LO, 0.5, HI]

    def quantiles(t: int):
        past = slice(0, max(t - h, 0))  # 목표일이 t보다 앞선 표본만
        zp, rp = z[past], r[past]
        zp, rp = zp[np.isfinite(zp)], rp[np.isfinite(rp)]
        if len(zp) < MIN_TRAIN or not np.isfinite(sigma[t]):
            return None, None
        return np.quantile(zp, qs) * sigma[t], np.quantile(rp, qs)

    rows = []
    for t in range(split, n - h):
        a, b = quantiles(t)
        if a is not None:
            rows.append((r[t], a[0], a[2], b[0], b[2]))
    out: Dict = {"target_coverage": 80, "n": len(rows), "n_independent": len(rows) // h}
    if len(rows) >= 20:
        y, alo, ahi, blo, bhi = np.array(rows).T
        alpha = 1 - (HI - LO)
        wa, wb = _winkler(y, alo, ahi, alpha).mean(), _winkler(y, blo, bhi, alpha).mean()
        out.update({
            "coverage": _nan(np.mean((y >= alo) & (y <= ahi)) * 100, 1),
            "base_coverage": _nan(np.mean((y >= blo) & (y <= bhi)) * 100, 1),
            "avg_width_pct": _nan(np.mean(np.exp(ahi) - np.exp(alo)) * 100, 1),
            "base_avg_width_pct": _nan(np.mean(np.exp(bhi) - np.exp(blo)) * 100, 1),
            "score_vs_base_pct": _nan((1 - wa / wb) * 100, 1),  # 양수면 기준보다 구간 품질이 좋다
        })
    past = z[: n - h]
    past = past[np.isfinite(past)]
    if len(past) >= MIN_TRAIN and np.isfinite(sigma[-1]):
        q = np.quantile(past, qs) * sigma[-1]  # 현재 시점: 확정된 모든 표본 사용
        last = float(df["close"].iloc[-1])
        out.update({"low": _nan(last * np.exp(q[0]), 0), "median": _nan(last * np.exp(q[1]), 0),
                    "high": _nan(last * np.exp(q[2]), 0),
                    "low_pct": _nan((np.exp(q[0]) - 1) * 100, 1), "high_pct": _nan((np.exp(q[2]) - 1) * 100, 1)})
    return out
