"""시간순 검증/테스트 분리 백테스트 + 기간별 앙상블 가중치.

원칙
- 가중치와 단일 모델 선택은 검증(VALIDATION) 구간에서만 정한다.
- 성능 보고는 그 뒤의 테스트(TEST) 구간에서만 한다. 검증 구간의 목표일은 분할 시점 이전에 끝난다.
- 모든 지표는 Random Walk(현재가 유지)와 나란히 보고한다.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models import BASELINE, MIN_HISTORY, MODEL_NAMES, predict_all

ENSEMBLE = "Ensemble"
SHRINK_K = 30  # 표본이 이 값보다 적을수록 균등 가중치 쪽으로 당긴다
PRUNE_RATIO = 1.25  # Random Walk 오차의 이 배수를 넘는 모델은 해당 기간 앙상블에서 제외


def prediction_table(df: pd.DataFrame, horizon: int, stride: int | None = None) -> pd.DataFrame:
    """예측 시점(origin)마다 모든 모델의 예측가와 실제가를 모은 표."""
    stride = stride or min(horizon, 30)  # 30일 초과 기간은 목표 구간이 서로 겹친다
    close = df["close"].to_numpy(dtype=float)
    rows = []
    for origin in range(MIN_HISTORY, len(df) - horizon, stride):
        row = {"origin": origin, "date": df.index[origin], "base": close[origin],
               "actual": close[origin + horizon]}
        row.update(predict_all(df, horizon, end=origin))
        rows.append(row)
    return pd.DataFrame(rows)


def _log_returns(tbl: pd.DataFrame, cols: List[str]) -> np.ndarray:
    return np.log(tbl[cols].to_numpy(dtype=float) / tbl[["base"]].to_numpy(dtype=float))


def fit_weights(tbl: pd.DataFrame) -> np.ndarray:
    """로그수익률 제곱오차를 최소화하는 가중치(합 1, 0 이상) → 균등 가중치 쪽으로 축소."""
    k = len(MODEL_NAMES)
    n = len(tbl)
    if n < 3:
        return np.full(k, 1.0 / k)
    # 같은 구간에서 Random Walk보다 오차가 PRUNE_RATIO배 넘게 큰 모델은 앙상블에서 뺀다.
    # (예: 장기 구간의 Momentum·Trend·Hybrid). 판단은 가중치를 맞추는 구간 안에서만 한다.
    actual = tbl["actual"].to_numpy(dtype=float)
    mape = {m: float(np.mean(np.abs((actual - tbl[m].to_numpy(dtype=float)) / actual))) for m in MODEL_NAMES}
    keep = np.array([mape[m] <= PRUNE_RATIO * mape[BASELINE] for m in MODEL_NAMES])
    uniform = keep / keep.sum()
    P = _log_returns(tbl, MODEL_NAMES)
    y = np.log(actual / tbl["base"].to_numpy(dtype=float))
    res = minimize(lambda w: float(np.mean((y - P @ w) ** 2)), uniform, method="SLSQP",
                   bounds=[(0.0, 1.0 if kp else 0.0) for kp in keep],
                   constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    w = np.clip(res.x, 0.0, None) * keep if res.success else uniform
    w = w / w.sum()
    lam = n / (n + SHRINK_K)
    return lam * w + (1.0 - lam) * uniform


def ensemble_price(tbl: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    return tbl["base"].to_numpy(dtype=float) * np.exp(_log_returns(tbl, MODEL_NAMES) @ w)


def metrics(actual, pred, base) -> Dict[str, float]:
    actual, pred, base = (np.asarray(x, dtype=float) for x in (actual, pred, base))
    if len(actual) == 0:
        return {"MAPE_pct": np.nan, "RMSE_USD": np.nan, "DirAcc_pct": np.nan, "n": 0}
    moved = np.sign(pred - base) != 0  # 방향을 제시한 예측만 채점 (Random Walk는 해당 없음)
    dir_acc = float(np.mean(np.sign(pred - base)[moved] == np.sign(actual - base)[moved]) * 100) \
        if moved.any() else np.nan
    return {"MAPE_pct": float(np.mean(np.abs((actual - pred) / actual)) * 100),
            "RMSE_USD": float(np.sqrt(np.mean((actual - pred) ** 2))),
            "DirAcc_pct": dir_acc, "n": int(len(actual))}


def _score(tbl: pd.DataFrame, w: np.ndarray, horizon: int, split: str) -> List[dict]:
    preds = {name: tbl[name].to_numpy() for name in MODEL_NAMES} if len(tbl) else {}
    if len(tbl):
        preds[ENSEMBLE] = ensemble_price(tbl, w)
    out = []
    for name, p in preds.items():
        m = metrics(tbl["actual"], p, tbl["base"])
        out.append({"horizon_days": horizon, "split": split, "model": name, **m})
    rw = next((r["MAPE_pct"] for r in out if r["model"] == BASELINE), np.nan)
    for r in out:  # 양수면 Random Walk보다 오차가 작다
        r["vs_RW_pct"] = (1.0 - r["MAPE_pct"] / rw) * 100 if rw and np.isfinite(rw) else np.nan
    return out


def run(df: pd.DataFrame, horizons: List[int], test_fraction: float = 0.25):
    """반환: (지표 DataFrame, 기간별 정보 dict)

    info[h] = {weights_live, weights_val, best_single, n_val, n_test}
    weights_val  : 검증 구간만으로 구한 가중치 (TEST 지표 계산에 사용)
    weights_live : 검증+테스트 전체로 다시 구한 가중치 (실제 예측에 사용)
    """
    split = int(len(df) * (1.0 - test_fraction))
    rows: List[dict] = []
    info: Dict[int, dict] = {}
    for h in horizons:
        tbl = prediction_table(df, h)
        val = tbl[tbl["origin"] + h < split]
        test = tbl[tbl["origin"] >= split]
        w_val = fit_weights(val)
        val_rows = _score(val, w_val, h, "VALIDATION")
        rows += val_rows + _score(test, w_val, h, "TEST")
        singles = [r for r in val_rows if r["model"] != ENSEMBLE]
        best = min(singles, key=lambda r: r["MAPE_pct"])["model"] if singles else BASELINE
        info[h] = {"weights_val": w_val, "weights_live": fit_weights(tbl),
                   "best_single": best, "n_val": len(val), "n_test": len(test)}
    return pd.DataFrame(rows), info


def forecast(df: pd.DataFrame, info: Dict[int, dict]) -> pd.DataFrame:
    """가장 최근 종가 기준, 기간별 앙상블 예측."""
    current = float(df["close"].iloc[-1])
    rows = []
    for h, meta in info.items():
        p = predict_all(df, h)
        logr = np.array([np.log(p[n] / current) for n in MODEL_NAMES])
        ens = current * float(np.exp(logr @ meta["weights_live"]))
        rows.append({"horizon_days": h,
                     "target_date": (df.index[-1] + pd.Timedelta(days=h)).strftime("%Y-%m-%d"),
                     "current_price": current, "ensemble_price": ens,
                     "ensemble_change_pct": (ens / current - 1) * 100,
                     "best_single_model": meta["best_single"],
                     "best_single_price": p[meta["best_single"]]})
    return pd.DataFrame(rows)


def daily_path(df: pd.DataFrame, weights: np.ndarray, days: int = 7) -> dict:
    """1~days일 뒤를 각각 직접 예측한 경로 (가짜 캔들을 만들어 이어 붙이지 않는다)."""
    current = float(df["close"].iloc[-1])
    out = {"dates": [], "ensemble": [], "models": {n: [] for n in MODEL_NAMES}}
    for d in range(1, days + 1):
        p = predict_all(df, d)
        logr = np.array([np.log(p[n] / current) for n in MODEL_NAMES])
        out["dates"].append((df.index[-1] + pd.Timedelta(days=d)).strftime("%Y-%m-%d"))
        out["ensemble"].append(round(current * float(np.exp(logr @ weights)), 1))
        for n in MODEL_NAMES:
            out["models"][n].append(round(p[n], 1))
    return out
