"""통합 예측 모델 8종.

모든 모델은 같은 형태를 가진다:  f(o, h, l, c, v, horizon) -> horizon일 뒤 예상 종가
입력은 예측 시점까지의 numpy 배열(과거 → 현재)이며, 미래 데이터는 쓰지 않는다.

- gpt 계열 5종 (종가 기반, 장기 예측용): Random Walk, Momentum, Trend, Mean Reversion, Hybrid
- gemini 계열 3종 (OHLCV 기반, 단기 신호): EMA Regime, VWAP Reversion, Candle Momentum
  원본은 1일 예측 전용이라, 신호가 매일 일정 비율로 약해진다고 보고 horizon일로 확장했다.
  horizon=1이면 원본 gemini 수식과 같은 값을 낸다.
  (원본 이름 'ETF·기관', '온체인'은 실제 입력과 달라 수식에 맞는 이름으로 바꿨다.)
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

MIN_HISTORY = 220  # 200일 이동평균 + 여유


def _ema_last(x: np.ndarray, span: int) -> float:
    """pandas ewm(span).mean()의 마지막 값과 동일 (adjust=True)."""
    alpha = 2.0 / (span + 1.0)
    w = (1.0 - alpha) ** np.arange(len(x) - 1, -1, -1)
    return float(np.sum(w * x) / np.sum(w))


def _damped(h: int, rho: float) -> float:
    """1일 신호가 매일 rho배로 약해질 때 h일 누적 배수 (h=1이면 1)."""
    return (1.0 - rho ** h) / (1.0 - rho)


# ---------- gpt 계열 ----------
def random_walk(o, h, l, c, v, horizon):
    return float(c[-1])


def momentum(o, h, l, c, v, horizon):
    r = np.diff(np.log(c[-22:]))
    drift = float(np.clip(r.mean(), -0.03, 0.03))
    return float(c[-1] * np.exp(drift * horizon))


def trend(o, h, l, c, v, horizon):
    logp = np.log(c[-60:])
    slope = float(np.polyfit(np.arange(len(logp), dtype=float), logp, 1)[0])
    return float(c[-1] * np.exp(np.clip(slope, -0.03, 0.03) * horizon))


def mean_reversion(o, h, l, c, v, horizon):
    z = np.log(c[-180:])
    a, b = z[:-1], z[1:]
    denom = float(np.sum((a - a.mean()) ** 2))
    phi = float(np.sum((a - a.mean()) * (b - b.mean())) / denom) if denom else 0.0
    phi = float(np.clip(phi, 0.0, 0.98))
    mu = float(z.mean())
    return float(np.exp(mu + (z[-1] - mu) * phi ** horizon))


def hybrid(o, h, l, c, v, horizon):
    m = momentum(o, h, l, c, v, horizon)
    t = trend(o, h, l, c, v, horizon)
    return float(np.exp(0.6 * np.log(m) + 0.4 * np.log(t)))


# ---------- gemini 계열 ----------
def ema_regime(o, h, l, c, v, horizon):
    """EMA(5) - EMA(15) 추세에 200일선 국면 가중(위 1.15 / 아래 0.85)."""
    win = c[-20:]
    slow = _ema_last(win, 15)
    drift = float(np.clip((_ema_last(win, 5) - slow) / slow, -0.025, 0.025))
    bias = 1.15 if c[-1] > c[-200:].mean() else 0.85
    step = drift * 0.45 * bias
    return float(c[-1] * (1.0 + np.clip(step * _damped(horizon, 0.85), -0.5, 0.5)))


def vwap_reversion(o, h, l, c, v, horizon):
    """20일 거래량 가중 평균가 쪽으로 매일 격차의 15%씩 회귀."""
    win_c, win_v = c[-20:], v[-20:]
    vwap = float(np.sum(win_c * win_v) / max(np.sum(win_v), 1e-6))
    gap = (vwap - c[-1]) / c[-1]
    step = float(np.clip(gap * 0.15, -0.02, 0.02))
    # 0.15 * (1 + 0.85 + 0.85^2 ...) : h가 커지면 격차 전체를 메우는 쪽으로 수렴
    return float(c[-1] * (1.0 + step * _damped(horizon, 0.85)))


def candle_momentum(o, h, l, c, v, horizon):
    """3일 수익률 + 직전 캔들 몸통 비율. 캔들 신호는 빨리 사라진다고 본다(rho=0.5)."""
    mom3 = (c[-1] - c[-4]) / c[-4]
    body = (c[-1] - o[-1]) / max(h[-1] - l[-1], 1e-6)
    step = float(np.clip(mom3 * 0.25 + body * 0.012, -0.03, 0.03))
    return float(c[-1] * (1.0 + step * _damped(horizon, 0.5)))


MODELS: Dict[str, Callable[..., float]] = {
    "Random Walk": random_walk,
    "Momentum": momentum,
    "Trend": trend,
    "Mean Reversion": mean_reversion,
    "Hybrid": hybrid,
    "EMA Regime": ema_regime,
    "VWAP Reversion": vwap_reversion,
    "Candle Momentum": candle_momentum,
}
MODEL_NAMES = list(MODELS)
BASELINE = "Random Walk"


def predict_all(df, horizon: int, end: int | None = None) -> Dict[str, float]:
    """df의 0..end 행(포함)만 사용해 모든 모델의 horizon일 뒤 예상가를 계산."""
    end = len(df) - 1 if end is None else end
    sl = slice(0, end + 1)
    arrs = [df[k].to_numpy(dtype=float)[sl] for k in ("open", "high", "low", "close", "volume")]
    if len(arrs[3]) < MIN_HISTORY:
        raise ValueError(f"과거 데이터가 {MIN_HISTORY}일 이상 필요합니다.")
    return {name: fn(*arrs, int(horizon)) for name, fn in MODELS.items()}
