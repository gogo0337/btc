"""점검용: horizon=1일 때 gemini 계열 3개 모델이 원본 pipeline.py 수식과 같은 값을 내는지 확인."""
import numpy as np
import pandas as pd

from data import fetch_daily
from models import predict_all

df = fetch_daily(years=1.2)
c, o, h, l, v = (df[k].values for k in ("close", "open", "high", "low", "volume"))
prev, wc, wv = c[-1], c[-20:], v[-20:]
ef = pd.Series(wc).ewm(span=5).mean().iloc[-1]
es = pd.Series(wc).ewm(span=15).mean().iloc[-1]
ma200 = df["close"].rolling(200).mean().iloc[-1]
p1 = prev * (1 + np.clip((ef - es) / es, -0.025, 0.025) * 0.45 * (1.15 if prev > ma200 else 0.85))
vwap = np.sum(wc * wv) / np.sum(wv)
p2 = prev * (1 + np.clip((vwap - prev) / prev * 0.15, -0.02, 0.02))
body = (c[-1] - o[-1]) / max(h[-1] - l[-1], 1e-6)
p3 = prev * (1 + np.clip((c[-1] - c[-4]) / c[-4] * 0.25 + body * 0.012, -0.03, 0.03))

mine = predict_all(df, 1)
for ref, name in zip((p1, p2, p3), ("EMA Regime", "VWAP Reversion", "Candle Momentum")):
    print(f"{name:16s} 원본 {ref:,.4f} / 통합 {mine[name]:,.4f} ->", "일치" if abs(ref - mine[name]) < 1e-6 else "불일치")
