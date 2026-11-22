"""Streamlit 대시보드:  streamlit run app.py  (먼저 pipeline.py를 실행해 data/latest.json 생성)"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BTC 통합 예측 대시보드", layout="wide")
st.title("📈 BTC 통합 예측 대시보드")

path = Path(__file__).parent / "data" / "latest.json"
if not path.exists():
    st.error("data/latest.json이 없습니다. 먼저 `python pipeline.py`를 실행하세요.")
    st.stop()
d = json.loads(path.read_text(encoding="utf-8"))
o = pd.DataFrame(d["ohlcv"])
st.caption(f"{d['symbol']} · 기준 종가일 {d['last_close_date']} · 갱신 {d['updated_at']} · 국면: {d['regime']}")

cols = st.columns(len(d["horizons"]) + 1)
cols[0].metric("현재 종가", f"${d['current_price']:,.0f}")
for c, h in zip(cols[1:], d["horizons"]):
    c.metric(f"{h['horizon_days']}일 뒤 앙상블", f"${h['ensemble_price']:,.0f}", f"{h['ensemble_change_pct']:+.2f}%")

shown = st.multiselect("7일 경로에 함께 볼 개별 모델", list(d["path"]["models"]), default=[])
last_date, last_close = o["date"].iloc[-1], o["close"].iloc[-1]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=o["date"], open=o["open"], high=o["high"], low=o["low"], close=o["close"], name=d["symbol"]))
fig.add_trace(go.Scatter(x=o["date"], y=o["MA50"], name="50일선", line=dict(color="#38bdf8", width=1.4)))
fig.add_trace(go.Scatter(x=o["date"], y=o["MA200"], name="200일선", line=dict(color="#ef4444", width=1.4)))
fig.add_trace(go.Scatter(x=[last_date] + d["path"]["dates"], y=[last_close] + d["path"]["ensemble"],
                         mode="lines+markers", name="앙상블 7일 경로", line=dict(color="#facc15", width=3)))
fig.add_trace(go.Scatter(x=[last_date] + [h["target_date"] for h in d["horizons"]],
                         y=[last_close] + [h["ensemble_price"] for h in d["horizons"]],
                         mode="lines+markers", name="기간별 앙상블", line=dict(color="#facc15", width=1.5, dash="dash")))
for name in shown:
    fig.add_trace(go.Scatter(x=[last_date] + d["path"]["dates"], y=[last_close] + d["path"]["models"][name],
                             mode="lines", name=name, line=dict(width=1.3, dash="dot")))
fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=600, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

if all("interval" in h for h in d["horizons"]):
    st.subheader("80% 변동 구간")
    st.dataframe(pd.DataFrame([{
        "기간(일)": h["horizon_days"], "하단": h["interval"].get("low"), "중앙": h["interval"].get("median"),
        "상단": h["interval"].get("high"), "테스트 적중률(%, 목표 80)": h["interval"].get("coverage"),
        "단순 방식 대비 품질(%)": h["interval"].get("score_vs_base_pct"),
        "독립 표본": h["interval"]["n_independent"],
        "판정": "신뢰 가능" if h["interval"]["calibrated"] else "주의",
    } for h in d["horizons"]]), hide_index=True, use_container_width=True)
    st.subheader("상승 확률")
    st.dataframe(pd.DataFrame([{
        "기간(일)": h["horizon_days"], "모델 상승 확률(%)": h["direction"].get("p_up"),
        "기준 확률(%)": h["direction"].get("base_p_up"), "테스트 정확도(%)": h["direction"].get("accuracy"),
        "기준 정확도(%)": h["direction"].get("base_accuracy"),
        "확률 오차 개선(%)": h["direction"].get("brier_skill_pct"),
        "독립 표본": h["direction"]["n_independent"],
        "판정": "사용 가능" if h["direction"]["usable"] else "신호 없음",
    } for h in d["horizons"]]), hide_index=True, use_container_width=True)
    st.caption("'확률 오차 개선'이 0 이하면 과거 상승 비율을 그대로 쓰는 것보다 못하다는 뜻입니다.")

st.subheader("가격 점 예측: 테스트 구간 성능 (가중치 결정에 쓰지 않은 구간)")
st.dataframe(pd.DataFrame([{
    "기간(일)": h["horizon_days"], "목표일": h["target_date"], "앙상블 예측": h["ensemble_price"],
    "변화율(%)": h["ensemble_change_pct"], "앙상블 MAPE(%)": h["test"]["ensemble_mape"],
    "Random Walk MAPE(%)": h["test"]["rw_mape"], "RW 대비(%)": h["test"]["vs_rw_pct"],
    "방향 적중(%)": h["test"]["dir_acc"], "표본 수": h["test"]["n"],
    "RW보다 나음": "예" if h["beats_rw"] else "아니오",
} for h in d["horizons"]]), hide_index=True, use_container_width=True)
st.subheader("기간별 앙상블 가중치 (%)")
st.dataframe(pd.DataFrame({f"{h['horizon_days']}일": h["weights"] for h in d["horizons"]}), use_container_width=True)
st.caption("'RW 대비'가 양수면 현재가 유지 예측보다 오차가 작습니다. 연구·참고용이며 투자 조언이 아닙니다.")
