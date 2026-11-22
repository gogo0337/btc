"""결과물 생성: 텔레그램용 PNG 차트, 단독 실행 HTML 대시보드."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def build_png(data: dict, path: Path) -> None:
    """최근 종가 + 50/200일선 + 7일 경로 + 기간별 앙상블 예측. (서버 폰트 문제로 라벨은 영문)"""
    o = pd.DataFrame(data["ohlcv"])
    o["date"] = pd.to_datetime(o["date"])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(o["date"], o["close"], color="#1f2937", lw=1.4, label="BTC close")
    ax.plot(o["date"], o["MA50"], color="#0ea5e9", lw=1, label="MA50")
    ax.plot(o["date"], o["MA200"], color="#ef4444", lw=1, label="MA200")
    last_d, last_c = o["date"].iloc[-1], o["close"].iloc[-1]
    pd_dates = [last_d] + list(pd.to_datetime(data["path"]["dates"]))
    ax.plot(pd_dates, [last_c] + data["path"]["ensemble"], color="#f59e0b", lw=2.5, label="Ensemble (7d path)")
    hz = data["horizons"]
    ax.plot([last_d] + [pd.Timestamp(h["target_date"]) for h in hz],
            [last_c] + [h["ensemble_price"] for h in hz],
            color="#f59e0b", ls="--", marker="o", lw=1.2, label="Ensemble by horizon")
    ax.set_title(f"{data['symbol']} unified forecast  (updated {data['updated_at']})")
    ax.set_ylabel("USDT")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC 통합 예측 대시보드</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{font-family:system-ui,'Malgun Gothic',sans-serif;margin:0;padding:16px;background:#0f172a;color:#e2e8f0}
 h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:22px 0 0} .sub{color:#94a3b8;font-size:13px;margin-bottom:12px}
 .cards{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px}
 .card{background:#1e293b;border-radius:8px;padding:10px 14px;min-width:150px}
 .card b{display:block;font-size:18px} .card span{font-size:12px;color:#94a3b8}
 .wrap{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}
 th,td{padding:6px 8px;border-bottom:1px solid #334155;text-align:right;white-space:nowrap}
 th:first-child,td:first-child{text-align:left} .good{color:#4ade80} .bad{color:#f87171}
 .note{color:#94a3b8;font-size:12px;margin-top:12px;line-height:1.6}
</style></head><body>
<h1>BTC 통합 예측 대시보드</h1><div class="sub" id="sub"></div>
<div class="cards" id="cards"></div><div id="chart" style="height:520px"></div>
<h2>80% 변동 구간</h2><div class="wrap"><table id="tblI"></table></div>
<div class="note">현재 변동성에 맞춰 폭을 조절한 범위입니다. 테스트 적중률이 75~85%면 '신뢰 가능'으로 표시합니다.
독립 표본은 서로 겹치지 않는 기간의 수로, 적을수록 우연일 가능성이 큽니다.</div>
<h2>상승 확률</h2><div class="wrap"><table id="tblD"></table></div>
<div class="note">펀딩비·공포탐욕·온체인 지표를 넣은 로지스틱 회귀입니다. '기준'은 과거 상승 비율을 그대로 확률로 쓰는 경우이고,
'확률 오차 개선'이 0 이하면 기준보다 못하다는 뜻이므로 그 기간의 확률은 쓰지 마세요.</div>
<h2>가격 점 예측 (앙상블)</h2><div class="wrap"><table id="tbl"></table></div>
<div class="note">TEST 지표는 가중치를 정할 때 쓰지 않은 뒤쪽 구간에서 측정했습니다.
'RW 대비'가 양수면 현재가 유지(Random Walk)보다 오차가 작다는 뜻입니다.
표본 수(n)가 작은 장기 구간은 참고용으로만 보세요. 투자 조언이 아닙니다.</div>
<script>
const D = __DATA__;
const f0 = x => '$' + Math.round(x).toLocaleString(), pc = x => (x >= 0 ? '+' : '') + x.toFixed(2) + '%';
document.getElementById('sub').textContent = `${D.symbol} · 기준 종가일 ${D.last_close_date} · 갱신 ${D.updated_at}`;
let cards = `<div class="card"><span>현재 종가</span><b>${f0(D.current_price)}</b></div>
 <div class="card"><span>시장 국면</span><b style="font-size:14px">${D.regime}</b></div>`;
const nv = (x, s = '') => x == null ? '-' : x + s, p1 = x => x == null ? '-' : (x >= 0 ? '+' : '') + x.toFixed(1) + '%';
const hasI = D.horizons.every(h => h.interval && h.interval.low != null);
D.horizons.forEach(h => cards += hasI
 ? `<div class="card"><span>${h.horizon_days}일 뒤 80% 구간</span><b>${f0(h.interval.low)} ~ ${f0(h.interval.high)}</b>
 <span class="${h.interval.calibrated ? 'good' : 'bad'}">${h.interval.calibrated ? '신뢰 가능' : '주의: 잘 안 맞음'}</span></div>`
 : `<div class="card"><span>${h.horizon_days}일 뒤 앙상블</span><b>${f0(h.ensemble_price)}</b><span>${pc(h.ensemble_change_pct)}</span></div>`);
document.getElementById('cards').innerHTML = cards;
if (hasI) {
 let ti = '<tr><th>기간</th><th>목표일</th><th>하단</th><th>중앙</th><th>상단</th><th>범위</th><th>테스트 적중률 (목표 80%)</th><th>단순 방식 적중률</th><th>평균 폭</th><th>단순 방식 대비 품질</th><th>독립 표본</th><th>판정</th></tr>';
 let td = '<tr><th>기간</th><th>모델 상승 확률</th><th>기준 확률</th><th>테스트 정확도</th><th>기준 정확도</th><th>확률 오차 개선</th><th>가격 피처만 썼을 때</th><th>AUC</th><th>독립 표본</th><th>판정</th><th>현재 주요 요인</th></tr>';
 D.horizons.forEach(h => { const i = h.interval, d = h.direction;
  ti += `<tr><td>${h.horizon_days}일</td><td>${h.target_date}</td><td>${f0(i.low)}</td><td>${f0(i.median)}</td><td>${f0(i.high)}</td><td>${p1(i.low_pct)} ~ ${p1(i.high_pct)}</td>
  <td>${nv(i.coverage, '%')}</td><td>${nv(i.base_coverage, '%')}</td><td>${nv(i.avg_width_pct, '%')}</td><td>${p1(i.score_vs_base_pct)}</td><td>약 ${i.n_independent}개</td>
  <td class="${i.calibrated ? 'good' : 'bad'}">${i.calibrated ? '신뢰 가능' : '주의'}</td></tr>`;
  td += `<tr><td>${h.horizon_days}일</td><td>${nv(d.p_up, '%')}</td><td>${nv(d.base_p_up, '%')}</td><td>${nv(d.accuracy, '%')}</td><td>${nv(d.base_accuracy, '%')}</td>
  <td>${p1(d.brier_skill_pct)}</td><td>${p1(d.price_only_brier_skill_pct)}</td><td>${nv(d.auc)}</td><td>약 ${d.n_independent}개</td>
  <td class="${d.usable ? 'good' : 'bad'}">${d.usable ? '사용 가능' : '신호 없음'}</td><td>${(d.drivers || []).map(v => `${v.feature}(${v.effect})`).join(', ')}</td></tr>`; });
 document.getElementById('tblI').innerHTML = ti; document.getElementById('tblD').innerHTML = td;
}
const o = D.ohlcv, x = o.map(r => r.date), last = o[o.length - 1];
const tr = [{type:'candlestick', x, open:o.map(r=>r.open), high:o.map(r=>r.high), low:o.map(r=>r.low), close:o.map(r=>r.close), name:D.symbol},
 {x, y:o.map(r=>r.MA50), mode:'lines', name:'50일선', line:{color:'#38bdf8', width:1.4}},
 {x, y:o.map(r=>r.MA200), mode:'lines', name:'200일선', line:{color:'#ef4444', width:1.4}},
 {x:[last.date, ...D.path.dates], y:[last.close, ...D.path.ensemble], mode:'lines+markers', name:'앙상블 7일 경로', line:{color:'#facc15', width:3}},
 {x:[last.date, ...D.horizons.map(h=>h.target_date)], y:[last.close, ...D.horizons.map(h=>h.ensemble_price)], mode:'lines+markers', name:'기간별 앙상블', line:{color:'#facc15', width:1.5, dash:'dash'}}];
if (hasI) { const tx = [last.date, ...D.horizons.map(h => h.target_date)];
 tr.push({x:tx, y:[last.close, ...D.horizons.map(h => h.interval.low)], mode:'lines', name:'80% 구간 하단', line:{color:'#a78bfa', width:1}},
  {x:tx, y:[last.close, ...D.horizons.map(h => h.interval.high)], mode:'lines', name:'80% 구간 상단', line:{color:'#a78bfa', width:1}, fill:'tonexty', fillcolor:'rgba(167,139,250,0.15)'}); }
Object.entries(D.path.models).forEach(([n, v]) => tr.push({x:[last.date, ...D.path.dates], y:[last.close, ...v], mode:'lines', name:n, visible:'legendonly', line:{width:1.3, dash:'dot'}}));
Plotly.newPlot('chart', tr, {template:'plotly_dark', paper_bgcolor:'#0f172a', plot_bgcolor:'#0f172a', font:{color:'#e2e8f0'},
 xaxis:{rangeslider:{visible:false}}, margin:{l:50, r:10, t:10, b:40}, legend:{orientation:'h', y:-0.12, yanchor:'top'}}, {responsive:true});
let t = '<tr><th>기간</th><th>목표일</th><th>앙상블 예측</th><th>변화율</th><th>검증 1위 단일 모델</th><th>TEST MAPE</th><th>RW MAPE</th><th>RW 대비</th><th>방향 적중</th><th>n</th><th>가중치 상위</th></tr>';
D.horizons.forEach(h => { const s = h.test, top = Object.entries(h.weights).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([n, w]) => `${n} ${w}%`).join(', ');
 t += `<tr><td>${h.horizon_days}일</td><td>${h.target_date}</td><td>${f0(h.ensemble_price)}</td><td>${pc(h.ensemble_change_pct)}</td>
 <td>${h.best_single_model} (${f0(h.best_single_price)})</td><td>${s.ensemble_mape ?? '-'}%</td><td>${s.rw_mape ?? '-'}%</td>
 <td class="${h.beats_rw ? 'good' : 'bad'}">${s.vs_rw_pct == null ? '-' : pc(s.vs_rw_pct)}</td><td>${s.dir_acc ?? '-'}%</td><td>${s.n}</td><td>${top}</td></tr>`; });
document.getElementById('tbl').innerHTML = t;
</script></body></html>
"""


def build_html(data: dict, path: Path) -> None:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(_HTML.replace("__DATA__", payload), encoding="utf-8")
