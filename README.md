# BTC 통합 예측 모델

`gpt/btc_forecast.py`와 `gemini/pipeline.py`를 하나로 합친 버전입니다.

- **gpt에서 가져온 것:** 시간순 검증/테스트 분리, Random Walk 기준선, 7/30/90/180일 예측, 종가 기반 모델 5종
- **gemini에서 가져온 것:** OHLCV 기반 단기 모델 3종, 가중치 앙상블, 대시보드, 텔레그램, GitHub Actions 매일 실행

## 파일

| 파일 | 역할 |
|---|---|
| `data.py` | Binance 공개 API에서 일봉 수집 (마감된 캔들만, 예비 호스트 포함) |
| `models.py` | 모델 8종. 모두 `f(o, h, l, c, v, horizon) -> 예상 종가` 형태 |
| `backtest.py` | 검증/테스트 분리 백테스트, 기간별 앙상블 가중치, 예측 |
| `pipeline.py` | 전체 실행: 수집 → 백테스트 → 예측 → 파일 저장 → 텔레그램 |
| `report.py` | `chart.png`, `dashboard.html` 생성 |
| `app.py` | Streamlit 대시보드 |
| `backtest_events.py` | 반감기·ETF 승인 전후 구간 검증 |
| `check_parity.py` | gemini 계열 모델이 원본 수식과 같은 값을 내는지 점검 |

## 실행

```bash
pip install -r requirements.txt
```

```bash
python pipeline.py
```

결과는 `data/` 폴더에 저장됩니다: `latest.json`, `dashboard.html`(더블클릭으로 열기), `chart.png`, `forecasts.csv`, `backtest_results.csv`.

옵션: `--years 8 --horizons 7 30 90 180 --out data --no-telegram`

```bash
streamlit run app.py
```

```bash
python backtest_events.py --horizon 1
```

## 텔레그램·자동 실행

`TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` 환경 변수가 있을 때만 발송합니다. 없으면 건너뜁니다.
GitHub에 올릴 때는 이 폴더를 저장소 루트로 올리고, 저장소 Secrets에 위 두 값(선택: `WEB_APP_URL`)을 등록하세요.
토큰은 코드나 파일에 적지 마세요. 매일 09:30 KST에 실행됩니다.

## 원본과 달라진 점

1. **가중치와 성능 측정 구간을 분리했습니다.** 가중치는 검증 구간에서만 정하고, 보고하는 수치는 그 뒤 테스트 구간(기본 뒤쪽 25%)에서 잽니다. 실제 예측에 쓰는 가중치는 전체 구간으로 다시 구합니다.
2. **'가격 일치율'을 없앴습니다.** 대신 MAPE를 Random Walk와 나란히 보여 주고, 'RW 대비'(양수면 더 나음)를 함께 표시합니다.
3. **gemini 모델 이름을 수식에 맞게 바꿨습니다.** ETF·기관 → EMA Regime, 온체인 → VWAP Reversion, 캔들 → Candle Momentum.
4. **7일 예측에서 가짜 캔들을 만들지 않습니다.** 1~7일 뒤를 각각 직접 예측합니다. gemini 모델은 1일 신호가 매일 일정 비율로 약해진다고 가정해 장기로 확장했습니다(감쇠율 0.85, 캔들은 0.5 — 임의로 정한 값이며 조정 대상).
5. **표본이 적으면 가중치를 균등 쪽으로 당깁니다**(`SHRINK_K = 30`).
6. **진행 중인 당일 캔들은 제외합니다.**
7. 30일을 넘는 기간은 예측 시점을 30일 간격으로 잡아 목표 구간이 겹칩니다. 표본 수(n)만큼 독립적인 관측은 아닙니다.

## 2차 개선: 가격 밖 데이터, 방향 확률, 변동 구간

| 파일 | 역할 |
|---|---|
| `features.py` | 펀딩비(Binance 선물), 공포·탐욕 지수(alternative.me), 온체인(Coin Metrics: MVRV, 활성 주소, 해시레이트, 거래소 입출금, 수수료) 수집과 피처 18개 생성. 모두 무료·키 불필요 |
| `signals.py` | 상승 확률(로지스틱 회귀)과 80% 변동 구간(현재 변동성으로 폭 조절) |

- 온체인 값은 다음 날 확정되므로 하루 늦춰 씁니다. 미국 현물 ETF 순유입은 무료 API가 없어 넣지 못했습니다.
- 테스트 구간에서는 30일마다 다시 학습하고, 그 시점에 결과가 확정된 표본만 씁니다.
- 앙상블에서는 같은 구간에서 Random Walk보다 오차가 1.25배 넘게 큰 모델을 자동으로 뺍니다(`PRUNE_RATIO`).
- 대시보드는 기간마다 **판정**을 붙입니다. 변동 구간은 테스트 적중률이 75~85%일 때 '신뢰 가능', 상승 확률은 기준(과거 상승 비율)보다 확률 오차가 작을 때만 '사용 가능'입니다.

2026-11-22 실행 결과: 7일·30일 변동 구간은 신뢰 가능(적중률 80.7%, 80.6%). 상승 확률은 모든 기간에서 기준보다 못했고, 외부 데이터를 넣어도 나아지지 않았습니다.

## ETF 전후 분석

```bash
python etf_analysis.py
```

`data/etf_report.html`과 `data/etf_analysis.json`을 만듭니다. `macro.py`가 Yahoo Finance에서 BTC·나스닥·금·달러·미국 10년 금리·이더리움 일별 종가를 받습니다.

1. ETF 승인(2024-01-10) 전후의 변동성·낙폭·상관 비교
2. 'ETF가 없었다면'의 가격 추정: ETF 이전의 BTC와 거시 지표의 관계를 이후로 연장. 가짜 시점 검증으로 방법의 오차를 함께 표시
3. 반감기 사이클 겹쳐 보기

ETF 일별 순유입은 무료 공개 API가 없어 쓰지 않았습니다(Farside는 자동 접근 차단, 다른 곳은 유료이거나 키 필요).

## 전체 기간 캔들 차트

```bash
python chart.py
```

`data/chart.html`을 만듭니다. 2014-09부터의 캔들(2017-08 이전은 Yahoo Finance, 이후는 Binance), 20/50/200일 이동평균선, 그리고 모델을 과거 전 구간에 적용한 결과(7일·30일 전 시점에 만든 80% 구간과 앙상블 예측)를 함께 보여 줍니다.

차트는 TradingView Lightweight Charts를 쓰며, 조작법은 페이지 아래에 있습니다. 표시 버튼으로 다음 오버레이를 켤 수 있습니다.

- 레인보우: 2010-07부터의 가격에 맞춘 로그 회귀선과 9단계 띠 (전체 기간으로 사후에 맞춘 곡선)
- 반감기 ±500일, ETF 승인 ±100일 구간 표시와 구간별 결과 표
- 이전 사이클 경로: 2016·2020 반감기 ±500일 흐름을 2024 반감기 가격에 맞춰 옮긴 선

## 매일 웹 게시

GitHub Actions가 매일 09:30(KST)에 `chart.py`를 실행해 GitHub Pages에 올립니다(`.github/workflows/chart_daily.yml`). 설정 방법은 [DEPLOY.md](DEPLOY.md)에 있습니다. 예측 파이프라인 워크플로(`daily_run.yml`)는 수동 실행으로 바꿨습니다.

## 한계

연구·참고용이며 투자 조언이 아닙니다. 모델은 모두 단순한 경험식이고, 테스트 구간에서 Random Walk를 뚜렷하게 이기지 못합니다. 결과를 업무에 쓰기 전에 담당자 검토가 필요합니다.
