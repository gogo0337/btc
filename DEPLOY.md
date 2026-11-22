# 차트를 매일 웹에 게시하기 (GitHub, 무료)

PC가 꺼져 있어도 GitHub가 매일 09:30(KST)에 `chart.py`를 실행해, 최신 종가로 만든 차트를 웹 주소에 올립니다.
저장소는 `gogo0337/btc-research` 하나만 씁니다.

- 웹 주소: `https://gogo0337.github.io/btc-research/`
- 설정 파일: `.github/workflows/chart_daily.yml`

> **공개 범위 주의**: 무료 계정은 공개(Public) 저장소에서만 웹 게시(GitHub Pages)를 쓸 수 있습니다.
> 저장소를 공개로 바꾸면 **코드 전체와 차트가 인터넷에 공개**됩니다.
> 사내 외부 공유 규정에 해당하는지 정보보안팀에 먼저 확인하세요.
> 코드를 비공개로 두려면 GitHub Pro(유료)가 필요합니다.

## 비용

무료입니다. 공개 저장소의 GitHub Actions와 Pages는 무료이고, 이 작업은 하루 2~3분을 씁니다.

## 1. 저장소를 공개로 바꾸기

`btc-research` → **Settings** → 맨 아래 **Danger Zone → Change repository visibility → Change to public**

## 2. 웹 게시 켜기

`btc-research` → **Settings → Pages → Build and deployment → Source**: **GitHub Actions**

## 3. 처음 한 번 수동 실행

`btc-research` → **Actions** 탭 → 왼쪽 **Daily BTC Chart** → **Run workflow** → **Run workflow**

2~3분 뒤 초록 체크가 뜨면 성공이고, 실행 화면의 deploy 단계에 웹 주소가 표시됩니다.
빨간 X가 뜨면 실행 기록을 열어 마지막 오류 메시지를 확인하세요.

## 이후

- 매일 09:30(KST) 전후로 자동 갱신됩니다. GitHub 사정에 따라 수십 분 늦어질 때가 있습니다.
- 차트 위쪽의 "갱신" 시각으로 마지막 갱신 시점을 확인할 수 있습니다.
- 공개 저장소는 60일 동안 커밋이 없으면 예약 실행이 꺼지므로, 50일이 지나면 작업이 빈 커밋을 하나 남겨 예약을 유지합니다.
- 코드를 고친 뒤에는 `git add .`, `git commit -m "..."`, `git push`로 올리면 다음 실행부터 반영됩니다.
- 갱신을 멈추려면 Actions → Daily BTC Chart → 오른쪽 위 **…** → **Disable workflow**.

## 데이터 출처가 막힐 때

GitHub 서버(미국)에서는 일부 출처가 막힐 수 있어, 각각 대체 경로를 두었습니다.

| 출처 | 막히면 |
|---|---|
| Binance (api.binance.com) | 공개 시세 전용 주소(data-api.binance.vision)로 다시 시도 |
| Yahoo Finance (2014~2017 캔들) | 그 구간은 Coin Metrics 종가로 그린 근사 캔들로 대체, 모델은 2017-08 이후 일봉만 사용 |
| Coin Metrics (2010~2014 가격) | 차트가 Binance 시작일(2017-08)부터 그려짐 |
| 뉴욕 연은·미 재무부 (금리) | 금리 관련 표시만 빠지고 나머지는 정상 |
| blockchain.info (블록 높이) | 다음 반감기 날짜를 4년 주기로 추정 |
