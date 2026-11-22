# 차트를 매일 웹에 게시하기 (GitHub, 무료)

PC가 꺼져 있어도 GitHub가 매일 09:30(KST)에 `chart.py`를 실행해, 최신 종가로 만든 차트를 웹 주소에 올립니다.

- **코드 저장소 (비공개)**: 이 폴더 전체. 매일 차트를 만드는 작업이 여기서 돕니다.
- **차트 저장소 (공개)**: 차트 페이지(`index.html`) 하나만 올라갑니다. 웹 주소는 `https://<사용자명>.github.io/<차트 저장소 이름>/` 입니다.

> **공개 범위 주의**: 차트 주소는 인터넷에 공개됩니다(주소를 아는 사람은 누구나 볼 수 있음). 사내 외부 공유 규정에 해당하는지 정보보안팀에 먼저 확인하세요. 코드는 비공개 저장소에 남습니다.

## 비용

모두 무료입니다. 비공개 저장소의 GitHub Actions는 무료 계정에 월 2,000분이 주어지고, 이 작업은 하루 2~3분(월 약 90분)을 씁니다. 공개 저장소의 GitHub Pages는 무료입니다.

## 1. 저장소 두 개 만들기

GitHub에 로그인한 뒤 오른쪽 위 **+ → New repository**에서 두 개를 만듭니다.

| 용도 | 이름 예시 | 공개 설정 | 기타 |
|---|---|---|---|
| 코드 | `btc-research` | **Private** | 나머지는 비워 둠 |
| 차트 | `btc-chart` | **Public** | "Add a README file" 체크 |

## 2. 코드 올리기

PowerShell에서 아래를 차례로 실행합니다. `<사용자명>`은 본인 GitHub 사용자명으로 바꿉니다.

```bash
cd C:\Claude\BTC\unified
```

```bash
git init
```

```bash
git add .
```

```bash
git commit -m "BTC research code"
```

```bash
git branch -M main
```

```bash
git remote add origin https://github.com/<사용자명>/btc-research.git
```

```bash
git push -u origin main
```

`.gitignore`에 따라 `.venv`, `data` 폴더(실행 결과물), Farside 파일은 올라가지 않습니다.

## 3. 차트 저장소에 쓸 토큰 만들기

1. GitHub 오른쪽 위 프로필 → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**
2. **Expiration**: 원하는 기간 (최대 1년. 만료되면 차트 갱신이 멈추므로 다시 만들어 넣어야 합니다)
3. **Repository access**: **Only select repositories** → `btc-chart`만 선택
4. **Permissions → Repository permissions → Contents**: **Read and write**
5. **Generate token** → 표시된 토큰을 복사 (다시 볼 수 없습니다)

토큰은 다음 단계의 Secrets에만 붙여 넣고, 다른 곳(채팅, 파일, 메모)에는 적지 마세요.

## 4. 코드 저장소에 설정 넣기

`btc-research` 저장소 → **Settings → Secrets and variables → Actions**

- **Secrets** 탭 → **New repository secret**
  - Name: `SITE_TOKEN`
  - Secret: 3단계에서 복사한 토큰
- **Variables** 탭 → **New repository variable**
  - Name: `SITE_REPO`
  - Value: `<사용자명>/btc-chart`

## 5. 처음 한 번 수동 실행

`btc-research` → **Actions** 탭 → 왼쪽 **Daily BTC Chart** → **Run workflow** → **Run workflow**

2~3분 뒤 초록 체크가 뜨면 성공입니다. 빨간 X가 뜨면 실행 기록을 열어 마지막 오류 메시지를 확인하세요.

## 6. 웹 게시 켜기 (처음 한 번)

`btc-chart` 저장소 → **Settings → Pages**

- **Source**: Deploy from a branch
- **Branch**: `gh-pages` / `(root)` → **Save**

1~2분 뒤 같은 화면 위쪽에 웹 주소가 표시됩니다. 이 주소를 즐겨찾기 하면 됩니다.

## 이후

- 매일 09:30(KST) 전후로 자동 갱신됩니다. GitHub 사정에 따라 수십 분 늦어질 때가 있습니다.
- 차트 위쪽의 "갱신" 시각으로 마지막 갱신 시점을 확인할 수 있습니다.
- 코드를 고친 뒤에는 `git add .`, `git commit -m "..."`, `git push`로 올리면 다음 실행부터 반영됩니다.
- 갱신을 멈추려면 `btc-research` → Actions → Daily BTC Chart → 오른쪽 위 **…** → **Disable workflow**.

## 데이터 출처가 막힐 때

GitHub 서버(미국)에서는 일부 출처가 막힐 수 있어, 각각 대체 경로를 두었습니다.

| 출처 | 막히면 |
|---|---|
| Binance (api.binance.com) | 공개 시세 전용 주소(data-api.binance.vision)로 다시 시도 |
| Yahoo Finance (2014~2017 캔들) | 그 구간은 Coin Metrics 종가로 그린 근사 캔들로 대체, 모델은 2017-08 이후 일봉만 사용 |
| Coin Metrics (2010~2014 가격) | 차트가 Binance 시작일(2017-08)부터 그려짐 |
| 뉴욕 연은·미 재무부 (금리) | 금리 관련 표시만 빠지고 나머지는 정상 |
| blockchain.info (블록 높이) | 다음 반감기 날짜를 4년 주기로 추정 |
