#!/usr/bin/env python3
"""BTC 통합 예측 파이프라인 (gpt 백테스트 체계 + gemini 단기 신호·자동화).

사용법
  python pipeline.py
  python pipeline.py --years 8 --horizons 7 30 90 180 --out data

TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 환경 변수가 있으면 텔레그램으로 요약을 보낸다(없으면 건너뜀).
연구·참고용 도구이며 투자 조언이 아니다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

import backtest as bt
import report
import signals
from data import fetch_daily
from features import EXTERNAL_FEATURES, PRICE_FEATURES, build_features, fetch_external
from models import BASELINE, MODEL_NAMES

KST = timezone(timedelta(hours=9))


def _num(x, nd=2):
    """JSON에 NaN이 들어가지 않도록 None으로 바꾼다."""
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def regime_label(close: float, ma50: float, ma200: float) -> str:
    if close > ma200 and ma50 > ma200:
        return "강세 (200일선 위, 50일선 > 200일선)"
    if close > ma200:
        return "회복/전환 (200일선 위)"
    return "약세/조정 (200일선 아래)"


def build_payload(df: pd.DataFrame, symbol: str, results: pd.DataFrame, info: dict) -> dict:
    ma50, ma200 = df["close"].rolling(50).mean(), df["close"].rolling(200).mean()
    close = float(df["close"].iloc[-1])
    view = df.assign(MA50=ma50, MA200=ma200).tail(180).round(2)
    view.insert(0, "date", view.index.strftime("%Y-%m-%d"))

    test = results[results["split"] == "TEST"].set_index(["horizon_days", "model"])
    horizons = []
    for row in bt.forecast(df, info).to_dict("records"):
        h = row["horizon_days"]
        ens = test.loc[(h, bt.ENSEMBLE)] if (h, bt.ENSEMBLE) in test.index else None
        rw = test.loc[(h, BASELINE)] if (h, BASELINE) in test.index else None
        vs = _num(ens["vs_RW_pct"]) if ens is not None else None
        horizons.append({
            **{k: (_num(v) if isinstance(v, float) else v) for k, v in row.items()},
            "weights": {n: round(float(w) * 100, 1) for n, w in zip(MODEL_NAMES, info[h]["weights_live"])},
            "test": {"ensemble_mape": _num(ens["MAPE_pct"]) if ens is not None else None,
                     "rw_mape": _num(rw["MAPE_pct"]) if rw is not None else None,
                     "vs_rw_pct": vs,
                     "dir_acc": _num(ens["DirAcc_pct"], 1) if ens is not None else None,
                     "n": info[h]["n_test"]},
            "beats_rw": bool(vs is not None and vs > 0),
        })

    path_h = min(info, key=lambda h: abs(h - 7))  # 7일에 가장 가까운 기간의 가중치로 일별 경로 생성
    return {
        "updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "symbol": symbol,
        "last_close_date": df.index[-1].strftime("%Y-%m-%d"),
        "current_price": round(close, 2),
        "ma50": _num(ma50.iloc[-1]), "ma200": _num(ma200.iloc[-1]),
        "regime": regime_label(close, float(ma50.iloc[-1]), float(ma200.iloc[-1])),
        "horizons": horizons,
        "path": bt.daily_path(df, info[path_h]["weights_live"], days=7),
        "ohlcv": json.loads(view.to_json(orient="records", date_format="iso")),
    }


def telegram_caption(d: dict) -> str:
    lines = [f"📊 *BTC 통합 예측 브리핑* ({d['last_close_date']} 종가 기준)",
             f"• 현재 종가: `${d['current_price']:,.0f}`",
             f"• 시장 국면: {d['regime']}", "", "🎯 *기간별 앙상블 예측*"]
    for h in d["horizons"]:
        lines.append(f"• {h['horizon_days']}일 뒤: `${h['ensemble_price']:,.0f}` ({h['ensemble_change_pct']:+.2f}%)")
    lines += ["", "🧪 *테스트 구간 성능 (Random Walk 대비)*"]
    for h in d["horizons"]:
        t = h["test"]
        if t["ensemble_mape"] is None:
            lines.append(f"• {h['horizon_days']}일: 테스트 표본 없음")
            continue
        mark = "✅" if h["beats_rw"] else "⚠️"
        lines.append(f"• {h['horizon_days']}일: MAPE `{t['ensemble_mape']}%` vs RW `{t['rw_mape']}%` "
                     f"{mark} (n={t['n']})")
    lines.append("\n⚠️는 현재가 유지 예측보다 나쁘다는 뜻입니다. 투자 조언이 아닙니다.")
    url = os.environ.get("WEB_APP_URL", "")
    if url:
        lines.append(f"🌐 [대시보드 열기]({url})")
    return "\n".join(lines)


def send_telegram(image: Path, caption: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("텔레그램 환경 변수가 없어 발송을 건너뜁니다.")
        return False
    with open(image, "rb") as photo:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto", timeout=60,
                          data={"chat_id": chat, "caption": caption, "parse_mode": "Markdown"},
                          files={"photo": photo})
    if not r.ok:  # 토큰이 로그에 남지 않도록 상태 코드와 본문만 출력
        raise RuntimeError(f"텔레그램 발송 실패: HTTP {r.status_code} {r.text[:200]}")
    print("텔레그램 발송 완료.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="BTC 통합 예측 파이프라인")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--years", type=float, default=8, help="과거 데이터 길이(년)")
    ap.add_argument("--horizons", type=int, nargs="+", default=[7, 30, 90, 180])
    ap.add_argument("--out", default="data", help="결과 폴더")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.symbol} 일봉 약 {args.years}년치 수집 중...")
    df = fetch_daily(args.symbol, args.years)
    print(f"{len(df):,}개 캔들: {df.index[0].date()} ~ {df.index[-1].date()}")

    results, info = bt.run(df, sorted(set(args.horizons)))
    payload = build_payload(df, args.symbol, results, info)

    print("외부 데이터(펀딩비·공포탐욕·온체인) 수집 및 방향·구간 신호 계산 중...")
    feats = build_features(df, fetch_external(out / "cache"))
    split = int(len(df) * 0.75)
    for h in payload["horizons"]:
        d = signals.direction(df, feats, h["horizon_days"], PRICE_FEATURES + EXTERNAL_FEATURES, split)
        p_only = signals.direction(df, feats, h["horizon_days"], PRICE_FEATURES, split)
        d["price_only_brier_skill_pct"] = p_only.get("brier_skill_pct")
        d["price_only_auc"] = p_only.get("auc")
        # 과거 상승 비율을 그대로 쓰는 것보다 확률 오차가 작을 때만 쓸 만하다고 표시
        d["usable"] = bool((d.get("brier_skill_pct") or 0) > 0)
        iv = signals.interval(df, h["horizon_days"], split)
        # 80% 구간이 테스트에서 실제로 75~85% 맞았을 때만 믿을 만하다고 표시
        iv["calibrated"] = bool(iv.get("coverage") is not None and abs(iv["coverage"] - 80) <= 5)
        h["direction"], h["interval"] = d, iv

    nested = ("weights", "test", "direction", "interval")
    results.to_csv(out / "backtest_results.csv", index=False)
    pd.DataFrame([{k: v for k, v in h.items() if k not in nested} | h["test"]
                  | {f"interval_{k}": v for k, v in h["interval"].items()}
                  | {f"direction_{k}": v for k, v in h["direction"].items() if k != "drivers"}
                  for h in payload["horizons"]]).to_csv(out / "forecasts.csv", index=False)
    (out / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report.build_png(payload, out / "chart.png")
    report.build_html(payload, out / "dashboard.html")

    pd.set_option("display.width", 200)
    print("\n=== 테스트 구간 성능 (가중치 결정에 쓰지 않은 구간) ===")
    test = results[results["split"] == "TEST"]
    for h in info:
        sub = test[test["horizon_days"] == h].sort_values("MAPE_pct")
        print(f"\n--- {h}일 (검증 n={info[h]['n_val']}, 테스트 n={info[h]['n_test']}) ---")
        print(sub[["model", "MAPE_pct", "vs_RW_pct", "DirAcc_pct", "RMSE_USD"]].round(2).to_string(index=False)
              if len(sub) else "테스트 표본 없음")
    print("\n=== 예측 ===")
    for h in payload["horizons"]:
        flag = "" if h["beats_rw"] else "  [주의: 테스트에서 Random Walk보다 나쁨]"
        print(f"{h['horizon_days']:>4}일 ({h['target_date']}): ${h['ensemble_price']:,.0f} "
              f"({h['ensemble_change_pct']:+.2f}%){flag}")
    print("\n=== 80% 변동 구간 (테스트 적중률 75~85%면 신뢰 가능) ===")
    for h in payload["horizons"]:
        iv = h["interval"]
        if iv.get("low") is None:
            continue
        mark = "신뢰 가능" if iv["calibrated"] else "주의: 구간이 잘 안 맞음"
        print(f"{h['horizon_days']:>4}일: ${iv['low']:,.0f} ~ ${iv['high']:,.0f} "
              f"({iv['low_pct']:+.1f}% ~ {iv['high_pct']:+.1f}%) | 테스트 적중률 {iv.get('coverage')}% "
              f"| 독립 표본 약 {iv['n_independent']}개 | {mark}")
    print("\n=== 상승 확률 (기준 = 과거 상승 비율) ===")
    for h in payload["horizons"]:
        d = h["direction"]
        mark = "사용 가능" if d["usable"] else "신호 없음: 기준보다 못함"
        print(f"{h['horizon_days']:>4}일: 모델 {d.get('p_up')}% / 기준 {d.get('base_p_up')}% | 테스트 정확도 "
              f"{d.get('accuracy')}% (기준 {d.get('base_accuracy')}%), 확률 오차 개선 {d.get('brier_skill_pct')}% "
              f"(가격 피처만: {d.get('price_only_brier_skill_pct')}%) | {mark}")
    print(f"\n결과 저장: {out.resolve()}")

    if not args.no_telegram:
        send_telegram(out / "chart.png", telegram_caption(payload))


if __name__ == "__main__":
    main()
