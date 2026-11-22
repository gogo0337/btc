#!/usr/bin/env python3
"""반감기·ETF 승인 전후 구간 검증.

원본(gemini)과 달리 구간 안에서 가중치를 맞추지 않는다.
가중치는 구간 시작 전 데이터로만 정하고, 구간 안에서는 채점만 한다.
Binance BTCUSDT는 2017-08부터 있어 2016년 2차 반감기는 다루지 못한다.

  python backtest_events.py --horizon 1
"""
import argparse

import pandas as pd

import backtest as bt
from data import fetch_daily

EVENTS = [("3차 반감기 ±500일", "2020-05-11", 500),
          ("미국 현물 ETF 승인 ±100일", "2024-01-10", 100),
          ("4차 반감기 ±500일", "2024-04-20", 500)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1, help="예측 기간(일)")
    ap.add_argument("--out", default="data/event_backtest.csv")
    args = ap.parse_args()
    h = args.horizon

    df = fetch_daily(years=12)
    tbl = bt.prediction_table(df, h, stride=1)
    target = tbl["date"] + pd.Timedelta(days=h)
    rows = []
    for name, day, span in EVENTS:
        center = pd.Timestamp(day)
        start, end = center - pd.Timedelta(days=span), center + pd.Timedelta(days=span)
        train = tbl[target < start]
        window = tbl[(tbl["date"] >= start) & (target <= end)]
        if len(train) < 100 or window.empty:
            print(f"{name}: 구간 이전 데이터가 부족해 건너뜁니다 (train n={len(train)}).")
            continue
        for r in bt._score(window, bt.fit_weights(train), h, name):
            rows.append({"event": name, "train_n": len(train), **r})
    if not rows:
        print("채점할 구간이 없습니다.")
        return
    res = pd.DataFrame(rows).drop(columns=["split"]).round(3)
    res.to_csv(args.out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    for name, sub in res.groupby("event", sort=False):
        print(f"\n=== {name} · {h}일 예측 (가중치 학습 n={sub['train_n'].iloc[0]}) ===")
        print(sub.sort_values("MAPE_pct")[["model", "MAPE_pct", "vs_RW_pct", "DirAcc_pct", "n"]].to_string(index=False))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
