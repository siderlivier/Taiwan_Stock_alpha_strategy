"""
régime 分解：LightGBM 選股『超額報酬』在不同市場狀態下的表現。

目的：檢驗高 IR 是否只是『剛好處在對因子友善的多頭régime』。
做法：把樣本外月份依『大盤(全池等權基準)漲跌』分組，分別算超額的 IR/t/勝率；
      並附逐年超額。若『大盤下跌/最差月份』仍有正且顯著的超額 → 是真選股技巧，非多頭順風車。

同時對『全體』與『可交易版(最流動 30%)』各做一次。

前置：先跑 src/ml/ml_model.py。
執行：python src/ml/ml_regime.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED
from mine_dfs import build_monthly_base, generate
import backtest as bt

ANN = 12
SCORE = "s_LightGBM"


def stats(ex):
    ex = ex.dropna()
    n = len(ex)
    if n < 4:
        return None
    ann = ex.mean() * ANN
    ir = (ex.mean() * ANN) / (ex.std() * np.sqrt(ANN)) if ex.std() > 0 else np.nan
    t = ex.mean() / (ex.std() / np.sqrt(n)) if ex.std() > 0 else np.nan
    return {"月數": n, "年化超額": ann, "IR": ir, "t": t, "勝率": (ex > 0).mean()}


def regime_report(name, rets):
    ex = (rets["long"] - rets["benchmark"]).dropna()
    bm = rets["benchmark"].reindex(ex.index)
    q25 = bm.quantile(0.25)
    buckets = [("全部月份", ex.index),
               ("大盤上漲月", bm[bm > 0].index),
               ("大盤下跌月", bm[bm <= 0].index),
               ("最差 25% 月", bm[bm <= q25].index)]
    print(f"\n===== {name} =====")
    print(f"{'狀態':>12} {'月數':>5} {'年化超額':>9} {'IR':>6} {'t':>6} {'勝率':>6}")
    for lab, idx in buckets:
        s = stats(ex.reindex(idx))
        if s:
            print(f"{lab:>12} {s['月數']:>5d} {s['年化超額']:>9.2%} "
                  f"{s['IR']:>6.2f} {s['t']:>6.2f} {s['勝率']:>6.1%}")
    # 逐年超額（複利）
    years = [p.year for p in ex.index]
    annual = [(y, (1 + g).prod() - 1) for y, g in ex.groupby(years)]
    print("  逐年超額：", "　".join(f"{int(y)}:{v:+.1%}" for y, v in annual))


def main():
    if not (DATA_PROCESSED / "ml_scores.parquet").exists():
        sys.exit("找不到 ml_scores.parquet，請先執行 src/ml/ml_model.py")
    print("建立面板 + 讀取模型分數與流動性…")
    m = build_monthly_base(pd.read_parquet(DATA_PROCESSED / "panel.parquet"))
    df, names = generate(m)
    ml = pd.read_parquet(DATA_PROCESSED / "ml_scores.parquet")
    liq = m[["stock_id", "ym", "amt_21"]]
    d = (df.merge(ml[["stock_id", "ym", SCORE]], on=["stock_id", "ym"])
           .merge(liq, on=["stock_id", "ym"]))
    oos = d[d[SCORE].notna()].dropna(subset=["amt_21", "fwd_ret_1m"]).copy()
    oos["liq_pct"] = oos.groupby("ym")["amt_21"].rank(pct=True)

    def run(sub):
        dd = sub[["stock_id", "group", "ym", "fwd_ret_1m", SCORE]].rename(
            columns={SCORE: "score"})
        return bt.portfolio_returns(dd, top_q=0.10, weighting="equal")

    regime_report("全體股票池（帳面 IR 2.85）", run(oos))
    regime_report("可交易版（最流動 30%）", run(oos[oos["liq_pct"] >= 0.7]))

    print("\n判讀：若『大盤下跌月/最差月』的超額 IR、t 仍為正且不小 → 選股 alpha 在逆風時"
          "也有效，代表不是多頭régime的順風車；若下跌月超額大幅轉負 → 高 IR 相當程度靠多頭。")


if __name__ == "__main__":
    main()
