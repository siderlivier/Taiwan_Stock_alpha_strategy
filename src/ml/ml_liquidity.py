"""
LightGBM 策略的『流動性歸因』與『流動性過濾 × 成本』壓力測試。

目的：檢驗 IR 2.85 有多少來自「難以實際交易的小型/低流動股」。
流動性用『成交金額』衡量（amt_21 = 過去 21 交易日平均成交金額，往回看、無前瞻）。

(A) 歸因：持股的流動性百分位分布、分低/中/高流動三層的報酬貢獻、貢獻最大個股(含名稱)。
(B) 過濾×成本：每月剔除底 0/30/50/70% 低流動股後重跑，並在每個過濾層再掃成本 0.4/0.8/1.2%。

前置：先跑 src/ml/ml_model.py 產生 ml_scores.parquet。
執行：python src/ml/ml_liquidity.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW
from mine_dfs import build_monthly_base, generate
import backtest as bt

ANN = 12
SCORE = "s_LightGBM"


def excess_stats(rets):
    a = (rets["long"] - rets["benchmark"]).dropna()
    t = a.mean() / (a.std() / np.sqrt(len(a))) if a.std() > 0 else np.nan
    return a.mean() * ANN, t


def main():
    if not (DATA_PROCESSED / "ml_scores.parquet").exists():
        sys.exit("找不到 ml_scores.parquet，請先執行 src/ml/ml_model.py")
    print("建立面板 + 讀取模型分數與流動性…")
    m = build_monthly_base(pd.read_parquet(DATA_PROCESSED / "panel.parquet"))
    df, names = generate(m)
    ml = pd.read_parquet(DATA_PROCESSED / "ml_scores.parquet")
    liq = m[["stock_id", "ym", "amt_21"]]
    uni = pd.read_parquet(DATA_RAW / "universe.parquet")
    name_map = dict(zip(uni["stock_id"], uni["stock_name"]))

    d = (df.merge(ml[["stock_id", "ym", SCORE]], on=["stock_id", "ym"])
           .merge(liq, on=["stock_id", "ym"]))
    oos = d[d[SCORE].notna()].dropna(subset=["amt_21", "fwd_ret_1m"]).copy()
    oos["liq_pct"] = oos.groupby("ym")["amt_21"].rank(pct=True)   # 當月流動性百分位

    # ================= (A) 歸因 =================
    holds = []
    for ym, g in oos.groupby("ym"):
        picks = []
        for grp, gg in g.groupby("group"):
            n = max(1, int(len(gg) * 0.10))
            picks.append(gg.sort_values(SCORE, ascending=False).head(n))
        h = pd.concat(picks).copy()
        h["w"] = 1.0 / len(h)
        h["contrib"] = h["w"] * h["fwd_ret_1m"]
        holds.append(h)
    H = pd.concat(holds)

    print("\n================ (A) 流動性歸因 ================")
    print(f"持股的『流動性百分位』平均：{H['liq_pct'].mean():.2f}"
          f"（0.5=市場中位；越低=越偏小型/低流動）")
    H["tier"] = pd.cut(H["liq_pct"], [0, 1/3, 2/3, 1.0],
                       labels=["低流動", "中流動", "高流動"], include_lowest=True)
    tier_sum = H.groupby("tier", observed=True)["contrib"].sum()
    tier_cnt = H.groupby("tier", observed=True).size()
    print("\n分流動性三層的『報酬貢獻』：")
    print(f"{'層級':>6} {'持股次數':>8} {'貢獻佔比':>8} {'平均月報酬':>10}")
    for t in ["低流動", "中流動", "高流動"]:
        share = tier_sum[t] / tier_sum.sum()
        avg = H[H["tier"] == t]["fwd_ret_1m"].mean()
        print(f"{t:>6} {tier_cnt[t]:>8} {share:>8.1%} {avg:>10.2%}")
    print("→ 若『低流動』層貢獻佔比過高，代表 edge 大半來自難交易的小型股。")

    tot = H.groupby("stock_id").agg(total_contrib=("contrib", "sum"),
                                    avg_liqpct=("liq_pct", "mean"),
                                    months=("contrib", "size"))
    tot["name"] = tot.index.map(name_map).fillna("?")
    top = tot.sort_values("total_contrib", ascending=False).head(20)
    print("\n貢獻最大的 20 檔（含流動性百分位）：")
    print(top[["name", "total_contrib", "avg_liqpct", "months"]].to_string(
        formatters={"total_contrib": "{:.3f}".format, "avg_liqpct": "{:.2f}".format}))

    # ============ (B) 流動性過濾 × 成本 ============
    print("\n============ (B) 流動性過濾 × 成本壓力測試 ============")
    print(f"{'剔除底X%':>8} {'持股~':>5} | {'CAGR@0.4%':>9} {'Sharpe':>6} "
          f"{'超額CAGR':>8} {'超額t':>6} {'換手':>5} | {'CAGR@0.8%':>9} {'CAGR@1.2%':>9}")
    orig = bt.COST
    for drop in [0.0, 0.3, 0.5, 0.7]:
        sub = oos if drop <= 0 else oos[oos["liq_pct"] >= drop]
        dd = sub[["stock_id", "group", "ym", "fwd_ret_1m", SCORE]].rename(
            columns={SCORE: "score"})
        res = {}
        for cst in [0.004, 0.008, 0.012]:
            bt.COST = cst
            res[cst] = bt.portfolio_returns(dd, top_q=0.10, weighting="equal")
        bt.COST = orig
        r0 = res[0.004]
        pf = bt.perf(r0["long"])
        exc, t = excess_stats(r0)
        nh = int(r0["nhold"].mean())
        print(f"{drop:>8.0%} {nh:>5d} | {pf['CAGR']:>9.2%} {pf['Sharpe']:>6.2f} "
              f"{exc:>8.2%} {t:>6.2f} {r0['turnover'].mean():>5.0%} | "
              f"{bt.perf(res[0.008]['long'])['CAGR']:>9.2%} "
              f"{bt.perf(res[0.012]['long'])['CAGR']:>9.2%}")
    print("\n判讀：若『剔除越多低流動股 → 超額/Sharpe 越掉』，代表 IR 2.85 大半是小型股假象；"
          "若過濾後仍穩健，代表 edge 較實在。年化 IR ≈ 超額CAGR/波動，可與 2.85 對照。")


if __name__ == "__main__":
    main()
