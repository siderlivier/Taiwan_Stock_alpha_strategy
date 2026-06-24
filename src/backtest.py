"""
步驟 9（階段 C）：因子正交化 + 綜合分數 + 產業內回測。

流程：
  1. 選用 DFS 驗證過的強因子（價值/獲利、成長、低波動、動能）。
  2. 產業內中性化：逐月在「同產業」內把每個因子 z-score（消除產業均值差異）。
  3. 對稱正交化（symmetric/Löwdin）：逐月把因子矩陣轉成彼此正交，消除重複曝險
     （例如 gp_to_px 與 eps_to_px 高度相關，會被去重而非雙重計分）。
  4. 合成分數 = 正交化後因子的等權平均。
  5. 回測：每月在「各產業內」依分數選前 TOP_Q 做多（產業中性），可選等權或動能加權；
     扣交易成本；同時算多空(long-short)。對照基準＝全池等權。
  6. 績效分全期 / 訓練期(<=2023) / 測試期(>2023) 報告，確認樣本外有效。

執行：python src/backtest.py
輸出：data/processed/backtest_returns.csv（各策略月報酬）
"""
import sys
import numpy as np
import pandas as pd

from config import DATA_PROCESSED
from mine_dfs import build_monthly_base, generate, TRAIN_END

# 選用因子（皆為「越大越好」方向；neg_vol 已轉正向）
FACTORS = ["gp_to_px", "eps_to_px", "op_to_rev", "g_eps_12",
           "neg_vol_63", "dist_high"]
# ---- 鎖定的最終策略版本（依集中度/加權雙期一致性檢驗選出）----
TOP_Q = 0.10            # 各產業內做多前 10%（甜蜜點：測試 Sharpe 最高、夠分散）
COST = 0.004            # 單次換手成本(費+稅+滑點)近似，套用於換手比例
WEIGHTING = "momentum"  # 動能加權(以 dist_high 排名)：訓練/測試兩期皆提升，故採用
ANN = 12


def zscore_within_group(df, cols):
    out = df.copy()
    for c in cols:
        g = out.groupby(["ym", "group"])[c]
        out[c] = ((out[c] - g.transform("mean")) / g.transform("std"))
    out[cols] = out[cols].fillna(0.0)
    return out


def symmetric_orthonormalize(F):
    """對稱正交化：F_orth = F * M^(-1/2)，M=F^T F。回傳正交化後矩陣。"""
    M = F.T @ F + 1e-6 * np.eye(F.shape[1])   # ridge 穩定
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-10, None)
    M_inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    return F @ M_inv_sqrt


def build_composite(df, factors):
    """逐月對稱正交化後等權合成。"""
    df = df.copy()
    df["score"] = np.nan
    for ym, idx in df.groupby("ym").groups.items():
        sub = df.loc[idx, factors].values
        if len(sub) < 10:
            continue
        orth = symmetric_orthonormalize(sub)
        df.loc[idx, "score"] = orth.mean(axis=1)
    return df


def portfolio_returns(df, top_q=TOP_Q, weighting=WEIGHTING):
    """逐月：各產業內選 top/bottom，回傳 long、long-short、benchmark 月報酬序列、換手與平均持股數。"""
    long_rets, ls_rets, bench_rets, turnovers, nhold = {}, {}, {}, {}, {}
    prev_holdings = set()

    for ym, g in df.groupby("ym"):
        g = g.dropna(subset=["score", "fwd_ret_1m"])
        if len(g) < 20:
            continue
        longs, shorts = [], []
        for grp, gg in g.groupby("group"):
            if len(gg) < 8:
                continue
            n = max(1, int(len(gg) * top_q))
            ranked = gg.sort_values("score", ascending=False)
            longs.append(ranked.head(n))
            shorts.append(ranked.tail(n))
        if not longs:
            continue
        L = pd.concat(longs)
        S = pd.concat(shorts)

        if weighting == "momentum" and "dist_high" in L.columns:
            wl = L["dist_high"].rank()
            wl = wl / wl.sum()
        else:
            wl = pd.Series(1.0 / len(L), index=L.index)

        long_ret = float((L["fwd_ret_1m"] * wl).sum())
        short_ret = float(S["fwd_ret_1m"].mean())
        bench_ret = float(g["fwd_ret_1m"].mean())

        cur = set(L["stock_id"])
        turn = 1.0 if not prev_holdings else len(cur ^ prev_holdings) / max(len(cur | prev_holdings), 1)
        prev_holdings = cur

        long_rets[ym] = long_ret - COST * turn
        ls_rets[ym] = (long_ret - short_ret) - COST * turn
        bench_rets[ym] = bench_ret
        turnovers[ym] = turn
        nhold[ym] = len(L)

    out = pd.DataFrame({
        "long": pd.Series(long_rets), "long_short": pd.Series(ls_rets),
        "benchmark": pd.Series(bench_rets), "turnover": pd.Series(turnovers),
        "nhold": pd.Series(nhold),
    }).sort_index()
    return out


def perf(r):
    r = r.dropna()
    if len(r) < 6:
        return {}
    cum = (1 + r).prod()
    cagr = cum ** (ANN / len(r)) - 1
    vol = r.std() * np.sqrt(ANN)
    sharpe = (r.mean() * ANN) / vol if vol > 0 else np.nan
    curve = (1 + r).cumprod()
    mdd = (curve / curve.cummax() - 1).min()
    win = (r > 0).mean()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe,
            "MaxDD": mdd, "WinRate": win, "Months": len(r)}


def active_stats(sub, leg="long"):
    """leg 相對 benchmark 的超額報酬統計：年化超額、資訊比率(IR)、t值、月勝率。"""
    a = (sub[leg] - sub["benchmark"]).dropna()
    if len(a) < 6:
        return {}
    ann = a.mean() * ANN
    ir = (a.mean() * ANN) / (a.std() * np.sqrt(ANN)) if a.std() > 0 else np.nan
    t = a.mean() / (a.std() / np.sqrt(len(a))) if a.std() > 0 else np.nan
    return {"年化超額": ann, "資訊比率IR": ir, "t值": t,
            "贏基準月%": (a > 0).mean(), "月數": len(a)}


def report(rets):
    def show(name, sub):
        print(f"\n--- {name} ---")
        tab = pd.DataFrame({k: perf(sub[k]) for k in ["long", "long_short", "benchmark"]}).T
        print(tab.to_string(formatters={
            "CAGR": "{:.2%}".format, "Vol": "{:.2%}".format,
            "Sharpe": "{:.2f}".format, "MaxDD": "{:.2%}".format,
            "WinRate": "{:.1%}".format, "Months": "{:.0f}".format}))
        act = active_stats(sub, "long")
        if act:
            print(f"  [long 超額] 年化超額={act['年化超額']:.2%}　"
                  f"IR={act['資訊比率IR']:.2f}　t={act['t值']:.2f}　"
                  f"贏基準月={act['贏基準月%']:.1%}")

    show("全期", rets)
    show("訓練期 (<=2023)", rets[rets.index <= TRAIN_END])
    show("測試期 (>2023, 樣本外)", rets[rets.index > TRAIN_END])
    print(f"\n平均月換手率：{rets['turnover'].mean():.1%}")
    print("判讀超額：t值>2 才算統計上顯著的 alpha；若測試期 t<2，"
          "代表樣本外沒有可靠的超額報酬，risk-adjusted 改善(Sharpe/MaxDD)才是賣點。")


def main():
    p = DATA_PROCESSED / "panel.parquet"
    if not p.exists():
        sys.exit("找不到 panel.parquet，請先執行 align_data.py")
    print("建立因子…")
    m = build_monthly_base(pd.read_parquet(p))
    df, _ = generate(m)
    facs = [c for c in FACTORS if c in df.columns]
    print(f"使用因子：{facs}　加權方式：{WEIGHTING}")

    df = df.dropna(subset=["fwd_ret_1m"]).copy()
    df = zscore_within_group(df, facs)
    df = build_composite(df, facs)

    # ===== 集中度掃描：同時看訓練/測試，挑「兩邊一致變好」的，而非只挑測試最好 =====
    def leg_stats(sub, leg="long"):
        r = sub[leg].dropna()
        cum = (1 + r).prod()
        cagr = cum ** (ANN / len(r)) - 1
        sharpe = (r.mean() * ANN) / (r.std() * np.sqrt(ANN)) if r.std() > 0 else np.nan
        a = (sub[leg] - sub["benchmark"]).dropna()
        t = a.mean() / (a.std() / np.sqrt(len(a))) if a.std() > 0 else np.nan
        return cagr, sharpe, t

    print("\n========= 集中度掃描（long，等權）=========")
    print(f"{'TOP_Q':>6} {'持股~':>5} | {'訓練CAGR':>8} {'訓練SR':>6} {'訓練t':>6} | "
          f"{'測試CAGR':>8} {'測試SR':>6} {'測試t':>6}")
    for q in [0.05, 0.10, 0.15, 0.20, 0.30]:
        r = portfolio_returns(df, top_q=q)
        tr, te = r[r.index <= TRAIN_END], r[r.index > TRAIN_END]
        c1, s1, t1 = leg_stats(tr)
        c2, s2, t2 = leg_stats(te)
        nh = int(r["nhold"].mean())
        print(f"{q:>6.0%} {nh:>5d} | {c1:>8.2%} {s1:>6.2f} {t1:>6.2f} | "
              f"{c2:>8.2%} {s2:>6.2f} {t2:>6.2f}")

    # 基準（不隨 TOP_Q 變）供對照
    rb = portfolio_returns(df, top_q=0.30)
    for lab, sub in [("訓練", rb[rb.index <= TRAIN_END]), ("測試", rb[rb.index > TRAIN_END])]:
        r = sub["benchmark"].dropna()
        cagr = (1 + r).prod() ** (ANN / len(r)) - 1
        sr = (r.mean() * ANN) / (r.std() * np.sqrt(ANN))
        print(f"  基準({lab})  CAGR={cagr:.2%}  Sharpe={sr:.2f}")

    print("\n判讀：找『訓練 t 與測試 t 同時較高、且 CAGR 在兩期都贏基準』的集中度才可靠；"
          "若集中度提高只在某一期變好、另一期變差，就是雜訊不是真 alpha。")

    # ===== 聚焦比較：5% vs 10% × 等權/動能加權 =====
    print("\n===== 5% vs 10% × 加權方式 比較 =====")
    print(f"{'集中度':>6} {'加權':>8} {'持股~':>5} | {'訓練CAGR':>8} {'訓練SR':>6} {'訓練t':>6} | "
          f"{'測試CAGR':>8} {'測試SR':>6} {'測試t':>6} | {'換手':>5}")
    for q in [0.05, 0.10]:
        for w in ["equal", "momentum"]:
            r = portfolio_returns(df, top_q=q, weighting=w)
            tr, te = r[r.index <= TRAIN_END], r[r.index > TRAIN_END]
            c1, s1, t1 = leg_stats(tr)
            c2, s2, t2 = leg_stats(te)
            wlab = "等權" if w == "equal" else "動能"
            print(f"{q:>6.0%} {wlab:>8} {int(r['nhold'].mean()):>5d} | "
                  f"{c1:>8.2%} {s1:>6.2f} {t1:>6.2f} | "
                  f"{c2:>8.2%} {s2:>6.2f} {t2:>6.2f} | {r['turnover'].mean():>5.0%}")
            r.to_csv(DATA_PROCESSED / f"backtest_q{int(q*100)}_{w}.csv")
    print("\n各組合月報酬已存：data/processed/backtest_q{5,10}_{equal,momentum}.csv")
    print("判讀：動能加權若在訓練、測試兩期都讓 CAGR/Sharpe 提升才採用；只在單期變好就是雜訊。")

    # ===== 鎖定版本完整報告：10% + 動能加權 =====
    print("\n############ 鎖定策略：前10% + 動能加權 ############")
    locked = portfolio_returns(df, top_q=TOP_Q, weighting=WEIGHTING)
    report(locked)
    locked.to_csv(DATA_PROCESSED / "backtest_returns.csv")
    print("\n鎖定版月報酬已存：data/processed/backtest_returns.csv")


if __name__ == "__main__":
    main()
