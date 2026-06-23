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
TOP_Q = 0.30            # 各產業內做多前 30%、做空後 30%
COST = 0.004            # 單次換手成本(費+稅+滑點)近似，套用於換手比例
WEIGHTING = "equal"     # "equal" 或 "momentum"(以 dist_high 排名加權)
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


def build_composite(df):
    """逐月對稱正交化後等權合成。"""
    df = df.copy()
    df["score"] = np.nan
    for ym, idx in df.groupby("ym").groups.items():
        sub = df.loc[idx, FACTORS].values
        if len(sub) < 10:
            continue
        orth = symmetric_orthonormalize(sub)
        df.loc[idx, "score"] = orth.mean(axis=1)
    return df


def portfolio_returns(df):
    """逐月：各產業內選 top/bottom，回傳 long、long-short、benchmark 月報酬序列與換手。"""
    long_rets, ls_rets, bench_rets, turnovers = {}, {}, {}, {}
    prev_holdings = set()

    for ym, g in df.groupby("ym"):
        g = g.dropna(subset=["score", "fwd_ret_1m"])
        if len(g) < 20:
            continue
        longs, shorts = [], []
        for grp, gg in g.groupby("group"):
            if len(gg) < 8:
                continue
            n = max(1, int(len(gg) * TOP_Q))
            ranked = gg.sort_values("score", ascending=False)
            longs.append(ranked.head(n))
            shorts.append(ranked.tail(n))
        if not longs:
            continue
        L = pd.concat(longs)
        S = pd.concat(shorts)

        # 權重
        if WEIGHTING == "momentum" and "dist_high" in L.columns:
            wl = L["dist_high"].rank()
            wl = wl / wl.sum()
        else:
            wl = pd.Series(1.0 / len(L), index=L.index)

        long_ret = float((L["fwd_ret_1m"] * wl).sum())
        short_ret = float(S["fwd_ret_1m"].mean())
        bench_ret = float(g["fwd_ret_1m"].mean())

        # 換手：與上月持股的差異比例
        cur = set(L["stock_id"])
        turn = 1.0 if not prev_holdings else len(cur ^ prev_holdings) / max(len(cur | prev_holdings), 1)
        prev_holdings = cur

        long_rets[ym] = long_ret - COST * turn
        ls_rets[ym] = (long_ret - short_ret) - COST * turn
        bench_rets[ym] = bench_ret
        turnovers[ym] = turn

    out = pd.DataFrame({
        "long": pd.Series(long_rets), "long_short": pd.Series(ls_rets),
        "benchmark": pd.Series(bench_rets), "turnover": pd.Series(turnovers),
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


def report(rets):
    def show(name, sub):
        print(f"\n--- {name} ---")
        tab = pd.DataFrame({k: perf(sub[k]) for k in ["long", "long_short", "benchmark"]}).T
        print(tab.to_string(formatters={
            "CAGR": "{:.2%}".format, "Vol": "{:.2%}".format,
            "Sharpe": "{:.2f}".format, "MaxDD": "{:.2%}".format,
            "WinRate": "{:.1%}".format, "Months": "{:.0f}".format}))

    show("全期", rets)
    show("訓練期 (<=2023)", rets[rets.index <= TRAIN_END])
    show("測試期 (>2023, 樣本外)", rets[rets.index > TRAIN_END])
    print(f"\n平均月換手率：{rets['turnover'].mean():.1%}")


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
    global FACTORS
    FACTORS = facs
    df = build_composite(df)

    rets = portfolio_returns(df)
    out = DATA_PROCESSED / "backtest_returns.csv"
    rets.to_csv(out)

    print("\n================ 回測績效 ================")
    report(rets)
    print(f"\n月報酬序列已存：{out}")
    print("\n判讀：看『測試期』long 與 long_short 是否仍明顯優於 benchmark、"
          "Sharpe 是否夠高、MaxDD 是否可接受。樣本外站得住才算數。")


if __name__ == "__main__":
    main()
