"""
步驟 5：建立月頻因子面板。

由每日 panel.parquet 取「每月最後交易日」快照，計算一組經典因子與
「未來 1 個月報酬」標的（label）。所有因子只用當月底(含)以前的資料，無前瞻偏差；
基本面欄位在 panel 已是 point-in-time（依可得日對齊），直接取用即可。

輸出：data/processed/factors_monthly.parquet

執行：
    python src/build_factors.py
"""
import sys
import numpy as np
import pandas as pd

from config import DATA_PROCESSED


def winsorize_by_month(s: pd.Series, ym: pd.Series, lo=0.01, hi=0.99):
    """逐月橫斷面 winsorize（截尾），降低極端值（含減資假跳空）影響。"""
    def clip(x):
        ql, qh = x.quantile(lo), x.quantile(hi)
        return x.clip(ql, qh)
    return s.groupby(ym).transform(clip)


def main():
    p = DATA_PROCESSED / "panel.parquet"
    if not p.exists():
        sys.exit("找不到 panel.parquet，請先執行 align_data.py")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    df["ym"] = df["date"].dt.to_period("M")

    fcols = [c for c in df.columns if c.startswith("f_")]
    print(f"可用財報欄位（{len(fcols)}）：{fcols}")

    # --- 日頻滾動指標（之後取月底快照）---
    grp = df.groupby("stock_id")
    df["vol_63"] = grp["ret"].transform(lambda s: s.rolling(63, min_periods=40).std())
    df["amt_21"] = grp["amount"].transform(lambda s: s.rolling(21, min_periods=10).mean())
    # 過去一年最大回撤距離（價格動能輔助）
    df["hi_252"] = grp["close"].transform(lambda s: s.rolling(252, min_periods=120).max())

    # --- 取每月最後交易日快照 ---
    m = df.groupby(["stock_id", "ym"]).tail(1).copy()
    m = m.sort_values(["stock_id", "ym"]).reset_index(drop=True)
    gc = m.groupby("stock_id")

    # === 技術面因子（用月底還原收盤）===
    m["mom_1m"] = gc["close"].pct_change(1, fill_method=None)    # 短期反轉用
    m["mom_3m"] = gc["close"].pct_change(3, fill_method=None)
    m["mom_6m"] = gc["close"].pct_change(6, fill_method=None)
    m["mom_12m"] = gc["close"].pct_change(12, fill_method=None)
    # 12-1 動能（跳過最近一個月，經典動能定義）
    m["mom_12_1"] = gc["close"].shift(1) / gc["close"].shift(12) - 1
    m["vol_63"] = m["vol_63"]                        # 低波動因子（取負相關）
    m["liq_amt"] = np.log(m["amt_21"] + 1)          # 流動性/規模代理
    m["dist_high"] = m["close"] / m["hi_252"] - 1   # 距一年高點（動能）

    # === 基本面因子（panel 已 PIT 對齊）===
    m["rev_yoy"] = gc["month_revenue"].pct_change(12, fill_method=None)   # 年增率
    m["rev_mom"] = gc["month_revenue"].pct_change(1, fill_method=None)    # 月增率
    if "f_EPS" in m.columns:
        m["earn_yield"] = m["f_EPS"] / m["close"]       # 盈餘殖利率(近似)
    if "f_GrossProfit" in m.columns and "f_Revenue" in m.columns:
        m["gross_margin"] = m["f_GrossProfit"] / m["f_Revenue"]
    if "f_EquityAttributableToOwnersOfParent" in m.columns and "f_IncomeAfterTaxes" in m.columns:
        m["roe_proxy"] = m["f_IncomeAfterTaxes"] / m["f_EquityAttributableToOwnersOfParent"]

    # === 標的：未來 1 個月報酬 ===
    m["ret_1m"] = gc["close"].pct_change(1, fill_method=None)
    m["fwd_ret_1m"] = gc["ret_1m"].shift(-1)            # 下個月報酬
    # 對標的逐月 winsorize（壓制減資/特殊事件假極端）
    m["fwd_ret_1m"] = winsorize_by_month(m["fwd_ret_1m"], m["ym"])

    factor_cols = [c for c in [
        "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_1",
        "vol_63", "liq_amt", "dist_high",
        "rev_yoy", "rev_mom", "earn_yield", "gross_margin", "roe_proxy",
    ] if c in m.columns]

    # 因子也逐月 winsorize
    for c in factor_cols:
        m[c] = winsorize_by_month(m[c], m["ym"])

    keep = ["stock_id", "stock_name", "group", "ym", "date", "close",
            "fwd_ret_1m"] + factor_cols
    out = m[keep].copy()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    op = DATA_PROCESSED / "factors_monthly.parquet"
    out.to_parquet(op, index=False)

    print(f"\n=== 月頻因子面板：{op} ===")
    print(f"列數：{len(out):,}　股票：{out['stock_id'].nunique()}　"
          f"月份：{out['ym'].nunique()}")
    print(f"因子（{len(factor_cols)}）：{factor_cols}")
    print("\n各因子非缺值比例：")
    for c in factor_cols:
        print(f"  {c:14} {out[c].notna().mean():.1%}")
    print(f"\nfwd_ret_1m 非缺值：{out['fwd_ret_1m'].notna().mean():.1%}")


if __name__ == "__main__":
    main()
