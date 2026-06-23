"""
步驟 6：因子評估（產業內 rank IC / ICIR）。

對每個因子，逐月在「同產業」橫斷面計算 Spearman rank IC
（因子值 vs 未來1月報酬的等級相關），再跨產業平均成當月 IC。
彙總成 IC 平均、ICIR、t 值、勝率，輸出因子排名。

ICIR = mean(IC) / std(IC)，衡量因子的「穩定解釋力」，是後續 GP 的適應度核心。

執行：
    python src/factor_eval.py
"""
import sys
import numpy as np
import pandas as pd

from config import DATA_PROCESSED

MIN_STOCKS = 8   # 每個產業每月至少要這麼多檔才計算 IC，避免小樣本雜訊


def monthly_ic(df, factor, ret="fwd_ret_1m"):
    """回傳每月 IC 的時間序列（產業內計算後跨產業平均）。"""
    recs = []
    for ym, g in df.groupby("ym"):
        ics = []
        for grp, gg in g.groupby("group"):
            sub = gg[[factor, ret]].dropna()
            if len(sub) < MIN_STOCKS:
                continue
            if sub[factor].nunique() < 2 or sub[ret].nunique() < 2:
                continue   # 零變異無法算相關，跳過（避免 divide warning）
            ic = sub[factor].rank().corr(sub[ret].rank())  # Spearman
            if pd.notna(ic):
                ics.append(ic)
        if ics:
            recs.append((ym, np.mean(ics)))
    return pd.Series(dict(recs)).sort_index()


def monthly_ic_groups(df, factor, ret="fwd_ret_1m"):
    """回傳 {產業: 該產業逐月 IC 時間序列}，用來看因子是否為產業專屬。"""
    out = {}
    for grp, g in df.groupby("group"):
        recs = []
        for ym, gg in g.groupby("ym"):
            sub = gg[[factor, ret]].dropna()
            if len(sub) < MIN_STOCKS:
                continue
            if sub[factor].nunique() < 2 or sub[ret].nunique() < 2:
                continue
            ic = sub[factor].rank().corr(sub[ret].rank())
            if pd.notna(ic):
                recs.append((ym, ic))
        if recs:
            out[grp] = pd.Series(dict(recs)).sort_index()
    return out


def summarize(ic: pd.Series):
    n = len(ic)
    mean = ic.mean()
    std = ic.std()
    icir = mean / std if std > 0 else np.nan
    t = icir * np.sqrt(n) if n > 0 else np.nan      # 近似 t 值
    hit = (ic > 0).mean()
    return mean, std, icir, t, hit, n


def main():
    p = DATA_PROCESSED / "factors_monthly.parquet"
    if not p.exists():
        sys.exit("找不到 factors_monthly.parquet，請先執行 build_factors.py")
    df = pd.read_parquet(p)

    non_factor = {"stock_id", "stock_name", "group", "ym", "date",
                  "close", "fwd_ret_1m"}
    factors = [c for c in df.columns if c not in non_factor]

    rows = []
    for f in factors:
        ic = monthly_ic(df, f)
        if len(ic) == 0:
            continue
        mean, std, icir, t, hit, n = summarize(ic)
        rows.append({
            "factor": f, "IC_mean": mean, "IC_std": std,
            "ICIR": icir, "t_stat": t, "hit_rate": hit, "months": n,
        })

    res = pd.DataFrame(rows).sort_values("ICIR", key=lambda s: s.abs(),
                                         ascending=False)
    pd.set_option("display.width", 140)
    print("=== 因子評估（依 |ICIR| 排序）===")
    print(res.to_string(index=False,
          formatters={"IC_mean": "{:.4f}".format, "IC_std": "{:.4f}".format,
                      "ICIR": "{:.3f}".format, "t_stat": "{:.2f}".format,
                      "hit_rate": "{:.1%}".format}))

    out = DATA_PROCESSED / "factor_ic_summary.csv"
    res.to_csv(out, index=False)
    print(f"\n已存：{out}")
    print("\n判讀：|IC_mean|>0.02、|ICIR|>0.3、|t|>2 通常代表有意義的因子；"
          "正負號代表方向（負號=因子越大未來報酬越低）。")


if __name__ == "__main__":
    main()
