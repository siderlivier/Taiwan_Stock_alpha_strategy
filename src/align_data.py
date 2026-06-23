"""
步驟 4：發布日對齊（Point-in-Time），產出每日合併面板。

核心原則：任何基本面資料，只有在「實際可得日」之後才能用於當天，避免預知未來。
  - 月營收：所屬月份由 revenue_year/revenue_month 決定；法規須於次月 10 日前公告，
            故保守設「可得日 = 次月 10 日」，之後才 forward-fill 到每個交易日。
  - 財報(季)：以財報期末 + 法定申報期限近似公告日：
            Q1(3/31)->5/15、Q2(6/30)->8/14、Q3(9/30)->11/14、Q4(12/31)->隔年 3/31。
  - 對齊用 merge_asof(backward)：每個交易日取「可得日 <= 當日」的最新一筆。

輸出：data/processed/panel.parquet
  欄位：date, stock_id, industry(group), 還原 OHLCV, ret(日報酬),
        month_revenue(+可得日), 財報各科目 f_*(+可得日)

執行：
    python src/align_data.py
"""
import sys
import numpy as np
import pandas as pd

from config import DATA_RAW, DATA_PROCESSED


def load(name):
    p = DATA_RAW / name
    if not p.exists():
        sys.exit(f"找不到 {p}，請先完成抓取步驟")
    return pd.read_parquet(p)


def build_price_base(prices, universe):
    """每日還原價基底 + 日報酬，掛上產業別。"""
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    # 用還原價為標準價格欄
    out = df[["date", "stock_id", "Trading_Volume", "Trading_money",
              "adj_open", "adj_max", "adj_min", "adj_close"]].rename(columns={
        "adj_open": "open", "adj_max": "high",
        "adj_min": "low", "adj_close": "close", "Trading_Volume": "volume",
        "Trading_money": "amount"})
    out["ret"] = out.groupby("stock_id")["close"].pct_change()
    # 掛產業別
    g = universe[["stock_id", "group", "stock_name"]]
    out = out.merge(g, on="stock_id", how="left")
    return out


def prep_revenue(rev):
    """月營收 → 加可得日(次月10日)。"""
    df = rev.copy()
    df = df[["stock_id", "revenue", "revenue_year", "revenue_month"]].dropna(
        subset=["revenue_year", "revenue_month"])
    df["revenue_year"] = df["revenue_year"].astype(int)
    df["revenue_month"] = df["revenue_month"].astype(int)
    # 可得日 = 所屬月的次月 10 日（法規公告期限，保守）
    df["rev_avail"] = (
        pd.to_datetime(dict(year=df["revenue_year"], month=df["revenue_month"], day=1))
        + pd.DateOffset(months=1, days=9))
    df = df.rename(columns={"revenue": "month_revenue"})
    df = df.sort_values("rev_avail")
    return df[["stock_id", "rev_avail", "month_revenue",
               "revenue_year", "revenue_month"]]


def _stmt_deadline(period_end: pd.Timestamp) -> pd.Timestamp:
    """財報期末 → 法定申報期限（近似公告日）。"""
    m = period_end.month
    y = period_end.year
    if m == 3:
        return pd.Timestamp(y, 5, 15)
    if m == 6:
        return pd.Timestamp(y, 8, 14)
    if m == 9:
        return pd.Timestamp(y, 11, 14)
    if m == 12:
        return pd.Timestamp(y + 1, 3, 31)
    # 非標準期末，保守 +90 天
    return period_end + pd.Timedelta(days=90)


def prep_financials(fin):
    """綜合損益表 long → wide，加可得日(法定期限)。"""
    df = fin.copy()
    df = df[df["type"].astype(str) != "-"]
    df["date"] = pd.to_datetime(df["date"])
    # long -> wide（同一 期末/公司/科目 取一筆）
    wide = df.pivot_table(index=["stock_id", "date"], columns="type",
                          values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    # 科目欄前綴 f_
    acc_cols = [c for c in wide.columns if c not in ("stock_id", "date")]
    wide = wide.rename(columns={c: f"f_{c}" for c in acc_cols})
    wide["fin_avail"] = wide["date"].apply(_stmt_deadline)
    wide = wide.rename(columns={"date": "fin_period_end"})
    wide = wide.sort_values("fin_avail")
    return wide, [f"f_{c}" for c in acc_cols]


def asof_merge(base, right, right_time_col):
    """以 merge_asof(backward) 把 right 對齊到 base 的每個交易日（同 stock_id）。"""
    left = base.sort_values("date")
    right = right.sort_values(right_time_col)
    merged = pd.merge_asof(
        left, right,
        left_on="date", right_on=right_time_col,
        by="stock_id", direction="backward")
    return merged


def main():
    prices = load("prices.parquet")
    rev = load("fund_revenue.parquet")
    fin = load("fund_financials.parquet")
    universe = load("universe.parquet")

    print("建立每日價格基底…")
    base = build_price_base(prices, universe)
    print(f"  基底 {len(base):,} 列、{base['stock_id'].nunique()} 檔")

    print("對齊月營收（可得日=次月10日）…")
    rev_p = prep_revenue(rev)
    base = asof_merge(base, rev_p, "rev_avail")

    print("對齊財報（法定申報期限近似）…")
    fin_w, fcols = prep_financials(fin)
    base = asof_merge(base, fin_w, "fin_avail")

    # 整理欄位順序
    base = base.sort_values(["stock_id", "date"]).reset_index(drop=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "panel.parquet"
    base.to_parquet(out, index=False)

    # 覆蓋率報告
    n = len(base)
    print(f"\n=== 對齊完成：{out} ===")
    print(f"總列數：{n:,}　股票數：{base['stock_id'].nunique()}")
    print(f"有月營收的列：{base['month_revenue'].notna().mean():.1%}")
    fin_have = base[fcols].notna().any(axis=1).mean()
    print(f"有財報資料的列：{fin_have:.1%}")
    print(f"財報科目數：{len(fcols)}")
    print("\n抽查（2330 中間一段，確認財報是發布後才出現）：")
    s = base[base["stock_id"] == "2330"]
    cols = ["date", "stock_id", "close", "month_revenue",
            "fin_period_end", "fin_avail", "f_EPS"]
    cols = [c for c in cols if c in base.columns]
    print(s[cols].iloc[600:606].to_string(index=False))


if __name__ == "__main__":
    main()
