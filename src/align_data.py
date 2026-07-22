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


def load_optional(name):
    p = DATA_RAW / name
    return pd.read_parquet(p) if p.exists() else None


def build_price_base(prices, universe):
    """每日還原價基底 + 日報酬，掛上產業別。"""
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    # 用還原價為標準價格欄
    out = df[["date", "stock_id", "Trading_Volume", "Trading_money",
              "adj_open", "adj_max", "adj_min", "adj_close", "close"]].rename(columns={
        "adj_open": "open", "adj_max": "high", "adj_min": "low",
        "adj_close": "close", "close": "close_raw",
        "Trading_Volume": "volume", "Trading_money": "amount"})
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


def prep_statement(stmt, prefix, drop_per=False):
    """通用：long 財務報表 → wide，加可得日(法定期限)。供資產負債表/現金流量表共用。
    drop_per：資產負債表有 *_per(占比)欄位，預設可丟掉只留原值，控制面板寬度。"""
    df = stmt.copy()
    df = df[df["type"].astype(str) != "-"]
    if drop_per:
        df = df[~df["type"].astype(str).str.endswith("_per")]
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index=["stock_id", "date"], columns="type",
                          values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    acc = [c for c in wide.columns if c not in ("stock_id", "date")]
    wide = wide.rename(columns={c: f"{prefix}{c}" for c in acc})
    wide[f"{prefix}avail"] = wide["date"].apply(_stmt_deadline)
    wide = wide.rename(columns={"date": f"{prefix}period_end"})
    return wide.sort_values(f"{prefix}avail"), [f"{prefix}{c}" for c in acc]


def asof_merge(base, right, right_time_col):
    """以 merge_asof(backward) 把 right 對齊到 base 的每個交易日（同 stock_id）。"""
    left = base.sort_values("date")
    right = right.sort_values(right_time_col)
    merged = pd.merge_asof(
        left, right,
        left_on="date", right_on=right_time_col,
        by="stock_id", direction="backward")
    return merged


# ---------- 擴充：日頻籌碼 / 估值（皆為收盤後公布，故一律 lag 1 日）----------
# 可得日 = 資料日 + 1 天；asof(backward) 於交易日 T 只會取到「資料日 <= T-1」的最新值，
# 保證不會用到當日尚未公布的籌碼（no look-ahead）。

def _avail_next_day(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["avail"] = df["date"] + pd.Timedelta(days=1)
    return df


def prep_inst(inst):
    """個股三大法人(long) → 每日 外資/投信/自營 淨買超(股數)。"""
    df = inst.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ("buy", "sell"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["net"] = df["buy"] - df["sell"]
    nm = df["name"].astype(str)
    df["cat"] = np.select(
        [nm.str.contains("Foreign"), nm.str.contains("Trust"), nm.str.contains("Dealer")],
        ["foreign", "trust", "dealer"], default="other")
    w = (df[df["cat"] != "other"]
         .groupby(["stock_id", "date", "cat"])["net"].sum()
         .unstack("cat").reset_index())
    w = w.rename(columns={"foreign": "inst_foreign_net",
                          "trust": "inst_trust_net", "dealer": "inst_dealer_net"})
    cols = [c for c in ["inst_foreign_net", "inst_trust_net", "inst_dealer_net"]
            if c in w.columns]
    w[cols] = w[cols].fillna(0.0)
    w = _avail_next_day(w)
    return w[["stock_id", "avail"] + cols].sort_values("avail"), cols


def prep_margin(margin):
    """融資融券 → 每日 融資餘額 / 融券餘額（張）。"""
    df = margin.copy()
    df["date"] = pd.to_datetime(df["date"])
    keep = {"MarginPurchaseTodayBalance": "margin_balance",
            "ShortSaleTodayBalance": "short_balance"}
    for c in keep:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    w = df[["stock_id", "date"] + list(keep)].rename(columns=keep)
    cols = list(keep.values())
    w = _avail_next_day(w)
    return w[["stock_id", "avail"] + cols].sort_values("avail"), cols


def prep_sharehold(sh):
    """外資持股 → 每日 外資持股比率(%)。"""
    df = sh.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["foreign_ratio"] = pd.to_numeric(
        df.get("ForeignInvestmentSharesRatio"), errors="coerce")
    w = _avail_next_day(df[["stock_id", "date", "foreign_ratio"]])
    return w[["stock_id", "avail", "foreign_ratio"]].sort_values("avail"), ["foreign_ratio"]


def prep_lending(lend):
    """借券成交明細 → 每日 借券成交量(加總)。注意此資料僅 2020-09 起。"""
    df = lend.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
    w = (df.groupby(["stock_id", "date"])["volume"].sum().reset_index()
           .rename(columns={"volume": "lending_vol"}))
    w = _avail_next_day(w)
    return w[["stock_id", "avail", "lending_vol"]].sort_values("avail"), ["lending_vol"]


def prep_per(per):
    """PER / PBR / 殖利率 → 每日。"""
    df = per.copy()
    df["date"] = pd.to_datetime(df["date"])
    ren = {"PER": "per", "PBR": "pbr", "dividend_yield": "div_yield"}
    for c in ren:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    w = _avail_next_day(df[["stock_id", "date"] + list(ren)].rename(columns=ren))
    cols = list(ren.values())
    return w[["stock_id", "avail"] + cols].sort_values("avail"), cols


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

    # 資產負債表(b_)、現金流量表(c_)：同樣用法定期限近似公告日對齊
    bcols, ccols = [], []
    bal = load_optional("fund_balance.parquet")
    if bal is not None:
        print("對齊資產負債表…")
        bal_w, bcols = prep_statement(bal, "b_", drop_per=True)
        base = asof_merge(base, bal_w, "b_avail")
    cf = load_optional("fund_cashflow.parquet")
    if cf is not None:
        print("對齊現金流量表…")
        cf_w, ccols = prep_statement(cf, "c_", drop_per=False)
        base = asof_merge(base, cf_w, "c_avail")

    # ---- 擴充：日頻籌碼 / 估值（皆 lag 1 日，asof backward）----
    chip_specs = [("chip_inst.parquet", prep_inst),
                  ("chip_margin.parquet", prep_margin),
                  ("chip_sharehold.parquet", prep_sharehold),
                  ("chip_lending.parquet", prep_lending),
                  ("chip_per.parquet", prep_per)]
    chip_cols = []
    for i, (fname, fn) in enumerate(chip_specs):
        raw = load_optional(fname)
        if raw is None or raw.empty:
            print(f"（略過 {fname}：未下載）")
            continue
        print(f"對齊 {fname}（lag 1 日）…")
        right, cols = fn(raw)
        tcol = f"_chip_avail_{i}"
        right = right.rename(columns={"avail": tcol})
        base = asof_merge(base, right, tcol)
        base = base.drop(columns=[tcol])
        chip_cols += cols

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
    print(f"有損益表的列：{base[fcols].notna().any(axis=1).mean():.1%}　科目數：{len(fcols)}")
    if bcols:
        print(f"有資產負債表的列：{base[bcols].notna().any(axis=1).mean():.1%}　科目數：{len(bcols)}")
    if ccols:
        print(f"有現金流量表的列：{base[ccols].notna().any(axis=1).mean():.1%}　科目數：{len(ccols)}")
    for c in chip_cols:
        if c in base.columns:
            print(f"有 {c} 的列：{base[c].notna().mean():.1%}")
    print("\n抽查（2330 中間一段，確認財報是發布後才出現）：")
    s = base[base["stock_id"] == "2330"]
    cols = ["date", "stock_id", "close", "month_revenue",
            "fin_period_end", "fin_avail", "f_EPS"]
    cols = [c for c in cols if c in base.columns]
    print(s[cols].iloc[600:606].to_string(index=False))


if __name__ == "__main__":
    main()
