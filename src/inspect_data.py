"""
資料品質檢查（QC）+ 人工抽查片段。
下載完（或先用 --limit 抓樣本後）執行，會：
  1. 對每個 parquet 印出 概況（筆數/欄位/型別/日期範圍/檔數）
  2. 缺值、重複主鍵、日期連續性檢查
  3. 價格：還原來源分佈、極端單日報酬、非正價格 等異常
  4. 印出幾檔的 head/tail 供眼睛掃
  5. 把樣本另存成 CSV（data/processed/samples/）方便用 Excel 開

執行：
    python src/inspect_data.py
"""
import sys
import pandas as pd

from config import DATA_RAW, DATA_PROCESSED, END_DATE

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

SAMPLE_DIR = DATA_PROCESSED / "samples"


def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def overview(df: pd.DataFrame, name: str, date_col="date", id_col="stock_id"):
    print(f"\n--- {name} ---")
    print(f"筆數：{len(df):,}   欄位數：{df.shape[1]}")
    print(f"欄位：{list(df.columns)}")
    if id_col in df.columns:
        print(f"股票檔數：{df[id_col].nunique()}")
    if date_col in df.columns:
        d = pd.to_datetime(df[date_col])
        print(f"日期範圍：{d.min()} ~ {d.max()}")
    # 缺值
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print("缺值欄位：")
        print(nulls.to_string())
    else:
        print("缺值：無")
    # 重複主鍵
    if date_col in df.columns and id_col in df.columns:
        dup = df.duplicated(subset=[date_col, id_col]).sum()
        print(f"重複 (date, stock_id)：{dup}")


def coverage_report(df: pd.DataFrame, name_map: dict):
    """每檔的資料起訖日與筆數，並標記提前結束（多為下市）的個股。
    以『全市場最後交易日』為基準，比固定 END_DATE 更準。"""
    line("各檔資料涵蓋範圍（含結束日檢查）")
    g = df.groupby("stock_id")["date"]
    cov = pd.DataFrame({
        "first": g.min(),
        "last": g.max(),
        "days": g.count(),
    }).reset_index()
    cov["name"] = cov["stock_id"].map(name_map).fillna("?")

    market_last = df["date"].max()
    cutoff = market_last - pd.Timedelta(days=30)
    cov["ends_early"] = cov["last"] < cutoff

    n_active = int((~cov["ends_early"]).sum())
    n_early = int(cov["ends_early"].sum())
    print(f"全市場最後交易日：{market_last.date()}")
    print(f"抓到接近最新（在市）：{n_active} 檔")
    print(f"提前結束（多為已下市，正常）：{n_early} 檔")
    print(f"在市比例：{n_active / len(cov):.1%}")

    if n_early:
        early = cov[cov["ends_early"]].sort_values("last")
        print(f"\n提前結束的個股（依結束日排序，最多列 30 檔供抽查）：")
        print(early[["stock_id", "name", "first", "last", "days"]]
              .head(30).to_string(index=False))
        print("\n→ 這些多半是真的下市股（保留它們可避免倖存者偏差）。"
              "若發現某檔你確定仍在交易卻提前結束，才需懷疑截斷。")


def check_prices(df: pd.DataFrame, name_map: dict):
    line("價格 QC")
    overview(df, "prices.parquet")
    coverage_report(df, name_map)

    # 還原來源
    if "adj_source" in df.columns:
        print("\n還原來源分佈：")
        print(df["adj_source"].value_counts().to_string())

    # 選還原收盤欄位（自行回推為 adj_close；付費表可能直接是 close）
    close_col = "adj_close" if "adj_close" in df.columns else "close"
    print(f"\n用於檢查的收盤欄位：{close_col}")

    # 非正價格
    bad = df[df[close_col] <= 0]
    print(f"非正收盤價筆數：{len(bad)}")

    # 極端單日報酬（可能是未還原的除權息跳空或資料錯誤）
    df = df.sort_values(["stock_id", "date"]).copy()
    df["ret"] = df.groupby("stock_id")[close_col].pct_change()
    extreme = df[df["ret"].abs() > 0.4]
    print(f"|單日報酬| > 40% 筆數：{len(extreme)}（若很多，要懷疑還原沒做對）")
    if len(extreme):
        print(extreme[["date", "stock_id", close_col, "ret"]].head(10).to_string(index=False))

    # 日期連續性：每檔交易日數 vs 整體交易日數
    all_days = df["date"].nunique()
    per_stock = df.groupby("stock_id")["date"].nunique()
    print(f"\n整體交易日數：{all_days}")
    print("各檔交易日數（最少 5 檔，過少代表新上市或資料缺漏）：")
    print(per_stock.sort_values().head(5).to_string())


def check_fundamentals(df: pd.DataFrame, name: str):
    line(f"基本面 QC：{name}")
    overview(df, name, date_col="date", id_col="stock_id")
    if "revenue_year" in df.columns and "revenue_month" in df.columns:
        print("（月營收）date=揭露日；所屬期間由 revenue_year/revenue_month 決定")
    if "type" in df.columns:
        print(f"（財報）type 種類數：{df['type'].nunique()}；範例："
              f"{sorted(df['type'].astype(str).unique())[:10]}")


def show_samples(prices, funds, name_map):
    line("人工抽查片段")
    sample_ids = list(prices["stock_id"].unique()[:3]) if prices is not None else []
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    for sid in sample_ids:
        nm = name_map.get(sid, "?")
        sub = prices[prices["stock_id"] == sid].sort_values("date")
        print(f"\n[{sid} {nm}] 價格 head：")
        print(sub.head(3).to_string(index=False))
        print(f"[{sid} {nm}] 價格 tail：")
        print(sub.tail(3).to_string(index=False))
        sub.to_csv(SAMPLE_DIR / f"price_sample_{sid}.csv", index=False)

    for name, df in funds.items():
        if df is None or df.empty:
            continue
        print(f"\n[{name}] 前 3 筆：")
        print(df.head(3).to_string(index=False))
        df.head(50).to_csv(SAMPLE_DIR / f"{name}_sample.csv", index=False)

    print(f"\nCSV 樣本已輸出到：{SAMPLE_DIR}")


def load(path):
    return pd.read_parquet(path) if path.exists() else None


def main():
    prices = load(DATA_RAW / "prices.parquet")
    funds = {
        "fund_revenue": load(DATA_RAW / "fund_revenue.parquet"),
        "fund_financials": load(DATA_RAW / "fund_financials.parquet"),
    }
    universe = load(DATA_RAW / "universe.parquet")
    name_map = {}
    if universe is not None and "stock_name" in universe.columns:
        name_map = dict(zip(universe["stock_id"], universe["stock_name"]))

    if universe is not None:
        line("股票池")
        overview(universe, "universe.parquet", date_col="(none)", id_col="stock_id")
        print(universe["group"].value_counts().to_string())

    if prices is None:
        print("\n找不到 prices.parquet，請先執行 fetch_prices.py")
    else:
        check_prices(prices, name_map)

    for name, df in funds.items():
        if df is not None and not df.empty:
            check_fundamentals(df, name)

    if prices is not None:
        show_samples(prices, funds, name_map)

    print("\nQC 完成。請特別確認：還原來源、極端報酬筆數、財報發布日欄位三項。")


if __name__ == "__main__":
    main()
