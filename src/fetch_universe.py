"""
步驟 1：建立股票池。
從 FinMind TaiwanStockInfo 撈全市場，篩出半導體/電子/生技/金融四大產業的
上市櫃普通股，存成 data/raw/universe.parquet。

執行：
    python src/fetch_universe.py
"""
import sys
import pandas as pd

from config import DATA_RAW, INDUSTRY_GROUPS, ALLOWED_MARKET_TYPES
from finmind_client import get_stock_info, FinMindError


def classify(industry_category: str) -> str | None:
    """把 FinMind 的 industry_category 對映到我們的四大群組，找不到回 None。"""
    if not isinstance(industry_category, str):
        return None
    for group, keywords in INDUSTRY_GROUPS.items():
        if any(kw in industry_category for kw in keywords):
            return group
    return None


def main():
    print("抓取 TaiwanStockInfo …")
    try:
        info = get_stock_info()
    except FinMindError as e:
        sys.exit(f"抓取失敗：{e}")

    # 排除非普通股：4 碼數字代碼、市場別在白名單內
    info = info[info["type"].isin(ALLOWED_MARKET_TYPES)].copy()
    info = info[info["stock_id"].str.fullmatch(r"\d{4}")]

    # 同一 stock_id 可能有多列（更新日不同），取最後一筆
    info = info.sort_values("date").drop_duplicates("stock_id", keep="last")

    print("\n=== 實際出現的 industry_category（供你核對 config 的關鍵字）===")
    for cat, n in info["industry_category"].value_counts().items():
        print(f"  {cat}: {n}")

    info["group"] = info["industry_category"].apply(classify)
    universe = info[info["group"].notna()].copy()
    universe = universe[["stock_id", "stock_name", "industry_category",
                         "group", "type"]].reset_index(drop=True)

    print("\n=== 篩選後股票池（依群組）===")
    print(universe["group"].value_counts())
    print(f"\n股票池共 {len(universe)} 檔")

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / "universe.parquet"
    universe.to_parquet(out, index=False)
    print(f"已存檔：{out}")


if __name__ == "__main__":
    main()
