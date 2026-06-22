"""
步驟 2：抓還原日價。
讀 universe.parquet，逐檔向 FinMind 抓還原日價（免費會員會自動回推還原），
邊抓邊存，可中斷續抓，最後合併成 data/raw/prices.parquet。

執行：
    python src/fetch_prices.py
"""
import sys
import time
import argparse
import pandas as pd

from config import DATA_RAW, START_DATE, END_DATE
from finmind_client import get_price_adjusted, FinMindError

SLEEP_SEC = 0.6          # 每檔之間稍微停一下，避免觸發限速
PER_STOCK_DIR = DATA_RAW / "prices_by_stock"   # 逐檔暫存，支援續抓


def main(limit: int | None = None):
    uni_path = DATA_RAW / "universe.parquet"
    if not uni_path.exists():
        sys.exit("找不到 universe.parquet，請先執行 fetch_universe.py")
    universe = pd.read_parquet(uni_path)
    if limit:
        universe = universe.head(limit)
        print(f"※ 取樣模式：只處理前 {limit} 檔")
    PER_STOCK_DIR.mkdir(parents=True, exist_ok=True)

    todo = [sid for sid in universe["stock_id"]
            if not (PER_STOCK_DIR / f"{sid}.parquet").exists()]
    print(f"股票池 {len(universe)} 檔，待抓 {len(todo)} 檔（已抓的會跳過）")

    for i, sid in enumerate(todo, 1):
        try:
            df = get_price_adjusted(sid, START_DATE, END_DATE)
        except FinMindError as e:
            print(f"  [{i}/{len(todo)}] {sid} 失敗：{e}")
            time.sleep(SLEEP_SEC)
            continue
        if df is None or df.empty:
            print(f"  [{i}/{len(todo)}] {sid} 無資料")
        else:
            df.to_parquet(PER_STOCK_DIR / f"{sid}.parquet", index=False)
            src = df["adj_source"].iloc[0] if "adj_source" in df else "?"
            print(f"  [{i}/{len(todo)}] {sid} 已存 {len(df)} 筆 ({src})")
        time.sleep(SLEEP_SEC)

    # 合併
    parts = []
    for sid in universe["stock_id"]:
        p = PER_STOCK_DIR / f"{sid}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d["stock_id"] = sid
            parts.append(d)
    if not parts:
        sys.exit("沒有任何價格資料可合併")
    prices = pd.concat(parts, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    out = DATA_RAW / "prices.parquet"
    prices.to_parquet(out, index=False)
    print(f"\n合併完成：{out}（{prices['stock_id'].nunique()} 檔、{len(prices)} 筆）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只抓前 N 檔（測試用）")
    args = ap.parse_args()
    main(limit=args.limit)
