"""
步驟 3：抓基本面（FinMind，免費、全覆蓋、歷史完整）。
  - 月營收 TaiwanStockMonthRevenue：date=揭露日(約次月1日)、revenue_year/month=所屬期間
  - 綜合損益表 TaiwanStockFinancialStatements：date=財報期末、type/value (long 格式)

逐檔暫存可中斷續抓，最後合併成 fund_revenue.parquet / fund_financials.parquet。
發布日對齊留到下一步：月營收加保守 lag、財報用法定申報期限近似。

執行：
    python src/fetch_fundamentals.py            # 全部
    python src/fetch_fundamentals.py --limit 5  # 先抓 5 檔測試
"""
import sys
import time
import argparse
import pandas as pd

from config import DATA_RAW, START_DATE, END_DATE
from finmind_client import get_month_revenue, get_financial_statement, FinMindError

SLEEP_SEC = 0.6


def fetch(kind, fn, stock_ids):
    out_dir = DATA_RAW / f"fund_{kind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stock_ids if not (out_dir / f"{s}.parquet").exists()]
    print(f"\n[{kind}] 待抓 {len(todo)} 檔")
    printed = False
    for i, sid in enumerate(todo, 1):
        try:
            df = fn(sid, START_DATE, END_DATE)
        except FinMindError as e:
            print(f"  [{i}/{len(todo)}] {sid} 失敗：{e}")
            time.sleep(SLEEP_SEC)
            continue
        if df is None or df.empty:
            pd.DataFrame().to_parquet(out_dir / f"{sid}.parquet")
            print(f"  [{i}/{len(todo)}] {sid} 無資料")
        else:
            if not printed:
                print(f"  欄位：{list(df.columns)}")
                printed = True
            df.to_parquet(out_dir / f"{sid}.parquet", index=False)
            print(f"  [{i}/{len(todo)}] {sid} {len(df)} 筆")
        time.sleep(SLEEP_SEC)


def merge(kind):
    out_dir = DATA_RAW / f"fund_{kind}"
    parts = [pd.read_parquet(p) for p in out_dir.glob("*.parquet")]
    parts = [d for d in parts if len(d)]
    if parts:
        m = pd.concat(parts, ignore_index=True)
        out = DATA_RAW / f"fund_{kind}.parquet"
        m.to_parquet(out, index=False)
        print(f"[{kind}] 合併 {len(m)} 筆 → {out}")


def main(limit=None):
    uni = DATA_RAW / "universe.parquet"
    if not uni.exists():
        sys.exit("找不到 universe.parquet，請先執行 fetch_universe.py")
    ids = pd.read_parquet(uni)["stock_id"].tolist()
    if limit:
        ids = ids[:limit]
        print(f"sample mode: first {limit} stocks only")

    fetch("revenue", get_month_revenue, ids)
    fetch("financials", get_financial_statement, ids)
    merge("revenue")
    merge("financials")
    print("\n完成。下一步：發布日對齊（月營收加 lag、財報用法定期限近似）。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只抓前 N 檔（測試用）")
    args = ap.parse_args()
    main(limit=args.limit)
