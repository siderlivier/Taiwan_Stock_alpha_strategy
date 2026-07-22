"""
步驟 3b：抓「籌碼面 + 資料品質 + 估值」擴充資料（FinMind，皆免費資料集）。

Tier1 籌碼（逐日、逐檔續抓，資料量大）：
  - TaiwanStockInstitutionalInvestorsBuySell  個股三大法人買賣（long：每日每法人一列）
  - TaiwanStockMarginPurchaseShortSale        個股融資融券
  - TaiwanStockShareholding                   外資持股
  - TaiwanStockSecuritiesLending              借券成交明細
  - TaiwanStockPER                            個股 PER / PBR / 殖利率
Tier2 資料品質（事件表，量小，全市場單次抓）：
  - TaiwanStockCapitalReductionReferencePrice 減資恢復買賣參考價（用來根治減資造成的極端日報酬）
  - TaiwanStockSplitPrice                     分割後參考價
  - TaiwanStockParValueChange                 變更面額恢復買賣參考價
  - TaiwanStockDelisting                      下市櫃表（確認下市日、佐證倖存者偏差處理）
Tier3 股利（事件表，全市場單次抓）：
  - TaiwanStockDividend                       股利政策表（宣告股利，餵殖利率 / 除息事件）

逐檔資料可中斷續抓：已存在的 chip_<name>/<stock_id>.parquet 會跳過，最後合併成 chip_<name>.parquet。
事件表資料量小，直接全市場單次抓成 event_<name>.parquet。

注意：FinMind 免費版有每日請求次數上限；Tier1 為逐日×近千檔，量大，可能需分幾次跑
（斷點續傳會自動接續）。_get() 已內建 402/429 限速重試。

執行：
    python src/fetch_extra.py                 # 全部
    python src/fetch_extra.py --limit 5       # Tier1 只抓前 5 檔測試
    python src/fetch_extra.py --only per      # 只抓逐檔（籌碼/PER）
    python src/fetch_extra.py --only market   # 只抓事件表（減資/分割/面額/下市/股利）
"""
import sys
import time
import argparse
import pandas as pd

from config import DATA_RAW, START_DATE, END_DATE
from finmind_client import _get, FinMindError

SLEEP_SEC = 0.6

# 逐檔（逐日資料）：name -> FinMind dataset
PER_STOCK = {
    "inst":      "TaiwanStockInstitutionalInvestorsBuySell",
    "margin":    "TaiwanStockMarginPurchaseShortSale",
    "sharehold": "TaiwanStockShareholding",
    "lending":   "TaiwanStockSecuritiesLending",
    "per":       "TaiwanStockPER",
}

# 全市場單次（事件表 / 小資料）：name -> FinMind dataset
MARKET_WIDE = {
    "capreduce": "TaiwanStockCapitalReductionReferencePrice",
    "split":     "TaiwanStockSplitPrice",
    "parvalue":  "TaiwanStockParValueChange",
    "delisting": "TaiwanStockDelisting",
    "dividend":  "TaiwanStockDividend",
}


def fetch_per_stock(name, dataset, stock_ids):
    out_dir = DATA_RAW / f"chip_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stock_ids if not (out_dir / f"{s}.parquet").exists()]
    print(f"\n[{name}] {dataset} 待抓 {len(todo)} 檔")
    printed = False
    for i, sid in enumerate(todo, 1):
        try:
            df = _get(dataset, {"data_id": sid,
                                "start_date": START_DATE, "end_date": END_DATE})
        except FinMindError as e:
            print(f"  [{i}/{len(todo)}] {sid} 失敗：{e}")
            time.sleep(SLEEP_SEC)
            continue
        if df is None or df.empty:
            pd.DataFrame().to_parquet(out_dir / f"{sid}.parquet")   # 佔位，續抓時跳過
        else:
            if not printed:
                print(f"  欄位：{list(df.columns)}")
                printed = True
            df.to_parquet(out_dir / f"{sid}.parquet", index=False)
        if i % 50 == 0 or i == len(todo):
            print(f"  進度 [{i}/{len(todo)}]")
        time.sleep(SLEEP_SEC)


def merge_per_stock(name):
    out_dir = DATA_RAW / f"chip_{name}"
    if not out_dir.exists():
        return
    parts = [pd.read_parquet(p) for p in out_dir.glob("*.parquet")]
    parts = [d for d in parts if len(d)]
    if parts:
        m = pd.concat(parts, ignore_index=True)
        out = DATA_RAW / f"chip_{name}.parquet"
        m.to_parquet(out, index=False)
        print(f"[{name}] 合併 {len(m):,} 筆 → {out.name}")


def fetch_market_wide(name, dataset, stock_ids):
    """先試全市場單次抓；若不支援（空/報錯）則自動退回逐檔抓後合併。"""
    out = DATA_RAW / f"event_{name}.parquet"
    if out.exists():
        print(f"[{name}] {out.name} 已存在，跳過（要重抓請先刪該檔）")
        return
    # (1) 全市場單次
    df = None
    try:
        df = _get(dataset, {"start_date": START_DATE, "end_date": END_DATE})
    except FinMindError as e:
        print(f"[{name}] 全市場單抓失敗，改逐檔：{e}")
    if df is not None and not df.empty:
        df.to_parquet(out, index=False)
        print(f"[{name}] {dataset} {len(df):,} 筆 → {out.name}　欄位：{list(df.columns)}")
        return
    # (2) 退回逐檔（適用需 data_id 的資料集，如減資/股利政策）
    print(f"[{name}] 改用逐檔抓 {dataset}（近千檔，資料量小）…")
    parts = []
    for i, sid in enumerate(stock_ids, 1):
        try:
            d = _get(dataset, {"data_id": sid,
                               "start_date": START_DATE, "end_date": END_DATE})
        except FinMindError:
            time.sleep(SLEEP_SEC)
            continue
        if d is not None and not d.empty:
            parts.append(d)
        if i % 100 == 0 or i == len(stock_ids):
            print(f"  進度 [{i}/{len(stock_ids)}]　已收集 {len(parts)} 檔")
        time.sleep(SLEEP_SEC)
    if parts:
        m = pd.concat(parts, ignore_index=True)
        m.to_parquet(out, index=False)
        print(f"[{name}] 逐檔合併 {len(m):,} 筆 → {out.name}　欄位：{list(m.columns)}")
    else:
        print(f"[{name}] 逐檔仍無資料")


def main(limit=None, only=None):
    uni = DATA_RAW / "universe.parquet"
    if not uni.exists():
        sys.exit("找不到 universe.parquet，請先執行 fetch_universe.py")
    ids = pd.read_parquet(uni)["stock_id"].tolist()
    if limit:
        ids = ids[:limit]
        print(f"sample mode: first {limit} stocks only")

    if only in (None, "per"):
        for name, ds in PER_STOCK.items():
            fetch_per_stock(name, ds, ids)
            merge_per_stock(name)

    if only in (None, "market"):
        for name, ds in MARKET_WIDE.items():
            fetch_market_wide(name, ds, ids)

    print("\n完成。下一步：")
    print("  1) 籌碼(三大法人/融資融券/外資持股/借券) 與 PER → 加 point-in-time lag(逐日資料"
          "至少 lag 1 交易日) 後併入 panel，做成新因子候選。")
    print("  2) 減資/分割/面額 參考價 → 修正還原價，根治殘留的極端日報酬（README 已知侷限）。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Tier1 只抓前 N 檔（測試用）")
    ap.add_argument("--only", choices=["per", "market"], default=None,
                    help="per=只抓逐檔籌碼/PER；market=只抓事件表")
    args = ap.parse_args()
    main(limit=args.limit, only=args.only)
