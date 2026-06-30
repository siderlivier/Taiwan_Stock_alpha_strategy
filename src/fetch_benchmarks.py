"""
抓取市場基準指數的月報酬，供回測比較超額報酬與回撤。
  - 0050（元大台灣50）：用還原價 → 含息總報酬，最貼近「買大盤」的可投資替代。
  - 大盤報酬指數 TAIEX、櫃買報酬指數 TPEx（TaiwanStockTotalReturnIndex，已含息）。

輸出：data/raw/benchmarks.parquet（欄位 ym + 各基準月報酬 BM_*）

執行：python src/fetch_benchmarks.py
"""
import sys
import numpy as np
import pandas as pd

from config import DATA_RAW, START_DATE, END_DATE
from finmind_client import _get, get_price_adjusted, FinMindError


def monthly_ret(df, price_col):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["ym"] = df["date"].dt.to_period("M")
    last = df.groupby("ym")[price_col].last()
    return last.pct_change(fill_method=None)


def main():
    series = {}

    # 0050 總報酬 — 用 yfinance 的 0050.TW（auto_adjust 已正確處理「分割」與配息），
    # 避免自製還原不處理 2025/6 1拆4 造成的失真。需先 pip install yfinance。
    print("抓 0050（yfinance 0050.TW，含分割/配息調整）…")
    try:
        import yfinance as yf
        h = yf.Ticker("0050.TW").history(start=START_DATE, end=END_DATE, auto_adjust=True)
        if h is not None and len(h):
            h = h.reset_index()[["Date", "Close"]].rename(
                columns={"Date": "date", "Close": "close"})
            h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None)
            r = monthly_ret(h, "close")
            series["BM_0050"] = r
            big = int((r.abs() > 0.25).sum())
            print(f"  0050 OK（{len(h)} 日）；|月報酬|>25% 有 {big} 個月（應為 0）")
        else:
            print("  0050 無資料")
    except ImportError:
        print("  未安裝 yfinance，跳過 0050。請執行： pip install yfinance")
    except Exception as e:
        print(f"  0050（yfinance）失敗：{e}")

    # 大盤(TAIEX) / 櫃買(TPEx) 報酬指數（已含息，無分割問題，最乾淨）
    print("抓 報酬指數 TaiwanStockTotalReturnIndex（需指定 data_id）…")
    for sid, label in [("TAIEX", "BM_TAIEX_TR"), ("TPEx", "BM_TPEx_TR")]:
        try:
            idx = _get("TaiwanStockTotalReturnIndex",
                       {"data_id": sid, "start_date": START_DATE, "end_date": END_DATE})
            if idx is None or len(idx) == 0:
                print(f"  {sid}: 無資料")
                continue
            numcol = next((c for c in ["price", "close", "value", "TotalReturnIndex"]
                           if c in idx.columns), None)
            if numcol is None:
                numcol = idx.select_dtypes("number").columns[-1]
            series[label] = monthly_ret(idx, numcol)
            print(f"  {label} OK（欄位 {numcol}，{len(idx)} 日）")
        except FinMindError as e:
            print(f"  {sid} 失敗：{e}")

    if not series:
        sys.exit("沒有抓到任何基準，請檢查 token 與資料表權限")

    out = pd.DataFrame(series)
    # 清除明顯的來源壞點：ETF/指數單月 |報酬|>40% 不可能（漲跌停限制），多為資料錯誤
    # （例：yfinance 0050.TW 2014-01 出現 -75% 假跌）。設為缺值，後續自動排除。
    for c in out.columns:
        bad = out[c].abs() > 0.40
        if int(bad.sum()):
            print(f"  清除 {c} 異常月：{list(out.index[bad].astype(str))}"
                  f"（|月報酬|>40%，視為資料錯誤）")
            out.loc[bad, c] = np.nan
    out.index = out.index.astype(str)          # Period → str 以便存 parquet
    out = out.reset_index().rename(columns={"index": "ym"})
    path = DATA_RAW / "benchmarks.parquet"
    out.to_parquet(path, index=False)
    print(f"\n已存 {path}：{list(series.keys())}")
    print(out.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
