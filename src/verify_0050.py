"""
0050 資料交叉核對：yfinance(調整後) vs FinMind(原始未還原)。
兩個獨立來源比對月報酬，用來確認資料可信、並看清楚壞點/分割/除息的影響。

說明：
  - FinMind 原始價：未還原 → 2025/6 分割當月會出現約 -75% 假跌（正常，因未處理分割）。
  - yfinance 調整後：已處理分割與配息 → 分割月正常；但 2014-01 有來源壞點(-75%)。
  - 除「分割月」與「壞點」外，兩者月報酬應大致一致（差異主要來自除息）。

執行：python src/verify_0050.py
"""
import sys
import pandas as pd

from config import START_DATE, END_DATE
from finmind_client import get_price_raw, FinMindError


def mret(df, pcol, dcol):
    df = df.copy()
    df[dcol] = pd.to_datetime(df[dcol]).dt.tz_localize(None)
    df["ym"] = df[dcol].dt.to_period("M")
    return df.groupby("ym")[pcol].last().pct_change(fill_method=None)


def main():
    # FinMind 原始（未還原）
    try:
        fm = get_price_raw("0050", START_DATE, END_DATE)
    except FinMindError as e:
        sys.exit(f"FinMind 抓取失敗：{e}")
    fm_px = fm.copy()
    fm_px["date"] = pd.to_datetime(fm_px["date"])
    fm_ret = mret(fm, "close", "date")

    # yfinance 調整後
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("未安裝 yfinance，請先 pip install yfinance")
    h = yf.Ticker("0050.TW").history(start=START_DATE, end=END_DATE,
                                     auto_adjust=True).reset_index()
    yf_ret = mret(h, "Close", "Date")

    cmp = pd.DataFrame({"FinMind原始": fm_ret, "yfinance調整": yf_ret})
    cmp["差異"] = cmp["yfinance調整"] - cmp["FinMind原始"]

    print("=== 價格水準對照（說明還原 vs 實際成交）===")
    h2 = h.copy()
    h2["Date"] = pd.to_datetime(h2["Date"]).dt.tz_localize(None)
    for d in ["2014-01-31", "2020-12-31", "2025-12-31"]:
        fmrow = fm_px[fm_px["date"] <= d].tail(1)
        yrow = h2[h2["Date"] <= d].tail(1)
        if len(fmrow) and len(yrow):
            print(f"  ~{d}: FinMind實際成交≈{fmrow['close'].iloc[0]:.1f}　"
                  f"yfinance還原後≈{yrow['Close'].iloc[0]:.1f}")

    print("\n=== 兩來源月報酬差異最大的 10 個月（通常=除息/分割/壞點）===")
    big = cmp.reindex(cmp["差異"].abs().sort_values(ascending=False).index).head(10)
    print(big.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== 幾個指標月份核對（除這些外兩者應接近）===")
    for ym in ["2014-01", "2020-03", "2022-10", "2024-06", "2025-06"]:
        p = pd.Period(ym, "M")
        if p in cmp.index:
            row = cmp.loc[p]
            print(f"  {ym}: FinMind={row['FinMind原始']:+.4f}  "
                  f"yfinance={row['yfinance調整']:+.4f}  差異={row['差異']:+.4f}")

    corr = cmp[["FinMind原始", "yfinance調整"]].dropna().corr().iloc[0, 1]
    print(f"\n兩來源月報酬相關係數：{corr:.4f}（接近 1 代表一致可信）")
    print("判讀：除『2025-06 分割月(FinMind假跌)』與『2014-01 壞點(yfinance假跌)』外，"
          "其餘月份兩者應高度一致——代表資料可信，我們清掉那兩個點即可。")


if __name__ == "__main__":
    main()
