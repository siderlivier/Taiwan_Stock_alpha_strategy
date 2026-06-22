"""
FinMind 截斷診斷。
用台積電(2330，必定一直在交易)測試免費版會不會把歷史資料截斷，
並回報 token 狀態與用量上限。

執行：
    python src/diagnose_finmind.py
"""
from datetime import date

from config import FINMIND_TOKEN, START_DATE, END_DATE
from finmind_client import get_price_raw, FinMindError


def main():
    print("=== Token 狀態 ===")
    if FINMIND_TOKEN:
        print(f"已設定 token（結尾 ...{FINMIND_TOKEN[-4:]}）")
    else:
        print("未設定 token！匿名呼叫額度極低且可能截斷資料 → 請到 "
              "https://finmindtrade.com 註冊取得 token 填入 .env 的 FINMIND_TOKEN")

    print(f"\n=== 測試 2330 原始日價 {START_DATE} ~ {END_DATE} ===")
    try:
        df = get_price_raw("2330", START_DATE, END_DATE)
    except FinMindError as e:
        print(f"抓取失敗：{e}")
        return

    if df.empty:
        print("沒有資料")
        return

    df = df.sort_values("date")
    first, last = df["date"].iloc[0], df["date"].iloc[-1]
    print(f"筆數：{len(df)}")
    print(f"最早：{first}")
    print(f"最晚：{last}")

    today = date.today().isoformat()
    if str(last) < "2025-01-01":
        print("\n⚠️ 2330 資料竟然停在 2025 之前 → 確認是『免費版截斷歷史資料』。")
        print("   解法：(1) 升級 FinMind 贊助會員(可拿完整歷史+還原表) ；")
        print("        (2) 改用 yfinance 抓價(免費無限，但缺已下市股) ；")
        print("        (3) 退回 TEJ 未還原價。")
    else:
        print(f"\n✅ 2330 資料到 {last}，免費版沒有截斷。"
              f"那麼之前停在 2020 的個股就是真的下市股，資料正確。")


if __name__ == "__main__":
    main()
