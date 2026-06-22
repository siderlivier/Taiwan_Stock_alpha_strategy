"""
FinMind 基本面覆蓋率探測。
同時測『大型在市股』與『我們抽到的下市樣本』，看月營收/財報各自的覆蓋與歷史長度。

執行：
    python src/probe_finmind_fund.py
"""
from config import START_DATE, END_DATE
from finmind_client import get_month_revenue, get_financial_statement, FinMindError

# 前段為大型在市股（半導體/生技/金融/電子），後段為我們抽到的下市樣本
TESTS = ["2330", "2454", "3008", "6505", "2891", "1101",
         "6238", "5349", "6497", "5304", "4152"]


def show(sid, label, fn):
    try:
        df = fn(sid, START_DATE, END_DATE)
    except FinMindError as e:
        print(f"  {sid:6} {label:11} 失敗 {e}")
        return
    if df is None or len(df) == 0:
        print(f"  {sid:6} {label:11} 空")
    else:
        print(f"  {sid:6} {label:11} {len(df):5} 筆  {df['date'].min()} ~ {df['date'].max()}")


def main():
    print("=== 月營收 TaiwanStockMonthRevenue ===")
    for sid in TESTS:
        show(sid, "revenue", get_month_revenue)
    print("\n=== 綜合損益表 TaiwanStockFinancialStatements ===")
    for sid in TESTS:
        show(sid, "financials", get_financial_statement)


if __name__ == "__main__":
    main()
