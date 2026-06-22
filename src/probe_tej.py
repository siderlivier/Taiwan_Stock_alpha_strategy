"""
TEJ 試用財報表探測。
目的：搞清楚 TRAIL 財報表裡『實際有哪些公司、有哪些欄位、發布日欄位叫什麼』，
以及試用帳號是否只開放少數示範公司。

執行：
    python src/probe_tej.py
"""
import tejapi
from config import TEJ_API_KEY

tejapi.ApiConfig.api_key = TEJ_API_KEY
tejapi.ApiConfig.api_base = "https://api.tej.com.tw"

CODES = ["TRAIL/TAIM1A", "TRAIL/TAIM1AQ", "TRAIL/TASALE"]
TEST_IDS = ["2330", "2317", "1101", "2412"]  # 都是大型權值股，一定有資料


def main():
    for code in CODES:
        print("\n" + "=" * 64)
        print(code)
        print("=" * 64)

        # (A) 不加任何過濾，看表裡前幾列 → 得知欄位與實際存在的 coid
        try:
            df = tejapi.get(code, paginate=False)
            print(f"[不過濾] 取回 {len(df)} 列")
            print(f"[不過濾] 欄位：{list(df.columns)}")
            if "coid" in df.columns:
                print(f"[不過濾] 出現的 coid（前 30 個）：")
                print(sorted(df['coid'].astype(str).unique())[:30])
            print("[不過濾] 前 3 列：")
            print(df.head(3).to_string())
        except Exception as e:
            print(f"[不過濾] 失敗：{e}")

        # (B) 逐一測大型股
        for sid in TEST_IDS:
            try:
                df = tejapi.get(code, coid=sid, paginate=True)
                cols = list(df.columns) if len(df) else "-"
                print(f"  {sid}: {len(df)} 列  欄位={cols}")
            except Exception as e:
                print(f"  {sid}: 失敗 {e}")


if __name__ == "__main__":
    main()
