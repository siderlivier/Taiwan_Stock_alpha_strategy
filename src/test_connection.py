"""
TEJ API 連線與權限測試。
目的：確認金鑰有效、確認你的方案能存取哪些資料表，並抓一小段資料驗證格式。

執行前先安裝套件：
    pip install tejapi pandas

執行：
    python src/test_connection.py
"""
import sys

try:
    import tejapi
except ImportError:
    sys.exit("尚未安裝 tejapi，請先執行： pip install tejapi pandas")

from config import TEJ_API_KEY

# 設定金鑰
tejapi.ApiConfig.api_key = TEJ_API_KEY
# 回傳 pandas DataFrame
tejapi.ApiConfig.api_base = "https://api.tej.com.tw"


def check_quota():
    """查詢目前 API 用量/額度。"""
    print("=== API 用量 ===")
    try:
        info = tejapi.ApiConfig.info()
        print(info)
    except Exception as e:
        print(f"查詢用量失敗：{e}")


def list_tables():
    """從 info() 列出本帳號實際可存取的資料表。"""
    print("\n=== 可用資料表（依本帳號權限）===")
    try:
        info = tejapi.ApiConfig.info()
        tables = info.get("user", {}).get("tables", {})
        for code in sorted(tables):
            yrs = tables[code]
            print(f"  {code}  (data {yrs.get('dataStartYear')}~{yrs.get('dataEndYear')})")
    except Exception as e:
        print(f"查詢資料表失敗：{e}")


def sample_financials():
    """抓台積電(2330)財報一小段，確認試用表格式與發布日欄位。"""
    print("\n=== 樣本：2330 財報 (TRAIL/TAIM1A) ===")
    try:
        df = tejapi.get(
            "TRAIL/TAIM1A",
            coid="2330",
            mdate={"gte": "2023-01-01", "lte": "2023-12-31"},
            paginate=True,
        )
        print("欄位：", list(df.columns))
        print(df.head(5))
        print(f"\n共 {len(df)} 筆")
    except Exception as e:
        print(f"抓取財報失敗（可能無此表權限）：{e}")


if __name__ == "__main__":
    print(f"金鑰結尾：...{TEJ_API_KEY[-4:]}\n")
    check_quota()
    list_tables()
    sample_financials()
    print("\n如果以上都成功，代表 API 與權限正常，可以進入資料抓取階段。")
