"""
集中管理設定與金鑰。
金鑰一律從 .env 讀取，不寫死在程式碼裡。
"""
import os
from pathlib import Path

# 專案根目錄（config.py 在 src/ 下，往上一層）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def _load_dotenv(env_path: Path) -> None:
    """極簡 .env 載入器，避免額外相依套件。"""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(PROJECT_ROOT / ".env")

# ---- 金鑰 ----
TEJ_API_KEY = os.environ.get("TEJ_API_KEY")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")  # 可留空
# 是否為 FinMind 贊助/付費會員：True 才會去抓現成還原表 TaiwanStockPriceAdj，
# 免費會員設 False（預設），直接用原始價+除權息回推，省下無謂的請求次數。
FINMIND_IS_SPONSOR = os.environ.get("FINMIND_IS_SPONSOR", "0") == "1"

if not TEJ_API_KEY:
    raise RuntimeError(
        "找不到 TEJ_API_KEY，請確認專案根目錄有 .env 且內含 TEJ_API_KEY=..."
    )

# ---- 回測 / 抓取範圍 ----
START_DATE = "2012-01-01"   # 擴大樣本：FinMind 價格回1994、財報1990、資產負債表自2012
END_DATE = "2026-06-25"

# ---- 目標產業 ----
# FinMind TaiwanStockInfo 的 industry_category 在上市/上櫃命名略有差異，
# 因此用「關鍵字子字串比對」而非完全相等，較不易漏抓。
# 跑 fetch_universe.py 時會印出實際抓到的產業別，可再回來微調這份對照。
INDUSTRY_GROUPS = {
    "半導體": ["半導體"],
    "電子": [
        "電子零組件", "電腦及週邊", "光電", "通信網路",
        "其他電子", "電子通路", "資訊服務",
    ],
    "生技": ["生技", "醫療", "製藥"],
    "金融": ["金融", "保險", "證券", "銀行"],
}

# 只保留上市(twse)與上櫃(tpex)的普通股，排除 ETF、權證、興櫃等
ALLOWED_MARKET_TYPES = {"twse", "tpex"}
