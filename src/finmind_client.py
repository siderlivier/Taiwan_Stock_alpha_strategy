"""
FinMind API 輕量封裝（只用 requests，不強制裝 FinMind 套件）。
- 免費資料集：TaiwanStockInfo、TaiwanStockPrice、TaiwanStockDividendResult
- 還原股價：免費版沒有 TaiwanStockPriceAdj 表，這裡用「未還原價 + 除權息」自行回推。
"""
import time
import requests
import pandas as pd

from config import FINMIND_TOKEN, FINMIND_IS_SPONSOR

BASE_URL = "https://api.finmindtrade.com/api/v4/data"


class FinMindError(RuntimeError):
    pass


def _get(dataset: str, params: dict | None = None, retries: int = 3) -> pd.DataFrame:
    """呼叫 FinMind /data 端點，回傳 DataFrame。內建限速重試。"""
    payload = {"dataset": dataset}
    if params:
        payload.update(params)
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"} if FINMIND_TOKEN else {}

    for attempt in range(retries):
        resp = requests.get(BASE_URL, headers=headers, params=payload, timeout=60)
        # 402 / 429：超過免費額度或限速 → 等待後重試
        if resp.status_code in (402, 429):
            wait = 30 * (attempt + 1)
            print(f"  [FinMind] 達到限速/額度({resp.status_code})，等 {wait}s 後重試…")
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            raise FinMindError(f"{dataset} HTTP {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        if body.get("status") not in (200, None):
            # FinMind 用 msg 回報權限/參數問題
            raise FinMindError(f"{dataset}: {body.get('msg')}")
        return pd.DataFrame(body.get("data", []))
    raise FinMindError(f"{dataset}: 多次重試後仍失敗（可能額度用盡）")


def get_stock_info() -> pd.DataFrame:
    """台股總覽：industry_category / stock_id / stock_name / type / date。"""
    return _get("TaiwanStockInfo")


def get_price_raw(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """未還原日價：date/stock_id/open/max/min/close/Trading_Volume/...。
    會剔除 close<=0 的停牌/無效列，避免後續報酬率出現 -1 / inf 假極端值。"""
    df = _get("TaiwanStockPrice",
              {"data_id": stock_id, "start_date": start, "end_date": end})
    if not df.empty and "close" in df.columns:
        df = df[df["close"] > 0].reset_index(drop=True)
    return df


def get_dividend_result(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """除權息結果（含除權息參考價、權值息值），用來回推還原價。"""
    return _get("TaiwanStockDividendResult",
                {"data_id": stock_id, "start_date": start, "end_date": end})


def get_month_revenue(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """月營收：date(揭露日,約次月1日)/stock_id/revenue/revenue_month/revenue_year。
    對齊時以 revenue_year/revenue_month 定所屬期間，並加保守 lag 才視為可用。"""
    return _get("TaiwanStockMonthRevenue",
                {"data_id": stock_id, "start_date": start, "end_date": end})


def get_financial_statement(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """綜合損益表(long)：date(財報期末)/stock_id/type/value/origin_name。
    無公告日，對齊時用法定申報期限近似發布日。"""
    return _get("TaiwanStockFinancialStatements",
                {"data_id": stock_id, "start_date": start, "end_date": end})


def get_balance_sheet(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """資產負債表(long)：date(期末)/stock_id/type/value/origin_name；資料自 2011-12 起。"""
    return _get("TaiwanStockBalanceSheet",
                {"data_id": stock_id, "start_date": start, "end_date": end})


def get_cashflow(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """現金流量表(long)：date(期末)/stock_id/type/value/origin_name；資料自 2008-06 起。"""
    return _get("TaiwanStockCashFlowsStatement",
                {"data_id": stock_id, "start_date": start, "end_date": end})


def back_adjust(price: pd.DataFrame, dividend: pd.DataFrame) -> pd.DataFrame:
    """
    後復權（back-adjust）：用除權息「參考價 / 除權息前一日收盤」當每次事件的調整係數，
    將係數累乘成還原因子，乘回 OHLC。這是最常見、無前瞻性的還原作法。

    回傳新增欄位：adj_factor（還原因子）、adj_close/adj_open/adj_high/adj_low。
    若 dividend 為空，係數=1（等於未還原）。
    """
    price = price.sort_values("date").reset_index(drop=True).copy()
    price["date"] = pd.to_datetime(price["date"])
    price["adj_factor"] = 1.0

    if (dividend is not None and not dividend.empty
            and {"before_price", "after_price"}.issubset(dividend.columns)):
        div = dividend.copy()
        div["date"] = pd.to_datetime(div["date"])  # 除權息交易日(ex-date)
        div = div.sort_values("date")
        for _, ev in div.iterrows():
            b, a = ev.get("before_price"), ev.get("after_price")
            # 標準後復權係數 = 除權息後收盤價 / 除權息前收盤價
            if pd.isna(b) or pd.isna(a) or float(b) <= 0 or float(a) <= 0:
                continue
            factor = float(a) / float(b)
            if factor <= 0:
                continue
            # ex-date 之前(不含當日)的價格全部乘上係數
            price.loc[price["date"] < ev["date"], "adj_factor"] *= factor

    for col in ["open", "max", "min", "close"]:
        if col in price.columns:
            price[f"adj_{col}"] = price[col] * price["adj_factor"]
    return price


def get_price_adjusted(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """
    取得還原日價。贊助會員優先用現成還原表 TaiwanStockPriceAdj；
    免費會員（FINMIND_IS_SPONSOR=False）直接用未還原價 + 除權息自行回推，
    省下一次必然失敗的請求。
    """
    if FINMIND_IS_SPONSOR:
        try:
            df = _get("TaiwanStockPriceAdj",
                      {"data_id": stock_id, "start_date": start, "end_date": end})
            if not df.empty:
                df["adj_source"] = "TaiwanStockPriceAdj"
                return df
        except FinMindError:
            pass  # 萬一還是失敗，往下走自行回推

    raw = get_price_raw(stock_id, start, end)
    if raw.empty:
        return raw
    try:
        div = get_dividend_result(stock_id, start, end)
    except FinMindError:
        div = pd.DataFrame()
    adj = back_adjust(raw, div)
    adj["adj_source"] = "self_back_adjust"
    return adj
