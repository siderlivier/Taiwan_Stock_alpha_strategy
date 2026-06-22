# 台股 Alpha 策略專案

同產業中性化的多因子 Alpha 策略。第一階段鎖定四個產業：**半導體、電子、生技、金融**。

資料來源分工：
- **價格**：FinMind（免費）。還原股價（`TaiwanStockPriceAdj`）為付費會員專屬，免費版用「未還原價＋除權息」自動回推（`taiwan_stock_daily_adj` 同原理）。
- **財報 / 月營收**：TEJ API（試用帳號），依**發布日**對映，避免預知未來。

## 資料夾結構

```
tw_alpha_strategy/
├── .env                    # TEJ / FinMind 金鑰（不進版控）
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                # 抓下來的原始資料（parquet）
│   └── processed/          # 對齊、清理後的特徵
├── notebooks/
└── src/
    ├── config.py           # 設定、金鑰、四產業關鍵字、日期範圍
    ├── finmind_client.py   # FinMind 封裝 + 還原價回推
    ├── test_connection.py  # TEJ 連線/權限測試
    ├── fetch_universe.py   # 步驟1：建立四產業股票池
    ├── fetch_prices.py     # 步驟2：抓還原日價
    └── fetch_fundamentals.py # 步驟3：抓財報/月營收（含發布日）
```

## 環境準備

```bash
pip install -r requirements.txt
```

到 https://finmindtrade.com 免費註冊，取得 token 後填入 `.env` 的 `FINMIND_TOKEN`（留空也能跑，但會限速）。

## 執行順序

```bash
python src/test_connection.py     # 0. 確認 TEJ 權限與財報欄位
python src/fetch_universe.py      # 1. 建立股票池（會印出實際產業別，可回 config 微調）
python src/fetch_prices.py        # 2. 抓還原日價（可中斷續抓）
python src/fetch_fundamentals.py  # 3. 抓財報/月營收（每天 5 萬筆上限，會自動分日續抓）
```

## 已知限制（TEJ 試用帳號）

- 只有未還原日價、財報資料自 **2018** 年起、每天上限 **50,000 筆**。
- 因此價格走 FinMind、財報用 TEJ；全市場研究須升級付費或 FinMind 贊助會員。

## Pipeline 後續步驟（尚未實作）

1. **發布日對齊**：財報依「發布日 + 保守 lag」對齊到交易日，價格用還原價。
2. **特徵生成**：DFS 深度特徵合成 + GP 基因規劃。
3. **因子初篩**：產業內 rank IC / ICIR。
4. **中性化與正交化**：先對市值/動能/產業中性化，再彼此正交去重。
5. **回測**：walk-forward + 過擬合檢定（Deflated Sharpe / PBO），同產業範圍。

## 安全提醒

`.env` 含金鑰，已被 `.gitignore` 排除。金鑰曾貼於對話中，建議到 TEJ 後台重新產生一組。
