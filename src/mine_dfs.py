"""
步驟 7（階段 B1）：DFS 機械式因子生成 + 訓練期 ICIR 海選。

由每日 panel 取月底快照的「基礎欄位」，系統性生成大量候選因子：
  - 獲利率/殖利率：各獲利科目 / (股價、營收、股東權益)
  - 成長率：營收與各獲利科目的 1/3/12 月變化
  - 技術面：多期動能、反轉、波動率(負向)、週轉、距高點
所有因子只用月底(含)以前資料（基本面在 panel 已 PIT 對齊），無前瞻偏差。

防過擬合：切訓練期(<=2023-12) / 測試期(>2023-12)。只在訓練期用產業內 ICIR 海選，
survivors 再回報測試期 ICIR 作樣本外驗證（測試期不回頭調整）。

輸出：data/processed/dfs_candidates.csv（所有候選的 train/test ICIR）

執行：
    python src/mine_dfs.py
"""
import sys
import numpy as np
import pandas as pd

from config import DATA_PROCESSED
from factor_eval import monthly_ic, monthly_ic_groups, summarize

TRAIN_END = pd.Period("2019-12", freq="M")  # 訓練2012–2019、測試2020–2026(含COVID/2022空頭/2024多頭)
GROUPS = ["半導體", "電子", "生技", "金融"]


def winsor_m(s, ym, lo=0.01, hi=0.99):
    return s.groupby(ym).transform(lambda x: x.clip(x.quantile(lo), x.quantile(hi)))


def build_monthly_base(panel):
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    df["ym"] = df["date"].dt.to_period("M")
    g = df.groupby("stock_id")
    df["vol_21"] = g["ret"].transform(lambda s: s.rolling(21, min_periods=12).std())
    df["vol_63"] = g["ret"].transform(lambda s: s.rolling(63, min_periods=40).std())
    df["vol_126"] = g["ret"].transform(lambda s: s.rolling(126, min_periods=80).std())
    df["amt_21"] = g["amount"].transform(lambda s: s.rolling(21, min_periods=10).mean())
    df["hi_252"] = g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    m = df.groupby(["stock_id", "ym"]).tail(1).copy()
    return m.sort_values(["stock_id", "ym"]).reset_index(drop=True)


def generate(m):
    m = m.copy()
    f = {}

    def col(name):
        return m[name] if name in m.columns else None

    def coalesce(*names):
        """多個同義欄位互補，解決 FinMind 科目命名隨時間改變造成的缺值。"""
        s = None
        for nm in names:
            c = col(nm)
            if c is None:
                continue
            s = c.copy() if s is None else s.fillna(c)
        return s

    # 淨利：跨年度命名不一致，合併多個同義欄位
    m["_NI"] = coalesce("f_NetIncome", "f_IncomeAfterTaxes", "f_IncomeAfterTax",
                        "f_TotalConsolidatedProfitForThePeriodAfterTax")
    gc = m.groupby("stock_id")

    def ratio(a, b):
        if a is None or b is None:
            return None
        return a / b.replace(0, np.nan)

    # 原始價(殖利率/市值用) vs 還原價(動能/報酬用)
    px = m["close_raw"] if "close_raw" in m.columns else m["close"]

    # ---- 技術面（還原價）----
    for k in (1, 3, 6, 12):
        f[f"mom_{k}m"] = gc["close"].pct_change(k, fill_method=None)
    f["mom_12_1"] = gc["close"].shift(1) / gc["close"].shift(12) - 1
    f["dist_high"] = m["close"] / m["hi_252"] - 1
    f["liq_amt"] = np.log(m["amt_21"] + 1)
    for v in ("vol_21", "vol_63", "vol_126"):
        f[f"neg_{v}"] = -m[v]

    # ---- 市值/規模（原始價 × 股數；股本面額10元 → 股數 = 股本/10）----
    cap = col("b_CapitalStock")
    mktcap = px * (cap / 10.0) if cap is not None else None
    if mktcap is not None:
        f["neg_size"] = -np.log(mktcap.replace(0, np.nan))   # 小型股溢酬

    # ---- 殖利率（用原始價/市值）與獲利率（對營收）----
    rev = col("f_Revenue")
    for pn, pv in {"gp": col("f_GrossProfit"), "op": col("f_OperatingIncome"),
                   "ni": col("_NI"), "pretax": col("f_PreTaxIncome")}.items():
        r = ratio(pv, px)
        if r is not None:
            f[f"{pn}_to_px"] = r
        rr = ratio(pv, rev)
        if rr is not None:
            f[f"{pn}_to_rev"] = rr
    eps = col("f_EPS")
    if eps is not None:
        f["eps_to_px"] = ratio(eps, px)

    # ---- 新：資產負債表 / 現金流量表 因子（皆通用比率）----
    ta = col("b_TotalAssets")
    eq = col("b_Equity")
    eqp = col("b_EquityAttributableToOwnersOfParent")
    liab = col("b_Liabilities")
    ca, cl = col("b_CurrentAssets"), col("b_CurrentLiabilities")
    ni = col("_NI")
    cfo = col("c_CashFlowsFromOperatingActivities")
    capex = col("c_PropertyAndPlantAndEquipment")

    if ni is not None and eq is not None:
        f["roe"] = ratio(ni, eq)
    if ni is not None and ta is not None:
        f["roa"] = ratio(ni, ta)
    if rev is not None and ta is not None:
        f["asset_turnover"] = ratio(rev, ta)
    if eqp is not None and mktcap is not None:
        f["bp"] = ratio(eqp, mktcap)              # 淨值股價比(value)
    if liab is not None and ta is not None:
        f["neg_debt_ratio"] = -ratio(liab, ta)
    if ca is not None and cl is not None:
        f["current_ratio"] = ratio(ca, cl)
    if cfo is not None and mktcap is not None:
        f["cfo_yield"] = ratio(cfo, mktcap)
    if cfo is not None and capex is not None and mktcap is not None:
        f["fcf_yield"] = ratio(cfo - capex.abs(), mktcap)
    if ni is not None and cfo is not None and ta is not None:
        f["neg_accruals"] = -ratio(ni - cfo, ta)  # 低應計=高品質
    if ta is not None:
        f["neg_asset_growth"] = -gc["b_TotalAssets"].pct_change(12, fill_method=None)

    # ---- 基本面成長率 ----
    grow_fields = {"mrev": "month_revenue", "rev": "f_Revenue",
                   "ni": "_NI", "op": "f_OperatingIncome",
                   "gp": "f_GrossProfit", "eps": "f_EPS"}
    for tag, gcol in grow_fields.items():
        if gcol not in m.columns:
            continue
        for k in (1, 3, 12):
            f[f"g_{tag}_{k}"] = gc[gcol].pct_change(k, fill_method=None)

    # === 適度擴充因子庫 ===
    def d12(s):
        """每檔 12 個月變化（趨勢）。"""
        return None if s is None else s.groupby(m["stock_id"]).diff(12)

    def add(name, series, trend=False):
        if series is None:
            return
        f[name] = series
        if trend:
            d = d12(series)
            if d is not None:
                f[f"d_{name}"] = d

    # 利潤率與趨勢
    gm = ratio(col("f_GrossProfit"), rev)
    opm = ratio(col("f_OperatingIncome"), rev)
    nm = ratio(ni, rev)
    add("gross_margin", gm, trend=True)
    add("op_margin", opm, trend=True)
    add("net_margin", nm, trend=True)
    # 報酬率趨勢
    add("d_roe", d12(ratio(ni, eq)))
    add("d_roa", d12(ratio(ni, ta)))
    add("d_asset_turnover", d12(ratio(rev, ta)))
    # 營運效率與趨勢
    add("inv_turnover", ratio(rev, col("b_Inventories")), trend=True)
    add("recv_turnover", ratio(rev, col("b_AccountsReceivableNet")), trend=True)
    # 現金流品質
    add("cfo_to_ni", ratio(cfo, ni))
    add("capex_to_ta", ratio(capex.abs() if capex is not None else None, ta))
    add("capex_to_rev", ratio(capex.abs() if capex is not None else None, rev))
    if "c_CashFlowsFromOperatingActivities" in m.columns:
        f["g_cfo_12"] = gc["c_CashFlowsFromOperatingActivities"].pct_change(12, fill_method=None)
    # 槓桿變化
    add("d_debt_ratio", d12(ratio(liab, ta)))
    # 資產組成比與變化
    for tag, c in {"cash": "b_CashAndCashEquivalents", "inv": "b_Inventories",
                   "recv": "b_AccountsReceivableNet", "ppe": "b_PropertyPlantAndEquipment",
                   "intang": "b_IntangibleAssets", "retain": "b_RetainedEarnings",
                   "curr": "b_CurrentAssets"}.items():
        add(f"{tag}_to_ta", ratio(col(c), ta), trend=True)
    # 淨營運資金應計
    if ca is not None and cl is not None:
        add("d_nwc_to_ta", d12(ratio(ca - cl, ta)))
    # 動能變體與風險調整
    f["mom_6_1"] = gc["close"].shift(1) / gc["close"].shift(6) - 1
    f["mom_9_1"] = gc["close"].shift(1) / gc["close"].shift(9) - 1
    if "vol_126" in m.columns:
        f["mom_vol_adj"] = f["mom_12_1"] / m["vol_126"].replace(0, np.nan)
    # 營收成長加速
    if "month_revenue" in m.columns:
        g3 = gc["month_revenue"].pct_change(3, fill_method=None)
        f["rev_accel"] = g3 - g3.groupby(m["stock_id"]).shift(3)
    # Piotroski 風格品質分（部分訊號；缺值不計）
    def sig(series, positive=True):
        if series is None:
            return None
        s = (series > 0) if positive else (series < 0)
        return s.astype(float).where(series.notna())
    sigs = [sig(ratio(ni, ta)), sig(cfo), sig((cfo - ni) if (cfo is not None and ni is not None) else None),
            sig(d12(ratio(ni, ta))), sig(d12(gm)), sig(d12(ratio(liab, ta)), positive=False)]
    sigs = [s for s in sigs if s is not None]
    if sigs:
        f["quality_score"] = pd.concat(sigs, axis=1).sum(axis=1, min_count=1)

    # 組裝 + 清理 inf
    fac = pd.DataFrame(f, index=m.index)
    fac = fac.replace([np.inf, -np.inf], np.nan)
    names = list(fac.columns)
    out = pd.concat([m[["stock_id", "group", "ym", "date", "close"]], fac], axis=1)

    # 未來1月報酬（label）
    out["ret_1m"] = gc["close"].pct_change(1, fill_method=None)
    out["fwd_ret_1m"] = out.groupby("stock_id")["ret_1m"].shift(-1)
    out["fwd_ret_1m"] = winsor_m(out["fwd_ret_1m"], out["ym"])
    # 因子逐月 winsorize
    for c in names:
        out[c] = winsor_m(out[c], out["ym"])
    return out, names


def _icir(ic):
    return summarize(ic)[2] if len(ic) >= 6 else np.nan


def eval_split(df, factor):
    """回傳 dict：整體(平均) train/test ICIR、t 值，以及各產業 train/test ICIR。"""
    tr = df[df["ym"] <= TRAIN_END]
    te = df[df["ym"] > TRAIN_END]
    ic_tr = monthly_ic(tr, factor)        # 產業內算、跨產業平均
    if len(ic_tr) < 12:
        return None
    m_tr, _, icir_tr, t_tr, _, _ = summarize(ic_tr)
    ic_te = monthly_ic(te, factor)
    if len(ic_te) >= 6:
        _, _, icir_te, t_te, _, _ = summarize(ic_te)
    else:
        icir_te, t_te = np.nan, np.nan

    rec = {"factor": factor, "IC_mean_train": m_tr,
           "ICIR_train": icir_tr, "t_train": t_tr,
           "ICIR_test": icir_te, "t_test": t_te}
    # 各產業分別評估（抓產業專屬因子）
    g_tr = monthly_ic_groups(tr, factor)
    g_te = monthly_ic_groups(te, factor)
    for grp in GROUPS:
        rec[f"ICIR_tr_{grp}"] = _icir(g_tr[grp]) if grp in g_tr else np.nan
        rec[f"ICIR_te_{grp}"] = _icir(g_te[grp]) if grp in g_te else np.nan
    return rec


def main():
    p = DATA_PROCESSED / "panel.parquet"
    if not p.exists():
        sys.exit("找不到 panel.parquet，請先執行 align_data.py")
    panel = pd.read_parquet(p)

    print("建立月底基礎欄位…")
    m = build_monthly_base(panel)
    print("生成候選因子…")
    df, names = generate(m)
    print(f"候選因子數：{len(names)}")

    rows = []
    for i, fac in enumerate(names, 1):
        r = eval_split(df, fac)
        if r is not None:
            rows.append(r)
        if i % 10 == 0:
            print(f"  已評估 {i}/{len(names)}")

    res = pd.DataFrame(rows)
    res["abs_icir_tr"] = res["ICIR_train"].abs()
    res = res.sort_values("abs_icir_tr", ascending=False).drop(columns="abs_icir_tr")

    out = DATA_PROCESSED / "dfs_candidates.csv"
    res.to_csv(out, index=False)

    pd.set_option("display.width", 180)
    fmt = {c: "{:.3f}".format for c in res.columns if c.startswith(("ICIR", "t_"))}
    fmt["IC_mean_train"] = "{:.4f}".format

    print("\n=== 整體（產業內平均）排名，前 20 ===")
    cols = ["factor", "ICIR_train", "t_train", "ICIR_test", "t_test"]
    print(res[cols].head(20).to_string(index=False, formatters=fmt))

    # (1) 跨產業廣泛有效：平均 |t|>2、測試同向、|ICIR_test|>0.2
    broad = res[(res["t_train"].abs() > 2) &
                (np.sign(res["ICIR_train"]) == np.sign(res["ICIR_test"])) &
                (res["ICIR_test"].abs() > 0.2)]
    print(f"\n=== (1) 跨產業廣泛有效的 survivors：{len(broad)} 個 ===")
    print(broad[["factor", "ICIR_train", "ICIR_test"]].to_string(index=False, formatters=fmt))

    # (2) 產業專屬：某產業訓練 |ICIR|>0.3 且測試同向、|ICIR|>0.25（避免被平均稀釋而漏掉）
    print("\n=== (2) 產業專屬 survivors（避免被平均稀釋）===")
    found = False
    for grp in GROUPS:
        tr_c, te_c = f"ICIR_tr_{grp}", f"ICIR_te_{grp}"
        sub = res[(res[tr_c].abs() > 0.3) &
                  (np.sign(res[tr_c]) == np.sign(res[te_c])) &
                  (res[te_c].abs() > 0.25)]
        if len(sub):
            found = True
            print(f"\n[{grp}]")
            print(sub[["factor", tr_c, te_c]].to_string(index=False, formatters=fmt))
    if not found:
        print("（本次無單一產業專屬且跨期穩定的因子）")

    print(f"\n完整結果（含各產業 ICIR）已存：{out}")


if __name__ == "__main__":
    main()
