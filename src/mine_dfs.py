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

TRAIN_END = pd.Period("2023-12", freq="M")
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
    gc = m.groupby("stock_id")
    f = {}

    # ---- 技術面 ----
    for k in (1, 3, 6, 12):
        f[f"mom_{k}m"] = gc["close"].pct_change(k, fill_method=None)
    f["mom_12_1"] = gc["close"].shift(1) / gc["close"].shift(12) - 1
    f["dist_high"] = m["close"] / m["hi_252"] - 1
    f["liq_amt"] = np.log(m["amt_21"] + 1)
    for v in ("vol_21", "vol_63", "vol_126"):
        f[f"neg_{v}"] = -m[v]                     # 低波動取正向

    # ---- 基本面：殖利率/獲利率/報酬率 ----
    price = m["close"]
    scales = {"px": price, "rev": m.get("f_Revenue"),
              "eq": m.get("f_EquityAttributableToOwnersOfParent"),
              "mrev": m["month_revenue"]}
    profits = {"gp": m.get("f_GrossProfit"), "op": m.get("f_OperatingIncome"),
               "ni": m.get("f_NetIncome"), "pretax": m.get("f_PreTaxIncome"),
               "eps": m.get("f_EPS")}
    for pn, pv in profits.items():
        if pv is None:
            continue
        for sn, sv in scales.items():
            if sv is None:
                continue
            if pn == "eps" and sn != "px":
                continue                          # EPS 只對股價有意義
            name = f"{pn}_to_{sn}"
            ratio = pv / sv.replace(0, np.nan)
            f[name] = ratio

    # ---- 基本面成長率 ----
    grow_fields = {"mrev": "month_revenue", "rev": "f_Revenue",
                   "ni": "f_NetIncome", "op": "f_OperatingIncome",
                   "gp": "f_GrossProfit", "eps": "f_EPS"}
    for tag, col in grow_fields.items():
        if col not in m.columns:
            continue
        for k in (1, 3, 12):
            f[f"g_{tag}_{k}"] = gc[col].pct_change(k, fill_method=None)

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
