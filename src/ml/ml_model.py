"""
ML 因子合成模型（LightGBM）+ 模型階梯比較 + 換手/成本敏感度 + 規模中性驗證。

放在 src/ml/ 底下；因需匯入 src/ 的 config、mine_dfs、factor_eval、backtest，
於檔頭把上層 src/ 加入模組搜尋路徑。

流程：85 個 DFS 因子當特徵、未來報酬(產業內去均值)當 label；walk-forward + embargo
時序 CV 產生樣本外分數；比較 等權 / Ridge / LightGBM / LightGBM(規模中性)；
每個模型報告 ICIR、回測績效與『換手率』；再對 LightGBM 做成本敏感度掃描。

需安裝：pip install lightgbm scikit-learn shap
執行：python src/ml/ml_model.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED
from mine_dfs import build_monthly_base, generate
from factor_eval import monthly_ic, summarize
import backtest as bt

try:
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
except ImportError:
    sys.exit("請先安裝： pip install lightgbm scikit-learn shap")

MIN_TRAIN_MONTHS = 48
RETRAIN_EVERY = 12
EMBARGO = 1
SIZE_FEATURES = ["liq_amt", "neg_size"]   # 規模/流動性代理，規模中性版會排除


def prep(df, feats):
    df = df.copy()
    g = df.groupby(["ym", "group"])
    for c in feats:
        df[c] = (df[c] - g[c].transform("mean")) / g[c].transform("std")
    df[feats] = df[feats].fillna(0.0)
    df["y"] = df["fwd_ret_1m"] - g["fwd_ret_1m"].transform("mean")
    return df


def lgbm():
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=31, max_depth=5,
        min_child_samples=200, subsample=0.8, colsample_bytree=0.6,
        reg_lambda=5.0, random_state=0, n_jobs=-1, verbose=-1)


def walk_forward(df, feats, kind):
    months = sorted(df["ym"].unique())
    pred = pd.Series(np.nan, index=df.index)
    i = MIN_TRAIN_MONTHS
    while i < len(months):
        train_end = months[i - 1 - EMBARGO]
        test_months = months[i:i + RETRAIN_EVERY]
        tr = df[df["ym"] <= train_end]
        te = df[df["ym"].isin(test_months)]
        if len(tr) >= 1000 and len(te):
            if kind == "equal":
                pred.loc[te.index] = te[feats].mean(axis=1)
            elif kind == "ridge":
                mdl = Ridge(alpha=10.0).fit(tr[feats].values, tr["y"].values)
                pred.loc[te.index] = mdl.predict(te[feats].values)
            else:  # lgbm
                mdl = lgbm().fit(tr[feats].values, tr["y"].values)
                pred.loc[te.index] = mdl.predict(te[feats].values)
        i += RETRAIN_EVERY
    return pred


def evaluate(base, score_col):
    """回傳 (ICIR, perf dict, 平均換手率)。回測用等權前10%，隔離分數品質。"""
    d = base.dropna(subset=[score_col]).copy()
    icir = summarize(monthly_ic(d, score_col))[2] if d[score_col].notna().any() else np.nan
    d = d.rename(columns={score_col: "score"})
    rets = bt.portfolio_returns(d, top_q=0.10, weighting="equal")
    return icir, bt.perf(rets["long"]), rets["turnover"].mean()


def main():
    p = DATA_PROCESSED / "panel.parquet"
    if not p.exists():
        sys.exit("找不到 panel.parquet，請先執行 align_data.py")
    print("建立因子面板…")
    m = build_monthly_base(pd.read_parquet(p))
    df, names = generate(m)
    feats = [c for c in names if df[c].notna().mean() > 0.3]
    df = df.dropna(subset=["fwd_ret_1m"]).reset_index(drop=True)
    proc = prep(df, feats)
    base = proc[["stock_id", "group", "ym", "fwd_ret_1m"]].copy()
    print(f"特徵數：{len(feats)}（全 DFS 因子；規模中性版排除 {SIZE_FEATURES}）")

    specs = [("等權baseline", "equal", feats),
             ("Ridge線性", "ridge", feats),
             ("LightGBM", "lgbm", feats),
             ("LightGBM(規模中性)", "lgbm", [f for f in feats if f not in SIZE_FEATURES])]

    print(f"\n{'模型':>16} {'ICIR':>7} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'換手/月':>7}")
    scorecols = {}
    for label, kind, fl in specs:
        col = "s_" + label
        base[col] = walk_forward(proc, fl, kind)
        scorecols[label] = col
        icir, pf, turn = evaluate(base[["stock_id", "group", "ym", "fwd_ret_1m", col]], col)
        print(f"{label:>16} {icir:>7.3f} {pf['CAGR']:>8.2%} {pf['Sharpe']:>7.2f} "
              f"{pf['MaxDD']:>8.2%} {turn:>7.1%}")

    # 成本敏感度（對 LightGBM 全特徵版）
    print("\n=== LightGBM 成本敏感度（等權前10%）===")
    lgbm_col = scorecols["LightGBM"]
    dcol = base[["stock_id", "group", "ym", "fwd_ret_1m", lgbm_col]].dropna(
        subset=[lgbm_col]).rename(columns={lgbm_col: "score"})
    print(f"{'單次成本':>8} {'CAGR':>8} {'Sharpe':>7}")
    orig_cost = bt.COST
    for cst in [0.002, 0.004, 0.008, 0.012, 0.016]:
        bt.COST = cst
        rets = bt.portfolio_returns(dcol, top_q=0.10, weighting="equal")
        pf = bt.perf(rets["long"])
        print(f"{cst:>8.1%} {pf['CAGR']:>8.2%} {pf['Sharpe']:>7.2f}")
    bt.COST = orig_cost

    # 特徵重要性 + SHAP（訓練期擬合一次）
    months = sorted(proc["ym"].unique())
    tr = proc[proc["ym"] <= months[MIN_TRAIN_MONTHS]]
    final = lgbm().fit(tr[feats].values, tr["y"].values)
    imp = pd.Series(final.feature_importances_, index=feats).sort_values(ascending=False)
    print("\n=== LightGBM 特徵重要性（前 15）===")
    print(imp.head(15).to_string())
    try:
        import shap
        samp = tr[feats].sample(min(2000, len(tr)), random_state=0)
        sv = np.abs(shap.TreeExplainer(final).shap_values(samp.values)).mean(0)
        print("\n=== SHAP 平均絕對貢獻（前 15）===")
        print(pd.Series(sv, index=feats).sort_values(ascending=False).head(15).to_string())
    except ImportError:
        print("\n(未安裝 shap，略過)")

    base.to_parquet(DATA_PROCESSED / "ml_scores.parquet", index=False)
    print(f"\n樣本外分數已存：{DATA_PROCESSED / 'ml_scores.parquet'}")
    print("\n判讀：(1) 規模中性版若仍勝 Ridge/等權 → edge 不只是小型股傾斜。"
          "(2) 成本拉高後 LightGBM 若快速衰退 → 報酬多來自高換手，實盤存疑。")


if __name__ == "__main__":
    main()
