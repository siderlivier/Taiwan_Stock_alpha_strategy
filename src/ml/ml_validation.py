"""
過擬合正式檢定：Deflated Sharpe Ratio (DSR) + Probability of Backtest Overfitting (PBO)。
（Bailey & López de Prado）

概念：
  - DSR：一個回測 Sharpe 之所以「看起來好」，可能是「試了很多策略挑出的幸運兒」。
    DSR 依『試驗數 N、各試驗 Sharpe 的變異、樣本長度、報酬偏態峰態』把 Sharpe 打折，
    給出「真實 Sharpe > 去膨脹門檻」的機率。>0.95 表示扣掉多重測試後仍顯著。
  - PBO：用 CSCV 把時間切成 S 塊、窮舉半數當 IS 半數當 OOS；看『在 IS 最好的策略，
    到 OOS 是否落在中位數以下』的比例。PBO 越低越好（<0.5 代表選擇流程沒有嚴重過擬合）。

策略母體（試驗集）：以 ~86 個 DFS 因子各自的『產業內前 10% 等權』策略 + 4 個模型，
代表我們搜尋過的空間。所有策略取相同的樣本外月份。

需安裝：pip install scipy （lightgbm 等前一步已裝）
前置：先跑 src/ml/ml_model.py 產生 data/processed/ml_scores.parquet
執行：python src/ml/ml_validation.py
"""
import os
import sys
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import DATA_PROCESSED
from mine_dfs import build_monthly_base, generate
from factor_eval import monthly_ic
import backtest as bt

MODEL_COLS = {"等權": "s_等權baseline", "Ridge": "s_Ridge線性",
              "LightGBM": "s_LightGBM", "LightGBM_規模中性": "s_LightGBM(規模中性)"}


def psr(sr, sr_star, T, skew, kurt):
    """Probabilistic Sharpe：P(真實 Sharpe > sr_star)。sr 為每期(月) Sharpe。"""
    den = np.sqrt(1 - skew * sr + ((kurt - 1) / 4.0) * sr ** 2)
    return norm.cdf((sr - sr_star) * np.sqrt(T - 1) / den)


def deflated_sharpe(rets, trial_sharpes):
    r = rets.dropna()
    T = len(r)
    sr = r.mean() / r.std()
    skew = r.skew()
    kurt = r.kurtosis() + 3.0                  # pandas 為超額峰態，轉回非超額
    ts = np.array([s for s in trial_sharpes if np.isfinite(s)])
    N = len(ts)
    var_sr = ts.var(ddof=1)
    g = 0.5772156649                            # Euler–Mascheroni
    z1 = norm.ppf(1 - 1.0 / N)
    z2 = norm.ppf(1 - 1.0 / (N * np.e))
    sr0 = np.sqrt(var_sr) * ((1 - g) * z1 + g * z2)   # 去膨脹門檻(期望最大 Sharpe)
    return psr(sr, sr0, T, skew, kurt), sr, sr0, N


def cscv_pbo(R, S=12):
    """CSCV 估 PBO。R: 月報酬矩陣(index=月, columns=策略)。"""
    R = R.dropna(axis=0, how="any")
    T, N = R.shape
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for isin in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[b] for b in isin])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in isin])
        Ris, Roos = R.iloc[is_rows], R.iloc[oos_rows]
        sr_is = Ris.mean() / Ris.std()
        best = sr_is.idxmax()
        sr_oos = Roos.mean() / Roos.std()
        rank = sr_oos.rank().loc[best]          # 1..N，越大越好
        omega = rank / (N + 1)
        logits.append(np.log(omega / (1 - omega)))
    logits = np.array(logits)
    return (logits <= 0).mean(), logits


def strat_returns(oos, col, orient=False):
    """某策略的樣本外『主動報酬』= 產業內前10%等權 − 全池等權基準。
    減掉共享的市場 beta，DSR/PBO 才是測『相對選股技巧』而非大家一起吃多頭。"""
    d = oos[["stock_id", "group", "ym", "fwd_ret_1m", col]].dropna(
        subset=[col, "fwd_ret_1m"]).copy()
    if len(d) < 100:
        return None
    if orient:                                  # 因子依樣本外 IC 正負定方向
        ic = monthly_ic(d, col)
        sign = 1.0 if (len(ic) and ic.mean() >= 0) else -1.0
        d[col] = sign * d[col]
    d = d.rename(columns={col: "score"})
    rets = bt.portfolio_returns(d, top_q=0.10, weighting="equal")
    return rets["long"] - rets["benchmark"]     # 主動(超額)報酬


def main():
    if not (DATA_PROCESSED / "ml_scores.parquet").exists():
        sys.exit("找不到 ml_scores.parquet，請先執行 src/ml/ml_model.py")
    print("建立因子面板 + 讀取模型分數…")
    m = build_monthly_base(pd.read_parquet(DATA_PROCESSED / "panel.parquet"))
    df, names = generate(m)
    ml = pd.read_parquet(DATA_PROCESSED / "ml_scores.parquet")
    feats = [c for c in names if df[c].notna().mean() > 0.3]

    merged = df.merge(ml[["stock_id", "ym"] + list(MODEL_COLS.values())],
                      on=["stock_id", "ym"], how="left")
    oos = merged[merged["s_LightGBM"].notna()].copy()
    print(f"樣本外月份：{oos['ym'].nunique()}　建立 {len(feats)} 因子 + "
          f"{len(MODEL_COLS)} 模型 的策略報酬矩陣（需幾分鐘）…")

    cols = {}
    for f in feats:                              # 因子策略(依 IC 定向)
        r = strat_returns(oos, f, orient=True)
        if r is not None:
            cols[f] = r
    for name, col in MODEL_COLS.items():         # 模型策略
        r = strat_returns(oos, col, orient=False)
        if r is not None:
            cols[name] = r
    R = pd.DataFrame(cols).sort_index()

    ANN = 12
    trial_sr = [(R[c].dropna().mean() / R[c].dropna().std())
                for c in R.columns if R[c].dropna().std() > 0]
    dsr, sr, sr0, N = deflated_sharpe(R["LightGBM"], trial_sr)

    print("\n===== Deflated Sharpe（LightGBM『主動報酬』= 超額，已剝離市場 beta）=====")
    print(f"樣本外月數 T：{R['LightGBM'].dropna().shape[0]}　試驗數 N：{N}")
    print(f"LightGBM 超額每月 Sharpe(≈月IR)：{sr:.3f}（年化 {sr*np.sqrt(ANN):.2f}）")
    print(f"去膨脹門檻 SR0（期望最大，每月）：{sr0:.3f}（年化 {sr0*np.sqrt(ANN):.2f}）")
    print(f"**Deflated Sharpe（真實超額SR>門檻之機率）：{dsr:.3f}**")
    print("判讀：>0.95 表示扣掉『試了 N 個策略』的多重測試後，『選股超額』仍顯著。")

    pbo, logits = cscv_pbo(R, S=12)
    print("\n===== PBO（回測過擬合機率，CSCV，用『超額報酬』）=====")
    print(f"策略母體 N={R.shape[1]}　切塊 S=12（C(12,6)=924 組）")
    print(f"**PBO：{pbo:.3f}**（IS 最佳『超額』策略在 OOS 落到中位數以下的比例）")
    print("判讀：<0.5 代表選股技巧的持續性沒有嚴重過擬合；越接近 0 越好。"
          "（已減基準 → 測的是相對技巧，非共享的多頭 beta）")


if __name__ == "__main__":
    main()
