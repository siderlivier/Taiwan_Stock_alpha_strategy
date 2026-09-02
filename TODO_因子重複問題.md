# 待處理清單：DFS 因子重複問題

> 建立日期：2026-09-01
> 發現途徑：`alpha_mining_agent` 專案在做因子組合合成時，對匯入的參考因子
> （來自本專案 `mine_dfs.py`）做兩兩相關度檢查，發現數對 ρ ≥ 0.95 的因子。
> 追查後確認其中三對是**程式碼層級的重複**，不是巧合。

---

## A. 確定的重複：同一條公式掛兩個名字（3 對）

### 成因

`src/mine_dfs.py` 的 `generate()` 在兩個不同區塊算了同一條公式：

```python
# 約第 95~102 行：通用「獲利對營收 / 對股價」比率迴圈
for pn, pv in {"gp": col("f_GrossProfit"), "op": col("f_OperatingIncome"),
               "ni": col("_NI"), "pretax": col("f_PreTaxIncome")}.items():
    r = ratio(pv, px)
    if r is not None:
        f[f"{pn}_to_px"] = r
    rr = ratio(pv, rev)
    if rr is not None:
        f[f"{pn}_to_rev"] = rr        # ← gp_to_rev / op_to_rev / ni_to_rev

# 約第 165~168 行：後來新增的「=== 適度擴充因子庫 ===」區塊
gm  = ratio(col("f_GrossProfit"), rev)
opm = ratio(col("f_OperatingIncome"), rev)
nm  = ratio(ni, rev)
add("gross_margin", gm,  trend=True)   # ← 與 gp_to_rev 完全相同
add("op_margin",    opm, trend=True)   # ← 與 op_to_rev 完全相同
add("net_margin",   nm,  trend=True)   # ← 與 ni_to_rev 完全相同
```

擴充區塊寫的時候沒有回頭檢查前面那個迴圈已經算過同樣的東西。

### 證據

`data/processed/dfs_candidates.csv` 裡，三對的評估指標**小數點後 16 位完全相同**
（浮點數逐位相等，不可能是兩條不同公式算出來的巧合）：

| 因子 A | 因子 B | ICIR_train | ICIR_test | t_train | ρ（月內排名） |
|---|---|---|---|---|---|
| `net_margin` | `ni_to_rev` | +0.5224 | +0.3877 | 5.01 | **1.000** |
| `op_margin` | `op_to_rev` | +0.4596 | +0.5276 | 4.41 | **1.000** |
| `gross_margin` | `gp_to_rev` | +0.1207 | −0.0396 | 1.16 | **1.000** |

（完整精度：`net_margin` 與 `ni_to_rev` 的 ICIR_train 皆為
`0.5224053537899057`；`op_margin` 與 `op_to_rev` 皆為 `0.4596369922253636`。）

已檢查全部 86 個候選，**就只有這三對**指標完全相同。

### 影響範圍

1. **`src/ml/ml_model.py`** — 把 85 個 DFS 因子全部當特徵。
   - 「等權 baseline」那一欄，淨利率與營益率各被算了兩次，
     等於這兩個概念被賦予兩倍權重。**這是目前最實質的影響**，
     因為等權 baseline 是用來判斷「LightGBM 有沒有比笨方法好」的基準線，
     基準線本身被扭曲了。
   - Ridge / LightGBM 受影響較小（完全共線的特徵會被正則化拆分權重，
     樹模型則會隨機選其中一個），但特徵重要性與 SHAP 的解讀會被稀釋：
     同一個訊號的重要性被拆成兩半，排名會低估。
2. **因子數的宣稱** — 「85 個 DFS 因子」實際上只有 82 個獨立公式。
3. **`mine_gp.py`（若有用到同一份因子池）** — 尚未確認，需檢查。

### 建議修法（擇一）

- **（推薦）** 刪掉「適度擴充因子庫」區塊裡的 `add("gross_margin", ...)`、
  `add("op_margin", ...)`、`add("net_margin", ...)` 三行的**本體**，
  但**保留 `trend=True` 產生的 `d_*` 衍生因子**——`d_gross_margin` /
  `d_op_margin` / `d_net_margin` 是 12 期變化，前面那個迴圈沒有算，
  它們不是重複的，不能一起刪掉。
  作法：改成 `f["d_gross_margin"] = d12(gm)` 之類的直接寫法。
- 或者保留擴充區塊、改成刪掉迴圈裡的 `{pn}_to_rev`——但這會動到
  `pretax_to_rev`，牽連較廣，不建議。

⚠️ 改完之後 `dfs_candidates.csv` 需要重跑，`alpha_mining_agent` 那邊的
`data/dfs_snapshot.parquet` 快照也要重新匯入。

### 加一道防呆

`generate()` 回傳前，加一個「同名/同值檢查」，例如：

```python
# 兩兩比較太慢的話，用「每欄的雜湊」就夠抓完全重複
import hashlib
sig = {}
for c in names:
    h = hashlib.md5(df[c].fillna(-9e99).values.tobytes()).hexdigest()
    if h in sig:
        raise ValueError(f"因子重複：{c} 與 {sig[h]} 的值完全相同")
    sig[h] = c
```

---

## B. 高度重合但公式不同（4 對，非 bug，但需留意）

這些**不是程式錯誤**——公式確實不一樣，只是在台股樣本上實證高度重合。
不需要修程式，但在做**等權合成**或解讀特徵重要性時要意識到它們不是獨立訊號。

| 因子 A | 因子 B | ρ | 為什麼會這麼像 |
|---|---|---|---|
| `roa`（淨利/總資產） | `roe`（淨利/股東權益） | 0.952 | 分母差在槓桿；台股同產業內公司的負債比相近，橫斷面排名幾乎同序 |
| `d_roa` | `d_roe` | 0.958 | 同上，取 12 期變化後仍然同序 |
| `pretax_to_px`（稅前/股價） | `ni_to_px`（稅後/股價） | 0.984 | 分子差在所得稅；台股有效稅率變異小 |
| `pretax_to_rev`（稅前/營收） | `ni_to_rev`（稅後/營收） | 0.988 | 同上 |

### 建議

- **不要刪**。它們在某些情境（例如金融業 vs 製造業的槓桿差異）確實會分開，
  刪掉會失去資訊。
- 但在**等權合成**時應該擇一，或把每一組視為一個概念先取平均再進合成。
- `ml_model.py` 的 SHAP / 特徵重要性報告，這幾對應該**合併呈現**，
  否則「淨利率」這個概念的真實重要性會被低估。

---

## C. 下游專案已經做的處理（供參考，本專案不需跟進）

`alpha_mining_agent/src/seed_reference.py` 新增了第四道篩選
`--max-ref-corr`（預設 0.95）：匯入參考因子時做兩兩去重，
口徑是「逐月橫斷面排名相關」，按 `|ICIR_train|` 由大到小保留。

實測 31 → 25 個，去重後合成的資訊比率（IR）從 1.663 升到 1.741（Ridge）。

那只是下游的補救；**根因在本專案的 `mine_dfs.py`，還是應該修**。

---

## 處理順序建議

- [ ] 1. 修 `mine_dfs.py` 的三對重複（A 節），保留 `d_*` 衍生因子
- [ ] 2. 加同值防呆檢查
- [ ] 3. 重跑 `mine_dfs.py` → 更新 `dfs_candidates.csv`
- [ ] 4. 重跑 `src/ml/ml_model.py`，確認「等權 baseline」的數字有變
      （若沒變就代表某個環節沒吃到修正）
- [ ] 5. 檢查 `mine_gp.py` 是否共用同一份因子池
- [ ] 6. 通知 `alpha_mining_agent` 重新匯入參考因子快照
- [ ] 7. （選配）B 節那四對，在 ml_model 的重要性報告裡合併呈現
