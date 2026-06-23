"""
步驟 8（階段 B2）：GP 符號回歸因子挖掘（自製輕量 GP，零外部相依）。

為何自製：gplearn 與新版 scikit-learn 不相容且久未維護。自製 GP 可完全掌控
運算子、適應度、隨機種子，最穩定也最好除錯。

做法：以一組標準化的基礎運算元為輸入，用運算式樹(add/sub/mul/div/neg/abs/sqrt/log)
組合，適應度 = 訓練期「產業內 ICIR」− parsimony×公式大小（懲罰複雜度，抑制過擬合）。

解決 GP 隨機性：跑多個種子各自演化，到測試期驗證，只保留「跨種子穩定、
訓練+測試 ICIR 同向且夠強」的公式。

執行：python src/mine_gp.py
輸出：data/processed/gp_factors.csv
"""
import sys
import random
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_PROCESSED
from factor_eval import monthly_ic, summarize
from mine_dfs import build_monthly_base, generate, TRAIN_END

OPERANDS = ["eps_to_px", "gp_to_px", "op_to_rev", "g_eps_12",
            "neg_vol_63", "dist_high", "mom_12_1", "mom_1m",
            "rev_yoy", "liq_amt"]
SEEDS = [0, 1, 2, 3, 4]

# GP 超參數（warm-start 版：放寬複雜度懲罰、加長世代）
POP, GENS, TOURN = 400, 20, 15
P_CROSS, P_MUT = 0.7, 0.25
MAX_DEPTH = 5
PARSIMONY = 0.0008

BIN = ["add", "sub", "mul", "div"]
UN = ["neg", "abs", "sqrt", "log"]


# ---------- 運算式樹 ----------
def gen_tree(rng, n_op, depth, full):
    if depth == 0 or (not full and rng.random() < 0.3):
        if rng.random() < 0.8:
            return ("var", rng.randrange(n_op))
        return ("const", round(rng.uniform(-1, 1), 3))
    if rng.random() < 0.7:
        op = rng.choice(BIN)
        return ("b", op, gen_tree(rng, n_op, depth - 1, full),
                gen_tree(rng, n_op, depth - 1, full))
    op = rng.choice(UN)
    return ("u", op, gen_tree(rng, n_op, depth - 1, full))


def ev(node, X):
    t = node[0]
    if t == "var":
        return X[:, node[1]]
    if t == "const":
        return np.full(X.shape[0], node[1])
    if t == "u":
        a = ev(node[2], X)
        op = node[1]
        if op == "neg":
            return -a
        if op == "abs":
            return np.abs(a)
        if op == "sqrt":
            return np.sqrt(np.abs(a))
        if op == "log":
            return np.log(np.abs(a) + 1e-9)
    a = ev(node[2], X)
    b = ev(node[3], X)
    op = node[1]
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return np.where(np.abs(b) > 1e-9, a / b, 1.0)


def size(node):
    if node[0] in ("var", "const"):
        return 1
    if node[0] == "u":
        return 1 + size(node[2])
    return 1 + size(node[2]) + size(node[3])


def all_nodes(node, path=()):
    yield path, node
    if node[0] == "u":
        yield from all_nodes(node[2], path + (2,))
    elif node[0] == "b":
        yield from all_nodes(node[2], path + (2,))
        yield from all_nodes(node[3], path + (3,))


def get_at(node, path):
    for p in path:
        node = node[p]
    return node


def replace_at(node, path, new):
    if not path:
        return new
    node = list(node)
    node[path[0]] = replace_at(node[path[0]], path[1:], new)
    return tuple(node)


def crossover(rng, a, b):
    pa = rng.choice([p for p, _ in all_nodes(a)])
    pb = rng.choice([p for p, _ in all_nodes(b)])
    return replace_at(a, pa, get_at(b, pb))


def mutate(rng, a, n_op):
    pa = rng.choice([p for p, _ in all_nodes(a)])
    return replace_at(a, pa, gen_tree(rng, n_op, rng.randint(1, 3), False))


def render(node, names):
    if node[0] == "var":
        return names[node[1]]
    if node[0] == "const":
        return str(node[1])
    if node[0] == "u":
        return f"{node[1]}({render(node[2], names)})"
    return f"({render(node[2], names)} {node[1]} {render(node[3], names)})"


# ---------- 適應度（訓練期產業內 ICIR）----------
_CELL = _YMc = _RR = None   # 預先算好的訓練期分組與報酬排名


def setup_fitness(train):
    global _CELL, _YMc, _RR
    ymc = train["ym"].astype("category").cat.codes.values
    gc = train["group"].astype("category").cat.codes.values
    _CELL = ymc * 100 + gc            # (月,產業) 細格
    _YMc = ymc
    rr = pd.Series(train["fwd_ret_1m"].values).groupby(_CELL).rank()
    _RR = rr.values


def fitness(f):
    if not np.all(np.isfinite(f)):
        f = np.where(np.isfinite(f), f, np.nan)
    s = pd.Series(f)
    if s.notna().sum() < 200 or s.nunique() < 5:
        return -9.0
    rf = s.groupby(_CELL).rank().values
    d = pd.DataFrame({"ym": _YMc, "rf": rf, "rr": _RR}).dropna()
    ics = d.groupby("ym").apply(
        lambda x: x["rf"].corr(x["rr"]) if x["rf"].nunique() > 1 else np.nan).dropna()
    if len(ics) < 12:
        return -9.0
    std = ics.std()
    return abs(ics.mean() / std) if std > 0 else -9.0


def warm_init(rng, n_op):
    """warm-start 初始族群：先放單因子與簡單組合，保證起跑點不低於 DFS。"""
    pop = []
    for i in range(n_op):            # 每個單因子放兩份，增加存活機會
        pop += [("var", i), ("var", i)]
    for _ in range(n_op * 4):        # 隨機兩因子組合
        pop.append(("b", rng.choice(BIN),
                    ("var", rng.randrange(n_op)), ("var", rng.randrange(n_op))))
    for _ in range(n_op * 2):        # 單因子的一元轉換
        pop.append(("u", rng.choice(UN), ("var", rng.randrange(n_op))))
    while len(pop) < POP:            # 其餘用隨機樹補滿
        pop.append(gen_tree(rng, n_op, rng.randint(2, MAX_DEPTH), rng.random() < 0.5))
    return pop[:POP]


def run_gp(rng, X, n_op):
    pop = warm_init(rng, n_op)

    def score(tree):
        return fitness(ev(tree, X)) - PARSIMONY * size(tree)

    scores = [score(t) for t in pop]
    for _ in range(GENS):
        new = []
        # 菁英保留
        best_i = int(np.argmax(scores))
        new.append(pop[best_i])
        while len(new) < POP:
            a = pop[max(rng.sample(range(POP), TOURN), key=lambda i: scores[i])]
            r = rng.random()
            if r < P_CROSS:
                b = pop[max(rng.sample(range(POP), TOURN), key=lambda i: scores[i])]
                child = crossover(rng, a, b)
            elif r < P_CROSS + P_MUT:
                child = mutate(rng, a, n_op)
            else:
                child = a
            if size(child) <= 25:
                new.append(child)
        pop = new
        scores = [score(t) for t in pop]
    best_i = int(np.argmax(scores))
    return pop[best_i]


def eval_tree_icir(tree, Xall, meta):
    d = meta.copy()
    d["gpf"] = ev(tree, Xall)
    tr = d[d["ym"] <= TRAIN_END]
    te = d[d["ym"] > TRAIN_END]
    ic_tr = monthly_ic(tr, "gpf")
    ic_te = monthly_ic(te, "gpf")
    icir_tr = summarize(ic_tr)[2] if len(ic_tr) >= 12 else np.nan
    icir_te = summarize(ic_te)[2] if len(ic_te) >= 6 else np.nan
    return icir_tr, icir_te


def zscore_by_month(df, cols):
    out = df.copy()
    for c in cols:
        g = out.groupby("ym")[c]
        out[c] = ((out[c] - g.transform("mean")) / g.transform("std")).fillna(0.0)
    return out


def main():
    p = DATA_PROCESSED / "panel.parquet"
    if not p.exists():
        sys.exit("找不到 panel.parquet，請先執行 align_data.py")
    print("建立基礎欄位 + 候選因子…")
    m = build_monthly_base(pd.read_parquet(p))
    df, _ = generate(m)
    ops = [c for c in OPERANDS if c in df.columns]
    print(f"GP 運算元（{len(ops)}）：{ops}")

    df = zscore_by_month(df, ops)
    df = df.dropna(subset=["fwd_ret_1m"]).reset_index(drop=True)
    train = df[df["ym"] <= TRAIN_END].reset_index(drop=True)
    setup_fitness(train)

    Xtr = train[ops].values
    Xall = df[ops].values
    meta = df[["ym", "group", "fwd_ret_1m"]].copy()

    # 基準：最佳單一運算元（GP 至少要贏過它才有意義）
    print("\n--- 單因子基準（GP 要超越這個）---")
    base = []
    for i, c in enumerate(ops):
        tr_i, te_i = eval_tree_icir(("var", i), Xall, meta)
        base.append((c, tr_i, te_i))
    base.sort(key=lambda x: abs(x[1]) if pd.notna(x[1]) else 0, reverse=True)
    for c, tr_i, te_i in base[:3]:
        print(f"  {c:12} ICIR_train={tr_i:.3f}  ICIR_test={te_i:.3f}")
    best_base = base[0]

    rows = []
    for seed in SEEDS:
        rng = random.Random(seed)
        np.random.seed(seed)
        print(f"\n=== GP 種子 {seed} 演化中（pop={POP}, gen={GENS}）… ===")
        best = run_gp(rng, Xtr, len(ops))
        icir_tr, icir_te = eval_tree_icir(best, Xall, meta)
        expr = render(best, ops)
        rows.append({"seed": seed, "ICIR_train": icir_tr,
                     "ICIR_test": icir_te, "size": size(best), "program": expr})
        print(f"  ICIR_train={icir_tr:.3f}  ICIR_test={icir_te:.3f}")
        print(f"  公式：{expr}")

    res = pd.DataFrame(rows)
    out = DATA_PROCESSED / "gp_factors.csv"
    res.to_csv(out, index=False)

    stable = res[(np.sign(res["ICIR_train"]) == np.sign(res["ICIR_test"])) &
                 (res["ICIR_test"].abs() > 0.2)]
    print("\n=== 多種子穩定性 ===")
    print(f"{len(SEEDS)} 個種子中，訓練+測試一致且 |ICIR_test|>0.2 的：{len(stable)} 個")
    print(res[["seed", "ICIR_train", "ICIR_test", "size"]].to_string(index=False,
          formatters={"ICIR_train": "{:.3f}".format, "ICIR_test": "{:.3f}".format}))
    print(f"\n最佳單因子基準：{best_base[0]} "
          f"(train={best_base[1]:.3f}, test={best_base[2]:.3f})")
    gp_best_test = res.loc[res["ICIR_test"].abs().idxmax()]
    print(f"GP 最佳(依|測試|)：種子{int(gp_best_test['seed'])} "
          f"train={gp_best_test['ICIR_train']:.3f} test={gp_best_test['ICIR_test']:.3f}")
    print(f"\n完整結果（含公式）已存：{out}")
    if len(stable) == 0:
        print("\n→ 跨種子不穩定，代表 GP 在現有運算元上找不到勝過 DFS 的穩健新因子，"
              "建議直接用 DFS survivors 進正交化 + 回測。")


if __name__ == "__main__":
    main()
