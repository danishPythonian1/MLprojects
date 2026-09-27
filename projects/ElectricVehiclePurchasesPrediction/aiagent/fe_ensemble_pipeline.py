# %% [markdown]
# EV purchases (Playground S6E9) — artifact features + nested target encoding + blend
#
# 1. Income digit features
# 2. Frequency (+ optional lift vs original dataset) of income & commute values
# 3. Nested target encoding of income / commute at several bin widths
# 4. Target encoding of every raw column at two smoothing levels
# 5. LightGBM + CatBoost + MLP, blended with greedy hill-climbing on OOF
#
# All target encodings are computed inside each outer fold (inner K-fold for the
# training rows), so OOF AUC is an honest estimate. Test predictions = mean of fold models.

# %% Config
import os, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

TRAIN_PATH = "train_cleaned.csv"
TEST_PATH  = "test.csv"
ORIG_PATH  = "ev_adoption.csv"   # original dataset (kaggle: itzzomkar/ev-adoption-behavior-and-range-anxiety); skipped if missing
OUT_DIR    = "preds"
TARGET     = "Will_Buy_EV"
SEED, N_FOLDS, INNER_FOLDS = 42, 5, 5

# feature groups on/off (for ablation)
USE_DIGITS, USE_FREQ, USE_NESTED_TE, USE_ALL_TE = True, True, True, True
MODELS = ["lgb", "cat", "mlp"]    # drop "mlp" to save time
BASELINE_OOF = 0.94218            # your CatBoost OOF on the 17-feature set

CAT_COLS = ["Gender", "City_Type", "Current_Car_Type",
            "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
os.makedirs(OUT_DIR, exist_ok=True)

# %% Load
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
for df in (train, test):
    for c in CAT_COLS:
        df[c] = df[c].astype(str).str.strip()

y = (train.pop(TARGET) == "Yes").astype(int).to_numpy()
test_ids = test["id"].to_numpy()
full = pd.concat([train.drop(columns="id"), test.drop(columns="id")], ignore_index=True)
RAW_COLS = list(full.columns)
ntr, nte = len(train), len(test)
tr_rows, te_rows = np.arange(ntr), np.arange(ntr, ntr + nte)
print(f"train {ntr} | test {nte} | pos rate {y.mean():.4f}")

# base features: categoricals -> int codes (fitted on train+test, no target involved)
X_base = full.copy()
for c in CAT_COLS:
    X_base[c] = pd.factorize(X_base[c])[0]

inc = full["Annual_Income_USD"].round().astype(np.int64)
com = (full["Daily_Commute_km"] * 10).round().astype(np.int64)   # commute in 0.1 km units

# %% 1. Income digit features
if USE_DIGITS:
    for i in range(6):   # ones, tens, hundreds, thousands, ten-thousands, hundred-thousands
        X_base[f"inc_d{i}"] = ((inc // 10**i) % 10).astype(np.int8)

# %% 2. Frequency + lift features (target-free -> computed on train+test together)
if USE_FREQ:
    for name, s in [("inc", inc), ("com", com)]:
        X_base[f"{name}_freq"] = s.map(s.value_counts()).astype(np.float32)

    if os.path.exists(ORIG_PATH):
        orig = pd.read_csv(ORIG_PATH)
        for name, col in [("inc", "Annual_Income_USD"), ("com", "Daily_Commute_km")]:
            if col not in orig.columns:
                continue
            o = orig[col].round().astype(np.int64) if name == "inc" else (orig[col] * 10).round().astype(np.int64)
            s = inc if name == "inc" else com
            p_here = s.map(s.value_counts(normalize=True))
            p_orig = s.map(o.value_counts(normalize=True)).fillna(0.0)
            X_base[f"{name}_lift"] = np.log((p_here + 1e-6) / (p_orig + 1e-6)).astype(np.float32)
            X_base[f"{name}_in_orig"] = (p_orig > 0).astype(np.int8)
        print("lift features added from", ORIG_PATH)
    else:
        print(f"[info] {ORIG_PATH} not found -> lift features skipped")

print("base feature count:", X_base.shape[1])

# %% 3 & 4. Target-encoding specs: name -> (int codes over train+test, n_codes, smoothing m)
te_specs = {}

def add_spec(name, key, m):
    codes, uniq = pd.factorize(key)
    te_specs[name] = (codes.astype(np.int64), len(uniq), m)

if USE_NESTED_TE:
    # 3. income / commute treated as categories at several bin widths
    for w in [1, 100, 1000, 5000]:
        add_spec(f"te_inc_b{w}", inc // w, m=20)
    for w in [1, 10, 50]:                      # 0.1 km, 1 km, 5 km
        add_spec(f"te_com_b{w}", com // w, m=20)

if USE_ALL_TE:
    # 4. every raw column, two smoothing levels
    for c in RAW_COLS:
        for m in (10, 300):
            add_spec(f"te_{c}_m{m}", full[c], m=m)

def te_map(codes_fit, y_fit, codes_apply, n, m):
    """Smoothed mean target per code, fitted on (codes_fit, y_fit)."""
    prior = y_fit.mean()
    s = np.bincount(codes_fit, weights=y_fit, minlength=n)
    c = np.bincount(codes_fit, minlength=n)
    return ((s + prior * m) / (c + m))[codes_apply].astype(np.float32)

def build_te(fit_idx, apply_idx_list):
    """Nested TE: fit rows get inner-OOF encodings; apply rows get encodings fitted on all fit rows."""
    fit_out = {}
    apply_out = [{} for _ in apply_idx_list]
    y_fit = y[fit_idx]
    inner = list(KFold(INNER_FOLDS, shuffle=True, random_state=SEED).split(fit_idx))
    for name, (codes, n, m) in te_specs.items():
        c_fit = codes[fit_idx]
        col = np.empty(len(fit_idx), dtype=np.float32)
        for a, b in inner:
            col[b] = te_map(c_fit[a], y_fit[a], c_fit[b], n, m)
        fit_out[name] = col
        for k, idx in enumerate(apply_idx_list):
            apply_out[k][name] = te_map(c_fit, y_fit, codes[idx], n, m)
    return pd.DataFrame(fit_out), [pd.DataFrame(d) for d in apply_out]

def make_xy(fit_idx, apply_idx_list):
    base_fit = X_base.iloc[fit_idx].reset_index(drop=True)
    base_apply = [X_base.iloc[i].reset_index(drop=True) for i in apply_idx_list]
    if not te_specs:
        return base_fit, base_apply
    te_fit, te_apply = build_te(fit_idx, apply_idx_list)
    return (pd.concat([base_fit, te_fit], axis=1),
            [pd.concat([b, t], axis=1) for b, t in zip(base_apply, te_apply)])

# %% 5a. Models
def fit_lgb(Xtr, ytr, Xva, yva):
    m = lgb.LGBMClassifier(
        objective="binary", n_estimators=5000, learning_rate=0.03,
        num_leaves=31, min_child_samples=100, colsample_bytree=0.7,
        subsample=0.8, subsample_freq=1, reg_lambda=5.0,
        random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m, m.best_iteration_

def fit_cat(Xtr, ytr, Xva, yva):
    m = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", iterations=5000,
        early_stopping_rounds=200, bootstrap_type="Bernoulli",
        learning_rate=0.06485, depth=3, l2_leaf_reg=29.6, random_strength=2.017,
        subsample=0.988, rsm=0.842, border_count=254, one_hot_max_size=4,
        cat_features=CAT_COLS, random_seed=SEED, verbose=0, thread_count=-1)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    return m, m.get_best_iteration()

def fit_mlp(Xtr, ytr, Xva, yva):
    num_cols = [c for c in Xtr.columns if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                    subsample=200_000, random_state=SEED), num_cols)])
    m = make_pipeline(pre, MLPClassifier(
        hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=2048,
        learning_rate_init=1e-3, max_iter=40, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=5, random_state=SEED))
    m.fit(Xtr, ytr)
    return m, m[-1].n_iter_

FITTERS = {"lgb": fit_lgb, "cat": fit_cat, "mlp": fit_mlp}

# %% 5b. Outer CV loop
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)   # same folds as your notebooks
oof = {k: np.zeros(ntr) for k in MODELS}
tst = {k: np.zeros(nte) for k in MODELS}
fold_auc = {k: [] for k in MODELS}

for f, (tr, va) in enumerate(skf.split(tr_rows, y), 1):
    t0 = time.time()
    Xtr, (Xva, Xte) = make_xy(tr, [va, te_rows])
    print(f"\nfold {f}: {Xtr.shape[1]} features, built in {time.time()-t0:.0f}s")
    for k in MODELS:
        t1 = time.time()
        model, it = FITTERS[k](Xtr, y[tr], Xva, y[va])
        oof[k][va] = model.predict_proba(Xva)[:, 1]
        tst[k] += model.predict_proba(Xte)[:, 1] / N_FOLDS
        fold_auc[k].append(roc_auc_score(y[va], oof[k][va]))
        print(f"  {k}: AUC={fold_auc[k][-1]:.5f}  iters={it}  ({time.time()-t1:.0f}s)", flush=True)

for k in MODELS:
    np.save(f"{OUT_DIR}/oof_{k}.npy", oof[k]); np.save(f"{OUT_DIR}/test_{k}.npy", tst[k])
    s = roc_auc_score(y, oof[k])
    print(f"{k}: mean fold {np.mean(fold_auc[k]):.5f} ± {np.std(fold_auc[k]):.5f} | "
          f"OOF {s:.5f} | vs baseline {s - BASELINE_OOF:+.5f}")
    pd.DataFrame({"id": test_ids, TARGET: tst[k]}).to_csv(f"submission_{k}.csv", index=False)

# %% 5c. Greedy hill-climbing blend (Caruana, with replacement) on rank-normalised OOF
def to_rank(p):
    return rankdata(p) / len(p)

def hill_climb(oofs, y, max_steps=100):
    names = list(oofs)
    P = np.column_stack([to_rank(oofs[n]) for n in names])
    counts = np.zeros(len(names))
    blend, best = np.zeros(len(y)), -1.0
    for step in range(max_steps):
        cand = [(blend * step + P[:, j]) / (step + 1) for j in range(len(names))]
        scores = [roc_auc_score(y, c) for c in cand]
        j = int(np.argmax(scores))
        if scores[j] <= best + 1e-7:
            break
        best, blend = scores[j], cand[j]
        counts[j] += 1
    return {n: float(w) for n, w in zip(names, counts / counts.sum())}, best

weights, blend_auc = hill_climb(oof, y)
print("\nblend weights:", {k: round(v, 3) for k, v in weights.items()})
print(f"blend OOF AUC: {blend_auc:.5f} | vs baseline {blend_auc - BASELINE_OOF:+.5f}")

test_blend = sum(w * to_rank(tst[k]) for k, w in weights.items())
pd.DataFrame({"id": test_ids, TARGET: test_blend}).to_csv("submission_blend.csv", index=False)
print("saved submission_blend.csv (+ submission_<model>.csv per model)")
