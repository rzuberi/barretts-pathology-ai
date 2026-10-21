"""Exploration E3 (docs/paper_explore_E3.md): tile-score pooling of a fitted slide-mean elastic net. Each stored fit of ex_e3_refit.R (FEAT, outer and
inner, raw-scale coefficients c0 + w . x) scores every tile of its held-out samples (stored UNI2-h bags, killcoyne_mm/bags; union of the sample's slides);
the sample score is the max or the mean of the top-k tile logits (k = 8, 16, 32), mapped onto the slide-mean model's logit scale by an affine map fitted on the
fit's own training samples (mean and SD; ranks unchanged; without it max-tile probabilities saturate at 1). The mean of all tile logits equals the slide-mean model (check).
No refit: the lambda and coefficients are those chosen on the training rows. Writes killcoyne_mm/explore_E3/refit/<FEAT>_tp<pool>/{outer,inner}_rep_XX.csv
in the refit format. FEAT REP. Row-level, cluster only."""
import os, json, numpy as np, pandas as pd
M = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm"; E3 = M + "/explore_E3"; FEAT = os.environ.get("FEAT", "base"); REP = int(os.environ["REP"])
assert FEAT == "base", "stored bags are UNI2-h level 2 (the base block)"
RD = f"{E3}/refit/{FEAT}"; POOLS = ["max", "top8", "top16", "top32", "all"]
O = pd.read_csv(f"{RD}/outer_rep_{REP:02d}.csv", dtype={"Sample": str}); I = pd.read_csv(f"{RD}/inner_rep_{REP:02d}.csv", dtype={"Sample": str})
CF = pd.read_csv(f"{RD}/coef_rep_{REP:02d}.csv.gz"); W = CF[[c for c in CF.columns if c.startswith("w")]].values; c0 = CF.c0.values
key = {(int(r.rep), int(r.fold), int(r.inner_fold)): i for i, r in enumerate(CF.itertuples())}
bi = pd.read_csv(M + "/bag_index.csv", dtype=str).set_index("sample_id").npz_path
ALL = sorted(set(O.Sample)); bags = [np.load(bi[s_])["embeddings"].astype(np.float64) for s_ in ALL]; cut = np.cumsum([0] + [len(b) for b in bags]); pos = {s_: i for i, s_ in enumerate(ALL)}
Z = np.vstack(bags) @ W.T + c0[None, :]   # every tile x every fit: tile logits
def pooled_all(p):
    """(samples, fits) pooled tile logits"""
    out = np.zeros((len(ALL), Z.shape[1]))
    for i in range(len(ALL)):
        z = np.sort(Z[cut[i]:cut[i + 1]], 0)[::-1]
        out[i] = {"max": z[0], "top8": z[:8].mean(0), "top16": z[:16].mean(0), "top32": z[:32].mean(0), "all": z.mean(0)}[p]
    return out
PA = {p: pooled_all(p) for p in POOLS}; ofold = O.drop_duplicates("Sample").set_index("Sample").fold
sig = lambda x: 1 / (1 + np.exp(-x)); res = {p: ([], []) for p in POOLS}; chk = []; cal = {}
from functools import lru_cache
@lru_cache(maxsize=None)
def train_rows(rep, fold, inner):   # positions (in ALL) of the fit's training samples
    if inner == 0: return np.array([pos[s_] for s_ in ALL if ofold[s_] != fold])
    d = I[(I.fold == fold) & (I.inner_fold != inner)]; return np.array([pos[s_] for s_ in d.Sample])
def mapped(p, i, tr):
    """pooled logit mapped onto the slide-mean model's logit scale by the mean and SD over the fit's own training samples (training-only affine map; ranks unchanged)"""
    if (p, i) not in cal:
        a_, b_ = PA[p][tr, i], PA["all"][tr, i]; cal[(p, i)] = (b_.mean() - a_.mean() * b_.std() / a_.std(), b_.std() / a_.std())
    return cal[(p, i)]
for which, D in (("outer", O), ("inner", I)):
    for r in D.itertuples():
        inner = int(r.inner_fold) if which == "inner" else 0; i = key[(int(r.rep), int(r.fold), inner)]; tr = train_rows(int(r.rep), int(r.fold), inner); q = pos[r.Sample]
        chk.append(abs(sig(PA["all"][q, i]) - r.pred) / max(r.pred, 1e-12))
        for p in POOLS: a0, a1 = mapped(p, i, tr); res[p][0 if which == "outer" else 1].append(sig(a0 + a1 * PA[p][q, i]))
for p in POOLS:
    od = f"{RD}_tp{p}"; os.makedirs(od, exist_ok=True)
    o = O.copy(); o["pred"] = res[p][0]; o["pred_alt"] = np.nan; o.to_csv(f"{od}/outer_rep_{REP:02d}.csv", index=False)
    i_ = I.copy(); i_["pred"] = res[p][1]; i_["pred_alt"] = np.nan; i_.to_csv(f"{od}/inner_rep_{REP:02d}.csv", index=False)
print(json.dumps({"rep": REP, "mean_tile_logit_vs_slide_mean_model_max_rel_diff": float(np.max(chk))}), "\nEXE3 TILEPOOL DONE", REP, flush=True)
