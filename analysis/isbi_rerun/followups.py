"""Follow-ups 1-3 (read-only on preds): T2 identity check, per-class OvR AUROC (A/B/C + B-A per class, clustered CIs),
seed stability (B-A per seed with clustered CIs), transposed T2 confusion matrices. -> followups.json"""
import glob, json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun"; CLS = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLS)}; NB = 2000
s = pd.read_csv(A + "/slides.csv", dtype={"idx": int}).fillna(""); s["vote_frac"] = pd.to_numeric(s.vote_frac, errors="coerce")
excl = set(pd.read_csv(A + "/qc/exclusions.csv").idx); s = s[~s.idx.isin(excl)].set_index("idx")
def load(hold, arm, task, seed=None):
    te, ho = {}, {}
    for f in glob.glob(A + f"/preds/{hold}_{arm}_{task}_seed{'*' if seed is None else seed}_fold*.npz"):
        z = np.load(f)
        for i, p in zip(z["test_idx"], z["test_prob"]): te.setdefault(int(i), []).append(p)
        for i, p in zip(z["holdout_idx"], z["holdout_prob"]): ho.setdefault(int(i), []).append(p)
    return {k: np.mean(v, 0) for k, v in te.items()}, {k: np.mean(v, 0) for k, v in ho.items()}
def truth(ids, task, col):
    g = s.loc[ids, col]
    if task == "bin": return g.isin(["NORMAL_OTHER", "NDBE", "LGD", "HGD", "CANCER"]).values, g.isin(["LGD", "HGD", "CANCER"]).astype(int).values
    return g.isin(CLS).values, g.map(C_OF).fillna(-1).astype(int).values
def sets(hold):
    fcol = f"fold_{hold}"; rng0 = np.random.RandomState(1); hp = sorted(s[s.human_patient].anon_id.unique()); fold_hp = {a: i % 5 for i, a in enumerate(rng0.permutation(hp))}
    fr = np.where(s.human_report, -1, np.where(s.fold >= 0, s.fold, s.anon_id.map(fold_hp))); f = fr if hold == "report" else s.fold.values
    test = s[f >= 0]; ho = s[(f == -1) & (s.human_grade != "")]
    return {"T1": test[(test.vote_frac >= 0.75) & (test.worst_grade != "")].index.tolist(), "T2": ho.index.tolist()}
def assemble(hold, task, T, seed=None):
    ids = sets(hold)[T]; src = 1 if T == "T2" else 0; preds = {a: load(hold, a, task, seed)[src] for a in "ABC"}
    common = [i for i in ids if all(i in preds[a] for a in "ABC")]; keep, y = truth(common, task, "human_grade" if T == "T2" else "worst_grade")
    common = np.array(common)[keep]; y = y[keep]; pats = s.loc[common, "anon_id"].values
    return y, pats, {a: np.stack([preds[a][i] for i in common]) for a in "ABC"}
def macro(P, y):
    v = [roc_auc_score((y == c).astype(int), P[:, c]) for c in range(P.shape[1]) if 0 < (y == c).sum() < len(y)]; return float(np.mean(v))
def pcls(P, y, c): yy = (y == c).astype(int); return roc_auc_score(yy, P[:, c]) if 0 < yy.sum() < len(yy) else np.nan
def ci(v): v = np.asarray(v, float); return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
def resample(pats, rng):
    up = np.unique(pats); groups = {p: np.where(pats == p)[0] for p in up}
    for _ in range(NB): yield np.concatenate([groups[p] for p in rng.choice(up, len(up), replace=True)])
out = {"t2_identity": {}, "per_class": {}, "seed_stability": {}, "t2_confusion_truth_rows_pred_cols": {}}
# 1. identity: one holdout set per (hold, task) across all 45 files; truth row sums equal across arms
for hold in ["patient", "report"]:
    for task in ["bin", "six"]:
        hs = {frozenset(np.load(f)["holdout_idx"].tolist()) for f in glob.glob(A + f"/preds/{hold}_*_{task}_seed*_fold*.npz")}
        y, pats, P = assemble(hold, task, "T2"); out["t2_identity"][f"{hold}_{task}"] = {"distinct_holdout_sets_across_45_files": len(hs), "n_T2_scored": int(len(y)), "same_truth_for_all_arms": True}
# transposed (correct) T2 confusion matrices
for hold in ["patient", "report"]:
    y, pats, P = assemble(hold, "six", "T2")
    out["t2_confusion_truth_rows_pred_cols"][hold] = {a: pd.crosstab(pd.Series(y).map(dict(enumerate(CLS))), pd.Series(P[a].argmax(1)).map(dict(enumerate(CLS)))).reindex(index=CLS, columns=CLS, fill_value=0).astype(int).to_dict("split") for a in "ABC"}
# 2. per-class OvR, patient holdout, T1 and T2
for T in ["T1", "T2"]:
    y, pats, P = assemble("patient", "six", T); rng = np.random.RandomState(0); res = {"n": int(len(y)), "n_patients": int(len(np.unique(pats))), "class_n": {CLS[c]: int((y == c).sum()) for c in range(6)}}
    pt = {a: [pcls(P[a], y, c) for c in range(6)] for a in "ABC"}; bs = {a: [[] for _ in range(6)] for a in "ABC"}
    for ix in resample(pats, rng):
        for a in "ABC":
            for c in range(6): bs[a][c].append(pcls(P[a][ix], y[ix], c))
    for c, name in enumerate(CLS):
        res[name] = {a: {"point": round(float(pt[a][c]), 4), "ci": ci(bs[a][c])} for a in "ABC"}
        res[name]["B_minus_A"] = {"point": round(float(pt["B"][c] - pt["A"][c]), 4), "ci": ci(np.array(bs["B"][c]) - np.array(bs["A"][c]))}
        res[name]["C_minus_A"] = {"point": round(float(pt["C"][c] - pt["A"][c]), 4), "ci": ci(np.array(bs["C"][c]) - np.array(bs["A"][c]))}
    out["per_class"][T] = res; print("per-class", T, "done", flush=True)
# 3. seed stability: B-A per seed (binary T1, six T1, six T2), patient holdout
for task, T in [("bin", "T1"), ("six", "T1"), ("six", "T2")]:
    key = f"{task}_{T}"; out["seed_stability"][key] = {}
    for seed in [0, 1, 2, None]:
        y, pats, P = assemble("patient", task, T, seed); m = (lambda p, yy: roc_auc_score(yy, p)) if task == "bin" else macro
        rng = np.random.RandomState(0); pa, pb = m(P["A"], y), m(P["B"], y); d = []
        for ix in resample(pats, rng): d.append(m(P["B"][ix], y[ix]) - m(P["A"][ix], y[ix]))
        out["seed_stability"][key]["seed_mean" if seed is None else f"seed{seed}"] = {"A": round(float(pa), 4), "B": round(float(pb), 4), "B_minus_A": {"point": round(float(pb - pa), 4), "ci": ci(d)}, "n": int(len(y))}
    print("seed", key, "done", flush=True)
json.dump(out, open(A + "/followups.json", "w"), indent=1); os.chmod(A + "/followups.json", 0o600); print("wrote followups.json")
