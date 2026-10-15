"""2.27: cross-foundation-model disagreement as an uncertainty biomarker.

Five encoders' tile features exist for every ERIN slide. Pool each, train
identical PCA64+logistic grade probes per encoder on SHARED folds, and define
per-slide disagreement = std of the five OOF probabilities. Readouts:
Spearman vs jury entropy; AUC for flagging unsure-holdout membership
(the stratum where model confidence failed, results/unsure_scoring.json);
IND enrichment; progression association.
"""
import json, os, sys
import h5py, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
ENCODERS = os.environ.get(
    "ENCODERS",
    "features_uni_v2,features_virchow2,features_gigapath,features_phikon2,features_hoptimus0").split(",")
POS = {"LGD", "HGD", "CANCER"}

m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
lab = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv",
                  dtype=str)[["CaseName", "jury_entropy", "jury_label"]].drop_duplicates("CaseName")
m = m.merge(lab, on="CaseName", how="left", suffixes=("", "_jl"))
if "jury_entropy" not in m.columns and "jury_entropy_jl" in m.columns:
    m["jury_entropy"] = m["jury_entropy_jl"]

pooled = {}
for enc in ENCODERS:
    cache = os.path.join(OUT, f"pooled_{enc}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        pooled[enc] = dict(zip(d["h5s"], d["X"]))
        print(enc, "cached", len(pooled[enc]), flush=True); continue
    P = {}
    for i, h5p in enumerate(m["h5"]):
        path = h5p.replace("features_uni_v2", enc)
        try:
            with h5py.File(path) as h:
                P[h5p] = np.asarray(h["features"]).mean(0)
        except Exception:
            pass
        if i % 400 == 0: print(enc, i, flush=True)
    if len(P) < 0.9 * len(m):
        print(f"SKIP {enc}: only {len(P)}/{len(m)} slides found", flush=True); continue
    np.savez(cache, h5s=np.array(list(P)), X=np.stack(list(P.values())))
    pooled[enc] = P
    print(enc, "pooled", len(P), flush=True)
assert len(pooled) >= 4, f"need >=4 encoders, got {list(pooled)}"

common = sorted(set.intersection(*(set(v) for v in pooled.values())))
mm = m.set_index("h5").loc[common]
elig = mm["label_status"].isin(["train_eligible", "adjudicated"]) & mm["final_label"].isin(
    ["NDBE", "LGD", "HGD", "CANCER"])
unsure = mm["label_status"] == "unsure_held_out"
y = mm["final_label"].isin(POS).astype(int).values
pats = mm["anon_id"].values
print(f"common slides={len(common)} eligible={int(elig.sum())} unsure={int(unsure.sum())}", flush=True)

uniq = sorted(set(pats[elig.values]))
rng = np.random.RandomState(0)
fold_of = {a: i % 5 for i, a in enumerate(rng.permutation(uniq))}

probs = {}
for enc in pooled:
    Xe = np.stack([pooled[enc][h] for h in common])
    pr = np.full(len(common), np.nan)
    ei = np.where(elig.values)[0]
    fm = np.array([fold_of[pats[i]] for i in ei])
    for f in range(5):
        tr = ei[fm != f]
        pl = Pipeline([("s", StandardScaler()), ("p", PCA(64)),
                       ("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000))])
        pl.fit(Xe[tr], y[tr])
        te = ei[fm == f]
        pr[te] = pl.predict_proba(Xe[te])[:, 1]
        ui = np.where(unsure.values)[0]
        if len(ui):  # unsure slides scored by every fold model, averaged
            add = pl.predict_proba(Xe[ui])[:, 1] / 5
            pr[ui] = np.where(np.isnan(pr[ui]), 0, pr[ui]) + add
    probs[enc] = pr
    ok = ~np.isnan(pr[elig.values])
    print(enc, "probe AUC:", round(roc_auc_score(y[elig.values][ok], pr[elig.values][ok]), 4), flush=True)

P = np.stack([probs[e] for e in probs])
disagree = np.nanstd(P, axis=0)
mean_p = np.nanmean(P, axis=0)

je = pd.to_numeric(mm["jury_entropy"], errors="coerce").values
ok = ~np.isnan(je) & ~np.isnan(disagree)
rho, pv = spearmanr(disagree[ok], je[ok])
scored = ~np.isnan(disagree)
u = unsure.values
res = {"_meta": {"encoders": list(probs), "n_slides": len(common),
                 "n_eligible": int(elig.sum()), "n_unsure": int(u.sum())},
       "disagreement_vs_jury_entropy": {"spearman": round(float(rho), 4),
                                        "p": float(pv), "n": int(ok.sum())},
       "flag_unsure_auc_disagreement": round(float(
           roc_auc_score(u[scored], disagree[scored])), 4),
       "flag_unsure_auc_confidence": round(float(
           roc_auc_score(u[scored], -np.abs(mean_p[scored] - 0.5))), 4),
       "ind_enrichment": {
           "disagree_mean_IND": round(float(np.nanmean(disagree[(mm["final_label"] == "IND").values])), 4),
           "disagree_mean_other": round(float(np.nanmean(disagree[(mm["final_label"] != "IND").values])), 4)}}
prog = pd.read_csv(T + "/labeller/erin_progression_cohort_v3.csv", dtype=str)
pmap = dict(zip(prog["anon_id"], (prog["progressed_to_HGDplus"] == "True").astype(int)))
pi = [i for i, a in enumerate(pats) if a in pmap and not np.isnan(disagree[i])]
if len(pi) > 50:
    dd = pd.DataFrame({"a": pats[pi], "d": disagree[pi]}).groupby("a")["d"].max()
    yy = np.array([pmap[a] for a in dd.index])
    res["progression_auc_of_disagreement"] = round(float(roc_auc_score(yy, dd.values)), 4) \
        if 0 < yy.sum() < len(yy) else None
np.savez(os.path.join(OUT, "disagreement_scores.npz"),
         h5s=np.array(common), disagree=disagree, mean_p=mean_p)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
