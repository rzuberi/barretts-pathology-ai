"""1.22: unsure-holdout sensitivity — score the 206 held-out 'unsure' reports'
slides through a model trained only on train-eligible jury labels.

The holdout was never trained on, so this is a pure deployment-population test.
Readouts: (a) prediction distribution on unsure vs train-eligible OOF,
(b) AUC on unsure using jury-plurality as a proxy reference (clearly labelled
proxy — R.4 pathologist grades will replace it), (c) calibration shift,
(d) abstention curve: performance on train-eligible when deferring the
top-k%-uncertain model predictions, the triage framing.
"""
import json, os, sys
import h5py, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_clf import train_abmil_clf_fold, bootstrap_auc, patient_folds

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]
POS = {"LGD", "HGD", "CANCER"}

m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
jl = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str)[["CaseName", "jury_label"]].drop_duplicates("CaseName")
if "jury_label" not in m.columns:
    m = m.merge(jl, on="CaseName", how="left")
tr_m = m[m["label_status"].isin(["train_eligible", "adjudicated"])
         & m["final_label"].isin(["NDBE", "LGD", "HGD", "CANCER"])]
un_m = m[m["label_status"] == "unsure_held_out"]
print(f"train-eligible slides={len(tr_m)} unsure slides={len(un_m)}", flush=True)

bags = {}
for h5p in list(tr_m["h5"]) + list(un_m["h5"]):
    try:
        with h5py.File(h5p) as h:
            bags[h5p] = np.asarray(h["features"])
    except Exception as e:
        print("skip", h5p, e)
tr_keys = [k for k in tr_m["h5"] if k in bags]
un_keys = [k for k in un_m["h5"] if k in bags]
y_tr = {r["h5"]: int(r["final_label"] in POS) for _, r in tr_m.iterrows() if r["h5"] in bags}
# proxy reference on unsure: jury plurality label (explicitly NOT truth)
y_un = {r["h5"]: int(str(r["jury_label"]) in POS) for _, r in un_m.iterrows()
        if r["h5"] in bags and pd.notna(r["jury_label"])}
pat = dict(zip(m["h5"], m["anon_id"]))

folds = patient_folds(tr_keys, pat, y_tr, 5, seed=0)
oof_tr, pred_un = {}, {k: [] for k in un_keys}
per = []
for s in SEEDS:
    o = {}
    for i in range(5):
        te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
        # each fold model also scores the whole unsure set; average later
        o.update(train_abmil_clf_fold(bags, tr, te + un_keys, {**y_tr, **{k: 0 for k in un_keys}}, s))
    per.append(o); print("seed", s, "done", flush=True)
for k in tr_keys:
    oof_tr[k] = float(np.mean([p[k] for p in per]))
for k in un_keys:
    pred_un[k] = float(np.mean([p[k] for p in per]))

p_tr = np.array([oof_tr[k] for k in tr_keys]); yt = np.array([y_tr[k] for k in tr_keys])
p_un = np.array([pred_un[k] for k in un_keys])
res = {"_meta": {"n_train_eligible": len(tr_keys), "n_unsure": len(un_keys),
                 "seeds": SEEDS, "unsure_reference": "jury plurality (PROXY, not truth)"},
       "train_eligible_oof": bootstrap_auc(oof_tr, y_tr),
       "pred_distribution": {
           "train_eligible": {"mean": round(float(p_tr.mean()), 3),
                              "frac_confident": round(float(np.mean((p_tr < 0.1) | (p_tr > 0.9))), 3)},
           "unsure": {"mean": round(float(p_un.mean()), 3),
                      "frac_confident": round(float(np.mean((p_un < 0.1) | (p_un > 0.9))), 3)}}}
if len(y_un) >= 30 and 0 < sum(y_un.values()) < len(y_un):
    res["unsure_vs_jury_proxy"] = bootstrap_auc({k: pred_un[k] for k in y_un}, y_un)
# abstention curve on train-eligible: defer the least-confident k%
curve = {}
conf = np.abs(p_tr - 0.5)
for frac in [0.0, 0.05, 0.1, 0.2]:
    keep = conf >= np.quantile(conf, frac)
    if len(set(yt[keep])) < 2: continue
    from sklearn.metrics import roc_auc_score
    curve[f"defer_{int(frac*100)}pct"] = {"n": int(keep.sum()),
                                          "auc": round(float(roc_auc_score(yt[keep], p_tr[keep])), 4)}
res["abstention_curve"] = curve
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
