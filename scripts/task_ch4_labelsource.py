"""EXECUTION_PLAN 2.3 (ERIN side): does label source change what the model learns?

Same ABMIL classifier, same folds, three label sources for TRAINING:
  jury (adopted), pathladder-only, feasibility-grader-only.
Evaluation is always against the adopted jury labels (with unsure held out), so
the comparison isolates training-label quality. Endpoint: NDBE vs LGD+.
"""
import json, os, sys
import h5py, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_clf import train_abmil_clf_fold, bootstrap_auc, patient_folds

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1]
POS = {"LGD", "HGD", "CANCER"}

m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
lab = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str)
lab = lab.merge(pd.read_csv(T + "/labeller/erin_consensus_labels.csv", dtype=str)[["CaseName", "plad", "feas"]],
                on="CaseName", how="left")
m = m.merge(lab[["CaseName", "plad", "feas"]], on="CaseName", how="left")

# eval set: jury train-eligible, primary grades only
ev = m[m["label_status"].isin(["train_eligible", "adjudicated"])
       & m["final_label"].isin(["NDBE", "LGD", "HGD", "CANCER"])].copy()
bags = {}
for _, r in ev.iterrows():
    with h5py.File(r["h5"]) as h:
        bags[r["h5"]] = np.asarray(h["features"])
keys = sorted(bags)
y_eval = {r["h5"]: int(r["final_label"] in POS) for _, r in ev.iterrows()}
pat = dict(zip(ev["h5"], ev["anon_id"]))
folds = patient_folds(keys, pat, y_eval, 5, seed=0)
print(f"eval slides={len(keys)}", flush=True)

sources = {"jury": dict(zip(ev["h5"], [int(l in POS) for l in ev["final_label"]])),
           "pathladder": {r["h5"]: int(r["plad"] in POS) for _, r in ev.iterrows() if pd.notna(r["plad"]) and r["plad"] != "None"},
           "feas_grader": {r["h5"]: int(r["feas"] in POS) for _, r in ev.iterrows() if pd.notna(r["feas"]) and r["feas"] != "None"}}

res = {"_meta": {"n": len(keys), "endpoint": "NDBE vs LGD+ (eval labels = jury)", "seeds": SEEDS}}
oof = {}
for src, ytr in sources.items():
    per = []
    for s in SEEDS:
        o = {}
        for i in range(5):
            te = folds[i]
            tr = [k for j, f in enumerate(folds) if j != i for k in f if k in ytr]
            o.update(train_abmil_clf_fold(bags, tr, te, {**y_eval, **ytr}, s))
        per.append(o); print(f"{src} seed{s} done", flush=True)
    oof[src] = {k: float(np.mean([p[k] for p in per])) for k in keys}
    ref = oof.get("jury") if src != "jury" else None
    res[src] = bootstrap_auc(oof[src], y_eval, prob_b=ref)
    print(src, res[src], flush=True)

# 1.15 cross-evaluation matrix: every trained model scored against every label
# source (restricted to slides that source labels), so no single source is both
# teacher and judge unchallenged.
res["cross_eval"] = {}
for esrc, ye in sources.items():
    ek = [k for k in keys if k in ye]
    for tsrc in oof:
        sub_o = {k: oof[tsrc][k] for k in ek}
        res["cross_eval"][f"train_{tsrc}_eval_{esrc}"] = bootstrap_auc(sub_o, ye)
        print(f"train={tsrc} eval={esrc} n={len(ek)}",
              res["cross_eval"][f"train_{tsrc}_eval_{esrc}"], flush=True)
np.savez(os.path.join(OUT, "oof_preds.npz"),
         **{s: np.array([oof[s][k] for k in keys]) for s in oof},
         keys=np.array(keys))
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
