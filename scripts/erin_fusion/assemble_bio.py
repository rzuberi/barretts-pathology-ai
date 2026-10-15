"""Item 30a: assemble the biopsy-only T3 re-run (T3a_bio, T3b_bio) exactly as assemble.py does for T3a/T3b, and
compare with the original all-specimen result. Run when all 30 img units + 2 tab tasks have results."""
import glob, json, os, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; Q = f"{T}/feasibility/erin_fusion/queue"; TD = f"{T}/feasibility/erin_fusion/tasks"; NB = 2000
sys.path.insert(0, T + "/scripts"); from abmil_clf import patient_folds
orig = json.load(open(f"{T}/results/erin_progression_fusion/results.json"))["tasks"]
res = {}
for t in ("T3a_bio", "T3b_bio"):
    d = pd.read_csv(f"{TD}/{t}.csv", dtype=str); keys = list(d.sample_id); pats = list(d.anon_id); y = d.y.astype(int).values; idx = {k: i for i, k in enumerate(keys)}
    folds = patient_folds(keys, dict(zip(keys, pats)), dict(zip(keys, y)), 5, seed=0)
    files = sorted(glob.glob(f"{Q}/results/img_{t}_f*_s*.json")); tab = f"{Q}/results/tab_{t}.json"
    if len(files) < 15 or not os.path.exists(tab): print(t, "incomplete:", len(files), "img units,", "tab" if os.path.exists(tab) else "no tab"); continue
    c = np.zeros(len(keys))
    for f in range(5):
        rs = [json.load(open(x)) for x in files if f"_f{f}_" in x]
        for k in folds[f]: c[idx[k]] = np.mean([r["preds"][k] for r in rs])
    a = np.array([json.load(open(tab))["arms"]["a"][k] for k in keys])
    g = np.array(pats); up = np.unique(g); idx_of = {u: np.where(g == u)[0] for u in up}; rng = np.random.RandomState(0); B = []
    while len(B) < NB:
        s = np.concatenate([idx_of[u] for u in rng.choice(up, len(up))])
        if len(set(y[s])) < 2: continue
        B.append(s)
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    o = orig[t.replace("_bio", "")]
    res[t] = {"n": int(len(y)), "pos": int(y.sum()), "patients": int(len(up)), "image_auroc": round(float(roc_auc_score(y, c)), 4), "image_ci": ci([roc_auc_score(y[s], c[s]) for s in B]),
              "baseline_auroc": round(float(roc_auc_score(y, a)), 4), "delta_image_minus_baseline": round(float(roc_auc_score(y, c) - roc_auc_score(y, a)), 4), "delta_ci": ci([roc_auc_score(y[s], c[s]) - roc_auc_score(y[s], a[s]) for s in B]),
              "original_all_specimens": {"n": o["n"], "pos": o["pos"], "image_auroc": o["arms"]["c"]["auroc"], "baseline_auroc": o["arms"]["a"]["auroc"], "delta": o["deltas"]["image - baseline"]}}
    print(t, json.dumps(res[t]))
if res:
    os.makedirs(f"{T}/results/numbers", exist_ok=True); json.dump(res, open(f"{T}/results/numbers/erin_t3_biopsy_only.json", "w"), indent=1); print("wrote results/numbers/erin_t3_biopsy_only.json")
