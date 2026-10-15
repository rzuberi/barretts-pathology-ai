"""2.38: does section-resolved (slide-level) supervision beat case-max labels?

On the 1,538 dual-labelled slides (erin_slide_labels_v2.csv): train identical
ABMIL classifiers under (a) case-max labels and (b) slide-level labels, same
patient-disjoint folds, and evaluate BOTH against slide-level truth (and, for
symmetry, case-max truth). 32% of slides differ between label schemes — this
quantifies what that noise costs a model. Paired bootstrap on the primary
contrast (slide-truth evaluation).
"""
import json, os, sys
import h5py, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_clf import train_abmil_clf_fold, bootstrap_auc, patient_folds

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]
POS = {"LGD", "HGD", "CANCER"}

lab = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str)
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
lab = lab.merge(m[["h5", "anon_id"]], on="h5")
lab["y_slide"] = lab["worst_grade"].isin(POS).astype(int)
lab["y_case"] = lab["case_max"].isin(POS).astype(int) if "case_max" in lab.columns else None
if lab["y_case"].isna().all():
    raise SystemExit("case_max column missing")
print(f"slides {len(lab)}, disagree on binary target: "
      f"{(lab.y_slide != lab.y_case).mean():.3f}", flush=True)

bags = {}
for h5p in lab["h5"]:
    with h5py.File(h5p) as h:
        bags[h5p] = np.asarray(h["features"])
keys = sorted(bags)
lab = lab.set_index("h5").loc[keys]
pat = dict(zip(keys, lab["anon_id"]))
y_slide = dict(zip(keys, lab["y_slide"].astype(int)))
y_case = dict(zip(keys, lab["y_case"].astype(int)))
folds = patient_folds(keys, pat, y_slide, 5, seed=0)

# ---- UNIT mode: train exactly one (arm, seed, fold) and save its test preds ----
UNIT = os.environ.get("UNIT", "")
if UNIT:
    arm, seed, fold = UNIT.split("_")
    seed, fold = int(seed), int(fold)
    ydict = y_case if arm == "case" else y_slide
    te = folds[fold]; tr = [k for j, f in enumerate(folds) if j != fold for k in f]
    o = train_abmil_clf_fold(bags, tr, te, ydict, seed)
    ud = T + "/feasibility/svc_units"
    os.makedirs(ud, exist_ok=True)
    np.savez(os.path.join(ud, f"{arm}_{seed}_{fold}.npz"),
             keys=np.array(list(o.keys())), preds=np.array(list(o.values())))
    json.dump({"unit": UNIT, "n_test": len(o)}, open(os.path.join(OUT, "results.json"), "w"))
    print("unit done", UNIT, flush=True)
    raise SystemExit(0)

def run(train_labels):
    per = []
    for s in SEEDS:
        o = {}
        for i in range(5):
            te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
            o.update(train_abmil_clf_fold(bags, tr, te, train_labels, s))
        per.append(o)
        print("seed", s, "done", flush=True)
    return {k: float(np.mean([p[k] for p in per])) for k in keys}

oof_case = run(y_case)
oof_slide = run(y_slide)
res = {"_meta": {"n_slides": len(keys), "n_patients": len(set(pat.values())),
                 "binary_disagreement": round(float((lab.y_slide != lab.y_case).mean()), 4),
                 "seeds": SEEDS},
       "eval_on_slide_truth": {
           "trained_case_max": bootstrap_auc(oof_case, y_slide),
           "trained_slide_level": bootstrap_auc(oof_slide, y_slide, prob_b=oof_case)},
       "eval_on_case_max_truth": {
           "trained_case_max": bootstrap_auc(oof_case, y_case),
           "trained_slide_level": bootstrap_auc(oof_slide, y_case)}}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
