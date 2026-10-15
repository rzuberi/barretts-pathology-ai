"""2.38 aggregator: assemble the 30 unit predictions into OOF per arm,
evaluate on slide-level and case-max truth, paired bootstrap."""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_clf import bootstrap_auc

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
POS = {"LGD", "HGD", "CANCER"}
lab = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str)
lab["y_slide"] = lab["worst_grade"].isin(POS).astype(int)
lab["y_case"] = lab["case_max"].isin(POS).astype(int)
y_slide = dict(zip(lab["h5"], lab["y_slide"]))
y_case = dict(zip(lab["h5"], lab["y_case"]))
units = glob.glob(T + "/feasibility/svc_units/*.npz")
print(f"units found: {len(units)}/30", flush=True)
acc = {"case": {}, "slide": {}}
for f in units:
    arm, seed, fold = os.path.basename(f)[:-4].split("_")
    z = np.load(f, allow_pickle=True)
    for k, p in zip(z["keys"], z["preds"]):
        acc[arm].setdefault(str(k), []).append(float(p))
oof = {a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in acc.items()}
n_c, n_s = len(oof["case"]), len(oof["slide"])
assert n_c == n_s and n_c > 1000, f"incomplete OOF: case={n_c} slide={n_s}"
res = {"_meta": {"units": len(units), "n_slides": n_c},
       "eval_on_slide_truth": {
           "trained_case_max": bootstrap_auc(oof["case"], y_slide),
           "trained_slide_level": bootstrap_auc(oof["slide"], y_slide, prob_b=oof["case"])},
       "eval_on_case_max_truth": {
           "trained_case_max": bootstrap_auc(oof["case"], y_case),
           "trained_slide_level": bootstrap_auc(oof["slide"], y_case)}}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
