"""2.46 (Astra A8): pooled vs within-fold concordance on OCCAMS v3, and
fold-local late-fusion normalisation.

Pooling OOF Cox risks from independently fitted folds and computing one C
mixes risk scales across folds (each fold's model has its own offset); the
within-fold C only compares pairs from the same fold. Also recomputes late
fusion with FOLD-LOCAL z-scoring (the original z-scored across the pooled OOF,
i.e. with statistics that include every fold's test risks). Reads oof.json
dumped by task_occams_v3.py.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_cox import cindex, bootstrap_c

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
SRC = os.environ.get("OOF_JSON", T + "/feasibility/runs/occams_v3_oof/output/oof.json")
d = json.load(open(SRC))
fold_of, time, event, oof = d["fold_of"], d["time"], d["event"], d["oof"]
cases = sorted(fold_of); folds = sorted(set(fold_of.values()))
t = np.array([time[k] for k in cases]); e = np.array([event[k] for k in cases])

def within_fold_c(risk):
    num = den = 0.0
    for f in folds:
        ks = [k for k in cases if fold_of[k] == f]
        r = np.array([risk[k] for k in ks]); tt = np.array([time[k] for k in ks]); ee = np.array([event[k] for k in ks])
        for i in range(len(ks)):
            if not ee[i]: continue
            later = tt > tt[i]
            den += later.sum(); num += (r[later] < r[i]).sum() + 0.5 * (r[later] == r[i]).sum()
    return float(num / den) if den else 0.5

def zfold(risk):
    out = {}
    for f in folds:
        ks = [k for k in cases if fold_of[k] == f]
        v = np.array([risk[k] for k in ks]); mu, sd = v.mean(), v.std() or 1.0
        out.update({k: (risk[k] - mu) / sd for k in ks})
    return out

res = {"_meta": {"n": len(cases), "events": int(e.sum()), "folds": len(folds), "source": SRC}}
for name, o in oof.items():
    r = np.array([o[k] for k in cases])
    res[name] = {"pooled_c": round(float(cindex(r, t, e)), 4), "within_fold_c": round(within_fold_c(o), 4),
                 "per_fold_c": [round(float(cindex(np.array([o[k] for k in cases if fold_of[k] == f]),
                                                    np.array([time[k] for k in cases if fold_of[k] == f]),
                                                    np.array([event[k] for k in cases if fold_of[k] == f]))), 4)
                                for f in folds],
                 "per_fold_risk_mean": [round(float(np.mean([o[k] for k in cases if fold_of[k] == f])), 4) for f in folds]}
zh = zfold(oof["hist_abmil"])
for other, tag in (("gen_cox", "late_hist_gen"), ("clin_cox", "late_hist_clin")):
    zo = zfold(oof[other])
    fused = {k: (zh[k] + zo[k]) / 2 for k in cases}
    res[tag + "_foldlocal_z"] = bootstrap_c(fused, {k: time[k] for k in cases}, {k: event[k] for k in cases},
                                            risk_b=oof["hist_abmil"])
    res[tag + "_foldlocal_z"]["within_fold_c"] = round(within_fold_c(fused), 4)
res["verdict"] = {n: round(res[n]["within_fold_c"] - res[n]["pooled_c"], 4)
                  for n in oof}  # positive = pooling UNDERSTATED concordance (scale mixing)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
