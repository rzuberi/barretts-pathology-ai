"""TCGA ABMIL fusion (EXECUTION_PLAN 1.2): same v3 machinery on TCGA-OAC.

Native censoring from CDR (os_event/os_days). Arms: ABMIL-Cox histology,
linear-Cox genomics (TP53/ploidy/WGD), linear-Cox clinical (stage/age), late
fusions. Harrell's C + bootstrap. Small n (~65) — treated as replication arm.
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_cox import (train_abmil_fold, train_linear_cox_fold, zscore_oof,
                       bootstrap_c, stratified_folds, cindex)

FEAT = "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2"
LABELS = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/data/tcga_oac_labels.csv"
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]

bags = {}
for f in sorted(glob.glob(os.path.join(FEAT, "*.h5"))):
    m = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
    if m:
        with h5py.File(f) as h:
            bags[m.group(1)] = np.asarray(h["features"])

lab = pd.read_csv(LABELS)
lab = lab[(lab["os_days"] > 0) & lab["barcode"].isin(bags)].drop_duplicates("barcode").set_index("barcode")
time = lab["os_days"].to_dict(); event = lab["os_event"].astype(int).to_dict()
stage = lab["ajcc_stage"].astype(str).str.extract(r"Stage (I{1,3}V?)", expand=False)
lab["stage_num"] = stage.map({"I": 1, "II": 2, "III": 3, "IV": 4}).fillna(2.5)
cases = sorted(lab.index)
print(f"n={len(cases)} events={sum(event[c] for c in cases)}")

G = {"keys": cases, "X": np.nan_to_num(np.column_stack(
    [lab["tp53_mut"].fillna(0), lab["ploidy"].fillna(2.0), lab["wgd"].astype(float).fillna(0)]), nan=0.0)}
C = {"keys": cases, "X": np.nan_to_num(np.column_stack(
    [lab["stage_num"], lab["age"].fillna(lab["age"].median())]), nan=0.0)}

folds = stratified_folds(cases, event, 5, seed=0)
arms = {"hist_abmil": lambda tr, te, s: train_abmil_fold(bags, tr, te, time, event, s),
        "gen_cox": lambda tr, te, s: train_linear_cox_fold(G, tr, te, time, event, s),
        "clin_cox": lambda tr, te, s: train_linear_cox_fold(C, tr, te, time, event, s)}
oof = {}
for name, fn in arms.items():
    per_seed = []
    for s in SEEDS:
        o = {}
        for i in range(5):
            te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
            o.update(fn(tr, te, s))
        per_seed.append(o)
    oof[name] = {k: float(np.mean([p[k] for p in per_seed])) for k in cases}
    arr = lambda d: np.array([d[k] for k in cases])
    print(name, "C=%.3f" % cindex(arr(oof[name]), arr(time), arr(event)), flush=True)

z = {n: zscore_oof(o) for n, o in oof.items()}
oof["late_hist_gen"] = {k: (z["hist_abmil"][k] + z["gen_cox"][k]) / 2 for k in cases}
oof["late_hist_clin"] = {k: (z["hist_abmil"][k] + z["clin_cox"][k]) / 2 for k in cases}

res = {"_meta": {"n": len(cases), "events": int(sum(event[c] for c in cases)), "seeds": SEEDS}}
for name, o in oof.items():
    ref = oof["hist_abmil"] if name.startswith("late") else None
    res[name] = bootstrap_c(o, time, event, risk_b=ref)
    print(name, res[name], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
