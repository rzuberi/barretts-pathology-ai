"""Deep-dive A (extends 2.8): why did histology C collapse in the OAC+GEJ pool?

Within-cohort vs cross-cohort ABMIL-Cox: train OAC->test OAC, GEJ->GEJ (5-fold
within each), and transfer OAC->GEJ / GEJ->OAC. Plus pooled fusion with a cohort
indicator added to the clinical arm. Distinguishes site-effect from no-signal.
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_cox import train_abmil_fold, bootstrap_c, stratified_folds, cindex

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
FEATS = {"OAC": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2",
         "GEJ": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/features/20x_224px/features_uni_v2"}
LABELS = {"OAC": T + "/data/tcga_oac_labels.csv", "GEJ": T + "/data/tcga_stad_labels.csv"}
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]

bags, grp, time_, event = {}, {}, {}, {}
for g, fd in FEATS.items():
    lab = pd.read_csv(LABELS[g])
    lab = lab[lab["os_days"] > 0].drop_duplicates("barcode").set_index("barcode")
    for f in sorted(glob.glob(os.path.join(fd, "*.h5"))):
        m = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
        if not m or m.group(1) not in lab.index or m.group(1) in bags: continue
        with h5py.File(f) as h:
            bags[m.group(1)] = np.asarray(h["features"])
        grp[m.group(1)] = g
        time_[m.group(1)] = float(lab.loc[m.group(1), "os_days"])
        event[m.group(1)] = int(lab.loc[m.group(1), "os_event"])

def arr(d, ks): return np.array([d[k] for k in ks])
res = {"_meta": {c: sum(1 for k in bags if grp[k] == c) for c in ("OAC", "GEJ")}}

# within-cohort CV
for c in ("OAC", "GEJ"):
    ks = sorted(k for k in bags if grp[k] == c)
    folds = stratified_folds(ks, event, 5, seed=0)
    per = []
    for s in SEEDS:
        o = {}
        for i in range(5):
            te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
            o.update(train_abmil_fold(bags, tr, te, time_, event, s))
        per.append(o)
    oof = {k: float(np.mean([p[k] for p in per])) for k in ks}
    res[f"within_{c}"] = bootstrap_c(oof, time_, event)
    print(f"within_{c}", res[f"within_{c}"], flush=True)

# cross-cohort transfer
for src, dst in (("OAC", "GEJ"), ("GEJ", "OAC")):
    tr = sorted(k for k in bags if grp[k] == src)
    te = sorted(k for k in bags if grp[k] == dst)
    per = [train_abmil_fold(bags, tr, te, time_, event, s) for s in SEEDS]
    oof = {k: float(np.mean([p[k] for p in per])) for k in te}
    res[f"transfer_{src}_to_{dst}"] = bootstrap_c(oof, time_, event)
    print(f"transfer_{src}->{dst}", res[f"transfer_{src}_to_{dst}"], flush=True)

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
