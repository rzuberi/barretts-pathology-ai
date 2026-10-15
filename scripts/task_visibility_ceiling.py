"""Visibility ceiling: ABMIL (attention over tiles) predicting TP53/WGD on the
full TCGA pool — the upper bound to the linear-probe floor (0.678/0.703)."""
import glob, json, os, re, sys
import numpy as np, pandas as pd, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_clf import train_abmil_clf_fold, bootstrap_auc, patient_folds

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
FEATS = {"OAC": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2",
         "GEJ": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/features/20x_224px/features_uni_v2"}
LABELS = {"OAC": T + "/data/tcga_oac_labels.csv", "GEJ": T + "/data/tcga_stad_labels.csv"}
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]

bags, lab_rows = {}, {}
for grp, fd in FEATS.items():
    lab = pd.read_csv(LABELS[grp]).drop_duplicates("barcode").set_index("barcode")
    for f in sorted(glob.glob(os.path.join(fd, "*.h5"))):
        m = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
        if not m or m.group(1) not in lab.index or m.group(1) in bags: continue
        with h5py.File(f) as h:
            bags[m.group(1)] = np.asarray(h["features"])
        lab_rows[m.group(1)] = lab.loc[m.group(1)]

res = {}
for target in ["tp53_mut", "wgd"]:
    y = {k: int(float(v[target])) for k, v in lab_rows.items()
         if pd.notna(v[target])}
    keys = sorted(set(bags) & set(y))
    pat = {k: k for k in keys}  # one bag per patient here
    folds = patient_folds(keys, pat, y, 5, seed=0)
    per = []
    for s in SEEDS:
        o = {}
        for i in range(5):
            te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
            o.update(train_abmil_clf_fold(bags, tr, te, y, s))
        per.append(o); print(target, "seed", s, "done", flush=True)
    oof = {k: float(np.mean([p[k] for p in per])) for k in keys}
    res[target] = {"n": len(keys), "pos": int(sum(y.values())),
                   **bootstrap_auc(oof, y)}
    print(target, res[target], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
