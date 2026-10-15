"""EXECUTION_PLAN R.2 payoff: the POWERED Ch2 fusion test on TCGA OAC + STAD/GEJ pool.

Same pre-registered machinery as occams v3 / tcga_abmil (ABMIL-Cox, linear-Cox
genomics + clinical, late fusion, Harrell's C, patient-disjoint folds), now at
n≈446. Primary contrast: late(hist+gen) minus hist. OAC-only subgroup reported.
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_cox import (train_abmil_fold, train_linear_cox_fold, zscore_oof,
                       bootstrap_c, stratified_folds, cindex)

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
FEATS = {"OAC": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2",
         "GEJ": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/features/20x_224px/features_uni_v2"}
LABELS = {"OAC": T + "/data/tcga_oac_labels.csv", "GEJ": T + "/data/tcga_stad_labels.csv"}
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]

bags, meta = {}, []
for grp, fd in FEATS.items():
    lab = pd.read_csv(LABELS[grp])
    lab = lab[lab["os_days"] > 0].drop_duplicates("barcode").set_index("barcode")
    for f in sorted(glob.glob(os.path.join(fd, "*.h5"))):
        mm = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
        if not mm or mm.group(1) not in lab.index or mm.group(1) in bags: continue
        with h5py.File(f) as h:
            bags[mm.group(1)] = np.asarray(h["features"])
        r = lab.loc[mm.group(1)]
        stage = re.search(r"Stage (I{1,3}V?)", str(r["ajcc_stage"]))
        meta.append({"barcode": mm.group(1), "grp": grp,
                     "time": float(r["os_days"]), "event": int(r["os_event"]),
                     "tp53": float(r["tp53_mut"]) if pd.notna(r["tp53_mut"]) else 0.0,
                     "ploidy": float(r["ploidy"]) if pd.notna(r["ploidy"]) else 2.0,
                     "wgd": float(r["wgd"]) if pd.notna(r["wgd"]) else 0.0,
                     "stage": {"I": 1, "II": 2, "III": 3, "IV": 4}.get(stage.group(1) if stage else "", 2.5),
                     "age": float(r["age"]) if pd.notna(r["age"]) else np.nan})
md = pd.DataFrame(meta).set_index("barcode")
md["age"] = md["age"].fillna(md["age"].median())
cases = sorted(md.index)
time = md["time"].to_dict(); event = md["event"].to_dict()
print(f"pool n={len(cases)} (OAC {sum(md.grp=='OAC')}, GEJ {sum(md.grp=='GEJ')}) events={md.event.sum()}", flush=True)

G = {"keys": cases, "X": md.loc[cases, ["tp53", "ploidy", "wgd"]].values}
C = {"keys": cases, "X": md.loc[cases, ["stage", "age"]].values}
folds = stratified_folds(cases, event, 5, seed=0)
arms = {"hist_abmil": lambda tr, te, s: train_abmil_fold(bags, tr, te, time, event, s),
        "gen_cox": lambda tr, te, s: train_linear_cox_fold(G, tr, te, time, event, s),
        "clin_cox": lambda tr, te, s: train_linear_cox_fold(C, tr, te, time, event, s)}
oof = {}
for name, fn in arms.items():
    per = []
    for s in SEEDS:
        o = {}
        for i in range(5):
            te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
            o.update(fn(tr, te, s))
        per.append(o)
        arr = lambda d: np.array([d[k] for k in cases])
        print(f"{name} seed{s} C={cindex(arr(o), arr(time), arr(event)):.3f}", flush=True)
    oof[name] = {k: float(np.mean([p[k] for p in per])) for k in cases}
z = {n: zscore_oof(o) for n, o in oof.items()}
oof["late_hist_gen"] = {k: (z["hist_abmil"][k] + z["gen_cox"][k]) / 2 for k in cases}
oof["late_hist_clin"] = {k: (z["hist_abmil"][k] + z["clin_cox"][k]) / 2 for k in cases}

res = {"_meta": {"n": len(cases), "events": int(md["event"].sum()),
                 "oac_n": int((md.grp == "OAC").sum()), "seeds": SEEDS}}
for name, o in oof.items():
    ref = oof["hist_abmil"] if name.startswith("late") else None
    res[name] = bootstrap_c(o, time, event, risk_b=ref)
    print(name, res[name], flush=True)
oac = [c for c in cases if md.loc[c, "grp"] == "OAC"]
res["oac_subgroup_hist_c"] = round(float(cindex(
    np.array([oof["hist_abmil"][k] for k in oac]),
    np.array([time[k] for k in oac]), np.array([event[k] for k in oac]))), 4)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
