"""2.35: final-pipeline permutation controls (wave-3 gate item 3).

Runs N_PERM label permutations through the IDENTICAL final pipelines for the
two arms flagged as lacking controls:
  - ERIN progression (hist ABMIL, n=153/28) — the reframed Ch3 primary
  - TCGA-OAC ABMIL survival (n=65)
Reports the empirical null mean/sd and the real result's percentile. PERM env
selects arm; N_PERM default 25 per job (two jobs per arm = 50 total).
"""
import json, os, sys
import h5py, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
ARM = os.environ.get("ARM", "erin_prog")
N_PERM = int(os.environ.get("N_PERM", 25))
SEED0 = int(os.environ.get("SEED0", 0))

if ARM == "erin_prog":
    from abmil_clf import train_abmil_clf_fold, patient_folds
    from sklearn.metrics import roc_auc_score
    m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
    m["date"] = pd.to_datetime(m["CollectedOrOrdered"], errors="coerce", dayfirst=True)
    coh = pd.read_csv(T + "/labeller/erin_progression_cohort_v3.csv", dtype=str)
    coh["index_date"] = pd.to_datetime(coh["index_date"], errors="coerce")
    coh["y"] = (coh["progressed_to_HGDplus"] == "True").astype(int)
    rows = []
    for _, r in coh.iterrows():
        s = m[(m["anon_id"] == r["anon_id"]) & (m["date"] <= r["index_date"])]
        if s.empty: continue
        s = s[s["date"] == s["date"].max()]
        for _, sl in s.iterrows():
            rows.append({"h5": sl["h5"], "anon_id": r["anon_id"], "y": int(r["y"])})
    idx = pd.DataFrame(rows).drop_duplicates("h5")
    bags = {}
    for _, r in idx.iterrows():
        with h5py.File(r["h5"]) as h:
            bags[r["h5"]] = np.asarray(h["features"])
    keys = sorted(bags)
    pat = dict(zip(idx["h5"], idx["anon_id"]))
    y0 = dict(zip(idx["h5"], idx["y"].astype(int)))
    pats_u = sorted(set(pat.values()))
    ylist = [max(y0[k] for k in keys if pat[k] == a) for a in pats_u]
    nulls = []
    for p_i in range(N_PERM):
        rng = np.random.RandomState(SEED0 + p_i)
        perm_pat = dict(zip(pats_u, rng.permutation(ylist)))
        y = {k: int(perm_pat[pat[k]]) for k in keys}
        folds = patient_folds(keys, pat, y, 5, seed=0)
        o = {}
        for i in range(5):
            te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
            o.update(train_abmil_clf_fold(bags, tr, te, y, 0))
        auc = roc_auc_score([y[k] for k in keys], [o[k] for k in keys])
        nulls.append(float(auc))
        print(f"perm {p_i}: {auc:.4f}", flush=True)
    real = 0.8191
else:  # tcga_abmil survival
    from abmil_cox import train_abmil_fold, stratified_folds, cindex
    import glob, re
    lab = pd.read_csv(T + "/data/tcga_oac_labels.csv").drop_duplicates("barcode").set_index("barcode")
    lab = lab[lab["os_days"] > 0]
    FEAT = "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2"
    bags, time_, event = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(FEAT, "*.h5"))):
        mm = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
        if not mm or mm.group(1) not in lab.index or mm.group(1) in bags: continue
        with h5py.File(f) as h:
            bags[mm.group(1)] = np.asarray(h["features"])
        time_[mm.group(1)] = float(lab.loc[mm.group(1), "os_days"])
        event[mm.group(1)] = int(lab.loc[mm.group(1), "os_event"])
    cases = sorted(bags)
    nulls = []
    for p_i in range(N_PERM):
        rng = np.random.RandomState(SEED0 + p_i)
        perm = rng.permutation(cases)
        t = {c: time_[p] for c, p in zip(cases, perm)}
        e = {c: event[p] for c, p in zip(cases, perm)}
        folds = stratified_folds(cases, e, 5, seed=0)
        o = {}
        for i in range(5):
            te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
            o.update(train_abmil_fold(bags, tr, te, t, e, 0))
        c = cindex(np.array([o[k] for k in cases]), np.array([t[k] for k in cases]),
                   np.array([e[k] for k in cases]))
        nulls.append(float(c))
        print(f"perm {p_i}: {c:.4f}", flush=True)
    real = None  # read from results/tcga_abmil.json at analysis time

json.dump({"arm": ARM, "seed0": SEED0, "n_perm": len(nulls), "nulls": nulls,
           "null_mean": round(float(np.mean(nulls)), 4),
           "null_sd": round(float(np.std(nulls)), 4),
           "real_reference": real},
          open(os.path.join(OUT, f"results_{ARM}_{SEED0}.json"), "w"), indent=2)
# task-level results.json written by whichever shard finishes last
shards = [f for f in os.listdir(OUT) if f.startswith("results_")]
if len(shards) >= 4:
    allr = [json.load(open(os.path.join(OUT, f))) for f in shards]
    json.dump({"shards": allr}, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("done", flush=True)
