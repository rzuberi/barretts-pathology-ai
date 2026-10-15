"""OCCAMS v3 — THE PRE-REGISTERED Ch2 run (EXECUTION_PLAN 1.1).

Design per docs/occams_v2_decision_tree.md: ABMIL over tile-level UNI2 features
with Cox loss; linear-Cox genomics arm; clinical-only arm; late fusion of z-scored
OOF risks. Harrell's C primary; signal = C lower CI > 0.55; fusion benefit =
paired delta CI excludes 0. Native censoring (no exclusions). 5 folds x 3 seeds.
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_cox import (train_abmil_fold, train_linear_cox_fold, zscore_oof,
                       bootstrap_c, stratified_folds, cindex)

BASE = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data"
FEAT = BASE + "/slides/features/20x_224px/features_uni_v2"
TSV = BASE + "/genomics/clinical_data_wgs_cases_therapy_tp53status_ploidy_wgd_status.tsv"
MASTER = "/home/zuberi01/occams_work/occams_master_20260511.csv"
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]

def norm_id(s):
    s = str(s).strip().upper().replace("/", "-")
    m = re.search(r"(?:OCCAMS|OC)[-_ ]?([A-Z]{2})[-_ ]?0*([0-9]+)", s)
    return f"{m.group(1)}{int(m.group(2)):04d}" if m else s

# --- survival from master (native censoring) ---
mast = pd.read_csv(MASTER, dtype=str, low_memory=False)
mast["cid"] = mast["occams_id"].map(norm_id)
dsd = pd.to_numeric(mast["deceased_survival_days"], errors="coerce")
lkd = pd.to_numeric(mast["last_known_survival_days"], errors="coerce")
mast["time"] = dsd.fillna(lkd); mast["event"] = dsd.notna().astype(int)
mast = mast[mast["time"] > 0].drop_duplicates("cid").set_index("cid")
time = mast["time"].to_dict(); event = mast["event"].to_dict()

# --- histology bags (tile-level, RES-preferred, per case) ---
bags, bag_spec = {}, {}
for f in sorted(glob.glob(os.path.join(FEAT, "*.h5"))):
    b = os.path.basename(f)
    c = norm_id(b.split("_")[0])
    if c in bags and "_RES" not in b:
        continue
    with h5py.File(f) as h:
        bags[c] = np.asarray(h["features"])
    bag_spec[c] = "RES" if "_RES" in b.upper() else ("OGD" if "_OGD" in b.upper() else "OTHER")
print(f"bags={len(bags)} master_labelled={len(mast)}")

# --- genomics arm ---
th = pd.read_csv(TSV, sep="\t", dtype=str)
th["cid"] = th["OCCAMS_ID"].map(norm_id)
th = th.drop_duplicates("cid").set_index("cid")
def fl(s): return s.astype(str).str.strip().str.lower().isin(("1","true","yes","y")).astype(float)

# --- clinical arm from master (regex-selected, printed for the record) ---
clin_cols = [c for c in mast.columns if re.search(
    r"age_at_diag|performance_status|charlson|(^|_)[tnm]stage.*pretreat", c, re.I)][:8]
print("clinical columns:", clin_cols)
def code(col):
    v = mast[col].astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(v, errors="coerce")

cases = sorted(set(bags) & set(mast.index) & set(th.index))
# 2026-09-09 audit: the RES-preference rule only fires when a RES slide EXISTS,
# so OGD-only cases enter the cohort as biopsies. SPECIMEN=res_only restricts to
# true resections; SPECIMEN=ogd_only is the complement. Default: unchanged.
SPECIMEN = os.environ.get("SPECIMEN", "all")
spec_mix = {t: sum(1 for c in cases if bag_spec.get(c) == t) for t in ("RES", "OGD", "OTHER")}
print("specimen mix in cohort:", spec_mix)
if SPECIMEN == "res_only":
    cases = [c for c in cases if bag_spec.get(c) == "RES"]
elif SPECIMEN == "ogd_only":
    cases = [c for c in cases if bag_spec.get(c) == "OGD"]
print(f"final case set n={len(cases)} events={sum(event[c] for c in cases)} specimen={SPECIMEN}")
attrition = {"h5_bags": len(bags), "master_with_survival": len(mast),
             "genomics_tsv": len(th),
             "bags_and_master": len(set(bags) & set(mast.index)),
             "bags_and_genomics": len(set(bags) & set(th.index)),
             "master_and_genomics": len(set(mast.index) & set(th.index)),
             "all_three_final": len(cases)}
print("attrition:", attrition)
SHUFFLE = bool(os.environ.get("SHUFFLE"))
if SHUFFLE:  # label-permutation control: same cases, survival pairs reassigned
    perm = np.random.RandomState(0).permutation(cases)
    time = {**time, **{c: time[p] for c, p in zip(cases, perm)}}
    event = {**event, **{c: int(event[p]) for c, p in zip(cases, perm)}}
G = {"keys": cases, "X": np.nan_to_num(np.column_stack(
    [fl(th.loc[cases]["TP53_SNV"]), fl(th.loc[cases]["TP53_indel"]),
     fl(th.loc[cases]["TP53_deletion"]), fl(th.loc[cases]["TP53_knockout"]),
     pd.to_numeric(th.loc[cases]["ploidy"], errors="coerce").fillna(2.0),
     fl(th.loc[cases]["WGD"])]), nan=0.0)}
C = {"keys": cases, "X": np.nan_to_num(np.column_stack(
    [code(c).loc[cases].fillna(code(c).median()) for c in clin_cols]), nan=0.0)}

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
        print(f"{name} seed{s} C={cindex(np.array([o[k] for k in cases]), np.array([time[k] for k in cases]), np.array([event[k] for k in cases])):.3f}", flush=True)
    oof[name] = {k: float(np.mean([p[k] for p in per_seed])) for k in cases}

z = {n: zscore_oof(o) for n, o in oof.items()}
oof["late_hist_gen"] = {k: (z["hist_abmil"][k] + z["gen_cox"][k]) / 2 for k in cases}
oof["late_hist_clin"] = {k: (z["hist_abmil"][k] + z["clin_cox"][k]) / 2 for k in cases}
# 2.46 (Astra A8): persist OOF risks + fold membership so pooled-vs-within-fold
# concordance and fold-local fusion normalisation can be audited without retraining
json.dump({"fold_of": {k: i for i, f in enumerate(folds) for k in f},
           "time": {k: float(time[k]) for k in cases}, "event": {k: int(event[k]) for k in cases},
           "oof": {n: {k: float(v) for k, v in o.items()} for n, o in oof.items()}},
          open(os.path.join(OUT, "oof.json"), "w"))

res = {"_meta": {"n": len(cases), "events": int(sum(event[c] for c in cases)),
                 "seeds": SEEDS, "clinical_cols": clin_cols, "shuffle": SHUFFLE,
                 "attrition": attrition, "specimen": SPECIMEN, "specimen_mix": spec_mix,
                 "design": "pre-registered v3: ABMIL-Cox / linear-Cox / Harrell C"}}
for name, o in oof.items():
    ref = oof["hist_abmil"] if name.startswith("late") else None
    res[name] = bootstrap_c(o, time, event, risk_b=ref)
    print(name, res[name], flush=True)

fn = "results.json" if SPECIMEN == "all" else f"results_{SPECIMEN}.json"
json.dump(res, open(os.path.join(OUT, fn), "w"), indent=2)
print("wrote", fn)
