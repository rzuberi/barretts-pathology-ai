"""2.25: predicted-WGD/TP53 transfer — resection teachers -> surveillance biopsies.

Train ABMIL teachers on the pooled labelled resections (TCGA-OAC + TCGA-STAD/GEJ
+ OCCAMS), gate on 5-fold CV reproducing the visibility ceiling (WGD >= 0.68,
else abort — no inference from an unvalidated teacher), retrain on all cases,
then score ERIN progression-cohort index slides. Decisive readout: does
predicted genomic state in a pre-progression biopsy mark future progressors?
"""
import glob, json, os, re, sys
import h5py, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_clf import train_abmil_clf_fold, bootstrap_auc, patient_folds
from sklearn.metrics import roc_auc_score

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]
GATE = {"wgd": 0.68, "tp53_mut": 0.62}

FEATS = {"tcga_oac": ("/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2",
                      T + "/data/tcga_oac_labels.csv"),
         "stad_gej": ("/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/features/20x_224px/features_uni_v2",
                      T + "/data/tcga_stad_labels.csv")}
OCC_FEAT = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/slides/features/20x_224px/features_uni_v2"
OCC_TSV = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/genomics/clinical_data_wgs_cases_therapy_tp53status_ploidy_wgd_status.tsv"

def norm_occ(s):
    s = str(s).strip().upper().replace("/", "-")
    mm = re.search(r"(?:OCCAMS|OC)[-_ ]?([A-Z]{2})[-_ ]?0*([0-9]+)", s)
    return f"{mm.group(1)}{int(mm.group(2)):04d}" if mm else s

bags, labels = {}, {}
for grp, (fd, lp) in FEATS.items():
    lab = pd.read_csv(lp).drop_duplicates("barcode").set_index("barcode")
    for f in sorted(glob.glob(os.path.join(fd, "*.h5"))):
        mm = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
        if not mm or mm.group(1) not in lab.index or mm.group(1) in bags: continue
        with h5py.File(f) as h:
            bags[mm.group(1)] = np.asarray(h["features"])
        r = lab.loc[mm.group(1)]
        labels[mm.group(1)] = {"tp53_mut": r["tp53_mut"], "wgd": r["wgd"]}
th = pd.read_csv(OCC_TSV, sep="\t", dtype=str)
th["cid"] = th["OCCAMS_ID"].map(norm_occ)
th = th.drop_duplicates("cid").set_index("cid")
def fl(v): return float(str(v).strip().lower() in ("1", "true", "yes", "y"))
for f in sorted(glob.glob(os.path.join(OCC_FEAT, "*.h5"))):
    c = norm_occ(os.path.basename(f).split("_")[0])
    if c in bags or c not in th.index: continue
    with h5py.File(f) as h:
        bags[c] = np.asarray(h["features"])
    r = th.loc[c]
    labels[c] = {"tp53_mut": max(fl(r["TP53_SNV"]), fl(r["TP53_indel"]),
                                 fl(r["TP53_deletion"]), fl(r["TP53_knockout"])),
                 "wgd": fl(r["WGD"]) if pd.notna(r["WGD"]) else np.nan}
print(f"teacher cases: {len(bags)}", flush=True)

# ERIN progression index slides (same construction as task_erin_progression)
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
m["date"] = pd.to_datetime(m["CollectedOrOrdered"], errors="coerce", dayfirst=True)
coh = pd.read_csv(T + "/labeller/erin_progression_cohort_v3.csv", dtype=str)
coh["index_date"] = pd.to_datetime(coh["index_date"], errors="coerce")
coh["y"] = (coh["progressed_to_HGDplus"] == "True").astype(int)
idx_rows = []
for _, r in coh.iterrows():
    s = m[(m["anon_id"] == r["anon_id"]) & (m["date"] <= r["index_date"])]
    if s.empty: continue
    s = s[s["date"] == s["date"].max()]
    for _, sl in s.iterrows():
        idx_rows.append({"h5": sl["h5"], "anon_id": r["anon_id"], "y": int(r["y"])})
idx = pd.DataFrame(idx_rows).drop_duplicates("h5")
erin_bags = {}
for _, r in idx.iterrows():
    with h5py.File(r["h5"]) as h:
        erin_bags[r["h5"]] = np.asarray(h["features"])
print(f"erin index slides: {len(erin_bags)}", flush=True)

res = {"_meta": {"teacher_cases": len(bags), "erin_index_slides": len(erin_bags),
                 "gates": GATE, "seeds": SEEDS}}
for target in ["wgd", "tp53_mut"]:
    y = {k: int(float(v[target])) for k, v in labels.items()
         if k in bags and pd.notna(v[target])}
    keys = sorted(y)
    folds = patient_folds(keys, {k: k for k in keys}, y, 5, seed=0)
    per = []
    for s in SEEDS:
        o = {}
        for i in range(5):
            te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
            o.update(train_abmil_clf_fold(bags, tr, te, y, s))
        per.append(o); print(target, "cv seed", s, flush=True)
    oof = {k: float(np.mean([p[k] for p in per])) for k in keys}
    cv = bootstrap_auc(oof, y)
    res[target] = {"teacher_cv": {"n": len(keys), "pos": int(sum(y.values())), **cv}}
    print(target, "teacher CV:", cv, flush=True)
    if cv["auc"] < GATE[target]:
        res[target]["transfer"] = f"ABORTED: teacher CV {cv['auc']} below gate {GATE[target]}"
        print(res[target]["transfer"], flush=True); continue
    # final teachers trained on ALL labelled cases score the ERIN slides
    preds = []
    for s in SEEDS:
        allb = {**bags, **erin_bags}
        o = train_abmil_clf_fold(allb, keys, sorted(erin_bags), {**y, **{k: 0 for k in erin_bags}}, s)
        preds.append(o); print(target, "final seed", s, flush=True)
    score = {k: float(np.mean([p[k] for p in preds])) for k in erin_bags}
    per_pat = pd.DataFrame({"a": [idx.set_index("h5").loc[k, "anon_id"] for k in score],
                            "p": list(score.values())}).groupby("a")["p"].max()
    yy = np.array([int(coh.set_index("anon_id").loc[a, "y"]) for a in per_pat.index])
    auc = roc_auc_score(yy, per_pat.values) if 0 < yy.sum() < len(yy) else None
    res[target]["transfer"] = {"n_patients": len(per_pat), "progressors": int(yy.sum()),
                               "progression_auc_of_predicted": round(float(auc), 4) if auc else None,
                               "score_mean_prog": round(float(per_pat.values[yy == 1].mean()), 4),
                               "score_mean_nonprog": round(float(per_pat.values[yy == 0].mean()), 4)}
    print(target, "transfer:", res[target]["transfer"], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
