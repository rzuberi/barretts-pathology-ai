"""2.22 (joint, Rehan 2026-08-25): genotype-visibility learning curve with n and
population disentangled — the 7/10-model consensus proposal.

Strata: TCGA-OAC (n<=65), OCCAMS-OAC (n<=~225), OAC-combined (cross-cohort,
batch caveat), STAD/GEJ, MIXED-ALL. At each (stratum, n) draw 20 patient
subsamples, run the standard probe (mean-pooled UNI2 -> PCA64 -> logistic),
report AUC mean/sd plus a permutation null (5 perms per replicate = 100 null
draws per cell). Decisive readout: STAD/GEJ at n=65 vs OAC at n=65 (matched-n
population contrast) and whether OAC-only rises with n or stays flat.
"""
import glob, json, os, re
import numpy as np, pandas as pd, h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
CACHE = os.path.join(OUT, "pooled_features.npz")
GRID = [50, 65, 100, 141, 200, 300, 381]
REPS, PERMS_PER_REP = 20, 5

SOURCES = {
    "tcga_oac": ("/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2",
                 T + "/data/tcga_oac_labels.csv"),
    "stad_gej": ("/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/features/20x_224px/features_uni_v2",
                 T + "/data/tcga_stad_labels.csv"),
}
OCCAMS_FEAT = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/slides/features/20x_224px/features_uni_v2"
OCCAMS_TSV = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/genomics/clinical_data_wgs_cases_therapy_tp53status_ploidy_wgd_status.tsv"

def norm_occ(s):
    s = str(s).strip().upper().replace("/", "-")
    m = re.search(r"(?:OCCAMS|OC)[-_ ]?([A-Z]{2})[-_ ]?0*([0-9]+)", s)
    return f"{m.group(1)}{int(m.group(2)):04d}" if m else s

if os.path.exists(CACHE):
    d = np.load(CACHE, allow_pickle=True)
    X = d["X"]; meta = pd.DataFrame({"key": d["key"], "stratum": d["stratum"],
                                     "tp53": d["tp53"], "wgd": d["wgd"]})
else:
    rows, feats = [], []
    for grp, (fd, lp) in SOURCES.items():
        lab = pd.read_csv(lp).drop_duplicates("barcode").set_index("barcode")
        for f in sorted(glob.glob(os.path.join(fd, "*.h5"))):
            m = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
            if not m or m.group(1) not in lab.index: continue
            if any(r["key"] == m.group(1) for r in rows): continue
            with h5py.File(f) as h:
                feats.append(np.asarray(h["features"]).mean(0))
            r = lab.loc[m.group(1)]
            rows.append({"key": m.group(1), "stratum": grp,
                         "tp53": float(r["tp53_mut"]) if pd.notna(r["tp53_mut"]) else np.nan,
                         "wgd": float(r["wgd"]) if pd.notna(r["wgd"]) else np.nan})
            if len(rows) % 100 == 0: print("pooled", len(rows), flush=True)
    th = pd.read_csv(OCCAMS_TSV, sep="\t", dtype=str)
    th["cid"] = th["OCCAMS_ID"].map(norm_occ)
    th = th.drop_duplicates("cid").set_index("cid")
    def fl(v): return float(str(v).strip().lower() in ("1", "true", "yes", "y"))
    seen = set()
    for f in sorted(glob.glob(os.path.join(OCCAMS_FEAT, "*.h5"))):
        c = norm_occ(os.path.basename(f).split("_")[0])
        if c in seen or c not in th.index: continue
        if c in seen: continue
        if "_RES" not in os.path.basename(f) and any(
                norm_occ(os.path.basename(g).split("_")[0]) == c and "_RES" in os.path.basename(g)
                for g in glob.glob(os.path.join(OCCAMS_FEAT, "*.h5"))): pass
        with h5py.File(f) as h:
            feats.append(np.asarray(h["features"]).mean(0))
        seen.add(c)
        r = th.loc[c]
        tp53 = max(fl(r["TP53_SNV"]), fl(r["TP53_indel"]), fl(r["TP53_deletion"]), fl(r["TP53_knockout"]))
        rows.append({"key": c, "stratum": "occams_oac", "tp53": tp53,
                     "wgd": fl(r["WGD"]) if pd.notna(r["WGD"]) else np.nan})
        if len(seen) % 50 == 0: print("occams", len(seen), flush=True)
    meta = pd.DataFrame(rows); X = np.vstack(feats)
    np.savez(CACHE, X=X, key=meta["key"].values, stratum=meta["stratum"].values,
             tp53=meta["tp53"].values, wgd=meta["wgd"].values)
print("cases:", meta["stratum"].value_counts().to_dict(), flush=True)

STRATA = {"tcga_oac": meta["stratum"] == "tcga_oac",
          "occams_oac": meta["stratum"] == "occams_oac",
          "oac_combined": meta["stratum"].isin(["tcga_oac", "occams_oac"]),
          "stad_gej": meta["stratum"] == "stad_gej",
          "mixed_all": meta["stratum"].notna()}

def probe_auc(Xs, ys, seed):
    oof = np.zeros(len(ys))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xs, ys):
        p = Pipeline([("s", StandardScaler()), ("p", PCA(min(64, len(tr) - 1))),
                      ("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))])
        p.fit(Xs[tr], ys[tr]); oof[te] = p.predict_proba(Xs[te])[:, 1]
    return roc_auc_score(ys, oof)

res = {"_meta": {"grid": GRID, "reps": REPS, "perms_per_rep": PERMS_PER_REP,
                 "counts": meta["stratum"].value_counts().to_dict()}}
for target in ["tp53", "wgd"]:
    res[target] = {}
    for sname, mask in STRATA.items():
        ok = mask.values & meta[target].notna().values
        Xs_all, ys_all = X[ok], meta[target][ok].astype(int).values
        n_avail = len(ys_all)
        res[target][sname] = {"n_available": int(n_avail), "pos": int(ys_all.sum()), "curve": {}}
        for n in GRID:
            if n > n_avail and n != GRID[-1]: continue
            n_use = min(n, n_avail)
            if n_use < 40 or n_use in [c["n"] for c in res[target][sname]["curve"].values()]: continue
            aucs, nulls = [], []
            for rep in range(REPS):
                rng = np.random.RandomState(1000 * rep + n_use)
                idx = rng.choice(n_avail, n_use, replace=False)
                ys = ys_all[idx]
                if ys.sum() < 8 or (len(ys) - ys.sum()) < 8: continue
                aucs.append(probe_auc(Xs_all[idx], ys, rep))
                for pi in range(PERMS_PER_REP):
                    nulls.append(probe_auc(Xs_all[idx], rng.permutation(ys), rep * 10 + pi))
            if not aucs: continue
            cell = {"n": n_use, "reps": len(aucs),
                    "auc": round(float(np.mean(aucs)), 3), "auc_sd": round(float(np.std(aucs)), 3),
                    "null_mean": round(float(np.mean(nulls)), 3), "null_sd": round(float(np.std(nulls)), 3),
                    "pctile_vs_null": round(float(np.mean([nl <= np.mean(aucs) for nl in nulls])), 3)}
            res[target][sname]["curve"][str(n_use)] = cell
            print(target, sname, cell, flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
