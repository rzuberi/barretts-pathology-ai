"""2.34 downstream: pre-registered encoder sweep on the SURVIVAL tasks
(OCCAMS v3 n=87, TCGA-OAC n=65, TCGA pool) — the wave-3 gate's one genuinely
new compute item. For each encoder with complete features: histology-only
ABMIL-Cox + late fusion with genomics, identical folds/machinery to the UNI2
runs. Encoders with incomplete features are skipped and named in results
(no silent truncation).
"""
import glob, json, os, re, sys
import h5py, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_cox import train_abmil_fold, train_linear_cox_fold, stratified_folds, cindex, bootstrap_c

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]
ENCODERS = {"virchow2": 2560, "gigapath": 1536, "phikon2": 1024, "hoptimus0": 1536}
MIN_FRAC = 0.95

OCC_FEAT = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/slides/features/20x_224px"
OCC_TSV = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/genomics/clinical_data_wgs_cases_therapy_tp53status_ploidy_wgd_status.tsv"
MASTER = "/home/zuberi01/occams_work/occams_master_20260511.csv"
TCGA = {"oac": ("/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px",
                T + "/data/tcga_oac_labels.csv"),
        "stad": ("/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/features/20x_224px",
                 T + "/data/tcga_stad_labels.csv")}

def norm_occ(s):
    s = str(s).strip().upper().replace("/", "-")
    mm = re.search(r"(?:OCCAMS|OC)[-_ ]?([A-Z]{2})[-_ ]?0*([0-9]+)", s)
    return f"{mm.group(1)}{int(mm.group(2)):04d}" if mm else s

# reference case lists from UNI2 (the frozen cohort definitions)
def occ_cases_uni2():
    th = pd.read_csv(OCC_TSV, sep="\t", dtype=str)
    th["cid"] = th["OCCAMS_ID"].map(norm_occ)
    th = th.drop_duplicates("cid").set_index("cid")
    mast = pd.read_csv(MASTER, dtype=str, low_memory=False)
    mast["cid"] = mast["occams_id"].map(norm_occ)
    dsd = pd.to_numeric(mast["deceased_survival_days"], errors="coerce")
    lkd = pd.to_numeric(mast["last_known_survival_days"], errors="coerce")
    mast["time"] = dsd.fillna(lkd); mast["event"] = dsd.notna().astype(int)
    mast = mast[mast["time"] > 0].drop_duplicates("cid").set_index("cid")
    out = {}
    for f in sorted(glob.glob(os.path.join(OCC_FEAT, "features_uni_v2", "*.h5"))):
        c = norm_occ(os.path.basename(f).split("_")[0])
        if c in out or c not in th.index or c not in mast.index: continue
        g = th.loc[c]
        def fl(v): return float(str(v).strip().lower() in ("1", "true", "yes", "y"))
        gen = [max(fl(g["TP53_SNV"]), fl(g["TP53_indel"]), fl(g["TP53_deletion"]), fl(g["TP53_knockout"])),
               (float(g["ploidy"]) - 2.0) if pd.notna(g["ploidy"]) else 0.0,
               fl(g["WGD"]) if pd.notna(g["WGD"]) else 0.0]
        out[c] = {"stem": os.path.basename(f)[:-3], "gen": np.array(gen, dtype=np.float32),
                  "time": float(mast.loc[c, "time"]), "event": int(mast.loc[c, "event"])}
    return out

def tcga_cases_uni2():
    out = {}
    for grp, (fd, lp) in TCGA.items():
        lab = pd.read_csv(lp).drop_duplicates("barcode").set_index("barcode")
        lab = lab[lab["os_days"] > 0]
        for f in sorted(glob.glob(os.path.join(fd, "features_uni_v2", "*.h5"))):
            mm = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
            if not mm or mm.group(1) in out or mm.group(1) not in lab.index: continue
            r = lab.loc[mm.group(1)]
            gen = [float(r["tp53_mut"]) if pd.notna(r["tp53_mut"]) else 0.0,
                   (float(r["ploidy"]) - 2.0) if pd.notna(r["ploidy"]) else 0.0,
                   float(r["wgd"]) if pd.notna(r["wgd"]) else 0.0]
            out[mm.group(1)] = {"stem": os.path.basename(f)[:-3], "grp": grp,
                                "gen": np.array(gen, dtype=np.float32),
                                "time": float(r["os_days"]), "event": int(r["os_event"])}
    return out

OCC, TC = occ_cases_uni2(), tcga_cases_uni2()
print(f"cohorts: occams={len(OCC)} tcga_pool={len(TC)} "
      f"tcga_oac={sum(1 for v in TC.values() if v['grp']=='oac')}", flush=True)

def load_bags(cases, featdir, enc):
    bags, miss = {}, 0
    for c, v in cases.items():
        # encoder h5 shares the stem; occams sits in one dir, tcga in per-group dirs
        if "grp" in v:
            path = os.path.join(TCGA[v["grp"]][0], f"features_{enc}", v["stem"] + ".h5")
        else:
            path = os.path.join(OCC_FEAT, f"features_{enc}", v["stem"] + ".h5")
        if not os.path.exists(path):
            # occams stems may differ (case_RES_x vs case_OGD): fall back to any h5 for the case
            alt = glob.glob(os.path.join(OCC_FEAT, f"features_{enc}", c.replace("", "") + "*RES*.h5")) \
                if "grp" not in v else []
            if alt: path = alt[0]
            else: miss += 1; continue
        with h5py.File(path) as h:
            bags[c] = np.asarray(h["features"])
    return bags, miss

def run_cohort(name, cases, enc):
    bags, miss = load_bags(cases, None, enc)
    frac = len(bags) / len(cases)
    if frac < MIN_FRAC:
        return {"skipped": f"features incomplete: {len(bags)}/{len(cases)}"}
    keys = sorted(bags)
    t = {k: cases[k]["time"] for k in keys}
    e = {k: cases[k]["event"] for k in keys}
    import numpy as _np
    G = {"keys": keys, "X": _np.stack([cases[k]["gen"] for k in keys])}
    folds = stratified_folds(keys, e, 5, seed=0)
    def cv(train_fn):
        per = []
        for s in SEEDS:
            o = {}
            for i in range(5):
                te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
                o.update(train_fn(tr, te, s))
            per.append(o)
        return {k: float(np.mean([p[k] for p in per])) for k in keys}
    hist = cv(lambda tr, te, s: train_abmil_fold(bags, tr, te, t, e, s))
    gen = cv(lambda tr, te, s: train_linear_cox_fold(G, tr, te, t, e, s))
    fuse = {k: 0.5 * ((hist[k] - np.mean(list(hist.values()))) / (np.std(list(hist.values())) + 1e-9)
                      + (gen[k] - np.mean(list(gen.values()))) / (np.std(list(gen.values())) + 1e-9))
            for k in keys}
    res = {}
    for arm, risk in [("hist", hist), ("late_fusion", fuse)]:
        res[arm] = bootstrap_c(risk, t, e, risk_b=hist if arm != "hist" else None)
    res["n"] = len(keys); res["missing_features"] = miss
    return res

res = {"_meta": {"encoders_attempted": list(ENCODERS), "min_frac": MIN_FRAC, "seeds": SEEDS}}
for enc in ENCODERS:
    res[enc] = {}
    for cname, cases in [("occams", OCC), ("tcga_pool", TC),
                         ("tcga_oac", {k: v for k, v in TC.items() if v["grp"] == "oac"})]:
        res[enc][cname] = run_cohort(cname, cases, enc)
        print(enc, cname, res[enc][cname].get("skipped") or
              {a: res[enc][cname][a] for a in ("hist", "late_fusion")}, flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
