"""2.25b: WGD/TP53 teacher scored on SWG biopsies, where MEASURED sWGS exists.

Diagnoses the ERIN transfer inversion: if predicted genomic state correlates
with measured CNV complexity (cx) on biopsies, the teacher's RANKING survives
specimen-type shift and the ERIN inversion needs another explanation; if the
correlation is absent/negative, the transfer is broken at ranking level and
'virtual biomarkers' should not cross specimen types at all.
Teachers identical to task_wgd_transfer (gate-checked there: WGD 0.730,
TP53 0.764). SWG tile embeddings via the release npz index.
"""
import glob, json, os, re, sys
import h5py, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_clf import train_abmil_clf_fold
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]

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

# SWG samples: tile embeddings + measured cx + progression labels
man = pd.read_csv(F + "/training_manifest.csv", dtype=str)
uidx = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str)
uidx = uidx[uidx["status"] == "ok"]
npz_of = dict(zip(uidx["sample_id"], uidx["npz_path"]))
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str)
key = next(c for c in cx.columns if c.lower() in ("sampleid", "sample_id"))
cx_of = {r[key]: pd.to_numeric(r["cx"], errors="coerce") for _, r in cx.iterrows()}
swg_bags, swg_meta = {}, []
for _, r in man.iterrows():
    sid = r["sample_id"]
    if sid not in npz_of: continue
    z = np.load(npz_of[sid])
    swg_bags["SWG_" + sid] = np.asarray(z["embeddings"])
    swg_meta.append({"k": "SWG_" + sid, "sid": sid, "patient": r["patient_id"],
                     "y_prog": int(r["y_progressor"]), "cx": cx_of.get(sid, np.nan)})
swg = pd.DataFrame(swg_meta)
print(f"swg samples: {len(swg)} with cx: {swg['cx'].notna().sum()}", flush=True)

res = {"_meta": {"teacher_cases": len(bags), "swg_samples": len(swg), "seeds": SEEDS}}
for target in ["wgd", "tp53_mut"]:
    y = {k: int(float(v[target])) for k, v in labels.items()
         if k in bags and pd.notna(v[target])}
    keys = sorted(y)
    preds = []
    for s in SEEDS:
        allb = {**{k: bags[k] for k in keys}, **swg_bags}
        o = train_abmil_clf_fold(allb, keys, sorted(swg_bags),
                                 {**y, **{k: 0 for k in swg_bags}}, s)
        preds.append(o); print(target, "seed", s, flush=True)
    swg[f"pred_{target}"] = [float(np.mean([p[k] for p in preds])) for k in swg["k"]]
    ok = swg["cx"].notna()
    rho, pv = spearmanr(swg.loc[ok, f"pred_{target}"], swg.loc[ok, "cx"])
    pat = swg.groupby("patient").agg(p=(f"pred_{target}", "max"), y=("y_prog", "max"))
    res[target] = {
        "pred_vs_measured_cx": {"spearman": round(float(rho), 4), "p": float(pv),
                                "n": int(ok.sum())},
        "progression_auc_of_predicted": round(float(roc_auc_score(
            pat["y"], pat["p"])), 4) if 0 < pat["y"].sum() < len(pat) else None}
    print(target, res[target], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
