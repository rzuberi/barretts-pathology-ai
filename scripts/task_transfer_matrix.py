"""2.32: cross-cohort transfer matrix (4-model consensus proposal).

Train in one cohort, evaluate in the others, per endpoint class:
  progression:  SWG <-> ERIN (binary progressed_to_HGDplus-style)
  grade:        ERIN -> SWG (pathologist Label on SWG as truth)
  survival:     OCCAMS <-> TCGA-pool (24-month landmark mortality)
All arms: pooled UNI2 embedding + scaler/PCA64/logistic (identical machinery,
so cells are comparable). Within-cohort CV AUC is reported beside each
transfer AUC — the gap IS the readout.
"""
import glob, json, os, re
import h5py, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
OUT = os.environ.get("OUTDIR", ".")

def make_pipe():
    return Pipeline([("s", StandardScaler()), ("p", PCA(64)),
                     ("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000))])

def cv_auc(X, y, groups, seed=0):
    uniq = sorted(set(groups)); rng = np.random.RandomState(seed)
    fold = {g: i % 5 for i, g in enumerate(rng.permutation(uniq))}
    fm = np.array([fold[g] for g in groups])
    oof = np.zeros(len(y))
    for f in range(5):
        tr, te = fm != f, fm == f
        pl = make_pipe(); pl.fit(X[tr], y[tr]); oof[te] = pl.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))

# ---- SWG: pooled embeddings, progression + grade ----
man = pd.read_csv(F + "/training_manifest.csv", dtype=str)
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).merge(
    man, left_on="SampleID", right_on="sample_id")
uidx = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str)
npz_of = dict(zip(uidx[uidx["status"] == "ok"]["sample_id"], uidx[uidx["status"] == "ok"]["npz_path"]))
# 2.40 follow-up: drop SWG patients who are also ERIN patients (overlap audit list)
EXCL_F = os.environ.get("EXCLUDE_PATIENTS", "")
if EXCL_F and not os.path.exists(EXCL_F):
    raise SystemExit(f"EXCLUDE_PATIENTS set but missing: {EXCL_F}")
EXCL = set(open(EXCL_F).read().split()) if EXCL_F else set()
rows = []; n_excl = 0
for _, r in coh.iterrows():
    if r["sample_id"] not in npz_of: continue
    if EXCL and str(r["patient_id"]) in EXCL:
        n_excl += 1; continue
    z = np.load(npz_of[r["sample_id"]])
    rows.append({"emb": np.asarray(z["slide_embedding_mean"]), "pat": r["patient_id"],
                 "y_prog": int(r["y_progressor"]),
                 "grade_pos": int(str(r.get("Label", "")).upper() not in ("NDBE", "ND", "NEGATIVE", "0", "IM"))})
swg = pd.DataFrame(rows)
Xs = np.stack(swg["emb"]); print(f"SWG {len(swg)}", flush=True)
swg_pat = swg.groupby("pat").agg(y=("y_prog", "max")).reset_index()

# ---- ERIN: pooled cache, progression cohort + grade labels ----
d = np.load(T + "/feasibility/runs/erin_probes/output/erin_pooled_uni2.npz", allow_pickle=True)
emb_of = {h: d["X"][i] for i, h in enumerate(d["h5s"])}
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
m["date"] = pd.to_datetime(m["CollectedOrOrdered"], errors="coerce", dayfirst=True)
pc = pd.read_csv(T + "/labeller/erin_progression_cohort_v3.csv", dtype=str)
pc["index_date"] = pd.to_datetime(pc["index_date"], errors="coerce")
er_rows = []
for _, r in pc.iterrows():
    s = m[(m["anon_id"] == r["anon_id"]) & (m["date"] <= r["index_date"])]
    s = s[s["h5"].isin(emb_of)]
    if s.empty: continue
    s = s[s["date"] == s["date"].max()]
    er_rows.append({"emb": np.mean([emb_of[h] for h in s["h5"]], axis=0),
                    "pat": r["anon_id"], "y": int(r["progressed_to_HGDplus"] == "True")})
erin_prog = pd.DataFrame(er_rows)
Xe = np.stack(erin_prog["emb"]); print(f"ERIN prog {len(erin_prog)}", flush=True)
elig = m[m["label_status"].isin(["train_eligible", "adjudicated"])
         & m["final_label"].isin(["NDBE", "LGD", "HGD", "CANCER"]) & m["h5"].isin(emb_of)]
Xg = np.stack([emb_of[h] for h in elig["h5"]])
yg = elig["final_label"].isin({"LGD", "HGD", "CANCER"}).astype(int).values
gg = elig["anon_id"].values

# ---- survival cohorts: 24-month landmark ----
def landmark_cohort(bag_iter, time_of, event_of, months=24):
    rows = []
    for cid, emb in bag_iter:
        t, e = time_of(cid), event_of(cid)
        if t is None: continue
        if t >= months * 30.44:
            rows.append({"emb": emb, "cid": cid, "y": 0 if not (e and t < months * 30.44) else 1})
            rows[-1]["y"] = 0
        elif e:
            rows.append({"emb": emb, "cid": cid, "y": 1})
    return pd.DataFrame(rows)

def pooled_from_h5(fd, keyre=r"(TCGA-\w{2}-\w{4})"):
    for f in sorted(glob.glob(os.path.join(fd, "*.h5"))):
        mm = re.search(keyre, os.path.basename(f))
        if not mm: continue
        with h5py.File(f) as h:
            yield mm.group(1), np.asarray(h["features"]).mean(0)

lab_o = pd.read_csv(T + "/data/tcga_oac_labels.csv").drop_duplicates("barcode").set_index("barcode")
lab_s = pd.read_csv(T + "/data/tcga_stad_labels.csv").drop_duplicates("barcode").set_index("barcode")
def t_time(c):
    for lb in (lab_o, lab_s):
        if c in lb.index and pd.notna(lb.loc[c, "os_days"]): return float(lb.loc[c, "os_days"])
    return None
def t_event(c):
    for lb in (lab_o, lab_s):
        if c in lb.index: return int(lb.loc[c, "os_event"])
    return 0
seen = {}
for fd in ["/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2",
           "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/features/20x_224px/features_uni_v2"]:
    for cid, emb in pooled_from_h5(fd):
        seen.setdefault(cid, emb)
tcga = landmark_cohort(seen.items(), t_time, t_event)
print(f"TCGA landmark {len(tcga)} pos={tcga['y'].sum()}", flush=True)

OCC_TSV = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/genomics/clinical_data_wgs_cases_therapy_tp53status_ploidy_wgd_status.tsv"
MASTER = "/home/zuberi01/occams_work/occams_master_20260511.csv"
def norm_occ(s):
    s = str(s).strip().upper().replace("/", "-")
    mm = re.search(r"(?:OCCAMS|OC)[-_ ]?([A-Z]{2})[-_ ]?0*([0-9]+)", s)
    return f"{mm.group(1)}{int(mm.group(2)):04d}" if mm else s
mast = pd.read_csv(MASTER, dtype=str, low_memory=False)
mast["cid"] = mast["occams_id"].map(norm_occ)
dsd = pd.to_numeric(mast["deceased_survival_days"], errors="coerce")
lkd = pd.to_numeric(mast["last_known_survival_days"], errors="coerce")
mast["time"] = dsd.fillna(lkd); mast["event"] = dsd.notna().astype(int)
mast = mast[mast["time"] > 0].drop_duplicates("cid").set_index("cid")
occ_seen = dict(pooled_from_h5("/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/slides/features/20x_224px/features_uni_v2",
                               keyre=r"^([^_]+)"))
occ_seen = {norm_occ(k): v for k, v in occ_seen.items()}
occ = landmark_cohort(((c, e) for c, e in occ_seen.items() if c in mast.index),
                      lambda c: float(mast.loc[c, "time"]),
                      lambda c: int(mast.loc[c, "event"]))
print(f"OCCAMS landmark {len(occ)} pos={occ['y'].sum()}", flush=True)

res = {"_meta": {"machinery": "pooled UNI2 + scaler/PCA64/logistic everywhere",
                 "swg_samples_excluded_for_erin_overlap": n_excl, "exclude_file": EXCL_F,
                 "swg_patients_used": int(swg["pat"].nunique())}}
def cell(name, Xtr, ytr, Xte, yte, gtr):
    within = cv_auc(Xtr, ytr, gtr)
    pl = make_pipe(); pl.fit(Xtr, ytr)
    transfer = float(roc_auc_score(yte, pl.predict_proba(Xte)[:, 1])) \
        if 0 < yte.sum() < len(yte) else None
    res[name] = {"within_train_cohort_cv": round(within, 4),
                 "transfer_auc": round(transfer, 4) if transfer else None,
                 "n_train": len(ytr), "n_test": len(yte)}
    print(name, res[name], flush=True)

ys, ye = swg["y_prog"].values, erin_prog["y"].values
cell("prog_SWG_to_ERIN", Xs, ys, Xe, ye, swg["pat"].values)
cell("prog_ERIN_to_SWG", Xe, ye, Xs, ys, erin_prog["pat"].values)
cell("grade_ERIN_to_SWG", Xg, yg, Xs, swg["grade_pos"].values, gg)
Xt, yt = np.stack(tcga["emb"]), tcga["y"].values
Xo, yo = np.stack(occ["emb"]), occ["y"].values
cell("surv24_OCCAMS_to_TCGA", Xo, yo, Xt, yt, occ["cid"].values)
cell("surv24_TCGA_to_OCCAMS", Xt, yt, Xo, yo, tcga["cid"].values)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
