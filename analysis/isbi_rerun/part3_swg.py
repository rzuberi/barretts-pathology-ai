"""Part 3: SWG transfer diagnosis (binary LGD+). READ-ONLY on sources; outputs -> analysis/isbi_rerun/part3/.
(i) SWG->SWG 5-fold by patient (same ABMIL/hparams, 3 seeds); (ii) ERIN arm-B binary fold models (patient holdout) -> SWG,
54 overlapping patients removed; (iii) SWG models -> ERIN T1 and T2.
SWG truth: xlsx block grade when the slide's accession has exactly one xlsx grade, else the release per-sample grade
(CurrentGradeNorm); source recorded. ID/IND excluded from the binary task (as in Part 2)."""
import glob, json, os, re, time
import numpy as np, pandas as pd, h5py, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun"; P3 = A + "/part3"; os.makedirs(P3, exist_ok=True)
REL = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
XL = "/mnt/scratche/slow/fmlab/zuberi01/phd/Leanne_shared_docs/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3).xlsx"
SWGF = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort/features_uni2h_05um"
OVER = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/runs/overlap_audit/output/swg_overlap_patients.txt"
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
CAP, D, K = 1200, 1536, 800; MAP = {"BE": "NDBE", "ID": "IND", "LGD": "LGD", "HGD": "HGD", "IMC": "CANCER", "OAC": "CANCER", "NDBE": "NDBE"}

def acc_norm(s):
    s = re.sub(r"[\s_][A-Za-z]{1,2}\d{0,2}\s*(\(.*\))?\s*$", "", str(s).strip()); m = re.match(r"^\s*([A-Za-z]{1,3})\s*(\d{2})\s*[-./]?\s*(\d{3,6})\s*$", s)
    return f"{m.group(1).upper()}{m.group(2)}-{int(m.group(3)):05d}" if m else None
# ---- SWG slides, labels ----
coh = pd.read_csv(REL + "/pre_event_cohort.csv", dtype=str); coh["base"] = coh.ImageAbsPath.map(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
coh = coh.drop_duplicates("base"); coh["h5"] = coh.base.map(lambda b: f"{SWGF}/{b}.h5"); coh = coh[coh.h5.map(os.path.exists)].copy()
coh["acc"] = coh.BiopsyID_real.map(acc_norm); coh["rel_grade"] = coh.CurrentGradeNorm.map(lambda g: MAP.get(g, g))
x = pd.read_excel(XL, dtype=str); x["acc"] = x["Path ID"].map(acc_norm); x["g"] = x.Pathology.str.upper().str.strip().map(MAP)
xg = x.dropna(subset=["acc", "g"]).groupby("acc").g.agg(lambda v: v.iloc[0] if v.nunique() == 1 else None)
coh["xlsx_grade"] = coh.acc.map(xg); coh["grade"] = coh.xlsx_grade.fillna(coh.rel_grade); coh["grade_source"] = np.where(coh.xlsx_grade.notna(), "xlsx_unique_block", "release_sample")
overlap = set(l.strip() for l in open(OVER) if l.strip()); coh["overlap"] = coh.PatientID_real.isin(overlap) | coh.PatientID.isin(overlap)
coh["y"] = coh.grade.isin(["LGD", "HGD", "CANCER"]).astype(int); coh["keep_bin"] = coh.grade.isin(["NORMAL_OTHER", "NDBE", "LGD", "HGD", "CANCER"])
coh = coh.reset_index(drop=True); coh["sidx"] = np.arange(len(coh))
# ---- pack SWG into GPU ----
t0 = time.time(); XS = torch.zeros((len(coh), CAP, D), dtype=torch.float16, device=dev); NS = torch.zeros(len(coh), dtype=torch.long, device=dev)
for i, r in coh.iterrows():
    with h5py.File(r.h5) as h: F = np.asarray(h["features"], dtype=np.float32)
    if len(F) > CAP: F = F[np.sort(np.random.RandomState(10000 + i).choice(len(F), CAP, replace=False))]
    XS[i, :len(F)] = torch.from_numpy(F.astype(np.float16)).to(dev); NS[i] = len(F)
print(f"SWG resident: {len(coh)} slides in {time.time()-t0:.0f}s | grade source: {coh.grade_source.value_counts().to_dict()} | grades: {coh.grade.value_counts().to_dict()} | overlap patients' slides: {int(coh.overlap.sum())}", flush=True)

class ABMIL(nn.Module):
    def __init__(self, d_in=1536, d=384, d_attn=128, p_drop=0.25, n_out=1):
        super().__init__(); self.emb = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.attn_v = nn.Sequential(nn.Linear(d, d_attn), nn.Tanh()); self.attn_u = nn.Sequential(nn.Linear(d, d_attn), nn.Sigmoid()); self.attn_w = nn.Linear(d_attn, 1); self.head = nn.Linear(d, n_out)
    def forward(self, bag, mask):
        h = self.emb(bag) * mask.unsqueeze(-1); a = self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(1); return self.head((a.unsqueeze(-1) * h).sum(1))
def gather(X, N, ids, g, k):
    B = len(ids); out = torch.zeros((B, k, D), dtype=torch.float16, device=dev)   # zeros, not empty: uninitialised padding leaked NaN/inf through 0*x
    mask = torch.zeros((B, k), dtype=torch.bool, device=dev)
    for b, i in enumerate(ids.tolist()):
        ni = int(N[i]); kk = min(k, ni); sel = torch.randperm(ni, generator=g)[:kk].to(dev) if ni > k else torch.arange(kk, device=dev)
        out[b, :kk] = X[i, sel]; mask[b, :kk] = True
    return out, mask
def train(X, N, ids, y, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); rng = np.random.RandomState(seed); m = ABMIL().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-5); yt = torch.tensor(y, device=dev, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((len(y) - y.sum()) / max(y.sum(), 1)), device=dev)); m.train()
    for ep in range(25):
        order = rng.permutation(len(ids))
        for i in range(0, len(order), 32):
            ch = order[i:i + 32]; bags, mask = gather(X, N, torch.tensor(ids[ch]), g, K)
            lg = m(bags.float(), mask); loss = lossf(lg.squeeze(-1), yt[ch])
            if not torch.isfinite(loss): opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    return m
@torch.inference_mode()
def predict(m, X, N, ids):
    m.eval(); g = torch.Generator().manual_seed(0); out = []
    for i in range(0, len(ids), 64):
        bags, mask = gather(X, N, torch.tensor(ids[i:i + 64]), g, CAP)
        out.append(torch.sigmoid(m(bags.float(), mask).squeeze(-1)).cpu().numpy())
    return np.concatenate(out)
def boot_auc(p, y, pats, nb=2000, seed=0):
    rng = np.random.RandomState(seed); up = np.unique(pats); grp = {u: np.where(pats == u)[0] for u in up}; v = []
    for _ in range(nb):
        ix = np.concatenate([grp[u] for u in rng.choice(up, len(up), replace=True)])
        if 0 < y[ix].sum() < len(ix): v.append(roc_auc_score(y[ix], p[ix]))
    return {"auc": round(roc_auc_score(y, p), 4), "ci": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)], "n": int(len(y)), "pos": int(y.sum()), "neg": int(len(y) - y.sum()), "patients": int(len(up))}
R = {"swg_slides": int(len(coh)), "grade_source": coh.grade_source.value_counts().to_dict(), "grade_dist": coh.grade.value_counts().to_dict(), "overlap_slides": int(coh.overlap.sum())}
# (i) SWG -> SWG, 5-fold by patient, 3 seeds, all SWG patients (overlap irrelevant within-cohort)
d = coh[coh.keep_bin]; pats = d.PatientID.values; up = np.unique(pats); rng = np.random.RandomState(0); rng.shuffle(up); fold = {u: i % 5 for i, u in enumerate(up)}
d = d.assign(fold=d.PatientID.map(fold)); preds = {}
for seed in [0, 1, 2]:
    for f in range(5):
        tr = d[d.fold != f]; te = d[d.fold == f]; m = train(XS, NS, tr.sidx.values, tr.y.values, seed); p = predict(m, XS, NS, te.sidx.values)
        for i, pp in zip(te.sidx.values, p): preds.setdefault(int(i), []).append(pp)
        torch.save(m.state_dict(), f"{P3}/swg_seed{seed}_fold{f}.pt")
ids = np.array(sorted(preds)); p = np.array([np.mean(preds[i]) for i in ids]); dd = coh.set_index("sidx").loc[ids]
R["swg_to_swg"] = boot_auc(p, dd.y.values, dd.PatientID.values); pd.DataFrame({"sidx": ids, "prob": p, "y": dd.y.values, "patient": dd.PatientID.values}).to_csv(P3 + "/pred_swg_to_swg.csv", index=False)
print("SWG->SWG", R["swg_to_swg"], flush=True)
# (ii) ERIN arm-B binary models (patient holdout) -> SWG, overlap patients removed
dn = d[~d.overlap]; ps = []
for f in sorted(glob.glob(A + "/models/patient_B_bin_seed*_fold*.pt")):
    m = ABMIL().to(dev); m.load_state_dict(torch.load(f, map_location=dev)); ps.append(predict(m, XS, NS, dn.sidx.values))
if ps:
    p = np.mean(ps, 0); R["erin_B_to_swg"] = dict(boot_auc(p, dn.y.values, dn.PatientID.values), n_models=len(ps))
    pd.DataFrame({"sidx": dn.sidx.values, "prob": p, "y": dn.y.values, "patient": dn.PatientID.values}).to_csv(P3 + "/pred_erin_to_swg.csv", index=False); print("ERIN->SWG", R["erin_B_to_swg"], flush=True)
else: R["erin_B_to_swg"] = "no ERIN arm-B models found yet"
# (iii) SWG models -> ERIN T1 and T2 (ERIN resident set loaded here; overlap patients removed from ERIN side too)
s = pd.read_csv(A + "/slides.csv", dtype={"idx": int}).fillna(""); s["vote_frac"] = pd.to_numeric(s.vote_frac, errors="coerce")
excl = set(pd.read_csv(A + "/qc/exclusions.csv").idx) if os.path.getsize(A + "/qc/exclusions.csv") > 0 else set()
idxf = json.load(open(A + "/pack/index.json")); NE = torch.tensor(idxf["n_tiles"], device=dev)
del XS; torch.cuda.empty_cache()
XE = torch.empty((len(s), CAP, D), dtype=torch.float16, device=dev)
for c in idxf["chunks"]:
    arr = np.load(A + "/pack/" + c["file"], mmap_mode="r")
    for a in range(0, c["hi"] - c["lo"], 256):
        e = min(a + 256, c["hi"] - c["lo"]); XE[c["lo"] + a:c["lo"] + e] = torch.from_numpy(np.ascontiguousarray(arr[a:e])).to(dev)
# ERIN-side exclusion of the shared patients: SWG accessions -> ERIN anon_id, plus the Barrett's-DB participant bridge
rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv", dtype=str, low_memory=False).drop_duplicates("CaseName")
racc = pd.Series(rep.anon_id.values, index=rep.CaseName.map(acc_norm)); ov_erin = set(racc.reindex(coh.acc.dropna()).dropna())
cohp = pd.read_csv(REL + "/pre_event_cohort.csv", dtype=str); pid = set(cohp.participant_id.dropna().map(lambda v: re.sub(r"\.0$", "", str(v))))
db = pd.read_csv("/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export/pathology_text_normalised_full.csv", dtype=str, usecols=["participant_id", "specimennumber"])
db["acc"] = db.specimennumber.map(acc_norm); db["pid"] = db.participant_id.map(lambda v: re.sub(r"\.0$", "", str(v)) if pd.notna(v) else None)
ov_erin |= set(racc.reindex(db[db.pid.isin(pid)].acc.dropna()).dropna()); R["erin_patients_excluded_as_shared"] = len(ov_erin)
def erin_truth(col, sub):
    g = sub[col]; keep = g.isin(["NORMAL_OTHER", "NDBE", "LGD", "HGD", "CANCER"]); return sub[keep], g[keep].isin(["LGD", "HGD", "CANCER"]).astype(int).values
base = s[(~s.idx.isin(excl)) & (s.worst_grade != "") & (~s.anon_id.isin(ov_erin))]
T1, y1 = erin_truth("worst_grade", base[(base.fold >= 0) & (base.vote_frac >= 0.75)]); T2, y2 = erin_truth("human_grade", base[(base.fold == -1) & (base.human_grade != "")])
swg_models = [torch.load(f, map_location=dev) for f in sorted(glob.glob(P3 + "/swg_seed*_fold*.pt"))]
for name, Tset, y in [("T1", T1, y1), ("T2", T2, y2)]:
    ps = []
    for sd in swg_models:
        m = ABMIL().to(dev); m.load_state_dict(sd); ps.append(predict(m, XE, NE, Tset.idx.values))
    p = np.mean(ps, 0); R[f"swg_to_erin_{name}"] = dict(boot_auc(p, y, Tset.anon_id.values), n_models=len(ps))
    pd.DataFrame({"idx": Tset.idx.values, "prob": p, "y": y, "patient": Tset.anon_id.values}).to_csv(P3 + f"/pred_swg_to_erin_{name}.csv", index=False); print(f"SWG->ERIN {name}", R[f"swg_to_erin_{name}"], flush=True)
json.dump(R, open(P3 + "/results.json", "w"), indent=1, default=str)
for f in glob.glob(P3 + "/*"): os.chmod(f, 0o600)
print("part3 complete", flush=True)
