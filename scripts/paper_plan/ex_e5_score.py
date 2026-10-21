"""E5: score the analysed cohort's pre-event slides with the ERIN models (GPU; reads only UNI2-h feature files, never a slide).
Models: ex_e5_train.py arm-B fold models (lgd, hgd, six; 5 folds x 3 seeds each, SWG-overlap patients excluded) and the existing leak-free P32 report-label
grade head (feasibility/runs/p32_head_noov, model_grade_LGDplus_f0-4, abmil_clf.ABMIL, overlap patients excluded). Each slide is scored on all its tiles.
Per slide: mean over models of the logit (binary) or of the class probabilities (six). Also ERIN validation of the new fold models (aggregate).
Row-level -> explore_E5/erin_scores_slides.csv (cluster only); aggregate -> explore_E5/erin_validation.json."""
import os, sys, glob, json, numpy as np, pandas as pd, h5py, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun"; E5 = T + "/feasibility/paper_plan/killcoyne_mm/explore_E5"
sys.path.insert(0, T + "/scripts"); from abmil_clf import ABMIL as ABMIL_CLF
dev = "cuda" if torch.cuda.is_available() else "cpu"
PARTIAL = bool(os.environ.get("E5_PARTIAL")); SFX = "_partial" if PARTIAL else ""; torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
class ABMIL(nn.Module):   # train_fold.py
    def __init__(self, d_in=1536, d=384, d_attn=128, p_drop=0.25, n_out=1):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.attn_v = nn.Sequential(nn.Linear(d, d_attn), nn.Tanh()); self.attn_u = nn.Sequential(nn.Linear(d, d_attn), nn.Sigmoid())
        self.attn_w = nn.Linear(d_attn, 1); self.head = nn.Linear(d, n_out)
    def forward(self, bag, mask):
        h = self.emb(bag) * mask.unsqueeze(-1); a = self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf")).softmax(dim=1); return self.head((a.unsqueeze(-1) * h).sum(1))
MODELS = {}
for task in ("lgd", "hgd", "six"):
    L = []
    for f in sorted(glob.glob(f"{E5}/erin_models/{task}_seed*_fold*.pt")):
        m = ABMIL(n_out=6 if task == "six" else 1).to(dev); m.load_state_dict(torch.load(f, map_location=dev)); L.append(m.eval())
    assert len(L) == 15, (task, len(L)); MODELS[task] = L
P32 = []
for k in range(5):
    m = ABMIL_CLF(d_in=1536); m.load_state_dict(torch.load(f"{T}/feasibility/runs/p32_head_noov/output/model_grade_LGDplus_f{k}.pt", map_location="cpu")); P32.append(m.to(dev).eval())
ss = pd.read_csv(E5 + "/sample_slides.csv", dtype=str); rows = []
PREV = E5 + "/erin_scores_slides_partial.csv"   # slides already scored by the partial run (same models, deterministic): reused
if not PARTIAL and os.path.exists(PREV): rows = pd.read_csv(PREV).to_dict("records")
done_h5 = set(r["h5"] for r in rows)
with torch.inference_mode():
    for h5 in sorted(set(ss.h5)):
        if PARTIAL and not os.path.exists(h5): continue
        if h5 in done_h5: continue
        with h5py.File(h5) as h: Xn = np.asarray(h["features"], np.float32); mpp = float(h.attrs.get("mpp_used", np.nan))
        X = torch.from_numpy(Xn).to(dev)[None]; mk = torch.ones(X.shape[:2], dtype=torch.bool, device=dev); r = {"h5": h5, "n_tiles": int(Xn.shape[0]), "mpp": mpp}
        if Xn.shape[0] == 0: rows.append(r); continue
        for task in ("lgd", "hgd"): r[f"logit_{task}"] = float(np.mean([m(X, mk).item() for m in MODELS[task]]))
        tl = torch.stack([m.head(m.emb(X[0])).squeeze(-1) for m in MODELS["lgd"]]).mean(0).cpu().numpy()   # per-tile LGD+ logit (instance head), mean over models
        r["tile_frac_lgd"] = float((tl > 0).mean()); r["tile_top16_lgd"] = float(np.sort(tl)[::-1][:16].mean())
        p6 = np.mean([torch.softmax(m(X, mk), 1)[0].cpu().numpy() for m in MODELS["six"]], 0)
        for i, c in enumerate(["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]): r[f"p6_{c}"] = float(p6[i])
        r["logit_p32"] = float(np.mean([m(X[0])[0].item() for m in P32])); rows.append(r)
S = pd.DataFrame(rows); S.to_csv(E5 + f"/erin_scores_slides{SFX}.csv", index=False)
# ---------------- ERIN validation of the fold models (overlap patients already excluded from test)
s = pd.read_csv(A + "/slides.csv", dtype={"idx": int}).fillna(""); s["vote_frac"] = pd.to_numeric(s.vote_frac, errors="coerce"); s = s.set_index("idx")
V = {}
for task in ("lgd", "hgd"):
    te = {}; ho = {}
    for f in glob.glob(f"{E5}/erin_preds/{task}_seed*_fold*.npz"):
        z = np.load(f)
        for i, p in zip(z["test_idx"], z["test_prob"]): te.setdefault(int(i), []).append(float(p))
        for i, p in zip(z["holdout_idx"], z["holdout_prob"]): ho.setdefault(int(i), []).append(float(p))
    pos = ["LGD", "HGD", "CANCER"] if task == "lgd" else ["HGD", "CANCER"]; allowed = ["NORMAL_OTHER", "NDBE", "LGD", "HGD", "CANCER"]
    def ev(dct, col, extra=None):
        ii = np.array(sorted(dct)); d = s.loc[ii]; ok = d[col].isin(allowed).values & (np.ones(len(d), bool) if extra is None else extra(d))
        yy = d[col].isin(pos).values[ok].astype(int); pp = np.array([np.mean(dct[i]) for i in ii])[ok]
        return {"n": int(ok.sum()), "pos": int(yy.sum()), "auroc": round(float(roc_auc_score(yy, pp)), 4) if 0 < yy.sum() < len(yy) else None}
    V[task] = {"T1_heldout_fold_votefrac_ge_0.75": ev(te, "worst_grade", lambda d: (d.vote_frac >= 0.75).values), "T3_heldout_fold_all": ev(te, "worst_grade"),
               "T2_human_holdout": ev(ho, "human_grade")}
V["swg_slides_scored"] = int(S.logit_lgd.notna().sum()); V["swg_slides"] = int(len(S)); V["swg_mpp_used"] = S.mpp.describe().round(3).to_dict(); V["swg_tiles_per_slide"] = S.n_tiles.describe().round(1).to_dict()
json.dump(V, open(E5 + f"/erin_validation{SFX}.json", "w"), indent=1); print(json.dumps(V, indent=1)); print("E5 SCORE DONE")
