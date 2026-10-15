"""Killcoyne final, items 1-2 (docs/paper_plan_killcoyne_final.md @ 8342ac0): fold-stratified AUROC and max-T adjusted p on the stratified
10 x 10 CV predictions (kv_cv.R), no refit. TASK = fs_<src>_<subset> | maxt. Parts -> feasibility/paper_plan/killcoyne_mm/final/parts/.
Subset definitions and the pooled-metric evaluation class are copied from kc_merge.py."""
import json, glob, os, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; K = T + "/feasibility/paper_plan/killcoyne"; M = K + "_mm"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
OUTP = M + "/final/parts"; os.makedirs(OUTP, exist_ok=True); TASK = os.environ["TASK"]
outf = f"{OUTP}/{TASK}.json"
if os.path.exists(outf): print("exists", outf); sys.exit(0)
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
C = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample"); C["y"] = (C.Status == "P").astype(int)
pats = np.array(sorted(C.Patient.unique())); pid = pd.Series(np.arange(len(pats)), index=pats)[C.Patient.values].values; yv = C.y.values; NP_ = len(pats)
ypat_all = np.zeros(NP_, int); np.maximum.at(ypat_all, pid, yv)
# ---- subsets (item 3)
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
A = pd.read_csv(K + "/kr_samples.csv", dtype=str).set_index("Sample"); A["mbf"] = pd.to_numeric(A.index.map(fd["Months before final"]), errors="coerce"); A["ogd"] = A.index.map(fd["Path_class_per_OGD"])
hg = A[(A.Pathology.isin(["HGD", "IMC"])) | (A.ogd.isin(["HGD", "IMC", "HGD/IMC"]))]; tev = hg.groupby("Patient").mbf.max()
Ppats = sorted(C[C.y == 1].Patient.unique()); fallback = [p for p in Ppats if p not in tev.index]; tev = pd.concat([tev, pd.Series(0.0, index=fallback)])
mbfC = C.index.map(A.mbf).astype(float).values
pre = np.array([True if y_ == 0 else (m > tev[p]) for y_, m, p in zip(yv, mbfC, C.Patient.values)])
ndbe = (C.Pathology == "NDBE").values
SUB = {"all_C": np.ones(len(C), bool), "a_NDBE_only": ndbe, "b_before_first_HGD_IMC": pre, "c_NDBE_and_before": ndbe & pre}
# ---- statistics restricted to a row subset
class Ev:
    def __init__(self, mask):
        self.r = np.where(mask)[0]; self.p = pid[self.r]; self.pp = np.unique(self.p); self.y = yv[self.r]; self.yp = ypat_all[self.pp]
        self.rows_of = {q: self.r[self.p == q] for q in self.pp}
        rng = np.random.RandomState(0); self.boot = []
        while len(self.boot) < 2000:
            ix = rng.choice(len(self.pp), len(self.pp))
            if 0 < self.yp[ix].sum() < len(ix): self.boot.append(ix)
        self.bootrows = [np.concatenate([self.rows_of[self.pp[i]] for i in ix]) for ix in self.boot]
        rng = np.random.RandomState(0); self.masks = []
        for _ in range(2000):
            sw = rng.rand(len(self.pp)) < 0.5; mk = np.zeros(len(yv), bool)
            for i in np.where(sw)[0]: mk[self.rows_of[self.pp[i]]] = True
            self.masks.append(mk)
    def pm(self, s): out = np.full(NP_, -np.inf); np.maximum.at(out, self.p, s[self.r]); return out[self.pp]
    def stats(self, s): return auc(self.y, s[self.r]), auc(self.yp, self.pm(s))
    def summary(self, s, ci=True):
        a, b = self.stats(s); o = {"n_samples": int(len(self.r)), "n_patients": int(len(self.pp)), "n_P_patients": int(self.yp.sum()), "per_sample_auroc": r3(a), "patient_max_auroc": r3(b)}
        if ci:
            pmv = self.pm(s); bs = np.array([(auc(yv[rr], s[rr]), auc(self.yp[ix], pmv[ix])) for ix, rr in zip(self.boot, self.bootrows)])
            o["ci95"] = {"per_sample_auroc": [r3(np.percentile(bs[:, 0], 2.5)), r3(np.percentile(bs[:, 0], 97.5))], "patient_max_auroc": [r3(np.percentile(bs[:, 1], 2.5)), r3(np.percentile(bs[:, 1], 97.5))]}
        return o
    def perm_deltas(self, a, b): return np.array([np.subtract(self.stats(np.where(mk, b, a)), self.stats(np.where(mk, a, b))) for mk in self.masks])
    def delta(self, a, b, pdist=None):
        obs = np.subtract(self.stats(a), self.stats(b)); pa, pb = self.pm(a), self.pm(b)
        bs = np.array([(auc(yv[rr], a[rr]) - auc(yv[rr], b[rr]), auc(self.yp[ix], pa[ix]) - auc(self.yp[ix], pb[ix])) for ix, rr in zip(self.boot, self.bootrows)])
        pdist = self.perm_deltas(a, b) if pdist is None else pdist
        return {k: {"delta": r3(obs[j]), "ci95": [r3(np.percentile(bs[:, j], 2.5)), r3(np.percentile(bs[:, j], 97.5))], "perm_p": round(float((1 + (np.abs(pdist[:, j]) >= abs(obs[j]) - 1e-12).sum()) / 2001), 4)}
                for j, k in enumerate(["per_sample_auroc", "patient_max_auroc"])}, obs, pdist

# ---- CV predictions per repeat
V = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in sorted(glob.glob(M + "/cv/preds/*.csv"))])
REPS = list(range(1, 11))
FOLD = {r: V[(V.cfg == 0) & (V.rep == r)].set_index("Sample").fold.reindex(C.index).values for r in REPS}
for c in V.cfg.unique():   # every arm shares the outer folds
    for r in REPS: assert (V[(V.cfg == c) & (V.rep == r)].set_index("Sample").fold.reindex(C.index).values == FOLD[r]).all(), (c, r)
def per_rep(cfg): return {r: V[(V.cfg == cfg) & (V.rep == r)].set_index("Sample").pred.reindex(C.index).values for r in REPS}
CFG = {"their": {"L-CNV": 0, "L-EARLY": 3}, "pkg": {"L-CNV": 1, "L-EARLY": 4}}
def arms_for(src):
    cn, im, ea = per_rep(CFG[src]["L-CNV"]), per_rep(2), per_rep(CFG[src]["L-EARLY"]); return {"L-CNV": cn, "L-IMG": im, "L-EARLY": ea, "L-LATE": {r: (cn[r] + im[r]) / 2 for r in REPS}}
IO = {r: V[(V.cfg == 0) & (V.rep == r)].set_index("Sample").train_prev.reindex(C.index).values for r in REPS}
def fs_auc(sc, rows):   # fold-stratified AUROC: within-fold P-NP pairs pooled over folds, mean over repeats; rows may repeat (bootstrap)
    y = yv[rows]; out = []
    for r in REPS:
        f = FOLD[r][rows]; s = sc[r][rows]; num = den = 0.0
        for k in range(1, 11):
            m = f == k; yk = y[m]; n1 = yk.sum(); n0 = len(yk) - n1
            if n1 == 0 or n0 == 0: continue
            rk = rankdata(s[m]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
        out.append(num / den)
    return float(np.mean(out))
def swap(a, b, mk): return {r: np.where(mk, b[r], a[r]) for r in REPS}, {r: np.where(mk, a[r], b[r]) for r in REPS}
res = {"task": TASK}
if TASK.startswith("fs_"):
    _, src, sub = TASK.split("_", 2); ev = Ev(SUB[sub]); A_ = arms_for(src); rows = ev.r
    o = {"n_samples": int(len(rows)), "n_patients": int(len(ev.pp)), "n_P_patients": int(ev.yp.sum())}
    for a, sc in A_.items():
        pt = fs_auc(sc, rows); bs = np.array([fs_auc(sc, rr) for rr in ev.bootrows]); d = {"fold_stratified_auroc": r3(pt), "ci95": [r3(np.percentile(bs, 2.5)), r3(np.percentile(bs, 97.5))]}
        if a != "L-CNV":
            ref = A_["L-CNV"]; obs = pt - fs_auc(ref, rows); bd = bs - np.array([fs_auc(ref, rr) for rr in ev.bootrows])
            pm = np.array([fs_auc(x, rows) - fs_auc(y_, rows) for x, y_ in (swap(sc, ref, mk) for mk in ev.masks)])
            d["delta_vs_L_CNV"] = {"delta": r3(obs), "ci95": [r3(np.percentile(bd, 2.5)), r3(np.percentile(bd, 97.5))], "perm_p": round(float((1 + (np.abs(pm) >= abs(obs) - 1e-12).sum()) / 2001), 4)}
        o[a] = d
    o["intercept_only_fold_stratified_auroc"] = r3(fs_auc(IO, rows)); res["result"] = o
elif TASK == "maxt":
    res["result"] = {}
    for src in ["their", "pkg"]:
        A_ = arms_for(src); mean = {a: np.mean([v[r] for r in REPS], 0) for a, v in A_.items()}
        for sub in ["all_C", "a_NDBE_only"]:
            ev = Ev(SUB[sub]); rows = ev.r; o = {}
            obs_p = {a: np.subtract(ev.stats(mean[a]), ev.stats(mean["L-CNV"])) for a in ["L-IMG", "L-EARLY", "L-LATE"]}
            pd_p = {a: ev.perm_deltas(mean[a], mean["L-CNV"]) for a in obs_p}
            obs_f = {a: fs_auc(A_[a], rows) - fs_auc(A_["L-CNV"], rows) for a in obs_p}
            pd_f = {a: np.array([fs_auc(x, rows) - fs_auc(y_, rows) for x, y_ in (swap(A_[a], A_["L-CNV"], mk) for mk in ev.masks)]) for a in obs_p}
            mxp = np.max(np.abs(np.stack([pd_p[a] for a in obs_p])), axis=0); mxf = np.max(np.abs(np.stack([pd_f[a] for a in obs_p])), axis=0)
            for a in obs_p:
                o[a] = {}
                for j, k in enumerate(["per_sample_auroc", "patient_max_auroc"]):
                    o[a][k] = {"delta": r3(obs_p[a][j]), "p_unadjusted": round(float((1 + (np.abs(pd_p[a][:, j]) >= abs(obs_p[a][j]) - 1e-12).sum()) / 2001), 4), "p_max_T_adjusted": round(float((1 + (mxp[:, j] >= abs(obs_p[a][j]) - 1e-12).sum()) / 2001), 4)}
                o[a]["fold_stratified_auroc"] = {"delta": r3(obs_f[a]), "p_unadjusted": round(float((1 + (np.abs(pd_f[a]) >= abs(obs_f[a]) - 1e-12).sum()) / 2001), 4), "p_max_T_adjusted": round(float((1 + (mxf >= abs(obs_f[a]) - 1e-12).sum()) / 2001), 4)}
            res["result"][f"{src}_{sub}"] = o; print("maxt", src, sub, flush=True)
json.dump(res, open(outf, "w"), indent=1); print("KF STATS DONE", TASK)
