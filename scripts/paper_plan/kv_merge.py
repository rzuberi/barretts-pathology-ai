"""Killcoyne stratified CV, items 1-4 merge (docs/paper_plan_killcoyne_cv.md @ 76879b8). Stratified-CV predictions (kv_cv.R), LOPO intercepts
(kv_lopo_intercepts.R), LOPO values from results/paper_final/killcoyne_checks.json, frozen package models. Aggregates only ->
results/paper_final/killcoyne_cv.json. Subset definitions and the evaluation class are copied from kc_merge.py."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import rankdata, pearsonr, spearmanr
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; K = T + "/feasibility/paper_plan/killcoyne"; M = K + "_mm"; OUT = os.environ.get("OUTDIR", T + "/results/paper_final")
S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
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
E = {k: Ev(m) for k, m in SUB.items()}
V = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in sorted(glob.glob(M + "/cv/preds/*.csv"))])
assert V.groupby("cfg").rep.nunique().eq(10).all() and V.cfg.nunique() == 10, "incomplete CV"
def cvscore(cfg):
    w = V[V.cfg == cfg].pivot_table(index="Sample", columns="rep", values="pred").reindex(C.index); assert w.notna().all().all(); return w.mean(1).values
arms = {}
for src, c_cnv, c_early, c_cg, c_eg in [("their", 0, 3, 6, 8), ("pkg", 1, 4, 7, 9)]:
    cn, im, cg = cvscore(c_cnv), cvscore(2), cvscore(c_cg)
    arms[src] = {"L-CNV": cn, "L-IMG": im, "L-EARLY": cvscore(c_early), "L-LATE": (cn + im) / 2, "L-GRADE": cvscore(5), "L-CNV+GRADE": cg, "L-EARLY+GRADE": cvscore(c_eg), "L-LATE with grade": (cg + im) / 2}
chk = json.load(open(T + "/results/paper_final/killcoyne_checks.json"))   # committed checks results (OUTDIR may be a run folder)
res = {"_spec": "docs/paper_plan_killcoyne_cv.md @ 76879b8", "subsets": {k: {"n_samples": int(m.sum()), "n_patients": int(len(E[k].pp)), "n_P_patients": int(E[k].yp.sum())} for k, m in SUB.items()},
       "fold_sizes": V[V.rep == 1].drop_duplicates("Patient").groupby("fold").agg(patients=("Patient", "size"), P=("y", "sum")).to_dict("index")}
LOPO_KEY = {"L-CNV": "L-CNV", "L-IMG": "L-IMG (fold-honest)", "L-EARLY": "L-EARLY (fold-honest)", "L-LATE": "L-LATE (fold-honest)"}
res["item1"] = {}
for src in ["their", "pkg"]:
    res["item1"][src] = {}
    for sub in SUB:
        o = {}
        for a in ["L-CNV", "L-IMG", "L-EARLY", "L-LATE"]:
            o[a] = E[sub].summary(arms[src][a])
            if a != "L-CNV": o[a]["delta_vs_L_CNV"] = E[sub].delta(arms[src][a], arms[src]["L-CNV"])[0]
            lo = chk["items2_3"][src][sub][LOPO_KEY[a]]; o[a]["LOPO"] = {"per_sample_auroc": lo["per_sample_auroc"], "patient_max_auroc": lo["patient_max_auroc"], "delta_vs_L_CNV": lo.get("delta_vs_L_CNV")}
        res["item1"][src][sub] = o; print("item1", src, sub, flush=True)
# item 2
LI = pd.concat([pd.read_csv(f, dtype={"Patient": str}) for f in sorted(glob.glob(M + "/cv/lopo_intercepts/*.csv"))])
cvint = V.groupby(["cfg", "Patient"]).apply(lambda d: d.drop_duplicates("rep").intercept.mean()).rename("intercept").reset_index()
CVNAME = {0: "L-CNV their", 1: "L-CNV pkg", 2: "L-IMG fold-honest", 3: "L-EARLY fold-honest their", 4: "L-EARLY fold-honest pkg", 5: "L-GRADE", 6: "L-CNV+GRADE their", 7: "L-CNV+GRADE pkg", 8: "L-EARLY+GRADE their", 9: "L-EARLY+GRADE pkg"}
lab = pd.Series(ypat_all, index=pats)
def corr(d): x = lab.reindex(d.Patient).values; v = d.intercept.values; return {"pearson": r3(pearsonr(x, v)[0]), "spearman": r3(spearmanr(x, v).correlation), "n_patients": int(len(d))}
res["item2"] = {"intercept_label_correlation": {}, "lopo_refit_max_abs_pred_diff": r3(float(LI.max_abs_pred_diff.max())) if len(LI) else None}
for c, nm in CVNAME.items():
    res["item2"]["intercept_label_correlation"][nm] = {"LOPO": corr(LI[LI.arm == nm]) if (LI.arm == nm).any() else None, "stratified_CV": corr(cvint[cvint.cfg == c])}
lopo_io = np.array([yv[pid != pid[i]].mean() for i in range(len(yv))])
cv_io = V[V.cfg == 0].pivot_table(index="Sample", columns="rep", values="train_prev").reindex(C.index).mean(1).values
res["item2"]["intercept_only_per_sample_auroc"] = {sub: {"LOPO": r3(auc(yv[E[sub].r], lopo_io[E[sub].r])), "stratified_CV": r3(auc(yv[E[sub].r], cv_io[E[sub].r]))} for sub in SUB}
# item 3
gr = C.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).astype(float).values
res["item3"] = {"raw_grade": {sub: E[sub].summary(gr) for sub in ["all_C", "a_NDBE_only"]}}
for src in ["their", "pkg"]:
    res["item3"][src] = {}
    for sub in ["all_C", "a_NDBE_only"]:
        o = {a: E[sub].summary(arms[src][a]) for a in ["L-GRADE", "L-CNV+GRADE", "L-EARLY+GRADE", "L-LATE with grade", "L-CNV"]}
        for a in ["L-EARLY+GRADE", "L-LATE with grade"]: o[a]["delta_vs_L_CNV+GRADE"] = E[sub].delta(arms[src][a], arms[src]["L-CNV+GRADE"])[0]
        o["L-CNV+GRADE"]["delta_vs_L_CNV"] = E[sub].delta(arms[src]["L-CNV+GRADE"], arms[src]["L-CNV"])[0]; res["item3"][src][sub] = o
# item 4: development estimate of the ACE-B primary contrast (one-sided lower bound from the same patient bootstrap)
ev = E["a_NDBE_only"]; a, b = arms["pkg"]["L-LATE"], arms["pkg"]["L-CNV"]
bs = np.array([auc(yv[rr], a[rr]) - auc(yv[rr], b[rr]) for rr in ev.bootrows]); obs = ev.stats(a)[0] - ev.stats(b)[0]
res["item4_development_estimate"] = {"contrast": "L-LATE (pkg) minus L-CNV (pkg), per-sample AUROC, NDBE samples, stratified CV", "delta": r3(obs), "one_sided_95_lower": r3(np.percentile(bs, 5)), "two_sided_95": [r3(np.percentile(bs, 2.5)), r3(np.percentile(bs, 97.5))], "n_samples": int(len(ev.r)), "n_patients": int(len(ev.pp)), "n_P_patients": int(ev.yp.sum())}
fz = T + "/models/killcoyne_frozen_pkg_v1/MANIFEST.json"
if os.path.exists(fz): res["item4_bundle_sha256"] = json.load(open(fz))["bundle_sha256"]
os.makedirs(OUT, exist_ok=True); json.dump(res, open(OUT + "/killcoyne_cv.json", "w"), indent=1); print("KV MERGE DONE")
