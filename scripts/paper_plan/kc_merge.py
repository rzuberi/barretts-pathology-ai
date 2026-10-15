"""Killcoyne checks, items 1-4 (docs/paper_plan_killcoyne_checks.md @ 828ebf7). Existing multimodal predictions + the kc_linear.R arms.
Aggregates only -> results/paper_final/killcoyne_checks.json."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import rankdata
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; K = T + "/feasibility/paper_plan/killcoyne"; M = K + "_mm"; OUT = os.environ.get("OUTDIR", T + "/results/paper_final")
S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
C = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample"); C["y"] = (C.Status == "P").astype(int)
pats = np.array(sorted(C.Patient.unique())); pid = pd.Series(np.arange(len(pats)), index=pats)[C.Patient.values].values; yv = C.y.values; NP_ = len(pats)
ypat_all = np.zeros(NP_, int); np.maximum.at(ypat_all, pid, yv)
# ---- predictions
L = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in sorted(glob.glob(M + "/linear/*.csv"))])
Q = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in sorted(glob.glob(M + "/checks/linear/*.csv"))])
N = pd.concat([pd.read_csv(f, dtype={"sample_id": str, "patient_id": str}) for f in sorted(glob.glob(M + "/neural/*.csv"))])
def lin(D, cfg, rule="class_min"):
    v = D[(D.cfg == cfg) & (D.rule == rule)].drop_duplicates("Sample").set_index("Sample").pred.reindex(C.index); assert v.notna().all(), (cfg, rule); return v.values
def neu(m):
    d = N[N.model == m]; ps = [q.drop_duplicates("sample_id").set_index("sample_id").y_prob.reindex(C.index).values for _, q in d.groupby("seed")]; assert len(ps) == 3 and all(np.isfinite(p).all() for p in ps), m; return np.mean(ps, 0)
arms = {}
for src, cc, ce, ci_, fe, cg, eg in [("their", 0, 2, 3, 1, 4, 6), ("pkg", 4, 5, 6, 2, 5, 7)]:
    cn, im = lin(L, cc), lin(L, 1); nimg = neu("img")
    arms[src] = {"L-CNV": cn, "L-IMG": im, "L-EARLY": lin(L, ce), "L-INTER": lin(L, ci_), "L-LATE": (cn + im) / 2, "N-IMG": nimg, "N-EARLY": neu(f"early_{src}"), "N-INTER": neu(f"inter_{src}"), "N-LATE": (nimg + cn) / 2,
                 "L-IMG (fold-honest)": lin(Q, 0), "L-EARLY (fold-honest)": lin(Q, fe), "L-LATE (fold-honest)": (cn + lin(Q, 0)) / 2,
                 "L-GRADE": lin(Q, 3), "L-CNV+GRADE": lin(Q, cg), "L-EARLY+GRADE": lin(Q, eg), "L-LATE with grade": (lin(Q, cg) + im) / 2}
EIGHT = ["L-IMG", "L-EARLY", "L-INTER", "L-LATE", "N-IMG", "N-EARLY", "N-INTER", "N-LATE"]
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
res = {"_spec": "docs/paper_plan_killcoyne_checks.md @ 828ebf7", "subsets": {k: {"n_samples": int(m.sum()), "n_patients": int(len(E[k].pp)), "n_P_patients": int(E[k].yp.sum())} for k, m in SUB.items()},
       "event_definition": {"P_patients_in_C": len(Ppats), "dated_from_HGD_IMC_record": int(len([p for p in Ppats if p not in fallback])), "fallback_final_endoscopy": len(fallback)}}
# item 1: max-T over the eight arms, full set C
res["item1_max_T"] = {}
for src in ["their", "pkg"]:
    ev = E["all_C"]; ref = arms[src]["L-CNV"]; obs = {}; pd_ = {}
    for a in EIGHT: _, obs[a], pd_[a] = ev.delta(arms[src][a], ref)
    mx = np.max(np.abs(np.stack([pd_[a] for a in EIGHT])), axis=0)   # 2000 x 2
    res["item1_max_T"][src] = {a: {k: {"delta": r3(obs[a][j]), "p_unadjusted": round(float((1 + (np.abs(pd_[a][:, j]) >= abs(obs[a][j]) - 1e-12).sum()) / 2001), 4),
                                       "p_max_T_adjusted": round(float((1 + (mx[:, j] >= abs(obs[a][j]) - 1e-12).sum()) / 2001), 4)} for j, k in enumerate(["per_sample_auroc", "patient_max_auroc"])} for a in EIGHT}
# items 2 and 3: every arm on every subset, delta vs L-CNV
ARMS23 = EIGHT + ["L-IMG (fold-honest)", "L-EARLY (fold-honest)", "L-LATE (fold-honest)"]
res["items2_3"] = {src: {sub: {"L-CNV": E[sub].summary(arms[src]["L-CNV"])} for sub in SUB} for src in ["their", "pkg"]}
for src in ["their", "pkg"]:
    for sub in SUB:
        for a in ARMS23:
            o = E[sub].summary(arms[src][a]); o["delta_vs_L_CNV"] = E[sub].delta(arms[src][a], arms[src]["L-CNV"])[0]; res["items2_3"][src][sub][a] = o
        print("done", src, sub, flush=True)
# item 4: grade
res["item4"] = {}
for src in ["their", "pkg"]:
    res["item4"][src] = {}
    for sub in ["all_C", "a_NDBE_only"]:
        o = {a: E[sub].summary(arms[src][a]) for a in ["L-GRADE", "L-CNV+GRADE", "L-EARLY+GRADE", "L-LATE with grade", "L-CNV"]}
        for a in ["L-EARLY+GRADE", "L-LATE with grade"]: o[a]["delta_vs_L_CNV+GRADE"] = E[sub].delta(arms[src][a], arms[src]["L-CNV+GRADE"])[0]
        o["L-CNV+GRADE"]["delta_vs_L_CNV"] = E[sub].delta(arms[src]["L-CNV+GRADE"], arms[src]["L-CNV"])[0]
        res["item4"][src][sub] = o
fz = glob.glob(T + "/models/killcoyne_frozen_v1/MANIFEST.json")
if fz: res["item5_manifest_bundle_sha256"] = json.load(open(fz[0]))["bundle_sha256"]
os.makedirs(OUT, exist_ok=True); json.dump(res, open(OUT + "/killcoyne_checks.json", "w"), indent=1); print("CHECKS MERGE DONE")
