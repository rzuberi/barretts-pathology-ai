"""Fold-stratified IPCW time-dependent AUROC (docs/paper_horizon_foldstrat.md @ 2d0014f). For each repeat, only case-control pairs whose two samples
were scored by the same outer fold model; IPCW case weights 1/G(T-) from the reverse Kaplan-Meier over the evaluated samples; mean over 10 repeats.
No refit: kv_cv.R (cfg 0-4), hz_fit.R outer (L-INTER), ha_clin.py (L-CLIN), intercept-only = train_prev. Patient bootstrap (2,000, RandomState(0)),
swap permutation (2,000, seed 0) with single-step max-T. TASK = <src>_<pop>_<t>, src their|pkg, pop pre|pre_ndbe, t 1|3|5.
Aggregates -> results/paper_final/horizon_foldstrat/fs_<TASK>.json"""
import json, os, glob, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"; HA = M + "/horizon_answers"
OUT = T + "/results/paper_final/horizon_foldstrat"; os.makedirs(OUT, exist_ok=True); TASK = os.environ["TASK"]; NB = 2000; REPS = list(range(1, 11))
SRC, rest = TASK.split("_", 1); POP, TT = rest.rsplit("_", 1); t = float(TT)
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM)
def per_rep(files, model=None, col="pred"):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files])
    if model is not None: D = D[D.model == model]
    P = D.pivot_table(index="Sample", columns="rep", values=col).reindex(SM.Sample); Fo = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(SM.Sample)
    assert P.shape == (n, 10) and P.notna().all().all(); return {r: P[r].values for r in REPS}, {r: Fo[r].values.astype(int) for r in REPS}
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv"))
A = {}; A["L-CNV"], FOLD = per_rep(kvf(0 if SRC == "their" else 1)); A["L-IMG"], f2 = per_rep(kvf(2)); A["L-EARLY"], f3_ = per_rep(kvf(3 if SRC == "their" else 4))
A["L-INTER"], f4 = per_rep(sorted(glob.glob(f"{HZ}/outer/inter_{SRC}_rep_*.csv"))); A["L-CLIN"], f5 = per_rep([HA + "/clin_outer.csv"], "clin"); A["L-CLIN-nograde"], f6 = per_rep([HA + "/clin_outer.csv"], "clin_nograde")
A["INTERCEPT-ONLY"], _ = per_rep(kvf(0), col="train_prev")
fold_agree = {nm: int(sum((FOLD[r] == ff[r]).all() for r in REPS)) for nm, ff in (("L-IMG", f2), ("L-EARLY", f3_), ("L-INTER", f4), ("L-CLIN", f5), ("L-CLIN-nograde", f6))}
assert all(v == 10 for v in fold_agree.values()), fold_agree
A["L-LATE"] = {r: (A["L-CNV"][r] + A["L-IMG"][r]) / 2 for r in REPS}
ARMS = ["L-CLIN", "L-CLIN-nograde", "L-CNV", "L-IMG", "L-EARLY", "L-INTER", "L-LATE", "INTERCEPT-ONLY"]; TABLE = ["L-CLIN", "L-CNV", "L-IMG", "L-EARLY", "L-INTER", "L-LATE"]
FAM = {"L-LATE": ["L-CNV", "L-IMG", "L-EARLY", "L-INTER"], "L-CNV": ["L-IMG", "L-EARLY", "L-INTER", "L-LATE"]}
pre = SM.pre.astype(bool).values; ndbe = SM.ndbe.astype(bool).values; mask = {"pre": pre, "pre_ndbe": pre & ndbe}[POP]
time = SM.time.values.astype(float); ev = SM.event.values.astype(int); pat = SM.Patient.values
def Gminus(tt, dd, at):
    u = np.unique(tt); nr = (tt[None, :] >= u[:, None]).sum(1); dc = ((tt[None, :] == u[:, None]) & (dd[None, :] == 0)).sum(1); sv = np.cumprod(1 - dc / nr); pv = np.concatenate([[1.0], sv[:-1]])
    k = np.searchsorted(u, at, side="left"); o = np.ones(len(at)); i = k < len(u); o[i] = pv[k[i]]; o[~i] = sv[-1]; return np.clip(o, 1e-12, None)
def setup(rows):
    tt, dd = time[rows], ev[rows]; case = (dd == 1) & (tt <= t); ctrl = ((dd == 0) & (tt >= t)) | ((dd == 1) & (tt > t))
    if not case.sum() or not ctrl.sum(): return None
    ci, co = np.where(case)[0], np.where(ctrl)[0]; return {"rows": rows, "ci": ci, "co": co, "w": 1 / Gminus(tt, dd, tt[case])}
def fs(arm_rep, S, weighted=True, swap=None):
    """arm_rep: dict r -> scores over all n samples (or a callable returning per-row scores); S: setup dict"""
    rows, ci, co = S["rows"], S["ci"], S["co"]; w = S["w"] if weighted else np.ones(len(ci)); out = []
    for r in REPS:
        s = arm_rep[r][rows] if swap is None else swap(r); f = FOLD[r][rows]
        kc = np.sort(f[co] * 4.0 + s[co]); fi, si = f[ci] * 4.0, s[ci]
        base = np.searchsorted(kc, fi, "left"); lo = np.searchsorted(kc, fi + si, "left") - base; hi = np.searchsorted(kc, fi + si, "right") - base; nct = np.searchsorted(kc, fi + 2.0, "left") - base
        den = (w * nct).sum(); out.append((w * (lo + 0.5 * (hi - lo))).sum() / den if den > 0 else np.nan)
    return float(np.nanmean(out))
def pctl(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return [r3(np.percentile(a, 2.5)), r3(np.percentile(a, 97.5))] if len(a) else [None, None]
rows0 = np.where(mask)[0]; S0 = setup(rows0); pp = pat[rows0]; up = np.unique(pp); ro = {q: rows0[pp == q] for q in up}
res = {"task": TASK, "fold_agreement_repeats": fold_agree, "n_cases": int(len(S0["ci"])), "n_controls": int(len(S0["co"])), "n_case_patients": int(len(set(pat[rows0][S0["ci"]]))), "n_control_patients": int(len(set(pat[rows0][S0["co"]]))), "arms": {}}
f_ = FOLD[1][rows0]; res["within_fold_pairs_repeat1"] = int(sum(((f_[S0["ci"]] == k).sum()) * ((f_[S0["co"]] == k).sum()) for k in range(1, 11))); res["all_pairs"] = int(len(S0["ci"]) * len(S0["co"]))
obs = {a: fs(A[a], S0) for a in ARMS}; obs_u = {a: fs(A[a], S0, weighted=False) for a in ARMS}
rng = np.random.RandomState(0); boots = []
while len(boots) < NB:
    rb = np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))]); boots.append(setup(rb))
valid = [b for b in boots if b is not None]
bs = {a: np.array([fs(A[a], b) for b in valid]) for a in ARMS}; bsu = {a: np.array([fs(A[a], b, weighted=False) for b in valid]) for a in ARMS}
for a in ARMS:
    o = {"ipcw": r3(obs[a]), "ipcw_ci95": pctl(bs[a]), "unweighted": r3(obs_u[a]), "unweighted_ci95": pctl(bsu[a])}
    for ref in ("L-LATE", "L-CNV"):
        if a != ref and a != "INTERCEPT-ONLY": o[f"delta_vs_{ref}"] = {"delta": r3(obs[a] - obs[ref]), "ci95": pctl(bs[a] - bs[ref])}
    res["arms"][a] = o
res["valid_bootstrap_draws"] = len(valid)
prs = np.random.RandomState(0); idx_of = {q: np.where(pp == q)[0] for q in up}; masks = []
for _ in range(NB):
    sw = prs.rand(len(up)) < 0.5; mk = np.zeros(len(rows0), bool)
    for i in np.where(sw)[0]: mk[idx_of[up[i]]] = True
    masks.append(mk)
for ref, fam in FAM.items():
    pdl = {}
    for a in fam:
        d = []
        for mk in masks:
            x = fs(None, S0, swap=lambda r, mk=mk: np.where(mk, A[ref][r][rows0], A[a][r][rows0])); y = fs(None, S0, swap=lambda r, mk=mk: np.where(mk, A[a][r][rows0], A[ref][r][rows0])); d.append(x - y)
        pdl[a] = np.array(d)
    mx = np.max(np.abs(np.stack([pdl[a] for a in fam])), 0)
    for a in fam:
        d0 = obs[a] - obs[ref]; o = res["arms"][a][f"delta_vs_{ref}"]
        o["p_unadjusted"] = round(float((1 + (np.abs(pdl[a]) >= abs(d0) - 1e-12).sum()) / (NB + 1)), 4); o["p_max_T"] = round(float((1 + (mx >= abs(d0) - 1e-12).sum()) / (NB + 1)), 4)
res["ranking"] = sorted(TABLE, key=lambda a: -obs[a]); res["best"] = res["ranking"][0]
json.dump(res, open(f"{OUT}/fs_{TASK}.json", "w"), indent=1); print("HF DONE", TASK, res["best"], {a: res["arms"][a]["ipcw"] for a in ARMS})
