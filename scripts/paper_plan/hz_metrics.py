"""Horizons H1 (docs/paper_survival_horizons.md @ ee51db8): time-dependent AUROC (IPCW Uno-type and unweighted) at 1/3/5 years, Harrell's and
Uno's C, paired deltas vs L-CNV and vs L-LATE, swap-permutation p and single-step max-T p over the arms vs L-LATE. No refit: scores are the
stratified-CV out-of-fold predictions (kv_cv.R cfg 0-5; hz_fit.R outer for L-INTER and L-GRADE+age+sex), mean over the 10 repeats.
TASK = <src>_<pop>, src in their|pkg, pop in pre|pre_ndbe|pre_nofallback. Aggregates only -> results/paper_final/horizons/h1_<TASK>.json."""
import json, os, sys, glob, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"
OUT = T + "/results/paper_final/horizons"; os.makedirs(OUT, exist_ok=True)
TASK = os.environ["TASK"]; SRC, POP = TASK.split("_", 1)
HORIZONS = [1.0, 3.0, 5.0]; NB = 2000
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); C = pd.read_csv(M + "/set_C.csv", dtype=str)
assert list(SM.Sample) == list(C.Sample); n = len(SM)
idx = pd.Series(np.arange(n), index=SM.Sample)
# ---------- scores (mean over repeats), arms of this CNV source
def load(files, model=None, cnvsrc=None):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in files])
    if model is not None: D = D[(D.model == model) & (D.cnvsrc == cnvsrc)]
    P = D.pivot_table(index="Sample", columns="rep", values="pred"); assert P.shape == (n, 10), (files[0], P.shape); return P.reindex(SM.Sample).values, D
kvf = lambda cfg: sorted(glob.glob(f"{M}/cv/preds/cfg_{cfg:02d}_rep_*.csv"))
ci_cnv = 0 if SRC == "their" else 1; ci_early = 3 if SRC == "their" else 4
R = {}; R["L-CNV"], kvd = load(kvf(ci_cnv)); R["L-IMG"], _ = load(kvf(2)); R["L-EARLY"], _ = load(kvf(ci_early)); R["L-GRADE"], _ = load(kvf(5))
R["L-INTER"], di = load(sorted(glob.glob(f"{HZ}/outer/inter_{SRC}_rep_*.csv"))); R["L-GRADE+age+sex"], _ = load(sorted(glob.glob(f"{HZ}/outer/gradeagesex_rep_*.csv")))
R["L-LATE"] = (R["L-CNV"] + R["L-IMG"]) / 2
fk = kvd.set_index(["Sample", "rep"]).fold; fi = di.set_index(["Sample", "rep"]).fold
fold_check = int((fk.reindex(fi.index).values == fi.values).sum()), int(len(fi))
ARMS = ["L-GRADE", "L-GRADE+age+sex", "L-CNV", "L-IMG", "L-EARLY", "L-INTER", "L-LATE"]; OTHERS = [a for a in ARMS if a != "L-LATE"]
S = {a: R[a].mean(1) for a in ARMS}
# ---------- population
pre = SM.pre.astype(bool).values; ndbe = SM.ndbe.astype(bool).values; fb = SM.fallback_endpoint.astype(bool).values
mask = {"pre": pre, "pre_ndbe": pre & ndbe, "pre_nofallback": pre & ~SM.Patient.isin(SM.Patient[fb]).values}[POP]
time = SM.time.values.astype(float); ev = SM.event.values.astype(int); pat = SM.Patient.values
def earliest_ndbe(m):
    r = np.where(m & ndbe)[0]; d = pd.DataFrame({"r": r, "p": pat[r], "mbf": SM.mbf.values[r]}).sort_values(["p", "mbf", "r"], ascending=[True, False, True])
    return np.sort(d.groupby("p").r.first().values)
UNITS = {"sample": np.where(mask)[0], "patient": earliest_ndbe(mask)}
# ---------- estimators
def Gminus(tt, dd, at):
    """reverse KM of censoring evaluated just before each time in `at`"""
    u = np.unique(tt); nrisk = (tt[None, :] >= u[:, None]).sum(1); dc = ((tt[None, :] == u[:, None]) & (dd[None, :] == 0)).sum(1)
    surv = np.cumprod(1 - dc / nrisk); prev = np.concatenate([[1.0], surv[:-1]])   # value just before u
    k = np.searchsorted(u, at, side="left"); out = np.ones(len(at)); inside = k < len(u)
    out[inside] = prev[k[inside]]; out[~inside] = surv[-1]; return np.clip(out, 1e-12, None)
def labels(tt, dd, t):
    case = (dd == 1) & (tt <= t); ctrl = ((dd == 0) & (tt >= t)) | ((dd == 1) & (tt > t)); return case, ctrl
def tdauc(sc, tt, dd, t, weighted):
    case, ctrl = labels(tt, dd, t)
    if case.sum() == 0 or ctrl.sum() == 0: return np.nan
    w = 1 / Gminus(tt, dd, tt[case]) if weighted else np.ones(case.sum())
    cs = np.sort(sc[ctrl]); m = sc[case]; conc = np.searchsorted(cs, m, "left") + 0.5 * (np.searchsorted(cs, m, "right") - np.searchsorted(cs, m, "left"))
    return float((w * conc).sum() / (w.sum() * len(cs)))
def cidx(sc, tt, dd, uno):
    ii = np.where(dd == 1)[0]
    if len(ii) == 0: return np.nan
    tau = tt[ii].max(); w = (1 / Gminus(tt, dd, tt[ii]) ** 2) if uno else np.ones(len(ii)); keep = tt[ii] <= tau
    comp = (tt[ii][:, None] < tt[None, :]); conc = (sc[ii][:, None] > sc[None, :]) + 0.5 * (sc[ii][:, None] == sc[None, :])
    num = (w[:, None] * comp * conc)[keep].sum(); den = (w[:, None] * comp)[keep].sum(); return float(num / den) if den > 0 else np.nan
def pctl(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return [r3(np.percentile(a, 2.5)), r3(np.percentile(a, 97.5))] if len(a) else [None, None]
res = {"task": TASK, "fold_check_inter_vs_kv": fold_check, "arms": ARMS, "horizons": HORIZONS, "units": {}}
for unit, rows in UNITS.items():
    tt, dd, pp = time[rows], ev[rows], pat[rows]; up = np.unique(pp); rows_of = {q: np.where(pp == q)[0] for q in up}
    rng = np.random.RandomState(0); boots = [np.concatenate([rows_of[up[i]] for i in rng.choice(len(up), len(up))]) for _ in range(NB)]
    sc = {a: S[a][rows] for a in ARMS}; U = {"n_rows": int(len(rows)), "n_patients": int(len(up)), "n_event_rows": int(dd.sum()), "n_event_patients": int(len(set(pp[dd == 1]))), "horizons": {}, "cindex": {}}
    for t in HORIZONS:
        case, ctrl = labels(tt, dd, t)
        H = {"n_cases": int(case.sum()), "n_controls": int(ctrl.sum()), "n_excluded": int((~case & ~ctrl).sum()), "n_case_patients": int(len(set(pp[case]))), "n_control_patients": int(len(set(pp[ctrl]))), "arms": {}}
        if case.sum() == 0 or ctrl.sum() == 0: U["horizons"][str(int(t))] = H; continue
        for wtd, key in ((True, "ipcw"), (False, "unweighted")):
            obs = {a: tdauc(sc[a], tt, dd, t, wtd) for a in ARMS}
            bs = {a: np.array([tdauc(sc[a][b], tt[b], dd[b], t, wtd) for b in boots]) for a in ARMS}
            for a in ARMS:
                o = H["arms"].setdefault(a, {}); o[key] = {"auroc": r3(obs[a]), "ci95": pctl(bs[a]), "valid_draws": int(np.isfinite(bs[a]).sum())}
                for ref in ("L-CNV", "L-LATE"):
                    if a != ref: o[key][f"delta_vs_{ref}"] = {"delta": r3(obs[a] - obs[ref]), "ci95": pctl(bs[a] - bs[ref])}
            if unit == "sample" and wtd:   # swap permutation, single-step max-T over the 6 arms vs L-LATE
                prs = np.random.RandomState(0); w = 1 / Gminus(tt, dd, tt[case]); cs_idx = np.where(ctrl)[0]; ca_idx = np.where(case)[0]
                def fast(s): cs = np.sort(s[cs_idx]); m = s[ca_idx]; lo = np.searchsorted(cs, m, "left"); hi = np.searchsorted(cs, m, "right"); return float((w * (lo + 0.5 * (hi - lo))).sum() / (w.sum() * len(cs)))
                pmask = []
                for _ in range(NB):
                    sw = prs.rand(len(up)) < 0.5; mk = np.zeros(len(rows), bool)
                    for i in np.where(sw)[0]: mk[rows_of[up[i]]] = True
                    pmask.append(mk)
                pdl = {}
                for a in OTHERS:
                    x, z = sc[a], sc["L-LATE"]; pdl[a] = np.array([fast(np.where(mk, z, x)) - fast(np.where(mk, x, z)) for mk in pmask])
                mx = np.max(np.abs(np.stack([pdl[a] for a in OTHERS])), 0)
                for a in OTHERS:
                    d0 = obs[a] - obs["L-LATE"]; o = H["arms"][a]["ipcw"]["delta_vs_L-LATE"]
                    o["p_unadjusted"] = round(float((1 + (np.abs(pdl[a]) >= abs(d0) - 1e-12).sum()) / (NB + 1)), 4); o["p_max_T"] = round(float((1 + (mx >= abs(d0) - 1e-12).sum()) / (NB + 1)), 4)
        U["horizons"][str(int(t))] = H; print(unit, t, H["n_cases"], H["n_controls"], flush=True)
    for a in ARMS:
        for key, uno in (("harrell", False), ("uno", True)):
            obs = cidx(sc[a], tt, dd, uno); bs = [cidx(sc[a][b], tt[b], dd[b], uno) for b in boots]
            U["cindex"].setdefault(a, {})[key] = {"c": r3(obs), "ci95": pctl(bs)}
    res["units"][unit] = U
json.dump(res, open(f"{OUT}/h1_{TASK}.json", "w"), indent=1); print("HZ METRICS DONE", TASK)
