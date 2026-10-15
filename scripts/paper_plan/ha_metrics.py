"""Horizon answers Q0/Q1 (docs/paper_horizon_answers.md @ 81473df): IPCW and unweighted time-dependent AUROC at 1/3/5 years, Harrell's C, paired deltas
vs L-LATE and vs L-CNV, swap-permutation p with single-step max-T over the 5 non-clinical arms (Δ vs L-LATE: family = L-CNV, L-IMG, L-EARLY, L-INTER;
Δ vs L-CNV: family = L-IMG, L-EARLY, L-INTER, L-LATE). Estimators copied from hz_metrics.py. Scores: kv_cv.R out-of-fold (cfg 0-4), hz_fit.R outer
(L-INTER), ha_clin.py (L-CLIN, L-CLIN without grade); mean over 10 repeats. TASK = <src>_<pop> (pre | pre_ndbe | pre_nofallback) or check.
Aggregates -> results/paper_final/horizon_answers/q0_<TASK>.json"""
import json, os, glob, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"; HA = M + "/horizon_answers"
OUT = T + "/results/paper_final/horizon_answers"; os.makedirs(OUT, exist_ok=True); TASK = os.environ["TASK"]; NB = 2000; HORIZONS = [1.0, 3.0, 5.0]
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM)
if TASK == "check":   # Months before final: 777 sheet (used) vs Source Data MOESM4 Fig 2d and MOESM11
    out = {}
    for f, sh in (("41591_2020_1033_MOESM4_ESM.xlsx", "Supporting data for Figure 2d"), ("41591_2020_1033_MOESM11_ESM.xlsx", "Ex Data Fig 5")):
        X0 = pd.read_excel("/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper/" + f, sheet_name=sh, header=None)
        h = next(i for i in range(len(X0)) if str(X0.iloc[i, 0]).strip().lower() in ("patient", "samplename", "sample")); X = X0.iloc[h + 1:].copy(); X.columns = [str(c).strip() for c in X0.iloc[h]]; cols = list(X.columns)
        sc = next((c for c in cols if str(c).lower().startswith("sample")), None); mc = next(c for c in cols if "months" in str(c).lower())
        o = {"columns": [str(c) for c in cols], "rows": int(len(X)), "sample_column": None if sc is None else str(sc)}
        if sc is not None:
            m = SM.set_index("Sample").mbf; X["mbf_sheet"] = X[sc].astype(str).map(m); ok = X.mbf_sheet.notna()
            o.update({"matched_rows": int(ok.sum()), "equal": int((pd.to_numeric(X.loc[ok, mc], errors="coerce") == X.loc[ok, "mbf_sheet"]).sum())})
        else:   # no sample id: compare the per-patient multiset of months
            o["note"] = "no sample column; per-patient comparison"; pc = next(c for c in cols if str(c).lower().startswith("patient"))
            A = X.groupby(X[pc].astype(str).str.replace(r"\.0$", "", regex=True))[mc].apply(lambda s: sorted(pd.to_numeric(s, errors="coerce").dropna().round(3)))
            fdm = SM.groupby("Patient").mbf.apply(lambda s: sorted(s.round(3)))
            o["patients_compared"] = int(sum(p in A.index for p in fdm.index)); o["patients_months_subset_of_published"] = int(sum(p in A.index and all(v in A[p] for v in fdm[p]) for p in fdm.index))
        out[f] = o
    json.dump(out, open(f"{OUT}/q0_check.json", "w"), indent=1, default=str); print("HA CHECK DONE"); raise SystemExit
SRC, POP = TASK.split("_", 1)
def load(files, model=None):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files])
    if model is not None: D = D[D.model == model]
    P = D.pivot_table(index="Sample", columns="rep", values="pred"); assert P.shape == (n, 10), (files[0], P.shape); return P.reindex(SM.Sample).values
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv"))
R = {"L-CNV": load(kvf(0 if SRC == "their" else 1)), "L-IMG": load(kvf(2)), "L-EARLY": load(kvf(3 if SRC == "their" else 4)),
     "L-INTER": load(sorted(glob.glob(f"{HZ}/outer/inter_{SRC}_rep_*.csv"))), "L-CLIN": load([HA + "/clin_outer.csv"], "clin"), "L-CLIN-nograde": load([HA + "/clin_outer.csv"], "clin_nograde")}
R["L-LATE"] = (R["L-CNV"] + R["L-IMG"]) / 2
ARMS = ["L-CLIN", "L-CLIN-nograde", "L-CNV", "L-IMG", "L-EARLY", "L-INTER", "L-LATE"]; NONCLIN = ["L-CNV", "L-IMG", "L-EARLY", "L-INTER", "L-LATE"]
S = {a: R[a].mean(1) for a in ARMS}
pre = SM.pre.astype(bool).values; ndbe = SM.ndbe.astype(bool).values; fb = SM.fallback_endpoint.astype(bool).values
mask = {"pre": pre, "pre_ndbe": pre & ndbe, "pre_nofallback": pre & ~SM.Patient.isin(SM.Patient[fb]).values}[POP]
time = SM.time.values.astype(float); ev = SM.event.values.astype(int); pat = SM.Patient.values
def earliest_ndbe(m):
    r = np.where(m & ndbe)[0]; d = pd.DataFrame({"r": r, "p": pat[r], "mbf": SM.mbf.values[r]}).sort_values(["p", "mbf", "r"], ascending=[True, False, True]); return np.sort(d.groupby("p").r.first().values)
UNITS = {"sample": np.where(mask)[0], "patient": earliest_ndbe(mask)}
def Gminus(tt, dd, at):
    u = np.unique(tt); nrisk = (tt[None, :] >= u[:, None]).sum(1); dc = ((tt[None, :] == u[:, None]) & (dd[None, :] == 0)).sum(1)
    surv = np.cumprod(1 - dc / nrisk); prev = np.concatenate([[1.0], surv[:-1]]); k = np.searchsorted(u, at, side="left"); out = np.ones(len(at)); ins = k < len(u)
    out[ins] = prev[k[ins]]; out[~ins] = surv[-1]; return np.clip(out, 1e-12, None)
def labels(tt, dd, t): return (dd == 1) & (tt <= t), ((dd == 0) & (tt >= t)) | ((dd == 1) & (tt > t))
def tdauc(sc, tt, dd, t, wtd):
    case, ctrl = labels(tt, dd, t)
    if case.sum() == 0 or ctrl.sum() == 0: return np.nan
    w = 1 / Gminus(tt, dd, tt[case]) if wtd else np.ones(case.sum()); cs = np.sort(sc[ctrl]); m = sc[case]; lo = np.searchsorted(cs, m, "left"); hi = np.searchsorted(cs, m, "right")
    return float((w * (lo + 0.5 * (hi - lo))).sum() / (w.sum() * len(cs)))
def harrell(sc, tt, dd):
    ii = np.where(dd == 1)[0]
    if not len(ii): return np.nan
    comp = tt[ii][:, None] < tt[None, :]; conc = (sc[ii][:, None] > sc[None, :]) + 0.5 * (sc[ii][:, None] == sc[None, :]); den = comp.sum(); return float((comp * conc).sum() / den) if den else np.nan
def pctl(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return [r3(np.percentile(a, 2.5)), r3(np.percentile(a, 97.5))] if len(a) else [None, None]
res = {"task": TASK, "arms": ARMS, "families": {"vs_L-LATE": ["L-CNV", "L-IMG", "L-EARLY", "L-INTER"], "vs_L-CNV": ["L-IMG", "L-EARLY", "L-INTER", "L-LATE"]}, "units": {}}
for unit, rows in UNITS.items():
    tt, dd, pp = time[rows], ev[rows], pat[rows]; up = np.unique(pp); ro = {q: np.where(pp == q)[0] for q in up}
    rng = np.random.RandomState(0); boots = [np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))]) for _ in range(NB)]
    prs = np.random.RandomState(0); masks = []
    for _ in range(NB):
        sw = prs.rand(len(up)) < 0.5; mk = np.zeros(len(rows), bool)
        for i in np.where(sw)[0]: mk[ro[up[i]]] = True
        masks.append(mk)
    sc = {a: S[a][rows] for a in ARMS}; U = {"n_rows": int(len(rows)), "n_patients": int(len(up)), "n_event_patients": int(len(set(pp[dd == 1]))), "horizons": {}, "harrell": {}}
    for t in HORIZONS:
        case, ctrl = labels(tt, dd, t)
        H = {"n_cases": int(case.sum()), "n_controls": int(ctrl.sum()), "n_case_patients": int(len(set(pp[case]))), "n_control_patients": int(len(set(pp[ctrl]))), "n_dropped": int((~case & ~ctrl).sum()), "arms": {}}
        if case.sum() == 0 or ctrl.sum() == 0: U["horizons"][str(int(t))] = H; continue
        for wtd, key in ((True, "ipcw"), (False, "unweighted")):
            obs = {a: tdauc(sc[a], tt, dd, t, wtd) for a in ARMS}; bs = {a: np.array([tdauc(sc[a][b], tt[b], dd[b], t, wtd) for b in boots]) for a in ARMS}
            for a in ARMS:
                o = H["arms"].setdefault(a, {}); o[key] = {"auroc": r3(obs[a]), "ci95": pctl(bs[a])}
                for ref in ("L-LATE", "L-CNV"):
                    if a != ref: o[key][f"delta_vs_{ref}"] = {"delta": r3(obs[a] - obs[ref]), "ci95": pctl(bs[a] - bs[ref])}
            if unit == "sample" and wtd:
                for ref, fam in res["families"].items():
                    ref = ref.replace("vs_", ""); pdl = {a: np.array([tdauc(np.where(mk, sc[ref], sc[a]), tt, dd, t, True) - tdauc(np.where(mk, sc[a], sc[ref]), tt, dd, t, True) for mk in masks]) for a in fam}
                    mx = np.max(np.abs(np.stack([pdl[a] for a in fam])), 0)
                    for a in fam:
                        d0 = obs[a] - obs[ref]; o = H["arms"][a]["ipcw"][f"delta_vs_{ref}"]
                        o["p_unadjusted"] = round(float((1 + (np.abs(pdl[a]) >= abs(d0) - 1e-12).sum()) / (NB + 1)), 4); o["p_max_T"] = round(float((1 + (mx >= abs(d0) - 1e-12).sum()) / (NB + 1)), 4)
        H["ranking_ipcw"] = sorted(ARMS[:1] + NONCLIN, key=lambda a: -H["arms"][a]["ipcw"]["auroc"]); H["best"] = H["ranking_ipcw"][0]
        U["horizons"][str(int(t))] = H; print(TASK, unit, t, H["n_cases"], H["n_controls"], H["best"], flush=True)
    for a in ARMS: U["harrell"][a] = {"c": r3(harrell(sc[a], tt, dd)), "ci95": pctl([harrell(sc[a][b], tt[b], dd[b]) for b in boots])}
    res["units"][unit] = U
json.dump(res, open(f"{OUT}/q0_{TASK}.json", "w"), indent=1); print("HA METRICS DONE", TASK)
