"""Pathology grade (raw score) row (docs/paper_horizon_grade_row.md @ 487b2a7). Score = ordinal sample grade (NDBE 0, ID 1, LGD 2), no model.
Pooled and fold-stratified IPCW td-AUROC (estimators copied from ha_metrics.py and hf_foldstrat.py), paired Δ vs L-CNV and L-LATE on the same
patient-bootstrap draws (2,000, RandomState(0)), unadjusted swap-permutation p (2,000, seed 0). L-CNV / L-LATE recomputed here and checked against
q0_*.json and fs_*.json. TASK = <src>_<t> (src their|pkg, t 1|3|5) or ndbecheck. -> results/paper_final/horizon_grade_row/hg_<TASK>.json"""
import json, os, glob, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"; OUT = T + "/results/paper_final/horizon_grade_row"; os.makedirs(OUT, exist_ok=True)
TASK = os.environ["TASK"]; NB = 2000; REPS = list(range(1, 11)); r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM)
grade = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).astype(float).values
pre = SM.pre.astype(bool).values; ndbe = SM.ndbe.astype(bool).values
if TASK == "ndbecheck":
    g = grade[pre & ndbe]; json.dump({"population": "NDBE pre-event samples", "n": int(len(g)), "distinct_grades": sorted(set(g.tolist())), "pre_event_grades": pd.Series(SM.Pathology[pre]).value_counts().to_dict()},
                                     open(f"{OUT}/hg_ndbecheck.json", "w"), indent=1); print("HG NDBE CHECK DONE", sorted(set(g.tolist()))); raise SystemExit
SRC, TT = TASK.split("_"); t = float(TT)
def per_rep(files):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files]); P = D.pivot_table(index="Sample", columns="rep", values="pred").reindex(SM.Sample); Fo = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(SM.Sample)
    return {r: P[r].values for r in REPS}, {r: Fo[r].values.astype(int) for r in REPS}
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv"))
CN, FOLD = per_rep(kvf(0 if SRC == "their" else 1)); IM, _ = per_rep(kvf(2))
A = {"GRADE": {r: grade for r in REPS}, "L-CNV": CN, "L-LATE": {r: (CN[r] + IM[r]) / 2 for r in REPS}}
POOL = {a: np.mean([A[a][r] for r in REPS], 0) for a in A}
time = SM.time.values.astype(float); ev = SM.event.values.astype(int); pat = SM.Patient.values
def Gm(tt, dd, at):
    u = np.unique(tt); nr = (tt[None, :] >= u[:, None]).sum(1); dc = ((tt[None, :] == u[:, None]) & (dd[None, :] == 0)).sum(1); sv = np.cumprod(1 - dc / nr); pv = np.concatenate([[1.0], sv[:-1]])
    k = np.searchsorted(u, at, side="left"); o = np.ones(len(at)); i = k < len(u); o[i] = pv[k[i]]; o[~i] = sv[-1]; return np.clip(o, 1e-12, None)
def setup(rows):
    tt, dd = time[rows], ev[rows]; case = (dd == 1) & (tt <= t); ctrl = ((dd == 0) & (tt >= t)) | ((dd == 1) & (tt > t))
    if not case.sum() or not ctrl.sum(): return None
    return {"rows": rows, "ci": np.where(case)[0], "co": np.where(ctrl)[0], "w": 1 / Gm(tt, dd, tt[case])}
def pooled(s, S):
    rows, ci, co, w = S["rows"], S["ci"], S["co"], S["w"]; x = s[rows]; cs = np.sort(x[co]); m = x[ci]; lo = np.searchsorted(cs, m, "left"); hi = np.searchsorted(cs, m, "right")
    return float((w * (lo + 0.5 * (hi - lo))).sum() / (w.sum() * len(cs)))
def fstrat(arm, S, swapfn=None):
    rows, ci, co, w = S["rows"], S["ci"], S["co"], S["w"]; out = []
    for r in REPS:
        s = arm[r][rows] if swapfn is None else swapfn(r); f = FOLD[r][rows]; kc = np.sort(f[co] * 8.0 + s[co]); fi, si = f[ci] * 8.0, s[ci]   # scores in [0, 2] for grade -> spacing 8
        base = np.searchsorted(kc, fi, "left"); lo = np.searchsorted(kc, fi + si, "left") - base; hi = np.searchsorted(kc, fi + si, "right") - base; nct = np.searchsorted(kc, fi + 4.0, "left") - base
        den = (w * nct).sum(); out.append((w * (lo + 0.5 * (hi - lo))).sum() / den if den > 0 else np.nan)
    return float(np.nanmean(out))
def pctl(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return [r3(np.percentile(a, 2.5)), r3(np.percentile(a, 97.5))] if len(a) else [None, None]
rows0 = np.where(pre)[0]; S0 = setup(rows0); pp = pat[rows0]; up = np.unique(pp); ro = {q: rows0[pp == q] for q in up}
rng = np.random.RandomState(0); boots = [setup(np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))])) for _ in range(NB)]; boots = [b for b in boots if b is not None]
res = {"task": TASK, "n_cases": int(len(S0["ci"])), "n_controls": int(len(S0["co"])), "valid_draws": len(boots), "arms": {}}
for metric, fn in (("pooled", lambda a, S: pooled(POOL[a], S)), ("foldstrat", lambda a, S: fstrat(A[a], S))):
    obs = {a: fn(a, S0) for a in A}; bs = {a: np.array([fn(a, b) for b in boots]) for a in A}
    for a in A: res["arms"].setdefault(a, {})[metric] = {"auroc": r3(obs[a]), "ci95": pctl(bs[a])}
    prs = np.random.RandomState(0); idx = {q: np.where(pp == q)[0] for q in up}; masks = []
    for _ in range(NB):
        sw = prs.rand(len(up)) < 0.5; mk = np.zeros(len(rows0), bool)
        for i in np.where(sw)[0]: mk[idx[up[i]]] = True
        masks.append(mk)
    for ref in ("L-CNV", "L-LATE"):
        d0 = obs["GRADE"] - obs[ref]
        if metric == "pooled":
            x, z = POOL["GRADE"], POOL[ref]; full = lambda v: v
            pd_ = np.array([pooled(np.where(np.isin(np.arange(n), rows0[mk]), z, x), S0) - pooled(np.where(np.isin(np.arange(n), rows0[mk]), x, z), S0) for mk in masks])
        else:
            pd_ = np.array([fstrat(None, S0, lambda r, mk=mk: np.where(mk, A[ref][r][rows0], A["GRADE"][r][rows0])) - fstrat(None, S0, lambda r, mk=mk: np.where(mk, A["GRADE"][r][rows0], A[ref][r][rows0])) for mk in masks])
        res["arms"]["GRADE"][metric][f"delta_vs_{ref}"] = {"delta": r3(d0), "ci95": pctl(bs["GRADE"] - bs[ref]), "p_unadjusted_swap": round(float((1 + (np.abs(pd_) >= abs(d0) - 1e-12).sum()) / (NB + 1)), 4)}
q0 = json.load(open(f"{T}/results/paper_final/horizon_answers/q0_{SRC}_pre.json"))["units"]["sample"]["horizons"][TT]["arms"]; fs = json.load(open(f"{T}/results/paper_final/horizon_foldstrat/fs_{SRC}_pre_{TT}.json"))["arms"]
res["check_against_existing"] = {a: {"pooled_here": res["arms"][a]["pooled"]["auroc"], "pooled_q0": q0[a]["ipcw"]["auroc"], "foldstrat_here": res["arms"][a]["foldstrat"]["auroc"], "foldstrat_fs": fs[a]["ipcw"]} for a in ("L-CNV", "L-LATE")}
json.dump(res, open(f"{OUT}/hg_{TASK}.json", "w"), indent=1); print("HG DONE", TASK, res["arms"]["GRADE"], res["check_against_existing"])
