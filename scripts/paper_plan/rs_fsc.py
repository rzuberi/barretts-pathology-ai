"""Addendum A of docs/paper_risk_strata.md (pre-specification f29bd1d): fold-stratified Harrell's and Uno's C (tau 8 y). TASK = fsc_pre | fsc_pre_ndbe.
Comparable pairs restricted to the same outer fold of a repeat, scored with that repeat's out-of-fold predictions, pooled within folds, mean over 10 repeats.
Stored predictions only; nothing refitted. Aggregates -> results/paper_final/risk_strata/fsc_<pop>.json."""
import os, glob, json, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"; OUT = T + "/results/paper_final/risk_strata"; os.makedirs(OUT, exist_ok=True)
TASK = os.environ["TASK"]; NB = 2000; REPS = list(range(1, 11)); TAU = 8.0; r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM); ids = SM.Sample.values
yv = SM.y.values.astype(int); pat = SM.Patient.values; time = SM.time.values.astype(float); ev = SM.event.values.astype(int)
pre = SM.pre.astype(bool).values; ndbe = SM.ndbe.astype(bool).values
grade = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).astype(float).values
def per_rep(files, col="pred"):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files]); P = D.pivot_table(index="Sample", columns="rep", values=col).reindex(ids); Fo = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)
    assert P.notna().all().all(); return {r: P[r].values for r in REPS}, {r: Fo[r].values.astype(int) for r in REPS}
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv"))
REP, FO = {}, {}
for k, fl, col in (("C", kvf(0), "pred"), ("C_pkg", kvf(1), "pred"), ("WSI", kvf(2), "pred"), ("EARLY", kvf(3), "pred"), ("EARLY_pkg", kvf(4), "pred"),
                   ("INTER", sorted(glob.glob(f"{HZ}/outer/inter_their_rep_*.csv")), "pred"), ("INTER_pkg", sorted(glob.glob(f"{HZ}/outer/inter_pkg_rep_*.csv")), "pred"), ("INTERCEPT", kvf(0), "train_prev")):
    REP[k], FO[k] = per_rep(fl, col)
FOLD = FO["C"]
match = {k: bool(all((FO[k][r] == FOLD[r]).all() for r in REPS)) for k in FO}
REP["L"] = {r: (REP["C"][r] + REP["WSI"][r]) / 2 for r in REPS}; REP["L_pkg"] = {r: (REP["C_pkg"][r] + REP["WSI"][r]) / 2 for r in REPS}
FO["L"] = FOLD if match["WSI"] else None; FO["L_pkg"] = FOLD if match["C_pkg"] and match["WSI"] else None
assert FO["L"] is not None and FO["L_pkg"] is not None, "L folds ill-defined: C/WSI folds differ"
kp = pd.to_numeric(SM.k_prob, errors="coerce").values; assert np.isfinite(kp).all()
REP["P"] = {r: kp for r in REPS}; FO["P"] = FOLD; REP["GRADE"] = {r: grade for r in REPS}; FO["GRADE"] = FOLD
MODELS = ["P", "C", "L", "WSI", "EARLY", "INTER", "GRADE", "C_pkg", "L_pkg", "EARLY_pkg", "INTER_pkg", "INTERCEPT"]
FKEY = {m: ("C" if m in ("L", "L_pkg", "P", "GRADE") else m) for m in MODELS}   # fold source per model
def Gm_fun(t, d):
    u = np.unique(t); nr = (t[None, :] >= u[:, None]).sum(1); dc = ((t[None, :] == u[:, None]) & (d[None, :] == 0)).sum(1); sv = np.cumprod(1 - dc / nr)
    return lambda a: np.clip(np.concatenate([[1.0], sv])[np.searchsorted(u, a, "left")], 1e-12, None)
def stats(rr, pooled=False):
    """returns {model: (harrell, uno)} fold-stratified (or pooled if pooled=True, as a cross-check against Section 4)"""
    t, d = time[rr], ev[rr]; Gm = Gm_fun(t, d); evi = (d == 1) & (t < TAU); base = evi[:, None] & (t[:, None] < t[None, :]); w = np.where(evi, 1 / Gm(t) ** 2, 0.0)
    pairs = {}
    out = {}
    for m in MODELS:
        H, U = [], []
        for r in ([1] if pooled else REPS):
            fk = FKEY[m]
            if pooled: I, J = np.nonzero(base)
            else:
                key = (fk, r)
                if key not in pairs: f = FO[fk][r][rr]; pairs[key] = np.nonzero(base & (f[:, None] == f[None, :]))
                I, J = pairs[key]
            s = (np.mean([REP[m][q] for q in REPS], 0) if pooled else REP[m][r])[rr]
            c = (s[I] > s[J]) + 0.5 * (s[I] == s[J]); wi = w[I]
            H.append(c.mean() if len(I) else np.nan); U.append((wi * c).sum() / wi.sum() if len(I) else np.nan)
        out[m] = (float(np.nanmean(H)), float(np.nanmean(U)))
    return out
POP = {"pre": pre, "pre_ndbe": pre & ndbe}; popn = TASK[4:]; rows = np.where(POP[popn])[0]
def boots_for(rows):   # identical to rs_strata.py disc (seed 0, need_both=True)
    pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(0); out = []
    while len(out) < NB:
        ix = rng.choice(len(up), len(up)); b = np.concatenate([ro[up[i]] for i in ix])
        if not (0 < yv[b].sum() < len(b)): continue
        out.append(b)
    return out
def pctl(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return [r3(np.percentile(a, 2.5)), r3(np.percentile(a, 97.5))] if len(a) else [None, None]
obs = stats(rows); pool = stats(rows, pooled=True); boots = boots_for(rows); BS = []
for i, b in enumerate(boots):
    BS.append(stats(b))
    if i % 200 == 0: print("draw", i, flush=True)
grade_const = bool(np.unique(grade[rows]).size == 1)
res = {"task": TASK, "prespec": "f29bd1d", "n_samples": int(len(rows)), "n_patients": int(len(set(pat[rows]))), "n_events": int(ev[rows].sum()), "tau": TAU, "draws": NB,
       "fold_match_vs_cfg0": match, "fold_source": FKEY, "grade_constant": grade_const,
       "intercept_check": {"harrell": r3(obs["INTERCEPT"][0]), "uno": r3(obs["INTERCEPT"][1]), "pass": bool(abs(obs["INTERCEPT"][0] - 0.5) < 1e-9 and abs(obs["INTERCEPT"][1] - 0.5) < 1e-9)},
       "pooled_crosscheck": {m: {"harrell": r3(pool[m][0]), "uno": r3(pool[m][1])} for m in MODELS}, "models": {}, "paired": {}}
for m in MODELS:
    A = np.array([x[m] for x in BS]); res["models"][m] = {"harrell": {"value": r3(obs[m][0]), "ci95": pctl(A[:, 0])}, "uno": {"value": r3(obs[m][1]), "ci95": pctl(A[:, 1])}}
    if m == "GRADE" and grade_const: res["models"][m] = "not estimable: grade constant in this population"
for a, ref in (("L", "C"), ("L", "P"), ("L_pkg", "C_pkg")):
    D = np.array([x[a] for x in BS]) - np.array([x[ref] for x in BS]); d0 = np.array(obs[a]) - np.array(obs[ref])
    res["paired"][f"{a}_vs_{ref}"] = {k: {"delta": r3(d0[j]), "ci95": pctl(D[:, j]), "p_bootstrap_unadjusted": round(float(min(1.0, 2 * min((D[:, j] <= 0).mean(), (D[:, j] >= 0).mean()))), 4)} for j, k in enumerate(("harrell", "uno"))}
json.dump(res, open(f"{OUT}/{TASK}.json", "w"), indent=1); print("FSC DONE", TASK, res["intercept_check"], flush=True)
