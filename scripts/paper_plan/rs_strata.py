"""Risk stratification (docs/paper_risk_strata.md @ b1f788a). TASK = sep_<pop> | disc_<pop> | km, pop in all676 | pre | pre_ndbe.
Stored out-of-fold predictions only (kv_cv.R cfg 0-5, hz_fit.R outer inter_*), mean over 10 repeats; (P) = Killcoyne's published probability (k_prob).
Survival analysis models (Cox, KM) are fitted to the stored predictions; no prediction model is refitted. Aggregates -> results/paper_final/risk_strata/."""
import os, glob, json, numpy as np, pandas as pd
from scipy.stats import rankdata, chi2, norm
from statsmodels.duration.hazard_regression import PHReg
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"; OUT = T + "/results/paper_final/risk_strata"; os.makedirs(OUT, exist_ok=True)
TASK = os.environ["TASK"]; NB = 2000; REPS = list(range(1, 11)); r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM); ids = SM.Sample.values
C = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample").reindex(ids); SM["endo"] = C.Endoscopy.values
yv = SM.y.values.astype(int); pat = SM.Patient.values; time = SM.time.values.astype(float); ev = SM.event.values.astype(int)
pre = SM.pre.astype(bool).values; ndbe = SM.ndbe.astype(bool).values; mbf = SM.mbf.values.astype(float); tev = SM.tev.values.astype(float)
grade = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).astype(float).values
def per_rep(files, col="pred"):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files]); P = D.pivot_table(index="Sample", columns="rep", values=col).reindex(ids); Fo = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)
    assert P.notna().all().all(); return {r: P[r].values for r in REPS}, {r: Fo[r].values.astype(int) for r in REPS}
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv"))
REP, FOLD = {}, None
REP["C"], FOLD = per_rep(kvf(0)); REP["C_pkg"], _ = per_rep(kvf(1)); REP["WSI"], _ = per_rep(kvf(2)); REP["EARLY"], _ = per_rep(kvf(3)); REP["EARLY_pkg"], _ = per_rep(kvf(4))
REP["INTER"], _ = per_rep(sorted(glob.glob(f"{HZ}/outer/inter_their_rep_*.csv"))); REP["INTER_pkg"], _ = per_rep(sorted(glob.glob(f"{HZ}/outer/inter_pkg_rep_*.csv")))
REP["INTERCEPT"], _ = per_rep(kvf(0), "train_prev")
REP["L"] = {r: (REP["C"][r] + REP["WSI"][r]) / 2 for r in REPS}; REP["L_pkg"] = {r: (REP["C_pkg"][r] + REP["WSI"][r]) / 2 for r in REPS}
S = {k: np.mean([v[r] for r in REPS], 0) for k, v in REP.items()}; S["P"] = pd.to_numeric(SM.k_prob, errors="coerce").values; S["GRADE"] = grade
assert np.isfinite(S["P"]).all()
POP = {"all676": np.ones(n, bool), "pre": pre, "pre_ndbe": pre & ndbe}
def pctl(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return [r3(np.percentile(a, 2.5)), r3(np.percentile(a, 97.5))] if len(a) else [None, None]
def boots_for(rows, need_both=False):
    pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(0); out = []
    while len(out) < NB:
        ix = rng.choice(len(up), len(up)); b = np.concatenate([ro[up[i]] for i in ix])
        if need_both and not (0 < yv[b].sum() < len(b)): continue
        out.append(b)
    return out
def classes_primary(s): return np.where(s <= 0.3, 0, np.where(s < 0.5, 1, 2))
def classes_matched(s, props):
    o = np.argsort(s, kind="stable"); k = np.zeros(len(s), int); n0 = int(round(props[0] * len(s))); n1 = int(round((props[0] + props[1]) * len(s)))
    k[o[n0:n1]] = 1; k[o[n1:]] = 2; cut = [float(s[o[n0 - 1]]) if n0 > 0 else None, float(s[o[n1 - 1]]) if n1 > 0 else None]; return k, cut
res = {"task": TASK}
SEPM = ["P", "C", "L", "C_pkg", "L_pkg"]
if TASK.startswith("sep_"):
    popn = TASK[4:]; rows = np.where(POP[popn])[0]; res["n_samples"] = int(len(rows)); res["n_patients"] = int(len(set(pat[rows]))); res["n_progressor_patients"] = int(len(set(pat[rows][yv[rows] == 1])))
    surv = popn != "all676"; boots = boots_for(rows)
    pclass = classes_primary(S["P"][rows]); props = [float((pclass == g).mean()) for g in range(3)]; res["P_class_proportions"] = [r3(p) for p in props]
    CLS = {}
    for m in SEPM:
        CLS[(m, "primary")] = classes_primary(S[m][rows])
        if m == "P": CLS[(m, "matched")] = CLS[(m, "primary")]; res.setdefault("matched_cutpoints", {})[m] = [0.3, 0.5]
        else: k, cut = classes_matched(S[m][rows], props); CLS[(m, "matched")] = k; res.setdefault("matched_cutpoints", {})[m] = [r3(c) for c in cut]
    def cox(kls, rr):
        X = np.column_stack([(kls == 1).astype(float), (kls == 2).astype(float)]); t, d = time[rr], ev[rr]
        if d[kls == 0].sum() == 0 or d.sum() == 0: return None
        try:
            f = PHReg(t, X, status=d, ties="breslow").fit(groups=pd.factorize(pat[rr])[0])
            if np.isfinite(f.bse).all(): return {"src": "statsmodels PHReg, Breslow ties, cluster-robust", "params": f.params, "ci": f.conf_int(), "p": f.pvalues}
        except Exception: pass
        from lifelines import CoxPHFitter   # fallback: Efron ties, cluster-robust sandwich (stated in the output)
        df = pd.DataFrame({"t": t, "d": d, "mod": X[:, 0], "high": X[:, 1], "pid": pd.factorize(pat[rr])[0]})
        cf = CoxPHFitter().fit(df, "t", "d", cluster_col="pid", robust=True)
        ci = cf.confidence_intervals_.values; return {"src": "lifelines CoxPHFitter, Efron ties, cluster-robust (fallback)", "params": cf.params_.values, "ci": ci, "p": cf.summary.p.values}
    def cox_plain_logHR_high(kls, rr):
        X = np.column_stack([(kls == 1).astype(float), (kls == 2).astype(float)]); t, d = time[rr], ev[rr]
        if d[kls == 0].sum() == 0 or d[kls == 2].sum() == 0: return np.nan
        try: return float(PHReg(t, X, status=d, ties="breslow").fit().params[1])
        except Exception: return np.nan
    def trend_logrank(kls, t, d):
        w = np.array([0.0, 1.0, 2.0]); U = 0.0; V = 0.0
        for tt in np.unique(t[d == 1]):
            atr = t >= tt; nt = atr.sum(); dt = ((t == tt) & (d == 1)).sum(); ng = np.array([(atr & (kls == g)).sum() for g in range(3)]); og = np.array([((t == tt) & (d == 1) & (kls == g)).sum() for g in range(3)])
            eg = dt * ng / nt; U += (w * (og - eg)).sum()
            if nt > 1: V += dt * (nt - dt) / (nt - 1) * ((w ** 2 * ng / nt).sum() - ((w * ng / nt).sum()) ** 2)
        z = U / np.sqrt(V) if V > 0 else np.nan; return float(z), float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else None
    yy = yv[rows]
    for (m, sch), kls in CLS.items():
        o = {"classes": []}
        for g, nm in enumerate(("low", "moderate", "high")):
            msk = kls == g; o["classes"].append({"class": nm, "samples": int(msk.sum()), "patients": int(len(set(pat[rows][msk]))), "progressor_samples": int(yy[msk].sum()), "progressor_patients": int(len(set(pat[rows][msk & (yy == 1)]))),
                                                 "event_samples": int(ev[rows][msk].sum()) if surv else None})
        # (i) sensitivity / specificity at the two boundaries (label = progressor status, per sample)
        def sesp(k, y):
            out = {}
            for nm, pos in (("low_vs_above", k >= 1), ("high_vs_below", k == 2)): out[nm] = (pos[y == 1].mean() if (y == 1).any() else np.nan, (~pos)[y == 0].mean() if (y == 0).any() else np.nan)
            return out
        bs_ = [sesp(classes_primary(S[m][b]) if sch == "primary" else None, yv[b]) for b in boots] if sch == "primary" else None
        ob = sesp(kls, yy); o["sens_spec"] = {}
        for nm in ob:
            o["sens_spec"][nm] = {"sensitivity": r3(ob[nm][0]), "specificity": r3(ob[nm][1])}
            if bs_: o["sens_spec"][nm].update({"sensitivity_ci95": pctl([b[nm][0] for b in bs_]), "specificity_ci95": pctl([b[nm][1] for b in bs_])})
        # (ii) class proportions by status, overall and in NDBE samples
        nd = ndbe[rows]; o["class_share_by_status"] = {lab: {sub: [r3((kls[(yy == yl) & sm] == g).mean()) if ((yy == yl) & sm).any() else None for g in range(3)] + [int(((yy == yl) & sm).sum())]
                                                               for sub, sm in (("all", np.ones(len(rows), bool)), ("NDBE", nd))} for lab, yl in (("progressor", 1), ("nonprogressor", 0))}
        if sch == "primary":
            # (iii) decile calibration
            dec = pd.qcut(pd.Series(S[m][rows]).rank(method="first"), 10, labels=False); o["deciles"] = [{"decile": int(k + 1), "n": int((dec == k).sum()), "P": int(yy[dec == k].sum()), "NP": int((yy[dec == k] == 0).sum()), "mean_pred": r3(S[m][rows][dec == k].mean())} for k in range(10)]
            # (iv) endoscopy-level sensitivity for high risk, progressor endoscopies, months before endpoint
            prog = yy == 1; mb = (mbf[rows] - tev[rows]); df = pd.DataFrame({"pat": pat[rows], "endo": SM.endo.values[rows], "mb": mb, "high": kls == 2})[prog & (mb >= 0)]
            E = df.groupby(["pat", "endo"]).agg(mb=("mb", "max"), high=("high", "max"), s=("high", "size")).reset_index()
            bins = [("endpoint", lambda x: x == 0), ("1-12", lambda x: (x > 0) & (x <= 12)), ("12-24", lambda x: (x > 12) & (x <= 24)), ("24-48", lambda x: (x > 24) & (x <= 48)), ("48-72", lambda x: (x > 48) & (x <= 72)), ("72-96", lambda x: (x > 72) & (x <= 96)), (">96", lambda x: x > 96)]
            o["endoscopy_sensitivity"] = [{"months_before_endpoint": nm, "endoscopies": int(f(E.mb).sum()), "samples": int(E.s[f(E.mb)].sum()), "patients": int(E.pat[f(E.mb)].nunique()), "share_high": r3(E.high[f(E.mb)].mean()) if f(E.mb).any() else None} for nm, f in bins]
        if surv:
            f = cox(kls, rows)
            if f is not None:
                ci = np.exp(np.asarray(f["ci"], float)); o["cox"] = {"source": f["src"], "HR_moderate_vs_low": r3(np.exp(f["params"][0])), "HR_moderate_ci95": [r3(ci[0, 0]), r3(ci[0, 1])], "HR_high_vs_low": r3(np.exp(f["params"][1])), "HR_high_ci95": [r3(ci[1, 0]), r3(ci[1, 1])], "p_moderate": float(f"{f['p'][0]:.3g}"), "p_high": float(f"{f['p'][1]:.3g}")}
            else: o["cox"] = "not estimable (no events in the low class)"
            z, p = trend_logrank(kls, time[rows], ev[rows]); o["logrank_trend"] = {"z": r3(z), "p": float(f"{p:.3g}") if p is not None else None}
            try:
                fo = PHReg(time[rows], kls.reshape(-1, 1).astype(float), status=ev[rows], ties="breslow").fit(groups=pd.factorize(pat[rows])[0])
                if np.isfinite(fo.bse).all(): o["ordinal_cox_robust"] = {"HR_per_class": r3(np.exp(fo.params[0])), "ci95": [r3(v) for v in np.exp(fo.conf_int()[0])], "p": float(f"{fo.pvalues[0]:.3g}"), "source": "statsmodels PHReg, Breslow, cluster-robust"}
                else:
                    from lifelines import CoxPHFitter
                    cf = CoxPHFitter().fit(pd.DataFrame({"t": time[rows], "d": ev[rows], "cls": kls.astype(float), "pid": pd.factorize(pat[rows])[0]}), "t", "d", cluster_col="pid", robust=True)
                    o["ordinal_cox_robust"] = {"HR_per_class": r3(float(np.exp(cf.params_.iloc[0]))), "ci95": [r3(float(v)) for v in np.exp(cf.confidence_intervals_.values[0])], "p": float(f"{cf.summary.p.iloc[0]:.3g}"), "source": "lifelines, Efron, cluster-robust (fallback)"}
            except Exception as ex: o["ordinal_cox_robust"] = f"failed: {ex}"[:100]
            o["share_events_in_high"] = r3(ev[rows][kls == 2].sum() / max(ev[rows].sum(), 1))
        res.setdefault("models", {}).setdefault(m, {})[sch] = o
    if surv:   # paired comparisons, patient bootstrap, Cox refitted per draw (plain variance)
        def stat(m, sch, b):
            s = S[m][b]
            if sch == "primary": k = classes_primary(s)
            else: k = classes_primary(s) if m == "P" else classes_matched(s, [float((classes_primary(S["P"][b]) == g).mean()) for g in range(3)])[0]
            return cox_plain_logHR_high(k, b), (ev[b][k == 2].sum() / ev[b].sum()) if ev[b].sum() else np.nan
        for sch in ("primary", "matched"):
            obs = {m: (cox_plain_logHR_high(CLS[(m, sch)], rows), ev[rows][CLS[(m, sch)] == 2].sum() / ev[rows].sum()) for m in ("P", "C", "L", "C_pkg", "L_pkg")}
            bst = {m: np.array([stat(m, sch, b) for b in boots]) for m in ("P", "C", "L", "C_pkg", "L_pkg")}
            for a, ref in (("L", "C"), ("L", "P"), ("L_pkg", "C_pkg"), ("L_pkg", "P")):
                o = {}
                for j, nm in ((0, "delta_logHR_high_vs_low"), (1, "delta_share_events_in_high")):
                    d0 = obs[a][j] - obs[ref][j]; db = bst[a][:, j] - bst[ref][:, j]; db = db[np.isfinite(db)]
                    o[nm] = {"delta": r3(d0), "ci95": pctl(db), "p_bootstrap_unadjusted": round(float(min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean()))), 4) if len(db) else None, "valid_draws": int(len(db))}
                res.setdefault("paired", {})[f"{a}_vs_{ref}|{sch}"] = o
            print("paired", sch, flush=True)
elif TASK.startswith("disc_"):
    popn = TASK[5:]; rows = np.where(POP[popn])[0]; tt, dd = time[rows], ev[rows]; yy = yv[rows]
    MODELS = ["P", "C", "L", "WSI", "EARLY", "INTER", "GRADE", "C_pkg", "L_pkg", "EARLY_pkg", "INTER_pkg", "INTERCEPT"]; CV = {k for k in REP}
    def fs_auc(m, rr):
        y = yv[rr]; out = []
        for r in REPS:
            f = FOLD[r][rr]; x = REP[m][r][rr]; num = den = 0.0
            for k in range(1, 11):
                mk = f == k; yk = y[mk]; n1 = yk.sum(); n0 = len(yk) - n1
                if n1 == 0 or n0 == 0: continue
                rk = rankdata(x[mk]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
            out.append(num / den if den else np.nan)
        return float(np.nanmean(out))
    def auc(y, s):
        r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else np.nan
    def Gfun(t, d):
        u = np.unique(t); nr = (t[None, :] >= u[:, None]).sum(1); dc = ((t[None, :] == u[:, None]) & (d[None, :] == 0)).sum(1); sv = np.cumprod(1 - dc / nr)
        def Gm(a): k = np.searchsorted(u, a, "left"); prev = np.concatenate([[1.0], sv]); return np.clip(prev[k], 1e-12, None)          # G(a-)
        def Gat(a): k = np.searchsorted(u, a, "right"); prev = np.concatenate([[1.0], sv]); return np.clip(prev[k], 1e-12, None)        # G(a)
        return Gm, Gat
    def km(t, d, grid):
        u = np.unique(t[d == 1]); nr = np.array([(t >= x).sum() for x in u]); de = np.array([((t == x) & (d == 1)).sum() for x in u]); sv = np.cumprod(1 - de / nr)
        k = np.searchsorted(u, grid, "right"); return np.concatenate([[1.0], sv])[k]
    def cidx(s, t, d, uno, tau=8.0):
        Gm, _ = Gfun(t, d); ii = np.where((d == 1) & (t < tau))[0]
        if not len(ii): return np.nan
        w = 1 / Gm(t[ii]) ** 2 if uno else np.ones(len(ii)); comp = t[ii][:, None] < t[None, :]; conc = (s[ii][:, None] > s[None, :]) + 0.5 * (s[ii][:, None] == s[None, :])
        den = (w[:, None] * comp).sum(); return float((w[:, None] * comp * conc).sum() / den) if den else np.nan
    GRID = np.round(np.arange(0.05, 5.0001, 0.05), 4)
    def brier(F, t, d, tgrid):
        Gm, Gat = Gfun(t, d); out = []
        for x in tgrid:
            Fi = F(x) if callable(F) else F; ev_ = (t <= x) & (d == 1); al = t > x
            out.append(float(((1 - Fi) ** 2 * ev_ / Gm(t) + Fi ** 2 * al / Gat(np.array([x]))[0]).mean()))
        return np.array(out)
    def metrics(m, rr):
        t, d = time[rr], ev[rr]; s = S[m][rr]; o = {}
        o["auroc"] = fs_auc(m, rr) if m in CV else auc(yv[rr], s)
        o["harrell"] = cidx(s, t, d, False); o["uno"] = cidx(s, t, d, True)
        b = brier(s, t, d, GRID); o["brier"] = {str(h): float(b[int(round(h / 0.05)) - 1]) for h in (1, 3, 5)}; o["ibs5"] = float(np.trapz(np.concatenate([[0.0], b]), np.concatenate([[0.0], GRID])) / 5.0)
        return o
    def null_metrics(rr):
        t, d = time[rr], ev[rr]; F = lambda x: 1 - km(t, d, np.array([x]))[0]; b = brier(F, t, d, GRID)
        return {"brier": {str(h): float(b[int(round(h / 0.05)) - 1]) for h in (1, 3, 5)}, "ibs5": float(np.trapz(np.concatenate([[0.0], b]), np.concatenate([[0.0], GRID])) / 5.0)}
    boots = boots_for(rows, need_both=True); obs = {m: metrics(m, rows) for m in MODELS}; nul = null_metrics(rows)
    BS = {m: [metrics(m, b) for b in boots] for m in MODELS}; BN = [null_metrics(b) for b in boots]
    flat = lambda o: [o["auroc"], o["harrell"], o["uno"], o["brier"]["1"], o["brier"]["3"], o["brier"]["5"], o["ibs5"]]; keys = ["auroc", "harrell", "uno", "brier_1", "brier_3", "brier_5", "ibs5"]
    for m in MODELS:
        A = np.array([flat(x) for x in BS[m]]); res.setdefault("models", {})[m] = {k: {"value": r3(v), "ci95": pctl(A[:, j])} for j, (k, v) in enumerate(zip(keys, flat(obs[m])))}
    An = np.array([[x["brier"]["1"], x["brier"]["3"], x["brier"]["5"], x["ibs5"]] for x in BN]); res["km_null"] = {k: {"value": r3(v), "ci95": pctl(An[:, j])} for j, (k, v) in enumerate(zip(["brier_1", "brier_3", "brier_5", "ibs5"], [nul["brier"]["1"], nul["brier"]["3"], nul["brier"]["5"], nul["ibs5"]]))}
    for a, ref in (("L", "C"), ("L_pkg", "C_pkg"), ("L", "P")):
        Aa = np.array([flat(x) for x in BS[a]]); Ar = np.array([flat(x) for x in BS[ref]]); d0 = np.array(flat(obs[a])) - np.array(flat(obs[ref])); D = Aa - Ar
        res.setdefault("paired", {})[f"{a}_vs_{ref}"] = {k: {"delta": r3(d0[j]), "ci95": pctl(D[:, j]), "p_bootstrap_unadjusted": round(float(min(1.0, 2 * min((D[:, j] <= 0).mean(), (D[:, j] >= 0).mean()))), 4)} for j, k in enumerate(keys)}
    res.update({"n_samples": int(len(rows)), "n_patients": int(len(set(pat[rows]))), "n_events": int(dd.sum()), "auroc_type": {m: ("fold-stratified" if m in CV else "plain") for m in MODELS}})
elif TASK == "km":
    GRID = np.round(np.arange(0, 10.0001, 0.05), 4); AT = [0, 2, 4, 6, 8]
    def km(t, d, grid):
        u = np.unique(t[d == 1]); nr = np.array([(t >= x).sum() for x in u]); de = np.array([((t == x) & (d == 1)).sum() for x in u]); sv = np.cumprod(1 - de / nr)
        k = np.searchsorted(u, grid, "right"); return np.concatenate([[1.0], sv])[k]
    rows = np.where(pre)[0]
    d_ = pd.DataFrame({"r": rows, "p": pat[rows], "mbf": mbf[rows]}).sort_values(["p", "mbf", "r"], ascending=[True, False, True]); prow = np.sort(d_.groupby("p").r.first().values)
    for level, rr in (("sample", rows), ("patient", prow)):
        pp = pat[rr]; up = np.unique(pp); ro = {q: rr[pp == q] for q in up}; rng = np.random.RandomState(0); boots = [np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))]) for _ in range(NB)]
        for m in ("P", "C", "L"):
            o = {}
            for g, nm in enumerate(("low", "moderate", "high")):
                msk = classes_primary(S[m][rr]) == g; t, d = time[rr][msk], ev[rr][msk]
                if not msk.any(): o[nm] = None; continue
                curve = km(t, d, GRID); bb = []
                for b in boots:
                    mb = classes_primary(S[m][b]) == g
                    if mb.sum() > 0: bb.append(km(time[b][mb], ev[b][mb], GRID))
                bb = np.array(bb); o[nm] = {"n": int(msk.sum()), "patients": int(len(set(pat[rr][msk]))), "events": int(d.sum()), "grid": GRID.tolist(), "km": np.round(curve, 4).tolist(),
                                            "lo": np.round(np.percentile(bb, 2.5, 0), 4).tolist(), "hi": np.round(np.percentile(bb, 97.5, 0), 4).tolist(), "at_risk": [int((t >= a).sum()) for a in AT]}
            res.setdefault(level, {})[m] = o
        res[level + "_n"] = {"samples": int(len(rr)), "patients": int(len(up)), "events": int(ev[rr].sum())}
json.dump(res, open(f"{OUT}/{TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RS DONE", TASK)
