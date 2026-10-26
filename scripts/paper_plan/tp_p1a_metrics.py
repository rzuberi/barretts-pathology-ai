"""Two papers, agent P1A (docs/two_papers_P1A.md): Paper 1 core estimates from stored and refitted held-out predictions.
TASK = smoke | point | boot_<shard 0..19> | merge | f3
  smoke  : reproduce known numbers (L-IMG 0.847, L-CNV 0.801, L-LATE 0.889, package 0.845, per-sample 0.785, past-only 0.800, P 0.823 fold-strat,
           L-LATE - L-IMG +0.042 with max-T 0.066 over the old fusion family) and check the refit anchors against stored L-CNV predictions.
  point  : observed values (all estimands, P-refit variant table, thresholds, calibration, design checks).
  boot_s : patient-bootstrap draws 100*s .. 100*s+99 of the tb_common sequence (predictions fixed); per-draw statistics -> cluster pickle.
  merge  : CIs, p, max-T -> results/paper_final/two_papers_P1A/main.json
  f3     : refit-bootstrap recomputation from review_round2/f3 (and two_papers_P1A/prefit_boot if present) -> refit_boot.json
Aggregates only leave the cluster."""
import os, sys, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan")
from tb_common import *
from tb_common import _outer
TASK = os.environ["TASK"]; OUTR = T + "/results/paper_final/two_papers_P1A"; ROWD = M + "/two_papers_P1A"; os.makedirs(OUTR, exist_ok=True); os.makedirs(ROWD + "/boot", exist_ok=True)
ROW4 = M + "/review_round4"; RR2 = M + "/review_round2"; PF = ROWD + "/prefit"; PFB = ROWD + "/prefit_boot"
rows = np.where(PRE)[0]; FO0 = load("their")[2]; NS = 20; PER = NB // NS
def dump(res, name): json.dump(res, open(f"{OUTR}/{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("TP P1A DONE", name, flush=True)
def need(files, k): assert len(files) == k, (len(files), k, files[:2]); return files
# ------------------------------------------------------------------ prediction sets (n x 10, rows = horizons/samples.csv order)
O, _, FO = load("their"); S = {"L-IMG": O["L-IMG"], "L-CNV": O["L-CNV"], "L-LATE": O["L-LATE"]}
Op, _, Fp = load("pkg"); assert (Fp == FO0).all(); S["CN joint package"] = Op["L-CNV"]
for key, tag in (("CN past-only", "pkgpast"), ("CN per-sample", "pkgps")):
    Oc, F = _outer(need(sorted(glob.glob(f"{ROW4}/r36_outer/cnv_{tag}_rep_*.csv")), 10)); assert (F == FO0).all(); S[key] = Oc
KP = SM.k_prob.values.astype(float); S["P published"] = np.tile(KP[:, None], (1, 10))
def prefit(feat, univ, nlam, rule):
    fs = sorted(glob.glob(f"{PF}/{feat}_{univ}_nl{nlam}_rep_*_fold_*.csv"))
    if len(fs) < 100: return None
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in fs]); D = D[D.rule == rule]
    P_ = D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values; Fq = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values
    assert np.isfinite(P_[rows]).all() and (Fq[rows] == FO0[rows]).all(); return P_
VAR = {}
for feat in ("their", "pkg"):
    for univ in ("s773", "C"):
        for rule in ("class_1se", "class_min", "dev_min", "dev_1se", "published"):
            v = prefit(feat, univ, 1000, rule)
            if v is not None: VAR[f"{feat}|{univ}|nl1000|{rule}"] = v
PRIMARY_P = "their|s773|nl1000|class_1se"
if PRIMARY_P in VAR: S["P refit"] = VAR[PRIMARY_P]
MODELS = list(S); CNS = [m for m in ("L-CNV", "P refit", "CN joint package", "CN past-only", "CN per-sample", "P published") if m in S]
CONTRASTS = [("L-LATE", "L-IMG")] + [("L-IMG", c) for c in CNS]
# ------------------------------------------------------------------ units: endoscopies and patients of the pre-event population
up = np.unique(pat[rows]); pidx = {q: i for i, q in enumerate(up)}
NPRE = pd.Series(1, index=pat[rows]).groupby(level=0).sum(); WS = np.zeros(n); WS[rows] = 1.0 / NPRE.reindex(pat[rows]).values   # 1 / n pre-event samples of the patient
eu = pd.unique(ENDO[rows]); E_rows = [rows[ENDO[rows] == e] for e in eu]; E_pat = np.array([pat[r[0]] for r in E_rows]); E_y = np.array([y[r[0]] for r in E_rows]); E_F = np.array([FO0[r[0]] for r in E_rows])
P_rows = [rows[pat[rows] == q] for q in up]; P_y = np.array([y[r[0]] for r in P_rows]); P_F = np.array([FO0[r[0]] for r in P_rows])
agg = lambda X, groups, f: np.stack([f(X[g], 0) for g in groups])
UNITS = {}
def set_units(m, X):
    UNITS[(m, "endo_max")] = agg(X, E_rows, np.max); UNITS[(m, "endo_mean")] = agg(X, E_rows, np.mean); UNITS[(m, "pat_max")] = agg(X, P_rows, np.max); UNITS[(m, "pat_mean")] = agg(X, P_rows, np.mean)
for m, X in S.items(): set_units(m, X)
E_of_pat = {q: np.where(E_pat == q)[0] for q in up}; S_of_pat = {q: rows[pat[rows] == q] for q in up}
def draws(seed=0, nb=NB):   # tb_common.boots_for sequence, keeping patient draws (so copies can be tagged as distinct units)
    rng = np.random.RandomState(seed); out = []
    while len(out) < nb:
        c = rng.choice(len(up), len(up)); b = np.concatenate([S_of_pat[up[i]] for i in c])
        if 0 < y[b].sum() < len(b): out.append(c)
    return out
# ------------------------------------------------------------------ fold-stratified concordance on units (optionally pair-weighted, optionally extra grouping)
def fsc(X, yv, F, w=None, grp=None, fold=True):
    """X, F: units x 10; yv: units; w: unit weights (pair weight = product); grp: extra grouping code per unit (pairs within the same group only)."""
    out = []
    for j in range(10):
        g = F[:, j] if fold else np.zeros(len(yv), int)
        if grp is not None: g = g * 1000 + grp
        x = X[:, j]; num = den = 0.0
        for k in np.unique(g):
            mk = g == k; yk = yv[mk]; n1 = int(yk.sum()); n0 = len(yk) - n1
            if n1 == 0 or n0 == 0: continue
            if w is None:
                rk = rankdata(x[mk]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
            else:
                xs = x[mk]; ws = w[mk]; xp, xn = xs[yk == 1], xs[yk == 0]; W = ws[yk == 1][:, None] * ws[yk == 0][None, :]
                num += (((xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])) * W).sum(); den += W.sum()
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
def unit_index(c):   # resampled units for a patient draw c (indices into up); copies are distinct units
    si = np.concatenate([S_of_pat[up[i]] for i in c]); ei = np.concatenate([E_of_pat[up[i]] for i in c]); return si, ei, np.asarray(c)
def auc_all(si, ei, pi, models=None):
    o = {}
    for m in (models or MODELS):
        X = S[m]; o[(m, "sample")] = fsc(X[si], y[si], FO0[si]); o[(m, "patient_weighted")] = fsc(X[si], y[si], FO0[si], w=WS[si])
        for lv in ("endo_max", "endo_mean"): o[(m, lv)] = fsc(UNITS[(m, lv)][ei], E_y[ei], E_F[ei])
        for lv in ("pat_max", "pat_mean"): o[(m, lv)] = fsc(UNITS[(m, lv)][pi], P_y[pi], P_F[pi])
    return o
LEVELS = ["sample", "patient_weighted", "endo_max", "endo_mean", "pat_max", "pat_mean"]
# ------------------------------------------------------------------ classification at P's thresholds (low <= 0.3, high >= 0.5), per repeat then averaged
def cls_stats(si, ei, share_ref="P refit"):
    o = {}
    for m in [x for x in ("P published", "P refit", "L-CNV", "CN joint package", "L-IMG", "L-LATE") if x in S]:
        for lvl, X, yy in (("sample", S[m][si], y[si]), ("endo", UNITS[(m, "endo_max")][ei], E_y[ei])):
            hi = X >= 0.5; nl = X > 0.3
            o[(m, lvl, "high")] = (hi[yy == 1].mean(), 1 - hi[yy == 0].mean()); o[(m, lvl, "not_low")] = (nl[yy == 1].mean(), 1 - nl[yy == 0].mean())
    if share_ref in S:   # L-IMG cut-points reproducing the reference model's held-out class shares (label-free; descriptive)
        for lvl, Xr, Xi, yy in (("sample", S[share_ref][si], S["L-IMG"][si], y[si]), ("endo", UNITS[(share_ref, "endo_max")][ei], UNITS[("L-IMG", "endo_max")][ei], E_y[ei])):
            for cl, rel in (("high", lambda X: X >= 0.5), ("not_low", lambda X: X > 0.3)):
                sens, spec = [], []
                for j in range(10):
                    share = rel(Xr[:, j]).mean(); xi = Xi[:, j]; t = np.quantile(xi, 1 - share) if share > 0 else np.inf; c = xi >= t
                    sens.append(c[yy == 1].mean()); spec.append(1 - c[yy == 0].mean())
                o[("L-IMG share-matched to " + share_ref, lvl, cl)] = (np.mean(sens), np.mean(spec))
    return o
# ------------------------------------------------------------------ calibration: logistic recalibration of status on logit(p), per repeat
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def irls(x, yy, w, offset_only=False):
    b = np.zeros(1 if offset_only else 2)
    for _ in range(50):
        eta = (b[0] + x) if offset_only else (b[0] + b[1] * x); p = 1 / (1 + np.exp(-eta)); Z = np.ones((len(x), 1)) if offset_only else np.column_stack([np.ones(len(x)), x])
        g = Z.T @ (w * (yy - p)); H = (Z * (w * p * (1 - p))[:, None]).T @ Z; step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g); b += step
        if np.abs(step).max() < 1e-8: break
    return b
def calib(si, models=None):
    o = {}
    for m in (models or [x for x in ("L-IMG", "L-CNV", "L-LATE", "P refit", "CN joint package", "P published") if x in S]):
        for wn, w in (("unweighted", np.ones(len(si))), ("patient_weighted", WS[si])):
            sl, it = [], []
            for j in range(10):
                x = logit(S[m][si, j]); sl.append(irls(x, y[si].astype(float), w)[1]); it.append(irls(x, y[si].astype(float), w, True)[0])
            o[(m, wn, "slope")] = np.mean(sl); o[(m, wn, "citl")] = np.mean(it)
    return o
# ------------------------------------------------------------------ matching strata (sex x age at diagnosis x Prague M), per patient
def strata():
    Sx = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
    fd = pd.read_csv(Sx + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
    hr = pd.Series(ids).map(fd["Hospital_Research_ID_updated"].str.strip()).values
    dm = pd.read_csv(Sx + "/Demographics_full.csv", dtype=str); dm.columns = [c.strip() for c in dm.columns]; dmi = {}
    for _, r in dm.iterrows():
        for c in ("Study Number", "Alternate Study Number"):
            v = r[c].strip() if isinstance(r[c], str) else ""
            if v and v not in dmi: dmi[v] = r
    rec = []
    for q in up:
        h = pd.Series(hr[pat == q]).dropna().mode(); d = dmi.get(h.iloc[0]) if len(h) else None
        num = lambda c: pd.to_numeric(pd.Series([d[c] if d is not None and c in d.index else None]), errors="coerce").iloc[0]
        rec.append({"Patient": q, "age_dx": num("Age at diagnosis"), "sex": (d["Sex"].strip() if d is not None and isinstance(d["Sex"], str) else None), "prague_m": num("Maximal"), "y": int(P_y[pidx[q]])})
    R = pd.DataFrame(rec); med = R.age_dx.median()
    R["age_band"] = np.where(R.age_dx.isna(), "missing", np.where(R.age_dx <= med, "le_median", "gt_median")); R["m_band"] = np.where(R.prague_m.isna(), "missing", np.where(R.prague_m <= 4, "le4", "gt4"))
    R["stratum"] = R.sex.fillna("missing") + "|" + R.age_band + "|" + R.m_band; codes = {s: i for i, s in enumerate(sorted(R.stratum.unique()))}; R["code"] = R.stratum.map(codes)
    return R, float(med)
def strat_conc(si, pcode, models=None):
    o = {}; g = pcode[[pidx[q] for q in pat[si]]]
    for m in (models or ["L-IMG", "L-CNV", "L-LATE", "P refit", "CN joint package"]):
        if m not in S: continue
        X = S[m][si]
        o[(m, "stratum_across_folds", "sample")] = fsc(X, y[si], FO0[si], grp=g, fold=False); o[(m, "stratum_across_folds", "patient_weighted")] = fsc(X, y[si], FO0[si], w=WS[si], grp=g, fold=False)
        o[(m, "stratum_within_fold", "sample")] = fsc(X, y[si], FO0[si], grp=g); o[(m, "stratum_within_fold", "patient_weighted")] = fsc(X, y[si], FO0[si], w=WS[si], grp=g)
        o[(m, "unstratified_across_folds", "sample")] = fsc(X, y[si], FO0[si], fold=False); o[(m, "unstratified_across_folds", "patient_weighted")] = fsc(X, y[si], FO0[si], w=WS[si], fold=False)
    return o
def flat(d): return {"::".join(map(str, k)): (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
r3v = lambda v: [r3(x) for x in v] if isinstance(v, (list, tuple)) else r3(v)
# ================================================================== tasks
if TASK == "smoke":
    si, ei, pi = unit_index(np.arange(len(up))); A = auc_all(si, ei, pi, ["L-IMG", "L-CNV", "L-LATE", "CN joint package", "CN past-only", "CN per-sample", "P published"])
    res = {"sample_level": {m: r3(A[(m, "sample")]) for m in ("L-IMG", "L-CNV", "L-LATE", "CN joint package", "CN past-only", "CN per-sample", "P published")},
           "expected": {"L-IMG": 0.847, "L-CNV": 0.801, "L-LATE": 0.889, "CN joint package": 0.845, "CN past-only": 0.800, "CN per-sample": 0.785, "P published (fold-strat)": 0.823},
           "units": {"samples": int(len(rows)), "endoscopies": int(len(eu)), "patients": int(len(up)), "progressor_patients": int(P_y.sum())}}
    # old fusion family max-T (tr_triage.py, sample level): L-EARLY, L-INTER, L-LATE vs L-IMG
    Oe, Fe = _outer(sorted(glob.glob(f"{M}/cv/preds/cfg_03_rep_*.csv"))); Oi, Fi = _outer(sorted(glob.glob(f"{HZ}/outer/inter_their_rep_*.csv"))); assert (Fe == FO0).all() and (Fi == FO0).all()
    S["L-EARLY"] = Oe; S["L-INTER"] = Oi; fam = ["L-EARLY", "L-INTER", "L-LATE"]; bs = boots_for(rows)
    st = lambda b: [fsc(S[m][b], y[b], FO0[b]) - fsc(S["L-IMG"][b], y[b], FO0[b]) for m in fam]
    D0 = np.array(st(rows)); DB = np.array([st(b) for b in bs]); sd = DB.std(0, ddof=1); Tb = np.max(np.abs((DB - D0) / sd), 1)
    res["old_family_late_minus_img"] = {"delta": r3(D0[2]), "ci95": pct(DB[:, 2]), "p_unadjusted": p2(DB[:, 2]), "p_maxT_family_EARLY_INTER_LATE": round(float(np.mean(Tb >= abs(D0[2] / sd[2]))), 4), "expected": "+0.042 [+0.002, +0.078], max-T 0.066"}
    # refit anchors: 100-lambda class_min on set C rows must equal the stored L-CNV predictions (their: cfg 0; pkg: cfg 1), repeat 1 fold 1
    for feat, cfg in (("their", 0), ("pkg", 1)):
        f = f"{PF}/{feat}_C_nl100_rep_01_fold_01.csv"
        if os.path.exists(f):
            D = pd.read_csv(f, dtype={"Sample": str}); D = D[D.rule == "class_min"].set_index("Sample"); st0 = pd.read_csv(f"{M}/cv/preds/cfg_{cfg:02d}_rep_01.csv", dtype={"Sample": str}).set_index("Sample")
            st0 = st0[st0.fold == 1]; assert set(D.index) == set(st0.index)
            res[f"anchor_{feat}_C_nl100_class_min_vs_stored_cfg{cfg}"] = {"n": int(len(D)), "max_abs_diff": float(np.abs(D.pred - st0.pred.reindex(D.index)).max()), "lambda_refit": float(D["lambda"].iloc[0]), "lambda_stored": float(st0["lambda"].iloc[0])}
    dump(res, "smoke")
elif TASK == "point":
    si, ei, pi = unit_index(np.arange(len(up))); A = auc_all(si, ei, pi); res = {"models": MODELS, "primary_P_refit": PRIMARY_P, "contrasts": [f"{a} - {b}" for a, b in CONTRASTS]}
    res["units"] = {"samples": int(len(rows)), "progressor_samples": int(y[rows].sum()), "endoscopies": int(len(eu)), "progressor_endoscopies": int(E_y.sum()), "patients": int(len(up)), "progressor_patients": int(P_y.sum())}
    res["auroc"] = {m: {lv: r3(A[(m, lv)]) for lv in LEVELS} for m in MODELS}
    VT = {}
    for k, v in VAR.items():
        S["_v"] = v; set_units("_v", v); A1 = auc_all(si, ei, pi, ["_v"]); VT[k] = {lv: r3(A1[("_v", lv)]) for lv in LEVELS}
        D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(f"{PF}/{k.split('|')[0]}_{k.split('|')[1]}_nl1000_rep_*_fold_*.csv"))]); D = D[D.rule == k.split("|")[3]].drop_duplicates(["rep", "fold"])
        VT[k]["lambda_median"] = r3(D["lambda"].median()); VT[k]["nnz_median"] = float(D.nnz.median()); VT[k]["nnz_zero_folds"] = int((D.nnz == 0).sum()); VT[k]["n_train_mean"] = r3(D.n_train.mean())
    S.pop("_v", None); res["P_refit_variants"] = VT
    res["classification"] = {k: r3v(v) for k, v in flat(cls_stats(si, ei)).items()}
    res["calibration"] = {k: r3(v) for k, v in flat(calib(si)).items()}
    R, med = strata(); pcode = R.set_index("Patient").code.reindex(up).values.astype(int)
    res["matching"] = {"age_dx_median_75": r3(med), "strata": R.groupby("stratum").y.agg(["size", "sum"]).rename(columns={"size": "patients", "sum": "progressors"}).reset_index().assign(
        patients=lambda d: d.patients.map(lambda v: v if v >= 5 else "<5"), progressors=lambda d: d.progressors.map(lambda v: v if (v >= 5 or v == 0) else "<5")).to_dict("records"),
        "missing": {"age_dx": int(R.age_dx.isna().sum()), "sex": int(R.sex.isna().sum()), "prague_m": int(R.prague_m.isna().sum())}}
    sc = strat_conc(si, pcode); res["stratum_concordance"] = {k: r3(v) for k, v in flat(sc).items()}
    g = pcode[[pidx[q] for q in pat[si]]]; pr = 0
    for j in range(10):
        for k in np.unique(g):
            mk = g == k; pr += (y[si][mk] == 1).sum() * (y[si][mk] == 0).sum()
    res["stratum_pairs_share_of_all_pairs"] = r3(pr / 10 / ((y[si] == 1).sum() * (y[si] == 0).sum()))
    # conditional logistic (patient-mean logit, per SD), stratum = matching cell; unconditional logistic alongside
    from statsmodels.discrete.conditional_models import ConditionalLogit; import statsmodels.api as sm
    CL = {}
    for m in [x for x in ("L-IMG", "L-CNV", "L-LATE", "P refit", "CN joint package") if x in S]:
        z = np.array([logit(S[m][r]).mean() for r in P_rows]); z = (z - z.mean()) / z.std(ddof=1)
        fc = ConditionalLogit(P_y, z[:, None], groups=pcode).fit(disp=0); fu = sm.Logit(P_y, sm.add_constant(z)).fit(disp=0)
        CL[m] = {"conditional_OR_per_SD": r3(np.exp(fc.params[0])), "conditional_ci95": [r3(np.exp(fc.conf_int()[0][0])), r3(np.exp(fc.conf_int()[0][1]))], "unconditional_OR_per_SD": r3(np.exp(fu.params[1])),
                 "unconditional_ci95": [r3(np.exp(fu.conf_int()[1][0])), r3(np.exp(fu.conf_int()[1][1]))]}
    res["conditional_logistic_patient_mean_logit"] = CL; dump(res, "point")
elif TASK.startswith("boot_"):
    sh = int(TASK.split("_")[1]); DR = draws()[sh * PER:(sh + 1) * PER]; R, _ = strata(); pcode = R.set_index("Patient").code.reindex(up).values.astype(int); out = []
    for c in DR:
        si, ei, pi = unit_index(c); o = flat(auc_all(si, ei, pi)); o.update({k: v for k, v in flat(cls_stats(si, ei)).items()}); o.update(flat(calib(si))); o.update(flat(strat_conc(si, pcode))); out.append(o)
    pd.to_pickle(out, f"{ROWD}/boot/shard_{sh:02d}.pkl"); print("TP P1A DONE", TASK, len(out), flush=True)
elif TASK == "merge":
    PT = json.load(open(f"{OUTR}/point.json")); B = []
    for sh in range(NS): B += pd.read_pickle(need([f"{ROWD}/boot/shard_{sh:02d}.pkl"], 1)[0])
    assert len(B) == NB; si, ei, pi = unit_index(np.arange(len(up))); obs = flat(auc_all(si, ei, pi)); obs.update(flat(cls_stats(si, ei))); obs.update(flat(calib(si)))
    R, _ = strata(); pcode = R.set_index("Patient").code.reindex(up).values.astype(int); obs.update(flat(strat_conc(si, pcode)))
    arr = lambda k: np.array([b[k] for b in B], float)
    res = {"draws": NB, "primary_P_refit": PRIMARY_P, "auroc": {}, "differences": {}, "classification": {}, "calibration": {}, "stratum_concordance": {}}
    for m in MODELS: res["auroc"][m] = {lv: {"value": r3(obs[f"{m}::{lv}"]), "ci95": pct(arr(f"{m}::{lv}"))} for lv in LEVELS}
    for lv in LEVELS:
        D0 = np.array([obs[f"{a}::{lv}"] - obs[f"{b}::{lv}"] for a, b in CONTRASTS]); DB = np.column_stack([arr(f"{a}::{lv}") - arr(f"{b}::{lv}") for a, b in CONTRASTS])
        sd = np.nanstd(DB, 0, ddof=1); Tb = np.nanmax(np.abs((DB - D0) / sd), 1); To = np.abs(D0 / sd)
        res["differences"][lv] = {f"{a} - {b}": {"delta": r3(D0[i]), "ci95": pct(DB[:, i]), "se_boot": r3(sd[i]), "p_unadjusted": p2(DB[:, i]), "p_maxT": round(float(np.mean(Tb >= To[i])), 4)} for i, (a, b) in enumerate(CONTRASTS)}
        res["differences"][lv]["maxT_family"] = [f"{a} - {b}" for a, b in CONTRASTS]
        FAMS = {"two_primary_LATE-IMG_and_IMG-LCNV": [0, 1], "LATE-IMG_and_IMG-best_segmentation": [0, CONTRASTS.index(("L-IMG", "CN joint package"))], "CN_contrasts_only": list(range(1, len(CONTRASTS)))}
        if ("L-IMG", "P refit") in CONTRASTS: FAMS["LATE-IMG_and_IMG-Prefit"] = [0, CONTRASTS.index(("L-IMG", "P refit"))]
        res["differences"][lv]["maxT_other_families"] = {}
        for fn, ix in FAMS.items():
            Tf = np.nanmax(np.abs((DB[:, ix] - D0[ix]) / sd[ix]), 1); res["differences"][lv]["maxT_other_families"][fn] = {f"{CONTRASTS[i][0]} - {CONTRASTS[i][1]}": round(float(np.mean(Tf >= To[i])), 4) for i in ix}
    for k in obs:
        if k.count("::") == 2 and (k.endswith("::high") or k.endswith("::not_low")):
            a = np.array([b[k] for b in B], float); res["classification"][k] = {"sensitivity": r3(obs[k][0]), "sens_ci95": pct(a[:, 0]), "specificity": r3(obs[k][1]), "spec_ci95": pct(a[:, 1])}
        elif k.endswith("::slope") or k.endswith("::citl"): res["calibration"][k] = {"value": r3(obs[k]), "ci95": pct(arr(k))}
        elif "stratum" in k or "unstratified" in k: res["stratum_concordance"][k] = {"value": r3(obs[k]), "ci95": pct(arr(k))}
    for m in ("L-IMG", "L-CNV", "P refit", "CN joint package"):   # paired differences in stratum-restricted concordance
        if m == "L-IMG" or m not in S: continue
        for sc in ("stratum_across_folds", "stratum_within_fold"):
            for lv in ("sample", "patient_weighted"):
                d = arr(f"L-IMG::{sc}::{lv}") - arr(f"{m}::{sc}::{lv}"); res["stratum_concordance"][f"L-IMG - {m}::{sc}::{lv}"] = {"delta": r3(obs[f"L-IMG::{sc}::{lv}"] - obs[f"{m}::{sc}::{lv}"]), "ci95": pct(d), "p_unadjusted": p2(d)}
        for sc in ("stratum_across_folds", "stratum_within_fold"):
            for lv in ("sample", "patient_weighted"):
                d = arr(f"L-LATE::{sc}::{lv}") - arr(f"L-IMG::{sc}::{lv}"); res["stratum_concordance"][f"L-LATE - L-IMG::{sc}::{lv}"] = {"delta": r3(obs[f"L-LATE::{sc}::{lv}"] - obs[f"L-IMG::{sc}::{lv}"]), "ci95": pct(d), "p_unadjusted": p2(d)}
    dump(res, "main")
elif TASK == "f3":
    PREs = pd.Series(PRE, index=ids); ENDs = pd.Series(ENDO, index=ids)
    def rd(f): D = pd.read_csv(f, dtype={"Sample": str, "Patient": str}); D["pre"] = PREs.reindex(D.Sample).values; return D[D.pre].reset_index(drop=True)
    def lev(E, cols):
        yy = E.y.values.astype(int); F = np.tile(E.fold.values[:, None], (1, 10)); o = {}
        npre = E.groupby(["Patient", "copy"])["Sample"].transform("size").values; w = 1.0 / npre
        E = E.assign(endo=E["Sample"].map(ENDs).values + "|" + E["copy"].astype(str), unit=E["Patient"] + "|" + E["copy"].astype(str))
        for c in cols:
            X = np.tile(E[c].values[:, None], (1, 10)); o[(c, "sample")] = fsc(X, yy, F); o[(c, "patient_weighted")] = fsc(X, yy, F, w=w)
            for nm, key in (("endo", "endo"), ("pat", "unit")):
                G = E.groupby(key).agg(y=("y", "first"), fold=("fold", "first"), mx=(c, "max"), mn=(c, "mean"))
                Fg = np.tile(G.fold.values[:, None], (1, 10)); o[(c, f"{nm}_max")] = fsc(np.tile(G.mx.values[:, None], (1, 10)), G.y.values.astype(int), Fg); o[(c, f"{nm}_mean")] = fsc(np.tile(G.mn.values[:, None], (1, 10)), G.y.values.astype(int), Fg)
        return o
    out = []; okp = 0
    for d in range(200):
        fc, fi = f"{RR2}/f3/cnv_their_draw_{d:03d}_outer.csv", f"{RR2}/f3/img_draw_{d:03d}_outer.csv"
        if not (os.path.exists(fc) and os.path.exists(fi)): continue
        oc, oi = rd(fc), rd(fi); E = oc.merge(oi[["Sample", "copy", "pred"]], on=["Sample", "copy"], suffixes=("_cnv", "_img")); assert len(E) == len(oc) == len(oi); E["late"] = (E.pred_cnv + E.pred_img) / 2; cols = ["pred_cnv", "pred_img", "late"]
        pb = sorted(glob.glob(f"{PFB}/their_s773_nl1000_draw_{d:03d}_fold_*.csv"))
        if len(pb) == 10:
            Pa = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in pb])
            for rule, nm in (("class_1se", "prefit"), ("class_min", "prefit_min")):
                Pb = Pa[Pa.rule == rule][["Sample", "copy", "fold", "pred"]].rename(columns={"pred": nm, "fold": "fold_p"})
                E = E.merge(Pb, on=["Sample", "copy"], how="left"); assert E[nm].notna().all() and (E.fold_p == E.fold).all(); E = E.drop(columns="fold_p"); cols.append(nm)
            okp += 1
        o = lev(E, cols); o["draw"] = d; out.append(o)
    assert len(out) >= 100; res = {"draws_scored": len(out), "draws_with_P_refit": okp, "levels": {}}
    si, ei, pi = unit_index(np.arange(len(up))); obs = auc_all(si, ei, pi, [m for m in ("L-IMG", "L-CNV", "L-LATE", "P refit") if m in S])
    MB = json.load(open(f"{OUTR}/main.json")) if os.path.exists(f"{OUTR}/main.json") else None
    pairs = [("late", "pred_img", "L-LATE - L-IMG"), ("pred_img", "pred_cnv", "L-IMG - L-CNV")] + ([("pred_img", "prefit", "L-IMG - P refit"), ("pred_img", "prefit_min", "L-IMG - P refit (class_min)")] if okp == len(out) else [])
    for lv in LEVELS:
        L = {}
        for a, b, nm in pairs:
            d = np.array([o[(a, lv)] - o[(b, lv)] for o in out]); A_, B_ = nm.replace(" (class_min)", "").split(" - "); stored = obs[(A_, lv)] - obs[(B_, lv)] if (A_, lv) in obs and (B_, lv) in obs else None
            L[nm] = {"refit_draw_mean": r3(d.mean()), "refit_draw_median": r3(np.median(d)), "refit_ci95_percentile": pct(d), "refit_sd": r3(d.std(ddof=1)), "share_draws_le_0": r3(np.mean(d <= 0)),
                     "stored_delta": r3(stored) if "class_min" not in nm else None, "fixed_prediction_se": (MB["differences"][lv][nm]["se_boot"] if MB and nm in MB["differences"][lv] else None)}
        L["draw_mean_auroc"] = {c: r3(np.mean([o[(c, lv)] for o in out])) for c in ("pred_cnv", "pred_img", "late") + (("prefit", "prefit_min") if okp == len(out) else ())}
        res["levels"][lv] = L
    res["note"] = "one stratified 10-fold CV per draw (rv2_refit.R MODE=f3: lambda re-chosen; copies of a patient share every split); ~48 distinct patients per draw, so refit models are weaker and the interval pessimistic"
    dump(res, "refit_boot")
