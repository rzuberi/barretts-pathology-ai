"""Exploration E4 (docs/paper_explore_E4.md): evaluate time-aware refits (ex_e4_fit.R) against the stored R49 gate and H&E alone (P1).
rv8b_step3.py is executed unchanged up to its task dispatch (which executes rv8_landscape.py the same way), so endoscopy units, thresholds (thr_arr),
the R49 gate (TC.triage + r49_calls), patient outcomes (patient_arrays, parts, met) and bootstrap draws (pdraws, RandomState(0)) are round 8's.
The new scores are swapped in for the stored ones by replacing `load` before setup(): every threshold still comes from the inner out-of-fold predictions
of the fold's training patients. Row-level files stay on the cluster; aggregates (no identifiers, no dates) -> results/paper_final/explore_E4/<TASK>.json.
TASK = eval_<scheme> | merge.  EXE4_NB=k: smoke mode (k draws, aggregates written aside)."""
import os, sys, json, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
TASK_E4 = os.environ["TASK"]; os.environ["TASK"] = "none"
_s = open(HERE + "/rv8b_step3.py").read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], HERE + "/rv8b_step3.py", "exec"))
TASK = TASK_E4; OUTE = T + "/results/paper_final/explore_E4"; RE = M + "/explore_E4"; FITS = RE + "/fits"
if os.environ.get("EXE4_NB"): NB = int(os.environ["EXE4_NB"]); TC.NB = NB; OUTE = RE + "/smoke_json"
os.makedirs(OUTE, exist_ok=True)
SV = pd.read_csv(RE + "/surv.csv", dtype={"Sample": str, "Patient": str}).set_index("Sample").reindex(ids); assert (SV.y.values == y).all()
T_STUDY = SV.t_study.values; T_EXT = SV.t_ext.values; T_SHEET = SM.time.values.astype(float); EVT = y.copy()
STORED = None
def stored():
    global STORED
    if STORED is None: STORED = TC.load("their")
    return STORED
def read_cfg(cfg):
    fo = sorted(glob.glob(f"{FITS}/{cfg}/outer_rep_*.csv")); fi = sorted(glob.glob(f"{FITS}/{cfg}/inner_rep_*_fold_*.csv"))
    assert len(fo) == 10 and len(fi) == 100, (cfg, len(fo), len(fi))
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); P = D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values
    F = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int); assert np.isfinite(P).all()
    Di = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fi]); I = {(r, k): pd.Series(d.pred.values, index=ix[d.Sample].values) for (r, k), d in Di.groupby(["rep", "fold"])}
    meta = {"lambda_median": r3(D.lambda_.median()) if "lambda_" in D else r3(D["lambda"].median()), "nnz_median": r3(D.nnz.median()), "n_train_median": r3(D.n_train.median()) if "n_train" in D else None}
    return P, I, F, meta
def logit_z(P, I):   # stored probabilities -> logit (used only when a stored block is fused with a z-scored Cox block)
    return lg(P), {k: pd.Series(lg(v.values), index=v.index) for k, v in I.items()}
# scheme -> (H&E source, copy-number source, fusion rule, gate band source, gate middle source)
KINDS = ("coxx", "coxs", "coxp", "h3", "h5", "wt", "wt2")
SCHEMES = {k: {"img": f"{k}_img", "cnv": f"{k}_cnv"} for k in KINDS}
SCHEMES["stored"] = {"img": "stored", "cnv": "stored"}   # control: must reproduce R49 / P1 with all deltas 0
SCHEMES.update({"coxx_img+stored_cnv": {"img": "coxx_img", "cnv": "stored"}, "stored_img+coxx_cnv": {"img": "stored", "cnv": "coxx_cnv"},
                "wt_img+stored_cnv": {"img": "wt_img", "cnv": "stored"}, "h5_img+stored_cnv": {"img": "h5_img", "cnv": "stored"}})
# added after the first results: ensembles of the stored and a time-aware model per modality, and the stored gate bands with a time-aware fusion in the middle band
SCHEMES.update({"coxp_img+stored_cnv": {"img": "coxp_img", "cnv": "stored"},
                "ens_h3": {"img": ["stored", "h3_img"], "cnv": ["stored", "h3_cnv"]}, "ens_h5": {"img": ["stored", "h5_img"], "cnv": ["stored", "h5_cnv"]},
                "ens_coxx": {"img": ["stored", "coxx_img"], "cnv": ["stored", "coxx_cnv"]}, "ens_wt2": {"img": ["stored", "wt2_img"], "cnv": ["stored", "wt2_cnv"]},
                "stored+h3late": {"img": "stored", "cnv": "stored", "late": ["h3_img", "h3_cnv"]}, "stored+h5late": {"img": "stored", "cnv": "stored", "late": ["h5_img", "h5_cnv"]},
                "stored+ensh3late": {"img": "stored", "cnv": "stored", "late": [["stored", "h3_img"], ["stored", "h3_cnv"]]}})
def build(sch):
    """O, I, FO with keys L-IMG, L-CNV, L-LATE for the scheme. A component is 'stored' (L-IMG / L-CNV by modality), a fitted config, or a list of these (mean).
    Combining: probabilities only -> mean of probabilities (as the stored L-LATE); if any component is a Cox linear predictor (z-scored on its model's training rows),
    every binary component is put on the logit scale and z-scored per (repeat, fold) on that fold's inner OOF values (training patients), then z's are averaged.
    L-LATE = combination of the H&E and copy-number components (or of the scheme's 'late' override)."""
    O0, I0, FO0_ = stored(); spec = SCHEMES[sch]; meta = {}
    def get(src, mod):
        if src == "stored": meta.setdefault(mod, "stored"); return O0[mod], I0[mod], False
        P, II, F, mt = read_cfg(src); assert (F == FO0_).all(), "folds differ from the stored folds"; meta[src] = mt; return P, II, src.startswith("cox")
    def zify(P, II):
        Pl, Il = logit_z(P, II); Oz = np.zeros_like(Pl); Iz = {}
        for j, r in enumerate(REPS):
            for k in range(1, 11):
                v = Il[(r, k)]; mu, sd = v.mean(), v.std(ddof=1); te = FO0_[:, j] == k; Oz[te, j] = (Pl[te, j] - mu) / sd; Iz[(r, k)] = (v - mu) / sd
        return Oz, Iz
    def leaves(x, mod): return [(x, mod)] if isinstance(x, str) else [l for e in x for l in leaves(e, mod)]
    def combine(parts):
        G = [get(s, m) for s, m in parts]
        if len(G) == 1: return G[0][0], G[0][1]
        if any(g[2] for g in G): G = [(g[0], g[1]) if g[2] else zify(g[0], g[1]) for g in G]
        Oc = sum(g[0] for g in G) / len(G); keys = G[0][1].keys(); Ic = {k: sum(g[1][k].reindex(G[0][1][k].index) for g in G) / len(G) for k in keys}
        return Oc, Ic
    O, I = {}, {}
    O["L-IMG"], I["L-IMG"] = combine(leaves(spec["img"], "L-IMG")); O["L-CNV"], I["L-CNV"] = combine(leaves(spec["cnv"], "L-CNV"))
    late = spec.get("late", [spec["img"], spec["cnv"]]); O["L-LATE"], I["L-LATE"] = combine(leaves(late[0], "L-IMG") + leaves(late[1], "L-CNV"))
    meta = {"components": {k: str(v) for k, v in spec.items()}, "fits": meta}
    return O, I, FO0_, meta
def setup_with(O, I, FO):
    global load
    _old = load; load = lambda src: (O, I, FO)
    try: return setup()
    finally: load = _old
# ------------------------------------------------------------------ discrimination: fold-stratified AUROC, IPCW time-dependent AUROC (hf_foldstrat.py), Harrell C
def auc_fs(X, lab, F, idx):
    """fold-stratified AUROC (pairs within the same outer fold), mean over repeats; X, F (N, 10); idx rows (bootstrap duplicates allowed)"""
    out = []
    for j in range(10):
        x = X[idx, j]; f = F[idx, j]; l = lab[idx]; num = den = 0.0
        for k in np.unique(f):
            m = f == k; xp = x[m & (l == 1)]; xn = x[m & (l == 0)]
            if len(xp) == 0 or len(xn) == 0: continue
            xs = np.sort(xn); num += (np.searchsorted(xs, xp, "left") + 0.5 * (np.searchsorted(xs, xp, "right") - np.searchsorted(xs, xp, "left"))).sum(); den += len(xp) * len(xn)
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
def Gminus(tt, dd, at):
    u = np.unique(tt); nr = (tt[None, :] >= u[:, None]).sum(1); dc = ((tt[None, :] == u[:, None]) & (dd[None, :] == 0)).sum(1); sv = np.cumprod(1 - dc / nr); pv = np.concatenate([[1.0], sv[:-1]])
    k = np.searchsorted(u, at, side="left"); o = np.ones(len(at)); i = k < len(u); o[i] = pv[k[i]]; o[~i] = sv[-1]; return np.clip(o, 1e-12, None)
def td_auc(X, F, idx, t, tt):
    """hf_foldstrat.py fs(): IPCW time-dependent AUROC at t, cases event & time <= t, controls time >= t without event or event after t; within-fold pairs; repeat mean"""
    ti, di = tt[idx], EVT[idx]; case = (di == 1) & (ti <= t); ctrl = ((di == 0) & (ti >= t)) | ((di == 1) & (ti > t))
    if not case.any() or not ctrl.any(): return np.nan
    w = 1 / Gminus(ti, di, ti[case]); out = []
    for j in range(10):
        x = X[idx, j]; f = F[idx, j]; num = den = 0.0; xc, fc = x[case], f[case]
        for k in np.unique(fc):
            xn = np.sort(x[ctrl & (f == k)]); m = fc == k
            if not len(xn): continue
            xp = xc[m]; ww = w[m]; lo = np.searchsorted(xn, xp, "left"); hi = np.searchsorted(xn, xp, "right"); num += (ww * (lo + 0.5 * (hi - lo))).sum(); den += (ww * len(xn)).sum()
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
def cindex(X, F, idx, tt):
    """fold-stratified Harrell C: comparable pairs (i event, t_i < t_j) within the same outer fold; ties in score 0.5; repeat mean"""
    ti, di = tt[idx], EVT[idx]; out = []
    for j in range(10):
        x = X[idx, j]; f = F[idx, j]; num = den = 0.0
        for k in np.unique(f):
            m = f == k; a, b, s = ti[m], di[m], x[m]; cmp_ = (b[:, None] == 1) & (a[:, None] < a[None, :])
            num += (cmp_ * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))).sum(); den += cmp_.sum()
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
def disc(O, FO, S, boots, O_ref):
    """discrimination table for the scheme's scores against the stored scores (paired bootstrap of the deltas over the same patient draws)"""
    rows = np.where(PRE)[0]; EF = S["EF"]; elab = S["elab"]; Erows = np.arange(S["E"])
    emax = lambda X: np.stack([np.array([X[s, j].max() for s in S["es"]]) for j in range(10)], 1)
    def metrics(OO, b, eb):
        r = {}
        for m in ("L-IMG", "L-CNV", "L-LATE"):
            X = OO[m]; r[f"auroc_sample_{m}"] = auc_fs(X, y, FO, b); r[f"auroc_endoscopy_max_{m}"] = auc_fs(OO["_E"][m], elab, EF, eb)
            for h in (1, 3, 5): r[f"td_auroc_{h}y_{m}"] = td_auc(X, FO, b, h, T_SHEET)
            r[f"cindex_study_{m}"] = cindex(X, FO, b, T_STUDY); r[f"cindex_ext_{m}"] = cindex(X, FO, b, T_EXT)
        return r
    for OO in (O, O_ref): OO["_E"] = {m: emax(OO[m]) for m in ("L-IMG", "L-CNV", "L-LATE")}
    ob = metrics(O, rows, Erows); obr = metrics(O_ref, rows, Erows)
    e_of_s = S["SEi"]; bn = []; br = []
    for b in boots:
        eb = e_of_s[b]; eb = eb[eb >= 0]; bn.append(metrics(O, b, eb)); br.append(metrics(O_ref, b, eb))
    out = {}
    for k in ob:
        d = np.array([x[k] - z[k] for x, z in zip(bn, br)]); out[k] = {"scheme": r3(ob[k]), "stored": r3(obr[k]), "delta": dstat(ob[k] - obr[k], d), "scheme_ci95": ci2([x[k] for x in bn])}
    for OO in (O, O_ref): OO.pop("_E")
    return out
# ------------------------------------------------------------------ rules and paired contrasts against R49 / P1 (stored)
def rule_set(S, O, I, FO):
    X = S["OUTS"]; Z = np.zeros((S["E"], 10), int); R = {}
    for q in Q3: R[f"P1-new {int(q * 100)}%"] = (X["img"] >= thr_arr(S, "img", q)[0], Z)
    R["P3-new 90%"] = (X["late"] >= thr_arr(S, "late", 0.90)[0], np.repeat(S["nsz"][:, None], 10, 1))
    for ql, qh in ((0.95, 0.90), (0.90, 0.90), (0.95, 0.95), (0.90, 0.95)):
        _, _, info = TC.triage(O, I, FO, S["popm"], ql, qh); F, NS, _ = r49_calls(O, info, FO, S["es"]); R[f"R49-new t_low {int(ql * 100)}/t_high {int(qh * 100)}"] = (F, NS)
    return R
def contrasts(S, Sref, RU, RUref, D):
    nP = int((S["plab"] == 1).sum()); nN = int((S["plab"] == 0).sum()); one = np.ones(S["P_"]); Z = np.zeros((S["E"], 10), int); Pe = S["elab"] == 1
    MET = {"endoscopy_specificity": lambda p: met(p, "endo_spec"), "endoscopy_sensitivity": lambda p: met(p, "endo_sens"), "non_progressor_patients_flagged": lambda p: nN * met(p, "np_flagged_share"),
           "progressor_patients_caught": lambda p: nP * met(p, "caught_share"), "samples_sequenced_per_patient": lambda p: met(p, "seq_per_patient"), "flags_per_np_patient_year": lambda p: met(p, "flags_per_np_py")}
    AR = {k: patient_arrays(S, F, NS) for k, (F, NS) in RU.items()}; ARr = {k: patient_arrays(Sref, F, NS) for k, (F, NS) in RUref.items()}
    def mstats(p): return {k: f(p) for k, f in MET.items()}
    def matched(Xe, m, sens):   # endoscopy score Xe (E, 10) thresholded per repeat to the target realised endoscopy sensitivity on the (weighted) evaluated endoscopies
        w = m[S["pidx"]].astype(int); F = np.zeros((S["E"], 10), bool)
        for j in range(10): F[:, j] = Xe[:, j] >= tmatch(Xe[Pe, j], sens[j], w[Pe])
        return F
    res = {}
    PO = {k: parts(S, AR[k], one) for k in AR}; PB = {k: parts(S, AR[k], D) for k in AR}; POr = {k: parts(Sref, ARr[k], one) for k in ARr}; PBr = {k: parts(Sref, ARr[k], D) for k in ARr}
    for k in AR:
        o = {"rule": {m: r3(v[0]) for m, v in mstats(PO[k]).items()}, "rule_ci95": {m: ci2(v) for m, v in mstats(PB[k]).items()}}
        for ref in ("R49", "P1 90%"):
            a = mstats(PO[k]); b = mstats(POr[ref]); ab = mstats(PB[k]); bb = mstats(PBr[ref])
            o[f"delta_vs_{ref}"] = {m: dstat(a[m][0] - b[m][0], ab[m] - bb[m]) for m in MET}
        sk = "stored " + k   # the same rule (same cut-off cell) built on the stored scores: the gain from the new scores alone
        if sk in POr:
            a = mstats(PO[k]); b = mstats(POr[sk]); ab = mstats(PB[k]); bb = mstats(PBr[sk])
            o["delta_vs_stored_same_rule"] = {m: dstat(a[m][0] - b[m][0], ab[m] - bb[m]) for m in MET}; o["stored_same_rule"] = {m: r3(v[0]) for m, v in b.items()}
        res[k] = o
    # matched endoscopy sensitivity: new endoscopy scores (max H&E = P1-new; max fusion = P3-new) and the stored max L-IMG, each thresholded to R49's / P1 90%'s realised sensitivity
    X = S["OUTS"]; Xr = Sref["OUTS"]; MATCH = {}
    for ref in ("R49", "P1 90%"):
        def one_draw(m):
            pr = parts(Sref, ARr[ref], m); sens = per_rep(pr, "endo_sens")[0]; out = {"ref": {k: f(pr)[0] for k, f in MET.items()}}
            for nm, Xe, SS in (("new H&E max (P1-new score)", X["img"], S), ("new fusion max (P3-new score, sequence all)", X["late"], S), ("stored H&E max (P1 score)", Xr["img"], Sref), ("stored fusion max (P3 score)", Xr["late"], Sref)):
                Z_ = np.zeros((S["E"], 10), int) if "H&E" in nm else np.repeat(S["nsz"][:, None], 10, 1)
                p = parts(SS, patient_arrays(SS, matched(Xe, m, sens), Z_), m); out[nm] = {k: f(p)[0] for k, f in MET.items()}
            return out
        ob = one_draw(one); bd = [one_draw(m) for m in D]; R_ = {"reference": {k: r3(v) for k, v in ob["ref"].items()}}
        for nm in ob:
            if nm == "ref": continue
            R_[nm] = {"at_matched": {k: r3(v) for k, v in ob[nm].items()},
                      "delta_vs_reference": {k: dstat(ob[nm][k] - ob["ref"][k], np.array([b_[nm][k] - b_["ref"][k] for b_ in bd])) for k in ("endoscopy_specificity", "non_progressor_patients_flagged", "progressor_patients_caught", "endoscopy_sensitivity")}}
        for nm in ("new H&E max (P1-new score)", "new fusion max (P3-new score, sequence all)"):
            st = "stored H&E max (P1 score)" if "H&E" in nm else "stored fusion max (P3 score)"
            R_[nm]["delta_vs_stored_score_at_same_sensitivity"] = {k: dstat(ob[nm][k] - ob[st][k], np.array([b_[nm][k] - b_[st][k] for b_ in bd])) for k in ("endoscopy_specificity", "non_progressor_patients_flagged", "progressor_patients_caught")}
        MATCH[ref] = R_
    # gate against P1 (stored) matched to the gate's own realised sensitivity (S1 (a)(i) form)
    GM = {}
    for k in [x for x in AR if x.startswith("R49-new")]:
        def gd(m):
            pr = parts(S, AR[k], m); sens = per_rep(pr, "endo_sens")[0]; p1 = parts(Sref, patient_arrays(Sref, matched(Xr["img"], m, sens), Z), m)
            return {kk: (f(pr)[0], f(p1)[0]) for kk, f in MET.items()}
        ob = gd(one); bd = [gd(m) for m in D]
        GM[k] = {kk: dstat(ob[kk][0] - ob[kk][1], np.array([b_[kk][0] - b_[kk][1] for b_ in bd])) for kk in ("endoscopy_specificity", "non_progressor_patients_flagged", "progressor_patients_caught")}
    return res, MATCH, GM
def apparent(S):
    """optimistic: P1-new threshold set on the evaluated (held-out) endoscopies themselves at 90% endoscopy sensitivity, per repeat"""
    X = S["OUTS"]["img"]; Pe = S["elab"] == 1; F = np.zeros_like(X, bool)
    for j in range(10): F[:, j] = X[:, j] >= thr_sens(X[Pe, j], 0.90)
    p = parts(S, patient_arrays(S, F, np.zeros_like(X, int)), np.ones(S["P_"])); nP = int((S["plab"] == 1).sum()); nN = int((S["plab"] == 0).sum())
    return {"P1-new at 90% endoscopy sensitivity set on the evaluated patients (OPTIMISTIC)": {"caught": r2(nP * met(p, "caught_share")[0]), "np_flagged": r2(nN * met(p, "np_flagged_share")[0]),
            "endoscopy_sensitivity": r3(met(p, "endo_sens")[0]), "endoscopy_specificity": r3(met(p, "endo_spec")[0])}}
# ================================================================== tasks
res = {"task": TASK}
if TASK.startswith("eval_"):
    sch = TASK[5:]; O, I, FO, meta = build(sch); S = setup_with(O, I, FO)
    O0, I0, FO0_ = stored(); Sref = setup_with(O0, I0, FO0_)
    D = pdraws(S); RU = rule_set(S, O, I, FO); RUref = rules(Sref); RUref.update({"stored " + k: v for k, v in rule_set(Sref, O0, I0, FO0_).items()})
    C, MATCH, GM = contrasts(S, Sref, RU, RUref, D)
    boots = boots_for(np.where(PRE)[0], 0, NB)
    DI = disc(O, FO, S, boots, O0)
    chk = {k: {"caught": r2(32 * met(parts(Sref, patient_arrays(Sref, *RUref[k]), np.ones(Sref["P_"])), "caught_share")[0]), "np_flagged": r2(43 * met(parts(Sref, patient_arrays(Sref, *RUref[k]), np.ones(Sref["P_"])), "np_flagged_share")[0])} for k in ("R49", "P1 90%")}
    res.update({"scheme": sch, "sources": SCHEMES[sch], "fit_meta": meta, "rules": C, "matched_sensitivity": MATCH, "gate_vs_P1_matched_to_gate": GM, "discrimination": DI, "apparent": apparent(S),
                "check_reference": {"got": chk, "expected": {"R49": [31.1, 31.8], "P1 90%": [31.1, 36.0]}, "reproduced": bool(abs(chk["R49"]["caught"] - 31.1) < 0.051 and abs(chk["R49"]["np_flagged"] - 31.8) < 0.051 and abs(chk["P1 90%"]["np_flagged"] - 36.0) < 0.051)},
                "draws": int(len(D)), "auroc_draws": int(len(boots)),
                "note": "delta = scheme - stored reference; single model (per-repeat values averaged over 10 repeats); paired patient bootstrap (rv8 pdraws, RandomState(0)) with thresholds fixed, "
                        "except matched rules whose per-repeat threshold is re-matched in every draw; discrimination deltas use tb_common.boots_for (RandomState(0)) on pre-event samples"})
    json.dump(res, open(f"{OUTE}/{TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("EXE4 DONE", TASK, flush=True)
else: raise SystemExit(f"unknown TASK {TASK}")
