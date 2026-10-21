"""Exploration E8 (docs/paper_explore_E8.md; rules paper/reviews_round7/decisions/EXPLORE_RULES.md): is the quiet-genome check real biology, can it be sharper or cheaper?
EXPLORATORY. The E2 module (ex_e2_veto.py) is executed unchanged up to its dispatch (as paper/story_deck/cluster/story_aggs_v2.py does), so A12 (stored round-9 flags) and the
final rule (A12 + ps_oac k2 cn90) are rebuilt with the exact E2 code; TASK=soft additionally executes rv9_falseflags.py unchanged up to its dispatch and re-runs the A12 nested
engine per outer fold (asserted equal to the stored flags) to get training-patient running scores. Every bar / step size is set on the fold's training patients; evaluation on
held-out patients; paired patient-bootstrap (round-8 pdraws, 2,000, RandomState(0)). Aggregates only -> results/paper_final/explore_E8/<TASK>.json.
TASK = base | qc | noise | arms | rand | conc | flat | seq | soft | time"""
import os, sys, json, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); TASK = os.environ["TASK"]; sys.path.insert(0, HERE)
os.environ["TASK"] = "__ex_e8_import__"; NS2 = {"__file__": HERE + "/ex_e2_veto.py", "__name__": "ex_e2_e8"}
_s = open(HERE + "/ex_e2_veto.py").read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], HERE + "/ex_e2_veto.py", "exec"), NS2)
os.environ["TASK"] = TASK
g = NS2; M = g["M"]; T = g["T"]; lg = g["lg"]; thr_sens = g["thr_sens"]; dstat = g["dstat"]; m5 = g["m5"]; r2 = g["r2"]; r3 = g["r3"]; ci2 = g["ci2"]; wauc = g["wauc"]
patient_arrays = g["patient_arrays"]; lstats = g["lstats"]; ids = g["ids"]; n = g["n"]; REPS = g["REPS"]
OUT = T + "/results/paper_final/explore_E8"; XD = M + "/explore_E8"
for d_ in (OUT, XD): os.makedirs(d_, exist_ok=True)
def dump(res, name=None):
    json.dump(res, open(f"{OUT}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("EXE8 DONE", name or TASK, flush=True)
S = g["build"](); FT = g["load_feat"](); E = S["E"]; EF = S["EF"]; TRX = S["TRX"]; TOPO = S["TOPO"]; GB = S["GB"]; nsz = S["nsz"]; elab = S["elab"]; pidx = S["pidx"]; plab = S["plab"]
D = S["D"]; INC = S["INC"]; P_ = S["P_"]; isP = plab == 1; isN = ~isP; nP = int(isP.sum()); nN = int(isN.sum()); one = np.ones((1, P_)); TTE = S["TTE"]; ORD = S["ORD"]
R9 = M + "/review_round9"; Z = np.load(R9 + "/grp_A12.npz"); FM = Z["A12__sel_np_F"].astype(bool); NM = Z["A12__sel_np_NS"].astype(int)
F8 = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(XD + "/feat8_*.csv"))]).drop_duplicates("Sample").set_index("Sample").reindex(ids)
k2 = np.minimum(2, nsz)[:, None]; T1 = TOPO[:, :, 0]; T2 = TOPO[:, :, 1]
# ------------------------------------------------------------------ arms (per-sample segmentation, E2 tile_feats deviations)
ARMS = [c[5:] for c in FT.columns if c.startswith("ps_A_")]; AD = {a: FT[f"ps_A_{a}"].values.astype(float) for a in ARMS}
ARMS = [a for a in ARMS if np.isfinite(AD[a]).mean() > 0.9 and a not in ("13p", "14p", "15p", "21p", "22p")]
_h = pd.read_csv(M + "/review_round4/reseg/tiles_persample_analysed.csv", nrows=0).columns; _ac = [c for c in _h if c not in ("Sample", "qc_pass", "cx_raw") and int(c.split(":")[1].split("-")[1]) - int(c.split(":")[1].split("-")[0]) >= 6e6]
ALEN = dict(zip(g["arm_names"](_ac), [int(c.split(":")[1].split("-")[1]) - int(c.split(":")[1].split("-")[0]) for c in _ac]))
LOSS, GAIN = g["LOSS"], g["GAIN"]; OAC = LOSS + GAIN; SIGN = {**{a: -1 for a in LOSS}, **{a: 1 for a in GAIN}}
def armscore(arms, signs=None, absolute=False):
    """sum over arms of the change in the given direction (relative copy number deviation; missing = 0, as E2's ps_oac)"""
    x = np.zeros(n)
    for a in arms:
        v = np.nan_to_num(AD[a]); x += np.abs(v) if absolute else np.maximum(0, (signs or SIGN)[a] * v)
    return x
q_oac = FT.ps_oac.values.astype(float); assert np.allclose(armscore(OAC), q_oac), "OAC arm score does not reproduce ps_oac"
ARMSD = 1.4826 * np.nanmedian(np.abs(np.stack([AD[a] for a in ARMS], 1) - np.nanmedian(np.stack([AD[a] for a in ARMS], 1), 1, keepdims=True)), 1)   # robust arm-level noise
# ------------------------------------------------------------------ veto machinery on A12
def V_of(f, k=2): return g["vs_test"](S, f, k), {kk: g["vs_train"](S, f, tx["top"], k) for kk, tx in TRX.items()}
def bar(vtr, q=0.90):
    TH = np.full((E, 10), np.nan)
    for (j, kk), tx in TRX.items():
        x = vtr[(j, kk)]; x = x[(tx["lab"] == 1) & np.isfinite(x)]; TH[EF[:, j] == kk, j] = thr_sens(x, q) if len(x) else -np.inf
    return TH
def apply(V, TH, can=True, k=2):
    veto = FM & (V < TH) & can; F = FM & ~veto; kk = np.minimum(k, nsz)[:, None]; return F, NM + np.where(FM, np.where(GB == 1, kk - 1, kk), 0)
V_oac, vtr_oac = V_of(q_oac); TH_oac = bar(vtr_oac); _, _, TH_e2 = g["veto_V"](S, "R49hm", V_oac, vtr_oac, "cn90", 2); assert np.array_equal(TH_oac, TH_e2)
QUIET = V_oac < TH_oac; F_fin, NS_fin = apply(V_oac, TH_oac)
KEYS = (("np_flagged_share", nN, "np_flagged"), ("caught_share", nP, "caught"), ("seq_per_patient", 1, "seq"), ("endo_spec", 1, "endo_spec"), ("cost_per_patient", 1, "cost"))
RA = {}
for nm_, (F_, NS_) in (("A12", (FM, NM)), ("final", (F_fin, NS_fin))): A_ = patient_arrays(S, F_, NS_); RA[nm_] = (lstats(S, A_, one), lstats(S, A_, D))
def obs(F, NS):
    a = lstats(S, patient_arrays(S, F, NS), one); return {"caught": float(a["caught_share"][0] * nP), "np_flagged": float(a["np_flagged_share"][0] * nN), "seq": float(a["seq_per_patient"][0])}
def full(F, NS, label, params=None, extra=None):
    c = g["summarize"](S, F, NS, label, params or {}); A = patient_arrays(S, F, NS); a0 = lstats(S, A, one); ab = lstats(S, A, D)
    for ref, (r0, rb) in RA.items(): c["contrasts"][ref] = {nm: dstat(mult * float(a0[key][0] - r0[key][0]), mult * (ab[key] - rb[key])) for key, mult, nm in KEYS}
    s = g["short"](c); s.update({"label": label, "params": params or {}, "dNP_vs_A12": c["contrasts"]["A12"]["np_flagged"], "dP_vs_A12": c["contrasts"]["A12"]["caught"],
                                 "dNP_vs_final": c["contrasts"]["final"]["np_flagged"], "dP_vs_final": c["contrasts"]["final"]["caught"], "dseq_vs_final": c["contrasts"]["final"]["seq"],
                                 "dspec_matched_vs_P1": c["contrasts"].get("P1_matched_sens_d_endo_spec"), "cost_per_patient_gbp": c["cost_per_patient_gbp"]})
    if extra: s.update(extra)
    return s
def apparent_bar(V, q=0.90, can=True, k=2):
    """OPTIMISTIC: bar at share q of the held-out progressor endoscopies of each repeat (evaluation patients)"""
    TH = np.full((E, 10), np.nan)
    for j in range(10): x = V[(elab == 1) & np.isfinite(V[:, j]), j]; TH[:, j] = thr_sens(x, q)
    F, NS = apply(V, TH, can, k); o = obs(F, NS); return {**{kk: r2(v) for kk, v in o.items()}, "label": "OPTIMISTIC (bar set on the evaluation patients)"}
def veto_cell(f, label, k=2, q=0.90, can=None, params=None, extra=None, app=True):
    V, vtr = V_of(f, k); TH = bar(vtr, q); F, NS = apply(V, TH, True if can is None else can, k); s = full(F, NS, label, params, extra)
    if app: s["apparent"] = apparent_bar(V, q, True if can is None else can, k)
    return s, F, NS
def pair_feat(x, how="mean"):
    """endoscopy x repeat value of a sample feature over the two top-H&E samples (mean / max / min / all-true)"""
    a = x[T1]; b = np.where(T2 >= 0, x[np.maximum(T2, 0)], np.nan)
    if how == "mean": return np.nanmean(np.stack([a, b]), 0)
    if how == "max": return np.fmax(a, b)
    if how == "min": return np.fmin(a, b)
    if how == "all": return a.astype(bool) & np.where(T2 >= 0, b, 1).astype(bool)
def pooled_auc(X, lab, mask):
    """AUROC of X (higher = label 1) within mask, per repeat pooled over folds, mean over repeats; patient-bootstrap CI (multiplicity weights)"""
    ob = []; bs = np.zeros((len(D), 10))
    for j in range(10):
        mk = mask[:, j] & np.isfinite(X[:, j]); lb = lab[:, j][mk] if lab.ndim == 2 else lab[mk]
        if len(np.unique(lb)) < 2: ob.append(np.nan); bs[:, j] = np.nan; continue
        ob.append(wauc(X[mk, j], lb, np.ones(mk.sum())))
        for bi in range(len(D)): bs[bi, j] = wauc(X[mk, j], lb, D[bi][pidx[mk]])
    return {"auroc": r3(np.nanmean(ob)), "ci95": ci2(np.nanmean(bs, 1))}
def gmed(X, mask): v = X[mask]; v = v[np.isfinite(v)]; return r3(np.median(v)) if len(v) else None
res = {"task": TASK, "baseline": {"A12": {k_: r2(v) for k_, v in obs(FM, NM).items()}, "final": {k_: r2(v) for k_, v in obs(F_fin, NS_fin).items()}}}
CLS = (("non_progressor", 0), ("progressor", 1)); EL = np.repeat(elab[:, None], 10, 1)
# ================================================================== base: reproduction + reference rows
if TASK == "base":
    res["A12_full"] = full(FM, NM, "A12 (stored round-9 memory + certificate)"); res["final_full"] = full(F_fin, NS_fin, "final rule: A12 + quiet-genome check (ps_oac k2 cn90)")
    res["reproduced"] = bool(abs(res["baseline"]["final"]["caught"] - 30.5) < 0.051 and abs(res["baseline"]["final"]["np_flagged"] - 17.2) < 0.051 and abs(res["baseline"]["final"]["seq"] - 3.82) < 0.006 and abs(res["baseline"]["A12"]["np_flagged"] - 21.5) < 0.051)
    res["A12_flagged_endoscopies_per_repeat"] = {cl: r2(float((FM & (EL == v)).sum(0).mean())) for cl, v in CLS}
    res["quiet_among_A12_flagged_per_repeat"] = {cl: r2(float((FM & QUIET & (EL == v)).sum(0).mean())) for cl, v in CLS}
    dump(res)
# ================================================================== 1. technical confounding
elif TASK == "qc":
    R = {}; cell = FT.cellularity.values.astype(float); reads = np.where(np.isfinite(F8.total_reads.values), F8.total_reads.values, FT.reads.values.astype(float))
    MET = {"log10_reads": np.log10(reads), "log_mapd_50kb": np.log(F8.mapd.values), "log_segment_residual_mad": np.log(F8.segmad.values), "qdnaseq_expected_variance": F8.expvar.values,
           "arm_level_robust_sd": ARMSD, "barretts_cellularity_pct": cell, "tile_qc_pass": FT.ps_qc.values.astype(float)}
    R["missing_samples"] = {k_: int(np.isnan(v[S["rows"]]).sum()) for k_, v in MET.items()}
    desc = {}
    for nm, x in MET.items():
        X = pair_feat(x, "mean"); o = {}
        for cl, v in CLS:
            mq = FM & QUIET & (EL == v); mn = FM & ~QUIET & (EL == v)
            o[cl] = {"median_quiet": gmed(X, mq), "median_not_quiet": gmed(X, mn), "auroc_metric_higher_in_quiet": pooled_auc(X, QUIET.astype(int), FM & (EL == v))}
        o["all_endoscopies_quiet_vs_not_by_class"] = {cl: pooled_auc(X, QUIET.astype(int), (EL == v) & np.isfinite(V_oac)) for cl, v in CLS}
        desc[nm] = o
    R["quiet_vs_not_among_A12_flagged"] = desc
    R["n_A12_flagged_endoscopies_per_repeat"] = {cl: {"quiet": r2(float((FM & QUIET & (EL == v)).sum(0).mean())), "not_quiet": r2(float((FM & ~QUIET & (EL == v)).sum(0).mean()))} for cl, v in CLS}
    # endoscopy-level logistic model of quiet (majority over repeats among A12-flagged repeats), cluster-robust by patient
    import statsmodels.api as sm
    fl = FM.sum(1) >= 5; qmaj = np.array([QUIET[e, FM[e]].mean() >= 0.5 if FM[e].any() else False for e in range(E)])
    cov = {nm: np.array([np.nanmean(pair_feat(MET[nm], "mean")[e, FM[e]]) if FM[e].any() else np.nan for e in range(E)]) for nm in ("log_mapd_50kb", "log10_reads", "log_segment_residual_mad", "barretts_cellularity_pct")}
    df = pd.DataFrame({"quiet": qmaj.astype(float), "prog": elab.astype(float), "pid": pidx, **cov})[fl]; df["cell_missing"] = df.barretts_cellularity_pct.isna().astype(float)
    df["barretts_cellularity_pct"] = df.barretts_cellularity_pct.fillna(df.barretts_cellularity_pct.median())
    for c_ in cov: df[c_] = (df[c_] - df[c_].mean()) / df[c_].std()
    lm = {}
    for nm, cols in (("class only", ["prog"]), ("class + noise + depth", ["prog", "log_mapd_50kb", "log10_reads", "log_segment_residual_mad"]), ("class + noise + depth + cellularity", ["prog", "log_mapd_50kb", "log10_reads", "log_segment_residual_mad", "barretts_cellularity_pct", "cell_missing"]),
                     ("QC only", ["log_mapd_50kb", "log10_reads", "log_segment_residual_mad", "barretts_cellularity_pct", "cell_missing"])):
        try:
            d_ = df.dropna(subset=cols); f_ = sm.Logit(d_.quiet, sm.add_constant(d_[cols])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d_.pid}); ci = f_.conf_int()
            lm[nm] = {c_: {"odds_ratio": r3(np.exp(f_.params[c_])), "ci95": [r3(np.exp(ci.loc[c_, 0])), r3(np.exp(ci.loc[c_, 1]))], "p": round(float(f_.pvalues[c_]), 4)} for c_ in cols}; lm[nm]["n_endoscopies"] = int(len(d_)); lm[nm]["n_quiet"] = m5(int(d_.quiet.sum()))
        except Exception as ex: lm[nm] = f"failed: {str(ex)[:150]}"
    R["logit_quiet_endoscopy_level"] = lm; R["logit_note"] = "endoscopies flagged by A12 in >= 5 of 10 repeats; quiet = quiet in >= half of those repeats; covariates = mean over the two top-H&E samples, standardised (OR per SD); cluster-robust SE by patient"
    from scipy.stats import spearmanr
    rows = S["rows"]; R["sample_spearman_ps_oac_vs"] = {cl: {nm: r3(spearmanr(q_oac[rows][g["y"][rows] == v], MET[nm][rows][g["y"][rows] == v], nan_policy="omit")[0]) for nm in MET} for cl, v in CLS}
    # noise-scaled quietness and QC-gated cancel
    cells = []
    for nm, f in (("ps_oac / segment-residual MAD", q_oac / F8.segmad.values), ("ps_oac / 50-kb MAPD", q_oac / F8.mapd.values), ("ps_oac / arm-level robust SD", q_oac / ARMSD),
                  ("OAC arms beyond 3 arm-level SDs (count)", sum((np.maximum(0, SIGN[a] * np.nan_to_num(AD[a])) > 3 * ARMSD).astype(float) for a in OAC)),
                  ("OAC arms beyond 2 arm-level SDs (count)", sum((np.maximum(0, SIGN[a] * np.nan_to_num(AD[a])) > 2 * ARMSD).astype(float) for a in OAC)),
                  ("sum of OAC-arm z (arm-level SD), z > 2 only", sum(np.where(np.maximum(0, SIGN[a] * np.nan_to_num(AD[a])) / ARMSD > 2, np.maximum(0, SIGN[a] * np.nan_to_num(AD[a])) / ARMSD, 0) for a in OAC))):
        s, _, _ = veto_cell(np.where(np.isfinite(f), f, np.nan), f"A12 + noise-scaled check: {nm} k2 cn90", params={"score": nm}); cells.append(s)
    # QC gates (cancel only if both sequenced samples pass); fold thresholds from the fold's training samples (no labels)
    V, vtr = V_oac, vtr_oac; mp = F8.mapd.values; rd = reads
    for nm in ("tile QC flag", "MAPD <= training p90", "reads >= training p10", "MAPD <= training p75", "cellularity >= training p10 (missing passes)", "all of tile QC, MAPD p90, reads p10"):
        can = np.zeros((E, 10), bool)
        for (j, kk), tx in TRX.items():
            tri = np.concatenate([S["es"][e] for e in tx["idx"]]); te = EF[:, j] == kk
            ok_tile = FT.ps_qc.values.astype(bool); ok_m90 = ~(mp > np.nanpercentile(mp[tri], 90)); ok_m75 = ~(mp > np.nanpercentile(mp[tri], 75)); ok_r = ~(rd < np.nanpercentile(rd[tri], 10))
            ok_c = ~(cell < np.nanpercentile(cell[tri], 10))
            ok = {"tile QC flag": ok_tile, "MAPD <= training p90": ok_m90, "reads >= training p10": ok_r, "MAPD <= training p75": ok_m75, "cellularity >= training p10 (missing passes)": ok_c,
                  "all of tile QC, MAPD p90, reads p10": ok_tile & ok_m90 & ok_r}[nm]
            can[te, j] = pair_feat(ok.astype(float), "all")[te, j]
        F, NS = apply(V, TH_oac, can); s = full(F, NS, f"A12 + QC-gated check ({nm})", {"gate": nm}, {"vetoes_blocked_by_gate_per_repeat": {cl: r2(float((FM & QUIET & ~can & (EL == v)).sum(0).mean())) for cl, v in CLS}})
        cells.append(s)
    R["cells"] = cells; res["qc"] = R; dump(res)
# ================================================================== 1b. noise-residualised quietness (fitted per fold on training non-progressor samples)
elif TASK == "noise":
    R = {}; cells = []; rows = S["rows"]
    for nm, z in (("log 50-kb MAPD", np.log(F8.mapd.values)), ("log segment-residual MAD", np.log(F8.segmad.values)), ("arm-level robust SD", ARMSD), ("log reads", -np.log(F8.total_reads.values))):
        for form in ("linear", "linear, only lowered"):
            ft = np.full((n, 10), np.nan); ftr = {}
            for (j, kk), tx in TRX.items():
                tri = np.concatenate([S["es"][e] for e in tx["idx"]]); npr = tri[g["y"][tri] == 0]; ok = np.isfinite(z[npr]) & np.isfinite(q_oac[npr]); b_ = np.polyfit(z[npr][ok], q_oac[npr][ok], 1)[0]
                adj = b_ * (z - np.nanmedian(z[npr])); adj = np.maximum(adj, 0) if form.endswith("lowered") else adj; x = q_oac - np.where(np.isfinite(adj), adj, 0); ftr[(j, kk)] = x; te = EF[:, j] == kk
                ft[np.concatenate([S["es"][e] for e in np.where(te)[0]]), j] = x[np.concatenate([S["es"][e] for e in np.where(te)[0]])]
            V = g["vs_test"](S, ft, 2); vtr = {kk: g["vs_train"](S, ftr[kk], tx["top"], 2) for kk, tx in TRX.items()}; TH = bar(vtr); F, NS = apply(V, TH)
            cells.append(full(F, NS, f"A12 + noise-residualised check: ps_oac minus slope x ({nm}) [{form}], slope from training NP samples, k2 cn90", {"noise": nm, "form": form}))
    R["cells"] = cells; R["note"] = "per outer fold: slope of ps_oac on the noise measure among the training patients' pre-event non-progressor samples; score = ps_oac - slope * (noise - training NP median) ('only lowered': noisier-than-median samples only)"
    res["noise"] = R; dump(res)
# ================================================================== 2. arm specificity (fixed sets, GC-rich, fragile sites)
elif TASK == "arms":
    R = {}; cells = []
    sets = [("17p loss only", ["17p"], None, False), ("17p + 9p losses", ["17p", "9p"], None, False), ("7 OAC losses only (no gains)", LOSS, None, False),
            ("5 losses 17p 9p 18q 5q 13q (no 3p, 4q, gains)", ["17p", "9p", "18q", "5q", "13q"], None, False), ("10 OAC arms minus 17p", [a for a in OAC if a != "17p"], None, False),
            ("10 OAC arms minus 20q", [a for a in OAC if a != "20q"], None, False), ("10 OAC arms, any direction", OAC, None, True),
            ("GC-rich non-OAC arms 19p 19q 22q 16p, any change", ["19p", "19q", "22q", "16p"], None, True), ("GC-rich arms 19p 19q 22q 16p, losses", ["19p", "19q", "22q", "16p"], {a: -1 for a in ("19p", "19q", "22q", "16p")}, False),
            ("GC-rich arms 19p 19q 22q 16p, gains", ["19p", "19q", "22q", "16p"], {a: 1 for a in ("19p", "19q", "22q", "16p")}, False)]
    for a in OAC: sets.append((f"single arm {a} ({'loss' if SIGN[a] < 0 else 'gain'})", [a], None, False))
    for nm, arms, sg, ab in sets: s, _, _ = veto_cell(armscore(arms, sg, ab), f"A12 + check on {nm} k2 cn90", params={"arms": arms, "absolute": ab}); cells.append(s)
    fh = F8.fhit_dev.values; ww = F8.wwox_dev.values
    for nm, f in (("FHIT focal loss (3p14.2)", np.maximum(0, -fh)), ("WWOX focal loss (16q23.1)", np.maximum(0, -ww)), ("FHIT + WWOX focal loss", np.maximum(0, -np.nan_to_num(fh)) + np.maximum(0, -np.nan_to_num(ww))),
                  ("FHIT + WWOX focal loss (segmented)", np.maximum(0, -np.nan_to_num(F8.fhit_segdev.values)) + np.maximum(0, -np.nan_to_num(F8.wwox_segdev.values)))):
        s, _, _ = veto_cell(f, f"A12 + check on {nm} k2 cn90", params={"score": nm}); cells.append(s)
    R["cells"] = cells
    # GC co-movement: does 17p 'loss' track GC-rich arms?
    from scipy.stats import spearmanr
    rows = S["rows"]; gc_ = np.nanmean(np.stack([AD[a] for a in ("19p", "19q", "22q", "16p", "17q")]), 0)
    R["spearman_17p_vs_mean_GC_rich_arms"] = {cl: r3(spearmanr(AD["17p"][rows][g["y"][rows] == v], gc_[rows][g["y"][rows] == v], nan_policy="omit")[0]) for cl, v in CLS}
    R["spearman_17p_vs_17q"] = {cl: r3(spearmanr(AD["17p"][rows][g["y"][rows] == v], AD["17q"][rows][g["y"][rows] == v], nan_policy="omit")[0]) for cl, v in CLS}
    R["share_samples_17p_loss_gt5pct"] = {cl: r3(float((AD["17p"][rows][g["y"][rows] == v] < -0.05).mean())) for cl, v in CLS}
    R["share_samples_fhit_focal_loss_gt5pct"] = {cl: r3(float((fh[rows][g["y"][rows] == v] < np.log2(0.95)).mean())) for cl, v in CLS}
    R["share_samples_wwox_focal_loss_gt5pct"] = {cl: r3(float((ww[rows][g["y"][rows] == v] < np.log2(0.95)).mean())) for cl, v in CLS}
    R["note"] = "arm deviations = per-sample-segmentation relative copy number of the arm / sample median tile - 1 (E2 tile_feats); fragile sites = median 50-kb log2 copy number in the gene body minus the median of a +-10 Mb flank (hg38)"
    res["arms"] = R; dump(res)
# ================================================================== 2b. random arm sets and random sample pairs (observed only, no bootstrap)
elif TASK == "rand":
    R = {}; real = obs(F_fin, NS_fin); a12 = obs(FM, NM); L0 = sum(ALEN[a] for a in OAC); rng = np.random.RandomState(8)
    out = {}
    for pool_nm, pool in (("non-OAC arms only", [a for a in ARMS if a not in OAC]), ("any arms (set != OAC)", list(ARMS))):
        rows_ = []; tries = 0
        while len(rows_) < 1000 and tries < 2_000_000:
            tries += 1; arms = list(rng.choice(pool, 10, replace=False))
            if set(arms) == set(OAC) or abs(sum(ALEN[a] for a in arms) / L0 - 1) > 0.15: continue
            sg = dict(zip(arms, rng.permutation([-1] * 7 + [1] * 3))); V, vtr = V_of(armscore(arms, sg)); F, NS = apply(V, bar(vtr)); o = obs(F, NS)
            rows_.append((o["np_flagged"] - a12["np_flagged"], o["caught"] - a12["caught"], o["seq"]))
        A = np.array(rows_); dnp, dc = A[:, 0], A[:, 1]; rdnp = real["np_flagged"] - a12["np_flagged"]; rdc = real["caught"] - a12["caught"]
        out[pool_nm] = {"n_sets": int(len(A)), "real_dNP_vs_A12": r2(rdnp), "real_dcaught_vs_A12": r2(rdc), "random_dNP_median": r2(np.median(dnp)), "random_dNP_p5_p95": [r2(np.percentile(dnp, 5)), r2(np.percentile(dnp, 95))],
                        "random_dcaught_median": r2(np.median(dc)), "random_dcaught_p5_p95": [r2(np.percentile(dc, 5)), r2(np.percentile(dc, 95))],
                        "share_random_clearing_at_least_as_many_NP": r3(float((dnp <= rdnp + 1e-9).mean())), "share_random_losing_no_more_progressors": r3(float((dc >= rdc - 1e-9).mean())),
                        "share_random_dominating_or_equal_real": r3(float(((dnp <= rdnp + 1e-9) & (dc >= rdc - 1e-9)).mean())), "share_random_clearing_ge3_NP_with_no_loss": r3(float(((dnp <= -3) & (dc >= -0.05)).mean())),
                        "percentile_of_real_NP_reduction_among_random": r3(float(100 * (dnp > rdnp + 1e-9).mean())), "random_net_NP_cleared_minus_progressors_lost_median": r2(np.median(-dnp + dc)), "real_net": r2(-rdnp + rdc),
                        "share_random_net_ge_real": r3(float(((-dnp + dc) >= (-rdnp + rdc) - 1e-9).mean())), "draws_dNP_dcaught": [[r2(x_), r2(y_)] for x_, y_ in zip(dnp, dc)]}
        np.save(f"{XD}/rand_{'nonoac' if pool_nm.startswith('non') else 'any'}.npy", A)
    R["random_arm_sets"] = out
    # random two samples per endoscopy (instead of the two top-H&E samples), OAC arms, same bar rule (training bar from the same random pairs is not defined for training -> the training bar uses the top-H&E pairs as in the real check)
    rr = []; es = S["es"]
    for b_ in range(200):
        rg = np.random.RandomState(1000 + b_); Vr = np.full((E, 10), np.nan)
        for e, s_ in enumerate(es):
            pk = rg.permutation(s_)[:2]; Vr[e, :] = q_oac[pk].max()
        F, NS = apply(Vr, TH_oac); o = obs(F, NS); rr.append((o["np_flagged"] - a12["np_flagged"], o["caught"] - a12["caught"]))
    rr = np.array(rr); R["random_two_samples_instead_of_top_HE"] = {"draws": 200, "dNP_vs_A12_median": r2(np.median(rr[:, 0])), "dNP_p5_p95": [r2(np.percentile(rr[:, 0], 5)), r2(np.percentile(rr[:, 0], 95))],
                                                                   "dcaught_vs_A12_median": r2(np.median(rr[:, 1])), "dcaught_p5_p95": [r2(np.percentile(rr[:, 1], 5)), r2(np.percentile(rr[:, 1], 95))],
                                                                   "real_top_HE": {"dNP": r2(real["np_flagged"] - a12["np_flagged"]), "dcaught": r2(real["caught"] - a12["caught"])}}
    R["note"] = ("random sets: 10 arms drawn without replacement from the pool (acrocentric p arms excluded), 7 assigned 'loss' and 3 'gain' at random, total arm length within +-15% of the OAC set; "
                 "same veto (two top-H&E samples, bar at 90% of the fold's training progressor endoscopies) on A12; observed held-out values, no bootstrap; RandomState(8)")
    res["rand"] = R; dump(res)
# ================================================================== 3. concordance of the two sequenced samples
elif TASK == "conc":
    R = {}; two = (T2 >= 0); base = FM & two
    for thr in (0.05, 0.03):
        o = {}
        for cl, v in CLS:
            mk = base & (EL == v); sh = 0.0; pr = 0.0; per = {}
            for a in OAC:
                c = np.maximum(0, SIGN[a] * np.nan_to_num(AD[a])) > thr; c1 = c[T1]; c2 = np.where(two, c[np.maximum(T2, 0)], False); s_ = (c1 & c2 & mk).sum(0).mean(); p_ = ((c1 ^ c2) & mk).sum(0).mean(); sh += s_; pr += p_
                per[a] = {"shared": r2(s_), "private": r2(p_)}
            o[cl] = {"altered_arm_instances_shared_per_repeat": r2(sh), "private_per_repeat": r2(pr), "share_shared": r3(sh / (sh + pr)) if sh + pr > 0 else None, "by_arm": per}
        R[f"arm_calls_threshold_{int(thr * 100)}pct"] = o
    q1 = q_oac[T1] < TH_oac; q2 = np.where(two, q_oac[np.maximum(T2, 0)] < TH_oac, True)
    R["quiet_call_agreement_A12_flagged_two_samples_per_repeat"] = {cl: {"both_quiet": r2(float((base & q1 & q2 & (EL == v)).sum(0).mean())), "one_quiet": r2(float((base & (q1 ^ q2) & (EL == v)).sum(0).mean())),
                                                                         "neither": r2(float((base & ~q1 & ~q2 & (EL == v)).sum(0).mean()))} for cl, v in CLS}
    cells = []
    # variants: k1 (top only), min of two (cancel if either quiet), mean of two, agreement-required arm score
    for k_ in (1,):
        s, _, _ = veto_cell(q_oac, "A12 + check on the top-H&E sample only (k1) cn90", k=1); cells.append(s)
    def pair_V(fn):
        V = np.full((E, 10), np.nan)
        for j in range(10): V[:, j] = fn(TOPO[:, j, 0], TOPO[:, j, 1])
        vtr = {kk: fn(tx["top"][:, 0], tx["top"][:, 1]) for kk, tx in TRX.items()}; return V, vtr
    CA = np.stack([np.maximum(0, SIGN[a] * np.nan_to_num(AD[a])) for a in OAC], 1)
    fns = {"min of the two samples (cancel if either quiet)": lambda a, b: np.where(b >= 0, np.fmin(q_oac[a], q_oac[np.maximum(b, 0)]), q_oac[a]),
           "mean of the two samples": lambda a, b: np.where(b >= 0, (q_oac[a] + q_oac[np.maximum(b, 0)]) / 2, q_oac[a]),
           "agreement-required arm score (arm counts only if changed in both)": lambda a, b: np.where(b >= 0, np.minimum(CA[a], CA[np.maximum(b, 0)]).sum(1), CA[a].sum(1)),
           "pooled-pair arm score (mean deviation of the two samples per arm)": lambda a, b: np.where(b >= 0, ((CA[a] + CA[np.maximum(b, 0)]) / 2).sum(1), CA[a].sum(1))}
    for nm, fn in fns.items():
        V, vtr = pair_V(fn); TH = bar(vtr); F, NS = apply(V, TH); s = full(F, NS, f"A12 + check: {nm} k2 cn90"); s["apparent"] = apparent_bar(V); cells.append(s)
    R["cells"] = cells; res["conc"] = R; dump(res)
# ================================================================== 4. flattened / genome-doubling proxy
elif TASK == "flat":
    R = {}
    MET = {"segments >= 1 Mb": F8.nseg_1mb.values.astype(float), "small-amplitude segments (0.03-0.10 log2)": F8.nseg_small.values.astype(float), "tiny-amplitude segments (0.02-0.06 log2)": F8.nseg_tiny.values.astype(float),
           "genome share in small-amplitude segments": F8.frac_small.values, "distinct segment levels (0.05 log2 steps)": F8.n_levels05.values.astype(float), "segment amplitude SD": F8.seg_amp_sd.values,
           "large segments (> 0.10 log2)": F8.nseg_large.values.astype(float), "fraction genome altered > 5% (tiles)": FT.ps_fga05.values.astype(float), "altered arms > 5%": FT.ps_narm05.values.astype(float),
           "all segments": F8.nseg.values.astype(float)}
    desc = {}
    for nm, x in MET.items():
        X = pair_feat(x, "max")
        desc[nm] = {"A12_flagged_quiet_progressor_vs_np": pooled_auc(X, EL, FM & QUIET), "all_quiet_progressor_vs_np": pooled_auc(X, EL, QUIET),
                    "median_quiet_progressor": gmed(X, QUIET & (EL == 1)), "median_quiet_np": gmed(X, QUIET & (EL == 0)), "median_not_quiet_progressor": gmed(X, ~QUIET & (EL == 1)), "median_not_quiet_np": gmed(X, ~QUIET & (EL == 0))}
    R["quiet_endoscopies_flattened_metrics"] = desc
    R["n_quiet_per_repeat"] = {"A12_flagged": {cl: r2(float((FM & QUIET & (EL == v)).sum(0).mean())) for cl, v in CLS}, "all": {cl: r2(float((QUIET & (EL == v)).sum(0).mean())) for cl, v in CLS}}
    cells = []
    for nm, x in MET.items():
        s, _, _ = veto_cell(np.where(np.isfinite(x), x, np.nan), f"A12 + check on global measure: {nm} k2 cn90", params={"score": nm}); cells.append(s)
    # double: cancel only if OAC-quiet AND not flattened (flattened-score below its own cn90 bar)
    for nm in ("small-amplitude segments (0.03-0.10 log2)", "segments >= 1 Mb", "distinct segment levels (0.05 log2 steps)", "segment amplitude SD"):
        for qf in (0.90, 0.75, 0.50):
            Vf, vtrf = V_of(np.where(np.isfinite(MET[nm]), MET[nm], np.nan)); THf = bar(vtrf, qf); can = Vf < THf
            F, NS = apply(V_oac, TH_oac, can); cells.append(full(F, NS, f"A12 + OAC-quiet AND not flattened ({nm} below {int(qf * 100)}% training-progressor bar)", {"flat_score": nm, "q_flat": qf},
                                    {"vetoes_blocked_per_repeat": {cl: r2(float((FM & QUIET & ~can & (EL == v)).sum(0).mean())) for cl, v in CLS}}))
    R["cells"] = cells; res["flat"] = R; dump(res)
# ================================================================== 5. sequential check
elif TASK == "seq":
    R = {}; v1 = q_oac[T1]; has2 = T2 >= 0
    need2 = has2 & (v1 < TH_oac)   # second sample sequenced only if the first is quiet
    kk = np.where(need2, 2, 1); NSs = NM + np.where(FM, np.where(GB == 1, kk - 1, kk), 0); Fs = F_fin
    R["decisions_identical_to_final"] = bool(np.array_equal(Fs, F_fin))
    R["sequential"] = full(Fs, NSs, "A12 + sequential quiet-genome check (second sample only if the first is quiet)")
    R["second_sample_needed_share_of_checked_endoscopies"] = {cl: r3(float((FM & need2 & (EL == v)).sum() / max((FM & has2 & (EL == v)).sum(), 1))) for cl, v in CLS}
    R["note"] = "decisions identical by construction (cancel iff both quiet); extra samples: 1 at every A12 flag (0 in R49's middle band, whose top sample is already sequenced) + 1 more only if the first is quiet"
    res["seq"] = R; dump(res)
# ================================================================== 7. timing
elif TASK == "time":
    R = {}; Q = QUIET; TT = np.repeat(TTE[:, None], 10, 1); W = np.ones(P_)
    def share(mask_num, mask_den, Wp):   # patient-weighted share pooled over repeats
        w = Wp[pidx][:, None]; return float((w * mask_num).sum() / max((w * mask_den).sum(), 1e-9))
    o = {}
    for pop_nm, base in (("all progressor endoscopies", (EL == 1) & np.isfinite(V_oac)), ("A12-flagged progressor endoscopies", FM & (EL == 1))):
        bands = {}
        for lo, hi, nm in ((0, 1, "<1 y before event"), (1, 3, "1-3 y"), (3, 99, ">=3 y")):
            m = base & (TT >= lo) & (TT < hi); ob = share(Q & m, m, W); bs = np.array([share(Q & m, m, D[b]) for b in range(len(D))])
            bands[nm] = {"endoscopies_per_repeat": r2(float(m.sum(0).mean())), "quiet_share": r3(ob), "ci95": ci2(bs)}
        tq = TT[base & Q]; tn = TT[base & ~Q]
        o[pop_nm] = {"bands": bands, "median_years_to_event_quiet": r2(np.median(tq)) if len(tq) else None, "median_years_to_event_not_quiet": r2(np.median(tn)), "quiet_endoscopies_per_repeat": r2(float((base & Q).sum(0).mean()))}
        if len(tq):
            d0 = np.median(tq) - np.median(tn); bs = []
            for b in range(len(D)):
                w = np.repeat(D[b][pidx][:, None], 10, 1); a_ = np.repeat(TT[base & Q], w[base & Q].astype(int)); c_ = np.repeat(TT[base & ~Q], w[base & ~Q].astype(int))
                if len(a_) and len(c_): bs.append(np.median(a_) - np.median(c_))
            o[pop_nm]["median_difference_quiet_minus_not"] = {"delta": r2(d0), "ci95": ci2(bs)}
    R["progressor_quiet_by_time_to_event"] = o
    from scipy.stats import spearmanr
    R["spearman_years_to_event_vs_top2_oac_progressor_endoscopies"] = r3(np.mean([spearmanr(TTE[elab == 1], V_oac[elab == 1, j], nan_policy="omit")[0] for j in range(10)]))
    # patient view: progressors quiet at first vs last pre-event endoscopy (patients with >= 2)
    pp = [i for i in range(P_) if isP[i] and len(ORD[i]) >= 2]; fq = np.mean([Q[ORD[i][0]].mean() for i in pp]); lq = np.mean([Q[ORD[i][-1]].mean() for i in pp])
    R["progressors_ge2_endoscopies"] = {"patients": len(pp), "quiet_share_first_endoscopy": r3(fq), "quiet_share_last_endoscopy": r3(lq)}
    # non-progressors: persistence of quietness between consecutive endoscopies
    qq = {"quiet_then_quiet": 0.0, "quiet_then_not": 0.0, "not_then_quiet": 0.0, "not_then_not": 0.0}
    for i in range(P_):
        if isP[i] or len(ORD[i]) < 2: continue
        for a, b in zip(ORD[i][:-1], ORD[i][1:]):
            qa, qb = Q[a], Q[b]; qq["quiet_then_quiet"] += (qa & qb).mean(); qq["quiet_then_not"] += (qa & ~qb).mean(); qq["not_then_quiet"] += (~qa & qb).mean(); qq["not_then_not"] += (~qa & ~qb).mean()
    R["non_progressor_consecutive_pairs"] = {k_: r2(v) for k_, v in qq.items()}; tq_ = qq["quiet_then_quiet"] + qq["quiet_then_not"]; tn_ = qq["not_then_quiet"] + qq["not_then_not"]
    R["non_progressor_P_quiet_next"] = {"given_quiet": r3(qq["quiet_then_quiet"] / tq_) if tq_ else None, "given_not_quiet": r3(qq["not_then_quiet"] / tn_) if tn_ else None}
    R["np_quiet_share_all_endoscopies"] = r3(float(Q[(EL == 0)].mean())); R["progressor_quiet_share_all_endoscopies"] = r3(float(Q[(EL == 1)].mean()))
    # single-look projection: the check applied only at each patient's FIRST A12-flagged endoscopy, no later rescue
    lost = np.zeros(10); cleared = np.zeros(10); one_endo = np.zeros(10)
    for j in range(10):
        for i in range(P_):
            fl = [e for e in ORD[i] if FM[e, j]]
            if not fl: continue
            if QUIET[fl[0], j]:
                if isP[i]: lost[j] += 1; one_endo[j] += len(ORD[i]) == 1
                else: cleared[j] += 1
    R["single_look_first_flag_only"] = {"progressors_whose_first_A12_flag_is_quiet": m5(round(float(lost.mean()), 1)), "of_which_with_one_pre_event_endoscopy": m5(round(float(one_endo.mean()), 1)),
                                        "non_progressors_whose_first_A12_flag_is_quiet": m5(round(float(cleared.mean()), 1)),
                                        "note": "ACE-B-like projection: a single look with no later endoscopy to rescue; mean over repeats"}
    R["note"] = "quiet = max OAC score of the two top-H&E samples below the fold's cn90 bar (defined for every endoscopy, as if sequenced); TTE = years before the reference date (event for progressors)"
    res["time"] = R; dump(res)
# ================================================================== 6. soft veto / in-fold bar (A12 nested engine rebuilt from rv9_falseflags.py)
elif TASK == "soft":
    os.environ["TASK"] = "__ex_e8_rv9__"; NS9 = {"__file__": HERE + "/rv9_falseflags.py", "__name__": "rv9_e8"}
    _s9 = open(HERE + "/rv9_falseflags.py").read(); exec(compile(_s9[:_s9.index('\nres = {"task": TASK}')], HERE + "/rv9_falseflags.py", "exec"), NS9); os.environ["TASK"] = TASK
    S9, ST, INFO = NS9["prep_state"](); assert len(S9["es"]) == E and all(np.array_equal(a, b) for a, b in zip(S9["es"], S["es"])) and np.array_equal(S9["pidx"], pidx)
    mk_ctx = NS9["make_ctx"]; fA12 = NS9["f_A12"]; pat_scores = NS9["pat_scores"]; thr_for = NS9["thr_for"]; cert_flags = NS9["cert_flags"]
    DL = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]; QS = [0.80, 0.85, 0.90, 0.95]
    U = np.full((E, 10), np.nan); TT_ = np.full((E, 10), np.nan); chosen_ok = 0; flag_ok = 0; TH_chk = 0
    VAR = {}; PICKS = {}
    def put(nm, j, te, F_, ns_, pk=None):
        if nm not in VAR: VAR[nm] = (np.zeros((E, 10), bool), np.zeros((E, 10), int))
        VAR[nm][0][te, j] = F_[te]; VAR[nm][1][te, j] = ns_[te]
        if pk is not None: PICKS.setdefault(nm, {}); PICKS[nm][str(pk)] = PICKS[nm].get(str(pk), 0) + 1
    chz = Z["A12__chosen_np"]
    for j in range(10):
        for k in range(1, 11):
            c = mk_ctx(S9, ST, j, k, INFO); te = c.te_e; cands = fA12(c); st = []
            for x in cands:
                pm = pat_scores(c, x["u"]); t = thr_for(c, pm); fl = x["u"] >= t; inc = (c.S["INC"] @ fl.astype(float)) > 0
                st.append((c.patient_det(fl), int((inc & ~c.isP & c.trP).sum()), int((fl & c.tr_e & (c.elab == 0)).sum()), float(x["ns"][c.tr_e].mean()), t))
            feas = [q for q in range(len(cands)) if st[q][0] >= c.D]
            pick = sorted(feas, key=lambda q: (st[q][1], st[q][2], st[q][3], q))[0] if feas else sorted(range(len(cands)), key=lambda q: (-st[q][0], st[q][1], st[q][2], q))[0]
            chosen_ok += int(pick == chz[j * 10 + k - 1]); u = cands[pick]["u"]; t = st[pick][4]; flag = u >= t; ns0 = cands[pick]["ns"]
            flag_ok += int(np.array_equal(flag[te], FM[te, j])); U[te, j] = u[te]; TT_[te, j] = t
            # top-2 H&E samples of every endoscopy in this fold (inner L-IMG for training, outer for held-out), quiet score
            x_ = np.full(n, np.nan); s_ = S9["I"]["L-IMG"][(REPS[j], k)]; x_[s_.index.values] = s_.values; tes = S9["FO"][:, j] == k; x_[tes] = S9["O"]["L-IMG"][tes, j]
            t1 = np.zeros(E, int); t2 = np.full(E, -1)
            for e, sm_ in enumerate(S9["es"]):
                od = sm_[np.argsort(-x_[sm_], kind="mergesort")]; t1[e] = od[0]; t2[e] = od[1] if len(od) > 1 else -1
            v = np.fmax(q_oac[t1], np.where(t2 >= 0, q_oac[np.maximum(t2, 0)], -np.inf)); v = np.where(np.isfinite(v) | (v == -np.inf), v, np.inf)
            assert np.array_equal(t1[te], TOPO[te, j, 0])
            trm = c.tr_e; Pm = c.elab == 1
            bars = {q: thr_sens(v[trm & Pm & np.isfinite(v)], q) for q in QS}; TH_chk += int(np.allclose(bars[0.90], TH_oac[te, j][0])) if te.any() else 1
            det = c.patient_det; npf = lambda f_: int((((c.S["INC"] @ f_.astype(float)) > 0) & ~c.isP & c.trP).sum()); d0 = det(flag); kk_ = np.minimum(2, c.nsz)
            seqall = ns0 + np.where(flag, np.where(c.r49_band == 1, kk_ - 1, kk_), 0)
            def seq_band(dl):   # sequence only where a quiet result can change the decision (u < t + dl)
                need = flag & (u < t + dl); return ns0 + np.where(need, np.where(c.r49_band == 1, kk_ - 1, kk_), 0)
            q1_ = q_oac[t1] < bars[0.90]; n_chk = np.where(c.r49_band == 1, 0, 1) + ((t2 >= 0) & q1_).astype(int)   # sequential: top sample, second only if the top is quiet
            def seq_band_seq(dl): return ns0 + np.where(flag & (u < t + dl), n_chk, 0)
            hv = flag & ~(v < bars[0.90]); hnp = npf(hv); hdet = det(hv); dsel = np.inf
            for dl in DL:
                fq = flag & ~((v < bars[0.90]) & (u < t + dl))
                if npf(fq) <= hnp and det(fq) >= hdet: dsel = dl; break
            fq = flag & ~((v < bars[0.90]) & (u < t + dsel)); put("band check: cheapest band matching the hard veto on training, sequence in band", j, te, fq, seq_band(dsel), dsel)
            put("band check: cheapest band matching the hard veto on training, sequence in band, sequential", j, te, fq, seq_band_seq(dsel))
            put("hard veto cn90, sequential (= final decisions)", j, te, hv, seq_band_seq(np.inf))
            for dl in (2.0, 3.0, 5.0):
                fq = flag & ~((v < bars[0.90]) & (u < t + dl)); put(f"band check, fixed delta {dl} (post hoc grid point), sequence in band, sequential", j, te, fq, seq_band_seq(dl))
            # (a) hard veto, cn90 bar (= final rule)
            qt = {q: v < b for q, b in bars.items()}; put("hard veto, cn90 bar (= final rule)", j, te, flag & ~qt[0.90], seqall)
            # (b) hard veto, bar re-chosen in-fold on A12's training flags (strict / loss1 / target)
            cand_b = np.unique(v[trm & flag & np.isfinite(v)])[::-1]
            for rule in ("strict", "loss1", "target"):
                need = d0 if rule == "strict" else (d0 - 1 if rule == "loss1" else c.D); bb = -np.inf
                for b in cand_b:
                    if det(flag & ~(v < b)) >= need: bb = b; break
                put(f"hard veto, bar re-chosen in-fold ({rule})", j, te, flag & ~(v < bb), seqall)
            # (c) soft veto: a quiet result lowers the running score by delta (equivalently: cancel only if u < t + delta)
            for crit in ("noloss", "target"):
                need = d0 if crit == "noloss" else c.D
                for qmode in ("cn90", "q in-fold"):
                    best = None
                    for q in ([0.90] if qmode == "cn90" else QS):
                        for dl in DL:
                            fq = flag & ~(qt[q] & (u < t + dl)); dd = det(fq)
                            if dd < need: continue
                            key = (npf(fq), -q, dl)   # fewest training NP flagged, then the safer bar, then the smaller step
                            if best is None or key < best[0]: best = (key, q, dl)
                    if best is None: q, dl = 0.90, 0.0
                    else: _, q, dl = best
                    fq = flag & ~(qt[q] & (u < t + dl)); nm = f"soft veto ({crit}, bar {qmode})"
                    put(nm + ", sequence every flag", j, te, fq, seqall, (q, dl)); put(nm + ", sequence only where it can change the call", j, te, fq, seq_band(dl))
            # (d) soft veto carried forward: a quiet result lowers the running score at this and later endoscopies within 2 years
            qual = flag & qt[0.90]; cr = qual | cert_flags(ST, qual, 2.0); best = None
            for dl in DL[:-1]:
                fq = (u - dl * cr) >= t; dd = det(fq)
                if dd < d0: continue
                key = (npf(fq), dl)
                if best is None or key < best[0]: best = (key, dl)
            dl = best[1] if best else 0.0; put("soft veto carried 2 y (noloss, cn90)", j, te, (u - dl * cr) >= t, seqall, dl)
    res["engine_check"] = {"chosen_candidate_equal_folds": chosen_ok, "held_out_flags_equal_folds": flag_ok, "cn90_bar_equal_folds": TH_chk, "of": 100}
    cells = []
    for nm, (F_, NS_) in VAR.items():
        ex = {"picks_(q, delta)_over_100_folds": dict(sorted(PICKS.get(nm, {}).items(), key=lambda x: -x[1])[:8])} if nm in PICKS else None; cells.append(full(F_, NS_, f"A12 + {nm}", {}, ex))
    # OPTIMISTIC: delta chosen per repeat on the evaluation patients (largest NP reduction keeping A12's caught)
    app = {}
    for dl in DL:
        Fd = FM & ~(QUIET & (U < TT_ + dl)); o = obs(Fd, NS_fin); app[str(dl)] = {k_: r2(v) for k_, v in o.items()}
    res["apparent_soft_by_delta_OPTIMISTIC"] = app
    res["cells"] = cells; res["note"] = ("A12 nested engine re-run per outer fold (rv9 f_A12 candidates, 'np' selection) to obtain running scores u and thresholds t for training and held-out endoscopies; "
                                         "soft veto: flagged endoscopy (u >= t) with a quiet top-2 check is cancelled only if u - delta < t; delta (and bar quantile when 'q in-fold') chosen on the fold's training patients: "
                                         "noloss = no training progressor patient lost relative to A12's training detection, target = detection >= R49's training detection; then fewest training NP patients flagged; "
                                         "training outcomes are resubstitution for the bar (copy-number feature fixed, H&E scores inner out-of-fold)")
    dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
