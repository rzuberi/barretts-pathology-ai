# Exploration E1 analysis tasks; executed inside ex_e1_main.py's namespace (rv8_landscape / rv8b_step3 helpers available). See ex_e1_main.py docstring.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
if os.environ.get("EXE1_NB"): TC.boots_for.__defaults__ = (0, NB)
S = setup(); D = pdraws(S); PR = pickle.load(open(X1 + "/prep.pkl", "rb")); C = PR["C"]; TS = PR["TS"]; TSN = PR["TSN"]; TQ = {q: i for i, q in enumerate(TSN)}
es = S["es"]; E = S["E"]; pidx = S["pidx"]; plab = S["plab"]; elab = S["elab"]; Pe = elab == 1; Ne = ~Pe; FO = S["FO"]; O = S["O"]; I = S["I"]
one = np.ones(S["P_"]); nP = int((plab == 1).sum()); nN = int((plab == 0).sum()); Z = np.zeros((E, 10), int); WD = D[:, pidx]
_, _, INFO = TC.triage(O, I, FO, S["popm"], 0.95, 0.90); FR49, NSR49, GB49 = r49_calls(O, INFO, FO, es)
TH1, _ = thr_arr(S, "img", 0.90); FP1 = S["OUTS"]["img"] >= TH1
RULES = {"R49": (FR49, NSR49), "P1 90%": (FP1, Z)}
TOPj = np.array([[s[np.argmax(O["L-IMG"][s, j])] for j in range(10)] for s in es]); MIMG = O["L-IMG"].mean(1); TOPm = np.array([s[np.argmax(MIMG[s])] for s in es])
EF = S["EF"]
def ts_out(q, top=TOPj):
    """(E, 10): tile summary q of the endoscopy's top-H&E sample under its outer model, per repeat"""
    return np.stack([TS[top[:, j], j, EF[:, j] - 1, TQ[q]] for j in range(10)], 1).astype(float)
# ------------------------------------------------------------------ endoscopy covariates (E1.1)
def endo_cov():
    V = {}; agg = lambda a, f: np.array([f(a[s]) for s in es])
    for k in ("tissue_tiles", "log_tissue_tiles", "bg_share", "edge_share", "ink_share", "blur_share", "lapvar_median", "sat_mean", "dark_share", "slide_age_years", "scanner_C13210",
              "batch_single_class", "batch_progressor_share", "note_z_levels", "note_marker", "same_block", "n_slides"):
        V[f"{k} (top-H&E sample)"] = C[k][TOPm].astype(float)
    V["tissue_tiles (endoscopy mean)"] = agg(C["tissue_tiles"], np.mean); V["bg_share (endoscopy max)"] = agg(C["bg_share"], np.max)
    V["biopsy_year"] = agg(C["biopsy_year"], np.min); V["samples_per_endoscopy"] = S["nsz"].astype(float); V["position_in_series (0 = first)"] = S["POS"].astype(float)
    NEp = np.bincount(pidx); V["position_from_last (0 = last)"] = (NEp[pidx] - 1 - S["POS"]).astype(float); V["first_endoscopy"] = S["FIRST"].astype(float)
    V["years_to_reference (event or last endoscopy)"] = S["TTE"].astype(float); V["endoscopies_of_patient"] = NEp[pidx].astype(float)
    g = agg(C["grade_sheet"], np.max); V["grade_sheet_max (0 NDBE, 1 ID, 2 LGD)"] = g; V["grade_sheet_ID_or_LGD"] = (g >= 1).astype(float); V["grade_sheet_LGD"] = (g >= 2).astype(float)
    V["p53_aberrant (any, of tested)"] = agg(C["p53_aberrant"], lambda a: np.nanmax(a) if np.isfinite(a).any() else np.nan); V["p53_tested"] = agg(C["p53_available"], np.max)
    dbg = agg(C["db_grade_at_endoscopy"], lambda a: np.nanmax(a) if np.isfinite(a).any() else np.nan); V["db_grade_at_endoscopy (0 NDBE .. 3 HGD)"] = dbg; V["db_grade_ID_or_worse"] = np.where(np.isfinite(dbg), (dbg >= 1).astype(float), np.nan)
    V["db_report_found"] = np.isfinite(dbg).astype(float); V["age"] = agg(C["age"], np.mean); V["sex_M"] = agg(C["sex_M"], np.max); V["prague_M"] = agg(C["prague_M"], np.max)
    for gname in ("no later report", "NDBE/ID", "LGD", "HGD+", "not linked"): V[f"later_db_group = {gname}"] = np.array([float(PR["LGRP"][s[0]] == gname) for s in es])
    V["H&E score: max logit L-IMG (repeat mean)"] = np.array([lg(O["L-IMG"][s]).max(0).mean() for s in es]); V["copy number: max logit L-CNV (repeat mean)"] = np.array([lg(O["L-CNV"][s]).max(0).mean() for s in es])
    return V
def wauc(xp, wp, xn, wn):
    """weighted P(x_p > x_n) + 0.5 ties"""
    ok_p = np.isfinite(xp) & (wp > 0); ok_n = np.isfinite(xn) & (wn > 0); xp, wp, xn, wn = xp[ok_p], wp[ok_p], xn[ok_n], wn[ok_n]
    if not len(xp) or not len(xn): return None
    W = wp[:, None] * wn[None, :]; return r3((((xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])) * W).sum() / W.sum())
def gcompare(x, G, pairs, Wd=WD):
    """x (E,) or (E, 10); G: group -> weights (E,) or (E, 10). Group means (patient-bootstrap CIs) and paired differences; for (E, 10) the per-repeat means are averaged"""
    X = x if x.ndim == 2 else np.repeat(x[:, None], 10, 1); v = np.isfinite(X); XX = np.where(v, X, 0.0); o = {"groups": {}, "contrasts": {}}; ob = {}; bs = {}
    for gname, w in G.items():
        W_ = (w if w.ndim == 2 else np.repeat(w[:, None], 10, 1)) * v
        with np.errstate(invalid="ignore", divide="ignore"):
            ob[gname] = np.nanmean((W_ * XX).sum(0) / W_.sum(0)); bs[gname] = np.nanmean((Wd @ (W_ * XX)) / (Wd @ W_), 1)
            sd_ = np.sqrt(np.nanmean(((W_ * (XX - ob[gname]) ** 2).sum(0)) / W_.sum(0)))
        o["groups"][gname] = {"mean": r3(ob[gname]), "ci95": ci2(bs[gname]), "n_endoscopies_weighted": r2(float(W_.sum(0).mean())), "sd": r3(sd_)}
    for a, b in pairs:
        sa, sb_ = o["groups"][a]["sd"], o["groups"][b]["sd"]; sp = np.sqrt(((sa or 0) ** 2 + (sb_ or 0) ** 2) / 2)
        o["contrasts"][f"{a} - {b}"] = {**dstat(ob[a] - ob[b], bs[a] - bs[b]), "standardised": r3((ob[a] - ob[b]) / sp) if sp and np.isfinite(sp) and sp > 0 else None}
    return o
def classes(F):
    f = F.astype(float); return {"FP-NP": f * Ne[:, None], "TN-NP": (1 - f) * Ne[:, None], "TP-P": f * Pe[:, None], "FN-P": (1 - f) * Pe[:, None]}
PAIRS = (("FP-NP", "TN-NP"), ("FP-NP", "TP-P"), ("TP-P", "FN-P"))
# ------------------------------------------------------------------ evaluation of a rule against R49 and P1 (rv8b machinery)
XIMG = S["OUTS"]["img"]; AR0 = {k: patient_arrays(S, F, NS) for k, (F, NS) in RULES.items()}; P0 = {k: (parts(S, A, one), parts(S, A, D)) for k, A in AR0.items()}
MET = {"progressor_patients_caught": lambda p: nP * met(p, "caught_share"), "non_progressor_patients_flagged": lambda p: nN * met(p, "np_flagged_share"),
       "endoscopy_sensitivity": lambda p: met(p, "endo_sens"), "endoscopy_specificity": lambda p: met(p, "endo_spec"), "samples_sequenced_per_patient": lambda p: met(p, "seq_per_patient")}
def p1_matched_parts(sens, m):
    w = m[pidx].astype(int); F = np.zeros((E, 10), bool)
    for j in range(10): F[:, j] = XIMG[:, j] >= tmatch(XIMG[Pe, j], sens[j], w[Pe])
    return parts(S, patient_arrays(S, F, Z), m)
FSAVE = {}
def evaluate(F, NS, label, matched=True, extra=None):
    FSAVE[label] = (np.asarray(F, bool), np.asarray(NS)); A = patient_arrays(S, F, NS); po = parts(S, A, one); pb = parts(S, A, D); o = {"label": label}
    o["value"] = {k: r3(f(po)[0]) for k, f in MET.items()}; o["ci95"] = {k: ci2(f(pb)) for k, f in MET.items()}
    for ref in ("R49", "P1 90%"):
        r0, rb = P0[ref]; o[f"delta_vs_{ref}"] = {k: dstat(f(po)[0] - f(r0)[0], f(pb) - f(rb)) for k, f in MET.items()}
    if matched:   # P1 re-matched (per repeat, per draw) to this rule's realised endoscopy sensitivity
        so = per_rep(po, "endo_sens")[0]; pm = p1_matched_parts(so, one); ob_ = {k: f(pm)[0] for k, f in MET.items()}
        sb = per_rep(pb, "endo_sens"); dd = {k: [] for k in MET}
        for i_, m in enumerate(D):
            pm_ = p1_matched_parts(sb[i_], m)
            for k, f in MET.items(): dd[k].append(f(parts(S, A, m))[0] - f(pm_)[0])
        o["delta_vs_P1_matched_endoscopy_sensitivity"] = {k: dstat(MET[k](po)[0] - ob_[k], dd[k]) for k in ("endoscopy_specificity", "non_progressor_patients_flagged", "progressor_patients_caught", "endoscopy_sensitivity")}
        o["P1_matched_value"] = {k: r3(v) for k, v in ob_.items()}
    if extra: o.update(extra)
    return o
# ------------------------------------------------------------------ training-side (inner out-of-fold) flags and features per (repeat, fold)
def train_side(j, k, img_inner=None):
    """training endoscopies of split (repeat j, outer fold k): indices, labels, R49 and P1 90% flags from inner out-of-fold scores and the fold's cut-offs, inner top sample"""
    r = j + 1; tr = S["TR"][(j, k)]; tidx = tr["idx"]; si = I["L-IMG"][(r, k)] if img_inner is None else img_inner[(r, k)]; v = np.full(n, np.nan); v[si.index.values] = si.values
    vl = np.full(n, np.nan); sl = I["L-LATE"][(r, k)]; vl[sl.index.values] = sl.values; i_ = j * 10 + k - 1; tl, cs, ts = INFO["t_low"][i_], INFO["c_star"][i_], INFO["t_seq"][i_]
    top = np.array([es[e][np.argmax(v[es[e]])] for e in tidx]); sc = np.array([v[es[e]].max() for e in tidx])
    g = np.where(sc > cs, 2, 0) if tl > cs else np.where(sc < tl, 0, np.where(sc > cs, 2, 1)); f49 = np.where(g == 1, vl[top] >= ts, g == 2)
    lab = elab[tidx]; t1 = thr_sens(sc[lab == 1], 0.90); f1 = sc >= t1
    return {"idx": tidx, "lab": lab, "top": top, "F": {"R49": f49, "P1 90%": f1}, "score": sc}
TRS = {(j, k): train_side(j, k) for j in range(10) for k in range(1, 11)}
SAMPLE_FEATS = ["log_tissue_tiles", "bg_share", "edge_share", "ink_share", "blur_share", "lapvar_median", "sat_mean", "dark_share"]
TILE_FEATS = ["mean", "median", "trim10", "nonbg_mean", "core_mean", "sd", "p90", "max", "top16_mean", "rest_mean", "share_pos", "gini_pos", "frac_u_gt2", "bg_contrib", "edge_contrib", "ink_contrib", "blur_contrib", "conc_top16_of_pos"]
ARTEFACT_FEATS = ["scanner_C13210", "slide_age_years", "batch_single_class", "batch_progressor_share", "biopsy_year"]
def endo_level_clin(e_idx):
    g = np.array([C["grade_sheet"][es[e]].max() for e in e_idx]); p53 = np.array([np.nanmax(C["p53_aberrant"][es[e]]) if np.isfinite(C["p53_aberrant"][es[e]]).any() else 0.0 for e in e_idx])
    pa = np.array([C["p53_available"][es[e]].max() for e in e_idx]); age = np.array([C["age"][es[e]].mean() for e in e_idx]); sex = np.array([C["sex_M"][es[e][0]] for e in e_idx])
    pm = np.array([C["prague_M"][es[e][0]] for e in e_idx]); return np.column_stack([g, p53, pa, age, sex, pm, S["nsz"][e_idx], S["POS"][e_idx]])
CLIN_NAMES = ["grade_max", "p53_aberrant", "p53_tested", "age", "sex_M", "prague_M", "samples_per_endoscopy", "position_in_series"]
def feats(e_idx, top, j, kmodel, names):
    """feature matrix for endoscopies e_idx with top samples `top`; tile summaries under model (repeat j, fold kmodel[i])"""
    cols = []
    for nm in names:
        if nm in TQ: cols.append(TS[top, j, kmodel - 1, TQ[nm]].astype(float))
        elif nm in C: cols.append(C[nm][top].astype(float))
        else: raise KeyError(nm)
    return np.column_stack(cols) if cols else np.zeros((len(e_idx), 0))
def clin_block(e_idx): return endo_level_clin(e_idx)
# ================================================================== E1.1 characterise false flags
if TASK == "char":
    V = endo_cov(); R = {"rules": {}}
    for rn, (F, NS) in RULES.items():
        G = classes(F); out = {}
        cnt = {g: r2(float(w.sum(0).mean())) for g, w in G.items()}
        for nm, x in V.items():
            o = gcompare(x, G, PAIRS); fp = G["FP-NP"].mean(1); tp = G["TP-P"].mean(1); tn = G["TN-NP"].mean(1)
            o["auroc_TP-P_vs_FP-NP"] = wauc(x[Pe], tp[Pe], x[Ne], fp[Ne]); o["auroc_FP-NP_vs_TN-NP"] = wauc(x[Ne], fp[Ne], x[Ne], tn[Ne]); o["n_missing"] = int((~np.isfinite(x)).sum())
            out[nm] = o
        # patient level: non-progressor patients ever flagged vs never, and caught progressors
        A = AR0[rn]; FL = A["FL"].mean(1); NEp = np.bincount(pidx)
        PV = {"endoscopies": NEp.astype(float), "years_followed (first endoscopy to reference)": S["PY"], "samples": np.array([sum(len(es[e]) for e in np.where(pidx == i)[0]) for i in range(S["P_"])], float)}
        for k_ in ("age", "sex_M", "prague_M", "tissue_tiles", "slide_age_years", "scanner_C13210", "batch_single_class", "bg_share", "grade_sheet"):
            PV[k_ + (" (max)" if k_ == "grade_sheet" else " (mean)")] = np.array([np.nanmax(C[k_][np.concatenate([es[e] for e in np.where(pidx == i)[0]])]) if k_ == "grade_sheet" else np.nanmean(C[k_][np.concatenate([es[e] for e in np.where(pidx == i)[0]])]) for i in range(S["P_"])])
        for gname in ("no later report", "NDBE/ID", "LGD", "HGD+", "not linked"): PV[f"later_db_group = {gname}"] = np.array([float(PR["LGRP"][es[np.where(pidx == i)[0][0]][0]] == gname) for i in range(S["P_"])])
        isN = plab == 0; isP = ~isN; GP = {"NP flagged": FL * isN, "NP never flagged": (1 - FL) * isN, "P caught": FL * isP, "P missed": (1 - FL) * isP}
        pout = {nm: gcompare(x, GP, (("NP flagged", "NP never flagged"), ("NP flagged", "P caught")), Wd=D) for nm, x in PV.items()}
        NFm = A["NF"]; FLb = A["FL"] > 0; rep_ = {}
        for cl, msk in (("non_progressor", isN), ("progressor", isP)):
            fl_ = FLb[msk]; nf_ = NFm[msk]; ne_ = NEp[msk][:, None] + 0 * nf_
            rep_[cl] = {"flagged_endoscopies_per_ever_flagged_patient": r2(float(nf_[fl_].mean())), "share_of_ever_flagged_with_exactly_one_flag": r3(float((nf_[fl_] == 1).mean())),
                        "share_of_endoscopies_flagged_among_ever_flagged": r3(float((nf_[fl_] / ne_[fl_]).mean())), "ever_flagged_with_one_endoscopy_only": r2(float((fl_ & (ne_ == 1)).sum(0).mean()))}
        R["rules"].setdefault(rn, {})
        R["rules"][rn] = {"flag_repetition": rep_, "endoscopy_class_counts_mean_over_repeats": cnt, "endoscopy": out, "patient": pout,
                          "patient_class_counts": {g: r2(float(w.sum())) for g, w in GP.items()}}
    # among NP endoscopies flagged by P1 90%, which does R49 clear? (the sequencing step's own error pattern)
    both = (FP1 & FR49).astype(float) * Ne[:, None]; p1only = (FP1 & ~FR49).astype(float) * Ne[:, None]; r49only = (~FP1 & FR49).astype(float) * Ne[:, None]
    R["np_flags_P1_vs_R49"] = {"both": r2(float(both.sum(0).mean())), "P1 only (cleared by sequencing)": r2(float(p1only.sum(0).mean())), "R49 only": r2(float(r49only.sum(0).mean())),
                               "covariates": {nm: gcompare(x, {"both": both, "P1 only": p1only}, (("both", "P1 only"),)) for nm, x in V.items() if not nm.startswith("later_db")}}
    # persistence: share of variance of the endoscopy H&E score (repeat-mean max logit) between patients (one-way ANOVA ICC), and concentration of false flags in patients
    sc_ = np.array([lg(O["L-IMG"][s_]).max(0).mean() for s_ in es])
    def icc(mask, m=None):
        w = np.ones(S["P_"]) if m is None else m; num_b = num_w = 0.0; nt = 0.0; grand = []; groups = []
        for i in np.where(np.bincount(pidx[mask], minlength=S["P_"]) >= 2)[0]:
            if w[i] == 0: continue
            v_ = sc_[mask & (pidx == i)]; groups.append((v_, w[i]))
        if len(groups) < 3: return np.nan
        allv = np.concatenate([np.repeat(v_[None, :], int(ww), 0).ravel() for v_, ww in groups]); gm = allv.mean(); k_ = sum(int(ww) for _, ww in groups); N_ = len(allv)
        ssb = sum(int(ww) * len(v_) * (v_.mean() - gm) ** 2 for v_, ww in groups); ssw = sum(int(ww) * ((v_ - v_.mean()) ** 2).sum() for v_, ww in groups)
        n0 = (N_ - sum(int(ww) * len(v_) ** 2 for v_, ww in groups) / N_) / (k_ - 1); msb = ssb / (k_ - 1); msw = ssw / (N_ - k_); return float((msb - msw) / (msb + (n0 - 1) * msw))
    R["persistence"] = {}
    for cl, mk in (("non_progressor", Ne), ("progressor", Pe)):
        ob_ = icc(mk); bd_ = [icc(mk, m) for m in D[:200]]
        R["persistence"][f"icc_endoscopy_hne_score_{cl}"] = {"value": r3(ob_), "ci95_200_draws": ci2(bd_)}
    for rn, (F, NS) in RULES.items():
        fpp = np.array([F[(pidx == i) & Ne].sum() for i in range(S["P_"])]) / 10.0; fpp = np.sort(fpp[plab == 0])[::-1]; tot = fpp.sum()
        R["persistence"][f"{rn}_false_flag_concentration"] = {"np_patients_with_any_false_flag_mean": r2(float((AR0[rn]["FL"][plab == 0]).sum(0).mean())), "false_flagged_np_endoscopies_mean": r2(float(tot)),
                                                             "share_of_false_flags_from_top_10_np_patients": r3(float(fpp[:10].sum() / tot)), "share_from_top_20": r3(float(fpp[:20].sum() / tot))}
    R["check_round8"] = {k: {"caught": r2(nP * met(P0[k][0], "caught_share")[0]), "np_flagged": r2(nN * met(P0[k][0], "np_flagged_share")[0])} for k in RULES}
    R["note"] = ("endoscopy classes per repeat (FP-NP = non-progressor endoscopy flagged; TN-NP not flagged; TP-P progressor flagged; FN-P not flagged), covariate means weighted by the share of repeats in "
                 "each class (averaged over repeats), paired patient bootstrap (rv8 pdraws, RandomState(0)); sample covariates taken from the endoscopy's top-H&E sample (highest repeat-mean outer L-IMG); "
                 "standardised = difference / sqrt(mean of the two group variances); AUROC = weighted P(x in first group > x in second); patient level: non-progressor patients weighted by the share of repeats flagged")
    res["E1_1"] = R; dump(res)
# ================================================================== E1.2 inside the H&E score
elif TASK == "tiles":
    R = {"rules": {}}
    TSX = {q: ts_out(q) for q in TSN}
    # check: b0 + mean u of the top sample = its outer logit
    lgt = np.stack([lg(O["L-IMG"][TOPj[:, j], j]) for j in range(10)], 1); ok = (O["L-IMG"][TOPj, np.arange(10)[None, :]] > 1e-6) & (O["L-IMG"][TOPj, np.arange(10)[None, :]] < 1 - 1e-6)
    R["check_logit_decomposition_max_abs_diff"] = r3(float(np.abs((TSX["b0"] + TSX["mean"] - lgt)[ok]).max()) * 1000)
    der = {"mean_without_top16": TSX["rest_mean"],
           "score_change_if_bg_removed (nonbg_mean - mean)": TSX["nonbg_mean"] - TSX["mean"], "score_change_if_edge_and_bg_removed (core_mean - mean)": TSX["core_mean"] - TSX["mean"],
           "score_change_trimmed10 (trim10 - mean)": TSX["trim10"] - TSX["mean"], "score_change_median (median - mean)": TSX["median"] - TSX["mean"], "score_change_drop_top16 (rest_mean - mean)": TSX["rest_mean"] - TSX["mean"]}
    ALL = {**{q: TSX[q] for q in TSN if q != "b0"}, **der}
    for rn, (F, NS) in RULES.items():
        G = {g: w for g, w in classes(F).items()}; out = {}
        for nm, x in ALL.items():
            o = gcompare(x, G, PAIRS); fpw = G["FP-NP"]; tpw = G["TP-P"]
            o["auroc_TP-P_vs_FP-NP_per_repeat_mean"] = r3(np.nanmean([wauc(x[Pe, j], tpw[Pe, j], x[Ne, j], fpw[Ne, j]) or np.nan for j in range(10)])); out[nm] = o
        R["rules"][rn] = out
    R["note"] = ("tile score u_t = w . z(e_t) (z-scored with the fit's training mean / sd), so the sample logit = b0 + mean_t u_t exactly; contributions = u_t / N (logit units); "
                 "summaries of the endoscopy's top-H&E sample in each repeat under its outer model; bg = tile with > 50% white pixels (R55); edge = non-background tile with tissue-mask share < 0.8 in "
                 "its 3 x 3-tile neighbourhood or > 25% white; ink = > 2% saturated green-blue pixels; blur = non-background tile below the 10th percentile of Laplacian variance; "
                 "gini_pos = Gini of positive contributions; conc_top16_of_pos = share of positive contribution from the 16 highest tiles; check value x 1000")
    res["E1_2"] = R; dump(res)
# ================================================================== E1.3 fixes
elif TASK.startswith("fix_"):
    which = TASK[4:]; R = {"base": {k: evaluate(F, NS, k, matched=(k == "R49")) for k, (F, NS) in RULES.items()}, "fixes": []}
    def heldout_top(): return TOPj
    if which == "score":
        # (a) substitute the sample H&E score by a robust aggregate of tile scores; P1 and R49 re-run with thresholds / cut-offs from the new inner out-of-fold scores
        from scipy.stats import rankdata as _rk
        rows = S["rows"]; boots = boots_for(rows)
        for agg in ("nonbg_mean", "core_mean", "trim10", "median", "noink_mean", "rest_mean"):
            Onew = np.stack([sig(TS[np.arange(n), j, FO[:, j] - 1, TQ["b0"]] + TS[np.arange(n), j, FO[:, j] - 1, TQ[agg]]) for j in range(10)], 1).astype(float)
            Inew = {}
            for (r, k), si in I["L-IMG"].items():
                ii = si.index.values; Inew[(r, k)] = pd.Series(sig(TS[ii, r - 1, k - 1, TQ["b0"]] + TS[ii, r - 1, k - 1, TQ[agg]]).astype(float), index=ii)
            O2 = {**O, "L-IMG": Onew}; I2 = {**I, "L-IMG": Inew}
            # P1 with the new score
            Xn = np.stack([np.array([Onew[s, j].max() for s in es]) for j in range(10)], 1); Fn = np.zeros((E, 10), bool)
            for j in range(10):
                for k in range(1, 11):
                    ts_ = train_side(j, k, img_inner=Inew); t = thr_sens(ts_["score"][ts_["lab"] == 1], 0.90); Fn[EF[:, j] == k, j] = Xn[EF[:, j] == k, j] >= t
            _, _, inf2 = TC.triage(O2, I2, FO, S["popm"], 0.95, 0.90); F49n, NS49n, _ = r49_calls(O2, inf2, FO, es)
            au = {"value_new": r3(fsa(Onew, rows)), "value_original": r3(fsa(O["L-IMG"], rows))}; bd = [fsa(Onew, b) - fsa(O["L-IMG"], b) for b in boots]
            au["delta"] = dstat(fsa(Onew, rows) - fsa(O["L-IMG"], rows), bd)
            R["fixes"].append({"family": "score", "agg": agg, "sample_auroc_fold_stratified_pre_event": au, "P1_new_score_90": evaluate(Fn, Z, f"P1 90% on {agg}"), "R49_new_score": evaluate(F49n, NS49n, f"R49 on {agg}")})
            print("score", agg, flush=True)
    elif which == "nsamp":
        # (d) more samples per endoscopy = more chances for a high maximum (FP-NP endoscopies carry more samples than TN-NP / TP-P): penalise the endoscopy maximum by c * log(samples)
        CG = np.round(np.arange(0, 3.01, 0.25), 2); lk = np.log(S["nsz"]).astype(float); TL_, CS_, TSq = cutoff_arrays(INFO, FO); LATE = O["L-LATE"]
        def r49_adj(sc_logit, late_top, tl, cs, ts):
            sc = sig(sc_logit); g = np.where(sc > cs, 2, 0) if tl > cs else np.where(sc < tl, 0, np.where(sc > cs, 2, 1)); return np.where(g == 1, late_top >= ts, g == 2), (g == 1).astype(int)
        mx = np.stack([np.array([lg(O["L-IMG"][s_, j]).max() for s_ in es]) for j in range(10)], 1); lt = np.stack([LATE[TOPj[:, j], j] for j in range(10)], 1)
        for eps in (0.0, 0.02):
            F1 = np.zeros((E, 10), bool); F4 = np.zeros((E, 10), bool); N4 = np.zeros((E, 10), int); ch1 = []; ch4 = []
            for j in range(10):
                for k in range(1, 11):
                    t = TRS[(j, k)]; lab = t["lab"]; tl_ = lk[t["idx"]]; sl = lg(t["score"]); i_ = j * 10 + k - 1; tl, cs, ts = INFO["t_low"][i_], INFO["c_star"][i_], INFO["t_seq"][i_]
                    vl = np.full(n, np.nan); sl_ = I["L-LATE"][(j + 1, k)]; vl[sl_.index.values] = sl_.values; ltt = vl[t["top"]]
                    best1 = (-1, 0.0, None); f0, _ = r49_adj(sl, ltt, tl, cs, ts); s0 = f0[lab == 1].mean(); best4 = (-1, 0.0)
                    for c_ in CG:
                        a_ = sl - c_ * tl_; th = thr_sens(a_[lab == 1], 0.90); sp = (a_[lab == 0] < th).mean()
                        if sp > best1[0] + 1e-12: best1 = (sp, c_, th)
                        f_, _ = r49_adj(a_, ltt, tl, cs, ts)
                        if f_[lab == 1].mean() >= s0 - eps - 1e-12:
                            nv = (~f_[lab == 0]).mean()
                            if nv > best4[0] + 1e-12: best4 = (nv, c_)
                    te = EF[:, j] == k; ch1.append(best1[1]); ch4.append(best4[1])
                    F1[te, j] = (mx[te, j] - best1[1] * lk[te]) >= best1[2]
                    F4[te, j], N4[te, j] = r49_adj(mx[te, j] - best4[1] * lk[te], lt[te, j], tl, cs, ts)
            if eps == 0.0:
                # P1 with max - c1 log(samples) - c2 log(1 + earlier endoscopies in the cohort): (c1, c2) on training specificity at 90% training sensitivity
                lp = np.log1p(S["POS"]).astype(float); F2 = np.zeros((E, 10), bool); SC2 = np.zeros((E, 10)); SC1 = np.zeros((E, 10)); ch2 = []
                for j in range(10):
                    for k in range(1, 11):
                        t = TRS[(j, k)]; lab = t["lab"]; sl = lg(t["score"]); best = (-1, 0.0, 0.0, None); te = EF[:, j] == k
                        for c1 in CG[::2]:
                            for c2 in CG[::2]:
                                a_ = sl - c1 * lk[t["idx"]] - c2 * lp[t["idx"]]; th = thr_sens(a_[lab == 1], 0.90); sp = (a_[lab == 0] < th).mean()
                                if sp > best[0] + 1e-12: best = (sp, c1, c2, th)
                        ch2.append([best[1], best[2]]); SC2[te, j] = mx[te, j] - best[1] * lk[te] - best[2] * lp[te]; F2[te, j] = SC2[te, j] >= best[3]
                        SC1[te, j] = mx[te, j] - ch1[j * 10 + k - 1] * lk[te]
                ch2 = np.array(ch2); R["fixes"].append({"family": "nsamp", "rule": "P1 90%", "eps": None, "c_chosen_median": [float(np.median(ch2[:, 0])), float(np.median(ch2[:, 1]))],
                                                        "nested": evaluate(F2, Z, "P1 90% on max - c1 log(samples) - c2 log(1 + position) (design-sensitive)")})
                def eauc(Sc, m=None):
                    w = np.ones(E) if m is None else m[pidx]; out = []
                    for j in range(10):
                        num = den = 0.0
                        for k in range(1, 11):
                            f = (EF[:, j] == k) & (w > 0); xp, xn, wp, wn = Sc[f & Pe, j], Sc[f & Ne, j], w[f & Pe], w[f & Ne]
                            if not len(xp) or not len(xn): continue
                            W_ = wp[:, None] * wn[None, :]; num += (((xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])) * W_).sum(); den += W_.sum()
                        out.append(num / den)
                    return float(np.mean(out))
                au = {}
                for nm_, Sc in (("max - c log(samples)", SC1), ("max - c1 log(samples) - c2 log(1 + position)", SC2)):
                    o0 = eauc(mx); o1 = eauc(Sc); bd = np.array([eauc(Sc, m) - eauc(mx, m) for m in D]); au[nm_] = {"auroc_max": r3(o0), "auroc_new": r3(o1), "delta": dstat(o1 - o0, bd)}
                R["endoscopy_auroc_fold_stratified"] = au
                R["fixes"].append({"family": "nsamp", "rule": "P1 90%", "eps": None, "c_chosen_median": float(np.median(ch1)), "c_chosen_iqr": [float(np.percentile(ch1, 25)), float(np.percentile(ch1, 75))], "nested": evaluate(F1, Z, "P1 90% on max - c log(samples)")})
            R["fixes"].append({"family": "nsamp", "rule": "R49", "eps": eps, "c_chosen_median": float(np.median(ch4)), "c_chosen_iqr": [float(np.percentile(ch4, 25)), float(np.percentile(ch4, 75))], "nested": evaluate(F4, N4, f"R49 on max - c log(samples) eps {eps}")})
    elif which in ("veto", "filter"):
        EPS = (0.0, 0.02, 0.05); NAMES_V = SAMPLE_FEATS + TILE_FEATS
        def ho_feats(j, k, names):
            """held-out endoscopies of fold k in repeat j: features of the outer top sample under the outer model"""
            e_idx = np.where(EF[:, j] == k)[0]; top = TOPj[e_idx, j]; return e_idx, feats(e_idx, top, j, np.full(len(top), k), names)
        def tr_feats(j, k, names):
            t = TRS[(j, k)]; return feats(t["idx"], t["top"], j, np.full(len(t["top"]), k), names)
        def pick_tau(xp, xn, allowed):
            """veto if d * x > tau; tau keeps at most `allowed` flagged progressor endoscopies vetoed; returns (tau, d, n NP vetoed) for the better direction"""
            best = (np.inf, 1, -1)
            for d in (1, -1):
                a = np.sort(d * xp[np.isfinite(xp)])[::-1]; tau = a[allowed] if allowed < len(a) else -np.inf; nv = int(np.sum(d * xn[np.isfinite(xn)] > tau))
                if nv > best[2]: best = (tau, d, nv)
            return best
        if which == "veto":
            for rn, (F, NS) in RULES.items():
                for eps in EPS:
                    for nm in NAMES_V + ["auto"]:
                        Fv = F.copy(); chosen = {}; Fa = F.copy()
                        for j in range(10):
                            xh_all = {};
                            for k in range(1, 11):
                                t = TRS[(j, k)]; fl = t["F"][rn]; allowed = int(np.floor(eps * fl[t["lab"] == 1].sum()))
                                cand = NAMES_V if nm == "auto" else [nm]; best = None
                                for c_ in cand:
                                    x = tr_feats(j, k, [c_])[:, 0]; tau, d, nv = pick_tau(x[fl & (t["lab"] == 1)], x[fl & (t["lab"] == 0)], allowed)
                                    if best is None or nv > best[3]: best = (c_, tau, d, nv)
                                c_, tau, d, nv = best; chosen[c_] = chosen.get(c_, 0) + 1
                                e_idx, xh = ho_feats(j, k, [c_]); xh = xh[:, 0]; Fv[e_idx, j] = F[e_idx, j] & ~(np.isfinite(xh) & (d * xh > tau))
                            if nm != "auto":   # apparent (optimistic): tau chosen on the evaluated flagged endoscopies of this repeat
                                xe = np.concatenate([ho_feats(j, k, [nm])[1][:, 0] for k in range(1, 11)]); ee = np.concatenate([np.where(EF[:, j] == k)[0] for k in range(1, 11)])
                                xa = np.full(E, np.nan); xa[ee] = xe; fl = F[:, j]; allowed = int(np.floor(eps * fl[Pe].sum())); tau, d, _ = pick_tau(xa[fl & Pe], xa[fl & Ne], allowed)
                                Fa[:, j] = F[:, j] & ~(np.isfinite(xa) & (d * xa > tau))
                        o = {"family": "veto", "rule": rn, "eps": eps, "feature": nm, "nested": evaluate(Fv, NS, f"{rn} veto {nm} eps {eps}", matched=False), "features_chosen": chosen if nm == "auto" else None}
                        if nm != "auto": A_ = patient_arrays(S, Fa, NS); po = parts(S, A_, one); o["apparent_optimistic"] = {k: r3(f(po)[0]) for k, f in MET.items()}
                        R["fixes"].append(o)
                    print("veto", rn, eps, flush=True)
        else:
            SETS_F = {"image quality": SAMPLE_FEATS, "tile contributions": TILE_FEATS, "image quality + tile": SAMPLE_FEATS + TILE_FEATS,
                      "image quality + tile + clinical": SAMPLE_FEATS + TILE_FEATS + ["CLIN"], "clinical": ["CLIN"],
                      "slide metadata: scanner, slide age, biopsy year (not deployable)": ["scanner_C13210", "slide_age_years", "biopsy_year"],
                      "sequencing batch progressor share, training labels only (not deployable)": ["BATCHN"],
                      "artefact incl. batch shares from all labels (LEAKS LABELS; demonstration only)": ARTEFACT_FEATS}
            fdu_ = sheet().drop_duplicates("combined_name").set_index("combined_name"); BATCH = pd.Series(ids).map(fdu_.Batch).astype(str).values
            def batch_share(top, j, k):
                """share of progressor samples in the top sample's sequencing batch among the training samples of split (j, k), the sample's own patient excluded"""
                tr = FO[:, j] != k; out = np.full(len(top), np.nan)
                for i_, s_ in enumerate(top):
                    m = tr & (BATCH == BATCH[s_]) & (pat != pat[s_])
                    if m.any(): out[i_] = y[m].mean()
                return out
            def build(e_idx, top, j, kvec, names, score, kb=None):
                nm_ = [x for x in names if x not in ("CLIN", "BATCHN")]; X_ = feats(e_idx, top, j, kvec, nm_)
                if "CLIN" in names: X_ = np.column_stack([X_, clin_block(e_idx)])
                if "BATCHN" in names: X_ = np.column_stack([X_, batch_share(top, j, kb)])
                return np.column_stack([X_, score])
            def hscore(e_idx, j): return lg(np.array([O["L-IMG"][es[e], j].max() for e in e_idx]))
            def fitpred(Xtr, ytr, Xte):
                med = np.nanmedian(Xtr, 0); med = np.where(np.isfinite(med), med, 0); Xtr = np.where(np.isfinite(Xtr), Xtr, med); Xte = np.where(np.isfinite(Xte), Xte, med)
                mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1; m_ = LogisticRegression(C=1.0, max_iter=5000).fit((Xtr - mu) / sd, ytr); return m_.predict_proba((Xte - mu) / sd)[:, 1]
            for rn, (F, NS) in RULES.items():
                for sname, names in SETS_F.items():
                    Fv = {eps: F.copy() for eps in EPS}; Fa = {eps: F.copy() for eps in EPS}; aucs = []
                    for j in range(10):
                        ph = np.full(E, np.nan)
                        for k in range(1, 11):
                            t = TRS[(j, k)]; fl = t["F"][rn]; ti = t["idx"][fl]; ty = t["lab"][fl]; tp_ = pat[[es[e][0] for e in ti]]
                            Xtr = build(ti, t["top"][fl], j, np.full(fl.sum(), k), names, lg(t["score"][fl]), kb=k)
                            oof = np.full(len(ty), np.nan)
                            for a_, b_ in GroupKFold(5).split(Xtr, ty, tp_):
                                if len(set(ty[a_])) < 2: oof[b_] = ty[a_].mean(); continue
                                oof[b_] = fitpred(Xtr[a_], ty[a_], Xtr[b_])
                            e_idx = np.where((EF[:, j] == k) & F[:, j])[0]
                            if not len(e_idx): continue
                            Xh = build(e_idx, TOPj[e_idx, j], j, np.full(len(e_idx), k), names, hscore(e_idx, j), kb=k)
                            ph[e_idx] = fitpred(Xtr, ty, Xh)
                            for eps in EPS:
                                allowed = int(np.floor(eps * (ty == 1).sum())); a = np.sort(oof[ty == 1]); tau = a[allowed] if allowed < len(a) else np.inf
                                Fv[eps][e_idx, j] = ph[e_idx] >= tau
                        fl = F[:, j]; aucs.append(wauc(ph[fl & Pe], np.ones((fl & Pe).sum()), ph[fl & Ne], np.ones((fl & Ne).sum())))
                        # apparent (optimistic): fitted and thresholded on the evaluated flagged endoscopies of this repeat (in-sample)
                        ee = np.where(fl)[0]; Xe = build(ee, TOPj[ee, j], j, EF[ee, j], names, hscore(ee, j), kb=0)
                        pa = fitpred(Xe, elab[ee], Xe)
                        for eps in EPS:
                            allowed = int(np.floor(eps * Pe[ee].sum())); a = np.sort(pa[Pe[ee]]); tau = a[allowed] if allowed < len(a) else np.inf; Fa[eps][ee, j] = pa >= tau
                    for eps in EPS:
                        o = {"family": "filter", "rule": rn, "eps": eps, "feature_set": sname, "auroc_heldout_flagged_P_vs_NP_per_repeat_mean": r3(np.nanmean([a_ if a_ is not None else np.nan for a_ in aucs])),
                             "nested": evaluate(Fv[eps], NS, f"{rn} filter {sname} eps {eps}", matched=False)}
                        A_ = patient_arrays(S, Fa[eps], NS); po = parts(S, A_, one); o["apparent_optimistic"] = {k: r3(f(po)[0]) for k, f in MET.items()}; R["fixes"].append(o)
                    print("filter", rn, sname, flush=True)
    R["note"] = ("nested: every threshold / model fitted on the training patients of each outer fold (R49 / P1 flags of training endoscopies from inner out-of-fold scores and the fold's own cut-offs; "
                 "tile summaries of training samples under the inner models of that split), applied to the held-out fold; eps = largest share of training flagged progressor endoscopies a veto may remove; "
                 "vetoed endoscopies keep their sequencing; apparent = threshold or model chosen on the evaluated endoscopies themselves (optimistic, labelled); deltas: paired patient bootstrap "
                 "(rv8 pdraws, RandomState(0)), thresholds and calls fixed; P1 matched = P1 threshold re-matched per repeat and per draw to the rule's realised endoscopy sensitivity (rv8b S1 (i))")
    res["E1_3"] = R; dump(res); pickle.dump(FSAVE, open(f"{X1}/fixF_{which}.pkl", "wb"))
elif TASK == "summary":
    rows_ = []; FS = {}
    for w in ("score", "veto", "filter", "nsamp"):
        f_ = f"{OUT}/fix_{w}.json"
        if not os.path.exists(f_): continue
        FS.update(pickle.load(open(f"{X1}/fixF_{w}.pkl", "rb")))
        for fx in json.load(open(f_))["E1_3"]["fixes"]:
            evs = [fx["P1_new_score_90"], fx["R49_new_score"]] if w == "score" else [fx["nested"]]
            for ev in evs:
                g = lambda a, b="delta_vs_R49": ev[b][a]["delta"]; c = lambda a, b="delta_vs_R49": ev[b][a]["ci95"]
                rows_.append({"label": ev["label"], "family": w, "caught": ev["value"]["progressor_patients_caught"], "np_flagged": ev["value"]["non_progressor_patients_flagged"],
                              "endo_sens": ev["value"]["endoscopy_sensitivity"], "endo_spec": ev["value"]["endoscopy_specificity"], "seq_per_patient": ev["value"]["samples_sequenced_per_patient"],
                              "d_np_vs_R49": g("non_progressor_patients_flagged"), "d_np_vs_R49_ci": c("non_progressor_patients_flagged"), "d_caught_vs_R49": g("progressor_patients_caught"), "d_caught_vs_R49_ci": c("progressor_patients_caught"),
                              "d_np_vs_P1": g("non_progressor_patients_flagged", "delta_vs_P1 90%"), "d_np_vs_P1_ci": c("non_progressor_patients_flagged", "delta_vs_P1 90%"),
                              "d_caught_vs_P1": g("progressor_patients_caught", "delta_vs_P1 90%"), "d_caught_vs_P1_ci": c("progressor_patients_caught", "delta_vs_P1 90%"),
                              "apparent_optimistic": fx.get("apparent_optimistic") if w != "score" else None, "eps": fx.get("eps"), "rule": fx.get("rule", "R49" if "R49" in ev["label"] else "P1 90%")})
    DF = pd.DataFrame(rows_); dep = ~DF.label.str.contains("not deployable|LEAKS"); DF["deployable"] = dep
    for r_, d_ in zip(rows_, dep): r_["deployable"] = bool(d_)
    keep = DF[dep & (DF.d_caught_vs_R49 >= -0.5)].sort_values("d_np_vs_R49"); top = list(keep.label.head(10))
    top += [l_ for l_ in DF[~dep & DF.label.str.contains("training labels only") & (DF.eps == 0.05)].label]
    for lab in ("R49 veto auto eps 0.0", "P1 90% veto auto eps 0.0", "P1 90% on max - c log(samples)", "P1 90% on max - c1 log(samples) - c2 log(1 + position) (design-sensitive)"):
        if lab in FS and lab not in top: top.append(lab)
    re_ = []
    for lab in top:
        F_, NS_ = FS[lab]; re_.append(evaluate(F_, NS_, lab, matched=True)); print("summary", lab, flush=True)
    res["E1_3_summary"] = {"all": rows_, "n_fixes_evaluated": len(rows_), "selected_matched": re_, "base": {k: evaluate(F, NS, k, matched=True) for k, (F, NS) in RULES.items()},
                           "note": "every nested fix evaluated (rows; deployable = uses no study-design metadata or labels); the 10 deployable fixes with the largest reduction in non-progressor patients flagged against R49 among those losing at most 0.5 progressor patients (point estimate), plus the nested batch-share filter at eps 0.05, the automatic-feature vetoes at eps 0 and the P1 sample-count penalties, re-evaluated with P1 matched to their realised endoscopy sensitivity. Selection over many fixes on the same data: the selected rows are optimistic as a group."}
    dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
