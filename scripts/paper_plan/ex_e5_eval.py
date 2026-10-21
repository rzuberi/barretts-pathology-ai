"""E5 evaluation (docs/paper_explore_E5.md): ERIN "dysplasia-likeness" scores on the 571 pre-event samples, (a) alone, (b) stacked with L-IMG,
(c) as a veto / confirmation inside the R49 gate. Policy machinery, thresholds, outcomes and bootstrap draws are round 8 / 8b's: rv8b_step3.py's module
body up to its task dispatch is executed unchanged (which executes rv8_landscape.py's). ERIN models never saw SWG labels; every threshold and stacker is
fitted on the outer fold's training patients only (inner out-of-fold L-IMG; stacker scores for training patients cross-fitted in 5 patient-grouped folds).
Apparent (non-nested) versions are labelled optimistic. Aggregates -> results/paper_final/explore_E5/<E5TASK>.json; nothing row-level leaves the cluster.
E5TASK = auc | a | b | c | mem (round-9 EW memory rule with ERIN inside; rv9_falseflags.py engine).  E5_NB=k: smoke mode (k draws, aggregates written to killcoyne_mm/explore_E5/smoke_json)."""
import os, sys, json, numpy as np, pandas as pd
PP = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"; E5T = os.environ["E5TASK"]; os.environ["TASK"] = "e5_" + E5T
PRE_ = "rv9_falseflags.py" if E5T == "mem" else "rv8b_step3.py"   # mem: the round-9 nested engine (which itself executes rv8b / rv8 unchanged)
__file__ = PP + "/" + PRE_; sys.path.insert(0, PP)
_s = open(PP + "/" + PRE_).read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], PP + "/" + PRE_, "exec"))
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
E5D = M + "/explore_E5"; OUT = T + "/results/paper_final/explore_E5"; NB = 2000
if os.environ.get("E5_NB"): NB = int(os.environ["E5_NB"]); TC.NB = NB; OUT = E5D + "/smoke_json"
os.makedirs(OUT, exist_ok=True)
def dump(res, name):
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("E5 DONE", name, flush=True)
# ------------------------------------------------------------------ ERIN sample scores (max over a sample's slides)
SS = pd.read_csv(E5D + "/sample_slides.csv", dtype=str)
if os.environ.get("E5_FAKE"):   # pipeline smoke test only: random scores, aggregates to smoke_json
    rf = np.random.RandomState(1); hh = sorted(set(SS.h5)); SC = pd.DataFrame({"h5": hh, "logit_lgd": rf.randn(len(hh)), "logit_hgd": rf.randn(len(hh)), "logit_p32": rf.randn(len(hh)), "tile_top16_lgd": rf.randn(len(hh)), "tile_frac_lgd": rf.rand(len(hh)),
                                                                              **{f"p6_{c}": rf.dirichlet(np.ones(6), len(hh))[:, i] for i, c in enumerate(["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"])}}).set_index("h5")
    OUT = E5D + "/smoke_json"; os.makedirs(OUT, exist_ok=True)
else: SC = pd.read_csv(E5D + "/erin_scores_slides.csv").set_index("h5")
SC["logit_six_lgdplus"] = lg(SC.p6_LGD + SC.p6_HGD + SC.p6_CANCER); SC["logit_six_hgdplus"] = lg(SC.p6_HGD + SC.p6_CANCER); SC["logit_six_lgd"] = lg(SC.p6_LGD); SC["logit_tile_frac"] = lg(SC.tile_frac_lgd)
EKEYS = {"erin_lgd": "logit_lgd", "erin_hgd": "logit_hgd", "erin_six_lgdplus": "logit_six_lgdplus", "erin_six_hgdplus": "logit_six_hgdplus", "erin_six_lgd": "logit_six_lgd", "erin_p32": "logit_p32", "erin_tile_top16": "tile_top16_lgd", "erin_tile_frac": "logit_tile_frac"}
ES = {}
for k, c in EKEYS.items():
    v = SS.assign(v=SS.h5.map(SC[c])).groupby("cnv").v.max(); ES[k] = pd.Series(ids).map(v).values.astype(float)
PRIMARY = "erin_lgd"
S = setup(); D = pdraws(S, NB); one = np.ones(S["P_"]); rows = S["rows"]; es = S["es"]; E = S["E"]; O, I, FO = S["O"], S["I"], S["FO"]
assert all(np.isfinite(ES[k][rows]).all() for k in ES), "pre-event sample without an ERIN score"
nP = int((S["plab"] == 1).sum()); nN = int((S["plab"] == 0).sum()); Z = np.zeros((E, 10), int)
def add_key(key, Oarr, Idict):
    """sample-level score -> endoscopy max per repeat (outer) and per (repeat, fold) over the training endoscopies (Idict: (r, k) -> Series by sample position)"""
    S["OUTS"][key] = np.stack([np.array([Oarr[s, j].max() for s in es]) for j in range(10)], 1)
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            tr = S["TR"][(j, k)]; v = np.full(n, np.nan); si = Idict[(r, k)]; v[si.index.values] = si.values
            x = np.array([v[es[e]].max() for e in tr["idx"]]); assert np.isfinite(x).all(); tr[key] = x
def const_inner(x):
    return {(r, k): pd.Series(x[I["L-IMG"][(r, k)].index.values], index=I["L-IMG"][(r, k)].index.values) for r in REPS for k in range(1, 11)}
for k in ES: add_key(k, np.repeat(ES[k][:, None], 10, 1), const_inner(ES[k]))
def cstar_arr(key, q):
    """per endoscopy x repeat: the fold's c* on the training NON-progressor endoscopies' score (q specificity; high = score > c*)"""
    C = np.full((E, 10), np.nan)
    for j in range(10):
        for k in range(1, 11): tr = S["TR"][(j, k)]; C[S["EF"][:, j] == k, j] = c_star(tr[key][tr["lab"] == 0], q)
    return C
# ------------------------------------------------------------------ outcome machinery (rv8b parts / met)
MET = {"progressor_patients_caught": lambda p: nP * met(p, "caught_share"), "non_progressor_patients_flagged": lambda p: nN * met(p, "np_flagged_share"),
       "endoscopy_sensitivity": lambda p: met(p, "endo_sens"), "endoscopy_specificity": lambda p: met(p, "endo_spec"),
       "samples_sequenced_per_patient": lambda p: met(p, "seq_per_patient"), "flags_per_np_patient_year": lambda p: met(p, "flags_per_np_py")}
def outcomes(F, NS):
    A = patient_arrays(S, F, NS); po = parts(S, A, one); pb = parts(S, A, D)
    return A, {k: f(po)[0] for k, f in MET.items()}, {k: f(pb) for k, f in MET.items()}
_, _, info49 = TC.triage(O, I, FO, S["popm"], 0.95, 0.90); F49, NS49, GB49 = r49_calls(O, info49, FO, es)
TH90, _ = thr_arr(S, "img", 0.90); F1 = S["OUTS"]["img"] >= TH90
REF = {"R49": outcomes(F49, NS49), "P1 90%": outcomes(F1, Z)}
chk = {k: [r2(REF[k][1]["progressor_patients_caught"]), r2(REF[k][1]["non_progressor_patients_flagged"])] for k in REF}
assert abs(chk["R49"][0] - 31.1) < 0.051 and abs(chk["R49"][1] - 31.8) < 0.051 and abs(chk["P1 90%"][1] - 36.0) < 0.051, chk
Pe = S["elab"] == 1
def matched_spec(X, sens, m):
    """endoscopy score X (E, 10): per repeat, the largest threshold reaching >= sens[j] endoscopy sensitivity on the evaluated (multiplicity-weighted) endoscopies
    (rv8b S1(i) matching) -> endoscopy specificity (repeat mean)"""
    w = m[S["pidx"]].astype(int); out = []
    for j in range(10):
        t = tmatch(X[Pe, j], sens[j], w[Pe]); fl = X[:, j] >= t; wn = w[~Pe]; out.append(1 - (fl[~Pe] * wn).sum() / wn.sum())
    return float(np.mean(out))
MS = np.vstack([one[None, :], D])
def sens_reps(A): return per_rep(parts(S, A, MS), "endo_sens")   # (1 + draws, 10): observed row first
S49 = sens_reps(REF["R49"][0]); SP49 = np.concatenate([[REF["R49"][1]["endoscopy_specificity"]], REF["R49"][2]["endoscopy_specificity"]])
def rule_result(label, F, NS, score=None, extra=None):
    """outcomes with CIs; paired deltas against R49 and P1 90%; delta specificity against P1 matched to the rule's realised endoscopy sensitivity (per draw);
    for score rules also the score's specificity matched to R49's realised sensitivity minus R49's"""
    A, ob, bs = outcomes(F, NS); o = {"label": label, "observed": {k: r3(v) for k, v in ob.items()}, "ci95": {k: ci2(v) for k, v in bs.items()}}
    o["share_samples_sequenced"] = r3(float(A["NS"].sum(0).mean() / S["nsz"].sum()))
    for rn, (_, rob, rbs) in REF.items(): o[f"delta_vs_{rn}"] = {k: dstat(ob[k] - rob[k], bs[k] - rbs[k]) for k in MET}
    sr = sens_reps(A); spec_r = np.concatenate([[ob["endoscopy_specificity"]], bs["endoscopy_specificity"]])
    p1m = np.array([matched_spec(S["OUTS"]["img"], sr[i], MS[i]) for i in range(len(MS))])
    o["delta_spec_vs_P1_matched_to_rule_sensitivity"] = dstat(spec_r[0] - p1m[0], spec_r[1:] - p1m[1:])
    if score is not None:
        xm = np.array([matched_spec(score, S49[i], MS[i]) for i in range(len(MS))]); o["delta_spec_score_matched_to_R49_sensitivity_minus_R49"] = dstat(xm[0] - SP49[0], xm[1:] - SP49[1:])
    if extra: o.update(extra)
    print(label, o["observed"]["progressor_patients_caught"], o["observed"]["non_progressor_patients_flagged"], flush=True); return o
def apparent_thr(X, q):
    """optimistic: one threshold from ALL patients' progressor endoscopies (repeat-specific X)"""
    return np.stack([np.full(E, thr_sens(X[Pe, j], q)) for j in range(10)], 1)
res = {"task": E5T, "draws": int(len(D)), "check_round8_caught_npflagged": chk, "n_patients": S["P_"], "n_endoscopies": E, "n_samples": int(len(rows))}
# ================================================================== AUROC and transfer checks
if E5T == "auc":
    from tb_common import boots_for
    B = boots_for(rows, 0, NB); out = {}
    SCO = {"L-IMG": O["L-IMG"], **{k: np.repeat(ES[k][:, None], 10, 1) for k in ES}}
    bl0 = fsa(O["L-IMG"], rows); bl = np.array([fsa(O["L-IMG"], b) for b in B])
    for k, X in SCO.items():
        ob = fsa(X, rows); bd = bl if k == "L-IMG" else np.array([fsa(X, b) for b in B]); o = {"sample_fold_stratified_auroc": r3(ob), "ci95": ci2(bd)}
        if k != "L-IMG": o["delta_vs_L-IMG"] = dstat(ob - bl0, bd - bl)
        out[k] = o
    def wauc(x, yv, w):
        p = yv == 1; xp, xn, wp, wn = x[p], x[~p], w[p], w[~p]; W = wp[:, None] * wn[None, :]
        return float((((xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])) * W).sum() / W.sum())
    INC = S["INC"]; ENDO = S["OUTS"]
    def lvl_auc(Xe, m, level):
        if level == "endoscopy": return np.mean([wauc(Xe[:, j], S["elab"], m[S["pidx"]]) for j in range(Xe.shape[1])])
        return np.mean([wauc(Xe[:, j], S["plab"], m) for j in range(Xe.shape[1])])
    PM = lambda Xe: np.stack([np.array([Xe[S["pidx"] == i, j].max() for i in range(S["P_"])]) for j in range(Xe.shape[1])], 1)
    for level in ("endoscopy", "patient"):
        cv = (lambda Xe: Xe) if level == "endoscopy" else PM; Xb = cv(ENDO["img"]); base = np.array([lvl_auc(Xb, m, level) for m in MS])
        for k in ["img"] + list(ES):
            Xk = cv(ENDO[k]); v = base if k == "img" else np.array([lvl_auc(Xk, m, level) for m in MS]); o = {"auroc": r3(v[0]), "ci95": ci2(v[1:])}
            if k != "img": o["delta_vs_L-IMG"] = dstat(v[0] - base[0], v[1:] - base[1:])
            out.setdefault(f"{level}_max", {})["L-IMG" if k == "img" else k] = o
    # transfer sanity on our slides: ERIN score vs the sample's recorded grade (pre-event samples), and agreement with L-IMG
    G = SM.Pathology.values; tr_ = {}
    for k in ES:
        m_ = np.isin(G[rows], ["NDBE", "ID", "LGD"]); x = ES[k][rows][m_]; g = G[rows][m_]
        def a2(pos, neg): mm = np.isin(g, pos + neg); yy = np.isin(g[mm], pos).astype(int); return r3(wauc(x[mm], yy, np.ones(mm.sum()))) if 0 < yy.sum() < mm.sum() else None
        tr_[k] = {"LGD_vs_NDBE": a2(["LGD"], ["NDBE"]), "ID_or_LGD_vs_NDBE": a2(["ID", "LGD"], ["NDBE"]), "spearman_with_repeat_mean_L-IMG": r3(spearmanr(ES[k][rows], O["L-IMG"][rows].mean(1)).correlation),
                  "n_LGD": "<5" if 0 < (g == "LGD").sum() < 5 else int((g == "LGD").sum()), "n_NDBE": int((g == "NDBE").sum())}
    out["transfer_vs_recorded_grade_pre_event"] = tr_; res["AUC"] = out; dump(res, "e5_auc")
# ================================================================== (a) ERIN score alone as an H&E rule (P1-like)
elif E5T == "a":
    out = {}
    for k in ES:
        for q in Q3:
            TH, miss = thr_arr(S, k, q); X = S["OUTS"][k]
            out[f"{k} {int(q * 100)}%"] = rule_result(f"(a) flag if max {k} >= t ({int(q * 100)}% training endoscopy sensitivity)", X >= TH, Z, score=X if q == 0.90 else None, extra={"folds_target_not_met": miss})
        X = S["OUTS"][k]; out[f"{k} 90% APPARENT"] = rule_result(f"(a) {k} 90%, threshold from all patients (OPTIMISTIC)", X >= apparent_thr(X, 0.90), Z)
    res["A"] = out; dump(res, "e5_a")
# ================================================================== (b) stacked: logistic(logit inner-OOF L-IMG, ERIN logit) per outer fold
elif E5T == "b":
    out = {}; coefs = {}
    FEAT = {k: ES[k][:, None] for k in (PRIMARY, "erin_p32", "erin_hgd", "erin_tile_top16")}
    FEAT["erin_profile"] = np.c_[ES["erin_lgd"], ES["erin_hgd"], ES["erin_six_lgd"], ES["erin_six_hgdplus"], ES["erin_p32"], ES["erin_tile_top16"]]   # ERIN grade profile (6 columns)
    for k in FEAT:
        e = FEAT[k]; STK = np.full((n, 10), np.nan); STKI = {}; cf = []
        for j, r in enumerate(REPS):
            for f in range(1, 11):
                si = I["L-IMG"][(r, f)]; tri = si.index.values[S["popm"][si.index.values]]; Xtr = np.c_[lg(si.reindex(tri).values), e[tri]]; ytr = y[tri]
                te = np.where((FO[:, j] == f) & S["popm"])[0]; Xte = np.c_[lg(O["L-IMG"][te, j]), e[te]]
                sc = StandardScaler().fit(Xtr); lr = LogisticRegression(C=1.0).fit(sc.transform(Xtr), ytr); STK[te, j] = lr.predict_proba(sc.transform(Xte))[:, 1]; cf.append(lr.coef_[0].tolist())
                cv = np.full(len(tri), np.nan)
                for a, b in GroupKFold(5).split(Xtr, ytr, pat[tri]):
                    s2 = StandardScaler().fit(Xtr[a]); l2 = LogisticRegression(C=1.0).fit(s2.transform(Xtr[a]), ytr[a]); cv[b] = l2.predict_proba(s2.transform(Xtr[b]))[:, 1]
                STKI[(r, f)] = pd.Series(cv, index=tri)
        cf = np.array(cf); coefs[k] = {"standardised_coef_L-IMG_mean_range": [r3(cf[:, 0].mean()), r3(cf[:, 0].min()), r3(cf[:, 0].max())],
                    "standardised_coef_ERIN_mean_range": [[r3(cf[:, c].mean()), r3(cf[:, c].min()), r3(cf[:, c].max())] for c in range(1, cf.shape[1])], "share_folds_ERIN_coef_positive": [r3((cf[:, c] > 0).mean()) for c in range(1, cf.shape[1])]}
        key = "stk_" + k; add_key(key, STK, STKI); X = S["OUTS"][key]
        a_ = fsa(STK, rows); coefs[k]["sample_fold_stratified_auroc_stack"] = r3(a_); coefs[k]["sample_fold_stratified_auroc_L-IMG"] = r3(fsa(O["L-IMG"], rows))
        for q in Q3:
            TH, miss = thr_arr(S, key, q); out[f"stack({k}) P1 {int(q * 100)}%"] = rule_result(f"(b)(i) flag if max stack(L-IMG, {k}) >= t ({int(q * 100)}%)", X >= TH, Z, score=X if q == 0.90 else None, extra={"folds_target_not_met": miss})
        O2 = dict(O); I2 = dict(I); O2["L-IMG"] = STK; I2["L-IMG"] = STKI
        _, _, inf2 = TC.triage(O2, I2, FO, S["popm"], 0.95, 0.90); F2, NS2, _ = r49_calls(O2, inf2, FO, es)
        out[f"R49 banding by stack({k})"] = rule_result(f"(b)(ii) R49 with stack(L-IMG, {k}) in place of L-IMG for banding; L-LATE unchanged", F2, NS2, extra={"folds_no_middle_band": int(inf2["no_mid"]), "t_seq_fallback": int(inf2["fallback"])})
        O3 = dict(O2); I3 = dict(I2); O3["L-LATE"] = (O["L-CNV"] + STK) / 2; I3["L-LATE"] = {kk: (I["L-CNV"][kk] + STKI[kk].reindex(I["L-CNV"][kk].index)) / 2 for kk in STKI}
        _, _, inf3 = TC.triage(O3, I3, FO, S["popm"], 0.95, 0.90); F3, NS3, _ = r49_calls(O3, inf3, FO, es)
        out[f"R49 banding + L-LATE by stack({k})"] = rule_result(f"(b)(iii) R49 with stack in place of L-IMG everywhere (L-LATE = (L-CNV + stack)/2)", F3, NS3)
        # apparent: stacker fitted on all pre-event samples with outer L-IMG, threshold from all patients (OPTIMISTIC)
        SA = np.full((n, 10), np.nan)
        for j in range(10):
            Xa = np.c_[lg(O["L-IMG"][rows, j]), e[rows]]; sc = StandardScaler().fit(Xa); SA[rows, j] = LogisticRegression(C=1.0).fit(sc.transform(Xa), y[rows]).predict_proba(sc.transform(Xa))[:, 1]
        XA = np.stack([np.array([SA[s, j].max() for s in es]) for j in range(10)], 1)
        out[f"stack({k}) P1 90% APPARENT"] = rule_result(f"(b) stack({k}) fitted and thresholded on all patients (OPTIMISTIC)", XA >= apparent_thr(XA, 0.90), Z)
    res["B"] = out; res["stack_coefficients"] = coefs; dump(res, "e5_b")
# ================================================================== (c) ERIN veto / confirmation inside R49 (and P1)
elif E5T == "c":
    TL, CSt, TS = cutoff_arrays(info49, FO); TSe = np.stack([TS[s[0]] for s in es]); seq = S["OUTS"]["top"] >= TSe
    assert ((GB49 == 2) | ((GB49 == 1) & seq) == F49).all(), "R49 reconstruction"
    out = {}; NS1 = (GB49 == 1).astype(int)
    for k in ES:
        X = S["OUTS"][k]
        for q in (0.90, 0.95, 0.98):
            if k != PRIMARY and q != 0.95: continue
            TAU, miss = thr_arr(S, k, q); ok = X >= TAU; tg = f"{k} q{int(q * 100)}"
            out[f"c1 {tg}"] = rule_result(f"(c1) R49 flag kept only if max {k} >= tau ({int(q * 100)}% of training progressor endoscopies)", F49 & ok, NS49, extra={"folds_target_not_met": miss})
            out[f"c2 {tg}"] = rule_result(f"(c2) R49, high-band (H&E-only) flags need {k} >= tau ({int(q * 100)}%)", ((GB49 == 2) & ok) | ((GB49 == 1) & seq), NS49)
            mid = (GB49 == 1) | ((GB49 == 2) & ~ok)
            out[f"c3 {tg}"] = rule_result(f"(c3) R49, high-band flags with {k} < tau ({int(q * 100)}%) sequenced instead (top sample, L-LATE >= t_seq)", ((GB49 == 2) & ok) | (mid & seq), mid.astype(int))
            out[f"c5 {tg}"] = rule_result(f"(c5) P1 90% flag kept only if max {k} >= tau ({int(q * 100)}%)", F1 & ok, Z)
        CE = cstar_arr(k, 0.90); resc = (GB49 == 0) & (X > CE); mid = (GB49 == 1) | resc
        out[f"c4 {k}"] = rule_result(f"(c4) R49, low-band endoscopies with {k} above the training non-progressor 90% cut-off rescued into the middle band", (GB49 == 2) | (mid & seq), mid.astype(int))
        if k == PRIMARY:
            TAU, _ = thr_arr(S, k, 0.95); ok = X >= TAU; mid = (GB49 == 1) | ((GB49 == 2) & ~ok) | resc
            out[f"c3+c4 {k} q95"] = rule_result("(c3+c4) both: weak-ERIN high-band flags sequenced, strong-ERIN low-band endoscopies sequenced", ((GB49 == 2) & ok) | (mid & seq), mid.astype(int))
            TA = apparent_thr(X, 0.95); out[f"c1 {k} q95 APPARENT"] = rule_result("(c1) tau from all patients' progressor endoscopies (OPTIMISTIC)", F49 & (X >= TA), NS49)
            out[f"c5 {k} q95 APPARENT"] = rule_result("(c5) P1 90% with tau from all patients (OPTIMISTIC)", F1 & (X >= TA), Z)
            # how many R49 flags the veto touches (endoscopy level, repeat mean)
            res["veto_touch_primary_q95"] = {"R49_flagged_endoscopies": r2(F49.sum(0).mean()), "vetoed_progressor_endoscopies": r2((F49 & ~ok & Pe[:, None]).sum(0).mean()),
                                             "vetoed_non_progressor_endoscopies": r2((F49 & ~ok & ~Pe[:, None]).sum(0).mean()), "high_band_flags": r2((GB49 == 2).sum(0).mean())}
    res["C"] = out; dump(res, "e5_c")
# ================================================================== round-9 memory rule (EW memory of the R49-aligned gate score) with ERIN inside
elif E5T == "mem":
    ST = statics(S); INFO = TC.triage(O, I, FO, S["popm"], 0.95, 0.90)[2]; EE = {k: np.array([ES[k][s_].max() for s_ in es]) for k in (PRIMARY, "erin_p32")}
    def veto_score(c, u, k, q=0.95):   # endoscopies whose ERIN score is below tau (q of the fold's training progressor endoscopies) can never flag
        tau = thr_sens(EE[k][c.tr_e & (c.elab == 1)], q); return np.where(EE[k] >= tau, u, NEG)
    def f_mem_veto(k):
        def b(c): s_, ns = base(c, "gate_al"); return [sc(f"ew h={h}y, ERIN veto", {"h": h}, veto_score(c, memory(c.ST, s_, "ew", h), k), ns) for h in (1, 2, 4)]
        return b
    def f_mem_stack(k):
        def b(c):   # endoscopy-level logistic (gate_al score, ERIN logit) cross-fitted in the training patients (rv9 lr_scores), then EW memory
            s_, ns = base(c, "gate_al"); e1, _ = lr_scores(c, [s_, EE[k]], [EE[k]], "e5stk_" + k); return [sc(f"ew h={h}y over stack", {"h": h}, memory(c.ST, e1, "ew", h), ns) for h in (1, 2, 4)]
        return b
    def f_mem_erin(k):
        def b(c): z = np.zeros(len(EE[k]), int); return [sc(f"ew h={h}y over ERIN alone", {"h": h}, memory(c.ST, EE[k], "ew", h), z) for h in (1, 2, 4)]
        return b
    BL = {"A1:gate_al:ew (round-9 row 2)": FAMS["A1:gate_al:ew"]}
    for k in (PRIMARY, "erin_p32"): BL[f"EW memory + {k} veto q95"] = f_mem_veto(k); BL[f"EW memory over stack(gate, {k})"] = f_mem_stack(k); BL[f"EW memory over {k} alone"] = f_mem_erin(k)
    R = run_families(S, ST, list(BL), INFO, builders=BL); out = {}
    for f, Rf in R.items():
        F_, NS_ = Rf["sel"]["np"]; ch = np.bincount(Rf["chosen"]["np"], minlength=len(Rf["labels"]))
        out[f] = rule_result(f"(m) {f} [nested: half-life and threshold chosen on training patients]", F_, NS_, extra={"chosen_counts": dict(zip(Rf["labels"], ch.tolist())), "infeasible_folds": Rf["infeasible"]["np"]})
    rf = list(R)[0]; _, rob, rbs = outcomes(*R[rf]["sel"]["np"])
    for f in list(R)[1:]:
        _, ob, bs = outcomes(*R[f]["sel"]["np"]); out[f]["delta_vs_memory_rule"] = {k: dstat(ob[k] - rob[k], bs[k] - rbs[k]) for k in MET}
    res["MEM"] = out; dump(res, "e5_mem")
