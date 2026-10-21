"""Exploration E3 (docs/paper_explore_E3.md): re-score R49 and P1 with a new L-IMG. The module bodies of rv8b_step3.py (which runs rv8_landscape.py) are
executed unchanged up to their task dispatch (setup, thr_arr, family, r49_calls, patient_arrays, pdraws, parts, met, tmatch, rules, dstat, ...); only the
loader is replaced so that L-IMG (outer predictions and inner out-of-fold predictions) comes from an E3 refit, and L-LATE = (L-CNV + new L-IMG) / 2.
The copy-number arm, folds, cut-off rules, bootstrap draws and patient outcomes are round 8's.
VAR (one task per variant):
  <name>                  refit killcoyne_mm/explore_E3/refit/<name>/ (ex_e3_refit.R or ex_e3_tilepool.py)
  stored                  the stored L-IMG (check: R49 31.1 / 31.8, P1 90% 31.1 / 36.0)
  ens:<a>+<b>+...         mean of the variants' probabilities (outer and inner alike)
  alt:<name>              endoscopy score = the model applied to the endoscopy-pooled embedding (pred_alt); top-H&E sample still by the sample model
  mlogit|medlogit|mprob|top2logit:<V>   endoscopy score = mean / median sample logit, mean probability, mean of the top-2 logits of the endoscopy's
                          samples, applied to the predictions of variant V (sample model kept for L-LATE and the top-H&E pick)
  sel:<a>+<b>+...         per (repeat, outer fold) the candidate with the best fold-stratified inner out-of-fold AUROC on the fold's training patients
                          (pre-event samples, inner folds as strata); its outer and inner predictions are used for that fold
TASK = score (VAR) | merge. Aggregates (no identifiers) -> results/paper_final/explore_E3/<file>.json. EXE3_NB=k: smoke (k draws, written aside)."""
import os, sys, json, glob, re, numpy as np, pandas as pd
TH = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; PP = TH + "/scripts/paper_plan"; sys.path.insert(0, PP)
ETASK = os.environ["TASK"]; os.environ["TASK"] = "exe3"
_f = __file__; __file__ = PP + "/rv8b_step3.py"; _s = open(__file__).read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], __file__, "exec")); __file__ = _f
E3 = M + "/explore_E3"; RD = E3 + "/refit"; OUTE = T + "/results/paper_final/explore_E3"
if os.environ.get("EXE3_NB"): NB = int(os.environ["EXE3_NB"]); TC.NB = NB; OUTE = E3 + "/smoke_json"
os.makedirs(OUTE, exist_ok=True); load0 = load; O0, I0, FO = load0("their")
fname = lambda v: re.sub(r"[^A-Za-z0-9_.-]", "_", v.replace("+", "__").replace(":", "--"))
def jdump(o, name): json.dump(o, open(f"{OUTE}/{name}.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)); print("EXE3 DONE", name, flush=True)
# ------------------------------------------------------------------ readers
def read_refit(name):
    """outer (n x 10) pred and pred_alt, inner {(rep, fold): Series} pred, pred_alt and inner fold (positions in set C)"""
    fo = sorted(glob.glob(f"{RD}/{name}/outer_rep_*.csv")); fi = sorted(glob.glob(f"{RD}/{name}/inner_rep_*.csv")); assert len(fo) == 10 and len(fi) == 10, (name, len(fo), len(fi))
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); po = lambda c: D.pivot_table(index="Sample", columns="rep", values=c, dropna=False).reindex(ids)[REPS].values
    Of = po("fold").astype(int); assert (Of == FO).all(), "folds differ from the stored folds"
    Di = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fi]); assert Di.groupby(["rep", "fold"]).ngroups == 100
    I, IA, IF = {}, {}, {}
    for (r, k), d in Di.groupby(["rep", "fold"]):
        pos = ix[d.Sample].values; I[(r, k)] = pd.Series(d.pred.values, index=pos); IA[(r, k)] = pd.Series(d.pred_alt.values, index=pos); IF[(r, k)] = pd.Series(d.inner_fold.values, index=pos)
    return {"O": po("pred"), "OA": po("pred_alt") if D.pred_alt.notna().any() else None, "I": I, "IA": IA if Di.pred_alt.notna().any() else None, "IF": IF}
lg_ = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda z: 1 / (1 + np.exp(-z))
ENDOPP = np.array([f"{e}|{int(p)}" for e, p in zip(ENDO, PRE)])
def wauc(x, l, f, w=None):
    """fold-stratified AUROC within one repeat (fsa convention: concordant pairs summed over folds); w = integer weights"""
    num = den = 0.0
    for k in np.unique(f):
        m = (f == k) & np.isfinite(x); xp, xn = x[m & (l == 1)], x[m & (l == 0)]
        if len(xp) == 0 or len(xn) == 0: continue
        wp = np.ones(len(xp)) if w is None else w[m & (l == 1)]; wn = np.ones(len(xn)) if w is None else w[m & (l == 0)]
        W = wp[:, None] * wn[None, :]; num += (((xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])) * W).sum(); den += W.sum()
    return num / den if den else np.nan
def inner_auc(sr, ifo, rows):
    """fold-stratified AUROC of inner out-of-fold predictions over the given training rows, inner folds as strata"""
    v = sr.reindex(rows).values; f = ifo.reindex(rows).values; return wauc(v, y[rows], f)
AGG = {"mlogit": np.mean, "medlogit": np.median, "mprob": lambda z, a: lg_(np.mean(sg(z), a)),
       "top2logit": lambda z, a: np.sort(z, axis=a)[-2:].mean(a) if np.shape(z)[a] >= 2 else np.max(z, a)}   # endoscopy aggregations of sample logits (current rule = max)
def variant(V):
    """-> (O_img n x 10, I_img dict, O_pick or None, I_pick or None, info). O_pick: the sample model (L-LATE and top-H&E pick) when L-IMG is an endoscopy score"""
    if V == "stored": return O0["L-IMG"], I0["L-IMG"], None, None, {}
    if V.startswith("ens:"):
        L = [read_refit(a) for a in V[4:].split("+")]; Oi = np.mean([x["O"] for x in L], 0); Ii = {k: pd.concat([x["I"][k] for x in L], axis=1).mean(1) for k in L[0]["I"]}
        return Oi, Ii, None, None, {"members": V[4:].split("+")}
    if V.startswith("alt:"):
        R = read_refit(V[4:]); assert R["OA"] is not None; return R["OA"], R["IA"], R["O"], R["I"], {}
    if V.split(":")[0] in AGG:
        how, nm = V.split(":", 1); Oo, Io = (variant(nm)[:2]); f = AGG[how]; Oa = np.full((n, 10), np.nan)
        for e in np.unique(ENDOPP): m = ENDOPP == e; Oa[m] = sg(f(lg_(Oo[m]), 0))[None, :]
        Ia = {}
        for k, sr in Io.items():
            g = pd.Series(lg_(sr.values), index=sr.index).groupby(ENDOPP[sr.index.values]).transform(lambda v: f(v.values, 0)); Ia[k] = pd.Series(sg(g.values), index=sr.index)
        return Oa, Ia, Oo, Io, {"endoscopy_aggregation": how}
    if V.startswith("sel:"):
        C = V[4:].split("+"); L = {c: read_refit(c) for c in C}; Oi = np.full((n, 10), np.nan); Ii = {}; pick = {c: 0 for c in C}
        for j, r in enumerate(REPS):
            for k in range(1, 11):
                tri = L[C[0]]["I"][(r, k)].index.values; tri = tri[PRE[tri]]; a = {c: inner_auc(L[c]["I"][(r, k)], L[c]["IF"][(r, k)], tri) for c in C}
                b = max(C, key=lambda c: (a[c], -C.index(c))); pick[b] += 1; te = FO[:, j] == k; Oi[te, j] = L[b]["O"][te, j]; Ii[(r, k)] = L[b]["I"][(r, k)]
        return Oi, Ii, None, None, {"candidates": C, "times_chosen_of_100_folds": pick}
    R = read_refit(V); return R["O"], R["I"], None, None, {}
def make_S(Oi, Ii, Op=None, Ip=None):
    """round-8 setup() with L-IMG replaced; L-LATE from the sample model (Op / Ip when L-IMG is an endoscopy-level score)"""
    O = dict(O0); I = dict(I0); Op = Oi if Op is None else Op; Ip = Ii if Ip is None else Ip
    O["L-IMG"] = Oi; O["L-LATE"] = (O["L-CNV"] + Op) / 2; I["L-IMG"] = Ii; I["L-LATE"] = {k: (I["L-CNV"][k] + Ip[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    g = globals(); g["load"] = lambda src: (O, I, FO)
    try: S = setup()
    finally: g["load"] = load0
    S["PICK_IMG"] = Op; return S
def r49_pick(S):
    """R49 exactly as r49_calls (rv7_round7.py), except that the sequenced (top-H&E) sample is chosen by the sample model when L-IMG is an endoscopy score"""
    _, _, info = TC.triage(S["O"], S["I"], S["FO"], S["popm"], 0.95, 0.90)
    if S["PICK_IMG"] is None: F, NS, _ = r49_calls(S["O"], info, S["FO"], S["es"]); return F, NS, info
    TL, CSt, TS = cutoff_arrays(info, S["FO"]); img = S["O"]["L-IMG"]; late = S["O"]["L-LATE"]; pk = S["PICK_IMG"]; E = len(S["es"]); GC = np.zeros((E, 10), bool); NS = np.zeros((E, 10), int)
    for e, s in enumerate(S["es"]):
        for j in range(10):
            tl, cs, ts = TL[s[0], j], CSt[s[0], j], TS[s[0], j]; sc = img[s, j].max(); g = (2 if sc > cs else 0) if tl > cs else (0 if sc < tl else (2 if sc > cs else 1))
            if g == 1: top = s[np.argmax(pk[s, j])]; GC[e, j] = late[top, j] >= ts; NS[e, j] = 1
            else: GC[e, j] = g == 2
    return GC, NS, info
def rule_set(S):
    X = S["OUTS"]["img"]; Z = np.zeros((S["E"], 10), int); R = {f"P1 {int(q * 100)}%": (X >= thr_arr(S, "img", q)[0], Z) for q in Q3}
    F, NS, info = r49_pick(S); R["R49"] = (F, NS); return R, info
# ------------------------------------------------------------------ AUROC (sample, endoscopy, patient), fold-stratified, repeat mean
def aurocs(S, w_p=None):
    rows = S["rows"]; sp = np.searchsorted(S["up"], pat[rows]); ws = None if w_p is None else w_p[sp]; we = None if w_p is None else w_p[S["pidx"]]
    Oi = S["O"]["L-IMG"]; X = S["OUTS"]["img"]; ol = S["O"]["L-LATE"]
    if "PX" not in S:
        S["PX"] = np.stack([np.array([X[S["pidx"] == i, j].max() for i in range(S["P_"])]) for j in range(10)], 1); S["PF"] = np.stack([np.array([S["EF"][S["pidx"] == i, j][0] for i in range(S["P_"])]) for j in range(10)], 1)
    PX, PF = S["PX"], S["PF"]
    o = {"sample": np.nanmean([wauc(Oi[rows, j], y[rows], FO[rows, j], ws) for j in range(10)]),
         "sample_L_LATE": np.nanmean([wauc(ol[rows, j], y[rows], FO[rows, j], ws) for j in range(10)]),
         "endoscopy": np.nanmean([wauc(X[:, j], S["elab"], S["EF"][:, j], we) for j in range(10)]),
         "patient": np.nanmean([wauc(PX[:, j], S["plab"], PF[:, j], w_p) for j in range(10)])}
    return o
# ------------------------------------------------------------------ one variant
if ETASK == "score":
    V = os.environ["VAR"]; Oi, Ii, Op, Ip, vinfo = variant(V); Sv = make_S(Oi, Ii, Op, Ip); S0 = make_S(O0["L-IMG"], I0["L-IMG"])
    D = pdraws(S0); nP = int((S0["plab"] == 1).sum()); nN = int((S0["plab"] == 0).sum()); one = np.ones(S0["P_"]); Pe = S0["elab"] == 1; Z = np.zeros((S0["E"], 10), int)
    assert (Sv["plab"] == S0["plab"]).all() and Sv["E"] == S0["E"]
    R0, info0 = rule_set(S0); RV, infov = rule_set(Sv)
    A0 = {k: patient_arrays(S0, F, NS) for k, (F, NS) in R0.items()}; AV = {k: patient_arrays(Sv, F, NS) for k, (F, NS) in RV.items()}
    MET = {"progressor_patients_caught": lambda p: nP * met(p, "caught_share"), "non_progressor_patients_flagged": lambda p: nN * met(p, "np_flagged_share"),
           "endoscopy_sensitivity": lambda p: met(p, "endo_sens"), "endoscopy_specificity": lambda p: met(p, "endo_spec"), "samples_sequenced_per_patient": lambda p: met(p, "seq_per_patient")}
    Xv = Sv["OUTS"]["img"]
    def p1_matched(m, sens):
        w = m[Sv["pidx"]].astype(int); F = np.zeros((Sv["E"], 10), bool)
        for j in range(10): F[:, j] = Xv[:, j] >= tmatch(Xv[Pe, j], sens[j], w[Pe])
        return F
    def contrast(new, ref, matched=False):
        """new rule of the variant against a reference rule of the current model; matched: the variant's P1 threshold re-matched per draw (and per repeat) to the reference's realised endoscopy sensitivity"""
        def one_draw(m):
            pr = parts(S0, A0[ref], m)
            pn = parts(Sv, patient_arrays(Sv, p1_matched(m, per_rep(pr, "endo_sens")[0]), Z), m) if matched else parts(Sv, AV[new], m)
            return {k: (f(pn)[0], f(pr)[0]) for k, f in MET.items()}
        ob = one_draw(one); bd = [one_draw(m) for m in D]
        return {"new": {k: r3(v[0]) for k, v in ob.items()}, "ref": {k: r3(v[1]) for k, v in ob.items()},
                "delta_new_minus_ref": {k: dstat(ob[k][0] - ob[k][1], np.array([b[k][0] - b[k][1] for b in bd])) for k in MET}}
    C = {"R49_new_vs_R49": contrast("R49", "R49"), "P1_90_new_vs_P1_90": contrast("P1 90%", "P1 90%"), "P1_90_new_vs_R49": contrast("P1 90%", "R49"),
         "R49_new_vs_P1_90": contrast("R49", "P1 90%"), "P1_new_matched_to_R49": contrast(None, "R49", True), "P1_new_matched_to_P1_90": contrast(None, "P1 90%", True)}
    cells = {}
    for k, A in AV.items():
        p = parts(Sv, A, one); cells[k] = {m: r2(f(p)[0]) if "patients" in m else r3(f(p)[0]) for m, f in MET.items()}
    # AUROC with paired patient-bootstrap delta against the current L-IMG
    a0 = aurocs(S0); av = aurocs(Sv); ad = {k: [] for k in av}
    for m in D:
        b0 = aurocs(S0, m.astype(int)); bv = aurocs(Sv, m.astype(int))
        for k in av: ad[k].append(bv[k] - b0[k])
    AU = {k: {"new": r3(av[k]), "current": r3(a0[k]), "delta": dstat(av[k] - a0[k], np.array(ad[k]))} for k in av}
    # apparent (optimistic): P1 threshold set on the evaluated endoscopies themselves to reach 90% endoscopy sensitivity, per repeat
    def apparent(S):
        X = S["OUTS"]["img"]; F = np.stack([X[:, j] >= thr_sens(X[Pe, j], 0.90) for j in range(10)], 1); p = parts(S, patient_arrays(S, F, Z), one); return {m: r3(f(p)[0]) for m, f in MET.items()}
    chk = {}
    if V == "stored":
        obs = {k: [cells[k]["progressor_patients_caught"], cells[k]["non_progressor_patients_flagged"]] for k in ("R49", "P1 90%")}
        chk = {"got": obs, "expected_round8": {"R49": [31.1, 31.8], "P1 90%": [31.1, 36.0]}, "reproduced": bool(abs(obs["R49"][0] - 31.1) < 0.051 and abs(obs["R49"][1] - 31.8) < 0.051 and abs(obs["P1 90%"][0] - 31.1) < 0.051 and abs(obs["P1 90%"][1] - 36.0) < 0.051)}
    if V == "base":
        I_ = I0["L-IMG"]; chk = {"outer_max_abs_diff_vs_stored": float(np.nanmax(np.abs(Oi - O0["L-IMG"]))), "inner_max_abs_diff_vs_stored": float(max(np.nanmax(np.abs(Ii[k].reindex(I_[k].index).values - I_[k].values)) for k in I_))}
    res = {"variant": V, "info": vinfo, "cells": cells, "auroc": AU, "contrasts": C, "apparent_optimistic_P1_at_90_eval_endoscopy_sensitivity": {"new": apparent(Sv), "current": apparent(S0)},
           "r49_info": {"folds_no_middle_band": int(infov["no_mid"]), "t_seq_fallback": int(infov["fallback"])}, "check": chk, "draws": int(len(D)),
           "n_progressor_patients": nP, "n_non_progressor_patients": nN,
           "note": "new = the variant's L-IMG (L-LATE = (L-CNV + new L-IMG) / 2); ref/current = the stored L-IMG; single model = repeat mean; paired patient bootstrap (rv8 pdraws, RandomState(0)); thresholds from inner out-of-fold predictions on training patients; matched = the variant's P1 threshold set per draw and repeat to the reference rule's realised endoscopy sensitivity"}
    jdump(res, "v_" + fname(V))
# ------------------------------------------------------------------ diag: why the endoscopy aggregation matters (samples per endoscopy)
elif ETASK == "diag":
    out = {}
    for V in ("stored", "mlogit:stored"):
        Oi, Ii, Op, Ip, _ = variant(V); S = make_S(Oi, Ii, Op, Ip); X = S["OUTS"]["img"]; TH, _ = thr_arr(S, "img", 0.90); F = X >= TH
        sz = np.minimum(S["nsz"], 4); o = {}
        for k in (1, 2, 3, 4):
            m = sz == k; nPe = int((m & (S["elab"] == 1)).sum()); nNe = int((m & (S["elab"] == 0)).sum())
            o[f"{k}{'+' if k == 4 else ''} samples"] = {"progressor_endoscopies": m5(nPe), "non_progressor_endoscopies": m5(nNe),
                "endoscopy_sensitivity_P1_90": r3(F[m & (S["elab"] == 1)].mean()) if nPe >= 5 else None, "endoscopy_specificity_P1_90": r3(1 - F[m & (S["elab"] == 0)].mean()) if nNe >= 5 else None}
        out[V] = o
    S0 = make_S(O0["L-IMG"], I0["L-IMG"]); out["mean_samples_per_endoscopy"] = {"progressor": r3(S0["nsz"][S0["elab"] == 1].mean()), "non_progressor": r3(S0["nsz"][S0["elab"] == 0].mean())}
    jdump({"diag": out, "note": "P1 at 90% training endoscopy sensitivity, repeat-pooled shares; endoscopy strata by number of pre-event samples; counts < 5 shown as <5"}, "diag_endoscopy_size")
# ------------------------------------------------------------------ merge: ranked table
elif ETASK == "merge":
    L = [json.load(open(f)) for f in sorted(glob.glob(f"{OUTE}/v_*.json"))]; rows = []
    for r in L:
        c = r["contrasts"]; g = lambda nm, k: c[nm]["delta_new_minus_ref"][k]
        rows.append({"variant": r["variant"], "auroc_sample": r["auroc"]["sample"]["new"], "d_auroc_sample": r["auroc"]["sample"]["delta"]["delta"], "d_auroc_sample_ci": r["auroc"]["sample"]["delta"]["ci95"],
                     "auroc_endoscopy": r["auroc"]["endoscopy"]["new"], "d_auroc_endoscopy": r["auroc"]["endoscopy"]["delta"]["delta"], "d_auroc_endoscopy_ci": r["auroc"]["endoscopy"]["delta"]["ci95"],
                     "auroc_patient": r["auroc"]["patient"]["new"], "d_auroc_patient": r["auroc"]["patient"]["delta"]["delta"],
                     "R49_caught": r["cells"]["R49"]["progressor_patients_caught"], "R49_np_flagged": r["cells"]["R49"]["non_progressor_patients_flagged"], "R49_seq": r["cells"]["R49"]["samples_sequenced_per_patient"],
                     "R49_spec": r["cells"]["R49"]["endoscopy_specificity"], "R49_sens": r["cells"]["R49"]["endoscopy_sensitivity"],
                     "d_R49_np": g("R49_new_vs_R49", "non_progressor_patients_flagged"), "d_R49_caught": g("R49_new_vs_R49", "progressor_patients_caught"), "d_R49_spec": g("R49_new_vs_R49", "endoscopy_specificity"),
                     "P1_caught": r["cells"]["P1 90%"]["progressor_patients_caught"], "P1_np_flagged": r["cells"]["P1 90%"]["non_progressor_patients_flagged"],
                     "d_P1_np": g("P1_90_new_vs_P1_90", "non_progressor_patients_flagged"), "d_P1_caught": g("P1_90_new_vs_P1_90", "progressor_patients_caught"),
                     "d_P1m_R49_np": g("P1_new_matched_to_R49", "non_progressor_patients_flagged"), "d_P1m_R49_spec": g("P1_new_matched_to_R49", "endoscopy_specificity"),
                     "d_P1m_P1_np": g("P1_new_matched_to_P1_90", "non_progressor_patients_flagged"), "d_P1m_P1_spec": g("P1_new_matched_to_P1_90", "endoscopy_specificity"),
                     "apparent_P1_np": r["apparent_optimistic_P1_at_90_eval_endoscopy_sensitivity"]["new"]["non_progressor_patients_flagged"], "info": r["info"], "check": r["check"]})
    rows.sort(key=lambda x: (x["d_R49_np"]["delta"], -x["d_R49_caught"]["delta"]))
    jdump({"table": rows, "n_variants": len(rows), "note": "sorted by delta non-progressor patients flagged under R49 (new - current), then delta caught"}, "merge")
