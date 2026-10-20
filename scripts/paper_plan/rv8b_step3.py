"""Review round 8b, step 3 (docs/paper_review_round8b_step3.md @ c39fd82): uncertainty and transportability of the fixed main rule (R49 = P6 t_low 95% /
t_high 90%) against H&E alone (P1). Stored patient-grouped outer and inner out-of-fold predictions are re-scored; nothing is refitted, no ACE-B file is read,
frozen bundles are not touched. Policy definitions, thresholds, outcomes and helpers are those of rv8_landscape.py: its module body up to the task dispatch
is executed here unchanged (setup, thr_arr, family, r49_calls, p8_calls, patient_arrays, pdraws, cutoff_arrays, dstat, ci2, ...).
Per-draw files only under killcoyne_mm/review_round8b/ on the cluster; aggregates (no identifiers, no dates) -> results/paper_final/review_round8b/<TASK>.json.
TASK = s1 | s2_<P1..P10> (per-family draws) | s2 (merge) | s3 | s4.  RV8B_NB=k: smoke mode (k bootstrap draws, 20k S4 draws, aggregates written aside)."""
import os, sys, json, numpy as np
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
assert not os.environ.get("RV8_NB"), "the round-8 smoke switch must not be set"
_s = open(HERE + "/rv8_landscape.py").read(); exec(compile(_s[:_s.index('res = {"task": TASK}')], HERE + "/rv8_landscape.py", "exec"))
PRESPEC = "c39fd82"; OUT8 = OUT; OUT = T + "/results/paper_final/review_round8b"; R8B = M + "/review_round8b"; NS4 = 10000; NP4, NN4, K4 = 15, 67, 2
if os.environ.get("RV8B_NB"):   # smoke mode: fewer draws, aggregates written aside (never to results/)
    NB = int(os.environ["RV8B_NB"]); TC.NB = NB; NS4 = 20 * NB; OUT = R8B + "/smoke_json"; R8B = R8B + "/smoke"
for d in (OUT, R8B): os.makedirs(d, exist_ok=True)
FAMS = [f"P{i}" for i in range(1, 11)]; COSTS = (85.0, 150.0, 250.0, 400.0); FLAGC = (535.0, 1605.0); WP10 = 0.10
def dump(res, name):
    res["prespec"] = PRESPEC; json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV8B DONE", name, flush=True)
def tmatch(x, q, w=None):
    """largest threshold reaching >= q sensitivity on scores x (integer weights w); q <= 0 -> nothing flagged"""
    return np.inf if not np.isfinite(q) or q <= 1e-12 else thr_sens(x, q - 1e-9, w)
def parts(S, A, Mb):
    """per multiplicity row (B, P_) and repeat: the weighted sums behind lstats (rv8_landscape.py) -> dict of (B, 10)"""
    Mb = np.atleast_2d(Mb) * A["pinc"][None, :]; isP = S["plab"] == 1; sP = Mb * isP; sN = Mb * ~isP
    return {"nP": sP.sum(1)[:, None] + 0 * A["FL"][0], "nN": sN.sum(1)[:, None] + 0 * A["FL"][0], "caught": sP @ A["FL"], "npfl": sN @ A["FL"], "NSP": sP @ A["NS"], "NSN": sN @ A["NS"],
            "NFP": sP @ A["NF"], "NFN": sN @ A["NF"], "NEP": sP @ A["NE"][:, None] + 0 * A["FL"][0], "NEN": sN @ A["NE"][:, None] + 0 * A["FL"][0], "PYN": (sN @ S["PY"])[:, None] + 0 * A["FL"][0]}
def per_rep(p, k, cs=COST_SEQ, cf=COST_FLAG, unit="patient", wp=None):
    """per-repeat metric (B, 10) from parts; cost units: per patient (cohort weights, or progressor patients reweighted to share wp) or per non-progressor patient-year"""
    with np.errstate(invalid="ignore", divide="ignore"):
        if k == "caught_share": return p["caught"] / p["nP"]
        if k == "np_flagged_share": return p["npfl"] / p["nN"]
        if k == "flags_per_np_py": return p["NFN"] / p["PYN"]
        if k == "seq_per_patient": return (p["NSP"] + p["NSN"]) / (p["nP"] + p["nN"])
        if k == "endo_sens": return p["NFP"] / p["NEP"]
        if k == "endo_spec": return 1 - p["NFN"] / p["NEN"]
        if k == "cost":
            if unit == "np_py": return (cs * p["NSN"] + cf * p["NFN"]) / p["PYN"]
            if wp is None: return (cs * (p["NSP"] + p["NSN"]) + cf * (p["NFP"] + p["NFN"])) / (p["nP"] + p["nN"])
            return wp * (cs * p["NSP"] + cf * p["NFP"]) / p["nP"] + (1 - wp) * (cs * p["NSN"] + cf * p["NFN"]) / p["nN"]
    raise KeyError(k)
def met(p, k, **kw): return np.nanmean(per_rep(p, k, **kw), 1)   # single model: repeat mean -> (B,)
def rules(S):
    """the S1 / S3 rules, each (F, NS) per endoscopy x repeat, as rv8_landscape.py defines the cells"""
    X = S["OUTS"]; Z = np.zeros((S["E"], 10), int); R = {}
    for q in Q3: R[f"P1 {int(q * 100)}%"] = (X["img"] >= thr_arr(S, "img", q)[0], Z)
    _, _, info = TC.triage(S["O"], S["I"], S["FO"], S["popm"], 0.95, 0.90); F, NS, _ = r49_calls(S["O"], info, S["FO"], S["es"]); R["R49"] = (F, NS)
    R["P3 90%"] = (X["late"] >= thr_arr(S, "late", 0.90)[0], np.repeat(S["nsz"][:, None], 10, 1))
    F, NS, _ = p8_calls(S, 0.90, 0.80); R["P8 t_b 90% / t_mid 80%"] = (F, NS)
    g = family(S, "P10")[0]; assert g[1] == {"rule": "grade"}; R["grade ID/LGD"] = (g[2], g[3])
    return R
res = {"task": TASK}
# ================================================================== S1: paired contrasts against H&E alone
if TASK == "s1":
    S = setup(); D = pdraws(S); RU = rules(S); Pe = S["elab"] == 1; X = S["OUTS"]["img"]; Z = np.zeros((S["E"], 10), int); one = np.ones(S["P_"])
    nP = int((S["plab"] == 1).sum()); nN = int((S["plab"] == 0).sum()); AR = {k: patient_arrays(S, F, NS) for k, (F, NS) in RU.items()}
    MET = {"endoscopy_specificity": lambda p: met(p, "endo_spec"), "non_progressor_patients_flagged": lambda p: nN * met(p, "np_flagged_share"),
           "flagged_np_endoscopies_per_np_patient_year": lambda p: met(p, "flags_per_np_py"), "progressor_patients_caught": lambda p: nP * met(p, "caught_share"),
           "samples_sequenced_per_patient": lambda p: met(p, "seq_per_patient"), "endoscopy_sensitivity": lambda p: met(p, "endo_sens")}
    def p1_matched(m, sens):
        """P1 (max L-IMG) with the per-repeat threshold reaching the rule's realised endoscopy sensitivity on the evaluated (multiplicity-weighted) endoscopies"""
        w = m[S["pidx"]].astype(int); F = np.zeros((S["E"], 10), bool)
        for j in range(10): F[:, j] = X[:, j] >= tmatch(X[Pe, j], sens[j], w[Pe])
        return F
    def contrast(rule, matched):
        A = AR[rule]; A1 = AR["P1 90%"]
        def one_draw(m):
            pr = parts(S, A, m)
            if matched: p1 = parts(S, patient_arrays(S, p1_matched(m, per_rep(pr, "endo_sens")[0]), Z), m)
            else: p1 = parts(S, A1, m)
            return {k: (f(pr)[0], f(p1)[0]) for k, f in MET.items()}
        ob = one_draw(one); bd = [one_draw(m) for m in D]
        o = {"rule": {k: r3(v[0]) for k, v in ob.items()}, "P1": {k: r3(v[1]) for k, v in ob.items()},
             "delta_rule_minus_P1": {k: dstat(ob[k][0] - ob[k][1], np.array([b[k][0] - b[k][1] for b in bd])) for k in MET}}
        if matched:
            gap = np.array([b["endoscopy_sensitivity"][1] - b["endoscopy_sensitivity"][0] for b in bd])
            o["matching"] = {"P1_minus_rule_endoscopy_sensitivity_observed": r3(ob["endoscopy_sensitivity"][1] - ob["endoscopy_sensitivity"][0]),
                             "P1_minus_rule_endoscopy_sensitivity_draws_median_max": [r3(np.median(gap)), r3(np.max(gap))]}
        return o
    C = {}
    for nm, rule in (("a_R49_vs_P1", "R49"), ("b_P3_90_vs_P1", "P3 90%")):
        C[nm] = {"i_evaluation_matched": contrast(rule, True), "ii_P1_at_90": contrast(rule, False)}
    up = C["a_R49_vs_P1"]["i_evaluation_matched"]["delta_rule_minus_P1"]["non_progressor_patients_flagged"]["ci95"][1]
    st = {"rule": "manuscript states 'sequencing reduces false flags compared with the slide alone' only if the upper 95% limit of the (a)(i) difference in non-progressor patients flagged is below 0",
          "upper_95_limit": up, "met": bool(up is not None and up < 0)}
    st["statement"] = "Sequencing reduces false flags compared with the slide alone." if st["met"] else "Whether sequencing reduces false flags compared with the slide alone: the reduction was not demonstrated."
    obs = {k: {"caught": r2(nP * met(parts(S, AR[k], one), "caught_share")[0]), "np_flagged": r2(nN * met(parts(S, AR[k], one), "np_flagged_share")[0])} for k in ("R49", "P1 90%", "P3 90%")}
    exp = {"R49": [31.1, 31.8], "P1 90%": [31.1, 36.0], "P3 90%": [31.2, 31.8]}
    chk = {"got": obs, "expected_round8": exp, "reproduced": bool(all(abs(obs[k]["caught"] - e[0]) < 0.051 and abs(obs[k]["np_flagged"] - e[1]) < 0.051 for k, e in exp.items()))}
    res["S1"] = {"contrasts": C, "statement": st, "check_round8": chk, "draws": int(len(D)), "n_progressor_patients": nP, "n_non_progressor_patients": nN,
                 "note": "delta = rule - P1; single model (per-repeat values averaged over the 10 repeats); patient bootstrap (rv8 pdraws, RandomState(0), one-class draws rejected), thresholds and calls fixed "
                         "except P1 in (i), whose per-repeat threshold is re-matched in every draw to the rule's realised endoscopy sensitivity on that draw's endoscopies; counts = shares x 32 or x 43"}
    dump(res, "s1")
# ================================================================== S2: per-family draws of every round-8 cell (row-level-free, kept on the cluster)
elif TASK.startswith("s2_"):
    fam = TASK[3:]; S = setup(); D = pdraws(S); one = np.ones(S["P_"]); L = []; ob = []; bd = []
    for label, params, F, NS, em, ms, extra in family(S, fam):
        A = patient_arrays(S, F, NS, em); nP = int(((S["plab"] == 1) & A["pinc"]).sum()); nN = int(((S["plab"] == 0) & A["pinc"]).sum()); po = parts(S, A, one); pb = parts(S, A, D)
        f = lambda p: np.stack([nP * met(p, "caught_share"), nN * met(p, "np_flagged_share"), met(p, "seq_per_patient"), met(p, "cost")])
        L.append({"family": fam, "label": label, "subset": params.get("subset") is not None}); ob.append(f(po)[:, 0]); bd.append(f(pb))
    np.savez(f"{R8B}/s2_{fam}.npz", meta=json.dumps(L), obs=np.array(ob), draws=np.array(bd)); print("RV8B DONE", TASK, flush=True)
# ================================================================== S2 merge: frontier stability
elif TASK == "s2":
    META = []; OB = []; BD = []; missing = []
    for f in FAMS:
        p = f"{R8B}/s2_{f}.npz"
        if not os.path.exists(p): missing.append(f); continue
        z = np.load(p); META += json.loads(str(z["meta"])); OB.append(z["obs"]); BD.append(z["draws"])
    OB = np.concatenate(OB); BD = np.concatenate(BD); keep = np.array([not m["subset"] for m in META]); PT = np.where(keep)[0]
    RND = np.array([2, 2, 3, 2])[:, None]   # round-8 rounding of caught, NP flagged, sequenced / patient, cost (cell_result)
    def frontier(V, keys):
        """V (4, C) or (4, C, B) rounded; keys (index, sign): True where not dominated (>= on all, > on one) by another point"""
        V = V[:, :, None] if V.ndim == 2 else V; C = V.shape[1]; ge = np.ones((C, C, V.shape[2]), bool); gt = np.zeros_like(ge)
        for k, s in keys:
            a = s * V[k][:, None, :]; b = s * V[k][None, :, :]; ge &= b >= a; gt |= b > a   # [a, b]: b at least as good as a
        dm = ge & gt; dm[np.arange(C), np.arange(C)] = False; return ~dm.any(1), dm
    K3 = ((0, 1), (1, -1), (2, -1)); KC = ((0, 1), (3, -1))
    Vo = np.stack([np.round(OB[PT, i], int(RND[i, 0])) for i in range(4)]); Vb = np.stack([np.round(BD[PT, i], int(RND[i, 0])) for i in range(4)])
    f3o, _ = frontier(Vo, K3); fco, _ = frontier(Vo, KC); f3b, d3 = frontier(Vb, K3); fcb, dc = frontier(Vb, KC); f3o = f3o[:, 0]; fco = fco[:, 0]
    l1 = J(f"{OUT8}/l1.json"); cells = []
    for q, c in enumerate(PT):
        m = META[c]; o = {"family": m["family"], "label": m["label"], "observed": {"caught": r2(OB[c, 0]), "np_flagged": r2(OB[c, 1]), "seq_per_patient": r3(OB[c, 2]), "cost": r2(OB[c, 3])},
                          "observed_on_pareto": bool(f3o[q]), "observed_on_cost_frontier": bool(fco[q]), "share_draws_on_pareto": r3(f3b[q].mean()), "share_draws_on_cost_frontier": r3(fcb[q].mean())}
        cells.append(o)
    def dominated(fam, label):
        q = [i for i, c in enumerate(PT) if META[c]["family"] == fam and META[c]["label"] == label][0]
        top = lambda dm: [{"family": META[PT[b]]["family"], "label": META[PT[b]]["label"], "share_draws_dominating": r3(dm[q, b].mean())} for b in np.argsort(-dm[q].mean(1))[:3] if dm[q, b].mean() > 0]
        return {"share_draws_dominated_pareto": r3(1 - f3b[q].mean()), "share_draws_dominated_cost": r3(1 - fcb[q].mean()), "top_dominators_pareto": top(d3), "top_dominators_cost": top(dc)}
    ref = {(c["family"], c["label"]): (c["pareto_caught_falseflags_sequencing"], c["pareto_caught_cost"]) for c in l1["L1"]["cells"]} if l1 else {}
    mism = [f"{c['family']} {c['label']}" for c in cells if (c["family"], c["label"]) in ref and ref[(c["family"], c["label"])] != (c["observed_on_pareto"], c["observed_on_cost_frontier"])]
    res["S2"] = {"cells": cells, "missing_families": missing, "not_available": missing, "draws": int(BD.shape[2]), "n_points": int(len(PT)),
                 "R49": dominated("P6", "t_low 95% / t_high 90%"), "P1_90": dominated("P1", "90%"),
                 "frontier_size_per_draw": {"pareto_median_iqr": [float(np.median(f3b.sum(0))), float(np.percentile(f3b.sum(0), 25)), float(np.percentile(f3b.sum(0), 75))],
                                            "cost_median_iqr": [float(np.median(fcb.sum(0))), float(np.percentile(fcb.sum(0), 25)), float(np.percentile(fcb.sum(0), 75))]},
                 "check_round8_frontier": {"cells_compared": len([c for c in cells if (c["family"], c["label"]) in ref]), "mismatches": mism, "reproduced": bool(ref) and not mism},
                 "note": "per draw (rv8 pdraws, RandomState(0)): every round-8 cell's caught / NP flagged / samples sequenced per patient / cost per patient (GBP 85 per sample, GBP 535 per flagged endoscopy) recomputed with "
                         "thresholds and calls fixed, rounded as round 8 (counts 2 dp, sequenced 3 dp, cost 2 dp); Pareto = not dominated on caught (higher), NP flagged and sequenced (lower); cost frontier on caught and cost; "
                         "the p53-subset current-care cell excluded as in round 8; dominated = not on that frontier"}
    dump(res, "s2")
# ================================================================== S3: cost sensitivity (descriptive)
elif TASK == "s3":
    S = setup(); D = pdraws(S); RU = rules(S); one = np.ones(S["P_"]); nP = int((S["plab"] == 1).sum()); names = list(RU)
    PO = {}; PB = {}
    for k, (F, NS) in RU.items(): A = patient_arrays(S, F, NS); PO[k] = parts(S, A, one); PB[k] = parts(S, A, D)
    caught = {k: r2(nP * met(PO[k], "caught_share")[0]) for k in names}; G = []
    for cs in COSTS:
        for cf in FLAGC:
            for unit in ("patient", "np_py"):
                for wt in ("cohort", "10% progressors"):
                    kw = {"cs": cs, "cf": cf, "unit": unit, "wp": None if wt == "cohort" else WP10}
                    co = {k: met(PO[k], "cost", **kw)[0] for k in names}; cb = {k: met(PB[k], "cost", **kw) for k in names}
                    fr = [k for k in names if not any((caught[b] >= caught[k]) and (r2(co[b]) <= r2(co[k])) and ((caught[b] > caught[k]) or (r2(co[b]) < r2(co[k]))) for b in names if b != k)]
                    G.append({"seq_price": cs, "flag_cost": cf, "unit": unit, "progressor_weight": wt, "reweighting_applies": unit == "patient",
                              "cost": {k: {"value": r2(co[k]), "ci95": [r2(x) for x in np.nanpercentile(cb[k], [2.5, 97.5])]} for k in names},
                              "cost_frontier": fr, "delta_cost_R49_minus_P1_90": {**dstat(co["R49"] - co["P1 90%"], cb["R49"] - cb["P1 90%"]), "delta": r2(co["R49"] - co["P1 90%"]),
                                                                                    "ci95": [r2(x) for x in np.nanpercentile(cb["R49"] - cb["P1 90%"], [2.5, 97.5])]}})
    l1 = J(f"{OUT8}/l1.json"); lab8 = {"P1 80%": ("P1", "80%"), "P1 90%": ("P1", "90%"), "P1 95%": ("P1", "95%"), "R49": ("P6", "t_low 95% / t_high 90%"), "P3 90%": ("P3", "90%"),
                                       "P8 t_b 90% / t_mid 80%": ("P8", "t_b 90%, t_mid 80%, t_c 90%"), "grade ID/LGD": ("P10", "grade ID/LGD (all pre-event endoscopies)")}
    c8 = {(c["family"], c["label"]): c for c in l1["L1"]["cells"]}; base = G[0]
    got = {k: [caught[k], base["cost"][k]["value"]] for k in names}; exp = {k: [c8[lab8[k]]["progressor_patients_caught"], c8[lab8[k]]["cost_per_patient_gbp"]] for k in names}
    res["S3"] = {"rules": names, "progressor_patients_caught": caught, "grid": G, "draws": int(len(D)),
                 "check_round8_cost": {"got_caught_cost": got, "expected_round8": exp, "reproduced": bool(all(abs(got[k][0] - exp[k][0]) < 0.006 and abs(got[k][1] - exp[k][1]) < 0.006 for k in names))},
                 "note": "cost = sequencing price x samples sequenced + flagged-endoscopy cost x flagged endoscopies; per patient (all 75 at cohort weights, or progressor and non-progressor patient means mixed 10% / 90%) or "
                         "per non-progressor patient-year (costs incurred in non-progressor patients / their patient-years; progressor weighting does not apply, so its two rows repeat); "
                         "cost frontier among these rules on progressor patients caught (higher) and cost (lower), observed single-model means, rounded as round 8; delta = R49 - P1 90%, paired patient bootstrap"}
    dump(res, "s3")
# ================================================================== S4: ACE-B-shaped precision (no ACE-B data)
elif TASK == "s4":
    S = setup(); O = S["O"]; es = S["es"]; plab = S["plab"]; P_ = S["P_"]; img = O["L-IMG"]; late = O["L-LATE"]
    _, _, info = TC.triage(O, S["I"], S["FO"], S["popm"], 0.95, 0.90); TL, CSt, TS = cutoff_arrays(info, S["FO"])
    EP = [np.where(S["pidx"] == i)[0] for i in range(P_)]; Pp = np.where(plab == 1)[0]; Np = np.where(plab == 0)[0]; U = NP4 + NN4; yy = np.r_[np.ones(NP4, bool), np.zeros(NN4, bool)]
    rng = np.random.RandomState(0); R = {k: np.zeros(NS4) for k in ("auroc", "sens", "spec", "dspec", "share_seq", "detected", "pass12_mean", "pass12_repeat_share", "hm_lower", "spec_p1")}
    ar = np.arange(U)[:, None]; rr = np.arange(10)[None, :]
    for d in range(NS4):
        u = np.concatenate([rng.choice(Pp, NP4, replace=False), rng.choice(Np, NN4, replace=True)]); idx = np.zeros((U, K4), int); nsel = 0
        for q, i in enumerate(u):
            s = es[EP[i][rng.randint(len(EP[i]))]]; k = min(K4, len(s)); pick = rng.choice(s, k, replace=False); idx[q] = np.r_[pick, np.repeat(pick[:1], K4 - k)]; nsel += k
        Xs = img[idx]; sc = Xs.max(1); top = idx[ar, Xs.argmax(1)]; s0 = idx[:, 0]; tl, cs, ts = TL[s0], CSt[s0], TS[s0]
        g = np.where(tl > cs, np.where(sc > cs, 2, 0), np.where(sc < tl, 0, np.where(sc > cs, 2, 1))); call = np.where(g == 1, late[top, rr] >= ts, g == 2)
        sens = call[yy].mean(0); spec = 1 - call[~yy].mean(0); det = call[yy].sum(0)
        rk = rankdata(sc, axis=0); auc = (rk[yy].sum(0) - NP4 * (NP4 + 1) / 2) / (NP4 * NN4)
        sp1 = np.array([1 - (sc[~yy, j] >= tmatch(sc[yy, j], sens[j])).mean() for j in range(10)])
        a = auc.mean(); q1 = a / (2 - a); q2 = 2 * a * a / (1 + a); se = np.sqrt((a * (1 - a) + (NP4 - 1) * (q1 - a * a) + (NN4 - 1) * (q2 - a * a)) / (NP4 * NN4))
        R["auroc"][d] = a; R["sens"][d] = sens.mean(); R["spec"][d] = spec.mean(); R["spec_p1"][d] = sp1.mean(); R["dspec"][d] = (spec - sp1).mean(); R["share_seq"][d] = (g == 1).sum(0).mean() / nsel
        R["detected"][d] = det.mean(); R["pass12_mean"][d] = det.mean() >= 12 - 1e-9; R["pass12_repeat_share"][d] = (det >= 12).mean(); R["hm_lower"][d] = a - 1.96 * se
    np.savez(f"{R8B}/s4_draws.npz", **R)
    def dist(v, lo=None, hi=None, nb=40):
        lo = float(np.min(v)) if lo is None else lo; hi = float(np.max(v)) if hi is None else hi; c, e = np.histogram(v, bins=nb, range=(lo, hi))
        return {"median": r3(np.median(v)), "p2_5": r3(np.percentile(v, 2.5)), "p97_5": r3(np.percentile(v, 97.5)), "mean": r3(np.mean(v)), "hist_counts": c.tolist(), "hist_edges": [round(float(x), 4) for x in e]}
    o = {"auroc_patient_max_L_IMG": dist(R["auroc"], 0.4, 1.0, 60), "r49_endoscopy_sensitivity": dist(R["sens"], 0.0, 1.0, 50), "r49_endoscopy_specificity": dist(R["spec"], 0.0, 1.0, 50),
         "p1_specificity_matched": dist(R["spec_p1"], 0.0, 1.0, 50), "delta_specificity_r49_minus_p1_matched": dist(R["dspec"], -0.5, 0.5, 50), "r49_share_samples_sequenced": dist(R["share_seq"], 0.0, 1.0, 50),
         "r49_progressor_patients_detected_of_15": dist(R["detected"], 0.0, 15.0, 60), "auroc_hanley_mcneil_lower_95": dist(R["hm_lower"], 0.2, 1.0, 64)}
    o["share_pass_ge12_of_15_repeat_mean"] = r3(R["pass12_mean"].mean()); o["share_pass_ge12_of_15_per_repeat_mean"] = r3(R["pass12_repeat_share"].mean())
    kmin = int(np.floor(NP4 * np.percentile(R["sens"], 2.5) + 1e-9)); smax = float(np.ceil(100 * np.percentile(R["share_seq"], 97.5) - 1e-9) / 100)
    o["proposed_pass_bounds"] = {
        "primary": {"endpoint": "H&E model transfer: patient-level AUROC (score = max L-IMG), lower 95% bound above 0.70", "bound": 0.70, "expected_auroc_median": r3(np.median(R["auroc"])),
                    "share_draws_lower_bound_above_0_70_hanley_mcneil": r3((R["hm_lower"] > 0.70).mean()), "share_draws_point_auroc_above_0_70": r3((R["auroc"] > 0.70).mean())},
        "secondary_sensitivity": {"endpoint": "R49 endoscopy sensitivity (one endoscopy per progressor patient)", "proposed_bound_progressors_detected_at_least": kmin, "of": NP4,
                                  "rule": "floor(15 x 2.5th percentile of the ACE-B-shaped sensitivity)", "share_draws_passing": r3((R["detected"] >= kmin - 1e-9).mean()),
                                  "share_draws_passing_ge12": r3(R["pass12_mean"].mean())},
        "secondary_share_sequenced": {"endpoint": "R49 share of samples sequenced", "proposed_upper_bound": smax, "rule": "97.5th percentile of the ACE-B-shaped share, rounded up to 0.01",
                                      "share_draws_passing": r3((R["share_seq"] <= smax + 1e-9).mean())}}
    o["design"] = {"draws": NS4, "progressor_patients": NP4, "non_progressor_patients": NN4, "max_samples_per_endoscopy": K4, "cohort_progressor_patients": int(len(Pp)), "cohort_non_progressor_patients": int(len(Np)),
                   "non_progressor_sampling": "with replacement (67 > 43 available)", "progressor_sampling": "without replacement", "rng": "RandomState(0), one stream"}
    o["note"] = ("per draw: 15 progressor patients without replacement and 67 non-progressor patients with replacement (only 43 exist); one pre-event endoscopy per drawn patient at random, at most 2 of its samples at random; "
                 "stored outer predictions and the round-7 R49 fold cut-offs (t_low 95%, c* 90%, t_seq) fixed; per repeat, then averaged over the 10 repeats (single model); AUROC pooled over folds on the draw's 82 patients; "
                 "P1 = max L-IMG over the chosen samples with its per-repeat threshold matched to R49's realised sensitivity in that draw; Hanley-McNeil lower bound from the repeat-mean AUROC (assurance, added for the proposal)")
    res["S4"] = o; dump(res, "s4")
else:
    raise SystemExit(f"unknown TASK {TASK}")
