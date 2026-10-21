"""Exploration E7 (docs/paper_explore_E7.md; rules paper/reviews_round7/decisions/EXPLORE_RULES.md): is the memory / quiet-genome result an artefact?
Exploratory. Rules under test, rebuilt with their own code: A1 = EW memory of the R49-aligned gate score (rv9_falseflags.py family 'A1:gate_al:ew', per-fold 'np'
selection), A12 = A1 + copy-number certificate (family 'A12'), final = A12 + quiet-OAC check (ex_e2_veto.py ps_oac, top-2 samples, cn90 bar; story_aggs_v2.py).
The rv9 nested engine is re-implemented as run_families (criterion 'np' only) so that it can also return the chosen candidate's training-patient flags (needed
for the in-fold check bar) and evaluate held-out patients on modified histories; base reproduces the stored flags exactly (asserted).
Row-level files only under killcoyne_mm/explore_E7/ on the cluster; aggregates (no identifiers, no dates, patient cells < 5 shown as '<5') ->
results/paper_final/explore_E7/<TASK>.json.   TASK = base | shuf_<b> | miss_<pct>_<b> | perm_<free|strat>_<b> | trunc_<b> | merge"""
import os, sys, json, time, glob, numpy as np, pandas as pd
P = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, P); TASK = os.environ["TASK"]
os.environ["TASK"] = "__ex_e7_e2__"; G2 = {"__file__": P + "/ex_e2_veto.py", "__name__": "ex_e2_e7"}
_s = open(P + "/ex_e2_veto.py").read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], P + "/ex_e2_veto.py", "exec"), G2)
os.environ["TASK"] = "__ex_e7_rv9__"; G9 = {"__file__": P + "/rv9_falseflags.py", "__name__": "rv9_e7"}
_s = open(P + "/rv9_falseflags.py").read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], P + "/rv9_falseflags.py", "exec"), G9)
os.environ["TASK"] = TASK
TC = G2["TC"]; lg = G2["lg"]; thr_sens = G2["thr_sens"]; patient_arrays = G2["patient_arrays"]; lstats = G2["lstats"]; dstat = G2["dstat"]; ci2 = G2["ci2"]; m5 = G2["m5"]
r49_calls = G2["r49_calls"]; M = G2["M"]; T = G2["T"]; SM = G2["SM"]; y0 = TC.y.copy()
OUT = T + "/results/paper_final/explore_E7"; XD = M + "/explore_E7"
for d in (OUT, XD): os.makedirs(d, exist_ok=True)
r2 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 2); r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
QS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975); FAM1 = "A1:gate_al:ew"; FAM12 = "A12"
def dump(res, name=None):
    json.dump(res, open(f"{OUT}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("EXE7 DONE", name or TASK, flush=True)
# ------------------------------------------------------------------ certificate respects missing records (only used when ST carries MISS)
_cf0 = G9["cert_flags"]
def _cert_e7(ST, qual, Y): return _cf0(ST, qual & ~ST["MISS"] if "MISS" in ST else qual, Y)
G9["cert_flags"] = _cert_e7
# ------------------------------------------------------------------ nested engine (rv9 run_families, criterion 'np'), with training flags and separate evaluation histories
def engine(S, STs, STe, INFO, fam, trkeep=None):
    mk = G9["make_ctx"]; B = G9["FAMS"][fam]; pat_scores = G9["pat_scores"]; thr_for = G9["thr_for"]; E = S["E"]; elab = S["elab"]
    F = np.zeros((E, 10), bool); NS = np.zeros((E, 10), int); CF = CNS = None; TRF = {}; CH = []
    for j in range(10):
        for k in range(1, 11):
            c = mk(S, STs, j, k, INFO)
            if trkeep is not None: c.tr_e = c.tr_e & trkeep
            te = c.te_e; cands = B(c); cands_e = cands if STe is STs else B(mk(S, STe, j, k, INFO))
            if CF is None: CF = np.zeros((len(cands), E, 10), bool); CNS = np.zeros((len(cands), E, 10), np.int8)
            ts_ = []; fl = []
            for q, x in enumerate(cands):
                assert x["kind"] == "score"
                pm = pat_scores(c, x["u"]); t = thr_for(c, pm); flag = x["u"] >= t
                det = c.patient_det(flag); inc = (S["INC"] @ flag.astype(float)) > 0; npf = int((inc & ~c.isP & c.trP).sum()); nfe = int((flag & c.tr_e & (elab == 0)).sum()); nsm = float(x["ns"][c.tr_e].mean())
                ts_.append((det, npf, nfe, nsm)); fl.append(flag); fe = cands_e[q]["u"] >= t; CF[q][:, j][te] = fe[te]; CNS[q][:, j][te] = cands_e[q]["ns"][te]
            ts_ = np.array(ts_); feas = ts_[:, 0] >= c.D; nC = len(cands)
            pick = (sorted([q for q in range(nC) if feas[q]], key=lambda q: (ts_[q, 1], ts_[q, 2], ts_[q, 3], q)) if feas.any() else sorted(range(nC), key=lambda q: (-ts_[q, 0], ts_[q, 1], ts_[q, 2], q)))[0]
            CH.append(int(pick)); F[:, j][te] = CF[pick][:, j][te]; NS[:, j][te] = CNS[pick][:, j][te]; TRF[(j, k)] = fl[pick]
    return dict(F=F, NS=NS, CF=CF, CNS=CNS, TRF=TRF, chosen=CH)
# ------------------------------------------------------------------ quiet-OAC check (E2 ps_oac, k = 2, threshold = share q of training progressor endoscopies)
def check_parts(S, FT):
    q = FT.ps_oac.values.astype(float); V = G2["vs_test"](S, q, 2); VTR = {kk: G2["vs_train"](S, q, tx["top"], 2) for kk, tx in S["TRX"].items()}; return V, VTR
def bars(S, VTR, keepE=None):
    TH = {qq: np.full((S["E"], 10), np.nan) for qq in QS}; EF = S["EF"]; TV = {}
    for (j, kk), tx in S["TRX"].items():
        v = VTR[(j, kk)]; m = (tx["lab"] == 1) & np.isfinite(v)
        if keepE is not None: m &= keepE[tx["idx"]]
        for qq in QS: t = thr_sens(v[m], qq) if m.any() else -np.inf; TH[qq][EF[:, j] == kk, j] = t; TV[(j, kk, qq)] = t
    return TH, TV
def apply_check(S, FM, NM, GB, V, TH):
    k2 = np.minimum(2, S["nsz"])[:, None]; return FM & ~(FM & (V < TH)), NM + np.where(FM, np.where(GB == 1, k2 - 1, k2), 0)
def first_flag(S, F):
    """(E, 10) True at each patient's first flagged endoscopy (time order) per repeat"""
    out = np.zeros_like(F)
    for o in S["ORD"]:
        f = F[o]; has = f.any(0); idx = f.argmax(0); out[o[idx[has]], np.where(has)[0]] = True
    return out
def one_look(S, FM, NM, GB, V, TH):
    """check only at each patient's first memory flag; a cancelled flag is final; sequencing only there"""
    FF = first_flag(S, FM); k2 = np.minimum(2, S["nsz"])[:, None]; keep = FF & (V >= TH)
    return keep, _ns_until_first(S, FM, NM) + np.where(FF, np.where(GB == 1, k2 - 1, k2), 0)
def _ns_until_first(S, FM, NM):
    """gate sequencing (middle band) counted up to and including the first memory flag; afterwards the patient is decided"""
    out = np.zeros_like(NM)
    for o in S["ORD"]:
        f = FM[o]; c = np.cumsum(f, 0); before = (c == 0) | ((c == 1) & f); out[o] = np.where(before, NM[o], 0)
    return out
def infold_bar(S, TRF, VTR, TV, slack=0, onelook=False, keepE=None):
    """per outer fold: the check bar (or no check) with the fewest training non-progressor patients flagged, subject to training progressor-patient detection
    >= the memory rule's own training detection minus slack (ties: the higher share q, i.e. fewer cancellations). Returns TH (E, 10) and pick counts"""
    TH = np.full((S["E"], 10), np.nan); EF = S["EF"]; picks = {}
    for (j, kk), tx in S["TRX"].items():
        f = TRF[(j, kk)][tx["idx"]] & (True if keepE is None else keepE[tx["idx"]]); v = VTR[(j, kk)]; lab = tx["lab"]; pid = tx["pid"]; POS = S["POS"][tx["idx"]]
        def outcome(t):
            if onelook:
                kept = np.zeros(len(f), bool)
                for p in np.unique(pid[f]):
                    ii = np.where((pid == p) & f)[0]; e0 = ii[np.argmin(POS[ii])]; kept[e0] = v[e0] >= t
            else: kept = f & ~(v < t)
            return len(np.unique(pid[kept & (lab == 1)])), len(np.unique(pid[kept & (lab == 0)]))
        d0 = len(np.unique(pid[f & (lab == 1)])); best = (outcome(-np.inf)[1], 0, -np.inf, "none")
        for qq in QS:
            d, npf = outcome(TV[(j, kk, qq)])
            if d >= d0 - slack and (npf, -qq) < best[:2]: best = (npf, -qq, TV[(j, kk, qq)], qq)
        TH[EF[:, j] == kk, j] = best[2]; picks[str(best[3])] = picks.get(str(best[3]), 0) + 1
    return TH, picks
# ------------------------------------------------------------------ evaluation
def counts(S):
    return int((S["plab"] == 1).sum()), int((S["plab"] == 0).sum())
def pt(S, F, NS, em=None):
    A = patient_arrays(S, F, NS, em); o = lstats(S, A, np.ones((1, S["P_"]))); nP, nN = counts(S); pi = A["pinc"]
    nPe = int(((S["plab"] == 1) & pi).sum()); nNe = int(((S["plab"] == 0) & pi).sum())
    return {"caught": r2(o["caught_share"][0] * nPe), "np_flagged": r2(o["np_flagged_share"][0] * nNe), "seq_per_patient": r3(o["seq_per_patient"][0]), "flags_per_np_py": r3(o["flags_per_np_py"][0]),
            "endo_spec": r3(o["endo_spec"][0]), "endo_sens": r3(o["endo_sens"][0])}
def avgA(L):
    o = {k: np.mean([a[k] for a in L], 0) for k in ("FL", "NF", "NS", "W2", "NE")}; o["pinc"] = np.any([a["pinc"] for a in L], 0); o["TFmed"] = np.nanmean([a["TFmed"] for a in L], 0); return o
KEYS = (("np_flagged_share", "np_flagged", "N"), ("caught_share", "caught", "P"), ("flags_per_np_py", "flags_per_np_py", 1), ("seq_per_patient", "seq_per_patient", 1), ("endo_spec", "endo_spec", 1))
def full(S, A, D):
    one = np.ones((1, S["P_"])); ob = lstats(S, A, one); bs = lstats(S, A, D); pi = A["pinc"]; nPe = int(((S["plab"] == 1) & pi).sum()); nNe = int(((S["plab"] == 0) & pi).sum())
    mul = lambda m: nNe if m == "N" else (nPe if m == "P" else 1)
    return {nm: {"value": r3(mul(m) * ob[k][0]), "ci95": ci2(mul(m) * bs[k])} for k, nm, m in KEYS}
def delta(S, A, Ar, D):
    one = np.ones((1, S["P_"])); a0, r0, ab, rb = lstats(S, A, one), lstats(S, Ar, one), lstats(S, A, D), lstats(S, Ar, D); pi = A["pinc"] | Ar["pinc"]
    nPe = int(((S["plab"] == 1) & pi).sum()); nNe = int(((S["plab"] == 0) & pi).sum()); mul = lambda m: nNe if m == "N" else (nPe if m == "P" else 1)
    return {nm: dstat(mul(m) * float(a0[k][0] - r0[k][0]), mul(m) * (ab[k] - rb[k])) for k, nm, m in KEYS}
def caught32(S, A, Ar, D):
    """progressors caught counted out of all 32 (patients with no evaluated endoscopy count as missed), paired difference against Ar"""
    isP = S["plab"] == 1; one = np.ones((1, S["P_"]))
    def cc(Aa, Mb): return 32 * ((Mb * isP) @ Aa["FL"]).mean(1) / (Mb * isP).sum(1)
    return {"value": r2(cc(A, one)[0]), "ref": r2(cc(Ar, one)[0]), "delta_vs_ref": dstat(float(cc(A, one)[0] - cc(Ar, one)[0]), cc(A, D) - cc(Ar, D))}
# ------------------------------------------------------------------ setup shared by every task
def base_setup():
    S = G2["build"](); FT = G2["load_feat"](); ST = G9["statics"](S); INFO = S["info"]; V, VTR = check_parts(S, FT); TH, TV = bars(S, VTR)
    return S, FT, ST, INFO, V, VTR, TH, TV
def rules_for(S, ST_s, ST_e, INFO, GB, V, VTR, TH, TV, trkeep=None):
    R1 = engine(S, ST_s, ST_e, INFO, FAM1, trkeep); R12 = engine(S, ST_s, ST_e, INFO, FAM12, trkeep)
    FF, NF = apply_check(S, R12["F"], R12["NS"], GB, V, TH[0.90]); THi, pk = infold_bar(S, R12["TRF"], VTR, TV, keepE=trkeep); FI, NI = apply_check(S, R12["F"], R12["NS"], GB, V, THi)
    return R1, R12, {"A1": (R1["F"], R1["NS"]), "A12": (R12["F"], R12["NS"]), "final": (FF, NF), "final_infold": (FI, NI)}, pk
def shuffled_ST(S, ST, rng):
    ST2 = dict(ST); PADI = ST["PADI"].copy()
    for i, o in enumerate(S["ORD"]): PADI[i, :len(o)] = rng.permutation(PADI[i, :len(o)])
    ST2["PADI"] = PADI; ST2["PADS"] = np.where(ST["VAL"], PADI, 0); return ST2
def missing_ST(S, ST, rng, p):
    ST2 = dict(ST); MISS = rng.rand(S["E"]) < p; ST2["MISS"] = MISS; Mm = np.where(ST["VAL"], MISS[ST["PADS"]], False); W = ST["W"]; ST2["EW"] = {}
    for h, A in ST["EW"].items():
        A2 = A.copy(); off = ~np.eye(W, dtype=bool)[None]; A2[np.broadcast_to(Mm[:, None, :], A2.shape) & off] = 0.0; ST2["EW"][h] = A2
    return ST2, MISS
def trunc_S(S, keep):
    S2 = dict(S); S2["ORD"] = [o[keep[o]] for o in S["ORD"]]; S2["INC"] = S["INC"] * keep[None, :]; ks = np.zeros(len(S["popm"]), bool)
    for e in np.where(keep)[0]: ks[S["es"][e]] = True
    S2["popm"] = S["popm"] & ks; return S2
def trunc_py(S, ST, keep):
    """non-progressor follow-up of a truncated history: first endoscopy to the first dropped endoscopy (or the reference date if none dropped)"""
    PY = S["PY"].copy()
    for i, o in enumerate(S["ORD"]):
        kk = keep[o]
        if not kk.all(): PY[i] = ST["TY"][i, int(np.argmin(kk))]
    return PY
def stored_check(S):
    R9 = M + "/review_round9"; Z = np.load(R9 + "/grp_A12.npz"); Za = np.load(R9 + "/grp_A1ga.npz")
    return (Za["A1_gate_al_ew__sel_np_F"].astype(bool), Za["A1_gate_al_ew__sel_np_NS"].astype(int)), (Z["A12__sel_np_F"].astype(bool), Z["A12__sel_np_NS"].astype(int))
def grid_apparent(S, R1, R12, GB, V, TH, ref_pt):
    """OPTIMISTIC best-of-grid: every A1 / A12 candidate (training-chosen thresholds) x {no check, each bar}, picked on the evaluation patients:
    fewest NP flagged with caught >= R49's - 0.5"""
    best = None; bx = None; n = 0; nP, nN = counts(S)
    for nm, R in (("A1", R1), ("A12", R12)):
        for q in range(R["CF"].shape[0]):
            Fq = R["CF"][q]; Nq = R["CNS"][q].astype(int)
            for b in (None,) + QS:
                F, NS = (Fq, Nq) if b is None else apply_check(S, Fq, Nq, GB, V, TH[b]); o = pt(S, F, NS); n += 1
                if o["caught"] >= ref_pt["caught"] - 0.5 and (best is None or o["np_flagged"] < best["np_flagged"]): best = dict(o, rule=nm, cand=q, bar=b)
                ex = (o["np_flagged"] - ref_pt["np_flagged"]) / nN - (o["caught"] - ref_pt["caught"]) / nP
                if bx is None or ex < bx["excess"]: bx = dict(o, rule=nm, cand=q, bar=b, excess=round(ex, 4))
    return best, bx, n
res = {"task": TASK}
# ================================================================== base: reproduction + tests 1, 3, 4, 5, 8 and the real-label reference for test 7
if TASK == "base":
    t0 = time.time(); S, FT, ST, INFO, V, VTR, TH, TV = base_setup(); D = S["D"]; GB = S["GB"]
    R1, R12, RU, pk = rules_for(S, ST, ST, INFO, GB, V, VTR, TH, TV); (sa1, sn1), (sa12, sn12) = stored_check(S)
    res["reproduction"] = {"A1_flag_mismatches": int((R1["F"] != sa1).sum()), "A1_ns_mismatches": int((R1["NS"] != sn1).sum()), "A12_flag_mismatches": int((R12["F"] != sa12).sum()), "A12_ns_mismatches": int((R12["NS"] != sn12).sum())}
    RU = {"R49": (S["GC"], S["NS49"]), "P1": (S["P1F"], np.zeros((S["E"], 10), int)), **RU}
    GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); dys = (GR == 1) | (GR == 2); G = np.array([dys[s].any() for s in S["es"]])
    RU["grade"] = (np.repeat(G[:, None], 10, 1), np.zeros((S["E"], 10), int))
    res["check"] = {k: pt(S, *v) for k, v in RU.items()}; res["infold_bar_picks_real"] = pk
    AR = {k: patient_arrays(S, *v) for k, v in RU.items()}
    res["all_patients"] = {k: {"values": full(S, AR[k], D), "vs_R49": delta(S, AR[k], AR["R49"], D), "vs_P1": delta(S, AR[k], AR["P1"], D)} for k in RU}
    np.savez_compressed(XD + "/base_flags.npz", **{f"{k}__F": v[0] for k, v in RU.items()}, **{f"{k}__NS": v[1] for k, v in RU.items()}, **{f"TRF12_{j}_{kk}": f for (j, kk), f in R12["TRF"].items()})
    print("reproduction", res["reproduction"], res["check"], round(time.time() - t0), flush=True)
    # ---------- test 1: exposure equalised (frozen per-fold rules; memory is causal so kept endoscopies' scores are unchanged)
    isP = S["plab"] == 1; NE = S["NE"].astype(int); POS = S["POS"]; pidx = S["pidx"]; T1 = {}
    def masks():
        rng = np.random.RandomState(7001); NEP = NE[isP]
        for d in range(200):
            m = np.where(isP, NE, np.minimum(NE, rng.choice(NEP, S["P_"]))); yield "np_matched_to_progressor_counts", POS < m[pidx]
        for m_ in (2, 3): yield f"first_{m_}_everyone", POS < m_
        for m_ in (2, 3): yield f"first_{m_}_non_progressors_only", (POS < m_) | isP[pidx]
    acc = {}
    for nm, keep in masks():
        Se = dict(S, PY=trunc_py(S, ST, keep)); a = acc.setdefault(nm, {"A": {k: [] for k in RU}, "PY": [], "pt": {k: [] for k in RU}, "ne_np": []}); a["PY"].append(Se["PY"]); a["ne_np"].append(float((keep & ~isP[pidx]).sum() / (~isP).sum()))
        for k, v in RU.items():
            A = patient_arrays(Se, v[0], v[1], keep); a["A"][k].append(A); o = lstats(Se, A, np.ones((1, S["P_"]))); a["pt"][k].append((32 * o["caught_share"][0], 43 * o["np_flagged_share"][0]))
    for nm, a in acc.items():
        Se = dict(S, PY=np.mean(a["PY"], 0)); AA = {k: avgA(L) for k, L in a["A"].items()}; o = {"draws": len(a["PY"]), "np_endoscopies_per_np_patient": r2(np.mean(a["ne_np"]))}
        for k in RU:
            if k == "R49": continue
            dd = np.array(a["pt"][k]) - np.array(a["pt"]["R49"])
            o[k] = {"values": full(Se, AA[k], D), "vs_R49": delta(Se, AA[k], AA["R49"], D), "vs_P1": delta(Se, AA[k], AA["P1"], D),
                    "per_draw_delta_np_vs_R49": {"mean": r2(dd[:, 1].mean()), "p2.5_p97.5": [r2(np.percentile(dd[:, 1], 2.5)), r2(np.percentile(dd[:, 1], 97.5))]},
                    "per_draw_delta_caught_vs_R49": {"mean": r2(dd[:, 0].mean())}}
        o["R49"] = {"values": full(Se, AA["R49"], D)}; T1[nm] = o
    T1["note"] = ("frozen per-fold rules (thresholds and hyper-parameters chosen on training patients with full histories) evaluated on truncated held-out histories; "
                  "non-progressor follow-up for flags per patient-year = first endoscopy to the first dropped endoscopy; draws averaged as fractional patient outcomes, then the paired patient bootstrap")
    res["test1_exposure_frozen"] = T1; print("test1", round(time.time() - t0), flush=True)
    # ---------- test 3: exclusion of endoscopies close to the event (frozen rules)
    TTE = S["TTE"]; T3 = {}
    for yr in (1, 2):
        for sym in (False, True):
            keep = (TTE >= yr) | (~isP[pidx] if not sym else np.zeros(S["E"], bool)); nm = f"drop_within_{yr}y_of_event" + ("_and_np_within_{}y_of_last".format(yr) if sym else "_progressors_only")
            AA = {k: patient_arrays(S, v[0], v[1], keep) for k, v in RU.items()}; elig = int((AA["R49"]["pinc"] & isP).sum())
            o = {"progressors_with_an_evaluated_endoscopy": m5(elig), "non_progressors_with_an_evaluated_endoscopy": m5(int((AA["R49"]["pinc"] & ~isP).sum()))}
            for k in RU:
                o[k] = {"caught_of_32": caught32(S, AA[k], AA["R49"], D), "caught_of_32_vs_grade": caught32(S, AA[k], AA["grade"], D)["delta_vs_ref"]}
                if sym and k != "R49": o[k]["vs_R49"] = delta(S, AA[k], AA["R49"], D)
            T3[nm] = o
    res["test3_prevalent_exclusion"] = T3; print("test3", round(time.time() - t0), flush=True)
    # ---------- tests 4 and 5: check bar sweep, in-fold bar, one look per patient, first endoscopy only
    FM, NM = RU["A12"]; A49 = AR["R49"]; A12 = AR["A12"]; T5 = {}
    for qq in QS:
        F, NS = apply_check(S, FM, NM, GB, V, TH[qq]); A = patient_arrays(S, F, NS); T5[f"bar_{qq}"] = {"values": pt(S, F, NS), "vs_R49": delta(S, A, A49, D), "vs_A12": delta(S, A, A12, D)}
    for sl in (0, 1):
        THi, pk = infold_bar(S, R12["TRF"], VTR, TV, slack=sl); F, NS = apply_check(S, FM, NM, GB, V, THi); A = patient_arrays(S, F, NS)
        T5[f"infold_bar_slack{sl}"] = {"values": pt(S, F, NS), "vs_R49": delta(S, A, A49, D), "vs_A12": delta(S, A, A12, D), "picks": pk}
    res["test5_bar_sweep"] = T5; T4 = {}
    for nm, THx in (("fixed_cn90", TH[0.90]), ("infold_bar", infold_bar(S, R12["TRF"], VTR, TV, onelook=True)[0]), ("infold_bar_chosen_multi_look", infold_bar(S, R12["TRF"], VTR, TV)[0])):
        F, NS = one_look(S, FM, NM, GB, V, THx); A = patient_arrays(S, F, NS); Fm, Nm = apply_check(S, FM, NM, GB, V, THx); Am = patient_arrays(S, Fm, Nm)
        T4[nm] = {"values": pt(S, F, NS), "vs_R49": delta(S, A, A49, D), "vs_A12": delta(S, A, A12, D), "vs_same_bar_multi_look": delta(S, A, Am, D)}
    first = S["FIRST"]; fm = np.repeat(first[:, None], 10, 1)
    for nm, THx in (("first_endoscopy_only_fixed_cn90", TH[0.90]),):
        Af = {k: patient_arrays(S, v[0], v[1], first) for k, v in RU.items()}; F, NS = apply_check(S, FM, NM, GB, V, THx); Ac = patient_arrays(S, F, NS, first)
        T4[nm] = {"R49": pt(S, S["GC"], S["NS49"], first), "A12": pt(S, FM, NM, first), "A12_plus_check": pt(S, F, NS, first), "check_vs_R49": delta(S, Ac, Af["R49"], D), "check_vs_A12": delta(S, Ac, Af["A12"], D),
                   "note": "only each patient's first pre-event endoscopy is evaluated (memory = that endoscopy alone; ACE-B-like); the certificate cannot act"}
    res["test4_one_look"] = T4; print("tests 4-5", round(time.time() - t0), flush=True)
    # ---------- test 7 reference (real labels): nested gains and the optimistic best-of-grid
    p49 = pt(S, S["GC"], S["NS49"]); best, bx, ng = grid_apparent(S, R1, R12, GB, V, TH, p49)
    res["test7_real"] = {"R49": p49, "nested": {k: pt(S, *RU[k]) for k in ("A1", "A12", "final", "final_infold")}, "best_of_grid_optimistic": best, "best_of_grid_excess_optimistic": bx, "grid_size": ng}
    # ---------- test 8: batch / scanner and the within-patient bounce
    import statsmodels.formula.api as smf
    SD_ = pd.read_csv(TC.HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(G2["ids"]); scn = SD_.scanner.astype(str).values; bt = FT.batch.astype(str).values
    es = S["es"]; esc = np.array([pd.Series(scn[s]).mode().iloc[0] for s in es]); ebt = np.array([pd.Series(bt[s]).mode().iloc[0] for s in es]); mixsc = np.array([len(set(scn[s])) > 1 for s in es])
    zi = lg(S["OUTS"]["img"]).mean(1); TL, CSt, TS = G2["g8"]["cutoff_arrays"](INFO, S["FO"]); O = S["O"]; TOP = S["TOPO"]; SR = np.full((S["E"], 10), np.nan)
    for e, s in enumerate(es):
        for j in range(10): SR[e, j] = lg(O["L-LATE"][TOP[e, j, 0], j]) - lg(TS[s[0], j]) if GB[e, j] == 1 else lg(O["L-IMG"][s, j].max()) - lg(CSt[s[0], j])
    gs = np.median(SR, 1); rows = []
    for i, o in enumerate(S["ORD"]):
        for a, b in zip(o[:-1], o[1:]):
            rows.append({"pid": i, "cls": int(S["plab"][i]), "d_img": abs(zi[b] - zi[a]), "d_gate": abs(gs[b] - gs[a]), "scanner_change": int(esc[a] != esc[b]), "batch_change": int(ebt[a] != ebt[b]),
                         "gap": float(ST["TY"][i, S["POS"][b]] - ST["TY"][i, S["POS"][a]]), "flip": int((gs[a] >= 0) != (gs[b] >= 0))})
    DF = pd.DataFrame(rows); T8 = {"endoscopies_with_mixed_scanner_samples": m5(int(mixsc.sum()))}
    for cl, v in (("non_progressor", 0), ("progressor", 1)):
        d = DF[DF.cls == v]; o = {"consecutive_pairs": int(len(d)), "pairs_scanner_change": m5(int(d.scanner_change.sum())), "pairs_batch_change": m5(int(d.batch_change.sum()))}
        for yv in ("d_img", "d_gate", "flip"):
            try:
                f = smf.ols(f"{yv} ~ scanner_change + batch_change + gap", d).fit(cov_type="cluster", cov_kwds={"groups": d.pid}); ci = f.conf_int()
                o[yv] = {t: {"coef": r3(f.params[t]), "ci95": [r3(ci.loc[t, 0]), r3(ci.loc[t, 1])], "p": round(float(f.pvalues[t]), 4)} for t in ("scanner_change", "batch_change", "gap")}
            except Exception as ex: o[yv] = f"failed: {str(ex)[:100]}"
        for yv in ("d_img", "d_gate", "flip"):
            o[f"{yv}_mean_by_scanner_change"] = {"no_change": r3(d[d.scanner_change == 0][yv].mean()), "change": r3(d[d.scanner_change == 1][yv].mean()) if d.scanner_change.sum() >= 5 else "<5 pairs"}
        T8[cl] = o
    def icc1(vals):
        k = len(vals); ni = np.array([len(v) for v in vals]); N = ni.sum(); m = np.concatenate(vals).mean(); mi = np.array([v.mean() for v in vals])
        msb = (ni * (mi - m) ** 2).sum() / (k - 1); msw = sum(((v - v.mean()) ** 2).sum() for v in vals) / (N - k); n0 = (N - (ni ** 2).sum() / N) / (k - 1); return (msb - msw) / (msb + (n0 - 1) * msw)
    ic = {}
    for cl, v in (("non_progressor", 0), ("progressor", 1)):
        pts = [i for i in range(S["P_"]) if S["plab"][i] == v]; msk = np.isin(pidx, pts); df = pd.DataFrame({"z": zi, "g": gs, "sc": esc, "bt": ebt, "pid": pidx})[msk]; o = {}
        for nm, col in (("logit max L-IMG", "z"), ("R49-aligned gate score", "g")):
            res_sc = smf.ols(f"{col} ~ C(sc)", df).fit().resid.values; res_both = smf.ols(f"{col} ~ C(sc) + C(bt)", df).fit().resid.values
            vals = lambda x: [x[(df.pid == i).values] for i in pts]
            o[nm] = {"icc1_unadjusted": r3(icc1(vals(df[col].values))), "icc1_scanner_adjusted": r3(icc1(vals(res_sc))), "icc1_scanner_and_batch_adjusted": r3(icc1(vals(res_both)))}
        ic[cl] = o
    T8["icc_on_repeat_mean_scores"] = ic
    one_sc = np.array([len(set(esc[o])) == 1 for o in S["ORD"]]); sub = {}
    for nm, pm in (("all_endoscopies_one_scanner", one_sc), ("scanner_changes_within_patient", ~one_sc)):
        em = pm[pidx]; Ar_ = patient_arrays(S, S["GC"], S["NS49"], em); Aa = patient_arrays(S, *RU["A1"], em); Af = patient_arrays(S, *RU["final"], em)
        sub[nm] = {"non_progressors": m5(int((pm & ~isP).sum())), "progressors": m5(int((pm & isP).sum()))}
        if (pm & ~isP).sum() >= 5: sub[nm].update({"A1_vs_R49": delta(S, Aa, Ar_, D), "final_vs_R49": delta(S, Af, Ar_, D)})
    T8["memory_gain_by_scanner_history"] = sub
    T8["note"] = ("consecutive pre-event endoscopies within patient; |change| of the repeat-mean logit max L-IMG and of the median R49-aligned gate score (0 = flag boundary); flip = gate flag differs; "
                  "endoscopy scanner / sequencing batch = commonest over its samples; OLS with patient-clustered SEs; ICC(1) on repeat-mean scores, adjusted = residuals on scanner (and batch) fixed effects")
    res["test8_batch"] = T8; res["seconds"] = round(time.time() - t0); dump(res)
# ================================================================== shuf_<b> / miss_<pct>_<b>: 10 draws per job, nested engine reruns
elif TASK.startswith("shuf_") or TASK.startswith("miss_"):
    S, FT, ST, INFO, V, VTR, TH, TV = base_setup(); GB = S["GB"]; b = int(TASK.split("_")[-1]); out = {}
    for d in range(10):
        draw = b * 10 + d; rng = np.random.RandomState(7100 + draw if TASK.startswith("shuf_") else 7300 + 1000 * int(TASK.split("_")[1]) + draw)
        if TASK.startswith("shuf_"): ST2 = shuffled_ST(S, ST, rng)
        else: ST2, MISS = missing_ST(S, ST, rng, int(TASK.split("_")[1]) / 100.0); out[f"d{draw}_MISS"] = MISS
        for var, (sts, ste) in (("both", (ST2, ST2)), ("deploy", (ST, ST2))):
            _, _, RU, pk = rules_for(S, sts, ste, INFO, GB, V, VTR, TH, TV)
            for k, (F, NS) in RU.items(): out[f"d{draw}_{var}_{k}__F"] = F; out[f"d{draw}_{var}_{k}__NS"] = NS.astype(np.int16)
        print("draw", draw, flush=True)
    np.savez_compressed(f"{XD}/{TASK}.npz", **out); print("EXE7 DONE", TASK, flush=True)
# ================================================================== trunc_<b>: whole selection rerun on a truncated cohort (training patients truncated too)
elif TASK.startswith("trunc_"):
    S, FT, ST, INFO, V, VTR, TH, TV = base_setup(); b = int(TASK.split("_")[1]); isP = S["plab"] == 1; NE = S["NE"].astype(int); POS = S["POS"]; pidx = S["pidx"]; out = {}
    if b == 0: L = [("first_2_everyone", POS < 2), ("first_3_everyone", POS < 3)]
    else:
        L = []; rng = np.random.RandomState(7500 + b)
        for d in range(5): m = np.where(isP, NE, np.minimum(NE, rng.choice(NE[isP], S["P_"]))); L.append((f"matched_{(b - 1) * 5 + d}", POS < m[pidx]))
    for nm, keep in L:
        S2 = trunc_S(S, keep); ST2 = G9["statics"](S2); INFO2 = TC.triage(S["O"], S["I"], S["FO"], S2["popm"], 0.95, 0.90)[2]; GC2, NS2, GB2 = r49_calls(S["O"], INFO2, S["FO"], S["es"])
        TH2, TV2 = bars(S, VTR, keep); _, _, RU, pk = rules_for(S2, ST2, ST2, INFO2, GB2, V, VTR, TH2, TV2, trkeep=keep)
        out[f"{nm}__keep"] = keep; out[f"{nm}_R49__F"] = GC2; out[f"{nm}_R49__NS"] = NS2
        for k, (F, NS) in RU.items(): out[f"{nm}_{k}__F"] = F; out[f"{nm}_{k}__NS"] = NS.astype(np.int16)
        print(nm, flush=True)
    np.savez_compressed(f"{XD}/{TASK}.npz", **out); print("EXE7 DONE", TASK, flush=True)
# ================================================================== perm_<free|strat>_<b>: label-permutation null of the whole selection (5 permutations per job)
elif TASK.startswith("perm_"):
    S, FT, ST, INFO, V, VTR, TH, TV = base_setup(); kind, b = TASK.split("_")[1], int(TASK.split("_")[2]); NE = S["NE"].astype(int); recs = []
    strata = np.digitize(NE, [2, 3, 4, 5, 7])
    for d in range(5):
        pn = b * 5 + d; rng = np.random.RandomState(7700 + pn + (500 if kind == "strat" else 0)); pl = S["plab"].copy()
        if kind == "free": pl = rng.permutation(pl)
        else:
            for s_ in np.unique(strata): ii = np.where(strata == s_)[0]; pl[ii] = rng.permutation(pl[ii])
        S2 = dict(S); S2["plab"] = pl; S2["elab"] = pl[S["pidx"]]; S2["TR"] = {kk: dict(v, lab=S2["elab"][v["idx"]]) for kk, v in S["TR"].items()}
        S2["TRX"] = {kk: dict(v, lab=S2["elab"][v["idx"]]) for kk, v in S["TRX"].items()}; ys = y0.copy(); rows = S["rows"]; ys[rows] = S2["elab"][S["SEi"][rows]]
        TC.y = ys; G9["y"] = ys; G2["g8"]["y"] = ys
        INFO2 = TC.triage(S["O"], S["I"], S["FO"], S["popm"], 0.95, 0.90)[2]; GC2, NS2, GB2 = r49_calls(S["O"], INFO2, S["FO"], S["es"]); TH2, TV2 = bars(S2, VTR)
        R1, R12, RU, pk = rules_for(S2, ST, ST, INFO2, GB2, V, VTR, TH2, TV2); p49 = pt(S2, GC2, NS2); best, bx, ng = grid_apparent(S2, R1, R12, GB2, V, TH2, p49)
        recs.append({"perm": pn, "kind": kind, "R49": p49, "nested": {k: pt(S2, *v) for k, v in RU.items()}, "best_of_grid_optimistic": best, "best_of_grid_excess_optimistic": bx, "label_agreement_with_truth": r3(float((pl == S["plab"]).mean()))})
        TC.y = y0; G9["y"] = y0; G2["g8"]["y"] = y0; print("perm", pn, flush=True)
    json.dump(recs, open(f"{XD}/{TASK}.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("EXE7 DONE", TASK, flush=True)
# ================================================================== merge: tests 2, 6, 7 and the full-rerun truncation
elif TASK == "merge":
    S, FT, ST, INFO, V, VTR, TH, TV = base_setup(); D = S["D"]; RUK = ("A1", "A12", "final", "final_infold"); A49 = patient_arrays(S, S["GC"], S["NS49"]); one = np.ones((1, S["P_"]))
    P1A = patient_arrays(S, S["P1F"], np.zeros((S["E"], 10), int)); base = json.load(open(OUT + "/base.json")); real = base["test7_real"]
    def draws_block(files, var):
        L = {k: [] for k in RUK}; dd = {k: [] for k in RUK}
        for f in files:
            z = np.load(f); ds = sorted({x.split("_")[0] for x in z.files if x.startswith("d") and "__F" in x})
            for dn in ds:
                for k in RUK:
                    A = patient_arrays(S, z[f"{dn}_{var}_{k}__F"], z[f"{dn}_{var}_{k}__NS"].astype(int)); L[k].append(A); o = lstats(S, A, one); r = lstats(S, A49, one)
                    dd[k].append((43 * (o["np_flagged_share"][0] - r["np_flagged_share"][0]), 32 * (o["caught_share"][0] - r["caught_share"][0])))
        o = {"draws": len(L["A1"])}
        for k in RUK:
            if not L[k]: continue
            A = avgA(L[k]); x = np.array(dd[k]); o[k] = {"values": full(S, A, D), "vs_R49": delta(S, A, A49, D), "vs_P1": delta(S, A, P1A, D), "per_draw_delta_np_vs_R49": {"mean": r2(x[:, 0].mean()), "p2.5_p97.5": [r2(np.percentile(x[:, 0], 2.5)), r2(np.percentile(x[:, 0], 97.5))]},
                         "per_draw_delta_caught_vs_R49": {"mean": r2(x[:, 1].mean()), "p2.5_p97.5": [r2(np.percentile(x[:, 1], 2.5)), r2(np.percentile(x[:, 1], 97.5))]}}
        return o
    T2 = {var: draws_block(sorted(glob.glob(XD + "/shuf_*.npz")), var) for var in ("both", "deploy")}
    T2["note"] = "both = the whole nested selection rerun on shuffled histories (dates kept, scores permuted across a patient's date slots); deploy = rules selected on true order, held-out histories shuffled"
    res["test2_order_shuffle"] = T2
    res["test6_missing_records"] = {f"{p}pct": {var: draws_block(sorted(glob.glob(XD + f"/miss_{p}_*.npz")), var) for var in ("deploy", "both")} for p in (20, 40)}
    res["test6_missing_records"]["note"] = "each endoscopy independently missing from later histories (memory and certificate) with probability p; it is still judged on its own score when it happens; deploy = selection on full histories"
    # full-rerun truncation
    TR = {}; zs = [np.load(f) for f in sorted(glob.glob(XD + "/trunc_*.npz"))]; isP = S["plab"] == 1
    groups = {}
    for z in zs:
        for key in z.files:
            if key.endswith("__keep"): nm = key[:-6]; groups.setdefault("matched" if nm.startswith("matched") else nm, []).append((z, nm))
    for g, L in groups.items():
        acc = {k: [] for k in ("R49",) + RUK}; PYs = []
        for z, nm in L:
            keep = z[nm + "__keep"]; Se = dict(S, PY=trunc_py(S, ST, keep)); PYs.append(Se["PY"])
            for k in acc: acc[k].append(patient_arrays(Se, z[f"{nm}_{k}__F"], z[f"{nm}_{k}__NS"].astype(int), keep))
        Se = dict(S, PY=np.mean(PYs, 0)); AA = {k: avgA(v) for k, v in acc.items()}; o = {"draws": len(L), "R49": {"values": full(Se, AA["R49"], D)}}
        for k in RUK: o[k] = {"values": full(Se, AA[k], D), "vs_R49": delta(Se, AA[k], AA["R49"], D)}
        TR[g] = o
    TR["note"] = "whole nested selection (R49 cut-offs, memory threshold and hyper-parameters, certificate, check bar) rerun with training AND held-out non-progressors truncated; models not refitted"
    res["test1_exposure_full_rerun"] = TR
    # permutation nulls
    T7 = {"real": real}
    for kind in ("free", "strat"):
        recs = sum([json.load(open(f)) for f in sorted(glob.glob(XD + f"/perm_{kind}_*.json"))], [])
        if not recs: continue
        o = {"permutations": len(recs), "label_agreement_with_truth_mean": r3(np.mean([r["label_agreement_with_truth"] for r in recs]))}
        def blk(dn, dc, rv, rc):
            ex = dn / 43 - dc / 32; rex = rv / 43 - rc / 32
            return {"null_delta_np_mean": r2(np.nanmean(dn)), "null_delta_np_p2.5_p97.5": [r2(np.nanpercentile(dn, 2.5)), r2(np.nanpercentile(dn, 97.5))], "null_delta_caught_mean": r2(np.nanmean(dc)),
                    "null_excess_mean": r3(np.nanmean(ex)), "null_excess_p2.5_p97.5": [r3(np.nanpercentile(ex, 2.5)), r3(np.nanpercentile(ex, 97.5))], "null_excess_min": r3(np.nanmin(ex)),
                    "real_delta_np": r2(rv), "real_delta_caught": r2(rc), "real_excess": r3(rex), "share_null_at_or_below_real_excess": r3(float(np.nanmean(ex <= rex))), "share_null_at_or_below_real_delta_np": r3(float(np.nanmean(dn <= rv))), "null_excess_values": [r3(x) for x in ex]}
        for k in RUK:
            dn = np.array([r["nested"][k]["np_flagged"] - r["R49"]["np_flagged"] for r in recs]); dc = np.array([r["nested"][k]["caught"] - r["R49"]["caught"] for r in recs])
            o[k] = blk(dn, dc, real["nested"][k]["np_flagged"] - real["R49"]["np_flagged"], real["nested"][k]["caught"] - real["R49"]["caught"])
        g = lambda r: r["best_of_grid_excess_optimistic"]; rb = real["best_of_grid_excess_optimistic"]
        o["best_of_grid_excess"] = blk(np.array([g(r)["np_flagged"] - r["R49"]["np_flagged"] for r in recs]), np.array([g(r)["caught"] - r["R49"]["caught"] for r in recs]), rb["np_flagged"] - real["R49"]["np_flagged"], rb["caught"] - real["R49"]["caught"])
        o["best_of_grid_constrained_found_in_null"] = int(sum(r["best_of_grid_optimistic"] is not None for r in recs))
        T7[kind] = o
    T7["note"] = ("labels permuted at patient level (free) or within endoscopy-count strata 1/2/3/4/5-6/7+ (strat: keeps the progressors' shorter histories, removes the signal); "
                  "everything that uses labels is rerun: R49 cut-offs, memory threshold rule, half-life / certificate selection, check bars; models' stored predictions unchanged; "
                  "excess = change in the share of NP flagged minus change in the share of progressors caught (0 expected when the rule only moves a threshold); "
                  "best_of_grid_excess = OPTIMISTIC pick on the evaluation patients, among the A1 / A12 candidates x check bars (243 cells), of the lowest excess")
    res["test7_permutation"] = T7; dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
