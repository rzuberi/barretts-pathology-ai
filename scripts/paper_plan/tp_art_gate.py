"""Two papers, agent ART (docs/two_papers_ART.md): Paper 2 rules rerun with the clean-tile L-IMG. No rule is re-implemented: the module body of
ex_e9_frontier.py is executed unchanged (definitions only; it executes rv9_falseflags.py, rv8b_step3.py, rv8_landscape.py and the helper section of
ex_e3_score.py), exactly as tp_reg_freeze.py does; pick() and ev() are copied verbatim from tp_reg_freeze.py (the round-9 operating rule).
L-IMG is swapped through E3's variant() reading the ART refits (RD redirected to killcoyne_mm/two_papers_ART/refit), as E9's state('ml') swaps in
the mean-logit score: st = sample-level L-IMG (R49 gate; memory and light tier on the R49-aligned gate score); ml = endoscopy mean-logit (tier 0 =
mean-logit + EW memory, E9 'ml_mem'). L-LATE = (stored L-CNV + clean L-IMG) / 2. The confirmatory tier is not rerun (its quiet-genome check arrays
are built on the stored R49 bands). Every threshold / hyper-parameter is chosen on training patients (inner out-of-fold predictions of the refit).
Paired patient bootstrap (rv8 pdraws, RandomState(0), 2,000 draws) for clean - stored. Aggregates -> results/paper_final/two_papers_ART/gate.json.
TASK = gate   (ART_NB=k: smoke, k draws, JSON aside)"""
import os, sys, json, numpy as np, pandas as pd
THESIS = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; PP = THESIS + "/scripts/paper_plan"; sys.path.insert(0, PP)
RTASK = os.environ["TASK"]; os.environ["TASK"] = "tpart_import"
_fself = __file__; __file__ = PP + "/ex_e9_frontier.py"; exec(compile(open(__file__).read(), __file__, "exec")); __file__ = _fself   # definitions only
os.environ["TASK"] = RTASK
A = M + "/two_papers_ART"; OUTA = T + "/results/paper_final/two_papers_ART"; NBG = 2000
if os.environ.get("ART_NB"): NBG = int(os.environ["ART_NB"]); OUTA = A + "/smoke_json"
os.makedirs(OUTA, exist_ok=True); RD = A + "/refit"   # read_refit() looks up RD at call time
# ------------------------------------------------------------------ copied verbatim from tp_reg_freeze.py
def pick(c, cands):
    INC = c.S["INC"]; trN = c.trP & ~c.isP; st = []; fl = []; th = []
    for x in cands:
        pm = pat_scores(c, x["u"]); t = thr_for(c, pm); f = x["u"] >= t; inc = INC @ f.astype(float) > 0
        st.append((c.patient_det(f), int((inc & trN).sum()), int((f & c.tr_e & (c.elab == 0)).sum()), float(x["ns"][c.tr_e].mean()))); fl.append(f); th.append(t)
    st = np.array(st); feas = st[:, 0] >= c.D; nC = len(cands)
    q = sorted([q for q in range(nC) if feas[q]], key=lambda q: (st[q, 1], st[q, 2], st[q, 3], q))[0] if feas.any() else sorted(range(nC), key=lambda q: (-st[q, 0], st[q, 1], st[q, 2], q))[0]
    return q, th[q], fl[q], st[q]
# ------------------------------------------------------------------ L-IMG swap
def state_v(var, name):
    if name == "stored": return state(var)
    g = globals()
    if var == "st":
        Oi, Ii = variant(name)[:2]; Op, Ip = Oi, Ii
    else:
        Oi, Ii, Op, Ip, _ = variant("mlogit:" + name)
        Oi = sg(lg_(Oi) + 1e-6 * lg_(Op)); Ii = {k: pd.Series(sg(lg_(Ii[k].values) + 1e-6 * lg_(Ip[k].reindex(Ii[k].index).values)), index=Ii[k].index) for k in Ii}
    O = dict(O0); I = dict(I0); O["L-IMG"] = Oi; O["L-LATE"] = (O["L-CNV"] + Op) / 2; I["L-IMG"] = Ii; I["L-LATE"] = {k: (I["L-CNV"][k] + Ip[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    g["load"] = lambda src: (O, I, FO)
    try: return prep_state()
    finally: g["load"] = load_stored
def run(var, name):
    S, ST, INFO = state_v(var, name); E = S["E"]; R = {}
    fams = {"st": {"R49": None, "memory (gate score, EW)": lambda c: f_score("gate_al", "ew")(c), "light tier (A12)": f_A12},
            "ml": {"R49 (mean-logit gate)": None, "tier 0: mean-logit + EW memory": lambda c: f_score("img", "ew")(c)}}[var]
    F = {k: np.zeros((E, 10), bool) for k in fams}; NS = {k: np.zeros((E, 10), int) for k in fams}
    for j in range(10):
        for k in range(1, 11):
            c = make_ctx(S, ST, j, k, INFO); te = c.te_e
            for nm, b in fams.items():
                if b is None: F[nm][te, j] = c.r49_flag[te]; NS[nm][te, j] = c.r49_ns[te]; continue
                cands = b(c); q, t, f, _ = pick(c, cands); F[nm][te, j] = f[te]; NS[nm][te, j] = cands[q]["ns"][te]
    return S, F, NS
D = None; out = {}; keep = {}
if RTASK == "gate":
    for var in ("st", "ml"):
        for name in ("stored", "clean", "rand"):
            S, F, NS = run(var, name); keep[(var, name)] = (S, F, NS); print("ran", var, name, flush=True)
    S0 = keep[("st", "stored")][0]; nP = int((S0["plab"] == 1).sum()); nN = int((S0["plab"] == 0).sum())
    rng_ = np.random.RandomState(0); Dd = []
    while len(Dd) < NBG:
        mm = np.bincount(rng_.choice(S0["P_"], S0["P_"]), minlength=S0["P_"])
        if (mm * S0["plab"]).sum() > 0 and (mm * (1 - S0["plab"])).sum() > 0: Dd.append(mm)
    Dd = np.array(Dd, float)   # = rv8 pdraws(S) with NB draws
    def stats(S, F_, NS_):
        Ap = patient_arrays(S, F_, NS_); p1 = parts(S, Ap, np.ones((1, S["P_"]))); pb = parts(S, Ap, Dd)
        return {k: (sc_ * met(p1, m)[0], sc_ * met(pb, m)) for k, m, sc_ in (("caught", "caught_share", nP), ("np_flagged", "np_flagged_share", nN), ("seq_per_patient", "seq_per_patient", 1.0))}
    for var in ("st", "ml"):
        for rule in keep[(var, "stored")][1]:
            St = {nm: stats(keep[(var, nm)][0], keep[(var, nm)][1][rule], keep[(var, nm)][2][rule]) for nm in ("stored", "clean", "rand")}
            o = {nm: {k: {"value": r2(v[0]) if k != "seq_per_patient" else r3(v[0]), "ci95": [r2(np.percentile(v[1], 2.5)), r2(np.percentile(v[1], 97.5))]} for k, v in St[nm].items()} for nm in St}
            for nm in ("clean", "rand"):
                o[f"{nm}_minus_stored"] = {k: {"delta": r2(St[nm][k][0] - St["stored"][k][0]), "ci95": [r2(np.percentile(St[nm][k][1] - St["stored"][k][1], 2.5)), r2(np.percentile(St[nm][k][1] - St["stored"][k][1], 97.5))]} for k in St[nm]}
            out[rule] = o
    exp = {"R49": (31.1, 31.8), "light tier (A12)": (30.5, 21.5), "memory (gate score, EW)": (30.7, 23.6), "tier 0: mean-logit + EW memory": (31.0, 26.6)}
    out["smoke_stored_reproduces_E9"] = {r: bool(abs(out[r]["stored"]["caught"]["value"] - e[0]) < 0.051 and abs(out[r]["stored"]["np_flagged"]["value"] - e[1]) < 0.051) for r, e in exp.items()}
    out["n_progressor_patients"] = nP; out["n_np_patients"] = nN; out["draws"] = int(len(Dd))
    out["note"] = ("nested: every threshold / hyper-parameter chosen on training patients (inner OOF of each refit); held-out, repeat mean; 'clean' = clean-tile L-IMG refit, 'rand' = random-subset control "
                   "(same number of tiles); stored = the published L-IMG; paired patient bootstrap on the same draws")
    json.dump({"task": "gate", "agent": "ART", "gate": out}, open(f"{OUTA}/gate.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)); print(json.dumps(out["smoke_stored_reproduces_E9"]), "ART DONE gate", flush=True)
