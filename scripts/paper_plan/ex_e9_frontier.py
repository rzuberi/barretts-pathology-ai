"""Exploration E9 (docs/paper_explore_E9.md; rules paper/reviews_round7/decisions/EXPLORE_RULES.md): the clinical and economic picture. EXPLORATORY.
No rule is re-implemented. Executed unchanged: the module body of rv9_falseflags.py (task name matching none of its branches: statics, memory, cert_flags,
make_ctx, base, f_score, f_A12, pat_scores, the R49 / P1 references and the round-8 helpers), the helper section of ex_e3_score.py (mean-logit endoscopy
score, variant("mlogit:stored"), E3's 1e-6 sample tie-break for the top-H&E pick) and ex_e2_veto.py up to its dispatch (as story_deck/cluster/story_aggs_v2.py:
quiet-genome check = ps_oac of the 2 top-H&E samples, cn90 bar, veto_V). The only new piece is the SWEEP: in every outer fold the round-9 operating rule
(highest threshold whose training progressor-patient detection >= target; among a family's candidates the fewest training non-progressor patients flagged,
ties: flagged training NP endoscopies, sequencing, index) is applied at several targets instead of only the R49 gate's own training detection D:
targets = [D] + [nTP - k for k in KS] (nTP = training progressor patients of the fold). Everything is chosen on training patients; held-out evaluation.
Row-level arrays only under killcoyne_mm/explore_E9/ on the cluster; aggregates (no identifiers, no dates, cells < 5 as '<5') -> results/paper_final/explore_E9/.
TASK = run_<variant>_<group> (variant st | ml) | merge.  EXE9_NB=k: smoke (k bootstrap draws, aggregates written aside)."""
import os, sys, json, glob, time, numpy as np, pandas as pd
THESIS = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; PP = THESIS + "/scripts/paper_plan"; sys.path.insert(0, PP)
ETASK = os.environ["TASK"]; os.environ["TASK"] = "exe9"
_f = __file__; __file__ = PP + "/rv9_falseflags.py"; exec(compile(open(__file__).read(), __file__, "exec")); __file__ = _f   # whole module, definitions only
_e = open(PP + "/ex_e3_score.py").read(); exec(compile(_e[_e.index('E3 = M + "/explore_E3"'):_e.index('\n# ------------------------------------------------------------------ one variant')], PP + "/ex_e3_score.py", "exec"))
os.environ["TASK"] = ETASK
XD9 = M + "/explore_E9"; OUT9 = T + "/results/paper_final/explore_E9"; NB9 = 2000
if os.environ.get("EXE9_NB"): NB9 = int(os.environ["EXE9_NB"]); OUT9 = XD9 + "/smoke_json"
for d_ in (XD9, OUT9): os.makedirs(d_, exist_ok=True)
KS = list(range(0, 25)); NPS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0); TNAMES = ["D"] + [f"k{k}" for k in KS] + [f"n{int(r * 100)}" for r in NPS]   # n<r>: threshold flagging >= r of training NP patients (extends curves past 'catch every training progressor')
def jdump9(o, name): json.dump(o, open(f"{OUT9}/{name}.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)); print("EXE9 DONE", name, flush=True)
# ------------------------------------------------------------------ the two H&E variants (stored max-logit; E3 mean-logit)
load_stored = load
def state(var):
    """(S, ST, INFO) with the stored L-IMG (st) or E3's mean-logit endoscopy score (ml; exactly ex_e3_memory.py's construction)"""
    g = globals()
    if var == "st": g["load"] = load_stored; return prep_state()
    Oi, Ii, Op, Ip, _ = variant("mlogit:stored")
    Oi = sg(lg_(Oi) + 1e-6 * lg_(Op)); Ii = {k: pd.Series(sg(lg_(Ii[k].values) + 1e-6 * lg_(Ip[k].reindex(Ii[k].index).values)), index=Ii[k].index) for k in Ii}
    O = dict(O0); I = dict(I0); O["L-IMG"] = Oi; O["L-LATE"] = (O["L-CNV"] + Op) / 2; I["L-IMG"] = Ii; I["L-LATE"] = {k: (I["L-CNV"][k] + Ip[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    g["load"] = lambda src: (O, I, FO)
    try: return prep_state()
    finally: g["load"] = load_stored
# ------------------------------------------------------------------ pathology scores (no model: identical in training and held-out endoscopies)
GRv = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); P53v = pd.to_numeric(SM["P53 IHC"], errors="coerce").values
def path_scores(S):
    es = S["es"]; g = np.array([GRv[s].max() for s in es]); hasp = np.isfinite(P53v); ab = np.array([(np.where(hasp, P53v, 0)[s] == 1).any() for s in es]); em = np.array([hasp[s].any() for s in es])
    assert g.max() <= 2, "pre-event endoscopy graded HGD/IMC"
    dys = (GRv == 1) | (GRv == 2); abs_ = np.where(hasp, P53v, 0) == 1; g53sub = np.array([(dys[s[hasp[s]]] | abs_[s[hasp[s]]]).any() for s in es])   # rv8 P10 subset definition
    return {"grade": g, "g53": np.maximum(g, 2.0 * ab), "p53ab": ab, "p53_any": em, "g53_subset": g53sub}
def f_path(kind, mf):
    def b(c):
        s = c.PATH[kind]; z = np.zeros(len(s), int)
        if mf == "none": return [sc("none", {}, s.astype(float), z)]
        if mf == "mean": return [sc("running mean", {}, memory(c.ST, s.astype(float), "mean"), z)]
        return [sc(f"ew h={h}y", {"h": h}, memory(c.ST, s.astype(float), "ew", h), z) for h in (1, 2, 4)]
    return b
GROUPS9 = {"core": {"img:none": f_score("img", "none"), "img:ew": f_score("img", "ew"), "gate_al:none": f_score("gate_al", "none"), "gate_al:ew": f_score("gate_al", "ew")},
           "A12": {"A12": f_A12},
           "path": {f"{k}:{m}": f_path(k, m) for k in ("grade", "g53") for m in ("none", "mean", "ew")}}
# ------------------------------------------------------------------ quiet-genome check (E2 module body, as story_aggs_v2.py)
def check_arrays():
    NS2 = {"__file__": PP + "/ex_e2_veto.py", "__name__": "ex_e2_e9"}; os.environ["TASK"] = "__ex_e9__"
    _s = open(PP + "/ex_e2_veto.py").read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], PP + "/ex_e2_veto.py", "exec"), NS2); os.environ["TASK"] = ETASK
    g = NS2; S2 = g["build"](); FT = g["load_feat"](); q = FT.ps_oac.values.astype(float); V = g["vs_test"](S2, q, 2); vtr = {kk: g["vs_train"](S2, q, tx["top"], 2) for kk, tx in S2["TRX"].items()}
    _, _, THc = g["veto_V"](S2, "R49hm", V, vtr, "cn90", 2)
    return dict(S2=S2, V=V, vtr=vtr, TH=THc, GB=S2["GB"], TRX=S2["TRX"], k2=np.minimum(2, S2["nsz"]))
def seq_sim(ST, u, Vf, THf, dl, Y, tg, k2):
    """check in the loop, time order per patient: running score u minus dl while a check-derived certificate holds (an EARLIER endoscopy whose check found
    both top samples quiet, <= Y years before); flag0 = score >= t; the check runs at every flag0 (k2 samples) and cancels the flag if quiet (V < bar).
    Vectorised over the threshold grid tg. Returns final flags (E, nT) and checks run (E, nT)"""
    VAL = ST["VAL"]; TY = ST["TY"]; Um = to_mat(ST, u); Vm = to_mat(ST, np.where(np.isfinite(Vf), Vf, np.inf)); P_, W = VAL.shape; nT = len(tg)
    cu = np.full((P_, nT), -np.inf); FF = np.zeros((P_, W, nT), bool); F0 = np.zeros((P_, W, nT), bool)
    for w in range(W):
        v = VAL[:, w]; scw = np.where(v, Um[:, w], -np.inf)[:, None] - dl * (TY[:, w][:, None] <= cu + 1e-9); f0 = v[:, None] & (scw >= tg[None, :]); qt = (np.where(v, Vm[:, w], np.inf) < THf)[:, None]
        FF[:, w] = f0 & ~qt; F0[:, w] = f0; cu = np.where(f0 & qt, TY[:, w][:, None] + Y, cu)
    E = len(u); o1 = np.zeros((E, nT), bool); o2 = np.zeros((E, nT), bool); o1[ST["PADI"][VAL]] = FF[VAL]; o2[ST["PADI"][VAL]] = F0[VAL]; return o1, o2
SEQC = {"SEQ0": [(h, 0.0, 0) for h in (1, 2, 4)], "SEQ": [(h, dl, Y) for h in (1, 2, 4) for dl in (0.5, 1.0) for Y in (2, 4)]}
# ------------------------------------------------------------------ the sweep engine (round-9 run_families selection at several targets)
def sweep(S, ST, INFO, fams, builders, CK=None):
    E = S["E"]; nT = len(TNAMES); R = {}; PATH = path_scores(S); BAND = np.full((E, 10), -1); DD = np.zeros((10, 10), int); NTP = np.zeros((10, 10), int)
    for f in fams: R[f] = dict(F=np.zeros((nT, E, 10), bool), NS=np.zeros((nT, E, 10), np.int16), U=None, chosen=np.zeros((nT, 100), int), labels=None, infeasible=np.zeros(nT, int), trainstats=np.zeros((nT, 100, 5)))
    for j in range(10):
        for k in range(1, 11):
            c = make_ctx(S, ST, j, k, INFO); c.PATH = PATH; te = c.te_e; BAND[te, j] = c.r49_band[te]; nTP = int((c.trP & c.isP).sum()); DD[j, k - 1] = c.D; NTP[j, k - 1] = nTP
            dl_ = [c.D] + [max(1, nTP - kk) for kk in KS]; fi = j * 10 + k - 1; nTN = int((c.trP & ~c.isP).sum()); nl_ = [max(1, int(np.ceil(r * nTN - 1e-9))) for r in NPS]
            trPm = c.trP & c.isP; trNm = c.trP & ~c.isP; INC = S["INC"]
            for f in fams:
                Rf = R[f]
                if f in SEQC:   # check in the loop (mean-logit memory + check, with or without a check-derived certificate)
                    s, ns = base(c, "img"); Vf = np.full(E, np.nan); tx = CK["TRX"][(j, k)]; Vf[tx["idx"]] = CK["vtr"][(j, k)]; Vf[te] = CK["V"][te, j]; THf = CK["TH"][te, j][0]
                    cands = SEQC[f]; Rf["labels"] = [f"ew h={h}y, cert -{dl} logit, Y={Y}y" if dl else f"ew h={h}y, no certificate" for h, dl, Y in cands]; stats = []; sims = []
                    for q, (h, dl, Y) in enumerate(cands):
                        u = memory(ST, s, "ew", h); tg = np.unique(np.r_[u[c.tr_e], u[c.tr_e] - dl])[::-1]; FFt, F0t = seq_sim(ST, u, Vf, THf, dl, Y, tg, CK["k2"])
                        PF = (INC @ FFt.astype(float)) > 0; det = PF[trPm].sum(0); npf = PF[trNm].sum(0); nfe = (FFt & (c.tr_e & (c.elab == 0))[:, None]).sum(0); nsm = (F0t * CK["k2"][:, None])[c.tr_e].mean(0)
                        sims.append((FFt, F0t, det, npf, nfe, nsm))
                    for ti, d in enumerate(dl_):
                        best = None
                        for q, (FFt, F0t, det, npf, nfe, nsm) in enumerate(sims):
                            ok = np.where(det >= d)[0]; ii = ok[0] if len(ok) else int(np.argmax(det)); key = (0 if len(ok) else 1, npf[ii] if len(ok) else -det[ii], nfe[ii], nsm[ii], q)
                            if best is None or key < best[0]: best = (key, q, ii)
                        _, q, ii = best; FFt, F0t, det, npf, nfe, nsm = sims[q]; Rf["chosen"][ti, fi] = q; Rf["infeasible"][ti] += int(best[0][0])
                        Rf["F"][ti][:, j][te] = FFt[te, ii]; Rf["NS"][ti][:, j][te] = (F0t[:, ii] * CK["k2"])[te]; Rf["trainstats"][ti, fi] = [det[ii], npf[ii], nfe[ii], nsm[ii], d]
                    for ui, m_ in enumerate(nl_):
                        ti = len(dl_) + ui; best = None
                        for q, (FFt, F0t, det, npf, nfe, nsm) in enumerate(sims):
                            ok = np.where(npf >= m_)[0]; ii = ok[0] if len(ok) else len(npf) - 1; key = (-det[ii], npf[ii], nfe[ii], nsm[ii], q)
                            if best is None or key < best[0]: best = (key, q, ii)
                        _, q, ii = best; FFt, F0t, det, npf, nfe, nsm = sims[q]; Rf["chosen"][ti, fi] = q
                        Rf["F"][ti][:, j][te] = FFt[te, ii]; Rf["NS"][ti][:, j][te] = (F0t[:, ii] * CK["k2"])[te]; Rf["trainstats"][ti, fi] = [det[ii], npf[ii], nfe[ii], nsm[ii], -m_]
                    continue
                cands = builders[f](c); nC = len(cands)
                if Rf["U"] is None: Rf["U"] = np.full((nC, E, 10), np.nan, np.float32); Rf["labels"] = [x["label"] for x in cands]
                PM = []; PN = []
                for q, x in enumerate(cands):
                    assert x["kind"] == "score"; Rf["U"][q][:, j][te] = x["u"][te]; pm = pat_scores(c, x["u"]); pp = np.sort(pm[trPm])[::-1]; PM.append(pp[pp > NEG / 2]); pn = np.sort(pm[trNm])[::-1]; PN.append(pn[pn > NEG / 2])
                for ti, d in enumerate(dl_):
                    ts_ = []; fl_ = []
                    for q, x in enumerate(cands):
                        pp = PM[q]; dd = min(d, len(pp)); t = pp[dd - 1] if dd >= 1 else np.inf; flag = x["u"] >= t
                        det = c.patient_det(flag); inc = INC @ flag.astype(float) > 0; npf = int((inc & trNm).sum()); nfe = int((flag & c.tr_e & (c.elab == 0)).sum()); nsm = float(x["ns"][c.tr_e].mean())
                        ts_.append((det, npf, nfe, nsm)); fl_.append(flag)
                    ts_ = np.array(ts_); feas = ts_[:, 0] >= d
                    if feas.any(): pick = sorted([q for q in range(nC) if feas[q]], key=lambda q: (ts_[q, 1], ts_[q, 2], ts_[q, 3], q))[0]
                    else: Rf["infeasible"][ti] += 1; pick = sorted(range(nC), key=lambda q: (-ts_[q, 0], ts_[q, 1], ts_[q, 2], q))[0]
                    Rf["chosen"][ti, fi] = pick; Rf["F"][ti][:, j][te] = fl_[pick][te]; Rf["NS"][ti][:, j][te] = cands[pick]["ns"][te]; Rf["trainstats"][ti, fi] = list(ts_[pick]) + [d]
                for ui, m_ in enumerate(nl_):   # NP-share targets: highest threshold flagging >= m_ training NP patients; candidate with the most training progressor patients caught
                    ti = len(dl_) + ui; ts_ = []; fl_ = []
                    for q, x in enumerate(cands):
                        pn = PN[q]; t = pn[min(m_, len(pn)) - 1] if len(pn) else -np.inf; flag = x["u"] >= t
                        det = c.patient_det(flag); inc = INC @ flag.astype(float) > 0; ts_.append((det, int((inc & trNm).sum()), int((flag & c.tr_e & (c.elab == 0)).sum()), float(x["ns"][c.tr_e].mean()))); fl_.append(flag)
                    ts_ = np.array(ts_); pick = sorted(range(nC), key=lambda q: (-ts_[q, 0], ts_[q, 1], ts_[q, 2], ts_[q, 3], q))[0]
                    Rf["chosen"][ti, fi] = pick; Rf["F"][ti][:, j][te] = fl_[pick][te]; Rf["NS"][ti][:, j][te] = cands[pick]["ns"][te]; Rf["trainstats"][ti, fi] = list(ts_[pick]) + [-m_]
    return R, BAND, DD, NTP
# ================================================================== run_<variant>_<group>
if ETASK.startswith("run_"):
    _, var, grp = ETASK.split("_", 2); t0 = time.time(); S, ST, INFO = state(var); CK = None
    if grp == "seq": CK = check_arrays(); assert CK["S2"]["E"] == S["E"] and (CK["S2"]["plab"] == S["plab"]).all(); fams = list(SEQC); builders = {}
    else: builders = GROUPS9[grp]; fams = list(builders)
    R, BAND, DD, NTP = sweep(S, ST, INFO, fams, builders, CK); arrs = {"BAND": BAND, "D": DD, "NTP": NTP}; meta = {}
    for f, Rf in R.items():
        key = f.replace(":", "_"); arrs.update({f"{key}__F": Rf["F"], f"{key}__NS": Rf["NS"], f"{key}__chosen": Rf["chosen"], f"{key}__train": Rf["trainstats"], f"{key}__infeasible": Rf["infeasible"]})
        if Rf["U"] is not None: arrs[f"{key}__U"] = Rf["U"]
        meta[f] = {"key": key, "labels": Rf["labels"]}
    np.savez_compressed(f"{XD9}/sweep_{var}_{grp}.npz", **arrs); json.dump({"meta": meta, "targets": TNAMES, "seconds": round(time.time() - t0, 1)}, open(f"{XD9}/sweep_{var}_{grp}.json", "w"))
    print("EXE9 DONE", ETASK, round(time.time() - t0, 1), "s", flush=True)
# ================================================================== merge: curves, bands, matched points, operating points, prevalence, break-even
def rates(S, A):
    """per multiplicity row (observed first, then the draws) class-specific per-patient rates, repeat-averaged as rv8 lstats (rv8b parts)"""
    p = parts(S, A, MB)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = {"sens": p["caught"] / p["nP"], "fpr": p["npfl"] / p["nN"], "nsp": p["NSP"] / p["nP"], "nsn": p["NSN"] / p["nN"], "nfp": p["NFP"] / p["nP"], "nfn": p["NFN"] / p["nN"],
             "nep": p["NEP"] / p["nP"], "nen": p["NEN"] / p["nN"], "pyn": p["PYN"] / p["nN"], "wP": p["nP"] / (p["nP"] + p["nN"])}
    o = {k: np.nanmean(v, 1).astype(np.float64) for k, v in r.items()}; o["tf"] = float(np.nanmean(A["TFmed"])); return o
def env(x, y):
    o = np.lexsort((y, x)); return x[o], np.maximum.accumulate(y[o])
def x_at(x, y, c):
    """fewest non-progressors flagged at which the (monotone envelope of the) swept curve reaches c progressors caught"""
    xs, ye = env(x, y); i = np.argmax(ye >= c - 1e-9)
    if ye[i] < c - 1e-9: return np.nan
    if i == 0: return xs[0]
    return xs[i - 1] + (xs[i] - xs[i - 1]) * (c - ye[i - 1]) / max(ye[i] - ye[i - 1], 1e-12)
def y_at(x, y, x0):
    """caught at x0 non-progressors flagged on the monotone envelope (x0 scalar or array; NaN outside the swept range)"""
    xs, ye = env(x, y); x0a = np.atleast_1d(np.asarray(x0, float)); v = np.interp(x0a, xs, ye); v[(x0a < xs[0] - 1e-9) | (x0a > xs[-1] + 1e-9)] = np.nan
    return v if np.ndim(x0) else float(v[0])
XG = np.round(np.arange(0, 43.01, 0.25), 2)
if ETASK == "merge":
    S, ST, INFO = state("st"); D = pdraws(S, NB9); MB = np.vstack([np.ones((1, S["P_"])), D]); nP = int((S["plab"] == 1).sum()); nN = int((S["plab"] == 0).sum()); E = S["E"]
    CK = check_arrays(); assert CK["S2"]["E"] == E and (CK["S2"]["plab"] == S["plab"]).all(); V = CK["V"]; THc = CK["TH"]; k2 = CK["k2"][:, None]
    Z = {}; META = {}
    for f in sorted(glob.glob(XD9 + "/sweep_*.npz")):
        nm = os.path.basename(f)[6:-4]; Z[nm] = np.load(f); META[nm] = json.load(open(f[:-4] + ".json"))
    assert (Z["st_core"]["BAND"] == CK["GB"]).all(), "engine gate band differs from r49_calls"
    def arr(grp, fam):
        key = fam.replace(":", "_"); return (Z[grp][f"{key}__F"], Z[grp][f"{key}__NS"].astype(int)) if grp in Z and f"{key}__F" in Z[grp].files else None
    def with_check(FN, band):
        """E2 / story_aggs_v2 final-rule construction: cancel a flag when both top-H&E samples are quiet (ps_oac < cn90 bar); k2 samples sequenced at every flag
        (k2 - 1 in the gate's middle band, whose top sample is already sequenced); band None = no gate"""
        if FN is None: return None
        F, NS = FN; veto = F & (V[None] < THc[None]); kk = k2[None] if band is None else np.where(band[None] == 1, k2[None] - 1, k2[None])
        return F & ~veto, NS + np.where(F, kk, 0)
    Bst = Z["st_core"]["BAND"]; Bml = Z["ml_core"]["BAND"]
    SW = {"hne_max": ("H&E max, single endoscopy", arr("st_core", "img:none")), "hne_mean": ("H&E mean-logit, single endoscopy", arr("ml_core", "img:none")),
          "gate": ("Today's gate (R49-aligned score)", arr("st_core", "gate_al:none")), "memory": ("Memory (gate score, EW)", arr("st_core", "gate_al:ew")),
          "memcert": ("Memory + certificate", arr("st_A12", "A12")), "final": ("Final rule (memory + certificate + check)", with_check(arr("st_A12", "A12"), Bst)),
          "memcheck": ("Memory + check (no certificate)", with_check(arr("st_core", "gate_al:ew"), Bst)), "hne_max_mem": ("H&E max + memory", arr("st_core", "img:ew")),
          "ml_mem": ("Mean-logit + memory", arr("ml_core", "img:ew")), "ml_gate": ("Mean-logit gate", arr("ml_core", "gate_al:none")), "ml_gatemem": ("Mean-logit gate + memory", arr("ml_core", "gate_al:ew")),
          "ml_memcert": ("Mean-logit gate + memory + certificate", arr("ml_A12", "A12")), "ml_final": ("Mean-logit full stack (gate + memory + certificate + check)", with_check(arr("ml_A12", "A12"), Bml)),
          "ml_memcheck": ("Mean-logit + memory + check (no gate)", with_check(arr("ml_core", "img:ew"), None)),
          "ml_seq0": ("Mean-logit + memory + check, threshold chosen with the check", arr("ml_seq", "SEQ0")), "ml_seq": ("Mean-logit + memory + check + check-derived certificate", arr("ml_seq", "SEQ")),
          "st_seq0": ("H&E max + memory + check, threshold chosen with the check", arr("st_seq", "SEQ0")), "st_seq": ("H&E max + memory + check + check-derived certificate", arr("st_seq", "SEQ")),
          "grade_none": ("Grade score, single endoscopy (swept)", arr("st_path", "grade:none")), "grade_mean": ("Grade, running mean", arr("st_path", "grade:mean")), "grade_ew": ("Grade, EW memory", arr("st_path", "grade:ew")),
          "g53_none": ("Grade + p53, single endoscopy (swept)", arr("st_path", "g53:none")), "g53_mean": ("Grade + p53, running mean", arr("st_path", "g53:mean")), "g53_ew": ("Grade + p53, EW memory", arr("st_path", "g53:ew"))}
    SW = {k: v for k, v in SW.items() if v[1] is not None}
    PS = path_scores(S); Zi = np.zeros((E, 10), int); rep = lambda b: np.repeat(b[:, None], 10, 1); RF = refs(S)
    FX = {"R49": ("Today's gate (R49)", RF["R49"], None), "P1": ("H&E alone, P1 90%", RF["P1 90%"], None), "grade": ("Pathologist grade IND/LGD ever", (rep(PS["grade"] >= 1), Zi), None),
          "lgd": ("LGD ever", (rep(PS["grade"] >= 2), Zi), None), "p53": ("p53 aberrant ever", (rep(PS["p53ab"]), Zi), None), "g53": ("Grade IND/LGD or p53 aberrant (p53 missing = normal)", (rep(PS["g53"] >= 1), Zi), None),
          "g53_lgd": ("LGD or p53 aberrant", (rep(PS["g53"] >= 2), Zi), None), "g53_subset": ("Grade or p53, endoscopies with p53 (round-8 P10)", (rep(PS["g53_subset"]), Zi), PS["p53_any"])}
    G53 = rep(PS["g53"] >= 1)
    for nm, src in (("final", "final"), ("memory", "memory"), ("ml_mem", "ml_mem"), ("R49", None)):
        Fb, NSb = (SW[src][1][0][0], SW[src][1][1][0]) if src else RF["R49"]; FX[f"{nm}_or_g53"] = (f"{SW[src][0] if src else 'R49'} OR grade/p53", (Fb | G53, NSb), None)
    t0 = time.time(); RT = {}; CUR = {}
    for nm, (lab, (F, NS)) in SW.items():
        RT[nm] = [rates(S, patient_arrays(S, F[t], NS[t])) for t in range(F.shape[0])]
    for nm, (lab, (F, NS), em) in FX.items(): RT[nm] = [rates(S, patient_arrays(S, F, NS, em))]
    print("rates", round(time.time() - t0, 1), flush=True)
    xy = lambda r: (nN * r["fpr"], nP * r["sens"])
    out = {"note": ("held-out predictions; every threshold / hyper-parameter chosen on training patients (round-9 operating rule at targets D and nTP - k); patient bootstrap "
                    f"{NB9} draws (rv8 pdraws, RandomState(0)), paired; counts = share x 32 / x 43; bands = vertical percentile bands of the monotone envelope on a common x grid; "
                    "costs: GBP 85 per sample sequenced, GBP 535 per flagged endoscopy (round 8)"), "targets": TNAMES, "n_draws": int(len(D))}
    # ---- operating points (target D) and reproduction
    def ci(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return [r3(np.percentile(a, 2.5)), r3(np.percentile(a, 97.5))] if len(a) > 0.9 * len(D) else None   # undefined when > 10% of draws never reach the point
    def opv(r):
        x, y_ = xy(r); seq = r["wP"] * r["nsp"] + (1 - r["wP"]) * r["nsn"]; fl = r["wP"] * r["nfp"] + (1 - r["wP"]) * r["nfn"]
        return {"caught": y_, "np_flagged": x, "seq_per_patient": seq, "flags_per_patient": fl, "cost_per_patient": COST_SEQ * seq + COST_FLAG * fl, "flags_per_np_py": r["nfn"] / r["pyn"], "endo_spec": 1 - r["nfn"] / r["nen"], "endo_sens": r["nfp"] / r["nep"]}
    OP = {}
    for nm in list(SW) + list(FX):
        r = RT[nm][0]; v = opv(r); OP[nm] = {"label": SW[nm][0] if nm in SW else FX[nm][0], **{k: r3(a[0]) for k, a in v.items()}, "ci95": {k: ci(a[1:]) for k, a in v.items() if k in ("caught", "np_flagged", "seq_per_patient", "cost_per_patient")},
                                              "years_first_flag_to_event_median": r3(r["tf"])}
        for ref in ("R49", "P1"):
            vr = opv(RT[ref][0]); OP[nm][f"vs_{ref}"] = {k: dstat(float(v[k][0] - vr[k][0]), v[k][1:] - vr[k][1:]) for k in ("np_flagged", "caught", "seq_per_patient", "cost_per_patient", "endo_spec")}
    out["operating_points_target_D"] = OP
    out["reproduction"] = {"expected": {"R49": [31.1, 31.8, 1.94], "P1": [31.1, 36.0, 0.0], "memory": [30.7, 23.6, 1.94], "final": [30.5, 17.2, 3.82], "ml_mem": [31.0, 26.6, 0.0], "ml_gatemem": [30.5, 22.6, 1.433], "hne_max": [31.5, 36.0, 0.0], "hne_mean": [30.9, 31.0, 0.0]},
                           "observed": {k: [OP[k]["caught"], OP[k]["np_flagged"], OP[k]["seq_per_patient"]] for k in ("R49", "P1", "memory", "final", "ml_mem", "ml_gatemem", "hne_max", "hne_mean", "memcert") if k in OP}}
    out["selected_candidates_target_D"] = {}
    for grp in Z:
        for f_, m_ in META[grp]["meta"].items():
            ch = Z[grp][m_["key"] + "__chosen"][0]; out["selected_candidates_target_D"][f"{grp}:{f_}"] = {m_["labels"][q]: int((ch == q).sum()) for q in np.unique(ch)}
    out["fold_D_and_nTP"] = {"D_mean": r3(Z["st_core"]["D"].mean()), "nTP_mean": r3(Z["st_core"]["NTP"].mean()), "ml_D_mean": r3(Z["ml_core"]["D"].mean())}
    # ---- curves, bands, matched points
    CUR = {}; MATCH = {}
    for nm in SW:
        pts = np.array([[nN * r["fpr"][0], nP * r["sens"][0]] for r in RT[nm]]); XB = np.stack([nN * r["fpr"][1:] for r in RT[nm]], 1); YB = np.stack([nP * r["sens"][1:] for r in RT[nm]], 1)
        yo = y_at(pts[:, 0], pts[:, 1], XG); YG = np.array([y_at(XB[b], YB[b], XG) for b in range(len(D))])
        ok = np.isfinite(YG).mean(0) >= 0.9; lo = np.where(ok, np.nanpercentile(np.where(np.isfinite(YG), YG, np.nan), 2.5, 0), np.nan); hi = np.where(ok, np.nanpercentile(np.where(np.isfinite(YG), YG, np.nan), 97.5, 0), np.nan)
        CUR[nm] = {"label": SW[nm][0], "points": [{"target": TNAMES[t], "np_flagged": r3(pts[t, 0]), "caught": r3(pts[t, 1]), "seq_per_patient": r3(opv(RT[nm][t])["seq_per_patient"][0])} for t in range(len(pts))],
                   "band_x": XG.tolist(), "band_obs": [r3(v) for v in yo], "band_lo": [r3(v) for v in lo], "band_hi": [r3(v) for v in hi]}
        MATCH[nm] = {"pts": pts, "XB": XB, "YB": YB}
    out["curves"] = CUR
    r49 = RT["R49"][0]; cR = nP * r49["sens"]; xR = nN * r49["fpr"]; mt = {}
    for nm, m in MATCH.items():
        o = {}
        for c in (31.1, 30.5, 30.0, 29.0):
            ob = x_at(m["pts"][:, 0], m["pts"][:, 1], c); bd = np.array([x_at(m["XB"][b], m["YB"][b], c) for b in range(len(D))]); o[f"np_flagged_at_caught_{c}"] = {"value": r3(ob), "ci95": ci(bd)}
            for ref in ("gate", "hne_max"):
                rm = MATCH[ref]; ro = x_at(rm["pts"][:, 0], rm["pts"][:, 1], c); rb = np.array([x_at(rm["XB"][b], rm["YB"][b], c) for b in range(len(D))]); dd = bd - rb
                o[f"np_flagged_at_caught_{c}"][f"vs_{ref}"] = {"delta": r3(ob - ro), "ci95": ci(dd)}
        ob = x_at(m["pts"][:, 0], m["pts"][:, 1], cR[0]); bd = np.array([x_at(m["XB"][b], m["YB"][b], cR[1 + b]) for b in range(len(D))])
        o["np_flagged_at_R49_caught_minus_R49"] = {"delta": r3(ob - xR[0]), "ci95": ci(bd - xR[1:])}
        for x0 in (17.2, 21.5, 23.6, 26.6, 31.8):
            ob = y_at(m["pts"][:, 0], m["pts"][:, 1], x0); bd = np.array([y_at(m["XB"][b], m["YB"][b], x0) for b in range(len(D))]); o[f"caught_at_np_{x0}"] = {"value": r3(ob), "ci95": ci(bd)}
            gm = MATCH["gate"]; go = y_at(gm["pts"][:, 0], gm["pts"][:, 1], x0); gb = np.array([y_at(gm["XB"][b], gm["YB"][b], x0) for b in range(len(D))]); o[f"caught_at_np_{x0}"]["vs_gate"] = {"delta": r3(ob - go), "ci95": ci(bd - gb)}
        mt[nm] = o
    out["matched"] = mt
    # ---- does memory help each base? paired difference memory - single-endoscopy on the same base, at matched points
    mg = {}
    for a, b in (("memory", "gate"), ("ml_mem", "hne_mean"), ("hne_max_mem", "hne_max"), ("grade_ew", "grade_none"), ("grade_mean", "grade_none"), ("g53_ew", "g53_none"), ("g53_mean", "g53_none"), ("ml_gatemem", "ml_gate")):
        if a not in MATCH or b not in MATCH: continue
        A_, B_ = MATCH[a], MATCH[b]; o = {}
        for x0 in (10.0, 16.0, 17.2, 23.6, 25.0, 31.8):
            ao = y_at(A_["pts"][:, 0], A_["pts"][:, 1], x0); bo = y_at(B_["pts"][:, 0], B_["pts"][:, 1], x0); ab = np.array([y_at(A_["XB"][q], A_["YB"][q], x0) for q in range(len(D))]); bb = np.array([y_at(B_["XB"][q], B_["YB"][q], x0) for q in range(len(D))])
            o[f"d_caught_at_np_{x0}"] = {"memory": r3(ao), "single": r3(bo), "delta": r3(ao - bo), "ci95": ci(ab - bb)}
        for c in (14.0, 17.0, 21.0, 30.5, 31.1):
            ao = x_at(A_["pts"][:, 0], A_["pts"][:, 1], c); bo = x_at(B_["pts"][:, 0], B_["pts"][:, 1], c); ab = np.array([x_at(A_["XB"][q], A_["YB"][q], c) for q in range(len(D))]); bb = np.array([x_at(B_["XB"][q], B_["YB"][q], c) for q in range(len(D))])
            o[f"d_np_at_caught_{c}"] = {"memory": r3(ao), "single": r3(bo), "delta": r3(ao - bo), "ci95": ci(ab - bb)}
        mg[f"{a} - {b}"] = o
    out["memory_gain_by_base"] = mg
    # ---- apparent (OPTIMISTIC) curves: per repeat, the held-out patient maximum of the modal candidate's score swept on the evaluated patients
    app = {}; isPp = S["plab"] == 1; INC = S["INC"]
    for nm, grp, fam in (("hne_max", "st_core", "img:none"), ("gate", "st_core", "gate_al:none"), ("memory", "st_core", "gate_al:ew"), ("memcert", "st_A12", "A12"), ("ml_mem", "ml_core", "img:ew"), ("hne_mean", "ml_core", "img:none"), ("grade_ew", "st_path", "grade:ew"), ("g53_ew", "st_path", "g53:ew")):
        key = fam.replace(":", "_"); U = Z[grp][key + "__U"]; q = int(np.bincount(Z[grp][key + "__chosen"][0]).argmax()); cur = []
        for j in range(10):
            Pm = np.where(INC > 0, U[q][:, j][None, :], -np.inf).max(1); th = np.unique(Pm)[::-1]; xs = np.array([nN * (Pm[~isPp] >= t).mean() for t in th]); ys = np.array([nP * (Pm[isPp] >= t).mean() for t in th])
            cur.append(y_at(xs, ys, XG))
        app[nm] = {"candidate": META[grp]["meta"][fam]["labels"][q], "y": [r3(v) for v in np.nanmean(np.array(cur, float), 0)]}
    out["apparent_curves_OPTIMISTIC"] = {"x": XG.tolist(), "curves": app, "note": "threshold swept on the evaluated (held-out) patients, per repeat, modal training-selected candidate; optimistic"}
    # ---- prevalence reweighting and break-even (target D operating points)
    PRULES = ["g53", "grade", "R49", "P1", "memory", "memcert", "final", "ml_mem", "ml_gatemem", "ml_memcert", "ml_final", "ml_memcheck", "ml_seq0", "ml_seq", "memcheck", "final_or_g53", "ml_mem_or_g53", "memory_or_g53", "R49_or_g53", "grade_ew", "g53_ew"]
    PRULES = [p for p in PRULES if p in RT]
    smpP = np.array([S["nsz"][S["pidx"] == i].sum() for i in range(S["P_"])]); spP = smpP[S["plab"] == 1].mean(); spN = smpP[S["plab"] == 0].mean()
    PREVS = {"1.5%": 0.015, "3%": 0.03, "5%": 0.05, "10%": 0.10, "cohort (43%)": None}
    def per1000(r, pi):
        pi = r["wP"] if pi is None else pi; tp = 1000 * pi * r["sens"]; fp = 1000 * (1 - pi) * r["fpr"]; fn = 1000 * pi - tp; tn = 1000 * (1 - pi) - fp
        with np.errstate(invalid="ignore", divide="ignore"):
            return {"patients_flagged": tp + fp, "progressors_caught": tp, "progressors_missed": fn, "non_progressors_flagged": fp, "non_progressors_never_flagged": tn, "ppv": tp / (tp + fp), "npv": tn / (tn + fn),
                    "flagged_per_progressor_caught": (tp + fp) / tp, "flagged_endoscopies": 1000 * (pi * r["nfp"] + (1 - pi) * r["nfn"]), "samples_sequenced": 1000 * (pi * r["nsp"] + (1 - pi) * r["nsn"]),
                    "endoscopies": 1000 * (pi * r["nep"] + (1 - pi) * r["nen"]), "samples_scored": 1000 * (pi * spP + (1 - pi) * spN) + 0 * tp}
    PV = {}
    for pn, pi in PREVS.items():
        PV[pn] = {}
        for nm in PRULES:
            v = per1000(RT[nm][0], pi); PV[pn][nm] = {k: {"value": r3(a[0]), "ci95": ci(a[1:])} for k, a in v.items()}
    out["per_1000_patients"] = {"prevalences": PV, "note": ("per 1,000 patients over the cohort's follow-up (non-progressors: median 5 pre-event endoscopies, progressors 2.5), class-specific per-patient rates of the held-out "
                                                            "rule reweighted to the progression share; 'flagged_endoscopies' = extra endoscopies if every flagged endoscopy triggers one; 'non_progressors_never_flagged' = de-escalation candidates; "
                                                            "progressor-share column 'cohort' uses the case-control mix")}
    BE = {}
    CSEQ = (85.0, 150.0, 250.0); CMISS = (0.0, 20000.0, 40000.0)
    for pn, pi in PREVS.items():
        BE[pn] = {}
        for nm in PRULES:
            if nm in ("g53", "R49"): continue
            v = per1000(RT[nm][0], pi); BE[pn][nm] = {}
            for ref in ("R49", "g53"):
                w = per1000(RT[ref][0], pi); dS = v["samples_sequenced"] - w["samples_sequenced"]; dM = v["progressors_missed"] - w["progressors_missed"]; o = {"d_samples": {"value": r3(dS[0]), "ci95": ci(dS[1:])}, "d_missed": {"value": r3(dM[0]), "ci95": ci(dM[1:])}}
                for conv, key in (("per_flagged_endoscopy", "flagged_endoscopies"), ("once_per_flagged_patient", "patients_flagged")):
                    dF = v[key] - w[key]; oc = {"d_flags": {"value": r3(dF[0]), "ci95": ci(dF[1:])}}
                    for cm in CMISS:
                        with np.errstate(invalid="ignore", divide="ignore"):
                            bs = np.where(dS > 1e-9, -(COST_FLAG * dF + cm * dM) / dS, np.nan); bf = np.where(dF < -1e-9, -(COST_SEQ * dS + cm * dM) / dF, np.nan)
                        dc = {f"{int(cs)}": cs * dS + COST_FLAG * dF + cm * dM for cs in CSEQ}
                        oc[f"miss_{int(cm)}"] = {"breakeven_seq_price_gbp": {"value": r2(bs[0]), "ci95": ci(bs[1:]), "share_draws_defined": r3(np.isfinite(bs[1:]).mean())},
                                                 "breakeven_flag_cost_gbp": {"value": r2(bf[0]), "ci95": ci(bf[1:]), "share_draws_defined": r3(np.isfinite(bf[1:]).mean())},
                                                 "d_cost_per_1000_gbp_at_seq_price": {k: {"value": r2(a[0]), "ci95": ci(a[1:]), "p_saves": r3((a[1:] < 0).mean())} for k, a in dc.items()}}
                    if ref == "g53":   # AI scanning / scoring cost on every sample (sensitivity)
                        for cai in (15.0,):
                            a = 85 * dS + COST_FLAG * dF + cai * v["samples_scored"]; oc[f"d_cost_per_1000_with_ai_{int(cai)}_per_sample_miss_0"] = {"value": r2(a[0]), "ci95": ci(a[1:])}
                    o[conv] = oc
                BE[pn][nm][f"vs_{ref}"] = o
    out["break_even"] = {"results": BE, "note": ("rule minus comparator per 1,000 patients; break-even sequencing price = -(flag cost x d_flags + miss cost x d_missed) / d_samples (the rule saves below it; defined when it sequences more); "
                                                 "break-even flag cost = -(GBP 85 x d_samples + miss cost x d_missed) / d_flags (the rule saves above it; defined when it flags less); flag = GBP 535; conventions: every flagged endoscopy "
                                                 "costs one action, or one pathway change per flagged patient; comparators: today's gate (R49) and standard care = grade IND/LGD or p53 aberrant (p53 missing = normal, no sequencing)")}
    # ---- tiered menu (target D): incremental cost per false flag avoided and per progressor lost, against the no-sequencing tier and against R49
    MENU = [m for m in ("ml_mem", "ml_gatemem", "memory", "memcert", "ml_memcert", "ml_memcheck", "ml_seq0", "ml_seq", "final", "ml_final", "R49", "P1", "g53") if m in RT]; TM = {}
    for nm in MENU:
        v = opv(RT[nm][0]); TM[nm] = {"label": OP[nm]["label"], "caught": OP[nm]["caught"], "np_flagged": OP[nm]["np_flagged"], "seq_per_patient": OP[nm]["seq_per_patient"], "cost_per_patient": OP[nm]["cost_per_patient"]}
        for ref in ("ml_mem", "R49"):
            w = opv(RT[ref][0]); dC = v["cost_per_patient"] - w["cost_per_patient"]; dN = v["np_flagged"] - w["np_flagged"]; dP = v["caught"] - w["caught"]; dS = v["seq_per_patient"] - w["seq_per_patient"]
            with np.errstate(invalid="ignore", divide="ignore"):
                TM[nm][f"vs_{ref}"] = {"d_cost_per_patient": dstat(float(dC[0]), dC[1:]), "d_np_flagged": dstat(float(dN[0]), dN[1:]), "d_caught": dstat(float(dP[0]), dP[1:]), "d_seq_per_patient": dstat(float(dS[0]), dS[1:]),
                                       "gbp_per_np_flag_avoided_cohort_75": r2(75 * dC[0] / -dN[0]) if dN[0] < -0.05 else None, "gbp_per_progressor_lost_cohort_75": r2(75 * dC[0] / -dP[0]) if dP[0] < -0.05 else None,
                                       "seq_samples_per_np_flag_avoided": r2(75 * dS[0] / -dN[0]) if dN[0] < -0.05 else None}
    out["tiered_menu"] = TM
    out["per_class_rates_target_D"] = {nm: {k: r3(RT[nm][0][k][0]) for k in ("sens", "fpr", "nsp", "nsn", "nfp", "nfn", "nep", "nen", "pyn")} for nm in PRULES}
    out["samples_per_patient_scored"] = {"progressor": r3(spP), "non_progressor": r3(spN)}
    out["p53_availability"] = {"endoscopies_with_p53": int(PS["p53_any"].sum()), "endoscopies": int(E), "endoscopies_p53_aberrant": m5(int(PS["p53ab"].sum()))}
    jdump9(out, "frontier")
