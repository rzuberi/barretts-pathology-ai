"""Two papers, agent P2N (docs/two_papers_P2N.md): selection-aware permutation null for Paper 2 (PAPER2_plan_v2.md new analysis 1).
For one label set (SHUF = 0 true labels, SHUF >= 1 a patient-level permutation from perms.csv) the whole rule search of round 9 and E2/E3/E7/E8/E9 is rerun
with its own code and every held-out cell is scored; the best cell under the selection criterion is taken in the merge.
Code reuse (executed unchanged, as ex_e7_stress.py / ex_e9_frontier.py do): ex_e2_veto.py up to its dispatch (which executes rv8_landscape.py: setup, triage, r49_calls,
patient_arrays, lstats; E2 build, vs_test, vs_train, veto_V, run_veto, double_V, FIXED, RULES) and rv9_falseflags.py up to its dispatch (statics, make_ctx, memory,
cert_flags, FAMS, run_families). The cell builders below copy the E2 task bodies (fixed, combo, grade, and, drift, stab, sel, mem), the E7 bar sweep / in-fold bar,
the E8 arm-set checks and the E3/E9 endoscopy aggregations (mean / median logit, mean probability, top-2 logit; full stack + check).
Label handling: every label-using step is rerun under the permuted labels (tb_common.y and the y of every executed namespace are replaced before build(); dates and
reference dates are cached from the true labels because they only feed lead-time and patient-year summaries, not the rules).
MODE a: L-IMG and L-CNV outer and inner predictions refitted under the permuted labels (tp_p2n_refit.R; folds re-stratified on the permuted labels);
MODE b: stored predictions (models trained on the true labels), only the rule search rerun.
Row-level outputs (cluster only): killcoyne_mm/two_papers_P2N/search/<MODE>_<KIND>_sXXXX.npz (per cell: caught, NP flagged, samples sequenced, per repeat).
TASK = perms | search (SHUF, MODE, KIND free|strat) | merge (tp_p2n_merge.py)"""
import os, sys, json, time, glob, numpy as np, pandas as pd
P = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"; sys.path.insert(0, P)
ETASK = os.environ["TASK"]
os.environ["TASK"] = "__tpp2n_e2__"; G2 = {"__file__": P + "/ex_e2_veto.py", "__name__": "ex_e2_p2n"}
_s = open(P + "/ex_e2_veto.py").read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], P + "/ex_e2_veto.py", "exec"), G2)
os.environ["TASK"] = "__tpp2n_rv9__"; G9 = {"__file__": P + "/rv9_falseflags.py", "__name__": "rv9_p2n"}
_s = open(P + "/rv9_falseflags.py").read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], P + "/rv9_falseflags.py", "exec"), G9)
os.environ["TASK"] = ETASK
g8 = G2["g8"]; TC = G2["TC"]; M = G2["M"]; T = G2["T"]; SM = G2["SM"]; ids = G2["ids"]; n = G2["n"]; REPS = G2["REPS"]; lg = G2["lg"]; thr_sens = G2["thr_sens"]
patient_arrays = G2["patient_arrays"]; lstats = G2["lstats"]; vs_test = G2["vs_test"]; vs_train = G2["vs_train"]; veto_V = G2["veto_V"]; double_V = G2["double_V"]; base_of = G2["base_of"]
LIMG = os.environ.get("LIMG", "stored"); assert LIMG in ("stored", "clean")   # clean (22 Oct): agent ART's clean-tile L-IMG (CLN's swap: L-IMG replaced, L-CNV stored, L-LATE their mean)
XD = M + "/two_papers_P2N"; SD_ = XD + ("/search" if LIMG == "stored" else "/search_clean")
for d_ in (XD, SD_): os.makedirs(d_, exist_ok=True)
y0 = TC.y.copy(); pat = TC.pat; PRE = TC.PRE; ix = TC.ix
DATES = g8["dates"](); g8["dates"] = lambda: DATES   # true-label dates (lead time / patient-years only)
def set_labels(ys):
    TC.y = ys; g8["y"] = ys; G2["y"] = ys; G9["y"] = ys
# ================================================================== permutations
if ETASK == "perms":
    PL = pd.Series(y0, index=pat).groupby(level=0).max(); prep = sorted(set(pat[PRE])); oth = sorted(set(PL.index) - set(prep))
    assert len(prep) == 75 and PL[prep].sum() == 32 and (PL[oth] == 1).all() and len(oth) == 5
    # endoscopy-count strata (E7 'strat'): pre-event endoscopies per patient
    ne = pd.Series(TC.ENDO[PRE], index=pat[PRE]).groupby(level=0).nunique().reindex(prep).values; strata = np.digitize(ne, [2, 3, 4, 5, 7])
    for kind, B in (("free", 1000), ("strat", 1000)):
        cols = {"Patient": list(PL.index), "s0000": PL.values.astype(int)}; base = PL[prep].values.astype(int)
        for b in range(1, B + 1):
            rng = np.random.RandomState(20261021 + b + (100000 if kind == "strat" else 0)); v = base.copy()
            if kind == "free": v = rng.permutation(v)
            else:
                for s_ in np.unique(strata): ii = np.where(strata == s_)[0]; v[ii] = rng.permutation(v[ii])
            lab = PL.copy(); lab[prep] = v; cols[f"s{b:04d}"] = lab.values.astype(int)
        pd.DataFrame(cols).to_csv(XD + ("/perms.csv" if kind == "free" else "/perms_strat.csv"), index=False)
    print("TPP2N DONE perms", flush=True); sys.exit(0)
SHUF = int(os.environ.get("SHUF", "0")); MODE = os.environ.get("MODE", "b"); KIND = os.environ.get("KIND", "free"); assert MODE in ("a", "b") and KIND in ("free", "strat")
assert not (MODE == "a" and KIND == "strat"), "the full-refit null uses the free permutations only"
OUTF = f"{SD_}/{MODE}_{KIND}_s{SHUF:04d}.npz"
if os.path.exists(OUTF) and not os.environ.get("FORCE"): print("TPP2N exists", OUTF); sys.exit(0)
# ------------------------------------------------------------------ labels and predictions
if SHUF > 0:
    PM = pd.read_csv(XD + ("/perms.csv" if KIND == "free" else "/perms_strat.csv"), dtype={"Patient": str}).set_index("Patient"); ys = PM[f"s{SHUF:04d}"].reindex(pat).values.astype(int)
else: ys = y0.copy()
set_labels(ys)
def load_refit(s):
    d = f"{XD}/refit/s{s:04d}"; O, I, FOs = {}, {}, []
    for mdl, key in (("cnv", "L-CNV"), ("img" if LIMG == "stored" else "imgclean", "L-IMG")):
        fo = sorted(glob.glob(f"{d}/{mdl}_outer_rep_*.csv.gz")); fi = sorted(glob.glob(f"{d}/{mdl}_inner_rep_*.csv.gz")); assert len(fo) == 10 and len(fi) == 10, (s, mdl, len(fo), len(fi))
        D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); O[key] = D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values
        FOs.append(D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int))
        Di = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fi]); assert Di.groupby(["rep", "fold"]).ngroups == 100
        I[key] = {(r, k): pd.Series(dd.pred.values, index=ix[dd.Sample].values) for (r, k), dd in Di.groupby(["rep", "fold"])}
    assert (FOs[0] == FOs[1]).all() and np.isfinite(O["L-IMG"]).all(); FO = FOs[0]
    O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2; I["L-LATE"] = {k: (I["L-CNV"][k] + I["L-IMG"][k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    return O, I, FO
def load_art_clean():
    """stored L-CNV with agent ART's clean-tile L-IMG refit (outer and inner, stored folds); L-LATE = mean (tp_cln_main.py swap)"""
    O, I, FO = TC.load("their"); d = M + "/two_papers_ART/refit/clean"; fo = sorted(glob.glob(d + "/outer_rep_*.csv")); fi = sorted(glob.glob(d + "/inner_rep_*.csv")); assert len(fo) == 10 and len(fi) == 10
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); Oi = D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values
    assert (D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int) == FO).all()
    Di = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fi]); Ii = {(r, k): pd.Series(dd.pred.values, index=ix[dd.Sample].values) for (r, k), dd in Di.groupby(["rep", "fold"])}
    O["L-IMG"] = Oi; O["L-LATE"] = (O["L-CNV"] + Oi) / 2; I["L-IMG"] = Ii; I["L-LATE"] = {k: (I["L-CNV"][k] + Ii[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}; return O, I, FO
O0, I0, FO0 = (TC.load("their") if LIMG == "stored" else load_art_clean()) if MODE == "b" else load_refit(SHUF)
# endoscopy aggregations of the sample H&E logit (E3 AGG; E3/E9 construction: every sample carries its endoscopy's score + 1e-6 x its own logit; L-LATE keeps the sample model)
lg_ = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda z: 1 / (1 + np.exp(-z))
ENDOPP = np.array([f"{e}|{int(p)}" for e, p in zip(TC.ENDO, PRE)]); _EG = pd.Series(np.arange(n)).groupby(ENDOPP).indices
AGG = {"ml": np.mean, "med": np.median, "mprob": lambda z, a: lg_(np.mean(sg(z), a)), "top2": lambda z, a: np.sort(z, axis=a)[-2:].mean(a) if np.shape(z)[a] >= 2 else np.max(z, a)}
def agg_state(how):
    f = AGG[how]; Oo, Io = O0["L-IMG"], I0["L-IMG"]; Oa = np.full((n, 10), np.nan)
    for e, m in _EG.items(): Oa[m] = sg(f(lg_(Oo[m]), 0))[None, :]
    Ia = {}
    for k, sr in Io.items():
        gg = pd.Series(lg_(sr.values), index=sr.index).groupby(ENDOPP[sr.index.values]).transform(lambda v: f(v.values, 0)); Ia[k] = pd.Series(sg(gg.values), index=sr.index)
    Oi = sg(lg_(Oa) + 1e-6 * lg_(Oo)); Ii = {k: pd.Series(sg(lg_(Ia[k].values) + 1e-6 * lg_(Io[k].reindex(Ia[k].index).values)), index=Ia[k].index) for k in Ia}
    O = dict(O0); I = dict(I0); O["L-IMG"] = Oi; O["L-LATE"] = (O["L-CNV"] + Oo) / 2; I["L-IMG"] = Ii; I["L-LATE"] = {k: (I["L-CNV"][k] + Io[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    return O, I
def build_state(O, I):
    g8["load"] = lambda src: (O, I, FO0); S = G2["build"](); ST = G9["statics"](S)
    assert int((S["plab"] == 1).sum()) == 32 and S["P_"] == 75 and S["E"] == 288; return S, ST
FT = G2["load_feat"]()
# ------------------------------------------------------------------ cell registry
CELLS = {}; ONE = None
def put(S, name, F, NS):
    A = patient_arrays(S, F, np.asarray(NS).astype(int)); isP = S["plab"] == 1; FL = A["FL"]
    CELLS[name] = np.stack([FL[isP].sum(0), FL[~isP].sum(0), A["NS"].sum(0) / S["P_"]])   # (3, 10): caught, NP flagged, samples per patient, per repeat
def inner_of(S, m, j, kk): x = np.full(n, np.nan); s_ = S["I"][m][(REPS[j], kk)]; x[s_.index.values] = s_.values; return x
def vcell(S, b, V, vtr, rule, k, vt=None, vtr_m=None):
    G2["TRAINSTATS"] = {}; F, NS, TH = veto_V(S, b, V, vtr, rule, k, ve_test=vt, ve_train=vtr_m); tr = G2["TRAINSTATS"]; G2["TRAINSTATS"] = None; return F, NS, TH, tr
def Vfix(S, x, k): return vs_test(S, x, k), {kk: vs_train(S, x, tx["top"], k) for kk, tx in S["TRX"].items()}
def Vmod(S, m, k): return vs_test(S, S["O"][m], k), {kk: vs_train(S, inner_of(S, m, *kk), tx["top"], k) for kk, tx in S["TRX"].items()}
GRv = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float)
# ================================================================== E2 (bases P1, R49h, R49hm on one state)
def e2_cells(S, tag):
    O = S["O"]; E = S["E"]; EF = S["EF"]; CAND = []
    sc = [(f, Vfix(S, FT[f].values.astype(float), 1), Vfix(S, FT[f].values.astype(float), 2), "quiet") for f in G2["FIXED"]] + [(m, Vmod(S, m, 1), Vmod(S, m, 2), "model") for m in ("L-CNV", "L-LATE")]
    for b in ("P1", "R49h", "R49hm"):   # fixed_<base> (648 cells over the three bases)
        for f, v1, v2, grp in sc:
            for k, (V, vtr) in ((1, v1), (2, v2)):
                for rule in G2["RULES"]:
                    F, NS, TH, tr = vcell(S, b, V, vtr, rule, k); put(S, f"{tag}|E2fixed|{b} veto {f} k{k} {rule}", F, NS); CAND.append(dict(b=b, name=f"{f} k{k} {rule}", grp=grp, F=F, NS=NS, tr=tr))
    EG = np.array([GRv[s].max() for s in S["es"]])
    for b in ("P1", "R49h", "R49hm"):   # sel_<base> candidates: trained models gated to NDBE/ID, double vetoes (the E2 sel task; elastic-net vetoes not rerun)
        vt = (EG <= 1)[:, None]; vtr_m = {kk: EG[tx["idx"]] <= 1 for kk, tx in S["TRX"].items()}
        for m in ("L-CNV", "L-LATE"):
            for k in (1, 2):
                V, vtr = Vmod(S, m, k)
                for rule in G2["RULES"]: F, NS, TH, tr = vcell(S, b, V, vtr, rule, k, vt, vtr_m); CAND.append(dict(b=b, name=f"{m} k{k} {rule} (NDBE/ID only)", grp="model_gated", F=F, NS=NS, tr=tr))
        for m in ("L-LATE", "L-CNV"):
            for qf in ("ps_oac", "jt_oac", "th_ntilez2", "jt_narm05"):
                for qq in (0.90, 0.80):
                    V, vtr = double_V(S, FT[qf].values.astype(float), O[m], lambda j, kk, m=m: inner_of(S, m, j, kk), 2, qq)
                    for rule in ("strict", "loss1", "endo95"): F, NS, TH, tr = vcell(S, b, V, vtr, rule, 2); CAND.append(dict(b=b, name=f"{m} low AND {qf} quiet<{int(qq * 100)} kq2 {rule}", grp="double", F=F, NS=NS, tr=tr))
        cs0 = [c for c in CAND if c["b"] == b]
        for fam, grps in (("double", ("double",)), ("quiet", ("quiet",)), ("model", ("model",)), ("model_gated", ("model_gated",)), ("model_any", ("model", "model_gated")), ("all", ("quiet", "model", "model_gated", "double"))):
            cs = [c for c in cs0 if c["grp"] in grps]
            for slack in (0, 1):
                F = np.zeros((E, 10), bool); NS = np.zeros((E, 10), int)
                for (j, kk) in S["TRX"]:
                    ok = [c for c in cs if c["tr"][(j, kk)][1] >= c["tr"][(j, kk)][2] - slack]
                    if not ok: continue
                    best = min(ok, key=lambda c: (c["tr"][(j, kk)][0], c["tr"][(j, kk)][3])); te = EF[:, j] == kk; F[te, j] = best["F"][te, j]; NS[te, j] = best["NS"][te, j]
                put(S, f"{tag}|E2sel|{b} selected per fold from {fam}, slack {slack}", F, NS)
    CAND.clear()
    for m in ("L-LATE", "L-CNV"):   # combo (192)
        for qf in ("ps_oac", "th_ntilez2", "jt_narm05", "ps_fga10"):
            q = FT[qf].values.astype(float)
            for kq in (1, 2):
                for qq in (0.90, 0.80):
                    Vq = vs_test(S, q, kq); Vm = vs_test(S, O[m], 1); V = np.full((E, 10), np.nan); vtr = {}
                    for (j, kk), tx in S["TRX"].items():
                        vq = vs_train(S, q, tx["top"], kq); t2 = thr_sens(vq[(tx["lab"] == 1) & np.isfinite(vq)], qq); vm = vs_train(S, inner_of(S, m, j, kk), tx["top"], 1)
                        vtr[(j, kk)] = np.where(vq >= t2, np.inf, vm); te = EF[:, j] == kk; V[te, j] = np.where(Vq[te, j] >= t2, np.inf, Vm[te, j])
                    for b in ("P1", "R49hm"):
                        for rule in ("strict", "loss1", "endo95"): F, NS, TH = veto_V(S, b, V, vtr, rule, max(kq, 1)); put(S, f"{tag}|E2combo|{b} double {m} low AND {qf} quiet<{int(qq * 100)} kq{kq} {rule}", F, NS)
    for gname, gmax in (("NDBE only", 0), ("NDBE or ID", 1)):   # grade (64)
        vt = (EG <= gmax)[:, None]; vtr_m = {kk: EG[tx["idx"]] <= gmax for kk, tx in S["TRX"].items()}
        for f, k in (("L-LATE", 1), ("L-CNV", 1), ("ps_oac", 2), ("jt_oac", 2)):
            V, vtr = Vmod(S, f, k) if f.startswith("L-") else Vfix(S, FT[f].values.astype(float), k)
            for b in ("P1", "R49hm"):
                for rule in G2["RULES"]: F, NS, TH = veto_V(S, b, V, vtr, rule, k, ve_test=vt, ve_train=vtr_m); put(S, f"{tag}|E2grade|{b} veto {f} k{k} {rule}, {gname}", F, NS)
    X = S["OUTS"]   # and (9)
    for qi in (0.80, 0.90, 0.95):
        Ti, _ = g8["thr_arr"](S, "img", qi); Fi = X["img"] >= Ti
        for qc in (0.80, 0.90, 0.95): Tc, _ = g8["thr_arr"](S, "cnv", qc); put(S, f"{tag}|E2and|AND img {qi} cnv {qc}", Fi & (X["cnv"] >= Tc), np.where(Fi, S["nsz"][:, None], 0))
    # drift: batch- and noise-adjusted vetoes (60), adjustment constants from each outer fold's training non-progressor samples (E2 drift task)
    bt = FT.batch.astype(str).values; lmapd = np.log(FT.mapd.values); es = S["es"]; trs_of = {kk: np.concatenate([es[e] for e in tx["idx"]]) for kk, tx in S["TRX"].items()}; yy = G2["y"]
    for f in ("ps_fga10", "ps_narm05", "jt_fga10", "th_cx", "L-CNV"):
        base = (lg(O["L-CNV"]) if f == "L-CNV" else FT[f].values.astype(float))
        for adj in ("batch", "noise"):
            ftest = np.full((n, 10), np.nan); ftr = {}
            for (j, kk), tx in S["TRX"].items():
                tri = trs_of[(j, kk)]; te = np.where(S["FO"][:, j] == kk)[0]
                if f == "L-CNV": xs = np.full(n, np.nan); s_ = S["I"]["L-CNV"][(REPS[j], kk)]; xs[s_.index.values] = lg(s_.values); xs[te] = base[te, j]
                else: xs = base.copy()
                nps = tri[yy[tri] == 0]
                if adj == "batch":
                    gm = np.nanmedian(xs[nps]); bm = pd.Series(xs[nps]).groupby(bt[nps]).agg(lambda v: np.nanmedian(v) if np.isfinite(v).sum() >= 5 else np.nan); xa = xs - pd.Series(bt).map(bm).fillna(gm).values
                else:
                    ok = np.isfinite(xs[nps]) & np.isfinite(lmapd[nps]); cf = np.polyfit(lmapd[nps][ok], xs[nps][ok], 1); xa = xs - np.polyval(cf, np.where(np.isfinite(lmapd), lmapd, np.nanmedian(lmapd)))
                ftr[(j, kk)] = xa; ftest[te, j] = xa[te]
            for b in ("P1", "R49hm"):
                for rule in ("strict", "loss1", "endo95"): F, NS, TH = G2["run_veto"](S, b, 1, ftest, lambda j, kk: ftr[(j, kk)], rule); put(S, f"{tag}|E2drift|{b} veto {f} {adj} k1 {rule}", F, NS)
    # stab (18)
    ARM = FT[[c for c in FT.columns if c.startswith("ps_A_")]].values.astype(float); ARM = np.where(np.isfinite(ARM), ARM, 0.0); FE = np.array([S["ORD"][S["pidx"][e]][0] for e in range(E)]); FIRST = S["FIRST"]; cur = FT.ps_armabs.values.astype(float)
    def pairscore(a, b_, kind):
        da, db = ARM[a], ARM[b_]
        if kind == "change": return np.abs(da - db).mean(-1)
        if kind == "new": return ((np.abs(da) > 0.05) & (np.abs(db) <= 0.05)).sum(-1).astype(float)
        return cur[a] + np.abs(da - db).sum(-1)
    for kind in ("change", "new", "cur_plus_change"):
        V = np.full((E, 10), np.nan)
        for j in range(10): V[:, j] = pairscore(S["TOPO"][:, j, 0], S["TOPO"][FE, j, 0], kind)
        vtr = {}; vet = {}
        for (j, kk), tx in S["TRX"].items():
            pos = {e: q for q, e in enumerate(tx["idx"])}; top = tx["top"][:, 0]; base_s = np.array([top[pos[FE[e]]] for e in tx["idx"]]); vtr[(j, kk)] = pairscore(top, base_s, kind); vet[(j, kk)] = ~FIRST[tx["idx"]]
        for b in ("P1", "R49hm"):
            for rule in ("strict", "loss1", "endo95"):
                F, NS, TH = veto_V(S, b, V, vtr, rule, 1, ve_test=~FIRST[:, None], ve_train=vet)
                BF, _, VEh, VEm, _ = base_of(S, b); used = (VEh | VEm) & BF & ~FIRST[:, None]; seqd = (VEh | VEm) & BF & FIRST[:, None]; extra = np.zeros((E, 10), int)
                for i in range(S["P_"]):
                    o = S["ORD"][i]; need = used[o].any(0) & ~seqd[o[0]] & ~((S["GB"][o[0]] == 1) if b != "P1" else np.zeros(10, bool)); extra[o[0]] += need
                put(S, f"{tag}|E2stab|{b} stability {kind} {rule}", F, NS + extra)
# ================================================================== memory-base vetoes / checks (E2 mem, E7 bar sweep and in-fold bar, E8 arm sets, E9 stacks)
def mem_vetoes(S):
    """the 17 E2 'mem' vetoes: (name, V, vtr, rule, k, gated)"""
    O = S["O"]; EG = np.array([GRv[s].max() for s in S["es"]]); L = []
    for m in ("L-LATE", "L-CNV"):
        V, vtr = Vmod(S, m, 1)
        for rule in ("strict", "loss1", "endo95"): L.append((f"{m} k1 {rule}", V, vtr, rule, 1, False)); L.append((f"{m} k1 {rule}, NDBE/ID only", V, vtr, rule, 1, True))
    q = FT.ps_oac.values.astype(float); V, vtr = Vfix(S, q, 2); L.append(("ps_oac k2 cn90", V, vtr, "cn90", 2, False))
    for qq in (0.80, 0.90):
        V, vtr = double_V(S, q, O["L-LATE"], lambda j, kk: inner_of(S, "L-LATE", j, kk), 2, qq)
        for rule in ("strict", "endo95"): L.append((f"L-LATE low AND ps_oac quiet<{int(qq * 100)} kq2 {rule}", V, vtr, rule, 2, False))
    return L, EG
def apply_veto(S, FM, NM, V, TH, k, gate, band=True):
    ve = FM & gate; veto = ve & (V < TH); k2 = np.minimum(k, S["nsz"])[:, None]
    return FM & ~veto, NM + np.where(ve, np.where(S["GB"] == 1, k2 - 1, k2) if band else k2, 0)
def mem_cells(S, tag, bases, VL, EG, only=None):
    gate_t = (EG <= 1)[:, None]; gate_tr = {kk: EG[tx["idx"]] <= 1 for kk, tx in S["TRX"].items()}; THC = {}
    for nm, V, vtr, rule, k, g in VL:
        if only and nm not in only: continue
        THC[nm] = veto_V(S, "R49hm", V, vtr, rule, k, ve_test=gate_t if g else None, ve_train=gate_tr if g else None)[2]
    for bn, (FM, NM, band) in bases.items():
        for nm, V, vtr, rule, k, g in VL:
            if only and nm not in only: continue
            F, NS = apply_veto(S, FM, NM, V, THC[nm], k, gate_t if g else True, band); put(S, f"{tag}|mem|{bn} + {nm}", F, NS)
def engine_trf(S, ST, fam):
    """rv9 nested engine for one family, criterion 'np' (as ex_e7_stress.py engine): held-out flags and the chosen candidate's training-patient flags per fold"""
    mk = G9["make_ctx"]; B = G9["FAMS"][fam]; E = S["E"]; elab = S["elab"]; F = np.zeros((E, 10), bool); NS = np.zeros((E, 10), int); TRF = {}
    for j in range(10):
        for k in range(1, 11):
            c = mk(S, ST, j, k, S["info"]); te = c.te_e; cands = B(c); ts_ = []; fl = []
            for x in cands:
                pm = G9["pat_scores"](c, x["u"]); t = G9["thr_for"](c, pm); flag = x["u"] >= t; inc = (S["INC"] @ flag.astype(float)) > 0
                ts_.append((c.patient_det(flag), int((inc & ~c.isP & c.trP).sum()), int((flag & c.tr_e & (elab == 0)).sum()), float(x["ns"][c.tr_e].mean()))); fl.append((flag, x["ns"]))
            ts_ = np.array(ts_); feas = ts_[:, 0] >= c.D; nC = len(cands)
            pick = (sorted([q for q in range(nC) if feas[q]], key=lambda q: (ts_[q, 1], ts_[q, 2], ts_[q, 3], q)) if feas.any() else sorted(range(nC), key=lambda q: (-ts_[q, 0], ts_[q, 1], ts_[q, 2], q)))[0]
            F[:, j][te] = fl[pick][0][te]; NS[:, j][te] = fl[pick][1][te]; TRF[(j, k)] = fl[pick][0]
    return F, NS, TRF
QS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975)
def bars_of(S, vtr):
    TH = {q: np.full((S["E"], 10), np.nan) for q in QS}; TV = {}
    for (j, kk), tx in S["TRX"].items():
        v = vtr[(j, kk)]; m = (tx["lab"] == 1) & np.isfinite(v)
        for q in QS: t = thr_sens(v[m], q) if m.any() else -np.inf; TH[q][S["EF"][:, j] == kk, j] = t; TV[(j, kk, q)] = t
    return TH, TV
def infold_bar(S, TRF, vtr, TV, slack):
    TH = np.full((S["E"], 10), np.nan)
    for (j, kk), tx in S["TRX"].items():
        f = TRF[(j, kk)][tx["idx"]]; v = vtr[(j, kk)]; lab = tx["lab"]; pid = tx["pid"]
        def outcome(t): kept = f & ~(v < t); return len(np.unique(pid[kept & (lab == 1)])), len(np.unique(pid[kept & (lab == 0)]))
        d0 = len(np.unique(pid[f & (lab == 1)])); best = (outcome(-np.inf)[1], 0, -np.inf)
        for q in QS:
            d, npf = outcome(TV[(j, kk, q)])
            if d >= d0 - slack and (npf, -q) < best[:2]: best = (npf, -q, TV[(j, kk, q)])
        TH[S["EF"][:, j] == kk, j] = best[2]
    return TH
def r9_cells(S, ST, tag, fams):
    R = G9["run_families"](S, ST, fams, S["info"]); out = {}
    for f, Rf in R.items():
        for q, lab in enumerate(Rf["labels"]): put(S, f"{tag}|r9|{f}|{lab}", Rf["F"][q], Rf["NS"][q])
        for cr in ("np", "pauc"): put(S, f"{tag}|r9|{f}|sel_{cr}", *Rf["sel"][cr])
        out[f] = Rf["sel"]["np"]
    return out
# ================================================================== run
t0 = time.time(); TIMES = {}
def tick(nm): TIMES[nm] = round(time.time() - t0, 1); print(nm, TIMES[nm], flush=True)
FAMS_ALL = list(G9["FAMS"]); FAMS_E3 = ["A1:img:none", "A1:img:ew", "A1:gate_al:none", "A1:gate_al:ew"]
# ---- st: stored-type aggregation (endoscopy max)
def p1_cells(S, tag):   # P1 at 80 / 90 / 95% training endoscopy sensitivity (round 8 / E3 cells)
    for q in (0.80, 0.95): put(S, f"{tag}|ref|P1_{int(q * 100)}", S["OUTS"]["img"] >= g8["thr_arr"](S, "img", q)[0], np.zeros((S["E"], 10), int))
S, ST = build_state(O0, I0); put(S, "st|ref|R49", S["GC"], S["NS49"]); put(S, "st|ref|P1_90", S["P1F"], np.zeros((S["E"], 10), int)); p1_cells(S, "st"); tick("build st")
SEL = r9_cells(S, ST, "st", FAMS_ALL); tick("rv9 st")
e2_cells(S, "st"); tick("E2 st")
VL, EG = mem_vetoes(S); FA12, NA12, TRF12 = engine_trf(S, ST, "A12"); assert (FA12 == SEL["A12"][0]).all()
mem_cells(S, "st", {"A1 ew": (*SEL["A1:gate_al:ew"], True), "A1 ew_ext": (*SEL["A1:gate_al:ew_ext"], True), "A12": (FA12, NA12, True)}, VL, EG)
q_oac = FT.ps_oac.values.astype(float); V_oac, vtr_oac = Vfix(S, q_oac, 2); THq, TVq = bars_of(S, vtr_oac)
for q in QS: put(S, f"st|E7bar|A12 + check bar {q}", *apply_veto(S, FA12, NA12, V_oac, THq[q], 2, True))
for sl in (0, 1): put(S, f"st|E7bar|A12 + check in-fold bar slack {sl}", *apply_veto(S, FA12, NA12, V_oac, infold_bar(S, TRF12, vtr_oac, TVq, sl), 2, True))
# E8 arm sets on A12 (k2, cn90 bar); controls (GC-rich arms, fragile sites, random sets) are not candidates and are left out
ARMSV = {c[5:]: np.nan_to_num(FT[c].values.astype(float)) for c in FT.columns if c.startswith("ps_A_")}; LOSS, GAIN = G2["LOSS"], G2["GAIN"]; OAC = LOSS + GAIN; SIGN = {**{a: -1 for a in LOSS}, **{a: 1 for a in GAIN}}
def armscore(arms, absolute=False): return sum(np.abs(ARMSV[a]) if absolute else np.maximum(0, SIGN[a] * ARMSV[a]) for a in arms)
assert np.allclose(armscore(OAC), q_oac)
SETS = [("17p", ["17p"], False), ("17p+9p", ["17p", "9p"], False), ("7 losses", LOSS, False), ("5 losses", ["17p", "9p", "18q", "5q", "13q"], False), ("OAC-17p", [a for a in OAC if a != "17p"], False),
        ("OAC-20q", [a for a in OAC if a != "20q"], False), ("OAC any direction", OAC, True)] + [(f"single {a}", [a], False) for a in OAC]
for nm, arms, ab in SETS:
    V, vtr = Vfix(S, armscore(arms, ab), 2); TH = bars_of(S, vtr)[0][0.90]; put(S, f"st|E8arms|A12 + check {nm} k2 cn90", *apply_veto(S, FA12, NA12, V, TH, 2, True))
for nm, x, k in (("OAC k1", q_oac, 1), ("ps_fga10 k2", FT.ps_fga10.values.astype(float), 2)):
    V, vtr = Vfix(S, x, k); TH = bars_of(S, vtr)[0][0.90]; put(S, f"st|E8arms|A12 + check {nm} cn90", *apply_veto(S, FA12, NA12, V, TH, k, True))
tick("mem/E7/E8 st"); del S, ST
# ---- E3 aggregations (mean logit = E9's base for the no-sequencing tier and the mean-logit full stack)
for how in ("ml", "med", "mprob", "top2"):
    O, I = agg_state(how); S, ST = build_state(O, I); put(S, f"{how}|ref|R49 rebuilt", S["GC"], S["NS49"]); put(S, f"{how}|ref|P1_90", S["P1F"], np.zeros((S["E"], 10), int)); p1_cells(S, how)
    SELm = r9_cells(S, ST, how, FAMS_E3 + (["A12"] if how == "ml" else []))
    if how == "ml":
        VL, EG = mem_vetoes(S)
        mem_cells(S, "ml", {"A12": (*SELm["A12"], True), "gate ew": (*SELm["A1:gate_al:ew"], True), "img ew (no gate)": (*SELm["A1:img:ew"], False)}, VL, EG, only={"ps_oac k2 cn90"})
    tick(f"{how}"); del S, ST
# ---- E3 extractor / pooling / ensemble variants (MODE b only: their stored refits; a full-refit null would need every block refitted per label set). Cells tagged e3x:
if MODE == "b" and LIMG == "stored" and os.environ.get("EXT", "1") == "1":   # not with the clean-tile L-IMG: the E3 variants were built on all tiles
    _e = open(P + "/ex_e3_score.py").read(); exec(compile(_e[_e.index('E3 = M + "/explore_E3"'):_e.index('\n# ------------------------------------------------------------------ one variant')], P + "/ex_e3_score.py", "exec"), G9)
    VV = [os.path.basename(f)[2:-5] for f in sorted(glob.glob(T + "/results/paper_final/explore_E3/v_*.json"))]; VM = [os.path.basename(f)[4:-5] for f in sorted(glob.glob(T + "/results/paper_final/explore_E3/mem_*.json"))]
    unf = lambda v: v.replace("--", ":").replace("__", "+")
    for V in sorted(set(VV) | set(VM)):
        if V == "stored": continue
        Oi, Ii, Op, Ip, _ = G9["variant"](unf(V))
        if Op is not None: Oi = sg(lg_(Oi) + 1e-6 * lg_(Op)); Ii = {k: pd.Series(sg(lg_(Ii[k].values) + 1e-6 * lg_(Ip[k].reindex(Ii[k].index).values)), index=Ii[k].index) for k in Ii}
        Ops = Oi if Op is None else Op; Ips = Ii if Ip is None else Ip; O = dict(O0); I = dict(I0)
        O["L-IMG"] = Oi; O["L-LATE"] = (O["L-CNV"] + Ops) / 2; I["L-IMG"] = Ii; I["L-LATE"] = {k: (I["L-CNV"][k] + Ips[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
        S, ST = build_state(O, I); tg = f"e3x:{V}"
        if V in VV: put(S, f"{tg}|ref|R49 rebuilt", S["GC"], S["NS49"]); put(S, f"{tg}|ref|P1_90", S["P1F"], np.zeros((S["E"], 10), int)); p1_cells(S, tg)
        if V in VM: r9_cells(S, ST, tg, FAMS_E3)
        del S, ST
    tick("E3 extended")
names = sorted(CELLS); arr = np.stack([CELLS[k] for k in names]).astype(np.float32)
np.savez_compressed(OUTF + ".tmp.npz", vals=arr, names=np.array(names), times=json.dumps(TIMES), shuf=SHUF, mode=MODE, kind=KIND, limg=LIMG); os.replace(OUTF + ".tmp.npz", OUTF)
print("TPP2N DONE search", MODE, KIND, SHUF, len(names), "cells", round(time.time() - t0, 1), "s", flush=True)
