"""Review round 7b: the per-endoscopy gate with nested selection (docs/paper_review_round7b_gate.md @ 0764a2e). Re-scores stored outer and inner
predictions (tb_common.py loaders and the paper_triage.md cut-off rules) and, for robustness row (d), the R52(a) refit outputs
(killcoyne_mm/review_round7/refit/r52a). No refit. Helpers copied unchanged from rv7_round7.py (R46 seeds, R49 bands, use_set, load_var).
Row-level files only under killcoyne_mm/review_round7b/ on the cluster; aggregates (no identifiers, no dates) -> results/paper_final/review_round7b/<TASK>.json.
TASK = sel_<row>_<rep> (rep 01..10: nested selection and outer application for one repeat) | merge_<row> (bootstrap, aggregates) | curve (trade-off grid)
       <row> = primary | a_ndbe | b_earliest | c_ge2 | d_r52a"""
import os, sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_common as TC
from tb_common import *
PRESPEC = "0764a2e"; OUT = T + "/results/paper_final/review_round7b"; OUT7 = T + "/results/paper_final/review_round7"
R7 = M + "/review_round7"; R7B = M + "/review_round7b"; SEL = R7B + "/sel"
if os.environ.get("RV7B_NB"):   # smoke mode: fewer bootstrap draws, aggregates written aside (never to results/)
    NB = int(os.environ["RV7B_NB"]); TC.NB = NB; OUT = R7B + "/smoke_json"
for d in (OUT, R7B, SEL): os.makedirs(d, exist_ok=True)
TASK = os.environ["TASK"]; MM = {"A*": "L-CNV", "B*": "L-LATE", "C*": "L-IMG"}; NSEEDS = 100
ROWS = ("primary", "a_ndbe", "b_earliest", "c_ge2", "d_r52a")
# ------------------------------------------------------------------ helpers copied from rv7_round7.py (unchanged)
def ci2(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return pct(a) if len(a) else [None, None]
def lo1(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return r3(np.percentile(a, 5)) if len(a) else None
def dstat(obs, draws, one_sided=False):
    o = {"delta": r3(obs), "ci95": ci2(draws), "p_two_sided": p2(draws)}
    if one_sided: o["one_sided_95_lower"] = lo1(draws); o["non_inferior_at_-0.05"] = bool(lo1(draws) is not None and lo1(draws) > -0.05)
    return o
def wmean(v, w): return float(v.mean()) if w is None else float((v * w).sum() / w.sum())
def need(files, k): assert len(files) == k, (len(files), k, files[:2]); return files
def nfo(rows): return {"n_samples": int(len(rows)), "n_progressor_samples": int(y[rows].sum()), "n_patients": int(len(set(pat[rows]))), "n_progressor_patients": int(len(set(pat[rows][y[rows] == 1])))}
def matched_calls(O, I, FO, popm, W=None, ql=0.95, qh=0.90, mid="L-LATE"):
    mc = {a: np.zeros((n, 10), bool) for a in MM}; sstar = []
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            si = I["L-IMG"][(r, k)]; tri = si.index.values[popm[si.index.values]]; w = None if W is None else W[tri]
            if w is not None: keep = w > 0; tri = tri[keep]; w = w[keep]
            s_tr = si.reindex(tri).values; yc = y[tri] == 1; wy = None if w is None else w[yc]; wn = None if w is None else w[~yc]
            tl = thr_sens(s_tr[yc], ql, wy); cs = c_star(s_tr[~yc], qh, wn); li = I[mid][(r, k)].reindex(tri).values
            md = (s_tr >= tl) & (s_tr <= cs) if tl <= cs else np.zeros(len(tri), bool); sel = md & yc; nsel = sel.sum() if w is None else w[sel].sum()
            ts = thr_sens(li[sel], 0.8, None if w is None else w[sel]) if nsel >= 5 else thr_sens(li[yc], 0.8, wy)
            tcall = np.where(s_tr < tl, False, np.where(s_tr > cs, True, li >= ts)) if tl <= cs else s_tr > cs
            s_star = 1 - wmean(tcall[~yc].astype(float), wn); sstar.append(s_star); te = FO[:, j] == k
            for a, mn in MM.items():
                xc = I[mn][(r, k)].reindex(tri).values[~yc]
                if w is not None: xc = np.repeat(xc, wn)
                xc = np.sort(xc); q = int(np.ceil(s_star * len(xc))); thr = xc[q - 1] if q > 0 else -np.inf; mc[a][te, j] = O[mn][te, j] > thr
    return mc, sstar
def all_calls(O, I, FO, popm, W=None, ql=0.95, qh=0.90, mid="L-LATE"):
    band, call, info = TC.triage(O, I, FO, popm, ql, qh, mid=mid, W=W); comp, thr = TC.comparators(O, I, FO, popm, W=W); mc, sstar = matched_calls(O, I, FO, popm, W=W, ql=ql, qh=qh, mid=mid)
    CALLS = {"T": call, "A": comp["A"], "B": comp["B"], "C": comp["C"], "A*": mc["A*"], "B*": mc["B*"], "C*": mc["C*"]}; return CALLS, band, info, thr, sstar
def use_set(V):
    """rebind the sample-level globals of tb_common and of this module to a variant set (refit metrics); V in the order of the refit's set"""
    g = {"n": len(V), "ids": V.Sample.values.astype(str), "y": V.y.values.astype(int), "pat": V.Patient.values.astype(str), "PRE": V.pre.astype(str).eq("True").values, "NDBE": V.ndbe.astype(str).eq("True").values}
    g["ix"] = pd.Series(np.arange(g["n"]), index=g["ids"]); g["ENDO"] = (pd.Series(g["pat"]) + "|" + V.Endoscopy.astype(str).values).values
    for k, v in g.items(): setattr(TC, k, v); globals()[k] = v
def load_var(var, reps=10):
    D = R7 + f"/refit/{var}"; f = lambda pat_: sorted(glob.glob(f"{D}/{pat_}"))
    oc, oi = need(f("outer_cnv_their_rep_*.csv"), reps), need(f("outer_img_rep_*.csv"), reps); ic, ii = need(f("inner_cnv_their_rep_*_fold_*.csv"), 10 * reps), need(f("inner_img_rep_*_fold_*.csv"), 10 * reps)
    def outer(files): Dd = pd.concat([pd.read_csv(x, dtype={"Sample": str}) for x in files]); return Dd.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values, Dd.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int)
    def inner(files): Dd = pd.concat([pd.read_csv(x, dtype={"Sample": str}) for x in files]); assert Dd.groupby(["rep", "fold"]).ngroups == 100; return {(r, k): pd.Series(d.pred.values, index=ix[d.Sample].values) for (r, k), d in Dd.groupby(["rep", "fold"])}
    O, I = {}, {}; O["L-CNV"], FO = outer(oc); O["L-IMG"], F2 = outer(oi); assert (FO == F2).all() and np.isfinite(O["L-CNV"]).all() and np.isfinite(O["L-IMG"]).all(); O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2
    I["L-CNV"] = inner(ic); I["L-IMG"] = inner(ii); I["L-LATE"] = {k: (I["L-CNV"][k] + I["L-IMG"][k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    for k in I["L-CNV"]: assert I["L-IMG"][k].index.sort_values().equals(I["L-CNV"][k].index.sort_values()), k
    return O, I, FO
def dump(res, name=None):
    res["prespec"] = PRESPEC; json.dump(res, open(f"{OUT}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV7B DONE", name or TASK, flush=True)
# ------------------------------------------------------------------ the 36 variants, in the pre-specified simplicity order (index order = tie-break order)
AGGS = ("max", "top2", "mean"); CUTS = ("sample", "endoscopy"); SEQS = ("highest", "uncertain", "all"); CALLM = ("L-LATE", "L-CNV")
VARS = [(a, c, s, m) for a in AGGS for c in CUTS for s in SEQS for m in CALLM]; NV = len(VARS); assert NV == 36
VLAB = [f"{a} | {c}-level cut-offs | {s} | {m}" for a, c, s, m in VARS]
V_R49 = VARS.index(("max", "sample", "highest", "L-LATE")); V_R49ALL = VARS.index(("max", "sample", "all", "L-LATE"))
def escore(x, a): xs = np.sort(x)[::-1]; return xs[0] if a == "max" else (xs[:2].mean() if a == "top2" else xs.mean())
def endo_cuts(groups, img, lab, ql=0.95, qh=0.90):
    """endoscopy-level cut-offs per aggregation: t_low at ql endoscopy sensitivity, t_high (c*) at qh endoscopy specificity, on these endoscopies' scores"""
    out = {}
    for a in AGGS: sc = np.array([escore(img[s], a) for s in groups]); out[a] = (thr_sens(sc[lab == 1], ql), c_star(sc[lab == 0], qh))
    return out
def eval_variants(groups, img, late, cnv, tl, cs, ce, ts, tsc):
    """groups: index arrays into img / late / cnv, one per endoscopy; (tl, cs) the sample-level fold cut-offs, ce[agg] the endoscopy-level ones; ts / tsc the
    L-LATE / L-CNV middle-band thresholds. Returns flag, samples sequenced and endoscopy band (0 low, 1 middle, 2 high), each G x 36."""
    G = len(groups); F = np.zeros((G, NV), bool); NS = np.zeros((G, NV), int); BD = np.zeros((G, NV), int)
    for g, s in enumerate(groups):
        xi = img[s]; sc = {a: escore(xi, a) for a in AGGS}
        for v, (a, c, q, m) in enumerate(VARS):
            lo, hi = (tl, cs) if c == "sample" else ce[a]
            b = (2 if sc[a] > hi else 0) if lo > hi else (0 if sc[a] < lo else (2 if sc[a] > hi else 1)); BD[g, v] = b
            if b != 1: F[g, v] = b == 2; continue
            pick = [int(np.argmax(xi))] if q == "highest" else ([int(np.argmin(np.abs(xi - (lo + hi) / 2)))] if q == "uncertain" else list(range(len(s))))
            F[g, v] = bool(((late if m == "L-LATE" else cnv)[s][pick] >= (ts if m == "L-LATE" else tsc)).any()); NS[g, v] = len(pick)
    return F, NS, BD
def most_chosen(CH): c = np.bincount(CH[CH >= 0], minlength=NV); return int(np.argmax(c))   # ties to the simpler (lower-index) variant
def groups_of(idx):
    """endoscopies of the sample indices idx, in sorted endoscopy order: (keys, list of positions into idx)"""
    ue, inv = np.unique(ENDO[idx], return_inverse=True); return ue, [np.where(inv == g)[0] for g in range(len(ue))]
def bstar_thr(I, r, k, popm, s_star):
    """B* fold threshold exactly as matched_calls (positive if L-LATE > thr)"""
    tri = I["L-IMG"][(r, k)].index.values; tri = tri[popm[tri]]; yc = y[tri] == 1; xc = np.sort(I["L-LATE"][(r, k)].reindex(tri).values[~yc]); q = int(np.ceil(s_star * len(xc))); return xc[q - 1] if q > 0 else -np.inf
def setup(row):
    if row == "d_r52a":
        V = pd.read_csv(R7 + "/sets/r52a.csv", dtype={"Sample": str, "Patient": str, "Endoscopy": str}); use_set(V); O, I, FO = load_var("r52a"); popm = PRE.copy()
    else:
        O, I, FO = load("their"); popm = PRE.copy()
        if row == "a_ndbe": popm &= NDBE
        elif row == "b_earliest":   # earliest pre-event endoscopy per patient = the one with the largest months-before-final
            mbf = SM.mbf.values.astype(float); em = pd.Series(mbf[PRE]).groupby(ENDO[PRE]).max(); ep = em.index.str.split("|").str[0]
            first = em.groupby(ep.values).idxmax(); popm &= np.isin(ENDO, first.values)
        elif row == "c_ge2": cnt = pd.Series(ENDO[PRE]).value_counts(); popm &= pd.Series(ENDO).map(cnt).fillna(0).values >= 2
        else: assert row == "primary", row
    rows = np.where(popm)[0]; CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm)
    bandc, _, infoc = TC.triage(O, I, FO, popm, mid="L-CNV"); assert (bandc == band).all()
    eu, es_ = groups_of(rows); es = [rows[g] for g in es_]
    return dict(O=O, I=I, FO=FO, popm=popm, rows=rows, CALLS=CALLS, band=band, info=info, thr=thr, sstar=sstar, infoc=infoc, eu=eu, es=es)
res = {"task": TASK}
# ================================================================== per repeat: nested selection on inner predictions, outer application of all 36 variants
if TASK.startswith("sel_"):
    row, rp = TASK[4:].rsplit("_", 1); j = int(rp) - 1; r = REPS[j]; S = setup(row); O, I, FO, popm, CALLS, band, info, infoc, es = (S[k] for k in ("O", "I", "FO", "popm", "CALLS", "band", "info", "infoc", "es"))
    E = len(es); efold = np.array([FO[s[0], j] for s in es]); assert all(len(set(FO[s, j])) == 1 for s in es)
    VF = np.zeros((E, NV), bool); VN = np.zeros((E, NV), int); VB = np.zeros((E, NV), int); NG = np.zeros(E, bool); NGN = np.zeros(E, int); NGB = np.full(E, -1)
    CH = np.full(10, -1); IS = np.full((10, NV), np.nan); ISP = np.full((10, NV), np.nan); ISH = np.full((10, NV), np.nan); IB = np.full((10, 2), np.nan); IT = np.full((10, 3), np.nan); INN = np.zeros((10, 3), int)
    img_o, late_o, cnv_o = O["L-IMG"][:, j], O["L-LATE"][:, j], O["L-CNV"][:, j]
    for k in range(1, 11):
        i = j * 10 + (k - 1); tl, cs, ts = info["t_low"][i], info["c_star"][i], info["t_seq"][i]; tsc = infoc["t_seq"][i]
        si = I["L-IMG"][(r, k)]; tri = si.index.values[popm[si.index.values]]; img_i = si.reindex(tri).values; late_i = I["L-LATE"][(r, k)].reindex(tri).values; cnv_i = I["L-CNV"][(r, k)].reindex(tri).values
        assert np.isfinite(img_i).all() and np.isfinite(late_i).all() and np.isfinite(cnv_i).all()
        _, gi = groups_of(tri); lab_i = np.array([y[tri][g].max() for g in gi]); ce = endo_cuts(gi, img_i, lab_i)
        Fi, Ni, _ = eval_variants(gi, img_i, late_i, cnv_i, tl, cs, ce, ts, tsc)
        tb = bstar_thr(I, r, k, popm, S["sstar"][i]); te = (FO[:, j] == k) & popm; assert ((late_o[te] > tb) == CALLS["B*"][te, j]).all(), "B* threshold not reproduced"
        Bi = np.array([(late_i[g] > tb).any() for g in gi]); sB, pB = Bi[lab_i == 1].mean(), 1 - Bi[lab_i == 0].mean()
        sV, pV, nV = Fi[lab_i == 1].mean(0), 1 - Fi[lab_i == 0].mean(0), Ni.sum(0)
        elig = (sV >= sB - 0.02 - 1e-9) & (pV >= pB - 0.05 - 1e-9); c = min(np.where(elig)[0], key=lambda v: (nV[v], v)) if elig.any() else -1; CH[k - 1] = c
        bi = np.where(img_i < tl, 0, np.where(img_i > cs, 2, 1)) if tl <= cs else np.where(img_i > cs, 2, 0); ti = np.where(bi == 1, late_i >= ts, bi == 2)
        Ti = np.array([ti[g].any() for g in gi]); IT[k - 1] = [Ti[lab_i == 1].mean(), 1 - Ti[lab_i == 0].mean(), (bi == 1).mean()]
        IS[k - 1], ISP[k - 1], ISH[k - 1] = sV, pV, nV / len(tri); IB[k - 1] = [sB, pB]; INN[k - 1] = [len(tri), len(gi), int(elig.sum())]
        idx = np.where(efold == k)[0]; Fo, No, Bo = eval_variants([es[e] for e in idx], img_o, late_o, cnv_o, tl, cs, ce, ts, tsc); VF[idx], VN[idx], VB[idx] = Fo, No, Bo
        if c >= 0: NG[idx], NGN[idx], NGB[idx] = Fo[:, c], No[:, c], Bo[:, c]
        else: NG[idx] = [CALLS["T"][es[e], j].any() for e in idx]; NGN[idx] = [(band[es[e], j] == 1).sum() for e in idx]
    np.savez(f"{SEL}/{row}_rep_{j + 1:02d}.npz", eu=S["eu"], VF=VF, VN=VN, VB=VB, NG=NG, NGN=NGN, NGB=NGB, CH=CH, IS=IS, ISP=ISP, ISH=ISH, IB=IB, IT=IT, INN=INN)
    print("RV7B SEL DONE", TASK, "endoscopies", E, "chosen", CH.tolist(), "eligible", INN[:, 2].tolist(), flush=True)
# ================================================================== merge: comparators, random-one, patient bootstrap (fitted models and selections fixed)
elif TASK.startswith("merge_"):
    row = TASK[6:]; S = setup(row); O, FO, rows, CALLS, band, es, eu = (S[k] for k in ("O", "FO", "rows", "CALLS", "band", "es", "eu"))
    Z = [np.load(f, allow_pickle=True) for f in need(sorted(glob.glob(f"{SEL}/{row}_rep_*.npz")), 10)]
    for z in Z: assert (z["eu"] == eu).all()
    VF = np.stack([z["VF"] for z in Z], 2); VN = np.stack([z["VN"] for z in Z], 2); VB = np.stack([z["VB"] for z in Z], 2); NG = np.stack([z["NG"] for z in Z], 1); NGN = np.stack([z["NGN"] for z in Z], 1); NGB = np.stack([z["NGB"] for z in Z], 1)
    CH = np.concatenate([z["CH"] for z in Z]); E = len(es); elab = np.array([int(y[s].max()) for s in es]); epat = np.array([pat[s][0] for s in es]); nsz = np.array([len(s) for s in es])
    up = np.unique(epat); P_ = len(up); pe = [np.where(epat == p)[0] for p in up]; plab = np.array([int(elab[i].max()) for i in pe])
    anyE = lambda C: np.array([C[s].any(0) for s in es]); midE = np.array([(band[s] == 1).sum(0) for s in es]); allE = np.repeat(nsz[:, None], 10, 1)
    RU = {"nested": (NG, NGN), "Bstar_endo": (anyE(CALLS["B*"]), allE), "B_endo": (anyE(CALLS["B"]), allE), "triage": (anyE(CALLS["T"]), midE), "r49_gate": (VF[:, V_R49], VN[:, V_R49]), "r49_all": (VF[:, V_R49ALL], VN[:, V_R49ALL])}
    for v in range(NV): RU[f"v{v:02d}"] = (VF[:, v], VN[:, v])
    PF = {nm: np.stack([F[i].any(0) for i in pe]) for nm, (F, _) in RU.items()}
    PICK = np.zeros((NSEEDS, E), int)
    for sd in range(NSEEDS):
        rng = np.random.RandomState(sd)
        for e, s in enumerate(es): PICK[sd, e] = s[rng.randint(len(s))]   # R41 draw order (rv6_round6.py r41, rv7_round7.py r46)
    LS = O["L-LATE"][PICK]; RBown = CALLS["B"][PICK]
    def rand_matched(eidx, spec_t):
        neg = eidx[elab[eidx] == 0]; Xs = np.sort(LS[:, neg, :], 1); q = np.ceil(spec_t * len(neg) - 1e-9).astype(int)
        thr = np.stack([Xs[:, q[j] - 1, j] if q[j] > 0 else np.full(NSEEDS, -np.inf) for j in range(10)], 1); return LS[:, eidx, :] > thr[:, None, :]
    def stats(eidx, pidx):
        le = elab[eidx]; lp = plab[pidx]; tot = nsz[eidx].sum(); o = {}
        for nm, (F, NS) in RU.items():
            Fe = F[eidx]; o[f"sens_{nm}"] = Fe[le == 1].mean(0); o[f"spec_{nm}"] = 1 - Fe[le == 0].mean(0); Pp = PF[nm][pidx]; o[f"pmiss_{nm}"] = (~Pp[lp == 1]).sum(0).astype(float); o[f"pflag_{nm}"] = Pp[lp == 0].sum(0).astype(float)
            Ne = NS[eidx]; o[f"shs_{nm}"] = Ne.sum(0) / tot; o[f"she_{nm}"] = (Ne > 0).mean(0)
        RM = rand_matched(eidx, o["spec_nested"]); Ro = RBown[:, eidx]
        for nm, A_ in (("rand_matched", RM), ("rand_own", Ro)):
            o[f"sens_{nm}"] = A_[:, le == 1, :].mean(1).mean(0); o[f"spec_{nm}"] = 1 - A_[:, le == 0, :].mean(1).mean(0); o[f"shs_{nm}"] = np.full(10, len(eidx) / tot); o[f"she_{nm}"] = np.ones(10)
        return o
    obs = stats(np.arange(E), np.arange(P_)); rng = np.random.RandomState(0); DR = []
    while len(DR) < NB:
        pidx = rng.choice(P_, P_); eidx = np.concatenate([pe[i] for i in pidx])
        if 0 < plab[pidx].sum() < P_: DR.append(stats(eidx, pidx))
    m_ = lambda k: float(np.mean(obs[k])); dm = lambda k: np.array([d[k].mean() for d in DR])
    # random-one patient columns (observed only, as R46 b)
    RMo = rand_matched(np.arange(E), obs["spec_nested"]); toP = lambda A_: np.stack([A_[:, i, :].any(1) for i in pe], 1)
    for nm, A_ in (("rand_matched", RMo), ("rand_own", RBown)): PA = toP(A_); obs[f"pmiss_{nm}"] = (~PA[:, plab == 1, :]).sum(1).mean(0).astype(float); obs[f"pflag_{nm}"] = PA[:, plab == 0, :].sum(1).mean(0).astype(float)
    def rule(nm):
        o = {"endoscopy_sensitivity": r3(m_(f"sens_{nm}")), "sens_ci95": ci2(dm(f"sens_{nm}")), "endoscopy_specificity": r3(m_(f"spec_{nm}")), "spec_ci95": ci2(dm(f"spec_{nm}")),
             "share_samples_sequenced": r3(m_(f"shs_{nm}")), "share_samples_ci95": ci2(dm(f"shs_{nm}")), "share_endoscopies_sequenced": r3(m_(f"she_{nm}"))}
        if f"pmiss_{nm}" in obs: o["progressor_patients_missed"] = round(m_(f"pmiss_{nm}"), 2); o["non_progressor_patients_flagged"] = round(m_(f"pflag_{nm}"), 2)
        return o
    def minus(a, b):
        o = {"d_endoscopy_sens": dstat(m_(f"sens_{a}") - m_(f"sens_{b}"), dm(f"sens_{a}") - dm(f"sens_{b}"), one_sided=True), "d_endoscopy_spec": dstat(m_(f"spec_{a}") - m_(f"spec_{b}"), dm(f"spec_{a}") - dm(f"spec_{b}")),
             "d_share_samples": dstat(m_(f"shs_{a}") - m_(f"shs_{b}"), dm(f"shs_{a}") - dm(f"shs_{b}")), "d_share_endoscopies": r3(m_(f"she_{a}") - m_(f"she_{b}"))}
        if f"pmiss_{b}" in obs and f"pmiss_{a}" in obs: o["d_patients_missed"] = round(m_(f"pmiss_{a}") - m_(f"pmiss_{b}"), 2); o["d_patients_flagged"] = round(m_(f"pflag_{a}") - m_(f"pflag_{b}"), 2)
        return o
    R = {"row": row, **nfo(rows), "n_endoscopies": int(E), "n_progressor_endoscopies": int(elab.sum()), "n_progressor_patients_in_units": int(plab.sum()), "n_non_progressor_patients_in_units": int((plab == 0).sum()), "draws": len(DR), "n_seeds": NSEEDS}
    MAIN = ("nested", "Bstar_endo", "B_endo", "triage", "rand_own", "rand_matched", "r49_gate", "r49_all"); R["rules"] = {nm: rule(nm) for nm in MAIN}
    R["nested_minus"] = {b: minus("nested", b) for b in MAIN[1:]}
    R["nested_endoscopy_bands"] = {nm: r3((NGB == g).mean()) for g, nm in ((0, "low"), (1, "middle"), (2, "high"), (-1, "fallback_triage"))}
    # (e) between-repeat variance (R33 / R15 logic) of the nested gate against B*-endo
    pts = obs["sens_nested"] - obs["sens_Bstar_endo"]; PD = np.array([d["sens_nested"] - d["sens_Bstar_endo"] for d in DR]); lows = np.percentile(PD, 5, axis=0); avg = PD.mean(1); vb = float(avg.var(ddof=1)); vr = float(pts.var(ddof=1)); dd = float(pts.mean())
    R["between_repeat"] = {"per_repeat_d_endoscopy_sens": [r3(v) for v in pts], "per_repeat_one_sided_lower": [r3(v) for v in lows], "n_repeats_lower_above_-0.05": int((lows > -0.05).sum()), "share_repeats_lower_above_-0.05": r3((lows > -0.05).mean()),
                           "delta_repeat_mean": r3(dd), "V_boot": round(vb, 6), "V_rep": round(vr, 6), "standard_one_sided_lower": lo1(avg), "variance_inflated_one_sided_lower": r3(dd - 1.645 * np.sqrt(vb + vr)),
                           "variance_inflated_ci95": [r3(dd - 1.96 * np.sqrt(vb + vr)), r3(dd + 1.96 * np.sqrt(vb + vr))], "non_inferior_at_-0.05": bool(dd - 1.645 * np.sqrt(vb + vr) > -0.05)}
    # selection: frequencies over the 100 outer folds, inner diagnostics
    vc = pd.Series(CH).value_counts(); R["choice_frequency"] = {("fallback: sample-level triage" if v < 0 else VLAB[v]): int(c) for v, c in vc.items()}; R["n_folds"] = int(len(CH)); R["fallback_folds"] = int((CH < 0).sum())
    R["choice_by_factor"] = {f: {lv: int(sum(1 for v in CH if v >= 0 and VARS[v][fi] == lv)) for lv in lvls} for fi, (f, lvls) in enumerate((("aggregation", AGGS), ("cutoffs", CUTS), ("sequenced", SEQS), ("call", CALLM)))}
    R["most_chosen_variant"] = VLAB[most_chosen(CH)] if (CH >= 0).any() else None
    IS = np.concatenate([z["IS"] for z in Z]); ISP = np.concatenate([z["ISP"] for z in Z]); ISH = np.concatenate([z["ISH"] for z in Z]); IB = np.concatenate([z["IB"] for z in Z]); IT = np.concatenate([z["IT"] for z in Z]); INN = np.concatenate([z["INN"] for z in Z])
    ch_ok = CH >= 0; md = lambda a: r3(np.median(a)) if len(a) else None
    R["inner"] = {"eligible_variants_per_fold": {"median": md(INN[:, 2]), "min": int(INN[:, 2].min()), "max": int(INN[:, 2].max())}, "training_samples_per_fold_median": md(INN[:, 0]), "training_endoscopies_per_fold_median": md(INN[:, 1]),
                  "Bstar_endo_sens_spec_median": [md(IB[:, 0]), md(IB[:, 1])], "triage_endo_sens_spec_share_median": [md(IT[:, 0]), md(IT[:, 1]), md(IT[:, 2])],
                  "chosen_sens_spec_share_median": [md(IS[ch_ok, CH[ch_ok]]), md(ISP[ch_ok, CH[ch_ok]]), md(ISH[ch_ok, CH[ch_ok]])],
                  "per_variant_median_sens_spec_share": {VLAB[v]: [md(IS[:, v]), md(ISP[:, v]), md(ISH[:, v])] for v in range(NV)}}
    # apparent (non-nested) results of all 36 variants on the outer predictions: optimistic
    R["apparent_optimistic"] = {VLAB[v]: {**rule(f"v{v:02d}"), "minus_Bstar_endo": {k_: x for k_, x in minus(f"v{v:02d}", "Bstar_endo").items() if k_ in ("d_endoscopy_sens", "d_endoscopy_spec")},
                                          "minus_triage_d_share_samples": minus(f"v{v:02d}", "triage")["d_share_samples"]["delta"], "times_chosen": int((CH == v).sum())} for v in range(NV)}
    if row == "primary":
        r49 = json.load(open(f"{OUT7}/r49.json"))["R49"]["rules"]; r46 = json.load(open(f"{OUT7}/r46.json"))["R46"]["a_matched_specificity"]["random_own_threshold"]
        chk = {"r49_gate": {"expected": [r49["gate"][k_] for k_ in ("endoscopy_sensitivity", "endoscopy_specificity", "share_samples_sequenced", "share_endoscopies_sequenced", "progressor_patients_missed", "non_progressor_patients_flagged")]},
               "r49_all": {"expected": [r49["gate_all_mid_sequenced"][k_] for k_ in ("endoscopy_sensitivity", "endoscopy_specificity", "share_samples_sequenced", "share_endoscopies_sequenced", "progressor_patients_missed", "non_progressor_patients_flagged")]},
               "triage": {"expected": [r49["triage"][k_] for k_ in ("endoscopy_sensitivity", "endoscopy_specificity", "share_samples_sequenced", "share_endoscopies_sequenced", "progressor_patients_missed", "non_progressor_patients_flagged")]},
               "Bstar_endo": {"expected": [0.923, 0.530]}, "rand_own": {"expected": [r46["endoscopy_sensitivity"]["mean_over_seeds"], r46["endoscopy_specificity"]["mean_over_seeds"]]}}
        for nm, c in chk.items():
            g = R["rules"][nm]; full = [g["endoscopy_sensitivity"], g["endoscopy_specificity"], g["share_samples_sequenced"], g["share_endoscopies_sequenced"], g.get("progressor_patients_missed"), g.get("non_progressor_patients_flagged")]
            c["got"] = full[:len(c["expected"])]; c["reproduced"] = bool(all(abs(a - b) < 0.0015 for a, b in zip(c["got"], c["expected"])))
        chk["r49_gate"]["ci_got_expected"] = [R["rules"]["r49_gate"]["sens_ci95"], r49["gate"]["sens_ci95"]]; R["checks"] = chk
    R["note"] = ("endoscopy = (patient, endoscopy number) over the row's pre-event samples; flagged if the rule flags it; single model (per-repeat values averaged over the 10 repeats); patient bootstrap 2,000 draws "
                 "(RandomState(0), R49 draw loop) with fitted models, cut-offs and variant selections fixed; random-one = R41 seeds 0-99, own threshold = stored (B) call of the picked sample, matched = one L-LATE threshold per seed x repeat "
                 "giving the nested gate's endoscopy specificity in that repeat (re-matched in each draw)")
    res["R7B"] = R; dump(res)
# ================================================================== trade-off curve (descriptive): 3 x 3 cut-off quantile grid
elif TASK == "curve":
    S = setup("primary"); O, I, FO, popm, rows, CALLS, es = (S[k] for k in ("O", "I", "FO", "popm", "rows", "CALLS", "es"))
    CH = np.concatenate([np.load(f)["CH"] for f in need(sorted(glob.glob(f"{SEL}/primary_rep_*.npz")), 10)]); vstar = most_chosen(CH)
    E = len(es); elab = np.array([int(y[s].max()) for s in es]); nsz = np.array([len(s) for s in es]); tot = nsz.sum()
    PICK = np.zeros((NSEEDS, E), int)
    for sd in range(NSEEDS):
        rng = np.random.RandomState(sd)
        for e, s in enumerate(es): PICK[sd, e] = s[rng.randint(len(s))]
    LS = O["L-LATE"][PICK]; RBown = CALLS["B"][PICK]; neg = elab == 0
    def es_(F): return float(F[elab == 1].mean()), float(1 - F[elab == 0].mean())
    out = []; own = es_(RBown.transpose(1, 0, 2).reshape(E, -1))
    for ql in (0.90, 0.95, 0.98):
        for qh in (0.85, 0.90, 0.95):
            band, call, info = TC.triage(O, I, FO, popm, ql, qh); _, _, infoc = TC.triage(O, I, FO, popm, ql, qh, mid="L-CNV")
            TE = np.array([call[s].any(0) for s in es]); tsh = float(np.mean([(band[rows, j] == 1).mean() for j in range(10)])); tshe = float(np.mean([(band[s] == 1).any(0) for s in es]))
            GF = np.zeros((E, 10), bool); GN = np.zeros((E, 10), int)
            for j, r in enumerate(REPS):
                efold = np.array([FO[s[0], j] for s in es])
                for k in range(1, 11):
                    i = j * 10 + (k - 1); si = I["L-IMG"][(r, k)]; tri = si.index.values[popm[si.index.values]]; img_i = si.reindex(tri).values; _, gi = groups_of(tri); lab_i = np.array([y[tri][g].max() for g in gi])
                    ce = endo_cuts(gi, img_i, lab_i, ql, qh); idx = np.where(efold == k)[0]
                    Fo, No, _ = eval_variants([es[e] for e in idx], O["L-IMG"][:, j], O["L-LATE"][:, j], O["L-CNV"][:, j], info["t_low"][i], info["c_star"][i], ce, info["t_seq"][i], infoc["t_seq"][i]); GF[idx, j], GN[idx, j] = Fo[:, vstar], No[:, vstar]
            gs, gp = es_(GF); spec_t = 1 - GF[neg].mean(0); Xs = np.sort(LS[:, neg, :], 1); q = np.ceil(spec_t * neg.sum() - 1e-9).astype(int)
            thr = np.stack([Xs[:, q[j] - 1, j] if q[j] > 0 else np.full(NSEEDS, -np.inf) for j in range(10)], 1); RM = LS > thr[:, None, :]
            ts_, tp_ = es_(TE)
            out.append({"t_low_training_sensitivity": ql, "t_high_training_specificity": qh,
                        "gate_most_chosen_variant": {"endoscopy_sensitivity": r3(gs), "endoscopy_specificity": r3(gp), "share_samples_sequenced": r3(GN.sum(0).mean() / tot), "share_endoscopies_sequenced": r3((GN > 0).mean())},
                        "triage": {"endoscopy_sensitivity": r3(ts_), "endoscopy_specificity": r3(tp_), "share_samples_sequenced": r3(tsh), "share_endoscopies_sequenced": r3(tshe)},
                        "random_one_matched_to_gate_endoscopy_specificity": {"endoscopy_sensitivity": r3(RM[:, elab == 1, :].mean()), "endoscopy_specificity": r3(1 - RM[:, neg, :].mean()), "share_samples_sequenced": r3(E / tot), "share_endoscopies_sequenced": 1.0}})
    res["curve"] = {"variant": VLAB[vstar], "grid": out, "random_one_own_threshold": {"endoscopy_sensitivity": r3(own[0]), "endoscopy_specificity": r3(own[1]), "share_samples_sequenced": r3(E / tot), "share_endoscopies_sequenced": 1.0},
                    "note": "descriptive, no intervals; the most-chosen variant applied in every fold (not nested) with cut-offs at each grid point (sample-level: the triage rule at those quantiles; endoscopy-level: the same quantiles on inner endoscopy scores); t_seq and the L-CNV threshold from the triage at that grid point; random-one has no cut-off quantiles, so it is shown at its own threshold and matched to the gate's endoscopy specificity at each grid point (evaluation-matched, per repeat)"}
    dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
