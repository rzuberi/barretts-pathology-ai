"""Review round 1, items R1-R12 (docs/paper_review_round1_analyses.md @ baa47ab). TASK = r1_<src>_<pop> | r2 | r3 | ... | r12.
Stored predictions only (tb_common.py loaders and paper_triage.md rules). Aggregates -> results/paper_final/review_round1/<TASK>.json."""
import os, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from tb_common import *
OUTR = T + "/results/paper_final/review_round1"; os.makedirs(OUTR, exist_ok=True); TASK = os.environ["TASK"]
CS = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample").reindex(ids)
YEAR = pd.to_numeric(CS.year, errors="coerce").values
GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float)
SDS = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); TILES = SDS.tissue_tiles.values.astype(float); SCAN = SDS.scanner.values
TIME = SM.time.values.astype(float); EV = SM.event.values.astype(int); MBF = SM.mbf.values.astype(float); KP = pd.to_numeric(SM.k_prob, errors="coerce").values
FO0 = load("their")[2]
def fsa(S, b, mask=None, w=None):
    """fold-stratified AUROC (mean over repeats of within-fold pooled pair concordance); S n x 10 or n; optional sample mask and per-sample weights (pair weight w_i w_j)"""
    out = []
    for j in range(10):
        x = (S[:, j] if np.ndim(S) == 2 else S)[b]; f = FO0[b, j]; l = y[b]; ok = np.ones(len(b), bool) if mask is None else mask[b]; num = den = 0.0
        for k in range(1, 11):
            m = ok & (f == k) & np.isfinite(x); yk = l[m]; n1 = int(yk.sum()); n0 = len(yk) - n1
            if n1 == 0 or n0 == 0: continue
            if w is None:
                rk = rankdata(x[m]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
            else:
                xs = x[m]; ws = w[b][m]; xp, xn, wp, wn = xs[yk == 1], xs[yk == 0], ws[yk == 1], ws[yk == 0]
                W = wp[:, None] * wn[None, :]; num += (((xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])) * W).sum(); den += W.sum()
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
def ci2(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return pct(a)
def lo1(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return r3(np.percentile(a, 5))
def ss(c, b): s, p = sens_spec(c, b); return s, p
def dump(res): res["prespec"] = "baa47ab"; json.dump(res, open(f"{OUTR}/{TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV DONE", TASK, flush=True)
def models(src):
    O, I, FO = load(src); return O, I, FO
res = {"task": TASK}
# ------------------------------------------------------------------ R1 matched-specificity comparators
if TASK.startswith("r1_"):
    _, src, pop = TASK.split("_", 2); popm = popmask(pop); rows = np.where(popm)[0]; O, I, FO = models(src)
    band, call, info = triage(O, I, FO, popm); comp, _ = comparators(O, I, FO, popm); CT = vote(call)
    MM = {"A*": "L-CNV", "B*": "L-LATE", "C*": "L-IMG"}; mc = {a: np.zeros((n, 10), bool) for a in MM}; sstar = []
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            si = I["L-IMG"][(r, k)]; tri = si.index.values[popm[si.index.values]]; s_tr = si.reindex(tri).values; yc = y[tri] == 1
            tl = thr_sens(s_tr[yc], 0.95); cs = c_star(s_tr[~yc], 0.90); li = I["L-LATE"][(r, k)].reindex(tri).values
            mid = (s_tr >= tl) & (s_tr <= cs) if tl <= cs else np.zeros(len(tri), bool)
            cm = li[mid & yc]; ts = thr_sens(cm, 0.8) if len(cm) >= 5 else thr_sens(li[yc], 0.8)
            tcall = np.where(s_tr < tl, False, np.where(s_tr > cs, True, li >= ts)) if tl <= cs else s_tr > cs
            s_star = 1 - tcall[~yc].mean(); sstar.append(s_star); te = FO[:, j] == k
            for a, mn in MM.items():
                xc = np.sort(I[mn][(r, k)].reindex(tri).values[~yc]); q = int(np.ceil(s_star * len(xc))); thr = xc[q - 1] if q > 0 else -np.inf
                mc[a][te, j] = O[mn][te, j] > thr
    CX = {a: vote(mc[a]) for a in MM}; boots = boots_for(rows)
    def met(b): o = {"T": ss(CT, b)}; o.update({a: ss(CX[a], b) for a in MM}); return o
    obs = met(rows); BS = [met(b) for b in boots]
    R1 = {"training_specificity_target": {"median": r3(np.median(sstar)), "range": [r3(min(sstar)), r3(max(sstar))]}, "triage": {"sensitivity": r3(obs["T"][0]), "specificity": r3(obs["T"][1])}, "matched": {}}
    for a in MM:
        ds = np.array([x["T"][0] - x[a][0] for x in BS]); dp = np.array([x["T"][1] - x[a][1] for x in BS])
        R1["matched"][a] = {"model": MM[a], "sensitivity": r3(obs[a][0]), "sens_ci95": ci2([x[a][0] for x in BS]), "specificity": r3(obs[a][1]), "spec_ci95": ci2([x[a][1] for x in BS]),
                            "d_sens_triage_minus": {"delta": r3(obs["T"][0] - obs[a][0]), "ci95": ci2(ds), "p_two_sided": p2(ds)}, "d_spec_triage_minus": {"delta": r3(obs["T"][1] - obs[a][1]), "ci95": ci2(dp), "p_two_sided": p2(dp)}}
    # secondary: test-set matching on repeat-mean scores
    SEC = {}
    for a, mn in MM.items():
        Sx = O[mn].mean(1)
        def at_spec(b, spec_t):
            xb = Sx[b]; yb = y[b]; xc = np.sort(xb[yb == 0]); q = int(np.ceil(spec_t * len(xc))); thr = xc[q - 1] if q > 0 else -np.inf
            return float(np.mean(xb[yb == 1] > thr)), float(np.mean(xb[yb == 0] <= thr))
        o = at_spec(rows, obs["T"][1]); d = []
        for b in boots:
            st, pt = ss(CT, b); sx, _ = at_spec(b, pt); d.append(st - sx)
        SEC[a] = {"sensitivity_at_triage_specificity": r3(o[0]), "realised_specificity": r3(o[1]), "d_sens_triage_minus": {"delta": r3(obs["T"][0] - o[0]), "ci95": ci2(d), "p_two_sided": p2(d)}}
    R1["secondary_test_set_matching"] = SEC; res.update({"src": src, "pop": pop, "n_samples": int(len(rows)), "R1": R1}); dump(res)
# ------------------------------------------------------------------ R2 patient-level discrimination
elif TASK == "r2":
    out = {}
    for src in ("their", "pkg"):
        O, I, FO = models(src); popm = popmask("pre"); rows = np.where(popm)[0]; up = np.unique(pat[rows]); pidx = {p: rows[pat[rows] == p] for p in up}
        PY = np.array([y[pidx[p]][0] for p in up]); EARL = np.array([pidx[p][np.argmax(MBF[pidx[p]])] for p in up]); PF = np.array([[FO[pidx[p][0], j] for j in range(10)] for p in up])
        SC = {m: O[m] for m in ("L-CNV", "L-IMG", "L-LATE")}; SC["P"] = np.repeat(KP[:, None], 10, 1)
        PM = {m: np.array([[S[pidx[p], j].mean() for j in range(10)] for p in up]) for m, S in SC.items()}; PE = {m: S[EARL] for m, S in SC.items()}
        rng = np.random.RandomState(0); draws = []
        while len(draws) < NB:
            ix = rng.choice(len(up), len(up)); bs = np.concatenate([pidx[up[i]] for i in ix])
            if 0 < y[bs].sum() < len(bs): draws.append(ix)
        def pfa(Pm, ix):
            outs = []
            for j in range(10):
                x = Pm[ix, j]; f = PF[ix, j]; l = PY[ix]; num = den = 0.0
                for k in range(1, 11):
                    m = f == k; yk = l[m]; n1 = yk.sum(); n0 = len(yk) - n1
                    if n1 == 0 or n0 == 0: continue
                    rk = rankdata(x[m]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
                if den: outs.append(num / den)
            return float(np.mean(outs))
        nps = np.array([len(pidx[p]) for p in up]); wsamp = np.zeros(n);
        for p in up: wsamp[pidx[p]] = 1.0 / len(pidx[p])
        allix = np.arange(len(up)); o = {}
        for lab, fn in (("patient_mean", lambda m, ix: pfa(PM[m], ix)), ("patient_earliest", lambda m, ix: pfa(PE[m], ix)),
                        ("patient_weighted_samples", lambda m, ix: fsa(SC[m], np.concatenate([pidx[up[i]] for i in ix]), w=wsamp))):
            obs = {m: fn(m, allix) for m in SC}; B = {m: np.array([fn(m, ix) for ix in draws]) for m in SC}
            o[lab] = {"auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in SC},
                      "IMG_minus_CNV": {"delta": r3(obs["L-IMG"] - obs["L-CNV"]), "ci95_two_sided": ci2(B["L-IMG"] - B["L-CNV"]), "one_sided_95_lower": lo1(B["L-IMG"] - B["L-CNV"]), "non_inferior_at_-0.05": bool(lo1(B["L-IMG"] - B["L-CNV"]) > -0.05)},
                      "LATE_minus_CNV": {"delta": r3(obs["L-LATE"] - obs["L-CNV"]), "ci95": ci2(B["L-LATE"] - B["L-CNV"]), "p_two_sided": p2(B["L-LATE"] - B["L-CNV"])}}
        out[src] = {"n_patients": int(len(up)), "n_progressor_patients": int(PY.sum()), **o}
    res["R2"] = out; dump(res)
# ------------------------------------------------------------------ R3 calibration
elif TASK == "r3":
    import statsmodels.api as sm
    O, I, FO = models("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); out = {}
    def cal(p, yy):
        lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        f = sm.GLM(yy, sm.add_constant(lp), family=sm.families.Binomial()).fit(); g = sm.GLM(yy, np.ones((len(yy), 1)), family=sm.families.Binomial(), offset=lp).fit()
        return f.params[1], f.params[0], g.params[0]
    for m in ("L-CNV", "L-IMG", "L-LATE"):
        S = O[m].mean(1); o = cal(S[rows], y[rows]); B = np.array([cal(S[b], y[b]) for b in boots])
        dq = pd.qcut(S[rows], 10, labels=False, duplicates="drop"); dec = [{"decile": int(d), "mean_pred": r3(S[rows][dq == d].mean()), "observed": r3(y[rows][dq == d].mean()), "n": int((dq == d).sum())} for d in np.unique(dq)]
        out[m] = {"slope": {"value": r3(o[0]), "ci95": ci2(B[:, 0])}, "intercept": {"value": r3(o[1]), "ci95": ci2(B[:, 1])}, "calibration_in_the_large": {"value": r3(o[2]), "ci95": ci2(B[:, 2])}, "mean_pred": r3(S[rows].mean()), "observed": r3(y[rows].mean()), "deciles": dec}
    res["R3"] = out; dump(res)
# ------------------------------------------------------------------ R4 scanner like-for-like + transfer triage
elif TASK == "r4":
    O, I, FO = models("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); out = {"a_main_cv_by_scanner": {}}
    for sc in ("C13210", "C13239-01"):
        mk = SCAN == sc; o = {}
        for m in ("L-IMG", "L-CNV", "L-LATE"):
            v = fsa(O[m], rows, mask=mk); B = [fsa(O[m], b, mask=mk) for b in boots]; o[m] = {"value": r3(v), "ci95": ci2(B)}
        out["a_main_cv_by_scanner"][sc] = {"n_samples": int((popm & mk).sum()), "n_patients": int(len(set(pat[popm & mk]))), **o}
    X = pd.read_csv(f"{ROW}/scanner/cross_b_s39_none.csv", dtype={"Sample": str, "Patient": str}); te = X[X.role == "test"]; inn = X[X.role == "inner"]
    yi = inn.y.values == 1; tl = thr_sens(inn.pred.values[yi], 0.95); cs = c_star(inn.pred.values[~yi], 0.90)
    FZ = json.load(open(T + "/results/paper_final/triage_robustness/frozen_cutoffs_their.json"))["cutoffs"]; tseq = FZ["t_seq"]
    tix = ix[te.Sample].values; img = te.pred.values; late = (img + O["L-CNV"].mean(1)[tix]) / 2
    bandt = np.where(img < tl, 0, np.where(img > cs, 2, 1)); callt = np.where(bandt == 1, late >= tseq, bandt == 2)
    band_m, call_m, _ = triage(O, I, FO, popm); CT = vote(call_m); s10 = rows[SCAN[rows] == "C13210"]
    def sp_on(c_full, idx): yy = y[idx]; return float(c_full[idx][yy == 1].mean()), float(1 - c_full[idx][yy == 0].mean())
    cfull = np.zeros(n, bool); cfull[tix] = callt; bfull = np.full(n, -1); bfull[tix] = bandt
    pp = pat[s10]; upp = np.unique(pp); ro = {q: s10[pp == q] for q in upp}; rng = np.random.RandomState(0); bb = []
    while len(bb) < NB:
        b = np.concatenate([ro[upp[i]] for i in rng.choice(len(upp), len(upp))])
        if 0 < y[b].sum() < len(b): bb.append(b)
    o_t = sp_on(cfull, s10); o_m = sp_on(CT, s10)
    out["b_transfer_triage_C13210"] = {"cutoffs": {"t_low": r3(tl), "c_star": r3(cs), "t_seq": r3(tseq)}, "n_samples": int(len(s10)), "n_patients": int(len(upp)), "n_P_patients": int(len(set(pat[s10][y[s10] == 1]))),
        "transfer": {"sensitivity": r3(o_t[0]), "sens_ci95": ci2([sp_on(cfull, b)[0] for b in bb]), "specificity": r3(o_t[1]), "spec_ci95": ci2([sp_on(cfull, b)[1] for b in bb]),
                     "band_shares": {nm: r3(np.mean(bfull[s10] == g)) for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}},
        "main_cv_triage_on_C13210": {"sensitivity": r3(o_m[0]), "sens_ci95": ci2([sp_on(CT, b)[0] for b in bb]), "specificity": r3(o_m[1]), "spec_ci95": ci2([sp_on(CT, b)[1] for b in bb]),
                     "band_shares_mean_over_repeats": {nm: r3((band_m[s10] == g).mean()) for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}}}
    W = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(f"{ROW}/scanner/within_s10_*.csv"))]); cc = []
    for rep, g in W.groupby("rep"):
        nf = 0; npairs = 0
        for k, h in g.groupby("fold"):
            n1 = int(h.y.sum()); n0 = len(h) - n1
            if n1 and n0: nf += 1; npairs += n1 * n0
        cc.append({"rep": int(rep), "folds_with_both_classes": nf, "within_fold_pairs": npairs})
    out["c_within_C13210_pairs"] = {"per_repeat": cc, "all_pairs_without_folds": int(W.drop_duplicates("Sample").y.sum() * (W.drop_duplicates("Sample").y == 0).sum())}
    res["R4"] = out; dump(res)
# ------------------------------------------------------------------ R5 technical confounders
elif TASK == "r5":
    O, I, FO = models("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); out = {}
    for nm, sc in (("year", YEAR), ("tissue_tiles", TILES)):
        out[f"a_auroc_{nm}"] = {"value": r3(fsa(sc, rows)), "ci95": ci2([fsa(sc, b) for b in boots])}
    for nm, mk in (("overlap_lt256", TILES < 256), ("no_overlap_ge256", TILES >= 256)):
        out[f"b_LIMG_{nm}"] = {"n_samples": int((popm & mk).sum()), "value": r3(fsa(O["L-IMG"], rows, mask=mk)), "ci95": ci2([fsa(O["L-IMG"], b, mask=mk) for b in boots])}
    Sm = O["L-IMG"].mean(1); lt = np.log(TILES)
    for nm, cov in (("year", YEAR), ("log_tiles", lt)):
        for cl in (1, 0):
            def rho(b): bb = b[y[b] == cl]; return spearmanr(Sm[bb], cov[bb]).correlation
            out[f"c_spearman_LIMG_{nm}_{'progressor' if cl else 'nonprogressor'}"] = {"value": r3(rho(rows)), "ci95": ci2([rho(b) for b in boots])}
    R = np.zeros_like(O["L-IMG"]); Xc = np.column_stack([np.ones(n), YEAR, lt])
    for j in range(10):
        coef, *_ = np.linalg.lstsq(Xc[rows], O["L-IMG"][rows, j], rcond=None); R[:, j] = O["L-IMG"][:, j] - Xc @ coef
    out["d_LIMG_residualised"] = {"value": r3(fsa(R, rows)), "ci95": ci2([fsa(R, b) for b in boots]), "unadjusted": r3(fsa(O["L-IMG"], rows))}
    res["R5"] = out; dump(res)
# ------------------------------------------------------------------ R6 timing
elif TASK == "r6":
    O, I, FO = models("their"); popm = popmask("pre"); rows = np.where(popm)[0]; cr = rows[y[rows] == 1]; tc = TIME[cr]
    out = {"time_to_event_progressor_samples": {"median": r3(np.median(tc)), "iqr": [r3(np.percentile(tc, 25)), r3(np.percentile(tc, 75))],
           "n_le_0.5y": int((tc <= 0.5).sum()), "patients_le_0.5y": int(len(set(pat[cr][tc <= 0.5]))), "n_le_1y": int((tc <= 1).sum()), "patients_le_1y": int(len(set(pat[cr][tc <= 1])))}}
    keep = popm & ~((y == 1) & (TIME <= 1)); rows2 = np.where(keep)[0]; boots = boots_for(rows2)
    SC = {m: O[m] for m in ("L-CNV", "L-IMG", "L-LATE")}; SC["P"] = KP
    obs = {m: fsa(S, rows2) for m, S in SC.items()}; B = {m: np.array([fsa(S, b) for b in boots]) for m, S in SC.items()}
    d = B["L-IMG"] - B["L-CNV"]
    out["excluding_progressor_samples_within_1y"] = {"n_samples": int(len(rows2)), "n_progressor_samples": int(y[rows2].sum()), "n_patients": int(len(set(pat[rows2]))), "auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in SC},
        "IMG_minus_CNV": {"delta": r3(obs["L-IMG"] - obs["L-CNV"]), "ci95": ci2(d), "one_sided_95_lower": lo1(d), "non_inferior_at_-0.05": bool(lo1(d) > -0.05)}}
    band, call, _ = triage(O, I, FO, popm); comp, _ = comparators(O, I, FO, popm)
    CALL = {"triage_LATE": vote(call), **{a: vote(comp[a]) for a in comp}}
    out["excluding_progressor_samples_within_1y"]["sensitivity_stored_calls"] = {s: r3(c[rows2][y[rows2] == 1].mean()) for s, c in CALL.items()}
    res["R6"] = out; dump(res)
# ------------------------------------------------------------------ R7 worst grade to date
elif TASK == "r7":
    O, I, FO = models("their"); out = {}
    WG = np.full(n, np.nan)
    for p in np.unique(pat):
        idx = np.where((pat == p) & PRE)[0]
        for i in idx: WG[i] = np.nanmax(GR[idx][MBF[idx] >= MBF[i]])
    for pop in ("pre", "pre_ndbe"):
        popm = popmask(pop); rows = np.where(popm)[0]; boots = boots_for(rows); SC = {"worst_grade_to_date": WG, "L-CNV": O["L-CNV"], "L-IMG": O["L-IMG"], "L-LATE": O["L-LATE"]}
        obs = {m: fsa(S, rows) for m, S in SC.items()}; B = {m: np.array([fsa(S, b) for b in boots]) for m, S in SC.items()}
        out[pop] = {"distinct_worst_grades": sorted(set(np.unique(WG[rows]).tolist())), "auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in SC},
                    "delta_vs_worst_grade": {m: {"delta": r3(obs[m] - obs["worst_grade_to_date"]), "ci95": ci2(B[m] - B["worst_grade_to_date"])} for m in ("L-CNV", "L-IMG", "L-LATE")}}
    res["R7"] = out; dump(res)
# ------------------------------------------------------------------ R8 no-vote triage
elif TASK == "r8":
    out = {}
    for src in ("their", "pkg"):
        O, I, FO = models(src); popm = popmask("pre"); rows = np.where(popm)[0]; band, call, _ = triage(O, I, FO, popm); comp, _ = comparators(O, I, FO, popm)
        CALLS = {"triage_LATE": call, **comp}; o = {}; ppat = pd.Series(y[rows], index=pat[rows]).groupby(level=0).max()
        for s, c in CALLS.items():
            se = [c[rows, j][y[rows] == 1].mean() for j in range(10)]; sp = [1 - c[rows, j][y[rows] == 0].mean() for j in range(10)]
            miss, fp = [], []
            for j in range(10):
                pos = pd.Series(c[rows, j], index=pat[rows]).groupby(level=0).max(); miss.append(int(((ppat == 1) & ~pos).sum())); fp.append(int(((ppat == 0) & pos).sum()))
            o[s] = {"sensitivity_mean": r3(np.mean(se)), "sensitivity_range": [r3(min(se)), r3(max(se))], "specificity_mean": r3(np.mean(sp)), "specificity_range": [r3(min(sp)), r3(max(sp))],
                    "patients_missed_mean": r3(np.mean(miss)), "patients_missed_range": [min(miss), max(miss)], "nonprog_patients_fp_mean": r3(np.mean(fp)), "nonprog_patients_fp_range": [min(fp), max(fp)]}
        sh = [(band[rows, j] == 1).mean() for j in range(10)]
        o["share_sequenced_per_repeat"] = {"mean": r3(np.mean(sh)), "range": [r3(min(sh)), r3(max(sh))]}; o["share_middle_band_in_at_least_one_repeat"] = r3((band[rows] == 1).any(1).mean())
        out[src] = o
    res["R8"] = out; dump(res)
# ------------------------------------------------------------------ R9 score correlation
elif TASK == "r9":
    out = {}
    for src in ("their", "pkg"):
        O, I, FO = models(src); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); a = O["L-IMG"].mean(1); c = O["L-CNV"].mean(1); o = {}
        for cl, nm in ((1, "progressor_samples"), (0, "nonprogressor_samples"), (None, "all")):
            def rho(b): bb = b if cl is None else b[y[b] == cl]; return spearmanr(a[bb], c[bb]).correlation
            o[nm] = {"spearman": r3(rho(rows)), "ci95": ci2([rho(b) for b in boots])}
        out[src] = o
    res["R9"] = out; dump(res)
# ------------------------------------------------------------------ R10 C-index without within-patient pairs
elif TASK == "r10":
    O, I, FO = models("their"); popm = popmask("pre"); rows = np.where(popm)[0]; TAU = 8.0
    REPM = {"L-CNV": O["L-CNV"], "L-LATE": O["L-LATE"], "P": np.repeat(KP[:, None], 10, 1)}
    def stats(rr):
        t, d = TIME[rr], EV[rr]; evi = (d == 1) & (t < TAU); pp = pat[rr]; base = evi[:, None] & (t[:, None] < t[None, :]) & (pp[:, None] != pp[None, :]); o = {}
        for m, S in REPM.items():
            H = []
            for j in range(10):
                f = FO[rr, j]; I_, J_ = np.nonzero(base & (f[:, None] == f[None, :])); s = S[rr, j]
                c = (s[I_] > s[J_]) + 0.5 * (s[I_] == s[J_]); H.append(c.mean() if len(I_) else np.nan)
            o[m] = float(np.nanmean(H))
        return o
    obs = stats(rows); boots = boots_for(rows); BS = [stats(b) for b in boots]
    out = {m: {"harrell_excl_within_patient": r3(obs[m]), "ci95": ci2([x[m] for x in BS])} for m in REPM}
    for a, b2 in (("L-LATE", "L-CNV"), ("L-LATE", "P")):
        dd = np.array([x[a] - x[b2] for x in BS]); out[f"{a}_minus_{b2}"] = {"delta": r3(obs[a] - obs[b2]), "ci95": ci2(dd), "p_two_sided": p2(dd)}
    res["R10"] = out; dump(res)
# ------------------------------------------------------------------ R11 proportional hazards
elif TASK == "r11":
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    O, I, FO = models("their"); popm = popmask("pre"); rows = np.where(popm)[0]; out = {}
    SC = {"P": KP, "L-CNV": O["L-CNV"].mean(1), "L-LATE": O["L-LATE"].mean(1)}
    for m, S in SC.items():
        s = S[rows]; k = np.where(s <= 0.3, 0, np.where(s < 0.5, 1, 2))
        df = pd.DataFrame({"t": TIME[rows], "d": EV[rows], "moderate": (k == 1).astype(float), "high": (k == 2).astype(float)})
        cph = CoxPHFitter().fit(df, "t", "d"); pt = proportional_hazard_test(cph, df, time_transform="rank")
        out[m] = {"p_moderate": r3(float(pt.summary.loc["moderate", "p"])), "p_high": r3(float(pt.summary.loc["high", "p"])), "note": "Efron ties, non-robust fit for the test; rank time transform"}
    res["R11"] = out; dump(res)
# ------------------------------------------------------------------ R12 band predictive values
elif TASK == "r12":
    O, I, FO = models("their"); popm = popmask("pre"); rows = np.where(popm)[0]; band, call, _ = triage(O, I, FO, popm); boots = boots_for(rows)
    def rates(b):
        yb = y[b]; bb = band[b]; return {g: ((bb[yb == 1] == g).mean(0).mean(), (bb[yb == 0] == g).mean(0).mean()) for g in (0, 1, 2)}
    def pv(rt, pi):
        h1, h0 = rt[2]; l1, l0 = rt[0]; m1, m0 = rt[1]
        return {"PPV_high_band": pi * h1 / (pi * h1 + (1 - pi) * h0), "NPV_low_band": (1 - pi) * l0 / ((1 - pi) * l0 + pi * l1), "progressor_share_middle_band": pi * m1 / (pi * m1 + (1 - pi) * m0)}
    obs_rt = rates(rows); B_rt = [rates(b) for b in boots]; out = {}
    for pi in (0.02, 0.05, 0.10, float(y[rows].mean())):
        o = pv(obs_rt, pi); Bv = [pv(r, pi) for r in B_rt]; key = "observed" if pi > 0.2 else f"{pi:.2f}"
        out[key] = {"prevalence": r3(pi), **{k: {"value": r3(v), "ci95": ci2([x[k] for x in Bv])} for k, v in o.items()}}
    res["R12"] = out; dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
