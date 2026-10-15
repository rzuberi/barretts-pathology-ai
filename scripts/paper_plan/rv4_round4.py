"""Review round 4, Part A, items R31-R35 (docs/paper_review_round4_analyses.md @ 45b77bf). Stored predictions, the sample sheet and the round-2 refit
predictions only (tb_common.py loaders and the paper_triage.md rules; rv2_round2.py conventions for the no-vote statistics, the training-matched A* B* C*
and the evaluation-matched comparators). Aggregates -> results/paper_final/review_round4/<TASK>.json; row-level shard outputs stay under
killcoyne_mm/review_round4/. Items 11 (deviance lambda, needs B5) and 19 (R36) of R33 are filled when their Part B files exist and marked pending otherwise;
r33_merge is re-runnable.
TASK = r31 | r32a | r32b | r33_base_<name> | r33_grid_<ql>_<qh> | r33_selci_shardNN | r33_selci_merge | r33_r15 | r33_f3 | r33_merge | r34 | r35
       (<name> primary | ndbe | pkg | f1 | f1pkg | neverdys | f5 | excl1y | earliest_endo | earliest_sample | devlam | r36 | r36past)"""
import os, sys, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, fisher_exact
from scipy.special import gammaln
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tb_common import *
from tb_common import _outer, _inner
PRESPEC = "45b77bf"; OUT4 = T + "/results/paper_final/review_round4"; ROW4 = M + "/review_round4"; RR2 = M + "/review_round2"; SR = M + "/scanner_robustness"; OUT2 = T + "/results/paper_final/review_round2"
os.makedirs(OUT4, exist_ok=True); os.makedirs(ROW4, exist_ok=True); TASK = os.environ["TASK"]
CS = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample").reindex(ids); YEAR = pd.to_numeric(CS.year, errors="coerce").values
GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float)
SDS = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); TILES = SDS.tissue_tiles.values.astype(float)
TIME = SM.time.values.astype(float); MBF = SM.mbf.values.astype(float)
FO0 = load("their")[2]; MM = {"A*": "L-CNV", "B*": "L-LATE", "C*": "L-IMG"}; COMPM = {"A": "L-CNV", "B": "L-LATE", "C": "L-IMG"}; MODELS = ("L-CNV", "L-IMG", "L-LATE")
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv")); hzi = lambda nm: sorted(glob.glob(f"{HZ}/inner/{nm}_rep_*_fold_*.csv"))
# ------------------------------------------------------------------ helpers (fsa / ci / dstat / dump as in rv2_round2.py)
def fsa(S, b, mask=None, w=None, cell=None, reps=None):
    """fold-stratified AUROC (mean over repeats of within-fold pooled pair concordance); S n x 10 or n; optional mask, pair weights w_i w_j, an extra
    per-sample cell key (pairs must share fold and cell; cell < 0 = excluded) and a subset of repeats"""
    out = []
    for j in (range(10) if reps is None else reps):
        x = (S[:, j] if np.ndim(S) == 2 else S)[b]; f = FO0[b, j]; l = y[b]; ok = np.ones(len(b), bool) if mask is None else mask[b].copy(); num = den = 0.0
        if cell is not None: cb = cell[b]; ok &= cb >= 0; f = f + 100 * np.where(cb < 0, 0, cb)
        for k in np.unique(f[ok]):
            m = ok & (f == k) & np.isfinite(x); yk = l[m]; n1 = int(yk.sum()); n0 = len(yk) - n1
            if n1 == 0 or n0 == 0: continue
            if w is None:
                rk = rankdata(x[m]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
            else:
                xs = x[m]; ws = w[b][m]; xp, xn, wp, wn = xs[yk == 1], xs[yk == 0], ws[yk == 1], ws[yk == 0]
                W = wp[:, None] * wn[None, :]; num += (((xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])) * W).sum(); den += W.sum()
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
def pairs_within(b, cell):   # per repeat: within-fold case-control pairs that also share the cell
    out = []
    for j in range(10):
        f = FO0[b, j] + 100 * cell[b]; l = y[b]; out.append(int(sum(int((l[f == k] == 1).sum()) * int((l[f == k] == 0).sum()) for k in np.unique(f))))
    return out
def auc_plain(yy, s):
    yy = np.asarray(yy).astype(int); ok = np.isfinite(s); yy, s = yy[ok], s[ok]; r = rankdata(s); n1 = yy.sum(); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(yy) else np.nan
def ci2(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return pct(a) if len(a) else [None, None]
def lo1(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return r3(np.percentile(a, 5)) if len(a) else None
def dstat(obs, draws, one_sided=False):
    o = {"delta": r3(obs), "ci95": ci2(draws), "p_two_sided": p2(draws)}
    if one_sided: o["one_sided_95_lower"] = lo1(draws); o["non_inferior_at_-0.05"] = bool(lo1(draws) is not None and lo1(draws) > -0.05)
    return o
def dump(res, name=None):
    res["prespec"] = PRESPEC; json.dump(res, open(f"{OUT4}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV4 DONE", name or TASK, flush=True)
def J(path):
    try: return json.load(open(path))
    except FileNotFoundError: return None
def sens_at_spec(x, yb, spec_t):   # R1 secondary rule: positive if score > the ceil(spec * n_controls)-th smallest held-out control score
    xc = np.sort(x[yb == 0]); q = int(np.ceil(spec_t * len(xc))); thr = xc[q - 1] if q > 0 else -np.inf; return float(np.mean(x[yb == 1] > thr))
def wmean(v, w): return float(v.mean()) if w is None else float((v * w).sum() / w.sum())
def frame(BS): return pd.DataFrame(BS)
def keys_of(d): return sorted(d.keys())
def nfo(rows): return {"n_samples": int(len(rows)), "n_progressor_samples": int(y[rows].sum()), "n_patients": int(len(set(pat[rows]))), "n_progressor_patients": int(len(set(pat[rows][y[rows] == 1])))}
# ------------------------------------------------------------------ calls: triage (any cut-off cell, any middle-band model), (A)(B)(C), training-matched A* B* C* re-matched to that triage's s*
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
    band, call, info = triage(O, I, FO, popm, ql, qh, mid=mid, W=W); comp, thr = comparators(O, I, FO, popm, W=W); mc, sstar = matched_calls(O, I, FO, popm, W=W, ql=ql, qh=qh, mid=mid)
    CALLS = {"T": call, "A": comp["A"], "B": comp["B"], "C": comp["C"], "A*": mc["A*"], "B*": mc["B*"], "C*": mc["C*"]}; return CALLS, band, info, thr, sstar
def draw_stats(b, CALLS, band, O, per_rep=False, w=None):
    """voted (v_) and no-vote (nv_, repeat-averaged) sensitivity / specificity of every call set (optionally sample-weighted), evaluation-matched A* B* C*
    (no vote: per repeat at that repeat's realised specificity; voted: repeat-mean scores at the voted specificity), band shares, class-wise band rates"""
    yb = y[b]; P = yb == 1; N = ~P; out = {}; rep = {}; wb = None if w is None else w[b]
    def cm(Cb, mk): return Cb[mk].mean(0) if wb is None else (Cb[mk] * wb[mk][:, None]).sum(0) / wb[mk].sum()
    for s, C in CALLS.items():
        Cb = C[b]; V = Cb.sum(1) >= 6; se = cm(Cb, P); sp = 1 - cm(Cb, N)
        out[f"v_sens_{s}"] = wmean(V[P].astype(float), wb if wb is None else wb[P]); out[f"v_spec_{s}"] = 1 - wmean(V[N].astype(float), wb if wb is None else wb[N])
        out[f"nv_sens_{s}"] = se.mean(); out[f"nv_spec_{s}"] = sp.mean(); rep[f"sens_{s}"] = se; rep[f"spec_{s}"] = sp
    spec_j = rep["spec_T"]
    for a, m in MM.items():
        out[f"v_sens_{a}_eval"] = sens_at_spec(O[m].mean(1)[b], yb, out["v_spec_T"]); sj = np.array([sens_at_spec(O[m][b, j], yb, spec_j[j]) for j in range(10)])
        out[f"nv_sens_{a}_eval"] = sj.mean(); rep[f"sens_{a}_eval"] = sj
    Bb = band[b]; mid = Bb == 1
    for nm, mk in (("", np.ones(len(b), bool)), ("_P", P), ("_N", N)):
        out[f"share_mid{nm}"] = cm(mid, mk).mean(); out[f"share_mid_any{nm}"] = wmean(mid[mk].any(1).astype(float), wb if wb is None else wb[mk])
    for g in (0, 1, 2): out[f"rate_P_{g}"] = cm(Bb == g, P).mean(); out[f"rate_N_{g}"] = cm(Bb == g, N).mean()
    return (out, rep) if per_rep else out
def paired(obs, D, s, mode, one_sided=False, eval_=False):
    k = f"{mode}_sens_{s}{'_eval' if eval_ else ''}"; o = {"d_sens_triage_minus": dstat(obs[f"{mode}_sens_T"] - obs[k], D[f"{mode}_sens_T"] - D[k], one_sided)}
    if not eval_: o["d_spec_triage_minus"] = dstat(obs[f"{mode}_spec_T"] - obs[f"{mode}_spec_{s}"], D[f"{mode}_spec_T"] - D[f"{mode}_spec_{s}"])
    o["comparator_sensitivity"] = r3(obs[k]); o["comparator_sens_ci95"] = ci2(D[k])
    if not eval_: o["comparator_specificity"] = r3(obs[f"{mode}_spec_{s}"]); o["comparator_spec_ci95"] = ci2(D[f"{mode}_spec_{s}"])
    return o
def trow(obs, D):
    """the R33 entries of one analysis: Claim T (no vote, training-matched B*), the pre-specified test (vs (B), voted and no vote) and the other matchings"""
    return {"triage_no_vote": {"sensitivity": r3(obs["nv_sens_T"]), "sens_ci95": ci2(D["nv_sens_T"]), "specificity": r3(obs["nv_spec_T"]), "spec_ci95": ci2(D["nv_spec_T"])},
            "triage_voted": {"sensitivity": r3(obs["v_sens_T"]), "sens_ci95": ci2(D["v_sens_T"]), "specificity": r3(obs["v_spec_T"]), "spec_ci95": ci2(D["v_spec_T"])},
            "share_sequenced": {"per_repeat_mean": r3(obs["share_mid"]), "ci95": ci2(D["share_mid"]), "in_at_least_one_repeat": r3(obs["share_mid_any"]), "cap_0.70_met": bool(obs["share_mid"] < 0.70)},
            "no_vote_vs_Bstar_training_matched": paired(obs, D, "B*", "nv", one_sided=True), "no_vote_vs_Bstar_evaluation_matched": paired(obs, D, "B*", "nv", one_sided=True, eval_=True),
            "no_vote_vs_B": paired(obs, D, "B", "nv", one_sided=True), "voted_vs_Bstar_training_matched": paired(obs, D, "B*", "v", one_sided=True), "voted_vs_B": paired(obs, D, "B", "v", one_sided=True),
            "no_vote_vs_Cstar_training_matched": paired(obs, D, "C*", "nv", one_sided=True), "no_vote_vs_C": paired(obs, D, "C", "nv")}
def hrow(O, rows, boots, mask=None, cell=None, w=None):
    """Claim H entries: fold-stratified AUROC of the three models and the paired differences"""
    obs = {m: fsa(O[m], rows, mask=mask, cell=cell, w=w) for m in MODELS}; B = frame([{m: fsa(O[m], b, mask=mask, cell=cell, w=w) for m in MODELS} for b in boots])
    return {"auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in MODELS}, "IMG_minus_CNV": dstat(obs["L-IMG"] - obs["L-CNV"], B["L-IMG"] - B["L-CNV"], one_sided=True), "LATE_minus_CNV": dstat(obs["L-LATE"] - obs["L-CNV"], B["L-LATE"] - B["L-CNV"])}
def analysis(O, I, FO, popm, rows, calls=None, claim_h=True, **kw):
    """one R33 analysis on the evaluation rows: calls from the stored (or given) predictions on the population popm, patient bootstrap over rows"""
    boots = boots_for(rows); CALLS, band, info, thr, sstar = calls if calls is not None else all_calls(O, I, FO, popm, **kw)
    obs = draw_stats(rows, CALLS, band, O); D = frame([draw_stats(b, CALLS, band, O) for b in boots])
    R = {**nfo(rows), "draws": int(len(D)), "cutoffs_median": {k: r3(np.median(v)) for k, v in info.items() if isinstance(v, list)}, "folds_no_middle_band": info["no_mid"], "t_seq_fallback": info["fallback"],
         "training_specificity_target_median": r3(np.median(sstar)), "claim_T": trow(obs, D)}
    if claim_h: R["claim_H"] = hrow(O, rows, boots)
    return R
def load_refit(cnv_outer, img_outer, cnv_inner, img_inner, col="pred"):
    """rv2_refit_metrics.load_refit: refit predictions in the tb_common.load() structure; folds must equal the stored folds"""
    def rd(files): return [pd.read_csv(f, dtype={"Sample": str}) if col == "pred" else pd.read_csv(f, dtype={"Sample": str}).drop(columns=["pred"], errors="ignore").rename(columns={col: "pred"}) for f in files]
    def outer(files): D = pd.concat(rd(files)); return D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values, D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int)
    def inner(files): D = pd.concat(rd(files)); assert D.groupby(["rep", "fold"]).ngroups == 100; return {(r, k): pd.Series(d.pred.values, index=ix[d.Sample].values) for (r, k), d in D.groupby(["rep", "fold"])}
    O, I = {}, {}; O["L-CNV"], FO = outer(cnv_outer); O["L-IMG"], F2 = outer(img_outer); assert (FO == F2).all() and (FO == FO0).all(); O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2
    I["L-CNV"] = inner(cnv_inner); I["L-IMG"] = inner(img_inner); I["L-LATE"] = {k: (I["L-CNV"][k] + I["L-IMG"][k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    for k in I["L-CNV"]: assert I["L-IMG"][k].index.sort_values().equals(I["L-CNV"][k].index.sort_values()), k
    return O, I, FO
def need(files, k): assert len(files) == k, (len(files), k, files[:2]); return files
def earliest_index():   # index of the pre-event endoscopy within the patient (1 = earliest), as rv2_round2.py R24
    EIDX = np.full(n, np.nan)
    for p in np.unique(pat):
        idx = np.where((pat == p) & PRE)[0]
        if not len(idx): continue
        em = pd.Series(MBF[idx]).groupby(ENDO[idx]).max().sort_values(ascending=False); rk = {e: i + 1 for i, e in enumerate(em.index)}; EIDX[idx] = [rk[e] for e in ENDO[idx]]
    return EIDX
def sheet():
    S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str)
    fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; f = fd.drop_duplicates("combined_name").set_index("combined_name").reindex(ids); assert f.Batch.notna().all(); return f
def fisher_p(cat, lab, B=10000, seed=0):
    """Fisher's exact test of a category x status table: exact for 2 x 2, otherwise Monte Carlo (B permutations of the status labels with fixed margins;
    p = (1 + #{table probability <= observed}) / (B + 1), as R's simulate.p.value)"""
    ci = pd.factorize(cat)[0]; li = np.asarray(lab).astype(int); nr, nc = ci.max() + 1, 2
    tab = np.zeros((nr, nc), int); np.add.at(tab, (ci, li), 1)
    if nr == 2: return {"method": "exact", "p": r3(fisher_exact(tab)[1])}
    lp = lambda t: -gammaln(t + 1).sum(); o = lp(tab); rng = np.random.RandomState(seed); c = 0
    for _ in range(B):
        t = np.zeros((nr, nc), int); np.add.at(t, (ci, rng.permutation(li)), 1); c += lp(t) <= o + 1e-9
    return {"method": f"monte_carlo_{B}", "p": r3((1 + c) / (B + 1))}
def xtab(cat, lab): return {str(k): {"progressor": int(v.get(1, 0)), "non_progressor": int(v.get(0, 0))} for k, v in pd.crosstab(pd.Series(cat), pd.Series(np.asarray(lab).astype(int))).iterrows()}
res = {"task": TASK}
# ================================================================== R31 sequencing batch and processing variables
if TASK == "r31":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); f = sheet(); R = {}
    BATCH = f.Batch.str.strip().values; SLX = f.SLXID1.str.strip().values; pil = f["Pilot for unpurifed lib"].fillna("").str.strip(); PILOT = np.array(["pilot" in b.lower() for b in BATCH])
    READS = pd.to_numeric(f["Number of reads"], errors="coerce").values; CELL = pd.to_numeric(f["% Barrett's cellularity"], errors="coerce").values
    notes = f["notes on label review"].fillna("").str.strip(); FLAGS = {"excluded": (f.excluded.fillna("").str.strip() == "1").values, "remove": (f["remove"].fillna("").str.strip() == "1").values, "inconsistency": (f.inconsistency.fillna("").str.strip() != "").values, "notes_on_label_review": (notes != "").values}
    R["join"] = {"samples_in_sheet": int(f.Batch.notna().sum()), "n_all": int(n), "n_pre_event": int(len(rows)), "pilot_flag_column_nonempty": int((pil != "").sum()),
                 "pilot_definition": "the sheet's 'Pilot for unpurifed lib' column is empty for all analysed samples; the pilot flag is taken from Batch values containing 'Pilot' (NP Pilot Study, Progressor Pilot Study)",
                 "reads_available": {"all": int(np.isfinite(READS).sum()), "pre_event": int(np.isfinite(READS[rows]).sum())}, "cellularity_available": {"all": int(np.isfinite(CELL).sum()), "pre_event": int(np.isfinite(CELL[rows]).sum())}}
    # (a) cross-tabulations
    A = {}
    for nm, v in (("batch", BATCH), ("slx_run", SLX), ("pilot", PILOT.astype(int))):
        pv = pd.Series(v[rows]).groupby(pat[rows]).agg(lambda s: s.value_counts().sort_index().idxmax()); nb = pd.Series(v[rows]).groupby(pat[rows]).nunique(); ppat = pd.Series(y[rows]).groupby(pat[rows]).max().reindex(pv.index)
        A[nm] = {"by_pre_event_sample": {"table": xtab(v[rows], y[rows]), "fisher": fisher_p(v[rows], y[rows])}, "by_all_676_samples": {"table": xtab(v, y), "fisher": fisher_p(v, y)},
                 "by_patient_modal": {"table": xtab(pv.values, ppat.values), "fisher": fisher_p(pv.values, ppat.values), "patients_spanning_several": int((nb > 1).sum()), "n_patients": int(len(pv)),
                                      "max_values_per_patient": int(nb.max()), "spanning_by_status": {"progressor": int(((nb > 1) & (ppat == 1)).sum()), "non_progressor": int(((nb > 1) & (ppat == 0)).sum())}}}
    R["a_cross_tabulation"] = A; print("R31 (a) done", flush=True)
    # (b) raw-score AUROC, with fold-honest target encoding for batch and SLX run
    def target_enc(cat):
        S = np.full((n, 10), np.nan)
        for j in range(10):
            for k in range(1, 11):
                tr = rows[FO[rows, j] != k]; te = FO[:, j] == k; share = pd.Series(y[tr]).groupby(cat[tr]).mean(); S[te, j] = pd.Series(cat[te]).map(share).fillna(y[tr].mean()).values
        return S
    SC = {"number_of_reads": READS, "pct_barretts_cellularity": CELL, "batch_target_encoded": target_enc(BATCH), "slx_run_target_encoded": target_enc(SLX), "pilot_flag": PILOT.astype(float)}
    obs = {m: fsa(S, rows) for m, S in SC.items()}; B = frame([{m: fsa(S, b) for m, S in SC.items()} for b in boots])
    R["b_raw_score_auroc"] = {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in SC}
    R["b_raw_score_auroc"]["note"] = "fold-stratified AUROC on the pre-event samples (missing values excluded within fold); target encoding per repeat x fold = progressor share of the category among the pre-event samples of that fold's training folds, unseen categories = the training share"
    # (c) within-batch pairs
    C = {}
    for nm, v in (("batch", BATCH), ("slx_run", SLX)):
        code = pd.factorize(v)[0]; npairs = pairs_within(rows, code); o = {m: fsa(O[m], rows, cell=code) for m in MODELS}; Bc = frame([{m: fsa(O[m], b, cell=code) for m in MODELS} for b in boots])
        C[nm] = {"within_pairs_per_repeat": npairs, "min_pairs_per_repeat": int(min(npairs)), "reported": bool(nm == "batch" or min(npairs) >= 100), "all_within_fold_pairs_repeat1": int(pairs_within(rows, np.zeros(n, int))[0]),
                 "auroc": {m: {"value": r3(o[m]), "ci95": ci2(Bc[m])} for m in MODELS}, "IMG_minus_CNV": dstat(o["L-IMG"] - o["L-CNV"], Bc["L-IMG"] - Bc["L-CNV"], one_sided=True), "LATE_minus_CNV": dstat(o["L-LATE"] - o["L-CNV"], Bc["L-LATE"] - Bc["L-CNV"])}
    C["unstratified_reference"] = {m: r3(fsa(O[m], rows)) for m in MODELS}; R["c_within_batch_pairs"] = C; print("R31 (c) done", flush=True)
    # (d) batch probe on the UNI2-h slide-mean embedding
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    IM = pd.read_csv(M + "/img_mean.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids).values.astype(float); assert np.isfinite(IM).all()
    def probe(idx):
        X = IM[idx]; g = pat[idx]; out = {}; npat = pd.Series(g).groupby(BATCH[idx]).nunique()
        for bch in sorted(npat.index[npat >= 5]):
            t = (BATCH[idx] == bch).astype(int); p = np.full(len(idx), np.nan)
            for tr, te in GroupKFold(5).split(X, t, g):
                if t[tr].min() == t[tr].max(): p[te] = 0.5; continue
                mdl = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=1.0, max_iter=5000)).fit(X[tr], t[tr]); p[te] = mdl.predict_proba(X[te])[:, 1]
            out[bch] = {"n_samples": int(t.sum()), "n_patients": int(npat[bch]), "auroc_one_vs_rest": r3(auc_plain(t, p))}
        vals = [v["auroc_one_vs_rest"] for v in out.values()]; return {"batches": out, "n_batches_probed": len(out), "auroc_mean_over_batches": r3(np.mean(vals)) if vals else None, "auroc_max": r3(max(vals)) if vals else None, "n_samples": int(len(idx)), "n_patients": int(len(set(g)))}
    R["d_batch_probe_uni2h"] = {"pre_event_samples": probe(rows), "non_progressor_pre_event_samples": probe(rows[y[rows] == 0]), "embedding_dim": int(IM.shape[1]),
                                "note": "one-vs-rest logistic regression (L2, C = 1, features standardised within the training folds), patient-grouped 5-fold CV, AUROC of the out-of-fold probabilities; batches with >= 5 patients"}
    print("R31 (d) done", flush=True)
    # (e) triage within the largest batch; if that batch holds one class only, the estimable side (specificity, share) with a patient bootstrap that does not
    # require both classes, plus the full comparison in the largest batch holding both classes
    bcount = pd.Series(BATCH[rows]).value_counts(); E = {"batch_sizes_pre_event": {str(k): int(v) for k, v in bcount.items()}}
    def batch_block(bname):
        rb = rows[BATCH[rows] == bname]; nP = int(y[rb].sum())
        if 0 < nP < len(rb):
            bb = boots_for(rb); ob = draw_stats(rb, CALLS, band, O); Db = frame([draw_stats(b, CALLS, band, O) for b in bb]); return {"batch": str(bname), **nfo(rb), "draws": int(len(Db)), "status": "done", "claim_T": trow(ob, Db), "claim_H": hrow(O, rb, bb)}
        pp = pat[rb]; up = np.unique(pp); ro = {q: rb[pp == q] for q in up}; rng = np.random.RandomState(0); bb = [np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))]) for _ in range(NB)]; cl = 1 if nP else 0
        st = lambda b: {**{s: float(CALLS[s][b].mean(0).mean() if cl else 1 - CALLS[s][b].mean(0).mean()) for s in ("T", "B", "B*", "C", "C*")}, "share_mid": float((band[b] == 1).mean())}
        o = st(rb); D = frame([st(b) for b in bb]); key = "sensitivity" if cl else "specificity"
        return {"batch": str(bname), **nfo(rb), "draws": int(len(D)), "status": f"d_sens not estimable: the batch holds {'progressor' if cl else 'non-progressor'} pre-event samples only", f"{key}_single_model": {s: {"value": r3(o[s]), "ci95": ci2(D[s])} for s in ("T", "B", "B*", "C", "C*")},
                f"d_{'sens' if cl else 'spec'}_triage_minus": {s: dstat(o["T"] - o[s], D["T"] - D[s]) for s in ("B", "B*")}, "share_sequenced": {"value": r3(o["share_mid"]), "ci95": ci2(D["share_mid"])}}
    mixed = [b_ for b_ in bcount.index if 0 < y[rows[BATCH[rows] == b_]].sum() < (BATCH[rows] == b_).sum()]
    E["largest_batch"] = batch_block(bcount.index[0]); E["largest_batch_with_both_classes"] = batch_block(mixed[0]) if mixed else None; E["batches_with_both_classes"] = [str(b_) for b_ in mixed]; R["e_triage_largest_batch"] = E
    # (f) exclusion flags
    R["f_exclusion_flags"] = {fl: {"all_676": {"progressor": int((m & (y == 1)).sum()), "non_progressor": int((m & (y == 0)).sum())}, "pre_event_571": {"progressor": int((m[rows] & (y[rows] == 1)).sum()), "non_progressor": int((m[rows] & (y[rows] == 0)).sum())}} for fl, m in FLAGS.items()}
    R["f_exclusion_flags"]["notes_on_label_review_distinct_pre_event"] = sorted(set(notes.values[rows][notes.values[rows] != ""]))
    res["R31"] = R; dump(res)
# ================================================================== R32(a) triage within tissue strata
elif TASK == "r32a":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); R = {"strata": {}}
    STR = {"lt256": TILES < 256, "ge256": TILES >= 256}
    def sstats(b, mk):
        bb = b[mk[b]]; o = {m: fsa(O[m], b, mask=mk) for m in MODELS}
        if len(bb) and 0 < y[bb].sum() < len(bb): o.update(draw_stats(bb, CALLS, band, O))
        return o
    obs = {s: sstats(rows, mk) for s, mk in STR.items()}; BS = {s: frame([sstats(b, mk) for b in boots]) for s, mk in STR.items()}
    for s, mk in STR.items():
        o, D = obs[s], BS[s]; R["strata"][s] = {"tissue_tiles": "< 256" if s == "lt256" else ">= 256", **nfo(rows[mk[rows]]), "claim_T": trow(o, D),
            "claim_H": {"auroc": {m: {"value": r3(o[m]), "ci95": ci2(D[m])} for m in MODELS}, "IMG_minus_CNV": dstat(o["L-IMG"] - o["L-CNV"], D["L-IMG"] - D["L-CNV"], one_sided=True), "LATE_minus_CNV": dstat(o["L-LATE"] - o["L-CNV"], D["L-LATE"] - D["L-CNV"])}}
    def inter(key, eval_=False):
        k = key + ("_eval" if eval_ else ""); g = lambda o: (o["nv_sens_T"] - o[k]); gi = lambda s: g(obs[s]); gd = lambda s: BS[s]["nv_sens_T"] - BS[s][k]
        return dstat(gi("lt256") - gi("ge256"), gd("lt256") - gd("ge256"))
    R["interaction_lt256_minus_ge256"] = {"d_sens_vs_Bstar_training_matched_no_vote": inter("nv_sens_B*"), "d_sens_vs_Bstar_evaluation_matched_no_vote": inter("nv_sens_B*", True), "d_sens_vs_B_no_vote": inter("nv_sens_B"),
                                          "d_auroc_IMG_minus_CNV": dstat((obs["lt256"]["L-IMG"] - obs["lt256"]["L-CNV"]) - (obs["ge256"]["L-IMG"] - obs["ge256"]["L-CNV"]), (BS["lt256"]["L-IMG"] - BS["lt256"]["L-CNV"]) - (BS["ge256"]["L-IMG"] - BS["ge256"]["L-CNV"]))}
    tert = np.full(n, -1); tert[rows] = pd.qcut(TILES[rows], 3, labels=False); R["tissue_matched_pairs_claim_H"] = {**hrow(O, rows, boots, cell=tert), "tertile_edges_tissue_tiles": [float(v) for v in np.quantile(TILES[rows], [0, 1 / 3, 2 / 3, 1])]}
    R["sanity_round2_r18"] = {"L-IMG_lt256_expected_0.905": R["strata"]["lt256"]["claim_H"]["auroc"]["L-IMG"]["value"], "L-IMG_ge256_expected_0.799": R["strata"]["ge256"]["claim_H"]["auroc"]["L-IMG"]["value"], "tissue_matched_IMG_minus_CNV_expected_+0.036_lower_-0.039": R["tissue_matched_pairs_claim_H"]["IMG_minus_CNV"]}
    res["R32a"] = R; dump(res)
# ================================================================== R32(b) tissue-adjusted gate
elif TASK == "r32b":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); assert np.isfinite(YEAR).all() and np.isfinite(TILES).all() and (TILES > 0).all()
    Xc = np.column_stack([np.ones(n), YEAR, np.log(TILES)]); O2 = dict(O); Rz = np.zeros_like(O["L-IMG"]); coefs = []
    for j in range(10): coef, *_ = np.linalg.lstsq(Xc[rows], O["L-IMG"][rows, j], rcond=None); Rz[:, j] = O["L-IMG"][:, j] - Xc @ coef; coefs.append(coef)
    O2["L-IMG"] = Rz; I2 = dict(I); I2["L-IMG"] = {}; icoefs = []
    for key, s in I["L-IMG"].items():
        idx = s.index.values; fit = idx[popm[idx]]; coef, *_ = np.linalg.lstsq(Xc[fit], s.reindex(fit).values, rcond=None); I2["L-IMG"][key] = pd.Series(s.values - Xc[idx] @ coef, index=idx); icoefs.append(coef)
    R = analysis(O2, I2, FO, popm, rows); R["gate"] = "L-IMG residualised on biopsy year and log tissue tiles (outer: per repeat on the held-out scores of the pre-event samples, as R5; inner: per (repeat, fold) set with coefficients fitted on that set's pre-event training rows); middle-band call L-LATE unchanged (stored L-CNV + unresidualised L-IMG); all cut-offs re-chosen by the primary rule"
    R["coefficients"] = {"outer_median_year_log_tiles": [r3(np.median([c[1] for c in coefs])), r3(np.median([c[2] for c in coefs]))], "inner_median_year_log_tiles": [r3(np.median([c[1] for c in icoefs])), r3(np.median([c[2] for c in icoefs]))]}
    R["claim_H"]["note"] = "L-IMG = residualised gate scores (per repeat); L-LATE unchanged"; R["sanity_round1_r5_residualised_LIMG_auroc_0.831"] = R["claim_H"]["auroc"]["L-IMG"]["value"]
    res["R32b"] = R; dump(res)
# ================================================================== R33 base analyses (one task per row)
elif TASK.startswith("r33_base_"):
    nm = TASK[9:]; popm = popmask("pre"); rows = np.where(popm)[0]; R = {"analysis": nm}
    if nm in ("primary", "ndbe", "pkg"):
        src, pop = ("pkg", "pre") if nm == "pkg" else ("their", "pre_ndbe" if nm == "ndbe" else "pre"); popm = popmask(pop); rows = np.where(popm)[0]; O, I, FO = load(src); R.update(analysis(O, I, FO, popm, rows))
        ref = J(f"{OUT2}/r14_{src}_{pop}.json")
        if ref: R["sanity_round2_r14"] = {"no_vote_vs_Bstar_expected": ref["R14"]["b_no_vote_paired"]["B*"]["d_sens_triage_minus"], "no_vote_vs_B_expected": ref["R14"]["b_no_vote_paired"]["B"]["d_sens_triage_minus"], "got_Bstar": R["claim_T"]["no_vote_vs_Bstar_training_matched"]["d_sens_triage_minus"], "got_B": R["claim_T"]["no_vote_vs_B"]["d_sens_triage_minus"]}
    elif nm in ("f1", "f1pkg"):
        src = "pkg" if nm == "f1pkg" else "their"; O, I, FO = load_refit(need(sorted(glob.glob(f"{RR2}/f1_outer/cnv_{src}_rep_*.csv")), 10), need(sorted(glob.glob(f"{RR2}/f1_outer/img_rep_*.csv")), 10), need(sorted(glob.glob(f"{RR2}/f1_inner/cnv_{src}_rep_*_fold_*.csv")), 100), need(sorted(glob.glob(f"{RR2}/f1_inner/img_rep_*_fold_*.csv")), 100))
        R.update(analysis(O, I, FO, popm, rows)); ref = J(f"{OUT2}/f1.json")
        if ref: R["sanity_round2_f1"] = {"no_vote_vs_Bstar_expected": ref["F1"][src]["pre"]["strategies"]["no_vote"]["triage"]["vs_B*"]["d_sens"], "IMG_minus_CNV_expected": ref["F1"][src]["pre"]["discrimination"]["IMG_minus_CNV"]}
    elif nm == "f5":
        O, I, FO = load_refit(kvf(0), need(sorted(glob.glob(f"{SR}/cv/cv_*_hoptimus0.csv")), 10), hzi("cnv_their"), need(sorted(glob.glob(f"{RR2}/f5_inner/img_rep_*_fold_*.csv")), 100)); R.update(analysis(O, I, FO, popm, rows))
        ref = J(f"{OUT2}/f5_r28.json")
        if ref: R["sanity_round2_f5"] = {"no_vote_vs_Bstar_expected": ref["F5_R28"]["pre"]["strategies"]["no_vote"]["triage"]["vs_B*"]["d_sens"], "IMG_minus_CNV_expected": ref["F5_R28"]["pre"]["discrimination"]["IMG_minus_CNV"]}
    elif nm in ("neverdys", "excl1y", "earliest_endo", "earliest_sample"):
        O, I, FO = load("their"); calls = all_calls(O, I, FO, popm)
        if nm == "neverdys": D1 = (GR == 1) | (GR == 2); dysp = set(pat[rows][D1[rows]]); keep = np.array([p not in dysp for p in pat]); R["definition"] = "patients with no pre-event ID or LGD sample (R24)"
        elif nm == "excl1y": keep = ~((y == 1) & (TIME <= 1)); R["definition"] = "progressor samples within 1 year of the first HGD/IMC excluded from the evaluation (calls and cut-offs as stored; single-model statistics recomputed on the remaining samples)"
        elif nm == "earliest_endo": keep = earliest_index() == 1; R["definition"] = "all samples of each patient's earliest pre-event endoscopy (calls as stored)"
        else: keep = np.zeros(n, bool); keep[[rows[pat[rows] == p][np.argmax(MBF[rows[pat[rows] == p]])] for p in np.unique(pat[rows])]] = True; R["definition"] = "each patient's single earliest pre-event sample (largest months before final; calls as stored)"
        rows2 = rows[keep[rows]]; R.update(analysis(O, I, FO, popm, rows2, calls=calls)); R["n_excluded_samples"] = int(len(rows) - len(rows2))
        if nm == "neverdys" and J(f"{OUT2}/r24.json"): R["sanity_round2_r24"] = {"no_vote_vs_Bstar_expected": J(f"{OUT2}/r24.json")["R24"]["never_dysplastic_patients"]["triage_no_vote_vs_Bstar"]["d_sens_triage_minus"], "IMG_minus_CNV_expected": J(f"{OUT2}/r24.json")["R24"]["never_dysplastic_patients"]["IMG_minus_CNV"]}
        if nm == "excl1y": R["sanity_round1_r6"] = {"auroc_expected_LIMG_0.842_LCNV_0.785_lower_-0.010": R["claim_H"]["IMG_minus_CNV"]}
        if nm == "earliest_sample": R["sanity_round1_r2"] = {"auroc_expected_LIMG_0.841_LCNV_0.812_lower_-0.075": {"auroc": {m: R["claim_H"]["auroc"][m]["value"] for m in MODELS}, "IMG_minus_CNV": R["claim_H"]["IMG_minus_CNV"]}}
    elif nm == "devlam":
        def f4_outer(cfg):
            D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in need(sorted(glob.glob(f"{RR2}/f4/{cfg}_rep_*.csv")), 10)]); Fd = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int); assert (Fd == FO0).all()
            return D.pivot_table(index="Sample", columns="rep", values="pred_dev").reindex(ids)[REPS].values
        O = {"L-CNV": f4_outer("cnv_their"), "L-IMG": f4_outer("img")}; O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2; boots = boots_for(rows); R.update(nfo(rows)); R["claim_H"] = hrow(O, rows, boots)
        R["sanity_round2_f4_expected_+0.021_lower_-0.050"] = R["claim_H"]["IMG_minus_CNV"]; fc = sorted(glob.glob(f"{ROW4}/b5_inner/cnv_their_rep_*_fold_*.csv")); fi = sorted(glob.glob(f"{ROW4}/b5_inner/img_rep_*_fold_*.csv"))
        if len(fc) == 100 and len(fi) == 100:
            col = "pred_dev" if "pred_dev" in pd.read_csv(fc[0], nrows=1).columns else "pred"; _, I, _ = load_refit(sorted(glob.glob(f"{RR2}/f4/cnv_their_rep_*.csv")), sorted(glob.glob(f"{RR2}/f4/img_rep_*.csv")), fc, fi, col=col)
            A = analysis(O, I, FO0, popm, rows, claim_h=False); R.update({k: v for k, v in A.items() if k != "claim_H"}); R["status"] = "done"; R["inner_source"] = f"killcoyne_mm/review_round4/b5_inner ({col})"; ref = J(f"{OUT4}/b5.json")
            try: R["sanity_part_b_b5"] = {"no_vote_vs_Bstar_expected": ref["B5"]["deviance_lambda"]["pre"]["strategies"]["no_vote"]["triage"]["vs_B*"]["d_sens"]}
            except Exception as e: R["sanity_part_b_b5"] = f"not compared: {type(e).__name__} {e}"[:200]
        else: R["status"] = "pending (Part B)"; R["claim_T"] = None; R["note"] = f"inner deviance-lambda predictions not found: {len(fc)} cnv and {len(fi)} img files under killcoyne_mm/review_round4/b5_inner (100 each needed); Claim H uses the F4 outer deviance predictions and is complete"
    elif nm in ("r36", "r36past"):
        src = "pkgps" if nm == "r36" else "pkgpast"; fo = sorted(glob.glob(f"{ROW4}/r36_outer/cnv_{src}_rep_*.csv")); fi = sorted(glob.glob(f"{ROW4}/r36_inner/cnv_{src}_rep_*_fold_*.csv"))
        R["definition"] = "L-CNV refitted on package features segmented " + ("per sample (R36 persample, cnv_pkgps)" if src == "pkgps" else "jointly with the patient's samples from the same or earlier endoscopies (R36 past-only, cnv_pkgpast)") + "; L-IMG stored; L-LATE = their mean; all cut-offs and B* re-chosen"
        if len(fo) == 10 and len(fi) == 100:
            O, I, FO = load_refit(fo, kvf(2), fi, hzi("img")); R.update(analysis(O, I, FO, popm, rows)); R["status"] = "done"; R["source"] = {"outer": f"killcoyne_mm/review_round4/r36_outer/cnv_{src}_rep_*.csv", "inner": f"killcoyne_mm/review_round4/r36_inner/cnv_{src}_rep_*_fold_*.csv"}
            ref = J(f"{OUT4}/r36.json")
            try: blk = ref["R36"]["persample" if src == "pkgps" else "past"]["pre"]; R["sanity_part_b_r36"] = {"no_vote_vs_Bstar_expected": blk["strategies"]["no_vote"]["triage"]["vs_B*"]["d_sens"], "IMG_minus_CNV_expected": blk["discrimination"]["IMG_minus_CNV"]}
            except Exception as e: R["sanity_part_b_r36"] = f"not compared: {type(e).__name__} {e}"[:200]
        else: R["status"] = "pending (Part B)"; R["claim_T"] = None; R["claim_H"] = None; R["note"] = f"{len(fo)} outer and {len(fi)} inner files for cnv_{src} under killcoyne_mm/review_round4/r36_outer and r36_inner (10 and 100 needed)"
    else: raise SystemExit(f"unknown base analysis {nm}")
    R.setdefault("status", "done"); res["R33_base"] = R; dump(res)
# ================================================================== R33 cut-off grid cells (B* re-matched to each cell's own training specificity)
elif TASK.startswith("r33_grid_"):
    _, _, ql, qh = TASK.split("_"); ql, qh = int(ql) / 100, int(qh) / 100; assert ql in (0.90, 0.95, 0.98) and qh in (0.85, 0.90, 0.95)
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; R = {"t_low_training_sensitivity": ql, "t_high_training_specificity": qh, **analysis(O, I, FO, popm, rows, claim_h=False, ql=ql, qh=qh)}
    g = J(f"{T}/results/paper_final/triage_robustness/grid_their_pre.json")
    if g: c = [c for c in g["cells"] if c["t_low_sens"] == ql and c["t_high_spec"] == qh][0]; R["sanity_tb_grid_voted"] = {"expected_sens_spec_share": [c["sensitivity"], c["specificity"], c["share_sequenced"]], "got": [R["claim_T"]["triage_voted"]["sensitivity"], R["claim_T"]["triage_voted"]["specificity"], R["claim_T"]["share_sequenced"]["per_repeat_mean"]]}
    res["R33_grid"] = R; dump(res)
# ================================================================== R33 selection-inclusive shards and merge (R16 logic, their matrix, all pre-event)
elif TASK.startswith("r33_selci_"):
    last = TASK.split("_")[-1]; popm = popmask("pre"); rows = np.where(popm)[0]; D16 = f"{ROW4}/selci"; os.makedirs(D16, exist_ok=True); O, I, FO = load("their"); boots = boots_for(rows); NSH = 20; per = NB // NSH
    if last == "merge":
        CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); obs = draw_stats(rows, CALLS, band, O); keys = keys_of(obs)
        files = sorted(glob.glob(f"{D16}/shard_*.npy")); A = np.vstack([np.load(f) for f in files]); assert A.shape == (NB, len(keys)), (A.shape, len(files)); D = pd.DataFrame(A, columns=keys)
        R = {"draws": int(len(D)), "n_shards": len(files), **nfo(rows), "claim_T": trow(obs, D), "note": "every cut-off (t_low, c*, t_seq, s*, comparator thresholds) re-chosen in each draw on the resampled inner training predictions (patient multiplicity = replication), as R16"}
        ref = J(f"{OUT2}/r16_their_pre.json")
        if ref: R["sanity_round2_r16"] = {"no_vote_vs_Bstar_expected": ref["R16"]["selection_inclusive_training_matched"]["no_vote"]["B*"]["d_sens_triage_minus"], "voted_vs_B_expected": ref["R16"]["selection_inclusive_training_matched"]["voted"]["vs_B"]["d_sens_triage_minus"]}
        res["R33_selci"] = R; dump(res, "r33_selci")
    else:
        sh = int(last.replace("shard", "")); outf = f"{D16}/shard_{sh:02d}.npy"
        if os.path.exists(outf): raise SystemExit(0)
        out = []
        for b in boots[sh * per:(sh + 1) * per]:
            W = np.bincount(b, minlength=n); CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm, W=W); d = draw_stats(b, CALLS, band, O); out.append([d[k] for k in keys_of(d)])
        np.save(outf, np.array(out)); print("RV4 SELCI SHARD DONE", TASK, flush=True)
# ================================================================== R33 between-repeat variance added (R15 logic)
elif TASK == "r33_r15":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); R = {}
    def per_rep_stats(b):
        o = {}; _, rep = draw_stats(b, CALLS, band, O, per_rep=True)
        for j in range(10):
            o[f"dAUC_IMG_{j}"] = fsa(O["L-IMG"], b, reps=[j]) - fsa(O["L-CNV"], b, reps=[j]); o[f"dsens_B_{j}"] = rep["sens_T"][j] - rep["sens_B"][j]; o[f"dsens_Bstar_{j}"] = rep["sens_T"][j] - rep["sens_B*"][j]
            o[f"dspec_B_{j}"] = rep["spec_T"][j] - rep["spec_B"][j]; o[f"dspec_Bstar_{j}"] = rep["spec_T"][j] - rep["spec_B*"][j]
        return o
    ob = per_rep_stats(rows); DB = frame([per_rep_stats(b) for b in boots])
    for nm, lab in (("dAUC_IMG", "dAUROC_LIMG_minus_LCNV"), ("dsens_B", "no_vote_d_sens_triage_minus_B"), ("dsens_Bstar", "no_vote_d_sens_triage_minus_Bstar"), ("dspec_B", "no_vote_d_spec_triage_minus_B"), ("dspec_Bstar", "no_vote_d_spec_triage_minus_Bstar")):
        pts = np.array([ob[f"{nm}_{j}"] for j in range(10)]); avg = DB[[f"{nm}_{j}" for j in range(10)]].mean(1).values; vb = float(avg.var(ddof=1)); vr = float(pts.var(ddof=1)); d = float(pts.mean()); lo = d - 1.645 * np.sqrt(vb + vr)
        R[lab] = {"delta_repeat_mean": r3(d), "sd_over_repeats": r3(pts.std(ddof=1)), "V_boot": round(vb, 6), "V_rep": round(vr, 6), "standard_one_sided_lower": lo1(avg), "variance_inflated_one_sided_lower": r3(lo), "variance_inflated_ci95": [r3(d - 1.96 * np.sqrt(vb + vr)), r3(d + 1.96 * np.sqrt(vb + vr))], "non_inferior_at_-0.05": bool(lo > -0.05)}
    R["share_sequenced"] = r3((band[rows] == 1).mean()); R["sanity_round2_r15_expected_Bstar_-0.046_IMG_-0.046"] = [R["no_vote_d_sens_triage_minus_Bstar"]["variance_inflated_one_sided_lower"], R["dAUROC_LIMG_minus_LCNV"]["variance_inflated_one_sided_lower"]]
    res["R33_r15"] = R; dump(res)
# ================================================================== R33 refit bootstrap (F3 draws) with Monte Carlo error
elif TASK == "r33_f3":
    PREs = pd.Series(PRE, index=ids)
    def rd(f): D = pd.read_csv(f, dtype={"Sample": str, "Patient": str}); D["pre"] = PREs.reindex(D.Sample).values; return D
    def draw_stats_f3(d):   # rv2_refit_metrics.py F3, unchanged
        oc, oi = rd(f"{RR2}/f3/cnv_their_draw_{d:03d}_outer.csv"), rd(f"{RR2}/f3/img_draw_{d:03d}_outer.csv"); ic, ii = rd(f"{RR2}/f3/cnv_their_draw_{d:03d}_inner.csv"), rd(f"{RR2}/f3/img_draw_{d:03d}_inner.csv")
        E = oc.merge(oi[["Sample", "copy", "pred"]], on=["Sample", "copy"], suffixes=("_cnv", "_img")); assert len(E) == len(oc) == len(oi); E = E[E.pre].reset_index(drop=True); E["late"] = (E.pred_cnv + E.pred_img) / 2
        Tr = ic.merge(ii[["Sample", "copy", "fold", "pred"]], on=["Sample", "copy", "fold"], suffixes=("_cnv", "_img")); assert len(Tr) == len(ic) == len(ii); Tr = Tr[Tr.pre].reset_index(drop=True); Tr["late"] = (Tr.pred_cnv + Tr.pred_img) / 2
        yy = E.y.values == 1; f = E.fold.values; auc = {}
        for m in ("pred_cnv", "pred_img", "late"):
            x = E[m].values; num = den = 0.0
            for k in range(1, 11):
                mk = f == k; yk = yy[mk]; n1 = int(yk.sum()); n0 = int((~yk).sum())
                if n1 == 0 or n0 == 0: continue
                rk = rankdata(x[mk]); num += rk[yk].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
            auc[m] = num / den
        call = {s: np.zeros(len(E), bool) for s in ("triage", "B", "B*")}; band = np.zeros(len(E), int)
        for k in range(1, 11):
            t = Tr[Tr.fold == k]; s_tr = t.pred_img.values; yc = t.y.values == 1; li = t.late.values
            tl = thr_sens(s_tr[yc], 0.95); cs = c_star(s_tr[~yc], 0.90); mid = (s_tr >= tl) & (s_tr <= cs) if tl <= cs else np.zeros(len(t), bool)
            cmid = li[mid & yc]; ts = thr_sens(cmid, 0.8) if len(cmid) >= 5 else thr_sens(li[yc], 0.8)
            tcall = np.where(s_tr < tl, False, np.where(s_tr > cs, True, li >= ts)) if tl <= cs else s_tr > cs; s_star = 1 - tcall[~yc].mean()
            xc = np.sort(li[~yc]); q = int(np.ceil(s_star * len(xc))); thr_bs = xc[q - 1] if q > 0 else -np.inf; thr_b = thr_sens(li[yc], 0.8)
            te = f == k; s = E.pred_img.values[te]; la = E.late.values[te]
            bd = np.where(s < tl, 0, np.where(s > cs, 2, 1)) if tl <= cs else np.where(s > cs, 2, 0); band[te] = bd
            call["triage"][te] = np.where(bd == 1, la >= ts, bd == 2); call["B"][te] = la >= thr_b; call["B*"][te] = la > thr_bs
        sp = {s: (c[yy].mean(), 1 - c[~yy].mean()) for s, c in call.items()}
        return {"d_auc_img_minus_cnv": auc["pred_img"] - auc["pred_cnv"], "d_sens_vs_B": sp["triage"][0] - sp["B"][0], "d_spec_vs_B": sp["triage"][1] - sp["B"][1], "d_sens_vs_Bstar": sp["triage"][0] - sp["B*"][0], "d_spec_vs_Bstar": sp["triage"][1] - sp["B*"][1], "share_mid": float((band == 1).mean())}
    S = [draw_stats_f3(d) for d in range(200) if all(os.path.exists(f"{RR2}/f3/{m}_draw_{d:03d}_{p}.csv") for m in ("cnv_their", "img") for p in ("outer", "inner"))]; assert len(S) == 200, len(S)
    A = {k: np.array([s[k] for s in S], float) for k in S[0]}; rng = np.random.RandomState(0); RS = [rng.randint(0, 200, 200) for _ in range(2000)]
    def mc(a): p5 = np.array([np.percentile(a[r], 5) for r in RS]); return {"mean": r3(a.mean()), "median": r3(np.median(a)), "ci95_percentile": ci2(a), "one_sided_5th_percentile": lo1(a), "sd": r3(a.std(ddof=1)), "non_inferior_at_-0.05": bool(np.percentile(a, 5) > -0.05), "mc_se_of_5th_percentile": round(float(p5.std(ddof=1)), 4), "mc_ci95_of_5th_percentile": ci2(p5), "share_of_draws_below_-0.05": r3((a < -0.05).mean())}
    R = {"draws_scored": 200, "mc_resamples": 2000, "claim_H_d_auroc_IMG_minus_CNV": mc(A["d_auc_img_minus_cnv"]), "claim_T_d_sens_triage_minus_Bstar": mc(A["d_sens_vs_Bstar"]), "d_spec_triage_minus_Bstar": mc(A["d_spec_vs_Bstar"]),
         "prespecified_d_sens_triage_minus_B": mc(A["d_sens_vs_B"]), "d_spec_triage_minus_B": mc(A["d_spec_vs_B"]), "share_sequenced": mc(A["share_mid"]), "note": "no vote; one stratified 10-fold CV per draw with lambda and all cut-offs re-chosen (rv2_refit_metrics.py F3); Monte Carlo SE = SD of the 5th percentile over 2,000 resamples of the 200 draws"}
    R["sanity_round2_f3_expected_Bstar_5th_-0.069_IMG_5th_-0.083"] = [R["claim_T_d_sens_triage_minus_Bstar"]["one_sided_5th_percentile"], R["claim_H_d_auroc_IMG_minus_CNV"]["one_sided_5th_percentile"]]
    res["R33_f3"] = R; dump(res)
# ================================================================== R33 merge: the two robustness tables and the pre-specified test
elif TASK == "r33_merge":
    def tcell(ct, key="no_vote_vs_Bstar_training_matched"):
        if ct is None: return None
        p = ct[key]; d = p["d_sens_triage_minus"]; sh = ct["share_sequenced"]
        return {"d_sens": d["delta"], "ci95": d["ci95"], "one_sided_lower": d.get("one_sided_95_lower"), "within_-0.05": d.get("non_inferior_at_-0.05"), "d_spec": p.get("d_spec_triage_minus", {}).get("delta"), "share_sequenced": sh["per_repeat_mean"], "cap_0.70_met": sh["cap_0.70_met"],
                "triage_sens_spec": [ct["triage_no_vote" if key.startswith("no_vote") else "triage_voted"]["sensitivity"], ct["triage_no_vote" if key.startswith("no_vote") else "triage_voted"]["specificity"]], "comparator_sens_spec": [p["comparator_sensitivity"], p.get("comparator_specificity")]}
    def hcell(ch):
        if ch is None: return None
        d = ch["IMG_minus_CNV"]; return {"L-IMG": ch["auroc"]["L-IMG"]["value"], "L-CNV": ch["auroc"]["L-CNV"]["value"], "d_auroc": d["delta"], "ci95": d["ci95"], "one_sided_lower": d["one_sided_95_lower"], "within_-0.05": d["non_inferior_at_-0.05"]}
    def base(nm): j = J(f"{OUT4}/r33_base_{nm}.json"); return j["R33_base"] if j else None
    TT, TH, P1 = [], [], []
    def add(label, item, T=None, H=None, P=None, status="done", na_T=None, na_H=None, na_P=None):
        TT.append({"item": item, "analysis": label, "status": status if na_T is None else f"not applicable: {na_T}", **(T or {})})
        TH.append({"item": item, "analysis": label, "status": status if na_H is None else f"not applicable: {na_H}", **(H or {})})
        P1.append({"item": item, "analysis": label, "status": status if na_P is None else f"not applicable: {na_P}", **(P or {})})
    def std(label, item, nm, na_H=None):
        b = base(nm)
        if b is None: add(label, item, status="missing (task not run)"); return
        st = b.get("status", "done"); ct = b.get("claim_T"); ch = b.get("claim_H")
        add(label, item, tcell(ct), hcell(ch) if na_H is None else None, {"voted": tcell(ct, "voted_vs_B"), "no_vote": tcell(ct, "no_vote_vs_B")} if ct else None, status=st, na_H=na_H)
        if st != "done" and ch is not None: TH[-1]["status"] = "done"
    std("primary: all pre-event samples", 1, "primary"); std("NDBE pre-event samples", 2, "ndbe"); std("package features", 3, "pkg"); std("pre-event-only training (F1)", 4, "f1"); std("patients never dysplastic before the event", 5, "neverdys")
    std("H-optimus-0 gate (F5)", 6, "f5"); std("excluding progressor samples within 1 year", 7, "excl1y")
    for ql in (90, 95, 98):
        for qh in (85, 90, 95):
            g = J(f"{OUT4}/r33_grid_{ql}_{qh}.json"); lab = f"cut-off grid cell: t_low {ql}% training sensitivity, t_high {qh}% training specificity" + (" (primary cell)" if (ql, qh) == (95, 90) else "")
            if g: ct = g["R33_grid"]["claim_T"]; add(lab, 8, tcell(ct), None, {"voted": tcell(ct, "voted_vs_B"), "no_vote": tcell(ct, "no_vote_vs_B")}, na_H="AUROC does not depend on the cut-offs")
            else: add(lab, 8, status="missing (task not run)", na_H="AUROC does not depend on the cut-offs")
    a = J(f"{OUT4}/r32a.json")
    for s, lab in (("lt256", "tissue stratum: < 256 tissue tiles (R32a)"), ("ge256", "tissue stratum: >= 256 tissue tiles (R32a)")):
        if a: ct = a["R32a"]["strata"][s]["claim_T"]; add(lab, 9, tcell(ct), hcell(a["R32a"]["strata"][s]["claim_H"]), {"voted": tcell(ct, "voted_vs_B"), "no_vote": tcell(ct, "no_vote_vs_B")})
        else: add(lab, 9, status="missing (task not run)")
    if a: add("tissue-matched pairs: same tissue tertile (Table 5)", "T5", None, hcell(a["R32a"]["tissue_matched_pairs_claim_H"]), None, na_T="pair-based AUROC only", na_P="pair-based AUROC only")
    b = J(f"{OUT4}/r32b.json")
    if b: ct = b["R32b"]["claim_T"]; add("tissue-adjusted gate (R32b)", 10, tcell(ct), hcell(b["R32b"]["claim_H"]), {"voted": tcell(ct, "voted_vs_B"), "no_vote": tcell(ct, "no_vote_vs_B")})
    else: add("tissue-adjusted gate (R32b)", 10, status="missing (task not run)")
    std("lambda by deviance (B5)", 11, "devlam")
    w = J(f"{OUT4}/r34.json")
    if w: W4 = w["R34"]; add("patient-weighted (R34)", 12, W4["claim_T_row"], hcell(W4["claim_H"]), W4["prespecified_row"])
    else: add("patient-weighted (R34)", 12, status="missing (task not run)")
    std("earliest pre-event endoscopy per patient (all its samples)", 13, "earliest_endo")
    sc = J(f"{OUT4}/r33_selci.json")
    if sc: ct = sc["R33_selci"]["claim_T"]; add("selection-inclusive (every cut-off re-chosen in each draw)", 14, tcell(ct), None, {"voted": tcell(ct, "voted_vs_B"), "no_vote": tcell(ct, "no_vote_vs_B")}, na_H="AUROC selects no cut-off")
    else: add("selection-inclusive (every cut-off re-chosen in each draw)", 14, status="missing (task not run)", na_H="AUROC selects no cut-off")
    pr = base("primary")
    if pr: add("evaluation-matched B*", 15, tcell(pr["claim_T"], "no_vote_vs_Bstar_evaluation_matched"), None, None, na_H="no threshold in AUROC", na_P="(B) is defined at its own threshold"); add("10-model vote", 16, tcell(pr["claim_T"], "voted_vs_Bstar_training_matched"), None, {"voted": tcell(pr["claim_T"], "voted_vs_B")}, na_H="AUROC is computed per repeat and averaged; the vote is a call rule")
    else: add("evaluation-matched B*", 15, status="missing (task not run)"); add("10-model vote", 16, status="missing (task not run)")
    r15 = J(f"{OUT4}/r33_r15.json")
    if r15:
        Q = r15["R33_r15"]; f_ = lambda q: {"d_sens": q["delta_repeat_mean"], "ci95": q["variance_inflated_ci95"], "one_sided_lower": q["variance_inflated_one_sided_lower"], "within_-0.05": q["non_inferior_at_-0.05"], "share_sequenced": Q["share_sequenced"], "cap_0.70_met": bool(Q["share_sequenced"] < 0.70)}
        TB = f_(Q["no_vote_d_sens_triage_minus_Bstar"]); TB["d_spec"] = Q["no_vote_d_spec_triage_minus_Bstar"]["delta_repeat_mean"]; PB = f_(Q["no_vote_d_sens_triage_minus_B"]); PB["d_spec"] = Q["no_vote_d_spec_triage_minus_B"]["delta_repeat_mean"]; qa = Q["dAUROC_LIMG_minus_LCNV"]
        add("between-repeat variance added", 17, TB, {"d_auroc": qa["delta_repeat_mean"], "ci95": qa["variance_inflated_ci95"], "one_sided_lower": qa["variance_inflated_one_sided_lower"], "within_-0.05": qa["non_inferior_at_-0.05"]}, {"no_vote": PB, "voted": None, "voted_note": "the vote pools the repeats; no between-repeat variance"})
    else: add("between-repeat variance added", 17, status="missing (task not run)")
    f3 = J(f"{OUT4}/r33_f3.json")
    if f3:
        Q = f3["R33_f3"]; g_ = lambda q, dq: {"d_sens": q["mean"], "ci95": q["ci95_percentile"], "one_sided_lower": q["one_sided_5th_percentile"], "within_-0.05": q["non_inferior_at_-0.05"], "d_spec": dq["mean"], "share_sequenced": Q["share_sequenced"]["mean"], "cap_0.70_met": bool(Q["share_sequenced"]["mean"] < 0.70), "mc_se_of_5th_percentile": q["mc_se_of_5th_percentile"], "share_of_draws_below_-0.05": q["share_of_draws_below_-0.05"]}
        qa = Q["claim_H_d_auroc_IMG_minus_CNV"]; add("refit bootstrap (F3, 200 draws)", 18, g_(Q["claim_T_d_sens_triage_minus_Bstar"], Q["d_spec_triage_minus_Bstar"]), {"d_auroc": qa["mean"], "ci95": qa["ci95_percentile"], "one_sided_lower": qa["one_sided_5th_percentile"], "within_-0.05": qa["non_inferior_at_-0.05"], "mc_se_of_5th_percentile": qa["mc_se_of_5th_percentile"], "share_of_draws_below_-0.05": qa["share_of_draws_below_-0.05"]},
                                                      {"no_vote": g_(Q["prespecified_d_sens_triage_minus_B"], Q["d_spec_triage_minus_B"]), "voted": None, "voted_note": "one model per draw; no vote"})
    else: add("refit bootstrap (F3, 200 draws)", 18, status="missing (task not run)")
    std("per-sample re-segmented copy number (R36)", 19, "r36"); std("past-only re-segmented copy number (R36, extra row)", "19b", "r36past")
    std("package features, pre-event-only training (Table 5)", "T5", "f1pkg"); std("earliest pre-event sample per patient (Table 5)", "T5", "earliest_sample")
    done_T = [r for r in TT if r["status"] == "done"]; done_H = [r for r in TH if r["status"] == "done"]
    R = {"claim_T_triage_vs_Bstar_training_matched_single_model": TT, "claim_H_LIMG_vs_LCNV_dAUROC": TH, "prespecified_test_triage_vs_B_voted_and_single_model": P1,
         "summary": {"claim_T": {"rows_done": len(done_T), "rows_within_-0.05": int(sum(bool(r.get("within_-0.05")) for r in done_T)), "rows_cap_met": int(sum(bool(r.get("cap_0.70_met")) for r in done_T)), "pending": [r["analysis"] for r in TT if r["status"].startswith("pending")], "missing": [r["analysis"] for r in TT if r["status"].startswith("missing")]},
                     "claim_H": {"rows_done": len(done_H), "rows_within_-0.05": int(sum(bool(r.get("within_-0.05")) for r in done_H)), "pending": [r["analysis"] for r in TH if r["status"].startswith("pending")], "missing": [r["analysis"] for r in TH if r["status"].startswith("missing")]}},
         "monte_carlo_error_refit_bootstrap": {k: f3["R33_f3"][k] for k in ("claim_H_d_auroc_IMG_minus_CNV", "claim_T_d_sens_triage_minus_Bstar", "prespecified_d_sens_triage_minus_B")} if f3 else "missing",
         "note": "no summary rule (pre-specification 45b77bf): both tables list which analyses pass and which fail against the -0.05 margin; the 70% cap applies to the share sequenced; 'within_-0.05' = one-sided 95% lower bound > -0.05"}
    res["R33"] = R; dump(res)
# ================================================================== R34 patient-weighted triage
elif TASK == "r34":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm)
    W = np.zeros(n); cnt = pd.Series(rows).groupby(pat[rows]).size()
    for p, c in cnt.items(): W[rows[pat[rows] == p]] = 1.0 / c
    obs = draw_stats(rows, CALLS, band, O, w=W); D = frame([draw_stats(b, CALLS, band, O, w=W) for b in boots]); R = {**nfo(rows), "draws": int(len(D)), "weight": "1 / (number of pre-event samples of the patient); weights fixed per sample, patients resampled with all their samples"}
    R["weighted_single_model"] = {s: {"sensitivity": r3(obs[f"nv_sens_{s}"]), "sens_ci95": ci2(D[f"nv_sens_{s}"]), "specificity": r3(obs[f"nv_spec_{s}"]), "spec_ci95": ci2(D[f"nv_spec_{s}"])} for s in ("T", "B", "C", "B*", "C*")}
    R["weighted_share_sequenced"] = {"value": r3(obs["share_mid"]), "ci95": ci2(D["share_mid"]), "cap_0.70_met": bool(obs["share_mid"] < 0.70)}
    R["weighted_delta_no_vote"] = {s: paired(obs, D, s, "nv", one_sided=True) for s in ("B", "B*", "C", "C*")}; R["weighted_delta_voted"] = {s: paired(obs, D, s, "v", one_sided=True) for s in ("B", "B*")}
    R["claim_T"] = trow(obs, D); u = lambda s, m, k: {"d_sens": R["claim_T"][k]["d_sens_triage_minus"]["delta"], "ci95": R["claim_T"][k]["d_sens_triage_minus"]["ci95"], "one_sided_lower": R["claim_T"][k]["d_sens_triage_minus"]["one_sided_95_lower"], "within_-0.05": R["claim_T"][k]["d_sens_triage_minus"]["non_inferior_at_-0.05"],
                                                        "d_spec": R["claim_T"][k]["d_spec_triage_minus"]["delta"], "share_sequenced": R["weighted_share_sequenced"]["value"], "cap_0.70_met": R["weighted_share_sequenced"]["cap_0.70_met"], "triage_sens_spec": [r3(obs[f"{m}_sens_T"]), r3(obs[f"{m}_spec_T"])], "comparator_sens_spec": [r3(obs[f"{m}_sens_{s}"]), r3(obs[f"{m}_spec_{s}"])]}
    R["claim_T_row"] = u("B*", "nv", "no_vote_vs_Bstar_training_matched"); R["prespecified_row"] = {"no_vote": u("B", "nv", "no_vote_vs_B"), "voted": u("B", "v", "voted_vs_B")}
    R["claim_H"] = hrow(O, rows, boots, w=W); R["claim_H"]["note"] = "pair weights w_i w_j (rv_round1.py R2 patient_weighted_samples)"; R["sanity_round1_r2_expected_LIMG_0.852_LCNV_0.822_lower_-0.034"] = {"auroc": {m: R["claim_H"]["auroc"][m]["value"] for m in MODELS}, "IMG_minus_CNV": R["claim_H"]["IMG_minus_CNV"]}
    res["R34"] = R; dump(res)
# ================================================================== R35 the deployable object and band stability
elif TASK == "r35":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); R = {}
    # (a) copy-number-only middle band (triage_CNV), B* re-matched to its own training specificity
    Cc, bandc, infoc, thrc, sstarc = all_calls(O, I, FO, popm, mid="L-CNV"); assert (bandc == band).all()
    oc = draw_stats(rows, Cc, bandc, O); Dc = frame([draw_stats(b, Cc, bandc, O) for b in boots])
    R["a_triage_CNV_middle_band"] = {"claim_T": trow(oc, Dc), "training_specificity_target_median": r3(np.median(sstarc)), "t_seq_median": r3(np.median(infoc["t_seq"])), "t_seq_fallback": infoc["fallback"],
                                     "reference_L-LATE_middle_band": {"no_vote_vs_Bstar": paired(draw_stats(rows, CALLS, band, O), frame([draw_stats(b, CALLS, band, O) for b in boots]), "B*", "nv", one_sided=True)["d_sens_triage_minus"]}}
    fx = J(f"{T}/results/paper_final/triage/their_pre.json")
    if fx: R["a_triage_CNV_middle_band"]["sanity_paper_triage_voted"] = {"expected": {k: fx["strategies"]["triage_CNV"][k]["value"] for k in ("sensitivity", "specificity")}, "got": {k: R["a_triage_CNV_middle_band"]["claim_T"]["triage_voted"][k] for k in ("sensitivity", "specificity")}}
    # (b) frozen cut-offs on held-out predictions (not out of fold)
    FC = J(f"{T}/results/paper_final/triage_robustness/frozen_cutoffs_their.json")["cutoffs"]; img, late = O["L-IMG"], O["L-LATE"]
    bandf = np.where(img < FC["t_low"], 0, np.where(img > FC["t_high"], 2, 1)); callf = np.where(bandf == 1, late >= FC["t_seq"], bandf == 2); Bf = late >= FC["t_B"]
    Cf = dict(CALLS); Cf["T"] = callf; Cf["B"] = Bf; of = draw_stats(rows, Cf, bandf, O); Df = frame([draw_stats(b, Cf, bandf, O) for b in boots])
    R["b_frozen_cutoffs_on_heldout"] = {"label": "frozen cut-offs on held-out predictions, not out of fold (medians over folds that include the evaluated fold)", "cutoffs": {k: r3(v) for k, v in FC.items()}, "t_B_is_0.314": bool(r3(FC["t_B"]) == 0.314),
                                        "triage_no_vote": {"sensitivity": r3(of["nv_sens_T"]), "sens_ci95": ci2(Df["nv_sens_T"]), "specificity": r3(of["nv_spec_T"]), "spec_ci95": ci2(Df["nv_spec_T"]), "share_sequenced": r3(of["share_mid"]), "share_ci95": ci2(Df["share_mid"]), "cap_0.70_met": bool(of["share_mid"] < 0.70)},
                                        "band_shares_mean_over_repeats": {nm: r3((bandf[rows] == g).mean()) for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}, "vs_B_frozen_0.314_no_vote": paired(of, Df, "B", "nv", one_sided=True), "vs_Bstar_training_matched_no_vote": paired(of, Df, "B*", "nv", one_sided=True),
                                        "voted_for_reference": {"vs_B_frozen": paired(of, Df, "B", "v", one_sided=True), "vs_Bstar": paired(of, Df, "B*", "v", one_sided=True)}, "all_repeats_identical_cutoffs": True}
    # (c) band stability
    Bq = band[rows]; modal = np.array([np.bincount(r_, minlength=3).argmax() for r_ in Bq]); nmod = np.array([(r_ == m).sum() for r_, m in zip(Bq, modal)]); yr = y[rows]; Tc = CALLS["T"][rows]; diff = Tc.min(1) != Tc.max(1)
    dist = lambda mk: {str(k): int(v) for k, v in pd.Series(nmod[mk]).value_counts().sort_index(ascending=False).items()}
    R["c_band_stability"] = {"repeats_in_modal_band_distribution": {"all": dist(np.ones(len(rows), bool)), **{f"{cl}_{bn}": dist((yr == v) & (modal == g)) for cl, v in (("progressor", 1), ("non_progressor", 0)) for g, bn in ((0, "low"), (1, "mid"), (2, "high"))}},
                             "mean_repeats_in_modal_band": {"all": r3(nmod.mean()), "progressor": r3(nmod[yr == 1].mean()), "non_progressor": r3(nmod[yr == 0].mean())}, "share_all_10_repeats_same_band": r3((nmod == 10).mean()),
                             "modal_band_counts": {cl: {bn: int(((yr == v) & (modal == g)).sum()) for g, bn in ((0, "low"), (1, "mid"), (2, "high"))} for cl, v in (("progressor", 1), ("non_progressor", 0))},
                             "share_single_model_call_differs_between_repeats": {"all": r3(diff.mean()), "progressor": r3(diff[yr == 1].mean()), "non_progressor": r3(diff[yr == 0].mean()), "by_modal_band": {bn: r3(diff[modal == g].mean()) for g, bn in ((0, "low"), (1, "mid"), (2, "high"))}}}
    res["R35"] = R; dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
