"""Review round 6, items R37-R41 (docs/paper_review_round6_analyses.md @ 5c10c27). Stored predictions, the analysed-sample table, Killcoyne's sample
sheet, the round-2 / round-4 refit predictions and the stored hospital-record check only (tb_common.py loaders and the paper_triage.md rules; the
rv4_round4.py helpers are copied here, with the vote-pivotal share of rv2_round2.py added to draw_stats). Nothing is refitted. Aggregates ->
results/paper_final/review_round6/<TASK>.json (no row-level data, no identifiers).
TASK = r37 | r38_<sub> | r38_merge | r39_<row> | r39_merge | r40 | r41
       <sub> excl | exclall | fu1y | fu2y | laterhgd | combined
       <row> primary | ndbe | pkg | f1 | f1pkg | neverdys | f5 | excl1y | earliest_endo | earliest_sample | grid_<ql>_<qh> | lt256 | ge256 | tgate | devlam | weighted | selci | r36 | r36past"""
import os, sys, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, fisher_exact, spearmanr, beta as beta_dist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tb_common import *
from tb_common import _outer, _inner
PRESPEC = "5c10c27"; OUT6 = T + "/results/paper_final/review_round6"; OUT4 = T + "/results/paper_final/review_round4"; OUT2 = T + "/results/paper_final/review_round2"
ROW4 = M + "/review_round4"; RR2 = M + "/review_round2"; SR = M + "/scanner_robustness"; KD = T + "/feasibility/paper_plan/killcoyne"; PP = T + "/feasibility/paper_plan"
os.makedirs(OUT6, exist_ok=True); TASK = os.environ["TASK"]
CS = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample").reindex(ids); YEAR = pd.to_numeric(CS.year, errors="coerce").values; P53 = pd.to_numeric(CS["P53 IHC"], errors="coerce").values
GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); STUDY = SM.study_number.astype(str).values
SDS = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); TILES = SDS.tissue_tiles.values.astype(float)
TIME = SM.time.values.astype(float); MBF = SM.mbf.values.astype(float)
FO0 = load("their")[2]; MM = {"A*": "L-CNV", "B*": "L-LATE", "C*": "L-IMG"}; COMPM = {"A": "L-CNV", "B": "L-LATE", "C": "L-IMG"}; MODELS = ("L-CNV", "L-IMG", "L-LATE")
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv")); hzi = lambda nm: sorted(glob.glob(f"{HZ}/inner/{nm}_rep_*_fold_*.csv"))
# ------------------------------------------------------------------ helpers (rv4_round4.py, unchanged unless stated)
def fsa(S, b, mask=None, w=None, cell=None, reps=None):
    """fold-stratified AUROC (mean over repeats of within-fold pooled pair concordance); S n x 10 or n; optional mask, pair weights, cell key, repeat subset"""
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
def ci2(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return pct(a) if len(a) else [None, None]
def lo1(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return r3(np.percentile(a, 5)) if len(a) else None
def dstat(obs, draws, one_sided=False):
    o = {"delta": r3(obs), "ci95": ci2(draws), "p_two_sided": p2(draws)}
    if one_sided: o["one_sided_95_lower"] = lo1(draws); o["non_inferior_at_-0.05"] = bool(lo1(draws) is not None and lo1(draws) > -0.05)
    return o
def cp_ci(k, nn):   # exact Clopper-Pearson 95% CI (rv2_round2.py)
    if nn == 0: return [None, None]
    lo = beta_dist.ppf(0.025, k, nn - k + 1) if k > 0 else 0.0; hi = beta_dist.ppf(0.975, k + 1, nn - k) if k < nn else 1.0; return [r3(lo), r3(hi)]
def dump(res, name=None):
    res["prespec"] = PRESPEC; json.dump(res, open(f"{OUT6}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV6 DONE", name or TASK, flush=True)
def J(path):
    try: return json.load(open(path))
    except FileNotFoundError: return None
def sens_at_spec(x, yb, spec_t):
    xc = np.sort(x[yb == 0]); q = int(np.ceil(spec_t * len(xc))); thr = xc[q - 1] if q > 0 else -np.inf; return float(np.mean(x[yb == 1] > thr))
def wmean(v, w): return float(v.mean()) if w is None else float((v * w).sum() / w.sum())
def frame(BS): return pd.DataFrame(BS)
def keys_of(d): return sorted(d.keys())
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
    band, call, info = triage(O, I, FO, popm, ql, qh, mid=mid, W=W); comp, thr = comparators(O, I, FO, popm, W=W); mc, sstar = matched_calls(O, I, FO, popm, W=W, ql=ql, qh=qh, mid=mid)
    CALLS = {"T": call, "A": comp["A"], "B": comp["B"], "C": comp["C"], "A*": mc["A*"], "B*": mc["B*"], "C*": mc["C*"]}; return CALLS, band, info, thr, sstar
def draw_stats(b, CALLS, band, O, per_rep=False, w=None):
    """rv4_round4.py draw_stats plus the vote-pivotal share of rv2_round2.py (>= 6-of-10 call changes between all middle-band calls positive and all negative)"""
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
    Bb = band[b]; mid = Bb == 1; Tb = CALLS["T"][b]; up = np.where(mid, True, Tb).sum(1) >= 6; dn = np.where(mid, False, Tb).sum(1) >= 6; piv = up != dn
    for nm, mk in (("", np.ones(len(b), bool)), ("_P", P), ("_N", N)):
        out[f"share_mid{nm}"] = cm(mid, mk).mean(); out[f"share_mid_any{nm}"] = wmean(mid[mk].any(1).astype(float), wb if wb is None else wb[mk]); out[f"share_pivotal{nm}"] = wmean(piv[mk].astype(float), wb if wb is None else wb[mk])
    for g in (0, 1, 2): out[f"rate_P_{g}"] = cm(Bb == g, P).mean(); out[f"rate_N_{g}"] = cm(Bb == g, N).mean()
    return (out, rep) if per_rep else out
def paired(obs, D, s, mode, one_sided=False, eval_=False):
    k = f"{mode}_sens_{s}{'_eval' if eval_ else ''}"; o = {"d_sens_triage_minus": dstat(obs[f"{mode}_sens_T"] - obs[k], D[f"{mode}_sens_T"] - D[k], one_sided)}
    if not eval_: o["d_spec_triage_minus"] = dstat(obs[f"{mode}_spec_T"] - obs[f"{mode}_spec_{s}"], D[f"{mode}_spec_T"] - D[f"{mode}_spec_{s}"])
    o["comparator_sensitivity"] = r3(obs[k]); o["comparator_sens_ci95"] = ci2(D[k])
    if not eval_: o["comparator_specificity"] = r3(obs[f"{mode}_spec_{s}"]); o["comparator_spec_ci95"] = ci2(D[f"{mode}_spec_{s}"])
    return o
def trow(obs, D):
    return {"triage_no_vote": {"sensitivity": r3(obs["nv_sens_T"]), "sens_ci95": ci2(D["nv_sens_T"]), "specificity": r3(obs["nv_spec_T"]), "spec_ci95": ci2(D["nv_spec_T"])},
            "triage_voted": {"sensitivity": r3(obs["v_sens_T"]), "sens_ci95": ci2(D["v_sens_T"]), "specificity": r3(obs["v_spec_T"]), "spec_ci95": ci2(D["v_spec_T"])},
            "share_sequenced": {"per_repeat_mean": r3(obs["share_mid"]), "ci95": ci2(D["share_mid"]), "in_at_least_one_repeat": r3(obs["share_mid_any"]), "vote_pivotal": r3(obs["share_pivotal"]), "cap_0.70_met": bool(obs["share_mid"] < 0.70)},
            "no_vote_vs_Bstar_training_matched": paired(obs, D, "B*", "nv", one_sided=True), "no_vote_vs_Bstar_evaluation_matched": paired(obs, D, "B*", "nv", one_sided=True, eval_=True),
            "no_vote_vs_B": paired(obs, D, "B", "nv", one_sided=True), "voted_vs_Bstar_training_matched": paired(obs, D, "B*", "v", one_sided=True), "voted_vs_Bstar_evaluation_matched": paired(obs, D, "B*", "v", one_sided=True, eval_=True), "voted_vs_B": paired(obs, D, "B", "v", one_sided=True),
            "no_vote_vs_Cstar_training_matched": paired(obs, D, "C*", "nv", one_sided=True), "no_vote_vs_C": paired(obs, D, "C", "nv")}
def hrow(O, rows, boots, mask=None, cell=None, w=None):
    obs = {m: fsa(O[m], rows, mask=mask, cell=cell, w=w) for m in MODELS}; B = frame([{m: fsa(O[m], b, mask=mask, cell=cell, w=w) for m in MODELS} for b in boots])
    return {"auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in MODELS}, "IMG_minus_CNV": dstat(obs["L-IMG"] - obs["L-CNV"], B["L-IMG"] - B["L-CNV"], one_sided=True), "LATE_minus_CNV": dstat(obs["L-LATE"] - obs["L-CNV"], B["L-LATE"] - B["L-CNV"])}
def analysis(O, I, FO, popm, rows, calls=None, claim_h=True, w=None, **kw):
    boots = boots_for(rows); CALLS, band, info, thr, sstar = calls if calls is not None else all_calls(O, I, FO, popm, **kw)
    obs = draw_stats(rows, CALLS, band, O, w=w); D = frame([draw_stats(b, CALLS, band, O, w=w) for b in boots])
    R = {**nfo(rows), "draws": int(len(D)), "cutoffs_median": {k: r3(np.median(v)) for k, v in info.items() if isinstance(v, list)}, "folds_no_middle_band": info["no_mid"], "t_seq_fallback": info["fallback"],
         "training_specificity_target_median": r3(np.median(sstar)), "claim_T": trow(obs, D)}
    if claim_h: R["claim_H"] = hrow(O, rows, boots, w=w)
    return R, obs, D, CALLS, band
def load_refit(cnv_outer, img_outer, cnv_inner, img_inner, col="pred"):
    def rd(files): return [pd.read_csv(f, dtype={"Sample": str}) if col == "pred" else pd.read_csv(f, dtype={"Sample": str}).drop(columns=["pred"], errors="ignore").rename(columns={col: "pred"}) for f in files]
    def outer(files): D = pd.concat(rd(files)); return D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values, D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int)
    def inner(files): D = pd.concat(rd(files)); assert D.groupby(["rep", "fold"]).ngroups == 100; return {(r, k): pd.Series(d.pred.values, index=ix[d.Sample].values) for (r, k), d in D.groupby(["rep", "fold"])}
    O, I = {}, {}; O["L-CNV"], FO = outer(cnv_outer); O["L-IMG"], F2 = outer(img_outer); assert (FO == F2).all() and (FO == FO0).all(); O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2
    I["L-CNV"] = inner(cnv_inner); I["L-IMG"] = inner(img_inner); I["L-LATE"] = {k: (I["L-CNV"][k] + I["L-IMG"][k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    for k in I["L-CNV"]: assert I["L-IMG"][k].index.sort_values().equals(I["L-CNV"][k].index.sort_values()), k
    return O, I, FO
def earliest_index():
    EIDX = np.full(n, np.nan)
    for p in np.unique(pat):
        idx = np.where((pat == p) & PRE)[0]
        if not len(idx): continue
        em = pd.Series(MBF[idx]).groupby(ENDO[idx]).max().sort_values(ascending=False); rk = {e: i + 1 for i, e in enumerate(em.index)}; EIDX[idx] = [rk[e] for e in ENDO[idx]]
    return EIDX
def sheet():
    S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str)
    fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; f = fd.drop_duplicates("combined_name").set_index("combined_name").reindex(ids); assert f.Batch.notna().all(); return f
def sheet_flags():
    f = sheet(); return {"excluded": (f.excluded.fillna("").str.strip() == "1").values, "remove": (f["remove"].fillna("").str.strip() == "1").values, "inconsistency": (f.inconsistency.fillna("").str.strip() != "").values}
def boots_tagged(rows, seed=0, nb=NB):   # rv2_round2.py R23: patient bootstrap with a draw tag so that resampled copies of a patient form distinct endoscopy / patient units
    pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(seed); out = []
    while len(out) < nb:
        dr = rng.choice(len(up), len(up)); b = np.concatenate([ro[up[i]] for i in dr]); t = np.concatenate([np.full(len(ro[up[i]]), q) for q, i in enumerate(dr)])
        if 0 < y[b].sum() < len(b): out.append((b, t))
    return out
def later_hgd_patients():
    """non-progressor patients with any later HGD+ report in the stored hospital-record check (pk_main.py K1 row-level source, joined by study number)"""
    LT = pd.read_csv(PP + "/later_disease_patient.csv", dtype={"patient_id": str}).set_index("patient_id")
    laterH = ((LT.db_max_grade_after >= 3) | (LT.release_excluded_max_grade >= 3) | (LT.slidematch_max_grade_after >= 3) | (LT.hgd_table_entries_after > 0))
    npp = sorted(set(pat[(y == 0)])); st = {p: STUDY[pat == p][0] for p in npp}; assert all(len(set(STUDY[pat == p])) == 1 for p in npp)
    matched = {p: s for p, s in st.items() if s in LT.index}; hit = [p for p, s in matched.items() if bool(laterH[s])]
    return hit, {"n_nonprogressor_patients": len(npp), "n_matched_in_hospital_record_check": len(matched), "n_not_identifiable": len(npp) - len(matched), "n_with_later_HGDplus": len(hit),
                 "n_with_later_LGDplus": int(sum(bool(((LT.db_max_grade_after >= 2) | (LT.release_excluded_max_grade >= 2) | (LT.slidematch_max_grade_after >= 2))[s]) for s in matched.values())),
                 "definition": "pk_main.py K1: db_max_grade_after >= 3 or release_excluded_max_grade >= 3 or slidematch_max_grade_after >= 3 or hgd_table_entries_after > 0; joined to the analysed patients by study number",
                 "k1_stored_reference": (J(T + "/results/paper_final/final_checks.json") or {}).get("K1", {}).get("discovery_nonprogressors")}
res = {"task": TASK}
# ================================================================== R37 p53 immunohistochemistry
if TASK == "r37":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); R = {}
    avail = np.isfinite(P53); p53 = np.where(avail, P53, 0) == 1; D1 = (GR == 1) | (GR == 2); GN = {0: "NDBE", 1: "ID", 2: "LGD"}
    # (a) availability
    ar = avail[rows]; yr = y[rows]; tab = [[int((ar & (yr == 1)).sum()), int((~ar & (yr == 1)).sum())], [int((ar & (yr == 0)).sum()), int((~ar & (yr == 0)).sum())]]
    ppat = pd.Series(yr, index=pat[rows]).groupby(level=0).max(); pav = pd.Series(ar, index=pat[rows]).groupby(level=0).max()
    R["a_availability"] = {"pre_event_samples": {"progressor": {"with_p53": tab[0][0], "without_p53": tab[0][1]}, "non_progressor": {"with_p53": tab[1][0], "without_p53": tab[1][1]}, "fisher_exact_p": r3(fisher_exact(tab)[1]), "share_available": r3(ar.mean())},
                           "patients_with_any_p53": {"progressor": int(pav[ppat == 1].sum()), "non_progressor": int(pav[ppat == 0].sum())}, "aberrant_among_available": {"progressor": int((p53[rows] & ar & (yr == 1)).sum()), "non_progressor": int((p53[rows] & ar & (yr == 0)).sum())},
                           "grade_distribution": {wo: {cl: {GN[g]: int(((ar == (wo == "with_p53")) & (yr == v) & (GR[rows] == g)).sum()) for g in (0, 1, 2)} for cl, v in (("progressor", 1), ("non_progressor", 0))} for wo in ("with_p53", "without_p53")},
                           "all_676_available": int(avail.sum()), "all_676_aberrant": int(p53.sum())}
    # rules on a given row set: deterministic vectors (n) and single-model matrices (n x 10); sample / endoscopy / patient levels
    Tv = vote(CALLS["T"]); RULES = lambda p: {"p53": p, "D": D1, "D+p53": D1 | p, "T_single": CALLS["T"], "T_voted": Tv, "T|p53_single": CALLS["T"] | p[:, None], "T|p53_voted": Tv | p, "B_single": CALLS["B"], "Bstar_single": CALLS["B*"]}
    def level_stats(b, t, rules):
        o = {}; yb = y[b]; keys = {"sample": np.arange(len(b)).astype(str), "endoscopy": pd.Series(ENDO[b]).astype(str).values + "#" + t.astype(str), "patient": pd.Series(pat[b]).astype(str).values + "#" + t.astype(str)}
        for lv, key in keys.items():
            lab = pd.Series(yb).groupby(key).max()
            for nm, c in rules.items():
                C = pd.DataFrame(c[b] if np.ndim(c) == 2 else c[b][:, None]).groupby(key).max(); se = C[lab.values == 1].mean(0).mean(); sp = 1 - C[lab.values == 0].mean(0).mean(); o[f"{lv}_sens_{nm}"] = float(se); o[f"{lv}_spec_{nm}"] = float(sp)
            o[f"{lv}_n_pos"] = int((lab == 1).sum()); o[f"{lv}_n_neg"] = int((lab == 0).sum())
        return o
    def block(rows_, rules, label):
        BT = boots_tagged(rows_); obs = level_stats(rows_, np.zeros(len(rows_), int), rules); D = frame([level_stats(b, t, rules) for b, t in BT]); out = {"label": label, **nfo(rows_), "draws": int(len(D)), "n_endoscopies": obs["endoscopy_n_pos"] + obs["endoscopy_n_neg"], "n_progressor_endoscopies": obs["endoscopy_n_pos"], "rules": {}}
        for nm in rules:
            o = {}
            for lv in ("sample", "endoscopy", "patient"):
                se, sp = obs[f"{lv}_sens_{nm}"], obs[f"{lv}_spec_{nm}"]; npos, nneg = obs[f"{lv}_n_pos"], obs[f"{lv}_n_neg"]
                o[lv] = {"sensitivity": r3(se), "sens_ci95_bootstrap": ci2(D[f"{lv}_sens_{nm}"]), "specificity": r3(sp), "spec_ci95_bootstrap": ci2(D[f"{lv}_spec_{nm}"])}
                if np.ndim(rules[nm]) == 1: o[lv]["sens_ci95_exact"] = cp_ci(int(round(se * npos)), npos); o[lv]["spec_ci95_exact"] = cp_ci(int(round(sp * nneg)), nneg); o[lv]["counts"] = {"true_positive": int(round(se * npos)), "positive": npos, "true_negative": int(round(sp * nneg)), "negative": nneg}
            out["rules"][nm] = o
        pr = {}
        for a, bb in (("T_single", "D"), ("T_single", "D+p53"), ("T_single", "p53"), ("T_voted", "D"), ("T_voted", "D+p53"), ("T_voted", "p53"), ("T|p53_single", "D+p53"), ("T|p53_single", "T_single"), ("T|p53_voted", "D+p53"), ("T|p53_voted", "T_voted"), ("T_single", "Bstar_single"), ("T|p53_single", "Bstar_single")):
            pr[f"{a}_minus_{bb}"] = {lv: {"d_sens": dstat(obs[f"{lv}_sens_{a}"] - obs[f"{lv}_sens_{bb}"], D[f"{lv}_sens_{a}"] - D[f"{lv}_sens_{bb}"], one_sided=True), "d_spec": dstat(obs[f"{lv}_spec_{a}"] - obs[f"{lv}_spec_{bb}"], D[f"{lv}_spec_{a}"] - D[f"{lv}_spec_{bb}"])} for lv in ("sample", "endoscopy", "patient")}
        out["paired"] = pr; out["share_sequenced_triage"] = r3((band[rows_] == 1).mean()); return out
    rows_p = rows[avail[rows]]; R["b_c_d_samples_with_p53"] = block(rows_p, RULES(p53), "pre-event samples with a p53 result; p53 = flag if aberrant; D = grade ID or LGD; D+p53 = D or p53 aberrant; T = triage (single model / voted); T|p53 = triage positive or p53 aberrant; B* training-matched")
    # (e) p53 among samples the triage clears
    modal = np.array([np.bincount(band[i][band[i] >= 0], minlength=3).argmax() for i in range(n)]); low_modal = modal == 0; low_all = (band == 0).all(1)
    def clears(mk, lab):
        m = mk & popm; return {"definition": lab, "n_cleared": int(m.sum()), "n_cleared_with_p53": int((m & avail).sum()), "progressor": {"cleared_with_p53": int((m & avail & (y == 1)).sum()), "p53_aberrant": int((m & p53 & (y == 1)).sum())}, "non_progressor": {"cleared_with_p53": int((m & avail & (y == 0)).sum()), "p53_aberrant": int((m & p53 & (y == 0)).sum())}}
    pm = {cl: float(np.mean([((band[:, j] == 0) & popm & p53 & (y == v)).sum() for j in range(10)])) for cl, v in (("progressor", 1), ("non_progressor", 0))}
    R["e_p53_among_triage_cleared"] = {"modal_low_band": clears(low_modal, "low band in the majority of the 10 repeats"), "low_band_all_repeats": clears(low_all, "low band in all 10 repeats"), "per_repeat_mean_p53_aberrant_in_low_band": {k: r3(v) for k, v in pm.items()},
                                       "note": "counts of pre-event samples with p53 available among those cleared; a p53-aberrant progressor sample in the low band is one the triage misses and p53 would flag"}
    # (f) missing p53 treated as wild-type, all pre-event samples
    R["f_missing_as_wild_type_all_pre_event"] = block(rows, RULES(p53), "all pre-event samples; missing p53 treated as wild-type (not flagged)")
    R["sanity_round2_r24_D1_expected_0.354_0.815"] = {k: R["f_missing_as_wild_type_all_pre_event"]["rules"]["D"]["sample"][k] for k in ("sensitivity", "specificity")}
    res["R37"] = R; dump(res)
# ================================================================== R38 label sensitivity analyses (stored calls re-evaluated on subsets, as R33 row 7)
elif TASK.startswith("r38_") and TASK != "r38_merge":
    nm = TASK[4:]; O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; calls = all_calls(O, I, FO, popm); FL = sheet_flags(); R = {"analysis": nm}
    hit, info = later_hgd_patients(); LH = np.array([p in set(hit) for p in pat]) & (y == 0)
    SUB = {"excl": (~FL["excluded"], "samples carrying the sheet's 'excluded' flag removed"), "exclall": (~(FL["excluded"] | FL["remove"] | FL["inconsistency"]), "samples carrying the sheet's 'excluded', 'remove' or 'inconsistency' flag removed"),
           "fu1y": (~((y == 0) & (MBF < 12)), "non-progressor samples taken less than 1 year (months before final < 12) before the patient's last sequenced endoscopy removed"), "fu2y": (~((y == 0) & (MBF < 24)), "non-progressor samples taken less than 2 years before the patient's last sequenced endoscopy removed"),
           "laterhgd": (~LH, "all samples of non-progressor patients with any later HGD+ report in the stored hospital-record check removed")}
    SUB["combined"] = (SUB["excl"][0] & SUB["fu1y"][0] & SUB["laterhgd"][0], "(a) 'excluded' flag, (b) 1-year follow-up and (c) later HGD+ together")
    keep, R["definition"] = SUB[nm]; rows2 = rows[keep[rows]]; assert 0 < y[rows2].sum() < len(rows2)
    A, obs, D, CALLS, band = analysis(O, I, FO, popm, rows2, calls=calls); R.update(A); R["n_excluded"] = {"samples": int(len(rows) - len(rows2)), "progressor_samples": int(y[rows].sum() - y[rows2].sum()), "patients": int(len(set(pat[rows])) - len(set(pat[rows2])))}
    R["later_hgd_check"] = info; R["flag_counts_pre_event"] = {k: {"progressor": int((v[rows] & (y[rows] == 1)).sum()), "non_progressor": int((v[rows] & (y[rows] == 0)).sum())} for k, v in FL.items()}
    R["note"] = "calls and cut-offs as stored (chosen on the full training folds); single-model statistics, the vote and the patient bootstrap recomputed on the remaining samples"
    res["R38"] = R; dump(res)
elif TASK == "r38_merge":
    base = J(f"{OUT4}/r33_base_primary.json")["R33_base"]; rows_ = []
    def cell(R, lab):
        ct = R["claim_T"]; ch = R["claim_H"]; b = ct["no_vote_vs_Bstar_training_matched"]; bb = ct["no_vote_vs_B"]; h = ch["IMG_minus_CNV"]
        return {"analysis": lab, **{k: R[k] for k in ("n_samples", "n_progressor_samples", "n_patients", "n_progressor_patients")}, "n_excluded": R.get("n_excluded"), "triage": {"sensitivity": ct["triage_no_vote"]["sensitivity"], "specificity": ct["triage_no_vote"]["specificity"], "share_sequenced": ct["share_sequenced"]["per_repeat_mean"]},
                "vs_Bstar": {"d_sens": b["d_sens_triage_minus"]["delta"], "ci95": b["d_sens_triage_minus"]["ci95"], "one_sided_lower": b["d_sens_triage_minus"]["one_sided_95_lower"], "within_-0.05": b["d_sens_triage_minus"]["non_inferior_at_-0.05"], "d_spec": b["d_spec_triage_minus"]["delta"]},
                "vs_B": {"d_sens": bb["d_sens_triage_minus"]["delta"], "ci95": bb["d_sens_triage_minus"]["ci95"], "one_sided_lower": bb["d_sens_triage_minus"]["one_sided_95_lower"], "d_spec": bb["d_spec_triage_minus"]["delta"]},
                "IMG_minus_CNV": {"delta": h["delta"], "ci95": h["ci95"], "one_sided_lower": h["one_sided_95_lower"], "within_-0.05": h["non_inferior_at_-0.05"], "auroc": {m: ch["auroc"][m]["value"] for m in MODELS}}}
    rows_.append(cell(base, "primary (R33 row 1, for reference)")); later = None
    for nm, lab in (("excl", "(a) 'excluded' flag removed"), ("exclall", "(a) 'excluded', 'remove' and 'inconsistency' flags removed"), ("fu1y", "(b) non-progressor samples < 1 year before the last sequenced endoscopy removed"), ("fu2y", "(b) non-progressor samples < 2 years before the last sequenced endoscopy removed"),
                    ("laterhgd", "(c) non-progressor patients with a later HGD+ report removed"), ("combined", "(d) (a) + (b, 1 year) + (c)")):
        j = J(f"{OUT6}/r38_{nm}.json")
        if j: rows_.append(cell(j["R38"], lab)); later = j["R38"]["later_hgd_check"]; rows_[-1]["definition"] = j["R38"]["definition"]
        else: rows_.append({"analysis": lab, "status": "missing (task not run)"})
    res["R38"] = {"rows": rows_, "later_hgd_check": later}; dump(res)
# ================================================================== R39 the 10-model vote as the frozen rule (R33 Claim T rows, voted triage vs voted B*)
elif TASK.startswith("r39_") and TASK != "r39_merge":
    nm = TASK[4:]; popm = popmask("pre"); rows = np.where(popm)[0]; R = {"analysis": nm}; kw = {}; w = None; calls = None; sub = rows
    if nm in ("primary", "ndbe", "pkg"): src, pop = ("pkg", "pre") if nm == "pkg" else ("their", "pre_ndbe" if nm == "ndbe" else "pre"); popm = popmask(pop); rows = sub = np.where(popm)[0]; O, I, FO = load(src)
    elif nm in ("f1", "f1pkg"): src = "pkg" if nm == "f1pkg" else "their"; O, I, FO = load_refit(need(sorted(glob.glob(f"{RR2}/f1_outer/cnv_{src}_rep_*.csv")), 10), need(sorted(glob.glob(f"{RR2}/f1_outer/img_rep_*.csv")), 10), need(sorted(glob.glob(f"{RR2}/f1_inner/cnv_{src}_rep_*_fold_*.csv")), 100), need(sorted(glob.glob(f"{RR2}/f1_inner/img_rep_*_fold_*.csv")), 100))
    elif nm == "f5": O, I, FO = load_refit(kvf(0), need(sorted(glob.glob(f"{SR}/cv/cv_*_hoptimus0.csv")), 10), hzi("cnv_their"), need(sorted(glob.glob(f"{RR2}/f5_inner/img_rep_*_fold_*.csv")), 100))
    elif nm in ("neverdys", "excl1y", "earliest_endo", "earliest_sample", "lt256", "ge256"):
        O, I, FO = load("their"); calls = all_calls(O, I, FO, popm)
        if nm == "neverdys": D1 = (GR == 1) | (GR == 2); dysp = set(pat[rows][D1[rows]]); keep = np.array([p not in dysp for p in pat])
        elif nm == "excl1y": keep = ~((y == 1) & (TIME <= 1))
        elif nm == "earliest_endo": keep = earliest_index() == 1
        elif nm == "earliest_sample": keep = np.zeros(n, bool); keep[[rows[pat[rows] == p][np.argmax(MBF[rows[pat[rows] == p]])] for p in np.unique(pat[rows])]] = True
        elif nm == "lt256": keep = TILES < 256
        else: keep = TILES >= 256
        sub = rows[keep[rows]]
    elif nm.startswith("grid_"): _, ql, qh = nm.split("_"); kw = {"ql": int(ql) / 100, "qh": int(qh) / 100}; O, I, FO = load("their")
    elif nm == "tgate":
        O, I, FO = load("their"); Xc = np.column_stack([np.ones(n), YEAR, np.log(TILES)]); O2 = dict(O); Rz = np.zeros_like(O["L-IMG"])
        for j in range(10): coef, *_ = np.linalg.lstsq(Xc[rows], O["L-IMG"][rows, j], rcond=None); Rz[:, j] = O["L-IMG"][:, j] - Xc @ coef
        O2["L-IMG"] = Rz; I2 = dict(I); I2["L-IMG"] = {}
        for key, s in I["L-IMG"].items(): idx = s.index.values; fit = idx[popm[idx]]; coef, *_ = np.linalg.lstsq(Xc[fit], s.reindex(fit).values, rcond=None); I2["L-IMG"][key] = pd.Series(s.values - Xc[idx] @ coef, index=idx)
        O, I = O2, I2
    elif nm == "devlam":
        fc = need(sorted(glob.glob(f"{ROW4}/b5_inner/cnv_their_rep_*_fold_*.csv")), 100); fi = need(sorted(glob.glob(f"{ROW4}/b5_inner/img_rep_*_fold_*.csv")), 100); col = "pred_dev" if "pred_dev" in pd.read_csv(fc[0], nrows=1).columns else "pred"
        O, I, FO = load_refit(need(sorted(glob.glob(f"{RR2}/f4/cnv_their_rep_*.csv")), 10), need(sorted(glob.glob(f"{RR2}/f4/img_rep_*.csv")), 10), fc, fi, col=col); R["inner_source"] = f"killcoyne_mm/review_round4/b5_inner ({col})"
    elif nm == "weighted":
        O, I, FO = load("their"); w = np.zeros(n); cnt = pd.Series(rows).groupby(pat[rows]).size()
        for p, c in cnt.items(): w[rows[pat[rows] == p]] = 1.0 / c
    elif nm in ("r36", "r36past"): src = "pkgps" if nm == "r36" else "pkgpast"; O, I, FO = load_refit(need(sorted(glob.glob(f"{ROW4}/r36_outer/cnv_{src}_rep_*.csv")), 10), kvf(2), need(sorted(glob.glob(f"{ROW4}/r36_inner/cnv_{src}_rep_*_fold_*.csv")), 100), hzi("img"))
    elif nm == "selci": O, I, FO = load("their")
    else: raise SystemExit(f"unknown R39 row {nm}")
    boots = boots_for(sub); CALLS, band, info, thr, sstar = calls if calls is not None else all_calls(O, I, FO, popm, **kw); obs = draw_stats(sub, CALLS, band, O, w=w)
    if nm == "selci":   # stored selection-inclusive draws (rv4_round4.py r33_selci shards: every cut-off re-chosen in each draw; same draws, same key order without the pivotal share)
        keys = [k for k in keys_of(obs) if "pivotal" not in k]; files = sorted(glob.glob(f"{ROW4}/selci/shard_*.npy")); A = np.vstack([np.load(f) for f in files]); assert A.shape == (NB, len(keys)), (A.shape, len(files)); D = pd.DataFrame(A, columns=keys)
        R["note"] = "voted statistics from the stored round-4 selection-inclusive draws (R16 logic); the vote-pivotal share is the observed value only"
    else: D = frame([draw_stats(b, CALLS, band, O, w=w) for b in boots])
    R.update({**nfo(sub), "draws": int(len(D)), "n_excluded_samples": int(len(rows) - len(sub)), "claim_T": trow(obs, D), "share_pivotal": {"value": r3(obs["share_pivotal"]), "ci95": ci2(D["share_pivotal"]) if "share_pivotal" in D else None, "progressor": r3(obs["share_pivotal_P"]), "non_progressor": r3(obs["share_pivotal_N"])},
              "training_specificity_target_median": r3(np.median(sstar)), "cutoffs_median": {k: r3(np.median(v)) for k, v in info.items() if isinstance(v, list)}})
    if nm == "primary":   # between-repeat stability of a voted call
        Tc = CALLS["T"][sub]; V = Tc.sum(1) >= 6; agree = (Tc == V[:, None]).sum(1); yr = y[sub]; Bq = band[sub]; modal = np.array([np.bincount(r_, minlength=3).argmax() for r_ in Bq]); nmod = np.array([(r_ == m).sum() for r_, m in zip(Bq, modal)]); BN = {0: "low", 1: "mid", 2: "high"}
        st = lambda mk: {"n": int(mk.sum()), "unanimous_10_of_10": r3((agree[mk] == 10).mean()), "near_unanimous_ge_8_of_10": r3((agree[mk] >= 8).mean()), "mean_repeats_agreeing_with_vote": r3(agree[mk].mean())}
        R["stability_of_voted_call"] = {"all": st(np.ones(len(sub), bool)), "progressor": st(yr == 1), "non_progressor": st(yr == 0), "by_modal_band": {bn: st(modal == g) for g, bn in BN.items()}, "by_modal_band_and_class": {cl: {bn: st((modal == g) & (yr == v)) for g, bn in BN.items()} for cl, v in (("progressor", 1), ("non_progressor", 0))},
                                        "agreement_distribution": {str(k): int(v) for k, v in pd.Series(agree).value_counts().sort_index(ascending=False).items()}, "note": "agreement = number of the 10 per-repeat triage calls equal to the >= 6-of-10 voted call; modal band = the sample's most frequent band over the repeats"}
        Bc = CALLS["B*"][sub]; VB = Bc.sum(1) >= 6; ag = (Bc == VB[:, None]).sum(1); R["stability_of_voted_call"]["Bstar_for_reference"] = {"unanimous_10_of_10": r3((ag == 10).mean()), "near_unanimous_ge_8_of_10": r3((ag >= 8).mean())}
        R["sanity"] = {"round4_r35_share_all_10_repeats_same_band_expected_0.35": r3((nmod == 10).mean()), "single_model_vs_Bstar_expected_-0.016_-0.032": [R["claim_T"]["no_vote_vs_Bstar_training_matched"]["d_sens_triage_minus"]["delta"], R["claim_T"]["no_vote_vs_Bstar_training_matched"]["d_sens_triage_minus"]["one_sided_95_lower"]],
                       "single_model_vs_B_expected_+0.033": R["claim_T"]["no_vote_vs_B"]["d_sens_triage_minus"]["delta"], "round4_r33_row16_voted_vs_Bstar_expected_-0.006_-0.044": [R["claim_T"]["voted_vs_Bstar_training_matched"]["d_sens_triage_minus"]["delta"], R["claim_T"]["voted_vs_Bstar_training_matched"]["d_sens_triage_minus"]["one_sided_95_lower"]]}
    res["R39_row"] = R; dump(res)
elif TASK == "r39_merge":
    r33 = J(f"{OUT4}/r33_merge.json")["R33"]["claim_T_triage_vs_Bstar_training_matched_single_model"]; r33by = {r["analysis"]: r for r in r33}
    def vcell(ct, key="voted_vs_Bstar_training_matched"):
        p = ct[key]; d = p["d_sens_triage_minus"]; sh = ct["share_sequenced"]
        return {"d_sens": d["delta"], "ci95": d["ci95"], "one_sided_lower": d.get("one_sided_95_lower"), "within_-0.05": d.get("non_inferior_at_-0.05"), "d_spec": p.get("d_spec_triage_minus", {}).get("delta"), "share_sequenced": sh["per_repeat_mean"], "share_pivotal": sh["vote_pivotal"], "cap_0.70_met": sh["cap_0.70_met"],
                "triage_sens_spec": [ct["triage_voted"]["sensitivity"], ct["triage_voted"]["specificity"]], "comparator_sens_spec": [p["comparator_sensitivity"], p.get("comparator_specificity")]}
    TT = []
    def add(label, item, nm=None, key="voted_vs_Bstar_training_matched", na=None, r33label=None):
        row = {"item": item, "analysis": label}; s = r33by.get(r33label or label)
        if s and s.get("status") == "done": row["single_model_R33"] = {"d_sens": s["d_sens"], "one_sided_lower": s["one_sided_lower"], "within_-0.05": s["within_-0.05"], "d_spec": s.get("d_spec")}
        if na: row["status"] = f"not applicable: {na}"; TT.append(row); return
        j = J(f"{OUT6}/r39_{nm}.json")
        if j is None: row["status"] = "missing (task not run)"; TT.append(row); return
        row.update({"status": "done", **vcell(j["R39_row"]["claim_T"], key), "voted_vs_B": vcell(j["R39_row"]["claim_T"], "voted_vs_B")}); TT.append(row)
    add("primary: all pre-event samples", 1, "primary"); add("NDBE pre-event samples", 2, "ndbe"); add("package features", 3, "pkg"); add("pre-event-only training (F1)", 4, "f1"); add("patients never dysplastic before the event", 5, "neverdys")
    add("H-optimus-0 gate (F5)", 6, "f5"); add("excluding progressor samples within 1 year", 7, "excl1y")
    for ql in (90, 95, 98):
        for qh in (85, 90, 95): add(f"cut-off grid cell: t_low {ql}% training sensitivity, t_high {qh}% training specificity" + (" (primary cell)" if (ql, qh) == (95, 90) else ""), 8, f"grid_{ql}_{qh}")
    add("tissue stratum: < 256 tissue tiles (R32a)", 9, "lt256"); add("tissue stratum: >= 256 tissue tiles (R32a)", 9, "ge256"); add("tissue-adjusted gate (R32b)", 10, "tgate"); add("lambda by deviance (B5)", 11, "devlam"); add("patient-weighted (R34)", 12, "weighted")
    add("earliest pre-event endoscopy per patient (all its samples)", 13, "earliest_endo"); add("selection-inclusive (every cut-off re-chosen in each draw)", 14, "selci"); add("evaluation-matched B*", 15, "primary", key="voted_vs_Bstar_evaluation_matched")
    add("10-model vote", 16, na="this table is the voted rule; identical to row 1"); add("between-repeat variance added", 17, na="the vote pools the repeats; no between-repeat variance"); add("refit bootstrap (F3, 200 draws)", 18, na="one model per draw; no vote")
    add("per-sample re-segmented copy number (R36)", 19, "r36"); add("past-only re-segmented copy number (R36, extra row)", "19b", "r36past"); add("package features, pre-event-only training (Table 5)", "T5", "f1pkg"); add("earliest pre-event sample per patient (Table 5)", "T5", "earliest_sample")
    done = [r for r in TT if r["status"] == "done"]; both = [r for r in done if "single_model_R33" in r]
    pr = J(f"{OUT6}/r39_primary.json"); R = {"claim_T_voted_triage_vs_voted_Bstar_training_matched": TT, "stability_of_voted_call": pr["R39_row"]["stability_of_voted_call"] if pr else None, "sanity": pr["R39_row"].get("sanity") if pr else None,
         "summary": {"rows_done": len(done), "rows_within_-0.05": int(sum(bool(r.get("within_-0.05")) for r in done)), "rows_cap_met": int(sum(bool(r.get("cap_0.70_met")) for r in done)), "rows_with_single_model_counterpart": len(both),
                     "single_model_within_-0.05_on_those_rows": int(sum(bool(r["single_model_R33"].get("within_-0.05")) for r in both)), "voted_within_-0.05_on_those_rows": int(sum(bool(r.get("within_-0.05")) for r in both)),
                     "mean_d_sens_voted": r3(np.mean([r["d_sens"] for r in both])), "mean_d_sens_single_model": r3(np.mean([r["single_model_R33"]["d_sens"] for r in both])), "mean_ci_width_voted": r3(np.mean([r["ci95"][1] - r["ci95"][0] for r in both])),
                     "missing": [r["analysis"] for r in TT if r["status"].startswith("missing")]},
         "note": "voted = >= 6-of-10 calls for both the triage and the training-matched B*; share_pivotal = share of samples whose voted call depends on the middle-band calls; single_model_R33 from results/paper_final/review_round4/r33_merge.json"}
    res["R39"] = R; dump(res)
# ================================================================== R40 paired comparison of copy-number versions
elif TASK == "r40":
    popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); V = {}
    V["shipped_matrix"] = load("their")[0]["L-CNV"]; V["joint_package"] = load("pkg")[0]["L-CNV"]
    for key, tag in (("past_only", "pkgpast"), ("per_sample", "pkgps")):
        Oc, F = _outer(need(sorted(glob.glob(f"{ROW4}/r36_outer/cnv_{tag}_rep_*.csv")), 10)); assert (F == FO0).all(); V[key] = Oc
    names = list(V); obs = {k: fsa(S, rows) for k, S in V.items()}; B = frame([{k: fsa(S, b) for k, S in V.items()} for b in boots])
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]; PW = {f"{a}_minus_{b}": dstat(obs[a] - obs[b], B[a] - B[b]) for a, b in pairs}
    order = sorted(PW, key=lambda k: PW[k]["p_two_sided"]); run = 0.0
    for i, k in enumerate(order): run = max(run, min(1.0, (len(order) - i) * PW[k]["p_two_sided"])); PW[k]["p_holm"] = round(run, 4)
    R = {**nfo(rows), "draws": int(len(B)), "auroc_L_CNV": {k: {"value": r3(obs[k]), "ci95": ci2(B[k])} for k in names}, "pairwise": PW, "note": "fold-stratified AUROC of L-CNV on the pre-event samples; same patient-bootstrap draws for every version (paired); p = two-sided bootstrap percentile p; Holm over the six pairs",
         "sanity_round4_r36_expected_per_sample_0.785_joint_0.845": [r3(obs["per_sample"]), r3(obs["joint_package"])]}
    # sampling intensity: per-sample correlation of joint vs per-sample final features by the number of samples segmented together (joint call = all of the patient's sheet samples)
    rb = lambda f: pd.read_csv(f, dtype={"sample_id": str}).set_index("sample_id").reindex(ids); XJ = rb(f"{M}/cnv_pkg_C.csv"); XP = rb(f"{ROW4}/r36_cnv_pkgps_C.csv"); assert list(XJ.columns) == list(XP.columns) and XJ.notna().all().all() and XP.notna().all().all()
    def rowcorr(A, Bm): A = A - A.mean(1, keepdims=True); Bm = Bm - Bm.mean(1, keepdims=True); return (A * Bm).sum(1) / np.sqrt((A ** 2).sum(1) * (Bm ** 2).sum(1))
    rr = rowcorr(XJ.values, XP.values); KR = pd.read_csv(KD + "/kr_samples.csv", dtype=str); NJ = KR.groupby("Patient").size().reindex(pat).values.astype(float); assert np.isfinite(NJ).all()
    PA = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(f"{ROW4}/reseg/past/pat_*.csv"))]).set_index("Sample").reindex(ids); NPAST = PA.n_segmented_together.values.astype(float)
    e1, e2 = np.quantile(NJ[rows], [1 / 3, 2 / 3]); tert = np.where(NJ <= e1, 0, np.where(NJ <= e2, 1, 2)); q = lambda v: {"n": int(len(v)), "median": r3(np.median(v)) if len(v) else None, "iqr": [r3(np.percentile(v, 25)), r3(np.percentile(v, 75))] if len(v) else None}
    SI = {"n_segmented_together_joint": {"definition": "number of the patient's samples in the stored joint segmentation call (all of the patient's sheet samples, rv4_reseg.R MODE=joint)", "pre_event_by_class": {cl: q(NJ[rows][y[rows] == v]) for cl, v in (("progressor", 1), ("non_progressor", 0))}, "tertile_edges_pre_event": [float(e1), float(e2)],
                                         "max_past_only_call_equals_joint_share": r3((pd.Series(NPAST[rows]).groupby(pat[rows]).max().values == pd.Series(NJ[rows]).groupby(pat[rows]).first().values).mean())},
          "per_sample_r_joint_vs_per_sample_final_features": {"pre_event_all": q(rr[rows]), "pre_event_by_class": {cl: q(rr[rows][y[rows] == v]) for cl, v in (("progressor", 1), ("non_progressor", 0))},
                                                              "by_tertile": {f"tertile_{t + 1}": {"n_segmented_range": [int(NJ[rows][tert[rows] == t].min()), int(NJ[rows][tert[rows] == t].max())], "all": q(rr[rows][tert[rows] == t]), **{cl: q(rr[rows][(tert[rows] == t) & (y[rows] == v)]) for cl, v in (("progressor", 1), ("non_progressor", 0))}} for t in range(3)},
                                                              "spearman_r_vs_n_segmented_pre_event": {"all": r3(spearmanr(rr[rows], NJ[rows]).correlation), **{cl: r3(spearmanr(rr[rows][y[rows] == v], NJ[rows][y[rows] == v]).correlation) for cl, v in (("progressor", 1), ("non_progressor", 0))}}}}
    def cls_diff(mk):   # class difference in the median r within a mask, patient bootstrap over the pre-event draws
        f = lambda b: (np.median(rr[b][(y[b] == 1) & mk[b]]) - np.median(rr[b][(y[b] == 0) & mk[b]])) if ((y[b] == 1) & mk[b]).any() and ((y[b] == 0) & mk[b]).any() else np.nan
        return dstat(f(rows), [f(b) for b in boots])
    SI["class_difference_median_r_progressor_minus_non_progressor"] = {"all_pre_event": cls_diff(np.ones(n, bool)), **{f"tertile_{t + 1}": cls_diff(tert == t) for t in range(3)}}
    SI["counts_by_tertile_and_class"] = {f"tertile_{t + 1}": {"progressor": int(((tert == t) & (y == 1))[rows].sum()), "non_progressor": int(((tert == t) & (y == 0))[rows].sum()), "patients": int(len(set(pat[rows][tert[rows] == t])))} for t in range(3)}
    R["sampling_intensity"] = SI; res["R40"] = R; dump(res)
# ================================================================== R41 one random biopsy per endoscopy
elif TASK == "r41":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); NS = 100
    eu, einv = np.unique(ENDO[rows], return_inverse=True); E = len(eu); es = [rows[einv == e] for e in range(E)]; elab = np.array([int(y[s].max()) for s in es]); epat = np.array([pat[s][0] for s in es])
    up = np.unique(epat); P_ = len(up); pe = [np.where(epat == p)[0] for p in up]; plab = np.array([int(elab[e].max()) for e in pe])
    anyE = lambda C: np.array([C[s].any(0) for s in es])                                         # endoscopy x 10: any positive sample
    TE = anyE(CALLS["T"]); BE = anyE(CALLS["B"]); CE = anyE(CALLS["C"]); BSE = anyE(CALLS["B*"])
    RB = np.zeros((NS, E, 10), bool); RV = np.zeros((NS, E, 10), bool)
    for sd in range(NS):
        rng = np.random.RandomState(sd)
        for e, s in enumerate(es):
            pick = s[rng.randint(len(s))]; other = s[s != pick]; RB[sd, e] = CALLS["B"][pick]; RV[sd, e] = CALLS["B"][pick] | (CALLS["C"][other].any(0) if len(other) else False)
    toP = lambda A: np.stack([A[..., e, :].any(-2) for e in pe], -2)                             # ... x E x 10 -> ... x P x 10
    TP, BP, RBP, RVP = toP(TE), toP(BE), toP(RB), toP(RV)
    def stats(eidx, pidx):   # per seed (first axis) or scalar: endoscopy- and patient-level single-model sensitivity and specificity
        o = {}
        for nm, A, Ap in (("random_biopsy", RB, RBP), ("random_biopsy_rest_HE", RV, RVP), ("triage", TE, TP), ("B_all_sequenced", BE, BP)):
            o[f"endo_sens_{nm}"] = A[..., eidx[elab[eidx] == 1], :].mean((-2, -1)); o[f"endo_spec_{nm}"] = 1 - A[..., eidx[elab[eidx] == 0], :].mean((-2, -1))
            o[f"pat_sens_{nm}"] = Ap[..., pidx[plab[pidx] == 1], :].mean((-2, -1)); o[f"pat_spec_{nm}"] = 1 - Ap[..., pidx[plab[pidx] == 0], :].mean((-2, -1))
        return o
    obs = stats(np.arange(E), np.arange(P_)); rng = np.random.RandomState(0); DR = []
    while len(DR) < NB:
        pidx = rng.choice(P_, P_); eidx = np.concatenate([pe[i] for i in pidx])
        if 0 < plab[pidx].sum() < P_: DR.append(stats(eidx, pidx))
    R = {**nfo(rows), "n_endoscopies": int(E), "n_progressor_endoscopies": int(elab.sum()), "n_patients_units": int(P_), "n_seeds": NS, "draws": int(len(DR)), "share_sequenced": {"random_biopsy": r3(E / len(rows)), "triage_per_repeat_mean": r3((band[rows] == 1).mean()), "triage_in_at_least_one_repeat": r3((band[rows] == 1).any(1).mean())},
         "rule": "at each pre-event endoscopy one sample is drawn at random (seeds 0-99); the endoscopy is called by that sample's L-LATE result at (B)'s fold threshold (its stored (B) call); variant: the other samples of the endoscopy are called by the H&E model (C) at its fold threshold and the endoscopy is positive if any call is; triage = any sample positive; single model = per repeat, averaged; patient = any endoscopy positive", "levels": {}}
    for lv in ("endo", "pat"):
        L = {}
        for nm in ("random_biopsy", "random_biopsy_rest_HE", "triage", "B_all_sequenced"):
            for st in ("sens", "spec"):
                k = f"{lv}_{st}_{nm}"; v = np.atleast_1d(obs[k]); Dk = np.array([np.atleast_1d(d[k]).mean() for d in DR])
                L[f"{st}_{nm}"] = {"mean_over_seeds": r3(v.mean()), "range_2.5_97.5_over_seeds": ci2(v) if len(v) > 1 else None, "ci95_bootstrap_of_seed_mean": ci2(Dk)}
        for nm in ("random_biopsy", "random_biopsy_rest_HE"):
            for st in ("sens", "spec"):
                dv = obs[f"{lv}_{st}_{nm}"] - obs[f"{lv}_{st}_triage"]; Dd = np.array([(d[f"{lv}_{st}_{nm}"] - d[f"{lv}_{st}_triage"]).mean() for d in DR]); pooled = np.concatenate([d[f"{lv}_{st}_{nm}"] - d[f"{lv}_{st}_triage"] for d in DR])
                L[f"d_{st}_{nm}_minus_triage"] = {"mean_over_seeds": r3(dv.mean()), "range_2.5_97.5_over_seeds": ci2(dv), "ci95_bootstrap_of_seed_mean": ci2(Dd), "one_sided_95_lower_seed_mean": lo1(Dd), "p_two_sided_seed_mean": p2(Dd), "ci95_pooled_seeds_and_draws": ci2(pooled), "one_sided_95_lower_pooled": lo1(pooled)}
        R["levels"]["endoscopy" if lv == "endo" else "patient"] = L
    R["sanity_round2_r23_endoscopy_no_vote_T_expected_0.909_0.549"] = [r3(obs["endo_sens_triage"]), r3(obs["endo_spec_triage"])]; res["R41"] = R; dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
