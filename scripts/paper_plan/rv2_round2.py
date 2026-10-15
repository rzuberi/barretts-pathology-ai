"""Review round 2, Part A, items R14-R30 (docs/paper_review_round2_analyses.md @ d681783). Stored predictions only (tb_common.py loaders and the
paper_triage.md rules); nothing is refitted here (Part B is rv2_refit.R). Aggregates -> results/paper_final/review_round2/<TASK>.json; row-level shard
outputs stay under killcoyne_mm/review_round2/.
TASK = r14_<src>_<pop> | r15 | r16_<src>_<pop>_shardNN | r16_<src>_<pop>_merge | r17_<src>_<pop> | r18 | r19 | r20 | r21 | r22 | r23 | r24 |
       r25_shardNN | r25_merge | r26 | r27_<pop> | r28 | r29 | r30      (src their|pkg; pop pre|pre_ndbe)"""
import os, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, beta as beta_dist
from tb_common import *
PRESPEC = "d681783"; OUTR = T + "/results/paper_final/review_round2"; ROW2 = M + "/review_round2"; os.makedirs(OUTR, exist_ok=True); os.makedirs(ROW2, exist_ok=True)
TASK = os.environ["TASK"]
CS = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample").reindex(ids)
GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float)
SDS = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); TILES = SDS.tissue_tiles.values.astype(float); SCAN = SDS.scanner.values
TIME = SM.time.values.astype(float); EV = SM.event.values.astype(int); MBF = SM.mbf.values.astype(float); KP = pd.to_numeric(SM.k_prob, errors="coerce").values
FO0 = load("their")[2]; MM = {"A*": "L-CNV", "B*": "L-LATE", "C*": "L-IMG"}; COMPM = {"A": "L-CNV", "B": "L-LATE", "C": "L-IMG"}
PCODE = pd.Series(pat).astype("category").cat.codes.values
# ------------------------------------------------------------------ helpers (fsa / ci / lo1 / dump as in rv_round1.py)
def fsa(S, b, mask=None, w=None, cell=None, reps=None):
    """fold-stratified AUROC (mean over repeats of within-fold pooled pair concordance); S n x 10 or n; optional mask, pair weights w_i w_j, an extra
    per-sample cell key (pairs must share fold and cell; cell < 0 = excluded) and a subset of repeats (fold structures) to use"""
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
def auc_plain(yy, s):
    yy = np.asarray(yy).astype(int); ok = np.isfinite(s); yy, s = yy[ok], s[ok]; r = rankdata(s); n1 = yy.sum(); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(yy) else np.nan
def ci2(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return pct(a) if len(a) else [None, None]
def lo1(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return r3(np.percentile(a, 5)) if len(a) else None
def ci90(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return [r3(np.percentile(a, 5)), r3(np.percentile(a, 95))] if len(a) else [None, None]
def dstat(obs, draws, one_sided=False, tost=False):
    o = {"delta": r3(obs), "ci95": ci2(draws), "p_two_sided": p2(draws)}
    if one_sided: o["one_sided_95_lower"] = lo1(draws); o["non_inferior_at_-0.05"] = bool(lo1(draws) is not None and lo1(draws) > -0.05)
    if tost: o["ci90_tost"] = ci90(draws); o["equivalent_within_0.05_tost"] = bool(ci90(draws)[0] is not None and ci90(draws)[0] > -0.05 and ci90(draws)[1] < 0.05)
    return o
def cp_ci(k, nn):   # exact Clopper-Pearson 95% CI for a proportion
    lo = beta_dist.ppf(0.025, k, nn - k + 1) if k > 0 else 0.0; hi = beta_dist.ppf(0.975, k + 1, nn - k) if k < nn else 1.0; return [r3(lo), r3(hi)]
def dump(res, name=None):
    res["prespec"] = PRESPEC; json.dump(res, open(f"{OUTR}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV2 DONE", name or TASK, flush=True)
def sens_at_spec(x, yb, spec_t):   # R1 secondary rule: positive if score > the ceil(spec * n_controls)-th smallest held-out control score
    xc = np.sort(x[yb == 0]); q = int(np.ceil(spec_t * len(xc))); thr = xc[q - 1] if q > 0 else -np.inf; return float(np.mean(x[yb == 1] > thr))
def wmean(v, w): return float(v.mean()) if w is None else float((v * w).sum() / w.sum())
# ------------------------------------------------------------------ calls: triage, (A)(B)(C), training-matched A* B* C* (R1 rule, optional bootstrap weights)
def matched_calls(O, I, FO, popm, W=None):
    mc = {a: np.zeros((n, 10), bool) for a in MM}; sstar = []; held = []
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            si = I["L-IMG"][(r, k)]; tri = si.index.values[popm[si.index.values]]; w = None if W is None else W[tri]
            if w is not None: keep = w > 0; tri = tri[keep]; w = w[keep]
            s_tr = si.reindex(tri).values; yc = y[tri] == 1; wy = None if w is None else w[yc]; wn = None if w is None else w[~yc]
            tl = thr_sens(s_tr[yc], 0.95, wy); cs = c_star(s_tr[~yc], 0.90, wn); li = I["L-LATE"][(r, k)].reindex(tri).values
            mid = (s_tr >= tl) & (s_tr <= cs) if tl <= cs else np.zeros(len(tri), bool); sel = mid & yc; nsel = sel.sum() if w is None else w[sel].sum()
            ts = thr_sens(li[sel], 0.8, None if w is None else w[sel]) if nsel >= 5 else thr_sens(li[yc], 0.8, wy)
            tcall = np.where(s_tr < tl, False, np.where(s_tr > cs, True, li >= ts)) if tl <= cs else s_tr > cs
            s_star = 1 - wmean(tcall[~yc].astype(float), wn); sstar.append(s_star); te = FO[:, j] == k
            for a, mn in MM.items():
                xc = I[mn][(r, k)].reindex(tri).values[~yc]
                if w is not None: xc = np.repeat(xc, wn)
                xc = np.sort(xc); q = int(np.ceil(s_star * len(xc))); thr = xc[q - 1] if q > 0 else -np.inf; mc[a][te, j] = O[mn][te, j] > thr
    return mc, sstar
def all_calls(O, I, FO, popm, W=None):
    band, call, info = triage(O, I, FO, popm, W=W); comp, thr = comparators(O, I, FO, popm, W=W); mc, sstar = matched_calls(O, I, FO, popm, W=W)
    CALLS = {"T": call, "A": comp["A"], "B": comp["B"], "C": comp["C"], "A*": mc["A*"], "B*": mc["B*"], "C*": mc["C*"]}; return CALLS, band, info, thr, sstar
def draw_stats(b, CALLS, band, O, per_rep=False):
    """voted (v_) and no-vote (nv_, repeat-averaged) sensitivity / specificity of every call set, evaluation-matched A* B* C* (voted: repeat-mean scores at the
    voted triage's realised specificity; no-vote: per repeat at that repeat's realised specificity), band shares, vote-pivotal share, class-wise band rates"""
    yb = y[b]; P = yb == 1; N = ~P; out = {}; rep = {}
    for s, C in CALLS.items():
        Cb = C[b]; V = Cb.sum(1) >= 6; se = Cb[P].mean(0); sp = 1 - Cb[N].mean(0)
        out[f"v_sens_{s}"] = V[P].mean(); out[f"v_spec_{s}"] = 1 - V[N].mean(); out[f"nv_sens_{s}"] = se.mean(); out[f"nv_spec_{s}"] = sp.mean(); rep[f"sens_{s}"] = se; rep[f"spec_{s}"] = sp
    spec_j = rep["spec_T"]
    for a, m in MM.items():
        out[f"v_sens_{a}_eval"] = sens_at_spec(O[m].mean(1)[b], yb, out["v_spec_T"]); sj = np.array([sens_at_spec(O[m][b, j], yb, spec_j[j]) for j in range(10)])
        out[f"nv_sens_{a}_eval"] = sj.mean(); rep[f"sens_{a}_eval"] = sj
    Bb = band[b]; mid = Bb == 1; Tb = CALLS["T"][b]
    up = np.where(mid, True, Tb).sum(1) >= 6; dn = np.where(mid, False, Tb).sum(1) >= 6; piv = up != dn
    for nm, mk in (("", np.ones(len(b), bool)), ("_P", P), ("_N", N)):
        out[f"share_mid{nm}"] = mid[mk].mean(0).mean(); out[f"share_mid_any{nm}"] = mid[mk].any(1).mean(); out[f"share_pivotal{nm}"] = piv[mk].mean()
    for g in (0, 1, 2): out[f"rate_P_{g}"] = (Bb[P] == g).mean(0).mean(); out[f"rate_N_{g}"] = (Bb[N] == g).mean(0).mean()
    return (out, rep) if per_rep else out
KEYS = None
def keys_of(d): return sorted(d.keys())
def frame(BS): return pd.DataFrame(BS)
def paired(obs, D, s, mode, one_sided=False, tost=False, eval_=False):
    k = f"{mode}_sens_{s}{'_eval' if eval_ else ''}"; o = {"d_sens_triage_minus": dstat(obs[f"{mode}_sens_T"] - obs[k], D[f"{mode}_sens_T"] - D[k], one_sided, tost)}
    if not eval_: o["d_spec_triage_minus"] = dstat(obs[f"{mode}_spec_T"] - obs[f"{mode}_spec_{s}"], D[f"{mode}_spec_T"] - D[f"{mode}_spec_{s}"])
    o["comparator_sensitivity"] = r3(obs[k]); o["comparator_sens_ci95"] = ci2(D[k])
    if not eval_: o["comparator_specificity"] = r3(obs[f"{mode}_spec_{s}"]); o["comparator_spec_ci95"] = ci2(D[f"{mode}_spec_{s}"])
    return o
def maxt(obs, D, names, mode, eval_):
    """max-T (centred bootstrap-t over the joint draws) and Holm adjustment across the three d sens triage - comparator"""
    d = {}; Z = {}; T_ = {}
    for a in names:
        k = f"{mode}_sens_{a}{'_eval' if eval_ else ''}"; dd = D[f"{mode}_sens_T"].values - D[k].values; d0 = obs[f"{mode}_sens_T"] - obs[k]; se = dd.std(ddof=1)
        d[a] = (d0, dd); Z[a] = (dd - dd.mean()) / se if se > 0 else np.zeros(len(dd)); T_[a] = d0 / se if se > 0 else np.inf
    mx = np.max(np.abs(np.stack([Z[a] for a in names])), 0); p_un = {a: p2(d[a][1]) for a in names}; order = sorted(names, key=lambda a: p_un[a]); holm = {}; run = 0.0
    for i, a in enumerate(order): run = max(run, min(1.0, (len(names) - i) * p_un[a])); holm[a] = round(run, 4)
    return {a: {"delta": r3(d[a][0]), "ci95": ci2(d[a][1]), "p_unadjusted_percentile": p_un[a], "p_holm": holm[a], "p_bootstrap_t_unadjusted": round(float(np.mean(np.abs(Z[a]) >= abs(T_[a]))), 4),
                "p_max_T": round(float(np.mean(mx >= abs(T_[a]))), 4)} for a in names}
def endo_pos(c_rows, key):   # any positive sample within an endoscopy unit (key per row)
    return pd.Series(c_rows).groupby(key).max()
res = {"task": TASK}
# ================================================================== R14 (and the shared no-vote draws), R17
def r14_core(src, pop):
    popm = popmask(pop); rows = np.where(popm)[0]; O, I, FO = load(src); CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); boots = boots_for(rows)
    obs = draw_stats(rows, CALLS, band, O); D = frame([draw_stats(b, CALLS, band, O) for b in boots]); return rows, O, I, FO, CALLS, band, info, sstar, obs, D, boots
if TASK.startswith("r14_"):
    _, src, pop = TASK.split("_", 2); rows, O, I, FO, CALLS, band, info, sstar, obs, D, boots = r14_core(src, pop)
    R = {"n_samples": int(len(rows)), "n_patients": int(len(set(pat[rows]))), "n_progressor_samples": int(y[rows].sum()), "draws": int(len(D))}
    R["a_sequencing_cost"] = {k: {"value": r3(obs[k]), "ci95": ci2(D[k])} for k in ("share_mid", "share_mid_any", "share_pivotal", "share_mid_P", "share_mid_any_P", "share_pivotal_P", "share_mid_N", "share_mid_any_N", "share_pivotal_N")}
    R["a_sequencing_cost"]["note"] = "share_mid = per-repeat mean share in the middle band (sequenced without the vote); share_mid_any = in the middle band in >= 1 repeat; share_pivotal = >= 6-of-10 call changes between all middle-band calls positive and all negative"
    R["triage_no_vote"] = {"sensitivity": r3(obs["nv_sens_T"]), "sens_ci95": ci2(D["nv_sens_T"]), "specificity": r3(obs["nv_spec_T"]), "spec_ci95": ci2(D["nv_spec_T"])}
    R["triage_voted"] = {"sensitivity": r3(obs["v_sens_T"]), "sens_ci95": ci2(D["v_sens_T"]), "specificity": r3(obs["v_spec_T"]), "spec_ci95": ci2(D["v_spec_T"])}
    R["b_no_vote_paired"] = {s: paired(obs, D, s, "nv", one_sided=s in ("B", "B*")) for s in ("A", "B", "C", "A*", "B*", "C*")}
    R["c_no_vote_evaluation_matched"] = {s: paired(obs, D, s, "nv", one_sided=True, eval_=True) for s in ("A*", "B*", "C*")}
    R["c_no_vote_evaluation_matched"]["note"] = "per repeat, threshold on that repeat's held-out scores at the triage's realised specificity in that repeat (R1 secondary rule), re-chosen in every bootstrap draw"
    R["voted_for_reference"] = {"training_matched": {s: paired(obs, D, s, "v", one_sided=s in ("B", "B*")) for s in ("A", "B", "C", "A*", "B*", "C*")}, "evaluation_matched": {s: paired(obs, D, s, "v", one_sided=True, eval_=True) for s in ("A*", "B*", "C*")}}
    R["sanity_round1"] = {"voted_d_sens_vs_Bstar": r3(obs["v_sens_T"] - obs["v_sens_B*"]), "voted_d_sens_vs_B": r3(obs["v_sens_T"] - obs["v_sens_B"]), "share_mid_any": r3(obs["share_mid_any"]), "share_mid_mean": r3(obs["share_mid"]), "nv_sens_T": r3(obs["nv_sens_T"]), "training_specificity_target_median": r3(np.median(sstar))}
    res.update({"src": src, "pop": pop, "R14": R}); dump(res)
elif TASK.startswith("r17_"):
    _, src, pop = TASK.split("_", 2); rows, O, I, FO, CALLS, band, info, sstar, obs, D, boots = r14_core(src, pop); names = ["A*", "B*", "C*"]
    R = {m + ("_evaluation_matched" if e else "_training_matched"): maxt(obs, D, names, mode, e) for mode, m in (("v", "voted"), ("nv", "no_vote")) for e in (False, True)}
    R["note"] = "p_unadjusted_percentile = two-sided bootstrap percentile p (round-1 convention); p_holm from it; p_max_T from the centred bootstrap-t draws, single-step over the three comparators"
    res.update({"src": src, "pop": pop, "draws": int(len(D)), "R17": R}); dump(res)
# ================================================================== R15 between-repeat variability
elif TASK == "r15":
    rows, O, I, FO, CALLS, band, info, sstar, obs, D, boots = r14_core("their", "pre"); R = {}
    def per_rep_stats(b):
        o = {}
        for j in range(10):
            o[f"dAUC_IMG_{j}"] = fsa(O["L-IMG"], b, reps=[j]) - fsa(O["L-CNV"], b, reps=[j]); o[f"dAUC_LATE_{j}"] = fsa(O["L-LATE"], b, reps=[j]) - fsa(O["L-CNV"], b, reps=[j])
        _, rep = draw_stats(b, CALLS, band, O, per_rep=True)
        for j in range(10): o[f"dsens_B_{j}"] = rep["sens_T"][j] - rep["sens_B"][j]; o[f"dsens_Bstar_{j}"] = rep["sens_T"][j] - rep["sens_B*"][j]
        return o
    ob = per_rep_stats(rows); DB = frame([per_rep_stats(b) for b in boots])
    for nm, lab in (("dAUC_IMG", "dAUROC_LIMG_minus_LCNV"), ("dAUC_LATE", "dAUROC_LLATE_minus_LCNV"), ("dsens_B", "no_vote_d_sens_triage_minus_B"), ("dsens_Bstar", "no_vote_d_sens_triage_minus_Bstar")):
        pts = np.array([ob[f"{nm}_{j}"] for j in range(10)]); lows = [float(np.percentile(DB[f"{nm}_{j}"], 5)) for j in range(10)]
        avg = DB[[f"{nm}_{j}" for j in range(10)]].mean(1).values; vb = float(avg.var(ddof=1)); vr = float(pts.var(ddof=1)); d = float(pts.mean())
        R[lab] = {"delta_repeat_mean": r3(d), "per_repeat": [r3(v) for v in pts], "sd_over_repeats": r3(pts.std(ddof=1)), "range_over_repeats": [r3(pts.min()), r3(pts.max())],
                  "per_repeat_one_sided_lower": [r3(v) for v in lows], "n_repeats_lower_above_-0.05": int(sum(v > -0.05 for v in lows)),
                  "V_boot": round(vb, 6), "V_rep": round(vr, 6), "standard_one_sided_lower": lo1(avg), "variance_inflated_one_sided_lower": r3(d - 1.645 * np.sqrt(vb + vr))}
    res["R15"] = R; dump(res)
# ================================================================== R16 selection-inclusive shards and merge
elif TASK.startswith("r16_"):
    body = TASK[4:]; src, rest = body.split("_", 1); pop, last = rest.rsplit("_", 1); popm = popmask(pop); rows = np.where(popm)[0]; D16 = f"{ROW2}/selci"; os.makedirs(D16, exist_ok=True)
    O, I, FO = load(src); boots = boots_for(rows); NSH = 20; per = NB // NSH
    if last == "merge":
        CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); obs = draw_stats(rows, CALLS, band, O); keys = keys_of(obs)
        files = sorted(glob.glob(f"{D16}/{src}_{pop}_shard_*.npy")); A = np.vstack([np.load(f) for f in files]); assert A.shape == (NB, len(keys)), (A.shape, len(files))
        D = pd.DataFrame(A, columns=keys); R = {"draws": int(len(D)), "n_shards": len(files), "selection_inclusive_training_matched": {}, "selection_inclusive_evaluation_matched": {}}
        for mode, m in (("v", "voted"), ("nv", "no_vote")):
            R["selection_inclusive_training_matched"][m] = {"triage": {"sensitivity": r3(obs[f"{mode}_sens_T"]), "sens_ci95": ci2(D[f"{mode}_sens_T"]), "specificity": r3(obs[f"{mode}_spec_T"]), "spec_ci95": ci2(D[f"{mode}_spec_T"])},
                                                              **{s: paired(obs, D, s, mode, one_sided=True, tost=s == "B*") for s in ("A*", "B*", "C*")}, "vs_B": paired(obs, D, "B", mode, one_sided=True, tost=True)}
            R["selection_inclusive_evaluation_matched"][m] = {s: paired(obs, D, s, mode, one_sided=True, tost=s == "B*", eval_=True) for s in ("A*", "B*", "C*")}
        R["share_sequenced_ci95"] = {"share_mid": ci2(D["share_mid"]), "share_mid_any": ci2(D["share_mid_any"]), "share_pivotal": ci2(D["share_pivotal"])}
        dB = D["v_sens_T"] - D["v_sens_B"]; d0 = obs["v_sens_T"] - obs["v_sens_B"]
        R["bias_primary_criterion_voted_vs_B"] = {"observed": r3(d0), "bootstrap_median": r3(np.median(dB)), "bootstrap_mean": r3(dB.mean()), "bias_corrected_2obs_minus_mean": r3(2 * d0 - dB.mean()), "ci95": ci2(dB), "one_sided_95_lower": lo1(dB)}
        dBn = D["nv_sens_T"] - D["nv_sens_B"]; d0n = obs["nv_sens_T"] - obs["nv_sens_B"]
        R["bias_primary_criterion_no_vote_vs_B"] = {"observed": r3(d0n), "bootstrap_median": r3(np.median(dBn)), "bootstrap_mean": r3(dBn.mean()), "bias_corrected_2obs_minus_mean": r3(2 * d0n - dBn.mean()), "ci95": ci2(dBn), "one_sided_95_lower": lo1(dBn)}
        old = sorted(glob.glob(f"{ROW}/selci/{src}_{pop}_shard_*.npy"))
        if old:
            A0 = np.vstack([np.load(f) for f in old]); d_old = A0[:, 0] - A0[:, 4]
            R["check_stored_tb_selci_draws"] = {"draws": int(len(A0)), "d_sens_vs_B_voted": {"median": r3(np.median(d_old)), "mean": r3(d_old.mean()), "ci95": ci2(d_old), "one_sided_95_lower": lo1(d_old)}, "max_abs_diff_vs_recomputed": r3(np.max(np.abs(d_old - dB.values)))}
        R["sanity_round1"] = {"voted_d_sens_vs_Bstar_fixed": r3(obs["v_sens_T"] - obs["v_sens_B*"]), "voted_d_sens_vs_B_fixed": r3(d0)}
        res.update({"src": src, "pop": pop, "R16": R}); dump(res, f"r16_{src}_{pop}")
    else:
        sh = int(last.replace("shard", "")); outf = f"{D16}/{src}_{pop}_shard_{sh:02d}.npy"
        if os.path.exists(outf): raise SystemExit(0)
        out = []
        for b in boots[sh * per:(sh + 1) * per]:
            W = np.bincount(b, minlength=n); CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm, W=W); d = draw_stats(b, CALLS, band, O); out.append([d[k] for k in keys_of(d)])
        np.save(outf, np.array(out)); print("RV2 R16 SHARD DONE", TASK, flush=True)
# ================================================================== R18 tissue amount
elif TASK == "r18":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); R = {"strata": {}}
    STR = {"overlap_lt256": TILES < 256, "no_overlap_ge256": TILES >= 256}
    def stratum_stats(b, mk):
        o = {m: fsa(O[m], b, mask=mk) for m in ("L-CNV", "L-IMG", "L-LATE")}; bb = b[mk[b]]
        if len(bb) and 0 < y[bb].sum() < len(bb): o.update(draw_stats(bb, {"T": CALLS["T"]}, band, O))
        return o
    obs = {s: stratum_stats(rows, mk) for s, mk in STR.items()}; BS = {s: frame([stratum_stats(b, mk) for b in boots]) for s, mk in STR.items()}
    for s, mk in STR.items():
        o, D = obs[s], BS[s]; d = D["L-IMG"] - D["L-CNV"]
        R["strata"][s] = {"n_samples": int((popm & mk).sum()), "n_progressor_samples": int(y[popm & mk].sum()), "n_patients": int(len(set(pat[popm & mk]))),
                          "auroc": {m: {"value": r3(o[m]), "ci95": ci2(D[m])} for m in ("L-CNV", "L-IMG", "L-LATE")}, "IMG_minus_CNV": dstat(o["L-IMG"] - o["L-CNV"], d, one_sided=True),
                          "triage_voted": {"sensitivity": r3(o["v_sens_T"]), "sens_ci95": ci2(D["v_sens_T"]), "specificity": r3(o["v_spec_T"]), "spec_ci95": ci2(D["v_spec_T"]), "share_mid_any": r3(o["share_mid_any"]), "share_pivotal": r3(o["share_pivotal"])},
                          "triage_no_vote": {"sensitivity": r3(o["nv_sens_T"]), "sens_ci95": ci2(D["nv_sens_T"]), "specificity": r3(o["nv_spec_T"]), "spec_ci95": ci2(D["nv_spec_T"]), "share_sequenced": r3(o["share_mid"]), "share_ci95": ci2(D["share_mid"])}}
    di = (obs["overlap_lt256"]["L-IMG"] - obs["overlap_lt256"]["L-CNV"]) - (obs["no_overlap_ge256"]["L-IMG"] - obs["no_overlap_ge256"]["L-CNV"])
    dd = (BS["overlap_lt256"]["L-IMG"] - BS["overlap_lt256"]["L-CNV"]) - (BS["no_overlap_ge256"]["L-IMG"] - BS["no_overlap_ge256"]["L-CNV"])
    R["interaction_dIMGCNV_lt256_minus_ge256"] = dstat(di, dd)
    tert = np.full(n, -1); tert[rows] = pd.qcut(TILES[rows], 3, labels=False); edges = [float(v) for v in np.quantile(TILES[rows], [0, 1 / 3, 2 / 3, 1])]
    om = {m: fsa(O[m], rows, cell=tert) for m in ("L-CNV", "L-IMG", "L-LATE")}; Bm = frame([{m: fsa(O[m], b, cell=tert) for m in ("L-CNV", "L-IMG", "L-LATE")} for b in boots])
    R["tissue_matched_auroc"] = {"tertile_edges_tissue_tiles": edges, "auroc": {m: {"value": r3(om[m]), "ci95": ci2(Bm[m])} for m in om}, "IMG_minus_CNV": dstat(om["L-IMG"] - om["L-CNV"], Bm["L-IMG"] - Bm["L-CNV"], one_sided=True),
                                 "counts_by_tertile_and_class": {f"tertile_{t + 1}": {"progressor": int(((tert == t) & (y == 1))[rows].sum()), "non_progressor": int(((tert == t) & (y == 0))[rows].sum())} for t in range(3)}}
    R["unstratified_for_reference"] = {m: r3(fsa(O[m], rows)) for m in ("L-CNV", "L-IMG", "L-LATE")}; res["R18"] = R; dump(res)
# ================================================================== R19 scanner fallback
elif TASK == "r19":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; band_m, call_m, _ = triage(O, I, FO, popm); CT = vote(call_m); R = {}
    X = pd.read_csv(f"{ROW}/scanner/cross_b_s39_none.csv", dtype={"Sample": str, "Patient": str}); te = X[X.role == "test"]; inn = X[X.role == "inner"]
    yi = inn.y.values == 1; tl0 = thr_sens(inn.pred.values[yi], 0.95); cs0 = c_star(inn.pred.values[~yi], 0.90)
    FZ = json.load(open(T + "/results/paper_final/triage_robustness/frozen_cutoffs_their.json"))["cutoffs"]; tseq = FZ["t_seq"]
    tix = ix[te.Sample].values; img = te.pred.values; late = (img + O["L-CNV"].mean(1)[tix]) / 2; s10 = rows[SCAN[rows] == "C13210"]; same = bool(set(tix) == set(s10))
    ref = json.load(open(T + "/results/paper_final/triage/their_pre.json"))["bands"]; low_t, high_t = ref["low"]["all"]["value"], ref["high"]["all"]["value"]
    tl1 = float(np.quantile(img, low_t)); cs1 = float(np.quantile(img, 1 - high_t))   # label-free reset: quantiles of the transfer scores reproducing the internal low / high shares
    pp = pat[s10]; upp = np.unique(pp); ro = {q: s10[pp == q] for q in upp}; rng = np.random.RandomState(0); bb = []
    while len(bb) < NB:
        b = np.concatenate([ro[upp[i]] for i in rng.choice(len(upp), len(upp))])
        if 0 < y[b].sum() < len(b): bb.append(b)
    def rule(tl, cs):
        bandt = np.where(img < tl, 0, np.where(img > cs, 2, 1)) if tl <= cs else np.where(img > cs, 2, 0); callt = np.where(bandt == 1, late >= tseq, bandt == 2)
        cf = np.zeros(n, bool); cf[tix] = callt; bf = np.full(n, -1); bf[tix] = bandt; return cf, bf
    def summ(cf, bf):
        st = lambda b: (float(cf[b][y[b] == 1].mean()), float(1 - cf[b][y[b] == 0].mean()), *[float((bf[b] == g).mean()) for g in (0, 1, 2)])
        o = st(s10); B = np.array([st(b) for b in bb]); return {"sensitivity": r3(o[0]), "sens_ci95": ci2(B[:, 0]), "specificity": r3(o[1]), "spec_ci95": ci2(B[:, 1]), "band_shares": {nm: r3(o[2 + g]) for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}}
    R["setting"] = {"n_samples": int(len(s10)), "n_transfer_test_rows": int(len(tix)), "test_rows_equal_C13210_pre_event": same, "n_patients": int(len(upp)), "n_P_patients": int(len(set(pat[s10][y[s10] == 1]))), "t_seq_frozen": r3(tseq), "internal_target_shares": {"low": low_t, "high": high_t}}
    R["transfer_unreset"] = {"cutoffs": {"t_low": r3(tl0), "c_star": r3(cs0)}, **summ(*rule(tl0, cs0))}
    R["transfer_fallback_reset"] = {"cutoffs": {"t_low": r3(tl1), "c_star": r3(cs1)}, **summ(*rule(tl1, cs1))}
    stm = lambda b: (float(CT[b][y[b] == 1].mean()), float(1 - CT[b][y[b] == 0].mean())); o = stm(s10); B = np.array([stm(b) for b in bb])
    R["mixed_scanner_cv_same_samples"] = {"sensitivity": r3(o[0]), "sens_ci95": ci2(B[:, 0]), "specificity": r3(o[1]), "spec_ci95": ci2(B[:, 1]), "band_shares_mean_over_repeats": {nm: r3((band_m[s10] == g).mean()) for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}}
    r1 = {g: (band_m[rows][y[rows] == 1] == g).mean(0).mean() for g in (0, 1, 2)}; r0 = {g: (band_m[rows][y[rows] == 0] == g).mean(0).mean() for g in (0, 1, 2)}
    R["class_wise_band_rates"] = {"progressor": {nm: r3(r1[g]) for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}, "non_progressor": {nm: r3(r0[g]) for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}}
    R["expected_shares_by_prevalence"] = {}
    for pi in (0.02, 0.05, 0.10, 0.15, 0.20, float(y[rows].mean())):
        R["expected_shares_by_prevalence"]["observed_%.3f" % pi if pi > 0.2 else f"{pi:.2f}"] = {nm: r3(pi * r1[g] + (1 - pi) * r0[g]) for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}
    up = np.unique(pat[rows]); ro2 = {q: rows[pat[rows] == q] for q in up}; rng = np.random.RandomState(0); sh = []
    for _ in range(NB):
        j = rng.randint(10); b = np.concatenate([ro2[up[i]] for i in rng.choice(len(up), 117)]); sh.append([(band_m[b, j] == g).mean() for g in (0, 1, 2)])
    sh = np.array(sh); R["trigger_prediction_interval_117_patients"] = {"draws": NB, "mean_samples_per_draw": r3(np.mean([len(ro2[q]) for q in up]) * 117), "n_internal_patients": int(len(up)),
        **{nm: {"mean": r3(sh[:, g].mean()), "pi95": ci2(sh[:, g])} for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}, "note": "117 internal pre-event patients resampled with replacement (all their samples), calls from a randomly chosen repeat per draw"}
    res["R19"] = R; dump(res)
# ================================================================== R20 the 8 patients without a slide (sheet + published P)
elif TASK == "r20":
    S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; KD = T + "/feasibility/paper_plan/killcoyne"
    fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
    A = pd.read_csv(KD + "/kr_samples.csv", dtype=str).set_index("Sample"); A["k"] = pd.to_numeric(A.k_prob, errors="coerce"); A["yy"] = (A.Status == "P").astype(int)
    A["mbf"] = pd.to_numeric(A.index.map(fd["Months before final"]), errors="coerce"); A["ogd"] = A.index.map(fd["Path_class_per_OGD"])
    hg = A[(A.Pathology.isin(["HGD", "IMC"])) | (A.ogd.isin(["HGD", "IMC", "HGD/IMC"]))]; tev = hg.groupby("Patient").mbf.max()
    A["tev"] = [tev.get(p, 0.0) if yy == 1 else np.nan for p, yy in zip(A.Patient, A.yy)]; A["pre"] = [True if yy == 0 else (m > t) for yy, m, t in zip(A.yy, A.mbf, A.tev)]
    p88 = sorted(A.Patient.unique()); p80 = sorted(set(SM.Patient)); miss = sorted(set(p88) - set(p80)); inP = sorted(set(SM.Patient[SM.y == 1]))
    assert len(p88) == 88 and len(p80) == 80 and len(miss) == 8 and all(A[A.Patient == p].yy.max() == 1 for p in miss)
    def grp(ps, pre_only):
        d = A[A.Patient.isin(ps)]; d = d[d.pre] if pre_only else d; pm = d.groupby("Patient").k.mean(); ns = d.groupby("Patient").size()
        q = lambda v: {"n": int(len(v)), "mean": r3(np.mean(v)), "median": r3(np.median(v)), "iqr": [r3(np.percentile(v, 25)), r3(np.percentile(v, 75))]} if len(v) else {"n": 0}
        return {"n_patients": int(d.Patient.nunique()), "n_samples": int(len(d)), "P_per_sample": q(d.k.dropna().values), "P_per_patient_mean": q(pm.dropna().values), "samples_per_patient": q(ns.values),
                "grade_mix": {k: int(v) for k, v in d.Pathology.value_counts().items()}, "share_P_ge_0.5_samples": r3((d.k >= 0.5).mean())}
    R = {"n_patients_sheet": len(p88), "n_patients_analysed": len(p80), "n_without_slide": len(miss), "all_without_slide_are_progressors": True, "n_included_progressor_patients": len(inP)}
    for lab, ps in (("without_slide_8", miss), ("included_progressors_37", inP)): R[lab] = {"all_samples": grp(ps, False), "pre_event_samples": grp(ps, True)}
    def pauc(d):
        pats = np.array(sorted(d.Patient.unique())); rb = {p: np.where(d.Patient.values == p)[0] for p in pats}; yy, ss = d.yy.values, d.k.values
        o = auc_plain(yy, ss); rng = np.random.RandomState(0); bs = []
        while len(bs) < NB:
            idx = np.concatenate([rb[pats[i]] for i in rng.choice(len(pats), len(pats))])
            if 0 < yy[idx].sum() < len(idx): bs.append(auc_plain(yy[idx], ss[idx]))
        return {"n_samples": int(len(d)), "n_patients": int(len(pats)), "n_P_patients": int(d[d.yy == 1].Patient.nunique()), "auroc": r3(o), "ci95": ci2(bs)}
    pre88 = A[A.pre & A.k.notna()]; pre80 = pre88[pre88.Patient.isin(p80)]; an = A.loc[ids[PRE]].copy()
    R["plain_auroc_P_pre_event"] = {"all_88_patients": pauc(pre88), "80_patients_all_sheet_pre_event_samples": pauc(pre80), "80_patients_571_analysed_samples": pauc(an[an.k.notna()])}
    rows8 = fd.loc[A.index[A.Patient.isin(miss)].intersection(fd.index)]
    notes = {}
    for c in ("Pathology notes", "inconsistency", "notes on label review", "excluded", "remove", "Replicate", "Set", "Batch"):
        if c in rows8.columns: v = rows8[c].fillna("").astype(str).str.strip(); notes[c] = {"n_nonempty": int((v != "").sum()), "distinct_values": sorted(set(v[v != ""].str[:80]))[:12]}
    try:
        sl = pd.read_csv(S + "/slide_matching.csv", dtype=str); key = sl["New Name (Nov 2020)"].fillna("").str.strip(); pid = rows8["Path ID"].fillna("").str.strip()
        m = sl[key.isin(set(pid))]; notes["slide_matching"] = {"n_sheet_rows_8_patients": int(len(rows8)), "n_matched_slide_rows": int(len(m)), "n_with_slide_file": int(m["Slide file"].fillna("").str.strip().ne("").sum()),
                                                                  "slide_notes_distinct": sorted(set(m["Slide Notes"].fillna("").str.strip()[lambda s: s != ""].str[:80]))[:12], "notes_distinct": sorted(set(m["Notes"].fillna("").str.strip()[lambda s: s != ""].str[:80]))[:12]}
    except Exception as e: notes["slide_matching"] = {"error": str(e)[:200]}
    R["reason_no_slide"] = {"source_columns_checked": notes, "conclusion": "not recorded" if not any(v.get("n_nonempty", 0) for k, v in notes.items() if k != "slide_matching") else "see columns"}
    res["R20"] = R; dump(res)
# ================================================================== R21 LOPO bias
elif TASK == "r21":
    O, I, FO = load("their"); R = {}
    LI = pd.read_csv(M + "/cv/lopo_intercepts/arm_00.csv", dtype={"Patient": str}); assert (LI.max_abs_pred_diff < 1e-6).all(); io = pd.Series(LI.train_prev.values, index=LI.Patient).reindex(pat).values
    LP = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in sorted(glob.glob(f"{M}/linear/cfg_00_chunk_*.csv"))]); LP = LP[LP.rule == "class_min"].set_index("Sample").reindex(ids); lopo = LP.pred.values.astype(float)
    assert np.isfinite(lopo).all() and np.isfinite(io).all(); R["stored_lopo_L_CNV"] = {"exists": True, "path": "killcoyne_mm/linear/cfg_00_chunk_*.csv (rule class_min; their matrix; 676 rows)", "lopo_lambda_range": [r3(LP["lambda"].min()), r3(LP["lambda"].max())]}
    R["intercept_only_lopo"] = {"path": "killcoyne_mm/cv/lopo_intercepts/arm_00.csv (train_prev of each leave-one-patient-out fit)", "n_patients": int(LI.Patient.nunique())}
    for pop in ("pre", "pre_ndbe"):
        popm = popmask(pop); rows = np.where(popm)[0]; boots = boots_for(rows); SC = {"intercept_only_LOPO": io, "P_published": KP, "L-CNV_LOPO_stored": lopo, "L-CNV_10x10": O["L-CNV"]}
        obs = {m: fsa(S, rows) for m, S in SC.items()}; B = frame([{m: fsa(S, b) for m, S in SC.items()} for b in boots])
        op = {m: auc_plain(y[rows], (S.mean(1) if np.ndim(S) == 2 else S)[rows]) for m, S in SC.items()}; Bp = frame([{m: auc_plain(y[b], (S.mean(1) if np.ndim(S) == 2 else S)[b]) for m, S in SC.items()} for b in boots])
        R[pop] = {"n_samples": int(len(rows)), "fold_stratified_10x10_pairs": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in SC}, "plain_pooled_auroc": {m: {"value": r3(op[m]), "ci95": ci2(Bp[m])} for m in SC},
                  "a_intercept_only_vs_P": dstat(obs["intercept_only_LOPO"] - obs["P_published"], B["intercept_only_LOPO"] - B["P_published"]),
                  "a_intercept_only_vs_0.5_fold_stratified": r3(obs["intercept_only_LOPO"] - 0.5),
                  "b_LOPO_minus_10x10_L_CNV": dstat(obs["L-CNV_LOPO_stored"] - obs["L-CNV_10x10"], B["L-CNV_LOPO_stored"] - B["L-CNV_10x10"]),
                  "b_LOPO_minus_10x10_L_CNV_plain": dstat(op["L-CNV_LOPO_stored"] - op["L-CNV_10x10"], Bp["L-CNV_LOPO_stored"] - Bp["L-CNV_10x10"]),
                  "P_minus_10x10_L_CNV": dstat(obs["P_published"] - obs["L-CNV_10x10"], B["P_published"] - B["L-CNV_10x10"])}
    res["R21"] = R; dump(res)
# ================================================================== R22 package-feature triage tables (reporting only)
elif TASK == "r22":
    J = lambda p: json.load(open(T + p)); tp = J("/results/paper_final/triage/pkg_pre.json"); lv = J("/results/paper_final/triage_robustness/levels_pkg_pre.json"); r1 = J("/results/paper_final/review_round1/r1_pkg_pre.json"); r8 = J("/results/paper_final/review_round1/r8.json")
    R = {"sample_level_voted": {s: {k: tp["strategies"][s][k] for k in ("sensitivity", "specificity") if k in tp["strategies"][s]} for s in ("triage_LATE", "A", "B", "C")},
         "primary_criterion": tp["primary_criterion"], "bands": tp["bands"], "matched_comparators_voted_R1": r1["R1"], "endoscopy_level_voted": lv["endoscopy_level"], "patient_level_voted": lv["patient_level"], "sequenced": lv["sequenced"],
         "sample_level_no_vote_R8": r8["R8"]["pkg"]}
    f14 = f"{OUTR}/r14_pkg_pre.json"; R["no_vote_paired_R14"] = json.load(open(f14))["R14"] if os.path.exists(f14) else "r14_pkg_pre.json not yet available"
    res["R22"] = R; dump(res)
# ================================================================== R23 endoscopy and patient level
elif TASK == "r23":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); R = {}
    def boots_tagged(rows, seed=0, nb=NB):
        pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(seed); out = []
        while len(out) < nb:
            dr = rng.choice(len(up), len(up)); b = np.concatenate([ro[up[i]] for i in dr]); t = np.concatenate([np.full(len(ro[up[i]]), q) for q, i in enumerate(dr)])
            if 0 < y[b].sum() < len(b): out.append((b, t))
        return out
    ES = ["T", "B*", "C*", "A", "B", "C"]
    def endo_metrics(b, t):
        key = pd.Series(ENDO[b]).astype(str).values + "#" + t.astype(str); lab = pd.Series(y[b]).groupby(key).max(); o = {}
        for s in ES:
            C = CALLS[s][b]; pv = pd.Series(C.sum(1) >= 6).groupby(key).max(); o[f"v_sens_{s}"] = pv[lab == 1].mean(); o[f"v_spec_{s}"] = 1 - pv[lab == 0].mean()
            se = []; sp = []
            for j in range(10): pj = pd.Series(C[:, j]).groupby(key).max(); se.append(pj[lab == 1].mean()); sp.append(1 - pj[lab == 0].mean())
            o[f"nv_sens_{s}"] = np.mean(se); o[f"nv_spec_{s}"] = np.mean(sp)
        return o
    BT = boots_tagged(rows); obs = endo_metrics(rows, np.zeros(len(rows), int)); D = frame([endo_metrics(b, t) for b, t in BT])
    eu = np.unique(ENDO[rows]); R["endoscopy_level"] = {"n_endoscopies": int(len(eu)), "n_progressor_endoscopies": int(pd.Series(y[rows]).groupby(ENDO[rows]).max().sum())}
    for mode, m in (("v", "voted"), ("nv", "no_vote")):
        R["endoscopy_level"][m] = {s: {"sensitivity": r3(obs[f"{mode}_sens_{s}"]), "sens_ci95": ci2(D[f"{mode}_sens_{s}"]), "specificity": r3(obs[f"{mode}_spec_{s}"]), "spec_ci95": ci2(D[f"{mode}_spec_{s}"])} for s in ES}
        R["endoscopy_level"][m]["paired_triage_minus"] = {s: {"d_sens": dstat(obs[f"{mode}_sens_T"] - obs[f"{mode}_sens_{s}"], D[f"{mode}_sens_T"] - D[f"{mode}_sens_{s}"]), "d_spec": dstat(obs[f"{mode}_spec_T"] - obs[f"{mode}_spec_{s}"], D[f"{mode}_spec_T"] - D[f"{mode}_spec_{s}"])} for s in ES if s != "T"}
    # patient level (voted)
    ppat = pd.Series(y[rows], index=pat[rows]).groupby(level=0).max(); nP = int((ppat == 1).sum()); nN = int((ppat == 0).sum()); PL = {"n_progressor_patients": nP, "n_nonprogressor_patients": nN}
    def pstats(b, t):
        key = pd.Series(pat[b]).astype(str).values + "#" + t.astype(str); lab = pd.Series(y[b]).groupby(key).max(); o = {}
        for s in ES:
            pv = pd.Series(vote(CALLS[s])[b]).groupby(key).max(); o[f"miss_{s}"] = float((~pv[lab == 1]).mean()); o[f"fp_{s}"] = float(pv[lab == 0].mean())
        return o
    po = pstats(rows, np.zeros(len(rows), int)); PD = frame([pstats(b, t) for b, t in BT])
    for s in ES:
        km = int(round(po[f"miss_{s}"] * nP)); kf = int(round(po[f"fp_{s}"] * nN))
        PL[s] = {"progressor_patients_missed": km, "missed_share": r3(po[f"miss_{s}"]), "missed_ci95_clopper_pearson": cp_ci(km, nP), "nonprogressor_patients_flagged": kf, "flagged_share": r3(po[f"fp_{s}"]), "flagged_ci95_clopper_pearson": cp_ci(kf, nN)}
    PL["paired_triage_minus"] = {s: {"d_missed_share": dstat(po["miss_T"] - po[f"miss_{s}"], PD["miss_T"] - PD[f"miss_{s}"]), "d_flagged_share": dstat(po["fp_T"] - po[f"fp_{s}"], PD["fp_T"] - PD[f"fp_{s}"]),
                                     "d_missed_count": int(round((po["miss_T"] - po[f"miss_{s}"]) * nP)), "d_flagged_count": int(round((po["fp_T"] - po[f"fp_{s}"]) * nN))} for s in ("B", "C", "B*", "C*")}
    R["patient_level_voted"] = PL
    # burden per non-progressor
    er = pd.DataFrame({"e": ENDO[rows], "p": pat[rows], "mbf": MBF[rows], "y": y[rows]}); ep = er.groupby("e").agg(p=("p", "first"), mbf=("mbf", "max"), y=("y", "max"))
    span = ep.groupby("p").mbf.agg(lambda v: (v.max() - v.min()) / 12.0); nend = ep.groupby("p").size(); pst = ep.groupby("p").y.max(); BU = {}
    for s in ("T", "B", "C", "B*", "C*"):
        def flagged(c_rows):
            fe = pd.Series(c_rows).groupby(er.e.values).max(); return fe.groupby(ep.p.reindex(fe.index).values).sum().reindex(pst.index).fillna(0).astype(float)
        fv = flagged(vote(CALLS[s])[rows]); fnv = sum(flagged(CALLS[s][rows, j]) for j in range(10)) / 10.0
        o = {}
        for mode, f in (("voted", fv), ("no_vote_mean", fnv)):
            npm = (pst == 0) & (nend > 1); single = (pst == 0) & (nend == 1)
            o[mode] = {"flagged_endoscopies_per_nonprogressor_patient_year": r3(f[npm].sum() / span[npm].sum()), "patient_years": r3(span[npm].sum()), "n_nonprogressors_multi_endoscopy": int(npm.sum()),
                       "single_endoscopy_nonprogressors": {"n": int(single.sum()), "flagged_endoscopies": r3(f[single].sum())},
                       "distribution_flagged_per_patient": {st: {c: r3((f[pst == v].clip(upper=3).round() == c).mean()) if mode == "voted" else None for c in (0, 1, 2, 3)} for st, v in (("progressor", 1), ("non_progressor", 0))}}
            if mode == "no_vote_mean":
                dist = {st: {c: 0.0 for c in (0, 1, 2, 3)} for st in ("progressor", "non_progressor")}
                for j in range(10):
                    fj = flagged(CALLS[s][rows, j]).clip(upper=3)
                    for st, v in (("progressor", 1), ("non_progressor", 0)):
                        for c in (0, 1, 2, 3): dist[st][c] += (fj[pst == v] == c).mean() / 10.0
                o[mode]["distribution_flagged_per_patient"] = {st: {c: r3(dist[st][c]) for c in (0, 1, 2, 3)} for st in dist}
        BU[s] = o
    BU["note"] = "distribution keys 0, 1, 2, 3 = 0, 1, 2, >= 3 flagged pre-event endoscopies per patient (share of patients); follow-up = first to last pre-event endoscopy (months before final / 12)"
    R["burden"] = BU; res["R23"] = R; dump(res)
# ================================================================== R24 current care, grade, sampling intensity
elif TASK == "r24":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); R = {}
    WG = np.full(n, np.nan)
    for p in np.unique(pat):
        idx = np.where((pat == p) & PRE)[0]
        for i in idx: WG[i] = np.nanmax(GR[idx][MBF[idx] >= MBF[i]])
    D1 = (GR == 1) | (GR == 2); D2 = (WG == 1) | (WG == 2); DC = {"D1_grade_ID_or_LGD": D1, "D2_worst_grade_to_date_ID_or_LGD": D2}
    ppat = pd.Series(y[rows], index=pat[rows]).groupby(level=0).max(); R["comparator_D"] = {}
    for nm, c in DC.items():
        cr = c[rows]; sens = cr[y[rows] == 1].mean(); spec = 1 - cr[y[rows] == 0].mean(); B = [(c[b][y[b] == 1].mean(), 1 - c[b][y[b] == 0].mean()) for b in boots]
        ep = pd.Series(cr).groupby(ENDO[rows]).max(); el = pd.Series(y[rows]).groupby(ENDO[rows]).max(); pp_ = pd.Series(cr).groupby(pat[rows]).max()
        R["comparator_D"][nm] = {"sample": {"sensitivity": r3(sens), "sens_ci95": ci2([x[0] for x in B]), "specificity": r3(spec), "spec_ci95": ci2([x[1] for x in B]), "share_flagged": r3(cr.mean())},
                                 "endoscopy": {"sensitivity": r3(ep[el == 1].mean()), "specificity": r3(1 - ep[el == 0].mean())},
                                 "patient": {"progressor_caught": int(pp_[ppat == 1].sum()), "progressor_missed": int((~pp_[ppat == 1].astype(bool)).sum()), "nonprogressor_flagged": int(pp_[ppat == 0].sum()), "nonprogressor_unflagged": int((~pp_[ppat == 0].astype(bool)).sum())}}
    R["table4_reference_voted"] = {s: {"sensitivity": r3(vote(CALLS[s])[rows][y[rows] == 1].mean()), "specificity": r3(1 - vote(CALLS[s])[rows][y[rows] == 0].mean())} for s in ("T", "A", "B", "C")}
    modal = np.array([np.bincount(band[i][band[i] >= 0], minlength=3).argmax() for i in range(n)]); GN = {0: "NDBE", 1: "ID", 2: "LGD"}
    R["modal_band_by_grade_and_class"] = {cl: {GN[g]: {bn: int(((modal == bi) & (GR == g) & (y == (1 if cl == "progressor" else 0)))[rows].sum()) for bi, bn in ((0, "low"), (1, "mid"), (2, "high"))} for g in (0, 1, 2)} for cl in ("progressor", "non_progressor")}
    er = pd.DataFrame({"e": ENDO[rows], "p": pat[rows], "mbf": MBF[rows], "y": y[rows], "dys": D1[rows]}); ep = er.groupby("e").agg(p=("p", "first"), mbf=("mbf", "max")); q = lambda v: {"mean": r3(np.mean(v)), "median": r3(np.median(v)), "range": [r3(np.min(v)), r3(np.max(v))]}
    R["worst_grade_and_sampling_by_status"] = {}
    for st, v in (("progressor", 1), ("non_progressor", 0)):
        d = er[er.y == v]; ps = d.p.unique(); ne = d.groupby("p").e.nunique(); ns = d.groupby("p").size(); sp = ep[ep.p.isin(ps)].groupby("p").mbf.agg(lambda x: (x.max() - x.min()) / 12.0)
        R["worst_grade_and_sampling_by_status"][st] = {"n_patients": int(len(ps)), "patients_with_any_pre_event_ID_or_LGD": int(d.groupby("p").dys.max().sum()), "endoscopies_per_patient": q(ne.values), "samples_per_patient": q(ns.values), "span_years_first_to_last_pre_event_endoscopy": q(sp.values)}
    EIDX = np.full(n, np.nan)
    for p in np.unique(pat):
        idx = np.where((pat == p) & PRE)[0]; em = pd.Series(MBF[idx]).groupby(ENDO[idx]).max().sort_values(ascending=False); rk = {e: i + 1 for i, e in enumerate(em.index)}; EIDX[idx] = [rk[e] for e in ENDO[idx]]
    R["endoscopy_index_auroc"] = {"value": r3(fsa(EIDX, rows)), "ci95": ci2([fsa(EIDX, b) for b in boots]), "note": "index of the pre-event endoscopy within the patient, 1 = earliest, as a raw score"}
    dysp = set(pat[rows][D1[rows]]); nd = np.array([p not in dysp for p in pat]); rows2 = rows[nd[rows]]; boots2 = boots_for(rows2)
    obs = {m: fsa(O[m], rows2) for m in ("L-CNV", "L-IMG", "L-LATE")}; B = frame([{m: fsa(O[m], b) for m in ("L-CNV", "L-IMG", "L-LATE")} for b in boots2])
    o2 = draw_stats(rows2, CALLS, band, O); D2b = frame([draw_stats(b, CALLS, band, O) for b in boots2])
    R["never_dysplastic_patients"] = {"n_samples": int(len(rows2)), "n_progressor_samples": int(y[rows2].sum()), "n_patients": int(len(set(pat[rows2]))), "n_progressor_patients": int(len(set(pat[rows2][y[rows2] == 1]))),
        "auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in obs}, "IMG_minus_CNV": dstat(obs["L-IMG"] - obs["L-CNV"], B["L-IMG"] - B["L-CNV"], one_sided=True),
        "triage_criterion_voted_vs_B": paired(o2, D2b, "B", "v", one_sided=True), "triage_no_vote_vs_B": paired(o2, D2b, "B", "nv", one_sided=True), "triage_no_vote_vs_Bstar": paired(o2, D2b, "B*", "nv", one_sided=True),
        "share_sequenced_no_vote": r3(o2["share_mid"])}
    res["R24"] = R; dump(res)
# ================================================================== R25 random gate (shards of seeds) and merge
elif TASK.startswith("r25_"):
    last = TASK.split("_")[1]; D25 = f"{ROW2}/randgate"; os.makedirs(D25, exist_ok=True); O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]
    if last == "merge":
        files = sorted(glob.glob(f"{D25}/seed_*.json")); assert len(files) == 100, len(files); S = [json.load(open(f)) for f in files]; R = {"n_seeds": len(S), "target_real_no_vote_sensitivity": S[0]["target"]}
        for k in ("share_sequenced", "sensitivity", "specificity"): v = np.array([s[k] for s in S]); R[f"random_gate_{k}"] = {"mean": r3(v.mean()), "p2.5_97.5": ci2(v)}
        w = np.array([s["share_to_reach_target"] if s["share_to_reach_target"] is not None else np.nan for s in S]); R["share_to_reach_real_sensitivity"] = {"mean": r3(np.nanmean(w)), "p2.5_97.5": ci2(w), "n_seeds_not_reached": int(np.isnan(w).sum()), "n_seeds": len(S)}
        R["sensitivity_at_full_sequencing_mean"] = r3(np.mean([s["sens_at_full"] for s in S])); R["real_gate_reference"] = S[0]["real"]; res["R25"] = R; dump(res, "r25")
    else:
        sh = int(last.replace("shard", "")); band0, call0, info0 = triage(O, I, FO, popm); real = {"share_sequenced": float((band0[rows] == 1).mean(0).mean()), "sensitivity": float(call0[rows][y[rows] == 1].mean(0).mean()), "specificity": float(1 - call0[rows][y[rows] == 0].mean(0).mean())}
        for seed in range(sh * 10, sh * 10 + 10):
            outf = f"{D25}/seed_{seed:03d}.json"
            if os.path.exists(outf): continue
            rng = np.random.RandomState(seed); O2 = dict(O); O2["L-IMG"] = rng.uniform(size=(n, 10)); I2 = dict(I); I2["L-IMG"] = {k: pd.Series(rng.uniform(size=len(v)), index=v.index) for k, v in I["L-IMG"].items()}
            band, call, info = triage(O2, I2, FO, popm); sens = call[rows][y[rows] == 1].mean(0); spec = 1 - call[rows][y[rows] == 0].mean(0); share = (band[rows] == 1).mean(0)
            # widening: per repeat x fold, the middle band as a rank interval on the random held-out scores, widened by w ranks on each side
            maxw = 150; SENS = np.zeros((10, maxw + 1)); SHARE = np.zeros((10, maxw + 1))
            for j, r in enumerate(REPS):
                tp = np.zeros(maxw + 1); seq = np.zeros(maxw + 1); npos = (y[rows] == 1).sum()
                for k in range(1, 11):
                    te = rows[FO[rows, j] == k]; s = O2["L-IMG"][te, j]; o = np.argsort(s); te = te[o]; s = s[o]; late = O2["L-LATE"][te, j]; ts = info["t_seq"][j * 10 + k - 1]; tl = info["t_low"][j * 10 + k - 1]; cs = info["c_star"][j * 10 + k - 1]
                    lo0 = int((s < tl).sum()); hi0 = max(int((s <= cs).sum()), lo0); m = len(te); pos = y[te] == 1; lc = late >= ts
                    for w in range(maxw + 1):
                        lo, hi = max(0, lo0 - w), min(m, hi0 + w); c = np.zeros(m, bool); c[hi:] = True; c[lo:hi] = lc[lo:hi]; tp[w] += (c & pos).sum(); seq[w] += hi - lo
                SENS[j] = tp / npos; SHARE[j] = seq / len(rows)
            ms = SENS.mean(0); msh = SHARE.mean(0); hit = np.where(ms >= real["sensitivity"] - 1e-12)[0]
            json.dump({"seed": seed, "share_sequenced": float(share.mean()), "sensitivity": float(sens.mean()), "specificity": float(spec.mean()), "no_mid": info["no_mid"], "fallback": info["fallback"], "target": real["sensitivity"], "real": real,
                       "share_to_reach_target": float(msh[hit[0]]) if len(hit) else None, "w_to_reach_target": int(hit[0]) if len(hit) else None, "sens_at_full": float(ms[-1]), "share_at_full": float(msh[-1]), "share_w0_check": float(msh[0]), "sens_w0_check": float(ms[0])}, open(outf, "w"))
        print("RV2 R25 SHARD DONE", sh, flush=True)
# ================================================================== R26 cut-off transfer and the frozen model in sample
elif TASK == "r26":
    R = {"spread_of_fold_cutoffs": {}, "training_vs_heldout_operating_points": {}}; popm = popmask("pre"); rows = np.where(popm)[0]
    for src in ("their", "pkg"):
        O, I, FO = load(src); band, call, info = triage(O, I, FO, popm); comp, thr = comparators(O, I, FO, popm); mc, sstar = matched_calls(O, I, FO, popm)
        qq = lambda v: {"median": r3(np.median(v)), "iqr": [r3(np.percentile(v, 25)), r3(np.percentile(v, 75))], "range": [r3(min(v)), r3(max(v))]}
        R["spread_of_fold_cutoffs"][src] = {"t_low": qq(info["t_low"]), "c_star": qq(info["c_star"]), "t_seq": qq(info["t_seq"]), "t_B": qq(thr["B"])}
        tr_spec = {"T": sstar, "A": [], "B": [], "C": []}; ho_spec = {s: [] for s in ("T", "A", "B", "C")}; CC = {"T": call, **comp}; i = 0
        for j, r in enumerate(REPS):
            for k in range(1, 11):
                tri = I["L-IMG"][(r, k)].index.values; tri = tri[popm[tri]]; yc = y[tri] == 1; te = rows[(FO[rows, j] == k) & (y[rows] == 0)]
                for a, m in COMPM.items(): tr_spec[a].append(float(1 - (I[m][(r, k)].reindex(tri).values[~yc] >= thr[a][i]).mean()))
                for s in CC: ho_spec[s].append(float(1 - CC[s][te, j].mean()) if len(te) else np.nan)
                i += 1
        R["training_vs_heldout_operating_points"][src] = {s: {"training_specificity": qq(tr_spec[s]), "heldout_specificity": qq([v for v in ho_spec[s] if np.isfinite(v)]), "heldout_minus_training": qq([h - t for h, t in zip(ho_spec[s], tr_spec[s]) if np.isfinite(h)])} for s in CC}
        if src == "their": lam_cnv = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{M}/cv/preds/cfg_00_rep_*.csv"))]).groupby(["rep", "fold"])["lambda"].first().values; lam_img = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{M}/cv/preds/cfg_02_rep_*.csv"))]).groupby(["rep", "fold"])["lambda"].first().values
    FZD = M + "/checks/frozen"; cf_c = pd.read_csv(FZD + "/coef_L_CNV.csv"); cf_i = pd.read_csv(FZD + "/coef_L_IMG.csv"); mi = json.load(open(FZD + "/model_info.json"))
    CN = pd.read_csv(M + "/cnv_their_C.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(ids); assert list(CN.columns) == list(cf_c.feature[1:]), "feature order"
    IM = pd.read_csv(M + "/img_mean.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); sc = pd.read_csv(FZD + "/image_scaling.csv").set_index("feature").reindex(IM.columns); assert list(IM.columns) == list(cf_i.feature[1:])
    IMz = (IM.values - sc["mean"].values) / np.where(sc["sd"].values > 0, sc["sd"].values, 1.0)
    sig = lambda z: 1 / (1 + np.exp(-z)); p_c = sig(cf_c.coefficient.values[0] + CN.values @ cf_c.coefficient.values[1:]); p_i = sig(cf_i.coefficient.values[0] + IMz @ cf_i.coefficient.values[1:]); p_l = (p_c + p_i) / 2
    chk = {"auroc_L_CNV_676": r3(auc_plain(y, p_c)), "auroc_L_IMG_676": r3(auc_plain(y, p_i)), "model_info": {k: r3(mi["insample_check"][k]) for k in ("auroc_L_CNV", "auroc_L_IMG")}, "mean_prob_L_CNV": r3(p_c.mean()), "mean_prob_L_IMG": r3(p_i.mean()), "model_info_mean": [r3(mi["L_CNV"]["insample_mean_prob"]), r3(mi["L_IMG"]["insample_mean_prob"])]}
    FC = json.load(open(T + "/results/paper_final/triage_robustness/frozen_cutoffs_their.json"))["cutoffs"]; img, late = p_i[rows], p_l[rows]; yr = y[rows]
    bandf = np.where(img < FC["t_low"], 0, np.where(img > FC["t_high"], 2, 1)); callf = np.where(bandf == 1, late >= FC["t_seq"], bandf == 2); bcall = late >= FC["t_B"]
    R["frozen_model_in_sample"] = {"label": "IN-SAMPLE (frozen bundle fitted on all 676 samples; not a performance estimate)", "cutoffs": {k: r3(v) for k, v in FC.items()}, "reproduction_check": chk,
        "band_shares_571": {nm: r3((bandf == g).mean()) for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}, "band_shares_by_class": {cl: {nm: r3((bandf[yr == v] == g).mean()) for g, nm in ((0, "low"), (1, "mid"), (2, "high"))} for cl, v in (("progressor", 1), ("non_progressor", 0))},
        "triage_in_sample": {"sensitivity": r3(callf[yr == 1].mean()), "specificity": r3(1 - callf[yr == 0].mean())}, "B_in_sample": {"sensitivity": r3(bcall[yr == 1].mean()), "specificity": r3(1 - bcall[yr == 0].mean())},
        "nonzero_coefficients": {"frozen_L_CNV": int((cf_c.coefficient.values[1:] != 0).sum()), "frozen_L_IMG": int((cf_i.coefficient.values[1:] != 0).sum()), "fold_models": "coefficient counts of the 100 fold models are not stored (cv/preds keeps lambda only); lambda ranges given instead",
                                 "lambda": {"frozen_L_CNV": r3(mi["L_CNV"]["lambda"]), "fold_L_CNV_range": [r3(lam_cnv.min()), r3(lam_cnv.max())], "frozen_L_IMG": r3(mi["L_IMG"]["lambda"]), "fold_L_IMG_range": [r3(lam_img.min()), r3(lam_img.max())]}}}
    res["R26"] = R; dump(res)
# ================================================================== R27 pairs and time-dependent AUROC without within-patient pairs
elif TASK.startswith("r27_"):
    pop = TASK[4:]; O, I, FO = load("their"); popm = popmask(pop); rows = np.where(popm)[0]; R = {}
    pr = []
    for j in range(10):
        f = FO[rows, j]; yr = y[rows]; ns = sum(int((yr[f == k] == 1).sum()) * int((yr[f == k] == 0).sum()) for k in range(1, 11)); lack = sum(1 for k in range(1, 11) if (yr[f == k] == 1).sum() == 0 or (yr[f == k] == 0).sum() == 0)
        pf = pd.DataFrame({"p": pat[rows], "f": f, "y": yr}).groupby("p").first(); npair = sum(int((pf.y[pf.f == k] == 1).sum()) * int((pf.y[pf.f == k] == 0).sum()) for k in range(1, 11))
        pr.append({"repeat": j + 1, "sample_level_within_fold_pairs": ns, "patient_level_within_fold_pairs": npair, "fold_cells_lacking_one_class": lack})
    R["pairs"] = {"per_repeat": pr, "sample_pairs_mean": r3(np.mean([x["sample_level_within_fold_pairs"] for x in pr])), "patient_pairs_mean": r3(np.mean([x["patient_level_within_fold_pairs"] for x in pr])), "cells_lacking_total": int(sum(x["fold_cells_lacking_one_class"] for x in pr)),
                  "all_sample_pairs_without_folds": int(y[rows].sum() * (len(rows) - y[rows].sum())), "n_samples": int(len(rows)), "n_patients": int(len(set(pat[rows])))}
    def Gminus(tt, dd, at):
        u = np.unique(tt); nr = (tt[None, :] >= u[:, None]).sum(1); dc = ((tt[None, :] == u[:, None]) & (dd[None, :] == 0)).sum(1); sv = np.cumprod(1 - dc / nr); pv = np.concatenate([[1.0], sv[:-1]])
        k = np.searchsorted(u, at, side="left"); o = np.ones(len(at)); i = k < len(u); o[i] = pv[k[i]]; o[~i] = sv[-1]; return np.clip(o, 1e-12, None)
    def setup(b, t):
        tt, dd = TIME[b], EV[b]; case = (dd == 1) & (tt <= t); ctrl = ((dd == 0) & (tt >= t)) | ((dd == 1) & (tt > t))
        if not case.sum() or not ctrl.sum(): return None
        return {"b": b, "ci": np.where(case)[0], "co": np.where(ctrl)[0], "w": 1 / Gminus(tt, dd, tt[case])}
    def conc(keyc, keya, s_co, s_ci):
        kc = np.sort(keyc * 4.0 + s_co); base = np.searchsorted(kc, keya * 4.0, "left"); lo = np.searchsorted(kc, keya * 4.0 + s_ci, "left") - base; hi = np.searchsorted(kc, keya * 4.0 + s_ci, "right") - base; nct = np.searchsorted(kc, keya * 4.0 + 2.0, "left") - base
        return lo + 0.5 * (hi - lo), nct
    def td(Sm, St, weighted, excl):
        b, ci, co = St["b"], St["ci"], St["co"]; w = St["w"] if weighted else np.ones(len(ci)); pc = PCODE[b]; out = []
        for j in range(10):
            s = Sm[b, j]; f = FO[b, j]; c1, n1 = conc(f[co], f[ci], s[co], s[ci])
            if excl: c2, n2 = conc(pc[co], pc[ci], s[co], s[ci]); c1 = c1 - c2; n1 = n1 - n2
            den = (w * n1).sum(); out.append((w * c1).sum() / den if den > 0 else np.nan)
        return float(np.nanmean(out))
    ARMS = ["L-CNV", "L-IMG", "L-LATE"]; up = np.unique(pat[rows]); ro = {q: rows[pat[rows] == q] for q in up}; rng = np.random.RandomState(0); draws = [np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))]) for _ in range(NB)]
    R["time_dependent_excluding_within_patient_pairs"] = {}
    for t in (1.0, 3.0, 5.0):
        S0 = setup(rows, t); H = {"n_cases": int(len(S0["ci"])), "n_controls": int(len(S0["co"]))}; valid = [s for s in (setup(b, t) for b in draws) if s is not None]; H["valid_draws"] = len(valid)
        f0 = FO[rows, 0]; H["within_fold_pairs_repeat1"] = int(sum(((f0[S0["ci"]] == k).sum()) * ((f0[S0["co"]] == k).sum()) for k in range(1, 11))); pc0 = PCODE[rows]; H["within_patient_pairs_removed"] = int(sum(((pc0[S0["ci"]] == q).sum()) * ((pc0[S0["co"]] == q).sum()) for q in np.unique(pc0)))
        for wtd, key in ((True, "ipcw"), (False, "unweighted")):
            for excl, ek in ((True, "excluding_within_patient"), (False, "all_within_fold_pairs_reference")):
                obs = {a: td(O[a], S0, wtd, excl) for a in ARMS}; B = frame([{a: td(O[a], s, wtd, excl) for a in ARMS} for s in valid])
                H[f"{key}_{ek}"] = {a: {"auroc": r3(obs[a]), "ci95": ci2(B[a])} for a in ARMS}; H[f"{key}_{ek}"]["IMG_minus_CNV"] = dstat(obs["L-IMG"] - obs["L-CNV"], B["L-IMG"] - B["L-CNV"], one_sided=True); H[f"{key}_{ek}"]["LATE_minus_CNV"] = dstat(obs["L-LATE"] - obs["L-CNV"], B["L-LATE"] - B["L-CNV"])
        R["time_dependent_excluding_within_patient_pairs"][str(int(t))] = H; print("R27", pop, t, flush=True)
    res.update({"pop": pop, "R27": R}); dump(res)
# ================================================================== R28 H-optimus-0 downstream predictions
elif TASK == "r28":
    O, I, FO = load("their"); files = sorted(glob.glob(f"{M}/scanner_robustness/cv/cv_*_hoptimus0.csv")); assert len(files) == 10, files
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files]); HI = D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values; FH = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int)
    assert (FH == FO).all() and np.isfinite(HI).all(); HL = (O["L-CNV"] + HI) / 2; R = {"source": "killcoyne_mm/scanner_robustness/cv/cv_<rep>_hoptimus0.csv (docs/paper_scanner_robustness.md downstream run D1)", "folds_equal_stored": True}
    for pop in ("pre", "pre_ndbe"):
        popm = popmask(pop); rows = np.where(popm)[0]; boots = boots_for(rows); SC = {"L-CNV": O["L-CNV"], "L-IMG_UNI2h": O["L-IMG"], "L-IMG_Hoptimus0": HI, "L-LATE_UNI2h": O["L-LATE"], "L-LATE_Hoptimus0": HL}
        obs = {m: fsa(S, rows) for m, S in SC.items()}; B = frame([{m: fsa(S, b) for m, S in SC.items()} for b in boots])
        R[pop] = {"n_samples": int(len(rows)), "auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in SC},
                  "IMG_Hoptimus0_minus_CNV": dstat(obs["L-IMG_Hoptimus0"] - obs["L-CNV"], B["L-IMG_Hoptimus0"] - B["L-CNV"], one_sided=True), "IMG_UNI2h_minus_CNV_reference": dstat(obs["L-IMG_UNI2h"] - obs["L-CNV"], B["L-IMG_UNI2h"] - B["L-CNV"], one_sided=True),
                  "LATE_Hoptimus0_minus_CNV": dstat(obs["L-LATE_Hoptimus0"] - obs["L-CNV"], B["L-LATE_Hoptimus0"] - B["L-CNV"]), "IMG_Hoptimus0_minus_IMG_UNI2h": dstat(obs["L-IMG_Hoptimus0"] - obs["L-IMG_UNI2h"], B["L-IMG_Hoptimus0"] - B["L-IMG_UNI2h"])}
    res["R28"] = R; dump(res)
# ================================================================== R29 likelihood ratios and natural frequencies
elif TASK == "r29":
    rows, O, I, FO, CALLS, band, info, sstar, obs, D, boots = r14_core("their", "pre"); R = {"class_wise_band_rates": {}, "likelihood_ratios": {}, "natural_frequencies_per_1000": {}}; BN = {0: "low", 1: "mid", 2: "high"}
    for g, nm in BN.items():
        R["class_wise_band_rates"][nm] = {"progressor": {"value": r3(obs[f"rate_P_{g}"]), "ci95": ci2(D[f"rate_P_{g}"])}, "non_progressor": {"value": r3(obs[f"rate_N_{g}"]), "ci95": ci2(D[f"rate_N_{g}"])}}
        R["likelihood_ratios"][nm] = {"value": r3(obs[f"rate_P_{g}"] / obs[f"rate_N_{g}"]), "ci95": ci2(D[f"rate_P_{g}"] / D[f"rate_N_{g}"])}
    def nf(o, pi, mode):
        se, sp = o[f"{mode}_sens_T"], o[f"{mode}_spec_T"]; seq = pi * o["rate_P_1"] + (1 - pi) * o["rate_N_1"] if mode == "nv" else pi * o["share_pivotal_P"] + (1 - pi) * o["share_pivotal_N"]
        return {"sequenced": 1000 * seq, "progressor_biopsies_caught": 1000 * pi * se, "progressor_biopsies_missed": 1000 * pi * (1 - se), "false_flags": 1000 * (1 - pi) * (1 - sp), "sequenced_any_repeat": 1000 * (pi * o["share_mid_any_P"] + (1 - pi) * o["share_mid_any_N"])}
    for pi in (0.02, 0.05):
        R["natural_frequencies_per_1000"][f"{pi:.2f}"] = {}
        for mode, m in (("nv", "no_vote"), ("v", "voted")):
            o = nf(obs, pi, mode); Bv = frame([nf(dict(r), pi, mode) for _, r in D.iterrows()])
            R["natural_frequencies_per_1000"][f"{pi:.2f}"][m] = {k: {"value": round(float(v), 1), "ci95": [round(float(np.percentile(Bv[k], 2.5)), 1), round(float(np.percentile(Bv[k], 97.5)), 1)]} for k, v in o.items()}
    R["note"] = "no_vote: sequenced = middle-band share at the prevalence (per-repeat mean rates); voted: sequenced = vote-pivotal share (copy number needed for the >= 6-of-10 call), sequenced_any_repeat = middle band in >= 1 repeat"
    R["sanity_round12"] = {"PPV_high_band_0.05": r3(0.05 * obs["rate_P_2"] / (0.05 * obs["rate_P_2"] + 0.95 * obs["rate_N_2"]))}; res["R29"] = R; dump(res)
# ================================================================== R30 lead time
elif TASK == "r30":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm); R = {}
    pr = rows[y[rows] == 1]; up = np.unique(pat[pr]); ip = {q: pr[pat[pr] == q] for q in up}
    def lead(c_full):   # per progressor patient: time (years) from the first positive pre-event endoscopy to the first HGD/IMC; None if not caught
        return {q: (float(TIME[ip[q]][c_full[ip[q]]].max()) if c_full[ip[q]].any() else None) for q in up}
    q = lambda v: {"n": int(len(v)), "median": r3(np.median(v)) if len(v) else None, "iqr": [r3(np.percentile(v, 25)), r3(np.percentile(v, 75))] if len(v) else None, "mean": r3(np.mean(v)) if len(v) else None}
    R["n_progressor_patients_with_pre_event_samples"] = int(len(up)); R["voted"] = {}
    LV = {s: lead(vote(CALLS[s])) for s in ("T", "B", "C", "B*", "C*")}
    for s, L in LV.items(): R["voted"][s] = q([v for v in L.values() if v is not None])
    both = [p for p in up if LV["T"][p] is not None and LV["C"][p] is not None]; R["voted"]["paired_T_minus_C_caught_by_both"] = q([LV["T"][p] - LV["C"][p] for p in both])
    bothB = [p for p in up if LV["T"][p] is not None and LV["B"][p] is not None]; R["voted"]["paired_T_minus_B_caught_by_both"] = q([LV["T"][p] - LV["B"][p] for p in bothB])
    R["no_vote_mean_over_repeats"] = {}
    for s in ("T", "B", "C"):
        st = [q([v for v in lead(CALLS[s][:, j]).values() if v is not None]) for j in range(10)]
        R["no_vote_mean_over_repeats"][s] = {"n_caught_mean": r3(np.mean([x["n"] for x in st])), "median_mean": r3(np.mean([x["median"] for x in st])), "iqr_mean": [r3(np.mean([x["iqr"][0] for x in st])), r3(np.mean([x["iqr"][1] for x in st]))], "mean_mean": r3(np.mean([x["mean"] for x in st]))}
    pd_ = []
    for j in range(10):
        LT, LC = lead(CALLS["T"][:, j]), lead(CALLS["C"][:, j]); d = [LT[p] - LC[p] for p in up if LT[p] is not None and LC[p] is not None]; pd_.append(q(d))
    R["no_vote_mean_over_repeats"]["paired_T_minus_C_caught_by_both"] = {"n_mean": r3(np.mean([x["n"] for x in pd_])), "median_mean": r3(np.mean([x["median"] for x in pd_])), "mean_mean": r3(np.mean([x["mean"] for x in pd_]))}
    R["note"] = "lead time = TIME of the earliest positive pre-event sample (years before the first HGD/IMC); patients counted once"; res["R30"] = R; dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
