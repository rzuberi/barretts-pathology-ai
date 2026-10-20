"""Review round 7, items R42-R57 (docs/paper_review_round7_analyses.md @ 724ff97). Stored predictions (tb_common.py loaders and the paper_triage.md
rules), the analysed-sample table, Killcoyne's sample sheets, the Barrett's database export, slide headers and pixels at the stored tile positions,
and the round-7 refits (rv7_refit.R, killcoyne_mm/review_round7/refit/). The rv6_round6.py helpers are copied here unchanged unless stated.
Aggregates -> results/paper_final/review_round7/<TASK>.json (no row-level data, no identifiers, no dates); row-level files only under
killcoyne_mm/review_round7/ on the cluster. R56 is an audit: the slide model is not scored on the validation cohort. R58 is registered only.
TASK = prep | r42 | r44 | r45 | r46 | r47 | r48 | r49 | r50 | r51 | r53 | r54db | refit_<var> | check | r55px_<chunk> | r55 | r56px_<chunk> | r56 | r57
       <var> r43a | r43b | r43c | r52a | r52b | r54"""
import os, sys, json, glob, re, itertools, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, binom, beta as beta_dist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_common as TC
from tb_common import *
from tb_common import _outer, _inner
PRESPEC = "724ff97"; OUT7 = T + "/results/paper_final/review_round7"; OUT6 = T + "/results/paper_final/review_round6"; OUT4 = T + "/results/paper_final/review_round4"
R7 = M + "/review_round7"; SETS = R7 + "/sets"; KD = T + "/feasibility/paper_plan/killcoyne"; PP = T + "/feasibility/paper_plan"
SWG = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; DBX = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; LOOK = "/mnt/scratche/slow/fmlab/zuberi01/analysis/clinical_lookup"
REL = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
if os.environ.get("RV7_NB"):   # smoke mode: fewer bootstrap draws, aggregates written aside (never to results/)
    NB = int(os.environ["RV7_NB"]); TC.NB = NB; TC.boots_for.__defaults__ = (0, NB); OUT7 = R7 + "/smoke_json"
for d in (OUT7, R7, SETS): os.makedirs(d, exist_ok=True)
TASK = os.environ["TASK"]
CS = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample").reindex(ids); YEAR = pd.to_numeric(CS.year, errors="coerce").values; P53 = pd.to_numeric(CS["P53 IHC"], errors="coerce").values
GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); STUDY = SM.study_number.astype(str).values
SDS = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); TILES = SDS.tissue_tiles.values.astype(float)
TIME = SM.time.values.astype(float); MBF = SM.mbf.values.astype(float)
FO0 = load("their")[2]; MM = {"A*": "L-CNV", "B*": "L-LATE", "C*": "L-IMG"}; MODELS = ("L-CNV", "L-IMG", "L-LATE")
remap = lambda p: p.replace("/scratchc/", "/mnt/scratche/fast/") if isinstance(p, str) and p.startswith("/scratchc/") else p
# ------------------------------------------------------------------ helpers copied from rv6_round6.py (unchanged)
def fsa(S, b, mask=None, w=None, cell=None, reps=None):
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
def cp_ci(k, nn):
    if nn == 0: return [None, None]
    lo = beta_dist.ppf(0.025, k, nn - k + 1) if k > 0 else 0.0; hi = beta_dist.ppf(0.975, k + 1, nn - k) if k < nn else 1.0; return [r3(lo), r3(hi)]
def dump(res, name=None):
    res["prespec"] = PRESPEC; json.dump(res, open(f"{OUT7}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV7 DONE", name or TASK, flush=True)
def J(path):
    try: return json.load(open(path))
    except FileNotFoundError: return None
def sens_at_spec(x, yb, spec_t):
    xc = np.sort(x[yb == 0]); q = int(np.ceil(spec_t * len(xc))); thr = xc[q - 1] if q > 0 else -np.inf; return float(np.mean(x[yb == 1] > thr))
def wmean(v, w): return float(v.mean()) if w is None else float((v * w).sum() / w.sum())
def frame(BS): return pd.DataFrame(BS)
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
def draw_stats(b, CALLS, band, O, per_rep=False, w=None):
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
def boots_tagged(rows, seed=0, nb=NB):
    pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(seed); out = []
    while len(out) < nb:
        dr = rng.choice(len(up), len(up)); b = np.concatenate([ro[up[i]] for i in dr]); t = np.concatenate([np.full(len(ro[up[i]]), q) for q, i in enumerate(dr)])
        if 0 < y[b].sum() < len(b): out.append((b, t))
    return out
def later_hgd_patients():
    LT = pd.read_csv(PP + "/later_disease_patient.csv", dtype={"patient_id": str}).set_index("patient_id")
    laterH = ((LT.db_max_grade_after >= 3) | (LT.release_excluded_max_grade >= 3) | (LT.slidematch_max_grade_after >= 3) | (LT.hgd_table_entries_after > 0))
    npp = sorted(set(pat[(y == 0)])); st = {p: STUDY[pat == p][0] for p in npp}
    matched = {p: s for p, s in st.items() if s in LT.index}; return [p for p, s in matched.items() if bool(laterH[s])]
def sheet():
    fd = pd.read_csv(SWG + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; return fd
# ------------------------------------------------------------------ round-7 helpers
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
    globals()["FO0"] = FO; return O, I, FO
def modal_band(band): return np.array([np.bincount(r_[r_ >= 0], minlength=3).argmax() if (r_ >= 0).any() else -1 for r_ in band])
def cutoff_arrays(info, FO):
    """per sample x repeat: t_low, c_star and t_seq of the sample's (repeat, fold) in tb_common.triage order (repeat-major, fold 1..10)"""
    TL = np.full((n, 10), np.nan); CSt = np.full((n, 10), np.nan); TS = np.full((n, 10), np.nan)
    for j in range(10):
        for k in range(1, 11):
            i = j * 10 + (k - 1); te = FO[:, j] == k; TL[te, j] = info["t_low"][i]; CSt[te, j] = info["c_star"][i]; TS[te, j] = info["t_seq"][i]
    return TL, CSt, TS
def level_cols(C, b, t=None):
    """single model (C n x 10, or n): sample sens/spec, endoscopy sens/spec (endoscopy positive if any sample flagged), progressor patients missed and
    non-progressor patients flagged (patient positive if any sample flagged), each a mean over the 10 repeats; t = bootstrap tag for resampled copies"""
    C = C if np.ndim(C) == 2 else np.repeat(np.asarray(C, bool)[:, None], 10, 1); t = np.zeros(len(b), int) if t is None else t; yb = y[b]; Cb = C[b]
    ek = pd.Series(ENDO[b]).astype(str).values + "#" + t.astype(str); pk = pd.Series(pat[b]).astype(str).values + "#" + t.astype(str)
    E = pd.DataFrame(Cb).groupby(ek).max(); le = pd.Series(yb).groupby(ek).max().reindex(E.index).values
    Pm = pd.DataFrame(Cb).groupby(pk).max(); lp = pd.Series(yb).groupby(pk).max().reindex(Pm.index).values
    return {"sens": float(Cb[yb == 1].mean()), "spec": float(1 - Cb[yb == 0].mean()), "endo_sens": float(E.values[le == 1].mean()), "endo_spec": float(1 - E.values[le == 0].mean()),
            "pat_missed": float((~Pm.values[lp == 1]).sum(0).mean()), "pat_flagged": float(Pm.values[lp == 0].sum(0).mean()), "n_endo": int(len(E)), "n_endo_P": int((le == 1).sum()), "n_pat_P": int((lp == 1).sum()), "n_pat_N": int((lp == 0).sum())}
def table1(rules, rows, shares, boots=None, pairs=()):
    """Table 1 style rows: share sequenced, sample sens / spec (patient-bootstrap CI), endoscopy sens / spec, patients missed / flagged; paired deltas"""
    obs = {k: level_cols(c, rows) for k, c in rules.items()}; out = {"rules": {}}
    BT = boots_tagged(rows) if boots is None else boots; DR = [{k: level_cols(c, b, t) for k, c in rules.items()} for b, t in BT]
    for k in rules:
        o = obs[k]; out["rules"][k] = {"share_sequenced": shares.get(k), "sensitivity": r3(o["sens"]), "sens_ci95": ci2([d[k]["sens"] for d in DR]), "specificity": r3(o["spec"]), "spec_ci95": ci2([d[k]["spec"] for d in DR]),
                                       "endoscopy_sensitivity": r3(o["endo_sens"]), "endoscopy_sens_ci95": ci2([d[k]["endo_sens"] for d in DR]), "endoscopy_specificity": r3(o["endo_spec"]), "endoscopy_spec_ci95": ci2([d[k]["endo_spec"] for d in DR]),
                                       "progressor_patients_missed": round(o["pat_missed"], 2), "non_progressor_patients_flagged": round(o["pat_flagged"], 2)}
    out["n"] = {**nfo(rows), "n_endoscopies": obs[list(rules)[0]]["n_endo"], "n_progressor_endoscopies": obs[list(rules)[0]]["n_endo_P"]}; out["draws"] = len(DR); out["paired"] = {}
    for a, b_ in pairs:
        out["paired"][f"{a}_minus_{b_}"] = {lv: {"d_sens": dstat(obs[a][f"{pre}sens"] - obs[b_][f"{pre}sens"], [d[a][f"{pre}sens"] - d[b_][f"{pre}sens"] for d in DR], one_sided=True),
                                                 "d_spec": dstat(obs[a][f"{pre}spec"] - obs[b_][f"{pre}spec"], [d[a][f"{pre}spec"] - d[b_][f"{pre}spec"] for d in DR])} for lv, pre in (("sample", ""), ("endoscopy", "endo_"))}
    return out
def tri_block(R, CALLS, band, rows):
    """compact claim-T/claim-H summary used for the refit and re-scoring rows (single model)"""
    ct = R["claim_T"]; o = {k: R[k] for k in ("n_samples", "n_progressor_samples", "n_patients", "n_progressor_patients")}
    o["triage"] = {"sensitivity": ct["triage_no_vote"]["sensitivity"], "specificity": ct["triage_no_vote"]["specificity"], "share_sequenced": ct["share_sequenced"]["per_repeat_mean"]}
    for ref, key in (("Bstar", "no_vote_vs_Bstar_training_matched"), ("B", "no_vote_vs_B")):
        p = ct[key]; o[f"vs_{ref}"] = {"d_sens": p["d_sens_triage_minus"]["delta"], "ci95": p["d_sens_triage_minus"]["ci95"], "one_sided_lower": p["d_sens_triage_minus"]["one_sided_95_lower"], "within_-0.05": p["d_sens_triage_minus"]["non_inferior_at_-0.05"],
                                       "d_spec": p["d_spec_triage_minus"]["delta"], "d_spec_ci95": p["d_spec_triage_minus"]["ci95"], "comparator_sens_spec": [p["comparator_sensitivity"], p["comparator_specificity"]]}
    if "claim_H" in R:
        h = R["claim_H"]; o["IMG_minus_CNV"] = {"delta": h["IMG_minus_CNV"]["delta"], "ci95": h["IMG_minus_CNV"]["ci95"], "one_sided_lower": h["IMG_minus_CNV"]["one_sided_95_lower"], "within_-0.05": h["IMG_minus_CNV"]["non_inferior_at_-0.05"]}
        o["auroc"] = {m: h["auroc"][m] for m in MODELS}
    return o
PCALLS = None
def primary():
    global PCALLS
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; PCALLS = all_calls(O, I, FO, popm); CALLS, band, info, thr, sstar = PCALLS; return O, I, FO, popm, rows, CALLS, band, info
# ------------------------------------------------------------------ Barrett's database linkage (copied from clinical_lookup/r2_main.py, read-only)
def db_link():
    fd = sheet(); fd = fd[fd.combined_name.notna()].copy(); fd["Patient"] = fd.Patient.astype(str)
    pe = pd.read_csv(REL + "/pre_event_cohort.csv", dtype=str); pid_of = pe.dropna(subset=["participant_id"]).groupby("PatientID_real").participant_id.agg(lambda s: s.str.replace(r"\.0$", "", regex=True).mode().iloc[0])
    IH = pd.read_parquet(DBX + "/initial_history.parquet"); IH["pid"] = IH.participant_id.astype(str).str.replace(r"\.0$", "", regex=True)
    dm = pd.read_csv(SWG + "/Demographics_full.csv", dtype=str)
    for c_ in ("Study Number", "Alternate Study Number"): dm[c_] = dm[c_].fillna("").str.strip()
    dmi = {}
    for _, r in dm.iterrows():
        for c_ in ("Study Number", "Alternate Study Number"):
            if r[c_] and r[c_] not in dmi: dmi[r[c_]] = r
    keys = {}
    for p, g in fd.groupby("Patient"):
        ks = []
        for c_ in ("Hospital_Research_ID_updated", "Hospital Research ID"):
            m_ = g[c_].dropna().str.strip(); m_ = m_[m_ != ""]
            if len(m_): ks.append(m_.mode().iloc[0])
        keys[p] = list(dict.fromkeys(ks))
    link = {}
    for p, ks in keys.items():
        pid = None
        for k in ks:
            if k in pid_of.index: pid = pid_of[k]; break
        if pid is None:
            d = next((dmi[k] for k in ks if k in dmi), None); ks2 = ks + ([d["Study Number"], d["Alternate Study Number"]] if d is not None else [])
            for k in ks2:
                if k and k in pid_of.index: pid = pid_of[k]; break
        if pid is None:
            for col in ("hospitalnumber", "patient_anonref"):
                m_ = IH[IH[col].astype(str).str.strip().isin(set(ks))]
                if len(m_): pid = m_.pid.iloc[0]; break
        link[p] = pid
    pt = pd.read_csv(DBX + "/pathology_text_normalised_full.csv", dtype=str, usecols=["participant_id", "receiveddatetime", "highestgradedysconf"])
    pt["pid"] = pt.participant_id.astype(str).str.replace(r"\.0$", "", regex=True); pt["d"] = pd.to_datetime(pt.receiveddatetime, errors="coerce")
    RK = {2.0: 0, 3.0: 1, 4.0: 2, 5.0: 3, 8.0: 4, 6.0: 5}; pt["g"] = pd.to_numeric(pt.highestgradedysconf, errors="coerce").map(lambda x: RK.get(x, np.nan))
    HG = pd.read_parquet(DBX + "/hgd_pathology_table.parquet"); HG["pid"] = HG.participant_id.astype(str).str.replace(r"\.0$", "", regex=True); HG["d"] = pd.to_datetime(HG.receiveddatetime, dayfirst=True, errors="coerce")
    # exact sample dates: pre_event_cohort.Date by CNV folder name; slide-table date (year only) as fallback, as r2_main.py
    pe["cnv"] = pe.CNVAbsPath.map(lambda p: os.path.basename(str(p))); pe["d"] = pd.to_datetime(pe.Date, errors="coerce"); pdate = pe.dropna(subset=["d"]).drop_duplicates("cnv").set_index("cnv").d
    mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str); smf = pd.read_csv(SWG + "/slide_matching.csv", dtype=str).drop_duplicates("Slide file").set_index("Slide file")
    mp = mp[mp.cnv.notna()].join(smf, on="base"); sdate_slide = pd.to_datetime(mp.EndoscopyDate, errors="coerce").groupby(mp.cnv).max()
    fd["sd"] = fd.combined_name.map(pdate).fillna(fd.combined_name.map(sdate_slide))
    return {"fd": fd, "link": link, "pt": pt, "HG": HG, "pdate": pdate, "sd_of": fd.drop_duplicates("combined_name").set_index("combined_name").sd}
def later_db(L, W=14):
    """per analysed patient: highest DB confirmed grade (0 NDBE .. 5 OAC) and HGD-register entries more than W days after the last sequenced endoscopy"""
    fd, pt, HG = L["fd"], L["pt"], L["HG"]; last = fd.groupby("Patient").sd.max(); out = {}
    for p in sorted(set(pat)):
        pid = L["link"].get(p); lp = last.get(p, pd.NaT)
        if pid is None or pd.isna(lp): out[p] = {"linked": pid is not None, "max_after": np.nan, "hgdtab_after": 0}; continue
        a = pt[(pt.pid == pid) & (pt.d > lp + pd.Timedelta(days=W))]; h = HG[(HG.pid == pid) & (HG.d > lp + pd.Timedelta(days=W))]
        out[p] = {"linked": True, "max_after": a.g.max() if a.g.notna().any() else np.nan, "hgdtab_after": int(len(h))}
    return pd.DataFrame(out).T
res = {"task": TASK}
# ================================================================== prep: variant sets for the refits (row-level -> killcoyne_mm/review_round7/sets/)
if TASK == "prep":
    C0 = pd.read_csv(M + "/set_C.csv", dtype=str); assert (C0.Sample.values == ids).all(), "set C and samples.csv order differ"
    base = pd.DataFrame({"Sample": ids, "Patient": pat, "y": y, "pre": PRE, "ndbe": NDBE, "Pathology": SM.Pathology.values, "mbf": MBF, "time": TIME, "Endoscopy": C0.Endoscopy.values})
    VS = {"base": base.copy()}; info = {}
    # R43: the R38 subsets, removed from training and evaluation
    VS["r43a"] = base[~((base.y == 0) & (base.mbf < 12))].copy(); VS["r43b"] = base[~((base.y == 0) & (base.mbf < 24))].copy()
    lh = later_hgd_patients(); VS["r43c"] = base[~base.Patient.isin(lh)].copy(); info["r43c_patients_removed"] = len(lh)
    # R52: the linked samples of the 8 progressors (clinical_lookup/a2_candidate_map_8pt.csv), rows from kr_samples.csv, hz_prep.py event logic
    a2 = pd.read_csv(LOOK + "/a2_candidate_map_8pt.csv", dtype=str); fd = sheet(); fdu = fd.drop_duplicates("combined_name").set_index("combined_name")
    A = pd.read_csv(KD + "/kr_samples.csv", dtype=str).set_index("Sample"); A["mbf"] = pd.to_numeric(A.index.map(fdu["Months before final"]), errors="coerce"); A["ogd"] = A.index.map(fdu["Path_class_per_OGD"])
    hg = A[(A.Pathology.isin(["HGD", "IMC"])) | (A.ogd.isin(["HGD", "IMC", "HGD/IMC"]))]; tev = hg.groupby("Patient").mbf.max()
    tiers = {"r52a": ["T1"], "r52b": ["T1", "T2b"]}; a2["t"] = a2.tier.str.split(" ").str[0]
    for v, tl in tiers.items():
        add = a2[a2.t.isin(tl)].copy(); assert add.Sample.isin(A.index).all() and not add.Sample.isin(ids).any() and (add.uni2_npz_all == "True").all()
        Ad = A.loc[sorted(add.Sample)].reset_index(); assert (Ad.Status == "P").all() and Ad.mbf.notna().all()
        fb = [p for p in Ad.Patient.unique() if p not in tev.index]; tv = Ad.Patient.map(lambda p: tev.get(p, 0.0)).astype(float)
        new = pd.DataFrame({"Sample": Ad.Sample, "Patient": Ad.Patient, "y": 1, "pre": Ad.mbf.values > tv.values, "ndbe": Ad.Pathology.values == "NDBE", "Pathology": Ad.Pathology, "mbf": Ad.mbf, "time": (Ad.mbf - tv) / 12.0, "Endoscopy": Ad.Endoscopy})
        assert not new.Patient.isin(set(pat)).any(); VS[v] = pd.concat([base, new], ignore_index=True)
        info[v] = {"added_samples": int(len(new)), "added_pre_event_samples": int(new.pre.sum()), "added_patients": int(new.Patient.nunique()), "added_pre_event_patients": int(new[new.pre].Patient.nunique()), "fallback_endpoint_patients": len(fb),
                   "added_grade_pre_event": new[new.pre].Pathology.value_counts().to_dict()}
    addall = sorted(set(VS["r52b"].Sample) - set(ids)); npz = a2.set_index("Sample").npz
    rows_ = []
    for s in addall:
        bags = []
        for f in str(npz[s]).split("|"):
            with np.load(f, allow_pickle=False) as z: bags.append(np.asarray(z["embeddings"], dtype=np.float32))
        b = np.concatenate(bags, 0); assert b.ndim == 2 and b.shape[1] == 1536 and np.isfinite(b).all(); rows_.append(b.mean(0))
    XA = pd.DataFrame(np.vstack(rows_), columns=[f"u{i}" for i in range(1536)]); XA.insert(0, "Sample", addall); XA.to_csv(SETS + "/img_added.csv", index=False)
    TF = pd.read_csv(KD + "/their_features.csv", dtype={"Sample": str}).set_index("Sample"); CT = pd.read_csv(M + "/cnv_their_C.csv", dtype={"sample_id": str}).set_index("sample_id")
    assert list(TF.columns) == list(CT.columns); info["their_features_vs_cnv_their_C_max_abs_diff"] = float(np.abs(TF.loc[CT.index].values - CT.values).max())
    TF.loc[addall].rename_axis("sample_id").reset_index().to_csv(SETS + "/cnv_their_added.csv", index=False)
    # reproduction of the slide-mean image row (km_prep.py) for 5 stored samples
    mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str); IMs = pd.read_csv(M + "/img_mean.csv", dtype={"Sample": str}).set_index("Sample"); dif = []
    for s in ids[:5]:
        g = mp[mp.cnv == s].sort_values("npz"); b = np.concatenate([np.asarray(np.load(f)["embeddings"], dtype=np.float32) for f in g.npz], 0); dif.append(float(np.abs(b.mean(0) - IMs.loc[s].values).max()))
    info["img_mean_reproduction_max_abs_diff_5_samples"] = max(dif)
    # R54: Barrett's database later grades after the last sequenced endoscopy
    L = db_link(); LD = later_db(L); npat = sorted(set(pat[y == 0])); ld = LD.loc[npat]
    relab = [p for p in npat if (ld.loc[p, "max_after"] >= 3) or (ld.loc[p, "hgdtab_after"] > 0)]; excl = [p for p in npat if p not in relab and ld.loc[p, "max_after"] == 2]
    V54 = base[~base.Patient.isin(excl)].copy(); V54.loc[V54.Patient.isin(relab), "y"] = 1; VS["r54"] = V54
    info["r54"] = {"non_progressor_patients": len(npat), "linked": int(ld.linked.sum()), "relabelled_later_HGDplus": len(relab), "excluded_later_LGD": len(excl),
                   "relabelled_also_in_stored_check": int(len(set(relab) & set(lh))), "excluded_samples": int(base.Patient.isin(excl).sum()), "relabelled_samples": int(base.Patient.isin(relab).sum())}
    for v, V in VS.items():
        V = V.copy(); V["pre"] = V.pre.astype(bool); V["ndbe"] = V.ndbe.astype(bool); V.to_csv(f"{SETS}/{v}.csv", index=False); pr = V[V.pre]
        info.setdefault(v, {}); info[v].update({"samples": int(len(V)), "patients": int(V.Patient.nunique()), "progressor_patients": int(V[V.y == 1].Patient.nunique()), "pre_event_samples": int(len(pr)), "pre_event_progressor_samples": int(pr.y.sum()),
                                                "pre_event_patients": int(pr.Patient.nunique()), "pre_event_progressor_patients": int(pr[pr.y == 1].Patient.nunique())})
    res["sets"] = info; dump(res)
# ================================================================== R42 stability of the 10-model vote
elif TASK == "r42":
    O, I, FO, popm, rows, CALLS, band, info = primary(); Tc = CALLS["T"][rows]; mb = modal_band(band)[rows]; BN = {0: "low", 1: "middle", 2: "high"}
    halves = [(list(a), [j for j in range(10) if j not in a]) for a in itertools.combinations(range(10), 5) if 0 in a]; assert len(halves) == 126
    agree5 = np.array([((Tc[:, a].sum(1) >= 3) == (Tc[:, b].sum(1) >= 3)) for a, b in halves]); pairs = list(itertools.combinations(range(10), 2)); agree1 = np.array([Tc[:, a] == Tc[:, b] for a, b in pairs])
    def st(A, mk): v = A[:, mk].mean(1); return {"n_samples": int(mk.sum()), "mean": r3(v.mean()), "range_2.5_97.5": [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))], "min": r3(v.min()), "max": r3(v.max())}
    groups = {"all": np.ones(len(rows), bool), **{f"modal_{bn}": mb == g for g, bn in BN.items()}, "progressor": y[rows] == 1, "non_progressor": y[rows] == 0}
    res["R42"] = {**nfo(rows), "n_splits": len(halves), "n_single_pairs": len(pairs), "two_5_model_votes_agree": {k: st(agree5, m) for k, m in groups.items()}, "two_single_repeats_agree": {k: st(agree1, m) for k, m in groups.items()},
                  "note": "each of the 126 splits of the 10 repeats into two sets of 5 (repeat 1 in the first set); 5-model vote = >= 3 of 5 per-repeat triage calls; agreement = share of pre-event samples with equal calls; range over splits (pairs); modal band = most frequent band over the 10 repeats"}
    dump(res)
# ================================================================== R44 p53 among all triage-negative samples
elif TASK == "r44":
    O, I, FO, popm, rows, CALLS, band, info = primary(); avail = np.isfinite(P53); ab = (np.where(avail, P53, 0) == 1); Tn = ~CALLS["T"]
    def per_rep(mk): return {cl: {"p53_aberrant_mean_over_repeats": r3(np.mean([(mk[:, j] & popm & ab & (y == v)).sum() for j in range(10)])), "negative_samples_mean": r3(np.mean([(mk[:, j] & popm & (y == v)).sum() for j in range(10)])),
                                  "negative_with_p53_mean": r3(np.mean([(mk[:, j] & popm & avail & (y == v)).sum() for j in range(10)])), "p53_aberrant_range": [int(min((mk[:, j] & popm & ab & (y == v)).sum() for j in range(10))), int(max((mk[:, j] & popm & ab & (y == v)).sum() for j in range(10)))]}
                                 for cl, v in (("progressor", 1), ("non_progressor", 0))}
    low_mid_neg = (band == 0) | ((band == 1) & Tn); assert (low_mid_neg == (Tn & (band >= 0))).all()
    mbd = modal_band(band) == 0; modal = {cl: {"cleared_with_p53": int((mbd & popm & avail & (y == v)).sum()), "p53_aberrant": int((mbd & popm & ab & (y == v)).sum())} for cl, v in (("progressor", 1), ("non_progressor", 0))}
    res["R44"] = {"triage_negative_per_repeat": per_rep(low_mid_neg), "low_band_only_per_repeat": per_rep(band == 0), "modal_low_band_R37e": modal, "p53_aberrant_pre_event": {"progressor": int((ab & popm & (y == 1)).sum()), "non_progressor": int((ab & popm & (y == 0)).sum())},
                  "note": "triage-negative = low band, or middle band called negative by L-LATE, in that repeat (single model; counts averaged over the 10 repeats); pre-event samples"}
    dump(res)
# ================================================================== R45 current care with p53 at matched specificity
elif TASK == "r45":
    O, I, FO, popm, rows, CALLS, band, info = primary(); avail = np.isfinite(P53); ab = np.where(avail, P53, 0) == 1; DP = ((GR == 1) | (GR == 2)) | ab; rp = rows[avail[rows]]
    def stats(b, t):
        yb = y[b]; o = {}; spec_d = 1 - DP[b][yb == 0].mean(); o["sens_DP"] = DP[b][yb == 1].mean(); o["spec_DP"] = spec_d
        for m in ("L-LATE", "L-IMG"):
            o[f"sens_{m}"] = np.mean([sens_at_spec(O[m][b, j], yb, spec_d) for j in range(10)])
        T_ = CALLS["T"][b]; o["sens_T"] = T_[yb == 1].mean(); o["spec_T"] = 1 - T_[yb == 0].mean()
        pk = pd.Series(pat[b]).astype(str).values + "#" + t.astype(str); lp = pd.Series(yb).groupby(pk).max(); idx = lp.index
        dp = pd.Series(DP[b]).groupby(pk).max().reindex(idx).values; lpv = lp.values; o["pat_sens_DP"] = dp[lpv == 1].mean(); ps = 1 - dp[lpv == 0].mean(); o["pat_spec_DP"] = ps
        for m in ("L-LATE", "L-IMG"):
            S_ = pd.DataFrame(O[m][b]).groupby(pk).max().reindex(idx).values; o[f"pat_sens_{m}"] = np.mean([sens_at_spec(S_[:, j], lpv, ps) for j in range(10)])
        Tp = pd.DataFrame(T_).groupby(pk).max().reindex(idx).values; o["pat_sens_T"] = Tp[lpv == 1].mean(); o["pat_spec_T"] = 1 - Tp[lpv == 0].mean()
        return o
    obs = stats(rp, np.zeros(len(rp), int)); D = frame([stats(b, t) for b, t in boots_tagged(rp)]); R = {**nfo(rp), "draws": int(len(D)), "target_specificity_sample": r3(obs["spec_DP"]), "target_specificity_patient": r3(obs["pat_spec_DP"])}
    R["grade_or_p53"] = {"sample": {"sensitivity": r3(obs["sens_DP"]), "specificity": r3(obs["spec_DP"])}, "patient": {"sensitivity": r3(obs["pat_sens_DP"]), "specificity": r3(obs["pat_spec_DP"])}}
    for lv, pre in (("per_sample", ""), ("per_patient", "pat_")):
        R[lv] = {m: {"sensitivity_at_matched_specificity": r3(obs[f"{pre}sens_{m}"]), "sens_ci95": ci2(D[f"{pre}sens_{m}"]), "d_sens_minus_grade_or_p53": dstat(obs[f"{pre}sens_{m}"] - obs[f"{pre}sens_DP"], D[f"{pre}sens_{m}"] - D[f"{pre}sens_DP"], one_sided=True)}
                 for m in ("L-LATE", "L-IMG")}
    R["triage_own_operating_point"] = {lv: {"sensitivity": r3(obs[f"{pre}sens_T"]), "specificity": r3(obs[f"{pre}spec_T"]), "d_sens_minus_grade_or_p53": dstat(obs[f"{pre}sens_T"] - obs[f"{pre}sens_DP"], D[f"{pre}sens_T"] - D[f"{pre}sens_DP"], one_sided=True),
                                           "d_spec_minus_grade_or_p53": dstat(obs[f"{pre}spec_T"] - obs[f"{pre}spec_DP"], D[f"{pre}spec_T"] - D[f"{pre}spec_DP"])} for lv, pre in (("sample", ""), ("patient", "pat_"))}
    R["share_sequenced_triage"] = r3((band[rp] == 1).mean())
    R["note"] = "pre-event samples with a p53 result; threshold of each model set per repeat on these samples so that its specificity equals that of grade ID/LGD or p53 aberrant in the same samples (evaluation-matched; re-matched in every bootstrap draw); patient score = maximum over the patient's samples; patient bootstrap with resampled copies as distinct patients"
    res["R45"] = R; dump(res)
# ================================================================== R46 one random sample per endoscopy at matched endoscopy specificity
elif TASK == "r46":
    O, I, FO, popm, rows, CALLS, band, info = primary(); NS = 100
    eu, einv = np.unique(ENDO[rows], return_inverse=True); E = len(eu); es = [rows[einv == e] for e in range(E)]; elab = np.array([int(y[s].max()) for s in es]); epat = np.array([pat[s][0] for s in es])
    up = np.unique(epat); P_ = len(up); pe = [np.where(epat == p)[0] for p in up]; plab = np.array([int(elab[e].max()) for e in pe])
    PICK = np.zeros((NS, E), int)
    for sd in range(NS):
        rng = np.random.RandomState(sd)
        for e, s in enumerate(es): PICK[sd, e] = s[rng.randint(len(s))]   # R41 draw order (rv6_round6.py r41)
    LS = O["L-LATE"][PICK]                                    # seeds x E x 10: L-LATE of the picked sample
    RBown = CALLS["B"][PICK]                                  # its stored (B) call (own fold threshold)
    TE = np.array([CALLS["T"][s].any(0) for s in es])         # E x 10
    toP = lambda A: np.stack([A[..., e, :].any(-2) for e in pe], -2)
    def matched_calls_e(eidx):
        """random rule at the threshold giving, per repeat, the triage's endoscopy specificity on these endoscopies (evaluation-matched)"""
        neg = eidx[elab[eidx] == 0]; spec_t = 1 - TE[neg].mean(0); X = LS[:, neg, :]; Xs = np.sort(X, 1); q = np.ceil(spec_t * len(neg)).astype(int)
        thr = np.stack([Xs[:, q[j] - 1, j] if q[j] > 0 else np.full(NS, -np.inf) for j in range(10)], 1); return LS[:, eidx, :] > thr[:, None, :]
    def stats(eidx, pidx):
        o = {}; RM = matched_calls_e(eidx); le = elab[eidx]; Tt = TE[eidx]
        o["endo_sens_T"] = Tt[le == 1].mean(); o["endo_spec_T"] = 1 - Tt[le == 0].mean()
        o["endo_sens_rand_matched"] = RM[:, le == 1, :].mean((1, 2)); o["endo_spec_rand_matched"] = 1 - RM[:, le == 0, :].mean((1, 2))
        o["endo_sens_rand_own"] = RBown[:, eidx][:, le == 1, :].mean((1, 2)); o["endo_spec_rand_own"] = 1 - RBown[:, eidx][:, le == 0, :].mean((1, 2))
        return o
    obs = stats(np.arange(E), np.arange(P_)); rng = np.random.RandomState(0); DR = []
    while len(DR) < NB:
        pidx = rng.choice(P_, P_); eidx = np.concatenate([pe[i] for i in pidx])
        if 0 < plab[pidx].sum() < P_: DR.append(stats(eidx, pidx))
    R = {**nfo(rows), "n_endoscopies": int(E), "n_progressor_endoscopies": int(elab.sum()), "n_seeds": NS, "draws": len(DR)}
    def sm_(k): v = np.atleast_1d(obs[k]); return {"mean_over_seeds": r3(v.mean()), "range_2.5_97.5_over_seeds": ci2(v) if len(v) > 1 else None, "ci95_bootstrap_of_seed_mean": ci2([np.atleast_1d(d[k]).mean() for d in DR])}
    R["a_matched_specificity"] = {"triage": {"endoscopy_sensitivity": sm_("endo_sens_T"), "endoscopy_specificity": sm_("endo_spec_T")},
                                  "random_matched": {"endoscopy_sensitivity": sm_("endo_sens_rand_matched"), "endoscopy_specificity": sm_("endo_spec_rand_matched")},
                                  "random_own_threshold": {"endoscopy_sensitivity": sm_("endo_sens_rand_own"), "endoscopy_specificity": sm_("endo_spec_rand_own")}}
    dv = obs["endo_sens_T"] - obs["endo_sens_rand_matched"]; Dd = np.array([(d["endo_sens_T"] - d["endo_sens_rand_matched"]).mean() for d in DR])
    R["a_matched_specificity"]["d_endoscopy_sensitivity_triage_minus_random"] = {"mean_over_seeds": r3(dv.mean()), "range_2.5_97.5_over_seeds": ci2(dv), "ci95_bootstrap_of_seed_mean": ci2(Dd), "one_sided_95_lower": lo1(Dd), "p_two_sided": p2(Dd)}
    # (b) per patient
    RMo = matched_calls_e(np.arange(E)); RPo = toP(RBown); RPm = toP(RMo); TPp = toP(TE)
    def pcount(A): return {"progressor_patients_missed": r3((~A[..., plab == 1, :]).sum(-2).mean()), "non_progressor_patients_flagged": r3(A[..., plab == 0, :].sum(-2).mean())}
    R["b_per_patient"] = {"random_own_threshold": pcount(RPo), "random_matched": pcount(RPm), "triage": pcount(TPp), "n_progressor_patients": int(plab.sum()), "n_non_progressor_patients": int((plab == 0).sum()),
                          "note": "counts averaged over seeds and repeats; patient positive if any of its endoscopies is"}
    # (c) sequencing burden
    mid = band == 1; R["c_burden"] = {"random": {"share_samples_sequenced": r3(E / len(rows)), "share_endoscopies_with_a_sequenced_sample": 1.0},
                                      "triage": {"share_samples_sequenced": r3(mid[rows].mean()), "share_endoscopies_with_a_sequenced_sample": r3(np.mean([np.mean([mid[s, j].any() for s in es]) for j in range(10)]))},
                                      "Bstar": {"share_samples_sequenced": 1.0, "share_endoscopies_with_a_sequenced_sample": 1.0}, "B": {"share_samples_sequenced": 1.0, "share_endoscopies_with_a_sequenced_sample": 1.0}}
    R["note"] = "seeds 0-99 with the R41 draw order; matched = one L-LATE threshold per seed x repeat over the endoscopies (pooled across folds), set so that the random rule's endoscopy specificity equals the triage's in that repeat; re-matched in each bootstrap draw; own threshold = the stored (B) call (fold threshold at 80% training sensitivity)"
    res["R46"] = R; dump(res)
# ================================================================== R47 / R48 burden, prevalence and cost; the two-band rule
elif TASK in ("r47", "r48"):
    O, I, FO, popm, rows, CALLS, band, info = primary(); TL, CSt, TS = cutoff_arrays(info, FO); late = O["L-LATE"]
    two = (band >= 1) & (late >= TS); assert np.isfinite(TS[rows]).all()
    avail = np.isfinite(P53); ab = np.where(avail, P53, 0) == 1; D1 = (GR == 1) | (GR == 2); DP = D1 | ab
    seq = {"T": band == 1, "2B": band >= 1, "B*": np.ones((n, 10), bool), "B": np.ones((n, 10), bool), "C": np.zeros((n, 10), bool), "D": np.zeros((n, 10), bool), "D+p53": np.zeros((n, 10), bool)}
    RU = {"T": CALLS["T"], "2B": two, "B*": CALLS["B*"], "B": CALLS["B"], "C": CALLS["C"], "D": np.repeat(D1[:, None], 10, 1), "D+p53": np.repeat(DP[:, None], 10, 1)}
    if TASK == "r48":
        R = {}; CAL2 = {**CALLS, "2B": two}; boots = boots_for(rows); obs = draw_stats(rows, CAL2, band, O); D = frame([draw_stats(b, CAL2, band, O) for b in boots])
        sh2 = lambda b: (band[b] >= 1).mean(); R["two_band"] = {"sensitivity": r3(obs["nv_sens_2B"]), "sens_ci95": ci2(D["nv_sens_2B"]), "specificity": r3(obs["nv_spec_2B"]), "spec_ci95": ci2(D["nv_spec_2B"]),
                                                               "share_sequenced": r3(sh2(rows)), "share_ci95": ci2([sh2(b) for b in boots]), "high_band_share": r3((band[rows] == 2).mean()),
                                                               "high_band_called_negative_share_of_high": r3(((band[rows] == 2) & ~two[rows]).sum() / max(1, (band[rows] == 2).sum()))}
        for ref in ("T", "B*", "B"):
            R[f"two_band_minus_{ref}"] = {"d_sens": dstat(obs["nv_sens_2B"] - obs[f"nv_sens_{ref}"], D["nv_sens_2B"] - D[f"nv_sens_{ref}"], one_sided=True), "d_spec": dstat(obs["nv_spec_2B"] - obs[f"nv_spec_{ref}"], D["nv_spec_2B"] - D[f"nv_spec_{ref}"])}
        R["table1"] = table1({k: RU[k] for k in ("2B", "T", "B*", "B")}, rows, {k: r3(seq[k][rows].mean()) for k in ("2B", "T", "B*", "B")}, pairs=(("2B", "T"), ("2B", "B*")))
        R["triage_reproduction_expected_0.865_0.697"] = [r3(obs["nv_sens_T"]), r3(obs["nv_spec_T"])]; R.update(nfo(rows))
        R["note"] = "two-band: low band cleared; middle and high bands sequenced and called by L-LATE at the triage's fold t_seq (no flag without sequencing); single model, means over repeats; patient bootstrap"
        res["R48"] = R; dump(res)
    else:
        rr = rows; yb = y[rr]; R = {"rules": {}}
        # (a) flag rate: flagged non-progressor endoscopies per non-progressor patient-year (span from the patient's first to last pre-event endoscopy)
        npats = np.unique(pat[rr][yb == 0]); py = sum((MBF[rr][pat[rr] == p].max() - MBF[rr][pat[rr] == p].min()) / 12.0 for p in npats)
        ek = ENDO[rr]; eu = np.unique(ek); elab = pd.Series(yb).groupby(ek).max().reindex(eu).values; spe = pd.Series(ek).value_counts().reindex(eu).values; kd = pd.Series(spe).value_counts(normalize=True).sort_index()
        PREV = (0.005, 0.02, 0.05); CPE = 85.0; CEN = 471.0
        for k in ("T", "2B", "B*", "B", "C", "D", "D+p53"):
            C_ = RU[k][rr]; Sq = seq[k][rr]; sens = C_[yb == 1].mean(); fpr = C_[yb == 0].mean(); sqP = Sq[yb == 1].mean(); sqN = Sq[yb == 0].mean()
            EF = pd.DataFrame(C_).groupby(ek).max().reindex(eu).values; fe = EF[elab == 0].sum(0).mean()
            o = {"sensitivity": r3(sens), "false_positive_rate": r3(fpr), "share_sequenced_progressor": r3(sqP), "share_sequenced_non_progressor": r3(sqN), "share_sequenced_cohort": r3(Sq.mean()),
                 "a_flagged_non_progressor_endoscopies_per_patient_year": r3(fe / py), "flagged_non_progressor_endoscopies_mean": r3(fe), "projection_per_1000_samples": {}}
            for pv in PREV:
                pf = pv * sens + (1 - pv) * fpr; nseq = 1000 * (pv * sqP + (1 - pv) * sqN); nfl = 1000 * pf; nendo = 1000 / spe.mean()
                fen = nendo * float(sum(w * (1 - (1 - pf) ** kk) for kk, w in kd.items())); miss = 1000 * pv * (1 - sens)
                o["projection_per_1000_samples"][str(pv)] = {"samples_sequenced": round(nseq, 1), "flagged_samples": round(nfl, 1), "flagged_endoscopies": round(fen, 1), "progressor_samples_missed": round(miss, 2),
                                                             "cost_gbp": round(CPE * nseq + CEN * fen), "cost_sequencing_gbp": round(CPE * nseq), "cost_repeat_endoscopy_gbp": round(CEN * fen), "c_share_sequenced_reweighted": r3(pv * sqP + (1 - pv) * sqN)}
            R["rules"][k] = o
        R["non_progressor_patient_years"] = r3(py); R["n_non_progressor_patients"] = int(len(npats)); R["samples_per_endoscopy"] = {"mean": r3(spe.mean()), "distribution": {str(int(k)): r3(v) for k, v in kd.items()}}
        R["costs"] = {"per_sequenced_sample_gbp": CPE, "per_flagged_endoscopy_gbp": CEN, "sources": "Gregson 2017 thesis (2017-18 prices); NHS Payment Scheme 2024/25 FE21Z, one repeat endoscopy per flagged endoscopy; illustrative"}
        R["note"] = "class-conditional per-sample rates from the pre-event samples (single model, means over repeats); flagged endoscopies = (1000 / mean samples per endoscopy) x sum_k f(k) [1 - (1 - p)^k], f = cohort samples-per-endoscopy distribution, p = per-sample flag rate at that prevalence (samples within an endoscopy taken as independent); grade-or-p53 with missing p53 as wild-type; (C), (D), grade-or-p53 sequence nothing"
        R.update(nfo(rows)); res["R47"] = R; dump(res)
# ================================================================== R49 per-endoscopy gate
elif TASK == "r49":
    O, I, FO, popm, rows, CALLS, band, info = primary(); TL, CSt, TS = cutoff_arrays(info, FO); img = O["L-IMG"]; late = O["L-LATE"]
    eu, einv = np.unique(ENDO[rows], return_inverse=True); E = len(eu); es = [rows[einv == e] for e in range(E)]; elab = np.array([int(y[s].max()) for s in es]); epat = np.array([pat[s][0] for s in es])
    GB = np.zeros((E, 10), int); GC = np.zeros((E, 10), bool); GV = np.zeros((E, 10), bool); NSEQ1 = np.zeros((E, 10)); NSEQV = np.zeros((E, 10)); TE = np.zeros((E, 10), bool); TSEQ = np.zeros((E, 10))
    for e, s in enumerate(es):
        for j in range(10):
            assert len(set(FO[s, j])) == 1; tl, cs, ts = TL[s[0], j], CSt[s[0], j], TS[s[0], j]; sc = img[s, j].max()
            g = (2 if sc > cs else 0) if tl > cs else (0 if sc < tl else (2 if sc > cs else 1)); GB[e, j] = g
            if g == 1: top = s[np.argmax(img[s, j])]; GC[e, j] = late[top, j] >= ts; GV[e, j] = (late[s, j] >= ts).any(); NSEQ1[e, j] = 1; NSEQV[e, j] = len(s)
            else: GC[e, j] = GV[e, j] = g == 2
            TE[e, j] = CALLS["T"][s, j].any(); TSEQ[e, j] = (band[s, j] == 1).sum()
    nsz = np.array([len(s) for s in es]); up = np.unique(epat); pe = [np.where(epat == p)[0] for p in up]; plab = np.array([int(elab[i].max()) for i in pe])
    def stats(eidx, pidx):
        le = elab[eidx]; o = {}
        for nm, A in (("gate", GC), ("gate_all_mid_sequenced", GV), ("triage", TE)):
            Ae = A[eidx]; o[f"sens_{nm}"] = Ae[le == 1].mean(); o[f"spec_{nm}"] = 1 - Ae[le == 0].mean()
            Ap = np.stack([A[pe[i]].any(0) for i in pidx]); lp = plab[pidx]; o[f"pmiss_{nm}"] = (~Ap[lp == 1]).sum(0).mean(); o[f"pflag_{nm}"] = Ap[lp == 0].sum(0).mean()
        tot = nsz[eidx].sum(); o["share_samples_gate"] = NSEQ1[eidx].sum(0).mean() / tot; o["share_samples_gate_all_mid_sequenced"] = NSEQV[eidx].sum(0).mean() / tot; o["share_samples_triage"] = TSEQ[eidx].sum(0).mean() / tot
        o["share_endo_gate"] = (GB[eidx] == 1).mean(); o["share_endo_gate_all_mid_sequenced"] = o["share_endo_gate"]; o["share_endo_triage"] = (TSEQ[eidx] > 0).mean(); return o
    obs = stats(np.arange(E), np.arange(len(up))); rng = np.random.RandomState(0); DR = []
    while len(DR) < NB:
        pidx = rng.choice(len(up), len(up)); eidx = np.concatenate([pe[i] for i in pidx])
        if 0 < plab[pidx].sum() < len(up): DR.append(stats(eidx, pidx))
    D = frame(DR); R = {**nfo(rows), "n_endoscopies": int(E), "n_progressor_endoscopies": int(elab.sum()), "draws": len(D), "endoscopy_band_shares": {nm: r3((GB == g).mean()) for g, nm in ((0, "low"), (1, "middle"), (2, "high"))}, "rules": {}}
    for nm in ("gate", "gate_all_mid_sequenced", "triage"):
        R["rules"][nm] = {"endoscopy_sensitivity": r3(obs[f"sens_{nm}"]), "sens_ci95": ci2(D[f"sens_{nm}"]), "endoscopy_specificity": r3(obs[f"spec_{nm}"]), "spec_ci95": ci2(D[f"spec_{nm}"]),
                          "share_samples_sequenced": r3(obs[f"share_samples_{nm}"]), "share_endoscopies_sequenced": r3(obs[f"share_endo_{nm}"]), "progressor_patients_missed": round(obs[f"pmiss_{nm}"], 2), "non_progressor_patients_flagged": round(obs[f"pflag_{nm}"], 2)}
        if nm != "triage":
            R["rules"][nm]["minus_triage"] = {"d_endoscopy_sens": dstat(obs[f"sens_{nm}"] - obs["sens_triage"], D[f"sens_{nm}"] - D["sens_triage"], one_sided=True), "d_endoscopy_spec": dstat(obs[f"spec_{nm}"] - obs["spec_triage"], D[f"spec_{nm}"] - D["spec_triage"]),
                                              "d_share_samples": dstat(obs[f"share_samples_{nm}"] - obs["share_samples_triage"], D[f"share_samples_{nm}"] - D["share_samples_triage"])}
    R["note"] = "endoscopy score = maximum L-IMG over its pre-event samples; the triage's fold t_low and c* (t_high) and t_seq; gate: middle-band endoscopy sequences its top-L-IMG sample and is called by its L-LATE; variant: all its samples sequenced, flagged if any L-LATE positive; triage aggregated to endoscopies (any sample positive); single model; patient bootstrap over endoscopies"
    res["R49"] = R; dump(res)
# ================================================================== R50 NDBE-only block
elif TASK == "r50":
    O, I, FO = load("their"); popm = popmask("pre_ndbe"); rows = np.where(popm)[0]; CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm)
    avail = np.isfinite(P53); ab = np.where(avail, P53, 0) == 1; DP = ((GR == 1) | (GR == 2)) | ab; assert not ((GR[rows] == 1) | (GR[rows] == 2)).any()
    RU = {"T": CALLS["T"], "B*": CALLS["B*"], "B": CALLS["B"], "C*": CALLS["C*"], "D+p53": np.repeat(DP[:, None], 10, 1)}; sh = {"T": r3((band[rows] == 1).mean()), "B*": 1.0, "B": 1.0, "C*": 0.0, "D+p53": 0.0}
    R = table1(RU, rows, sh, pairs=(("T", "B*"), ("T", "B"), ("T", "D+p53")))
    A_, obs, D, _, _ = analysis(O, I, FO, popm, rows, calls=(CALLS, band, info, thr, sstar)); R["claim_T_H"] = tri_block(A_, CALLS, band, rows)
    R["d_plus_p53_on_samples_with_p53"] = {"n_samples": int(avail[rows].sum()), **{k: r3(v) for k, v in zip(("sensitivity", "specificity"), (DP[rows][avail[rows] & (y[rows] == 1)].mean(), 1 - DP[rows][avail[rows] & (y[rows] == 0)].mean()))}}
    R["note"] = "pre-event NDBE samples; cut-offs and matched comparators chosen on NDBE training samples (R33 row 2 / R39 'ndbe' definition); (D) is empty on NDBE, so (D)+p53 = p53 aberrant (missing p53 as wild-type)"
    res["R50"] = R; dump(res)
# ================================================================== R51 falsely cleared progressor endoscopies
elif TASK == "r51":
    O, I, FO, popm, rows, CALLS, band, info = primary()
    eu, einv = np.unique(ENDO[rows], return_inverse=True); es = [rows[einv == e] for e in range(len(eu))]; elab = np.array([int(y[s].max()) for s in es])
    pe_ = [s for s, l in zip(es, elab) if l == 1]; neg = np.array([~CALLS["T"][s].any(0) for s in pe_]); tte = np.array([TIME[s].max() for s in pe_]); gmax = np.array([GR[s].max() for s in pe_])
    tb = np.where(tte < 1, "<1 year", np.where(tte <= 3, "1-3 years", ">3 years")); GN = {0: "NDBE", 1: "ID", 2: "LGD"}
    R = {"n_progressor_endoscopies": len(pe_), "by_time_to_event": {}}
    for b_ in ("<1 year", "1-3 years", ">3 years"):
        m = tb == b_; R["by_time_to_event"][b_] = {"progressor_endoscopies": int(m.sum()), "falsely_cleared_mean_over_repeats": r3(neg[m].sum(0).mean()), "falsely_cleared_in_any_repeat": int(neg[m].any(1).sum()), "falsely_cleared_in_majority": int((neg[m].sum(1) >= 6).sum()),
                                                   "grade_of_endoscopies_cleared_in_any_repeat": {GN[g]: int(((gmax == g) & neg.any(1) & m).sum()) for g in (0, 1, 2)}}
    R["grade_of_all_progressor_endoscopies"] = {GN[g]: int((gmax == g).sum()) for g in (0, 1, 2)}
    e1, e2 = np.quantile(TILES[rows], [1 / 3, 2 / 3]); ppats = np.unique(pat[rows][y[rows] == 1]); miss = []
    for p in ppats:
        ix_ = rows[pat[rows] == p]; called = CALLS["T"][ix_].any(0)
        if (~called).any():
            mt = np.median(TILES[ix_]); miss.append({"repeats_missed": int((~called).sum()), "n_pre_event_endoscopies": int(len(set(ENDO[ix_]))), "max_grade": GN[int(GR[ix_].max())], "tissue_tertile": int(0 if mt <= e1 else 1 if mt <= e2 else 2) + 1,
                                                     "years_first_sample_to_event": float(TIME[ix_].max())})
    MD = pd.DataFrame(miss)
    R["progressor_patients_missed_in_at_least_one_repeat"] = {"n": len(MD), "n_progressor_patients": int(len(ppats))}
    if len(MD):
        R["progressor_patients_missed_in_at_least_one_repeat"].update({"repeats_missed": {"median": float(MD.repeats_missed.median()), "min": int(MD.repeats_missed.min()), "max": int(MD.repeats_missed.max())},
            "n_pre_event_endoscopies": {"median": float(MD.n_pre_event_endoscopies.median()), "min": int(MD.n_pre_event_endoscopies.min()), "max": int(MD.n_pre_event_endoscopies.max())},
            "max_grade": MD.max_grade.value_counts().to_dict(), "tissue_tertile": {str(k): int(v) for k, v in MD.tissue_tertile.value_counts().sort_index().items()},
            "years_first_sample_to_event": {"median": r3(MD.years_first_sample_to_event.median()), "min": r3(MD.years_first_sample_to_event.min()), "max": r3(MD.years_first_sample_to_event.max())}})
    R["note"] = "falsely cleared = progressor pre-event endoscopy with every sample called negative by the triage in that repeat; time to event of the endoscopy = its samples' time (years); grade = highest sample grade at the endoscopy; tissue tertile = tertile (pre-event sample edges) of the patient's median tissue tiles; time from first sample to event = the patient's largest pre-event time; aggregates only (render shows counts < 5 as '<5')"
    res["R51"] = R; dump(res)
# ================================================================== R53 event dates from the Barrett's database (re-scoring, no refit)
elif TASK == "r53":
    O, I, FO, popm, rows, CALLS, band, info = primary(); L = db_link(); fd = L["fd"]; pdate = L["pdate"]; sd_of = L["sd_of"]; pt = L["pt"]
    A = pd.read_csv(KD + "/kr_samples.csv", dtype=str).set_index("Sample"); fdu = fd.drop_duplicates("combined_name").set_index("combined_name"); A["mbf"] = pd.to_numeric(A.index.map(fdu["Months before final"]), errors="coerce"); A["ogd"] = A.index.map(fdu["Path_class_per_OGD"])
    hgA = A[(A.Pathology.isin(["HGD", "IMC"])) | (A.ogd.isin(["HGD", "IMC", "HGD/IMC"]))]; tevA = hgA.groupby("Patient").mbf.max()
    SD_ = pd.Series(ids).map(pdate).values; assert pd.notna(SD_).all(), "analysed sample without an exact date"; SD_ = pd.to_datetime(SD_)
    ev = {}; evs = {}; evd = {}; src = {"sheet_only": 0, "db_only": 0, "both_db_earlier": 0, "both_sheet_earlier_or_same": 0, "neither": 0, "not_linked": 0}
    for p in sorted(set(pat[y == 1])):
        sh = pd.NaT
        if p in tevA.index:
            d_ = [sd_of.get(s) for s in A[(A.Patient == p) & (A.mbf == tevA[p])].index]; d_ = [x for x in d_ if pd.notna(x)]; sh = min(d_) if d_ else pd.NaT
        pid = L["link"].get(p); db = pt[(pt.pid == pid) & (pt.g >= 3)].d.min() if pid is not None else pd.NaT
        if pid is None: src["not_linked"] += 1
        if pd.notna(sh) and pd.notna(db): src["both_db_earlier" if db < sh else "both_sheet_earlier_or_same"] += 1
        elif pd.notna(sh): src["sheet_only"] += 1
        elif pd.notna(db): src["db_only"] += 1
        else: src["neither"] += 1
        ev[p] = min([x for x in (sh, db) if pd.notna(x)], default=pd.NaT); evs[p] = sh; evd[p] = db
    EV = pd.to_datetime(pd.Series(pat).map(ev).values); EVS = pd.to_datetime(pd.Series(pat).map(evs).values); EVD = pd.to_datetime(pd.Series(pat).map(evd).values); R = {"event_date_source_progressor_patients": src}
    pm1 = popm & (y == 1); R["diagnostics_pre_event_progressor_samples"] = {"dated_on_or_after_sheet_event_date": int((pm1 & pd.notna(EVS) & (SD_ >= EVS)).sum()), "dated_on_or_after_db_first_hgdplus": int((pm1 & pd.notna(EVD) & (SD_ >= EVD)).sum()),
                                                            "dated_more_than_14_days_after_db_first_hgdplus": int((pm1 & pd.notna(EVD) & (SD_ > EVD + pd.Timedelta(days=14))).sum()), "dated_within_14_days_of_db_first_hgdplus": int((pm1 & pd.notna(EVD) & (SD_ >= EVD - pd.Timedelta(days=14)) & (SD_ <= EVD + pd.Timedelta(days=14))).sum())}
    for nm, tol in (("primary_on_or_after", 0), ("same_endoscopy_within_14_days_or_after", 14)):
        drop = popm & (y == 1) & pd.notna(EV) & (SD_ >= EV - pd.Timedelta(days=tol)); rows2 = rows[~drop[rows]]
        A_, obs, D, _, _ = analysis(O, I, FO, popm, rows2, calls=PCALLS)
        R[nm] = {"n_removed_samples": int(drop[rows].sum()), "n_removed_patients_affected": int(len(set(pat[rows][drop[rows]]))), "removed_grade": pd.Series(SM.Pathology.values[rows][drop[rows]]).value_counts().to_dict(), **tri_block(A_, CALLS, band, rows2)}
    R["note"] = "event date = earlier of the sheet event endoscopy (exact date of the first HGD/IMC sample, hz_prep.py event) and the first HGD+ confirmed-grade pathology report in the Barrett's database (r2_main.py linkage); pre-event samples dated (pre_event_cohort.Date) on or after it are removed from evaluation; stored calls and cut-offs (no refit); the 14-day row also removes samples from the same endoscopy as the first database HGD+ report (lookup convention)"
    res["R53"] = R; dump(res)
# ================================================================== refit metrics (R43, R52, R54) and the reproduction check
elif TASK.startswith("refit_"):
    var = TASK[6:]; V = pd.read_csv(f"{SETS}/{var}.csv", dtype={"Sample": str, "Patient": str, "Endoscopy": str}); use_set(V)
    O, I, FO = load_var(var); popm = popmask("pre"); rows = np.where(popm)[0]; CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm)
    A_, obs, D, _, _ = analysis(O, I, FO, popm, rows, calls=(CALLS, band, info, thr, sstar)); R = {"variant": var, **tri_block(A_, CALLS, band, rows), "folds_no_middle_band": A_["folds_no_middle_band"], "t_seq_fallback": A_["t_seq_fallback"]}
    R["table1"] = table1({k: CALLS[k] for k in ("T", "B*", "B", "C*")}, rows, {"T": r3((band[rows] == 1).mean()), "B*": 1.0, "B": 1.0, "C*": 0.0})
    R["n_total"] = {"samples": int(n), "patients": int(len(set(pat))), "progressor_patients": int(len(set(pat[y == 1])))}
    res["refit"] = R; dump(res)
elif TASK == "check":
    out = {}
    def cmp(a, b, key):
        A = pd.read_csv(a, dtype={"Sample": str}); B = pd.read_csv(b, dtype={"Sample": str}); Mg = A.merge(B, on=key, suffixes=("_a", "_b")); assert len(Mg) == len(A) == len(B), (len(A), len(B), len(Mg))
        return {"n": int(len(Mg)), "max_abs_pred_diff": float(np.abs(Mg.pred_a - Mg.pred_b).max()), "folds_equal": bool((Mg.fold_a == Mg.fold_b).all()), "lambda_max_abs_diff": float(np.abs(Mg.lambda_a - Mg.lambda_b).max())}
    D_ = R7 + "/refit/base"
    out["outer_cnv_their_rep_01_vs_stored_cfg_00"] = cmp(f"{D_}/outer_cnv_their_rep_01.csv", f"{M}/cv/preds/cfg_00_rep_01.csv", ["Sample"])
    out["outer_img_rep_01_vs_stored_cfg_02"] = cmp(f"{D_}/outer_img_rep_01.csv", f"{M}/cv/preds/cfg_02_rep_01.csv", ["Sample"])
    for cfg, st in (("cnv_their", "cnv_their"), ("img", "img")):
        cs = [cmp(f"{D_}/inner_{cfg}_rep_01_fold_{k:02d}.csv", f"{HZ}/inner/{st}_rep_01_fold_{k:02d}.csv", ["Sample", "inner_fold"]) for k in range(1, 11)]
        out[f"inner_{cfg}_rep_01_10_folds_vs_stored"] = {"files": len(cs), "n": sum(c["n"] for c in cs), "max_abs_pred_diff": max(c["max_abs_pred_diff"] for c in cs), "folds_equal": all(c["folds_equal"] for c in cs), "lambda_max_abs_diff": max(c["lambda_max_abs_diff"] for c in cs)}
    out["all_passed"] = bool(all(v["max_abs_pred_diff"] <= 1e-5 and v["folds_equal"] and v["lambda_max_abs_diff"] <= 1e-9 for v in out.values()))
    # stored predictions through the round-7 scoring path: R33 row 1 and the R38 evaluation-only rows
    O, I, FO, popm, rows, CALLS, band, info = primary(); A_, obs, D, _, _ = analysis(O, I, FO, popm, rows, calls=PCALLS); out["r33_row1_expected_-0.016_-0.032_+0.033_+0.045"] = tri_block(A_, CALLS, band, rows)
    r38 = J(f"{OUT6}/r38_merge.json")["R38"]["rows"]; out["r38_triage_sensitivity_by_row"] = {r["analysis"]: r["triage"]["sensitivity"] for r in r38 if r.get("status", "done") == "done"}
    res["checks"] = out; dump(res)
# ================================================================== R55 image quality and within-class confounds
elif TASK.startswith("r55px_") or TASK.startswith("r56px_"):
    import openslide
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    which, ch = TASK.split("_"); ch = int(ch); NCH = 40; OD = R7 + ("/px_disc" if which == "r55px" else "/px_val"); os.makedirs(OD, exist_ok=True)
    if which == "r55px": mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(set(ids))].drop_duplicates(["cnv", "npz"]).sort_values(["cnv", "npz"]).reset_index(drop=True); key = "cnv"
    else: mp = pd.read_csv(R7 + "/val_slides.csv", dtype=str).sort_values(["sample_id", "npz"]).reset_index(drop=True); key = "sample_id"
    sub = mp.iloc[ch::NCH]; out = []
    for r in sub.itertuples():
        with np.load(r.npz, allow_pickle=True) as z: sp = remap(str(z["slide_path"])); lv = int(z["level"]); ts = int(z["tile_size"]); xy = np.asarray(z["coords_level"])
        s = openslide.OpenSlide(sp); prop = s.properties; ds = s.level_downsamples[lv]; bg = []
        for x, yy in xy:
            im = np.asarray(s.read_region((int(round(x * ds)), int(round(yy * ds))), lv, (ts, ts)).convert("RGB")); bg.append(((im > 220).all(-1)).mean() > 0.5)
        W_, H_ = s.level_dimensions[lv]; md = os.path.join(os.path.dirname(os.path.dirname(sp)), "masks"); cand = sorted(glob.glob(os.path.join(md, glob.escape(os.path.splitext(os.path.basename(sp))[0]) + "*"))); tt = np.nan
        if cand:
            mk = np.array(Image.open(cand[0]).convert("L")) > 0; mh, mw = mk.shape
            xi = np.clip(((np.arange(W_ // ts) * ts + ts / 2) * mw / W_).round().astype(int), 0, mw - 1); yi = np.clip(((np.arange(H_ // ts) * ts + ts / 2) * mh / H_).round().astype(int), 0, mh - 1); tt = int(mk[np.ix_(yi, xi)].sum())
        out.append({"key": getattr(r, key), "npz": r.npz, "n_tiles": len(bg), "n_background": int(np.sum(bg)), "scanner": prop.get("hamamatsu.Product"), "created": prop.get("hamamatsu.Created"), "tissue_tiles": tt}); s.close()
    pd.DataFrame(out).to_csv(f"{OD}/chunk_{ch:02d}.csv", index=False); res["chunk"] = {"slides": len(out)}; print("RV7 PX DONE", TASK, len(out), flush=True)
elif TASK == "r55":
    O, I, FO, popm, rows, CALLS, band, info = primary(); PX = pd.concat([pd.read_csv(f, dtype={"key": str, "created": str}) for f in sorted(glob.glob(R7 + "/px_disc/chunk_*.csv"))])
    mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(set(ids))].drop_duplicates(["cnv", "npz"]); assert len(PX) == len(mp) and PX.npz.nunique() == len(mp), (len(PX), len(mp))
    g = PX.groupby("key"); BG = (g.n_background.sum() / g.n_tiles.sum()).reindex(ids).values; SBG = (PX.n_background / PX.n_tiles); assert np.isfinite(BG).all()
    created = pd.to_datetime(PX.created, errors="coerce"); assert created.notna().all(); scan = created.groupby(PX.key.values).min().reindex(ids)
    L = db_link(); sdate = pd.to_datetime(pd.Series(ids).map(L["pdate"]).values); AGE = ((scan.values - sdate.values) / np.timedelta64(1, "D")) / 365.25; AGE = np.asarray(AGE, float); assert np.isfinite(AGE).all()
    e1, e2 = np.quantile(TILES[rows], [1 / 3, 2 / 3]); tert = np.where(TILES <= e1, 0, np.where(TILES <= e2, 1, 2)); R = {}
    q = lambda v: {"n": int(len(v)), "mean": r3(np.mean(v)), "median": r3(np.median(v)), "iqr": [r3(np.percentile(v, 25)), r3(np.percentile(v, 75))], "share_above_0.25": r3(np.mean(np.asarray(v) > 0.25)), "n_above_0.25": int(np.sum(np.asarray(v) > 0.25))}
    R["a_background"] = {"definition": "share of the stored 256 tile positions per slide whose tile (stored level and size) has > 50% pixels with all RGB > 220; per sample pooled over its slides",
                         "slides": {"n": int(len(PX)), "share_slides_above_0.25": r3((SBG > 0.25).mean()), "n_slides_above_0.25": int((SBG > 0.25).sum())},
                         "pre_event_samples_by_class": {cl: q(BG[rows][y[rows] == v]) for cl, v in (("progressor", 1), ("non_progressor", 0))},
                         "pre_event_samples_by_tissue_tertile": {f"tertile_{t + 1}": {cl: q(BG[rows][(tert[rows] == t) & (y[rows] == v)]) for cl, v in (("progressor", 1), ("non_progressor", 0))} for t in range(3)},
                         "all_676_by_class": {cl: q(BG[y == v]) for cl, v in (("progressor", 1), ("non_progressor", 0))}}
    keep = BG <= 0.25; rows2 = rows[keep[rows]]; A_, obs, D, _, _ = analysis(O, I, FO, popm, rows2, calls=PCALLS)
    R["a_background"]["excluding_samples_above_0.25"] = {"n_removed": int(len(rows) - len(rows2)), "n_removed_progressor": int(y[rows].sum() - y[rows2].sum()), **tri_block(A_, CALLS, band, rows2)}
    # (b) slide age, within class
    LI = O["L-IMG"].mean(1); R["b_slide_age"] = {"definition": "scan date (slide header hamamatsu.Created; earliest over the sample's slides) minus the exact endoscopy date (pre_event_cohort.Date), years",
                                                "pre_event_by_class": {cl: {**{k: r3(v) for k, v in zip(("median", "q1", "q3"), np.percentile(AGE[rows][y[rows] == vv], [50, 25, 75]))}, "min": r3(AGE[rows][y[rows] == vv].min()), "max": r3(AGE[rows][y[rows] == vv].max())} for cl, vv in (("progressor", 1), ("non_progressor", 0))},
                                                "spearman_L_IMG_within_class": {}}
    for cl, vv in (("progressor", 1), ("non_progressor", 0)):
        m = rows[y[rows] == vv]; R["b_slide_age"]["spearman_L_IMG_within_class"][cl] = {"slide_age": r3(spearmanr(LI[m], AGE[m]).correlation), "tissue_tiles": r3(spearmanr(LI[m], TILES[m]).correlation), "biopsy_year": r3(spearmanr(LI[m], YEAR[m], nan_policy="omit").correlation), "n": int(len(m))}
    boots = boots_for(rows); neg_age = -AGE
    R["b_slide_age"]["auroc_slide_age_alone"] = {"direction": "younger slide = higher score (score = -age)", "value": r3(fsa(neg_age, rows)), "ci95": ci2([fsa(neg_age, b) for b in boots]), "note": "fold-stratified AUROC (fixed score, folds of the stored CV)"}
    # (c), (d) metadata alone and L-IMG + metadata, out-of-fold under the same folds (training rows = the stored inner rows of each outer split; L-IMG enters as its inner / outer prediction)
    from sklearn.linear_model import LogisticRegression
    fdu = sheet().drop_duplicates("combined_name").set_index("combined_name"); BATCH = pd.Series(ids).map(fdu.Batch).values; bc = pd.Series(y).groupby(BATCH).agg(lambda s: s.nunique()); SINGLE = (pd.Series(BATCH).map(bc).values == 1).astype(float)
    SCAN = (SDS.scanner.values == "C13210").astype(float); assert set(SDS.scanner.unique()) <= {"C13210", "C13239-01"}
    MET = np.column_stack([AGE, np.log(TILES), SCAN, SINGLE]); lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    PM = np.full((n, 10), np.nan); PMI = np.full((n, 10), np.nan)
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            si = I["L-IMG"][(r, k)]; tri = si.index.values; te = np.where(FO[:, j] == k)[0]
            for out_, X_tr, X_te in ((PM, MET[tri], MET[te]), (PMI, np.column_stack([MET[tri], lg(si.values)]), np.column_stack([MET[te], lg(O["L-IMG"][te, j])]))):
                mu = X_tr.mean(0); sdv = X_tr.std(0); sdv[sdv == 0] = 1; clf = LogisticRegression(penalty=None, max_iter=5000).fit((X_tr - mu) / sdv, y[tri]); out_[te, j] = clf.predict_proba((X_te - mu) / sdv)[:, 1]
    obs_m = fsa(PM, rows); obs_mi = fsa(PMI, rows); obs_i = fsa(O["L-IMG"], rows); BM = np.array([fsa(PM, b) for b in boots]); BMI = np.array([fsa(PMI, b) for b in boots]); BI = np.array([fsa(O["L-IMG"], b) for b in boots])
    R["c_metadata_alone"] = {"features": ["slide age", "log tissue tiles", "scanner", "batch composition (single-class / mixed)"], "auroc": r3(obs_m), "ci95": ci2(BM),
                             "batch_groups": {"single_class_batches": int((bc == 1).sum()), "mixed_batches": int((bc > 1).sum()), "pre_event_samples_in_single_class_batches": int(SINGLE[rows].sum())},
                             "model": "unpenalised logistic regression on training-standardised features; trained on the outer training rows of each split (all set C samples), scored on the held-out fold; fold-stratified AUROC on pre-event samples, mean over repeats"}
    R["d_incremental"] = {"auroc_L_IMG_plus_metadata": r3(obs_mi), "ci95": ci2(BMI), "auroc_L_IMG_alone": r3(obs_i), "d_L_IMG_plus_metadata_minus_metadata": dstat(obs_mi - obs_m, BMI - BM, one_sided=True), "d_L_IMG_plus_metadata_minus_L_IMG": dstat(obs_mi - obs_i, BMI - BI),
                          "note": "L-IMG enters as the logit of its stored inner out-of-fold prediction on training rows and of its stored outer prediction on the held-out fold"}
    res["R55"] = R; dump(res)
# ================================================================== R56 Killcoyne validation cohort: audit only (no outcome-based score of the slide model)
elif TASK in ("r56prep", "r56"):
    v = pd.read_csv(SWG + "/sWGS_validation_cleaned_Leanne (4) (1).csv", dtype=str); v["PatientID"] = v.PatientID.fillna("").str.strip()
    def ckey(s):
        m = re.search(r"([A-Z]{0,3})\s*(\d{2})[.\s]+0*(\d{3,6})", str(s).upper()); return f"{m.group(2)}.{m.group(3)}" if m else None
    def bkey_slide(b):
        s = str(b).upper().strip()
        if s in ("", "NAN"): return None
        mb = re.match(r"B?\s*(\d+)", s)
        if mb: return "N" + str(int(mb.group(1)))
        ml = re.match(r"B([A-Z])\b", s) or re.match(r"^([A-Z])$", s); return ("L" + ml.group(1)) if ml else None
    def vblock(s):
        s = str(s).upper(); m = re.search(r"(\d{2})[.\s]+0*(\d{3,6})(.*)$", s); rest = m.group(3) if m else ""
        mb = re.search(r"[-\s]B?\s*(\d+)\b", rest)
        if mb: return "N" + str(int(mb.group(1)))
        ml = re.search(r"[-\s]B?([A-Z])\b", rest); return ("L" + ml.group(1)) if ml else None
    sm = pd.read_csv(SWG + "/slide_matching.csv", dtype=str); sm["c"] = sm.PathCaseID.map(ckey); sm["b"] = sm.Block.map(bkey_slide); sm["real"] = ~sm["Slide file"].astype(str).str.startswith("(") & sm["Slide file"].notna()
    v["c"] = v.Block.map(ckey); v["b"] = v.Block.map(vblock); mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str).drop_duplicates(["base", "npz"]); npz_of = mp.groupby("base").npz.first()
    smr = sm[sm.real & sm.c.notna() & sm.b.notna()]; lk = v.merge(smr[["c", "b", "Slide file"]], on=["c", "b"], how="inner"); lk = lk[lk["Slide file"].isin(npz_of.index)].copy(); lk["npz"] = lk["Slide file"].map(npz_of)
    if TASK == "r56prep":
        lk[["sample_id", "npz"]].drop_duplicates().to_csv(R7 + "/val_slides.csv", index=False); res["r56prep"] = {"slides": int(lk.npz.nunique()), "samples": int(lk.sample_id.nunique())}; dump(res)
    else:
        R = {}; vpat = v.groupby("PatientID").Status.agg(lambda s: "P" if (s == "P").any() else ("NP" if (s == "NP").any() else "other"))
        # (a) identity: release rows from this sheet (the release's validation batch) and overlap with the 80
        tm = pd.read_csv(REL + "/training_manifest.csv", dtype=str); mm = pd.read_csv(REL + "/matched_manifest.csv", dtype=str); rel = tm.merge(mm[["sample_id", "cnv_id"]].drop_duplicates("sample_id"), on="sample_id", how="left")
        vids = set(v.sample_id); relv = rel[rel.cnv_id.isin(vids)]; vsamp_in_rel = v.sample_id.isin(set(rel.cnv_id)); vpat_rel = set(v[vsamp_in_rel].PatientID)
        fd = sheet(); ids80 = set(fd[fd.Patient.astype(str).isin(set(pat))]["Hospital_Research_ID_updated"].dropna().str.strip()) | set(fd[fd.Patient.astype(str).isin(set(pat))]["Hospital Research ID"].dropna().str.strip())
        isval = lambda c: os.path.isdir(f"{SWG}/copy_number_hg38/val/500kb/{c}") or os.path.isdir(f"{SWG}/copy_number_hg38/val/perPatient/500kb/{c}")
        rel_val_folder = rel[[isval(c) for c in rel.cnv_id.fillna("")]]
        R["a_identity"] = {"validation_sheet": {"samples": int(v.sample_id.nunique()), "patients": int(v.PatientID.nunique()), "patients_by_status": vpat.value_counts().to_dict()},
                           "release_rows_from_validation_folders": {"rows": int(len(rel_val_folder)), "patients": int(rel_val_folder.cnv_id.map(v.set_index("sample_id").PatientID).nunique())},
                           "validation_samples_in_release": int(vsamp_in_rel.sum()), "validation_patients_with_a_sample_in_release": len(vpat_rel), "release_rows_matched_to_validation_samples": int(len(relv)),
                           "share_validation_patients_in_release_batch": r3(len(vpat_rel) / v.PatientID.nunique()), "validation_patients_overlapping_the_80": int(len(set(v.PatientID) & ids80)),
                           "definition": "release validation batch = release rows (training_manifest.csv) whose CNV sample is a sample of this sheet (docs/dataset_description.md 268-sheet stratum)"}
        rvp = int(R["a_identity"]["release_rows_from_validation_folders"]["patients"]); R["a_identity"]["is_the_rejected_batch"] = bool(len(vpat_rel) > 0 or rvp > 0)
        R["a_identity"]["rule"] = "the sheet is taken to be (or to contain) the rejected batch if any of its patients has a sample among the release rows" 
        # (b) linkage, slides, scanners, tissue tiles by class (slide headers and masks read in the r56px tasks)
        PX = pd.concat([pd.read_csv(f, dtype={"key": str}) for f in sorted(glob.glob(R7 + "/px_val/chunk_*.csv"))]); assert PX.npz.nunique() == lk.npz.nunique(), (PX.npz.nunique(), lk.npz.nunique())
        ls = v[v.sample_id.isin(set(lk.sample_id))].copy(); st = ls.Status.fillna("NA"); ls["cls"] = st
        g = PX.groupby("key"); ls["tiles"] = ls.sample_id.map(g.tissue_tiles.sum()); ls["scanner"] = ls.sample_id.map(g.scanner.agg(lambda s: "/".join(sorted(set(map(str, s))))))
        ls["year"] = pd.to_numeric(ls["Endoscopy Year"], errors="coerce"); ls["batch"] = ls.SLXID1.fillna("NA")
        lpat = ls.groupby("PatientID").Status.agg(lambda s: "P" if (s == "P").any() else ("NP" if (s == "NP").any() else "other"))
        R["b_linkage"] = {"samples_with_slide": int(len(ls)), "samples_with_slide_by_status": st.value_counts().to_dict(), "patients_with_slide_by_status": lpat.value_counts().to_dict(), "slides": int(PX.npz.nunique()),
                          "progressor_patients_with_slide": int((lpat == "P").sum()), "sequencing_runs_SLXID1": int(ls.batch.nunique()),
                          "sequencing_runs_by_status": {c: int(ls[ls.cls == c].batch.nunique()) for c in ("P", "NP")}, "runs_with_both_classes": int((ls[ls.cls.isin(["P", "NP"])].groupby("batch").cls.nunique() == 2).sum()),
                          "scanner_by_status": {c: ls[ls.cls == c].scanner.value_counts().to_dict() for c in ("P", "NP")},
                          "tissue_tiles_by_status": {c: {"median": r3(ls[ls.cls == c].tiles.median()), "iqr": [r3(ls[ls.cls == c].tiles.quantile(0.25)), r3(ls[ls.cls == c].tiles.quantile(0.75))], "n": int((ls.cls == c).sum())} for c in ("P", "NP")},
                          "grade_by_status": {c: ls[ls.cls == c].Pathology2.value_counts().to_dict() for c in ("P", "NP")},
                          "linkage": "sheet Block -> case + block (r2_main.py ckey / vblock) = slide_matching PathCaseID + Block, slide embedded in uni2_npz_map.csv; batch = sequencing run (SLXID1; the sheet has no Batch column)"}
        # (c) metadata alone for status: out-of-fold logistic (scanner, run, tissue tiles, biopsy year), patient-grouped stratified 10-fold x 10 repeats
        from sklearn.linear_model import LogisticRegression
        d = ls[ls.cls.isin(["P", "NP"])].copy(); d = d[d.tiles.notna() & d.year.notna()]; yy = (d.cls == "P").values.astype(int); pp = d.PatientID.values
        X = pd.get_dummies(pd.DataFrame({"scanner": d.scanner.values, "run": d.batch.values}), drop_first=False).values.astype(float); X = np.column_stack([X, np.log1p(d.tiles.values.astype(float)), d.year.values.astype(float)])
        def sfolds(seed):
            pl = pd.Series(yy).groupby(pp).max(); rng = np.random.RandomState(seed); Pp = rng.permutation(pl.index[pl == 1].values); Np = rng.permutation(pl.index[pl == 0].values); o = np.concatenate([Pp, Np]); return dict(zip(o, np.arange(len(o)) % 10))
        PR = np.full((len(d), 10), np.nan); FF = np.zeros((len(d), 10), int)
        for j in range(10):
            fo = sfolds(j + 1); f = np.array([fo[p] for p in pp]); FF[:, j] = f
            for k in range(10):
                te = f == k; tr = ~te; mu = X[tr].mean(0); sdv = X[tr].std(0); sdv[sdv == 0] = 1
                PR[te, j] = LogisticRegression(C=1.0, max_iter=5000).fit((X[tr] - mu) / sdv, yy[tr]).predict_proba((X[te] - mu) / sdv)[:, 1]
        def auc_fs(b):
            out = []
            for j in range(10):
                num = den = 0.0; x = PR[b, j]; f = FF[b, j]; l = yy[b]
                for k in np.unique(f):
                    m = f == k; n1 = int(l[m].sum()); n0 = int(m.sum()) - n1
                    if n1 and n0: rk = rankdata(x[m]); num += rk[l[m] == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
                if den: out.append(num / den)
            return float(np.mean(out))
        up = np.unique(pp); ro = {q: np.where(pp == q)[0] for q in up}; rng = np.random.RandomState(0); BS = []
        while len(BS) < NB:
            b = np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))])
            if 0 < yy[b].sum() < len(b): BS.append(auc_fs(b))
        av = auc_fs(np.arange(len(d))); lo, hi = np.percentile(BS, [2.5, 97.5]); lo1s = np.percentile(BS, 5)
        R["c_metadata_alone"] = {"auroc": r3(av), "ci95": [r3(lo), r3(hi)], "one_sided_95_lower": r3(np.percentile(BS, 5)), "n_samples": int(len(d)), "n_patients": int(len(up)), "n_progressor_patients": int(pd.Series(yy).groupby(pp).max().sum()),
                                 "model": "L2 logistic (C = 1) on scanner and sequencing run (one-hot), log(1 + tissue tiles) and biopsy year, training-standardised; patient-grouped folds stratified by patient status, 10 folds x 10 repeats (seeds 1-10); fold-stratified AUROC, mean over repeats; patient bootstrap 2,000 draws"}
        R["gate"] = {"not_the_rejected_batch": not R["a_identity"]["is_the_rejected_batch"], "at_least_10_progressor_patients_with_slides": bool(R["b_linkage"]["progressor_patients_with_slide"] >= 10),
                     "metadata_auroc_point_below_0.65": bool(av < 0.65), "metadata_auroc_one_sided_lower_95_below_0.65": bool(lo1s < 0.65)}
        R["gate"]["passes"] = bool(all(R["gate"].values())); R["note"] = "audit only; no outcome-based score of the slide model was computed"
        res["R56"] = R; dump(res)
# ================================================================== R57 achievable precision for an ACE-B-like set (simulated from the discovery cohort; no ACE-B data)
elif TASK == "r57":
    O, I, FO, popm, rows, CALLS, band, info = primary(); ND = int(os.environ.get("RV7_ND", 10000)); NPp, NNp = 15, 67; rng = np.random.RandomState(0)
    pr_ = {p: rows[pat[rows] == p] for p in np.unique(pat[rows])}; Pp = [p for p in pr_ if y[pr_[p][0]] == 1]; Np = [p for p in pr_ if y[pr_[p][0]] == 0]
    out = {"sens": [], "spec": [], "pdet": [], "auc": [], "pass": []}
    for _ in range(ND):
        j = rng.randint(10); sp = list(rng.choice(len(Pp), NPp, replace=True)); sn = list(rng.choice(len(Np), NNp, replace=True)); ss = []; ps = []; pl = []
        for lst, plist, lab in ((sp, Pp, 1), (sn, Np, 0)):
            for i in lst:
                s = pr_[plist[i]]; pick = s if len(s) <= 2 else rng.choice(s, 2, replace=False); ss.append(pick); ps.append(O["L-IMG"][pick, j].max()); pl.append(lab)
        C_ = [CALLS["T"][s, j] for s in ss]; pl = np.array(pl); allc = np.concatenate(C_); ally = np.concatenate([np.full(len(s), l) for s, l in zip(ss, pl)])
        out["sens"].append(allc[ally == 1].mean()); out["spec"].append(1 - allc[ally == 0].mean()); det = np.array([c.any() for c in C_])[pl == 1]; out["pdet"].append(det.mean()); out["pass"].append(det.sum() >= 12)
        rk = rankdata(ps); n1 = (pl == 1).sum(); n0 = (pl == 0).sum(); out["auc"].append((rk[pl == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    hw = lambda a: r3((np.percentile(a, 97.5) - np.percentile(a, 2.5)) / 2); R = {"draws": ND, "design": {"progressor_patients": NPp, "non_progressor_patients": NNp, "samples_per_patient": 2}}
    R["a_precision"] = {k: {"mean": r3(np.mean(out[k])), "ci_half_width": hw(out[k]), "range_2.5_97.5": [r3(np.percentile(out[k], 2.5)), r3(np.percentile(out[k], 97.5))]} for k in ("sens", "spec")}
    cohort_det = float(np.mean([np.mean([CALLS["T"][pr_[p], j].any() for p in Pp]) for j in range(10)])); sim_det = float(np.mean(out["pdet"]))
    R["b_pass_rule_ge_12_of_15"] = {"simulated_probability": r3(np.mean(out["pass"])), "mean_detection_rate_in_simulation": r3(sim_det), "cohort_per_patient_detection_rate_all_samples": r3(cohort_det),
                                    "binomial_probability": {f"{pv:.2f}": r3(binom.sf(11, 15, pv)) for pv in (sim_det, cohort_det, 0.70, 0.80, 0.90)}}
    R["c_slide_model_patient_auroc"] = {"mean": r3(np.mean(out["auc"])), "ci_half_width": hw(out["auc"]), "range_2.5_97.5": [r3(np.percentile(out["auc"], 2.5)), r3(np.percentile(out["auc"], 97.5))]}
    R["note"] = "each draw: one repeat at random, 15 progressor and 67 non-progressor patients drawn with replacement from the 32 / 43 pre-event patients, 2 pre-event samples per patient without replacement (all if fewer); per-sample calls = that repeat's stored triage calls; patient detected if any drawn sample is flagged; patient L-IMG score = maximum over its drawn samples; CI half-width = half the 2.5-97.5% range over draws"
    res["R57"] = R; dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
