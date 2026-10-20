"""Review round 7d: sensitivity first (docs/paper_review_round7d_sensitivity_first.md @ 832d184). Rule V0 ("clear on the slide, confirm by sequencing"),
its variants and the comparators re-scored on stored outer and inner out-of-fold predictions (tb_common.py loaders), the R52(a) refit outputs
(killcoyne_mm/review_round7/refit/r52a) and the stored F3 refit-bootstrap draws. Nothing is refitted; no ACE-B file is read; frozen bundles are not touched.
Helpers copied unchanged from rv7_round7.py / rv7b_gate.py / rv7c_gate_rob.py (matched_calls, all_calls, use_set, load_var, cutoff_arrays, groups_of, r49_calls,
sheet) and from clinical_lookup/q6_q1fix.py (same-block labels, read-only).
Row-level files only under killcoyne_mm/review_round7d/ on the cluster; aggregates (no identifiers, no dates) -> results/paper_final/review_round7d/<TASK>.json.
TASK = prereq | boot_<row>_<chunk> (threshold re-derivation bootstrap, draws chunk*NB/10 ...) | row_<row> | curve | f3 | merge
       <row> = primary | ndbe | earliest | ge2 | tis_lt256 | tis_ge256 | r52a | pkg"""
import os, sys, json, glob, re, numpy as np, pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats.mstats import hdquantiles
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_common as TC
from tb_common import *
PRESPEC = "832d184"; OUT = T + "/results/paper_final/review_round7d"; OUT7 = T + "/results/paper_final/review_round7"; OUT7C = T + "/results/paper_final/review_round7c"
R7 = M + "/review_round7"; R7D = M + "/review_round7d"; BOOT = R7D + "/boot"; RR2 = M + "/review_round2"; KD = T + "/feasibility/paper_plan/killcoyne"
SWG = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
NCH = 10; NF3 = 200
if os.environ.get("RV7D_NB"):   # smoke mode: fewer bootstrap draws (and F3 draws), aggregates and draws written aside (never to results/)
    NB = int(os.environ["RV7D_NB"]); TC.NB = NB; NF3 = int(os.environ.get("RV7D_F3N", "5")); OUT = R7D + "/smoke_json"; BOOT = R7D + "/smoke_boot"
for d in (OUT, R7D, BOOT): os.makedirs(d, exist_ok=True)
TASK = os.environ["TASK"]; MM = {"A*": "L-CNV", "B*": "L-LATE", "C*": "L-IMG"}; NSEEDS = 100
ROWS = ("primary", "ndbe", "earliest", "ge2", "tis_lt256", "tis_ge256", "r52a", "pkg")
# costs: the pre-specification's assumptions, used exactly as written
COST = {"sequencing_per_sample_gbp": 85.0, "sequencing_sensitivity_gbp": [150.0, 250.0], "sequencing_failure_rate": 0.03, "flag_gbp": 471.0 + 44.0 + 20.0,
        "flag_items_gbp": {"repeat_endoscopy": 471.0, "expert_pathology_review": 44.0, "p53": 20.0}, "slide_scanning_and_ai_per_slide_gbp": 5.0, "progressor_endoscopy_prevalence": 0.02}
V0Q = {"ql": 0.02, "target": 0.95}   # t_low = 2nd percentile (Harrell-Davis) of training progressor endoscopy scores; two-stage target 95%
# ------------------------------------------------------------------ helpers copied unchanged (rv7_round7.py / rv7b_gate.py / rv7c_gate_rob.py)
def ci2(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return pct(a) if len(a) else [None, None]
def lo1(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return r3(np.percentile(a, 5)) if len(a) else None
def dstat(obs, draws, one_sided=False):
    o = {"delta": r3(obs), "ci95": ci2(draws), "p_two_sided": p2(draws)}
    if one_sided: o["one_sided_95_lower"] = lo1(draws); o["non_inferior_at_-0.05"] = bool(lo1(draws) is not None and lo1(draws) > -0.05)
    return o
def J(path):
    try: return json.load(open(path))
    except FileNotFoundError: return None
def wmean(v, w): return float(v.mean()) if w is None else float((v * w).sum() / w.sum())
def need(files, k): assert len(files) == k, (len(files), k, files[:2]); return files
def nfo(rows): return {"n_samples": int(len(rows)), "n_progressor_samples": int(y[rows].sum()), "n_patients": int(len(set(pat[rows]))), "n_progressor_patients": int(len(set(pat[rows][y[rows] == 1])))}
def cp_ci(k, nn):
    if nn == 0: return [None, None]
    lo = beta_dist.ppf(0.025, k, nn - k + 1) if k > 0 else 0.0; hi = beta_dist.ppf(0.975, k + 1, nn - k) if k < nn else 1.0; return [r3(lo), r3(hi)]
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
def cutoff_arrays(info, FO):
    """per sample x repeat: t_low, c_star and t_seq of the sample's (repeat, fold) in tb_common.triage order (repeat-major, fold 1..10)"""
    TL = np.full((n, 10), np.nan); CSt = np.full((n, 10), np.nan); TS = np.full((n, 10), np.nan)
    for j in range(10):
        for k in range(1, 11):
            i = j * 10 + (k - 1); te = FO[:, j] == k; TL[te, j] = info["t_low"][i]; CSt[te, j] = info["c_star"][i]; TS[te, j] = info["t_seq"][i]
    return TL, CSt, TS
def groups_of(idx):
    """endoscopies of the sample indices idx, in sorted endoscopy order: (keys, list of positions into idx)"""
    ue, inv = np.unique(ENDO[idx], return_inverse=True); return ue, [np.where(inv == g)[0] for g in range(len(ue))]
def r49_calls(O, info, FO, es):
    """rv7_round7.py r49 (unchanged): endoscopy band from max L-IMG against the fold's t_low and c*, middle band -> top-L-IMG sample called by L-LATE at t_seq"""
    TL, CSt, TS = cutoff_arrays(info, FO); img = O["L-IMG"]; late = O["L-LATE"]; E = len(es)
    GB = np.zeros((E, 10), int); GC = np.zeros((E, 10), bool); NS = np.zeros((E, 10), int)
    for e, s in enumerate(es):
        for j in range(10):
            assert len(set(FO[s, j])) == 1; tl, cs, ts = TL[s[0], j], CSt[s[0], j], TS[s[0], j]; sc = img[s, j].max()
            g = (2 if sc > cs else 0) if tl > cs else (0 if sc < tl else (2 if sc > cs else 1)); GB[e, j] = g
            if g == 1: top = s[np.argmax(img[s, j])]; GC[e, j] = late[top, j] >= ts; NS[e, j] = 1
            else: GC[e, j] = g == 2
    return GC, NS, GB
def sheet():
    fd = pd.read_csv(SWG + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; return fd
def dump(res, name=None):
    res["prespec"] = PRESPEC; json.dump(res, open(f"{OUT}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV7D DONE", name or TASK, flush=True)
# ------------------------------------------------------------------ same-block labels: clinical_lookup/q6_q1fix.py (round-7 lookup Q1), unchanged; per analysed sample
def same_block_labels():
    def ckey(s):
        m = re.search(r"([A-Z]{0,3})\s*(\d{2})[.\s]+0*(\d{3,6})", str(s).upper()); return f"{m.group(2)}.{m.group(3)}" if m else None
    def bkey_sheet(pid):
        s = str(pid).upper(); m = re.search(r"(\d{2})[.\s]+0*(\d{3,6})(.*)$", s); rest = m.group(3) if m else s
        mb = re.search(r"B\s*(\d+)", rest) or re.search(r"[ _](\d+)\b", rest)
        if mb: return "N" + str(int(mb.group(1)))
        ml = re.search(r"[ _]B([A-Z])\b", rest); return ("L" + ml.group(1)) if ml else None
    def bkey_slide(b):
        s = str(b).upper().strip()
        if s in ("", "NAN"): return None
        mb = re.match(r"B?\s*(\d+)", s)
        if mb: return "N" + str(int(mb.group(1)))
        ml = re.match(r"B([A-Z])\b", s) or re.match(r"^([A-Z])$", s); return ("L" + ml.group(1)) if ml else None
    fd = sheet(); fd = fd[fd.combined_name.notna()].copy(); fi = fd.set_index("combined_name")
    sm = pd.read_csv(SWG + "/slide_matching.csv", dtype=str); mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(set(ids))].drop_duplicates(["cnv", "npz"]).copy()
    mp = mp.join(sm.drop_duplicates("Slide file").set_index("Slide file"), on="base", rsuffix="_sm")
    mp["s_case"] = mp.cnv.map(fi["Path ID"]).map(ckey); mp["sl_case"] = mp.PathCaseID.map(ckey); mp["s_b"] = mp.cnv.map(fi["Path ID"]).map(bkey_sheet); mp["sl_b"] = mp.Block.map(bkey_slide)
    def cat(r):
        if r.s_case is None or r.sl_case is None: return "case unparseable"
        if r.s_case != r.sl_case: return "different case"
        if r.s_b is None or r.sl_b is None: return "same case, block not recorded on one side"
        return "same case, same block" if r.s_b == r.sl_b else "same case, different block"
    mp["q1"] = [cat(r) for r in mp.itertuples()]
    return mp.groupby("cnv").q1.agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed").reindex(ids).values
# ------------------------------------------------------------------ round-7d machinery
def hd(x, q, w=None):
    """Harrell-Davis quantile (scipy.stats.mstats.hdquantiles); integer weights = replication"""
    x = np.asarray(x, float)
    if w is not None: x = np.repeat(x, w)
    if len(x) == 0: return -np.inf
    if len(x) == 1: return float(x[0])
    return float(np.asarray(hdquantiles(x, prob=[q]))[0])
def bands(sE, tl, th):
    """(high, cleared) masks: cleared = s_E < t_low; high = s_E > t_high (V4 only; th = inf otherwise); t_low > t_high -> no middle band (triage convention)"""
    if tl > th: hi = sE > th; return hi, ~hi
    return sE > th, sE < tl
def thr_v(tr, w, conf="ct", ql=V0Q["ql"], target=V0Q["target"], qh=None):
    """t_low (HD quantile ql of training progressor endoscopy scores), t_high (qh training specificity; V4) and the confirmation threshold: the highest threshold at
    which the two-stage rule as a whole flags >= target of the training progressor endoscopies (cleared -> never flagged, high band -> always flagged)"""
    P = tr["lab"] == 1; sE = tr["sE"]; wP = None if w is None else w[P]
    tl = hd(sE[P], ql, wP); th = c_star(sE[~P], qh, None if w is None else w[~P]) if qh else np.inf
    hi, cl = bands(sE, tl, th); ce = np.where(cl, -np.inf, np.where(hi, np.inf, tr[conf]))
    return tl, th, thr_sens(ce[P], target, wP)
def apply_v(sE, c, tl, th, t):
    hi, cl = bands(sE, tl, th); return ~cl & (hi | (c >= t)), ~cl & ~hi
def roc_spec(score, lab, w, s):
    """specificity at sensitivity s on the empirical ROC of score (positive if score >= threshold), linear interpolation between adjacent operating points;
    w = integer endoscopy multiplicities (0 = not in the evaluation set)"""
    k = w > 0; sc = score[k]; lb = lab[k].astype(float); ww = w[k].astype(float)
    o = np.argsort(-sc, kind="mergesort"); sc = sc[o]; lb = lb[o]; ww = ww[o]
    tp = np.cumsum(ww * lb); fp = np.cumsum(ww * (1 - lb))
    if not len(tp) or tp[-1] == 0 or fp[-1] == 0: return np.nan
    last = np.r_[sc[1:] != sc[:-1], True]
    tpr = np.r_[0.0, tp[last] / tp[-1]]; fpr = np.r_[0.0, fp[last] / fp[-1]]
    i = min(int(np.searchsorted(tpr, s - 1e-12)), len(tpr) - 1)
    if i == 0 or abs(tpr[i] - s) <= 1e-12: f = fpr[i]
    else: f = fpr[i - 1] + (s - tpr[i - 1]) / (tpr[i] - tpr[i - 1]) * (fpr[i] - fpr[i - 1])
    return 1.0 - f
def attrs(V=None):
    """grade, p53 (analysed-sample table 'P53 IHC': 1 aberrant, 0 wild-type, blank missing; as R37), tissue tiles, slides, months before final, years to event"""
    S0 = SM.set_index("Sample").reindex(ids); src = S0 if V is None else V.set_index("Sample").reindex(ids); SD = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids)
    return {"GR": src.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float), "P53": pd.to_numeric(S0["P53 IHC"], errors="coerce").values,
            "TILES": SD.tissue_tiles.values.astype(float), "NSL": SD.n_slides.fillna(1).values.astype(float), "MBF": pd.to_numeric(src.mbf, errors="coerce").values.astype(float), "TIME": pd.to_numeric(src.time, errors="coerce").values.astype(float)}
SUBLAB = {"primary": "all pre-event endoscopies", "ndbe": "NDBE endoscopies (all samples graded NDBE)", "earliest": "earliest pre-event endoscopy per patient", "ge2": "endoscopies with ≥ 2 samples",
          "tis_lt256": "< 256 tissue tiles (by top sample)", "tis_ge256": "≥ 256 tissue tiles (by top sample)", "r52a": "R52(a) cohort (refit predictions)", "pkg": "package copy-number features"}
def setup(row):
    """predictions, training population (all pre-event samples), the endoscopies, their outer score arrays, the evaluation subset and the per-fold training endoscopies"""
    if row == "r52a": V = pd.read_csv(R7 + "/sets/r52a.csv", dtype={"Sample": str, "Patient": str, "Endoscopy": str}); use_set(V); O, I, FO = load_var("r52a"); A = attrs(V)
    else: O, I, FO = load("pkg" if row == "pkg" else "their"); A = attrs()
    popm = PRE.copy(); rows = np.where(popm)[0]; eu, gi = groups_of(rows); es = [rows[g] for g in gi]; E = len(es); S = {"O": O, "I": I, "FO": FO, "popm": popm, "rows": rows, "eu": eu, "es": es, "E": E, "A": A}
    elab = np.array([int(y[s].max()) for s in es]); assert all(len(set(y[s])) == 1 for s in es); epat = np.array([pat[s][0] for s in es]); nsz = np.array([len(s) for s in es])
    up = np.unique(epat); pidx = np.searchsorted(up, epat); assert (up[pidx] == epat).all(); plab = np.array([int(elab[pidx == i].max()) for i in range(len(up))])
    INC = np.zeros((len(up), E)); INC[pidx, np.arange(E)] = 1
    img, late, cnv = O["L-IMG"], O["L-LATE"], O["L-CNV"]; cj = np.arange(10)
    for s in es: assert all(len(set(FO[s, j])) == 1 for j in range(10))
    SE = np.stack([img[s].max(0) for s in es]); TOP = np.stack([s[img[s].argmax(0)] for s in es]); EF = np.stack([FO[s[0]] for s in es])
    SC = {"H": SE, "SLATE": np.stack([late[s].max(0) for s in es]), "SCNV": np.stack([cnv[s].max(0) for s in es])}
    hasp = np.isfinite(A["P53"]); ab = np.where(hasp, A["P53"], 0) == 1; dys = (A["GR"] == 1) | (A["GR"] == 2)
    es53 = [s[hasp[s]] for s in es]; sub53 = np.array([len(x) > 0 for x in es53]); x53 = [x if len(x) else s[:1] for x, s in zip(es53, es)]
    SE53 = np.stack([img[x].max(0) for x in x53]); TOP53 = np.stack([x[img[x].argmax(0)] for x in x53])
    sub = np.ones(E, bool)
    if row == "ndbe": sub = np.array([NDBE[s].all() for s in es])
    elif row == "earliest":
        emb = np.array([A["MBF"][s].max() for s in es]); sub = np.zeros(E, bool)
        for i in range(len(up)): idx = np.where(pidx == i)[0]; sub[idx[np.argmax(emb[idx])]] = True
    elif row == "ge2": sub = nsz >= 2
    elif row.startswith("tis_"):
        mimg = img.mean(1); top = np.array([s[np.argmax(mimg[s])] for s in es]); sub = (A["TILES"][top] < 256) if row == "tis_lt256" else (A["TILES"][top] >= 256)
    PICK = np.zeros((NSEEDS, E), int)
    for sd in range(NSEEDS):
        rng = np.random.RandomState(sd)
        for e, s in enumerate(es): PICK[sd, e] = s[rng.randint(len(s))]   # R41 draw order (rv6_round6.py r41, rv7_round7.py r46)
    TR = {}
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            si = I["L-IMG"][(r, k)]; tri = si.index.values[popm[si.index.values]]; im = si.reindex(tri).values; cv = I["L-CNV"][(r, k)].reindex(tri).values; lt = I["L-LATE"][(r, k)].reindex(tri).values
            assert np.isfinite(im).all() and np.isfinite(cv).all() and np.isfinite(lt).all() and not (FO[tri, j] == k).any()
            _, g = groups_of(tri); tp = np.array([x[np.argmax(im[x])] for x in g]); tpat = np.array([pat[tri][x[0]] for x in g]); pi_ = np.searchsorted(up, tpat); assert (up[pi_] == tpat).all()
            TR[(j, k)] = {"sE": np.array([im[x].max() for x in g]), "lab": np.array([int(y[tri][x].max()) for x in g]), "pi": pi_, "ct": cv[tp], "lt": lt[tp], "cm": np.array([cv[x].max() for x in g])}
    spi = np.searchsorted(up, pat); spi = np.where((spi < len(up)) & (up[np.minimum(spi, len(up) - 1)] == pat), spi, -1)
    S.update({"elab": elab, "epat": epat, "nsz": nsz, "up": up, "P_": len(up), "pidx": pidx, "plab": plab, "INC": INC, "SE": SE, "TOP": TOP, "EF": EF, "SC": SC,
              "CT": cnv[TOP, cj], "LT": late[TOP, cj], "CM": SC["SCNV"], "sub": sub, "sub53": sub53, "SE53": SE53, "AB53": ab[TOP53], "CT53": cnv[TOP53, cj],
              "OVR": np.array([(dys[s] | ab[s]).any() for s in es]), "GRD": np.array([dys[s].any() for s in es]), "CC53": np.array([(dys[x] | ab[x]).any() if len(x) else False for x in es53]),
              "LS": late[PICK], "TR": TR, "spi": spi, "ab": ab, "hasp": hasp, "row": row,
              "emb": np.array([A["MBF"][s].max() for s in es]), "etime": np.array([A["TIME"][s].max() for s in es]), "eslides": np.array([A["NSL"][s].sum() for s in es]), "endo53_n": np.array([len(x) for x in es53])})
    return S
RULES = ("V0", "V1", "V2", "V4", "V0ovr", "R49", "Vp53", "V0p53")
MATCH = {"V0": ("H", "SLATE", "SCNV", "R1"), "V1": ("H", "SLATE"), "V2": ("H", "SLATE"), "V4": ("H", "SLATE"), "V0ovr": ("H", "SLATE"), "R49": ("H", "SLATE")}
def derive_all(S, m=None, ql=V0Q["ql"], target=V0Q["target"], rules=RULES, keep_thr=False):
    """calls (E x 10) and samples sequenced of every rule with all thresholds derived from the training endoscopies; m = patient multiplicities (None = observed)"""
    E = S["E"]; RU = {nm: [np.zeros((E, 10), bool), np.zeros((E, 10), int)] for nm in rules}; Z53 = np.full((E, 10), -np.inf); THR = []
    for j in range(10):
        for k in range(1, 11):
            tr = S["TR"][(j, k)]; w = None if m is None else m[tr["pi"]]; idx = S["EF"][:, j] == k; se = S["SE"][idx, j]
            tl, th, t = thr_v(tr, w, "ct", ql, target); F, sq = apply_v(se, S["CT"][idx, j], tl, th, t)
            if "V0" in RU: RU["V0"][0][idx, j] = F; RU["V0"][1][idx, j] = sq
            if "V0ovr" in RU: ov = S["OVR"][idx]; RU["V0ovr"][0][idx, j] = F | ov; RU["V0ovr"][1][idx, j] = sq & ~ov
            if "Vp53" in RU:
                s53 = S["SE53"][idx, j]; RU["Vp53"][0][idx, j] = (s53 >= tl) & S["AB53"][idx, j]; RU["Vp53"][1][idx, j] = s53 >= tl   # p53 read (no sequencing)
                RU["V0p53"][0][idx, j] = (s53 >= tl) & (S["CT53"][idx, j] >= t); RU["V0p53"][1][idx, j] = s53 >= tl; Z53[idx, j] = np.where(s53 >= tl, S["CT53"][idx, j], -np.inf)
            if keep_thr:
                P = tr["lab"] == 1; bl = P & (tr["sE"] < tl); Ft, _ = apply_v(tr["sE"], tr["ct"], tl, th, t); oP = idx & (S["elab"] == 1) & S["sub"]
                THR.append({"rep": j + 1, "fold": k, "t_low": float(tl), "t_cnv": float(t), "n_train_progressor_endoscopies": int(P.sum()), "n_train_progressor_endoscopies_below_t_low": int(bl.sum()),
                            "n_train_progressor_patients_below_t_low": int(len(set(tr["pi"][bl]))), "train_two_stage_sensitivity": float(Ft[P].mean()), "train_step1_sensitivity": float((tr["sE"][P] >= tl).mean()),
                            "infeasible_flag_all_uncleared": bool(t == -np.inf), "n_outer_progressor_endoscopies": int(oP.sum()),
                            "outer_two_stage_sensitivity": float(RU["V0"][0][oP, j].mean()) if oP.any() else None, "outer_step1_sensitivity": float((S["SE"][oP, j] >= tl).mean()) if oP.any() else None})
            if "V1" in RU: tl1, th1, t1 = thr_v(tr, w, "lt", ql, target); F, sq = apply_v(se, S["LT"][idx, j], tl1, th1, t1); RU["V1"][0][idx, j] = F; RU["V1"][1][idx, j] = sq
            if "V2" in RU: tl2, th2, t2 = thr_v(tr, w, "cm", ql, target); F, sq = apply_v(se, S["CM"][idx, j], tl2, th2, t2); RU["V2"][0][idx, j] = F; RU["V2"][1][idx, j] = sq * S["nsz"][idx]
            if "V4" in RU: tl4, th4, t4 = thr_v(tr, w, "ct", ql, target, qh=0.90); F, sq = apply_v(se, S["CT"][idx, j], tl4, th4, t4); RU["V4"][0][idx, j] = F; RU["V4"][1][idx, j] = sq
    if "R49" in RU:
        W = None if m is None else np.where(S["spi"] >= 0, m[np.maximum(S["spi"], 0)], 0)
        _, _, info = TC.triage(S["O"], S["I"], S["FO"], S["popm"], W=W); GC, NS, _ = r49_calls(S["O"], info, S["FO"], S["es"]); RU["R49"] = [GC, NS]
    RU = {k: (v[0], v[1]) for k, v in RU.items()}
    return (RU, Z53, THR) if keep_thr else (RU, Z53)
def stats(S, RU, Z53, m=None):
    """per-repeat endoscopy sens / spec, shares sequenced, patient detection and matched comparator specificities on the evaluation subset (multiplicities m)"""
    mm = np.ones(S["P_"]) if m is None else m.astype(float); w = mm[S["pidx"]] * S["sub"]; P = S["elab"] == 1; o = {}
    w53 = w * S["sub53"]; pw = mm * (S["INC"] @ (w > 0) > 0); pwP = pw * (S["plab"] == 1); pwN = pw * (S["plab"] == 0)
    for nm, (F, NS) in RU.items():
        ww = w53 if nm in ("Vp53", "V0p53") else w; wP = ww * P; wN = ww * ~P
        o[f"sens_{nm}"] = (F * wP[:, None]).sum(0) / wP.sum(); o[f"spec_{nm}"] = 1 - (F * wN[:, None]).sum(0) / wN.sum()
        o[f"she_{nm}"] = ((NS > 0) * ww[:, None]).sum(0) / ww.sum()
        if nm not in ("Vp53", "V0p53"):
            o[f"shs_{nm}"] = (NS * ww[:, None]).sum(0) / (S["nsz"] * ww).sum()
            PF = (S["INC"] @ (F * (ww > 0)[:, None])) > 0; o[f"pdet_{nm}"] = (PF * pwP[:, None]).sum(0) / pwP.sum(); o[f"pnp_{nm}"] = (PF * pwN[:, None]).sum(0) / pwN.sum()
    for nm, comps in MATCH.items():
        if nm not in RU: continue
        s_ = o[f"sens_{nm}"]
        for c in comps:
            if c == "R1": o[f"spec_R1_at_{nm}"] = np.array([np.mean([roc_spec(S["LS"][sd, :, j], P, w, s_[j]) for sd in range(NSEEDS)]) for j in range(10)])
            else: o[f"spec_{c}_at_{nm}"] = np.array([roc_spec(S["SC"][c][:, j], P, w, s_[j]) for j in range(10)])
    if "Vp53" in RU: o["spec_V0p53m_at_Vp53"] = np.array([roc_spec(Z53[:, j], P, w53, o["sens_Vp53"][j]) for j in range(10)])
    return o
def draws(S, nb=None):
    """patient bootstrap (RandomState(0), the R49 draw loop): multiplicities over the population's patients; draws whose evaluation subset is one-class are rejected"""
    nb = NB if nb is None else nb; rng = np.random.RandomState(0); P = S["elab"] == 1; out = []
    while len(out) < nb:
        pidx = rng.choice(S["P_"], S["P_"]); mm = np.bincount(pidx, minlength=S["P_"]); we = mm[S["pidx"]] * S["sub"]
        if (we * P).sum() > 0 and (we * ~P).sum() > 0: out.append(mm)
    return out
def stack(L): return {k: np.stack([d[k] for d in L]) for k in L[0]}
res = {"task": TASK}
# ================================================================== prerequisites (stored arrays only; nothing refitted)
if TASK == "prereq":
    R = {}
    def chk(nm, O, I, FO):
        o = {"outer_finite": bool(all(np.isfinite(O[m][PRE]).all() for m in ("L-CNV", "L-IMG"))), "outer_repeats": int(O["L-IMG"].shape[1]), "inner_repeat_folds_img": len(I["L-IMG"]), "inner_repeat_folds_cnv": len(I["L-CNV"])}
        bad = 0; leak = 0
        for j, r in enumerate(REPS):
            for k in range(1, 11):
                si = I["L-IMG"][(r, k)]; tri = si.index.values[PRE[si.index.values]]; leak += int((FO[tri, j] == k).sum())
                bad += int(not (np.isfinite(si.reindex(tri).values).all() and np.isfinite(I["L-CNV"][(r, k)].reindex(tri).values).all()))
        o["repeat_folds_with_missing_inner_values"] = bad; o["inner_rows_in_own_outer_fold"] = leak
        o["available"] = bool(o["outer_finite"] and o["outer_repeats"] == 10 and o["inner_repeat_folds_img"] == 100 and o["inner_repeat_folds_cnv"] == 100 and bad == 0 and leak == 0)
        o["pre_event_samples"] = int(PRE.sum()); return o
    O, I, FO = load("their"); R["L-IMG / L-CNV / L-LATE (stored, cfg_00 / cfg_02 outer; horizons/inner img and cnv_their)"] = chk("their", O, I, FO)
    A = attrs(); CS = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample").reindex(ids); p_c = pd.to_numeric(CS["P53 IHC"], errors="coerce").values
    R["p53_field"] = {"available_pre_event_samples": int(np.isfinite(A["P53"][PRE]).sum()), "aberrant_pre_event_samples": int((A["P53"][PRE] == 1).sum()), "equal_to_endoscopy_table_field": bool(np.array_equal(np.nan_to_num(A["P53"], nan=-9), np.nan_to_num(p_c, nan=-9)))}
    O, I, FO = load("pkg"); R["package L-CNV (cfg_01 outer; horizons/inner cnv_pkg)"] = chk("pkg", O, I, FO)
    try:
        V = pd.read_csv(R7 + "/sets/r52a.csv", dtype={"Sample": str, "Patient": str, "Endoscopy": str}); use_set(V); O, I, FO = load_var("r52a"); R["R52(a) refit (review_round7/refit/r52a)"] = chk("r52a", O, I, FO)
    except AssertionError as e: R["R52(a) refit (review_round7/refit/r52a)"] = {"available": False, "error": str(e)[:200]}
    fl = lambda m_, d, p: f"{RR2}/f3/{m_}_draw_{d:03d}_{p}.csv"; have = [d for d in range(200) if all(os.path.exists(fl(m_, d, p)) for m_ in ("cnv_their", "img") for p in ("outer", "inner"))]
    R["refit bootstrap draws (review_round2/f3)"] = {"draws_with_outer_and_inner_cnv_and_img": len(have), "of": 200, "available": len(have) == 200}
    R["R41 random-one seeds"] = {"available": True, "note": "seeds 0-99 regenerated by the R41 draw order (RandomState(seed).randint per endoscopy); no stored file needed"}
    res["prereq"] = R; dump(res)
# ================================================================== threshold re-derivation bootstrap: one chunk of draws
elif TASK.startswith("boot_"):
    row, c = TASK[5:].rsplit("_", 1); c = int(c); S = setup(row); D = draws(S); lo_, hi_ = c * NB // NCH, (c + 1) * NB // NCH; out = []
    for i in range(lo_, hi_): RU, Z = derive_all(S, D[i]); out.append(stats(S, RU, Z, D[i]))
    np.savez(f"{BOOT}/{row}_{c:02d}.npz", lo=lo_, hi=hi_, **stack(out)); print("RV7D BOOT DONE", row, c, lo_, hi_, flush=True)
# ================================================================== one row: observed, fixed-threshold bootstrap, threshold re-derivation bootstrap (chunks), aggregates
elif TASK.startswith("row_"):
    row = TASK[4:]; S = setup(row); RU, Z53, THR = derive_all(S, keep_thr=True); obs = stats(S, RU, Z53); D = draws(S)
    DF = stack([stats(S, RU, Z53, m) for m in D])
    ZB = [np.load(f) for f in need(sorted(glob.glob(f"{BOOT}/{row}_*.npz")), NCH)]; assert sum(int(z["hi"]) - int(z["lo"]) for z in ZB) == NB
    DR = {k: np.concatenate([z[k] for z in ZB]) for k in obs}
    m_ = lambda k: float(np.mean(obs[k])); dmF = lambda k: DF[k].mean(1); dmR = lambda k: DR[k].mean(1)
    def rule(nm):
        o = {"endoscopy_sensitivity": r3(m_(f"sens_{nm}")), "sens_ci95_rederived": ci2(dmR(f"sens_{nm}")), "sens_ci95_fixed": ci2(dmF(f"sens_{nm}")),
             "endoscopy_specificity": r3(m_(f"spec_{nm}")), "spec_ci95_rederived": ci2(dmR(f"spec_{nm}")), "spec_ci95_fixed": ci2(dmF(f"spec_{nm}")), "share_endoscopies_sequenced": r3(m_(f"she_{nm}"))}
        if f"shs_{nm}" in obs: o["share_samples_sequenced"] = r3(m_(f"shs_{nm}")); o["share_samples_ci95_rederived"] = ci2(dmR(f"shs_{nm}")); o["patient_detection"] = r3(m_(f"pdet_{nm}")); o["patient_detection_ci95_rederived"] = ci2(dmR(f"pdet_{nm}")); o["patient_detection_ci95_fixed"] = ci2(dmF(f"pdet_{nm}")); o["non_progressor_patients_flagged_share"] = r3(m_(f"pnp_{nm}"))
        return o
    def contrast(a, b, one_sided=False):
        o = {"delta": r3(np.mean(obs[a] - obs[b])), "per_repeat": [r3(v) for v in obs[a] - obs[b]], "rederived": dstat(np.mean(obs[a] - obs[b]), (DR[a] - DR[b]).mean(1), one_sided), "fixed": dstat(np.mean(obs[a] - obs[b]), (DF[a] - DF[b]).mean(1), one_sided)}
        o["comparator_specificity"] = r3(m_(b)); o["comparator_spec_ci95_rederived"] = ci2(dmR(b)); return o
    E = S["E"]; sub = S["sub"]; rows_ = np.concatenate([s for s, k in zip(S["es"], sub) if k])
    R = {"row": row, "label": SUBLAB[row], **nfo(rows_), "n_endoscopies": int(sub.sum()), "n_progressor_endoscopies": int((S["elab"][sub] == 1).sum()), "draws_fixed": len(D), "draws_rederived": int(len(DR["sens_V0"])),
         "rules": {nm: rule(nm) for nm in RULES}}
    R["n_p53_subset"] = {"endoscopies": int((sub & S["sub53"]).sum()), "progressor_endoscopies": int((sub & S["sub53"] & (S["elab"] == 1)).sum()), "samples": int(S["endo53_n"][sub].sum())}
    R["comparators"] = {nm: {c: {"specificity": r3(m_(f"spec_{c}_at_{nm}")), "spec_ci95_rederived": ci2(dmR(f"spec_{c}_at_{nm}")), "spec_ci95_fixed": ci2(dmF(f"spec_{c}_at_{nm}"))} for c in cs} for nm, cs in MATCH.items()}
    R["contrasts"] = {f"{nm}_minus_{c}": contrast(f"spec_{nm}", f"spec_{c}_at_{nm}", one_sided=(c == "SLATE")) for nm, cs in MATCH.items() for c in cs}
    R["contrasts"]["Vp53_minus_V0p53_matched"] = contrast("spec_Vp53", "spec_V0p53m_at_Vp53")
    # primary contrast and key secondary (fixed sequence)
    pc = R["contrasts"]["V0_minus_H"]; lo = pc["rederived"]["ci95"][0]; sv = m_("sens_V0"); ok = bool(lo is not None and lo > 0 and sv >= 0.90)
    R["primary_contrast"] = {"delta": pc["delta"], "ci95_rederived": pc["rederived"]["ci95"], "ci95_fixed": pc["fixed"]["ci95"], "v0_endoscopy_sensitivity": r3(sv), "lower_limit_above_0": bool(lo is not None and lo > 0),
                             "sensitivity_condition_met": bool(sv >= 0.90), "success": ok, "clinically_relevant_delta_ge_0.05": bool(np.mean(obs["spec_V0"] - obs["spec_H_at_V0"]) >= 0.05)}
    ks = R["contrasts"]["V0_minus_SLATE"]
    R["key_secondary"] = {"delta": ks["delta"], "one_sided_95_lower_rederived": ks["rederived"]["one_sided_95_lower"], "one_sided_95_lower_fixed": ks["fixed"]["one_sided_95_lower"], "ci95_rederived": ks["rederived"]["ci95"], "ci95_fixed": ks["fixed"]["ci95"],
                          "non_inferior_at_-0.05_rederived": ks["rederived"]["non_inferior_at_-0.05"], "status": "tested" if ok else "not tested (primary failed)"}
    # between-repeat variance (R33 / R15 rule) on the primary contrast, threshold re-derivation draws
    pts = obs["spec_V0"] - obs["spec_H_at_V0"]; avg = (DR["spec_V0"] - DR["spec_H_at_V0"]).mean(1); vb = float(np.var(avg, ddof=1)); vr = float(np.var(pts, ddof=1)); dd = float(pts.mean())
    R["between_repeat"] = {"delta_repeat_mean": r3(dd), "V_boot": round(vb, 6), "V_rep": round(vr, 6), "variance_inflated_ci95": [r3(dd - 1.96 * np.sqrt(vb + vr)), r3(dd + 1.96 * np.sqrt(vb + vr))], "per_repeat_delta": [r3(v) for v in pts]}
    if row == "primary":
        P = S["elab"] == 1; ALL = np.ones(E, bool)
        # ---- checks: R49 through this code; (H) at R49's sensitivity vs C*-endo; S-LATE at B*-endo's sensitivity
        CALLS, band, info, thr, sstar = all_calls(S["O"], S["I"], S["FO"], S["popm"]); anyE = lambda C: np.array([C[s].any(0) for s in S["es"]])
        BE = anyE(CALLS["B*"]); CE = anyE(CALLS["C*"]); se_ = lambda F: (F[P].mean(0), 1 - F[~P].mean(0))
        r49 = J(f"{OUT7}/r49.json")["R49"]["rules"]["gate"]; g = R["rules"]["R49"]
        bs, bp = se_(BE); cs_, cp_ = se_(CE); rs = obs["sens_R49"]; w1 = np.ones(E)
        hR = np.array([roc_spec(S["SC"]["H"][:, j], P, w1, rs[j]) for j in range(10)]); hC = np.array([roc_spec(S["SC"]["H"][:, j], P, w1, cs_[j]) for j in range(10)]); sB = np.array([roc_spec(S["SC"]["SLATE"][:, j], P, w1, bs[j]) for j in range(10)])
        R["checks"] = {"r49_vs_round7": {"got": [g["endoscopy_sensitivity"], g["endoscopy_specificity"], g["share_samples_sequenced"], g["share_endoscopies_sequenced"], g["sens_ci95_fixed"]],
                                         "expected": [r49["endoscopy_sensitivity"], r49["endoscopy_specificity"], r49["share_samples_sequenced"], r49["share_endoscopies_sequenced"], r49["sens_ci95"]]},
                       "H_at_R49_sensitivity_vs_Cstar_endo": {"R49_sensitivity": r3(rs.mean()), "H_specificity_evaluation_matched": r3(hR.mean()), "Cstar_endo_sens_spec_training_matched": [r3(cs_.mean()), r3(cp_.mean())], "H_specificity_at_Cstar_endo_sensitivity": r3(hC.mean())},
                       "SLATE_at_Bstar_endo_sensitivity": {"Bstar_endo_sens_spec": [r3(bs.mean()), r3(bp.mean())], "SLATE_specificity_evaluation_matched": r3(sB.mean()), "expected_Bstar_endo": [0.923, 0.530]}}
        c = R["checks"]["r49_vs_round7"]; c["reproduced"] = bool(all(abs(a - b) < 0.0015 for a, b in zip(c["got"][:4], c["expected"][:4])) and all(abs(a - b) < 0.0015 for a, b in zip(c["got"][4], c["expected"][4])))
        c2 = R["checks"]["SLATE_at_Bstar_endo_sensitivity"]; c2["Bstar_endo_reproduced"] = bool(abs(c2["Bstar_endo_sens_spec"][0] - 0.923) < 0.0015 and abs(c2["Bstar_endo_sens_spec"][1] - 0.530) < 0.0015)
        c2["SLATE_minus_Bstar_endo_spec"] = r3(sB.mean() - bp.mean()); c3 = R["checks"]["H_at_R49_sensitivity_vs_Cstar_endo"]; c3["H_minus_Cstar_endo_spec"] = r3(hR.mean() - cp_.mean())
        # ---- binary calls at matched sensitivity for the descriptive tables: largest threshold reaching the rule's per-repeat endoscopy sensitivity on the unit
        def bincall(score, s_, unit):
            F = np.zeros((E, 10), bool)
            for j in range(10): t = thr_sens(score[unit & P, j], s_[j] - 1e-9); F[:, j] = score[:, j] >= t
            return F
        def sens_on(F, unit): return F[unit & P].mean(0)
        CAL = {"V0": RU["V0"][0], "V0ovr": RU["V0ovr"][0], "H": bincall(S["SC"]["H"], obs["sens_V0"], ALL), "SLATE": bincall(S["SC"]["SLATE"], obs["sens_V0"], ALL), "R49": RU["R49"][0], "GP53": np.repeat(S["OVR"][:, None], 10, 1), "GRADE": np.repeat(S["GRD"][:, None], 10, 1)}
        SEQ = {"V0": RU["V0"][1], "V0ovr": RU["V0ovr"][1], "H": np.zeros((E, 10), int), "SLATE": np.repeat(S["nsz"][:, None], 10, 1), "R49": RU["R49"][1], "GP53": np.zeros((E, 10), int), "GRADE": np.zeros((E, 10), int)}
        TL_ = ["V0", "V0ovr", "H", "SLATE", "R49", "GP53"]
        # ---- samples (secondary unit): a sample takes its endoscopy's call
        ys = np.concatenate([y[s] for s in S["es"]]); rep = lambda F: np.concatenate([np.repeat(F[e:e + 1], len(s), 0) for e, s in enumerate(S["es"])])
        R["sample_unit"] = {nm: {"sensitivity": r3(rep(CAL[nm])[ys == 1].mean()), "specificity": r3(1 - rep(CAL[nm])[ys == 0].mean())} for nm in TL_}
        R["sample_unit"]["note"] = "each sample takes its endoscopy's call; (H) and S-LATE at the binary threshold reaching V0's per-repeat endoscopy sensitivity"
        # ---- patient level (bias-corrected, descriptive, unmatched)
        up, pidx, plab = S["up"], S["pidx"], S["plab"]; NP = plab == 0; PP = plab == 1; emb = S["emb"]
        npy = sum((emb[pidx == i].max() - emb[pidx == i].min()) / 12.0 for i in np.where(NP)[0])
        last2 = np.zeros(E, bool)
        for i in range(len(up)): idx = np.where(pidx == i)[0]; last2[idx[np.argsort(emb[idx], kind="mergesort")[:2]]] = True
        def patient(F):
            PF = (S["INC"] @ F) > 0; share = np.stack([F[pidx == i].mean(0) for i in range(len(up))]); P2 = (S["INC"] @ (F & last2[:, None])) > 0
            det = PF[PP].sum(0); med = int(np.floor(np.median(det)))
            return {"progressor_patients_detected_per_repeat": {"median": float(np.median(det)), "min": int(det.min()), "max": int(det.max()), "of": int(PP.sum())}, "exact_ci95_median_repeat": cp_ci(med, int(PP.sum())),
                    "progressor_patients_detected_mean": round(float(det.mean()), 2), "non_progressor_patients_ever_flagged_mean": round(float(PF[NP].sum(0).mean()), 2), "of_non_progressor": int(NP.sum()),
                    "per_patient_share_of_endoscopies_flagged_equal_weight": {"progressor": r3(share[PP].mean()), "non_progressor": r3(share[NP].mean())},
                    "ever_flagged_last2_endoscopies": {"progressor_share": r3(P2[PP].mean()), "non_progressor_share": r3(P2[NP].mean()), "non_progressor_mean_count": round(float(P2[NP].sum(0).mean()), 2)},
                    "flagged_non_progressor_endoscopies_per_patient_year": r3(F[~P].sum(0).mean() / npy)}
        R["patient_level"] = {nm: patient(CAL[nm]) for nm in TL_}
        R["patient_level"]["V0"]["detected_proportion_ci95_rederived"] = ci2(dmR("pdet_V0")); R["patient_level"]["V0"]["detected_proportion_ci95_fixed"] = ci2(dmF("pdet_V0")); R["patient_level"]["V0"]["detected_proportion"] = r3(m_("pdet_V0"))
        R["patient_level"]["n"] = {"patients": int(len(up)), "progressor": int(PP.sum()), "non_progressor": int(NP.sum()), "non_progressor_patient_years": r3(npy), "endoscopies_per_patient_mean": {"progressor": r3(np.bincount(pidx)[PP].mean()), "non_progressor": r3(np.bincount(pidx)[NP].mean())}}
        # ---- timing (descriptive)
        et = S["etime"]; tb = {"<2": P & (et < 2), "2-5": P & (et >= 2) & (et <= 5), ">5": P & (et > 5)}
        def timing(F, seq=None):
            o = {"endoscopy_sensitivity_by_years_to_event": {b: {"n_endoscopies": int(mk.sum()), "sensitivity": r3(F[mk].mean()) if mk.any() else None} for b, mk in tb.items()}}
            first, only_last, only_last_ge2, missed = [], [], [], []
            for j in range(10):
                fj, ol, ol2, ms = [], 0, 0, 0
                for i in np.where(PP)[0]:
                    idx = np.where(pidx == i)[0]; fl_ = idx[F[idx, j]]
                    if not len(fl_): ms += 1; continue
                    fj.append(float(et[fl_].max())); lastE = idx[np.argmin(emb[idx])]
                    if set(fl_) == {lastE}: ol += 1; ol2 += int(len(idx) >= 2)
                first.append(fj); only_last.append(ol); only_last_ge2.append(ol2); missed.append(ms)
            allf = np.concatenate([np.array(f) for f in first])
            o["years_first_flag_to_event"] = {"median_of_per_repeat_medians": r3(np.mean([np.median(f) for f in first])), "pooled_median": r3(np.median(allf)), "pooled_iqr": [r3(np.percentile(allf, 25)), r3(np.percentile(allf, 75))], "pooled_range": [r3(allf.min()), r3(allf.max())]}
            o["progressor_patients_first_flagged_only_at_last_pre_event_endoscopy_mean"] = round(float(np.mean(only_last)), 2); o["of_which_patients_with_ge2_endoscopies_mean"] = round(float(np.mean(only_last_ge2)), 2)
            o["missed_progressor_patients_mean"] = round(float(np.mean(missed)), 2); o["missed_progressor_patients_per_repeat_range"] = [int(min(missed)), int(max(missed))]
            if seq is not None:   # progressor endoscopies cleared by copy number at step 2: is a later pre-event endoscopy of the patient flagged?
                n_, lat, later_nf, none_ = [], [], [], []
                for j in range(10):
                    cc = np.where(P & seq[:, j] & ~F[:, j])[0]; a = b = c_ = 0
                    for e in cc:
                        idx = np.where((pidx == pidx[e]) & (emb < emb[e]))[0]
                        if not len(idx): c_ += 1
                        elif F[idx, j].any(): a += 1
                        else: b += 1
                    n_.append(len(cc)); lat.append(a); later_nf.append(b); none_.append(c_)
                o["progressor_endoscopies_cleared_by_copy_number"] = {"mean_per_repeat": round(float(np.mean(n_)), 2), "later_endoscopy_flagged_mean": round(float(np.mean(lat)), 2), "later_endoscopy_not_flagged_mean": round(float(np.mean(later_nf)), 2), "no_later_pre_event_endoscopy_mean": round(float(np.mean(none_)), 2)}
            return o
        R["timing"] = {nm: timing(CAL[nm], SEQ[nm].astype(bool) if nm in ("V0", "V0ovr") else None) for nm in TL_}
        # ---- clinical output: per 1,000 NDBE surveillance endoscopies (class-conditional rates on the NDBE endoscopies, projected to 2% prevalence)
        U = np.array([NDBE[s].all() for s in S["es"]]); sv0 = sens_on(RU["V0"][0], U)
        CALN = {**CAL, "H": bincall(S["SC"]["H"], sv0, U), "SLATE": bincall(S["SC"]["SLATE"], sv0, U)}; pv = COST["progressor_endoscopy_prevalence"]; fr = COST["sequencing_failure_rate"]
        UP = sorted(set(pidx[U & P])); eP = (U & P).sum() / len(UP)
        def clin(nm, scan=True):
            F = CALN[nm]; NS = SEQ[nm]; prob = F + (~F) * (1 - (1 - fr) ** NS)
            cls = lambda mk: {"flag": float(prob[mk].mean()), "flag_no_failure": float(F[mk].mean()), "seqE": float((NS[mk] > 0).mean()), "seqS": float(NS[mk].mean()), "slides": float(S["eslides"][mk].mean()) if scan else 0.0}
            a, b = cls(U & P), cls(U & ~P); per = lambda k_: 1000 * (pv * a[k_] + (1 - pv) * b[k_])
            pdet = float(np.mean([len(set(pidx[U & P & F[:, j]])) / len(UP) for j in range(10)]))
            o = {"endoscopies_with_sequencing": round(per("seqE"), 1), "samples_sequenced": round(per("seqS"), 1), "endoscopies_flagged": round(per("flag"), 1), "endoscopies_flagged_without_failures": round(per("flag_no_failure"), 1),
                 "progressor_endoscopies_missed": round(1000 * pv * (1 - a["flag"]), 2), "slides_scanned": round(per("slides"), 1), "progressor_endoscopy_sensitivity": r3(a["flag"]), "non_progressor_endoscopy_specificity": r3(1 - b["flag"]),
                 "progressor_patient_detection_ndbe_unit": r3(pdet), "_sens_raw": round(a["flag"], 6), "progressor_patients_detected_per_1000": round(1000 * pv / eP * pdet, 3)}
            for cs in [COST["sequencing_per_sample_gbp"]] + COST["sequencing_sensitivity_gbp"]:
                o[f"cost_gbp_seq_{int(cs)}"] = round(cs * per("seqS") + COST["flag_gbp"] * per("flag") + COST["slide_scanning_and_ai_per_slide_gbp"] * per("slides"))
            o["cost_components_gbp_seq_85"] = {"sequencing": round(85 * per("seqS")), "flags": round(COST["flag_gbp"] * per("flag")), "slides": round(COST["slide_scanning_and_ai_per_slide_gbp"] * per("slides"))}
            return o
        CT_ = {nm: clin(nm, scan=nm not in ("GP53", "GRADE")) for nm in TL_}
        for nm in TL_:
            if nm == "GP53": continue
            for cs in [85, 150, 250]:
                dc = CT_[nm][f"cost_gbp_seq_{cs}"] - CT_["GP53"][f"cost_gbp_seq_{cs}"]; dp = CT_[nm]["progressor_patients_detected_per_1000"] - CT_["GP53"]["progressor_patients_detected_per_1000"]
                de = 1000 * pv * (CT_[nm]["_sens_raw"] - CT_["GP53"]["_sens_raw"])
                CT_[nm][f"extra_cost_per_additional_progressor_patient_detected_vs_grade_p53_seq_{cs}"] = round(dc / dp) if dp > 1e-9 else None
                CT_[nm][f"extra_cost_per_additional_progressor_endoscopy_detected_vs_grade_p53_seq_{cs}"] = round(dc / de) if de > 1e-9 else None
        R["clinical_per_1000_ndbe"] = {"rules": CT_, "costs": COST, "ndbe_unit": {"endoscopies": int(U.sum()), "progressor_endoscopies": int((U & P).sum()), "progressor_patients": len(UP), "progressor_endoscopies_per_progressor_patient": r3(eP),
                                       "samples_per_endoscopy_mean": r3(S["nsz"][U].mean()), "slides_per_endoscopy_mean": r3(S["eslides"][U].mean())}, "V0_ndbe_sensitivity_used_for_matching": r3(sv0.mean())}
        # ---- current care (deterministic) and the V0 family at its operating points: in the curve task; same-block status of V0-sequenced samples
        SB = same_block_labels(); pre_lab = pd.Series(SB[S["rows"]]).value_counts().to_dict(); notsame = SB != "same case, same block"
        seqd = np.zeros((n, 10), bool)
        for j in range(10): sq = RU["V0"][1][:, j] > 0; seqd[S["TOP"][sq, j], j] = True
        ever = seqd.any(1)
        R["same_block"] = {"pre_event_sample_labels": {k: int(v) for k, v in pre_lab.items()}, "v0_sequenced_samples_per_repeat_mean": round(float(seqd.sum(0).mean()), 1),
                           "v0_sequenced_not_same_block_per_repeat_mean": round(float((seqd & notsame[:, None]).sum(0).mean()), 2), "v0_sequenced_in_any_repeat": int(ever.sum()), "v0_sequenced_in_any_repeat_not_same_block": int((ever & notsame).sum()),
                           "note": "labels from the round-7 lookup Q1 (case number + block on the sheet's Path ID vs the slide's PathCaseID + Block; clinical_lookup/q6_q1fix.py logic); 'not same block' = different block, block not recorded on one side, or case unparseable"}
        # ---- threshold stability (per fold)
        TH = pd.DataFrame(THR); qs = lambda v: {"median": r3(np.median(v)), "iqr": [r3(np.percentile(v, 25)), r3(np.percentile(v, 75))], "range": [r3(np.min(v)), r3(np.max(v))]}
        bl = TH.n_train_progressor_endoscopies_below_t_low.values; blp = TH.n_train_progressor_patients_below_t_low.values; osn = TH.outer_two_stage_sensitivity.dropna().values
        R["threshold_stability"] = {"folds": int(len(TH)), "t_low": qs(TH.t_low.values), "t_cnv": qs(TH.t_cnv[np.isfinite(TH.t_cnv)].values), "folds_t_cnv_infeasible": int(TH.infeasible_flag_all_uncleared.sum()),
                                    "train_progressor_endoscopies_per_fold": qs(TH.n_train_progressor_endoscopies.values),
                                    "folds_by_train_progressor_endoscopies_below_t_low": {"0": int((bl == 0).sum()), "1-4": int(((bl >= 1) & (bl <= 4)).sum()), ">=5": int((bl >= 5).sum())},
                                    "folds_by_train_progressor_patients_below_t_low": {"0": int((blp == 0).sum()), "1-4": int(((blp >= 1) & (blp <= 4)).sum()), ">=5": int((blp >= 5).sum())},
                                    "train_step1_sensitivity": qs(TH.train_step1_sensitivity.values), "train_two_stage_sensitivity": qs(TH.train_two_stage_sensitivity.values),
                                    "target_two_stage": V0Q["target"], "target_step1": 1 - V0Q["ql"], "outer_two_stage_sensitivity_per_fold": qs(osn), "outer_step1_sensitivity_per_fold": qs(TH.outer_step1_sensitivity.dropna().values),
                                    "outer_two_stage_sensitivity_per_repeat": [r3(v) for v in obs["sens_V0"]], "folds_without_outer_progressor_endoscopies": int(TH.outer_two_stage_sensitivity.isna().sum()),
                                    "per_fold": [{k: (r3(v) if isinstance(v, float) else v) for k, v in rr.items() if not k.startswith("n_")} for rr in THR]}
        R["current_care"] = {"grade_ID_LGD_all": {"endoscopy_sensitivity": r3(S["GRD"][P].mean()), "endoscopy_specificity": r3(1 - S["GRD"][~P].mean()), "sens_cp95": cp_ci(int(S["GRD"][P].sum()), int(P.sum())), "spec_cp95": cp_ci(int((~S["GRD"][~P]).sum()), int((~P).sum()))},
                             "grade_or_p53_on_p53_subset": {"endoscopy_sensitivity": r3(S["CC53"][S["sub53"] & P].mean()), "endoscopy_specificity": r3(1 - S["CC53"][S["sub53"] & ~P].mean()), "endoscopies": int(S["sub53"].sum()), "progressor_endoscopies": int((S["sub53"] & P).sum())},
                             "grade_or_p53_missing_as_wild_type_all": {"endoscopy_sensitivity": r3(S["OVR"][P].mean()), "endoscopy_specificity": r3(1 - S["OVR"][~P].mean())},
                             "override_endoscopies": int(S["OVR"].sum()), "override_progressor_endoscopies": int(S["OVR"][P].sum())}
    R["note"] = ("endoscopy = (patient, endoscopy number) over the pre-event samples; V0 thresholds per (repeat, fold) from the training endoscopies' inner out-of-fold L-IMG (score, top sample, cleared) and L-CNV / L-LATE (confirmation); "
                 "single model (per-repeat values averaged over the 10 repeats); comparators evaluation-matched: specificity read at the rule's per-repeat endoscopy sensitivity on the pooled (across folds) empirical ROC with linear interpolation; "
                 "patient bootstrap 2,000 draws (RandomState(0)); rederived = every threshold re-derived from the resampled training patients' inner predictions in each draw (primary interval); fixed = thresholds and calls fixed (rounds 4-7)")
    res["R7D"] = R; dump(res)
# ================================================================== trade-off curve (descriptive): the V0 family over the t_low x target grid; current care points
elif TASK == "curve":
    S = setup("primary"); P = S["elab"] == 1; E = S["E"]; w1 = np.ones(E); out = []; out53 = []; tot = S["nsz"].sum()
    for q1 in (0.90, 0.95, 0.98):
        for q2 in (0.90, 0.95, 0.98):
            RU, Z, TH_ = derive_all(S, ql=1 - q1, target=q2, rules=("V0", "Vp53", "V0p53"), keep_thr=True); F, NS = RU["V0"]; se, sp = F[P].mean(0), 1 - F[~P].mean(0)
            out.append({"t_low_training_step1_sensitivity": q1, "two_stage_training_target": q2, "endoscopy_sensitivity": r3(se.mean()), "endoscopy_specificity": r3(sp.mean()),
                        "share_samples_sequenced": r3(NS.sum(0).mean() / tot), "share_endoscopies_sequenced": r3((NS > 0).mean()), "folds_target_unreachable_all_uncleared_flagged": int(sum(t_["infeasible_flag_all_uncleared"] for t_ in TH_)),
                        "H_specificity_at_same_sensitivity": r3(np.mean([roc_spec(S["SC"]["H"][:, j], P, w1, se[j]) for j in range(10)])), "SLATE_specificity_at_same_sensitivity": r3(np.mean([roc_spec(S["SC"]["SLATE"][:, j], P, w1, se[j]) for j in range(10)]))})
            F53 = RU["V0p53"][0]; m53 = S["sub53"]; out53.append({"t_low_training_step1_sensitivity": q1, "two_stage_training_target": q2, "endoscopy_sensitivity": r3(F53[m53 & P].mean()), "endoscopy_specificity": r3(1 - F53[m53 & ~P].mean())})
    def frontier(pts):
        pts = sorted({(p["endoscopy_specificity"], p["endoscopy_sensitivity"]) for p in pts}); fr_ = [p for p in pts if not any((q[0] >= p[0] and q[1] >= p[1]) and q != p for q in pts)]; return sorted(fr_)
    def sens_at(fr_, s0):
        xs = [p[0] for p in fr_]; ys_ = [p[1] for p in fr_]
        return None if (s0 < xs[0] or s0 > xs[-1]) else r3(np.interp(s0, xs, ys_))
    def spec_at(fr_, s0):
        q = sorted((p[1], p[0]) for p in fr_); xs = [a for a, _ in q]; ys_ = [b for _, b in q]
        return None if (s0 < xs[0] or s0 > xs[-1]) else r3(np.interp(s0, xs, ys_))
    cc1 = (r3(S["GRD"][P].mean()), r3(1 - S["GRD"][~P].mean())); m53 = S["sub53"]; cc2 = (r3(S["CC53"][m53 & P].mean()), r3(1 - S["CC53"][m53 & ~P].mean()))
    f1, f2 = frontier(out), frontier(out53)
    res["curve"] = {"grid": out, "grid_p53_subset": out53,
                    "current_care": {"grade_ID_LGD": {"endoscopy_sensitivity": cc1[0], "endoscopy_specificity": cc1[1], "V0_family_sensitivity_at_this_specificity": sens_at(f1, cc1[1]), "V0_family_specificity_at_this_sensitivity": spec_at(f1, cc1[0])},
                                     "grade_or_p53_on_p53_subset": {"endoscopy_sensitivity": cc2[0], "endoscopy_specificity": cc2[1], "V0_family_sensitivity_at_this_specificity": sens_at(f2, cc2[1]), "V0_family_specificity_at_this_sensitivity": spec_at(f2, cc2[0])}},
                    "note": "descriptive, no intervals; V0 thresholds re-derived at each grid point (t_low = Harrell-Davis quantile 1 - step-1 sensitivity; t_cnv at the two-stage target); (H) and S-LATE evaluation-matched to the point's per-repeat sensitivity; "
                            "family values interpolated linearly along the non-dominated grid points (None = outside the family's range); the p53-subset family evaluates V0 on the endoscopies' p53 samples"}
    dump(res)
# ================================================================== refit bootstrap (stored F3 draws): V0 thresholds re-derived per draw from its stored inner predictions
elif TASK == "f3":
    fl = lambda m_, d, p: f"{RR2}/f3/{m_}_draw_{d:03d}_{p}.csv"; have = [d for d in range(200) if all(os.path.exists(fl(m_, d, p)) for m_ in ("cnv_their", "img") for p in ("outer", "inner"))]
    if len(have) < 200: res["R7D"] = {"row": "f3", "status": f"not available: stored outer and inner predictions exist for {len(have)} of 200 F3 draws"}; dump(res); raise SystemExit(0)
    PREs = pd.Series(PRE, index=ids); ENDOs = pd.Series(ENDO, index=ids)
    def rd(f): D = pd.read_csv(f, dtype={"Sample": str, "Patient": str}); D["pre"] = PREs.reindex(D.Sample).values; return D
    def draw(d):
        oc, oi = rd(fl("cnv_their", d, "outer")), rd(fl("img", d, "outer")); ic, ii = rd(fl("cnv_their", d, "inner")), rd(fl("img", d, "inner"))
        Ev = oc.merge(oi[["Sample", "copy", "pred"]], on=["Sample", "copy"], suffixes=("_cnv", "_img")); assert len(Ev) == len(oc) == len(oi); Ev = Ev[Ev.pre].reset_index(drop=True); Ev["late"] = (Ev.pred_cnv + Ev.pred_img) / 2
        Tr = ic.merge(ii[["Sample", "copy", "fold", "pred"]], on=["Sample", "copy", "fold"], suffixes=("_cnv", "_img")); assert len(Tr) == len(ic) == len(ii); Tr = Tr[Tr.pre].reset_index(drop=True)
        key = lambda D: (pd.Series(ENDOs.reindex(D.Sample).values).astype(str) + "#" + D["copy"].astype(str).reset_index(drop=True)).values
        Ev["e"] = key(Ev); Tr["e"] = key(Tr); F = []; L = []; HS = []; LS_ = []; NS = 0; G49 = []; Tr["late"] = (Tr.pred_cnv + Tr.pred_img) / 2
        for k in range(1, 11):
            t = Tr[Tr.fold == k]; g = t.groupby("e"); top = t.loc[g.pred_img.idxmax()]
            tr = {"sE": top.pred_img.values, "lab": g.y.max().reindex(top.e).values.astype(int), "ct": top.pred_cnv.values}
            tl, th, tc = thr_v(tr, None); te = Ev[Ev.fold == k]; ge = te.groupby("e"); tt = te.loc[ge.pred_img.idxmax()]
            s_tr = t.pred_img.values; yc = t.y.values == 1; li = t.late.values; tl9 = thr_sens(s_tr[yc], 0.95); cs9 = c_star(s_tr[~yc], 0.90); md9 = (s_tr >= tl9) & (s_tr <= cs9) if tl9 <= cs9 else np.zeros(len(t), bool)   # rv4 r33_f3 / rv7c f3 (R49 check)
            cm9 = li[md9 & yc]; ts9 = thr_sens(cm9, 0.8) if len(cm9) >= 5 else thr_sens(li[yc], 0.8); sc9 = tt.pred_img.values
            b9 = np.where(sc9 > cs9, 2, 0) if tl9 > cs9 else np.where(sc9 < tl9, 0, np.where(sc9 > cs9, 2, 1)); G49.append(np.where(b9 == 1, tt.late.values >= ts9, b9 == 2))
            Fk, sq = apply_v(tt.pred_img.values, tt.pred_cnv.values, tl, th, tc); F.append(Fk); L.append(ge.y.max().reindex(tt.e).values.astype(int)); HS.append(tt.pred_img.values); LS_.append(ge.late.max().reindex(tt.e).values); NS += int(sq.sum())
        F = np.concatenate(F); L = np.concatenate(L); HS = np.concatenate(HS); LS_ = np.concatenate(LS_); w1 = np.ones(len(L)); se = F[L == 1].mean(); sp = 1 - F[L == 0].mean()
        hs = roc_spec(HS, L, w1, se); ls = roc_spec(LS_, L, w1, se); G49 = np.concatenate(G49)
        return {"r49_sens": G49[L == 1].mean(), "r49_spec": 1 - G49[L == 0].mean(), "v0_sens": se, "v0_spec": sp, "H_spec": hs, "SLATE_spec": ls, "d_spec_vs_H": sp - hs, "d_spec_vs_SLATE": sp - ls, "share_samples_v0": NS / len(Ev), "share_endoscopies_v0": NS / len(L), "n_endoscopies": len(L)}
    Sd = [draw(d) for d in range(NF3)]; A_ = {k: np.array([s[k] for s in Sd], float) for k in Sd[0]}
    def mc(a): return {"mean": r3(a.mean()), "median": r3(np.median(a)), "ci95_percentile": ci2(a), "one_sided_5th_percentile": lo1(a), "share_of_draws_above_0": r3((a > 0).mean())}
    f7c = J(f"{OUT7C}/f3.json"); chk = None
    if f7c and NF3 == 200: g_ = [r3(A_["r49_sens"].mean()), r3(A_["r49_spec"].mean())]; e_ = [f7c["R7C"]["r49_sens"]["mean"], f7c["R7C"]["r49_spec"]["mean"]]; chk = {"r49_mean_sens_spec_vs_round7c_f3": {"got": g_, "expected": e_, "reproduced": bool(all(abs(a - b) < 0.0015 for a, b in zip(g_, e_)))}}
    res["R7D"] = {"row": "f3", "label": "refit bootstrap (F3, 200 draws)", "status": "done", "draws_scored": NF3, "checks": chk, **{k: mc(v) for k, v in A_.items() if k != "n_endoscopies"},
                  "n_endoscopies_per_draw": {"median": r3(np.median(A_["n_endoscopies"])), "min": int(A_["n_endoscopies"].min()), "max": int(A_["n_endoscopies"].max())},
                  "definition": "stored F3 refit-bootstrap draws (one stratified 10-fold CV per patient-bootstrap draw with lambda refitted); per draw and fold V0's t_low and t_cnv re-derived from the draw's stored inner predictions (training endoscopies = resampled copy x endoscopy); V0, (H) and S-LATE on the draw's held-out pre-event endoscopies, comparators evaluation-matched to V0's sensitivity in the draw"}
    dump(res)
# ================================================================== merge: robustness table of the primary contrast
elif TASK == "merge":
    TAB = []
    for r in ROWS:
        j = J(f"{OUT}/row_{r}.json")
        if j is None: TAB.append({"row": r, "label": SUBLAB[r], "status": "missing (task not run)"}); continue
        x = j["R7D"]; c = x["contrasts"]["V0_minus_H"]; v = x["rules"]["V0"]
        TAB.append({"row": r, "label": SUBLAB[r], "status": "done", "n": {k: x[k] for k in ("n_endoscopies", "n_progressor_endoscopies", "n_samples", "n_patients", "n_progressor_patients")},
                    "v0_sens_spec": [v["endoscopy_sensitivity"], v["endoscopy_specificity"]], "H_spec": c["comparator_specificity"], "d_spec": c["delta"], "ci95_rederived": c["rederived"]["ci95"], "ci95_fixed": c["fixed"]["ci95"],
                    "lower_above_0_rederived": bool(c["rederived"]["ci95"][0] is not None and c["rederived"]["ci95"][0] > 0), "share_samples_sequenced": v["share_samples_sequenced"], "share_endoscopies_sequenced": v["share_endoscopies_sequenced"],
                    "d_spec_vs_SLATE": x["contrasts"]["V0_minus_SLATE"]["delta"], "d_spec_vs_SLATE_lower_rederived": x["contrasts"]["V0_minus_SLATE"]["rederived"]["one_sided_95_lower"]})
    pr = J(f"{OUT}/row_primary.json")
    if pr:
        br = pr["R7D"]["between_repeat"]; TAB.append({"row": "between_repeat", "label": "between-repeat variance added (as R33)", "status": "done", "d_spec": br["delta_repeat_mean"], "ci95_rederived": br["variance_inflated_ci95"], "lower_above_0_rederived": bool(br["variance_inflated_ci95"][0] > 0)})
    f3 = J(f"{OUT}/f3.json")
    if f3 is None: TAB.append({"row": "f3", "label": "refit bootstrap (F3, 200 draws)", "status": "missing (task not run)"})
    elif f3["R7D"].get("status") != "done": TAB.append({"row": "f3", "label": "refit bootstrap (F3, 200 draws)", "status": f3["R7D"]["status"]})
    else:
        q = f3["R7D"]; TAB.append({"row": "f3", "label": "refit bootstrap (F3, 200 draws)", "status": "done", "v0_sens_spec": [q["v0_sens"]["mean"], q["v0_spec"]["mean"]], "H_spec": q["H_spec"]["mean"], "d_spec": q["d_spec_vs_H"]["mean"],
                                   "ci95_rederived": q["d_spec_vs_H"]["ci95_percentile"], "lower_above_0_rederived": bool(q["d_spec_vs_H"]["ci95_percentile"][0] > 0), "share_samples_sequenced": q["share_samples_v0"]["mean"], "d_spec_vs_SLATE": q["d_spec_vs_SLATE"]["mean"]})
    rob = [r for r in TAB if r["row"] != "primary"]; done = [r for r in rob if r["status"] == "done"]
    res["R7D"] = {"table": TAB, "summary": {"rows": len(rob), "lower_limit_above_0": int(sum(r["lower_above_0_rederived"] for r in done)), "not_above_0": int(sum(not r["lower_above_0_rederived"] for r in done)), "not_available": [r["label"] for r in rob if r["status"] != "done"]}}
    dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
