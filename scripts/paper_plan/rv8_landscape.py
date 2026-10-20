"""Review round 8: mapping the strategy space (docs/paper_review_round8_landscape.md @ b396cdc). Descriptive and hypothesis-generating only.
Stored patient-grouped outer and inner out-of-fold predictions are re-scored (tb_common.py loaders and the paper_triage.md cut-off rules); nothing is
refitted, no ACE-B file is read, frozen bundles are not touched. Helpers copied unchanged from rv7_round7.py / rv7b_gate.py / rv7d_sensfirst.py
(fsa, dstat, ci2, lo1, matched_calls, all_calls, cutoff_arrays, groups_of, r49_calls, sheet). Exact endoscopy dates: pre_event_cohort.Date by CNV
folder name (the rv7_round7.py db_link mapping), never the year-only slide-table date.
Row-level files only under killcoyne_mm/review_round8/ on the cluster; aggregates (no identifiers, no dates) -> results/paper_final/review_round8/<TASK>.json.
TASK = prep | pol_<P1..P10> | l1 | l2_<overall|heq|grade|tissue|tte|position|nsamp> | l3 | l4 | l5"""
import os, sys, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_common as TC
from tb_common import *
PRESPEC = "b396cdc"; OUT = T + "/results/paper_final/review_round8"; OUT7 = T + "/results/paper_final/review_round7"; OUT6 = T + "/results/paper_final/review_round6"
R8 = M + "/review_round8"; KD = T + "/feasibility/paper_plan/killcoyne"; SWG = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
REL = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
if os.environ.get("RV8_NB"):   # smoke mode: fewer bootstrap draws, aggregates written aside (never to results/)
    NB = int(os.environ["RV8_NB"]); TC.NB = NB; TC.boots_for.__defaults__ = (0, NB); OUT = R8 + "/smoke_json"
for d in (OUT, R8): os.makedirs(d, exist_ok=True)
TASK = os.environ["TASK"]; MM = {"A*": "L-CNV", "B*": "L-LATE", "C*": "L-IMG"}; NSEEDS = 100; MODELS = ("L-CNV", "L-IMG", "L-LATE")
Q3 = (0.80, 0.90, 0.95); GRID9 = [(ql, qh) for ql in (0.90, 0.95, 0.98) for qh in (0.85, 0.90, 0.95)]
COST_SEQ = 85.0; COST_FLAG = 471.0 + 44.0 + 20.0; DAY = np.timedelta64(1, "D")
r2 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 2)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def m5(k):
    """aggregate count shown as '<5' when 0 < k < 5 (patient cells)"""
    if k is None or not np.isfinite(k): return None
    return "<5" if 0 < k < 5 else (round(float(k), 1) if abs(k - round(k)) > 1e-9 else int(round(k)))
# ------------------------------------------------------------------ helpers copied unchanged (rv7_round7.py / rv7b_gate.py / rv7d_sensfirst.py)
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
def nfo(rows): return {"n_samples": int(len(rows)), "n_progressor_samples": int(y[rows].sum()), "n_patients": int(len(set(pat[rows]))), "n_progressor_patients": int(len(set(pat[rows][y[rows] == 1])))}
FO0 = None
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
    res["prespec"] = PRESPEC; json.dump(res, open(f"{OUT}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV8 DONE", name or TASK, flush=True)
# ------------------------------------------------------------------ exact dates (pre_event_cohort.Date), event and last-sequenced reference dates
def dates():
    """per analysed sample: exact date; per patient: reference date (progressor: the sheet event endoscopy = first HGD/IMC endoscopy, hz_prep.py event,
    exact date of its samples; non-progressor: last sequenced endoscopy = latest exact date over all the patient's sequenced samples on the sheet).
    Fallbacks (counted): implied from an analysed sample's exact date and months-before-final."""
    pe = pd.read_csv(REL + "/pre_event_cohort.csv", dtype=str); pe["cnv"] = pe.CNVAbsPath.map(lambda p: os.path.basename(str(p))); pe["d"] = pd.to_datetime(pe.Date, errors="coerce")
    pdate = pe.dropna(subset=["d"]).drop_duplicates("cnv").set_index("cnv").d
    SD = pd.to_datetime(pd.Series(ids).map(pdate).values); info = {"analysed_samples_with_exact_date": int(pd.notna(SD).sum()), "analysed_samples": int(n)}
    fd = sheet(); fd = fd[fd.combined_name.notna()].copy(); fd["Patient"] = fd.Patient.astype(str).str.replace(r"\.0$", "", regex=True); fdu = fd.drop_duplicates("combined_name").set_index("combined_name")
    A = pd.read_csv(KD + "/kr_samples.csv", dtype=str).set_index("Sample"); A["Patient"] = A.Patient.astype(str).str.replace(r"\.0$", "", regex=True)
    A["mbf"] = pd.to_numeric(A.index.map(fdu["Months before final"]), errors="coerce"); A["ogd"] = A.index.map(fdu["Path_class_per_OGD"])
    hg = A[(A.Pathology.isin(["HGD", "IMC"])) | (A.ogd.isin(["HGD", "IMC", "HGD/IMC"]))]; tevA = hg.groupby("Patient").mbf.max()
    TEV = pd.to_numeric(SM.tev, errors="coerce").values; MBF = SM.mbf.values.astype(float); REF = {}; src = {"progressor_exact_event_endoscopy": 0, "progressor_implied": 0, "non_progressor_sheet_exact": 0, "non_progressor_implied": 0}
    agree = {"progressor": [], "non_progressor": []}; tev_ok = 0; PPAT = set(pat[PRE])
    for p in sorted(PPAT):
        s = np.where((pat == p) & pd.notna(SD))[0]; implied = pd.to_datetime([SD[i] + pd.Timedelta(days=30.4375 * (MBF[i] - (TEV[i] if y[i] == 1 else 0.0))) for i in s])
        imp = implied.min() + (implied.max() - implied.min()) / 2
        if y[s[0]] == 1:
            tv = TEV[s[0]]; tev_ok += int(p in tevA.index and abs(tevA[p] - tv) < 1e-9)
            d_ = [pdate.get(x) for x in A[(A.Patient == p) & (A.mbf == tv)].index]; d_ = [x for x in d_ if x is not None and pd.notna(x)]
            if d_: REF[p] = min(d_); src["progressor_exact_event_endoscopy"] += 1; agree["progressor"].append(abs((REF[p] - imp) / DAY))
            else: REF[p] = imp; src["progressor_implied"] += 1
        else:
            d_ = [pdate.get(x) for x in fd[fd.Patient == p].combined_name]; d_ = [x for x in d_ if x is not None and pd.notna(x)] + list(SD[s])
            if len(d_): REF[p] = max(d_); src["non_progressor_sheet_exact"] += 1; agree["non_progressor"].append(abs((REF[p] - imp) / DAY)); lastA = SD[s].max(); agree.setdefault("np_last_sequenced_after_last_analysed", []).append(bool(REF[p] > lastA))
            else: REF[p] = imp; src["non_progressor_implied"] += 1
    info.update({"reference_date_source_patients": src, "progressor_event_months_before_final_equal_to_sheet_rule": tev_ok,
                 "abs_days_exact_vs_implied_reference": {cl: {"n": len(v), "median": r3(np.median(v)), "p90": r3(np.percentile(v, 90)), "max": r3(np.max(v)), "n_over_90_days": int(np.sum(np.asarray(v) > 90))} for cl, v in agree.items() if cl != "np_last_sequenced_after_last_analysed" and len(v)},
                 "non_progressor_last_sequenced_endoscopy_after_last_analysed_sample": int(np.sum(agree.get("np_last_sequenced_after_last_analysed", [])))})
    return SD, REF, info
# ------------------------------------------------------------------ setup: endoscopies, dates, training structures, endoscopy scores
def escore(img, cnv, late, idx, es, PICK):
    """endoscopy scores over the endoscopies idx from sample score vectors: max L-IMG / L-CNV / L-LATE, top-H&E sample's L-LATE, max L-LATE of the
    2 top-H&E samples, and L-LATE of the R41 random sample (seeds x endoscopies)"""
    o = {k: np.full(len(idx), np.nan) for k in ("img", "cnv", "late", "top", "top2")}
    for i, e in enumerate(idx):
        s = es[e]; xi = img[s]; od = np.argsort(-xi, kind="mergesort")
        o["img"][i] = xi.max(); o["cnv"][i] = cnv[s].max(); o["late"][i] = late[s].max(); o["top"][i] = late[s[od[0]]]; o["top2"][i] = late[s[od[:2]]].max()
    o["rnd"] = late[PICK[:, idx]]; return o
def setup():
    global FO0
    O, I, FO = load("their"); FO0 = FO; popm = PRE.copy(); rows = np.where(popm)[0]; eu, gi = groups_of(rows); es = [rows[g] for g in gi]; E = len(es)
    elab = np.array([int(y[s].max()) for s in es]); assert all(len(set(y[s])) == 1 for s in es); epat = np.array([pat[s][0] for s in es]); nsz = np.array([len(s) for s in es])
    up = np.unique(epat); pidx = np.searchsorted(up, epat); assert (up[pidx] == epat).all(); P_ = len(up); plab = np.array([int(elab[pidx == i].max()) for i in range(P_)])
    INC = np.zeros((P_, E)); INC[pidx, np.arange(E)] = 1
    for s in es: assert all(len(set(FO[s, j])) == 1 for j in range(10))
    EF = np.stack([FO[s[0]] for s in es]); SEi = np.full(n, -1);
    for e, s in enumerate(es): SEi[s] = e
    SD, REF, dinfo = dates(); assert pd.notna(SD[rows]).all(), "pre-event sample without an exact date"
    ED = pd.to_datetime([SD[s].min() for s in es]); spread = np.array([(SD[s].max() - SD[s].min()) / DAY for s in es]); dinfo["endoscopies_with_more_than_one_sample_date"] = int((spread > 0).sum()); dinfo["max_within_endoscopy_date_spread_days"] = r3(spread.max())
    emb = np.array([SM.mbf.values.astype(float)[s].max() for s in es]); REFp = pd.to_datetime([REF[p] for p in up])
    TTE = np.asarray((REFp[pidx] - ED) / DAY, float) / 365.25; dinfo["endoscopies_after_reference_date"] = int((TTE < -1e-9).sum()); dinfo["progressor_endoscopies_dated_on_the_event_date"] = int(((np.abs(TTE) < 1e-9) & (elab == 1)).sum())
    dinfo["endoscopy_date_spread_days_when_more_than_one_date"] = {"median": r3(np.median(spread[spread > 0])), "max": r3(spread.max()), "units_with_all_dates_in_one_calendar_year": int(sum(len(set(SD[s].year)) == 1 for s, sp in zip(es, spread) if sp > 0))} if (spread > 0).any() else None
    ORD = []; FIRST = np.zeros(E, bool); LAST2 = np.zeros(E, bool); POS = np.zeros(E, int)
    for i in range(P_):
        idx = np.where(pidx == i)[0]; o = idx[np.lexsort((-emb[idx], np.asarray(ED[idx].values, "datetime64[ns]")))]; ORD.append(o); FIRST[o[0]] = True; LAST2[o[-2:]] = True; POS[o] = np.arange(len(o))
    order_mbf_agree = int(sum(np.all(np.diff(emb[o]) <= 0) for o in ORD))
    PY = np.array([float((REFp[i] - ED[ORD[i][0]]) / DAY) / 365.25 for i in range(P_)]); NE = INC.sum(1)
    dinfo.update({"patients_with_date_order_equal_to_months_before_final_order": order_mbf_agree, "patient_years": {"progressor": r3(PY[plab == 1].sum()), "non_progressor": r3(PY[plab == 0].sum()), "all": r3(PY.sum())},
                  "non_progressor_patient_years_months_before_final_span_as_round7": r3(sum((emb[pidx == i].max() - emb[pidx == i].min()) / 12.0 for i in np.where(plab == 0)[0]))})
    PICK = np.zeros((NSEEDS, E), int)
    for sd in range(NSEEDS):
        rng = np.random.RandomState(sd)
        for e, s in enumerate(es): PICK[sd, e] = s[rng.randint(len(s))]   # R41 draw order (rv6_round6.py r41, rv7_round7.py r46)
    ALL = np.arange(E); OS = [escore(O["L-IMG"][:, j], O["L-CNV"][:, j], O["L-LATE"][:, j], ALL, es, PICK) for j in range(10)]
    OUTS = {k: np.stack([o[k] for o in OS], -1) for k in OS[0]}   # (E, 10); rnd (seeds, E, 10)
    TR = {}
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            tidx = np.where(EF[:, j] != k)[0]; tri = I["L-IMG"][(r, k)].index.values; tri = tri[popm[tri]]
            assert set(tri) == set(np.concatenate([es[e] for e in tidx])), "training endoscopies differ from the inner training samples"
            v = {}
            for m in MODELS: x = np.full(n, np.nan); x[tri] = I[m][(r, k)].reindex(tri).values; assert np.isfinite(x[tri]).all(); v[m] = x
            TR[(j, k)] = {**escore(v["L-IMG"], v["L-CNV"], v["L-LATE"], tidx, es, PICK), "lab": elab[tidx], "idx": tidx}
    return dict(O=O, I=I, FO=FO, popm=popm, rows=rows, eu=eu, es=es, E=E, elab=elab, epat=epat, nsz=nsz, up=up, pidx=pidx, P_=P_, plab=plab, INC=INC, EF=EF, SEi=SEi,
                ED=ED, SD=SD, TTE=TTE, ORD=ORD, FIRST=FIRST, LAST2=LAST2, POS=POS, PY=PY, NE=NE, emb=emb, PICK=PICK, OUTS=OUTS, TR=TR, dinfo=dinfo)
def thr_arr(S, key, q):
    """per endoscopy x repeat: the fold threshold reaching >= q training endoscopy sensitivity on score `key` (training = the fold's training patients'
    pre-event endoscopies, inner out-of-fold predictions); for 'rnd' one threshold per seed (seeds, E, 10). Returns (thresholds, folds with target not met)"""
    E = S["E"]; rnd = key == "rnd"; TH = np.full((NSEEDS, E, 10) if rnd else (E, 10), np.nan); miss = 0
    for j in range(10):
        for k in range(1, 11):
            tr = S["TR"][(j, k)]; P = tr["lab"] == 1; idx = S["EF"][:, j] == k
            if rnd:
                for sd in range(NSEEDS): x = tr["rnd"][sd, P]; t = thr_sens(x, q); TH[sd, idx, j] = t; miss += int(len(x) == 0 or (x >= t).mean() < q - 1e-9)
            else: x = tr[key][P]; t = thr_sens(x, q); TH[idx, j] = t; miss += int(len(x) == 0 or (x >= t).mean() < q - 1e-9)
    return TH, miss
# ------------------------------------------------------------------ patient outcomes and statistics
def patient_arrays(S, F, NS, em=None):
    """F, NS (E, 10) or (seeds, E, 10): per patient x repeat ever flagged, flagged endoscopies, samples sequenced, flagged within the last 2 pre-event endoscopies;
    years from first flag to the reference date per repeat (median over flagged progressors); seeds averaged. em = endoscopies evaluated (None = all)"""
    if F.ndim == 3:
        L = [patient_arrays(S, F[sd], NS[sd], em) for sd in range(F.shape[0])]; o = {k: np.mean([x[k] for x in L], 0) for k in ("FL", "NF", "NS", "W2")}
        o.update({k: L[0][k] for k in ("NE", "pinc")}); o["TFmed"] = np.mean([x["TFmed"] for x in L], 0); return o
    em = np.ones(S["E"], bool) if em is None else em; INC = S["INC"] * em[None, :]; Ff = F & em[:, None]
    o = {"FL": (INC @ Ff) > 0, "NF": INC @ Ff, "NS": INC @ (NS * em[:, None]), "W2": (INC @ (Ff & S["LAST2"][:, None])) > 0, "NE": INC.sum(1)}; o["pinc"] = o["NE"] > 0
    TT = np.where(Ff, S["TTE"][:, None], -np.inf); TF = np.stack([TT[S["pidx"] == i].max(0) if (S["pidx"] == i).any() else np.full(10, -np.inf) for i in range(S["P_"])])
    Pm = (S["plab"] == 1) & o["pinc"]; o["TFmed"] = np.array([np.median(TF[Pm & o["FL"][:, j], j]) if (Pm & o["FL"][:, j]).any() else np.nan for j in range(10)]); o["FL"] = o["FL"].astype(float); o["W2"] = o["W2"].astype(float)
    return o
def pdraws(S, nb=None):
    """patient bootstrap multiplicities (RandomState(0), the R49 / rv7d draw loop); one-class draws rejected"""
    nb = NB if nb is None else nb; rng = np.random.RandomState(0); out = []; P_ = S["P_"]; pl = S["plab"]
    while len(out) < nb:
        mm = np.bincount(rng.choice(P_, P_), minlength=P_)
        if (mm * pl).sum() > 0 and (mm * (1 - pl)).sum() > 0: out.append(mm)
    return np.array(out, float)
def lstats(S, A, Mb):
    """per-repeat outcomes for multiplicity rows Mb (B, P_), averaged over the 10 repeats -> dict of (B,)"""
    Mb = Mb * A["pinc"][None, :]; isP = S["plab"] == 1; sP = Mb * isP; sN = Mb * ~isP; PY = S["PY"]; NE = A["NE"]
    with np.errstate(invalid="ignore", divide="ignore"):
        o = {"caught_share": (sP @ A["FL"]) / sP.sum(1)[:, None], "np_flagged_share": (sN @ A["FL"]) / sN.sum(1)[:, None], "flags_per_np_py": (sN @ A["NF"]) / (sN @ PY)[:, None],
             "w2_caught_share": (sP @ A["W2"]) / sP.sum(1)[:, None], "w2_np_flagged_share": (sN @ A["W2"]) / sN.sum(1)[:, None], "seq_per_patient": (Mb @ A["NS"]) / Mb.sum(1)[:, None],
             "seq_per_100py": 100 * (Mb @ A["NS"]) / (Mb @ PY)[:, None], "cost_per_patient": (COST_SEQ * (Mb @ A["NS"]) + COST_FLAG * (Mb @ A["NF"])) / Mb.sum(1)[:, None],
             "endo_sens": (sP @ A["NF"]) / (sP @ NE)[:, None], "endo_spec": 1 - (sN @ A["NF"]) / (sN @ NE)[:, None], "flags_per_patient": (Mb @ A["NF"]) / Mb.sum(1)[:, None]}
    return {k: np.nanmean(v, 1) for k, v in o.items()}
def cell_result(S, A, D, fam, label, params, miss=0, extra=None):
    ob = lstats(S, A, np.ones((1, S["P_"]))); bs = lstats(S, A, D); pi = A["pinc"]; nP = int(((S["plab"] == 1) & pi).sum()); nN = int(((S["plab"] == 0) & pi).sum())
    g = lambda k: float(ob[k][0]); c = lambda k: ci2(bs[k])
    o = {"family": fam, "label": label, "params": params, "n_progressor_patients": nP, "n_non_progressor_patients": nN, "folds_target_not_met": int(miss),
         "progressor_patients_caught": r2(g("caught_share") * nP), "caught_share": r3(g("caught_share")), "caught_share_ci95": c("caught_share"), "progressor_patients_missed": r2((1 - g("caught_share")) * nP),
         "non_progressor_patients_ever_flagged": r2(g("np_flagged_share") * nN), "np_flagged_share": r3(g("np_flagged_share")), "np_flagged_share_ci95": c("np_flagged_share"),
         "flags_per_non_progressor_patient_year": r3(g("flags_per_np_py")), "flags_per_np_py_ci95": c("flags_per_np_py"),
         "years_first_flag_to_event_median": r2(float(np.nanmean(A["TFmed"]))),
         "last2_window": {"progressor_patients_caught": r2(g("w2_caught_share") * nP), "non_progressor_patients_flagged": r2(g("w2_np_flagged_share") * nN)},
         "samples_sequenced_per_patient": r3(g("seq_per_patient")), "seq_per_patient_ci95": c("seq_per_patient"), "samples_sequenced_per_100_patient_years": r2(g("seq_per_100py")),
         "share_samples_sequenced": r3(float((A["NS"][A["pinc"]].sum(0)).mean() / S["nsz"][np.isin(S["pidx"], np.where(pi)[0])].sum())),
         "cost_per_patient_gbp": r2(g("cost_per_patient")), "cost_ci95": [r2(x) if x is not None else None for x in c("cost_per_patient")], "flagged_endoscopies_per_patient": r3(g("flags_per_patient")),
         "endoscopy_sensitivity": r3(g("endo_sens")), "endo_sens_ci95": c("endo_sens"), "endoscopy_specificity": r3(g("endo_spec")), "endo_spec_ci95": c("endo_spec")}
    if extra: o.update(extra)
    return o
# ------------------------------------------------------------------ policies
KEY7 = {"1 top-H&E": "top", "1 random": "rnd", "2 top-H&E": "top2", "all": "late"}
def kcount(S, opt): return np.ones(S["E"], int) if opt in ("1 top-H&E", "1 random") else (np.minimum(2, S["nsz"]) if opt == "2 top-H&E" else S["nsz"].copy())
def p7_calls(S, opt, qb, qs, THC=None):
    """P7: first pre-event endoscopy sequenced (k samples), flagged if its score >= t_b and the patient becomes high risk (every later endoscopy flagged);
    otherwise later endoscopies flagged if max L-IMG >= t_s. Returns F, NS (seeds-first when the random sample is used)"""
    key = KEY7[opt]; THb, mb = thr_arr(S, key, qb); THs, ms = thr_arr(S, "img", qs); X = S["OUTS"][key]; FIRST = S["FIRST"]; kc = kcount(S, opt)
    def one(Xb, Tb):
        f0 = (Xb >= Tb) & FIRST[:, None]; hr = ((S["INC"] @ f0) > 0)[S["pidx"]]; F = np.where(FIRST[:, None], f0, hr | (S["OUTS"]["img"] >= THs))
        return F, np.where(FIRST[:, None], kc[:, None], 0)
    if key == "rnd": R = [one(X[sd], THb[sd]) for sd in range(NSEEDS)]; return np.stack([r[0] for r in R]), np.stack([r[1] for r in R]), mb + ms
    F, NS = one(X, THb); return F, NS, mb + ms
def p8_calls(S, qb, qmid, qc=0.90):
    THb, mb = thr_arr(S, "top", qb); THm, mm_ = thr_arr(S, "img", qmid); THc, mc = thr_arr(S, "top", qc); X = S["OUTS"]; FIRST = S["FIRST"][:, None]
    f0 = (X["top"] >= THb) & FIRST; hr = ((S["INC"] @ f0) > 0)[S["pidx"]]; sq = ~FIRST & ~hr & (X["img"] >= THm)
    F = np.where(FIRST, f0, hr | (sq & (X["top"] >= THc))); NS = np.where(FIRST, 1, sq.astype(int)); return F, NS, mb + mm_ + mc
def p9_calls(S, N, q=0.90, qs=0.90):
    THt, mt = thr_arr(S, "top", q); THs, ms = thr_arr(S, "img", qs); X = S["OUTS"]; E = S["E"]; F = np.zeros((E, 10), bool); NS = np.zeros((E, 10), int); ED = S["ED"]
    for j in range(10):
        for o in S["ORD"]:
            last = None
            for e in o:
                if last is None or (ED[e] - last) / DAY / 365.25 >= N - 1e-9: NS[e, j] = 1; F[e, j] = X["top"][e, j] >= THt[e, j]; last = ED[e]
                else: F[e, j] = X["img"][e, j] >= THs[e, j]
    return F, NS, mt + ms
def family(S, fam):
    """list of (label, params, F, NS, em, folds_target_not_met, extra)"""
    E = S["E"]; Z = np.zeros((E, 10), int); NSALL = np.repeat(S["nsz"][:, None], 10, 1); C = []; X = S["OUTS"]
    if fam in ("P1", "P2", "P3"):
        key = {"P1": "img", "P2": "cnv", "P3": "late"}[fam]
        for q in Q3: TH, ms = thr_arr(S, key, q); C.append((f"{int(q * 100)}%", {"q": q}, X[key] >= TH, Z if fam == "P1" else NSALL, None, ms, None))
    elif fam == "P4":
        for q in Q3: TH, ms = thr_arr(S, "rnd", q); C.append((f"{int(q * 100)}%", {"q": q, "seeds": NSEEDS}, X["rnd"] >= TH, np.ones((NSEEDS, E, 10), int), None, ms, None))
    elif fam in ("P5", "P6"):
        for ql, qh in GRID9:
            band, call, info = TC.triage(S["O"], S["I"], S["FO"], S["popm"], ql, qh)
            if fam == "P5": F = np.array([call[s].any(0) for s in S["es"]]); NS = np.array([(band[s] == 1).sum(0) for s in S["es"]])
            else: F, NS, _ = r49_calls(S["O"], info, S["FO"], S["es"])
            C.append((f"t_low {int(ql * 100)}% / t_high {int(qh * 100)}%", {"ql": ql, "qh": qh}, F, NS, None, 0, {"folds_no_middle_band": int(info["no_mid"]), "t_seq_fallback": int(info["fallback"])}))
    elif fam == "P7":
        for opt in KEY7:
            for qb in Q3:
                for qs in Q3: F, NS, ms = p7_calls(S, opt, qb, qs); C.append((f"k = {opt}; t_b {int(qb * 100)}%, t_s {int(qs * 100)}%", {"k": opt, "qb": qb, "qs": qs}, F, NS, None, ms, None))
    elif fam == "P8":
        for qb in Q3:
            for qm in (0.80, 0.90): F, NS, ms = p8_calls(S, qb, qm); C.append((f"t_b {int(qb * 100)}%, t_mid {int(qm * 100)}%, t_c 90%", {"k": "1 top-H&E", "qb": qb, "qmid": qm, "qc": 0.90}, F, NS, None, ms, None))
    elif fam == "P9":
        for N in (2, 3, 5): F, NS, ms = p9_calls(S, N); C.append((f"N = {N} years; t, t_s 90%", {"N": N, "q": 0.90, "qs": 0.90}, F, NS, None, ms, None))
    elif fam == "P10":
        GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); P53 = pd.to_numeric(SM["P53 IHC"], errors="coerce").values
        hasp = np.isfinite(P53); ab = np.where(hasp, P53, 0) == 1; dys = (GR == 1) | (GR == 2)
        G = np.array([dys[s].any() for s in S["es"]]); em = np.array([hasp[s].any() for s in S["es"]]); G53 = np.array([(dys[s[hasp[s]]] | ab[s[hasp[s]]]).any() for s in S["es"]])
        C.append(("grade ID/LGD (all pre-event endoscopies)", {"rule": "grade"}, np.repeat(G[:, None], 10, 1), Z, None, 0, None))
        C.append(("grade ID/LGD or p53 aberrant (samples with p53)", {"rule": "grade_or_p53", "subset": "samples with a p53 result"}, np.repeat(G53[:, None], 10, 1), Z, em, 0,
                  {"subset_samples": int(hasp[S["rows"]].sum()), "subset_endoscopies": int(em.sum())}))
    else: raise SystemExit(f"unknown family {fam}")
    return C
res = {"task": TASK}
# ================================================================== prep: dates, populations, availability (row-level date table on the cluster only)
if TASK == "prep":
    S = setup(); R = {**nfo(S["rows"]), "n_endoscopies": S["E"], "n_progressor_endoscopies": int(S["elab"].sum()), "n_non_progressor_patients": int((S["plab"] == 0).sum()), "dates": S["dinfo"]}
    O, I = S["O"], S["I"]; R["available"] = {"outer_repeats_finite": bool(all(np.isfinite(O[m][S["rows"]]).all() for m in ("L-CNV", "L-IMG"))), "inner_repeat_folds": {m: len(I[m]) for m in MODELS},
                                             "training_endoscopies_equal_inner_training_samples": True, "time_dependent_auroc_stored": {h: os.path.exists(f"{T}/results/paper_final/horizon_foldstrat/fs_their_pre_{h}.json") for h in (1, 3, 5)}}
    ne = np.bincount(S["pidx"]); R["endoscopies_per_patient"] = {"progressor": {"mean": r3(ne[S["plab"] == 1].mean()), "ge2": int((ne[S["plab"] == 1] >= 2).sum())}, "non_progressor": {"mean": r3(ne[S["plab"] == 0].mean()), "ge2": int((ne[S["plab"] == 0] >= 2).sum())}}
    R["years_first_endoscopy_to_reference"] = {cl: {"median": r3(np.median(S["PY"][S["plab"] == v])), "iqr": [r3(np.percentile(S["PY"][S["plab"] == v], 25)), r3(np.percentile(S["PY"][S["plab"] == v], 75))]} for cl, v in (("progressor", 1), ("non_progressor", 0))}
    R["years_endoscopy_to_reference"] = {cl: {"median": r3(np.median(S["TTE"][S["elab"] == v])), "min": r3(S["TTE"][S["elab"] == v].min()), "max": r3(S["TTE"][S["elab"] == v].max())} for cl, v in (("progressor", 1), ("non_progressor", 0))}
    tt = pd.Series(SM.time.values.astype(float))[S["rows"]].values; st = S["TTE"][S["SEi"][S["rows"]]]; pm = y[S["rows"]] == 1
    R["exact_vs_months_before_final_time_to_event_progressor_samples"] = {"median_abs_diff_years": r3(np.median(np.abs(tt[pm] - st[pm]))), "max_abs_diff_years": r3(np.max(np.abs(tt[pm] - st[pm])))}
    pd.DataFrame({"endoscopy": S["eu"], "patient": S["epat"], "label": S["elab"], "n_samples": S["nsz"], "date": S["ED"], "years_to_reference": S["TTE"], "first": S["FIRST"], "last2": S["LAST2"], "position": S["POS"]}).to_csv(R8 + "/endoscopies.csv", index=False)
    res["prep"] = R; dump(res)
# ================================================================== L1: one policy family
elif TASK.startswith("pol_"):
    fam = TASK[4:]; S = setup(); D = pdraws(S); cells = []
    for label, params, F, NS, em, ms, extra in family(S, fam):
        A = patient_arrays(S, F, NS, em); cells.append(cell_result(S, A, D, fam, label, params, ms, extra))
    R = {"family": fam, "cells": cells, "draws": int(len(D)), "n_endoscopies": S["E"], "n_patients": S["P_"]}
    if fam == "P3":   # check: the B* operating point (stored round-7 B* sample calls) through this endoscopy / patient code = B*-endo
        CALLS, band, info, thr, sstar = all_calls(S["O"], S["I"], S["FO"], S["popm"]); FB = np.array([CALLS["B*"][s].any(0) for s in S["es"]])
        A = patient_arrays(S, FB, np.repeat(S["nsz"][:, None], 10, 1)); cb = cell_result(S, A, D, fam, "B* operating point (round-7 B* sample threshold, any sample)", {"q": "B*"})
        near = min(cells, key=lambda c: abs(c["endoscopy_specificity"] - cb["endoscopy_specificity"]))
        R["check_Bstar"] = {"cell": cb, "expected_Bstar_endo": [0.923, 0.530], "got": [cb["endoscopy_sensitivity"], cb["endoscopy_specificity"]],
                            "reproduced": bool(abs(cb["endoscopy_sensitivity"] - 0.923) < 0.0015 and abs(cb["endoscopy_specificity"] - 0.530) < 0.0015), "nearest_grid_cell": near["label"],
                            "nearest_grid_cell_sens_spec": [near["endoscopy_sensitivity"], near["endoscopy_specificity"]]}
    if fam == "P6":
        r49 = J(f"{OUT7}/r49.json")["R49"]["rules"]["gate"]; c = [x for x in cells if x["params"] == {"ql": 0.95, "qh": 0.90}][0]
        got = [c["endoscopy_sensitivity"], c["endoscopy_specificity"], c["share_samples_sequenced"], c["progressor_patients_missed"], c["non_progressor_patients_ever_flagged"]]
        exp = [r49["endoscopy_sensitivity"], r49["endoscopy_specificity"], r49["share_samples_sequenced"], r49["progressor_patients_missed"], r49["non_progressor_patients_flagged"]]
        R["check_R49"] = {"got": got, "expected": exp, "reproduced": bool(all(abs(a - b) < 0.0015 for a, b in zip(got[:3], exp[:3])) and all(abs(a - b) < 0.051 for a, b in zip(got[3:], exp[3:])))}
    if fam == "P5":
        tri = J(f"{OUT7}/r49.json")["R49"]["rules"]["triage"]; c = [x for x in cells if x["params"] == {"ql": 0.95, "qh": 0.90}][0]
        got = [c["endoscopy_sensitivity"], c["endoscopy_specificity"], c["share_samples_sequenced"], c["progressor_patients_missed"], c["non_progressor_patients_ever_flagged"]]
        exp = [tri["endoscopy_sensitivity"], tri["endoscopy_specificity"], tri["share_samples_sequenced"], tri["progressor_patients_missed"], tri["non_progressor_patients_flagged"]]
        R["check_triage_R49"] = {"got": got, "expected": exp, "reproduced": bool(all(abs(a - b) < 0.0015 for a, b in zip(got[:3], exp[:3])) and all(abs(a - b) < 0.051 for a, b in zip(got[3:], exp[3:])))}
    if fam == "P10":
        r37 = J(f"{OUT6}/r37.json")["R37"]["f_missing_as_wild_type_all_pre_event"]["rules"]["D"]["patient"]["counts"]; c = cells[0]
        exp = [r37["positive"] - r37["true_positive"], r37["negative"] - r37["true_negative"]]; got = [c["progressor_patients_missed"], c["non_progressor_patients_ever_flagged"]]
        R["check_Table1_D_patient"] = {"got_missed_flagged": got, "expected_missed_flagged": exp, "reproduced": bool(all(abs(a - b) < 1e-6 for a, b in zip(got, exp)))}
    res["L1"] = R; dump(res)
# ================================================================== L1 merge: table, Pareto frontier, checks
elif TASK == "l1":
    FAMS = [f"P{i}" for i in range(1, 11)]; cells = []; checks = {}; missing = []
    for f in FAMS:
        j = J(f"{OUT}/pol_{f}.json")
        if j is None: missing.append(f); continue
        cells += j["L1"]["cells"]; checks.update({k: v for k, v in j["L1"].items() if k.startswith("check")})
    pts = [c for c in cells if c.get("params", {}).get("subset") is None]
    def dom(a, b, keys):   # b dominates a
        ge = all((b[k] >= a[k]) if s > 0 else (b[k] <= a[k]) for k, s in keys); gt = any((b[k] > a[k]) if s > 0 else (b[k] < a[k]) for k, s in keys); return ge and gt
    K3 = (("progressor_patients_caught", 1), ("non_progressor_patients_ever_flagged", -1), ("samples_sequenced_per_patient", -1)); KC = (("progressor_patients_caught", 1), ("cost_per_patient_gbp", -1))
    for c in cells: c["pareto_caught_falseflags_sequencing"] = bool(c in pts and not any(dom(c, b, K3) for b in pts if b is not c)); c["pareto_caught_cost"] = bool(c in pts and not any(dom(c, b, KC) for b in pts if b is not c))
    res["L1"] = {"cells": cells, "missing_families": missing, "checks": checks, "n_cells": len(cells),
                 "frontier": [{"family": c["family"], "label": c["label"]} for c in cells if c["pareto_caught_falseflags_sequencing"]], "frontier_cost": [{"family": c["family"], "label": c["label"]} for c in cells if c["pareto_caught_cost"]],
                 "note": "Pareto: not dominated on progressor patients caught (higher better), non-progressor patients ever flagged and samples sequenced per patient (lower better); cost frontier on caught and cost per patient; observed single-model means; the p53-subset current-care point (different patients) is excluded from both frontiers"}
    dump(res)
# ================================================================== L2: where copy number adds value (one stratum family per task)
elif TASK.startswith("l2_"):
    st = TASK[3:]; S = setup(); O = S["O"]; rows = S["rows"]; SEi = S["SEi"]; E = S["E"]; es = S["es"]
    GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); TILES = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids).tissue_tiles.values.astype(float)
    mimg = O["L-IMG"].mean(1); qe = np.quantile(mimg[rows], [0.2, 0.4, 0.6, 0.8]); te1, te2 = np.quantile(TILES[rows], [1 / 3, 2 / 3])
    stte = np.full(n, np.nan); stte[rows] = S["TTE"][SEi[rows]]; sfirst = np.zeros(n, bool); sfirst[rows] = S["FIRST"][SEi[rows]]; snsz = np.zeros(n, int); snsz[rows] = S["nsz"][SEi[rows]]
    tert = lambda v: np.where(v <= te1, 0, np.where(v <= te2, 1, 2)); quint = lambda v: np.searchsorted(qe, v, side="right")
    etop = np.array([s[np.argmax(mimg[s])] for s in es]); emax = np.array([mimg[s].max() for s in es]); egr = np.array([GR[s].max() for s in es]); P = y == 1; NP = y == 0
    eP = S["elab"] == 1; ett = S["TTE"]
    DEF = {"overall": [("all pre-event samples", PRE, np.ones(E, bool))],
           "heq": [(f"H&E score quintile {q + 1}", PRE & (quint(mimg) == q), quint(emax) == q) for q in range(5)],
           "grade": [(g, PRE & (GR == v), egr == v) for v, g in ((0, "NDBE"), (1, "ID"), (2, "LGD"))],
           "tissue": [(f"tissue tertile {t + 1}", PRE & (tert(TILES) == t), tert(TILES[etop]) == t) for t in range(3)],
           "tte": [(lab, PRE & ((P & mk(stte)) | NP), (eP & mk(ett)) | ~eP) for lab, mk in (("< 1 year", lambda v: v < 1), ("1-3 years", lambda v: (v >= 1) & (v <= 3)), ("> 3 years", lambda v: v > 3))],
           "position": [("first pre-event endoscopy", PRE & sfirst, S["FIRST"]), ("later endoscopies", PRE & ~sfirst, ~S["FIRST"])],
           "nsamp": [("endoscopies with 1 sample", PRE & (snsz == 1), S["nsz"] == 1), ("endoscopies with >= 2 samples", PRE & (snsz >= 2), S["nsz"] >= 2)]}[st]
    T1, _ = thr_arr(S, "img", 0.90); T3, _ = thr_arr(S, "late", 0.90); F1 = S["OUTS"]["img"] >= T1; F3 = S["OUTS"]["late"] >= T3; CHG = F1 != F3
    boots = boots_for(rows); out = []
    for lab, mk, emk in DEF:
        r_ = rows[mk[rows]]; obs = {m: fsa(O[m], rows, mask=mk) for m in MODELS}; B = pd.DataFrame([{m: fsa(O[m], b, mask=mk) for m in MODELS} for b in boots])
        def ch(sel): return r3(CHG[sel].mean()) if sel.any() else None
        o = {"stratum": lab, "n_samples": int(len(r_)), "n_progressor_samples": int(y[r_].sum()), "n_patients": int(len(set(pat[r_]))), "n_progressor_patients": int(len(set(pat[r_][y[r_] == 1]))),
             "auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in MODELS}, "LATE_minus_IMG": dstat(obs["L-LATE"] - obs["L-IMG"], B["L-LATE"] - B["L-IMG"]), "CNV_minus_IMG": dstat(obs["L-CNV"] - obs["L-IMG"], B["L-CNV"] - B["L-IMG"]),
             "call_change_P1_vs_P3_at_90": {"n_endoscopies": int(emk.sum()), "n_progressor_endoscopies": int((emk & eP).sum()), "share_changed": ch(emk), "share_changed_progressor": ch(emk & eP), "share_changed_non_progressor": ch(emk & ~eP),
                                            "progressor_gained_flag_share": r3((F3 & ~F1)[emk & eP].mean()) if (emk & eP).any() else None, "progressor_lost_flag_share": r3((F1 & ~F3)[emk & eP].mean()) if (emk & eP).any() else None,
                                            "non_progressor_gained_flag_share": r3((F3 & ~F1)[emk & ~eP].mean()) if (emk & ~eP).any() else None, "non_progressor_lost_flag_share": r3((F1 & ~F3)[emk & ~eP].mean()) if (emk & ~eP).any() else None}}
        out.append(o)
    res["L2"] = {"family": st, "strata": out, "draws": len(boots), "note": "fold-stratified AUROC on pre-event samples within the stratum (folds lacking a class skipped), single model; patient bootstrap (tb_common boots_for, RandomState(0)), "
                 "thresholds not involved; call change = share of endoscopy x repeat cells whose P1 (max L-IMG) and P3 (max L-LATE) calls at 90% training endoscopy sensitivity differ; endoscopy strata: H&E quintile of the endoscopy's maximum repeat-mean L-IMG (sample quintile edges), "
                 "highest grade, tissue tertile of the top-H&E sample, progressor endoscopy time to event (all non-progressor endoscopies in every time stratum)"}
    dump(res)
# ================================================================== L3: change over time within a patient
elif TASK == "l3":
    import statsmodels.formula.api as smf
    S = setup(); O = S["O"]; es = S["es"]; pidx = S["pidx"]; plab = S["plab"]; tte = S["TTE"]; R = {}
    SC = {m: np.array([lg(O[m][s]).max(0).mean() for s in es]) for m in ("L-IMG", "L-CNV")}   # endoscopy score: mean over repeats of the max logit
    ne = np.bincount(pidx, minlength=S["P_"]); keepP = ne >= 2; keepE = keepP[pidx]; PI = np.where(keepP)[0]
    R["n"] = {"patients_ge2_endoscopies": int(keepP.sum()), "progressor": int((keepP & (plab == 1)).sum()), "non_progressor": int((keepP & (plab == 0)).sum()), "endoscopies": int(keepE.sum())}
    def slopes(m):
        out = np.full(S["P_"], np.nan)
        for i in PI:
            ix = np.where(pidx == i)[0]; x = tte[ix]
            if np.ptp(x) > 0: out[i] = np.polyfit(x, SC[m][ix], 1)[0]
        return out
    rng = np.random.RandomState(0); DR = []
    while len(DR) < NB:
        d = PI[rng.choice(len(PI), len(PI))]
        if 0 < plab[d].sum() < len(d): DR.append(d)
    for m in ("L-IMG", "L-CNV"):
        sl = slopes(m); P = (plab == 1) & np.isfinite(sl); N_ = (plab == 0) & np.isfinite(sl)
        def dm(d): s = sl[d]; pl = plab[d]; ok = np.isfinite(s); return np.mean(s[ok & (pl == 1)]) - np.mean(s[ok & (pl == 0)])
        bd = np.array([dm(d) for d in DR]); o = {"slope_per_year_vs_time_to_event": {"progressor_mean": r3(sl[P].mean()), "progressor_median": r3(np.median(sl[P])), "non_progressor_mean": r3(sl[N_].mean()), "non_progressor_median": r3(np.median(sl[N_])),
                                                                                       "n_progressor": int(P.sum()), "n_non_progressor": int(N_.sum())},
                                                 "difference_progressor_minus_non_progressor": dstat(sl[P].mean() - sl[N_].mean(), bd)}
        df = pd.DataFrame({"score": SC[m][keepE], "tte": tte[keepE], "cls": (S["elab"][keepE] == 1).astype(int), "pid": pidx[keepE]})
        try:
            fit = smf.mixedlm("score ~ tte * cls", df, groups=df["pid"]).fit(reml=True); ci = fit.conf_int()
            o["mixed_model"] = {k: {"estimate": r3(fit.params[k]), "se": r3(fit.bse[k]), "ci95": [r3(ci.loc[k, 0]), r3(ci.loc[k, 1])], "p": round(float(fit.pvalues[k]), 4)} for k in ("Intercept", "tte", "cls", "tte:cls")}
            o["mixed_model"]["random_intercept_variance"] = r3(float(fit.cov_re.iloc[0, 0])); o["mixed_model"]["residual_variance"] = r3(float(fit.scale)); o["mixed_model"]["converged"] = bool(fit.converged)
            o["mixed_model"]["n_endoscopies"] = int(len(df)); o["mixed_model"]["n_patients"] = int(df.pid.nunique())
        except Exception as ex: o["mixed_model"] = {"status": f"failed: {str(ex)[:200]}"}
        # last against first pre-event endoscopy
        for cl, v in (("progressor", 1), ("non_progressor", 0)):
            dd = np.array([SC[m][S["ORD"][i][-1]] - SC[m][S["ORD"][i][0]] for i in PI if plab[i] == v]); pp = [i for i in PI if plab[i] == v]
            bb = []; rr = np.random.RandomState(0)
            for _ in range(NB): bb.append(dd[rr.choice(len(dd), len(dd))].mean())
            o[f"last_minus_first_{cl}"] = {"n_patients": len(pp), "mean": r3(dd.mean()), "ci95": ci2(bb), "median": r3(np.median(dd)), "share_increased": r3((dd > 0).mean())}
        # trajectory: mean endoscopy score by class and time-to-reference bin (patients with >= 2 endoscopies)
        BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 99)]; tr_ = []
        for cl, v in (("progressor", 1), ("non_progressor", 0)):
            for a, b in BINS:
                mk = keepE & (S["elab"] == v) & (tte >= a) & (tte < b); np_ = len(set(pidx[mk]))
                tr_.append({"class": cl, "bin_years": [a, b], "n_endoscopies": m5(int(mk.sum())), "n_patients": m5(np_), "mean": r3(SC[m][mk].mean()) if mk.sum() >= 1 and np_ >= 5 else None,
                            "se": r3(SC[m][mk].std(ddof=1) / np.sqrt(mk.sum())) if mk.sum() >= 2 and np_ >= 5 else None})
        o["trajectory"] = tr_; R[m] = o
    R["time_dependent_auroc_stored"] = {}
    for h in (1, 3, 5):
        f = J(f"{T}/results/paper_final/horizon_foldstrat/fs_their_pre_{h}.json")
        R["time_dependent_auroc_stored"][str(h)] = "not available" if f is None else {"n_cases": f["n_cases"], "n_controls": f["n_controls"], **{m: {"ipcw": f["arms"][m]["ipcw"], "ci95": f["arms"][m]["ipcw_ci95"]} for m in MODELS}}
    R["note"] = ("endoscopy score = mean over the 10 repeats of the maximum logit outer prediction over the endoscopy's pre-event samples; time = years from the endoscopy (exact date) to the event (progressors) or to the last sequenced endoscopy (non-progressors); "
                 "slopes = per-patient least squares; patient bootstrap of the patients with >= 2 pre-event endoscopies (RandomState(0)); mixed model = statsmodels MixedLM, REML, random patient intercept; "
                 "a negative slope against time to event = the score rises towards the event; time-dependent AUROC re-tabulated from results/paper_final/horizon_foldstrat (IPCW, fold-stratified)")
    res["L3"] = R; dump(res)
# ================================================================== L4: disagreement within an endoscopy
elif TASK == "l4":
    S = setup(); O = S["O"]; es = S["es"]; R = {}
    def fs_auc(score, lab, fold):
        out = []
        for j in range(10):
            num = den = 0.0
            for k in np.unique(fold[:, j]):
                m = (fold[:, j] == k) & np.isfinite(score[:, j]); l = lab[m]; n1 = int(l.sum()); n0 = len(l) - n1
                if n1 == 0 or n0 == 0: continue
                rk = rankdata(score[m, j]); num += rk[l == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
            if den: out.append(num / den)
        return float(np.mean(out)) if out else np.nan
    def unit_block(groups, lab, fold, upat, nm):
        """groups: sample index arrays per unit; returns AUROCs (max, SD, max+SD rank sum) for logit L-IMG and L-CNV with patient bootstrap"""
        o = {"n_units": len(groups), "n_progressor_units": int(lab.sum())}; ui = np.unique(upat); upos = {p: np.where(upat == p)[0] for p in ui}; rng = np.random.RandomState(0); DRw = []
        while len(DRw) < NB:
            b = np.concatenate([upos[ui[i]] for i in rng.choice(len(ui), len(ui))])
            if 0 < lab[b].sum() < len(b): DRw.append(b)
        for m in ("L-IMG", "L-CNV"):
            L = lg(O[m]); MX = np.stack([L[s].max(0) for s in groups]); SDv = np.stack([L[s].std(0, ddof=1) for s in groups])
            CB = np.stack([rankdata(MX[:, j]) + rankdata(SDv[:, j]) for j in range(10)], 1)
            sc = {"max": MX, "sd": SDv, "max_plus_sd": CB}; obs = {k: fs_auc(v, lab, fold) for k, v in sc.items()}; B = pd.DataFrame([{k: fs_auc(v[b], lab[b], fold[b]) for k, v in sc.items()} for b in DRw])
            o[m] = {"auroc": {k: {"value": r3(obs[k]), "ci95": ci2(B[k])} for k in sc}, "max_plus_sd_minus_max": dstat(obs["max_plus_sd"] - obs["max"], B["max_plus_sd"] - B["max"]), "sd_minus_max": dstat(obs["sd"] - obs["max"], B["sd"] - B["max"])}
        o["draws"] = len(DRw); return o
    sel = [e for e, s in enumerate(es) if len(s) >= 2]; G = [es[e] for e in sel]
    R["endoscopy"] = unit_block(G, S["elab"][sel], S["EF"][sel], S["epat"][sel], "endoscopy")
    pg = [np.concatenate([es[e] for e in np.where(S["pidx"] == i)[0]]) for i in range(S["P_"])]; ps = [i for i in range(S["P_"]) if len(pg[i]) >= 2]
    R["patient"] = unit_block([pg[i] for i in ps], S["plab"][ps], np.stack([S["FO"][pg[i][0]] for i in ps]), S["up"][ps], "patient")
    R["note"] = ("endoscopies with >= 2 pre-event samples (patients with >= 2 pre-event samples for the patient level); SD = within-unit standard deviation (ddof 1) of the logit outer prediction, per repeat; max = maximum logit; "
                 "max + SD = rank of the max plus rank of the SD among the units in that repeat (fixed equal weights); fold-stratified AUROC (unit fold = its patient's outer fold), single model; patient bootstrap 2,000 draws (RandomState(0)) with ranks fixed")
    res["L4"] = R; dump(res)
# ================================================================== L5: who each modality misses
elif TASK == "l5":
    S = setup(); P_ = S["P_"]; plab = S["plab"]; es = S["es"]; pidx = S["pidx"]; isP = plab == 1; isN = ~isP; X = S["OUTS"]
    FLm = {}
    for nm, key in (("H&E", "img"), ("copy number", "cnv")):
        FLm[nm] = [patient_arrays(S, X[key] >= thr_arr(S, key, q)[0], np.zeros((S["E"], 10), int))["FL"] for q in Q3]
    def interp(L, tau):
        f = np.array([l_[isN].sum() for l_ in L])
        if tau <= f[0]: return L[0], 0.0, int(tau < f[0] - 1e-9)
        if tau >= f[-1]: return L[-1], 2.0, int(tau > f[-1] + 1e-9)
        i = int(np.searchsorted(f, tau, side="right") - 1); i = min(i, len(f) - 2); w = (tau - f[i]) / (f[i + 1] - f[i]) if f[i + 1] > f[i] else 0.0
        return (1 - w) * L[i] + w * L[i + 1], i + w, 0
    PC = {"H&E": np.zeros((P_, 10)), "copy number": np.zeros((P_, 10))}; info = {"target_np_flagged": [], "grid_position": {"H&E": [], "copy number": []}, "clamped": {"H&E": 0, "copy number": 0}}
    for j in range(10):
        tau = (FLm["H&E"][1][isN, j].sum() + FLm["copy number"][1][isN, j].sum()) / 2; info["target_np_flagged"].append(tau)
        for nm in PC:
            v, pos, cl = interp([l_[:, j] for l_ in FLm[nm]], tau); PC[nm][:, j] = v; info["grid_position"][nm].append(pos); info["clamped"][nm] += cl
    h, c = PC["H&E"], PC["copy number"]; cells = {"both": h * c, "H&E only": h * (1 - c), "copy number only": (1 - h) * c, "neither": (1 - h) * (1 - c)}
    R = {"target_non_progressor_patients_flagged_mean": r2(np.mean(info["target_np_flagged"])), "realised_non_progressor_flagged_mean": {nm: r2(PC[nm][isN].sum(0).mean()) for nm in PC},
         "grid_position_mean_0_80_1_90_2_95": {nm: r3(np.mean(v)) for nm, v in info["grid_position"].items()}, "repeats_clamped_to_grid_end": info["clamped"],
         "progressor_patients_caught_mean": {nm: r2(PC[nm][isP].sum(0).mean()) for nm in PC}, "n_progressor_patients": int(isP.sum()), "n_non_progressor_patients": int(isN.sum())}
    R["cross_tab_progressor_patients_mean_over_repeats"] = {k: m5(round(float(v[isP].sum(0).mean()), 1)) for k, v in cells.items()}
    R["cross_tab_non_progressor_patients_mean_over_repeats"] = {k: m5(round(float(v[isN].sum(0).mean()), 1)) for k, v in cells.items()}
    # discordant patients: modal category over repeats (mean category probabilities); aggregate description
    prob = np.stack([cells[k].mean(1) for k in cells], 1); cat = np.array(list(cells))[prob.argmax(1)]
    GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); SDS = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids)
    TILES = SDS.tissue_tiles.values.astype(float); SCAN = SDS.scanner.astype(str).values; rows = S["rows"]; e1, e2 = np.quantile(TILES[rows], [1 / 3, 2 / 3])
    fdu = sheet().drop_duplicates("combined_name").set_index("combined_name"); BATCH = pd.Series(ids).map(fdu.Batch).values; bc = pd.Series(y).groupby(BATCH).agg(lambda s: s.nunique()); SINGLE = pd.Series(BATCH).map(bc).values == 1
    GN = {0: "NDBE", 1: "ID", 2: "LGD"}
    def pinfo(i):
        s = np.concatenate([es[e] for e in np.where(pidx == i)[0]]); mt = np.median(TILES[s]); sc = sorted(set(SCAN[s]))
        return {"grade": GN[int(GR[s].max())], "n_endo": int((pidx == i).sum()), "years": float(S["PY"][i]), "tert": int(0 if mt <= e1 else 1 if mt <= e2 else 2) + 1,
                "batch": "any sample in a single-class batch" if SINGLE[s].any() else "all samples in mixed batches", "scanner": sc[0] if len(sc) == 1 else "both scanners"}
    PIn = {i: pinfo(i) for i in np.where(isP)[0]}; desc = {}
    for k in cells:
        g = [PIn[i] for i in np.where(isP & (cat == k))[0]]; o = {"n_patients": m5(len(g))}
        if len(g) >= 5:
            df = pd.DataFrame(g); o.update({"highest_pre_event_grade": {a: m5(b) for a, b in df.grade.value_counts().items()}, "pre_event_endoscopies": {"median": float(df.n_endo.median()), "iqr": [float(df.n_endo.quantile(0.25)), float(df.n_endo.quantile(0.75))]},
                                            "years_first_sample_to_event": {"median": r2(df.years.median()), "iqr": [r2(df.years.quantile(0.25)), r2(df.years.quantile(0.75))]},
                                            "tissue_tertile": {str(a): m5(b) for a, b in df.tert.value_counts().sort_index().items()}, "batch_type": {a: m5(b) for a, b in df.batch.value_counts().items()}, "scanner": {a: m5(b) for a, b in df.scanner.value_counts().items()}})
        elif len(g): o["note"] = "fewer than 5 patients: not described"
        desc[k] = o
    anyd = isP & ((prob[:, 1] + prob[:, 2]) > 1e-9); g = [PIn[i] for i in np.where(anyd)[0]]; o = {"n_patients": m5(len(g)), "of_which_modal_both": m5(int((anyd & (cat == "both")).sum()))}
    if len(g) >= 5:
        df = pd.DataFrame(g); o.update({"highest_pre_event_grade": {a: m5(b) for a, b in df.grade.value_counts().items()}, "pre_event_endoscopies": {"median": float(df.n_endo.median()), "iqr": [float(df.n_endo.quantile(0.25)), float(df.n_endo.quantile(0.75))]},
                                        "years_first_sample_to_event": {"median": r2(df.years.median()), "iqr": [r2(df.years.quantile(0.25)), r2(df.years.quantile(0.75))]},
                                        "tissue_tertile": {str(a): m5(b) for a, b in df.tert.value_counts().sort_index().items()}, "batch_type": {a: m5(b) for a, b in df.batch.value_counts().items()}, "scanner": {a: m5(b) for a, b in df.scanner.value_counts().items()},
                                        "direction_mean_repeats": {"H&E only": r2(cells["H&E only"][anyd].sum(0).mean()), "copy number only": r2(cells["copy number only"][anyd].sum(0).mean())}})
    elif len(g): o["note"] = "fewer than 5 patients: not described"
    desc["discordant in at least one repeat (H&E only or copy number only)"] = o
    R["progressor_patients_by_modal_category"] = desc
    R["mean_probability_by_category_progressors"] = {k: r3(prob[isP, q].mean()) for q, k in enumerate(cells)}
    R["note"] = ("P1 (max L-IMG) and P2 (max L-CNV) at 80/90/95% training endoscopy sensitivity; per repeat the common target is the mean of the two policies' non-progressor ever-flagged counts at 90%, and each policy is interpolated linearly in that count "
                 "between adjacent grid cells (a patient's catch probability = the weighted mix of its two cells; clamped at the grid ends); cross-tab cells are expected counts (the two mixes taken as independent), mean over repeats; "
                 "a patient's category = the one with the highest mean probability over repeats; 'discordant in at least one repeat' = any repeat with a non-zero H&E-only or copy-number-only probability; groups of fewer than 5 patients are not described; tissue tertile = tertile (pre-event sample edges) of the patient's median tissue tiles; batch type from the sheet Batch (single-class batch = all its samples one class, as R55)")
    res["L5"] = R; dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
