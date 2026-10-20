"""Review round 7c: robustness of the R49 per-endoscopy gate (docs/paper_review_round7c_gate_robustness.md @ 6794545). R49 exactly as rv7_round7.py r49
(endoscopy score = max L-IMG over its pre-event samples, the triage's fold t_low / c* / t_seq, middle band sequences the top-L-IMG sample, called by L-LATE)
re-scored on every R33 Claim T row whose stored predictions exist: stored outer and inner predictions (tb_common.py loaders, paper_triage.md rules), the
round-2 / round-4 refit predictions, the round-7 refit outputs (killcoyne_mm/review_round7/refit/) and the stored F3 refit-bootstrap draws. Nothing is refitted.
Helpers copied unchanged from rv7_round7.py (matched_calls, all_calls, use_set, load_var, cutoff_arrays, db_link), rv7b_gate.py (comparator / bootstrap /
random-one engine of merge_<row>), rv4_round4.py (R32b residualisation, earliest_index, r33_f3 cut-off rule) and rv6_round6.py (load_refit, sheet flags).
Row-level files only on the cluster; aggregates (no identifiers, no dates) -> results/paper_final/review_round7c/<TASK>.json.
TASK = row_<name> | f3 | merge
       <name> primary | ndbe | neverdys | earliest_endo | ge2 | excl1y | grid_<ql>_<qh> | tis_lt256 | tis_ge256 | tis_lt256pure | tis_ge256pure | pkg | r36 | f5
              | devlam | tgate | r38a | r53 | r43a | r43b | r43c | r54 | r55 | r52a | r52b"""
import os, sys, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_common as TC
from tb_common import *
from tb_common import _outer
PRESPEC = "6794545"; OUT = T + "/results/paper_final/review_round7c"; OUT7 = T + "/results/paper_final/review_round7"; OUT7B = T + "/results/paper_final/review_round7b"; OUT4 = T + "/results/paper_final/review_round4"
R7 = M + "/review_round7"; R7C = M + "/review_round7c"; ROW4 = M + "/review_round4"; RR2 = M + "/review_round2"; SR = M + "/scanner_robustness"
KD = T + "/feasibility/paper_plan/killcoyne"; SWG = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; DBX = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
REL = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
NF3 = 200
if os.environ.get("RV7C_NB"):   # smoke mode: fewer bootstrap draws (and F3 draws), aggregates written aside (never to results/)
    NB = int(os.environ["RV7C_NB"]); TC.NB = NB; NF3 = int(os.environ.get("RV7C_F3N", "5")); OUT = R7C + "/smoke_json"
for d in (OUT, R7C): os.makedirs(d, exist_ok=True)
TASK = os.environ["TASK"]; MM = {"A*": "L-CNV", "B*": "L-LATE", "C*": "L-IMG"}; NSEEDS = 100
CS = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample").reindex(ids); YEAR = pd.to_numeric(CS.year, errors="coerce").values
GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float)
SDS = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); TILES = SDS.tissue_tiles.values.astype(float)
TIME = SM.time.values.astype(float); MBF = SM.mbf.values.astype(float)
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv")); hzi = lambda nm: sorted(glob.glob(f"{HZ}/inner/{nm}_rep_*_fold_*.csv"))
FO0 = _outer(kvf(0))[1]
REFIT = ("r43a", "r43b", "r43c", "r54", "r52a", "r52b")
# ------------------------------------------------------------------ helpers copied unchanged (rv7_round7.py / rv7b_gate.py)
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
def dump(res, name=None):
    res["prespec"] = PRESPEC; json.dump(res, open(f"{OUT}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV7C DONE", name or TASK, flush=True)
# ------------------------------------------------------------------ row loaders copied from rv6_round6.py / rv4_round4.py (unchanged)
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
def sheet_ids():
    fd = pd.read_csv(SWG + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str)
    fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; f = fd.drop_duplicates("combined_name").set_index("combined_name").reindex(ids); assert f.Batch.notna().all(); return f
def residualise(O, I, popm):
    """R32b: L-IMG residualised on biopsy year and log tissue tiles (outer per repeat on the pre-event held-out scores; inner per (repeat, fold) on that set's pre-event rows); L-LATE unchanged"""
    rows = np.where(popm)[0]; assert np.isfinite(YEAR).all() and np.isfinite(TILES).all() and (TILES > 0).all()
    Xc = np.column_stack([np.ones(n), YEAR, np.log(TILES)]); O2 = dict(O); Rz = np.zeros_like(O["L-IMG"])
    for j in range(10): coef, *_ = np.linalg.lstsq(Xc[rows], O["L-IMG"][rows, j], rcond=None); Rz[:, j] = O["L-IMG"][:, j] - Xc @ coef
    O2["L-IMG"] = Rz; I2 = dict(I); I2["L-IMG"] = {}
    for key, s in I["L-IMG"].items(): idx = s.index.values; fit = idx[popm[idx]]; coef, *_ = np.linalg.lstsq(Xc[fit], s.reindex(fit).values, rcond=None); I2["L-IMG"][key] = pd.Series(s.values - Xc[idx] @ coef, index=idx)
    return O2, I2
# ------------------------------------------------------------------ R53 event dates: rv7_round7.py db_link / sheet and the r53 event logic (unchanged)
def sheet():
    fd = pd.read_csv(SWG + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; return fd
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
    pe["cnv"] = pe.CNVAbsPath.map(lambda p: os.path.basename(str(p))); pe["d"] = pd.to_datetime(pe.Date, errors="coerce"); pdate = pe.dropna(subset=["d"]).drop_duplicates("cnv").set_index("cnv").d
    mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str); smf = pd.read_csv(SWG + "/slide_matching.csv", dtype=str).drop_duplicates("Slide file").set_index("Slide file")
    mp = mp[mp.cnv.notna()].join(smf, on="base"); sdate_slide = pd.to_datetime(mp.EndoscopyDate, errors="coerce").groupby(mp.cnv).max()
    fd["sd"] = fd.combined_name.map(pdate).fillna(fd.combined_name.map(sdate_slide))
    return {"fd": fd, "link": link, "pt": pt, "HG": HG, "pdate": pdate, "sd_of": fd.drop_duplicates("combined_name").set_index("combined_name").sd}
def r53_drop(popm):
    """R53 primary row: pre-event progressor samples dated on or after the earlier of the sheet event endoscopy and the first database HGD+ report"""
    L = db_link(); fd = L["fd"]; pdate = L["pdate"]; sd_of = L["sd_of"]; pt = L["pt"]
    A = pd.read_csv(KD + "/kr_samples.csv", dtype=str).set_index("Sample"); fdu = fd.drop_duplicates("combined_name").set_index("combined_name"); A["mbf"] = pd.to_numeric(A.index.map(fdu["Months before final"]), errors="coerce"); A["ogd"] = A.index.map(fdu["Path_class_per_OGD"])
    hgA = A[(A.Pathology.isin(["HGD", "IMC"])) | (A.ogd.isin(["HGD", "IMC", "HGD/IMC"]))]; tevA = hgA.groupby("Patient").mbf.max()
    SD_ = pd.Series(ids).map(pdate).values; assert pd.notna(SD_).all(), "analysed sample without an exact date"; SD_ = pd.to_datetime(SD_); ev = {}
    for p in sorted(set(pat[y == 1])):
        sh = pd.NaT
        if p in tevA.index:
            d_ = [sd_of.get(s) for s in A[(A.Patient == p) & (A.mbf == tevA[p])].index]; d_ = [x for x in d_ if pd.notna(x)]; sh = min(d_) if d_ else pd.NaT
        pid = L["link"].get(p); db = pt[(pt.pid == pid) & (pt.g >= 3)].d.min() if pid is not None else pd.NaT
        ev[p] = min([x for x in (sh, db) if pd.notna(x)], default=pd.NaT)
    EV = pd.to_datetime(pd.Series(pat).map(ev).values); return popm & (y == 1) & pd.notna(EV) & (SD_ >= EV)
def r55_background():
    PX = pd.concat([pd.read_csv(f, dtype={"key": str, "created": str}) for f in sorted(glob.glob(R7 + "/px_disc/chunk_*.csv"))])
    mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(set(ids))].drop_duplicates(["cnv", "npz"]); assert len(PX) == len(mp) and PX.npz.nunique() == len(mp), (len(PX), len(mp))
    g = PX.groupby("key"); BG = (g.n_background.sum() / g.n_tiles.sum()).reindex(ids).values; assert np.isfinite(BG).all(); return BG
# ------------------------------------------------------------------ rows
LABEL = {"primary": "primary: all pre-event endoscopies", "ndbe": "NDBE-only samples", "neverdys": "never-dysplastic patients", "earliest_endo": "earliest pre-event endoscopy per patient",
         "ge2": "endoscopies with ≥ 2 samples", "excl1y": "excluding progressor samples within 1 year of the event",
         "tis_lt256": "< 256 tissue tiles (endoscopy by its top-L-IMG sample)", "tis_ge256": "≥ 256 tissue tiles (endoscopy by its top-L-IMG sample)",
         "tis_lt256pure": "< 256 tissue tiles (all samples of the endoscopy in the stratum)", "tis_ge256pure": "≥ 256 tissue tiles (all samples of the endoscopy in the stratum)",
         "pkg": "package copy-number features (joint rebuild)", "r36": "per-sample segmentation (R36)", "f5": "H-optimus-0 extractor (F5, stored predictions)", "devlam": "λ by deviance (B5)",
         "tgate": "tissue- and year-adjusted gate scores (R32b rule)", "r38a": "R38(a): sheet 'excluded' flag removed", "r53": "R53: database event dates", "r43a": "R43(a): non-progressor samples < 1 year before last endoscopy, refit",
         "r43b": "R43(b): < 2 years, refit", "r43c": "R43(c): later-HGD non-progressor patient, refit", "r54": "R54: later-disease relabel, refit", "r55": "R55: background share > 25% excluded",
         "r52a": "R52(a): + 17 same-block samples, refit", "r52b": "R52(b): + 24 samples, refit"}
for ql in (90, 95, 98):
    for qh in (85, 90, 95): LABEL[f"grid_{ql}_{qh}"] = f"cut-off grid: t_low {ql}% / t_high {qh}%" + (" (primary cell)" if (ql, qh) == (95, 90) else "")
def setup(row):
    """predictions, population for the cut-offs (popm), calls and the evaluation endoscopies (index arrays) of one row"""
    kw = {}; keep = None; d = {}
    if row in REFIT:
        V = pd.read_csv(R7 + f"/sets/{row}.csv", dtype={"Sample": str, "Patient": str, "Endoscopy": str}); use_set(V); O, I, FO = load_var(row); popm = PRE.copy()
        d["definition"] = f"round-7 refit set {row} and its refit outer and inner predictions (killcoyne_mm/review_round7/refit/{row}); all cut-offs and comparators re-chosen"
    else:
        popm = PRE.copy()
        if row == "pkg": O, I, FO = load("pkg"); d["definition"] = "stored package-feature L-CNV (cfg_01, cnv_pkg inner); all cut-offs re-chosen (R33 row 3)"
        elif row == "r36": O, I, FO = load_refit(need(sorted(glob.glob(f"{ROW4}/r36_outer/cnv_pkgps_rep_*.csv")), 10), kvf(2), need(sorted(glob.glob(f"{ROW4}/r36_inner/cnv_pkgps_rep_*_fold_*.csv")), 100), hzi("img")); d["definition"] = "R36 per-sample segmented package L-CNV (cnv_pkgps); L-IMG stored; all cut-offs re-chosen"
        elif row == "f5": O, I, FO = load_refit(kvf(0), need(sorted(glob.glob(f"{SR}/cv/cv_*_hoptimus0.csv")), 10), hzi("cnv_their"), need(sorted(glob.glob(f"{RR2}/f5_inner/img_rep_*_fold_*.csv")), 100)); d["definition"] = "H-optimus-0 L-IMG (stored F5 outer and inner predictions); all cut-offs re-chosen"
        elif row == "devlam":
            fc = need(sorted(glob.glob(f"{ROW4}/b5_inner/cnv_their_rep_*_fold_*.csv")), 100); fi = need(sorted(glob.glob(f"{ROW4}/b5_inner/img_rep_*_fold_*.csv")), 100); col = "pred_dev" if "pred_dev" in pd.read_csv(fc[0], nrows=1).columns else "pred"
            O, I, FO = load_refit(need(sorted(glob.glob(f"{RR2}/f4/cnv_their_rep_*.csv")), 10), need(sorted(glob.glob(f"{RR2}/f4/img_rep_*.csv")), 10), fc, fi, col=col); d["definition"] = f"λ by deviance: F4 outer and B5 inner predictions ({col}); all cut-offs re-chosen"
        else: O, I, FO = load("their")
        if row == "ndbe": popm &= NDBE; d["definition"] = "pre-event NDBE samples; cut-offs and comparators re-chosen on NDBE training samples (R33 row 2); endoscopies keep their NDBE samples"
        elif row == "ge2": cnt = pd.Series(ENDO[PRE]).value_counts(); popm &= pd.Series(ENDO).map(cnt).fillna(0).values >= 2; d["definition"] = "endoscopies with ≥ 2 pre-event samples; cut-offs re-chosen on them (round 7b (c))"
        elif row.startswith("grid_"): _, ql, qh = row.split("_"); kw = {"ql": int(ql) / 100, "qh": int(qh) / 100}; d["definition"] = f"t_low at {ql}% training sensitivity, t_high at {qh}% training specificity; t_seq, B* and C* re-matched at that cell (R33 row 8)"
        elif row == "tgate": O, I = residualise(O, I, popm); d["definition"] = "L-IMG residualised on biopsy year and log tissue tiles (R32b); endoscopy score, bands and the sequenced sample use the residualised L-IMG; middle-band call by stored L-LATE; cut-offs re-chosen"
        elif row == "neverdys": D1 = (GR == 1) | (GR == 2); rows_ = np.where(popm)[0]; dysp = set(pat[rows_][D1[rows_]]); keep = np.array([p not in dysp for p in pat]); d["definition"] = "patients with no pre-event ID or LGD sample; calls and cut-offs as stored (R33 row 5)"
        elif row == "excl1y": keep = ~((y == 1) & (TIME <= 1)); d["definition"] = "progressor samples within 1 year of the event removed from evaluation; endoscopies keep their other samples; calls and cut-offs as stored (R33 row 7)"
        elif row == "earliest_endo": keep = earliest_index() == 1; d["definition"] = "each patient's earliest pre-event endoscopy (all its samples); calls and cut-offs as stored (R33 row 13)"
        elif row == "r38a": f = sheet_ids(); keep = ~(f.excluded.fillna("").str.strip() == "1").values; d["definition"] = "samples with the sheet's 'excluded' flag removed from evaluation; calls and cut-offs as stored (R38 a)"
        elif row == "r53": keep = ~r53_drop(popm); d["definition"] = "pre-event progressor samples dated on or after the earlier of the sheet event and the first database HGD+ report removed; calls and cut-offs as stored (R53 primary row)"
        elif row == "r55": keep = r55_background() <= 0.25; d["definition"] = "samples with background share > 0.25 of the stored tile positions removed; calls and cut-offs as stored (R55 a)"
        elif row.startswith("tis_") or row == "primary" or row in ("pkg", "r36", "f5", "devlam"): pass
        else: raise SystemExit(f"unknown row {row}")
    CALLS, band, info, thr, sstar = all_calls(O, I, FO, popm, **kw); rows = np.where(popm)[0]
    if keep is not None: rows = rows[keep[rows]]
    eu, es_ = groups_of(rows); es = [rows[g] for g in es_]
    if row.startswith("tis_"):
        lt = row in ("tis_lt256", "tis_lt256pure"); st = (TILES < 256) if lt else (TILES >= 256); mimg = O["L-IMG"].mean(1)
        top = np.array([s[np.argmax(mimg[s])] for s in es]); toprep = np.array([[s[np.argmax(O["L-IMG"][s, j])] for j in range(10)] for s in es])
        sel = np.array([st[s].all() for s in es]) if row.endswith("pure") else st[top]
        d["all_endoscopies_top_sample_stratum_changes_in_some_repeat_share"] = r3(np.mean([(st[toprep[e]] != st[top[e]]).any() for e in range(len(es))]))   # over all endoscopies, before selection
        d["definition"] = ("endoscopies whose pre-event samples are all " if row.endswith("pure") else "endoscopies whose top sample (highest L-IMG averaged over the 10 repeats) has ") + ("< 256" if lt else "≥ 256") + " tissue tiles; the endoscopy keeps all its samples; calls and cut-offs as stored (R32a strata)"
        eu = eu[sel]; es = [s for s, k in zip(es, sel) if k]
    return dict(O=O, I=I, FO=FO, popm=popm, CALLS=CALLS, band=band, info=info, sstar=sstar, eu=eu, es=es, meta=d)
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
def evaluate(S):
    """rv7b_gate.py merge engine with R49 in place of the nested gate: comparators, random-one (own / matched to R49's endoscopy specificity), patient bootstrap"""
    O, FO, CALLS, band, info, es = (S[k] for k in ("O", "FO", "CALLS", "band", "info", "es")); GC, NS, GB = r49_calls(O, info, FO, es)
    E = len(es); rows = np.concatenate(es); elab = np.array([int(y[s].max()) for s in es]); epat = np.array([pat[s][0] for s in es]); nsz = np.array([len(s) for s in es])
    up = np.unique(epat); P_ = len(up); pe = [np.where(epat == p)[0] for p in up]; plab = np.array([int(elab[i].max()) for i in pe])
    anyE = lambda C: np.array([C[s].any(0) for s in es]); midE = np.array([(band[s] == 1).sum(0) for s in es]); allE = np.repeat(nsz[:, None], 10, 1)
    RU = {"r49": (GC, NS), "Bstar_endo": (anyE(CALLS["B*"]), allE), "B_endo": (anyE(CALLS["B"]), allE), "triage": (anyE(CALLS["T"]), midE), "Cstar_endo": (anyE(CALLS["C*"]), np.zeros((E, 10), int))}
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
        for nm, (F, NS_) in RU.items():
            Fe = F[eidx]; o[f"sens_{nm}"] = Fe[le == 1].mean(0); o[f"spec_{nm}"] = 1 - Fe[le == 0].mean(0); Pp = PF[nm][pidx]; o[f"pmiss_{nm}"] = (~Pp[lp == 1]).sum(0).astype(float); o[f"pflag_{nm}"] = Pp[lp == 0].sum(0).astype(float)
            Ne = NS_[eidx]; o[f"shs_{nm}"] = Ne.sum(0) / tot; o[f"she_{nm}"] = (Ne > 0).mean(0)
        RM = rand_matched(eidx, o["spec_r49"]); Ro = RBown[:, eidx]
        for nm, A_ in (("rand_matched", RM), ("rand_own", Ro)):
            o[f"sens_{nm}"] = A_[:, le == 1, :].mean(1).mean(0); o[f"spec_{nm}"] = 1 - A_[:, le == 0, :].mean(1).mean(0); o[f"shs_{nm}"] = np.full(10, len(eidx) / tot); o[f"she_{nm}"] = np.ones(10)
        return o
    obs = stats(np.arange(E), np.arange(P_)); rng = np.random.RandomState(0); DR = []
    while len(DR) < NB:
        pidx = rng.choice(P_, P_); eidx = np.concatenate([pe[i] for i in pidx])
        if 0 < plab[pidx].sum() < P_: DR.append(stats(eidx, pidx))
    m_ = lambda k: float(np.mean(obs[k])); dm = lambda k: np.array([d[k].mean() for d in DR])
    RMo = rand_matched(np.arange(E), obs["spec_r49"]); toP = lambda A_: np.stack([A_[:, i, :].any(1) for i in pe], 1)
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
    R = {**nfo(rows), "n_endoscopies": int(E), "n_progressor_endoscopies": int(elab.sum()), "n_progressor_patients_in_units": int(plab.sum()), "n_non_progressor_patients_in_units": int((plab == 0).sum()), "draws": len(DR), "n_seeds": NSEEDS,
         "folds_no_middle_band": info["no_mid"], "t_seq_fallback": info["fallback"], "r49_endoscopy_band_shares": {nm: r3((GB == g).mean()) for g, nm in ((0, "low"), (1, "middle"), (2, "high"))}}
    MAIN = ("r49", "Bstar_endo", "B_endo", "triage", "Cstar_endo", "rand_matched", "rand_own"); R["rules"] = {nm: rule(nm) for nm in MAIN}; R["r49_minus"] = {b: minus("r49", b) for b in MAIN[1:]}
    # between-repeat variance (R33 / R15 logic, as rv7b (e)) of R49 against B*-endo; per-repeat bounds
    pts = obs["sens_r49"] - obs["sens_Bstar_endo"]; PD = np.array([d["sens_r49"] - d["sens_Bstar_endo"] for d in DR]); lows = np.percentile(PD, 5, axis=0); avg = PD.mean(1); vb = float(avg.var(ddof=1)); vr = float(pts.var(ddof=1)); dd = float(pts.mean())
    sp = obs["spec_r49"] - obs["spec_Bstar_endo"]; SPD = np.array([d["spec_r49"] - d["spec_Bstar_endo"] for d in DR]).mean(1)
    R["between_repeat"] = {"per_repeat_d_endoscopy_sens": [r3(v) for v in pts], "per_repeat_one_sided_lower": [r3(v) for v in lows], "per_repeat_ci95": [ci2(PD[:, j]) for j in range(10)], "n_repeats_lower_above_-0.05": int((lows > -0.05).sum()),
                           "share_repeats_lower_above_-0.05": r3((lows > -0.05).mean()), "delta_repeat_mean": r3(dd), "V_boot": round(vb, 6), "V_rep": round(vr, 6), "standard_one_sided_lower": lo1(avg),
                           "variance_inflated_one_sided_lower": r3(dd - 1.645 * np.sqrt(vb + vr)), "variance_inflated_ci95": [r3(dd - 1.96 * np.sqrt(vb + vr)), r3(dd + 1.96 * np.sqrt(vb + vr))], "non_inferior_at_-0.05": bool(dd - 1.645 * np.sqrt(vb + vr) > -0.05),
                           "d_endoscopy_spec_repeat_mean": r3(sp.mean()), "d_endoscopy_spec_variance_inflated_ci95": [r3(sp.mean() - 1.96 * np.sqrt(SPD.var(ddof=1) + sp.var(ddof=1))), r3(sp.mean() + 1.96 * np.sqrt(SPD.var(ddof=1) + sp.var(ddof=1)))]}
    R["note"] = ("endoscopy = (patient, endoscopy number) over the row's evaluated pre-event samples; flagged if the rule flags it; single model (per-repeat values averaged over the 10 repeats); patient bootstrap (RandomState(0), R49 draw loop, "
                 "one-class draws rejected) with fitted models and cut-offs fixed; B*-endo / (B)-endo / C*-endo = any sample positive by the stored-rule call; random-one = R41 seeds 0-99, own threshold = stored (B) call of the picked sample, "
                 "matched = one L-LATE threshold per seed x repeat giving R49's endoscopy specificity in that repeat (re-matched in each draw)")
    return R
APP = "max | sample-level cut-offs | highest | L-LATE"
res = {"task": TASK}
# ================================================================== one robustness row
if TASK.startswith("row_"):
    row = TASK[4:]; S = setup(row); R = {"row": row, "label": LABEL[row], "status": "done", **S["meta"], **evaluate(S)}
    def pick(g): return [g["endoscopy_sensitivity"], g["endoscopy_specificity"], g["share_samples_sequenced"], g["share_endoscopies_sequenced"], g.get("progressor_patients_missed"), g.get("non_progressor_patients_flagged")]
    def chk(got, exp): return {"got": got, "expected": exp, "reproduced": bool(all(a is not None and b is not None and abs(a - b) < 0.0015 for a, b in zip(got, exp)))}
    if row == "primary":
        r49 = J(f"{OUT7}/r49.json")["R49"]["rules"]; r46 = J(f"{OUT7}/r46.json")["R46"]["a_matched_specificity"]["random_own_threshold"]
        R["checks"] = {"r49_gate_vs_R49": {**chk(pick(R["rules"]["r49"]), pick(r49["gate"])), "sens_ci_got_expected": [R["rules"]["r49"]["sens_ci95"], r49["gate"]["sens_ci95"]]},
                       "r49_spec_samples_endoscopies_vs_prespec_0.907_0.560_0.255_0.505": chk(pick(R["rules"]["r49"])[:4], [0.907, 0.560, 0.255, 0.505]),
                       "Bstar_endo_vs_0.923_0.530": chk(pick(R["rules"]["Bstar_endo"])[:2], [0.923, 0.530]), "triage_vs_R49": chk(pick(R["rules"]["triage"]), pick(r49["triage"])),
                       "rand_own_vs_R46": chk(pick(R["rules"]["rand_own"])[:2], [r46["endoscopy_sensitivity"]["mean_over_seeds"], r46["endoscopy_specificity"]["mean_over_seeds"]])}
    b7 = {"primary": "primary", "ndbe": "a_ndbe", "ge2": "c_ge2", "r52a": "d_r52a"}.get(row)
    if b7 and J(f"{OUT7B}/merge_{b7}.json"):   # round 7b apparent result of the R49 variant on the same population (same cut-offs, same bootstrap)
        a = J(f"{OUT7B}/merge_{b7}.json")["R7B"]["apparent_optimistic"][APP]; s = R["r49_minus"]["Bstar_endo"]["d_endoscopy_sens"]
        R.setdefault("checks", {})["vs_round7b_apparent_R49_variant"] = chk(pick(R["rules"]["r49"]) + [s["delta"], s["one_sided_95_lower"]], pick(a) + [a["minus_Bstar_endo"]["d_endoscopy_sens"]["delta"], a["minus_Bstar_endo"]["d_endoscopy_sens"]["one_sided_95_lower"]])
    res["R7C"] = R; dump(res)
# ================================================================== refit bootstrap (stored F3 draws; cut-offs re-derived per draw by the rv4 r33_f3 rule)
elif TASK == "f3":
    fl = lambda m, d, p: f"{RR2}/f3/{m}_draw_{d:03d}_{p}.csv"; have = [d for d in range(200) if all(os.path.exists(fl(m, d, p)) for m in ("cnv_their", "img") for p in ("outer", "inner"))]
    if len(have) < 200:
        res["R7C"] = {"row": "f3", "status": f"not available: stored outer and inner predictions exist for {len(have)} of 200 F3 draws"}; dump(res); raise SystemExit(0)
    PREs = pd.Series(PRE, index=ids); ENDOs = pd.Series(ENDO, index=ids)
    def rd(f): D = pd.read_csv(f, dtype={"Sample": str, "Patient": str}); D["pre"] = PREs.reindex(D.Sample).values; return D
    def draw(d):
        oc, oi = rd(fl("cnv_their", d, "outer")), rd(fl("img", d, "outer")); ic, ii = rd(fl("cnv_their", d, "inner")), rd(fl("img", d, "inner"))
        E = oc.merge(oi[["Sample", "copy", "pred"]], on=["Sample", "copy"], suffixes=("_cnv", "_img")); assert len(E) == len(oc) == len(oi); E = E[E.pre].reset_index(drop=True); E["late"] = (E.pred_cnv + E.pred_img) / 2
        Tr = ic.merge(ii[["Sample", "copy", "fold", "pred"]], on=["Sample", "copy", "fold"], suffixes=("_cnv", "_img")); assert len(Tr) == len(ic) == len(ii); Tr = Tr[Tr.pre].reset_index(drop=True); Tr["late"] = (Tr.pred_cnv + Tr.pred_img) / 2
        yy = E.y.values == 1; f = E.fold.values; nE = len(E); call = {s: np.zeros(nE, bool) for s in ("triage", "B", "B*")}; band = np.zeros(nE, int); TL = np.zeros(nE); CSt = np.zeros(nE); TS = np.zeros(nE)
        for k in range(1, 11):   # rv4_round4.py r33_f3, unchanged
            t = Tr[Tr.fold == k]; s_tr = t.pred_img.values; yc = t.y.values == 1; li = t.late.values
            tl = thr_sens(s_tr[yc], 0.95); cs = c_star(s_tr[~yc], 0.90); mid = (s_tr >= tl) & (s_tr <= cs) if tl <= cs else np.zeros(len(t), bool)
            cmid = li[mid & yc]; ts = thr_sens(cmid, 0.8) if len(cmid) >= 5 else thr_sens(li[yc], 0.8)
            tcall = np.where(s_tr < tl, False, np.where(s_tr > cs, True, li >= ts)) if tl <= cs else s_tr > cs; s_star = 1 - tcall[~yc].mean()
            xc = np.sort(li[~yc]); q = int(np.ceil(s_star * len(xc))); thr_bs = xc[q - 1] if q > 0 else -np.inf; thr_b = thr_sens(li[yc], 0.8)
            te = f == k; s = E.pred_img.values[te]; la = E.late.values[te]
            bd = np.where(s < tl, 0, np.where(s > cs, 2, 1)) if tl <= cs else np.where(s > cs, 2, 0); band[te] = bd
            call["triage"][te] = np.where(bd == 1, la >= ts, bd == 2); call["B"][te] = la >= thr_b; call["B*"][te] = la > thr_bs; TL[te] = tl; CSt[te] = cs; TS[te] = ts
        eo = ENDOs.reindex(E.Sample).values; assert pd.notna(eo).all(); ek = (pd.Series(eo).astype(str) + "#" + E["copy"].astype(str).reset_index(drop=True)).values
        ue, inv = np.unique(ek, return_inverse=True); es = [np.where(inv == g)[0] for g in range(len(ue))]; img = E.pred_img.values; late = E.late.values
        G = len(es); F = {nm: np.zeros(G, bool) for nm in ("r49", "Bstar_endo", "B_endo", "triage")}; NSq = np.zeros(G, int); NT = np.zeros(G, int); el = np.zeros(G, bool)
        for g, s in enumerate(es):
            assert len(set(f[s])) == 1 and len(set(yy[s])) == 1; tl, cs, ts = TL[s[0]], CSt[s[0]], TS[s[0]]; sc = img[s].max(); el[g] = yy[s[0]]
            b = (2 if sc > cs else 0) if tl > cs else (0 if sc < tl else (2 if sc > cs else 1))
            if b == 1: F["r49"][g] = late[s[np.argmax(img[s])]] >= ts; NSq[g] = 1
            else: F["r49"][g] = b == 2
            F["Bstar_endo"][g] = call["B*"][s].any(); F["B_endo"][g] = call["B"][s].any(); F["triage"][g] = call["triage"][s].any(); NT[g] = (band[s] == 1).sum()
        se = {nm: (v[el].mean(), 1 - v[~el].mean()) for nm, v in F.items()}; sps = {s: (c[yy].mean(), 1 - c[~yy].mean()) for s, c in call.items()}
        o = {f"{a}_{nm}": se["r49"][i] - se[nm][i] for nm in ("Bstar_endo", "B_endo", "triage") for i, a in ((0, "d_sens_vs"), (1, "d_spec_vs"))}
        o.update({"r49_sens": se["r49"][0], "r49_spec": se["r49"][1], "Bstar_endo_sens": se["Bstar_endo"][0], "Bstar_endo_spec": se["Bstar_endo"][1], "share_samples_r49": NSq.sum() / nE, "share_endoscopies_r49": float((NSq > 0).mean()),
                  "share_samples_triage": NT.sum() / nE, "n_endoscopies": G, "sample_level_triage_minus_Bstar_d_sens": sps["triage"][0] - sps["B*"][0]})
        return o
    Sd = [draw(d) for d in range(NF3)]; A = {k: np.array([s[k] for s in Sd], float) for k in Sd[0]}; rng = np.random.RandomState(0); RS = [rng.randint(0, NF3, NF3) for _ in range(2000)]
    def mc(a): p5 = np.array([np.percentile(a[r], 5) for r in RS]); return {"mean": r3(a.mean()), "median": r3(np.median(a)), "ci95_percentile": ci2(a), "one_sided_5th_percentile": lo1(a), "sd": r3(a.std(ddof=1)), "non_inferior_at_-0.05": bool(np.percentile(a, 5) > -0.05), "mc_se_of_5th_percentile": round(float(p5.std(ddof=1)), 4), "mc_ci95_of_5th_percentile": ci2(p5), "share_of_draws_below_-0.05": r3((a < -0.05).mean())}
    R = {"row": "f3", "label": "refit bootstrap (F3, 200 draws)", "status": "done", "draws_scored": NF3, "mc_resamples": 2000, **{k: mc(v) for k, v in A.items() if k != "n_endoscopies"}, "n_endoscopies_per_draw": {"median": r3(np.median(A["n_endoscopies"])), "min": int(A["n_endoscopies"].min()), "max": int(A["n_endoscopies"].max())},
         "definition": "stored F3 refit-bootstrap draws (one stratified 10-fold CV per patient-bootstrap draw with lambda refitted); per draw the triage cut-offs, t_seq and the B* / (B) thresholds re-derived from the draw's stored inner predictions by the rv4 r33_f3 rule; R49 and the endoscopy comparators on the draw's held-out pre-event endoscopies (resampled copies = distinct endoscopies); no vote"}
    f3r = J(f"{OUT4}/r33_f3.json")
    if f3r and NF3 == 200: R["checks"] = {"sample_level_triage_minus_Bstar_mean_and_5th_vs_R33": {"got": [R["sample_level_triage_minus_Bstar_d_sens"]["mean"], R["sample_level_triage_minus_Bstar_d_sens"]["one_sided_5th_percentile"]], "expected": [f3r["R33_f3"]["claim_T_d_sens_triage_minus_Bstar"]["mean"], f3r["R33_f3"]["claim_T_d_sens_triage_minus_Bstar"]["one_sided_5th_percentile"]]}}
    res["R7C"] = R; dump(res)
# ================================================================== merge: the robustness table, summary counts and checks (aggregates of the row JSONs only)
elif TASK == "merge":
    ORDER = [("primary", "0")] + [(r, str(i)) for i, r in ((1, "ndbe"), (2, "neverdys"), (3, "earliest_endo"), (4, "ge2"), (5, "excl1y"))] + [(f"grid_{a}_{b}", "6") for a in (90, 95, 98) for b in (85, 90, 95)] + \
            [(r, "7") for r in ("tis_lt256", "tis_ge256", "tis_lt256pure", "tis_ge256pure")] + [(r, str(i)) for i, r in ((8, "pkg"), (9, "r36"), (10, "f5"), (11, "devlam"), (12, "tgate"))] + \
            [(r, "13") for r in ("r38a", "r53", "r43a", "r43b", "r43c", "r54", "r55")] + [(r, "14") for r in ("r52a", "r52b")]
    def verdict(lo, ci, within):
        if within: return "within"
        return "shift" if ci and ci[1] is not None and ci[1] < 0 else "width"
    TAB = []
    for r, it in ORDER:
        j = J(f"{OUT}/row_{r}.json")
        if j is None: TAB.append({"item": it, "row": r, "label": LABEL[r], "status": "missing (task not run)"}); continue
        x = j["R7C"]
        if x.get("status") != "done": TAB.append({"item": it, "row": r, "label": LABEL[r], "status": x.get("status")}); continue
        g = x["rules"]["r49"]; b = x["rules"]["Bstar_endo"]; t = x["rules"]["triage"]; d = x["r49_minus"]["Bstar_endo"]; s = d["d_endoscopy_sens"]
        TAB.append({"item": it, "row": r, "label": LABEL[r], "status": "done", "n": {k: x[k] for k in ("n_endoscopies", "n_progressor_endoscopies", "n_samples", "n_patients", "n_progressor_patients")},
                    "r49_sens_spec": [g["endoscopy_sensitivity"], g["endoscopy_specificity"]], "Bstar_endo_sens_spec": [b["endoscopy_sensitivity"], b["endoscopy_specificity"]], "d_sens": s["delta"], "ci95": s["ci95"], "one_sided_lower": s["one_sided_95_lower"],
                    "within_-0.05": s["non_inferior_at_-0.05"], "verdict": verdict(s["one_sided_95_lower"], s["ci95"], s["non_inferior_at_-0.05"]), "d_spec": d["d_endoscopy_spec"]["delta"], "d_spec_ci95": d["d_endoscopy_spec"]["ci95"],
                    "share_samples_sequenced": g["share_samples_sequenced"], "share_endoscopies_sequenced": g["share_endoscopies_sequenced"], "triage_share_samples_sequenced": t["share_samples_sequenced"],
                    "patients_missed_flagged": [g.get("progressor_patients_missed"), g.get("non_progressor_patients_flagged")], "d_sens_vs_triage": x["r49_minus"]["triage"]["d_endoscopy_sens"]["delta"], "d_sens_vs_triage_lower": x["r49_minus"]["triage"]["d_endoscopy_sens"]["one_sided_95_lower"]})
    pr = J(f"{OUT}/row_primary.json")
    if pr:
        br = pr["R7C"]["between_repeat"]; sh = pr["R7C"]["rules"]["r49"]
        TAB.append({"item": "15", "row": "between_repeat", "label": "between-repeat variance added (as R33)", "status": "done", "d_sens": br["delta_repeat_mean"], "ci95": br["variance_inflated_ci95"], "one_sided_lower": br["variance_inflated_one_sided_lower"],
                    "within_-0.05": br["non_inferior_at_-0.05"], "verdict": verdict(br["variance_inflated_one_sided_lower"], br["variance_inflated_ci95"], br["non_inferior_at_-0.05"]), "d_spec": br["d_endoscopy_spec_repeat_mean"], "share_samples_sequenced": sh["share_samples_sequenced"], "share_endoscopies_sequenced": sh["share_endoscopies_sequenced"]})
        TAB.append({"item": "15", "row": "per_repeat", "label": "per-repeat bounds (repeats within −0.05 of 10)", "status": "done", "n_repeats_within": br["n_repeats_lower_above_-0.05"], "per_repeat_d_sens": br["per_repeat_d_endoscopy_sens"], "per_repeat_one_sided_lower": br["per_repeat_one_sided_lower"],
                    "verdict": "count (not an interval; outside the summary counts)"})
    else:
        for rr, lab in (("between_repeat", "between-repeat variance added (as R33)"), ("per_repeat", "per-repeat bounds (repeats within −0.05 of 10)")): TAB.append({"item": "15", "row": rr, "label": lab, "status": "missing (task not run)"})
    f3 = J(f"{OUT}/f3.json")
    if f3 is None: TAB.append({"item": "15", "row": "f3", "label": "refit bootstrap (F3, 200 draws)", "status": "missing (task not run)"})
    elif f3["R7C"].get("status") != "done": TAB.append({"item": "15", "row": "f3", "label": "refit bootstrap (F3, 200 draws)", "status": f3["R7C"]["status"]})
    else:
        q = f3["R7C"]["d_sens_vs_Bstar_endo"]; ok = q["non_inferior_at_-0.05"]
        TAB.append({"item": "15", "row": "f3", "label": "refit bootstrap (F3, 200 draws)", "status": "done", "d_sens": q["mean"], "ci95": q["ci95_percentile"], "one_sided_lower": q["one_sided_5th_percentile"], "within_-0.05": ok,
                    "verdict": verdict(q["one_sided_5th_percentile"], q["ci95_percentile"], ok), "d_spec": f3["R7C"]["d_spec_vs_Bstar_endo"]["mean"], "share_samples_sequenced": f3["R7C"]["share_samples_r49"]["mean"],
                    "share_endoscopies_sequenced": f3["R7C"]["share_endoscopies_r49"]["mean"], "mc_se_of_5th_percentile": q["mc_se_of_5th_percentile"], "share_of_draws_below_-0.05": q["share_of_draws_below_-0.05"]})
    rob = [r for r in TAB if r["row"] != "primary" and r["row"] != "per_repeat"]; done = [r for r in rob if r["status"] == "done"]
    na = [r["label"] for r in rob if r["status"] != "done"]
    SUM = {"rows_in_summary": len(rob), "within_-0.05": int(sum(r["verdict"] == "within" for r in done)), "fail_by_shift": int(sum(r["verdict"] == "shift" for r in done)), "fail_by_width": int(sum(r["verdict"] == "width" for r in done)),
           "not_available": len(na), "not_available_rows": na, "rows_with_lower_share_sequenced_than_triage": int(sum(1 for r in done if r.get("triage_share_samples_sequenced") is not None and r["share_samples_sequenced"] < r["triage_share_samples_sequenced"])),
           "note": "robustness rows 1-15 (the primary row excluded, the per-repeat count row excluded because it is not an interval); within = one-sided 95% lower bound > -0.05; shift = not within and two-sided 95% CI entirely below 0; width = not within, CI includes 0"}
    CK = {}
    if pr: CK["primary"] = pr["R7C"].get("checks")
    g = J(f"{OUT}/row_grid_95_90.json")
    if pr and g:
        a, b = pr["R7C"], g["R7C"]; keys = ("rules", "r49_minus", "between_repeat", "n_endoscopies", "n_samples")
        CK["grid_primary_cell_equals_primary_row"] = bool(all(json.dumps(a[k], sort_keys=True) == json.dumps(b[k], sort_keys=True) for k in keys))
    for r in ("ndbe", "ge2", "r52a"):
        j = J(f"{OUT}/row_{r}.json")
        if j and j["R7C"].get("checks"): CK[f"{r}_vs_round7b_apparent"] = j["R7C"]["checks"].get("vs_round7b_apparent_R49_variant")
    if f3 and f3["R7C"].get("checks"): CK["f3"] = f3["R7C"]["checks"]
    res["R7C"] = {"table": TAB, "summary": SUM, "checks": CK, "primary": pr["R7C"] if pr else None}; dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
