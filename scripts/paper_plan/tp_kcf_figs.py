"""Two papers, agent KCF (docs/two_papers_KCF.md): Killcoyne et al. 2020 (Nat Med 26:1726) figures reproduced on our cohort, with the H&E score
and the combined score next to copy number. Runs on the cluster (row-level inputs); writes aggregates and figure panels only.
TASK = smoke | boot | figs | lcmerge
  smoke  : known numbers (fold-stratified sample AUROC on the 571 pre-event samples: L-CNV 0.801, stored L-IMG 0.847, clean L-IMG 0.811,
           clean L-LATE 0.875, published P 0.823), inner-prediction coverage, metadata checks -> smoke.json
  boot   : point estimates and 2,000-draw patient bootstrap (RandomState(0), predictions fixed) of the key numbers -> stats.json
  figs   : panel PNGs (one per model: cn / he / comb, same axes; or one 'ours' panel) -> killcoyne_mm/two_papers_KCF/figs/  (+ figs.json)
  lcmerge: Extended Data Fig. 6 learning curve from tp_kcf_lc.R refits -> lc.json
Models: CN = L-CNV (shipped Killcoyne matrix, our 10 x 10 patient-grouped folds, nested lambda); HE = clean-tile L-IMG (agent ART refit);
COMB = per-repeat mean of the two probabilities (clean L-LATE); PUB = Killcoyne's published leave-one-patient-out probability (reference only).
Risk classes: low / moderate / high. Published P: Killcoyne's fixed cuts (low <= 0.3, high >= 0.5). Our models: cuts chosen in each outer fold on the
training patients' inner out-of-fold predictions to hit Killcoyne's published operating points (87% of training progressor samples above the low
cut; 82% of training non-progressor samples below the high cut). Display classes = median class over the 10 repeats."""
import os, sys, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, mannwhitneyu
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan")
from tb_common import *
TASK = os.environ.get("TASK", "smoke")
KCF = M + "/two_papers_KCF"; OUTR = T + "/results/paper_final/two_papers_KCF"; FIG = KCF + "/figs" + os.environ.get("FIGSUF", "")
for d in (OUTR, FIG): os.makedirs(d, exist_ok=True)
def dump(o, name): json.dump(o, open(f"{OUTR}/{name}.json", "w"), indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)); print("KCF DONE", name, flush=True)
# ------------------------------------------------------------------ predictions
O, I, FO = load("their")
def read_refit(d):
    fo = sorted(glob.glob(f"{d}/outer_rep_*.csv")); fi = sorted(glob.glob(f"{d}/inner_rep_*.csv")); assert len(fo) == 10 and len(fi) == 10
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); po = lambda c: D.pivot_table(index="Sample", columns="rep", values=c, dropna=False).reindex(ids)[REPS].values
    assert (po("fold").astype(int) == FO).all(); Di = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fi])
    return po("pred"), {(r, k): pd.Series(d_.pred.values, index=ix[d_.Sample].values) for (r, k), d_ in Di.groupby(["rep", "fold"])}
CLO, CLI = read_refit(M + "/two_papers_ART/refit/clean")
KP = SM.k_prob.values.astype(float); assert np.isfinite(KP).all()
P = {"CN": O["L-CNV"], "HE": CLO, "COMB": (O["L-CNV"] + CLO) / 2, "PUB": np.tile(KP[:, None], (1, 10))}
IN = {"CN": I["L-CNV"], "HE": CLI}; IN["COMB"] = {k: (IN["CN"][k] + IN["HE"][k].reindex(IN["CN"][k].index)) / 2 for k in IN["CN"]}
MODS = ["CN", "HE", "COMB"]; ALLM = MODS + ["PUB"]
NAME = {"CN": "Copy number (L-CNV)", "HE": "H&E (clean-tile L-IMG)", "COMB": "H&E + copy number (mean)", "PUB": "Killcoyne published P"}
SHORT = {"CN": "copy number", "HE": "H&E", "COMB": "H&E + copy number"}
SENS_LO, SPEC_HI = 0.87, 0.82
TL, TH = {}, {}
for m in MODS:
    tl = np.zeros((10, 10)); th = np.zeros((10, 10))
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            s = IN[m][(r, k)]; v = s.values; yy = y[s.index.values]
            tl[j, k - 1] = np.quantile(v[yy == 1], 1 - SENS_LO); th[j, k - 1] = max(np.quantile(v[yy == 0], SPEC_HI), tl[j, k - 1])
    TL[m], TH[m] = tl, th
TL["PUB"], TH["PUB"] = np.full((10, 10), 0.3), np.full((10, 10), 0.5)
RI = np.arange(10)[None, :]
CUTLO = {m: TL[m][RI, FO - 1] for m in ALLM}; CUTHI = {m: TH[m][RI, FO - 1] for m in ALLM}
CLS = {m: np.where(P[m] <= CUTLO[m], 0, np.where(P[m] >= CUTHI[m], 2, 1)) for m in ALLM}
CLS_FIX = {m: np.where(P[m] <= 0.3, 0, np.where(P[m] >= 0.5, 2, 1)) for m in ALLM}
MC = {m: np.floor(np.median(CLS[m], 1) + 0.5).astype(int) for m in ALLM}
PM = {m: P[m].mean(1) for m in ALLM}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
# ------------------------------------------------------------------ metadata
SX = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
fd = pd.read_csv(SX + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]
fd = fd.drop_duplicates("combined_name").set_index("combined_name")
LOC = pd.to_numeric(pd.Series(ids).map(fd["Esophageal location"]), errors="coerce").values
PATH = SM.Pathology.values.astype(str); P53 = pd.to_numeric(SM["P53 IHC"], errors="coerce").values
MBF = SM.mbf.values.astype(float); TEV = SM.tev.values.astype(float)
CNV = pd.read_csv(M + "/cnv_their_C.csv").set_index("sample_id").reindex(ids); assert CNV.notna().all().all()
WIN = [c for c in CNV.columns if c != "cx" and (int(c.split(":")[1].split("-")[1]) - int(c.split(":")[1].split("-")[0])) < 6e6]
ARM = [c for c in CNV.columns if c != "cx" and c not in WIN]; CX = CNV["cx"].values
Zs = lambda A: (A - A.mean(0)) / np.where(A.std(0) > 0, A.std(0), 1)   # each window / arm standardised across the 676 samples; altered = |z| > 2
Zw = Zs(CNV[WIN].values); Za = Zs(CNV[ARM].values)
ALT_W = (np.abs(Zw) > 2).sum(1); ALT_A = (np.abs(Za) > 2).sum(1)
wchr = np.array([int(c.split(":")[0]) for c in WIN]); achr = np.array([int(c.split(":")[0]) for c in ARM])
# ------------------------------------------------------------------ units
up = np.unique(pat); NPAT = len(up); pidx = {q: i for i, q in enumerate(up)}
ENDO2 = np.array([f"{q}|{m_:.0f}|{int(p)}" for q, m_, p in zip(pat, SM.mbf.values.astype(float), PRE)])   # endoscopy unit = patient x months-before-final (the time axis Killcoyne plot); 28 of the P1A "Endoscopy" groups span several months
eu = pd.unique(ENDO2); E_rows = [np.where(ENDO2 == e)[0] for e in eu]; NE = len(eu)
E_pat = np.array([pidx[pat[r[0]]] for r in E_rows]); E_y = np.array([y[r[0]] for r in E_rows]); E_F = np.array([FO[r[0]] for r in E_rows])
E_mbf = np.array([MBF[r].max() for r in E_rows]); E_pre = np.array([PRE[r].all() for r in E_rows]); E_tev = np.array([TEV[r[0]] for r in E_rows])
GRADE = {"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}; G = np.array([GRADE.get(p, -1) for p in PATH]); assert (G >= 0).all()
E_maxg = np.array([G[r].max() for r in E_rows]); E_p53 = np.array([np.nanmax(P53[r]) if np.isfinite(P53[r]).any() else np.nan for r in E_rows])
P_y = np.array([y[pat == q][0] for q in up]); P_F = np.array([FO[pat == q][0] for q in up])
S_of = [np.where(pat == q)[0] for q in up]; E_of = [np.where(E_pat == i)[0] for i in range(NPAT)]
def eagg(X, f, rows_mask=None):
    return np.stack([f(X[r], 0) for r in E_rows])
EPmean = {m: eagg(P[m], np.mean) for m in ALLM}; EPmax = {m: eagg(P[m], np.max) for m in ALLM}
ECLS = {m: eagg(CLS[m], np.max) for m in ALLM}; EMC = {m: np.array([MC[m][r].max() for r in E_rows]) for m in ALLM}
E_order = {}   # endoscopy order within patient (by time: larger mbf = earlier)
for i in range(NPAT):
    es = E_of[i]; E_order[i] = es[np.argsort(-E_mbf[es])]
PRE_pat = np.array([PRE[S_of[i]].any() for i in range(NPAT)])
def fsc(X, yv, F):
    out = []
    for j in range(X.shape[1]):
        x = X[:, j]; num = den = 0.0
        for k in np.unique(F[:, j]):
            mk = F[:, j] == k; yk = yv[mk]; n1 = int(yk.sum()); n0 = len(yk) - n1
            if n1 == 0 or n0 == 0: continue
            rk = rankdata(x[mk]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
def pooled_auc(s, yy):
    yy = np.asarray(yy); n1 = yy.sum(); n0 = len(yy) - n1
    return float((rankdata(s)[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan
# ------------------------------------------------------------------ Fig 3b bins and Fig 4b rule
BINS = [(">96", 96, 1e9), ("72-96", 72, 96), ("48-72", 48, 72), ("24-48", 24, 48), ("12-24", 12, 24), ("1-12", 0, 12)]
def e_bin(e):
    if E_y[e] != 1 or not np.isfinite(E_tev[e]): return None
    d = E_mbf[e] - E_tev[e]
    if abs(d) < 1e-9: return "Endpoint"
    for nm, lo, hi in BINS:
        if lo < d <= hi: return nm
    return None
E_BIN = np.array([e_bin(e) for e in range(NE)], dtype=object); BIN_NAMES = [b[0] for b in BINS] + ["Endpoint"]
REC = ["Immediate RFA", "Recheck in 6-12 months", "Recheck in 12-24 months", "Regular surveillance in 3-5 years"]
def rule(c1, c2, p53=False):   # BarrettsProgressionRisk rxRules on the first two pre-event endoscopies (endoscopy class = highest sample class)
    c = [c1, c2]
    if c.count(2) == 2: return 0
    if c.count(2) == 1 or p53: return 1
    if 1 in c: return 2
    return 3
def recs(m):
    out = {}
    for i in range(NPAT):
        es = [e for e in E_order[i] if E_pre[e]]
        if len(es) < 2: continue
        if m == "GRADE":   # our mapping of grade/p53 into the same rule: LGD -> high, ID -> moderate, NDBE -> low; aberrant p53 -> rule 2
            cc = [{0: 0, 1: 1, 2: 2}.get(int(E_maxg[e]), 2) for e in es[:2]]; ab = any(E_p53[e] == 1 for e in es[:2])
            out[i] = rule(cc[0], cc[1], ab)
        else: out[i] = rule(int(EMC[m][es[0]]), int(EMC[m][es[1]]))
    return out
REC_OF = {m: recs(m) for m in ALLM + ["GRADE"]}
# ------------------------------------------------------------------ statistics on a patient draw (indices into up; copies are distinct units)
PUBLISHED = {"fig2b_ndbe_P_high": 0.605, "fig2b_ndbe_NP_low": 0.647, "sens_lo_sample": 0.87, "spec_lo_sample": 0.65, "sens_hi_sample": 0.72, "spec_hi_sample": 0.82,
             "ed10_NP": [0.65, 0.17, 0.18], "ed10_P": [0.13, 0.15, 0.72], "fig3b": dict(zip(BIN_NAMES, [0.5, 0.5, 0.625, 0.618, 0.783, 0.853, 0.930])),
             "fig3b_n": dict(zip(BIN_NAMES, [16, 12, 24, 34, 23, 34, 43])), "fig4b_P": [19, 9, 5, 2], "fig4b_NP": [7, 6, 7, 21],
             "ed2": {"per sample": [0.87, 0.87, 0.65], "excluding HGD/IMC": [0.93, 0.90, 0.76], "endoscopy mean": [0.88, 0.90, 0.69], "endoscopy max": [0.87, 0.90, 0.64],
                     "patient mean": [0.87, 0.82, 0.70], "patient max": [0.84, 0.90, 0.44]},
             "ed7a_p53": {"NDBE": 0.11, "ID": 0.17, "LGD": 0.25, "HGD": 0.64, "IMC": 0.75}, "ed1_cx_auc": 0.6, "ed1c_p": [6e-8, 5.54e-11], "ed1d_p": 7e-4,
             "auc_lopo": 0.87, "ed6_auc": {"28": 0.70, "38": 0.67, "48": 0.73, "58": 0.77, "68": 0.78, "78": 0.85, "88": 0.87}}
def stats(c):
    o = {}; si = np.concatenate([S_of[i] for i in c]); ei = np.concatenate([E_of[i] for i in c]); c = np.asarray(c)
    sp = si[PRE[si]]; ep = ei[E_pre[ei]]; cp = c[PRE_pat[c]]
    for m in ALLM:
        X = P[m]
        o[f"{m}|auc|sample_pre"] = fsc(X[sp], y[sp], FO[sp]); o[f"{m}|auc|sample_all"] = fsc(X[si], y[si], FO[si])
        o[f"{m}|auc|endo_mean"] = fsc(EPmean[m][ep], E_y[ep], E_F[ep]); o[f"{m}|auc|endo_max"] = fsc(EPmax[m][ep], E_y[ep], E_F[ep])
        PPm = np.stack([P[m][S_of[i][PRE[S_of[i]]]].mean(0) for i in cp]); PPx = np.stack([P[m][S_of[i][PRE[S_of[i]]]].max(0) for i in cp])
        o[f"{m}|auc|pat_mean"] = fsc(PPm, P_y[cp], P_F[cp]); o[f"{m}|auc|pat_max"] = fsc(PPx, P_y[cp], P_F[cp])
        # sensitivity / specificity at the low cut (positive = not low) per unit; high cut per sample
        lo = CUTLO[m]
        def ss(Xu, cut, yy):
            pos = Xu > cut; return float(pos[yy == 1].mean()), float(1 - pos[yy == 0].mean())
        for nm, Xu, cut, yy in (("sample_all", X[si], lo[si], y[si]), ("sample_pre", X[sp], lo[sp], y[sp]),
                                ("endo_mean", EPmean[m][ep], lo[[E_rows[e][0] for e in ep]], E_y[ep]), ("endo_max", EPmax[m][ep], lo[[E_rows[e][0] for e in ep]], E_y[ep]),
                                ("pat_mean", PPm, lo[[S_of[i][0] for i in cp]], P_y[cp]), ("pat_max", PPx, lo[[S_of[i][0] for i in cp]], P_y[cp])):
            o[f"{m}|sens_lo|{nm}"], o[f"{m}|spec_lo|{nm}"] = ss(Xu, cut, yy)
        hi = X[sp] >= CUTHI[m][sp]; o[f"{m}|sens_hi|sample_pre"] = float(hi[y[sp] == 1].mean()); o[f"{m}|spec_hi|sample_pre"] = float(1 - hi[y[sp] == 0].mean())
        Cs = CLS[m][si]; ys = y[si]; nd = PATH[si] == "NDBE"
        o[f"{m}|fig2b|ndbe_P_high"] = float((Cs[nd & (ys == 1)] == 2).mean()); o[f"{m}|fig2b|ndbe_NP_low"] = float((Cs[nd & (ys == 0)] == 0).mean())
        for lab, yy in (("NP", 0), ("P", 1)):
            for k, cn in enumerate(("low", "mod", "high")): o[f"{m}|ed10|{lab}_{cn}"] = float((Cs[ys == yy] == k).mean())
        Eb = E_BIN[ei]; Eh = ECLS[m][ei] == 2
        for b in BIN_NAMES:
            mk = Eb == b; o[f"{m}|fig3b|{b}"] = float(Eh[mk].mean()) if mk.any() else np.nan
        R = REC_OF[m]; rr = [(R[i], P_y[i]) for i in c if i in R]
        o[f"{m}|fig4b|P_RFA"] = sum(1 for v, yy in rr if yy == 1 and v == 0); o[f"{m}|fig4b|NP_3to5y"] = sum(1 for v, yy in rr if yy == 0 and v == 3)
        o[f"{m}|fig4b|P_n"] = sum(1 for v, yy in rr if yy == 1); o[f"{m}|fig4b|NP_n"] = sum(1 for v, yy in rr if yy == 0)
        # Extended Data Fig. 1c analogue: patient-level AUROC of the score at the first endoscopy (pre-event first endoscopies only)
        fe = np.array([E_order[i][0] for i in c]); keep = E_pre[fe]
        o[f"{m}|ed1c|auc_first_endo"] = pooled_auc(EPmean[m][fe[keep]].mean(1), E_y[fe[keep]])
    fe = np.array([E_order[i][0] for i in c]); keep = E_pre[fe]
    for nm, v in (("windows", ALT_W), ("arms", ALT_A)):
        o[f"CNcount|ed1c|auc_first_endo_{nm}"] = pooled_auc(np.array([v[E_rows[e]].mean() for e in fe[keep]]), E_y[fe[keep]])
    o["cx|auc|sample_pre"] = pooled_auc(CX[sp], y[sp])
    Eb = E_BIN[ei]; Eg = E_maxg[ei] >= 2
    for b in BIN_NAMES[:-1]:
        mk = Eb == b; o[f"GRADE|fig3b|{b}"] = float(Eg[mk].mean()) if mk.any() else np.nan
    R = REC_OF["GRADE"]; rr = [(R[i], P_y[i]) for i in c if i in R]
    o["GRADE|fig4b|P_RFA"] = sum(1 for v, yy in rr if yy == 1 and v == 0); o["GRADE|fig4b|NP_3to5y"] = sum(1 for v, yy in rr if yy == 0 and v == 3)
    return o
def pdraws(nb=NB, seed=0):
    rng = np.random.RandomState(seed); out = []
    while len(out) < nb:
        c = rng.choice(NPAT, NPAT); sp = np.concatenate([S_of[i] for i in c]); sp = sp[PRE[sp]]
        if 0 < y[sp].sum() < len(sp): out.append(c)
    return out
r3 = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 3)
# ================================================================== smoke
if TASK == "smoke":
    rows = np.where(PRE)[0]; o = {}
    for nm, X in (("L-CNV", P["CN"]), ("L-IMG stored", O["L-IMG"]), ("L-IMG clean", P["HE"]), ("L-LATE clean", P["COMB"]), ("L-LATE stored", O["L-LATE"]), ("P published", P["PUB"])):
        o[nm] = r3(fsc(X[rows], y[rows], FO[rows]))
    exp = {"L-CNV": 0.801, "L-IMG stored": 0.847, "L-IMG clean": 0.811, "L-LATE clean": 0.875, "L-LATE stored": 0.889, "P published": 0.823}
    o["match"] = {k: abs(o[k] - v) <= 0.0005 for k, v in exp.items()}
    cov = []
    for m in MODS:
        for j, r in enumerate(REPS):
            for k in range(1, 11):
                tr = np.where(FO[:, j] != k)[0]; cov.append(set(IN[m][(r, k)].index) == set(tr))
    o["inner_covers_all_training_rows"] = bool(all(cov))
    o["n"] = {"setC_samples": int(n), "setC_patients": NPAT, "pre_samples": int(PRE.sum()), "pre_patients": int(PRE_pat.sum()), "pre_P_patients": int(P_y[PRE_pat].sum()),
              "pre_endoscopies": int(E_pre.sum()), "setC_endoscopies": NE}
    o["loc_missing"] = int(np.isnan(LOC).sum()); o["loc_range"] = [float(np.nanmin(LOC)), float(np.nanmax(LOC))]
    o["endo_units_mixed_months"] = int(sum(np.ptp(MBF[r]) > 0 for r in E_rows)); o["mean_altered_windows"] = r3(ALT_W.mean()); o["mean_altered_arms"] = r3(ALT_A.mean()); o["windows"] = len(WIN); o["arms"] = len(ARM)
    o["cnv_col_sd_median"] = r3(np.median(CNV[WIN].std().values)); o["cnv_col_mean_abs_median"] = r3(np.median(np.abs(CNV[WIN].mean().values)))
    o["cut_lo_median"] = {m: r3(np.median(TL[m])) for m in MODS}; o["cut_hi_median"] = {m: r3(np.median(TH[m])) for m in MODS}
    o["fig3b_bins"] = {b: int((E_BIN == b).sum()) for b in BIN_NAMES}; o["fig4b_patients"] = {m: len(REC_OF[m]) for m in ("CN", "GRADE")}
    o["pathology_counts"] = pd.Series(PATH).value_counts().to_dict(); o["p53_known"] = int(np.isfinite(P53).sum())
    for ft, m in (("cnv", "CN"), ("clean", "HE")):   # tp_kcf_lc.R with K above the training size must equal the stored outer predictions (repeat 1)
        f = f"{KCF}/lc/{ft}_k999_rep_01.csv"
        if os.path.exists(f):
            Dl = pd.read_csv(f, dtype={"Sample": str}).set_index("Sample").reindex(ids); o[f"lc_smoke_{ft}_max_abs_diff"] = float(np.abs(Dl.pred.values - P[m][:, 0]).max())
    dump(o, "smoke")
# ================================================================== point estimates + bootstrap
if TASK == "boot":
    pt = stats(np.arange(NPAT)); D = pdraws(int(os.environ.get("NBOOT", NB))); B = [stats(c) for c in D]; keys = list(pt)
    A = {k: np.array([b[k] for b in B], float) for k in keys}
    res = {"point": {k: r3(v) for k, v in pt.items()}, "ci": {k: [r3(np.nanpercentile(A[k], 2.5)), r3(np.nanpercentile(A[k], 97.5))] for k in keys}, "diff": {}}
    for k in keys:
        if k.startswith("CN|"):
            for m in ("HE", "COMB"):
                k2 = m + k[2:]; d = A[k2] - A[k]
                res["diff"][f"{m}-CN|{k[3:]}"] = {"delta": r3(pt[k2] - pt[k]), "ci": [r3(np.nanpercentile(d, 2.5)), r3(np.nanpercentile(d, 97.5))]}
    res["published"] = PUBLISHED; res["n_draws"] = len(D)
    res["note"] = ("Patient bootstrap over the 80 set-C patients (RandomState(0), 2,000 draws, predictions fixed; pre-event statistics use the drawn patients' pre-event rows). "
                   "AUROC = fold-stratified concordance averaged over 10 repeats (P1A); rates averaged over repeats; fig4b counts on median classes. "
                   "Cuts: CN/HE/COMB chosen on training patients' inner OOF predictions (87% sensitivity low cut, 82% specificity high cut); PUB fixed 0.3/0.5.")
    res["fixed_cut_rates"] = {}
    for m in MODS:
        sp = np.where(PRE)[0]; C_ = CLS_FIX[m][sp]; yy = y[sp]
        res["fixed_cut_rates"][m] = {"sens_lo": r3(((C_ > 0)[yy == 1]).mean()), "spec_lo": r3(1 - ((C_ > 0)[yy == 0]).mean()), "sens_hi": r3(((C_ == 2)[yy == 1]).mean()), "spec_hi": r3(1 - ((C_ == 2)[yy == 0]).mean())}
    dump(res, os.environ.get("STATS", "stats"))
# ================================================================== learning curve merge (Extended Data Fig. 6)
if TASK == "lcmerge":
    rows = np.where(PRE)[0]; res = {}
    for K in sorted({int(f.split("_k")[1].split("_")[0]) for f in glob.glob(KCF + "/lc/cnv_k*_rep_*.csv")}):
        fs = {ft: sorted(glob.glob(f"{KCF}/lc/{ft}_k{K}_rep_*.csv")) for ft in ("cnv", "clean")}
        if len(fs["cnv"]) < 10 or len(fs["clean"]) < 10: res[str(K)] = {"incomplete": {ft: len(v) for ft, v in fs.items()}}; continue
        X = {}; NZ = {}; NT = None
        for ft, fl in fs.items():
            Dd = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fl]); X[ft] = Dd.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values
            Ff = Dd.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values; assert (Ff == FO).all()
            NZ[ft] = Dd.groupby(["rep", "fold"]).nonzero.first().groupby(level=0).mean().values; NT = Dd.groupby(["rep", "fold"]).n_train_pat.first().mean()
        X["comb"] = (X["cnv"] + X["clean"]) / 2; o = {"n_train_patients_mean": r3(NT)}
        for ft in ("cnv", "clean", "comb"):
            per = [fsc(X[ft][rows][:, [j]], y[rows], FO[rows][:, [j]]) for j in range(10)]
            o[ft] = {"auc_mean": r3(np.mean(per)), "auc_sem": r3(np.std(per, ddof=1) / np.sqrt(10)), "auc_per_rep": [r3(v) for v in per]}
            if ft in NZ: o[ft]["nonzero_mean"] = r3(NZ[ft].mean()); o[ft]["nonzero_sem"] = r3(NZ[ft].std(ddof=1) / np.sqrt(10))
        res[str(K)] = o
    full = {}
    for ft, X in (("cnv", P["CN"]), ("clean", P["HE"]), ("comb", P["COMB"])):
        per = [fsc(X[rows][:, [j]], y[rows], FO[rows][:, [j]]) for j in range(10)]; full[ft] = {"auc_mean": r3(np.mean(per)), "auc_sem": r3(np.std(per, ddof=1) / np.sqrt(10))}
    res["full_stored"] = full; res["note"] = "held-out = full outer test fold; training patients of each outer fold subsampled to K (status-stratified); 10 repeats; sem over repeats (as Killcoyne ED6)."
    dump(res, "lc")
# ================================================================== figures
if TASK == "figs":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "font.family": "DejaVu Sans"})
    ST = json.load(open(f"{OUTR}/{os.environ.get('STATS', 'stats')}.json")); PT, CI, DF = ST["point"], ST["ci"], ST["diff"]; PUBV = ST["published"]
    LC = json.load(open(f"{OUTR}/lc.json")) if os.path.exists(f"{OUTR}/lc.json") else None
    COL = {"CN": "#1f5fa8", "HE": "#e07b00", "COMB": "#2e8b3a", "GREY": "#808080", "PUB": "#1f5fa8"}
    SHADE = {"CN": ["#c6dbef", "#6baed6", "#08306b"], "HE": ["#fde0c2", "#fdae6b", "#a63603"], "COMB": ["#c7e9c0", "#74c476", "#00441b"]}
    PCOL, NPCOL = "#111111", "#9a9a9a"; CLN = ["Low", "Moderate", "High"]
    figj = {}
    def fmt(k, pct=False):
        v, c = PT.get(k), CI.get(k)
        if v is None: return "n/a"
        f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.2f}")
        return f"{f(v)} [{f(c[0])}, {f(c[1])}]" if c and c[0] is not None else f(v)
    def dfmt(m, k, pct=False):
        d = DF.get(f"{m}-CN|{k}")
        if not d or m == "CN": return ""
        f = (lambda x: f"{100 * x:+.0f}") if pct else (lambda x: f"{x:+.2f}")
        return f"vs CN {f(d['delta'])} [{f(d['ci'][0])}, {f(d['ci'][1])}]"
    def save(fig, key, m):
        fn = f"{FIG}/kcf_{key}_{m.lower()}.png"; fig.savefig(fn, dpi=300); plt.close(fig); return fn
    def title(ax, m, extra=""):
        ax.set_title(NAME[m] + (("\n" + extra) if extra else ""), color=COL[m], fontsize=10.5)
    rows_pre = np.where(PRE)[0]; allr = np.arange(n)
    # ---------------- Fig 1a: cohort design (ours) --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.4)); yy0 = 0; info = {}
    for lab, st, col in (("Non-progressors", 0, NPCOL), ("Progressors", 1, PCOL)):
        ps = [i for i in range(NPAT) if P_y[i] == st]; fu = []
        ps = sorted(ps, key=lambda i: E_mbf[E_of[i]].max() - E_mbf[E_of[i]].min())
        for i in ps:
            es = E_order[i]; t = (E_mbf[es].max() - E_mbf[es]) / 12.0; fu.append(t.max())
            ax.plot([0, t.max()], [yy0, yy0], color=col, lw=0.8); ax.scatter(t, np.full(len(t), yy0), s=6, color=col, zorder=3)
            if st == 1: te = [e for e in es if E_BIN[e] == "Endpoint"]; ax.scatter((E_mbf[es].max() - E_mbf[te]) / 12.0, np.full(len(te), yy0), marker="|", s=40, color="#c00000", zorder=4)
            yy0 += 1
        info[lab] = {"patients": len(ps), "follow_up_years_mean": r3(np.mean(fu)), "sd": r3(np.std(fu, ddof=1)), "min": r3(min(fu)), "max": r3(max(fu)),
                     "endoscopies": int(sum(len(E_of[i]) for i in ps)), "samples": int(sum(len(S_of[i]) for i in ps))}
        ax.text(16.3, yy0 - len(ps) / 2, f"{lab}\n{len(ps)} patients\n{info[lab]['endoscopies']} endoscopies\n{info[lab]['samples']} samples\nfollow-up {info[lab]['min']:.0f}-{info[lab]['max']:.0f} y\n(mean {info[lab]['follow_up_years_mean']:.1f}, s.d. {info[lab]['sd']:.1f})",
                va="center", fontsize=9, color=col)
        yy0 += 3
    ax.set_xlim(-0.5, 22); ax.set_yticks([]); ax.set_xlabel("Years since first endoscopy in the cohort"); ax.set_title("Our cohort: 80 patients, 676 samples with H&E and copy number\n(red tick = HGD/IMC endpoint endoscopy)")
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    fig.tight_layout(); save(fig, "fig1a", "ours"); figj["fig1a"] = info
    # ---------------- Fig 1b: per-sample CN profiles of one P and one NP patient, with the three model classes -------------
    def most(st):
        c_ = [i for i in range(NPAT) if P_y[i] == st]; return max(c_, key=lambda i: (len(E_of[i]), len(S_of[i])))
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 9.0)); info = {}
    for ax, (lab, st) in zip(axs, (("Non-progressor", 0), ("Progressor", 1))):
        i = most(st); es = E_order[i]; srows = [r for e in es for r in E_rows[e][np.argsort(np.nan_to_num(LOC[E_rows[e]]))]][:18]
        t0 = E_mbf[es].max(); xpos = np.arange(len(WIN))
        for k, r in enumerate(srows):
            v = np.clip(Zw[r], -6, 6) / 6 * 0.42; base = len(srows) - 1 - k
            ax.bar(xpos, v, bottom=base, width=1.0, color=PCOL if st else NPCOL, linewidth=0)
            ax.axhline(base, color="#dddddd", lw=0.4, zorder=0)
            for q, m in enumerate(MODS): ax.add_patch(Rectangle((len(WIN) + 6 + q * 14, base - 0.4), 12, 0.8, color=SHADE[m][MC[m][r]], clip_on=False))
            ax.text(-8, base, f"y{(t0 - MBF[r]) / 12:.0f} {PATH[r]}", ha="right", va="center", fontsize=7.5)
        bnd = np.r_[0, np.where(np.diff(wchr))[0] + 1]
        for b in bnd: ax.axvline(b - 0.5, color="#bbbbbb", lw=0.4)
        ax.set_xticks([(bnd[j] + (bnd[j + 1] if j + 1 < len(bnd) else len(WIN))) / 2 for j in range(len(bnd))]); ax.set_xticklabels([str(c) if c < 19 or c % 2 == 0 else "" for c in np.unique(wchr)], fontsize=7.5)
        ax.set_yticks([]); ax.set_xlim(-0.5, len(WIN) + 50); ax.set_ylim(-0.7, len(srows) - 0.3)
        for q, m in enumerate(MODS): ax.text(len(WIN) + 12 + q * 14, len(srows) - 0.2, ["CN", "H&E", "Comb"][q], rotation=90, fontsize=7.5, ha="center", va="bottom", color=COL[m])
        ax.set_title(f"{lab} ({len(es)} endoscopies, {len(S_of[i])} samples; first {len(srows)} shown)", fontsize=10)
        for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
        info[lab] = {"endoscopies": len(es), "samples": len(S_of[i]), "shown": len(srows)}
    axs[1].set_xlabel("Chromosome (5-Mb windows, arm-adjusted standardised copy number)")
    fig.text(0.5, 0.005, "Squares at right: risk class per sample (light = low, mid = moderate, dark = high). Labels: years since first endoscopy, grade.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); save(fig, "fig1b", "ours"); figj["fig1b"] = info
    # ---------------- Fig 1c/1d and ED7b: genome-wide frequency of gains / losses ------------------------------------------
    def freq_panel(key, mask, m, chrom=None, ylim=15):
        fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
        sel = np.ones(len(WIN), bool) if chrom is None else wchr == chrom; xpos = np.arange(sel.sum())
        if m == "CN": groups = (("Non-progressor samples", mask & (y == 0), NPCOL), ("Progressor samples", mask & (y == 1), PCOL))
        else: groups = ((f"{SHORT[m]} low-risk samples", mask & (MC[m] == 0), SHADE[m][1]), (f"{SHORT[m]} high-risk samples", mask & (MC[m] == 2), SHADE[m][2]))
        out = {}
        for ax, (lab, mk, col) in zip(axs, groups):
            Z = Zw[mk][:, sel]; g = 100 * (Z > 2).mean(0); l = 100 * (Z < -2).mean(0)
            ax.bar(xpos, g, width=1.0, color=col, linewidth=0); ax.bar(xpos, -l, width=1.0, color=col, alpha=0.55, linewidth=0); ax.axhline(0, color="k", lw=0.5)
            alt = 100 * (np.abs(Z) > 2).mean(); out[lab] = {"n": int(mk.sum()), "pct_windows_altered": r3(alt)}
            if chrom is not None:
                ap = [j for j, c in enumerate(ARM) if achr[j] == chrom]; A_ = Za[mk][:, ap]; p_loss = 100 * (A_[:, 0] < -2).mean(); q_gain = 100 * (A_[:, 1] > 2).mean()
                out[lab].update({"arm_p_loss_pct": r3(p_loss), "arm_q_gain_pct": r3(q_gain)}); ax.text(0.99, 0.92, f"arm level: {chrom}p loss {p_loss:.1f}%, {chrom}q gain {q_gain:.1f}%", transform=ax.transAxes, fontsize=8.5, va="top", ha="right")
            ax.set_ylim(-ylim, ylim); ax.set_ylabel("% samples\n(gain up, loss down)", fontsize=8.5); ax.text(0.01, 0.92, f"{lab} (n = {mk.sum()}): {alt:.1f}% of windows altered", transform=ax.transAxes, fontsize=8.5, va="top")
            if chrom is None:
                for b in np.where(np.diff(wchr))[0]: ax.axvline(b + 0.5, color="#cccccc", lw=0.4)
        if chrom is None:
            bnd = np.r_[0, np.where(np.diff(wchr))[0] + 1]; axs[1].set_xticks([(bnd[j] + (bnd[j + 1] if j + 1 < len(bnd) else len(WIN))) / 2 for j in range(len(bnd))])
            axs[1].set_xticklabels([str(c) if c < 19 or c % 2 == 0 else "" for c in np.unique(wchr)]); axs[1].set_xlabel("Chromosome (5-Mb windows; altered = |standardised CN| > 2)")
        else:
            st = [int(c.split(":")[1].split("-")[0]) / 1e6 for c in np.array(WIN)[sel]]; axs[1].set_xticks(xpos[::3]); axs[1].set_xticklabels([f"{v:.0f}" for v in st[::3]]); axs[1].set_xlabel(f"Chromosome {chrom} position (Mb; window start)")
        axs[0].set_title(NAME[m] if m != "CN" else "Split by outcome (copy-number data)", color=COL[m], fontsize=10.5)
        fig.tight_layout(); save(fig, key, m); return out
    figj["fig1c"] = {m: freq_panel("fig1c", np.ones(n, bool), m) for m in MODS}
    figj["fig1d"] = {m: freq_panel("fig1d", PATH == "NDBE", m) for m in MODS}
    figj["ed7b"] = {"CN": freq_panel("ed7b", np.ones(n, bool), "CN", chrom=17, ylim=15)}
    # ---------------- Fig 2a: risk histogram + calibration inset ------------------------------------------------------------
    LGR = {m: lg(PM[m]) for m in MODS}; lim = (-12, 12); bins = np.arange(-12, 12.01, 2)
    for m in MODS:
        fig, ax = plt.subplots(figsize=(6.4, 4.8)); bot = np.zeros(len(bins) - 1)
        for k in range(3):
            h, _ = np.histogram(np.clip(LGR[m][MC[m] == k], -11.99, 11.99), bins); ax.bar(bins[:-1] + 1, h, width=1.9, bottom=bot, color=SHADE[m][k], label=CLN[k]); bot += h
        ax.set_xlim(*lim); ax.set_ylim(0, 420); ax.set_xlabel("log odds of progression (held-out, repeat mean)"); ax.set_ylabel("Number of samples"); ax.legend(loc="upper left", title="Risk class", frameon=False)
        ins = ax.inset_axes([0.62, 0.5, 0.35, 0.45]); edges = np.linspace(0, 1, 11); cal = []
        for a, b in zip(edges[:-1], edges[1:]):
            mk = (PM[m] > a) & (PM[m] <= b) if a > 0 else (PM[m] >= a) & (PM[m] <= b)
            if mk.sum() < 5: cal.append(None); continue
            k_ = y[mk].sum(); nn = mk.sum(); ph = k_ / nn; z = 1.96; den = 1 + z * z / nn; cen = (ph + z * z / (2 * nn)) / den; hw = z * np.sqrt(ph * (1 - ph) / nn + z * z / (4 * nn * nn)) / den
            ins.errorbar(PM[m][mk].mean(), ph, yerr=[[max(0, ph - (cen - hw))], [max(0, (cen + hw) - ph)]], fmt="o", ms=3, color=COL[m], lw=0.8); cal.append([r3(PM[m][mk].mean()), r3(ph), int(nn)])
        ins.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.7); ins.set_xlim(0, 1); ins.set_ylim(0, 1); ins.set_xlabel("Predicted", fontsize=8); ins.set_ylabel("Observed P share", fontsize=8); ins.tick_params(labelsize=7.5)
        cnt = [int((MC[m] == k).sum()) for k in range(3)]; title(ax, m, f"676 samples; low / moderate / high = {cnt[0]} / {cnt[1]} / {cnt[2]}")
        fig.tight_layout(); save(fig, "fig2a", m); figj.setdefault("fig2a", {})[m] = {"class_counts": cnt, "calibration_deciles": cal}
    # ---------------- Fig 2b: risk classes by pathology and outcome --------------------------------------------------------
    groups = [("NP", "NDBE"), ("NP", "ID"), ("NP", "LGD"), ("P", "NDBE"), ("P", "ID"), ("P", "LGD"), ("P", "HGD"), ("P", "IMC")]
    for m in MODS:
        fig, ax = plt.subplots(figsize=(6.4, 4.8)); out = {}
        for gi, (st, pth) in enumerate(groups):
            mk = (y == (st == "P")) & (PATH == pth); sh = [(CLS[m][mk] == k).mean() for k in range(3)]; bot = 0
            for k in range(3):
                ax.bar(gi + (gi >= 3) * 0.6, 100 * sh[k], bottom=bot, color=SHADE[m][k], width=0.85, label=CLN[k] if gi == 0 else None)
                if sh[k] >= 0.06: ax.text(gi + (gi >= 3) * 0.6, bot + 50 * sh[k], f"{100 * sh[k]:.0f}", ha="center", va="center", fontsize=8, color="white" if k == 2 else "black")
                bot += 100 * sh[k]
            ax.text(gi + (gi >= 3) * 0.6, 102, f"n={mk.sum()}", ha="center", fontsize=7.5); out[f"{st}|{pth}"] = {"n": int(mk.sum()), "shares": [r3(v) for v in sh]}
        ax.set_xticks([g + (g >= 3) * 0.6 for g in range(len(groups))]); ax.set_xticklabels([p for _, p in groups]); ax.set_ylim(0, 112); ax.set_ylabel("% of samples (mean over 10 repeats)")
        ax.text(1, -16, "Non-progressors", ha="center", color=NPCOL, fontsize=9.5); ax.text(5.6, -16, "Progressors", ha="center", color=PCOL, fontsize=9.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
        title(ax, m, f"NDBE: P high {fmt(f'{m}|fig2b|ndbe_P_high', True)}; NP low {fmt(f'{m}|fig2b|ndbe_NP_low', True)}")
        fig.tight_layout(); save(fig, "fig2b", m); figj.setdefault("fig2b", {})[m] = out
    # ---------------- Fig 2d / ED5: per-patient risk heatmaps -----------------------------------------------------------
    def heat_layout():
        boxes = []
        for st in (0, 1):
            for i in sorted([i for i in range(NPAT) if P_y[i] == st], key=lambda i: -len(E_of[i])):
                es = E_order[i]; cells = []
                for xj, e in enumerate(es):
                    rr = E_rows[e]; locs = LOC[rr]; order = np.argsort(np.nan_to_num(locs, nan=99));
                    for yj, r in enumerate(rr[order]): cells.append((xj, yj, r))
                boxes.append((st, i, len(es), max(c[1] for c in cells) + 1, cells))
        return boxes
    BOX = heat_layout(); WMAX = 46
    def place():
        pos = []; x = 0; yb = 0; rowh = 0; cur = None
        for st, i, w, h, cells in BOX:
            if cur is not None and st != cur: yb += rowh + 3; x = 0; rowh = 0
            cur = st
            if x + w > WMAX: yb += rowh + 1; x = 0; rowh = 0
            pos.append((x, yb)); x += w + 1; rowh = max(rowh, h)
        return pos, yb + rowh
    POS, HT = place()
    for m in MODS:
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        for (st, i, w, h, cells), (x0, y0) in zip(BOX, POS):
            ax.add_patch(Rectangle((x0 - 0.1, y0 - 0.1), w + 0.2, h + 0.2, fill=False, ec=PCOL if st else NPCOL, lw=0.6))
            for xj, yj, r in cells: ax.add_patch(Rectangle((x0 + xj + 0.05, y0 + yj + 0.05), 0.9, 0.9, color=SHADE[m][MC[m][r]], lw=0))
        ax.set_xlim(-1, WMAX + 1); ax.set_ylim(HT + 1, -3); ax.set_aspect("equal"); ax.axis("off")
        ax.text(0, -1.5, "Non-progressors (grey boxes)", color=NPCOL, fontsize=9.5); yP = min(p[1] for (b, p) in zip(BOX, POS) if b[0] == 1); ax.text(0, yP - 1, "Progressors (black boxes)", color=PCOL, fontsize=9.5)
        allP = np.mean([np.any(MC[m][S_of[i]] == 2) for i in range(NPAT) if P_y[i] == 1]); allN = np.mean([np.all(MC[m][S_of[i]] == 0) for i in range(NPAT) if P_y[i] == 0])
        title(ax, m, f"P with >=1 high sample {100 * allP:.0f}%; NP with all samples low {100 * allN:.0f}%")
        fig.tight_layout(); save(fig, "fig2d", m); figj.setdefault("fig2d", {})[m] = {"P_any_high": r3(allP), "NP_all_low": r3(allN)}
    # ---------------- Fig 3a: per-endoscopy risk over time ---------------------------------------------------------------
    MK = {0: "o", 1: "s", 2: "D", 3: "^", 4: "v"}
    for m in MODS:
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 7.2), sharex=True); out = {}
        for ax, st in zip(axs, (0, 1)):
            ps = sorted([i for i in range(NPAT) if P_y[i] == st], key=lambda i: (np.mean([EMC[m][e] for e in E_order[i]]), len(E_of[i])))
            for yy_, i in enumerate(ps):
                es = E_order[i]; t = E_mbf[es].max() - E_mbf[es]; ax.plot(t, np.full(len(t), yy_), color=PCOL if st else NPCOL, lw=0.7, zorder=1)
                for e, tt in zip(es, t): ax.scatter(tt, yy_, marker=MK[int(E_maxg[e])], s=22, color=SHADE[m][EMC[m][e]], edgecolor="k", linewidth=0.3, zorder=3)
            ax.set_yticks([]); ax.set_title(["Non-progressors", "Progressors"][st], color=[NPCOL, PCOL][st], fontsize=10); ax.set_ylim(-1, len(ps))
            first_hi = np.mean([EMC[m][E_order[i][0]] == 2 for i in ps]); never_hi = np.mean([all(EMC[m][e] < 2 for e in E_order[i]) for i in ps])
            out[["NP", "P"][st]] = {"first_endoscopy_high": r3(first_hi), "never_high": r3(never_hi), "patients": len(ps)}
        axs[0].set_xlabel("Months since first endoscopy"); axs[1].set_xlabel("Months since first endoscopy"); axs[0].set_xlim(-5, 200)
        hs = [plt.Line2D([], [], marker=MK[g], ls="", color="w", markeredgecolor="k", label=lb) for g, lb in enumerate(["NDBE", "ID", "LGD", "HGD", "IMC"])] + \
             [plt.Line2D([], [], marker="o", ls="", color=SHADE[m][k], markeredgecolor="k", label=CLN[k]) for k in range(3)]
        fig.legend(handles=hs, loc="lower center", ncol=8, frameon=False, fontsize=8, handletextpad=0.2, columnspacing=0.8)
        fig.suptitle(NAME[m] + f"\nP high at first endoscopy {100 * out['P']['first_endoscopy_high']:.0f}%; NP never high {100 * out['NP']['never_high']:.0f}%", color=COL[m], fontsize=10.5)
        fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, "fig3a", m); figj.setdefault("fig3a", {})[m] = out
    # ---------------- Fig 3b: share of progressor endoscopies with >= 1 high-risk sample, by months before endpoint ------------
    for m in MODS:
        fig, ax = plt.subplots(figsize=(6.4, 4.8)); xs = np.arange(len(BIN_NAMES)); out = {}
        for xj, b in enumerate(BIN_NAMES):
            v = PT[f"{m}|fig3b|{b}"]; c = CI[f"{m}|fig3b|{b}"]; ne = int((E_BIN == b).sum()); ns = int(sum(len(E_rows[e]) for e in np.where(E_BIN == b)[0]))
            ax.bar(xj, 100 * v, color=COL[m], width=0.7); ax.errorbar(xj, 100 * v, yerr=[[100 * max(0, v - c[0])], [100 * max(0, c[1] - v)]], color="k", lw=0.8, capsize=2)
            ax.text(xj, 3, f"n={ne}\ns={ns}", ha="center", fontsize=7.5, color="white"); ax.text(xj, 100 * v - 6, f"{100 * v:.0f}%", ha="center", fontsize=8, color="white")
            ax.scatter(xj + 0.3, 100 * PUBV["fig3b"][b], marker="D", s=22, facecolor="white", edgecolor="k", zorder=4, label="Killcoyne 2020" if xj == 0 else None)
            if b != "Endpoint": ax.scatter(xj - 0.3, 100 * PT[f"GRADE|fig3b|{b}"], marker="s", s=22, color=COL["GREY"], zorder=4, label=">=1 LGD sample (grade)" if xj == 0 else None)
            out[b] = {"share": v, "ci": c, "endoscopies": ne, "samples": ns, "grade_LGD_share": PT.get(f"GRADE|fig3b|{b}")}
        ax.set_xticks(xs); ax.set_xticklabels(BIN_NAMES, rotation=30); ax.set_xlabel("Months before endpoint (HGD/IMC)"); ax.set_ylabel("% progressor endoscopies with\n>=1 high-risk sample"); ax.set_ylim(0, 105)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=8); title(ax, m, f">96 months {fmt(f'{m}|fig3b|>96', True)}; 12-24 months {fmt(f'{m}|fig3b|12-24', True)}")
        fig.tight_layout(); save(fig, "fig3b", m); figj.setdefault("fig3b", {})[m] = out
    # ---------------- Fig 4b: recommendation after the second endoscopy -----------------------------------------------------
    LAB4 = ["RFA", "6-12-month\nrecheck", "12-24-month\nrecheck", "3-5-year\nsurveillance"]
    for m in MODS:
        fig, ax = plt.subplots(figsize=(6.4, 4.8)); R = REC_OF[m]; out = {}
        for st, col, off in ((1, PCOL, 0.2), (0, NPCOL, -0.2)):
            cnt = [sum(1 for i, v in R.items() if P_y[i] == st and v == k) for k in range(4)]; out[["NP", "P"][st]] = cnt
            ax.barh(np.arange(4) + off, cnt, height=0.38, color=col, label=["Non-progressors", "Progressors"][st])
            pub = PUBV["fig4b_P"] if st else PUBV["fig4b_NP"]; gr = [sum(1 for i, v in REC_OF["GRADE"].items() if P_y[i] == st and v == k) for k in range(4)]; out[["NP", "P"][st] + "_grade"] = gr
            for k in range(4):
                ax.text(cnt[k] + 0.3, k + off, f"{cnt[k]}  (Killcoyne {pub[k]}; grade/p53 {gr[k]})", va="center", fontsize=7.5)
        ax.set_yticks(range(4)); ax.set_yticklabels(LAB4); ax.set_xlim(0, 40); ax.set_xlabel("Number of patients"); ax.legend(loc="lower right", frameon=False)
        nP = sum(1 for i in R if P_y[i] == 1); nN = sum(1 for i in R if P_y[i] == 0)
        title(ax, m, f"{nP} P / {nN} NP with 2 pre-event endoscopies; P RFA {out['P'][0]}, NP 3-5 y {out['NP'][3]}")
        fig.tight_layout(); save(fig, "fig4b", m); figj.setdefault("fig4b", {})[m] = out
    # ---------------- Fig 4c-e: three example patients ----------------------------------------------------------------------
    pre_cnt = lambda i: sum(1 for e in E_order[i] if E_pre[e])
    exP = sorted([i for i in range(NPAT) if P_y[i] == 1], key=lambda i: -pre_cnt(i))[:2]; exN = sorted([i for i in range(NPAT) if P_y[i] == 0], key=lambda i: -pre_cnt(i))[:1]
    EX = [(exP[0], "Progressor A"), (exP[1], "Progressor B"), (exN[0], "Non-progressor C")]
    for m in MODS:
        fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.2))
        for ax, (i, lab) in zip(axs, EX):
            es = E_order[i]; t0 = E_mbf[es].max()
            for e in es:
                rr = E_rows[e]; order = np.argsort(np.nan_to_num(LOC[rr], nan=99))
                for yj, r in enumerate(rr[order]):
                    tx = (t0 - MBF[r]) / 12; ax.add_patch(Rectangle((tx - 0.35, yj), 0.7, 0.85, color=SHADE[m][MC[m][r]]))
                    ax.scatter(tx, yj + 0.42, marker=MK[int(G[r])], s=12, facecolor="none", edgecolor="white" if MC[m][r] == 2 else "k", linewidth=0.6)
            rc = REC_OF[m].get(i); rg = REC_OF["GRADE"].get(i)
            SR = ["RFA", "6-12-month recheck", "12-24-month recheck", "3-5-year surveillance"]
            ax.set_title(f"{lab}. After 2nd endoscopy: model {SR[rc] if rc is not None else 'n/a'}; grade/p53 {SR[rg] if rg is not None else 'n/a'}", fontsize=8.5, loc="left")
            ax.set_xlim(-0.8, (t0 - E_mbf[es].min()) / 12 + 0.8); ax.set_ylim(-0.3, max(len(E_rows[e]) for e in es) + 0.3); ax.set_ylabel("Sample (EGJ up)", fontsize=8.5)
        axs[-1].set_xlabel("Years since first endoscopy"); fig.suptitle(NAME[m], color=COL[m], fontsize=10.5); fig.tight_layout(); save(fig, "fig4c", m)
    # ---------------- ED1a: score distributions P vs NP (sample level, pre-event) ------------------------------------------
    rp = rows_pre; ED1 = {"CN": ("Genomic complexity (cx)", CX), "HE": ("H&E log odds", lg(PM["HE"])), "COMB": ("Combined log odds", lg(PM["COMB"]))}
    for m in MODS:
        lab, v = ED1[m]; fig, ax = plt.subplots(figsize=(6.4, 4.8)); out = {}
        for k, (st, col) in enumerate(((0, NPCOL), (1, PCOL))):
            vv = v[rp][y[rp] == st]; ax.boxplot(vv, positions=[k], widths=0.5, showfliers=False, medianprops={"color": COL[m]}); ax.scatter(k + np.random.RandomState(k).uniform(-0.18, 0.18, len(vv)), vv, s=4, color=col, alpha=0.6)
        p = mannwhitneyu(v[rp][y[rp] == 1], v[rp][y[rp] == 0]).pvalue; a = pooled_auc(v[rp], y[rp])
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"Non-progressor\n(n={(y[rp] == 0).sum()})", f"Progressor\n(n={(y[rp] == 1).sum()})"]); ax.set_ylabel(lab)
        title(ax, m, f"Wilcoxon p = {p:.1e}; pooled AUROC {a:.2f}" + (" (Killcoyne cx: p 2.4e-06, AUC 0.6)" if m == "CN" else ""))
        fig.tight_layout(); save(fig, "ed1a", m); figj.setdefault("ed1a", {})[m] = {"wilcoxon_p": float(p), "pooled_auc": r3(a), "measure": lab}
    # ---------------- ED1b: ROC curves (pre-event samples; pooled curve of repeat-mean predictions) -----------------------
    def roc(s, yy):
        o_ = np.argsort(-s); yy = yy[o_]; tp = np.r_[0, np.cumsum(yy)] / yy.sum(); fp = np.r_[0, np.cumsum(1 - yy)] / (1 - yy).sum(); return fp, tp
    for m in MODS:
        fig, ax = plt.subplots(figsize=(4.8, 4.8)); fp, tp = roc(PM[m][rp], y[rp]); ax.plot(1 - fp, tp, color=COL[m], lw=1.8, label=f"{SHORT[m]}: {fmt(f'{m}|auc|sample_pre')}")
        if m != "CN": f2, t2 = roc(PM["CN"][rp], y[rp]); ax.plot(1 - f2, t2, color=COL["CN"], lw=0.8, alpha=0.5, label="copy number (reference)")
        else:
            f2, t2 = roc(CX[rp], y[rp]); ax.plot(1 - f2, t2, color=COL["CN"], ls=":", lw=1, label=f"cx alone: {PT['cx|auc|sample_pre']:.2f}")
            f3, t3 = roc(KP[rp], y[rp]); ax.plot(1 - f3, t3, color="k", ls="--", lw=0.8, label=f"published P on our samples: {fmt('PUB|auc|sample_pre')}")
        ax.plot([1, 0], [0, 1], color="#cccccc", lw=0.6); ax.set_xlim(1.02, -0.02); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("Specificity"); ax.set_ylabel("Sensitivity")
        ax.legend(loc="lower right", frameon=False, fontsize=7.5); title(ax, m, ("571 pre-event samples; " + dfmt(m, "auc|sample_pre")) if m != "CN" else "571 pre-event samples (Killcoyne LOPO 0.87 on 773)")
        fig.tight_layout(); save(fig, "ed1b", m)
    # ---------------- ED1c: initial endoscopy, P vs NP ----------------------------------------------------------------
    fe = np.array([E_order[i][0] for i in range(NPAT)]); fe = fe[E_pre[fe]]; fy = E_y[fe]
    for m in MODS:
        if m == "CN":
            fig, axs = plt.subplots(1, 3, figsize=(6.4, 4.8)); vals = [("Windows\naltered", np.array([ALT_W[E_rows[e]].mean() for e in fe])), ("Arms\naltered", np.array([ALT_A[E_rows[e]].mean() for e in fe])), ("CN probability\n(endoscopy mean)", EPmean["CN"][fe].mean(1))]
        else:
            fig, axs = plt.subplots(1, 1, figsize=(6.4, 4.8)); axs = [axs]; vals = [(f"{SHORT[m]} probability\n(endoscopy mean)", EPmean[m][fe].mean(1))]
        out = {}
        for ax, (lab, v) in zip(axs, vals):
            for k, (st, col) in enumerate(((0, NPCOL), (1, PCOL))):
                vv = v[fy == st]; ax.boxplot(vv, positions=[k], widths=0.5, showfliers=False, medianprops={"color": COL[m]}); ax.scatter(k + np.random.RandomState(k).uniform(-0.15, 0.15, len(vv)), vv, s=8, color=col)
            p = mannwhitneyu(v[fy == 1], v[fy == 0]).pvalue; a = pooled_auc(v, fy); out[lab] = {"wilcoxon_p": float(p), "auc": r3(a)}
            ax.set_xticks([0, 1]); ax.set_xticklabels([f"NP\n({(fy == 0).sum()})", f"P\n({(fy == 1).sum()})"]); ax.set_title(f"{lab}\np = {p:.1e}\nAUROC {a:.2f}", fontsize=8.5)
            if "probability" in lab: ax.set_ylim(-0.03, 1.03)
        fig.suptitle(NAME[m] + ": first endoscopy" + (" (Killcoyne p 6e-08 / 5.5e-11)" if m == "CN" else ""), color=COL[m], fontsize=10.5); fig.tight_layout(); save(fig, "ed1c", m); figj.setdefault("ed1c", {})[m] = out
    # ---------------- ED1d: first vs last endoscopy, per patient -------------------------------------------------------
    for m in MODS:
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 4.8), sharey=True); ch = {0: [], 1: []}; cha = {}
        for ax, st in zip(axs, (0, 1)):
            for i in [i for i in range(NPAT) if P_y[i] == st]:
                es = E_order[i]
                if len(es) < 2: continue
                e0, e1 = es[0], es[-1]; v0, v1 = EPmean[m][e0].mean(), EPmean[m][e1].mean(); cha.setdefault(st, []).append(ALT_A[E_rows[e1]].mean() - ALT_A[E_rows[e0]].mean())
                t1 = (E_mbf[e0] - E_mbf[e1]) / 12; ax.plot([0, t1], [v0, v1], color=PCOL if st else NPCOL, lw=0.7, marker="o", ms=2.5); ch[st].append(v1 - v0)
            ax.set_title(f"{['Non-progressors', 'Progressors'][st]} (n={len(ch[st])})", fontsize=9.5, color=[NPCOL, PCOL][st]); ax.set_xlabel("Years since first endoscopy")
        axs[0].set_ylabel(f"{SHORT[m]} probability (endoscopy mean)"); axs[0].set_ylim(-0.03, 1.03)
        p = mannwhitneyu(ch[1], ch[0]).pvalue; pa = mannwhitneyu(cha[1], cha[0]).pvalue
        fig.suptitle(NAME[m] + f": first vs last endoscopy\nchange P vs NP p = {p:.1e}" + (f"; altered arms p = {pa:.1e} (Killcoyne arms 7e-04)" if m == "CN" else ""), color=COL[m], fontsize=10)
        fig.tight_layout(); save(fig, "ed1d", m); figj.setdefault("ed1d", {})[m] = {"wilcoxon_p_change": float(p), "median_change_P": r3(np.median(ch[1])), "median_change_NP": r3(np.median(ch[0])), "arms_wilcoxon_p_change": float(pa), "n_P": len(ch[1]), "n_NP": len(ch[0])}
    # ---------------- ED2: AUROC / sensitivity / specificity at the low cut, by unit -------------------------------------
    UN = [("sample_all", "All samples\n(676)"), ("sample_pre", "Pre-event\nsamples (571)"), ("endo_mean", "Endoscopy\nmean"), ("endo_max", "Endoscopy\nmax"), ("pat_mean", "Patient\nmean"), ("pat_max", "Patient\nmax")]
    PUBMAP = {"sample_all": "per sample", "sample_pre": "excluding HGD/IMC", "endo_mean": "endoscopy mean", "endo_max": "endoscopy max", "pat_mean": "patient mean", "pat_max": "patient max"}
    for m in MODS:
        fig, ax = plt.subplots(figsize=(7.2, 4.8)); xs = np.arange(len(UN))
        for q, (met, a_) in enumerate((("auc", 1.0), ("sens_lo", 0.65), ("spec_lo", 0.35))):
            for xj, (u, _) in enumerate(UN):
                k = f"{m}|{met}|{u}"; v = PT[k]; c = CI[k]
                ax.bar(xj + (q - 1) * 0.27, v, width=0.26, color=COL[m], alpha=a_, label=["AUROC", "Sensitivity", "Specificity"][q] if xj == 0 else None)
                ax.errorbar(xj + (q - 1) * 0.27, v, yerr=[[max(0, v - c[0])], [max(0, c[1] - v)]], color="k", lw=0.7, capsize=1.5); ax.text(xj + (q - 1) * 0.27, 0.03, f"{v:.2f}", rotation=90, ha="center", va="bottom", fontsize=7, color="white" if q < 2 else "k")
                if m == "CN": ax.scatter(xj + (q - 1) * 0.27, PUBV["ed2"][PUBMAP[u]][q], marker="D", s=14, facecolor="white", edgecolor="k", zorder=4, label="Killcoyne 2020" if (xj == 0 and q == 0) else None)
        ax.set_xticks(xs); ax.set_xticklabels([u[1] for u in UN], fontsize=8.5); ax.set_ylim(0, 1.05); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
        title(ax, m, "sensitivity / specificity at the low-risk cut (training-matched)"); fig.tight_layout(); save(fig, "ed2", m)
    # ---------------- ED6: learning curve ----------------------------------------------------------------------------
    if LC:
        Ks = sorted(int(k) for k in LC if k.isdigit() and "cnv" in LC[k]); ft_of = {"CN": "cnv", "HE": "clean", "COMB": "comb"}; out = {}
        for m in MODS:
            fig, ax = plt.subplots(figsize=(6.4, 4.8)); ft = ft_of[m]
            xs = [LC[str(k)]["n_train_patients_mean"] for k in Ks] + [72]; vs = [LC[str(k)][ft]["auc_mean"] for k in Ks] + [LC["full_stored"][ft]["auc_mean"]]; es = [LC[str(k)][ft]["auc_sem"] for k in Ks] + [LC["full_stored"][ft]["auc_sem"]]
            ax.bar(range(len(xs)), vs, color=COL[m], width=0.7); ax.errorbar(range(len(xs)), vs, yerr=es, color="k", lw=0.8, capsize=2)
            for j, v in enumerate(vs): ax.text(j, 0.04, f"{v:.2f}", ha="center", color="white", fontsize=8.5)
            ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:.0f}" for x in xs[:-1]] + ["all\n(~72)"]); ax.set_xlabel("Training patients per outer fold"); ax.set_ylabel("Held-out AUROC (pre-event samples)"); ax.set_ylim(0, 1)
            title(ax, m, f"{xs[0]:.0f} -> all training patients: {vs[0]:.2f} -> {vs[-1]:.2f}" + ("  (Killcoyne 28 -> 88: 0.70 -> 0.87)" if m == "CN" else "")); fig.tight_layout(); save(fig, "ed6", m); out[m] = dict(zip([str(x) for x in xs], vs))
        figj["ed6"] = out
    # ---------------- ED7a: progressor samples by grade: p53 aberrant vs model high ----------------------------------------
    PTH = ["NDBE", "ID", "LGD", "HGD", "IMC"]
    for m in MODS:
        fig, ax = plt.subplots(figsize=(6.4, 4.8)); out = {}
        for xj, pth in enumerate(PTH):
            mk = (y == 1) & (PATH == pth); kn = mk & np.isfinite(P53); ab = (P53[kn] == 1).mean() if kn.any() else np.nan; hi = (CLS[m][mk] == 2).mean()
            ax.bar(xj - 0.2, 100 * ab, width=0.38, color=COL["GREY"], label="aberrant p53 IHC" if xj == 0 else None); ax.bar(xj + 0.2, 100 * hi, width=0.38, color=COL[m], label=f"{SHORT[m]} high risk" if xj == 0 else None)
            ax.text(xj - 0.2, 100 * ab + 1.5, f"{100 * ab:.0f}", ha="center", fontsize=7.5); ax.text(xj + 0.2, 100 * hi + 1.5, f"{100 * hi:.0f}", ha="center", fontsize=7.5)
            ax.scatter(xj - 0.2, 100 * PUBV["ed7a_p53"][pth], marker="D", s=18, facecolor="white", edgecolor="k", zorder=4, label="Killcoyne p53 2020" if xj == 0 else None)
            ax.text(xj, -9, f"n={mk.sum()} ({kn.sum()} IHC)", ha="center", fontsize=7); out[pth] = {"n": int(mk.sum()), "n_ihc": int(kn.sum()), "p53_aberrant": r3(ab), "model_high": r3(hi)}
        ax.set_xticks(range(5)); ax.set_xticklabels(PTH); ax.set_ylim(-12, 110); ax.set_ylabel("% of progressor samples"); ax.legend(loc="upper left", frameon=False, fontsize=8)
        title(ax, m, f"NDBE progressor samples: p53 aberrant {100 * out['NDBE']['p53_aberrant']:.0f}%, high risk {100 * out['NDBE']['model_high']:.0f}%"); fig.tight_layout(); save(fig, "ed7a", m); figj.setdefault("ed7a", {})[m] = out
    # ---------------- ED10a: class shares by outcome ----------------------------------------------------------------
    for m in MODS:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        for xj, st in enumerate(("NP", "P")):
            bot = 0
            for k, cn in enumerate(("low", "mod", "high")):
                v = PT[f"{m}|ed10|{st}_{cn}"]; ax.bar(xj, v, bottom=bot, color=SHADE[m][k], width=0.6, label=CLN[k] if xj == 0 else None)
                ax.text(xj, bot + v / 2, f"{v:.2f}\n(K {PUBV['ed10_' + st][k]:.2f})", ha="center", va="center", fontsize=8, color="white" if k == 2 else "black"); bot += v
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-progressor", "Progressor"]); ax.set_ylabel("Sample share (676 samples)"); ax.set_ylim(0, 1); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
        title(ax, m, "K = Killcoyne 2020, 50 kb discovery"); fig.tight_layout(); save(fig, "ed10a", m)
    json.dump(figj, open(f"{OUTR}/figs.json", "w"), indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)); print("KCF DONE figs", flush=True)
