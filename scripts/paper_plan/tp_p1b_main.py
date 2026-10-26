"""Two papers, agent P1B (docs/two_papers_P1B.md): Paper 1 confound and robustness analyses (paper/two_papers/PAPER1_plan_v2.md items 4, 5, 6, 7, 9).
Inputs, read-only: stored outer and inner predictions (tb_common.py loaders), the sample sheet, slide_desc, package QC, the E3 refits
(killcoyne_mm/explore_E3/refit; selection), the round-7 refits (killcoyne_mm/review_round7/refit; labels and added samples).
Helpers copied unchanged where stated: fsa / dstat / target encoding / batch probe (rv4_round4.py), read_refit / wauc / inner_auc / sel: choice (ex_e3_score.py),
depth (dd_describe.py), pre-event rule (hz_prep.py), same-block labels (rv7d_sensfirst.py, executed in its own namespace), R53 event dates (rv7_round7.py db_link).
Aggregates (no identifiers, no dates, cells < 5 shown as "<5") -> results/paper_final/two_papers_P1B/<TASK>.json; row-level files only under
killcoyne_mm/two_papers_P1B/ on the cluster.
TASK = smoke | batch | depthprep | depth | select_<set> | grade | sens | exclsel        P1B_NB=k: smoke mode (k bootstrap draws; JSON written aside)."""
import os, sys, json, glob, re, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
PP = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"; sys.path.insert(0, PP)
PTASK = os.environ["TASK"]
import tb_common as TC
from tb_common import *
OUT = T + "/results/paper_final/two_papers_P1B"; ROW = M + "/two_papers_P1B"; KD = T + "/feasibility/paper_plan/killcoyne"
SWG = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; ST = "/mnt/scratche/slow/fmlab/zuberi01/envs/killcoyne_r/bin/samtools"
NBX = NB
if os.environ.get("P1B_NB"): NBX = int(os.environ["P1B_NB"]); OUT = ROW + "/smoke_json"
for d in (OUT, ROW): os.makedirs(d, exist_ok=True)
MODELS = ("L-CNV", "L-IMG", "L-LATE"); DIFFS = (("IMG_minus_CNV", "L-IMG", "L-CNV"), ("LATE_minus_IMG", "L-LATE", "L-IMG"), ("LATE_minus_CNV", "L-LATE", "L-CNV"))
O0, I0, FO0 = load("their"); ROWS = np.where(PRE)[0]
def dump(res, name=None):
    res["agent"] = "P1B"; json.dump(res, open(f"{OUT}/{name or PTASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("P1B DONE", name or PTASK, flush=True)
def ci2(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return pct(a) if len(a) else [None, None]
def dstat(obs, draws): return {"delta": r3(obs), "ci95": ci2(draws), "p_two_sided": p2(draws)}
def cnt(v): v = int(v); return v if (v == 0 or v >= 5) else "<5"
def mtab(a, b):
    t = pd.crosstab(pd.Series(a, name="a"), pd.Series(b, name="b")); return {str(i): {str(c): cnt(t.loc[i, c]) for c in t.columns} for i in t.index}
# ------------------------------------------------------------------ evaluation environment (a sample set, its labels, patients and folds)
class Env:
    def __init__(s, ids_, y_, pat_, FO_, pre_): s.ids = np.asarray(ids_); s.y = np.asarray(y_).astype(int); s.pat = np.asarray(pat_).astype(str); s.FO = FO_; s.pre = np.asarray(pre_).astype(bool); s.n = len(s.ids)
EV0 = Env(ids, y, pat, FO0, PRE)
def fsa(S, b, ev=EV0, mask=None, w=None, cell=None):
    """fold-stratified AUROC (rv4_round4.fsa, with the set's folds and labels passed in): mean over repeats of within-fold pooled pair concordance;
    mask = evaluation subset; w = per-sample weights (pair weight w_i w_j); cell = pairs must also share the cell (cell < 0 excluded)"""
    out = []
    for j in range(10):
        x = (S[:, j] if np.ndim(S) == 2 else S)[b]; f = ev.FO[b, j].copy(); l = ev.y[b]; ok = np.ones(len(b), bool) if mask is None else mask[b].copy(); num = den = 0.0
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
def npairs(b, ev=EV0, mask=None, cell=None):
    """within-fold (and within-cell) case-control pairs, mean over repeats"""
    out = []
    for j in range(10):
        f = ev.FO[b, j].copy(); l = ev.y[b]; ok = np.ones(len(b), bool) if mask is None else mask[b].copy()
        if cell is not None: cb = cell[b]; ok &= cb >= 0; f = f + 100 * np.where(cb < 0, 0, cb)
        out.append(sum(int(((f == k) & ok & (l == 1)).sum()) * int(((f == k) & ok & (l == 0)).sum()) for k in np.unique(f[ok])))
    return float(np.mean(out))
def boots(rows, ev=EV0, seed=0, nb=None):
    """tb_common.boots_for with the set's patients and labels"""
    nb = nb or NBX; pp = ev.pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(seed); out = []
    while len(out) < nb:
        b = np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))])
        if 0 < ev.y[b].sum() < len(b): out.append(b)
    return out
def patw(rows, ev=EV0):
    """patient weight 1 / (number of the patient's evaluated samples)"""
    c = pd.Series(ev.pat[rows]).value_counts(); w = np.zeros(ev.n); w[rows] = 1.0 / pd.Series(ev.pat[rows]).map(c).values; return w
def hblock(O, rows, bb, ev=EV0, mask=None, w=None, cell=None, keep=False):
    """AUROC of each model in O and the paired differences, patient bootstrap"""
    ms = [m for m in MODELS if m in O] + [m for m in O if m not in MODELS]
    obs = {m: fsa(O[m], rows, ev, mask, w, cell) for m in ms}; B = pd.DataFrame([{m: fsa(O[m], b, ev, mask, w, cell) for m in ms} for b in bb])
    sub = rows if mask is None else rows[mask[rows]]
    R = {"n_samples": cnt(len(sub)), "n_progressor_samples": cnt(ev.y[sub].sum()), "n_patients": cnt(len(set(ev.pat[sub]))), "n_progressor_patients": cnt(len(set(ev.pat[sub][ev.y[sub] == 1]))),
         "pairs_per_repeat": round(npairs(rows, ev, mask, cell), 1), "auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in ms}}
    for nm, a, c in DIFFS:
        if a in obs and c in obs: R[nm] = dstat(obs[a] - obs[c], B[a] - B[c])
    return (R, obs, B) if keep else R
def did(o1, B1, o2, B2, a, c=None):
    """difference in differences (subgroup 1 minus subgroup 2) of a model AUROC or of a paired difference a - c"""
    f = (lambda o: o[a]) if c is None else (lambda o: o[a] - o[c]); return dstat(f(o1) - f(o2), f(B1) - f(B2))
def sheet():
    fd = pd.read_csv(SWG + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; return fd
def other_ns(fname):
    """execute another script's module body up to its task dispatch in its own namespace (as ex_e1_main.other_ns)"""
    old = os.environ.get("TASK"); os.environ["TASK"] = "none"
    ns = {"__file__": PP + "/" + fname, "__name__": "p1b_" + fname}; s_ = open(PP + "/" + fname).read(); exec(compile(s_[:s_.index('\nres = {"task": TASK}')], PP + "/" + fname, "exec"), ns)
    os.environ["TASK"] = old; return ns
F = sheet(); FI = F.drop_duplicates("combined_name").set_index("combined_name"); FS = FI.reindex(ids)
BATCH = FS.Batch.str.strip().values
SDS = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); SCAN = SDS.scanner.values; TILES = SDS.tissue_tiles.values.astype(float)
GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); P53 = pd.to_numeric(SM["P53 IHC"], errors="coerce").values
res = {"task": PTASK}
# ================================================================== smoke: the known numbers
if PTASK == "smoke":
    bb = boots(ROWS, nb=50); R = {"overall": hblock(O0, ROWS, bb), "within_batch": hblock(O0, ROWS, bb, cell=pd.factorize(BATCH)[0]),
                                   "ndbe": hblock(O0, ROWS, bb, mask=NDBE)}
    R["expected"] = {"overall": [0.801, 0.847, 0.889], "within_batch_round4": [0.790, 0.841, 0.893], "ndbe_killcoyne_final_c": [0.780, 0.827, 0.867]}
    res["smoke"] = R; dump(res)
# ================================================================== item 4: batch
elif PTASK == "batch":
    rows = ROWS; bb = boots(rows); W = patw(rows); bc = pd.factorize(BATCH)[0]; sc = pd.factorize(SCAN)[0]; bs = pd.factorize(pd.Series(BATCH) + "|" + pd.Series(SCAN))[0]; R = {}
    R["overall"] = {"sample": hblock(O0, rows, bb), "patient_weighted": hblock(O0, rows, bb, w=W)}
    for nm, cl in (("within_batch", bc), ("within_scanner", sc), ("within_batch_and_scanner", bs)):
        R[nm] = {"sample": hblock(O0, rows, bb, cell=cl), "patient_weighted": hblock(O0, rows, bb, w=W, cell=cl)}; print(nm, "done", flush=True)
    # sequencing batch vs scan campaign
    t = pd.crosstab(BATCH[rows], SCAN[rows]); chi = ((t.values - t.values.sum(1, keepdims=True) * t.values.sum(0, keepdims=True) / t.values.sum()) ** 2 / (t.values.sum(1, keepdims=True) * t.values.sum(0, keepdims=True) / t.values.sum())).sum()
    V = np.sqrt(chi / (t.values.sum() * (min(t.shape) - 1)))
    pb = pd.DataFrame({"p": pat[rows], "b": BATCH[rows], "s": SCAN[rows], "y": y[rows]})
    R["batch_by_scanner"] = {"pre_event_samples": mtab(BATCH[rows], SCAN[rows]), "cramers_v": r3(V), "batches_with_both_scanners": int((t > 0).all(1).sum()), "n_batches": int(len(t)),
                             "scanner_by_class_samples": mtab(SCAN[rows], np.where(y[rows] == 1, "progressor", "non_progressor")),
                             "patients_with_both_scanners": cnt(int((pb.groupby("p").s.nunique() > 1).sum())), "patients_with_several_batches": cnt(int((pb.groupby("p").b.nunique() > 1).sum())),
                             "note": "scanner = the slide's scanner (each scanner was used in one scan campaign, 2016 or 2017), i.e. the H&E batch"}
    # one-variable AUROCs with fold-honest target encoding (rv4_round4.r31 (b), unchanged)
    def target_enc(cat):
        S = np.full((n, 10), np.nan)
        for j in range(10):
            for k in range(1, 11):
                tr = rows[FO0[rows, j] != k]; te = FO0[:, j] == k; share = pd.Series(y[tr]).groupby(cat[tr]).mean(); S[te, j] = pd.Series(cat[te]).map(share).fillna(y[tr].mean()).values
        return S
    SCO = {"batch_target_encoded": target_enc(BATCH), "scanner_target_encoded": target_enc(SCAN), "batch_x_scanner_target_encoded": target_enc((pd.Series(BATCH) + "|" + pd.Series(SCAN)).values)}
    R["one_variable_auroc"] = hblock(SCO, rows, bb)["auroc"]
    # is copy number batch-associated? (a) batch probes from each modality (rv4_round4.r31 (d) probe, unchanged settings), overall and within class
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    XC = pd.read_csv(M + "/cnv_their_C.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(ids).values.astype(float)
    XI = pd.read_csv(M + "/img_mean.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids).values.astype(float); assert np.isfinite(XC).all() and np.isfinite(XI).all()
    def auc_plain(t, p):
        r = rankdata(p); n1 = t.sum(); n0 = len(t) - n1; return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    def probe(X, idx, minpat):
        g = pat[idx]; out = {}; npat = pd.Series(g).groupby(BATCH[idx]).nunique()
        for bch in sorted(npat.index[npat >= minpat]):
            t = (BATCH[idx] == bch).astype(int); p = np.full(len(idx), np.nan)
            if t.min() == t.max(): continue
            for tr, te in GroupKFold(5).split(X[idx], t, g):
                if t[tr].min() == t[tr].max(): p[te] = 0.5; continue
                mdl = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=1.0, max_iter=5000)).fit(X[idx][tr], t[tr]); p[te] = mdl.predict_proba(X[idx][te])[:, 1]
            out[bch] = {"n_patients": cnt(npat[bch]), "auroc_one_vs_rest": r3(auc_plain(t, p))}
        vals = [v["auroc_one_vs_rest"] for v in out.values()]; return {"batches": out, "n_batches_probed": len(out), "auroc_mean_over_batches": r3(np.mean(vals)) if vals else None}
    PR = {}
    for nm, idx, mp_ in (("pre_event_all", rows, 5), ("non_progressor_pre_event", rows[y[rows] == 0], 3), ("progressor_pre_event", rows[y[rows] == 1], 3)):
        PR[nm] = {"copy_number_matrix": probe(XC, idx, mp_), "uni2h_slide_mean": probe(XI, idx, mp_), "min_patients_per_batch": mp_}; print("probe", nm, flush=True)
    PR["note"] = "one-vs-rest L2 logistic (C = 1, standardised in the training folds), patient-grouped 5-fold CV, AUROC of out-of-fold probabilities; within class, batch is not confounded with the label"
    R["batch_probe"] = PR
    # (b) between-batch variance of each model's held-out score within class (patient level: mean repeat-mean logit, modal batch), permutation p
    lg = lambda p_: np.log(np.clip(p_, 1e-6, 1 - 1e-6) / (1 - np.clip(p_, 1e-6, 1 - 1e-6))); SV = {}
    for m in ("L-CNV", "L-IMG"):
        z = lg(O0[m]).mean(1); D_ = pd.DataFrame({"p": pat[rows], "z": z[rows], "b": BATCH[rows], "y": y[rows]})
        P_ = D_.groupby("p").agg(z=("z", "mean"), b=("b", lambda s: s.value_counts().sort_index().idxmax()), y=("y", "max")); o = {}
        for cls, nmc in ((0, "non_progressor"), (1, "progressor")):
            Q = P_[P_.y == cls]; Q = Q[Q.b.map(Q.b.value_counts()) >= 2]
            def eta(zz, bb_): g_ = pd.Series(zz).groupby(bb_); return float(((g_.transform("mean") - zz.mean()) ** 2).sum() / ((zz - zz.mean()) ** 2).sum())
            e0 = eta(Q.z.values, Q.b.values); rng = np.random.RandomState(0); null = np.array([eta(Q.z.values, rng.permutation(Q.b.values)) for _ in range(5000)])
            o[nmc] = {"eta_squared": r3(e0), "perm_p": round(float((1 + (null >= e0).sum()) / 5001), 4), "null_mean_eta_squared": r3(null.mean()), "n_patients": int(len(Q)), "n_batches": int(Q.b.nunique())}
        SV[m] = o
    SV["note"] = "patients' mean held-out logit (repeat mean) grouped by their modal batch, batches with >= 2 patients of that class; eta squared = between-batch share of variance; p from 5,000 permutations of batch across patients"
    R["score_batch_association_within_class"] = SV; res["batch"] = R; dump(res)
# ================================================================== item 5: depth per sample (row-level)
elif PTASK == "depthprep":
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    def rcs(cid, kb=50):   # dd_describe.py rcs, unchanged
        for d in (f"{SWG}/copy_number_hg38/train/perPatient/{kb}kb/{cid}", f"{SWG}/copy_number_hg38/val/perPatient/{kb}kb/{cid}", f"{SWG}/copy_number_hg38/val/{kb}kb/{cid}"):
            f = f"{d}/{kb}.readCountSummary.txt"
            if os.path.exists(f): x = pd.read_csv(f, sep="\t"); return float(x["total.reads"].iloc[0]), float(x["used.reads"].iloc[0])
        return np.nan, np.nan
    def bam(cid):   # dd_describe.py bam (genome length chr1-22, X, Y and modal read length)
        b = f"{SWG}/dna_seq_bam/{cid}.bam"
        if not os.path.exists(b): return np.nan, np.nan
        h = subprocess.run([ST, "view", "-H", b], capture_output=True, text=True).stdout.splitlines()
        sq = {l.split("\tSN:")[1].split("\t")[0]: int(l.split("\tLN:")[1].split("\t")[0]) for l in h if l.startswith("@SQ")}; glen = sum(v for k, v in sq.items() if re.fullmatch(r"chr([0-9]+|X|Y)", k))
        L = subprocess.run(f"{ST} view {b} | head -2000 | cut -f10 | awk '{{print length($0)}}'", shell=True, capture_output=True, text=True).stdout.split()
        return glen, (int(np.bincount([int(v) for v in L]).argmax()) if L else np.nan)
    def one(cid): t, u = rcs(cid); g, rl = bam(cid); return {"Sample": cid, "total": t, "used": u, "glen": g, "rl": rl}
    with ThreadPoolExecutor(8) as ex: D = pd.DataFrame(list(ex.map(one, ids)))
    D["depth_total"] = D.total * D.rl / D.glen; D["depth_used"] = D.used * D.rl / D.glen
    D["reads_sheet"] = pd.to_numeric(FS["Number of reads"].astype(str).str.replace(",", ""), errors="coerce").values; D["depth_sheet"] = D.reads_sheet * 50 / 3088269832
    D.to_csv(ROW + "/depth.csv", index=False); print("P1B DONE depthprep", int(D.depth_total.notna().sum()), float(D.depth_total.median()), flush=True)
# ================================================================== item 5: depth, QC and biopsy size
elif PTASK == "depth":
    D = pd.read_csv(ROW + "/depth.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); DEP = D.depth_total.values; DSH = D.depth_sheet.values
    QC = pd.read_csv(HZ + "/pkg_qc.csv", dtype={"Sample": str}).drop_duplicates("Sample").set_index("Sample").reindex(ids)
    QPASS = QC.Pass.astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).values.astype(float); VMAD = QC.varMAD_median.values.astype(float)
    rows = ROWS; bb = boots(rows); R = {}
    def desc(v):
        v = v[np.isfinite(v)]; q = np.percentile(v, [10, 25, 50, 75, 90]); return {"n": int(len(v)), "median": round(float(q[2]), 3), "iqr": [round(float(q[1]), 3), round(float(q[3]), 3)], "p10_p90": [round(float(q[0]), 3), round(float(q[4]), 3)],
                                                                                 "min": round(float(v.min()), 3), "max": round(float(v.max()), 3), "n_above_1x": cnt((v > 1).sum()), "n_below_0.1x": cnt((v < 0.1).sum())}
    R["depth_distribution"] = {"bam_based_pre_event": desc(DEP[rows]), "sheet_reads_x50bp_pre_event": desc(DSH[rows]), "bam_based_all_676": desc(DEP),
                               "by_class_pre_event": {"progressor": desc(DEP[rows][y[rows] == 1]), "non_progressor": desc(DEP[rows][y[rows] == 0])},
                               "by_batch_pre_event_median": {str(b): round(float(np.nanmedian(DEP[rows][BATCH[rows] == b])), 3) for b in sorted(set(BATCH[rows]))},
                               "spearman_bam_vs_sheet": r3(spearmanr(DEP[rows], DSH[rows], nan_policy="omit")[0]), "definition": "total reads (50-kb readCountSummary) x modal read length / genome length chr1-22, X, Y (dd_describe.py); sheet: Number of reads x 50 bp / 3,088,269,832"}
    def mdiff(v, a, b_):   # median difference between two groups with patient bootstrap
        o = float(np.nanmedian(v[rows][a[rows]]) - np.nanmedian(v[rows][b_[rows]])); d = [np.nanmedian(v[x][a[x]]) - np.nanmedian(v[x][b_[x]]) for x in bb]; return dstat(o, d)
    R["depth_progressor_minus_nonprogressor_median"] = mdiff(DEP, y == 1, y == 0)
    def strata(name, lab, levels):
        out = {}; KK = {}
        for lv in levels:
            mk = lab == lv
            if mk[rows].sum() == 0: continue
            sub = rows[mk[rows]]
            n1_ = int(y[sub].sum()); p1_ = len(set(pat[sub][y[sub] == 1])); p0_ = len(set(pat[sub][y[sub] == 0]))
            if min(n1_, len(sub) - n1_, p1_, p0_) < 5: out[str(lv)] = {"n_samples": cnt(len(sub)), "status": "not reported: fewer than 5 samples or patients in a class"}; continue
            r_, o_, B_ = hblock(O0, rows, bb, mask=mk, keep=True); out[str(lv)] = r_; KK[lv] = (o_, B_)
        return out, KK
    def t3(v):
        q = np.nanquantile(v[rows], [1 / 3, 2 / 3]); lab = np.where(~np.isfinite(v), -1, np.digitize(v, q)); return lab, [round(float(x), 3) for x in q]
    DT, dq = t3(DEP); out, KK = strata("depth", DT, (0, 1, 2)); out["cuts_x"] = dq
    if 0 in KK and 2 in KK:
        (o1, B1), (o2, B2) = KK[0], KK[2]; out["low_minus_high_tertile"] = {"L-CNV": did(o1, B1, o2, B2, "L-CNV"), "L-IMG": did(o1, B1, o2, B2, "L-IMG"), "IMG_minus_CNV": did(o1, B1, o2, B2, "L-IMG", "L-CNV"), "LATE_minus_IMG": did(o1, B1, o2, B2, "L-LATE", "L-IMG")}
    out["within_depth_tertile_pairs"] = hblock(O0, rows, bb, cell=DT); R["depth_tertiles"] = out; print("depth tertiles", flush=True)
    R["sWGS_QC_pass"] = {"counts_pre_event": mtab(np.where(QPASS[rows] == 1, "pass", np.where(QPASS[rows] == 0, "fail", "missing")), np.where(y[rows] == 1, "progressor", "non_progressor"))}
    o_, _ = strata("qc", QPASS, (1.0, 0.0)); R["sWGS_QC_pass"]["strata"] = o_
    VT, vq = t3(VMAD); o_, KV = strata("varmad", VT, (0, 1, 2)); o_["cuts"] = vq
    if 0 in KV and 2 in KV: (o1, B1), (o2, B2) = KV[0], KV[2]; o_["low_noise_minus_high_noise"] = {"L-CNV": did(o1, B1, o2, B2, "L-CNV"), "IMG_minus_CNV": did(o1, B1, o2, B2, "L-IMG", "L-CNV")}
    R["sWGS_noise_varMAD_tertiles"] = o_; print("qc", flush=True)
    # biopsy size: < 256 vs >= 256 tissue tiles (R32) and tissue tertiles (cuts 200 / 302 as round 2)
    SMALL = np.where(TILES < 256, 1, 0); o_, KS = strata("size", SMALL, (1, 0))
    (o1, B1), (o2, B2) = KS[1], KS[0]
    o_["small_minus_large"] = {"L-CNV": did(o1, B1, o2, B2, "L-CNV"), "L-IMG": did(o1, B1, o2, B2, "L-IMG"), "L-LATE": did(o1, B1, o2, B2, "L-LATE"),
                               "IMG_minus_CNV": did(o1, B1, o2, B2, "L-IMG", "L-CNV"), "LATE_minus_IMG": did(o1, B1, o2, B2, "L-LATE", "L-IMG")}
    # the same interaction with pairs matched on depth tertile (pairs share fold, size group and depth tertile)
    a1, oa1, Ba1 = hblock(O0, rows, bb, mask=SMALL == 1, cell=DT, keep=True); a2, oa2, Ba2 = hblock(O0, rows, bb, mask=SMALL == 0, cell=DT, keep=True)
    o_["depth_matched_pairs"] = {"small": a1, "large": a2, "small_minus_large": {"L-CNV": did(oa1, Ba1, oa2, Ba2, "L-CNV"), "IMG_minus_CNV": did(oa1, Ba1, oa2, Ba2, "L-IMG", "L-CNV")}}
    TT = np.where(TILES < 200, 0, np.where(TILES <= 302, 1, 2)); o2_, KT = strata("tt", TT, (0, 1, 2)); o_["tissue_tertiles_200_302"] = o2_
    if 0 in KT and 2 in KT: (p1, C1), (p2_, C2) = KT[0], KT[2]; o_["tissue_tertiles_200_302"]["low_minus_high"] = {"L-CNV": did(p1, C1, p2_, C2, "L-CNV"), "IMG_minus_CNV": did(p1, C1, p2_, C2, "L-IMG", "L-CNV")}
    R["biopsy_size"] = o_; print("size", flush=True)
    # are small biopsies low depth?
    sp = spearmanr(TILES[rows], DEP[rows], nan_policy="omit")[0]; spb = [spearmanr(TILES[x], DEP[x], nan_policy="omit")[0] for x in bb]
    R["size_vs_depth"] = {"spearman_tissue_tiles_vs_depth": {"value": r3(sp), "ci95": ci2(spb)}, "median_depth_small_minus_large": mdiff(DEP, SMALL == 1, SMALL == 0),
                          "median_depth_small": round(float(np.nanmedian(DEP[rows][SMALL[rows] == 1])), 3), "median_depth_large": round(float(np.nanmedian(DEP[rows][SMALL[rows] == 0])), 3),
                          "tissue_tertile_by_depth_tertile": mtab(TT[rows], DT[rows]), "small_by_depth_tertile": mtab(SMALL[rows], DT[rows]),
                          "spearman_tissue_tiles_vs_varMAD": r3(spearmanr(TILES[rows], VMAD[rows], nan_policy="omit")[0]), "qc_fail_small_vs_large": mtab(SMALL[rows], QPASS[rows])}
    res["depth"] = R; dump(res)
# ================================================================== item 6: nested choice among image variants (E3 refits)
elif PTASK.startswith("select_"):
    RD = M + "/explore_E3/refit"
    def read_refit(name):   # ex_e3_score.read_refit (outer pred, inner pred and inner fold), unchanged in substance
        fo = sorted(glob.glob(f"{RD}/{name}/outer_rep_*.csv")); fi = sorted(glob.glob(f"{RD}/{name}/inner_rep_*.csv")); assert len(fo) == 10 and len(fi) == 10, (name, len(fo), len(fi))
        D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); po = lambda c: D.pivot_table(index="Sample", columns="rep", values=c, dropna=False).reindex(ids)[REPS].values
        assert (po("fold").astype(int) == FO0).all(), "folds differ from the stored folds"
        Di = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fi]); assert Di.groupby(["rep", "fold"]).ngroups == 100; I, IF = {}, {}
        for (r, k), d in Di.groupby(["rep", "fold"]): pos = ix[d.Sample].values; I[(r, k)] = pd.Series(d.pred.values, index=pos); IF[(r, k)] = pd.Series(d.inner_fold.values, index=pos)
        return {"O": po("pred"), "I": I, "IF": IF}
    def wauc(x, l, f):   # ex_e3_score.wauc (unweighted)
        num = den = 0.0
        for k in np.unique(f):
            m = (f == k) & np.isfinite(x); xp, xn = x[m & (l == 1)], x[m & (l == 0)]
            if len(xp) == 0 or len(xn) == 0: continue
            num += ((xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])).sum(); den += len(xp) * len(xn)
        return num / den if den else np.nan
    def inner_auc(sr, ifo, rr): return wauc(sr.reindex(rr).values, y[rr], ifo.reindex(rr).values)
    CACHE = {}
    def cand(c):
        if c in CACHE: return CACHE[c]
        if c.startswith("ens:"):
            L = [cand(a) for a in c[4:].split("+")]; v = {"O": np.mean([x["O"] for x in L], 0), "I": {k: pd.concat([x["I"][k] for x in L], axis=1).mean(1) for k in L[0]["I"]}, "IF": L[0]["IF"]}
        else: v = read_refit(c)
        CACHE[c] = v; return v
    E3A = ["base", "u2l1_mean", "gpl1_mean", "gpl2_mean", "gpl3_mean", "v2l2_mean", "co20_mean", "mu20_mean", "cat_u2l12_mean", "cat_gp123_mean", "cat_u2gp2_mean"]
    E3B = ["base", "u2l1_mean", "gpl2_mean", "v2l2_mean", "ho20_mean", "v220_mean", "co20_mean", "ct20_mean", "mu20_mean"]
    POOL = ["base", "u2l2_max", "u2l2_meansd", "u2l2_q3", "u2l2_top8n", "u2l2_top16n", "u2l2_top32n", "base_tpmax", "base_tptop8", "base_tptop16", "base_tptop32"]
    ENS = ["ens:base+u2l1_mean", "ens:base+gpl2_mean", "ens:base+ho20_mean", "ens:base+ho20_mean+v220_mean+gpl2_mean", "ens:base+u2l1_mean+gpl2_mean+v2l2_mean+co20_mean+mu20_mean"]
    ALLD = sorted(os.path.basename(d) for d in glob.glob(RD + "/*") if os.path.isdir(d) and len(glob.glob(d + "/inner_rep_*.csv")) == 10)
    SETS = {"e3_11": E3A, "e3_9": E3B, "poolings_11": POOL, "extractors_scales_poolings_ensembles": sorted(set(ALLD) | set(ENS))}
    nm = PTASK[len("select_"):]; C = SETS[nm]; L = {c: cand(c) for c in C}
    chk = float(np.nanmax(np.abs(L["base"]["O"] - O0["L-IMG"]))); assert chk < 1e-9, chk
    Oi = np.full((n, 10), np.nan); pick = {c: 0 for c in C}
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            tri = L[C[0]]["I"][(r, k)].index.values; tri = tri[PRE[tri]]; a = {c: inner_auc(L[c]["I"][(r, k)], L[c]["IF"][(r, k)], tri) for c in C}
            b_ = max(C, key=lambda c: (a[c], -C.index(c))); pick[b_] += 1; te = FO0[:, j] == k; Oi[te, j] = L[b_]["O"][te, j]
    rows = ROWS; bb = boots(rows)
    OS = {"L-CNV": O0["L-CNV"], "L-IMG": O0["L-IMG"], "L-LATE": O0["L-LATE"], "IMG-nested": Oi, "LATE-nested": (O0["L-CNV"] + Oi) / 2}
    for c in C: OS["cand:" + c] = L[c]["O"]
    obs = {m: fsa(S_, rows) for m, S_ in OS.items()}; B = pd.DataFrame([{m: fsa(S_, b) for m, S_ in OS.items()} for b in bb])
    cn = ["cand:" + c for c in C]; best = max(cn, key=lambda c: obs[c]); Bmax = B[cn].max(1)
    R = {"candidates": C, "n_candidates": len(C), "base_reproduces_stored_L_IMG_max_abs_diff": chk, "times_chosen_of_100_folds": pick,
         "auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in ("L-CNV", "L-IMG", "L-LATE", "IMG-nested", "LATE-nested")},
         "nested_minus_L_IMG": dstat(obs["IMG-nested"] - obs["L-IMG"], B["IMG-nested"] - B["L-IMG"]), "nested_IMG_minus_L_CNV": dstat(obs["IMG-nested"] - obs["L-CNV"], B["IMG-nested"] - B["L-CNV"]),
         "L_IMG_minus_L_CNV": dstat(obs["L-IMG"] - obs["L-CNV"], B["L-IMG"] - B["L-CNV"]), "nested_LATE_minus_L_CNV": dstat(obs["LATE-nested"] - obs["L-CNV"], B["LATE-nested"] - B["L-CNV"]),
         "nested_LATE_minus_nested_IMG": dstat(obs["LATE-nested"] - obs["IMG-nested"], B["LATE-nested"] - B["IMG-nested"]),
         "apparent_best_candidate": {"name": best[5:], "auroc": r3(obs[best]), "max_over_candidates_in_draws_ci95": ci2(Bmax)},
         "optimism_apparent_best_minus_nested": dstat(obs[best] - obs["IMG-nested"], Bmax - B["IMG-nested"]),
         "optimism_L_IMG_minus_nested": dstat(obs["L-IMG"] - obs["IMG-nested"], B["L-IMG"] - B["IMG-nested"]),
         "candidate_auroc": {c: r3(obs["cand:" + c]) for c in C}, "rank_of_L_IMG_among_candidates": int(sorted([obs[c] for c in cn], reverse=True).index(obs["cand:base"]) + 1),
         "note": "nested = per (repeat, outer fold) the candidate with the best fold-stratified inner out-of-fold AUROC on the fold's pre-event training samples (ex_e3_score.py sel:); its held-out predictions are used for that fold; ensembles average probabilities (outer and inner alike); apparent best = the candidate with the highest held-out AUROC, chosen after seeing it (optimistic); in each draw the max over candidates is taken"}
    res["select"] = R; dump(res)
# ================================================================== item 7: grade and p53
elif PTASK == "grade":
    from sklearn.linear_model import LogisticRegression
    rows = ROWS; assert np.nanmax(GR[rows]) <= 2, "pre-event samples with HGD/IMC"; bb = boots(rows)
    lg = lambda p_: np.log(np.clip(p_, 1e-6, 1 - 1e-6) / (1 - np.clip(p_, 1e-6, 1 - 1e-6)))
    def design(r_, sc):
        g = GR[r_]; X = [(g == 1).astype(float), (g == 2).astype(float)]
        if "p53" in sc: p = P53[r_]; X += [np.where(p == 1, 1.0, 0.0), np.where(np.isfinite(p), 0.0, 1.0)]
        if "p53ab" in sc: X += [np.where(P53[r_] == 1, 1.0, 0.0)]
        for m in ("L-IMG", "L-CNV", "L-LATE"):
            if m in sc: X.append(sc[m])
        return np.column_stack(X)
    SPECS = {"GRADE": [], "GRADE+IMG": ["L-IMG"], "GRADE+CNV": ["L-CNV"], "GRADE+LATE": ["L-LATE"], "GRADE+P53": ["p53"], "GRADE+P53+IMG": ["p53", "L-IMG"], "GRADE+P53+LATE": ["p53", "L-LATE"], "GRADE+P53+CNV": ["p53", "L-CNV"],
             "GRADE+P53AB": ["p53ab"], "GRADE+P53AB+IMG": ["p53ab", "L-IMG"], "GRADE+P53AB+LATE": ["p53ab", "L-LATE"]}
    S = {k: np.full((n, 10), np.nan) for k in SPECS}; nconv = 0
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            tr = I0["L-IMG"][(r, k)].index.values; tr = tr[PRE[tr]]; te = np.where((FO0[:, j] == k) & PRE)[0]
            for nm_, sp_ in SPECS.items():
                sctr = {q: 1 for q in ("p53", "p53ab") if q in sp_}; scte = dict(sctr)
                for m in ("L-IMG", "L-CNV", "L-LATE"):
                    if m in sp_: sctr[m] = lg(I0[m][(r, k)].reindex(tr).values); scte[m] = lg(O0[m][te, j])
                mdl = LogisticRegression(penalty=None, max_iter=10000).fit(design(tr, sctr), y[tr]); nconv += int(mdl.n_iter_[0] >= 10000)
                S[nm_][te, j] = mdl.decision_function(design(te, scte))
    S.update({"L-IMG": O0["L-IMG"], "L-CNV": O0["L-CNV"], "L-LATE": O0["L-LATE"], "raw_grade": np.repeat(GR[:, None], 10, 1)})
    obs = {m: fsa(S_, rows) for m, S_ in S.items()}; B = pd.DataFrame([{m: fsa(S_, b) for m, S_ in S.items()} for b in bb])
    R = {"auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in S}, "fits_not_converged": nconv}
    PAIRS = [("GRADE+IMG", "GRADE"), ("GRADE+CNV", "GRADE"), ("GRADE+LATE", "GRADE"), ("GRADE+P53+IMG", "GRADE+P53"), ("GRADE+P53+CNV", "GRADE+P53"), ("GRADE+P53+LATE", "GRADE+P53"),
             ("GRADE+P53", "GRADE"), ("GRADE+P53AB+IMG", "GRADE+P53AB"), ("GRADE+P53AB+LATE", "GRADE+P53AB"), ("GRADE+P53AB", "GRADE"), ("GRADE+IMG", "L-IMG"), ("GRADE+LATE", "L-LATE"), ("GRADE+IMG", "GRADE+CNV"), ("L-IMG", "raw_grade")]
    R["differences"] = {f"{a}_minus_{b_}": dstat(obs[a] - obs[b_], B[a] - B[b_]) for a, b_ in PAIRS}
    R["note"] = "per (repeat, outer fold): unpenalised logistic regression on the fold's pre-event training samples (grade dummies ID, LGD; p53 aberrant and p53 missing dummies (P53), or p53 aberrant only with missing as not aberrant (P53AB); model scores as inner out-of-fold logits), applied to the held-out samples with their outer logits; fold-stratified AUROC on held-out pre-event samples"
    # within grade: NDBE only, pairs sharing grade, p53-normal only
    R["within_grade"] = {"ndbe_only": hblock(O0, rows, bb, mask=NDBE), "pairs_sharing_grade": hblock(O0, rows, bb, cell=GR.astype(int)), "p53_normal_only": hblock(O0, rows, bb, mask=(P53 == 0)),
                         "ndbe_and_p53_normal": hblock(O0, rows, bb, mask=NDBE & (P53 == 0)), "ndbe_only_patient_weighted": hblock(O0, rows, bb, mask=NDBE, w=patw(rows[NDBE[rows]]))}
    # apparent (non-nested, optimistic): one logistic per spec on all pre-event samples with repeat-mean outer logits, scored in sample
    AP = {}
    for nm_, sp_ in SPECS.items():
        sc = {q: 1 for q in ("p53", "p53ab") if q in sp_}
        for m in ("L-IMG", "L-CNV", "L-LATE"):
            if m in sp_: sc[m] = lg(O0[m]).mean(1)[rows]
        mdl = LogisticRegression(penalty=None, max_iter=10000).fit(design(rows, sc), y[rows]); z = np.full(n, np.nan); z[rows] = mdl.decision_function(design(rows, sc)); AP[nm_] = fsa(np.repeat(z[:, None], 10, 1), rows)
    R["apparent_optimistic"] = {k: r3(v) for k, v in AP.items()}; res["grade"] = R; dump(res)
# ================================================================== item 9: sensitivity rows (labels, same block, added samples, sheet flags)
elif PTASK == "sens":
    rows = ROWS; bb = boots(rows); R = {}
    # label provenance: sheet Status vs the sheet's original P / NP column
    st = FS.Status.values; orig = FS["Progressor/ Non Progressor"].str.strip().values
    R["label_check"] = {"pre_event_samples_status_equals_original_column": int((st[rows] == orig[rows]).sum()), "n": int(len(rows)), "note": "sheet Status (used throughout) against 'Progressor/ Non Progressor'; the published Fig 2a table carries no label column"}
    R["primary_original_labels"] = hblock(O0, rows, bb)
    # (a) Barrett's database event dates (R53 logic, rv7_round7.py, evaluation only)
    ns7 = other_ns("rv7_round7.py"); L7 = ns7["db_link"](); pdate = L7["pdate"]; sd_of = L7["sd_of"]; pt = L7["pt"]
    A = pd.read_csv(KD + "/kr_samples.csv", dtype=str).set_index("Sample"); A["mbf"] = pd.to_numeric(A.index.map(FI["Months before final"]), errors="coerce"); A["ogd"] = A.index.map(FI["Path_class_per_OGD"])
    hgA = A[(A.Pathology.isin(["HGD", "IMC"])) | (A.ogd.isin(["HGD", "IMC", "HGD/IMC"]))]; tevA = hgA.groupby("Patient").mbf.max()
    SD_ = pd.to_datetime(pd.Series(ids).map(pdate).values); ev_ = {}
    for p in sorted(set(pat[y == 1])):
        sh = pd.NaT
        if p in tevA.index:
            d_ = [sd_of.get(s) for s in A[(A.Patient == p) & (A.mbf == tevA[p])].index]; d_ = [x for x in d_ if pd.notna(x)]; sh = min(d_) if d_ else pd.NaT
        pid = L7["link"].get(p); db = pt[(pt.pid == pid) & (pt.g >= 3)].d.min() if pid is not None else pd.NaT
        ev_[p] = min([x for x in (sh, db) if pd.notna(x)], default=pd.NaT)
    EVD = pd.to_datetime(pd.Series(pat).map(ev_).values); drop = PRE & (y == 1) & pd.notna(EVD) & (SD_ >= EVD)
    R["db_event_dates_eval_only"] = {"removed_samples": cnt(drop[rows].sum()), **hblock(O0, rows, bb, mask=~drop)}
    # (b) sheet label-review flags (evaluation only)
    cons = FS["consistency after review"].astype(str).str.upper().values; exc = FS.excluded.astype(str).str.strip().values
    R["sheet_review_inconsistent_removed"] = {"removed_samples": cnt((cons[rows] == "FALSE").sum()), **hblock(O0, rows, bb, mask=cons != "FALSE")}
    R["sheet_excluded_flag_removed"] = {"removed_samples": cnt((exc[rows] == "1").sum()), **hblock(O0, rows, bb, mask=exc != "1")}
    # (c) same block only (evaluation only)
    ns7d = other_ns("rv7d_sensfirst.py"); SB = ns7d["same_block_labels"](); same = SB == "same case, same block"
    R["same_block_only"] = {"labels_pre_event": {k: cnt(v) for k, v in pd.Series(SB[rows]).value_counts().items()}, "removed_samples": int((~same[rows]).sum()), **hblock(O0, rows, bb, mask=same)}
    print("eval-only rows done", flush=True)
    # (d) refits: R54 database-updated labels, R43 control exclusions, R52 added samples (rv7_refit.R outputs)
    R7 = M + "/review_round7"
    def load_var(var):
        V = pd.read_csv(f"{R7}/sets/{var}.csv", dtype={"Sample": str, "Patient": str}); D = f"{R7}/refit/{var}"; vid = V.Sample.values; vix = pd.Series(np.arange(len(V)), index=vid)
        def outer(pat_):
            Dd = pd.concat([pd.read_csv(x, dtype={"Sample": str}) for x in sorted(glob.glob(f"{D}/{pat_}"))]); assert Dd.rep.nunique() == 10
            return Dd.pivot_table(index="Sample", columns="rep", values="pred").reindex(vid)[REPS].values, Dd.pivot_table(index="Sample", columns="rep", values="fold").reindex(vid)[REPS].values.astype(int)
        OC, FOv = outer("outer_cnv_their_rep_*.csv"); OI, F2 = outer("outer_img_rep_*.csv"); assert (FOv == F2).all() and np.isfinite(OC).all() and np.isfinite(OI).all()
        ev = Env(vid, V.y.values, V.Patient.values, FOv, V.pre.astype(str).eq("True").values); return {"L-CNV": OC, "L-IMG": OI, "L-LATE": (OC + OI) / 2}, ev
    REF = {}
    for var, desc_ in (("base", "refit of the primary set (reproduction check)"), ("r54", "Barrett's database follow-up: later HGD relabelled progressor, later LGD patients excluded"),
                       ("r43c", "the non-progressor with later HGD removed"), ("r43a", "non-progressor samples < 12 months before the final endoscopy removed"), ("r43b", "non-progressor samples < 24 months before the final endoscopy removed"),
                       ("r52a", "8 progressors' linkable samples added: same case and block"), ("r52b", "8 progressors' linkable samples added: also block unrecorded")):
        if not os.path.isdir(f"{R7}/refit/{var}"): REF[var] = {"status": "not available"}; continue
        if var == "base":   # repeat 1 only: reproduction check against the stored predictions
            chk = {}
            for mm, fn in (("L-CNV", "outer_cnv_their_rep_01.csv"), ("L-IMG", "outer_img_rep_01.csv")):
                Dd = pd.read_csv(f"{R7}/refit/base/{fn}", dtype={"Sample": str}).set_index("Sample").reindex(ids); chk[mm] = float(np.nanmax(np.abs(Dd.pred.values - O0[mm][:, 0])))
            REF[var] = {"description": desc_, "repeat1_max_abs_diff_vs_stored": chk}; continue
        Ov, ev = load_var(var); rv = np.where(ev.pre)[0]; bv = boots(rv, ev); REF[var] = {"description": desc_, **hblock(Ov, rv, bv, ev)}; print("refit", var, flush=True)
    R["refits"] = REF; res["sens"] = R; dump(res)
# ================================================================== item 9: samples without a slide vs samples with one (selection check)
elif PTASK == "exclsel":
    A = pd.read_csv(KD + "/kr_samples.csv", dtype=str); A["inc"] = A.Sample.isin(set(ids)); A["y"] = (A.Status == "P").astype(int); f = FI.reindex(A.Sample.values)
    A["mbf"] = pd.to_numeric(f["Months before final"].values, errors="coerce"); A["ogd"] = f["Path_class_per_OGD"].values
    hg = A[(A.Pathology.isin(["HGD", "IMC"])) | (A.ogd.isin(["HGD", "IMC", "HGD/IMC"]))]; tev = hg.groupby("Patient").mbf.max()   # hz_prep.py event rule
    A["pre"] = [True if yy == 0 else (m > (tev[p] if p in tev.index else 0.0)) for yy, p, m in zip(A.y, A.Patient, A.mbf)]   # hz_prep.py: fallback event time 0
    A["batch"] = f.Batch.str.strip().values; A["reads"] = pd.to_numeric(f["Number of reads"].astype(str).str.replace(",", "").values, errors="coerce"); A["p53"] = pd.to_numeric(A["P53 IHC"], errors="coerce")
    A["year"] = pd.to_numeric(A.year, errors="coerce"); A["k"] = pd.to_numeric(A.k_prob, errors="coerce"); A["grade"] = A.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4})
    X = pd.read_csv(KD + "/their_features.csv", dtype={"Sample": str}).set_index("Sample").reindex(A.Sample.values); A["cn_abs"] = np.abs(X.values).mean(1)
    pats_out = sorted(set(A.Patient[~A.inc]) - set(A.Patient[A.inc]))
    R = {"counts": {"published_samples": int(len(A)), "with_slide": int(A.inc.sum()), "without_slide": int((~A.inc).sum()), "patients_without_any_slide": len(pats_out),
                    "without_slide_by_class": {c: cnt(((~A.inc) & (A.y == v)).sum()) for c, v in (("progressor", 1), ("non_progressor", 0))},
                    "without_slide_pre_event_by_class": {c: cnt(((~A.inc) & A.pre & (A.y == v)).sum()) for c, v in (("progressor", 1), ("non_progressor", 0))},
                    "without_slide_from_patients_with_other_slides": cnt(((~A.inc) & ~A.Patient.isin(pats_out)).sum())}}
    up = A.Patient.unique(); rng = np.random.RandomState(0); BI = [rng.choice(len(up), len(up)) for _ in range(NBX)]; grp = {p: np.where(A.Patient.values == p)[0] for p in up}
    BR = [np.concatenate([grp[up[i]] for i in b]) for b in BI]
    def cmp(var, sel):
        v = A[var].values.astype(float); a = sel & ~A.inc.values; b = sel & A.inc.values
        def st(rr): x1 = v[rr][a[rr]]; x0 = v[rr][b[rr]]; x1 = x1[np.isfinite(x1)]; x0 = x0[np.isfinite(x0)]; return (np.mean(x1) - np.mean(x0)) / np.sqrt((np.var(x1) + np.var(x0)) / 2) if len(x1) > 1 and len(x0) > 1 else np.nan
        o = st(np.arange(len(A))); d = [st(rr) for rr in BR]; va = v[a]; vb = v[b]
        return {"n_without_slide": cnt(np.isfinite(va).sum()), "n_with_slide": int(np.isfinite(vb).sum()), "mean_without": r3(np.nanmean(va)) if np.isfinite(va).sum() >= 5 else None, "mean_with": r3(np.nanmean(vb)),
                "std_mean_difference_without_minus_with": dstat(o, d) if np.isfinite(va).sum() >= 5 else "cell < 5"}
    C = {}
    for cls, nmc in ((1, "progressor"), (0, "non_progressor")):
        for scope, s_ in (("all", np.ones(len(A), bool)), ("pre_event", A.pre.values)):
            sel = (A.y.values == cls) & s_; C[f"{nmc}_{scope}"] = {var: cmp(var, sel) for var in ("k", "cn_abs", "reads", "grade", "p53", "year")}
    R["within_class_comparison"] = C
    R["batch_without_slide_pre_event"] = {str(k): cnt(v) for k, v in A[(~A.inc) & A.pre].batch.value_counts().items()}
    def aucp(rr):
        t = A.y.values[rr]; s = A.k.values[rr]; ok = np.isfinite(s); t, s = t[ok], s[ok]; r_ = rankdata(s); n1 = t.sum(); n0 = len(t) - n1
        return float((r_[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(t) else np.nan
    PA = {}
    for nm_, m_ in (("all_773", np.ones(len(A), bool)), ("with_slide", A.inc.values), ("without_slide", ~A.inc.values), ("pre_event_all", A.pre.values), ("pre_event_with_slide", A.pre.values & A.inc.values), ("pre_event_without_slide", A.pre.values & ~A.inc.values)):
        rr = np.where(m_)[0]; o = aucp(rr); d = [aucp(b[m_[b]]) for b in BR]
        PA[nm_] = {"n_samples": int(len(rr)), "n_progressor": cnt(A.y.values[rr].sum()), "auroc_published_P_probability": r3(o), "ci95": ci2(d)}
    R["published_model_P_auroc"] = PA
    R["note"] = "published samples (Killcoyne Fig 2a, 773) with and without a linked H&E slide; pre-event by the hz_prep.py rule; k = published model P probability (their LOPO), cn_abs = mean |value| of the shipped standardised copy-number matrix; standardised mean differences (pooled SD) with a 2,000-draw patient bootstrap over the 88 patients"
    res["exclsel"] = R; dump(res)
