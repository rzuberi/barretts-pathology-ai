"""Two papers, agent CND (docs/two_papers_CND.md): which copy-number regions drive L-CNV (A) and does L-CNV survive shallower sequencing (B).
Metric helpers (fsa, boots, hblock, patw, dstat, ci2, cnt) come from tp_p1b_main.py, whose module body is executed unchanged with a task name matching
none of its branches (as tp_cnc_main.py does). Inputs, read-only: stored predictions (tb_common.load), the refits of tp_cnd_refit.R and the thinned
blocks of tp_cnd_thin.R under killcoyne_mm/two_papers_CND/.
TASK = smoke | score_A | score_B | score_ablate        CND_NB=k: smoke mode (k bootstrap draws; JSON written aside).
Aggregates only (no sample or patient identifiers, no dates; cells < 5 as "<5") -> results/paper_final/two_papers_CND/<TASK>.json."""
import os, sys, json, glob, numpy as np, pandas as pd
PP = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"; sys.path.insert(0, PP)
DTASK = os.environ["TASK"]; os.environ["TASK"] = "cnd_none"
exec(compile(open(PP + "/tp_p1b_main.py").read(), PP + "/tp_p1b_main.py", "exec"))   # definitions only
os.environ["TASK"] = DTASK
E_ = M + "/two_papers_CND"; OUTD = T + "/results/paper_final/two_papers_CND"; NBX = NB
if os.environ.get("CND_NB"): NBX = int(os.environ["CND_NB"]); OUTD = E_ + "/smoke_json"
os.makedirs(OUTD, exist_ok=True)
def ddump(res):
    res["agent"] = "CND"; json.dump(res, open(f"{OUTD}/{DTASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("CND DONE", DTASK, flush=True)
def outer(d):
    fo = sorted(glob.glob(f"{d}/outer_rep_*.csv")); assert len(fo) == 10, (d, len(fo))
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); pv = lambda c: D.pivot_table(index="Sample", columns="rep", values=c).reindex(ids)[REPS].values
    assert (pv("fold").astype(int) == FO0).all(), "folds differ from the stored folds"
    return pv("pred"), D.groupby(["rep", "fold"]).nonzero.first().values
O_PKG = load("pkg")[0]["L-CNV"]
def score(O, refs):
    """AUROC of every model in O and its paired difference against each reference, sample level and patient-weighted"""
    rows = ROWS; bb = boots(rows); W_ = patw(rows); R = {}
    for unit, w in (("sample", None), ("patient_weighted", W_)):
        r, o, B = hblock(O, rows, bb, w=w, keep=True)
        R[unit] = {"n": {k: r[k] for k in ("n_samples", "n_progressor_samples", "n_patients", "n_progressor_patients")}, "auroc": r["auroc"],
                   "delta": {m: {ref: dstat(o[m] - o[ref], B[m] - B[ref]) for ref in refs if ref != m} for m in O}}
    return R
res = {"task": DTASK}
# ------------------------------------------------------------------ smoke: refit path and thinning path reproduce stored numbers
if DTASK == "smoke":
    for tag, cfg, src in (("their", 0, "sub"), ("pkg", 1, "pkg")):
        f = f"{E_}/refit/{tag}/outer_rep_01.csv"
        if os.path.exists(f):
            a = pd.read_csv(f, dtype={"Sample": str}); b = pd.read_csv(f"{M}/cv/preds/cfg_{cfg:02d}_rep_01.csv", dtype={"Sample": str}); m = a.merge(b, on=["Sample", "fold"], suffixes=("", "_s")); assert len(m) == len(a) == len(b)
            res[f"refit_{tag}_rep01_vs_stored_cfg_{cfg:02d}"] = {"max_abs_pred": float(np.abs(m.pred - m.pred_s).max()), "lambda_equal": bool(np.allclose(m["lambda"], m["lambda_s"]))}
    ST = pd.read_csv(M + "/review_round4/reseg/tiles_joint_stored_analysed.csv", dtype={"Sample": str}).set_index("Sample"); tc = [c for c in ST.columns if c not in ("qc_pass", "cx_raw")]
    for tag in ("nat_force", "thin_010"):
        for f in sorted(glob.glob(f"{E_}/thin/{tag}/pat_*.csv")):
            D = pd.read_csv(f, dtype={"Sample": str}).set_index("Sample"); s = D.index.intersection(ST.index)
            dd = np.abs(D.loc[s, tc].values.astype(float) - ST.loc[s, tc].values.astype(float))
            res[f"{tag}_{os.path.basename(f)[:-4]}_vs_stored_tiles"] = {"n_analysed_samples": int(len(s)), "max_abs_tile": float(np.nanmax(dd)), "median_abs_tile": float(np.nanmedian(dd)),
                "nan_mismatch": int((np.isnan(D.loc[s, tc].values.astype(float)) != np.isnan(ST.loc[s, tc].values.astype(float))).sum()), "thinned": int(D.thinned.sum()),
                "keep_p_median": float(D.keep_p.median()), "depth_after_median": float(D.depth_after.median()), "qc_pass": int(D.qc_pass.sum()), "secs": float(D.secs.iloc[0])}
    print(json.dumps(res, indent=1)); ddump(res)
# ------------------------------------------------------------------ A: reduced feature sets
elif DTASK == "score_A":
    SETS = {"arms_cx": "44 arms + cx (fixed)", "top05": "top 5 by |coefficient| of the fold's full L-CNV fit", "top10": "top 10 (same rule)", "top20": "top 20 (same rule)"}
    Oth, nz_full = outer(f"{E_}/refit/their"); d0 = float(np.abs(Oth - O0["L-CNV"]).max()); assert d0 < 1e-9, d0
    O = {"L-CNV": O0["L-CNV"]}; NZ = {}
    for s in SETS: O[s], NZ[s] = outer(f"{E_}/refit/{s}")
    res["checks"] = {"refit_full_vs_stored_L_CNV_max_abs_pred": d0}
    res["score"] = score(O, ["L-CNV"]); res["sets"] = SETS
    res["nonzero_per_fit"] = {s: {"median": float(np.median(v)), "min": int(v.min()), "max": int(v.max())} for s, v in [("full L-CNV", nz_full)] + list(NZ.items())}
    # feature names: hg19 coordinates of the shipped matrix; arms labelled p / q (the arm starting at 1 is p), 5-Mb tiles labelled by the arm holding their midpoint
    cols = pd.read_csv(M + "/cnv_their_C.csv", nrows=0).columns[1:]; assert len(cols) == 634 and cols[-1] == "cx"
    loc = lambda nm: (nm.split(":")[0], *map(int, nm.split(":")[1].split("-"))); arms = {}
    for nm in cols[589:633]: c, s0, e0 = loc(nm); arms[nm] = (c, "p" if s0 == 1 else "q", s0, e0)
    def label(nm):
        if nm == "cx": return "cx (complexity count)"
        if nm in arms: c, a, _, _ = arms[nm]; return f"arm {c}{a}"
        c, s0, e0 = loc(nm); mid = (s0 + e0) / 2; a = [aa for (cc, aa, ss, ee) in arms.values() if cc == c and ss <= mid <= ee]
        return f"5-Mb tile {c}:{s0 / 1e6:.1f}-{e0 / 1e6:.1f} Mb ({c}{a[0] if a else 'cen'}, minus arm)"
    S = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{E_}/refit/sel/sel_rep_*.csv"))]); assert S.groupby(["rep", "fold"]).ngroups == 100
    res["full_fit_nonzero_below_20_folds"] = int((S[S.set == "top20"].groupby(["rep", "fold"]).full_nonzero.first() < 20).sum())
    F = {}
    for s in ("their_nonzero", "top05", "top10", "top20", "arms_cx_nonzero"):
        d = S[S.set == s]; g = d.groupby("feature").agg(n=("coef", "size"), pos=("coef", lambda v: int((v > 0).sum())), mean_coef=("coef", "mean")).sort_values(["n", "mean_coef"], ascending=[False, False]).head(20)
        F[s] = [{"feature": label(f), "coordinates_hg19": f, "selected_of_100_fits": int(r.n), "positive_coef_fits": int(r.pos), "mean_coef_when_selected": r3(r.mean_coef)} for f, r in g.iterrows()]
        F[s + "_distinct_features"] = int(d.feature.nunique())
    res["selection_frequency"] = F
    res["note"] = ("fold-stratified AUROC over the 571 pre-event samples (fsa); 2,000-draw patient bootstrap, paired deltas on the same draws; same 10 x 10 outer folds and class-error lambda rule "
                   "as L-CNV. Top-k chosen inside each outer training fold from that fold's full L-CNV fit (no test rows used). Selection counts are over the 100 outer fits "
                   "(10 repeats x 10 folds); for top-k sets 'selected' = in the top k (coef = refit coefficient, may be 0); for *_nonzero sets = non-zero coefficient. "
                   "5-Mb tile features are deviations from their arm (the shipped matrix subtracts the arm value). Coefficients are on the shipped standardised scale.")
    ddump(res)
# ------------------------------------------------------------------ B: thinned depth
elif DTASK == "score_B":
    Op, _ = outer(f"{E_}/refit/pkg"); d0 = float(np.abs(Op - O_PKG).max()); assert d0 < 1e-9, d0
    TAGS = [t for t in ("thin_020", "thin_010", "thin_005") if len(glob.glob(f"{E_}/refit/{t}/outer_rep_*.csv")) == 10]
    O = {"L-CNV": O0["L-CNV"], "native (joint package)": O_PKG}
    for t in TAGS: O[t] = outer(f"{E_}/refit/{t}")[0]
    res["checks"] = {"refit_pkg_vs_stored_cfg_01_max_abs_pred": d0}
    res["score"] = score(O, ["native (joint package)", "L-CNV"])
    SA = pd.read_csv(M + "/review_round4/reseg/tiles_joint_stored_analysed.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids)
    TH = {}
    for t in TAGS:
        D = pd.read_csv(f"{E_}/thin/{t}/samples_C.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); pre = PRE
        TH[t] = {pop: {"n_samples": int(m.sum()), "left_unthinned_below_target": cnt((~D.thinned.astype(bool) & m).sum()), "depth_after_median": r3(D.depth_after[m].median()),
                       "depth_after_range": [r3(D.depth_after[m].min()), r3(D.depth_after[m].max())], "qc_pass": cnt((D.qc_pass.astype(bool) & m).sum()),
                       "qc_pass_native": cnt((SA.qc_pass.astype(bool) & m).sum()), "cx_raw_median": r3(D.cx_raw[m].median()), "cx_raw_median_native": r3(SA.cx_raw[m].median())}
                 for pop, m in (("analysed_676", np.ones(n, bool)), ("pre_event_571", pre))}
        TH[t]["native_depth_median_pre_event"] = r3(D.depth_native[pre].median())
    res["thinning"] = TH; res["tags"] = {"thin_020": "target 0.2x", "thin_010": "target 0.1x", "thin_005": "target 0.05x"}
    res["note"] = ("native = the stored joint package block (BarrettsProgressionRisk joint segmentation of each patient's samples on our 50-kb QDNAseq counts), refitted with the L-CNV recipe; "
                   "it is the pipeline that can be rerun on thinned counts (the shipped L-CNV matrix cannot). Thinned = Binomial(count, target / native depth) per 50-kb bin for every "
                   "sample segmented with the patient (fixed seeds), QDNAseq fit re-estimated, same segmentation, tiling, standardisation over the 676 analysed samples, refit in the same folds. "
                   "Depth = total reads x 50 bp / 3.09 Gb. fsa over the 571 pre-event samples; 2,000-draw patient bootstrap; paired deltas.")
    ddump(res)
# ------------------------------------------------------------------ ablation: arms 4q, 7p, 17p removed
elif DTASK == "score_ablate":
    Of, _ = outer(f"{E_}/refit/abl_full"); d0 = float(np.abs(Of - O0["L-CNV"]).max()); assert d0 < 1e-9, d0
    O = {"L-CNV": O0["L-CNV"]}; NZ = {}
    for s_ in ("abl_no3arms", "abl_arms_cx_no3"): O[s_], NZ[s_] = outer(f"{E_}/refit/{s_}")
    cols = list(pd.read_csv(M + "/cnv_their_C.csv", nrows=0).columns[1:]); assert len(cols) == 634 and cols[-1] == "cx"
    lc = lambda nm: (nm.split(":")[0], *map(int, nm.split(":")[1].split("-")))
    A3 = {"4q": "4:52700000-191154276", "7p": "7:1-58000000", "17p": "17:1-22200000"}
    hit = {a: [i for i, c in enumerate(cols[:-1]) if lc(c)[0] == lc(v)[0] and lc(c)[1] <= lc(v)[2] and lc(c)[2] >= lc(v)[1]] for a, v in A3.items()}
    allhit = sorted(set(sum(hit.values(), []))); nk = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{E_}/refit/abl_no3arms/outer_rep_*.csv"))]).n_features.unique()
    assert list(nk) == [634 - len(allhit)], (nk, len(allhit))
    res["checks"] = {"refit_full_vs_stored_L_CNV_max_abs_pred": d0}
    res["removed"] = {"no3arms_total_removed": len(allhit), "arm_columns_removed": sum(i >= 589 for i in allhit), "tiles_5mb_removed": sum(i < 589 for i in allhit),
                      "per_arm": {a: {"arm_column": 1, "tiles_5mb": sum(i < 589 for i in v)} for a, v in hit.items()}, "no3arms_features_kept": 634 - len(allhit), "arms_cx_no3_features_kept": 42}
    res["score"] = score(O, ["L-CNV"])
    res["nonzero_per_fit"] = {s_: {"median": float(np.median(v)), "min": int(v.min()), "max": int(v.max())} for s_, v in NZ.items()}
    res["sets"] = {"abl_no3arms": "shipped matrix minus every feature overlapping hg19 arms 4q, 7p, 17p (arm columns and overlapping 5-Mb tiles); cx kept",
                   "abl_arms_cx_no3": "44 arms + cx minus the 4q, 7p and 17p arms (42 features)"}
    res["note"] = ("same stored 10 x 10 outer folds and L-CNV recipe (glmnet alpha 0.9, class-error lambda); abl_full refit equals stored L-CNV; fsa over the 571 pre-event samples; "
                   "2,000-draw patient bootstrap, paired deltas against L-CNV on the same draws (same draws as score_A).")
    ddump(res)
