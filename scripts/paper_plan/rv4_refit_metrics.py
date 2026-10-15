"""Review round 4, Part B metrics (docs/paper_review_round4_analyses.md @ 45b77bf). TASK = check | r36 | b5.
tb_common.py is imported unmodified: refit predictions (rv4_refit.R, killcoyne_mm/review_round4/) are loaded into the same (O, I, FO) structure as
tb_common.load() and scored with tb_common.triage() / comparators(); A*, B*, C* re-matched to the refit's own training specificity (rv_round1.py R1,
re-computed from the refit's inner predictions). Scoring helpers (fsa, matched, strategies, auc_block, triage_block) are copied from rv2_refit_metrics.py.
Single model = per-repeat calls, each statistic averaged over the 10 repeats ("no_vote"); voted = >= 6-of-10 calls; CIs from the patient bootstrap
(2,000 draws, RandomState(0), folds kept). Aggregates -> results/paper_final/review_round4/<TASK>.json."""
import os, sys, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tb_common import *
from tb_common import _outer, _inner
R4 = M + "/review_round4"; RS = R4 + "/reseg"; RR2 = M + "/review_round2"; OUT4 = T + "/results/paper_final/review_round4"; os.makedirs(OUT4, exist_ok=True); TASK = os.environ["TASK"]
PRESPEC = "45b77bf"
def fsa(S, b, FO, mask=None):
    """fold-stratified AUROC: within-fold pooled pair concordance per repeat, mean over repeats; S n x 10 or n (single model scored on every repeat's folds)"""
    out = []
    for j in range(10):
        x = (S[:, j] if np.ndim(S) == 2 else S)[b]; f = FO[b, j]; l = y[b]; ok = np.ones(len(b), bool) if mask is None else mask[b]; num = den = 0.0
        for k in range(1, 11):
            m = ok & (f == k) & np.isfinite(x); yk = l[m]; n1 = int(yk.sum()); n0 = len(yk) - n1
            if n1 == 0 or n0 == 0: continue
            rk = rankdata(x[m]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
def ci2(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return pct(a)
def lo1(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return r3(np.percentile(a, 5))
def dump(res): res["prespec"] = PRESPEC; json.dump(res, open(f"{OUT4}/{TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV4 DONE", TASK, flush=True)
def need(files, k): assert len(files) == k, (len(files), k, files[:2]); return files
FO0 = load("their")[2]
def _outer_col(files, col):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files]); F = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int); assert (F == FO0).all()
    return D.pivot_table(index="Sample", columns="rep", values=col).reindex(ids)[REPS].values
def _inner_col(files, col):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files]); assert D.groupby(["rep", "fold"]).ngroups == 100
    return {(r, k): pd.Series(d[col].values, index=ix[d.Sample].values) for (r, k), d in D.groupby(["rep", "fold"])}
def assemble(Oc, Oi, Ic, Ii):
    """same structure as tb_common.load(): O[model] n x 10 outer predictions, I[model][(rep, fold)] inner OOF predictions indexed by sample row"""
    O = {"L-CNV": Oc, "L-IMG": Oi}; O["L-LATE"] = (Oc + Oi) / 2; I = {"L-CNV": Ic, "L-IMG": Ii}
    I["L-LATE"] = {k: (Ic[k] + Ii[k].reindex(Ic[k].index)) / 2 for k in Ic}
    for k in Ic: assert Ii[k].index.sort_values().equals(Ic[k].index.sort_values()), k
    return O, I, FO0
def matched(O, I, FO, popm):
    """rv_round1.py R1: A*, B*, C* at the triage's training specificity s* (inner predictions), per repeat x fold"""
    MM = {"A*": "L-CNV", "B*": "L-LATE", "C*": "L-IMG"}; mc = {a: np.zeros((n, 10), bool) for a in MM}; sstar = []
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            si = I["L-IMG"][(r, k)]; tri = si.index.values[popm[si.index.values]]; s_tr = si.reindex(tri).values; yc = y[tri] == 1
            tl = thr_sens(s_tr[yc], 0.95); cs = c_star(s_tr[~yc], 0.90); li = I["L-LATE"][(r, k)].reindex(tri).values
            mid = (s_tr >= tl) & (s_tr <= cs) if tl <= cs else np.zeros(len(tri), bool)
            cm = li[mid & yc]; ts = thr_sens(cm, 0.8) if len(cm) >= 5 else thr_sens(li[yc], 0.8)
            tcall = np.where(s_tr < tl, False, np.where(s_tr > cs, True, li >= ts)) if tl <= cs else s_tr > cs
            s_star = 1 - tcall[~yc].mean(); sstar.append(s_star); te = FO[:, j] == k
            for a, mn in MM.items():
                xc = np.sort(I[mn][(r, k)].reindex(tri).values[~yc]); q = int(np.ceil(s_star * len(xc))); thr = xc[q - 1] if q > 0 else -np.inf
                mc[a][te, j] = O[mn][te, j] > thr
    return mc, sstar
def ss_v(c, b): return sens_spec(vote(c), b)
def ss_nv(c, b): yb = y[b]; cb = c[b]; return float(cb[yb == 1].mean(0).mean()), float(1 - cb[yb == 0].mean(0).mean())
def strategies(CALLS, rows, boots):
    out = {}
    for mode, fn in (("voted", ss_v), ("no_vote", ss_nv)):
        obs = {s: fn(c, rows) for s, c in CALLS.items()}; BS = {s: np.array([fn(c, b) for b in boots]) for s, c in CALLS.items()}
        o = {s: {"sensitivity": r3(obs[s][0]), "sens_ci95": ci2(BS[s][:, 0]), "specificity": r3(obs[s][1]), "spec_ci95": ci2(BS[s][:, 1])} for s in CALLS}
        for ref in CALLS:
            if ref == "triage": continue
            ds = BS["triage"][:, 0] - BS[ref][:, 0]; dp = BS["triage"][:, 1] - BS[ref][:, 1]
            o["triage"][f"vs_{ref}"] = {"d_sens": {"delta": r3(obs["triage"][0] - obs[ref][0]), "ci95": ci2(ds), "one_sided_95_lower": lo1(ds), "p_two_sided": p2(ds)},
                                        "d_spec": {"delta": r3(obs["triage"][1] - obs[ref][1]), "ci95": ci2(dp), "one_sided_95_lower": lo1(dp), "p_two_sided": p2(dp)}}
        out[mode] = o
    return out
def auc_block(O, FO, rows, boots, models=("L-CNV", "L-IMG", "L-LATE")):
    obs = {m: fsa(O[m], rows, FO) for m in models}; B = {m: np.array([fsa(O[m], b, FO) for b in boots]) for m in models}; o = {"auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in models}}
    if "L-IMG" in models and "L-CNV" in models:
        d = B["L-IMG"] - B["L-CNV"]; o["IMG_minus_CNV"] = {"delta": r3(obs["L-IMG"] - obs["L-CNV"]), "ci95": ci2(d), "one_sided_95_lower": lo1(d), "margin": -0.05, "non_inferior": bool(np.nanpercentile(d, 5) > -0.05), "p_two_sided": p2(d)}
    if "L-LATE" in models and "L-CNV" in models:
        d = B["L-LATE"] - B["L-CNV"]; o["LATE_minus_CNV"] = {"delta": r3(obs["L-LATE"] - obs["L-CNV"]), "ci95": ci2(d), "p_two_sided": p2(d)}
    return o
def triage_block(O, I, FO, pop, auc=True):
    popm = popmask(pop); rows = np.where(popm)[0]; boots = boots_for(rows)
    band, call, info = triage(O, I, FO, popm); comp, _ = comparators(O, I, FO, popm); mc, sstar = matched(O, I, FO, popm); CALLS = {"triage": call, **comp, **mc}
    res = {"n_samples": int(len(rows)), "n_patients": int(len(np.unique(pat[rows]))), "n_progressor_samples": int(y[rows].sum()),
           "share_sequenced": {"per_repeat_mean": r3((band[rows] == 1).mean()), "ci95": ci2([(band[b] == 1).mean() for b in boots]), "in_at_least_one_repeat": r3((band[rows] == 1).any(1).mean())},
           "folds_no_middle_band": info["no_mid"], "t_seq_fallback": info["fallback"], "cutoffs": {k: {"median": r3(np.median(v)), "range": [r3(min(v)), r3(max(v))]} for k, v in info.items() if isinstance(v, list)},
           "training_specificity_target": {"median": r3(np.median(sstar)), "range": [r3(min(sstar)), r3(max(sstar))]}, "strategies": strategies(CALLS, rows, boots)}
    if auc: res["discrimination"] = auc_block(O, FO, rows, boots)
    sh = res["share_sequenced"]["per_repeat_mean"]
    for mode in ("voted", "no_vote"):
        for ref in ("B", "B*", "C*"):
            v = res["strategies"][mode]["triage"][f"vs_{ref}"]["d_sens"]
            res[f"criterion_vs_{ref}_{mode}"] = {"delta_sens": v["delta"], "ci95": v["ci95"], "one_sided_95_lower": v["one_sided_95_lower"], "noninferior_sens": bool(v["one_sided_95_lower"] > -0.05),
                                                 "delta_spec": res["strategies"][mode]["triage"][f"vs_{ref}"]["d_spec"]["delta"], "share_sequenced": sh, "cap_70_met": bool(sh < 0.70), "met": bool(v["one_sided_95_lower"] > -0.05 and sh < 0.70)}
    return res
# ------------------------------------------------------------------ reproduction and feature-difference checks
def cmp_preds(a, b, col_a="pred", col_b="pred"):
    """rv2_refit_metrics.py check: same rows, same folds, same lambda, same predictions"""
    A = pd.read_csv(a, dtype={"Sample": str}); B = pd.read_csv(b, dtype={"Sample": str}); key = ["Sample"] + (["inner_fold"] if "inner_fold" in A and "inner_fold" in B else [])
    Mg = A.merge(B, on=key, suffixes=("_a", "_b")); assert len(Mg) == len(A) == len(B), (len(A), len(B), len(Mg))
    o = {"n": int(len(Mg)), "max_abs_pred_diff": float(np.abs(Mg[col_a + "_a"] - Mg[col_b + "_b"]).max()) if col_a == col_b else float(np.abs(Mg[col_a] - Mg[col_b]).max()),
         "folds_equal": bool((Mg.fold_a == Mg.fold_b).all()), "lambda_max_abs_diff": float(np.abs(Mg.lambda_a - Mg.lambda_b).max())}
    o["passed"] = bool(o["max_abs_pred_diff"] <= 1e-5 and o["folds_equal"] and o["lambda_max_abs_diff"] <= 1e-9); return o
def tile_cols(D): return [c for c in D.columns if ":" in c]
def joint_reproduction():
    """rv4_reseg.R MODE=joint: the stored joint segmentation re-run on one patient against the stored pkg_tiles.rds rows"""
    out = {}
    for f in sorted(glob.glob(f"{RS}/joint/pat_[0-9][0-9][0-9].csv")):
        A = pd.read_csv(f, dtype={"Sample": str}); B = pd.read_csv(f.replace(".csv", "_stored.csv"), dtype={"Sample": str}); tc = tile_cols(B); assert set(tc) <= set(A.columns), "tile columns differ"
        Mg = A.merge(B, on="Sample", suffixes=("_a", "_b")); d = np.abs(Mg[[c + "_a" for c in tc]].values - Mg[[c + "_b" for c in tc]].values)
        o = {"patient_index": int(A.pat_index.iloc[0]), "n_samples_segmented_together": int(A.n_segmented_together.iloc[0]), "n_samples_compared": int(len(Mg)), "n_stored_rows": int(len(B)),
             "n_tile_columns": len(tc), "max_abs_tile_diff": float(d.max()), "qc_pass_equal": bool((Mg.qc_pass_a == Mg.qc_pass_b).all()), "seconds": r3(A.secs.iloc[0])}
        o["passed"] = bool(len(Mg) == len(B) and o["max_abs_tile_diff"] <= 1e-8 and o["qc_pass_equal"]); out[os.path.basename(f)] = o
    assert out, "no joint re-run found"; out["all_passed"] = bool(all(v["passed"] for v in out.values() if isinstance(v, dict))); return out
def rowcorr(A, B): A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True); return (A * B).sum(1) / np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
def dist(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return {"mean": r3(a.mean()), "median": r3(np.median(a)), "iqr": [r3(np.percentile(a, 25)), r3(np.percentile(a, 75))], "min": r3(a.min()), "max": r3(a.max())}
def feature_differences(J, P, XJ, XP, label):
    """J, P: raw tiles tables (Sample, qc_pass, cx_raw, 5-Mb tiles, arms) of the stored joint and the re-segmented pipeline; XJ, XP the final feature blocks"""
    J = J.set_index("Sample").reindex(ids); P = P.set_index("Sample").reindex(ids); assert P.notna().all().all(), "missing re-segmented samples"
    tc = tile_cols(J); assert tc == tile_cols(P) and len(tc) == 587 + 44, len(tc); five = tc[:-44]; arm = tc[-44:]   # rv4_reseg.R tiles_table: 5-Mb tiles then the 44 arms
    o = {"what": label, "n_samples": int(len(ids)), "n_5mb_tiles": len(five), "n_arms": len(arm), "n_block_features": int(XJ.shape[1]),
         "per_sample_pearson_r": {"raw_5mb_tiles": dist(rowcorr(P[five].values, J[five].values)), "raw_arm_tiles": dist(rowcorr(P[arm].values, J[arm].values)), "final_feature_block": dist(rowcorr(XP, XJ))},
         "per_feature_pearson_r_across_samples": {"raw_5mb_tiles": dist(rowcorr(P[five].values.T, J[five].values.T)), "final_feature_block": dist(rowcorr(XP.T, XJ.T))},
         "max_abs_diff_raw_5mb_tiles": r3(np.abs(P[five].values - J[five].values).max()), "share_samples_with_identical_raw_tiles": r3((np.abs(P[five].values - J[five].values).max(1) <= 1e-8).mean())}
    cj = J.cx_raw.values; cp = P.cx_raw.values; d = cp - cj
    o["cx_raw"] = {"joint": dist(cj), "resegmented": dist(cp), "change_resegmented_minus_joint": dist(d), "pearson_r": r3(np.corrcoef(cj, cp)[0, 1]), "spearman_rho": r3(spearmanr(cj, cp).correlation),
                   "share_unchanged": r3((d == 0).mean()), "share_higher": r3((d > 0).mean()), "share_lower": r3((d < 0).mean()), "note": "cx = tiles beyond 2 s.d. within the sample's standardised 5-Mb tiles, before the final standardisation"}
    o["qc_pass"] = {"joint": int(J.qc_pass.sum()), "resegmented": int(P.qc_pass.sum()), "both": int((J.qc_pass & P.qc_pass).sum())}
    o["block_cx_feature_pearson_r"] = r3(np.corrcoef(XJ[:, -1], XP[:, -1])[0, 1]); return o
def read_tiles(f): return pd.read_csv(f, dtype={"Sample": str})
def read_block(f): X = pd.read_csv(f, dtype={"sample_id": str}).set_index("sample_id").reindex(ids); return X
def refit_outer_inner(cfg):
    return (need(sorted(glob.glob(f"{R4}/r36_outer/{cfg}_rep_*.csv")), 10), need(sorted(glob.glob(f"{R4}/r36_inner/{cfg}_rep_*_fold_*.csv")), 100))
def stored_ref(src, pop):
    try:
        fx = json.load(open(f"{T}/results/paper_final/triage/{src}_{pop}.json")); r1 = json.load(open(f"{T}/results/paper_final/review_round1/r1_{src}_{pop}.json"))["R1"]
        return {"auroc": {m: fx["section1"]["auroc"][m]["value"] for m in ("L-CNV", "L-IMG", "L-LATE")}, "img_vs_cnv": fx["section1"]["img_vs_cnv"], "triage_vs_B_sens_voted": fx["strategies"]["triage_LATE"]["vs_B"]["sens"],
                "matched_voted": {a: r1["matched"][a]["d_sens_triage_minus"] for a in ("A*", "B*", "C*")}}
    except FileNotFoundError: return None
res = {"task": TASK}
# ------------------------------------------------------------------ smoke-stage checks (joint re-run, per-sample re-run on the same patient, refit reproductions)
if TASK == "check":
    out = {"joint_segmentation_reproduction": joint_reproduction()}
    ps = sorted(glob.glob(f"{RS}/persample/pat_[0-9][0-9][0-9].csv"))
    if ps:
        A = read_tiles(ps[0]); f = f"{RS}/joint/{os.path.basename(ps[0]).replace('.csv', '_stored.csv')}"
        if os.path.exists(f):
            B = read_tiles(f); tc = [c for c in tile_cols(B) if c in A.columns]; Mg = A.merge(B, on="Sample", suffixes=("_a", "_b"))
            rc = rowcorr(Mg[[c + "_a" for c in tc]].values, Mg[[c + "_b" for c in tc]].values)
            out["persample_vs_joint_same_patient"] = {"n_samples": int(len(Mg)), "per_sample_pearson_r_raw_tiles": dist(rc), "max_abs_tile_diff": r3(np.abs(Mg[[c + "_a" for c in tc]].values - Mg[[c + "_b" for c in tc]].values).max()),
                                                     "qc_pass_joint": int(Mg.qc_pass_b.sum()), "qc_pass_persample": int(Mg.qc_pass_a.sum()), "seconds": r3(A.secs.iloc[0]), "n_segmented_together": int(A.n_segmented_together.iloc[0])}
    pa = sorted(glob.glob(f"{RS}/past/pat_[0-9][0-9][0-9].csv"))
    if pa: A = read_tiles(pa[0]); out["past_smoke"] = {"n_samples": int(len(A)), "n_segmented_together": sorted(A.n_segmented_together.astype(int).tolist()), "seconds": r3(A.secs.iloc[0])}
    f = f"{R4}/r36_outer/cnv_pkg_rep_01.csv"
    if os.path.exists(f): out["outer_cnv_pkg_rep_01_vs_stored_cfg_01_rep_01"] = cmp_preds(f, f"{M}/cv/preds/cfg_01_rep_01.csv")
    for cfg in ("cnv_their", "img"):
        fs = sorted(glob.glob(f"{R4}/b5_inner/{cfg}_rep_*_fold_*.csv"))
        if fs: out[f"b5_inner_{cfg}_class_lambda_vs_stored_inner"] = {"files": len(fs), "worst": max((cmp_preds(g, f"{HZ}/inner/{os.path.basename(g)}") for g in fs), key=lambda o: o["max_abs_pred_diff"])}
    flags = [out["joint_segmentation_reproduction"]["all_passed"]] + [v["passed"] for k, v in out.items() if k.startswith("outer_")] + [v["worst"]["passed"] for k, v in out.items() if k.startswith("b5_inner_")]
    out["all_passed"] = bool(all(flags)); res["checks"] = out; dump(res)
# ------------------------------------------------------------------ R36 per-sample segmentation
elif TASK == "r36":
    out = {"checks": {"joint_segmentation_reproduction": joint_reproduction(), "refit_outer_cnv_pkg_rep_01_vs_stored_cfg_01_rep_01": cmp_preds(f"{R4}/r36_outer/cnv_pkg_rep_01.csv", f"{M}/cv/preds/cfg_01_rep_01.csv")}}
    ai = pd.read_csv(f"{RS}/assemble_info.csv", dtype=str).set_index("key").value; out["checks"]["stored_block_reconstruction_max_abs_diff"] = float(ai["stored_block_reconstruction_max_abs_diff"])
    out["checks"]["all_passed"] = bool(out["checks"]["joint_segmentation_reproduction"]["all_passed"] and out["checks"]["refit_outer_cnv_pkg_rep_01_vs_stored_cfg_01_rep_01"]["passed"] and out["checks"]["stored_block_reconstruction_max_abs_diff"] < 1e-8)
    J = read_tiles(f"{RS}/tiles_joint_stored_analysed.csv"); XJ = read_block(f"{M}/cnv_pkg_C.csv"); cols = list(XJ.columns)
    variants = {"persample": ("pkgps", "each sample segmented on its own (segmentRawData single-sample branch)"), "past": ("pkgpast", "each sample segmented jointly with its patient's samples from the same or earlier endoscopies")}
    O_img = _outer(sorted(glob.glob(f"{M}/cv/preds/cfg_02_rep_*.csv")))[0]; I_img = _inner(sorted(glob.glob(f"{HZ}/inner/img_rep_*_fold_*.csv")))
    for mode, (tag, what) in variants.items():
        bf = f"{R4}/r36_cnv_{tag}_C.csv"
        if not os.path.exists(bf) or len(glob.glob(f"{R4}/r36_outer/cnv_{tag}_rep_*.csv")) < 10 or len(glob.glob(f"{R4}/r36_inner/cnv_{tag}_rep_*_fold_*.csv")) < 100:
            out[mode] = {"status": "not run", "what": what, "block_exists": os.path.exists(bf), "outer_files": len(glob.glob(f"{R4}/r36_outer/cnv_{tag}_rep_*.csv")), "inner_files": len(glob.glob(f"{R4}/r36_inner/cnv_{tag}_rep_*_fold_*.csv"))}; continue
        XP = read_block(bf); assert list(XP.columns) == cols and XP.notna().all().all(); P = read_tiles(f"{RS}/tiles_{mode}_analysed.csv")
        o = {"status": "complete", "what": what, "feature_differences_vs_joint": feature_differences(J, P, XJ.values, XP.values, f"{mode} vs stored joint (per patient) package features, 676 analysed samples"),
             "seconds_segmentation_total": float(ai.get(f"{mode}.secs_total", np.nan)), "mean_samples_segmented_together": float(ai.get(f"{mode}.mean_n_segmented_together", np.nan))}
        of, inf = refit_outer_inner(f"cnv_{tag}"); O, I, FO = assemble(_outer(of)[0], O_img, _inner(inf), I_img)
        for pop in ("pre", "pre_ndbe"): o[pop] = triage_block(O, I, FO, pop)
        out[mode] = o
    Oj, Ij, FOj = load("pkg"); out["joint_package_reference"] = {"what": "stored package features (joint per-patient segmentation), stored outer and inner predictions, scored the same way"}
    for pop in ("pre", "pre_ndbe"): out["joint_package_reference"][pop] = triage_block(Oj, Ij, FOj, pop); out["joint_package_reference"][pop]["stored_reference"] = stored_ref("pkg", pop)
    out["note"] = "L-IMG and the gate are the stored predictions (UNI2-h slide mean); L-LATE = mean of the refit L-CNV and stored L-IMG; cut-offs and A*/B*/C* matching re-chosen on the refit inner predictions"
    res["R36"] = out; dump(res)
# ------------------------------------------------------------------ B5 inner predictions with lambda by deviance
elif TASK == "b5":
    out = {"reproduction_class_lambda_vs_stored_inner": {}, "lambda_rules": {}}
    files = {cfg: need(sorted(glob.glob(f"{R4}/b5_inner/{cfg}_rep_*_fold_*.csv")), 100) for cfg in ("cnv_their", "img")}
    for cfg, fs in files.items():
        cm = [cmp_preds(g, f"{HZ}/inner/{os.path.basename(g)}") for g in fs]
        out["reproduction_class_lambda_vs_stored_inner"][cfg] = {"files": len(fs), "max_abs_pred_diff": max(c["max_abs_pred_diff"] for c in cm), "lambda_max_abs_diff": max(c["lambda_max_abs_diff"] for c in cm), "all_passed": bool(all(c["passed"] for c in cm))}
        D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fs]); fm = D.drop_duplicates(["rep", "fold", "inner_fold"]); assert len(fm) == 500
        out["lambda_rules"][cfg] = {"inner_fold_models": int(len(fm)), "share_with_no_selected_feature": {"class_error_lambda": r3((fm.nnz_class == 0).mean()), "deviance_lambda": r3((fm.nnz_dev == 0).mean())},
                                   "n_nonzero_median": {"class_error_lambda": r3(fm.nnz_class.median()), "deviance_lambda": r3(fm.nnz_dev.median())}, "lambda_median": {"class_error": r3(fm["lambda"].median()), "deviance": r3(fm.lambda_dev.median())},
                                   "deviance_lambda_smaller_share": r3((fm.lambda_dev < fm["lambda"]).mean()), "share_identical_lambda": r3((fm.lambda_dev == fm["lambda"]).mean())}
    out["reproduction_class_lambda_vs_stored_inner"]["all_passed"] = bool(all(v["all_passed"] for v in out["reproduction_class_lambda_vs_stored_inner"].values() if isinstance(v, dict)))
    f4 = {cfg: need(sorted(glob.glob(f"{RR2}/f4/{cfg}_rep_*.csv")), 10) for cfg in ("cnv_their", "img")}
    O, I, FO = assemble(_outer_col(f4["cnv_their"], "pred_dev"), _outer_col(f4["img"], "pred_dev"), _inner_col(files["cnv_their"], "pred_dev"), _inner_col(files["img"], "pred_dev"))
    Os, Is, FOs = load("their"); out["deviance_lambda"] = {"what": "outer predictions at the deviance-minimum lambda (round-2 F4, 10 x 5 inner curves) and inner predictions at the deviance-minimum lambda (B5, 1 x 5 inner curve) for L-CNV (their matrix) and L-IMG; L-LATE = their mean; gate, middle band, (B), B* and C* all from these predictions"}
    out["class_error_lambda_stored"] = {"what": "stored predictions (class-error lambda), scored the same way"}
    for pop in ("pre", "pre_ndbe"):
        out["deviance_lambda"][pop] = triage_block(O, I, FO, pop); out["class_error_lambda_stored"][pop] = triage_block(Os, Is, FOs, pop); out["class_error_lambda_stored"][pop]["stored_reference"] = stored_ref("their", pop)
    res["B5"] = out; dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
