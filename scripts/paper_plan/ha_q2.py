"""Horizon answers Q2 (docs/paper_horizon_answers.md @ 81473df): horizon-specific operating point (80% sensitivity among that horizon's training cases on
the inner out-of-fold predictions of each outer training fold, applied to the held-out fold), majority call over 10 repeats, FPR/TPR, reclassification,
categorical NRI for L-CNV vs the Q1 best arm (their matrix, all pre-event samples, from q0_their_pre.json) and vs L-LATE if different; captured and cleared
groups with descriptors. Aggregates -> results/paper_final/horizon_answers/q2.json; per-patient list (study numbers) cluster-only:
feasibility/paper_plan/killcoyne_mm/horizon_answers/q2_patient_list.csv. Bootstrap/NRI code adapted from hz_stack.py TASK=h2."""
import json, os, glob, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"; HA = M + "/horizon_answers"; OUT = T + "/results/paper_final/horizon_answers"
HORIZONS = [1.0, 3.0, 5.0]; NB = 2000; REPS = range(1, 11)
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM); ix = pd.Series(np.arange(n), index=SM.Sample)
time = SM.time.values.astype(float); ev = SM.event.values.astype(int); pat = SM.Patient.values; pre = SM.pre.astype(bool).values; ndbe = SM.ndbe.astype(bool).values
def labels(tt, dd, t): return (dd == 1) & (tt <= t), ((dd == 0) & (tt >= t)) | ((dd == 1) & (tt > t))
CASE = {t: labels(time, ev, t)[0] & pre for t in HORIZONS}
def pctl(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return [r3(np.percentile(a, 2.5)), r3(np.percentile(a, 97.5))] if len(a) else [None, None]
def outer(files, model=None):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files])
    if model is not None: D = D[D.model == model]
    return D.pivot_table(index="Sample", columns="rep", values="pred").reindex(SM.Sample).values, D.pivot_table(index="Sample", columns="rep", values="fold").reindex(SM.Sample).values.astype(int)
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv"))
def inner(files, model=None):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files])
    if model is not None: D = D[D.model == model]
    assert D.groupby(["rep", "fold"]).ngroups == 100; return {(r, k): pd.Series(d.pred.values, index=ix[d.Sample].values) for (r, k), d in D.groupby(["rep", "fold"])}
def arms(src):
    O, I = {}, {}
    O["L-CNV"], FO = outer(kvf(0 if src == "their" else 1)); O["L-IMG"], _ = outer(kvf(2)); O["L-EARLY"], _ = outer(kvf(3 if src == "their" else 4))
    O["L-INTER"], _ = outer(sorted(glob.glob(f"{HZ}/outer/inter_{src}_rep_*.csv"))); O["L-CLIN"], _ = outer([HA + "/clin_outer.csv"], "clin"); O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2
    I["L-CNV"] = inner(sorted(glob.glob(f"{HZ}/inner/cnv_{src}_rep_*_fold_*.csv"))); I["L-IMG"] = inner(sorted(glob.glob(f"{HZ}/inner/img_rep_*_fold_*.csv")))
    I["L-EARLY"] = inner(sorted(glob.glob(f"{HZ}/inner/early_{src}_rep_*_fold_*.csv"))); I["L-INTER"] = inner(sorted(glob.glob(f"{HZ}/inner/inter_{src}_rep_*_fold_*.csv"))); I["L-CLIN"] = inner([HA + "/clin_inner.csv"], "clin")
    I["L-LATE"] = {k: (I["L-CNV"][k] + I["L-IMG"][k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    return O, I, FO
q0 = json.load(open(f"{OUT}/q0_their_pre.json"))["units"]["sample"]["horizons"]; BEST = {t: q0[str(int(t))]["best"] for t in HORIZONS}
slide = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(SM.Sample)
qc = pd.read_csv(HZ + "/pkg_qc.csv", dtype={"Sample": str}).drop_duplicates("Sample").set_index("Sample").reindex(SM.Sample)
cxp = pd.read_csv(M + "/cnv_pkg_C.csv", usecols=["sample_id", "cx"], dtype={"sample_id": str}).set_index("sample_id").cx.reindex(SM.Sample).values
scl = pd.read_csv(T + "/models/killcoyne_frozen_pkg_v1/cnv_scaling_setC.csv"); cxr = scl[scl.kind == "cx"].iloc[0]; cx_raw = cxp * cxr["sd"] + cxr["mean"]
X0 = pd.read_excel("/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper/41591_2020_1033_MOESM4_ESM.xlsx", sheet_name="Supporting data for Figure 2a", header=1)
rc = X0.set_index(X0.Samplename.astype(str))["Risk class"].reindex(SM.Sample).values
res = {"best_by_horizon_their_pre": {str(int(t)): b for t, b in BEST.items()}, "risk_class_matched": int(pd.notna(rc).sum()), "results": {}}; plist = []
for src in ("their", "pkg"):
    O, I, FO = arms(src); R = {}
    for t in HORIZONS:
        comp = sorted(set([BEST[t], "L-LATE"]) - {"L-CNV"}); need = ["L-CNV"] + comp; calls, thr = {}, {}
        for a in need:
            c = np.zeros((n, 10), bool); th = []
            for j, r in enumerate(REPS):
                for k in range(1, 11):
                    s = I[a][(r, k)]; cs = np.sort(s.values[CASE[t][s.index]])[::-1]
                    tau = cs[int(np.ceil(0.8 * len(cs))) - 1] if len(cs) else np.inf; th.append(tau); te = FO[:, j] == k; c[te, j] = O[a][te, j] >= tau
            calls[a] = c.sum(1) >= 6; thr[a] = {"median": r3(np.median(th)), "range": [r3(min(th)), r3(max(th))], "n": len(th)}
        for unit in ("sample", "patient"):
            if unit == "sample": rows = np.where(pre)[0]
            else:
                r_ = np.where(pre & ndbe)[0]; d = pd.DataFrame({"r": r_, "p": pat[r_], "mbf": SM.mbf.values[r_]}).sort_values(["p", "mbf", "r"], ascending=[True, False, True]); rows = np.sort(d.groupby("p").r.first().values)
            pp = pat[rows]; up = np.unique(pp); ro = {q: np.where(pp == q)[0] for q in up}; rng = np.random.RandomState(0); boots = [np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))]) for _ in range(NB)]
            case, ctrl = labels(time[rows], ev[rows], t); o = {"n_cases": int(case.sum()), "n_controls": int(ctrl.sum()), "n_case_patients": int(len(set(pp[case]))), "n_control_patients": int(len(set(pp[ctrl]))), "thresholds": thr}
            def rates(v, cs, ct): return (v[cs].mean() if cs.sum() else np.nan), (v[ct].mean() if ct.sum() else np.nan)
            def nri(cs, ct, x0, x1):
                u, d_ = x1 & ~x0, x0 & ~x1; return (u[cs].mean() - d_[cs].mean() if cs.sum() else np.nan) + (d_[ct].mean() - u[ct].mean() if ct.sum() else np.nan)
            for a in need: tp, fp = rates(calls[a][rows], case, ctrl); o[a] = {"TPR": r3(tp), "FPR": r3(fp)}
            for f_ in comp:
                c0, c1 = calls["L-CNV"][rows], calls[f_][rows]; tp0, fp0 = rates(c0, case, ctrl); tp1, fp1 = rates(c1, case, ctrl)
                bd = []
                for b in boots:
                    cs, ct = labels(time[rows][b], ev[rows][b], t); a0 = rates(c0[b], cs, ct); a1 = rates(c1[b], cs, ct); bd.append([a1[1] - a0[1], a1[0] - a0[0], nri(cs, ct, c0[b], c1[b])])
                bd = np.array(bd); tab = {}
                for g, m in (("cases", case), ("controls", ctrl)):
                    tab[g] = {"CNVneg_bestpos": int((~c0 & c1)[m].sum()), "CNVpos_bestneg": int((c0 & ~c1)[m].sum()), "both_pos": int((c0 & c1)[m].sum()), "both_neg": int((~c0 & ~c1)[m].sum())}
                    if unit == "sample":
                        pr = pd.Series(pp)
                        tab[g + "_patients"] = {k: int(len(set(pr[(msk & m)]))) for k, msk in (("CNVneg_bestpos", ~c0 & c1), ("CNVpos_bestneg", c0 & ~c1), ("both_pos", c0 & c1), ("both_neg", ~c0 & ~c1))}
                o[f"{f_}_vs_L-CNV"] = {"dFPR": r3(fp1 - fp0), "dFPR_ci95": pctl(bd[:, 0]), "dTPR": r3(tp1 - tp0), "dTPR_ci95": pctl(bd[:, 1]), "NRI": r3(nri(case, ctrl, c0, c1)), "NRI_ci95": pctl(bd[:, 2]), "reclassification": tab}
                if unit == "sample":
                    grp = {"captured_cases_CNVneg_bestpos": rows[case & ~c0 & c1], "cases_both_pos": rows[case & c0 & c1], "cleared_controls_CNVpos_bestneg": rows[ctrl & c0 & ~c1], "controls_both_pos": rows[ctrl & c0 & c1]}
                    Dd = {}
                    for g, gi in grp.items():
                        med = lambda v: r3(np.nanmedian(np.asarray(v, float))) if len(gi) else None
                        Dd[g] = {"n_samples": int(len(gi)), "n_patients": int(len(set(pat[gi]))), "pathology": pd.Series(SM.Pathology.values[gi]).value_counts().to_dict(), "time_median_years": med(time[gi]),
                                 "cx_raw_pkg_median": med(cx_raw[gi]), "noise_varMAD_median": med(qc.varMAD_median.values[gi]), "tissue_tiles_median": med(slide.tissue_tiles.values[gi]),
                                 "scanner": pd.Series(slide.scanner.values[gi]).value_counts().to_dict(), "p53_ihc": pd.Series(SM["P53 IHC"].astype(str).values[gi]).value_counts().to_dict(), "killcoyne_risk_class_MOESM4": pd.Series(rc[gi]).astype(str).value_counts().to_dict()}
                        for i in gi: plist.append({"src": src, "horizon": int(t), "comparison": f"{f_} vs L-CNV", "group": g, "study_number": SM.study_number.values[i], "Sample": SM.Sample.values[i], "Pathology": SM.Pathology.values[i],
                                                   "time_years": round(time[i], 2), "event": int(ev[i]), "cx_raw_pkg": round(float(cx_raw[i]), 1), "noise_varMAD_median": qc.varMAD_median.values[i], "tissue_tiles": slide.tissue_tiles.values[i],
                                                   "scanner": slide.scanner.values[i], "p53_ihc": SM["P53 IHC"].values[i], "killcoyne_risk_class_MOESM4": rc[i]})
                    o[f"{f_}_groups"] = Dd
            R[f"{int(t)}|{unit}"] = o
        print("q2", src, t, BEST[t], flush=True)
    res["results"][src] = R
os.makedirs(HA, exist_ok=True); pd.DataFrame(plist).to_csv(HA + "/q2_patient_list.csv", index=False); res["patient_list_rows"] = len(plist); res["patient_list_path_cluster"] = HA + "/q2_patient_list.csv"
json.dump(res, open(f"{OUT}/q2.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("HA Q2 DONE")
