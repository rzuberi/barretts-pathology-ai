"""Horizons H2 and H3 (docs/paper_survival_horizons.md @ ee51db8). TASK = h2 | h3_<src> (src their|pkg).
H2: operating points (a) 80% sensitivity on the training samples' inner OOF predictions (hz_fit.R inner), applied to the outer held-out predictions;
(b) Pr >= 0.5. Majority call over the 10 repeats. FPR/TPR, reclassification, categorical NRI vs L-CNV, for the H1-best model and L-LATE.
H3: per (repeat, outer fold, horizon) logistic stack label_t ~ a + b*z(logit CNV) + c*z(logit IMG) on the inner OOF predictions of the outer training
pre-event cases/controls; patient-bootstrap refits; held-out stack AUROC vs L-LATE (exploratory). Aggregates -> results/paper_final/horizons/<TASK>.json;
row-level patient list (H2) on the cluster only."""
import json, os, glob, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"; OUT = T + "/results/paper_final/horizons"
TASK = os.environ["TASK"]; HORIZONS = [1.0, 3.0, 5.0]; NB = 2000; REPS = range(1, 11)
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM); ix = pd.Series(np.arange(n), index=SM.Sample)
time = SM.time.values.astype(float); ev = SM.event.values.astype(int); pat = SM.Patient.values; yv = SM.y.values.astype(int)
pre = SM.pre.astype(bool).values; ndbe = SM.ndbe.astype(bool).values
def labels(tt, dd, t): return (dd == 1) & (tt <= t), ((dd == 0) & (tt >= t)) | ((dd == 1) & (tt > t))
def Gminus(tt, dd, at):
    u = np.unique(tt); nrisk = (tt[None, :] >= u[:, None]).sum(1); dc = ((tt[None, :] == u[:, None]) & (dd[None, :] == 0)).sum(1)
    surv = np.cumprod(1 - dc / nrisk); prev = np.concatenate([[1.0], surv[:-1]]); k = np.searchsorted(u, at, side="left"); out = np.ones(len(at)); ins = k < len(u)
    out[ins] = prev[k[ins]]; out[~ins] = surv[-1]; return np.clip(out, 1e-12, None)
def tdauc(sc, tt, dd, t):
    case, ctrl = labels(tt, dd, t)
    if case.sum() == 0 or ctrl.sum() == 0: return np.nan
    w = 1 / Gminus(tt, dd, tt[case]); cs = np.sort(sc[ctrl]); m = sc[case]; lo = np.searchsorted(cs, m, "left"); hi = np.searchsorted(cs, m, "right")
    return float((w * (lo + 0.5 * (hi - lo))).sum() / (w.sum() * len(cs)))
def pctl(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return [r3(np.percentile(a, 2.5)), r3(np.percentile(a, 97.5))] if len(a) else [None, None]
def boots_for(rows):
    pp = pat[rows]; up = np.unique(pp); ro = {q: np.where(pp == q)[0] for q in up}; rng = np.random.RandomState(0)
    return [np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))]) for _ in range(NB)], up
# ---------- outer (per repeat) and inner OOF predictions
def outer(files, model=None, cnvsrc=None):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files]); P = D.pivot_table(index="Sample", columns="rep", values="pred").reindex(SM.Sample)
    Fo = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(SM.Sample); return P.values, Fo.values.astype(int)
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv"))
def arms_outer(src):
    O = {}; O["L-CNV"], FO = outer(kvf(0 if src == "their" else 1)); O["L-IMG"], _ = outer(kvf(2)); O["L-EARLY"], _ = outer(kvf(3 if src == "their" else 4))
    O["L-INTER"], _ = outer(sorted(glob.glob(f"{HZ}/outer/inter_{src}_rep_*.csv"))); O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2; return O, FO
def inner(cfg):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(f"{HZ}/inner/{cfg}_rep_*_fold_*.csv"))]); assert len(D) and D.groupby(["rep", "fold"]).ngroups == 100, cfg
    return {(r, k): pd.Series(d.pred.values, index=ix[d.Sample].values) for (r, k), d in D.groupby(["rep", "fold"])}
def arms_inner(src):
    I = {"L-CNV": inner(f"cnv_{src}"), "L-IMG": inner("img"), "L-EARLY": inner(f"early_{src}"), "L-INTER": inner(f"inter_{src}")}
    I["L-LATE"] = {key: (I["L-CNV"][key] + I["L-IMG"][key].reindex(I["L-CNV"][key].index)) / 2 for key in I["L-CNV"]}; return I
res = {"task": TASK}
if TASK == "h2":
    h1 = json.load(open(f"{OUT}/h1_their_pre.json"))["units"]["sample"]["horizons"]
    cand = ["L-IMG", "L-EARLY", "L-INTER", "L-LATE"]; meanauc = {a: float(np.mean([h1[str(int(t))]["arms"][a]["ipcw"]["auroc"] for t in HORIZONS])) for a in cand}
    best = max(cand, key=lambda a: meanauc[a]); res["best_model"] = best; res["best_selection_mean_ipcw_auroc"] = {a: r3(v) for a, v in meanauc.items()}
    slide = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(SM.Sample)
    qc = pd.read_csv(HZ + "/pkg_qc.csv", dtype={"Sample": str}).drop_duplicates("Sample").set_index("Sample").reindex(SM.Sample)
    noise_col = next((c for c in ("varMAD_median", "varMAD", "MAD") if c in qc.columns), None) or next(c for c in qc.columns if "var" in c.lower() or "mad" in c.lower()); res["noise_column"] = noise_col
    cxp = pd.read_csv(M + "/cnv_pkg_C.csv", usecols=["sample_id", "cx"], dtype={"sample_id": str}).set_index("sample_id").cx.reindex(SM.Sample).values
    cxt = pd.read_csv(M + "/cnv_their_C.csv", usecols=["sample_id", "cx"], dtype={"sample_id": str}).set_index("sample_id").cx.reindex(SM.Sample).values
    sc = pd.read_csv(T + "/models/killcoyne_frozen_pkg_v1/cnv_scaling_setC.csv"); cxr = sc[sc.kind == "cx"].iloc[0]; cx_raw = cxp * cxr["sd"] + cxr["mean"]
    kcls = pd.cut(SM.k_prob.astype(float), [-np.inf, 0.3, 0.5, np.inf], right=False, labels=["low", "moderate", "high"]).astype(str).values
    res["results"] = {}; plist = []
    for src in ("their", "pkg"):
        O, FO = arms_outer(src); I = arms_inner(src); calls = {}
        for a in ("L-CNV", best, "L-LATE"):
            ca = np.zeros((n, 10), bool); cb = np.zeros((n, 10), bool); thr = []
            for j, r in enumerate(REPS):
                cb[:, j] = O[a][:, j] >= 0.5
                for k in range(1, 11):
                    s = I[a][(r, k)]; ps = np.sort(s.values[yv[s.index] == 1])[::-1]; tau = ps[int(np.ceil(0.8 * len(ps))) - 1]; thr.append(tau)
                    te = FO[:, j] == k; ca[te, j] = O[a][te, j] >= tau
            calls[a] = {"a_sens80": ca, "b_pr05": cb}; res.setdefault("thresholds_a", {})[f"{src}_{a}"] = {"median": r3(np.median(thr)), "range": [r3(min(thr)), r3(max(thr))], "n": len(thr)}
        R2 = {}
        for op in ("a_sens80", "b_pr05"):
            maj = {a: calls[a][op].sum(1) >= 6 for a in calls}; per_rep = {a: calls[a][op] for a in calls}
            for popname, pm in (("pre", pre), ("pre_ndbe", pre & ndbe)):
                for unit in ("sample", "patient"):
                    if unit == "sample": rows = np.where(pm)[0]
                    else:
                        r_ = np.where(pm & ndbe)[0]; d = pd.DataFrame({"r": r_, "p": pat[r_], "mbf": SM.mbf.values[r_]}).sort_values(["p", "mbf", "r"], ascending=[True, False, True]); rows = np.sort(d.groupby("p").r.first().values)
                    boots, _ = boots_for(rows)
                    for t in HORIZONS:
                        case, ctrl = labels(time[rows], ev[rows], t); key = f"{op}|{popname}|{unit}|{int(t)}"; o = {"n_cases": int(case.sum()), "n_controls": int(ctrl.sum())}
                        def rates(cv, b=None):
                            cs, ct = (case, ctrl) if b is None else labels(time[rows][b], ev[rows][b], t); v = cv[rows] if b is None else cv[rows][b]
                            return (v[cs].mean() if cs.sum() else np.nan), (v[ct].mean() if ct.sum() else np.nan)
                        for a in calls:
                            tp, fp = rates(maj[a]); o[a] = {"TPR": r3(tp), "FPR": r3(fp), "TPR_mean_over_repeats": r3(np.mean([rates(per_rep[a][:, j])[0] for j in range(10)])), "FPR_mean_over_repeats": r3(np.mean([rates(per_rep[a][:, j])[1] for j in range(10)]))}
                        for f_ in sorted(set([best, "L-LATE"])):
                            c0, c1 = maj["L-CNV"][rows], maj[f_][rows]
                            def nri(cs, ct, x0, x1):
                                up_, dn = x1 & ~x0, x0 & ~x1
                                return (up_[cs].mean() - dn[cs].mean() if cs.sum() else np.nan) + (dn[ct].mean() - up_[ct].mean() if ct.sum() else np.nan)
                            tp0, fp0 = rates(maj["L-CNV"]); tp1, fp1 = rates(maj[f_])
                            bd = np.array([[*np.subtract(rates(maj[f_], b), rates(maj["L-CNV"], b)), nri(*labels(time[rows][b], ev[rows][b], t), c0[b], c1[b])] for b in boots])
                            tab = {g: {"CNVneg_fusionpos": int((~c0 & c1)[m].sum()), "CNVpos_fusionneg": int((c0 & ~c1)[m].sum()), "both_pos": int((c0 & c1)[m].sum()), "both_neg": int((~c0 & ~c1)[m].sum())} for g, m in (("cases", case), ("controls", ctrl))}
                            o[f"{f_}_vs_L-CNV"] = {"dTPR": r3(tp1 - tp0), "dTPR_ci95": pctl(bd[:, 0]), "dFPR": r3(fp1 - fp0), "dFPR_ci95": pctl(bd[:, 1]), "NRI": r3(nri(case, ctrl, c0, c1)), "NRI_ci95": pctl(bd[:, 2]), "reclassification": tab}
                            if op == "a_sens80" and popname == "pre" and unit == "sample" and t == 3.0:   # who fusion catches / clears
                                rr = rows; grp = {}
                                grp["cases_CNVneg_fusionpos"] = rr[case & ~c0 & c1]; grp["cases_both_pos"] = rr[case & c0 & c1]
                                grp["controls_CNVpos_fusionneg"] = rr[ctrl & c0 & ~c1]; grp["controls_both_pos"] = rr[ctrl & c0 & c1]
                                D = {}
                                for g, gi in grp.items():
                                    D[g] = {"n_samples": int(len(gi)), "n_patients": int(len(set(pat[gi]))), "pathology": SM.Pathology.values[gi].tolist() and pd.Series(SM.Pathology.values[gi]).value_counts().to_dict(),
                                            "time_to_event_or_censor_median": r3(np.median(time[gi])) if len(gi) else None, "cx_raw_pkg_median": r3(np.median(cx_raw[gi])) if len(gi) else None,
                                            "cx_z_their_median": r3(np.median(cxt[gi])) if len(gi) else None, "noise_median": r3(np.nanmedian(qc[noise_col].values[gi].astype(float))) if len(gi) else None,
                                            "tissue_tiles_median": r3(np.nanmedian(slide.tissue_tiles.values[gi].astype(float))) if len(gi) else None, "scanner": pd.Series(slide.scanner.values[gi]).value_counts().to_dict(),
                                            "p53_ihc": pd.Series(SM["P53 IHC"].astype(str).values[gi]).value_counts().to_dict(), "killcoyne_risk_class": pd.Series(kcls[gi]).value_counts().to_dict()}
                                    for i in gi: plist.append({"src": src, "fusion": f_, "group": g, "study_number": SM.study_number.values[i], "Sample": SM.Sample.values[i], "Pathology": SM.Pathology.values[i], "time": round(time[i], 2), "event": int(ev[i]), "k_prob": SM.k_prob.values[i], "killcoyne_risk_class": kcls[i], "cx_raw_pkg": round(cx_raw[i], 1), "scanner": slide.scanner.values[i], "tissue_tiles": slide.tissue_tiles.values[i], "p53_ihc": SM["P53 IHC"].values[i]})
                                o[f"{f_}_who"] = D
                        R2[key] = o
        res["results"][src] = R2; print("h2", src, flush=True)
    pd.DataFrame(plist).to_csv(HZ + "/h2_patient_list.csv", index=False); res["patient_list_rows"] = len(plist); res["patient_list_path_cluster"] = HZ + "/h2_patient_list.csv"
elif TASK.startswith("h3_"):
    src = TASK.split("_")[1]; O, FO = arms_outer(src); I = arms_inner(src)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    def irls(X, yy, w, lam=1e-4, it=25):
        Xd = np.column_stack([np.ones(len(yy)), X]); beta = np.zeros(3); P = np.diag([0, lam, lam])
        for _ in range(it):
            mu = 1 / (1 + np.exp(-Xd @ beta)); W = w * mu * (1 - mu); g = Xd.T @ (w * (yy - mu)) - P @ beta; Hh = Xd.T @ (Xd * W[:, None]) + P
            step = np.linalg.solve(Hh + 1e-10 * np.eye(3), g); beta = beta + step
            if np.abs(step).max() < 1e-8: break
        return beta
    pid = pd.Series(np.arange(len(np.unique(pat))), index=np.unique(pat))[pat].values; npat = pid.max() + 1
    prepats = np.unique(pat[pre]); rng = np.random.RandomState(0); BW = np.zeros((NB, npat))
    pidmap = {q: i for i, q in enumerate(np.unique(pat))}
    for b in range(NB):
        dr = rng.choice(len(prepats), len(prepats)); np.add.at(BW[b], [pidmap[prepats[i]] for i in dr], 1)   # patient multiplicities
    fits = {}; skipped = {}
    for t in HORIZONS:
        coefs, bcoefs, stack_scores = [], [], np.full((n, 10), np.nan); sk = 0
        for j, r in enumerate(REPS):
            for k in range(1, 11):
                sc_, si_ = I["L-CNV"][(r, k)], I["L-IMG"][(r, k)].reindex(I["L-CNV"][(r, k)].index); rows = sc_.index.values; keep = pre[rows]
                rows = rows[keep]; case, ctrl = labels(time[rows], ev[rows], t); el = case | ctrl; rows = rows[el]; yy = case[el].astype(float)
                if yy.sum() == 0 or yy.sum() == len(yy): sk += 1; continue
                xc, xi = lg(sc_.reindex(rows).values), lg(si_.reindex(rows).values); mc, sdc, mi, sdi = xc.mean(), xc.std(), xi.mean(), xi.std()
                X = np.column_stack([(xc - mc) / sdc, (xi - mi) / sdi]); beta = irls(X, yy, np.ones(len(yy))); coefs.append(beta)
                te = np.where(FO[:, j] == k)[0]; stack_scores[te, j] = beta[0] + beta[1] * (lg(O["L-CNV"][te, j]) - mc) / sdc + beta[2] * (lg(O["L-IMG"][te, j]) - mi) / sdi
                bb = []
                for b in range(NB):
                    w = BW[b][pid[rows]]
                    if w.sum() == 0 or (w * yy).sum() == 0 or (w * (1 - yy)).sum() == 0: bb.append([np.nan] * 3); continue
                    mcw, mii = np.average(xc, weights=w), np.average(xi, weights=w); sdcw, sdiw = np.sqrt(np.average((xc - mcw) ** 2, weights=w)), np.sqrt(np.average((xi - mii) ** 2, weights=w))
                    bb.append(irls(np.column_stack([(xc - mcw) / sdcw, (xi - mii) / sdiw]), yy, w))
                bcoefs.append(bb)
        C_ = np.array(coefs); B_ = np.array(bcoefs)            # fits x 3 ; fits x NB x 3
        ratio = C_[:, 2] / (C_[:, 1] + C_[:, 2]); bratio = B_[:, :, 2] / (B_[:, :, 1] + B_[:, :, 2])
        fits[t] = {"b": C_[:, 1].mean(), "c": C_[:, 2].mean(), "ratio": ratio.mean(), "b_boot": np.nanmean(B_[:, :, 1], 0), "c_boot": np.nanmean(B_[:, :, 2], 0), "ratio_boot": np.nanmean(bratio, 0),
                   "ratio_of_means": C_[:, 2].mean() / (C_[:, 1].mean() + C_[:, 2].mean()), "ratio_of_means_boot": np.nanmean(B_[:, :, 2], 0) / (np.nanmean(B_[:, :, 1], 0) + np.nanmean(B_[:, :, 2], 0)),
                   "n_fits": len(C_), "skipped": sk, "stack": np.nanmean(stack_scores, 1), "n_negative_b": int((C_[:, 1] < 0).sum()), "n_negative_c": int((C_[:, 2] < 0).sum())}
        print("h3", src, t, len(C_), sk, flush=True)
    late = O["L-LATE"].mean(1); rows = np.where(pre)[0]; boots, _ = boots_for(rows); tt, dd = time[rows], ev[rows]; H3 = {}
    for t in HORIZONS:
        f = fits[t]; st = f["stack"][rows]; okr = np.isfinite(st)
        a_s, a_l = tdauc(st[okr], tt[okr], dd[okr], t), tdauc(late[rows][okr], tt[okr], dd[okr], t)
        bd = [tdauc(st[b][okr[b]], tt[b][okr[b]], dd[b][okr[b]], t) - tdauc(late[rows][b][okr[b]], tt[b][okr[b]], dd[b][okr[b]], t) for b in boots]
        H3[str(int(t))] = {"b_mean": r3(f["b"]), "b_ci95": pctl(f["b_boot"]), "c_mean": r3(f["c"]), "c_ci95": pctl(f["c_boot"]), "ratio_mean": r3(f["ratio"]), "ratio_ci95": pctl(f["ratio_boot"]),
                           "ratio_of_means": r3(f["ratio_of_means"]), "ratio_of_means_ci95": pctl(f["ratio_of_means_boot"]), "n_fits": f["n_fits"], "skipped_fits": f["skipped"], "n_negative_b": f["n_negative_b"], "n_negative_c": f["n_negative_c"],
                           "exploratory_heldout": {"n_samples": int(okr.sum()), "stack_ipcw_auroc": r3(a_s), "late_ipcw_auroc": r3(a_l), "delta": r3(a_s - a_l), "delta_ci95": pctl(bd)}}
    H3["ratio_1y_minus_5y"] = {"delta": r3(fits[1.0]["ratio"] - fits[5.0]["ratio"]), "ci95": pctl(fits[1.0]["ratio_boot"] - fits[5.0]["ratio_boot"]),
                               "ratio_of_means_delta": r3(fits[1.0]["ratio_of_means"] - fits[5.0]["ratio_of_means"]), "ratio_of_means_ci95": pctl(fits[1.0]["ratio_of_means_boot"] - fits[5.0]["ratio_of_means_boot"])}
    res["results"] = H3
json.dump(res, open(f"{OUT}/{TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("HZ STACK DONE", TASK)
