"""Two papers, agent CNC (docs/two_papers_CNC.md): new copy-number representations scored like L-CNV.
Metric helpers (fsa, boots, hblock, patw, dstat, ci2, cnt) come from tp_p1b_main.py, whose module body is executed unchanged with a task name matching
none of its branches (as tp_art_main.py does). Inputs, read-only: stored predictions (tb_common.load), ART's clean-tile L-IMG refit
(killcoyne_mm/two_papers_ART/refit/clean), the round-4 re-segmentation refits (review_round4/r36_outer, r36_inner), this agent's refits
(killcoyne_mm/two_papers_CNC/refit/<FEAT>).
TASK = smoke_refit | score | select        CNC_NB=k: smoke mode (k bootstrap draws; JSON written aside).
Row-level files only under killcoyne_mm/two_papers_CNC/ on the cluster; aggregates (no identifiers, cells < 5 as "<5") -> results/paper_final/two_papers_CNC/."""
import os, sys, json, glob, numpy as np, pandas as pd
PP = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"; sys.path.insert(0, PP)
CTASK = os.environ["TASK"]; os.environ["TASK"] = "cnc_none"
exec(compile(open(PP + "/tp_p1b_main.py").read(), PP + "/tp_p1b_main.py", "exec"))   # definitions only (no P1B branch matches)
os.environ["TASK"] = CTASK
C_ = M + "/two_papers_CNC"; OUTC = T + "/results/paper_final/two_papers_CNC"; NBC = NB
if os.environ.get("CNC_NB"): NBC = int(os.environ["CNC_NB"]); OUTC = C_ + "/smoke_json"
NBX = NBC
os.makedirs(OUTC, exist_ok=True)
def cdump(res, name=None):
    res["agent"] = "CNC"; json.dump(res, open(f"{OUTC}/{name or CTASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("CNC DONE", name or CTASK, flush=True)
def read_refit(d, need_inner=True):
    """outer pred (n x 10), inner pred and inner fold per (rep, fold): tp_p1b_main select_ read_refit, directory given"""
    fo = sorted(glob.glob(f"{d}/outer_rep_*.csv")); fi = sorted(glob.glob(f"{d}/inner_rep_*.csv")); assert len(fo) == 10 and (len(fi) == 10 or not need_inner), (d, len(fo), len(fi))
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); po = lambda c: D.pivot_table(index="Sample", columns="rep", values=c, dropna=False).reindex(ids)[REPS].values
    assert (po("fold").astype(int) == FO0).all(), "folds differ from the stored folds"
    I, IF = {}, {}
    if fi:
        Di = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fi]); assert Di.groupby(["rep", "fold"]).ngroups == 100
        for (r, k), d_ in Di.groupby(["rep", "fold"]): pos = ix[d_.Sample].values; I[(r, k)] = pd.Series(d_.pred.values, index=pos); IF[(r, k)] = pd.Series(d_.inner_fold.values, index=pos)
    return {"O": po("pred"), "I": I, "IF": IF}
res = {"task": CTASK}
# ================================================================== smoke: the CNC refit script reproduces stored L-CNV (outer cfg_00, inner horizons/inner/cnv_their)
if CTASK == "smoke_refit":
    reps = sorted(int(f[-6:-4]) for f in glob.glob(f"{C_}/refit/their/outer_rep_*.csv"))
    out = {}
    for r in reps:
        a = pd.read_csv(f"{C_}/refit/their/outer_rep_{r:02d}.csv", dtype={"Sample": str}); b = pd.read_csv(f"{M}/cv/preds/cfg_00_rep_{r:02d}.csv", dtype={"Sample": str})
        m = a.merge(b, on=["Sample", "fold"], suffixes=("", "_s")); assert len(m) == len(a) == len(b)
        ai = pd.read_csv(f"{C_}/refit/their/inner_rep_{r:02d}.csv", dtype={"Sample": str})
        bi = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(f"{HZ}/inner/cnv_their_rep_{r:02d}_fold_*.csv"))])
        mi = ai.merge(bi, on=["Sample", "fold", "inner_fold"], suffixes=("", "_s")); assert len(mi) == len(ai) == len(bi), (len(mi), len(ai), len(bi))
        out[r] = {"outer_max_abs_pred": float(np.abs(m.pred - m.pred_s).max()), "outer_lambda_equal": bool(np.allclose(m["lambda"], m["lambda_s"])),
                  "inner_max_abs_pred": float(np.abs(mi.pred - mi.pred_s).max()), "inner_lambda_equal": bool(np.allclose(mi["lambda"], mi["lambda_s"]))}
    res["refit_vs_stored_L_CNV"] = out
    if len(reps) == 10:
        O = read_refit(f"{C_}/refit/their")["O"]; res["fsa_refit_their"] = r3(fsa(O, ROWS)); res["fsa_stored_L_CNV"] = r3(fsa(O0["L-CNV"], ROWS))
    print(json.dumps(res, indent=1)); cdump(res)
# ================================================================== shared: copy-number candidates (outer + inner predictions) and clean-tile L-IMG
NEW = [f for f in ("cbs", "ace", "ichor", "gis", "cin_drews", "cin_mac", "pkg_gis", "tessera_qn", "tessera_raw") if len(glob.glob(f"{C_}/refit/{f}/outer_rep_*.csv")) == 10]
LABEL = {"their": "L-CNV (shipped matrix)", "pkg": "joint package rebuild", "pkgps": "per-sample package segmentation", "pkgpast": "past-only package segmentation",
         "cbs": "QDNAseq CBS segments", "ace": "ACE absolute copy number", "ichor": "ichorCNA (1 Mb HMM)", "gis": "genome-instability scores", "cin_drews": "CIN signatures (Drews 2022)",
         "cin_mac": "CN signatures (Macintyre 2018)", "pkg_gis": "joint package + instability scores", "tessera_qn": "TESSERA embeddings (TCGA rank-mapped)", "tessera_raw": "TESSERA embeddings (raw log2)"}
def stored_cand(tag):
    """stored outer + inner predictions of the four existing L-CNV versions (read-only)"""
    if tag in ("their", "pkg"): fo = sorted(glob.glob(f"{M}/cv/preds/cfg_{0 if tag == 'their' else 1:02d}_rep_*.csv")); fi = sorted(glob.glob(f"{HZ}/inner/cnv_{tag}_rep_*_fold_*.csv"))
    else: fo = sorted(glob.glob(f"{M}/review_round4/r36_outer/cnv_{tag}_rep_*.csv")); fi = sorted(glob.glob(f"{M}/review_round4/r36_inner/cnv_{tag}_rep_*_fold_*.csv"))
    assert len(fo) == 10 and len(fi) == 100, (tag, len(fo), len(fi))
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); po = lambda c: D.pivot_table(index="Sample", columns="rep", values=c).reindex(ids)[REPS].values
    assert (po("fold").astype(int) == FO0).all(); Di = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fi]); I, IF = {}, {}
    for (r, k), d_ in Di.groupby(["rep", "fold"]): pos = ix[d_.Sample].values; I[(r, k)] = pd.Series(d_.pred.values, index=pos); IF[(r, k)] = pd.Series(d_.inner_fold.values, index=pos)
    return {"O": po("pred"), "I": I, "IF": IF}
def cand(tag): return stored_cand(tag) if tag in ("their", "pkg", "pkgps", "pkgpast") else read_refit(f"{C_}/refit/{tag}")
CLEAN = read_refit(M + "/two_papers_ART/refit/clean", need_inner=False)["O"]
def checks():
    c = {"stored_L_CNV": r3(fsa(O0["L-CNV"], ROWS)), "clean_L_IMG": r3(fsa(CLEAN, ROWS)), "clean_L_LATE": r3(fsa((O0["L-CNV"] + CLEAN) / 2, ROWS)), "stored_L_IMG": r3(fsa(O0["L-IMG"], ROWS))}
    assert (c["stored_L_CNV"], c["clean_L_IMG"], c["clean_L_LATE"], c["stored_L_IMG"]) == (0.801, 0.811, 0.875, 0.847), c
    return c
# ================================================================== score: each CN representation, alone and fused with clean-tile L-IMG
if CTASK == "score":
    res["checks"] = checks(); tags = ["their", "pkg", "pkgps", "pkgpast"] + NEW; L = {t: cand(t)["O"] for t in tags}
    assert np.abs(L["their"] - O0["L-CNV"]).max() == 0
    O = {"L-CNV": O0["L-CNV"], "L-IMG": CLEAN, "L-LATE": (O0["L-CNV"] + CLEAN) / 2, "L-IMG stored": O0["L-IMG"]}
    for t in tags: O["CN:" + t] = L[t]; O["LATE:" + t] = (L[t] + CLEAN) / 2
    rows = ROWS; bb = boots(rows); W_ = patw(rows); R = {}
    for unit, w in (("sample", None), ("patient_weighted", W_)):
        r, o, B = hblock(O, rows, bb, w=w, keep=True); T_ = {}
        for t in tags:
            T_[t] = {"label": LABEL[t], "auroc": r["auroc"]["CN:" + t], "delta_vs_L_CNV": dstat(o["CN:" + t] - o["L-CNV"], B["CN:" + t] - B["L-CNV"]),
                     "delta_vs_clean_L_IMG": dstat(o["CN:" + t] - o["L-IMG"], B["CN:" + t] - B["L-IMG"]), "late_auroc": r["auroc"]["LATE:" + t],
                     "late_minus_clean_L_IMG": dstat(o["LATE:" + t] - o["L-IMG"], B["LATE:" + t] - B["L-IMG"]), "late_minus_clean_L_LATE": dstat(o["LATE:" + t] - o["L-LATE"], B["LATE:" + t] - B["L-LATE"])}
        R[unit] = {"reference": {m: r["auroc"][m] for m in ("L-CNV", "L-IMG", "L-LATE", "L-IMG stored")}, "n": {k: r[k] for k in ("n_samples", "n_progressor_samples", "n_patients", "n_progressor_patients")},
                   "representations": T_}
    res["score"] = R; res["new_representations"] = NEW
    res["note"] = ("fold-stratified AUROC over the 571 pre-event samples (fsa), patient-weighted = pair weight 1/(n_i n_j); 2,000-draw patient bootstrap, predictions fixed, paired deltas on the same draws. "
                   "L-IMG = ART clean-tile refit; L-LATE (reference) = (L-CNV + clean L-IMG) / 2; LATE:<rep> = (CN representation + clean L-IMG) / 2. Same outer folds, recipe and nested lambda for every representation.")
    cdump(res)
# ================================================================== select: nested choice of the CN representation (tp_p1b_main select_ logic, unchanged)
elif CTASK == "select":
    res["checks"] = checks()
    def wauc(x, l, f):
        num = den = 0.0
        for k in np.unique(f):
            m = (f == k) & np.isfinite(x); xp, xn = x[m & (l == 1)], x[m & (l == 0)]
            if len(xp) == 0 or len(xn) == 0: continue
            num += ((xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])).sum(); den += len(xp) * len(xn)
        return num / den if den else np.nan
    def inner_auc(sr, ifo, rr): return wauc(sr.reindex(rr).values, y[rr], ifo.reindex(rr).values)
    SETS = {"existing_4": ["their", "pkg", "pkgps", "pkgpast"], "all": ["their", "pkg", "pkgps", "pkgpast"] + NEW}
    LC = {t: cand(t) for t in SETS["all"]}; assert np.abs(LC["their"]["O"] - O0["L-CNV"]).max() == 0
    rows = ROWS; bb = boots(rows); W_ = patw(rows); OUTR = {}
    for nm, C in SETS.items():
        Oi = np.full((n, 10), np.nan); pick = {c: 0 for c in C}
        for j, r in enumerate(REPS):
            for k in range(1, 11):
                tri = LC[C[0]]["I"][(r, k)].index.values; tri = tri[PRE[tri]]; a = {c: inner_auc(LC[c]["I"][(r, k)], LC[c]["IF"][(r, k)], tri) for c in C}
                b_ = max(C, key=lambda c: (a[c], -C.index(c))); pick[b_] += 1; te = FO0[:, j] == k; Oi[te, j] = LC[b_]["O"][te, j]
        OS = {"L-CNV": O0["L-CNV"], "L-IMG": CLEAN, "L-LATE": (O0["L-CNV"] + CLEAN) / 2, "CN-nested": Oi, "LATE-nested": (Oi + CLEAN) / 2}
        for c in C: OS["cand:" + c] = LC[c]["O"]
        R = {"candidates": C, "times_chosen_of_100_folds": pick}
        for unit, w in (("sample", None), ("patient_weighted", W_)):
            obs = {m: fsa(S_, rows, w=w) for m, S_ in OS.items()}; B = pd.DataFrame([{m: fsa(S_, b, w=w) for m, S_ in OS.items()} for b in bb])
            cn = ["cand:" + c for c in C]; best = max(cn, key=lambda c: obs[c]); Bmax = B[cn].max(1)
            R[unit] = {"auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in ("L-CNV", "L-IMG", "L-LATE", "CN-nested", "LATE-nested")},
                       "nested_minus_L_CNV": dstat(obs["CN-nested"] - obs["L-CNV"], B["CN-nested"] - B["L-CNV"]),
                       "nested_CN_minus_clean_L_IMG": dstat(obs["CN-nested"] - obs["L-IMG"], B["CN-nested"] - B["L-IMG"]),
                       "nested_LATE_minus_clean_L_IMG": dstat(obs["LATE-nested"] - obs["L-IMG"], B["LATE-nested"] - B["L-IMG"]),
                       "nested_LATE_minus_clean_L_LATE": dstat(obs["LATE-nested"] - obs["L-LATE"], B["LATE-nested"] - B["L-LATE"]),
                       "apparent_best_candidate": {"name": best[5:], "auroc": r3(obs[best]), "max_over_candidates_in_draws_ci95": ci2(Bmax)},
                       "optimism_apparent_best_minus_nested": dstat(obs[best] - obs["CN-nested"], Bmax - B["CN-nested"]),
                       "optimism_L_CNV_minus_nested": dstat(obs["L-CNV"] - obs["CN-nested"], B["L-CNV"] - B["CN-nested"]),
                       "candidate_auroc": {c: r3(obs["cand:" + c]) for c in C}}
        OUTR[nm] = R
    res["select"] = OUTR
    res["note"] = ("nested = per (repeat, outer fold) the CN representation with the best fold-stratified inner out-of-fold AUROC (unweighted) on the fold's pre-event training samples "
                   "(tp_p1b_main.py select_ logic); its held-out predictions are used for that fold. Apparent best = highest held-out AUROC chosen after seeing it (optimistic); "
                   "in each draw the max over candidates is taken. L-IMG = clean-tile refit (ART).")
    cdump(res)
