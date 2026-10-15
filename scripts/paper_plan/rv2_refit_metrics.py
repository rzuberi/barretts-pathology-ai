"""Review round 2, Part B metrics (docs/paper_review_round2_analyses.md @ d681783). TASK = check | f3_draws | f1 | f2 | f3 | f4 | f5_r28 | f6_r25b | r21b.
tb_common.py is imported unmodified: refit predictions (rv2_refit.R, killcoyne_mm/review_round2/) are loaded into the same (O, I, FO) structure as
tb_common.load() and scored with tb_common.triage() / comparators(); the training-matched comparators A*, B*, C* follow rv_round1.py R1. Voted =
>= 6-of-10 calls; no vote = per-repeat calls with each statistic averaged over the 10 repeats, CIs from the patient bootstrap (2,000 draws,
RandomState(0), folds kept) of the repeat-averaged statistic with calls fixed. Aggregates -> results/paper_final/review_round2/<TASK>.json."""
import os, sys, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tb_common import *
from tb_common import _outer, _inner
RR = M + "/review_round2"; SR = M + "/scanner_robustness"; OUT2 = T + "/results/paper_final/review_round2"; os.makedirs(OUT2, exist_ok=True); TASK = os.environ["TASK"]
SDS = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); SCAN = SDS.scanner.values
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv")); hzi = lambda nm: sorted(glob.glob(f"{HZ}/inner/{nm}_rep_*_fold_*.csv"))
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
def dump(res): res["prespec"] = "d681783"; json.dump(res, open(f"{OUT2}/{TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV2 DONE", TASK, flush=True)
def need(files, k): assert len(files) == k, (len(files), k, files[:2]); return files
def load_refit(cnv_outer, img_outer, cnv_inner, img_inner):
    """same structure as tb_common.load(): O[model] n x 10 outer predictions, I[model][(rep, fold)] inner OOF predictions indexed by sample row, FO n x 10 folds"""
    O, I = {}, {}; O["L-CNV"], FO = _outer(cnv_outer); O["L-IMG"], F2 = _outer(img_outer); assert (FO == F2).all(); O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2
    I["L-CNV"] = _inner(cnv_inner); I["L-IMG"] = _inner(img_inner); I["L-LATE"] = {k: (I["L-CNV"][k] + I["L-IMG"][k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    for k in I["L-CNV"]: assert I["L-IMG"][k].index.sort_values().equals(I["L-CNV"][k].index.sort_values()), k
    assert (FO == FO0).all(); return O, I, FO
FO0 = load("their")[2]
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
        v = res["strategies"][mode]["triage"]["vs_B"]["d_sens"]
        res[f"criterion_vs_B_{mode}"] = {"delta_sens": v["delta"], "one_sided_95_lower": v["one_sided_95_lower"], "noninferior_sens": bool(v["one_sided_95_lower"] > -0.05), "share_sequenced": sh, "met": bool(v["one_sided_95_lower"] > -0.05 and sh < 0.70)}
    return res
res = {"task": TASK}
# ------------------------------------------------------------------ reproduction checks
if TASK == "check":
    def cmp(a, b, col_a="pred", col_b="pred"):
        A = pd.read_csv(a, dtype={"Sample": str}); B = pd.read_csv(b, dtype={"Sample": str}); key = ["Sample"] + (["inner_fold"] if "inner_fold" in A and "inner_fold" in B else [])
        Mg = A.merge(B, on=key, suffixes=("_a", "_b")); assert len(Mg) == len(A) == len(B), (len(A), len(B), len(Mg))
        return {"n": int(len(Mg)), "max_abs_pred_diff": float(np.abs(Mg[col_a + "_a"] - Mg[col_b + "_b"]).max()) if col_a == col_b else float(np.abs(Mg[col_a] - Mg[col_b]).max()),
                "folds_equal": bool((Mg.fold_a == Mg.fold_b).all()), "lambda_max_abs_diff": float(np.abs(Mg["lambda_class" if "lambda_class" in Mg else "lambda_a"] - Mg["lambda_b" if "lambda_b" in Mg else "lambda"]).max())}
    out = {"f1_outer_unfiltered_vs_cfg_00_rep_01": cmp(f"{RR}/f1_outer_repro/cnv_their_rep_01.csv", f"{M}/cv/preds/cfg_00_rep_01.csv"),
           "f1_inner_unfiltered_vs_inner_img_rep_01_fold_01": cmp(f"{RR}/f1_inner_repro/img_rep_01_fold_01.csv", f"{HZ}/inner/img_rep_01_fold_01.csv"),
           "f5_inner_uni2_vs_inner_img_rep_01_fold_01": cmp(f"{RR}/f5_inner_repro/img_rep_01_fold_01.csv", f"{HZ}/inner/img_rep_01_fold_01.csv"),
           "f4_class_lambda_vs_cfg_00_rep_01": cmp(f"{RR}/f4/cnv_their_rep_01.csv", f"{M}/cv/preds/cfg_00_rep_01.csv", "pred_class", "pred")}
    for k, v in out.items(): v["passed"] = bool(v["max_abs_pred_diff"] <= 1e-5 and v["folds_equal"] and v["lambda_max_abs_diff"] <= 1e-9)
    H, FH = _outer(need(sorted(glob.glob(f"{SR}/cv/cv_*_hoptimus0.csv")), 10)); rows = np.where(popmask("pre"))[0]
    out["hoptimus0_outer_auroc_pre"] = {"value": r3(fsa(H, rows, FH)), "reference_docs_paper_scanner_robustness": 0.805, "folds_equal_stored": bool((FH == FO0).all())}
    out["hoptimus0_outer_auroc_pre"]["passed"] = bool(out["hoptimus0_outer_auroc_pre"]["value"] == 0.805 and out["hoptimus0_outer_auroc_pre"]["folds_equal_stored"])
    L = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in sorted(glob.glob(f"{M}/linear/cfg_00_chunk_*.csv"))]); L = L[L.rule == "class_min"]
    out["lopo_L-CNV_their_stored"] = {"rows": int(len(L)), "patients": int(L.Patient.nunique()), "all_samples_covered": bool(set(L.Sample) == set(ids)), "source": "km_linear.R cfg 0 (killcoyne_mm/linear), rule class_min", "F7_needed": not bool(set(L.Sample) == set(ids))}
    out["all_passed"] = bool(all(v.get("passed", True) for v in out.values() if isinstance(v, dict)) and not out["lopo_L-CNV_their_stored"]["F7_needed"])
    res["checks"] = out; dump(res)
# ------------------------------------------------------------------ F3 draws: first 200 of the standard patient-bootstrap sequence over the pre-event population
elif TASK == "f3_draws":
    rows = np.where(popmask("pre"))[0]; pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(0); ch = []
    while len(ch) < 200:
        c = rng.choice(len(up), len(up)); b = np.concatenate([ro[up[i]] for i in c])
        if 0 < y[b].sum() < len(b): ch.append(c)
    ref = boots_for(rows, nb=200); assert all((np.concatenate([ro[up[i]] for i in c]) == b).all() for c, b in zip(ch, ref))
    D = pd.DataFrame([(d, up[i], int(m)) for d, c in enumerate(ch) for i, m in zip(*np.unique(c, return_counts=True))], columns=["draw", "Patient", "mult"])
    os.makedirs(f"{RR}/f3", exist_ok=True); D.to_csv(f"{RR}/f3/draws.csv", index=False)
    res.update({"draws": 200, "population": "pre", "n_patients": int(len(up)), "consistent_with_tb_common_boots_for": True, "distinct_patients_per_draw": {"mean": r3(D.groupby("draw").size().mean()), "range": [int(D.groupby("draw").size().min()), int(D.groupby("draw").size().max())]}}); dump(res)
# ------------------------------------------------------------------ F1 pre-event-only training
elif TASK == "f1":
    out = {}
    for src in ("their", "pkg"):
        O, I, FO = load_refit(need(sorted(glob.glob(f"{RR}/f1_outer/cnv_{src}_rep_*.csv")), 10), need(sorted(glob.glob(f"{RR}/f1_outer/img_rep_*.csv")), 10),
                              need(sorted(glob.glob(f"{RR}/f1_inner/cnv_{src}_rep_*_fold_*.csv")), 100), need(sorted(glob.glob(f"{RR}/f1_inner/img_rep_*_fold_*.csv")), 100))
        nt = pd.concat([pd.read_csv(f, usecols=["rep", "fold", "n_train"]) for f in glob.glob(f"{RR}/f1_outer/img_rep_*.csv")]).drop_duplicates()
        o = {"training_rows_per_fold": {"mean": r3(nt.n_train.mean()), "range": [int(nt.n_train.min()), int(nt.n_train.max())], "stored_main_cv": "all samples of the training patients (about 608)"}}
        for pop in ("pre", "pre_ndbe"):
            o[pop] = triage_block(O, I, FO, pop)
            try:
                fx = json.load(open(f"{T}/results/paper_final/triage/{src}_{pop}.json")); r1 = json.load(open(f"{T}/results/paper_final/review_round1/r1_{src}_{pop}.json"))["R1"]
                o[pop]["stored_reference"] = {"auroc": {m: fx["section1"]["auroc"][m]["value"] for m in ("L-CNV", "L-IMG", "L-LATE")}, "img_vs_cnv": fx["section1"]["img_vs_cnv"], "triage_vs_B_sens_voted": fx["strategies"]["triage_LATE"]["vs_B"]["sens"],
                                              "matched_voted": {a: r1["matched"][a]["d_sens_triage_minus"] for a in ("A*", "B*", "C*")}}
            except FileNotFoundError: pass
        out[src] = o
    res["F1"] = out; dump(res)
# ------------------------------------------------------------------ F2 training-size control
elif TASK == "f2":
    F = need(sorted(glob.glob(f"{RR}/f2/img_rep_*_fold_*.csv")), 100); O2, F2 = _outer(F); assert (F2 == FO0).all(); O, _, FO = load("their"); rows = np.where(popmask("pre"))[0]; boots = boots_for(rows)
    meta = pd.concat([pd.read_csv(f, usecols=["rep", "fold", "n_train_P_patients", "full_train_P_patients", "full_train_patients"]) for f in F]).drop_duplicates()
    out = {"training_patients": 48, "P_patients_in_48": {"mean": r3(meta.n_train_P_patients.mean()), "range": [int(meta.n_train_P_patients.min()), int(meta.n_train_P_patients.max())]}, "full_training_patients": {"mean": r3(meta.full_train_patients.mean()), "P_mean": r3(meta.full_train_P_patients.mean())}, "by_scanner": {}}
    for sc in ("C13210", "C13239-01"):
        mk = SCAN == sc; v48 = fsa(O2, rows, FO, mask=mk); vf = fsa(O["L-IMG"], rows, FO, mask=mk); B48 = np.array([fsa(O2, b, FO, mask=mk) for b in boots]); Bf = np.array([fsa(O["L-IMG"], b, FO, mask=mk) for b in boots])
        out["by_scanner"][sc] = {"n_samples": int((popmask("pre") & mk).sum()), "size48_mixed_scanner": {"value": r3(v48), "ci95": ci2(B48)}, "full_size_main_cv": {"value": r3(vf), "ci95": ci2(Bf)}, "d_size48_minus_full": {"delta": r3(v48 - vf), "ci95": ci2(B48 - Bf)}}
    out["by_scanner"]["C13210"]["transfer_model_C13239-01_to_C13210_reference"] = 0.726; out["by_scanner"]["C13239-01"]["within_C13239-01_cv_reference"] = 0.841
    out["all_pre_event"] = {"size48": {"value": r3(fsa(O2, rows, FO)), "ci95": ci2([fsa(O2, b, FO) for b in boots])}, "full_size": r3(fsa(O["L-IMG"], rows, FO))}
    res["F2"] = out; dump(res)
# ------------------------------------------------------------------ F3 refit bootstrap
elif TASK == "f3":
    ND = int(os.environ.get("NDRAWS", "200")); PREs = pd.Series(PRE, index=ids); done = []
    def rd(f): D = pd.read_csv(f, dtype={"Sample": str, "Patient": str}); D["pre"] = PREs.reindex(D.Sample).values; return D
    def draw_stats(d):
        oc, oi = rd(f"{RR}/f3/cnv_their_draw_{d:03d}_outer.csv"), rd(f"{RR}/f3/img_draw_{d:03d}_outer.csv"); ic, ii = rd(f"{RR}/f3/cnv_their_draw_{d:03d}_inner.csv"), rd(f"{RR}/f3/img_draw_{d:03d}_inner.csv")
        E = oc.merge(oi[["Sample", "copy", "pred"]], on=["Sample", "copy"], suffixes=("_cnv", "_img")); assert len(E) == len(oc) == len(oi); E = E[E.pre].reset_index(drop=True); E["late"] = (E.pred_cnv + E.pred_img) / 2
        Tr = ic.merge(ii[["Sample", "copy", "fold", "pred"]], on=["Sample", "copy", "fold"], suffixes=("_cnv", "_img")); assert len(Tr) == len(ic) == len(ii); Tr = Tr[Tr.pre].reset_index(drop=True); Tr["late"] = (Tr.pred_cnv + Tr.pred_img) / 2
        yy = E.y.values == 1; f = E.fold.values; auc = {}
        for m in ("pred_cnv", "pred_img", "late"):
            x = E[m].values; num = den = 0.0
            for k in range(1, 11):
                mk = f == k; yk = yy[mk]; n1 = int(yk.sum()); n0 = int((~yk).sum())
                if n1 == 0 or n0 == 0: continue
                rk = rankdata(x[mk]); num += rk[yk].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
            auc[m] = num / den
        call = {s: np.zeros(len(E), bool) for s in ("triage", "B", "B*")}; band = np.zeros(len(E), int)
        for k in range(1, 11):
            t = Tr[Tr.fold == k]; s_tr = t.pred_img.values; yc = t.y.values == 1; li = t.late.values
            tl = thr_sens(s_tr[yc], 0.95); cs = c_star(s_tr[~yc], 0.90); mid = (s_tr >= tl) & (s_tr <= cs) if tl <= cs else np.zeros(len(t), bool)
            cm = li[mid & yc]; ts = thr_sens(cm, 0.8) if len(cm) >= 5 else thr_sens(li[yc], 0.8)
            tcall = np.where(s_tr < tl, False, np.where(s_tr > cs, True, li >= ts)) if tl <= cs else s_tr > cs; s_star = 1 - tcall[~yc].mean()
            xc = np.sort(li[~yc]); q = int(np.ceil(s_star * len(xc))); thr_bs = xc[q - 1] if q > 0 else -np.inf; thr_b = thr_sens(li[yc], 0.8)
            te = f == k; s = E.pred_img.values[te]; la = E.late.values[te]
            bd = np.where(s < tl, 0, np.where(s > cs, 2, 1)) if tl <= cs else np.where(s > cs, 2, 0); band[te] = bd
            call["triage"][te] = np.where(bd == 1, la >= ts, bd == 2); call["B"][te] = la >= thr_b; call["B*"][te] = la > thr_bs
        sp = {s: (c[yy].mean(), 1 - c[~yy].mean()) for s, c in call.items()}
        return {"d_auc_img_minus_cnv": auc["pred_img"] - auc["pred_cnv"], "auc_cnv": auc["pred_cnv"], "auc_img": auc["pred_img"], "auc_late": auc["late"], "d_sens_vs_B": sp["triage"][0] - sp["B"][0], "d_spec_vs_B": sp["triage"][1] - sp["B"][1],
                "d_sens_vs_Bstar": sp["triage"][0] - sp["B*"][0], "sens_triage": sp["triage"][0], "spec_triage": sp["triage"][1], "share_mid": float((band == 1).mean()), "n_eval_rows": int(len(E))}
    S = []
    for d in range(ND):
        if all(os.path.exists(f"{RR}/f3/{m}_draw_{d:03d}_{p}.csv") for m in ("cnv_their", "img") for p in ("outer", "inner")): S.append(draw_stats(d)); done.append(d)
    assert len(S) >= 100, len(S); A = {k: np.array([s[k] for s in S], float) for k in S[0]}
    fx = json.load(open(f"{T}/results/paper_final/triage/their_pre.json")); r1 = json.load(open(f"{T}/results/paper_final/review_round1/r1_their_pre.json"))["R1"]
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; boots = boots_for(rows); band0, call0, _ = triage(O, I, FO, popm); comp0, _ = comparators(O, I, FO, popm); mc0, _ = matched(O, I, FO, popm)
    cond_nv = strategies({"triage": call0, "B": comp0["B"], "B*": mc0["B*"]}, rows, boots)["no_vote"]["triage"]
    def pi(a): return {"mean": r3(np.mean(a)), "median": r3(np.median(a)), "ci95_percentile": ci2(a), "one_sided_95_lower": lo1(a), "sd": r3(np.std(a, ddof=1))}
    out = {"draws_scored": len(S), "draw_indices_scored_max": int(max(done)), "n_eval_rows_mean": r3(A["n_eval_rows"].mean()),
           "d_auroc_IMG_minus_CNV": {"refit_bootstrap": pi(A["d_auc_img_minus_cnv"]), "non_inferior_at_-0.05": bool(np.percentile(A["d_auc_img_minus_cnv"], 5) > -0.05), "conditional_stored": fx["section1"]["img_vs_cnv"]},
           "auroc_refit_draw_mean": {"L-CNV": r3(A["auc_cnv"].mean()), "L-IMG": r3(A["auc_img"].mean()), "L-LATE": r3(A["auc_late"].mean())},
           "d_sens_triage_vs_B": {"refit_bootstrap": pi(A["d_sens_vs_B"]), "non_inferior_at_-0.05": bool(np.percentile(A["d_sens_vs_B"], 5) > -0.05), "conditional_stored_voted": fx["strategies"]["triage_LATE"]["vs_B"]["sens"], "conditional_stored_no_vote": cond_nv["vs_B"]["d_sens"]},
           "d_spec_triage_vs_B": {"refit_bootstrap": pi(A["d_spec_vs_B"]), "conditional_stored_voted": fx["strategies"]["triage_LATE"]["vs_B"]["spec"], "conditional_stored_no_vote": cond_nv["vs_B"]["d_spec"]},
           "d_sens_triage_vs_Bstar": {"refit_bootstrap": pi(A["d_sens_vs_Bstar"]), "non_inferior_at_-0.05": bool(np.percentile(A["d_sens_vs_Bstar"], 5) > -0.05), "conditional_stored_voted": r1["matched"]["B*"]["d_sens_triage_minus"], "conditional_stored_no_vote": cond_nv["vs_B*"]["d_sens"]},
           "triage_per_draw": {"sensitivity": pi(A["sens_triage"]), "specificity": pi(A["spec_triage"]), "share_sequenced": pi(A["share_mid"])},
           "note": "single stratified 10-fold CV per draw (seed = draw index), lambda and all cut-offs re-chosen inside the draw; no vote; copies of a patient share the fold and every grouped split"}
    res["F3"] = out; dump(res)
# ------------------------------------------------------------------ F4 lambda by deviance
elif TASK == "f4":
    O, _, FO = load("their"); out = {"share_fold_models_with_no_selected_feature": {}, "reproduction_class_lambda_vs_stored_max_abs_pred_diff": {}}; Od = {}
    for cfg, m, c in (("cnv_their", "L-CNV", 0), ("img", "L-IMG", 2)):
        D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in need(sorted(glob.glob(f"{RR}/f4/{cfg}_rep_*.csv")), 10)])
        Pc = D.pivot_table(index="Sample", columns="rep", values="pred_class").reindex(ids)[REPS].values; Pd = D.pivot_table(index="Sample", columns="rep", values="pred_dev").reindex(ids)[REPS].values
        Fd = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int); assert (Fd == FO).all(); Od[m] = Pd
        out["reproduction_class_lambda_vs_stored_max_abs_pred_diff"][m] = float(np.abs(Pc - O[m]).max())
        fm = D.drop_duplicates(["rep", "fold"]); assert len(fm) == 100
        out["share_fold_models_with_no_selected_feature"][m] = {"class_error_lambda": r3((fm.nnz_class == 0).mean()), "deviance_lambda": r3((fm.nnz_dev == 0).mean()), "n_nonzero_median": {"class_error_lambda": r3(fm.nnz_class.median()), "deviance_lambda": r3(fm.nnz_dev.median())},
                                                                  "lambda_median": {"class_error": r3(fm.lambda_class.median()), "deviance": r3(fm.lambda_dev.median())}, "deviance_lambda_smaller_share": r3((fm.lambda_dev < fm.lambda_class).mean())}
    Od["L-LATE"] = (Od["L-CNV"] + Od["L-IMG"]) / 2
    for pop in ("pre", "pre_ndbe"):
        rows = np.where(popmask(pop))[0]; boots = boots_for(rows); out[pop] = {"deviance_lambda": auc_block(Od, FO, rows, boots), "class_error_lambda_stored": auc_block(O, FO, rows, boots)}
    res["F4"] = out; dump(res)
# ------------------------------------------------------------------ F5 / R28 H-optimus-0 gate
elif TASK == "f5_r28":
    O, I, FO = load_refit(kvf(0), need(sorted(glob.glob(f"{SR}/cv/cv_*_hoptimus0.csv")), 10), hzi("cnv_their"), need(sorted(glob.glob(f"{RR}/f5_inner/img_rep_*_fold_*.csv")), 100))
    out = {"gate": "L-IMG on H-optimus-0 sample embeddings (stored outer predictions of the scanner-robustness downstream run; inner predictions F5)", "L-CNV": "stored, their matrix"}
    for pop in ("pre", "pre_ndbe"):
        out[pop] = triage_block(O, I, FO, pop); fx = json.load(open(f"{T}/results/paper_final/triage/their_{pop}.json")); r1 = json.load(open(f"{T}/results/paper_final/review_round1/r1_their_{pop}.json"))["R1"]
        out[pop]["stored_reference_UNI2h"] = {"auroc": {m: fx["section1"]["auroc"][m]["value"] for m in ("L-CNV", "L-IMG", "L-LATE")}, "img_vs_cnv": fx["section1"]["img_vs_cnv"], "triage_vs_B_sens_voted": fx["strategies"]["triage_LATE"]["vs_B"]["sens"],
                                              "share_sequenced_voted_mean": fx["bands"]["mid"]["all"]["value"], "matched_voted": {a: r1["matched"][a]["d_sens_triage_minus"] for a in ("A*", "B*", "C*")}}
    res["F5_R28"] = out; dump(res)
# ------------------------------------------------------------------ F6 / R25(b) permutation-trained gate
elif TASK == "f6_r25b":
    O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; per = {}; keys = ["share_sequenced", "sensitivity", "specificity"]
    for p in range(1, 6):
        Op = dict(O); Ip = dict(I); Op["L-IMG"], Fp = _outer(need(sorted(glob.glob(f"{RR}/f6_outer/img_perm_{p}_rep_*.csv")), 10)); assert (Fp == FO).all()
        Ip["L-IMG"] = _inner(need(sorted(glob.glob(f"{RR}/f6_inner/img_perm_{p}_rep_*_fold_*.csv")), 100)); Op["L-LATE"] = (Op["L-CNV"] + Op["L-IMG"]) / 2
        Ip["L-LATE"] = {k: (Ip["L-CNV"][k] + Ip["L-IMG"][k].reindex(Ip["L-CNV"][k].index)) / 2 for k in Ip["L-CNV"]}
        band, call, info = triage(Op, Ip, FO, popm); s, sp = ss_nv(call, rows); sv, spv = ss_v(call, rows)
        per[p] = {"no_vote": {"share_sequenced": r3((band[rows] == 1).mean()), "sensitivity": r3(s), "specificity": r3(sp)}, "voted": {"share_sequenced_in_at_least_one_repeat": r3((band[rows] == 1).any(1).mean()), "sensitivity": r3(sv), "specificity": r3(spv)},
                  "folds_no_middle_band": info["no_mid"], "t_seq_fallback": info["fallback"], "gate_auroc_fold_stratified": r3(fsa(Op["L-IMG"], rows, FO)), "late_auroc_with_permuted_gate": r3(fsa(Op["L-LATE"], rows, FO))}
    out = {"permutations": 5, "per_permutation": per, "mean_over_permutations_no_vote": {k: r3(np.mean([per[p]["no_vote"][k] for p in per])) for k in keys}, "mean_over_permutations_voted": {k: r3(np.mean([per[p]["voted"][k] for p in per])) for k in ("sensitivity", "specificity")},
           "gate_auroc_mean": r3(np.mean([per[p]["gate_auroc_fold_stratified"] for p in per]))}
    band, call, _ = triage(O, I, FO, popm); s, sp = ss_nv(call, rows); out["real_gate_no_vote"] = {"share_sequenced": r3((band[rows] == 1).mean()), "sensitivity": r3(s), "specificity": r3(sp)}
    out["note"] = "labels permuted at the patient level within each outer training set (outer and inner fits of L-IMG); cut-offs chosen on the permuted gate's inner scores against the real labels; middle-band call = real L-LATE (stored L-CNV + permuted L-IMG)"
    res["F6_R25b"] = out; dump(res)
# ------------------------------------------------------------------ R21(b) leave-one-patient-out L-CNV from our pipeline
elif TASK == "r21b":
    L = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in sorted(glob.glob(f"{M}/linear/cfg_00_chunk_*.csv"))]); L = L[L.rule == "class_min"]
    S = L.set_index("Sample").pred.reindex(ids).values; assert np.isfinite(S).all() and len(L) == n; O, _, FO = load("their"); out = {"source": "km_linear.R cfg 0 (cnv, their matrix), leave-one-patient-out over the 80 patients, lambda = class-error minimum of the 10 x 5 patient-grouped CV (as kv_cv.R)", "rows": int(len(L))}
    def plain(x, b): yb = y[b]; rk = rankdata(x[b]); n1 = yb.sum(); n0 = len(yb) - n1; return float((rk[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    for pop in ("pre", "pre_ndbe"):
        rows = np.where(popmask(pop))[0]; boots = boots_for(rows); ol = fsa(S, rows, FO); oc = fsa(O["L-CNV"], rows, FO); Bl = np.array([fsa(S, b, FO) for b in boots]); Bc = np.array([fsa(O["L-CNV"], b, FO) for b in boots]); d = Bl - Bc
        out[pop] = {"n_samples": int(len(rows)), "fold_stratified_auroc_on_10x10_pairs": {"LOPO_L-CNV": {"value": r3(ol), "ci95": ci2(Bl)}, "CV_L-CNV_10x10": {"value": r3(oc), "ci95": ci2(Bc)}},
                    "d_LOPO_minus_CV": {"delta": r3(ol - oc), "ci95": ci2(d), "p_two_sided": p2(d)}, "plain_pooled_auroc": {"LOPO_L-CNV": r3(plain(S, rows)), "CV_L-CNV_repeat_mean_score": r3(plain(O["L-CNV"].mean(1), rows))}}
    res["R21b"] = out; dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")
