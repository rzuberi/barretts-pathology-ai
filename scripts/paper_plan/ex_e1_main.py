"""Exploration E1 (docs/paper_explore_E1.md): error analysis of non-progressor false flags under R49 and H&E alone (P1 90%), the H&E score split into
per-tile contributions, and candidate fixes evaluated nested (thresholds / models on training patients' inner out-of-fold quantities, scored on held-out
patients). Exploratory; not confirmatory.
Policy definitions, thresholds, outcomes and bootstrap draws are those of rv8_landscape.py / rv8b_step3.py (their module bodies are executed unchanged up
to the task dispatch). Barrett's database linkage (db_link, later_db) from rv7_round7.py and same-block labels from rv7d_sensfirst.py, executed in their
own namespaces. H&E model coefficients from ex_e1_coef.R (verbatim refits, checked against the stored predictions); per-tile pixel statistics from ex_e1_px.py.
Row-level files only under killcoyne_mm/explore_E1/ on the cluster; aggregates (no identifiers, no dates) -> results/paper_final/explore_E1/<TASK>.json.
TASK = u (tile scores per sample x repeat x fold model) | prep (covariates + tile summaries) | char (E1.1) | tiles (E1.2) | fix_score | fix_veto | fix_filter (E1.3) | summary
EXE1_NB=k: smoke mode (k bootstrap draws; aggregates written aside)."""
import os, sys, json, glob, pickle, re, numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
E1T = os.environ["TASK"]; os.environ["TASK"] = "none"
_s = open(HERE + "/rv8b_step3.py").read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], HERE + "/rv8b_step3.py", "exec"))
TASK = E1T; os.environ["TASK"] = E1T
OUT = T + "/results/paper_final/explore_E1"; X1 = M + "/explore_E1"; KD = T + "/feasibility/paper_plan/killcoyne"
if os.environ.get("EXE1_NB"): NB = int(os.environ["EXE1_NB"]); TC.NB = NB; OUT = X1 + "/smoke_json"
for d in (OUT, X1, X1 + "/u"): os.makedirs(d, exist_ok=True)
sig = lambda z: 1 / (1 + np.exp(-z))
def dump(res, name=None):
    res["exploration"] = "E1"; json.dump(res, open(f"{OUT}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("EXE1 DONE", name or TASK, flush=True)
def other_ns(fname):
    ns = {"__file__": HERE + "/" + fname, "__name__": "e1_" + fname}; s_ = open(HERE + "/" + fname).read(); exec(compile(s_[:s_.index('\nres = {"task": TASK}')], HERE + "/" + fname, "exec"), ns); return ns
# ------------------------------------------------------------------ coefficients and tile bags
def coefs():
    """A[(j, k, i)] = (a = w / sd, c0 = mu . a, b0): j repeat index 0..9, k outer fold 1..10, i inner fold (0 = outer model)"""
    A = {}
    for f in sorted(glob.glob(X1 + "/coef/*.csv")):
        D = pd.read_csv(f); uc = [c for c in D.columns if c.startswith("u")]
        for (r, k, i), g in D.groupby(["rep", "fold", "inner_fold"]):
            g = g.set_index("kind"); w = g.loc["w", uc].values.astype(float); mu = g.loc["mu", uc].values.astype(float); sd = g.loc["sd", uc].values.astype(float)
            a = np.where(sd > 0, w / np.where(sd > 0, sd, 1), 0.0); A[(int(r) - 1, int(k), int(i))] = (a, float(mu @ a), float(g.intercept.iloc[0]))
    assert len(A) == 10 * 10 + 10 * 10 * 5, len(A); return A
def inner_fold_map():
    """IF[s, j, k-1] = inner fold of sample s in split (repeat j, outer fold k) (0 where s is in fold k)"""
    IF = np.zeros((n, 10, 10), int)
    for f in sorted(glob.glob(HZ + "/inner/img_rep_*_fold_*.csv")):
        D = pd.read_csv(f, dtype={"Sample": str}); r = int(D.rep.iloc[0]); k = int(D.fold.iloc[0]); IF[ix[D.Sample].values, r - 1, k - 1] = D.inner_fold.values
    return IF
def px_table():
    P = pd.concat([pd.read_csv(f, dtype={"key": str}) for f in sorted(glob.glob(X1 + "/px/chunk_*.csv"))]); return P.sort_values(["key", "npz", "tile"]).reset_index(drop=True)
res = {"task": TASK}
# ================================================================== u: per-tile score u_t = w . z(e_t) for every sample under every (repeat, fold) model
if TASK.startswith("u_"):
    NCU = 24; cu = int(TASK[2:]); O, I, FO = load("their"); A = coefs(); IF = inner_fold_map(); BI = pd.read_csv(M + "/bag_index.csv", dtype={"sample_id": str}).set_index("sample_id")
    IL = {kk: dict(zip(v.index.values, v.values)) for kk, v in I["L-IMG"].items()}
    lg_ = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12))); dmax = {"outer": 0.0, "inner": 0.0}; nchk = {"outer": 0, "inner": 0}
    for s in range(cu, n, NCU):
        if os.path.exists(f"{X1}/u/{ids[s]}.npz"): continue
        E_ = np.load(BI.loc[ids[s], "npz_path"])["embeddings"].astype(np.float64); U = np.zeros((E_.shape[0], 10, 10), np.float32); B0 = np.zeros((10, 10))
        for j in range(10):
            for k in range(1, 11):
                i = 0 if FO[s, j] == k else int(IF[s, j, k - 1]); assert FO[s, j] == k or i > 0, (s, j, k)
                a, c0, b0 = A[(j, k, i)]; u = E_ @ a - c0; U[:, j, k - 1] = u; B0[j, k - 1] = b0; z = b0 + u.mean()
                ref = O["L-IMG"][s, j] if i == 0 else IL[(j + 1, k)].get(s, np.nan); key = "outer" if i == 0 else "inner"
                if 1e-9 < ref < 1 - 1e-9: dmax[key] = max(dmax[key], abs(z - lg_(ref))); nchk[key] += 1
        np.savez(f"{X1}/u/{ids[s]}.npz", U=U, B0=B0)
    res["check"] = {"max_abs_logit_diff_vs_stored": dmax, "n_checked": nchk}; assert max(dmax.values()) < 1e-4, dmax
    json.dump(res, open(f"{X1}/u/check_{cu:02d}.json", "w")); print("EXE1 DONE", TASK, flush=True)
# ================================================================== prep: covariates per sample / endoscopy / patient and tile summaries (row-level, cluster only)
elif TASK == "prep":
    ns7 = other_ns("rv7_round7.py"); ns7d = other_ns("rv7d_sensfirst.py")
    S = setup(); es = S["es"]; E = S["E"]; rows = S["rows"]; C = {}
    SDS = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids)
    C["tissue_tiles"] = SDS.tissue_tiles.values.astype(float); C["n_slides"] = SDS.n_slides.values.astype(float); C["scanner_C13210"] = (SDS.scanner.values == "C13210").astype(float)
    PX = px_table(); mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(set(ids))].sort_values(["cnv", "npz"])
    R55 = pd.concat([pd.read_csv(f, dtype={"key": str, "created": str}) for f in sorted(glob.glob(M + "/review_round7/px_disc/chunk_*.csv"))])
    created = pd.to_datetime(R55.created, errors="coerce").groupby(R55.key.values).min().reindex(ids)
    L = ns7["db_link"](); sdate = pd.to_datetime(pd.Series(ids).map(L["pdate"]).values)
    C["slide_age_years"] = np.asarray(((created.values - sdate.values) / np.timedelta64(1, "D")) / 365.25, float)
    C["biopsy_year"] = np.asarray(sdate.year + (sdate.dayofyear - 1) / 365.25, float)
    fdu = sheet().drop_duplicates("combined_name").set_index("combined_name"); BATCH = pd.Series(ids).map(fdu.Batch).values
    bc = pd.Series(y).groupby(BATCH).agg(lambda s_: s_.nunique()); C["batch_single_class"] = (pd.Series(BATCH).map(bc).values == 1).astype(float)
    bp = pd.Series(y).groupby(BATCH).mean(); C["batch_progressor_share"] = pd.Series(BATCH).map(bp).values.astype(float)
    GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); C["grade_sheet"] = GR
    P53 = pd.to_numeric(SM["P53 IHC"], errors="coerce").values; C["p53_aberrant"] = np.where(np.isfinite(P53), (P53 == 1).astype(float), np.nan); C["p53_available"] = np.isfinite(P53).astype(float)
    C["age"] = SM.age.values.astype(float); C["sex_M"] = SM.sex_M.values.astype(float)
    dm = pd.read_csv(ns7["SWG"] + "/Demographics_full.csv", dtype=str); dmi = {}
    for _, r_ in dm.iterrows():
        for c_ in ("Study Number", "Alternate Study Number"):
            v_ = r_[c_].strip() if isinstance(r_[c_], str) else ""
            if v_ and v_ not in dmi: dmi[v_] = r_
    C["prague_M"] = np.array([pd.to_numeric(pd.Series([dmi[s_]["Maximal"]]), errors="coerce").iloc[0] if s_ in dmi else np.nan for s_ in SM.study_number.astype(str)], float)
    # Barrett's database: highest confirmed grade in reports within 14 days of the sample date; later grade group after the last sequenced endoscopy
    pt = L["pt"]; dbg = np.full(n, np.nan); linked = np.zeros(n)
    for s_ in range(n):
        pid = L["link"].get(pat[s_])
        if pid is None or pd.isna(sdate[s_]): continue
        linked[s_] = 1; a_ = pt[(pt.pid == pid) & ((pt.d - sdate[s_]).abs() <= pd.Timedelta(days=14))]
        if a_.g.notna().any(): dbg[s_] = a_.g.max()
    C["db_grade_at_endoscopy"] = dbg; C["db_linked"] = linked
    LD = ns7["later_db"](L); lg_grp = {}
    for p_ in set(pat):
        r_ = LD.loc[p_]
        lg_grp[p_] = "not linked" if not r_.linked else ("HGD+" if (r_.max_after >= 3 or r_.hgdtab_after > 0) else ("LGD" if r_.max_after == 2 else ("NDBE/ID" if np.isfinite(r_.max_after) else "no later report")))
    LGRP = np.array([lg_grp[p_] for p_ in pat])
    # slide notes and same-block status
    sm = pd.read_csv(ns7["SWG"] + "/slide_matching.csv", dtype=str).drop_duplicates("Slide file").set_index("Slide file"); mpn = mp.join(sm, on="base")
    note = mpn["Slide Notes"].fillna("").str.lower() if "Slide Notes" in mpn else pd.Series("", index=mpn.index)
    C["note_z_levels"] = (note.str.contains("z level")).groupby(mpn.cnv.values).max().reindex(ids).fillna(False).values.astype(float)
    C["note_marker"] = (note.str.contains("marker")).groupby(mpn.cnv.values).max().reindex(ids).fillna(False).values.astype(float)
    SB = ns7d["same_block_labels"](); C["same_block"] = (SB == "same case, same block").astype(float)
    # pixel statistics per sample (tile order = bag order: slides sorted by npz, stored coordinate order)
    BI = pd.read_csv(M + "/bag_index.csv", dtype={"sample_id": str}).set_index("sample_id"); PXg = {k_: g for k_, g in PX.groupby("key")}
    lapq = float(np.nanpercentile(PX.lapvar[PX.white <= 0.5], 10)); TILE = {}
    for s_ in range(n):
        g = PXg[ids[s_]]; assert len(g) == int(BI.loc[ids[s_], "n_tiles"]), (ids[s_], len(g))
        bg = g.white.values > 0.5; edge = ~bg & ((g.mask_nb.fillna(1).values < 0.8) | (g.white.values > 0.25)); ink = g.ink.values > 0.02; blur = ~bg & (g.lapvar.values < lapq)
        TILE[s_] = {"bg": bg, "edge": edge, "ink": ink, "blur": blur}
        C.setdefault("bg_share", np.zeros(n))[s_] = bg.mean(); C.setdefault("edge_share", np.zeros(n))[s_] = edge.mean(); C.setdefault("ink_share", np.zeros(n))[s_] = ink.mean()
        C.setdefault("blur_share", np.zeros(n))[s_] = blur.mean(); C.setdefault("lapvar_median", np.zeros(n))[s_] = np.median(g.lapvar.values[~bg]) if (~bg).any() else np.nan
        C.setdefault("sat_mean", np.zeros(n))[s_] = g.sat.values[~bg].mean() if (~bg).any() else np.nan; C.setdefault("dark_share", np.zeros(n))[s_] = g.dark.values.mean()
    C["log_tissue_tiles"] = np.log(C["tissue_tiles"])
    # tile summaries of the score under every (repeat, fold) model: TS[s, j, k-1, q]
    TSN = ["mean", "b0", "median", "trim10", "nonbg_mean", "core_mean", "noink_mean", "sd", "p90", "max", "top16_mean", "rest_mean", "share_pos", "gini_pos",
           "n_u_gt2", "n_u_gt5", "frac_u_gt2", "bg_contrib", "edge_contrib", "ink_contrib", "blur_contrib", "top16_contrib", "pos_sum", "conc_top16_of_pos"]
    TS = np.full((n, 10, 10, len(TSN)), np.nan, np.float32)
    def gini(v):
        v = np.sort(v[v > 0]); m_ = len(v)
        return float((2 * np.arange(1, m_ + 1) - m_ - 1) @ v / (m_ * v.sum())) if m_ > 1 and v.sum() > 0 else 0.0
    for s_ in range(n):
        z = np.load(f"{X1}/u/{ids[s_]}.npz"); U = z["U"].astype(np.float64); B0 = z["B0"]; t_ = TILE[s_]; N_ = U.shape[0]; core = ~t_["bg"] & ~t_["edge"]
        for j in range(10):
            for k in range(10):
                u = U[:, j, k]; o_ = np.sort(u)[::-1]; nb = ~t_["bg"]; kk = min(16, N_); pos = np.clip(u, 0, None)
                v = [u.mean(), B0[j, k], np.median(u), np.mean(np.sort(u)[int(0.1 * N_):N_ - int(0.1 * N_)]), u[nb].mean() if nb.any() else u.mean(), u[core].mean() if core.any() else (u[nb].mean() if nb.any() else u.mean()),
                     u[~t_["ink"]].mean() if (~t_["ink"]).any() else u.mean(), u.std(), np.percentile(u, 90), u.max(), o_[:kk].mean(), o_[kk:].mean() if N_ > kk else np.nan, (u > 0).mean(), gini(u),
                     (u > 2).sum(), (u > 5).sum(), (u > 2).mean(), u[t_["bg"]].sum() / N_, u[t_["edge"]].sum() / N_, u[t_["ink"]].sum() / N_, u[t_["blur"]].sum() / N_, o_[:kk].sum() / N_, pos.sum() / N_,
                     np.sort(pos)[::-1][:kk].sum() / pos.sum() if pos.sum() > 0 else np.nan]
                TS[s_, j, k] = v
    PRE_ = {"C": C, "LGRP": LGRP, "TS": TS, "TSN": TSN, "lapvar_p10_nonbg": lapq, "same_block_label": SB}
    pickle.dump(PRE_, open(X1 + "/prep.pkl", "wb"))
    res["prep"] = {"n_samples": int(n), "tiles": int(len(PX)), "covariates": sorted(C), "missing_pre_event": {k_: int(np.isnan(v_[PRE]).sum()) for k_, v_ in C.items()},
                   "tile_summaries": TSN, "lapvar_p10_nonbg": r3(lapq),
                   "tile_flags_share_all_tiles": {k_: r3(np.mean(np.concatenate([TILE[s_][k_] for s_ in range(n)]))) for k_ in ("bg", "edge", "ink", "blur")},
                   "later_db_group_pre_event_patients": {k_: m5(v_) for k_, v_ in pd.Series([lg_grp[p_] for p_ in sorted(set(pat[PRE]))]).value_counts().items()}}
    dump(res)
else:
    exec(open(HERE + "/ex_e1_analysis.py").read())
