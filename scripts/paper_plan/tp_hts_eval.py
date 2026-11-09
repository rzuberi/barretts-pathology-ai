"""Two papers, agent HTS (docs/two_papers_HTS.md): step 3 of 3, aggregates from tp_hts_fit.R's held-out scores. CPU only, no pixels.
Metric as every L-IMG number: fold-stratified AUROC (fsa, tp_p1b_main.py), stored 10 repeats x 10 patient-grouped outer folds, 571 pre-event samples,
2,000-draw patient bootstrap (RandomState(0)), every delta paired on the same draws. tp_p1b_main.py's module body is executed unchanged with a task
name matching none of its branches (as tp_art_main.py does).
Outputs (aggregates only, no identifiers, cells < 5 as "<5") -> results/paper_final/two_papers_HTS/{A_noretrain,B_retrain,C_concentration}.json.
HTS_NB=k: smoke mode (k bootstrap draws; JSON written aside under killcoyne_mm/two_papers_HTS/smoke_json)."""
import os, sys, json, glob, numpy as np, pandas as pd
PP = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"; sys.path.insert(0, PP)
os.environ["TASK"] = "hts_none"
exec(compile(open(PP + "/tp_p1b_main.py").read(), PP + "/tp_p1b_main.py", "exec"))   # definitions only
W = M + "/two_papers_HTS"; OUTH = T + "/results/paper_final/two_papers_HTS"; NBX = NB
if os.environ.get("HTS_NB"): NBX = int(os.environ["HTS_NB"]); OUTH = W + "/smoke_json"
os.makedirs(OUTH, exist_ok=True)
def hdump(res, name):
    res["agent"] = "HTS"; json.dump(res, open(f"{OUTH}/{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("HTS DONE", name, flush=True)
def rd(pref):
    fo = sorted(glob.glob(f"{W}/fit/{pref}_rep_[0-9][0-9].csv")); assert len(fo) == 10, (pref, len(fo))
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); assert len(D) == 10 * n and D.groupby("rep").size().eq(n).all()
    Of = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int); assert (Of == FO0).all()
    return lambda c: D.pivot_table(index="Sample", columns="rep", values=c).reindex(ids)[REPS].values
def tmr(o, B):   # paired top-k minus random-k at the same k (does the model's own ranking beat chance?)
    return {f"k{k}": dstat(o[f"top_{k}"] - o[f"rnd_{k}"], B[f"top_{k}"] - B[f"rnd_{k}"]) for k in (1, 4, 16, 64) if f"top_{k}" in o and f"rnd_{k}" in o}
def block(O, ref, bb):
    r, o, B = hblock(O, ROWS, bb, keep=True)
    r["delta_vs_" + ref] = {m: dstat(o[m] - o[ref], B[m] - B[ref]) for m in O if m != ref}; r["top_minus_random"] = tmr(o, B); return r
bb = boots(ROWS); PART = os.environ.get("PART", "ABC")
fs = glob.glob(f"{T}/feasibility/paper_plan/killcoyne_mm/two_papers_ART/refit/clean/outer_rep_*.csv"); assert len(fs) == 10
SC = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fs]).pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values
NK = rd("A")("n_kept")[:, 0]
base = {"population": "pre-event samples", "n_kept_tiles_pre_event": {"median": float(np.median(NK[ROWS])), "min": int(NK[ROWS].min()),
        "samples_lt64_kept": cnt((NK[ROWS] < 64).sum()), "samples_lt16_kept": cnt((NK[ROWS] < 16).sum())}}
# ---------------------------------------------------------------- A: same fitted model, tile subsets
if "A" in PART:
    g = rd("A"); ALL = g("all"); chkA = float(np.nanmax(np.abs(ALL - np.log(SC / (1 - SC)))))
    O = {"all": ALL}; RDW = {}
    for k in (1, 4, 16, 64):
        for rl in ("top", "bot", "ext"): O[f"{rl}_{k}"] = g(f"{rl}_{k}")
        RDW[f"rnd_{k}"] = [g(f"rnd_{k}_d{d:02d}") for d in range(1, 21)]
    def ablock(O, RDW, ref, bb):   # as hblock, but a random-k AUROC is the mean of the AUROCs of 20 single draws (on every bootstrap draw too)
        f = lambda b: {**{m: fsa(O[m], b) for m in O}, **{m: float(np.mean([fsa(x, b) for x in L])) for m, L in RDW.items()}}
        obs = f(ROWS); B = pd.DataFrame([f(b) for b in bb])
        r = {"n_samples": cnt(len(ROWS)), "n_progressor_samples": cnt(y[ROWS].sum()), "n_patients": cnt(len(set(pat[ROWS]))),
             "n_progressor_patients": cnt(len(set(pat[ROWS][y[ROWS] == 1]))), "auroc": {m: {"value": r3(obs[m]), "ci95": ci2(B[m])} for m in obs}}
        r["delta_vs_" + ref] = {m: dstat(obs[m] - obs[ref], B[m] - B[ref]) for m in obs if m != ref}; r["top_minus_random"] = tmr(obs, B); return r
    RA = dict(base, check_all_tiles_vs_stored_clean_logit_max_abs=chkA, overall=ablock(O, RDW, "all", bb),
              note=("No retraining. Each held-out sample is scored by the outer-fold model that held it out (stored clean L-IMG coefficients), "
                    "score = mean tile score u_t = c0 + w.e_t over a subset of its kept tiles. top_k / bot_k = k highest / lowest u_t; ext_k = k tiles with the "
                    "largest |u_t - sample median u|; rnd_k = k random tiles, AUROC averaged over 20 separate draws (in each bootstrap draw too); all = all kept tiles (= the stored clean L-IMG). "
                    "k capped at the sample's kept-tile count. fsa AUROC, 2,000-draw patient bootstrap, paired delta vs all."))
    assert chkA < 1e-5, chkA
    hdump(RA, "A_noretrain")
# ---------------------------------------------------------------- B: select with stage 1, refit on selected tiles
if "B" in PART:
    g = rd("B"); FULL = g("full"); chkB = float(np.nanmax(np.abs(FULL - SC))); chkS = float(np.nanmax(np.abs(g("stored") - SC)))
    O = {"full": SC}
    for rl in ("top", "ext", "rnd"):
        for k in (4, 16, 64): O[f"{rl}_{k}"] = g(f"{rl}_{k}")
    RB = dict(base, check_full_refit_vs_stored_clean_pred_max_abs=chkB, check_stored_column_max_abs=chkS, overall=block(O, "full", bb),
              note=("Retraining. In each outer fold, stage 1 = that fold's stored clean L-IMG fit (training patients only) scores every tile of every sample; "
                    "each sample's mean embedding is taken over its top k tiles by u_t (top_k), its k tiles with the largest |u_t - sample median| (ext_k), "
                    "or k random tiles (rnd_k, one draw); the L-IMG recipe (glmnet alpha 0.9, class-error-minimum lambda from 10 x 5 patient-grouped CV, "
                    "training-row z-scaling) is refitted on the training rows and scores the held-out rows. full = stored clean L-IMG (a refit on all kept "
                    "tiles reproduces it, see check). fsa AUROC, 2,000-draw patient bootstrap, paired delta vs full."))
    assert chkB < 1e-5, chkB
    hdump(RB, "B_retrain")
# ---------------------------------------------------------------- C: concentration of the tile contributions
if "C" in PART:
    g = rd("A"); RC = dict(base, note=("Per held-out sample and repeat: share of the summed |u_t - c0| (share_c0, as specified) and of the summed "
                                       "|u_t - sample median u| (share_med) carried by the top 10% of kept tiles (ceil(0.1 n), ranked by the same quantity); "
                                       "averaged over the 10 repeats per sample; 0.10 = all tiles contribute equally. Difference = progressor minus "
                                       "non-progressor median, 2,000-draw patient bootstrap."))
    for c in ("share_c0", "share_med"):
        v = g(c).mean(1); d = {}
        for lab, yy in (("progressor", 1), ("non_progressor", 0)):
            x = v[ROWS][y[ROWS] == yy]; d[lab] = {"samples": cnt(len(x)), "median": r3(np.median(x)), "q1": r3(np.quantile(x, 0.25)), "q3": r3(np.quantile(x, 0.75)),
                                                  "min": r3(x.min()), "max": r3(x.max())}
        md = lambda b: np.median(v[b][y[b] == 1]) - np.median(v[b][y[b] == 0])
        d["median_difference_p_minus_np"] = dstat(md(ROWS), np.array([md(b) for b in bb])); RC[c] = d
    hdump(RC, "C_concentration")
