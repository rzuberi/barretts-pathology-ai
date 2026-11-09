"""One paper, Extended Data Fig. 4b: per tile-set version of the H&E model, sensitivity in progressor samples and specificity in non-progressor
samples at the high cut-off of Extended Data Fig. 1 (chosen in each training fold on the training patients' inner out-of-fold predictions:
82% of training non-progressor samples below it; tp_kcf_figs.py), on the 571 pre-event samples, mean of 10 repeats.
Also the AUROC difference of every version against the artefact-free model used in the paper (clean), for the Extended Data Fig. 4a value column.
2,000-draw patient bootstrap (tp_p1b_main.boots, RandomState(0)); paired differences on the same draws.
Versions: all tiles (stored L-IMG), random tiles removed (rand), glass/edge/pen removed but blur kept (clean_noblur), all artefact removed (clean),
artefact share alone (no model; cut-off = 82nd percentile of the training non-progressor samples' shares).
ART_NB=k: smoke (k draws, JSON written aside). Runs on the cluster; writes aggregates only -> two_papers_ART/ed4.json."""
import os, glob, numpy as np, pandas as pd
PP = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"
os.environ["TASK"] = "ed4_defs"   # tp_art_main definitions only (no ART branch matches)
exec(compile(open(PP + "/tp_art_main.py").read(), PP + "/tp_art_main.py", "exec"))
def rd2(name):
    fo = sorted(glob.glob(f"{A}/refit/{name}/outer_rep_*.csv")); fi = sorted(glob.glob(f"{A}/refit/{name}/inner_rep_*.csv")); assert len(fo) == 10 and len(fi) == 10, name
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); Of = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int)
    assert (Of == FO0).all(); X = D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values
    Di = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fi])
    return X, {(r, k): pd.Series(d_.pred.values, index=ix[d_.Sample].values) for (r, k), d_ in Di.groupby(["rep", "fold"])}
V = {"all": (O0["L-IMG"], I0["L-IMG"])}
for nm, d in (("rand", "rand"), ("blur_kept", "clean_noblur"), ("clean", "clean")): V[nm] = rd2(d)
Q = qc_table(); F, lt = flags(Q); SHARE = F.groupby(Q["sample"].values)["art"].mean().reindex(ids).values.astype(float); assert np.isfinite(SHARE).all()
V["art_share"] = (np.repeat(SHARE[:, None], 10, 1), None)
ORDER = ["all", "rand", "blur_kept", "clean", "art_share"]
HI = {}   # high cut-off per sample and repeat (its outer fold's cut)
for nm in ORDER:
    X, I_ = V[nm]; C = np.zeros((10, 10))
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            if I_ is None: tr = np.where(FO0[:, j] != k)[0]; v = SHARE[tr]; yy = y[tr]
            else: s = I_[(r, k)]; v = s.values; yy = y[s.index.values]
            tl = np.quantile(v[yy == 1], 1 - 0.87); C[j, k - 1] = max(np.quantile(v[yy == 0], 0.82), tl)
    HI[nm] = C[np.arange(10)[None, :], FO0 - 1]
POS = {nm: V[nm][0] >= HI[nm] for nm in ORDER}
def sens(nm, b): p = POS[nm][b]; return float(p[y[b] == 1].mean())
def spec(nm, b): p = POS[nm][b]; return float(1 - p[y[b] == 0].mean())
rows = ROWS; bb = boots(rows)
st = {"auroc": lambda nm, b: fsa(V[nm][0], b), "sens": sens, "spec": spec}
obs = {s: {nm: f(nm, rows) for nm in ORDER} for s, f in st.items()}
B = {s: pd.DataFrame([{nm: f(nm, b) for nm in ORDER} for b in bb]) for s, f in st.items()}
R = {"n_samples": int(len(rows)), "n_progressor_samples": int(y[rows].sum()), "n_draws": len(bb), "versions": ORDER}
for s in st:
    R[s] = {nm: {"value": r3(obs[s][nm]), "ci95": ci2(B[s][nm])} for nm in ORDER}
    R[s + "_minus_clean"] = {nm: dstat(obs[s][nm] - obs[s]["clean"], B[s][nm] - B[s]["clean"]) for nm in ORDER if nm != "clean"}
R["cut_hi_median"] = {nm: r4(np.median(HI[nm])) for nm in ORDER}
R["note"] = ("sens = share of progressor pre-event samples at or above the high cut (mean over repeats); spec = share of non-progressor pre-event samples below it; "
             "cut chosen per outer fold as tp_kcf_figs (82% training specificity, floored at the 87%-sensitivity low cut); fsa AUROC; patient bootstrap, paired on the same draws.")
res["ed4"] = R; adump(res, "ed4")
