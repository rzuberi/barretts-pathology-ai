"""P33 payoff test: are 13-d interpretable tile-grade summaries as good as ABMIL for the imminent-dysplasia tasks?
Tasks T2a, T2b (case bags: summaries pooled per case by tile-weighted mean), T3a, T3b, T3a_bio, T3b_bio (single slide).
Arm: logistic on the 13-d summary (+ age, grade as in the baseline), same queue folds (abmil_clf.patient_folds seed 0),
vs the queue's saved ABMIL image OOF (mean of 3 seeds); patient-clustered 2,000-boot paired deltas.
Env: P33DIR (has tile_summary_all.parquet), OUTDIR."""
import glob, json, os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; Q = f"{T}/feasibility/erin_fusion/queue/results"; TD = f"{T}/feasibility/erin_fusion/tasks"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
sys.path.insert(0, T + "/scripts"); from abmil_clf import patient_folds
S = pd.read_parquet(os.environ["P33DIR"] + "/tile_summary_all.parquet").set_index("h5"); feat_cols = [c for c in S.columns if c.startswith(("frac_", "mean_", "log_"))]
def summarise_case(hl):
    ps = [p for p in hl.split("|") if p in S.index]
    if not ps: return None
    w = np.expm1(S.loc[ps, "log_ntiles"].values); V = S.loc[ps, feat_cols].values; v = (V * w[:, None]).sum(0) / w.sum(); v[-1] = np.log1p(w.sum()); return v
res = {}
for t in ("T2a", "T2b", "T3a", "T3b", "T3a_bio", "T3b_bio"):
    f = f"{TD}/{t}.csv"
    if not os.path.exists(f): continue
    d = pd.read_csv(f, dtype=str); d["y"] = d.y.astype(int)
    files = sorted(glob.glob(f"{Q}/img_{t}_f*_s*.json"))
    if len(files) < 15: print(t, "no ABMIL OOF"); continue
    preds = {}
    for fp in files:
        for k, v in json.load(open(fp))["preds"].items(): preds.setdefault(k, []).append(float(v))
    d["abmil"] = d.sample_id.map(lambda k: np.mean(preds[k]) if k in preds else np.nan); d["V"] = [summarise_case(h) for h in d.h5_list]
    d = d[d.abmil.notna() & d.V.notna()].reset_index(drop=True); X = np.stack(d.V.values); print(t, "units with both summaries and ABMIL:", len(d), flush=True)
    keys = list(d.sample_id); pats = dict(zip(keys, d.anon_id)); y = dict(zip(keys, d.y)); folds = patient_folds(keys, pats, y, 5, seed=0); idx = {k: i for i, k in enumerate(keys)}
    age = pd.to_numeric(d.age, errors="coerce").fillna(pd.to_numeric(d.age, errors="coerce").median()).values; g = (d.grade.isin(["IND"]) if not t.startswith("T3") else d.grade.eq("NDBE")).astype(float).values
    Xa = np.column_stack([X, age, g]); yv = d.y.values; p = np.zeros(len(d)); pb = np.zeros(len(d))
    for fi in range(5):
        te = np.array([idx[k] for k in folds[fi]]); tr = np.array([idx[k] for j in range(5) if j != fi for k in folds[j]])
        mu, sd = Xa[tr].mean(0), Xa[tr].std(0) + 1e-9; p[te] = LogisticRegression(C=0.5, max_iter=3000).fit((Xa[tr] - mu) / sd, yv[tr]).predict_proba((Xa[te] - mu) / sd)[:, 1]
        Xb = np.column_stack([age, g]); mub, sdb = Xb[tr].mean(0), Xb[tr].std(0) + 1e-9; pb[te] = LogisticRegression(max_iter=3000).fit((Xb[tr] - mub) / sdb, yv[tr]).predict_proba((Xb[te] - mub) / sdb)[:, 1]
    a = d.abmil.values; gg = d.anon_id.values; up = np.unique(gg); idx_of = {u: np.where(gg == u)[0] for u in up}; rng = np.random.RandomState(0); B = []
    while len(B) < NB:
        s = np.concatenate([idx_of[u] for u in rng.choice(up, len(up))])
        if len(set(yv[s])) < 2: continue
        B.append(s)
    ci = lambda x, z=None: [round(float(np.percentile([roc_auc_score(yv[s], x[s]) - (roc_auc_score(yv[s], z[s]) if z is not None else 0) for s in B], q)), 4) for q in (2.5, 97.5)]
    fused = np.zeros(len(d))
    for fi in range(5):
        te = np.array([idx[k] for k in folds[fi]]); za = (a[te] - a[te].mean()) / (a[te].std() + 1e-9); zp = (p[te] - p[te].mean()) / (p[te].std() + 1e-9); fused[te] = (za + zp) / 2
    res[t] = {"n": int(len(d)), "pos": int(yv.sum()), "tile_summary_logistic": {"auroc": round(float(roc_auc_score(yv, p)), 4), "ci": ci(p)}, "abmil_image": {"auroc": round(float(roc_auc_score(yv, a)), 4), "ci": ci(a)},
              "baseline_grade_age": round(float(roc_auc_score(yv, pb)), 4), "delta_tile_minus_abmil": ci(p, a), "delta_tile_minus_baseline": ci(p, pb), "fused_tile_abmil": {"auroc": round(float(roc_auc_score(yv, fused)), 4), "delta_vs_abmil": ci(fused, a)},
              "coef_top": None}
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9; lr = LogisticRegression(C=0.5, max_iter=3000).fit((Xa - mu) / sd, yv); names = feat_cols + ["age", "grade_flag"]; top = np.argsort(-np.abs(lr.coef_[0]))[:5]; res[t]["coef_top"] = {names[i]: round(float(lr.coef_[0][i]), 3) for i in top}
    print(t, json.dumps(res[t]), flush=True)
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
