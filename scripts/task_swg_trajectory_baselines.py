"""2.42 (Astra A10): does current H&E predict the NEXT biopsy's CNV complexity
beyond what the CURRENT CNV state, elapsed time and current grade already
predict? The 2.24b secondary (rho 0.161) had no persistence baseline.

Transitions (sample i -> i+1 within patient) with cx known at both ends.
Arms (Ridge, frozen release patient folds):
  persist      : [cx_i, dt]
  persist_grade: [cx_i, dt, onehot(grade_i)]
  hist_only    : PCA64(emb_i)                       (the 2.24b arm)
  hist_plus    : PCA64(emb_i) + [cx_i, dt, onehot(grade_i)]
Targets: cx_next and delta = cx_next - cx_i. Uncertainty: patient-clustered
bootstrap on incremental rho (hist_plus - persist_grade); row-permutation
null for the increment as a secondary.
"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
OUT = os.environ.get("OUTDIR", ".")
N_BOOT, N_PERM = 2000, 2000

man = pd.read_csv(F + "/training_manifest.csv", dtype=str)
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).merge(man, left_on="SampleID", right_on="sample_id")
coh["Date"] = pd.to_datetime(coh["Date"], errors="coerce")
fold_of = dict(zip(man["patient_id"], man["fold_id_rep01"].astype(int)))
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str)
key = next(c for c in cx.columns if c.lower() in ("sampleid", "sample_id"))
num_cols = [c for c in cx.columns if c != key]
cx_val = {r[key]: pd.to_numeric(r[num_cols], errors="coerce").mean() for _, r in cx.iterrows()}
uidx = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str)
uidx = uidx[uidx["status"] == "ok"]
npz_of = dict(zip(uidx["sample_id"], uidx["npz_path"]))
emb = {}
for sid in coh["SampleID"]:
    if sid in npz_of:
        try: emb[sid] = np.asarray(np.load(npz_of[sid])["slide_embedding_mean"])
        except Exception: pass
coh = coh[coh["SampleID"].isin(emb)]
GR = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}

rows = []
for pid, g in coh.sort_values("Date").groupby("patient_id"):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 1):
        a, b = g.loc[i], g.loc[i + 1]
        ca, cb = cx_val.get(a["SampleID"], np.nan), cx_val.get(b["SampleID"], np.nan)
        if pd.isna(ca) or pd.isna(cb) or pd.isna(a["Date"]) or pd.isna(b["Date"]): continue
        dt = max((b["Date"] - a["Date"]).days / 365.25, 1 / 365.25)
        rows.append({"pid": pid, "emb": emb[a["SampleID"]], "cx": float(ca), "dt": dt,
                     "grade": GR.get(str(a["Label"]).strip(), -1), "cx_next": float(cb)})
n = len(rows)
pid = np.array([r["pid"] for r in rows]); fm = np.array([fold_of.get(p, -1) for p in pid])
E = np.vstack([r["emb"] for r in rows])
B = np.column_stack([[r["cx"] for r in rows], [r["dt"] for r in rows]])
G = np.zeros((n, 5))
for i, r in enumerate(rows):
    if r["grade"] >= 0: G[i, r["grade"]] = 1
y_next = np.array([r["cx_next"] for r in rows]); y_delta = y_next - B[:, 0]
print(f"transitions={n} patients={len(set(pid))} folds={sorted(set(fm))}", flush=True)

def oof(Xparts, y, use_pca):
    out = np.zeros(n)
    for f in sorted(set(fm)):
        tr, te = fm != f, fm == f
        if use_pca:
            pe = Pipeline([("s", StandardScaler()), ("p", PCA(min(64, int(tr.sum()) - 1)))]).fit(Xparts[0][tr])
            Xtr = np.hstack([pe.transform(Xparts[0][tr])] + [x[tr] for x in Xparts[1:]])
            Xte = np.hstack([pe.transform(Xparts[0][te])] + [x[te] for x in Xparts[1:]])
        else:
            Xtr = np.hstack([x[tr] for x in Xparts]); Xte = np.hstack([x[te] for x in Xparts])
        sc = StandardScaler().fit(Xtr)
        r = Ridge(alpha=10.0).fit(sc.transform(Xtr), y[tr])
        out[te] = r.predict(sc.transform(Xte))
    return out

ARMS = {"persist": ([B], False), "persist_grade": ([B, G], False),
        "hist_only": ([E], True), "hist_plus": ([E, B, G], True)}
res = {"_meta": {"n_transitions": n, "n_patients": len(set(pid)), "n_boot": N_BOOT, "n_perm": N_PERM}}
upats = sorted(set(pid)); rows_of = {p: np.where(pid == p)[0] for p in upats}
for tname, y in (("cx_next", y_next), ("cx_delta", y_delta)):
    P = {a: oof(x, y, pca) for a, (x, pca) in ARMS.items()}
    blk = {"rho": {a: round(float(spearmanr(P[a], y)[0]), 4) for a in ARMS}}
    def inc(idx, a="hist_plus", b="persist_grade"):
        return spearmanr(P[a][idx], y[idx])[0] - spearmanr(P[b][idx], y[idx])[0]
    rng = np.random.RandomState(0); boots = []
    for _ in range(N_BOOT):
        sp = rng.randint(0, len(upats), len(upats))
        idx = np.concatenate([rows_of[upats[j]] for j in sp]); boots.append(inc(idx))
    # row-permutation null: break the histology<->target link, keep baseline covariates
    rng = np.random.RandomState(1); nulls = []
    for _ in range(N_PERM):
        pe = rng.permutation(n)
        nulls.append(spearmanr(P["hist_plus"][pe], y)[0] - spearmanr(P["persist_grade"], y)[0])
    obs = inc(np.arange(n))
    blk["increment_hist_plus_minus_persist_grade"] = {
        "observed": round(float(obs), 4),
        "patient_clustered_ci": [round(float(np.percentile(boots, 2.5)), 4),
                                 round(float(np.percentile(boots, 97.5)), 4)],
        "perm_p_oof_shuffle": round((1 + sum(v >= obs for v in nulls)) / (N_PERM + 1), 5)}
    blk["hist_only_minus_persist"] = {
        "observed": round(float(inc(np.arange(n), "hist_only", "persist")), 4)}
    res[tname] = blk
    print(tname, blk, flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
