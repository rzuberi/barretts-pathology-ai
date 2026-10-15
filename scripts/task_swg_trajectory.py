"""2.24b (joint, Rehan 2026-08-25: "The Barrett's (image+cnv) dataset is a good
candidate for the trajectory idea"): SWG joint morphology+CNV trajectories.

SWG is longitudinally dense (~4.7 samples/patient, H&E AND sWGS per sample) —
the cohort ERIN couldn't be for this idea. Strict pre-event cohort, FROZEN
release folds (fold_id_rep01), so results sit directly beside the release arms.

Arms: last-sample histology / last-sample hist+CNV (snapshots) vs trajectory
features (embedding drift + CNV-complexity slope) vs GRU over joint sequences.
Secondary (Kimi): does current H&E predict the NEXT biopsy's CNV complexity?
"""
import json, os, sys
import h5py, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_clf import bootstrap_auc
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge

F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]

man = pd.read_csv(F + "/training_manifest.csv", dtype=str)
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str)
# pre_event_cohort.PatientID is an integer index; the manifest's patient_id is
# the real ID space — merge on sample_id and use the manifest columns throughout
coh = coh.merge(man, left_on="SampleID", right_on="sample_id", how="inner")
coh["PatientID"] = coh["patient_id"]
coh["Date"] = pd.to_datetime(coh["Date"], errors="coerce")
fold_of = dict(zip(man["patient_id"], man["fold_id_rep01"].astype(int)))
y_of = dict(zip(man["patient_id"], man["y_progressor"].astype(int)))
print(f"strict samples={len(coh)} patients={coh['PatientID'].nunique()}", flush=True)

# CNV complexity per sample (defensive column discovery)
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str)
key = next(c for c in cx.columns if c.lower() in ("sampleid", "sample_id"))
num_cols = [c for c in cx.columns if c != key]
cx_val = {r[key]: pd.to_numeric(r[num_cols], errors="coerce").mean()
          for _, r in cx.iterrows()}
print("cx cols:", num_cols[:5], flush=True)

uidx = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str)
uidx = uidx[uidx["status"] == "ok"]
npz_of = dict(zip(uidx["sample_id"], uidx["npz_path"]))
emb = {}
for i, sid in enumerate(coh["SampleID"]):
    if sid not in npz_of: continue
    try:
        z = np.load(npz_of[sid])
        emb[sid] = np.asarray(z["slide_embedding_mean"])
    except Exception as e:
        print("skip", sid, e, flush=True)
    if i % 100 == 0: print("loaded", i, flush=True)
coh = coh[coh["SampleID"].isin(emb)]
print(f"pooled samples={len(coh)}", flush=True)

seqs = {}
for pid, g in coh.sort_values("Date").groupby("PatientID"):
    if pid not in y_of: continue
    seqs[pid] = [(r.Date, emb[r.SampleID],
                  cx_val.get(r.SampleID, np.nan)) for r in g.itertuples()]
pats = sorted(seqs)
y = np.array([y_of[p] for p in pats])
fm = np.array([fold_of[p] for p in pats])
multi = sum(len(seqs[p]) > 1 for p in pats)
print(f"patients={len(pats)} progressors={int(y.sum())} multi-sample={multi}", flush=True)

def featurise(p):
    s = seqs[p]
    dates, embs, cxs = zip(*s)
    last = np.asarray(embs[-1])
    cx_last = np.nan_to_num(np.array([cxs[-1]]), nan=0.0)
    if len(s) > 1:
        drifts = []
        for i in range(1, len(s)):
            dt = max((dates[i] - dates[i-1]).days / 365.25, 1/365.25)
            drifts.append((np.asarray(embs[i]) - np.asarray(embs[i-1])) / dt)
        mean_drift = np.mean(drifts, axis=0)
        cser = pd.Series([c for c in cxs if pd.notna(c)])
        cx_slope = np.polyfit(range(len(cser)), cser, 1)[0] if len(cser) > 1 else 0.0
        span = (dates[-1] - dates[0]).days / 365.25
    else:
        mean_drift = np.zeros_like(last); cx_slope = 0.0; span = 0.0
    scal = np.array([len(s), span, cx_slope])
    return {"hist_last": last,
            "joint_last": np.concatenate([last, cx_last]),
            "traj_hist": np.concatenate([last, mean_drift, scal[:2]]),
            "traj_joint": np.concatenate([last, mean_drift, cx_last, scal])}

feats = {p: featurise(p) for p in pats}

def cv_arm(view, seed):
    X = np.vstack([feats[p][view] for p in pats])
    oof = np.zeros(len(pats))
    for f in sorted(set(fm)):
        tr, te = fm != f, fm == f
        pl = Pipeline([("s", StandardScaler()), ("p", PCA(min(64, int(tr.sum()) - 1))),
                       ("l", LogisticRegression(C=0.5, class_weight="balanced",
                                                max_iter=4000, random_state=seed))])
        pl.fit(X[tr], y[tr]); oof[te] = pl.predict_proba(X[te])[:, 1]
    return oof

def gru_arm(seed):
    import torch, torch.nn as nn
    L = max(len(seqs[p]) for p in pats)
    oof = np.zeros(len(pats))
    for f in sorted(set(fm)):
        tr, te = np.where(fm != f)[0], np.where(fm == f)[0]
        pca = PCA(min(64, len(tr) - 1)).fit(np.vstack([feats[pats[i]]["hist_last"] for i in tr]))
        def tens(i):
            s = seqs[pats[i]]
            steps = [np.concatenate([pca.transform(np.asarray(e)[None])[0],
                                     [0.0 if pd.isna(c) else float(c)]]) for _, e, c in s]
            pad = np.zeros((L, len(steps[0]))); pad[-len(steps):] = np.stack(steps)
            return pad
        Xtr = torch.tensor(np.stack([tens(i) for i in tr]), dtype=torch.float32)
        Xte = torch.tensor(np.stack([tens(i) for i in te]), dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.float32)
        torch.manual_seed(seed)
        gru = nn.GRU(Xtr.shape[-1], 32, batch_first=True); head = nn.Linear(32, 1)
        opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=1e-3)
        w = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        for _ in range(80):
            opt.zero_grad(); _, h = gru(Xtr)
            loss = nn.functional.binary_cross_entropy_with_logits(
                head(h[-1]).squeeze(-1), ytr, pos_weight=w)
            loss.backward(); opt.step()
        with torch.no_grad():
            _, h = gru(Xte); oof[te] = torch.sigmoid(head(h[-1]).squeeze(-1)).numpy()
    return oof

res = {"_meta": {"n_patients": len(pats), "progressors": int(y.sum()),
                 "multi_sample_patients": int(multi), "folds": "frozen fold_id_rep01",
                 "seeds": SEEDS}}
oof = {}
for arm, fn in [("hist_last", lambda s: cv_arm("hist_last", s)),
                ("joint_last", lambda s: cv_arm("joint_last", s)),
                ("traj_hist", lambda s: cv_arm("traj_hist", s)),
                ("traj_joint", lambda s: cv_arm("traj_joint", s)),
                ("gru_joint", gru_arm)]:
    per = [fn(s) for s in SEEDS]
    oof[arm] = {pats[i]: float(np.mean([p[i] for p in per])) for i in range(len(pats))}
    ref = oof.get("joint_last") if arm not in ("hist_last", "joint_last") else (
        oof.get("hist_last") if arm == "joint_last" else None)
    res[arm] = bootstrap_auc(oof[arm], dict(zip(pats, y)), prob_b=ref)
    print(arm, res[arm], flush=True)

# secondary (Kimi): current H&E -> NEXT biopsy's CNV complexity
pairs = []
for p in pats:
    s = seqs[p]
    for i in range(len(s) - 1):
        if pd.notna(s[i+1][2]):
            pairs.append((p, s[i][1], float(s[i+1][2])))
if len(pairs) > 100:
    gp = np.array([p for p, _, _ in pairs])
    Xp = np.vstack([e for _, e, _ in pairs])
    yp = np.array([c for _, _, c in pairs])
    fmp = np.array([fold_of[p] for p in gp])
    oofp = np.zeros(len(yp))
    for f in sorted(set(fmp)):
        tr, te = fmp != f, fmp == f
        pl = Pipeline([("s", StandardScaler()), ("p", PCA(64)), ("r", Ridge(alpha=10.0))])
        pl.fit(Xp[tr], yp[tr]); oofp[te] = pl.predict(Xp[te])
    rho, pv = spearmanr(oofp, yp)
    res["future_cnv_from_hist"] = {"n_transitions": len(pairs),
                                   "spearman": round(float(rho), 4), "p": float(pv)}
    print("future CNV:", res["future_cnv_from_hist"], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
