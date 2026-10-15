"""2.24: longitudinal trajectory modelling — does the biopsy SEQUENCE beat the
index snapshot? (9/10 blank-slate convergence; novel to the thesis.)

Arms on identical patient folds, progression cohort v3 endpoint:
  index_only  — pooled UNI2 embedding of the index slide(s)
  traj_feats  — index embedding + mean drift/year + last-step delta + scalars
  gru         — GRU over the (per-fold-PCA'd) embedding sequence
Secondary: next-timepoint upgrade (grade >=LGD at t+1 from slides <=t) on the
wider >=2-imaged-timepoint cohort.
"""
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_clf import bootstrap_auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
CACHE = os.environ.get("POOLED", T + "/feasibility/runs/erin_probes/output/erin_pooled_uni2.npz")
SEEDS = [0, 1, 2]
NUM = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}

d = np.load(CACHE, allow_pickle=True)
X, h5s = d["X"], list(d["h5s"])
emb = {h: X[i] for i, h in enumerate(h5s)}
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
m["date"] = pd.to_datetime(m["CollectedOrOrdered"], errors="coerce", dayfirst=True)
m = m[m["h5"].isin(emb) & m["date"].notna()]
coh = pd.read_csv(T + "/labeller/erin_progression_cohort_v3.csv", dtype=str)
coh["index_date"] = pd.to_datetime(coh["index_date"], errors="coerce")
coh["y"] = (coh["progressed_to_HGDplus"] == "True").astype(int)

def patient_seq(anon, cutoff):
    s = m[(m["anon_id"] == anon) & (m["date"] <= cutoff)].sort_values("date")
    if s.empty: return None
    by_tp = [(dt, np.mean([emb[h] for h in g["h5"]], axis=0))
             for dt, g in s.groupby("date")]
    return by_tp

rows = []
for _, r in coh.iterrows():
    seq = patient_seq(r["anon_id"], r["index_date"])
    if seq is None: continue
    rows.append({"anon_id": r["anon_id"], "y": int(r["y"]), "seq": seq})
print(f"progression patients with imaging: {len(rows)}; "
      f">=2 timepoints: {sum(len(r['seq']) > 1 for r in rows)}", flush=True)

def featurise(seq):
    dates, embs = zip(*seq)
    idx_emb = embs[-1]
    if len(embs) > 1:
        drifts, dts = [], []
        for i in range(1, len(embs)):
            dt = max((dates[i] - dates[i-1]).days / 365.25, 1/365.25)
            drifts.append((np.asarray(embs[i]) - np.asarray(embs[i-1])) / dt); dts.append(dt)
        mean_drift = np.mean(drifts, axis=0); last_delta = np.asarray(embs[-1]) - np.asarray(embs[-2])
        span = (dates[-1] - dates[0]).days / 365.25
    else:
        mean_drift = np.zeros_like(idx_emb); last_delta = np.zeros_like(idx_emb); span = 0.0
    scal = np.array([len(embs), span])
    return np.asarray(idx_emb), np.concatenate([idx_emb, mean_drift, last_delta, scal])

X_idx = np.vstack([featurise(r["seq"])[0] for r in rows])
X_trj = np.vstack([featurise(r["seq"])[1] for r in rows])
y = np.array([r["y"] for r in rows])
pats = [r["anon_id"] for r in rows]

def cv_arm(Xa, seed):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(y))
    folds = np.array_split(order, 5)
    oof = np.zeros(len(y))
    for f in folds:
        tr = np.setdiff1d(order, f)
        p = Pipeline([("s", StandardScaler()), ("p", PCA(min(64, len(tr) - 1))),
                      ("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000))])
        p.fit(Xa[tr], y[tr]); oof[f] = p.predict_proba(Xa[f])[:, 1]
    return oof

def gru_arm(seed):
    import torch, torch.nn as nn
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(y)); folds = np.array_split(order, 5)
    oof = np.zeros(len(y)); L = max(len(r["seq"]) for r in rows)
    for f in folds:
        tr = np.setdiff1d(order, f)
        pca = PCA(min(64, len(tr) - 1)).fit(X_idx[tr])
        def tens(i):
            s = [pca.transform(np.asarray(e)[None])[0] for _, e in rows[i]["seq"]]
            pad = np.zeros((L, s[0].shape[0])); pad[-len(s):] = np.stack(s)
            return pad
        Xtr = torch.tensor(np.stack([tens(i) for i in tr]), dtype=torch.float32)
        Xte = torch.tensor(np.stack([tens(i) for i in f]), dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.float32)
        torch.manual_seed(seed)
        gru = nn.GRU(Xtr.shape[-1], 32, batch_first=True); head = nn.Linear(32, 1)
        opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=1e-3)
        w = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        for _ in range(60):
            opt.zero_grad()
            _, h = gru(Xtr); logit = head(h[-1]).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, ytr, pos_weight=w)
            loss.backward(); opt.step()
        with torch.no_grad():
            _, h = gru(Xte); oof[f] = torch.sigmoid(head(h[-1]).squeeze(-1)).numpy()
    return oof

res = {"_meta": {"n_patients": len(rows), "progressors": int(y.sum()),
                 "multi_timepoint": int(sum(len(r["seq"]) > 1 for r in rows)), "seeds": SEEDS}}
oof = {}
for arm, fn in [("index_only", lambda s: cv_arm(X_idx, s)),
                ("traj_feats", lambda s: cv_arm(X_trj, s)),
                ("gru", gru_arm)]:
    per = [fn(s) for s in SEEDS]
    oof[arm] = {pats[i]: float(np.mean([p[i] for p in per])) for i in range(len(y))}
    ref = oof.get("index_only") if arm != "index_only" else None
    res[arm] = bootstrap_auc(oof[arm], dict(zip(pats, y)), prob_b=ref)
    print(arm, res[arm], flush=True)

# secondary: next-timepoint upgrade on the wider imaged cohort
lab = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str)
lab["date"] = pd.to_datetime(lab["CollectedOrOrdered"], errors="coerce", dayfirst=True)
lab["num"] = lab["final_label"].map(NUM)
lab = lab[lab["label_status"].isin(["train_eligible", "adjudicated"])].dropna(subset=["date", "num"])
pairs = []
for anon, g in m.groupby("anon_id"):
    tps = sorted(g["date"].unique())
    for i in range(len(tps) - 1):
        nxt = lab[(lab["anon_id"] == anon) & (lab["date"] > tps[i])].sort_values("date")
        if nxt.empty: continue
        seq = patient_seq(anon, pd.Timestamp(tps[i]))
        if seq is None: continue
        pairs.append({"anon": anon, "seq": seq, "y": int(nxt.iloc[0]["num"] >= 2)})
print(f"upgrade pairs: {len(pairs)} pos={sum(p['y'] for p in pairs)}", flush=True)
if len(pairs) > 200:
    Xi = np.vstack([featurise(p["seq"])[0] for p in pairs])
    Xt = np.vstack([featurise(p["seq"])[1] for p in pairs])
    yu = np.array([p["y"] for p in pairs]); pu = [p["anon"] for p in pairs]
    uniq = sorted(set(pu)); rng = np.random.RandomState(0)
    fold_of = {a: i % 5 for i, a in enumerate(rng.permutation(uniq))}
    fmask = np.array([fold_of[a] for a in pu])
    res["upgrade"] = {}
    o2 = {}
    for arm, Xa in [("index_only", Xi), ("traj_feats", Xt)]:
        oofu = np.zeros(len(yu))
        for f in range(5):
            tr, te = fmask != f, fmask == f
            pl = Pipeline([("s", StandardScaler()), ("p", PCA(64)),
                           ("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000))])
            pl.fit(Xa[tr], yu[tr]); oofu[te] = pl.predict_proba(Xa[te])[:, 1]
        keys = list(range(len(yu)))
        o2[arm] = dict(zip(keys, map(float, oofu)))
        ref = o2.get("index_only") if arm != "index_only" else None
        res["upgrade"][arm] = bootstrap_auc(o2[arm], dict(zip(keys, yu)), prob_b=ref)
        print("upgrade", arm, res["upgrade"][arm], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
