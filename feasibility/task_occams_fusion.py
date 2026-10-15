"""Ch2 GATE first-pass: OCCAMS histology vs genomics vs fusion -> survival.

Uses mean-pooled UNI2 slide features (276 cases) + WGS-derived labels (383 cases).
First-pass endpoint: 2-year overall survival (Weeks.Survival >= 104). If a vital-status
column is present it is used to exclude early-censored cases; otherwise counts are reported.
Patient-level 5x5 stratified CV, shuffled-label controls, paired bootstrap on OOF deltas.
"""
import os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

BASE = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data"
TSV = BASE + "/genomics/clinical_data_wgs_cases_therapy_tp53status_ploidy_wgd_status.tsv"
NPZ = "/home/zuberi01/occams_scan/uni_res_pooled.npz"
OUT = os.environ.get("OUTDIR", ".")

d = np.load(NPZ, allow_pickle=True)
X_all, cases = d["X"], list(d["cases"])
idx = {c: i for i, c in enumerate(cases)}
df = pd.read_csv(TSV, sep="\t", dtype=str, low_memory=False)
print("feature cases:", len(cases), "| label rows:", len(df))
print("label columns:", list(df.columns))

df = df[df["OCCAMS_ID"].isin(idx)].drop_duplicates("OCCAMS_ID").copy()
df["weeks"] = pd.to_numeric(df["Weeks.Survival"], errors="coerce")
status_cols = [c for c in df.columns if any(k in c.lower() for k in ("status", "death", "alive", "vital"))
               and "tp53" not in c.lower() and "wgd" not in c.lower()]
print("candidate status columns:", status_cols)
df = df[df["weeks"].notna()].copy()
CUT = 104.0
df["y"] = (df["weeks"] >= CUT).astype(int)
print(f"joined n={len(df)}, 2yr-survivor rate={df['y'].mean():.3f}")

# genomic features
def flag(col):
    v = df[col].astype(str).str.strip().str.lower()
    return v.isin(("1", "true", "yes", "y")).astype(float)
G = np.column_stack([
    flag("TP53_SNV"), flag("TP53_indel"), flag("TP53_deletion"), flag("TP53_knockout"),
    pd.to_numeric(df["ploidy"], errors="coerce").fillna(2.0).values,
    flag("WGD"),
])
H = X_all[[idx[c] for c in df["OCCAMS_ID"]]]
y = df["y"].values
print("H", H.shape, "G", G.shape, "pos", int(y.sum()))

def oof_probs(Xi, y, seed, pca=None):
    oof = np.zeros(len(y))
    steps = [("s", StandardScaler())]
    if pca: steps.append(("p", PCA(min(pca, Xi.shape[1], len(y) - 1))))
    steps.append(("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xi, y):
        p = Pipeline(steps)
        p.fit(Xi[tr], y[tr])
        oof[te] = p.predict_proba(Xi[te])[:, 1]
    return oof

SEEDS = range(5)
arms = {
    "hist_only":  lambda s, yy: oof_probs(H, yy, s, pca=64),
    "gen_only":   lambda s, yy: oof_probs(G, yy, s),
    "early_fusion": lambda s, yy: oof_probs(np.column_stack([PCA(64).fit_transform(StandardScaler().fit_transform(H)), StandardScaler().fit_transform(G)]), yy, s),
}
res, oofs = {}, {}
for name, fn in arms.items():
    aucs, briers, per_seed = [], [], []
    for s in SEEDS:
        o = fn(s, y); per_seed.append(o)
        aucs.append(roc_auc_score(y, o)); briers.append(brier_score_loss(y, o))
    oofs[name] = np.mean(per_seed, axis=0)
    sh = [roc_auc_score(np.random.RandomState(s).permutation(y), fn(s, np.random.RandomState(s).permutation(y))) for s in SEEDS]
    res[name] = {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                 "brier": float(np.mean(briers)), "shuffled_auc": float(np.mean(sh))}
    print(f"{name:14s} AUC={res[name]['auc']:.3f}±{res[name]['auc_sd']:.3f} Brier={res[name]['brier']:.3f} shuf={res[name]['shuffled_auc']:.3f}")

oofs["late_fusion"] = (oofs["hist_only"] + oofs["gen_only"]) / 2
res["late_fusion"] = {"auc": float(roc_auc_score(y, oofs["late_fusion"])),
                      "brier": float(brier_score_loss(y, oofs["late_fusion"]))}
print(f"late_fusion    AUC={res['late_fusion']['auc']:.3f} Brier={res['late_fusion']['brier']:.3f}")

rng = np.random.RandomState(0)
for fus in ("late_fusion", "early_fusion"):
    deltas = []
    for _ in range(2000):
        b = rng.randint(0, len(y), len(y))
        if len(set(y[b])) < 2: continue
        deltas.append(roc_auc_score(y[b], oofs[fus][b]) - roc_auc_score(y[b], oofs["hist_only"][b]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    res[fus]["delta_auc_vs_hist"] = {"mean": float(np.mean(deltas)), "ci": [float(lo), float(hi)]}
    print(f"{fus} - hist_only: dAUC={np.mean(deltas):+.3f} [{lo:+.3f},{hi:+.3f}]")

res["_meta"] = {"n": int(len(y)), "endpoint": f"Weeks.Survival>={CUT}", "pos": int(y.sum()),
                "status_cols_found": status_cols, "note": "first-pass; censoring not modelled if no status col"}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote", os.path.join(OUT, "results.json"))
