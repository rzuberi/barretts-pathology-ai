"""Follow-up F8 (pre-specified @ 0db4075): PCA/UMAP figures for all five folds per representation (projection fitted on each
fold's training patients, held-out patients shown), and the paired difference between the linear-probe patient AUROC and the
model's own OOF patient AUROC for early and intermediate fusion. Uses embeddings saved by pp_latent.py in
feasibility/paper_plan/latent/. Outputs: results/paper_plan/figs/latent_folds/*.png, results/paper_plan/f8_probe_vs_output.json."""
import glob, json, os, numpy as np, pandas as pd, warnings
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import umap
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; L = T + "/feasibility/paper_plan/latent"
AGG = os.environ.get("OUTDIR", T + "/results/paper_plan"); FIG = T + "/results/paper_plan/figs/latent_folds"; os.makedirs(FIG, exist_ok=True); os.makedirs(AGG, exist_ok=True); NB = 2000; SEED = 0
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def r3(x): return round(float(x), 3)
idx = pd.read_csv(L + "/index.csv", dtype={"sample_id": str, "patient_id": str}); ids = list(idx.sample_id); pid = idx.patient_id.values; fold = idx.fold.values; y = idx.y.values
pats = np.array(sorted(set(pid))); pfold = pd.Series(fold, index=pid).groupby(level=0).first().reindex(pats).values; py = pd.Series(y, index=pid).groupby(level=0).max().reindex(pats).values
pairs = pd.read_csv(T + "/feasibility/closeout/erin_swg_pairs.csv", dtype=str); psub = np.array(["also_in_ERIN" if p in set(pairs.swg_patient_id) else "never_in_ERIN" for p in pats])
inner = {k: pd.read_csv(f"{R}/image_only/fold{k}/inner_fold_assignments.csv", dtype=str).set_index("patient_id").inner_fold.astype(int) for k in range(1, 6)}
def pmean(M): return pd.DataFrame(M, index=pid).groupby(level=0).mean().reindex(pats).values
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{R}/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(ids).values
def pmax(s): return pd.Series(s, index=pid).groupby(level=0).max().reindex(pats).values
rng = np.random.RandomState(SEED); BI = []
while len(BI) < NB:
    s = rng.choice(len(py), len(py))
    if len(set(py[s])) > 1: BI.append(s)
ci = lambda a, b: [r3(np.percentile([auc(py[s], a[s]) - auc(py[s], b[s]) for s in BI], q)) for q in (2.5, 97.5)]
OUT = {"probe_vs_model_output": {}, "figures": {}}
for fam in ["image_only", "cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]:
    E = {k: np.load(f"{L}/emb_{fam}_fold{k}.npy") for k in range(1, 6)}; probe = np.zeros(len(pats))
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5))
    for k in range(1, 6):
        P = pmean(E[k]); tr = pfold != k; te = ~tr; sc = StandardScaler().fit(P[tr]); Xtr, Xte = sc.transform(P[tr]), sc.transform(P[te]); ytr = py[tr]
        inn = inner[k].reindex(pats[tr]).values; best, bestC = -1, 1.0
        for C in [0.01, 0.1, 1.0, 10.0]:
            pv = np.zeros(tr.sum())
            for j in np.unique(inn): v = inn == j; pv[v] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], ytr[~v]).predict_proba(Xtr[v])[:, 1]
            a = auc(ytr, pv)
            if a > best: best, bestC = a, C
        probe[te] = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        for row, (name, proj) in enumerate([("PCA", PCA(2, random_state=0).fit(Xtr)), ("UMAP", umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit(Xtr))]):
            Z = proj.transform(Xte); ax = axes[row, k - 1]
            for v, c, l in [(0, "#1f77b4", "non-progressor"), (1, "#d62728", "progressor")]: ax.scatter(Z[py[te] == v, 0], Z[py[te] == v, 1], s=16, c=c, label=l, alpha=.8, marker="o")
            ax.scatter(Z[psub[te] == "also_in_ERIN", 0], Z[psub[te] == "also_in_ERIN", 1], s=60, facecolors="none", edgecolors="k", linewidths=.6, label="also_in_ERIN (ring)")
            ax.set_title(f"{fam} {name} fold {k} (held-out n={te.sum()}, events={int(py[te].sum())})", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            if k == 1 and row == 0: ax.legend(fontsize=6)
    fn = f"{FIG}/{fam}_pca_umap_5folds.png"; fig.tight_layout(); fig.savefig(fn, dpi=110); plt.close(fig); OUT["figures"][fam] = fn.replace(T + "/", "")
    if fam in ["early_fusion", "intermediate_fusion", "image_only", "coattention_fusion"]:
        Pm = pmax(oof(fam)); OUT["probe_vs_model_output"][fam] = {"n": int(len(py)), "events": int(py.sum()), "probe_auroc": r3(auc(py, probe)), "model_oof_auroc": r3(auc(py, Pm)), "delta_probe_minus_model": r3(auc(py, probe) - auc(py, Pm)), "ci": ci(probe, Pm)}
    print(fam, flush=True)
OUT["_method"] = "probe: L2 logistic on patient-mean row embeddings, standardised and C-tuned (release inner folds) on training patients, applied to held-out patients (same procedure and seed as pp_latent.py); model output: release OOF, patient max; paired patient bootstrap; projections fitted on each fold's training patients"
json.dump(OUT, open(AGG + "/f8_probe_vs_output.json", "w"), indent=1); print(json.dumps(OUT["probe_vs_model_output"], indent=1))
