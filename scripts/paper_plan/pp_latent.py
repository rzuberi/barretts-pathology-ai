"""Paper plan items 6, 7 (attention arrays + per-row metrics) and 8b, pre-specified in docs/paper_plan_answers.md @ db236a0.
Loads the frozen release fold checkpoints (image_only, early_fusion, intermediate_fusion, coattention_fusion .pt; cnv_only
.joblib), computes for every row and every fold model: representation vectors, tile attention, CNV permutation importance
on held-out rows. Row-level arrays -> feasibility/paper_plan/latent/ (cluster only); aggregates -> results/paper_plan/
latent_item6.json, attention_item7_rows.csv (row metrics, no identifiers beyond sample_id), perm_importance_item8b.json;
figures -> results/paper_plan/figs/latent/. Env: erin (torch 2.0.1, sklearn 1.5.2 = release pickle version)."""
import glob, json, os, sys, numpy as np, pandas as pd, torch, joblib, warnings
from scipy.stats import spearmanr, rankdata, entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/multimodal-barretts-progression"; sys.path.insert(0, B + "/src")
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; ROW = T + "/feasibility/paper_plan/latent"; AGG = os.environ.get("OUTDIR", T + "/results/paper_plan"); FIG = T + "/results/paper_plan/figs/latent"
for p in (ROW, AGG, FIG): os.makedirs(p, exist_ok=True)
from barrett.models.image_mil import AttentionMIL; from barrett.models.early_fusion import EarlyFusionMLP; from barrett.models.intermediate_fusion import IntermediateABMILCNV; from barrett.models.coattention import CoAttentionABMILCNV
from barrett.training.data import load_cnv_matrix
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); NB = 2000; SEED = 0
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); ids = list(man.index); y_row = man.y_progressor.astype(int).values; fold = man.fold_id_rep01.astype(int).values; pid = man.patient_id.values
ui = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str).set_index("sample_id").reindex(ids)
cnv_df, feats = load_cnv_matrix(F + "/feature_views/cnv"); X_cnv = cnv_df.set_index("sample_id").loc[ids, feats].to_numpy(np.float64)   # float64: the release fed the sklearn pipeline a float64 DataFrame (float32 shifts RF probabilities by up to 0.14)
ARM = [i for i, f in enumerate(feats) if f.startswith("chr") and (f.endswith("p") or f.endswith("q")) or f == "cx"]
print("rows", len(ids), "cnv feats", len(feats), "arm+cx feats", len(ARM), flush=True)
bags = {}
for s, p in zip(ids, ui.npz_path):
    with np.load(p, allow_pickle=False) as z: bags[s] = torch.from_numpy(np.asarray(z["embeddings"], np.float32))
print("bags loaded", flush=True)
def build(fam, cfg, cnv_dim):
    if fam == "image_only": return AttentionMIL(1536, int(cfg["hidden_dim"]), int(cfg["attn_dim"]), float(cfg["dropout"]))
    if fam == "early_fusion": return EarlyFusionMLP(1536, cnv_dim, int(cfg["hidden_dim"]), float(cfg["dropout"]))
    cls = IntermediateABMILCNV if fam == "intermediate_fusion" else CoAttentionABMILCNV
    return cls(1536, cnv_dim, int(cfg["img_hidden"]), int(cfg["cnv_hidden"]), int(cfg["attn_dim"]), int(cfg["fusion_hidden"]), float(cfg["dropout"]))
def load(fam, k):
    ck = torch.load(f"{R}/{fam}/fold{k}/model.pt", map_location="cpu"); m = build(fam, ck["configuration"], len(feats)); m.load_state_dict(ck["state_dict"]); m.eval().to(DEV)
    return m, ck.get("cnv_median"), ck.get("cnv_mean"), ck.get("cnv_std")
def std_cnv(X, med, mu, sd):   # torch models: float32 standardised input, as in training
    X = np.where(np.isfinite(X), X, med) if med is not None else X; return ((X - mu) / sd).astype(np.float32) if mu is not None else X.astype(np.float32)
EMB = {}; ATT = {}; PROB = {}; PRE = {}
with torch.no_grad():
    for k in range(1, 6):
        for fam in ["image_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]:
            m, med, mu, sd = load(fam, k); Xc = torch.from_numpy(std_cnv(X_cnv, med, mu, sd)).to(DEV) if fam != "image_only" else None
            emb = []; att = []; prob = []; pre = {}
            for i, s in enumerate(ids):
                bag = bags[s].to(DEV)
                if fam == "image_only":
                    h = m.embed(bag); score = m.attn_c(torch.tanh(m.attn_a(h)) * torch.sigmoid(m.attn_b(h))).squeeze(-1); w = torch.softmax(score, 0); z = (h * w[:, None]).sum(0); logit = m.classifier(z)
                    emb.append(z.cpu().numpy()); att.append(w.cpu().numpy()); prob.append(torch.sigmoid(logit).item()); pre[s] = z
                elif fam == "early_fusion":
                    img = bag.mean(0); x = torch.cat([img, Xc[i]]); h1 = m.classifier[2](m.classifier[1](m.classifier[0](x))); logit = m.classifier[6](m.classifier[5](m.classifier[4](m.classifier[3](h1))))
                    emb.append(h1.cpu().numpy()); prob.append(torch.sigmoid(logit).item()); pre[s] = img
                elif fam == "intermediate_fusion":
                    h = m.img_embed(bag); score = m.attn_c(torch.tanh(m.attn_a(h)) * torch.sigmoid(m.attn_b(h))).squeeze(-1); w = torch.softmax(score, 0); zi = (h * w[:, None]).sum(0); zc = m.cnv_branch(Xc[i]); logit = m.fusion(torch.cat([zi, zc]))
                    emb.append(torch.cat([zi, zc]).cpu().numpy()); att.append(w.cpu().numpy()); prob.append(torch.sigmoid(logit).item()); pre[s] = zi
                else:
                    h = m.img_embed(bag); keys = m.key_proj(h); vals = m.value_proj(h); zc = m.cnv_embed(Xc[i]); q = m.query_proj(zc); sc = (keys @ q) / (keys.shape[1] ** 0.5); w = torch.softmax(sc, 0); zi = (vals * w[:, None]).sum(0); logit = m.fusion(torch.cat([zi, zc]))
                    emb.append(torch.cat([zi, zc]).cpu().numpy()); att.append(w.cpu().numpy()); prob.append(torch.sigmoid(logit).item()); pre[s] = (keys, vals)
            EMB[(fam, k)] = np.stack(emb); PROB[(fam, k)] = np.array(prob); PRE[(fam, k)] = (m, Xc, pre)
            if att: ATT[(fam, k)] = np.stack(att)
            np.save(f"{ROW}/emb_{fam}_fold{k}.npy", EMB[(fam, k)]); 
            if att: np.save(f"{ROW}/attn_{fam}_fold{k}.npy", ATT[(fam, k)])
            print("done", fam, k, flush=True)
        pipe = joblib.load(f"{R}/cnv_only/fold{k}/model.joblib"); Z = pipe.named_steps["pca"].transform(pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(X_cnv))); EMB[("cnv_only", k)] = Z; PROB[("cnv_only", k)] = pipe.predict_proba(X_cnv)[:, 1]; np.save(f"{ROW}/emb_cnv_only_fold{k}.npy", Z)
# sanity: held-out probabilities vs release OOF
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{R}/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(ids).values
SAN = {}
for fam in ["image_only", "early_fusion", "intermediate_fusion", "coattention_fusion", "cnv_only"]:
    rec = np.zeros(len(ids)); 
    for k in range(1, 6): rec[fold == k] = PROB[(fam, k)][fold == k]
    SAN[fam] = {"max_abs_diff_vs_release_oof": r3(np.nanmax(np.abs(rec - oof(fam)))), "spearman": r3(spearmanr(rec, oof(fam)).correlation)}
print("sanity", SAN, flush=True)
np.save(f"{ROW}/ids.npy", np.array(ids)); pd.DataFrame({"sample_id": ids, "patient_id": pid, "fold": fold, "y": y_row}).to_csv(f"{ROW}/index.csv", index=False)
# ------------------------------------------------------------- item 6
pairs = pd.read_csv(T + "/feasibility/closeout/erin_swg_pairs.csv", dtype=str); OV = set(pairs.swg_patient_id)
pats = np.array(sorted(set(pid))); pfold = pd.Series(fold, index=pid).groupby(level=0).first().reindex(pats).values; py = pd.Series(y_row, index=pid).groupby(level=0).max().reindex(pats).values; psub = np.array(["also_in_ERIN" if p in OV else "never_in_ERIN" for p in pats])
inner = {k: pd.read_csv(f"{R}/image_only/fold{k}/inner_fold_assignments.csv", dtype=str).set_index("patient_id").inner_fold.astype(int) for k in range(1, 6)}
def patient_mean(M): return pd.DataFrame(M, index=pid).groupby(level=0).mean().reindex(pats).values
def boots(y): rng = np.random.RandomState(SEED); out = []
def bidx(y):
    rng = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        s = rng.choice(len(y), len(y))
        if len(set(y[s])) > 1: out.append(s)
    return out
BI = bidx(py); ci = lambda a, b=None: [r3(np.percentile([auc(py[s], a[s]) - (auc(py[s], b[s]) if b is not None else 0) for s in BI], q)) for q in (2.5, 97.5)]
I6 = {}; PROBE = {}; KNN = {}
for fam in ["image_only", "cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]:
    probe = np.zeros(len(pats)); knn = np.zeros(len(pats)); sil = []
    for k in range(1, 6):
        P = patient_mean(EMB[(fam, k)]); tr = pfold != k; te = ~tr; sc = StandardScaler().fit(P[tr]); Xtr, Xte = sc.transform(P[tr]), sc.transform(P[te]); ytr = py[tr]
        inn = inner[k].reindex(pats[tr]).values; best, bestC = -1, 1.0
        for C in [0.01, 0.1, 1.0, 10.0]:
            pv = np.zeros(tr.sum())
            for j in np.unique(inn[~np.isnan(inn)] if inn.dtype.kind == "f" else inn):
                v = inn == j; pv[v] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], ytr[~v]).predict_proba(Xtr[v])[:, 1]
            a = auc(ytr, pv)
            if a > best: best, bestC = a, C
        probe[te] = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        knn[te] = KNeighborsClassifier(n_neighbors=10).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        sil.append(silhouette_score(Xte, py[te]) if len(set(py[te])) > 1 else np.nan)
    PROBE[fam] = probe; KNN[fam] = knn; I6[fam] = {"dim": int(EMB[(fam, 1)].shape[1]), "n": int(len(pats)), "events": int(py.sum()), "probe_auroc": r3(auc(py, probe)), "probe_ci": ci(probe), "knn10_auroc": r3(auc(py, knn)), "knn10_ci": ci(knn), "silhouette_heldout_per_fold": [r3(s) for s in sil], "silhouette_mean": r3(np.nanmean(sil))}
for fam in ["early_fusion", "intermediate_fusion", "coattention_fusion"]:
    for ref in ["image_only", "cnv_only"]: I6[fam][f"probe_delta_vs_{ref}"] = [r3(auc(py, PROBE[fam]) - auc(py, PROBE[ref]))] + ci(PROBE[fam], PROBE[ref]); I6[fam][f"knn_delta_vs_{ref}"] = [r3(auc(py, KNN[fam]) - auc(py, KNN[ref]))] + ci(KNN[fam], KNN[ref])
# figures (fold 1 projection fitted on training patients)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import umap
figs = {}
for fam in ["image_only", "cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]:
    P = patient_mean(EMB[(fam, 1)]); tr = pfold != 1; te = ~tr; sc = StandardScaler().fit(P[tr]); Xtr, Xte = sc.transform(P[tr]), sc.transform(P[te])
    for name, proj in [("pca", PCA(2, random_state=0).fit(Xtr)), ("umap", umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit(Xtr))]:
        Z = proj.transform(Xte); fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        for a, lab, col in [(ax[0], py[te], "status"), (ax[1], (psub[te] == "also_in_ERIN").astype(int), "subgroup")]:
            for v, c, l in [(0, "#1f77b4", "non-progressor" if col == "status" else "never_in_ERIN"), (1, "#d62728", "progressor" if col == "status" else "also_in_ERIN")]: a.scatter(Z[lab == v, 0], Z[lab == v, 1], s=18, c=c, label=l, alpha=.8)
            a.set_title(f"{fam} {name.upper()} held-out fold 1, by {col}"); a.legend(fontsize=7); a.set_xticks([]); a.set_yticks([])
        fn = f"{FIG}/{fam}_{name}_fold1.png"; fig.tight_layout(); fig.savefig(fn, dpi=130); plt.close(fig); figs[f"{fam}_{name}"] = fn.replace(T + "/", "")
I6["_figures"] = figs; I6["_method"] = "row vectors from each fold model (held-out rows = out-of-sample); patient = mean of rows; probe/kNN/scaler fitted on training-fold patients (C by release inner folds); silhouette on held-out standardised patients; PCA/UMAP fitted on fold-1 training patients, held-out fold-1 patients shown; late_mean excluded (no shared representation)"; I6["_sanity_recomputed_vs_release_oof"] = SAN
json.dump(I6, open(AGG + "/latent_item6.json", "w"), indent=1)
# ------------------------------------------------------------- item 7 row metrics
rows = []
for k in range(1, 6):
    te = np.where(fold == k)[0]; a0 = ATT[("image_only", k)]
    for fam in ["intermediate_fusion", "coattention_fusion"]:
        a1 = ATT[(fam, k)]
        for i in te:
            t5 = max(1, int(round(0.05 * len(a0[i])))); top0 = set(np.argsort(a0[i])[-t5:]); top1 = set(np.argsort(a1[i])[-t5:]); T0 = set(np.argsort(a0[i])[-50:]); T1 = set(np.argsort(a1[i])[-50:])
            rows.append({"sample_id": ids[i], "fold": k, "fusion_model": fam, "spearman": spearmanr(a0[i], a1[i]).correlation, "jaccard_top5pct": len(top0 & top1) / len(top0 | top1), "jaccard_top50": len(T0 & T1) / len(T0 | T1), "entropy_image_only": entropy(a0[i]), f"entropy_fusion": entropy(a1[i]), "n_tiles": int(len(a0[i]))})
pd.DataFrame(rows).to_csv(AGG + "/attention_item7_rows.csv", index=False)
# ------------------------------------------------------------- item 8b permutation importance (held-out rows, patient-level max AUROC)
def pat_auc(idx, p): g = pd.DataFrame({"p": pid[idx], "s": p, "y": y_row[idx]}).groupby("p"); return auc(g.y.max().values, g.s.max().values)
def fusion_prob(fam, k, Xc_std):
    m, _, pre = PRE[(fam, k)]; out = []
    with torch.no_grad():
        for i, s in enumerate(ids):
            if fam == "early_fusion": out.append(torch.sigmoid(m.classifier(torch.cat([pre[s], Xc_std[i]]))).item())
            elif fam == "intermediate_fusion": out.append(torch.sigmoid(m.fusion(torch.cat([pre[s], m.cnv_branch(Xc_std[i])]))).item())
            else:
                keys, vals = pre[s]; zc = m.cnv_embed(Xc_std[i]); q = m.query_proj(zc); w = torch.softmax((keys @ q) / (keys.shape[1] ** 0.5), 0); out.append(torch.sigmoid(m.fusion(torch.cat([(vals * w[:, None]).sum(0), zc]))).item())
    return np.array(out)
IMP = {}; rng = np.random.RandomState(SEED); NREP = 50
for fam in ["cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]:
    delta = np.zeros((len(ARM), NREP)); base_all = []
    for k in range(1, 6):
        te = np.where(fold == k)[0]
        if fam == "cnv_only":
            pipe = joblib.load(f"{R}/cnv_only/fold{k}/model.joblib"); base = pat_auc(te, pipe.predict_proba(X_cnv[te])[:, 1])
            for j, fi in enumerate(ARM):
                for r in range(NREP):
                    Xp = X_cnv[te].copy(); Xp[:, fi] = Xp[rng.permutation(len(te)), fi]; delta[j, r] += (base - pat_auc(te, pipe.predict_proba(Xp)[:, 1])) / 5
        else:
            ck = torch.load(f"{R}/{fam}/fold{k}/model.pt", map_location="cpu"); Xs = std_cnv(X_cnv, ck.get("cnv_median"), ck.get("cnv_mean"), ck.get("cnv_std")); base = pat_auc(te, fusion_prob(fam, k, torch.from_numpy(Xs).to(DEV))[te])
            for j, fi in enumerate(ARM):
                for r in range(NREP):
                    Xp = Xs.copy(); Xp[te, fi] = Xp[te[rng.permutation(len(te))], fi]; delta[j, r] += (base - pat_auc(te, fusion_prob(fam, k, torch.from_numpy(Xp).to(DEV))[te])) / 5
        print("perm importance", fam, k, flush=True)
    imp = delta.mean(1); order = np.argsort(-imp); IMP[fam] = {"importance_mean_delta_auroc": {feats[ARM[j]]: r3(imp[j]) for j in range(len(ARM))}, "top10": [{"feature": feats[ARM[j]], "delta_auroc": r3(imp[j]), "sd_over_repeats": r3(delta[j].std())} for j in order[:10]]}
for fam in ["early_fusion", "intermediate_fusion", "coattention_fusion"]:
    a = np.array([IMP["cnv_only"]["importance_mean_delta_auroc"][f] for f in [feats[i] for i in ARM]]); b = np.array([IMP[fam]["importance_mean_delta_auroc"][f] for f in [feats[i] for i in ARM]]); IMP[fam]["spearman_vs_cnv_only"] = r3(spearmanr(a, b).correlation)
IMP["_method"] = f"held-out rows per fold; one of {len(ARM)} arm-level features (39 arms + cx) permuted across the held-out rows, {NREP} repeats, RandomState(0); delta = baseline patient-level AUROC (max over rows) minus permuted, averaged over the 5 folds; 5-Mb features held fixed; fusion models: image parts precomputed, CNV branch re-run"
json.dump(IMP, open(AGG + "/perm_importance_item8b.json", "w"), indent=1); print("ALL DONE", flush=True)
