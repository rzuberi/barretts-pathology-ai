"""Latent space Section 2 (docs/paper_latent_attention.md @ dfac9ad): UMAP of R1 / R2 / R3 (their matrix), in-sample on the 571 pre-event samples (transforms fitted
on those rows), coloured by progressor status, patient, scanner and grade; silhouettes on the representation (cosine) and on the UMAP (Euclidean).
Supplementary: n_neighbors 50, and the held-out pre-event samples of repeat 1 fold 1 under that fold's transform. Coordinates stay on the cluster;
figures -> feasibility/.../latent_attention/figs (copied to be_paper_figs/v3), aggregates -> results/paper_final/latent_attention/umap.json."""
import os, json, numpy as np, pandas as pd, matplotlib, umap
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; LA = M + "/latent_attention"; FD = LA + "/figs"; OUT = T + "/results/paper_final/latent_attention"
os.makedirs(FD, exist_ok=True); os.makedirs(OUT, exist_ok=True); r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
C = pd.read_csv(M + "/set_C.csv", dtype=str); ids = C.Sample.values
SM = pd.read_csv(M + "/horizons/samples.csv", dtype={"Sample": str, "Patient": str}).set_index("Sample").reindex(ids); SD = pd.read_csv(M + "/horizons/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids)
pat = SM.Patient.values; y = SM.y.values.astype(int); pre = SM.pre.astype(bool).values; grade = SM.Pathology.values; scan = SD.scanner.values
blk = lambda f, k: pd.read_csv(f"{M}/{f}", dtype={k: str}).set_index(k).reindex(ids).values.astype(float)
IM = blk("img_mean.csv", "Sample"); CN = blk("cnv_their_C.csv", "sample_id")
def zs(X, tr): mu = X[tr].mean(0); sd = X[tr].std(0, ddof=1); Z = (X - mu) / np.where(sd > 0, sd, 1); Z[:, sd == 0] = 0; return Z
def pca_block(B, tr, k=32):   # hz_fit.R:13 (prcomp centred, unscaled; sign of components irrelevant for cosine geometry)
    mu = B[tr].mean(0); _, _, Vt = np.linalg.svd(B[tr] - mu, full_matrices=False); return zs((B - mu) @ Vt[:k].T, tr)
tr_all = np.where(pre)[0]
REPS = {"R1": zs(IM, tr_all), "R2": np.hstack([CN, zs(IM, tr_all)]), "R3": np.hstack([pca_block(CN, tr_all), pca_block(zs(IM, tr_all), tr_all)])}
FO = pd.read_csv(f"{M}/cv/preds/cfg_00_rep_01.csv", dtype={"Sample": str}).set_index("Sample").fold.reindex(ids).values.astype(int); trf = np.where(FO != 1)[0]
R3f = pd.read_csv(f"{LA}/refit/repr_inter_their_rep_01_fold_01.csv.gz", dtype={"Sample": str}).set_index("Sample").reindex(ids).values.astype(float)
HELD = {"R1": zs(IM, trf), "R2": np.hstack([CN, zs(IM, trf)]), "R3": R3f}; held_rows = np.where((FO == 1) & pre)[0]
NAME = {"R1": "WSI-only (R1)", "R2": "early fusion (R2)", "R3": "inter fusion (R3)"}
def emb(X, nn): return umap.UMAP(n_neighbors=nn, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X)
def sil(X, lab, metric):
    lab = np.asarray(lab); return r3(silhouette_score(X, lab, metric=metric)) if len(set(lab)) > 1 else None
def stats(X, U, rr):
    pp = pat[rr]; multi = np.isin(pp, [q for q in set(pp) if (pp == q).sum() >= 2])
    return {"progressor": {"repr_cosine": sil(X[rr], y[rr], "cosine"), "umap": sil(U, y[rr], "euclidean")},
            "patient": {"repr_cosine": sil(X[rr][multi], pp[multi], "cosine"), "umap": sil(U[multi], pp[multi], "euclidean"), "n_samples": int(multi.sum())}}
COLP = {0: "#1f77b4", 1: "#d62728"}; COLS = {"C13239-01": "#2ca02c", "C13210": "#9467bd"}; COLG = {"NDBE": "#440154", "ID": "#21918c", "LGD": "#fde725"}
def panel_row(axs, U, rr, label, st):
    pp = pat[rr]; cnt = pd.Series(pp).value_counts(); top = sorted(cnt.index, key=lambda q: (-cnt[q], q))[:10]; cmap = plt.get_cmap("tab10")
    for j, ax in enumerate(axs):
        if j == 0: c = [COLP[v] for v in y[rr]]
        elif j == 1: c = [cmap(top.index(q)) if q in top else "#cccccc" for q in pp]
        elif j == 2: c = [COLS.get(v, "#000000") for v in scan[rr]]
        else: c = [COLG.get(v, "#000000") for v in grade[rr]]
        order = np.argsort([0 if (j == 1 and pp[i] not in top) else 1 for i in range(len(rr))], kind="stable")
        ax.scatter(U[order, 0], U[order, 1], c=np.array(c, dtype=object)[order].tolist(), s=7, lw=0, alpha=0.85); ax.set_xticks([]); ax.set_yticks([])
        if j == 0: ax.set_ylabel(label, fontsize=9); ax.text(0.02, 0.02, f"silhouette (repr / UMAP): {st['progressor']['repr_cosine']} / {st['progressor']['umap']}", transform=ax.transAxes, fontsize=6.5)
        if j == 1: ax.text(0.02, 0.02, f"silhouette (repr / UMAP): {st['patient']['repr_cosine']} / {st['patient']['umap']}", transform=ax.transAxes, fontsize=6.5)
TITLES = ["progressor status (red = progressor)", "patient (10 largest coloured)", "scanner (green C13239-01, purple C13210)", "grade (purple NDBE, teal ID, yellow LGD)"]
res = {"prespec": "dfac9ad", "settings": {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine", "random_state": 0, "umap_version": umap.__version__}, "main": {}, "nn50": {}, "heldout_rep1_fold1": {"n_samples": int(len(held_rows))}}
coords = []
def figure(rowsets, fname, title):
    fig, A = plt.subplots(len(rowsets), 4, figsize=(13, 3.1 * len(rowsets)))
    for i, (U, rr, lab, st) in enumerate(rowsets): panel_row(A[i], U, rr, lab, st)
    for j in range(4): A[0, j].set_title(TITLES[j], fontsize=9)
    fig.suptitle(title, fontsize=10); fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"): fig.savefig(f"{FD}/{fname}.{ext}", dpi=200)
    plt.close(fig)
main, supp = [], []
for k, X in REPS.items():
    U = emb(X[tr_all], 15); st = stats(X, U, tr_all); res["main"][k] = st; main.append((U, tr_all, NAME[k], st))
    U50 = emb(X[tr_all], 50); st50 = stats(X, U50, tr_all); res["nn50"][k] = st50; supp.append((U50, tr_all, NAME[k] + ", n_neighbors 50", st50))
    coords += [pd.DataFrame({"Sample": ids[tr_all], "set": s, "repr": k, "u1": u[:, 0], "u2": u[:, 1]}) for s, u in (("main", U), ("nn50", U50))]
for k, X in HELD.items():
    U = emb(X[held_rows], 15); st = stats(X, U, held_rows); res["heldout_rep1_fold1"][k] = st; supp.append((U, held_rows, NAME[k] + ", held-out r1 f1", st))
    coords.append(pd.DataFrame({"Sample": ids[held_rows], "set": "heldout_r1f1", "repr": k, "u1": U[:, 0], "u2": U[:, 1]}))
figure(main, "11_F_latent_space", "In-sample, illustrative: UMAP of the linear-tier representations, 571 pre-event samples (their CNV matrix); no separation claim rests on this figure")
figure(supp, "11_F_latent_space_supp", "Supplementary, illustrative: n_neighbors 50 (rows 1-3); held-out pre-event samples of repeat 1, fold 1 under that fold's transform (rows 4-6)")
pd.concat(coords).to_csv(LA + "/umap_coords.csv", index=False); json.dump(res, open(f"{OUT}/umap.json", "w"), indent=1); print("LA UMAP DONE", flush=True)
