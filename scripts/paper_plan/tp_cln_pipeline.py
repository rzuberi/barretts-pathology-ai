"""Two papers, agent CLN (docs/two_papers_CLN.md): the image front end of frozen bundle v2 (killcoyne_mm/two_papers_REG/frozen_v2_clean/), written
independently of ART's analysis code (tp_art_main.py) so that it can check it. numpy / pandas only.

Pipeline per sample (all of its slides): UNI2-h tiles (224 px at level 2, tile centres as stored; pixel statistics by tp_art_qc.py pixel_stats)
  -> tile artefact flag with FIXED thresholds (qc_spec.json):
       glass / background: stained-tissue share < 0.15
       edge: not background and a straight band of dark unstained pixels covering >= 50% of a 3-line running mean (line_d >= 0.5)
       pen / dye: blue-green ink share > 0.02
       blur: not background and Laplacian variance over tissue pixels < LAP_THR (missing when < 500 tissue pixels: counted as blur)
  -> clean tiles = not flagged; if fewer than 32, top up to 32 with the flagged tiles of highest tissue share, then highest lapvar_t, then bag order
  -> sample embedding = mean of the kept tiles' embeddings
  -> L-IMG probability = sigmoid(b0 + sum_f w_f (x_f - mean_f) / sd_f) with the frozen coefficients and image scaling (sd 0 -> 1).
TASK = verify (cluster, CPU): recompute ART's clean-tile features from qc_all.parquet and the bags, and the in-sample probabilities from the v2 models."""
import os, sys, json, numpy as np, pandas as pd

def tile_flags(q, spec):
    """q: DataFrame with tissue, line_d, ink, lapvar_t (one row per tile). Returns DataFrame of booleans bg, edge, pen, blur, art."""
    t = q.tissue.to_numpy(float); bg = t < spec["tissue_min"]
    edge = ~bg & (q.line_d.to_numpy(float) >= spec["line_d_min"])
    pen = q.ink.to_numpy(float) > spec["ink_max"]
    lv = q.lapvar_t.to_numpy(float); lv = np.where(np.isfinite(lv), lv, 0.0); blur = ~bg & (lv < spec["lapvar_t_min"])
    return pd.DataFrame({"bg": bg, "edge": edge, "pen": pen, "blur": blur, "art": bg | edge | pen | blur}, index=q.index)

def kept_tiles(q, art, min_tiles):
    """boolean mask of the tiles used for the sample embedding (clean tiles, topped up to min_tiles with the least-artefactual flagged tiles)"""
    keep = ~np.asarray(art, bool); need = min_tiles - int(keep.sum())
    if need > 0:
        t = q.tissue.to_numpy(float); lv = q.lapvar_t.to_numpy(float); lv = np.where(np.isfinite(lv), lv, 0.0)
        cand = sorted(np.where(~keep)[0], key=lambda i: (-t[i], -lv[i], i)); keep[cand[:need]] = True
    return keep

def sample_embedding(q, emb, spec):
    f = tile_flags(q, spec); k = kept_tiles(q, f.art.values, spec["min_tiles"]); return np.asarray(emb, np.float64)[k].mean(0), int((~f.art).sum()), int(k.sum())

def l_img_prob(X, coef, scaling):
    """X: (n, 1536) sample embeddings in the scaling's feature order; coef: DataFrame(feature, coefficient) with (Intercept) first"""
    mu = scaling["mean"].to_numpy(float); sd = scaling["sd"].to_numpy(float); Z = (np.asarray(X, float) - mu) / np.where(sd > 0, sd, 1.0)
    c = coef.set_index("feature").coefficient; w = c.reindex(scaling.feature).to_numpy(float); return 1 / (1 + np.exp(-(c.iloc[0] + Z @ w)))

if __name__ == "__main__" and os.environ.get("TASK") == "verify":
    M = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm"; FZ = os.environ.get("CLN_FZ", M + "/two_papers_REG/frozen_v2_clean")
    spec = json.load(open(FZ + "/qc/qc_spec.json"))["thresholds"]; Q = pd.read_parquet(M + "/two_papers_ART/qc_all.parquet")
    BI = pd.read_csv(M + "/bag_index.csv", dtype={"sample_id": str}).set_index("sample_id"); C = pd.read_csv(M + "/set_C.csv", dtype=str); ids = C.Sample.values
    ART = pd.read_csv(M + "/two_papers_ART/feats/clean.csv.gz", dtype={"Sample": str}).set_index("Sample").reindex(ids)
    QI = Q.groupby("sample").indices; X = np.zeros((len(ids), 1536)); info = []
    for i, s in enumerate(ids):
        q = Q.iloc[np.sort(QI[s])].reset_index(drop=True); E = np.load(BI.loc[s, "npz_path"])["embeddings"]; assert len(E) == len(q) and (q.bag_pos.values == np.arange(len(q))).all()
        X[i], nc, nk = sample_embedding(q, E, spec); info.append((nc, nk))
    d = np.abs(X - ART.values); rel = d / np.maximum(np.abs(ART.values), 1e-3)
    out = {"samples": int(len(ids)), "embedding_vs_ART_clean_max_abs": float(d.max()), "embedding_vs_ART_clean_max_rel": float(rel.max()),
           "samples_topped_up": int(sum(nc < spec["min_tiles"] for nc, _ in info)), "flag_share_all_tiles": float(tile_flags(Q, spec).art.mean())}
    sc = pd.read_csv(FZ + "/models_their/image_scaling.csv"); pf = pd.DataFrame(X, columns=[f"u{j}" for j in range(1536)])[sc.feature.values].values
    out["image_scaling_mean_vs_pipeline_embedding_mean_max_abs"] = float(np.abs(sc["mean"].values - pf.mean(0)).max())
    from sklearn.metrics import roc_auc_score
    y = (C.Status == "P").astype(int).values
    for sub in ("models_their", "models_pkg"):
        sc = pd.read_csv(f"{FZ}/{sub}/image_scaling.csv"); p = l_img_prob(pd.DataFrame(X, columns=[f"u{j}" for j in range(1536)])[sc.feature.values].values, pd.read_csv(f"{FZ}/{sub}/coef_L_IMG.csv"), sc)
        mi = json.load(open(f"{FZ}/{sub}/model_info.json"))["insample_check"]; out[f"{sub}_insample_auroc_L_IMG"] = [round(float(roc_auc_score(y, p)), 6), round(float(mi["auroc_L_IMG"]), 6)]
    out["pass"] = bool(out["embedding_vs_ART_clean_max_abs"] < 1e-6 and all(abs(v[0] - v[1]) < 1e-6 for k, v in out.items() if k.endswith("insample_auroc_L_IMG")))
    json.dump(out, open(FZ + "/verify_pipeline.json", "w"), indent=1); os.makedirs(M + "/two_papers_CLN/freeze", exist_ok=True); json.dump(out, open(M + "/two_papers_CLN/freeze/verify_pipeline.json", "w"), indent=1)
    print(json.dumps(out, indent=1)); print("TPCLN DONE pipeline verify", out["pass"], flush=True)
