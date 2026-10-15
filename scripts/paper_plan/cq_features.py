"""CNV-quiet morphology (docs/paper_cnvquiet_morphology.md), step 3 (CPU): per-sample feature panel from the HoVer-Net nucleus tables.
Classes (PanNuke): 1 neoplastic, 2 inflammatory, 3 connective, 4 dead, 5 non-neoplastic epithelial. Epithelial = {1, 5}. Units: um (0.25 um/px).
Nuclei < 5 um^2 dropped everywhere. Border nuclei (contour within 2 px of the tile edge) are dropped from F1-F3 only. Neighbourhoods are within a tile.
F1 mean epithelial nuclear area (um^2); F2 SD of epithelial nuclear area (um^2); F3 mean epithelial nuclear circularity 4*pi*A/P^2;
F4 basal-layer crowding: basal-layer nuclei = epithelial nuclei with a connective nucleus centroid within 25 um; F4 = mean number of other epithelial
nuclei within 20 um of a basal-layer nucleus; F5 peri-epithelial inflammatory density: inflammatory nuclei with an epithelial centroid within 50 um,
per 100 epithelial nuclei (pooled counts). Eligibility: >= 200 non-border epithelial nuclei (F1-F3, F5); >= 50 basal-layer nuclei (F4).
Row-level output (cluster only): cnvquiet_morphology/features_sample.csv."""
import os, json, numpy as np, pandas as pd
from scipy.spatial import cKDTree
W = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm/cnvquiet_morphology"; MPP = 0.25
TD, ND, TAG, GR = os.environ.get("CQ_TILEDIR", "tiles"), os.environ.get("CQ_NUCDIR", "nuclei"), os.environ.get("CQ_TAG", ""), os.environ.get("CQ_GROUPS", "abc")
MIN_AREA, R_BASAL, R_CROWD, R_PEL, MIN_E, MIN_BASAL = 5.0, 25.0, 20.0, 50.0, 200, 50
def slide_stats(f):
    z = np.load(f); t = z["tile"]; x = z["x"] * MPP; y = z["y"] * MPP; area = z["area_px"] * MPP ** 2; per = z["perim_px"] * MPP; ty = z["type"]; edge = z["edge"]
    keep = area >= MIN_AREA; t, x, y, area, per, ty, edge = t[keep], x[keep], y[keep], area[keep], per[keep], ty[keep], edge[keep]
    E = np.isin(ty, [1, 5]); morph = E & ~edge; circ = np.clip(4 * np.pi * area[morph] / np.maximum(per[morph], 1e-6) ** 2, 0, 1)
    crowd, n_basal, n_pel, n_e = [], 0, 0, int(E.sum())
    for k in np.unique(t):
        m = t == k; pe = np.c_[x[m & E], y[m & E]]; ps = np.c_[x[m & (ty == 3)], y[m & (ty == 3)]]; pi = np.c_[x[m & (ty == 2)], y[m & (ty == 2)]]
        if len(pe) == 0: continue
        te = cKDTree(pe)
        if len(ps):
            dS, _ = cKDTree(ps).query(pe, k=1); bas = pe[dS <= R_BASAL]
            if len(bas): crowd.extend([len(v) - 1 for v in te.query_ball_point(bas, R_CROWD)]); n_basal += len(bas)
        if len(pi): dE, _ = te.query(pi, k=1); n_pel += int((dE <= R_PEL).sum())
    return {"area": area[morph], "circ": circ, "crowd": np.asarray(crowd, float), "n_basal": n_basal, "n_pel": n_pel, "n_e": n_e,
            "n_neo": int((ty == 1).sum()), "n_epi5": int((ty == 5).sum()), "n_inf": int((ty == 2).sum()), "n_con": int((ty == 3).sum()), "n_dead": int((ty == 4).sum()),
            "n_tiles": int(z["n_tiles"]), "tiles_with_e": int(len(np.unique(t[E])))}
R = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/results/paper_final/cnvquiet_morphology"
sl = pd.read_csv(W + "/slides.csv", dtype=str); sl = sl[sl.group.isin(list(GR))]; META = {k: json.load(open(f"{W}/{TD}/{k}/meta.json")) for k in sl.slide_key}
# Deviation (reported): the pre-specified coordinate check (all of the first 4 tiles r >= 0.95) is applied to tissue tiles only (grey SD >= 10 on the
# saved tile); every tile below 0.95 in the run was blank glass (grey SD about 2, no tissue), where the correlation is noise.
from PIL import Image
def gsd(k, i): return float(np.asarray(Image.open(f"{W}/{TD}/{k}/t{i:03d}.jpg").convert("L"), float).std())
def cmin(k):
    c = [r for i, r in enumerate(META[k]["coord_check_corr"]) if gsd(k, i) >= 10]; return min(c) if c else np.nan
sl["coord_min_all"] = sl.slide_key.map(lambda k: min(META[k]["coord_check_corr"])); sl["coord_min"] = sl.slide_key.map(cmin)
bad = sl[~(sl.coord_min >= 0.95)]; sl = sl[sl.coord_min >= 0.95]
assert all(os.path.exists(f"{W}/{ND}/{k}.npz") for k in sl.slide_key), "missing nucleus tables"
qc = {"slides": int(len(META)), "slides_failing_coord_check": int(len(bad)), "samples_affected": sorted(set(bad.group)), "coord_corr_min": float(min(min(m["coord_check_corr"]) for m in META.values())), "coord_corr_min_tissue_tiles": float(sl.coord_min.min()),
      "slides_failing_prespecified_all_tile_rule": int((pd.concat([sl, bad]).coord_min_all < 0.95).sum()),
      "out_mpp_range": [float(min(m["out_mpp"] for m in META.values())), float(max(m["out_mpp"] for m in META.values()))], "mpp0_range": [float(min(m["mpp0_x"] for m in META.values())), float(max(m["mpp0_x"] for m in META.values()))],
      "out_px_values": sorted({int(m["out_px"]) for m in META.values()}), "tiles_total": int(sum(m["n_tiles"] for m in META.values()))}
rows = []
for s, g in sl.groupby("Sample"):
    st = [slide_stats(f"{W}/{ND}/{k}.npz") for k in g.slide_key]
    area = np.concatenate([a["area"] for a in st]); circ = np.concatenate([a["circ"] for a in st]); crowd = np.concatenate([a["crowd"] for a in st])
    S = {k: sum(a[k] for a in st) for k in ["n_basal", "n_pel", "n_e", "n_neo", "n_epi5", "n_inf", "n_con", "n_dead", "n_tiles", "tiles_with_e"]}
    ok = len(area) >= MIN_E; okb = S["n_basal"] >= MIN_BASAL
    rows.append({"Sample": s, "group": g.group.iloc[0], "n_morph_e": len(area), **S,
                 "F1_area_mean": area.mean() if ok else np.nan, "F2_area_sd": area.std(ddof=1) if ok else np.nan, "F3_circularity": circ.mean() if ok else np.nan,
                 "F4_basal_crowding": crowd.mean() if okb else np.nan, "F5_pel_per100e": 100 * S["n_pel"] / S["n_e"] if ok else np.nan})
F = pd.DataFrame(rows); F.to_csv(W + f"/features_sample{TAG}.csv", index=False)
qc["nuclei_kept_total"] = int(F[["n_neo", "n_epi5", "n_inf", "n_con", "n_dead"]].sum().sum()); json.dump(qc, open(R + f"/segmentation_qc{TAG}.json", "w"), indent=1); print(qc); print(F.groupby("group")[[c for c in F.columns if c.startswith("F")]].mean().round(3)); print("CQ FEATURES DONE", len(F))
