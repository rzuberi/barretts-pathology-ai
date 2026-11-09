"""Two papers, agent BCN (docs/two_papers_BCN.md): CPU tiling of the Killcoyne discovery slides (set C, 683 slides of 676 samples) for BEACON's DACOR
(Ercan et al., github.com/caner-ercan/BEACON @ b2007a8). CPU only; the only BCN script that reads slide pixels. GPU jobs later read the arrays written here.

BEACON's DACOR tiles (their code, 1.aneuploid/mil): CLAM create_patches_fp.py with preset barrett(_gaus).csv on 20x slides (MDACC NanoZoomer 0.4548 um/px),
patch 224 px at level 0, step 224 (no overlap), four_pt contour check, padding on; tissue = their WholeSlideImage.segmentTissue gaus_otsu branch
(grey -> 7x7 Gaussian -> inverted Otsu, minus dilated ink mask grey < 50, close 2, contour area > a_t = 1 x (512 px)^2 at level 0, <= 2 holes > 1 x (512 px)^2)
at the level nearest downsample 64. The encoder sees the tile as float 0-255, 224 x 224, no stain or ImageNet normalisation.
Tissue: their detector, run on our whole NDPI slides, picks up pen-marker rings and coverslip edges and misses faint tissue (smoke, 2 of 3 slides;
it was built for ROI crops), so the default (TISSUE=mask) takes tissue from the Killcoyne tissue masks that L-IMG used, with their contour filters.
Our slides are 40x NDPI (0.2204 / 0.2209 um/px at level 0), so every pixel size in their code is scaled by r = 0.4548 / mpp0 (about 2.06) to keep the
same physical size: segmentation at the level nearest downsample 64 r, area unit (512 r px)^2, tile side 224 r px at level 0 (= 101.9 um).
  A (primary, their bag tiles): read 224 r / 2 = 231 px at our level 1 (0.441 um/px), resize to 224 (bicubic) -> 0.4548 um/px, 224 px.
  B (alternative): their published encoder was SimCLR fine-tuned on 448 px crops resized to 224 ("finetuned/448_ds2"), i.e. 0.91 um/px. If DACOR's bags
    were made that way, tiles are 448 r px at level 0 (203.8 um): read 231 px at our level 2 (0.882 um/px), resize to 224 -> 0.9096 um/px.
  SegFormer (tissue classes) and CellViT (cells): coordinates only (their tiles are 1024 px at 0.5 um/px, and 1024 px at slide-native 0.22 um/px with
    64 px overlap); counted here for the storage estimate.
Row-level outputs (cluster only): killcoyne_mm/two_papers_BCN/{slide_key.csv, tiles/{A,B}/s<k>.npy, coords/s<k>.npz, meta/s<k>.json}.
TASK = key | tile_<k> | summary | montage"""
import os, sys, json, time, glob, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; KD = T + "/feasibility/paper_plan/killcoyne"
MIL = "/mnt/scratche/slow/fmlab/zuberi01/phd/beacon/code/1.aneuploid/mil"
OD = M + "/two_papers_BCN"; RES = T + "/results/paper_final/two_papers_BCN"
THEIR_MPP = 0.4548  # MDACC NanoZoomer, 20x: the scanner DACOR was trained on (preprint Methods)
PRESET = dict(sthresh=30, mthresh=11, close=2, use_otsu=False, gaus_otsu=True, a_t=1, a_h=1, max_n_holes=2, contour_fn="four_pt", use_padding=True)  # barrett_gaus.csv
remap = lambda p: p.replace("/scratchc/", "/mnt/scratche/fast/") if isinstance(p, str) and p.startswith("/scratchc/") else p
TASK = os.environ.get("TASK", "")

def slide_key():
    """one row per slide (k = position in sorted npz order) with its sample(s); same slide set as the L-IMG bags (tp_art_qc.bag_slides)"""
    ids = pd.read_csv(M + "/set_C.csv", dtype=str).Sample.values; mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str)
    mp = mp[mp.cnv.isin(set(ids))].sort_values(["cnv", "npz"])
    g = mp.groupby("npz").cnv.apply(lambda s: ";".join(sorted(set(s)))).reset_index().sort_values("npz").reset_index(drop=True)
    sp = []
    for f in g.npz:
        with np.load(f, allow_pickle=True) as z: sp.append(remap(str(z["slide_path"])))
    g.insert(0, "k", range(len(g))); g["slide_path"] = sp; return g

def mask_contours(sp, s, unit0):
    """contours (level-0 px) of the stored Killcoyne tissue mask; their filters: keep a contour if area minus holes > a_t * unit0, keep its
    max_n_holes largest holes with area > a_h * unit0 (WholeSlideImage.segmentTissue._filter_contours, area in level-0 px^2)"""
    import cv2; from PIL import Image; Image.MAX_IMAGE_PIXELS = None
    md = os.path.join(os.path.dirname(os.path.dirname(sp)), "masks"); cand = sorted(glob.glob(os.path.join(md, glob.escape(os.path.splitext(os.path.basename(sp))[0]) + "*")))
    assert cand, "no tissue mask for slide"
    mk = (np.array(Image.open(cand[0]).convert("L")) > 0).astype(np.uint8); W0, H0 = s.level_dimensions[0]; fx, fy = W0 / mk.shape[1], H0 / mk.shape[0]
    cs, hi = cv2.findContours(mk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hi is None: return [], []
    hi = hi[0][:, 2:]; a0 = lambda c: cv2.contourArea(c) * fx * fy; fg, hs = [], []
    for i in np.flatnonzero(hi[:, 1] == -1):
        hl = [cs[j] for j in np.flatnonzero(hi[:, 1] == i)]; a = a0(cs[i]) - sum(a0(h) for h in hl)
        if a > PRESET["a_t"] * unit0:
            fg.append(cs[i]); hs.append([h for h in sorted(hl, key=cv2.contourArea, reverse=True)[:PRESET["max_n_holes"]] if a0(h) > PRESET["a_h"] * unit0])
    sc = np.array([fx, fy]); return [np.array(c * sc, dtype="int32") for c in fg], [[np.array(h * sc, dtype="int32") for h in hh] for hh in hs]

def coords_for(W, size0, step0):
    """their process_contour over every tissue contour, level-0 tile side size0 and step step0 (CLAM drops a contour with <= 1 tile)"""
    out = []
    for cont, holes in zip(W.contours_tissue, W.holes_tissue):
        a, _ = W.process_contour(cont, holes, 0, None, size0, step0, contour_fn=PRESET["contour_fn"], use_padding=PRESET["use_padding"])
        if len(a): out.append(a["coords"])
    return np.concatenate(out).astype(np.int64) if out else np.zeros((0, 2), np.int64)

def read_tiles(slide, xy, level, side, outp):
    """read side x side px at `level` from level-0 top-left xy, resize to 224 (bicubic), write uint8 (N, 224, 224, 3) .npy via memmap"""
    from PIL import Image
    tmp = f"{outp}.{os.getpid()}.{os.environ.get('SLURM_JOB_ID', 0)}.tmp.npy"; arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(xy), 224, 224, 3))  # tmp name unique: racing duplicates may run the same slide
    for i, (x, y) in enumerate(xy):
        im = slide.read_region((int(x), int(y)), level, (side, side)).convert("RGB")
        arr[i] = np.asarray(im.resize((224, 224), Image.BICUBIC) if side != 224 else im)
    arr.flush(); st = tile_stats(arr); del arr; os.replace(tmp, outp); return st

def tile_stats(arr):
    """per tile: sat = mean HSV saturation (0-255, CLAM isWhitePatch: < white_thresh 5 = blank), rgb = mean RGB (isBlackPatch: < black_thresh 50),
    tis = share of pixels with saturation > 0.15 and not all RGB > 220 (tp_art_qc tissue)"""
    import cv2; out = np.zeros((len(arr), 3), np.float32)
    for i in range(len(arr)):
        a = np.asarray(arr[i]); S = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[..., 1]
        out[i] = (S.mean(), a.mean(), ((S > 0.15 * 255) & ~(a > 220).all(-1)).mean())
    return out

KEEP_TIS = float(os.environ.get("KEEP_TIS", "0.15"))  # set by eye on smoke_tissue_bins.png: < 0.15 is glass with slivers, 0.15-0.25 real tissue edges
def keep(st):
    """tiles DACOR scores: not blank / black by their preset (sat >= 5, mean RGB >= 50) and tissue share >= KEEP_TIS (tp_art_qc tissue pixels). CLAM's sat < 5 rule removes nothing on our scans (glass sat 9-14)"""
    return (st[:, 0] >= 5) & (st[:, 1] >= 50) & (st[:, 2] >= KEEP_TIS)

def tile_one(k):
    meta_f = f"{OD}/meta/s{k:03d}.json"
    if os.path.exists(meta_f): return
    sys.path.insert(0, MIL); from wsi_core.WholeSlideImage import WholeSlideImage
    key = pd.read_csv(OD + "/slide_key.csv", dtype={"cnv": str}).set_index("k"); sp = key.loc[k, "slide_path"]; t0 = time.time()
    W = WholeSlideImage(sp); s = W.wsi; mpp0 = float(s.properties["openslide.mpp-x"]); r = THEIR_MPP / mpp0
    ds = [float(d) for d in s.level_downsamples]; assert abs(ds[1] - 2) < 0.01 and abs(ds[2] - 4) < 0.01, ds
    seg_level = s.get_best_level_for_downsample(64 * r); TISSUE = os.environ.get("TISSUE", "mask")
    if TISSUE == "theirs":   # their detector, physically scaled (smoke only: it takes pen rings and coverslip edges and misses faint tissue on our slides)
        W.segmentTissue(seg_level=seg_level, sthresh=PRESET["sthresh"], mthresh=PRESET["mthresh"], close=PRESET["close"], use_otsu=PRESET["use_otsu"],
                        filter_params={k_: PRESET[k_] for k_ in ("a_t", "a_h", "max_n_holes")}, ref_patch_size=int(round(512 * r)), gaus_otsu=PRESET["gaus_otsu"])
    else:                    # the Killcoyne tissue mask L-IMG used (slide_dir/../masks/<name>*), with their contour area / hole filters in the same physical units
        W.contours_tissue, W.holes_tissue = mask_contours(sp, s, (512 * r) ** 2)
    A0, B0 = int(round(224 * r)), int(round(448 * r)); S0 = int(round(1024 * 0.5 / mpp0))
    xyA = coords_for(W, A0, A0); xyB = coords_for(W, B0, B0); xyS = coords_for(W, S0, S0); xyC = coords_for(W, 1024, 960)
    t1 = time.time()
    for d in ("tiles/A", "tiles/B", "coords", "meta"): os.makedirs(f"{OD}/{d}", exist_ok=True)
    sA, sB = int(round(A0 / ds[1])), int(round(B0 / ds[2]))
    stA = read_tiles(s, xyA, 1, sA, f"{OD}/tiles/A/s{k:03d}.npy"); stB = read_tiles(s, xyB, 2, sB, f"{OD}/tiles/B/s{k:03d}.npy")
    np.savez(f"{OD}/coords/s{k:03d}.npz", A=xyA, B=xyB, segformer=xyS, cellvit=xyC, statsA=stA, statsB=stB, stats_cols=np.array(["sat", "rgb", "tis"]))
    tissue_px = sum(float(__import__("cv2").contourArea(c)) for c in W.contours_tissue)  # level-0 px^2, holes included
    m = {"k": k, "tissue": TISSUE, "mpp0": mpp0, "product": s.properties.get("hamamatsu.Product", ""), "r": r, "seg_level": seg_level, "seg_downsample": ds[seg_level],
         "n_contours": len(W.contours_tissue), "tissue_mm2": tissue_px * mpp0 ** 2 / 1e6,
         "A": {"n": len(xyA), "side_level0": A0, "read_level": 1, "read_px": sA, "out_mpp": ds[1] * mpp0 * sA / 224},
         "B": {"n": len(xyB), "side_level0": B0, "read_level": 2, "read_px": sB, "out_mpp": ds[2] * mpp0 * sB / 224},
         "segformer": {"n": len(xyS), "side_level0": S0}, "cellvit": {"n": len(xyC), "side_level0": 1024, "step_level0": 960},
         "A_keep": int(keep(stA).sum()), "B_keep": int(keep(stB).sum()), "bytes_A": os.path.getsize(f"{OD}/tiles/A/s{k:03d}.npy"), "bytes_B": os.path.getsize(f"{OD}/tiles/B/s{k:03d}.npy"),
         "sec_seg_coords": round(t1 - t0, 1), "sec_total": round(time.time() - t0, 1)}
    mt = f"{meta_f}.{os.getpid()}.tmp"; json.dump(m, open(mt, "w")); os.replace(mt, meta_f); print("TPBCN TILE", json.dumps(m), flush=True)

if TASK == "key":
    os.makedirs(OD, exist_ok=True); os.chmod(OD, 0o700); g = slide_key(); g.to_csv(OD + "/slide_key.csv", index=False); os.chmod(OD + "/slide_key.csv", 0o600)
    print("TPBCN KEY slides", len(g), "samples", len(set(";".join(g.cnv).split(";"))), "missing slide files", int((~g.slide_path.map(os.path.exists)).sum()))
elif TASK.startswith("tile_"):
    tile_one(int(TASK.split("_")[1]))
elif TASK == "estimate":     # storage estimate before the campaign: mask tissue area per slide (no slide pixels read)
    from PIL import Image; Image.MAX_IMAGE_PIXELS = None; import openslide; key = pd.read_csv(OD + "/slide_key.csv"); rows = []
    for r_ in key.itertuples():
        s = openslide.OpenSlide(r_.slide_path); mpp0 = float(s.properties["openslide.mpp-x"]); md = os.path.join(os.path.dirname(os.path.dirname(r_.slide_path)), "masks")
        cand = sorted(glob.glob(os.path.join(md, glob.escape(os.path.splitext(os.path.basename(r_.slide_path))[0]) + "*")))
        if not cand: rows.append({"k": r_.k, "mask": 0}); continue
        mk = np.array(Image.open(cand[0]).convert("L")) > 0; W0, H0 = s.level_dimensions[0]
        rows.append({"k": r_.k, "mask": 1, "mask_mm2": mk.sum() * (W0 / mk.shape[1]) * (H0 / mk.shape[0]) * mpp0 ** 2 / 1e6})
    E_ = pd.DataFrame(rows); tile_mm2 = (224 * THEIR_MPP / 1000) ** 2
    R = {"slides": len(E_), "slides_without_mask": int((E_["mask"] == 0).sum()), "mask_mm2": E_.mask_mm2.describe().round(2).to_dict(), "total_mask_mm2": float(E_.mask_mm2.sum()),
         "A_tiles_approx": float(E_.mask_mm2.sum() / tile_mm2), "A_GB_approx_uint8": float(E_.mask_mm2.sum() / tile_mm2 * 224 * 224 * 3 / 1e9)}
    R["B_GB_approx_uint8"] = R["A_GB_approx_uint8"] / 4; os.makedirs(RES, exist_ok=True); json.dump(R, open(RES + "/storage_estimate.json", "w"), indent=1); print("TPBCN ESTIMATE", json.dumps(R))
elif TASK == "smokestats":   # stats on already-written tile arrays + montage by tissue share (rule check)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; rng = np.random.default_rng(1); bins = [(0, .05), (.05, .15), (.15, .25), (.25, .4), (.4, 1.01)]
    fig, ax = plt.subplots(len(bins), 10, figsize=(13, 7.2)); R = {}
    st = np.concatenate([tile_stats(np.load(f, mmap_mode="r")) for f in sorted(glob.glob(OD + "/tiles/A/s*.npy"))]); arrs = [np.load(f, mmap_mode="r") for f in sorted(glob.glob(OD + "/tiles/A/s*.npy"))]
    idx = np.concatenate([np.stack([np.full(len(a), j), np.arange(len(a))], 1) for j, a in enumerate(arrs)])
    for b, (lo, hi) in enumerate(bins):
        w = np.flatnonzero((st[:, 2] >= lo) & (st[:, 2] < hi)); R[f"tis_{lo}_{hi}"] = int(len(w))
        for c, p in enumerate(rng.choice(w, min(10, len(w)), replace=False) if len(w) else []):
            j, t = idx[p]; ax[b, c].imshow(arrs[j][t]); ax[b, c].set_title(f"tis {st[p, 2]:.2f} sat {st[p, 0]:.0f}", fontsize=9)
    for x in ax.ravel(): x.axis("off")
    R["n"] = int(len(st)); R["sat_lt5"] = int((st[:, 0] < 5).sum()); R["rgb_lt50"] = int((st[:, 1] < 50).sum()); print("TPBCN SMOKESTATS", json.dumps(R))
    fig.suptitle("A tiles by tissue share (rows: < 0.05, 0.05-0.15, 0.15-0.25, 0.25-0.4, >= 0.4)", fontsize=10); fig.tight_layout(); fig.savefig(OD + "/smoke_tissue_bins.png", dpi=200)
elif TASK == "summary":
    ms = [json.load(open(f)) for f in sorted(glob.glob(OD + "/meta/s*.json"))]; D = pd.json_normalize(ms); n_key = len(pd.read_csv(OD + "/slide_key.csv"))
    q = lambda c: {"min": float(D[c].min()), "median": float(D[c].median()), "max": float(D[c].max()), "total": float(D[c].sum())}
    seg_bytes = 1024 * 1024 * 3; cv_bytes = 1024 * 1024 * 3
    R = {"slides_done": len(D), "slides_in_key": n_key, "products": D["product"].value_counts().to_dict(), "mpp0": D.mpp0.round(5).value_counts().to_dict(),
         "seg_level": D.seg_level.value_counts().to_dict(), "tissue_mm2": q("tissue_mm2"), "tiles_A": q("A.n"), "tiles_B": q("B.n"),
         "tiles_segformer": q("segformer.n"), "tiles_cellvit": q("cellvit.n"), "out_mpp_A": q("A.out_mpp"), "out_mpp_B": q("B.out_mpp"),
         "slides_with_0_tiles_A": int((D["A.n"] == 0).sum()), "storage_GB": {"A": D.bytes_A.sum() / 1e9, "B": D.bytes_B.sum() / 1e9,
         "segformer_raw_1024_if_extracted": D["segformer.n"].sum() * seg_bytes / 1e9, "cellvit_raw_1024_if_extracted": D["cellvit.n"].sum() * cv_bytes / 1e9},
         "sec_per_slide": q("sec_total")}
    bad = 0; kept = {"A": 0, "B": 0}
    for m_ in ms:   # completeness: every array has as many tiles as its coordinates and stats, all shaped 224 x 224 x 3 uint8
        z = np.load(f"{OD}/coords/s{m_['k']:03d}.npz")
        for v in ("A", "B"):
            a = np.load(f"{OD}/tiles/{v}/s{m_['k']:03d}.npy", mmap_mode="r"); kept[v] += int(keep(z[f"stats{v}"]).sum())
            bad += int(a.shape != (m_[v]["n"], 224, 224, 3) or a.dtype != np.uint8 or len(z[v]) != m_[v]["n"] or len(z[f"stats{v}"]) != m_[v]["n"])
    R["arrays_mismatched"] = bad; R["tmp_files_left"] = len(glob.glob(OD + "/tiles/*/*.tmp.npy")); R["tiles_kept_tis_ge_%.2f" % KEEP_TIS] = kept
    if len(D) < n_key: R["projected_all_slides_GB"] = {k_: v * n_key / len(D) for k_, v in R["storage_GB"].items()}; R["projected_tiles_A"] = R["tiles_A"]["total"] * n_key / len(D)
    os.makedirs(RES, exist_ok=True); fn = RES + ("/tiling_summary.json" if len(D) == n_key else "/tiling_smoke.json"); json.dump(R, open(fn, "w"), indent=1)
    print("TPBCN SUMMARY", json.dumps(R))
elif TASK == "montage":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(0); ks = sorted(int(os.path.basename(f)[1:4]) for f in glob.glob(OD + "/meta/s*.json"))[:3]
    fig, ax = plt.subplots(6, 12, figsize=(14, 7.6))
    for row, k in enumerate(ks):
        for j, (v, nper) in enumerate((("A", 8), ("B", 4))):
            a = np.load(f"{OD}/tiles/{v}/s{k:03d}.npy", mmap_mode="r"); pick = np.sort(rng.choice(len(a), min(nper * 2, len(a)), replace=False))
            for i, p in enumerate(pick):
                r_, c_ = 2 * row + i // nper, (i % nper) + (8 if v == "B" else 0); ax[r_, c_].imshow(a[p]); ax[r_, c_].set_title(f"slide {row + 1} {v}", fontsize=9)
    for x in ax.ravel(): x.axis("off")
    fig.suptitle("BEACON DACOR tiles (224 px): A = 0.455 um/px (cols 1-8), B = 0.910 um/px (cols 9-12); random tiles, 3 smoke slides", fontsize=10)
    fig.tight_layout(); fig.savefig(OD + "/montage_smoke.png", dpi=200); print("TPBCN MONTAGE", OD + "/montage_smoke.png")
    # tissue detection check: thumbnail cropped to the tissue, A tile boxes drawn (green)
    import openslide; from matplotlib.patches import Rectangle
    key = pd.read_csv(OD + "/slide_key.csv").set_index("k"); fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    for i, k in enumerate(ks):
        s = openslide.OpenSlide(key.loc[k, "slide_path"]); lv = s.get_best_level_for_downsample(32); d = float(s.level_downsamples[lv]); xy = np.load(f"{OD}/coords/s{k:03d}.npz")["A"]
        side = json.load(open(f"{OD}/meta/s{k:03d}.json"))["A"]["side_level0"]; x0, y0 = xy.min(0) - 4 * side; x1, y1 = xy.max(0) + 5 * side
        x0, y0 = max(int(x0), 0), max(int(y0), 0); w, h = int((x1 - x0) / d), int((y1 - y0) / d)
        ax[i].imshow(np.asarray(s.read_region((x0, y0), lv, (w, h)).convert("RGB")))
        for x, y in xy: ax[i].add_patch(Rectangle(((x - x0) / d, (y - y0) / d), side / d, side / d, fill=False, lw=0.3, ec="lime"))
        ax[i].set_title(f"slide {i + 1}: {len(xy)} A tiles", fontsize=10); ax[i].axis("off")
    fig.tight_layout(); fig.savefig(OD + "/tissue_smoke.png", dpi=200); print("TPBCN TISSUE", OD + "/tissue_smoke.png")
