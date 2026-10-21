"""Exploration E3 (docs/paper_explore_E3.md): per-sample pooled image features from tile embeddings that already exist (no slide pixels read).
SRC = one stored extractor/scale; writes killcoyne_mm/explore_E3/feats/<SRC>_<pool>.csv.gz (Sample + columns, set C order) for pools
mean | max | meansd | q3 (q10, q50, q90) | top8n / top16n / top32n (mean of the k tiles with the largest L2 norm) and <SRC>_endo.csv.gz
(the mean over every tile of the sample's endoscopy, pre-event and post-event samples pooled separately; the ALT rows of the refit).
Tile-score pooling uses the stored UNI2-h bags (killcoyne_mm/bags, km_prep.py).
A sample's bag is the union of its slides' tiles (km_prep.py); a sample with no slide in the source gets NaN (imputed with training means in the refit).
Row-level outputs stay on the cluster."""
import os, sys, json, numpy as np, pandas as pd, h5py
K = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"; M = K + "_mm"; E3 = M + "/explore_E3"; FD = E3 + "/feats"
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/data"; TRI = "/mnt/scratche/slow/fmlab/zuberi01/phd/TRIDENT/trident_processed"
SRCS = {  # name -> (kind, location); npz256 = 256 sampled tiles per slide (coords are tile centres, unused here); h5 = TRIDENT all-tissue tiles at 20x (0.5 um/px)
    "u2l2": ("npz", B + "/foundation_outputs/uni2_tile224_lvl2"), "u2l1": ("npz", B + "/foundation_outputs/uni2_tile224_lvl1"),
    "gpl1": ("npz", B + "/foundation_outputs/gigapath_tile224_lvl1"), "gpl2": ("npz", B + "/foundation_outputs/gigapath_tile224_lvl2"),
    "gpl3": ("npz", B + "/foundation_outputs/gigapath_tile224_lvl3"), "v2l2": ("npz", B + "/virchow2_outputs"),
    "ho20": ("h5", TRI + "/20x_224px_0px_overlap/features_hoptimus0"), "v220": ("h5", TRI + "/20x_224px_0px_overlap/features_virchow2"),
    "co20": ("h5", TRI + "/20x_448px_0px_overlap/features_conch_v15"), "ct20": ("h5", TRI + "/20x_224px_0px_overlap/features_ctranspath"),
    "mu20": ("h5", TRI + "/20x_384px_0px_overlap/features_musk")}
SRC = os.environ["SRC"]; os.makedirs(FD, exist_ok=True)
if SRC.startswith("cat_"):   # multi-scale / multi-extractor concatenation of existing mean blocks: SRC=cat_<name> PARTS="u2l1 u2l2" -> cat_<name>_mean, cat_<name>_endo
    P = os.environ["PARTS"].split()
    for how in ("mean", "endo"):
        L = [pd.read_csv(f"{FD}/{p}_{how}.csv.gz", dtype={"Sample": str}).set_index("Sample") for p in P]; assert all((l.index == L[0].index).all() for l in L)
        X = np.hstack([l.values for l in L]); df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]); df.insert(0, "Sample", L[0].index.values); df.to_csv(f"{FD}/{SRC}_{how}.csv.gz", index=False, float_format="%.7g")
    print("EXE3 FEATS DONE", SRC, P, flush=True); raise SystemExit(0)
kind, loc = SRCS[SRC]
C = pd.read_csv(M + "/set_C.csv", dtype=str); ids = C.Sample.values
SM = pd.read_csv(M + "/horizons/samples.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); PRE = SM.pre.astype(bool).values
ENDO = (C.Patient + "|" + C.Endoscopy + "|" + np.where(PRE, "pre", "post")).values
mp = pd.read_csv(K + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(ids)].sort_values(["cnv", "npz"])
def slide_file(r):
    if kind == "npz": return f"{loc}/slide_embeddings/{os.path.basename(r.npz)}"
    return f"{loc}/{os.path.splitext(r.base)[0]}.h5"
def read(f):
    if kind == "npz":
        with np.load(f, allow_pickle=True) as a: return np.asarray(a["embeddings"], dtype=np.float32)
    with h5py.File(f, "r") as h: return np.asarray(h["features"], dtype=np.float32)
bags = {}; miss_slides = 0
for s, g in mp.groupby("cnv"):
    L = []
    for r in g.itertuples():
        f = slide_file(r)
        if os.path.exists(f): L.append(read(f))
        else: miss_slides += 1
    if L: b = np.concatenate(L, 0); assert b.ndim == 2 and np.isfinite(b).all(), s; bags[s] = b
d = next(iter(bags.values())).shape[1]; nt = np.array([len(bags[s]) if s in bags else 0 for s in ids])
def pool(b, how):
    if how == "mean": return b.mean(0)
    if how == "max": return b.max(0)
    if how == "meansd": return np.concatenate([b.mean(0), b.std(0)])
    if how == "q3": return np.quantile(b, [0.1, 0.5, 0.9], axis=0).reshape(-1)
    if how.startswith("top"):
        k = int(how[3:-1]); o = np.argsort(-np.linalg.norm(b, axis=1), kind="mergesort")[:k]; return b[o].mean(0)
    raise KeyError(how)
POOLS = ["mean", "max", "meansd", "q3", "top8n", "top16n", "top32n"] if SRC in ("u2l2", "ho20", "v220", "gpl2", "co20") else ["mean"]
for how in POOLS:
    X = np.vstack([pool(bags[s], how) if s in bags else np.full(len(pool(np.zeros((2, d), np.float32), how)), np.nan) for s in ids])
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]); df.insert(0, "Sample", ids); df.to_csv(f"{FD}/{SRC}_{how}.csv.gz", index=False, float_format="%.7g")
# endoscopy-pooled mean over the union of the endoscopy's tiles (same pre/post status)
EX = np.full((len(ids), d), np.nan)
for e in np.unique(ENDO):
    m = np.where(ENDO == e)[0]; L = [bags[ids[i]] for i in m if ids[i] in bags]
    if L: EX[m] = np.concatenate(L, 0).mean(0)
df = pd.DataFrame(EX, columns=[f"f{i}" for i in range(d)]); df.insert(0, "Sample", ids); df.to_csv(f"{FD}/{SRC}_endo.csv.gz", index=False, float_format="%.7g")
if SRC == "u2l2":   # rebuild check against the stored slide-mean matrix (km_prep.py)
    IM = pd.read_csv(M + "/img_mean.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids).values; mx = float(np.nanmax(np.abs(IM - np.vstack([bags[s].mean(0) for s in ids]))))
else: mx = None
info = {"src": SRC, "kind": kind, "dim": int(d), "samples_with_tiles": int((nt > 0).sum()), "samples": int(len(ids)), "slides_missing": int(miss_slides),
        "tiles_per_sample": {"median": float(np.median(nt[nt > 0])), "min": int(nt[nt > 0].min()), "max": int(nt.max())}, "pools": POOLS,
        "img_mean_rebuild_max_abs_diff": mx, "endoscopies_pre": int(len({e for e in ENDO if e.endswith("pre")})),
        "mixed_pre_post_endoscopies": int(pd.Series(PRE).groupby((C.Patient + "|" + C.Endoscopy).values).nunique().gt(1).sum())}
json.dump(info, open(f"{FD}/{SRC}_info.json", "w"), indent=1); print(json.dumps(info), "\nEXE3 FEATS DONE", SRC, flush=True)
