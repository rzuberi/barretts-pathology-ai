"""QC (CPU): for each slide in a task list, thumbnail with the kept tile grid outlined + 48 random tiles -> logs/qc/{slide}.png"""
import json, os, sys
import numpy as np, pandas as pd, openslide
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; os.makedirs(A + "/logs/qc", exist_ok=True)
s = pd.read_csv(A + "/manifests/slides.csv").set_index("slide")
for line in open(sys.argv[1]):
    ds, sl = line.strip().split()[-2:]; m = json.load(open(f"{A}/data/tiles/{ds}/{sl}.json")); T = np.load(f"{A}/data/tiles/{ds}/{sl}.npy", mmap_mode="r")
    w = openslide.OpenSlide(f"{A}/data/wsi/{ds}/{s.loc[sl, 'fname']}"); th = w.get_thumbnail((1200, 1200)); f = th.size[0] / m["W0"]
    fig = plt.figure(figsize=(14, 7)); ax = fig.add_axes([0, 0, 0.5, 1]); ax.imshow(th); ax.axis("off"); S = m["tile_side_level0"] * f
    for x, y in m["coords_level0_topleft"]: ax.add_patch(plt.Rectangle((x * f, y * f), S, S, fill=False, lw=0.3, ec="lime"))
    ax.set_title(f"{sl[:40]}  n={m['n_tiles']}  label={s.loc[sl, 'label']}", fontsize=8)
    r = np.random.RandomState(0).choice(len(T), min(48, len(T)), replace=False)
    for k, i in enumerate(r): a = fig.add_axes([0.5 + (k % 8) * 0.0625, 0.84 - (k // 8) * 0.14, 0.06, 0.13]); a.imshow(T[i]); a.axis("off")
    fig.savefig(f"{A}/logs/qc/{ds}_{sl[:23]}.png", dpi=90); plt.close(fig); print(sl, m["n_tiles"], m["mpp0"], m["read_px"], flush=True)
