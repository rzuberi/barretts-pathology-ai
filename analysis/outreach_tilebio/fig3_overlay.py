"""Figure 3 (CPU): one CAMELYON16 test tumour slide, attention-top-16 tiles (UNI2-h ABMIL trained on all tiles, attention averaged over
3 seeds) over the tumour annotation. Slide chosen by a rule fixed before looking (DESIGN.md status log): the test tumour slide whose
tumour-tile fraction is the median over all test tumour slides (lower median if even)."""
import json, os, glob, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, openslide
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; MO = os.environ.get("MODEL", "uni2h")
L = json.load(open(A + "/results/c16_lesion_tiles.json")); L = {k: v for k, v in L.items() if v["n_tumour_tiles"] > 0}
order = sorted(L, key=lambda k: (L[k]["tumour_frac"], k)); sl = order[(len(order) - 1) // 2]; g = L[sl]
zs = [np.load(f) for f in sorted(glob.glob(f"{A}/preds/c16_{MO}_all_f0_s*.npz"))]; a = np.mean([z[f"attn|{sl}"].astype(np.float32) for z in zs], 0)
top = np.argsort(-a)[:16]; tum = set(g["tumour_idx"]); m = json.load(open(f"{A}/data/tiles/c16/{sl}.json")); C = np.array(m["coords_level0_topleft"]); S0 = m["tile_side_level0"]
w = openslide.OpenSlide(f"{A}/data/wsi/c16/{sl}.tif"); th = w.get_thumbnail((2400, 2400)); f = th.size[0] / m["W0"]
T = np.load(f"{A}/data/tiles/c16/{sl}.npy", mmap_mode="r")
plt.rcParams.update({"font.size": 7})
polys = []
for an in ET.parse(f"{A}/data/c16_meta/annotations/{sl}.xml").getroot().iter("Annotation"):
    pts = sorted(((int(c.get("Order")), float(c.get("X")), float(c.get("Y"))) for c in an.iter("Coordinate")))
    if len(pts) >= 3: polys.append((np.array([(x, y) for _, x, y in pts] + [pts[0][1:]]), an.get("PartOfGroup", "") in ("Exclusion", "_2")))
hits = sum(int(i in tum) for i in top)
fig = plt.figure(figsize=(7.0, 2.75))
# (a) whole slide, cropped to tissue, with the zoom window
ax = fig.add_axes([0.0, 0.0, 0.3, 0.86]); x0, y0 = C.min(0) * f; x1, y1 = (C.max(0) + S0) * f; pad = 20
ax.imshow(th); ax.set_xlim(max(0, x0 - pad), min(th.size[0], x1 + pad)); ax.set_ylim(min(th.size[1], y1 + pad), max(0, y0 - pad)); ax.axis("off")
for P, ex in polys: ax.plot(P[:, 0] * f, P[:, 1] * f, color="#555555" if ex else "#e0001b", lw=0.6)
T0 = C[top]; zx0, zy0 = T0.min(0) - 3 * S0; zx1, zy1 = T0.max(0) + 4 * S0; side = max(zx1 - zx0, zy1 - zy0); cx, cy = (zx0 + zx1) / 2, (zy0 + zy1) / 2
zx0, zy0 = cx - side / 2, cy - side / 2
ax.add_patch(plt.Rectangle((zx0 * f, zy0 * f), side * f, side * f, fill=False, lw=1.0, ec="black"))
ax.set_title("(a) whole slide; box = zoom", fontsize=6.5, loc="left")
# (b) zoom read from the WSI
lvl = w.get_best_level_for_downsample(side / 900); ds_ = w.level_downsamples[lvl]
Z = w.read_region((int(zx0), int(zy0)), lvl, (int(side / ds_), int(side / ds_))).convert("RGB")
az = fig.add_axes([0.31, 0.0, 0.33, 0.86]); az.imshow(Z, extent=(zx0, zx0 + side, zy0 + side, zy0)); az.axis("off")
for P, ex in polys: az.plot(P[:, 0], P[:, 1], color="#555555" if ex else "#e0001b", lw=0.9, ls=":" if ex else "-")
for i in top: az.add_patch(plt.Rectangle(tuple(C[i]), S0, S0, fill=False, lw=0.9, ec="#00a0ff"))
az.set_xlim(zx0, zx0 + side); az.set_ylim(zy0 + side, zy0)
az.set_title("(b) zoom: tumour (red), top-16 tiles (blue)", fontsize=6.5, loc="left")
# (c) the 16 tiles
for k, i in enumerate(top):
    axi = fig.add_axes([0.66 + (k % 4) * 0.085, 0.64 - (k // 4) * 0.205, 0.08, 0.195]); axi.imshow(T[i]); axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values(): sp.set_edgecolor("#e0001b" if i in tum else "#999999"); sp.set_linewidth(1.4)
fig.text(0.66, 0.87, "(c) the 16 tiles (red = in tumour)", fontsize=6.5)
fig.text(0.0, 0.955, f"{sl}, {g['lesion']}-metastasis, UNI2-h ABMIL: {g['n_tumour_tiles']} of {g['n_tiles']} tissue tiles ({100*g['tumour_frac']:.1f}%) lie in the annotated tumour; "
         f"{hits}/16 attention-top tiles do", fontsize=6.8)
[fig.savefig(f"{A}/figures/fig3_c16_attention_overlay.{e}", dpi=300) for e in ("pdf", "png")]
json.dump({"slide": sl, "model": MO, "rule": "median tumour-tile fraction among test tumour slides", "hits_top16": hits, **{k: g[k] for k in ("lesion", "n_tiles", "n_tumour_tiles", "tumour_frac")}},
          open(A + "/results/fig3_slide.json", "w"), indent=1); print(sl, hits)
