"""CPU unit: Figure 2. WGD ABMIL attention (repeat-0 OOF model of each patient's test fold) for 3 cases chosen by rule
(DESIGN.md): most confident correct WGD+, most confident correct WGD-, most confident error. Each: slide thumbnail with
attention overlay + top-8 attended tiles (re-read from the slide at 0.5 um/px). Writes figures/fig2_attention.{pdf,png} and
results/heatmap_cases.json."""
import json
import h5py, numpy as np, pandas as pd, openslide
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia"
p0 = pd.read_csv(A + "/preds/wgd_strat_abmil_r0.csv").set_index("patient")
at = np.load(A + "/attn/wgd_strat_abmil_r0.npz")
pos = p0[p0.y == 1].score.idxmax(); neg = p0[p0.y == 0].score.idxmin()
fp = p0[p0.y == 0].score.idxmax(); fn = p0[p0.y == 1].score.idxmin()
err = fp if p0.loc[fp, "score"] - 0.5 >= 0.5 - p0.loc[fn, "score"] else fn
cases = [(pos, "Correct: WGD+"), (neg, "Correct: WGD−"), (err, "Error: " + ("false positive" if err == fp else "false negative"))]
info = []
plt.rcParams.update({"font.size": 6, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(6.2, 4.4))
gs = fig.add_gridspec(3, 3, width_ratios=[1.7, 2.0, 0.05], wspace=0.05, hspace=0.32)
for r, (p, ttl) in enumerate(cases):
    with h5py.File(f"{A}/feats/{p}.h5") as h:
        xy = h["coords"][:]; S0 = float(h.attrs["tile_px_level0"]); sp = h.attrs["slide_path"]
    a = at[p]; assert len(a) == len(xy)
    sl = openslide.OpenSlide(sp); W0, H0 = sl.dimensions; th = sl.get_thumbnail((900, 900)); sc = th.size[0] / W0
    ax = fig.add_subplot(gs[r, 0]); ax.imshow(th); ax.axis("off")
    rk = a.argsort().argsort() / max(len(a) - 1, 1)   # attention percentile per tile
    sct = ax.scatter((xy[:, 0] + S0 / 2) * sc, (xy[:, 1] + S0 / 2) * sc, c=rk, cmap="magma", s=2.0, alpha=0.75, linewidths=0, vmin=0, vmax=1)
    ax.set_title(f"{ttl}  ·  {p}  ·  P(WGD) = {p0.loc[p, 'score']:.2f}", fontsize=6, loc="left")
    top = a.argsort()[::-1][:8]
    sub = gs[r, 1].subgridspec(2, 4, wspace=0.03, hspace=0.03)
    for j, t in enumerate(top):
        x, y = xy[t]; lvl = sl.get_best_level_for_downsample(S0 / 224 + 1e-6); ds = sl.level_downsamples[lvl]
        im = sl.read_region((int(x), int(y)), lvl, (int(round(S0 / ds)),) * 2).convert("RGB").resize((224, 224), Image.BILINEAR)
        axt = fig.add_subplot(sub[j // 4, j % 4]); axt.imshow(im); axt.set_xticks([]); axt.set_yticks([])
        for s_ in axt.spines.values(): s_.set_linewidth(0.4)
    info.append({"patient": p, "role": ttl, "p_wgd": round(float(p0.loc[p, "score"]), 4), "y": int(p0.loc[p, "y"]), "cohort": p0.loc[p, "cohort"],
                 "n_tiles": int(len(a)), "top8_attention_share": round(float(np.sort(a)[::-1][:8].sum()), 4)})
cax = fig.add_subplot(gs[:, 2]); cb = fig.colorbar(sct, cax=cax); cb.set_label("attention percentile", fontsize=6); cb.ax.tick_params(labelsize=5)
for f in ("pdf", "png"): fig.savefig(f"{A}/figures/fig2_attention.{f}", bbox_inches="tight", dpi=300)
json.dump(info, open(A + "/results/heatmap_cases.json", "w"), indent=1); print(json.dumps(info, indent=1))
