"""Tile-level grade maps for ERIN (Rehan, 21 Sep 2026: 'one patch at a time ... a colour per grade ...
a confidence per grade'). Trains the six-class gated-attention MIL (same as 2.38b) under case-max
labels AND under section-resolved labels on the folds that exclude the chosen slides, saves the
weights, then for each chosen slide scores EVERY tile individually through the trained head
(per-tile softmax over NORMAL_OTHER<NDBE<IND<LGD<HGD<CANCER) and records the attention weight.
Renders, per slide, side-by-side maps: thumbnail | case-max-model argmax grade | section-model argmax
grade | expected-grade (sliding scale) | attention. Same slides for every model, by construction.
Env: N_PER_CLASS (default 2), OUTDIR.
"""
import json, os
import h5py, numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", "."); DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLASSES)}
COLORS = ["#bdbdbd", "#4daf4a", "#ffff33", "#ff7f00", "#e41a1c", "#7b1fa2"]; CMAP = ListedColormap(COLORS)
NPC = int(os.environ.get("N_PER_CLASS", "2"))
lab = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str)
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
lab = lab.merge(m[["h5", "anon_id"]], on="h5"); lab = lab[lab.worst_grade.isin(C_OF) & lab.case_max.isin(C_OF)].reset_index(drop=True)
keys = sorted(lab.h5); lab = lab.set_index("h5").loc[keys]; pat = dict(zip(keys, lab.anon_id))
y_slide = {k: C_OF[g] for k, g in zip(keys, lab.worst_grade)}; y_case = {k: C_OF[g] for k, g in zip(keys, lab.case_max)}
rng = np.random.RandomState(0); uniq = sorted(set(pat.values())); fold_of = {a: i % 5 for i, a in enumerate(rng.permutation(uniq))}
# choose display slides: per class, prefer slides whose section grade differs from case-max (where the schemes disagree)
SHOW_FOLD = int(os.environ.get("SHOW_FOLD", "4"))   # display slides come from ONE held-out fold; train on the other four
rs = np.random.RandomState(1); chosen = []
inf = lab[[fold_of[pat[k]] == SHOW_FOLD for k in lab.index]]
for c in CLASSES:
    cand = inf[(inf.worst_grade == c)]; dis = cand[cand.worst_grade != cand.case_max]; pool = dis if len(dis) >= NPC else cand
    chosen += list(rs.choice(pool.index, min(NPC, len(pool)), replace=False))
show_folds = {SHOW_FOLD}
assert all(fold_of[pat[k]] == SHOW_FOLD for k in chosen)
print("chosen", len(chosen), "held-out folds", sorted(show_folds), flush=True)
bags = {}
for k in keys:
    with h5py.File(k) as h: bags[k] = np.asarray(h["features"], dtype=np.float32)
class MC_MIL(nn.Module):
    def __init__(self, d_in=1536, n_cls=6):
        super().__init__(); self.emb = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.1))
        self.att_v = nn.Linear(512, 128); self.att_u = nn.Linear(512, 128); self.att_w = nn.Linear(128, 1); self.head = nn.Linear(512, n_cls)
    def tiles(self, bag):
        h = self.emb(bag); a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).softmax(0)
        return h, a.squeeze(-1)
    def forward(self, bag):
        h, a = self.tiles(bag); return self.head((a.unsqueeze(-1) * h).sum(0))
def train(ydict, tr, seed=0):
    torch.manual_seed(seed); net = MC_MIL().to(DEV); cnt = np.bincount([ydict[k] for k in tr], minlength=6).astype(float)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32, device=DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    r = np.random.RandomState(seed)
    for ep in range(30):
        net.train()
        for k in r.permutation(tr):
            b = bags[k]; b = b[r.choice(len(b), 2000, replace=False)] if len(b) > 2000 else b
            loss = nn.functional.cross_entropy(net(torch.tensor(b, device=DEV)).unsqueeze(0), torch.tensor([ydict[k]], device=DEV), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        print("epoch", ep, flush=True)
    return net.eval()
tr = [k for k in keys if fold_of[pat[k]] not in show_folds]   # never trained on a displayed slide's patient
print(f"training on {len(tr)} slides (folds excluded: {sorted(show_folds)})", flush=True)
os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)
nets = {}
for name, yd in (("case_max", y_case), ("section", y_slide)):
    wp = os.path.join(OUT, f"mcmil_{name}.pt")
    if os.path.exists(wp):   # resume: weights from a previous (rendering-killed) run
        net = MC_MIL().to(DEV); net.load_state_dict(torch.load(wp, map_location=DEV)); nets[name] = net.eval(); print("loaded", wp, flush=True)
    else:
        nets[name] = train(yd, tr); torch.save(nets[name].state_dict(), wp)
keep = {k: bags[k] for k in chosen}; del bags; bags = keep; import gc; gc.collect()   # free ~20 GB before rendering
try: import openslide
except Exception: openslide = None
summary = []
for k in chosen:
    with h5py.File(k) as h:
        coords = np.asarray(h["coords"]); attrs = dict(h.attrs); sp = attrs.get("slide_path", "")
    bag = torch.tensor(bags[k], device=DEV); rec = {"h5": os.path.basename(k), "section_grade": lab.loc[k, "worst_grade"], "case_max": lab.loc[k, "case_max"], "n_tiles": len(coords)}
    tile_out = {}
    with torch.no_grad():
        for name, net in nets.items():
            hh, a = net.tiles(bag); p_tile = torch.softmax(net.head(hh), -1).cpu().numpy(); slide_p = torch.softmax(net(bag), -1).cpu().numpy()
            tile_out[name] = {"p": p_tile, "attn": a.cpu().numpy()}
            rec[f"{name}_slide_pred"] = CLASSES[int(slide_p.argmax())]; rec[f"{name}_slide_probs"] = [round(float(x), 3) for x in slide_p]
            rec[f"{name}_tile_argmax_dist"] = {CLASSES[i]: int((p_tile.argmax(1) == i).sum()) for i in range(6)}
            rec[f"{name}_expected_grade_mean"] = round(float((p_tile * np.arange(6)).sum(1).mean()), 3)
    summary.append(rec)
    # render
    thumb = None
    if openslide is not None and sp and os.path.exists(sp):
        try:
            sl = openslide.OpenSlide(sp); W0, H0 = sl.dimensions; tw = 2500
            thumb = np.asarray(sl.get_thumbnail((tw, int(tw * H0 / W0))).convert("RGB"))
            ds_lvl = sl.level_downsamples[int(attrs.get("level", 0))]   # coords are stored in the extraction LEVEL's pixel space
            scale = thumb.shape[1] / W0 * ds_lvl   # level-coords -> level-0 -> thumbnail
        except Exception as e: print("openslide fail", e, flush=True)
    if thumb is None:
        W, H = coords.max(0) + 224; scale = 2000.0 / max(W, H); thumb = np.full((int(H * scale) + 1, int(W * scale) + 1, 3), 255, np.uint8)
    xy = (coords * scale).astype(int); ts = max(1, int(224 * scale))   # 224-px tiles in level coords
    fig, axes = plt.subplots(1, 5, figsize=(22, 5)); axes[0].imshow(thumb); axes[0].set_title(f"{rec['h5'][:8]}  section={rec['section_grade']}  case-max={rec['case_max']}", fontsize=9)
    def overlay(ax, vals, cmap, vmin, vmax, title):
        ax.imshow(thumb, alpha=0.35)
        ax.scatter(xy[:, 0] + ts / 2, xy[:, 1] + ts / 2, c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=max(2, (ts * 0.9) ** 2 / 4), marker="s", linewidths=0); ax.set_title(title, fontsize=9)
    overlay(axes[1], tile_out["case_max"]["p"].argmax(1), CMAP, -0.5, 5.5, "per-tile grade (trained on case-max)")
    overlay(axes[2], tile_out["section"]["p"].argmax(1), CMAP, -0.5, 5.5, "per-tile grade (trained on section)")
    overlay(axes[3], (tile_out["section"]["p"] * np.arange(6)).sum(1), "RdYlGn_r", 0, 5, "expected grade 0..5 (sliding scale, section model)")
    overlay(axes[4], tile_out["section"]["attn"] / tile_out["section"]["attn"].max(), "magma", 0, 1, "attention (section model)")
    for ax in axes: ax.axis("off")
    fig.savefig(os.path.join(OUT, "maps", f"{rec['h5'][:8]}_{rec['section_grade']}_{rec['case_max']}.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    np.savez(os.path.join(OUT, "maps", f"{rec['h5'][:8]}_tiles.npz"), coords=coords, **{f"{n}_p": v["p"] for n, v in tile_out.items()}, **{f"{n}_attn": v["attn"] for n, v in tile_out.items()})
    print("rendered", rec["h5"][:8], flush=True)
json.dump({"classes": CLASSES, "colors": COLORS, "slides": summary, "trained_on_slides": len(tr), "held_out_folds": sorted(show_folds)}, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("done")
