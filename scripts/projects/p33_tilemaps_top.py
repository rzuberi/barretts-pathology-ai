"""P33: per-tile grade maps for the 8 benign-section T3a slides with the highest ABMIL field-effect score (all positives)
+ 4 highest-scoring negatives, using the saved tile MLP (fold model of the slide), drawn on the slide thumbnail.
Output PNGs to review_local/ (cluster) -> copy to laptop review/. Env: P33DIR, OUTDIR."""
import glob, json, os, sys, h5py, numpy as np, pandas as pd, torch, torch.nn as nn, openslide
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; Q = f"{T}/feasibility/erin_fusion/queue/results"; OUT = os.environ.get("OUTDIR", T + "/review_local/p33_tilemaps"); DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; COL = ["#bdbdbd", "#4daf4a", "#ffff33", "#ff7f00", "#e41a1c", "#7b1fa2"]
class TileMLP(nn.Module):
    def __init__(s, d=1536, n=6): super().__init__(); s.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, n))
    def forward(s, x): return s.net(x)
sd = torch.load(os.environ["P33DIR"] + "/tile_mlp_models.pt", map_location=DEV); S = pd.read_parquet(os.environ["P33DIR"] + "/tile_summary_all.parquet").set_index("uuid")
def nets_for(u):
    keys = [k for k in sd if k.startswith(f"{fold_of.get(u, 'final')}_s")] or [k for k in sd if k.startswith("final")]
    out = []
    for k in keys: m = TileMLP().to(DEV); m.load_state_dict(sd[k]); out.append(m.eval())
    return out
d = pd.read_csv(f"{T}/feasibility/erin_fusion/tasks/T3a.csv", dtype=str); d["y"] = d.y.astype(int); preds = {}
for fp in glob.glob(f"{Q}/img_T3a_f*_s*.json"):
    for k, v in json.load(open(fp))["preds"].items(): preds.setdefault(k, []).append(float(v))
d["abmil"] = d.sample_id.map(lambda k: np.mean(preds.get(k, [np.nan]))); d["uuid"] = d.h5_list.map(lambda p: os.path.basename(p).replace(".h5", ""))
lab = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str); m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
lab = lab.drop(columns=[c for c in ("anon_id",) if c in lab.columns]).merge(m[["h5", "anon_id"]], on="h5"); keys = sorted(lab.h5); pat = dict(zip(lab.h5, lab.anon_id))
rng = np.random.RandomState(0); uniq = sorted(set(pat.values())); fold_of_pat = {a: i % 5 for i, a in enumerate(rng.permutation(uniq))}; fold_of = {os.path.basename(k).replace(".h5", ""): fold_of_pat[pat[k]] for k in keys}
pick = pd.concat([d[d.y == 1].sort_values("abmil", ascending=False).head(8), d[d.y == 0].sort_values("abmil", ascending=False).head(4)])
os.makedirs(OUT, exist_ok=True); recs = []
for r in pick.itertuples():
    p = r.h5_list
    with h5py.File(p) as h: X = np.asarray(h["features"], np.float32); C = np.asarray(h["coords"]); lvl = int(h.attrs.get("level", 1)); sp = str(h.attrs.get("slide_path", ""))
    with torch.no_grad(): P = np.mean([torch.softmax(n(torch.tensor(X).to(DEV)), -1).cpu().numpy() for n in nets_for(r.uuid)], 0)
    am = P.argmax(1); frac = np.bincount(am, minlength=6) / len(am)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    try:
        sl = openslide.OpenSlide(sp); th = sl.get_thumbnail((1200, 1200)); ds = sl.level_downsamples[lvl]; W, H = sl.level_dimensions[lvl]; sc = th.size[0] / W
        ax[0].imshow(th); ax[0].scatter(C[:, 0] * sc, C[:, 1] * sc, c=[COL[i] for i in am], s=4, marker="s", linewidths=0); ax[0].set_axis_off()
    except Exception as e: ax[0].text(0.1, 0.5, f"thumbnail unavailable\n{e}"[:120]); ax[0].set_axis_off()
    ax[1].bar(CLASSES, frac, color=COL); ax[1].set_ylim(0, 1); ax[1].set_title(f"tile argmax fractions; ABMIL field-effect p={r.abmil:.2f}; section={r.grade}; y={r.y}", fontsize=8); plt.setp(ax[1].get_xticklabels(), rotation=30, fontsize=7)
    fig.suptitle(f"T3a slide {r.uuid[:8]} (case-max label {'LGD+' if r.y else 'benign'}); model {'fold' if r.uuid in fold_of else 'final'}", fontsize=9); fig.tight_layout(); fig.savefig(f"{OUT}/{r.y}_{r.abmil:.2f}_{r.uuid[:8]}.png", dpi=110); plt.close(fig)
    recs.append({"uuid": r.uuid, "y": int(r.y), "abmil": round(float(r.abmil), 3), "section": r.grade, **{f"frac_{c}": round(float(frac[i]), 3) for i, c in enumerate(CLASSES)}, "mean_p_LGDplus": round(float(P[:, 3:].sum(1).mean()), 3)})
json.dump(recs, open(f"{OUT}/index.json", "w"), indent=1); print(json.dumps(recs, indent=None))
