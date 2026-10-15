"""Item 29: is CONCH's weak SWG result a tile-SCALE artefact? Same slides, same tissue locations, three scales:
the stored level-2 tiles (~0.88 um/px, what task_conch_swg used), level-1 tiles (~0.44 um/px) and level-0 tiles
(~0.22 um/px, native 40x); 224 px each, centred on the same points. Readout: AUROC vs pathologist LGD+ per scale on
a stratified subset (all HGD+/IND-LGD slides + a matched number of NDBE), plus per-tile argmax fractions."""
import json, os, sys, numpy as np, pandas as pd, torch, openslide
from PIL import Image
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/CONCH")
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
SL = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort/slides"; OUT = os.environ.get("OUTDIR", "."); DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_PER = int(os.environ.get("N_PER", "40")); TILES = int(os.environ.get("TILES", "128"))
CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]
PROMPTS = {"NORMAL_OTHER": ["normal squamous oesophageal mucosa", "normal gastric mucosa", "benign oesophageal tissue without Barrett's"],
           "NDBE": ["Barrett's oesophagus with intestinal metaplasia and no dysplasia", "non-dysplastic Barrett's mucosa with goblet cells", "columnar-lined oesophagus without dysplasia"],
           "IND": ["Barrett's oesophagus indefinite for dysplasia", "columnar epithelium with reactive atypia, indefinite for dysplasia"],
           "LGD": ["Barrett's oesophagus with low-grade dysplasia", "low-grade glandular dysplasia in Barrett's mucosa"],
           "HGD": ["Barrett's oesophagus with high-grade dysplasia", "high-grade glandular dysplasia with marked nuclear atypia"],
           "CANCER": ["oesophageal adenocarcinoma", "invasive adenocarcinoma arising in Barrett's oesophagus"]}
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="/mnt/scratche/slow/fmlab/zuberi01/phd/CONCH/checkpoints/conch/pytorch_model.bin"); model = model.to(DEV).eval(); tok = get_tokenizer()
with torch.inference_mode():
    T = []
    for c in CLASSES:
        e = model.encode_text(tokenize(texts=PROMPTS[c], tokenizer=tok).to(DEV)); e = e / e.norm(dim=-1, keepdim=True); T.append(e.mean(0))
    T = torch.stack(T); T = T / T.norm(dim=-1, keepdim=True)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID")
uidx = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str); uidx = uidx[uidx.status == "ok"]; npz_of = dict(zip(uidx.sample_id, uidx.npz_path))
man["label"] = pd.to_numeric(pd.Series(coh.Label.reindex(man.sample_id).values), errors="coerce").fillna(0).astype(int)
rs = np.random.RandomState(0)
sub = pd.concat([man[man.label >= 3], man[(man.label >= 1) & (man.label < 3)].sample(min(N_PER, int(((man.label >= 1) & (man.label < 3)).sum())), random_state=0), man[man.label == 0].sample(N_PER, random_state=0)])
print("subset:", len(sub), sub.label.value_counts().to_dict(), flush=True)
out_csv = os.path.join(OUT, "conch_swg_scale_slides.csv"); done = set(pd.read_csv(out_csv, dtype=str).sample_id) if os.path.exists(out_csv) else set()
if not os.path.exists(out_csv): open(out_csv, "w").write("sample_id,label,scale_level,mpp,n_tiles,mean_plgd,max_plgd,mean_phgd," + ",".join(f"frac_{c}" for c in CLASSES) + "\n")
def score(tiles):
    with torch.inference_mode():
        x = torch.stack([preprocess(t) for t in tiles]).to(DEV); e = model.encode_image(x, proj_contrast=True, normalize=True); p = (e @ T.T * model.logit_scale.exp()).softmax(-1).cpu().numpy()
    return p
for r in sub.itertuples():
    if r.sample_id in done: continue
    z = np.load(npz_of[r.sample_id], allow_pickle=True); coords = np.asarray(z["coords_level"]); lvl2 = int(z["level"]); ts = int(z["tile_size"])
    path = os.path.join(SL, os.path.basename(str(z["slide_path"])))
    try: sl = openslide.OpenSlide(path)
    except Exception as e: print("[skip]", r.sample_id, e, flush=True); continue
    mpp0 = float(sl.properties.get("openslide.mpp-x", 0.22)); ds2 = sl.level_downsamples[lvl2]
    sel = coords[rs.choice(len(coords), min(TILES, len(coords)), replace=False)]
    centres0 = [((x + ts / 2) * ds2, (y + ts / 2) * ds2) for x, y in sel]   # level-0 centre of each stored tile
    for L in sorted({lvl2, max(lvl2 - 1, 0), 0}):
        dsL = sl.level_downsamples[L]; tiles = []
        for cx, cy in centres0:
            x0 = int(cx - 112 * dsL); y0 = int(cy - 112 * dsL); tiles.append(sl.read_region((x0, y0), L, (224, 224)).convert("RGB"))
        p = score(tiles); am = p.argmax(1)
        plgd = p[:, 3:].sum(1); phgd = p[:, 4:].sum(1)
        with open(out_csv, "a") as fh: fh.write(f"{r.sample_id},{r.label},{L},{mpp0*dsL:.3f},{len(tiles)},{plgd.mean():.4f},{plgd.max():.4f},{phgd.mean():.4f}," + ",".join(f"{(am==i).mean():.3f}" for i in range(len(CLASSES))) + "\n")
    print("done", r.sample_id, r.label, flush=True)
d = pd.read_csv(out_csv, dtype={"sample_id": str}); d["y"] = (d.label >= 2).astype(int); d["yh"] = (d.label >= 3).astype(int)
res = {"n_slides": int(d.sample_id.nunique()), "label_dist": d.drop_duplicates("sample_id").label.value_counts().to_dict(), "tiles_per_slide": TILES, "by_scale": {}}
for L, g in d.groupby("scale_level"):
    res["by_scale"][f"level_{L}"] = {"mpp": round(float(g.mpp.median()), 3), "n": int(len(g)),
        "auroc_LGDplus": {k: round(float(roc_auc_score(g.y, g[k])), 4) for k in ("mean_plgd", "max_plgd", "mean_phgd")} if g.y.nunique() > 1 else None,
        "auroc_HGDplus": {k: round(float(roc_auc_score(g.yh, g[k])), 4) for k in ("mean_plgd", "max_plgd", "mean_phgd")} if g.yh.nunique() > 1 else None,
        "mean_tile_argmax_frac_by_truth": {str(l): {c: round(float(gg[f"frac_{c}"].mean()), 3) for c in CLASSES} for l, gg in g.groupby("label")}}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2); print(json.dumps(res, indent=1))
