"""E: CONCH zero-shot tile grading on the same 100 pilot slides as the MedGemma image pilot (seed 7, stratified by
section grade), 64 highest-attention tiles per slide (section-trained MIL weights), prompt-ensemble zero-shot over
NORMAL_OTHER/NDBE/IND/LGD/HGD/CANCER. Slide score = mean/max/top-10% of tile P(LGD+); truth = section grade.
Gate (pre-registered): binary AUROC >= 0.70. Env: OUTDIR, N_SLIDES (100), TILES (64)."""
import json, os, sys
import h5py, numpy as np, pandas as pd, torch, torch.nn as nn, openslide
from PIL import Image
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/CONCH")
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", "."); DEV = "cuda" if torch.cuda.is_available() else "cpu"
NS = int(os.environ.get("N_SLIDES", "100")); K = int(os.environ.get("TILES", "64"))
CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLASSES)}
PROMPTS = {"NORMAL_OTHER": ["normal squamous oesophageal mucosa", "normal gastric mucosa", "benign squamous epithelium", "unremarkable mucosa without Barrett's"],
           "NDBE": ["Barrett's oesophagus with intestinal metaplasia, negative for dysplasia", "non-dysplastic Barrett's oesophagus", "intestinal metaplasia with goblet cells and no dysplasia", "columnar lined oesophagus without dysplasia"],
           "IND": ["Barrett's oesophagus indefinite for dysplasia", "columnar mucosa with atypia indefinite for dysplasia", "reactive atypia in Barrett's oesophagus, indefinite for dysplasia"],
           "LGD": ["Barrett's oesophagus with low-grade dysplasia", "low-grade glandular dysplasia in Barrett's oesophagus", "columnar epithelium with low-grade dysplasia"],
           "HGD": ["Barrett's oesophagus with high-grade dysplasia", "high-grade glandular dysplasia", "columnar epithelium with high-grade dysplasia and marked nuclear atypia"],
           "CANCER": ["oesophageal adenocarcinoma", "invasive adenocarcinoma arising in Barrett's oesophagus", "intramucosal adenocarcinoma", "adenocarcinoma with invasion"]}
TEMPLATES = ["CLASSNAME.", "a photomicrograph showing CLASSNAME.", "an H&E image of CLASSNAME.", "histopathology image of CLASSNAME.", "CLASSNAME is shown in this image."]
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="/mnt/scratche/slow/fmlab/zuberi01/phd/CONCH/checkpoints/conch/pytorch_model.bin"); model = model.to(DEV).eval()
tok = get_tokenizer()
with torch.inference_mode():
    W = []
    for c in CLASSES:
        texts = [t.replace("CLASSNAME", p) for p in PROMPTS[c] for t in TEMPLATES]
        e = model.encode_text(tokenize(texts=texts, tokenizer=tok).to(DEV)); e = e / e.norm(dim=-1, keepdim=True); e = e.mean(0); W.append(e / e.norm())
    W = torch.stack(W); scale = model.logit_scale.exp().item()
lab = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str); lab = lab[lab.worst_grade.isin(C_OF)]
rs = np.random.RandomState(7); per = max(1, NS // 6); chosen = []
if os.environ.get("ALL"):   # full run (gate passed on the 100-slide pilot): every dual-labelled slide
    chosen = sorted(lab.h5)
else:
    for c in CLASSES:
        pool = lab[lab.worst_grade == c]; chosen += list(pool.sample(min(per, len(pool)), random_state=rs).h5)
lab = lab.set_index("h5")
class MC_MIL(nn.Module):
    def __init__(self, d_in=1536, n_cls=6):
        super().__init__(); self.emb = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.1))
        self.att_v = nn.Linear(512, 128); self.att_u = nn.Linear(512, 128); self.att_w = nn.Linear(128, 1); self.head = nn.Linear(512, n_cls)
    def tiles(self, bag):
        h = self.emb(bag); a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).softmax(0); return h, a.squeeze(-1)
mil = MC_MIL(); mil.load_state_dict(torch.load(T + "/feasibility/runs/erin_tilemaps/output/mcmil_section.pt", map_location="cpu")); mil.eval()
rows = []; tile_rows = []
for h5p in chosen:
    try:
        with h5py.File(h5p) as h: X = np.asarray(h["features"], np.float32); coords = np.asarray(h["coords"]); attrs = dict(h.attrs)
        with torch.no_grad(): _, a = mil.tiles(torch.tensor(X)); a = a.numpy()
        top = np.argsort(-a)[:K]; sl = openslide.OpenSlide(attrs["slide_path"]); lvl = int(attrs.get("level", 0)); ds = sl.level_downsamples[lvl]
        imgs = [preprocess(sl.read_region((int(coords[i][0] * ds), int(coords[i][1] * ds)), lvl, (224, 224)).convert("RGB")) for i in top]
        with torch.inference_mode():
            f = model.encode_image(torch.stack(imgs).to(DEV), proj_contrast=True, normalize=True); P = torch.softmax(scale * f @ W.T, -1).cpu().numpy()
        plgd = P[:, 3:].sum(1); am = P.argmax(1)
        rows.append({"h5": h5p, "truth": lab.loc[h5p, "worst_grade"], "case_max": lab.loc[h5p, "case_max"], "n": len(top), "mean_plgd": float(plgd.mean()), "max_plgd": float(plgd.max()),
                     "top10_plgd": float(np.sort(plgd)[-max(1, len(plgd) // 10):].mean()), **{f"frac_{c}": float((am == i).mean()) for i, c in enumerate(CLASSES)}, "mean_probs": P.mean(0).round(4).tolist()})
        for i, idx in enumerate(top): tile_rows.append({"h5": os.path.basename(h5p), "tile": int(idx), "attn": float(a[idx]), "pred": CLASSES[int(am[i])], "p_lgdplus": float(plgd[i])})
        print("done", os.path.basename(h5p)[:8], lab.loc[h5p, "worst_grade"], "mean P(LGD+)", round(float(plgd.mean()), 3), flush=True)
    except Exception as e: print("ERR", h5p, e, flush=True)
d = pd.DataFrame(rows); pd.DataFrame(tile_rows).to_csv(os.path.join(OUT, "conch_tiles.csv"), index=False); d.to_csv(os.path.join(OUT, "conch_slides.csv"), index=False)
y = d.truth.isin(["LGD", "HGD", "CANCER"]).astype(int).values; yt = d.truth.map(C_OF).values; MP = np.stack(d.mean_probs.values)
res = {"n_slides": len(d), "tiles_per_slide": K, "truth_dist": d.truth.value_counts().to_dict(),
       "auroc_LGDplus": {a: round(float(roc_auc_score(y, d[a])), 4) for a in ("mean_plgd", "max_plgd", "top10_plgd")},
       "macro_auroc_6class_mean_probs": round(float(np.mean([roc_auc_score((yt == c).astype(int), MP[:, c]) for c in range(6) if 0 < (yt == c).sum() < len(yt)])), 4),
       "tile_pred_dist": pd.DataFrame(tile_rows).pred.value_counts().to_dict(), "mean_tile_argmax_frac_by_truth": d.groupby("truth")[[f"frac_{c}" for c in CLASSES]].mean().round(3).to_dict("index")}
res["gate_0.70"] = bool(max(res["auroc_LGDplus"].values()) >= 0.70)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2); print(json.dumps(res, indent=1))
