"""CONCH zero-shot on the SWG Barrett's (histology + CNV) release: all 707 release slides, the 256 stored tiles per
slide (coords_level = tile centres in level-2 pixel space, 224 px), Barrett's prompt ensemble -> per-tile class probabilities.
Readouts: (i) grade vs the release PATHOLOGIST label (LGD+ = Label >= 2; HGD+ = Label >= 3) — the human image anchor;
(ii) zero-shot IMAGE arm for progression (release endpoint y_progressor, sample and patient level) vs trained
image_only 0.739 (sample) / 0.731 (patient). Writes per-slide scores for later fusion with CNV. Env: OUTDIR."""
import json, os, sys
import numpy as np, pandas as pd, torch, openslide
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/CONCH")
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
SL = "/mnt/scratche/slow/fmlab/datasets/imaging/SWGCohort/slides"; OUT = os.environ.get("OUTDIR", "."); DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]
PROMPTS = {"NORMAL_OTHER": ["normal squamous oesophageal mucosa", "normal gastric mucosa", "benign squamous epithelium", "unremarkable mucosa without Barrett's"],
           "NDBE": ["Barrett's oesophagus with intestinal metaplasia, negative for dysplasia", "non-dysplastic Barrett's oesophagus", "intestinal metaplasia with goblet cells and no dysplasia", "columnar lined oesophagus without dysplasia"],
           "IND": ["Barrett's oesophagus indefinite for dysplasia", "columnar mucosa with atypia indefinite for dysplasia", "reactive atypia in Barrett's oesophagus, indefinite for dysplasia"],
           "LGD": ["Barrett's oesophagus with low-grade dysplasia", "low-grade glandular dysplasia in Barrett's oesophagus", "columnar epithelium with low-grade dysplasia"],
           "HGD": ["Barrett's oesophagus with high-grade dysplasia", "high-grade glandular dysplasia", "columnar epithelium with high-grade dysplasia and marked nuclear atypia"],
           "CANCER": ["oesophageal adenocarcinoma", "invasive adenocarcinoma arising in Barrett's oesophagus", "intramucosal adenocarcinoma", "adenocarcinoma with invasion"]}
TEMPLATES = ["CLASSNAME.", "a photomicrograph showing CLASSNAME.", "an H&E image of CLASSNAME.", "histopathology image of CLASSNAME.", "CLASSNAME is shown in this image."]
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="/mnt/scratche/slow/fmlab/zuberi01/phd/CONCH/checkpoints/conch/pytorch_model.bin"); model = model.to(DEV).eval(); tok = get_tokenizer()
with torch.inference_mode():
    W = []
    for c in CLASSES:
        e = model.encode_text(tokenize(texts=[t.replace("CLASSNAME", p) for p in PROMPTS[c] for t in TEMPLATES], tokenizer=tok).to(DEV)); e = e / e.norm(dim=-1, keepdim=True); e = e.mean(0); W.append(e / e.norm())
    W = torch.stack(W); scale = model.logit_scale.exp().item()
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).merge(man, left_on="SampleID", right_on="sample_id")
uidx = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str); uidx = uidx[uidx.status == "ok"]; npz_of = dict(zip(uidx.sample_id, uidx.npz_path))
out_csv = os.path.join(OUT, "conch_swg_slides.csv"); done = set(pd.read_csv(out_csv, dtype=str).sample_id) if os.path.exists(out_csv) else set()
if not os.path.exists(out_csv): open(out_csv, "w").write("sample_id,patient_id,label,y_progressor,n_tiles,mean_plgd,max_plgd,top10_plgd,mean_phgd," + ",".join(f"mean_p_{c}" for c in CLASSES) + "\n")
for r in coh.itertuples():
    sid = r.sample_id
    if sid in done or sid not in npz_of: continue
    try:
        z = np.load(npz_of[sid], allow_pickle=True); coords = np.asarray(z["coords_level"]); lvl = int(z["level"]); ts = int(z["tile_size"]); path = os.path.join(SL, os.path.basename(str(z["slide_path"])))
        sl = openslide.OpenSlide(path); ds = sl.level_downsamples[lvl]; tl = np.clip(coords - ts // 2, 0, np.array(sl.level_dimensions[lvl]) - ts)   # coords_level are tile centres
        imgs = [preprocess(sl.read_region((int(x * ds), int(y * ds)), lvl, (ts, ts)).convert("RGB")) for x, y in tl]
        P = []
        with torch.inference_mode():
            for i in range(0, len(imgs), 128):
                f = model.encode_image(torch.stack(imgs[i:i + 128]).to(DEV), proj_contrast=True, normalize=True); P.append(torch.softmax(scale * f @ W.T, -1).cpu().numpy())
        P = np.concatenate(P); pl = P[:, 3:].sum(1); ph = P[:, 4:].sum(1)
        open(out_csv, "a").write(f"{sid},{r.patient_id},{r.Label},{r.y_progressor},{len(P)},{pl.mean():.5f},{pl.max():.5f},{np.sort(pl)[-max(1, len(pl) // 10):].mean():.5f},{ph.mean():.5f}," + ",".join(f"{v:.5f}" for v in P.mean(0)) + "\n")
        print("done", sid, r.Label, round(float(pl.mean()), 3), flush=True)
    except Exception as e: print("ERR", sid, e, flush=True)
d = pd.read_csv(out_csv); d["lab"] = pd.to_numeric(d.label, errors="coerce"); d["y"] = pd.to_numeric(d.y_progressor, errors="coerce"); ok = d.lab.notna()
res = {"n_slides": len(d), "grade_dist": d.lab.value_counts().to_dict(), "grade_vs_pathologist": {}, "progression_zero_shot": {}}
for tag, thr in (("LGDplus", 2), ("HGDplus", 3)):
    yy = (d.lab[ok] >= thr).astype(int)
    if 0 < yy.sum() < len(yy): res["grade_vs_pathologist"][tag] = {a: round(float(roc_auc_score(yy, d.loc[ok, a])), 4) for a in ("mean_plgd", "max_plgd", "top10_plgd", "mean_phgd")}
for a in ("mean_plgd", "max_plgd", "top10_plgd", "mean_phgd"):
    pat = d.groupby("patient_id").agg(y=("y", "max"), s=(a, "max"))
    res["progression_zero_shot"][a] = {"sample_auroc": round(float(roc_auc_score(d.y, d[a])), 4), "patient_auroc_max": round(float(roc_auc_score(pat.y, pat.s)), 4)}
res["reference_trained_image_only"] = {"sample": 0.739, "patient": 0.7312}; res["reference_pathologist_grade_as_score_patient"] = 0.687
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2); print(json.dumps(res, indent=1))
