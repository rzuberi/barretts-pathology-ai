"""Slide table for the public tile-budget probe: manifests/slides.csv (ds, slide, url, size, label, split, fold).
tcga: LUAD=0 / LUSC=1, one diagnostic slide per patient, patient-level stratified 5-fold (seed 20261019).
c16:  CAMELYON16 official split; train labels from file names, test labels from evaluation/reference.csv; annotations fetched for the figure
      and the lesion-size analysis. Public data only."""
import os, urllib.request
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; S3 = "https://camelyon-dataset.s3.amazonaws.com/"
M = A + "/data/c16_meta"; [os.makedirs(M + d, exist_ok=True) for d in ("/annotations", "/evaluation")]
t = pd.read_csv(A + "/manifests/tcga_lung.csv")
t["fold"] = -9
for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=20261019).split(t, t.label)): t.loc[te, "fold"] = f
t = pd.DataFrame({"ds": "tcga", "slide": t.slide, "url": "https://api.gdc.cancer.gov/data/" + t.file_id, "size": t["size"], "label": t.label,
                  "split": "cv", "fold": t.fold, "patient": t.patient, "fname": t.file_name})
b = pd.read_csv(A + "/manifests/camelyon16_bucket.csv")
for k in b.key[b.key.str.contains("/annotations/|/evaluation/reference.csv|README.md|license.txt")]:
    dst = M + "/" + k.split("CAMELYON16/", 1)[1]
    if not os.path.exists(dst): urllib.request.urlretrieve(S3 + k, dst)
ref = pd.read_csv(M + "/evaluation/reference.csv"); ref["slide"] = ref.image.str.replace(".tif", "", regex=False)   # image,type,class(macro/micro/negative),center
c = pd.read_csv(A + "/manifests/camelyon16.csv")
lab = {r.slide: int(r.type == "tumor") for r in ref.itertuples()}; meta = ref.set_index("slide")
c["label"] = [1 if s.startswith("tumor") else 0 if s.startswith("normal") else lab.get(s, -1) for s in c.slide]
c["split"] = np.where(c.slide.str.startswith("test"), "test", "train")
c = pd.DataFrame({"ds": "c16", "slide": c.slide, "url": S3 + c.key, "size": c["size"], "label": c.label, "split": c.split, "fold": 0, "patient": c.slide, "fname": c.slide + ".tif",
                  "lesion": [meta["class"].get(x, "") for x in c.slide], "center": [meta["center"].get(x, -1) for x in c.slide]})
assert (c.label >= 0).all(), c[c.label < 0]
s = pd.concat([t, c], ignore_index=True); s.to_csv(A + "/manifests/slides.csv", index=False)
print(s.groupby(["ds", "split", "label"]).size().to_string()); print("annotations:", len(os.listdir(M + "/annotations")))
