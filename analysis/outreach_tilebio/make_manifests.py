"""Build download manifests (public data only): TCGA-LUAD/LUSC diagnostic slides (one per patient, GDC open access)
and CAMELYON16 (AWS open-data bucket). Writes manifests/{tcga_lung,camelyon16}.csv."""
import json, os, re, urllib.request, urllib.parse, xml.etree.ElementTree as ET
import pandas as pd
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; os.makedirs(A + "/manifests", exist_ok=True)
# ---- TCGA lung ----
filt = {"op": "and", "content": [
    {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LUAD", "TCGA-LUSC"]}},
    {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
    {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
    {"op": "in", "content": {"field": "access", "value": ["open"]}}]}
q = {"filters": json.dumps(filt), "fields": "file_id,file_name,file_size,cases.submitter_id,cases.project.project_id", "format": "JSON", "size": "5000"}
r = json.load(urllib.request.urlopen("https://api.gdc.cancer.gov/files?" + urllib.parse.urlencode(q), timeout=120))
rows = [{"file_id": h["file_id"], "file_name": h["file_name"], "size": h["file_size"], "patient": h["cases"][0]["submitter_id"],
         "project": h["cases"][0]["project"]["project_id"]} for h in r["data"]["hits"]]
d = pd.DataFrame(rows); print("TCGA diagnostic slides:", len(d), d.project.value_counts().to_dict())
d = d[d.file_name.str.contains(r"-01Z-00-DX", regex=True)]   # primary tumour diagnostic slides only
both = d.groupby("patient").project.nunique(); d = d[~d.patient.isin(both[both > 1].index)]   # drop any patient listed in both projects
d = d.sort_values(["patient", "file_name"]).groupby("patient").head(1).reset_index(drop=True)   # one slide per patient (lowest DX number)
d["label"] = (d.project == "TCGA-LUSC").astype(int); d["slide"] = d.file_name.str.replace(".svs", "", regex=False)
d.to_csv(A + "/manifests/tcga_lung.csv", index=False)
print("TCGA one-per-patient:", len(d), d.project.value_counts().to_dict(), f"{d['size'].sum()/1e12:.2f} TB")
# ---- CAMELYON16 ----
keys = []; tok = None
while True:
    p = {"list-type": "2", "prefix": "CAMELYON16/"}
    if tok: p["continuation-token"] = tok
    x = ET.fromstring(urllib.request.urlopen("https://camelyon-dataset.s3.amazonaws.com/?" + urllib.parse.urlencode(p), timeout=120).read())
    ns = {"s": x.tag.split("}")[0][1:]}
    for c in x.findall("s:Contents", ns): keys.append((c.find("s:Key", ns).text, int(c.find("s:Size", ns).text)))
    t = x.find("s:NextContinuationToken", ns)
    if t is None: break
    tok = t.text
k = pd.DataFrame(keys, columns=["key", "size"]); k.to_csv(A + "/manifests/camelyon16_bucket.csv", index=False)
print("CAMELYON16 bucket objects:", len(k)); print(k[~k.key.str.endswith(".tif")].to_string())
tif = k[k.key.str.endswith(".tif") & k.key.str.contains("/images/")].copy(); tif["slide"] = tif.key.str.extract(r"([^/]+)\.tif$")[0]
print("tif slides:", len(tif), f"{tif['size'].sum()/1e12:.2f} TB"); print(tif.slide.str.extract(r"^([a-z]+)_")[0].value_counts().to_dict())
tif.to_csv(A + "/manifests/camelyon16.csv", index=False)
