"""EXECUTION_PLAN 2.8: build download + extraction manifests for the TCGA-STAD/GEJ
extension pool (R.2 approved 2026-08-19). Diagnostic slides for complete-modality
STAD adenocarcinoma cases."""
import json, os, urllib.request
import pandas as pd

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
SLIDES_DIR = "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/slides"

man = pd.read_csv(os.path.join(T, "manifests", "TCGA_multimodal_manifest.csv"))
pool = man[(man["project"] == "TCGA-STAD")
           & (man["disease_type"] == "Adenomas and Adenocarcinomas")
           & (man["rna_seq"] == 1) & (man["wxs_wgs"] == 1) & (man["slide_image"] == 1)]
barcodes = set(pool["barcode"])
print("STAD complete-modality adeno cases:", len(barcodes))

body = json.dumps({
    "filters": {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-STAD"]}},
        {"op": "in", "content": {"field": "files.data_type", "value": ["Slide Image"]}},
    ]},
    "fields": "file_id,file_name,file_size,cases.submitter_id,experimental_strategy",
    "size": 5000, "format": "JSON"}).encode()
req = urllib.request.Request("https://api.gdc.cancer.gov/files", data=body,
                             headers={"Content-Type": "application/json"})
hits = json.loads(urllib.request.urlopen(req, timeout=180).read())["data"]["hits"]
rows = [{"file_id": h["file_id"], "file_name": h["file_name"], "size": h["file_size"],
         "barcode": h["cases"][0]["submitter_id"],
         "strategy": h.get("experimental_strategy", ""),
         "diagnostic": "-DX" in h["file_name"],
         "target_oac": h["cases"][0]["submitter_id"] in barcodes}
        for h in hits]
df = pd.DataFrame(rows)
tgt = df[df["target_oac"] & df["diagnostic"]].drop_duplicates("file_id")
print(f"diagnostic slides for pool: {len(tgt)} across {tgt['barcode'].nunique()} cases, "
      f"{tgt['size'].sum()/1e9:.0f} GB")
tgt.to_csv(os.path.join(T, "campaigns", "stad_dl_manifest.csv"), index=False)
with open(os.path.join(T, "campaigns", "stad_extract_manifest.txt"), "w") as f:
    for name in sorted(tgt["file_name"]):
        f.write(os.path.join(SLIDES_DIR, name) + "\n")
print("wrote stad_dl_manifest.csv + stad_extract_manifest.txt")
