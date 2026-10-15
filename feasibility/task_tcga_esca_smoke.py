"""Ch2 GATE support: enumerate TCGA-ESCA diagnostic slides for OAC cases via GDC API,
download ONE slide end-to-end, estimate the full transfer."""
import os, json, time, urllib.request, pandas as pd

OUT = os.environ.get("OUTDIR", ".")
MANIFEST = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/manifests/TCGA_multimodal_manifest.csv"

man = pd.read_csv(MANIFEST)
oac = man[(man["project"] == "TCGA-ESCA") & (man["disease_type"] == "Adenomas and Adenocarcinomas")
          & (man["rna_seq"] == 1) & (man["wxs_wgs"] == 1) & (man["slide_image"] == 1)]
barcodes = set(oac["barcode"])
print("target OAC complete-modality cases:", len(barcodes))

body = json.dumps({
    "filters": {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-ESCA"]}},
        {"op": "in", "content": {"field": "files.data_type", "value": ["Slide Image"]}},
    ]},
    "fields": "file_id,file_name,file_size,cases.submitter_id,experimental_strategy",
    "size": 3000, "format": "JSON",
}).encode()
req = urllib.request.Request("https://api.gdc.cancer.gov/files", data=body,
                             headers={"Content-Type": "application/json"})
hits = json.loads(urllib.request.urlopen(req, timeout=120).read())["data"]["hits"]
rows = []
for h in hits:
    bc = h["cases"][0]["submitter_id"]
    rows.append({"file_id": h["file_id"], "file_name": h["file_name"], "size": h["file_size"],
                 "barcode": bc, "strategy": h.get("experimental_strategy", ""),
                 "diagnostic": "-DX" in h["file_name"], "target_oac": bc in barcodes})
files = pd.DataFrame(rows)
files.to_csv(os.path.join(OUT, "esca_slide_files.csv"), index=False)
tgt = files[files["target_oac"] & files["diagnostic"]]
alt = files[files["target_oac"]]
summary = {
    "esca_slide_files_total": len(files),
    "oac_diagnostic_slides": len(tgt), "oac_diagnostic_cases": tgt["barcode"].nunique(),
    "oac_diagnostic_GB": round(tgt["size"].sum() / 1e9, 1),
    "oac_all_slides": len(alt), "oac_all_GB": round(alt["size"].sum() / 1e9, 1),
}
print(summary)

# smoke-download the smallest target slide
pick = (tgt if len(tgt) else alt).sort_values("size").iloc[0]
t0 = time.time()
data = urllib.request.urlopen(f"https://api.gdc.cancer.gov/data/{pick['file_id']}", timeout=1800).read()
dt = time.time() - t0
dest = os.path.join(OUT, pick["file_name"])
open(dest, "wb").write(data)
mbps = len(data) / 1e6 / dt
summary["smoke_download"] = {"file": pick["file_name"], "MB": round(len(data) / 1e6, 1),
                             "seconds": round(dt, 1), "MB_per_s": round(mbps, 1),
                             "est_full_hours": round(tgt["size"].sum() / 1e6 / mbps / 3600, 2)}
print(summary["smoke_download"])
json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
