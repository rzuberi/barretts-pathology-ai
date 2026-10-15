"""Ch4 GATE first-pass: download TCGA-Reports, join to TCGA multimodal manifest,
count usable oesophageal/GEJ cases with report + WSI."""
import os, json, io, urllib.request, pandas as pd

OUT = os.environ.get("OUTDIR", ".")
MANIFEST = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/manifests/TCGA_multimodal_manifest.csv"

def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()

# Mendeley public API: list files for dataset hyg5xkznpx v1
listing = json.loads(fetch("https://data.mendeley.com/public-api/datasets/hyg5xkznpx/files?folder_id=root&version=1"))
print("files in Mendeley dataset:")
target = None
for f in listing:
    name = f.get("filename", "")
    print(" -", name, f.get("size"))
    if name.endswith((".csv", ".csv.zip")):
        target = f
assert target, "no CSV found in dataset listing"
url = target["content_details"]["download_url"]
raw = fetch(url, timeout=600)
local = os.path.join(OUT, target["filename"])
open(local, "wb").write(raw)
print("downloaded", local, len(raw), "bytes")

if local.endswith(".zip"):
    import zipfile
    with zipfile.ZipFile(local) as z:
        member = next(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
        local = z.extract(member, OUT)
rep = pd.read_csv(local)
print("report rows:", len(rep), "| columns:", list(rep.columns)[:10])
# patient_filename like TCGA-XX-XXXX.<suffix>; derive 12-char barcode
idcol = next(c for c in rep.columns if "patient" in c.lower() or "filename" in c.lower() or "id" in c.lower())
rep["barcode"] = rep[idcol].astype(str).str.extract(r"(TCGA-\w{2}-\w{4})")[0]
rep = rep.dropna(subset=["barcode"]).drop_duplicates("barcode")
print("unique report barcodes:", len(rep))

man = pd.read_csv(MANIFEST)
merged = man.merge(rep[["barcode"]], on="barcode", how="inner")
summary = {}
for label, projects in [("ESCA", ["TCGA-ESCA"]), ("STAD", ["TCGA-STAD"]), ("ESCA+STAD", ["TCGA-ESCA", "TCGA-STAD"])]:
    sub = merged[merged["project"].isin(projects)]
    withslide = sub[sub["slide_image"] == 1]
    oac = withslide[withslide["disease_type"] == "Adenomas and Adenocarcinomas"]
    summary[label] = {"report+case": len(sub), "report+slide": len(withslide), "report+slide adenocarcinoma": len(oac)}
    print(label, summary[label])
summary["pan_cancer_report+slide"] = int((merged["slide_image"] == 1).sum())
print("pan-cancer report+slide:", summary["pan_cancer_report+slide"])
json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
