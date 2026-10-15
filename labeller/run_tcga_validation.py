"""Validate pathladder on TCGA ESCA+STAD reports against structured clinical truth.

Truth sources: cBioPortal pan-can study clinical attributes (GRADE) and the
manifest's disease_type (adenocarcinoma vs squamous from pathology coding).
Outputs labeller/tcga_validation_results.json + per-case CSV.
"""
import json, os, sys, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pathladder import label_frame, TCGA_GI

REPORTS = os.path.join(HERE, "..", "data", "TCGA_Reports.csv.zip")
MANIFEST = os.path.join(HERE, "..", "manifests", "TCGA_multimodal_manifest.csv")

import zipfile
with zipfile.ZipFile(REPORTS) as z:
    member = next(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
    rep = pd.read_csv(z.open(member))
idcol = next(c for c in rep.columns if "patient" in c.lower() or "filename" in c.lower())
rep["barcode"] = rep[idcol].astype(str).str.extract(r"(TCGA-\w{2}-\w{4})")[0]
man = pd.read_csv(MANIFEST)
gi = man[man["project"].isin(["TCGA-ESCA", "TCGA-STAD"])][["barcode", "project", "disease_type"]]
df = rep.merge(gi, on="barcode", how="inner").drop_duplicates("barcode")
textcol = next(c for c in df.columns if df[c].astype(str).str.len().mean() > 200)
print(f"GI reports matched: {len(df)} (text column: {textcol})")

labels = label_frame(df, textcol, TCGA_GI)
df = pd.concat([df[["barcode", "project", "disease_type"]].reset_index(drop=True),
                labels.reset_index(drop=True)], axis=1)

# --- truth 1: histologic type from GDC disease_type coding ---
truth_type = df["disease_type"].map({
    "Adenomas and Adenocarcinomas": "adenocarcinoma",
    "Squamous Cell Neoplasms": "squamous"})
m = truth_type.notna() & df["histologic_type"].notna()
type_acc = float((df.loc[m, "histologic_type"] == truth_type[m]).mean())
type_cov = float(df["histologic_type"].notna().mean())

# --- truth 2: grade from cBioPortal clinical data ---
def cbio_grades(study):
    url = f"https://www.cbioportal.org/api/studies/{study}/clinical-data?clinicalDataType=SAMPLE&projection=SUMMARY&pageSize=10000"
    d = json.loads(urllib.request.urlopen(url, timeout=120).read())
    return {r["patientId"]: r["value"] for r in d if r["clinicalAttributeId"] == "GRADE"}

grades = {}
for s in ["esca_tcga_pan_can_atlas_2018", "stad_tcga_pan_can_atlas_2018"]:
    grades.update(cbio_grades(s))
df["grade_truth"] = df["barcode"].map(grades).replace({"GX": None, "[Not Available]": None})
g = df["grade_truth"].notna() & df["grade"].notna()
grade_acc = float((df.loc[g, "grade"] == df.loc[g, "grade_truth"]).mean())
grade_cov = float(df["grade"].notna().mean())

res = {
    "n_reports": int(len(df)),
    "histologic_type": {"coverage": round(type_cov, 3), "accuracy_vs_gdc": round(type_acc, 3),
                        "n_compared": int(m.sum())},
    "grade": {"coverage": round(grade_cov, 3), "accuracy_vs_cbioportal": round(grade_acc, 3),
              "n_compared": int(g.sum())},
    "grade_confusion": pd.crosstab(df.loc[g, "grade"], df.loc[g, "grade_truth"]).to_dict(),
}
print(json.dumps(res, indent=2))
df.to_csv(os.path.join(HERE, "tcga_validation_per_case.csv"), index=False)
json.dump(res, open(os.path.join(HERE, "tcga_validation_results.json"), "w"), indent=2)
print("wrote tcga_validation_results.json + per-case CSV")
