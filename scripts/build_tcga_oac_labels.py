"""Build the Ch2 TCGA-OAC label table: survival (TCGA-CDR), TP53 (cBioPortal),
ploidy/WGD/purity (PanCanAtlas ABSOLUTE). Output: data/tcga_oac_labels.csv."""
import json, os, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
man = pd.read_csv(os.path.join(ROOT, "manifests", "TCGA_multimodal_manifest.csv"))
oac = man[(man["project"] == "TCGA-ESCA")
          & (man["disease_type"] == "Adenomas and Adenocarcinomas")].copy()
print("OAC cases:", len(oac))

# --- survival + stage from TCGA-CDR (canonical endpoints) ---
cdr = pd.read_excel(os.path.join(ROOT, "data", "TCGA-CDR.xlsx"), sheet_name="TCGA-CDR")
cdr = cdr[cdr["type"] == "ESCA"][["bcr_patient_barcode", "OS", "OS.time",
                                  "ajcc_pathologic_tumor_stage", "age_at_initial_pathologic_diagnosis"]]
cdr.columns = ["barcode", "os_event", "os_days", "ajcc_stage", "age"]
oac = oac.merge(cdr, on="barcode", how="left")

# --- TP53 somatic mutation status from cBioPortal ---
req = urllib.request.Request(
    "https://www.cbioportal.org/api/molecular-profiles/esca_tcga_pan_can_atlas_2018_mutations/mutations/fetch?projection=SUMMARY",
    data=json.dumps({"sampleListId": "esca_tcga_pan_can_atlas_2018_all",
                     "entrezGeneIds": [7157]}).encode(),
    headers={"Content-Type": "application/json"})
muts = json.loads(urllib.request.urlopen(req, timeout=120).read())
tp53_patients = {m["patientId"] for m in muts}
oac["tp53_mut"] = oac["barcode"].isin(tp53_patients).astype(int)

# --- ploidy / WGD / purity from ABSOLUTE ---
ab = pd.read_csv(os.path.join(ROOT, "data", "absolute_ploidy.txt"), sep="\t")
ab["barcode"] = ab["array"].str.extract(r"(TCGA-\w{2}-\w{4})")[0]
ab = ab[ab["array"].str.endswith("-01")].drop_duplicates("barcode")
oac = oac.merge(ab[["barcode", "purity", "ploidy", "Genome doublings"]], on="barcode", how="left")
oac = oac.rename(columns={"Genome doublings": "genome_doublings"})
oac["wgd"] = (oac["genome_doublings"] >= 1).astype("Int64").where(oac["genome_doublings"].notna())

cols = ["barcode", "rna_seq", "wxs_wgs", "slide_image", "os_event", "os_days",
        "ajcc_stage", "age", "tp53_mut", "purity", "ploidy", "genome_doublings", "wgd"]
out = oac[cols]
out.to_csv(os.path.join(ROOT, "data", "tcga_oac_labels.csv"), index=False)
print(out.notna().sum())
print("TP53 mutant:", int(out["tp53_mut"].sum()), "| WGD+:", int((out["wgd"] == 1).sum()),
      "| OS events:", int(out["os_event"].sum()))
print("complete rows (survival+tp53+wgd):",
      int((out["os_days"].notna() & out["wgd"].notna()).sum()))
