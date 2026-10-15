"""ERIN all-slides campaign, step 0: one row per WSI file.

Joins the WSIExport manifests (stain, tissue, case) with Shiv Sakthivel's
slide<->section table and the report table, records file size/path, and writes:
  erin_slides_all.csv        every slide (12,343) with metadata + flags
  erin_manifest_all_he.txt   extraction manifest = H&E slides not yet extracted
Also hard-links the existing per-case features (named by case folder) into the
new per-file feature dir so those slides are not re-extracted.
"""
import glob, os, re, json
import pandas as pd, h5py

E = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
C = T + "/campaigns/allslides"
OLD = E + "/features/20x_224px/features_uni_v2"
NEW = E + "/features/20x_224px/features_uni_v2_all"
os.makedirs(C, exist_ok=True); os.makedirs(NEW, exist_ok=True)
UUID = r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"

# --- files on disk ---
rows = []
for p in glob.glob(E + "/slides/*/*.tif*"):
    rows.append({"path": p, "case_folder": os.path.basename(os.path.dirname(p)).lower(),
                 "file_uuid": os.path.splitext(os.path.basename(p))[0].lower(),
                 "size_mb": round(os.path.getsize(p) / 1e6, 1)})
disk = pd.DataFrame(rows)
print("tiffs on disk", len(disk))

# --- WSIExport manifests ---
w = pd.concat([pd.read_csv(f, dtype=str) for f in glob.glob(E + "/slides/WSIExport_Result_*.csv")
               + glob.glob(E + "/raw/WSIExport_Result_*.csv")], ignore_index=True).fillna("")
w = w.drop_duplicates("WSI File Name")
w["file_uuid"] = w["WSI File Name"].str.replace(r"\.\w+$", "", regex=True).str.lower()
w = w.rename(columns={"Case Identifier": "CaseName", "Stain Type": "stain_type", "Stain Name": "stain_name",
                      "Tissue Name": "tissue_manifest", "Slide Identifier": "slide_identifier",
                      "Block Identifier": "block_id", "Acquisition DateTime": "acquired"})
w["CaseName"] = w["CaseName"].str.strip()
w = w[["file_uuid", "CaseName", "stain_type", "stain_name", "tissue_manifest", "slide_identifier", "block_id", "acquired"]]

# --- Shiv section table ---
s = pd.read_csv("/mnt/scratche/slow/fmlab/sakthi01/erin/data/matched_image_pathology.csv", dtype=str).fillna("")
s["file_uuid"] = s["Filepath"].str.extract(UUID + r"(?:\.\w+)?$", expand=False).str.lower()
s = s.drop_duplicates("file_uuid")[["file_uuid", "Section", "TissueName", "Stain"]].rename(
    columns={"Section": "section", "TissueName": "tissue_shiv", "Stain": "stain_shiv"})

# --- reports (patient, date) ---
rep = pd.read_csv(E + "/data/PathologyReport_AnonIds.csv", dtype=str, low_memory=False).fillna("")
rep = rep.drop_duplicates("CaseName")[["CaseName", "anon_id", "CollectedOrOrdered", "SpecimenProtocol"]]

df = disk.merge(w, on="file_uuid", how="left").merge(s, on="file_uuid", how="left").merge(rep, on="CaseName", how="left")
df["tissue_meta"] = df["tissue_manifest"].where(df["tissue_manifest"].fillna("") != "", df["tissue_shiv"]).fillna("")
df["organ_meta"] = df["tissue_meta"].str.split("|").str[0].str.strip().str.upper()
df["organ_meta"] = df["organ_meta"].replace({"LYMPH NODES": "LYMPH NODE", "RECTAL BIOPSY": "COLORECTAL", "COLON": "COLORECTAL"})
df["is_he"] = df["stain_type"].fillna("").str.contains("H&E")
df["section"] = df["section"].fillna("").str.upper().str.strip()

# --- existing features: map case folder -> file uuid of the extracted slide ---
old_map = {}
for h in glob.glob(OLD + "/*.h5"):
    try:
        with h5py.File(h) as f: sp = str(f.attrs.get("slide_path", ""))
    except Exception: continue
    m = re.search(UUID + r"(?:\.\w+)?$", sp)
    if m: old_map[m.group(1).lower()] = h
df["old_h5"] = df["file_uuid"].map(old_map)
linked = 0
for _, r in df[df["old_h5"].notna()].iterrows():
    dst = os.path.join(NEW, r["file_uuid"] + ".h5")
    if not os.path.exists(dst):
        try: os.link(r["old_h5"], dst); linked += 1
        except OSError: import shutil; shutil.copy2(r["old_h5"], dst); linked += 1
df["h5_new"] = df["file_uuid"].map(lambda u: os.path.join(NEW, u + ".h5"))
df["has_features"] = df["h5_new"].map(os.path.exists)
print("existing features linked into new dir:", linked, "| slides with features now:", int(df.has_features.sum()))

df = df.sort_values(["CaseName", "section", "file_uuid"])
df.to_csv(C + "/erin_slides_all.csv", index=False)

todo = df[df.is_he & ~df.has_features]
open(C + "/erin_manifest_all_he.txt", "w").write("\n".join(todo["path"]) + "\n")
summary = {"slides_on_disk": len(df), "in_manifest_tables": int(df.stain_type.notna().sum()),
           "he": int(df.is_he.sum()), "special_stain": int(df.stain_type.fillna("").str.contains("Special").sum()),
           "ihc": int(df.stain_type.fillna("").str.contains("IHC").sum()), "stain_blank": int((df.stain_type.fillna("") == "").sum()),
           "organ_meta_dist": df.organ_meta.replace("", "UNKNOWN").value_counts().to_dict(),
           "he_organ_meta_dist": df[df.is_he].organ_meta.replace("", "UNKNOWN").value_counts().to_dict(),
           "with_section": int((df.section != "").sum()), "already_extracted_he": int((df.is_he & df.has_features).sum()),
           "to_extract_he": len(todo), "to_extract_gb": round(todo.size_mb.sum() / 1e3, 1)}
json.dump(summary, open(C + "/slides_table_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
