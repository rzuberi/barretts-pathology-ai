"""Build (report x SWG Path ID) pairs for the specimen-scoped jury pass.

Each pair's text = a SPECIMEN OF INTEREST header (path id + parsed suffix) + the
full report. CaseName = "<path_id>@<pathology_text_id>" so votes join back to both.
Also carries the SWG coded grade for the downstream audit.
"""
import os, re
import pandas as pd

E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
SWG_XL = "/mnt/scratche/slow/fmlab/zuberi01/phd/Leanne_shared_docs/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3).xlsx"

rep = pd.read_parquet(os.path.join(E, "swg_matched_reports_v2.parquet"))
swg = pd.read_excel(SWG_XL, dtype=str)
swg["path_key"] = swg["Path ID"].str.strip()
grade_map = dict(zip(swg["path_key"], swg["Pathology"].str.strip().str.upper()))

rows = []
for _, r in rep.iterrows():
    rid = str(pd.Series([r["pathology_text_id"]]).astype("Int64").iloc[0])
    for pid in str(r["swg_path_ids"]).split(";"):
        pid = pid.strip()
        if not pid: continue
        m = re.search(r"_([A-Za-z]+\d*)\s*$", pid)
        suffix = m.group(1) if m else "(none)"
        header = (f"SPECIMEN OF INTEREST: laboratory sample id '{pid}', "
                  f"specimen/block designation '{suffix}'.\n\n")
        rows.append({"CaseName": f"{pid}@{rid}",
                     "path_id": pid, "report_id": rid, "suffix": suffix,
                     "swg_grade_code": grade_map.get(pid, ""),
                     "reporttext": header + str(r["reporttext"])})
df = pd.DataFrame(rows).drop_duplicates("CaseName")
out = os.path.join(E, "specimen_pairs.parquet")
df.to_parquet(out, index=False); os.chmod(out, 0o600)
print(f"pairs: {len(df)} across {df['report_id'].nunique()} reports; "
      f"with SWG code: {(df['swg_grade_code'] != '').sum()}")
