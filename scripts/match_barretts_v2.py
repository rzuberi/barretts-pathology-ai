"""Barretts DB -> SWG matching v2 (READ-ONLY: GET requests only).

Improvements over v1: match on the view's populated specimennumber column
(normalised (yy, num) keys) plus text-regex fallback incl. H-numbers; report
coverage by SWG Set/centre to quantify the Addenbrooke's-only ceiling.
"""
import json, os, re, urllib.request
import pandas as pd

API = "https://api.barrettsdatabase.org.uk"
TOKEN = open(os.path.expanduser("~/.barretts_token")).read().strip()
OUTD = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
SWG_XL = "/mnt/scratche/slow/fmlab/zuberi01/phd/Leanne_shared_docs/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3).xlsx"

def fetch(table, page=1000):
    rows, offset = [], 0
    while True:
        req = urllib.request.Request(f"{API}/{table}?limit={page}&offset={offset}",
                                     headers={"Authorization": f"Bearer {TOKEN}"})
        chunk = json.loads(urllib.request.urlopen(req, timeout=120).read())
        rows.extend(chunk)
        if len(chunk) < page: break
        offset += page
    print(f"{table}: {len(rows)} rows", flush=True)
    return pd.DataFrame(rows)

pt = fetch("view_patient_pathology_text_normalised")
# old-LIMS reports embed \r, which corrupts CSV round-trips: sanitize before saving
pt["reporttext"] = (pt["reporttext"].fillna("").astype(str)
                    .str.replace("\r\n", "\n").str.replace("\r", "\n"))
pt.to_csv(os.path.join(OUTD, "view_patient_pathology_full.csv"), index=False)
os.chmod(os.path.join(OUTD, "view_patient_pathology_full.csv"), 0o600)

PS = re.compile(r"PS\s?(\d{2})[.\-/ ]?(\d{3,6})(?!\d)", re.I)
HN = re.compile(r"(?<!\d)(\d{2})H(\d{5,10})(?!\d)", re.I)
def keys_from(s):
    s = str(s)
    out = {("PS", y, int(n)) for y, n in PS.findall(s)}
    out |= {("H", y, int(n)) for y, n in HN.findall(s)}
    return out

pt["keys"] = (pt["specimennumber"].fillna("") + " || " + pt["reporttext"].fillna("")).map(keys_from)

swg = pd.read_excel(SWG_XL, dtype=str)
swg["keys"] = swg["Path ID"].fillna("").map(keys_from)
key2paths = {}
for _, r in swg.iterrows():
    for k in r["keys"]:
        key2paths.setdefault(k, set()).add(r["Path ID"])

pt["swg_hits"] = pt["keys"].map(lambda ks: sorted({p for k in ks for p in key2paths.get(k, ())}))
matched = pt[pt["swg_hits"].str.len() > 0].copy()
matched["swg_path_ids"] = matched["swg_hits"].map(";".join)
out_df = matched.drop(columns=["keys", "swg_hits"])
out_df.to_csv(os.path.join(OUTD, "swg_matched_reports_v2.csv"), index=False)
out_df.to_parquet(os.path.join(OUTD, "swg_matched_reports_v2.parquet"), index=False)
os.chmod(os.path.join(OUTD, "swg_matched_reports_v2.csv"), 0o600); os.chmod(os.path.join(OUTD, "swg_matched_reports_v2.parquet"), 0o600)

hit_paths = {p for h in matched["swg_hits"] for p in h}
swg["matched"] = swg["Path ID"].isin(hit_paths)
by_set = swg.groupby("Set").agg(n=("Path ID", "size"), matched=("matched", "sum"))
rep = {"db_reports": int(len(pt)),
       "matched_report_rows": int(len(matched)),
       "swg_path_ids_matched": int(len(hit_paths)),
       "swg_patients_matched": int(swg.loc[swg["matched"], "PatientID"].nunique()),
       "swg_patients_total": int(swg["PatientID"].nunique()),
       "coverage_by_set": {str(k): {"samples": int(v["n"]), "matched": int(v["matched"])}
                           for k, v in by_set.iterrows()}}
json.dump(rep, open(os.path.join(OUTD, "match_report_v2.json"), "w"), indent=2)
print(json.dumps(rep, indent=2))
