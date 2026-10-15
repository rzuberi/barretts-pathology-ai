"""Export pathology free text from the Barretts database (Fitzgerald group) and
match to the SWG cohort. CLUSTER-SIDE; token read from ~/.barretts_token.

Output (chmod-700 dir, never in git): /mnt/scratche/slow/fmlab/zuberi01/barretts_db_export/
  - pathology_text_normalised_full.csv   (all rows)
  - swg_matched_reports.csv              (rows matched to SWG Path IDs)
  - match_report.json
"""
import json, os, re, urllib.request
import pandas as pd

API = "https://api.barrettsdatabase.org.uk"
TOKEN = open(os.path.expanduser("~/.barretts_token")).read().strip()
OUTD = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
SWG_XL = "/mnt/scratche/slow/fmlab/zuberi01/phd/Leanne_shared_docs/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3).xlsx"
os.makedirs(OUTD, exist_ok=True)
os.chmod(OUTD, 0o700)

def fetch(table, params="", page=1000):
    rows, offset = [], 0
    while True:
        url = f"{API}/{table}?limit={page}&offset={offset}" + (f"&{params}" if params else "")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
        chunk = json.loads(urllib.request.urlopen(req, timeout=120).read())
        rows.extend(chunk)
        print(f"{table}: {len(rows)} rows", flush=True)
        if len(chunk) < page: break
        offset += page
    return pd.DataFrame(rows)

pt = fetch("pathology_text_normalised")
pt.to_csv(os.path.join(OUTD, "pathology_text_normalised_full.csv"), index=False)

# PS accession numbers from report text: PSyy.nnnn / PSyy.nnnnn / PSyy-nnnnn styles
def ps_numbers(text):
    t = str(text)
    hits = re.findall(r"\bPS\s?(\d{2})[.\-/](\d{3,6})\b", t)
    return {f"PS{y}.{n}" for y, n in hits} | {f"PS{y}-{int(n):05d}" for y, n in hits}
pt["ps_ids"] = pt["reporttext"].map(ps_numbers)

# SWG Path IDs
swg = pd.read_excel(SWG_XL, dtype=str)
raw_ids = swg["Path ID"].dropna().unique()
def norm_ps(s):
    m = re.search(r"PS\s?(\d{2})[.\-/ ]?(\d{3,6})", str(s).upper())
    return {f"PS{m.group(1)}.{m.group(2)}", f"PS{m.group(1)}-{int(m.group(2)):05d}"} if m else set()
swg_ids = set()
id_map = {}
for r in raw_ids:
    ns = norm_ps(r)
    swg_ids |= ns
    for n in ns: id_map[n] = r
print("SWG Path IDs:", len(raw_ids), "-> normalised keys:", len(swg_ids))
print("sample raw:", list(raw_ids)[:5])

pt["swg_hits"] = pt["ps_ids"].map(lambda s: sorted(s & swg_ids))
matched = pt[pt["swg_hits"].str.len() > 0].copy()
matched["swg_path_id_raw"] = matched["swg_hits"].map(lambda h: ";".join(sorted({id_map[x] for x in h})))
matched.drop(columns=["ps_ids", "swg_hits"]).to_csv(os.path.join(OUTD, "swg_matched_reports.csv"), index=False)

matched_raw_ids = {id_map[x] for h in matched["swg_hits"] for x in h}
rep = {"db_reports": int(len(pt)),
       "db_reports_with_ps_id": int((pt["ps_ids"].str.len() > 0).sum()),
       "swg_path_ids": int(len(raw_ids)),
       "swg_path_ids_matched": int(len(matched_raw_ids)),
       "matched_report_rows": int(len(matched)),
       "swg_patients_matched": int(swg[swg["Path ID"].isin(matched_raw_ids)]["PatientID"].nunique())}
json.dump(rep, open(os.path.join(OUTD, "match_report.json"), "w"), indent=2)
print(json.dumps(rep, indent=2))
for f in os.listdir(OUTD): os.chmod(os.path.join(OUTD, f), 0o600)
