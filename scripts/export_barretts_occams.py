"""EXECUTION_PLAN 2.15: OCCAMS-side pull from the Barretts DB (READ-ONLY GETs)
plus snapshot of thesis-relevant auxiliary tables before token expiry.

Everything lands in the secured export dir (chmod 600), never in git.
"""
import json, os, urllib.request
import pandas as pd

API = "https://api.barrettsdatabase.org.uk"
TOKEN = open(os.path.expanduser("~/.barretts_token")).read().strip()
OUTD = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"

def get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())

def fetch(table, page=1000, max_rows=200000):
    rows, offset = [], 0
    while offset < max_rows:
        chunk = get(f"{API}/{table}?limit={page}&offset={offset}")
        rows.extend(chunk)
        if len(chunk) < page: break
        offset += page
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace("\r\n", "\n").str.replace("\r", "\n")
    out = os.path.join(OUTD, f"{table}.parquet")
    df.to_parquet(out, index=False); os.chmod(out, 0o600)
    print(f"{table}: {len(df)} rows -> parquet", flush=True)
    return df

# discover any OCCAMS-named endpoints from the OpenAPI root
paths = sorted(get(API + "/").get("paths", {}).keys())
occ = [p.strip("/") for p in paths if "occams" in p.lower() or "masterpath" in p.lower()]
print("occams/masterpath endpoints:", occ)

report = {}
for t in occ:
    if t.startswith("rpc/"): continue
    try:
        report[t] = int(len(fetch(t)))
    except Exception as e:
        report[t] = f"FAILED: {str(e)[:80]}"

# thesis-relevant auxiliary tables (small, structured)
AUX = ["endoscopy", "mat_endoscopy_full", "dysplasiagradehistory", "diagnosishistory",
       "initial_history", "studypatientbarrettssurv", "studypatientspongestudy",
       "hgd_pathology_table", "bo_details", "heightweighthistory", "symptomshistory"]
for t in AUX:
    try:
        report[t] = int(len(fetch(t)))
    except Exception as e:
        report[t] = f"FAILED: {str(e)[:80]}"

json.dump(report, open(os.path.join(OUTD, "extras_report.json"), "w"), indent=2)
print(json.dumps(report, indent=2))
