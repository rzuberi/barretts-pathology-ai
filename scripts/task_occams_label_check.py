"""Deep-dive C (extends 1.1/2.15): test the 'OCCAMS null = label quality' hypothesis.

Cross-check survival fields between our OCCAMS master export (weeks/days columns
used in v3) and the independent Barretts-database OCCAMS masterlist. Reports
coverage, agreement of death indicators, and time-field correlation. Discrepancy
=> justify v4 with corrected labels; agreement => the null is likely biological.
"""
import json, os, re
import numpy as np, pandas as pd

E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
MASTER = "/home/zuberi01/occams_work/occams_master_20260511.csv"
OUT = os.environ.get("OUTDIR", ".")

def norm_id(s):
    m = re.search(r"(?:OCCAMS|OC)[-_/ ]?([A-Z]{2})[-_/ ]?0*([0-9]+)", str(s).upper())
    return f"{m.group(1)}{int(m.group(2)):04d}" if m else None

mast = pd.read_csv(MASTER, dtype=str, low_memory=False)
mast["cid"] = mast["occams_id"].map(norm_id)
mast["dsd"] = pd.to_numeric(mast["deceased_survival_days"], errors="coerce")
mast["lkd"] = pd.to_numeric(mast["last_known_survival_days"], errors="coerce")
mast = mast.dropna(subset=["cid"]).drop_duplicates("cid").set_index("cid")

db = None
for cand in ("view_occams_masterlist", "occams_masterlist_modified_20200813"):
    p = os.path.join(E, cand + ".parquet")
    if os.path.exists(p):
        db = pd.read_parquet(p); src = cand; break
print("db source:", src, "| cols:", list(db.columns)[:25])
idc = next((c for c in db.columns if "occams" in c.lower() and "id" in c.lower()), db.columns[0])
db["cid"] = db[idc].map(norm_id)
db = db.dropna(subset=["cid"]).drop_duplicates("cid").set_index("cid")

both = mast.index.intersection(db.index)
res = {"master_n": int(len(mast)), "db_n": int(len(db)), "overlap": int(len(both)),
       "db_id_col": idc}

# find date/status-ish columns in db
surv_cols = [c for c in db.columns if any(k in c.lower() for k in
             ("death", "died", "deceas", "surviv", "last_seen", "follow", "status", "date_of"))]
res["db_survival_cols"] = surv_cols[:12]
for c in surv_cols[:6]:
    v = db.loc[both, c].astype(str)
    res[f"db.{c}.nonnull"] = int((v.notna() & (v != "None") & (v != "nan") & (v != "")).sum())

# if a death-date column exists, compare event indicator with master dsd
death_col = next((c for c in surv_cols if "death" in c.lower() or "deceas" in c.lower()), None)
if death_col is not None:
    db_dead = db.loc[both, death_col].astype(str).replace({"None": "", "nan": ""}).str.len() > 4
    m_dead = mast.loc[both, "dsd"].notna()
    agree = float((db_dead.values == m_dead.values).mean())
    res["death_indicator_agreement"] = round(agree, 4)
    res["db_dead"] = int(db_dead.sum()); res["master_dead"] = int(m_dead.sum())
    disagree = both[(db_dead.values != m_dead.values)]
    res["n_disagreements"] = int(len(disagree))
    pd.DataFrame({"cid": disagree}).to_csv(os.path.join(OUT, "occams_label_disagreements.csv"), index=False)

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
