"""ABD QC: parse quality of the Feb-2026 pathology scrape (ids/dates only; counts)."""
import re, pandas as pd
FEB = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/data/barretts_scrape_20260220_172725"
DB = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
for name, p in [("feb", f"{FEB}/view_patient_pathology_text_normalised.csv"), ("aug", f"{DB}/pathology_text_normalised_full.csv")]:
    d = pd.read_csv(p, usecols=["pathology_text_id", "participant_id", "receiveddatetime"], dtype=str)
    num_id = d.pathology_text_id.fillna("").str.fullmatch(r"\d+(\.0)?"); num_p = d.participant_id.fillna("").str.fullmatch(r"\d+(\.0)?")
    dt = pd.to_datetime(d.receiveddatetime, errors="coerce").notna()
    print(name, "rows", len(d), "numeric report id", int(num_id.sum()), "numeric pid", int(num_p.sum()), "valid date", int(dt.sum()),
          "distinct report ids", d.pathology_text_id[num_id].nunique(), "distinct numeric pids", d.participant_id[num_p].str.replace(r"\.0$", "", regex=True).nunique(),
          "null pid", int(d.participant_id.isna().sum()), "dup report ids", int(d.pathology_text_id[num_id].duplicated().sum()))
