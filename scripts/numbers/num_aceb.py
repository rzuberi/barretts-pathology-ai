"""NUMBERS A/25 for ACE-B: what exists = metadata matching from May 2026 (no slides/features on cluster)."""
import glob, json, os
import pandas as pd
D = "/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta"; OUT = os.environ.get("OUTDIR", ".")
res = {"dir": D, "files": sorted(os.path.basename(f) for f in glob.glob(D + "/*")), "reports_md": {}}
for f in glob.glob(D + "/*.md"): res["reports_md"][os.path.basename(f)] = open(f, errors="ignore").read()[:3000]
for f in glob.glob(D + "/*.csv"):
    try:
        d = pd.read_csv(f, dtype=str, low_memory=False); res.setdefault("csv_shapes", {})[os.path.basename(f)] = {"rows": len(d), "cols": list(d.columns)[:15]}
    except Exception as e: res.setdefault("csv_shapes", {})[os.path.basename(f)] = str(e)
for f in glob.glob(D + "/aceb_endoscopy_samples_per_patient_*.csv"):
    d = pd.read_csv(f, dtype=str); c = next((x for x in d.columns if "count" in x.lower()), d.columns[-1]); v = pd.to_numeric(d[c], errors="coerce")
    res["endoscopy_per_patient"] = {"patients": len(d), "records": int(v.sum()), "median": float(v.median()), "min": float(v.min()), "max": float(v.max())}
for f in glob.glob(D + "/aceb_official_case_presence_in_barretts_db_*.csv"):
    d = pd.read_csv(f, dtype=str); res["official_case_presence"] = {"rows": len(d), "cols": list(d.columns), "value_counts": {c: d[c].value_counts().head(6).to_dict() for c in d.columns if d[c].nunique() <= 6}}
for f in glob.glob(D + "/ACE-B cases for Rehan.csv"):
    d = pd.read_csv(f, dtype=str); res["cases_for_rehan"] = {"rows": len(d), "cols": list(d.columns)}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=str); print(json.dumps({k: v for k, v in res.items() if k != "reports_md"}, indent=1, default=str)[:3000])
