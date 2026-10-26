"""ABD step 1: column names only (no rows) of the ACE-B sheet, ACE-B May-2026 ID files,
the Barrett's DB export tables and the cross-cohort lists; plus a search for other DB extracts.
Header tokens containing 3+ consecutive digits are masked (a header-less file would otherwise
show a data row). Run on epyc via srun. Prints nothing but names, row counts and file dates."""
import os, re, glob, csv, datetime, sqlite3
import pyarrow.parquet as pq

Z = "/mnt/scratche/slow/fmlab/zuberi01"
META = f"{Z}/phd/aceb_meta"
DB = f"{Z}/barretts_db_export"
TH = f"{Z}/phd/thesis"
KM = f"{TH}/feasibility/paper_plan/killcoyne_mm"

def mask(tok):
    return "<masked>" if re.search(r"\d{3,}|PS\d|ACE", tok, re.I) else tok

def csv_header(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        hdr = next(csv.reader(f))
    return [mask(t) for t in hdr]

def mtime(p):
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d")

print("=== ACE-B meta CSV headers ===")
for p in sorted(glob.glob(f"{META}/*.csv")):
    try: print(os.path.basename(p), mtime(p), "|", csv_header(p))
    except Exception as e: print(os.path.basename(p), "ERR", e)
print("=== ACE-B meta xlsx sheets/headers ===")
import openpyxl
for p in sorted(glob.glob(f"{META}/*.xlsx")):
    wb = openpyxl.load_workbook(p, read_only=True)
    for ws in wb.worksheets:
        hdr = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
        print(os.path.basename(p), ws.title, ws.max_row, "|", [mask(str(h)) for h in hdr])

print("=== DB export parquet schemas ===")
for p in sorted(glob.glob(f"{DB}/*.parquet")):
    pf = pq.ParquetFile(p)
    print(os.path.basename(p), mtime(p), pf.metadata.num_rows, "|", pf.schema_arrow.names)
for p in sorted(glob.glob(f"{DB}/*.csv")):
    print(os.path.basename(p), mtime(p), "|", csv_header(p))
for sub in ("jury", "jury_specimen"):
    print(sub, sorted(os.listdir(f"{DB}/{sub}"))[:20])
SUM = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort/barretts_database_230809.csv"
print("summary extract:", mtime(SUM), csv_header(SUM))

print("=== cross-cohort lists ===")
for p in [f"{TH}/feasibility/closeout/erin_swg_pairs.csv"] + sorted(glob.glob(f"{KM}/two_papers_ERL/*.csv")):
    print(os.path.relpath(p, TH), mtime(p), "|", csv_header(p))

print("=== search for other Barrett's DB extracts ===")
pat = re.compile(r"barrett.*(database|_db|db_)|participant.*(table|scrape)|pathology_text|view_patient_pathology|endoscopy\.(parquet|csv)", re.I)
skip = re.compile(r"/(\.git|conda|miniforge|site-packages|envs|node_modules|aceb_seq|aceb_dna_seq|data-exchange)(/|$)")
for root in [Z, "/mnt/scratche/fast/fmlab/zuberi01", "/mnt/scratche/slow/fmlab/datasets", "/mnt/scratche/fast/fmlab/datasets", os.path.expanduser("~")]:
    base_depth = root.rstrip("/").count("/")
    for r, ds, fs in os.walk(root):
        if skip.search(r) or r.count("/") - base_depth >= 6:
            ds[:] = []; continue
        ds[:] = [d for d in ds if not skip.search(f"{r}/{d}")]
        for f in fs:
            if pat.search(f):
                p = f"{r}/{f}"
                try: print(mtime(p), os.path.getsize(p), p)
                except OSError: pass
