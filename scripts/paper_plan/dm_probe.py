"""Demographics completion (docs/demographics_completion.md), structure probe. Lists table/column names of the Barrett's DB export, the database
report JSONs, any column dictionary rows about smoking, candidate scraping code, ACE-B metadata file headers (column names only) and the SQLite
schema of BE_Progression_Project.db (names only), and the BarrettsProgressionRisk package files. Values are printed only for demographic-type
columns of the DB tables (no pathology, outcome or follow-up values). Output: feasibility/paper_plan/demographics/probe.txt (cluster)."""
import os, re, glob, json, sqlite3, subprocess, pandas as pd
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; A = "/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta"; OUT = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/demographics"
os.makedirs(OUT, exist_ok=True); log = open(OUT + "/probe.txt", "w")
def P(*a): print(*a, file=log, flush=True)
DEMO = re.compile(r"smok|pack|cigar|tobac|quit|sex|gender|bmi|height|weight|alcohol|drink|units|hernia|hiat|ppi|proton|fam|ethnic|race|diagnos|dob|birth|^age|_age|prague|circum|maxim", re.I)
P("== DB tables"); cols = []
for f in sorted(glob.glob(E + "/*.parquet")):
    d = pd.read_parquet(f); nm = os.path.basename(f); P(nm, d.shape)
    for c in d.columns:
        cols.append((nm, c, str(d[c].dtype), int(d[c].notna().sum())))
        if DEMO.search(c) and not re.search(r"patho|histol|outcome|followup|follow_up|diagnosis_text|report", c, re.I):
            v = d[c].astype(str); vc = v.value_counts(dropna=False)
            P(f"   {c} dtype={d[c].dtype} nonnull={int(d[c].notna().sum())} distinct={v.nunique()} top={vc.head(8).to_dict() if v.nunique() <= 40 else 'many'}")
pd.DataFrame(cols, columns=["table", "column", "dtype", "nonnull"]).to_csv(OUT + "/db_columns.csv", index=False)
P("== report JSONs")
for f in sorted(glob.glob(E + "/*.json")): P(f, open(f).read()[:1500].replace("\n", " "))
P("== occams_table_column rows mentioning smok")
try:
    oc = pd.read_parquet(E + "/occams_table_column.parquet"); P(oc.columns.tolist(), oc.shape)
    m = oc.apply(lambda r: r.astype(str).str.contains("smok", case=False).any(), axis=1); P(oc[m].to_string()[:4000])
except Exception as ex: P("ERR", ex)
P("== csv files in export (headers)")
for f in sorted(glob.glob(E + "/*.csv")): P(f, pd.read_csv(f, nrows=0).columns.tolist()[:60])
P("== candidate scraping / dictionary code (grep initial_history|initialhistory|barretts_db_export|smoking)")
roots = ["/mnt/scratche/slow/fmlab/zuberi01", "/home/zuberi01"]
cmd = ["grep", "-rIl", "--include=*.py", "--include=*.R", "--include=*.r", "--include=*.sql", "--include=*.sh", "--include=*.ipynb", "--include=*.md", "--include=*.txt", "--include=*.yaml", "--include=*.yml", "--include=*.json", "--include=*.js", "--include=*.ts", "--include=*.php", "-i", "-E", "initial_?history|barretts_db_export", "--exclude-dir=.git", "--exclude-dir=envs", "--exclude-dir=miniforge3", "--exclude-dir=.conda", "--exclude-dir=site-packages", "--exclude-dir=runs"]
hits = []
for r in roots:
    p = subprocess.run(cmd + [r], capture_output=True, text=True, timeout=7200); hits += p.stdout.splitlines()
P(len(hits)); [P("  ", h) for h in hits[:300]]
P("== smoking mentions in those files")
for h in hits[:300]:
    try:
        for i, line in enumerate(open(h, errors="ignore")):
            if re.search(r"smok", line, re.I): P(f"  {h}:{i+1}: {line.strip()[:220]}")
    except Exception: pass
P("== dictionary-like file names under zuberi01, barretts_db_export, aceb_meta")
for r in ["/mnt/scratche/slow/fmlab/zuberi01", "/home/zuberi01"]:
    p = subprocess.run(["lfs", "find", r, "-type", "f"], capture_output=True, text=True) if r.startswith("/mnt") else subprocess.run(["find", r, "-type", "f"], capture_output=True, text=True)
    for x in p.stdout.splitlines():
        b = os.path.basename(x)
        if re.search(r"dictionar|codebook|code_book|lookup|schema|enum|form_def|crf|\.sql$|\.mdb$|\.accdb$|\.bak$|data_dict", b, re.I) and not re.search(r"site-packages|/envs/|miniforge|\.git/|node_modules", x): P("  ", x)
P("== aceb_meta headers (names only)")
for f in sorted(glob.glob(A + "/*")):
    if f.endswith(".csv"):
        try: P(os.path.basename(f), pd.read_csv(f, nrows=0).columns.tolist())
        except Exception as ex: P(os.path.basename(f), "ERR", ex)
    elif f.endswith(".db"):
        con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        for (t,) in con.execute("select name from sqlite_master where type='table'"): P("  sqlite", t, [r[1] for r in con.execute(f"pragma table_info('{t}')")], con.execute(f"select count(*) from '{t}'").fetchone()[0])
    else: P(os.path.basename(f), "(not read)")
P("== BarrettsProgressionRisk package files")
pk = "/mnt/scratche/slow/fmlab/zuberi01/envs/killcoyne_r/lib/R/library/BarrettsProgressionRisk"
for x in sorted(subprocess.run(["find", pk], capture_output=True, text=True).stdout.splitlines()): P("  ", x)
for x in subprocess.run(["find", "/mnt/scratche/slow/fmlab/zuberi01", "-maxdepth", "6", "-iname", "*BarrettsProgressionRisk*", "-not", "-path", "*/envs/*"], capture_output=True, text=True).stdout.splitlines(): P("  clone?", x)
P("PROBE DONE")
