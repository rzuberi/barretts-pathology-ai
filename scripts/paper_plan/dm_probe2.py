"""Demographics completion D1.1/D2 (docs/demographics_completion.md @ 24b2972), replacing the slow recursive grep of dm_probe.py.
PART=code SHARD=<dir> [MAXD=1]: `lfs find` files < 5 MB with code/text extensions, grep -il for initial_history|initialhistory|barretts_db_export,
then the lines mentioning 'smok' in those files -> feasibility/paper_plan/demographics/code_shards/. PART=rest: dictionary-like file names,
ACE-B metadata headers (column names only), SQLite schema of BE_Progression_Project.db (names only), BarrettsProgressionRisk package files.
PART=merge: -> results/paper_final/demographics/d1_search.json."""
import os, re, sys, json, glob, sqlite3, subprocess, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; D = T + "/feasibility/paper_plan/demographics"; CS = D + "/code_shards"; os.makedirs(CS, exist_ok=True)
A = "/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta"; PART = os.environ["PART"]
CODE = re.compile(r"\.(py|r|R|sql|sh|ipynb|md|txt|yaml|yml|json|js|ts|php|cfg|ini|toml)$")
SKIP = re.compile(r"site-packages|/envs/|miniforge|/\.git/|node_modules|/\.pixi/|/lib/python|/R/library|/\.conda|/runs/|/feasibility/paper_plan/demographics/")
if PART == "code":
    d = os.environ["SHARD"]; md = os.environ.get("MAXD", ""); co = os.environ.get("CODEONLY", ""); key = re.sub(r"[^A-Za-z0-9]+", "_", d).strip("_") + ("_top" if md else "") + ("_codeonly" if co else ""); outf = f"{CS}/{key}.json"
    if co: CODE = re.compile(r"\.(py|r|R|sql|sh|ipynb|md|js|ts|php)$")   # narrower extension set for the slowest directories (output .json/.txt excluded)
    if os.path.exists(outf): sys.exit(0)
    lf = subprocess.run((["lfs", "find", d] if d.startswith("/mnt/") else ["find", d]) + (["-maxdepth", md] if md else []) + ["-type", "f", "-size", "-5M"], capture_output=True, text=True).stdout.splitlines()
    files = [f for f in lf if CODE.search(f) and not SKIP.search(f)]; hits, smok = [], []
    for i in range(0, len(files), 500):
        p = subprocess.run(["grep", "-ilE", "initial_?history|barretts_db_export"] + files[i:i + 500], capture_output=True, text=True); hits += p.stdout.splitlines()
    for h in hits:
        try:
            for n, line in enumerate(open(h, errors="ignore")):
                if re.search(r"smok|packyear|pack_year|quit", line, re.I): smok.append(f"{h}:{n + 1}: {line.strip()[:240]}")
        except Exception: pass
    json.dump({"shard": d, "codeonly": bool(co), "n_candidate_files": len(files), "hits": hits, "smoking_lines": smok[:400]}, open(outf, "w")); print("CODE SHARD DONE", d, len(files), len(hits))
elif PART == "rest":
    out = {"dictionary_like_names": [], "aceb_meta_headers": {}, "aceb_sqlite_schema": {}, "package_files": []}
    for r in ["/mnt/scratche/slow/fmlab/zuberi01", "/mnt/scratche/fast/fmlab/zuberi01", "/home/zuberi01"]:
        if not os.path.isdir(r): continue
        lf = subprocess.run((["lfs", "find", r] if r.startswith("/mnt/") else ["find", r]) + ["-type", "f"], capture_output=True, text=True).stdout.splitlines()
        out["dictionary_like_names"] += [x for x in lf if re.search(r"dictionar|codebook|code_book|lookup|schema|enum|form_def|\bcrf\b|\.sql$|\.mdb$|\.accdb$|\.bak$|data_dict", os.path.basename(x), re.I) and not SKIP.search(x)]
    for f in sorted(glob.glob(A + "/*")):
        if f.endswith(".csv"):
            try: out["aceb_meta_headers"][os.path.basename(f)] = pd.read_csv(f, nrows=0).columns.tolist()
            except Exception as ex: out["aceb_meta_headers"][os.path.basename(f)] = f"ERR {ex}"
        elif f.endswith(".db"):
            con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
            for (t,) in con.execute("select name from sqlite_master where type='table'"): out["aceb_sqlite_schema"][t] = {"columns": [r[1] for r in con.execute(f"pragma table_info('{t}')")], "rows": con.execute(f"select count(*) from '{t}'").fetchone()[0]}
    pk = "/mnt/scratche/slow/fmlab/zuberi01/envs/killcoyne_r/lib/R/library/BarrettsProgressionRisk"
    out["package_files"] = sorted(subprocess.run(["find", pk], capture_output=True, text=True).stdout.splitlines())
    out["package_clones"] = subprocess.run(["find", "/mnt/scratche/slow/fmlab/zuberi01", "-maxdepth", "7", "-type", "d", "-iname", "*BarrettsProgressionRisk*", "-not", "-path", "*/envs/*"], capture_output=True, text=True).stdout.splitlines()
    json.dump(out, open(D + "/probe_rest.json", "w"), indent=1); print("REST DONE")
else:
    S = [json.load(open(f)) for f in sorted(glob.glob(CS + "/*.json"))]; R = json.load(open(D + "/probe_rest.json"))
    res = {"code_search": {"roots": ["/mnt/scratche/slow/fmlab/zuberi01", "/mnt/scratche/fast/fmlab/zuberi01", "/home/zuberi01"], "n_shards": len(S), "n_candidate_files": sum(s["n_candidate_files"] for s in S),
           "codeonly_shards": [s["shard"] for s in S if s.get("codeonly")], "files_referencing_db": sorted(set(h for s in S for h in s["hits"])), "smoking_lines": sorted(set(l for s in S for l in s["smoking_lines"]))},
           "dictionary_like_names": R["dictionary_like_names"], "package_files": R["package_files"], "package_clones": R["package_clones"], "aceb_meta_files_with_headers": len(R["aceb_meta_headers"]), "aceb_sqlite_tables": list(R["aceb_sqlite_schema"])}
    os.makedirs(T + "/results/paper_final/demographics", exist_ok=True); json.dump(res, open(T + "/results/paper_final/demographics/d1_search.json", "w"), indent=1); print("MERGE DONE", len(S))
