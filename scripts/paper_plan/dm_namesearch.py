"""Demographics completion D2: sharded `lfs find` over the lab scratch (same shards as hz_search_shard.py) for files whose name suggests Killcoyne 2020
supplementary data (Nature MOESM files, the PMC 'Supp_Appendix', EGA dataset EGAD00001006033, 'supplementary'/'extended data' tables, 'killcoyne').
MODE=merge -> results/paper_final/demographics/d2_name_search.json (paths only)."""
import os, re, sys, json, glob, subprocess
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; SH = T + "/feasibility/paper_plan/demographics/name_shards"; os.makedirs(SH, exist_ok=True)
PAT = re.compile(r"moesm|41591|1033|supp(lement)?[_ -]?(ary)?[_ -]?(appendix|table|info|data)|supp_appendix|extended[_ -]?data|EGAD00001006033|EGAS|killcoyne|s41591|table[_ -]?s1\b|suppl", re.I)
EXT = re.compile(r"\.(xlsx|xls|csv|tsv|txt|pdf|docx|doc|rds|rda|rdata|zip|json)$", re.I)
SKIP = re.compile(r"site-packages|/envs/|miniforge|/\.git/|node_modules|/\.pixi/|/lib/python|/conda|/R/library/(?!BarrettsProgressionRisk)", re.I)
if os.environ.get("MODE") != "merge":
    d = os.environ["SHARD"]; md = os.environ.get("MAXD", ""); key = re.sub(r"[^A-Za-z0-9]+", "_", d).strip("_") + ("_top" if md else ""); outf = f"{SH}/{key}.json"
    if os.path.exists(outf): sys.exit(0)
    p = subprocess.run(["lfs", "find", d] + (["-maxdepth", md] if md else []) + ["-type", "f"], capture_output=True, text=True)
    hits = [x for x in p.stdout.splitlines() if PAT.search(os.path.basename(x)) and EXT.search(x) and not SKIP.search(x)]
    json.dump({"shard": d, "n_files": len(p.stdout.splitlines()), "stderr_lines": len(p.stderr.splitlines()), "hits": hits}, open(outf, "w")); print("SHARD DONE", d, len(hits))
else:
    S = [json.load(open(f)) for f in sorted(glob.glob(SH + "/*.json"))]; hits = sorted(set(h for s in S for h in s["hits"]))
    os.makedirs(T + "/results/paper_final/demographics", exist_ok=True)
    json.dump({"pattern": PAT.pattern, "extensions": EXT.pattern, "excluded_paths": SKIP.pattern, "n_shards": len(S), "n_files_listed": sum(s["n_files"] for s in S), "stderr_lines": sum(s["stderr_lines"] for s in S), "hits": hits},
              open(T + "/results/paper_final/demographics/d2_name_search.json", "w"), indent=1); print("MERGE DONE", len(S), len(hits))
