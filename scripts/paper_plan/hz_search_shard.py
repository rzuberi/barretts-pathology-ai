"""Horizons H4 (docs/paper_survival_horizons.md @ ee51db8): sharded version of hz_search.py (the single find ran > 3.5 h). SHARD = one directory;
`lfs find` (Lustre) lists files and directories; read files and depth-named directories are kept. MODE=merge combines the shards into
results/paper_final/horizons/h4_search.json with the same fields as hz_search.py. No outcome or pathology field is read."""
import os, re, sys, json, glob, subprocess, collections, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; OUT = T + "/feasibility/paper_plan/killcoyne_mm/horizons"; SH = OUT + "/h4_shards"; os.makedirs(SH, exist_ok=True)
ext = re.compile(r"\.(bam|cram|fastq|fq|fastq\.gz|fq\.gz|sra)$", re.I); depthdir = re.compile(r"(^|[_\-.])(4x|deep|reseq|resequenc|highdepth|high_depth)", re.I)
if os.environ.get("MODE") != "merge":
    d = os.environ["SHARD"]; md = os.environ.get("MAXD", ""); key = re.sub(r"[^A-Za-z0-9]+", "_", d).strip("_") + ("_top" if md else ""); outf = f"{SH}/{key}.json"
    if os.path.exists(outf): sys.exit(0)
    p = subprocess.run(["lfs", "find", d] + (["-maxdepth", md] if md else []), capture_output=True, text=True); files, dirs = [], []
    for path in p.stdout.splitlines():
        b = os.path.basename(path)
        if ext.search(b):
            try: files.append((path, os.path.getsize(path)))
            except OSError: files.append((path, -1))
        elif depthdir.search(b) and os.path.isdir(path): dirs.append(path)
    json.dump({"shard": d, "n_entries": len(p.stdout.splitlines()), "stderr_lines": len(p.stderr.splitlines()), "files": files, "dirs": dirs}, open(outf, "w")); print("SHARD DONE", d, len(files))
else:
    fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]
    disc = set(); [disc.update(re.findall(r"SLX-\d+", v)) for c in ("SLXID1", "SLXID2") for v in fd[c].dropna()]
    ac = pd.read_csv("/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta/ACEB_samples_for Rehan.csv", dtype=str, usecols=["SLX"]); aceb = set(); [aceb.update(re.findall(r"SLX-\d+", v)) for v in ac.SLX.dropna()]
    shards = [json.load(open(f)) for f in sorted(glob.glob(SH + "/*.json"))]; files = sorted(set(tuple(x) for s in shards for x in s["files"])); dirs = sorted(set(x for s in shards for x in s["dirs"]))
    ids = lambda p: set(re.findall(r"SLX-\d+", p))
    dfiles = [(p, z) for p, z in files if ids(p) & disc and "/SWGCohort/dna_seq_bam/" not in p]; afiles = [(p, z) for p, z in files if ids(p) & aceb]
    def agg(L):
        g = collections.defaultdict(list)
        for p, z in L: g[os.path.dirname(p)].append((p, z))
        return {k: {"files": len(v), "GB": round(sum(max(z, 0) for _, z in v) / 1e9, 2), "ext": dict(collections.Counter(ext.search(p).group(1).lower() for p, _ in v))} for k, v in sorted(g.items())}
    res = {"roots_searched": ["/mnt/scratche/fast/fmlab", "/mnt/scratche/slow/fmlab"], "method": "lfs find, sharded by depth-2 directory", "shards": [{"dir": s["shard"], "entries": s["n_entries"], "stderr_lines": s["stderr_lines"]} for s in shards],
           "find_stderr_lines": sum(s["stderr_lines"] for s in shards), "n_entries_listed": sum(s["n_entries"] for s in shards), "n_read_files_total": len(files), "discovery_slx_ids": len(disc), "aceb_slx_ids": sorted(aceb),
           "discovery_reads_outside_dna_seq_bam": agg(dfiles), "aceb_reads": agg(afiles), "depth_named_dirs": dirs[:200], "n_depth_named_dirs": len(dirs)}
    pd.DataFrame(files, columns=["path", "bytes"]).to_csv(OUT + "/h4_read_files_all.csv", index=False); pd.DataFrame(afiles, columns=["path", "bytes"]).to_csv(OUT + "/h4_aceb_read_files.csv", index=False)
    json.dump(res, open(T + "/results/paper_final/horizons/h4_search.json", "w"), indent=1); print("HZ SEARCH MERGE DONE", len(shards), len(files), len(dfiles), len(afiles), len(dirs))
