"""Horizons H4 (docs/paper_survival_horizons.md @ ee51db8): search for (i) 4x resequenced discovery reads and (ii) ACE-B reads. Lists every root
searched, every read file whose name carries a discovery or ACE-B SLX id (outside SWGCohort/dna_seq_bam for discovery), and depth-named directories.
No outcome or pathology field is read (ACE-B manifest: SLX and SampleID columns only)."""
import os, re, json, subprocess, glob, collections
import pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
OUT = T + "/feasibility/paper_plan/killcoyne_mm/horizons"; os.makedirs(OUT, exist_ok=True)
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]
disc = set()
for c in ("SLXID1", "SLXID2"):
    for v in fd[c].dropna(): disc |= set(re.findall(r"SLX-\d+", v))
ac = pd.read_csv("/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta/ACEB_samples_for Rehan.csv", dtype=str, usecols=["SLX", "SampleID"])
aceb = set(); [aceb.update(re.findall(r"SLX-\d+", v)) for v in ac.SLX.dropna()]
roots = sorted(set(d.rstrip("/") for d in glob.glob("/mnt/scratche/*/fmlab") + glob.glob("/mnt/scratch*/fmlab") + glob.glob("/mnt/scratche/*/fmlab/datasets")))
roots = [r for r in roots if not any(r != q and r.startswith(q + "/") for q in roots)]
ext = re.compile(r"\.(bam|cram|fastq|fq|fastq\.gz|fq\.gz|sra)$", re.I); depthdir = re.compile(r"(^|[_\-.])(4x|deep|reseq|resequenc|highdepth|high_depth)", re.I)
files, dirs, errs = [], [], {}
for r in roots:
    p = subprocess.run(["find", r, "-xdev", "(", "-name", ".snapshot", "-prune", ")", "-o", "(", "-type", "f", "-o", "-type", "d", ")", "-printf", "%y\t%s\t%p\n"], capture_output=True, text=True)
    errs[r] = len(p.stderr.splitlines())
    for line in p.stdout.splitlines():
        t, sz, path = line.split("\t", 2)
        if t == "f" and ext.search(path): files.append((path, int(sz)))
        elif t == "d" and depthdir.search(os.path.basename(path)): dirs.append(path)
pd.DataFrame(files, columns=["path", "bytes"]).to_csv(OUT + "/h4_read_files_all.csv", index=False)
def ids_in(p): return set(re.findall(r"SLX-\d+", p))
dfiles = [(p, s) for p, s in files if ids_in(p) & disc and "/SWGCohort/dna_seq_bam/" not in p]
afiles = [(p, s) for p, s in files if ids_in(p) & aceb]
def agg(L):
    g = collections.defaultdict(list)
    for p, sz in L: g[os.path.dirname(p)].append((p, sz))
    return {d: {"files": len(v), "GB": round(sum(sz for _, sz in v) / 1e9, 2), "ext": dict(collections.Counter(ext.search(p).group(1).lower() for p, _ in v))} for d, v in sorted(g.items())}
res = {"roots_searched": roots, "find_stderr_lines": errs, "n_read_files_total": len(files), "discovery_slx_ids": len(disc), "aceb_slx_ids": sorted(aceb),
       "discovery_reads_outside_dna_seq_bam": agg(dfiles), "aceb_reads": agg(afiles), "depth_named_dirs": dirs[:200], "n_depth_named_dirs": len(dirs)}
pd.DataFrame(afiles, columns=["path", "bytes"]).to_csv(OUT + "/h4_aceb_read_files.csv", index=False)
pd.DataFrame(dfiles, columns=["path", "bytes"]).to_csv(OUT + "/h4_discovery_other_read_files.csv", index=False)
os.makedirs(T + "/results/paper_final/horizons", exist_ok=True); json.dump(res, open(T + "/results/paper_final/horizons/h4_search.json", "w"), indent=1)
print("HZ SEARCH DONE", len(files), len(dfiles), len(afiles), len(dirs))
