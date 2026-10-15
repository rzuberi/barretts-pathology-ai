"""Horizons H4 (docs/paper_survival_horizons.md @ ee51db8): characterise the discovery-SLX read files found outside SWGCohort/dna_seq_bam by the
sharded search (h4_search.json): symlink or file, link target, and for a RandomState(0) sample of 20 real BAMs per directory group the read count,
modal read length, aligner and reference, and the nominal depth, against the dna_seq_bam BAM of the same name. Aggregates ->
results/paper_final/horizons/h4_bam_probe.json."""
import os, re, json, subprocess, collections, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; OUT = T + "/feasibility/paper_plan/killcoyne_mm/horizons"
ST = "/mnt/scratche/slow/fmlab/zuberi01/envs/killcoyne_r/bin/samtools"; GL = 3088269832
F = pd.read_csv(OUT + "/h4_read_files_all.csv"); fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]
disc = set(); [disc.update(re.findall(r"SLX-\d+", v)) for c in ("SLXID1", "SLXID2") for v in fd[c].dropna()]
F = F[F.path.map(lambda p: bool(set(re.findall(r"SLX-\d+", p)) & disc) and "/SWGCohort/dna_seq_bam/" not in p)].copy()
F["group"] = F.path.map(lambda p: "/".join(p.split("/")[:9])); F["islink"] = F.path.map(os.path.islink); F["target"] = F.path.map(lambda p: os.path.realpath(p))
F["target_dir"] = F.target.map(lambda p: "/".join(p.split("/")[:9])); F["exists"] = F.target.map(os.path.exists)
def stats(b):
    h = subprocess.run([ST, "view", "-H", b], capture_output=True, text=True).stdout.splitlines()
    pg = [l for l in h if l.startswith("@PG")]; sq = {l.split("\tSN:")[1].split("\t")[0]: int(l.split("\tLN:")[1].split("\t")[0]) for l in h if l.startswith("@SQ")}
    chr1 = sq.get("chr1", sq.get("1")); build = {248956422: "hg38", 249250621: "hg19"}.get(chr1, f"chr1 length {chr1}")
    n = subprocess.run(f"{ST} view -c -F 0x900 {b}", shell=True, capture_output=True, text=True).stdout.strip()
    L = subprocess.run(f"{ST} view {b} | head -2000 | cut -f10 | awk '{{print length($0)}}'", shell=True, capture_output=True, text=True).stdout.split()
    rl = int(np.bincount([int(x) for x in L]).argmax()) if L else 0; n = int(n) if n.isdigit() else -1
    return {"reads": n, "read_len_mode": rl, "build": build, "pg": [p.split("\tCL:")[1][:120] if "\tCL:" in p else p[:120] for p in pg][:3], "depth": round(n * rl / GL, 4) if n > 0 else None}
res = {"groups": {}}
for g, d in F.groupby("group"):
    o = {"files": int(len(d)), "symlinks": int(d.islink.sum()), "broken_links": int((d.islink & ~d.exists).sum()), "link_target_dirs": d[d.islink].target_dir.value_counts().head(5).to_dict(),
         "bytes_real_files_GB": round(d[~d.islink].bytes.clip(lower=0).sum() / 1e9, 1), "ext": d.path.str.extract(r"\.([a-z.]+)$")[0].value_counts().to_dict()}
    real = d[~d.islink & d.path.str.endswith(".bam")]
    if len(real):
        pick = real.iloc[np.random.RandomState(0).choice(len(real), min(20, len(real)), replace=False)]; rows = []
        for p in pick.path:
            s = stats(p); twin = f"{S}/dna_seq_bam/{os.path.basename(p)}"; s["twin_exists"] = os.path.exists(twin)
            if s["twin_exists"]: t = stats(twin); s.update({"twin_reads": t["reads"], "twin_depth": t["depth"], "twin_build": t["build"]})
            rows.append(s)
        R = pd.DataFrame(rows); o["probe_n"] = len(R); o["depth_median"] = float(R.depth.median()); o["depth_range"] = [float(R.depth.min()), float(R.depth.max())]
        o["read_len_mode"] = R.read_len_mode.value_counts().to_dict(); o["build"] = R.build.value_counts().to_dict(); o["pg_examples"] = rows[0]["pg"]
        if "twin_reads" in R: o["twin_exists"] = int(R.twin_exists.sum()); o["reads_ratio_to_dna_seq_bam_median"] = float((R.reads / R.twin_reads).median()); o["twin_depth_median"] = float(R.twin_depth.median())
        o["names_also_in_dna_seq_bam"] = int(sum(os.path.exists(f"{S}/dna_seq_bam/{os.path.basename(p)}") for p in real.path))
    res["groups"][g] = o; print(g, json.dumps(o)[:400], flush=True)
json.dump(res, open(T + "/results/paper_final/horizons/h4_bam_probe.json", "w"), indent=1); print("HZ BAM PROBE DONE")
