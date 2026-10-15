"""Dataset description, probe: field names and codes needed by dd_describe.py (DB columns, BAM header/read length, slide properties). Prints aggregates only."""
import glob, os, re, subprocess, collections, json
import pandas as pd
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
ST = "/mnt/scratche/slow/fmlab/zuberi01/envs/killcoyne_r/bin/samtools"
pe = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str)
pids = set(pe.participant_id.dropna())
print("participant_ids", len(pids))
pat = re.compile("sex|gender|smok|prague|circum|maxim|dob|birth", re.I)
for f in sorted(glob.glob(E + "/*.parquet")):
    try: d = pd.read_parquet(f)
    except Exception as e: print("ERR", f, e); continue
    hit = [c for c in d.columns if pat.search(c)]
    if hit: print(os.path.basename(f), d.shape, hit, "| id cols:", [c for c in d.columns if re.search("participant|patient", c, re.I)][:6])
ih = pd.read_parquet(E + "/initial_history.parquet")
print(ih.dtypes.to_string())
sub = ih[ih.participant_id.astype(str).isin(pids)]
print("initial_history rows for release pids", len(sub), "unique", sub.participant_id.nunique())
for c in ih.columns:
    if re.search("smok|prague|dob|sex|gender", c, re.I): print(c, sub[c].astype(str).value_counts(dropna=False).head(12).to_dict() if c != "dob" else sub[c].notna().sum())
dm = pd.read_csv(S + "/Demographics_full.csv", dtype=str)
print("Demographics smoking", dm["Smoking Status"].value_counts(dropna=False).to_dict()); print("Sex", dm["Sex"].value_counts(dropna=False).to_dict())
for c in ["Circumference ", "Maximal"]: print(c, dm[c].notna().sum(), dm[c].dropna().head(8).tolist())
# BAM
bams = sorted(glob.glob(S + "/dna_seq_bam/*.bam")); print("bams", len(bams))
for b in bams[:2] + bams[-2:]:
    h = subprocess.run([ST, "view", "-H", b], capture_output=True, text=True).stdout.splitlines()
    print(os.path.basename(b), [l[:200] for l in h if l.startswith("@PG")][:3], "SQ", sum(l.startswith("@SQ") for l in h), [l for l in h if l.startswith("@SQ")][:2])
    r = subprocess.run(f"{ST} view {b} | head -2000 | cut -f10 | awk '{{print length($0)}}' | sort -n | uniq -c | sort -rn | head -3", shell=True, capture_output=True, text=True).stdout
    print("  readlen", r.replace("\n", " | "))
# slides
import openslide
for p in pe.ImageAbsPath.dropna().unique()[:2]:
    p2 = p.replace("/scratchc/", "/mnt/scratche/fast/") if not os.path.exists(p) else p
    print(p2, os.path.exists(p2))
    if os.path.exists(p2):
        s = openslide.OpenSlide(p2); print({k: v for k, v in s.properties.items() if re.search("mpp|product|model|objective|serial|vendor|level-count", k, re.I)})
print("image path prefixes", collections.Counter(os.path.dirname(p) for p in pe.ImageAbsPath.dropna()).most_common(5))
print("cnv path prefixes", collections.Counter(os.path.dirname(p) for p in pe.CNVAbsPath.dropna()).most_common(5))
A = glob.glob("/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta/ACEB_samples*.csv")[0]
a = pd.read_csv(A, dtype=str); print(a.columns.tolist()); print(a.Batch.value_counts().to_dict()); print(a.SLX.value_counts().head(20).to_dict())
print("DONE_PROBE")
