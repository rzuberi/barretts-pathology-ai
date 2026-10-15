"""PORPOISE data build (1.4, step 1 of 2): construct the omics + feature inputs
their pipeline expects, for TCGA-ESCA + TCGA-STAD.

Downloads cBioPortal PanCanAtlas study archives (epyc has internet), extracts
mRNA z-scores / CNA / mutations, filters to PORPOISE's signatures.csv gene
sets, and writes an all_clean.csv in their column convention
({gene}_rnaseq | _cnv | _mut) with survival columns. Also converts our UNI2
h5 tile features to per-slide .pt files in their expected layout, and writes
custom split CSVs mirroring our frozen patient-disjoint folds.
Step 2 (training run with input-dim patch) follows once this lands.
"""
import glob, json, os, re, subprocess, tarfile
import h5py, numpy as np, pandas as pd
import torch

P = "/mnt/scratche/slow/fmlab/zuberi01/phd/mahmood_lab/PORPOISE"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
WORK = os.path.join(OUT, "cbioportal")
os.makedirs(WORK, exist_ok=True)
STUDIES = ["esca_tcga_pan_can_atlas_2018", "stad_tcga_pan_can_atlas_2018"]

FILES = ["data_clinical_patient.txt", "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt",
         "data_mrna_seq_v2_rsem.txt", "data_cna.txt", "data_mutations.txt"]
for st in STUDIES:
    d = os.path.join(WORK, st); os.makedirs(d, exist_ok=True)
    for fn in FILES:
        dst = os.path.join(d, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000: continue
        url = f"https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/{st}/{fn}"
        print("downloading", st, fn, flush=True)
        r = subprocess.run(["curl", "-sfL", "-o", dst, url])
        if r.returncode != 0 or (os.path.exists(dst) and os.path.getsize(dst) < 1000):
            print("  unavailable:", fn, flush=True)
            if os.path.exists(dst): os.remove(dst)

sig = pd.read_csv(os.path.join(P, "datasets_csv/signatures.csv"))
genes = sorted(set(sig.values.ravel()) - {np.nan}
               - {v for v in sig.values.ravel() if pd.isna(v)})
genes = [g for g in genes if isinstance(g, str)]
print(f"signature genes: {len(genes)}", flush=True)

def load_study(st):
    d = os.path.join(WORK, st)
    def table(name, idcol="Hugo_Symbol"):
        f = os.path.join(d, name)
        if not os.path.exists(f): return None
        df = pd.read_csv(f, sep="\t", low_memory=False)
        return df
    rna = table("data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt")
    if rna is None:
        rna = table("data_mrna_seq_v2_rsem.txt")
    cna = table("data_cna.txt")
    mut = table("data_mutations.txt")
    clin = pd.read_csv(os.path.join(d, "data_clinical_patient.txt"), sep="\t", comment="#")
    return rna, cna, mut, clin

rows = {}
for st in STUDIES:
    rna, cna, mut, clin = load_study(st)
    clin = clin.set_index("PATIENT_ID")
    def to_pat(s): return s[:12]
    def gene_matrix(df):
        if df is None: return {}
        df = df[df["Hugo_Symbol"].isin(genes)].drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")
        samp = [c for c in df.columns if c.startswith("TCGA")]
        return {to_pat(c): df[c] for c in samp}
    R, C = gene_matrix(rna), gene_matrix(cna)
    M = {}
    if mut is not None:
        mm = mut[mut["Hugo_Symbol"].isin(genes)]
        for pat, g in mm.groupby(mm["Tumor_Sample_Barcode"].str[:12]):
            M[pat] = set(g["Hugo_Symbol"])
    for pat in set(R) | set(C) | set(M):
        if pat not in clin.index: continue
        cl = clin.loc[pat]
        os_m = pd.to_numeric(cl.get("OS_MONTHS"), errors="coerce")
        os_s = str(cl.get("OS_STATUS", ""))
        if pd.isna(os_m): continue
        row = {"case_id": pat, "study": st,
               "survival_months": float(os_m),
               "censorship": 0 if "DECEASED" in os_s else 1}
        for g in genes:
            if pat in R and g in R[pat].index:
                row[f"{g}_rnaseq"] = float(pd.to_numeric(R[pat][g], errors="coerce"))
            if pat in C and g in C[pat].index:
                row[f"{g}_cnv"] = float(pd.to_numeric(C[pat][g], errors="coerce"))
            row[f"{g}_mut"] = 1.0 if (pat in M and g in M[pat]) else 0.0
        rows[pat] = row
omics = pd.DataFrame(list(rows.values()))
print(f"omics cases: {len(omics)}", flush=True)

# match to our slides + convert features to .pt
FEATS = {"esca": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2",
         "stad": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/features/20x_224px/features_uni_v2"}
PT = os.path.join(OUT, "pt_files")
os.makedirs(PT, exist_ok=True)
slide_rows = []
for grp, fd in FEATS.items():
    for f in sorted(glob.glob(os.path.join(fd, "*.h5"))):
        mm = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
        if not mm or mm.group(1) not in rows: continue
        sid = os.path.basename(f).replace(".h5", "")
        ptp = os.path.join(PT, sid + ".pt")
        if not os.path.exists(ptp):
            with h5py.File(f) as h:
                torch.save(torch.tensor(np.asarray(h["features"]), dtype=torch.float32), ptp)
        slide_rows.append({"case_id": mm.group(1), "slide_id": sid})
slides = pd.DataFrame(slide_rows).drop_duplicates("slide_id")
final = slides.merge(omics, on="case_id")
final.to_csv(os.path.join(OUT, "tcga_esca_stad_all_clean.csv"), index=False)
print(f"final: {len(final)} slides, {final['case_id'].nunique()} cases", flush=True)

# frozen patient-disjoint splits (5 fold), reproducible
cases = sorted(final["case_id"].unique())
rng = np.random.RandomState(0)
fold = {c: i % 5 for i, c in enumerate(rng.permutation(cases))}
os.makedirs(os.path.join(OUT, "splits"), exist_ok=True)
for k in range(5):
    tr = [c for c in cases if fold[c] != k]
    va = [c for c in cases if fold[c] == k]
    n = max(len(tr), len(va))
    pd.DataFrame({"train": tr + [""] * (n - len(tr)),
                  "val": va + [""] * (n - len(va))}).to_csv(
        os.path.join(OUT, "splits", f"splits_{k}.csv"), index=False)
json.dump({"n_slides": len(final), "n_cases": int(final["case_id"].nunique()),
           "n_genes": len(genes),
           "omics_cols": int(sum(c.endswith(("_rnaseq", "_cnv", "_mut"))
                                 for c in final.columns))},
          open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
