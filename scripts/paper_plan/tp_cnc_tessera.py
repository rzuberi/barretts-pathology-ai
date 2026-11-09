"""Two papers, agent CNC (docs/two_papers_CNC.md): TESSERA (Sidhom et al., bioRxiv 10.64898/2026.05.27.728319; pip tessera-foundation 0.1.7, weights
JW-Sidhom-Lab/tessera-foundation, CC-BY-NC-4.0) CNA-only embeddings of the discovery CBS segments written by tp_cnc_feats.R
(killcoyne_mm/two_papers_CNC/tessera_segments.csv.gz: segments >= 10 bins, log2 ratio clamped to [-3, 3], GRCh38 -> lifted to GRCh37 by the package).
Variant joint_snv_cna_noloh (no LOH from sWGS); per-sample features = mean and max pools of the 688-d per-segment embeddings + log segment count
(the paper's 'genomic fingerprint' construction). Segments capped at 1,000 per sample with the package's subsample_cna (largest |Segment_Mean|).
QN = 1: Segment_Mean rank-mapped onto the TCGA pretraining distribution (the package's recommendation for inputs whose log2 distribution differs);
QN = 0: raw. SMOKE = 1: first 5 samples only (prints shapes). -> killcoyne_mm/two_papers_CNC/feats/tessera_qn.csv.gz / tessera_raw.csv.gz."""
import os, sys, numpy as np, pandas as pd
CD = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm/two_papers_CNC"; CACHE = "/mnt/scratche/slow/fmlab/zuberi01/envs/cnc_tools/cache"
os.environ.setdefault("HF_HOME", CACHE + "/hf"); os.environ.setdefault("TESSERA_REF_GENOME_DIR", CACHE + "/ref"); os.environ.setdefault("TESSERA_LIFTOVER_CHAIN", CACHE + "/chain/hg38ToHg19.over.chain.gz")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import tessera
from tessera.data import preprocessing
QN = os.environ.get("QN", "1") == "1"; SMOKE = os.environ.get("SMOKE", "0") == "1"
seg = pd.read_csv(CD + "/tessera_segments.csv.gz", dtype={"Tumor_Sample_Barcode": str, "Chromosome": str})
order = list(pd.read_csv(CD + "/../set_C.csv", dtype=str).Sample)   # output rows = set C order (a sample without segments gets NaN)
if SMOKE: order = order[:5]; seg = seg[seg.Tumor_Sample_Barcode.isin(order)]
seg = preprocessing.subsample_cna(seg, max_segments=1000)
res = tessera.featurize(cna_df=seg, variant="joint_snv_cna_noloh", from_assembly="GRCh38", quantile_normalize_to_tcga=QN,
                        chain_file=os.environ["TESSERA_LIFTOVER_CHAIN"], batch_size=8)
F = np.asarray(res.cna_features); tab = res.cna_table.reset_index(drop=True); assert len(tab) == F.shape[0], (len(tab), F.shape)
print("segments in", len(seg), "embedded", len(tab), "dims", F.shape[1], "liftover", getattr(res, "liftover_stats", None), flush=True)
rows = []
for s in order:
    m = (tab.Tumor_Sample_Barcode.astype(str) == s).values
    if m.sum() == 0: rows.append(np.full(2 * F.shape[1] + 1, np.nan)); continue
    rows.append(np.concatenate([F[m].mean(0), F[m].max(0), [np.log1p(m.sum())]]))
X = pd.DataFrame(np.vstack(rows), columns=[f"m{i}" for i in range(F.shape[1])] + [f"x{i}" for i in range(F.shape[1])] + ["log_nseg"]); X.insert(0, "Sample", order)
if SMOKE: print(X.iloc[:, :6].describe().T, flush=True); print("TP CNC TESSERA SMOKE OK"); sys.exit(0)
os.makedirs(CD + "/feats", exist_ok=True); X.to_csv(f"{CD}/feats/tessera_{'qn' if QN else 'raw'}.csv.gz", index=False); print("TP CNC TESSERA DONE", X.shape, flush=True)
