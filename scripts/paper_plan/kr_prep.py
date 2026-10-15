"""Killcoyne reconciliation prep (docs/paper_plan_killcoyne_reconcile.md @ a655e08, plan A inputs): (1) the 773 published samples
with patient, sheet status, pathology, endoscopy order and count-file paths; (2) the package blacklist (hg19) lifted to hg38 with
the UCSC hg19ToHg38 chain (the package default is hg19; our counts are hg38). Row-level outputs stay on the cluster."""
import os, gzip, urllib.request, numpy as np, pandas as pd
S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; K = "/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper"; P = "/mnt/scratche/slow/fmlab/zuberi01/phd/BarrettsProgressionRisk"
O = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"; os.makedirs(O, exist_ok=True); D = S + "/copy_number_hg38/train/perPatient/50kb"
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
kp = pd.read_excel(K + "/41591_2020_1033_MOESM4_ESM.xlsx", sheet_name="Supporting data for Figure 2a", header=1)
s = pd.DataFrame({"Sample": kp.Samplename.values, "k_prob": kp.Probability.values}); s["Patient"] = s.Sample.map(fd.Patient); s["Status"] = s.Sample.map(fd.Status)
s["Pathology"] = s.Sample.map(fd.Pathology).replace({"BE": "NDBE"}); s["P53 IHC"] = s.Sample.map(fd["p53 status"]); s["year"] = pd.to_numeric(s.Sample.map(fd["Endoscopy Year"]), errors="coerce")
s["Endoscopy"] = s.groupby("Patient").year.rank(method="dense").fillna(1).astype(int)   # endoscopy order within patient (package needs an ordering column only)
s["raw"] = [f"{D}/{x}/50.raw_read_counts.txt" for x in s.Sample]; s["fit"] = [f"{D}/{x}/50.fitted_read_counts.txt" for x in s.Sample]
s["files_ok"] = [os.path.exists(a) and os.path.exists(b) for a, b in zip(s.raw, s.fit)]
print("samples", len(s), "patients", s.Patient.nunique(), "files ok", int(s.files_ok.sum()), "status", s.Status.value_counts().to_dict())
pats = sorted(s.Patient.unique()); s["pat_index"] = s.Patient.map({p: i for i, p in enumerate(pats)}); s.to_csv(O + "/kr_samples.csv", index=False)
# blacklist liftover (hg19 -> hg38): lift start and end separately; keep regions whose two ends land on the same chromosome in order
ch = O + "/hg19ToHg38.over.chain.gz"
if not os.path.exists(ch): urllib.request.urlretrieve("https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz", ch)
from pyliftover import LiftOver
lo = LiftOver(ch); bl = pd.read_csv(P + "/inst/extdata/qDNAseq_blacklistedRegions.txt", sep="\t"); out = []; lost = 0
def lift(c, x, step):   # first liftable position moving inward from x in 10-kb steps (endpoints often sit in unliftable gaps)
    for k in range(0, 2000):
        r = lo.convert_coordinate("chr" + str(c), int(x) - 1 + step * k * 10000)
        if r and r[0][0] == "chr" + str(c): return r[0][1] + 1, k
    return None, None
nudged = 0
for c, a, b in bl[["chromosome", "start", "end"]].itertuples(index=False):
    la, ka = lift(c, a, +1); lb, kb = lift(c, b, -1)
    if la is not None and lb is not None and lb > la and (ka * 10000 + kb * 10000) < (b - a): out.append((str(c), la, lb)); nudged += (ka > 0 or kb > 0)
    else: lost += 1
pd.DataFrame(out, columns=["chromosome", "start", "end"]).to_csv(O + "/blacklist_hg38.txt", sep="\t", index=False)
print("blacklist regions hg19", len(bl), "lifted", len(out), "of which endpoints nudged", nudged, "not lifted", lost, "bp hg19", int((bl.end - bl.start).sum()), "bp hg38", int(sum(b - a for _, a, b in out)))
