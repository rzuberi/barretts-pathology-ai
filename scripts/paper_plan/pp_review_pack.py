"""Paper plan item 10 review pack (pre-specified @ db236a0): 2,048-px thumbnails of every FN slide (late_mean) plus equal
numbers of TP and TN slides, shuffled, opaque names R###.png. Output stays on the cluster: feasibility/paper_plan/review_pack/
(the manifest with outcomes is review_pack_manifest_SECRET.csv, written by pp_main.py)."""
import os, pandas as pd, openslide
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = T + "/feasibility/paper_plan/review_pack"; os.makedirs(OUT, exist_ok=True)
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID"); rows = pd.read_csv(T + "/feasibility/paper_plan/review_pack_slides.csv", dtype=str)
for r in rows.itertuples():
    sp = coh.ImageAbsPath[r.sample_id].replace("/scratchc/fmlab", "/mnt/scratche/fast/fmlab"); sl = openslide.OpenSlide(sp); th = sl.get_thumbnail((2048, 2048)); th.save(f"{OUT}/{r.pack_id}.png"); print(r.pack_id, flush=True)
open(OUT + "/README.txt", "w").write("Blinded review pack: one thumbnail per slide, named R###. Outcomes and identities are in ../review_pack_manifest_SECRET.csv (do not open before grading). Grade each as NDBE / IND / LGD / HGD / other, and note any feature suggesting progression risk.\n"); print("pack done", len(rows))
