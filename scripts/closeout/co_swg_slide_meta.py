"""Closeout item C: scanner metadata for the 707 SWG release slides (openslide properties only; no pixel reads).
Row-level output stays on the cluster (feasibility/closeout/swg_slide_meta.csv: slide filename = accession-like identifier);
the aggregate summary is written to results/closeout/swg_slide_meta_summary.json."""
import json, os, pandas as pd, openslide
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; ROW = T + "/feasibility/closeout"; AGG = T + "/results/closeout"; os.makedirs(ROW, exist_ok=True); os.makedirs(AGG, exist_ok=True)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).merge(man, left_on="SampleID", right_on="sample_id")
KEYS = ["hamamatsu.Created", "hamamatsu.Updated", "hamamatsu.NDP.S/N", "hamamatsu.SourceLens", "hamamatsu.Objective.Lens.Magnificant", "openslide.mpp-x", "openslide.objective-power", "openslide.vendor", "tiff.DateTime", "tiff.Model", "tiff.Software", "openslide.level-count", "openslide.level[0].width", "openslide.level[0].height"]
rows = []
for r in coh.itertuples():
    p = r.ImageAbsPath.replace("/scratchc/fmlab", "/mnt/scratche/fast/fmlab"); d = {"sample_id": r.sample_id, "patient_id": r.patient_id, "biopsy_date": r.Date, "slide_file": os.path.basename(p)}
    try:
        pr = dict(openslide.OpenSlide(p).properties); d.update({k: pr.get(k) for k in KEYS}); d["ok"] = True
    except Exception as e: d["ok"] = False; d["err"] = str(e)[:100]
    rows.append(d)
df = pd.DataFrame(rows); df.to_csv(ROW + "/swg_slide_meta.csv", index=False)
ok = df[df.ok]; summ = {"n_slides": int(len(df)), "n_read_ok": int(len(ok)), "tiff.Model": ok["tiff.Model"].value_counts(dropna=False).to_dict(), "serial": ok["hamamatsu.NDP.S/N"].value_counts(dropna=False).to_dict(),
        "SourceLens": ok["hamamatsu.SourceLens"].value_counts(dropna=False).to_dict(), "software": ok["tiff.Software"].value_counts(dropna=False).to_dict(), "vendor": ok["openslide.vendor"].value_counts(dropna=False).to_dict(),
        "mpp_x_summary": pd.to_numeric(ok["openslide.mpp-x"], errors="coerce").describe().round(4).to_dict(), "scan_year_Created": pd.to_datetime(ok["hamamatsu.Created"], format="%Y/%m/%d", errors="coerce").dt.year.value_counts().sort_index().to_dict(),
        "scan_year_tiffDateTime": pd.to_datetime(ok["tiff.DateTime"], format="%Y:%m:%d %H:%M:%S", errors="coerce").dt.year.value_counts().sort_index().to_dict()}
json.dump(summ, open(AGG + "/swg_slide_meta_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
