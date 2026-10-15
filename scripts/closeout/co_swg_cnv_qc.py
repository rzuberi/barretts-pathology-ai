"""Closeout item C: per-sample CNV QC from the QDNAseq 50 kb outputs the release's CNV features were built from
(copy_number_hg38/train/perPatient/50kb/<cnv_id>/50.copy_number_segmented_output.csv). Metrics per cnv_id:
n_bins (non-NA), noise_mapd (median absolute pairwise difference of consecutive bin log-ratios; QDNAseq-style noise),
sd_logratio, n_segments (runs of constant segmented value), frac_altered (|segmented - median(segmented)| > 0.15 and > 0.30; values are relative copy-number ratios centred near 1.0, not log-ratios). Row-level table stays on the
cluster (feasibility/closeout/swg_cnv_qc.csv); no aggregate is written here (co_main.py summarises)."""
import os, numpy as np, pandas as pd
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; ROW = T + "/feasibility/closeout"; os.makedirs(ROW, exist_ok=True)
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str); rows = []
for cid, path in cx.drop_duplicates("cnv_id")[["cnv_id", "cnv_id"]].values:
    d = None
    for root in ["/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort/copy_number_hg38/train/perPatient/50kb/", "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort/copy_number_hg38/val/50kb/", "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort/copy_number_hg38/val/perPatient/50kb/"]:
        p = root + cid + "/50.copy_number_segmented_output.csv"
        if os.path.exists(p): d = pd.read_csv(p, usecols=["chromosome", "start", "copy_number", "segmented"]); break
    if d is None: rows.append({"cnv_id": cid, "found": False}); continue
    d = d.dropna(subset=["copy_number"]); v = d.copy_number.values.astype(float); s = d.segmented.values.astype(float)
    ch = d.chromosome.astype(str).values; same = ch[1:] == ch[:-1]
    diff = np.abs(np.diff(v))[same]; nseg = int((np.abs(np.diff(s))[same] > 1e-9).sum() + len(np.unique(ch)))
    rows.append({"cnv_id": cid, "found": True, "n_bins": int(len(v)), "noise_mapd": float(np.median(diff)) if len(diff) else np.nan, "sd_logratio": float(np.std(v)), "n_segments": nseg, "frac_altered_0p15": float((np.abs(s - np.median(s)) > 0.15).mean()), "frac_altered_0p3": float((np.abs(s - np.median(s)) > 0.3).mean()), "median_segmented": float(np.median(s))})
    if len(rows) % 100 == 0: print(len(rows), flush=True)
pd.DataFrame(rows).to_csv(ROW + "/swg_cnv_qc.csv", index=False); print("done", len(rows), "found", sum(r.get("found", False) for r in rows))
