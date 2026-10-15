"""Killcoyne reconciliation grid inputs (docs/paper_plan_killcoyne_reconcile.md @ a655e08, plan C factor 2): export the release CNV
feature table (load_cnv_matrix: 587 window-minus-arm + 44 arms + cx, unstandardised) keyed by cnv_id, and the release discovery rows.
Row-level outputs stay on the cluster."""
import sys, numpy as np, pandas as pd
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/multimodal-barretts-progression"; sys.path.insert(0, B + "/src"); from barrett.training.data import load_cnv_matrix
O = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"
cnv, feats = load_cnv_matrix(F + "/feature_views/cnv"); cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str)
print("feature view rows", len(cnv), "features", len(feats))
m = cnv[["sample_id"] + feats].merge(cx[["sample_id", "cnv_id"]], on="sample_id"); smp = pd.read_csv(O + "/kr_samples.csv", dtype=str)
m = m[m.cnv_id.isin(smp.Sample)].drop_duplicates("cnv_id"); out = m[["cnv_id"] + feats].rename(columns={"cnv_id": "Sample"})
out.to_csv(O + "/rel_features.csv", index=False); print("release feature rows among the 773 published", len(out), "NaN cells", int(out[feats].isna().sum().sum()))
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); r = man.merge(cx[["sample_id", "cnv_id"]], on="sample_id"); r = r[r.cnv_id.isin(smp.Sample)]
dp = set(smp.Patient); fdp = smp.set_index("Sample").Patient; r = r[r.cnv_id.map(fdp).isin(dp)]
pd.DataFrame({"Sample": r.cnv_id.values, "release_sample_id": r.sample_id.values}).to_csv(O + "/rel_rows.csv", index=False); print("release discovery rows matched to published samples", len(r), "patients", r.cnv_id.map(fdp).nunique())
