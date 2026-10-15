"""R13 of docs/paper_review_round1_analyses.md (baa47ab): N-IMG (release AttentionMIL, image_only) under the kv_cv.R stratified patient-grouped
10-fold x 10-repeat CV. Data loading, model, hyperparameters and training call are those of km_neural.py (release fit_neural / predict_neural,
imported unchanged); only the split changes: outer folds = the `fold` column of cv/preds/cfg_00_rep_XX.csv, training = all samples of the
training patients minus the validation group (early stopping only) from rv_abmil_val.R, test = all samples of the held-out fold. Seed = repeat.
Env: REP (1..10), FOLDS (optional comma list, default 1..10), TORCH_THREADS, DEVICE (default cpu, as the stored km_neural.py runs).
Rows -> feasibility/paper_plan/killcoyne_mm/review_round1/abmil/nimg_rep_XX_fold_YY.csv (cluster only); existing fold files are skipped."""
import os, sys, time, numpy as np, pandas as pd, torch
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/multimodal-barretts-progression"; sys.path.insert(0, B + "/src")
from barrett.training.data import CanonicalFeatureStore
from barrett.training.loops import fit_neural, predict_neural
M = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm"; O = M + "/review_round1/abmil"
family, config = "image_only", dict(hidden_dim=256, attn_dim=128, dropout=0.1, lr=1e-4, weight_decay=0.01, batch_size=8, max_epochs=20, patience=5)   # = km_neural.py CFG["img"]
rep = int(os.environ["REP"]); seed = rep; folds = [int(k) for k in os.environ.get("FOLDS", ",".join(map(str, range(1, 11)))).split(",")]
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2"))); device = torch.device(os.environ.get("DEVICE", "cpu"))
C = pd.read_csv(M + "/set_C.csv", dtype=str); frame = pd.DataFrame({"sample_id": C.Sample, "patient_id": C.Patient, "y_progressor": (C.Status == "P").astype(int)})
cnv = pd.read_csv(f"{M}/cnv_their_C.csv", dtype={"sample_id": str}); feats = [c for c in cnv.columns if c != "sample_id"]   # unused by image_only; store needs it
store = CanonicalFeatureStore(pd.read_csv(M + "/bag_index.csv", dtype=str), cnv, feats)
F = pd.read_csv(f"{M}/cv/preds/cfg_00_rep_{rep:02d}.csv", dtype={"Sample": str, "Patient": str}).set_index("Sample").reindex(frame.sample_id)
assert F.fold.notna().all() and (F.Patient.values == frame.patient_id.values).all() and (F.y.values == frame.y_progressor.values).all()
fold_of = F.fold.astype(int).values; assert (pd.Series(fold_of).groupby(frame.patient_id.values).nunique() == 1).all()
V = pd.read_csv(O + "/val_patients.csv", dtype={"val_patient": str})
for k in folds:
    outf = f"{O}/nimg_rep_{rep:02d}_fold_{k:02d}.csv"
    if os.path.exists(outf): print("exists", outf, flush=True); continue
    t0 = time.time(); vp = set(V[(V.rep == rep) & (V.fold == k)].val_patient); te = frame[fold_of == k]; trp = set(frame.patient_id[fold_of != k])
    assert vp and vp <= trp; va = frame[frame.patient_id.isin(vp)]; tr = frame[(fold_of != k) & ~frame.patient_id.isin(vp)]
    assert not set(tr.patient_id) & set(te.patient_id) and not set(va.patient_id) & set(te.patient_id)
    fit = fit_neural(family, tr, va, store, config, device, seed)
    pr = predict_neural(fit.model, family, te, store, device, int(config["batch_size"]), fit.cnv_median, fit.cnv_mean, fit.cnv_std)
    pr = pr.rename(columns={"sample_id": "Sample", "patient_id": "Patient", "y_true": "y", "y_prob": "pred", "y_logit": "logit"})
    pr["rep"] = rep; pr["fold"] = k; pr["seed"] = seed; pr["best_epoch"] = fit.best_epoch; pr["epochs_run"] = len(fit.history)
    pr["n_train"] = len(tr); pr["n_val"] = len(va); pr["val_patients"] = len(vp); pr["val_pos_patients"] = int(va.groupby("patient_id").y_progressor.max().sum())
    pr[["Sample", "Patient", "y", "rep", "fold", "seed", "pred", "logit", "best_epoch", "epochs_run", "n_train", "n_val", "val_patients", "val_pos_patients"]].to_csv(outf + ".tmp", index=False); os.replace(outf + ".tmp", outf)
    print(f"rep {rep} fold {k} train {len(tr)} val {len(va)} ({len(vp)} patients, {pr.val_pos_patients.iloc[0]} progressors) test {len(te)} "
          f"best_epoch {fit.best_epoch} epochs {len(fit.history)} secs {time.time() - t0:.0f} device {device}", flush=True)
print("ABMIL REP DONE", rep, folds, flush=True)
