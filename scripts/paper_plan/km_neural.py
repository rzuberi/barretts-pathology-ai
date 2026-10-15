"""Killcoyne protocol with imaging, neural tier (docs/paper_plan_killcoyne_multimodal.md @ 3904c44). Release classes and release fit_neural /
predict_neural imported unchanged; fixed release hyperparameters; LOPO over the 80 patients of set C; validation (early stopping only) =
group 1 of the r = 1 patient shuffle from km_linear.R; label = sheet Status per sample. Env: MODEL in {img, early_their, inter_their,
early_pkg, inter_pkg}, SEED, CHUNK_ID, N_CHUNKS. Rows -> feasibility/paper_plan/killcoyne_mm/neural/."""
import os, sys, time, numpy as np, pandas as pd, torch
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/multimodal-barretts-progression"; sys.path.insert(0, B + "/src")
from barrett.training.data import CanonicalFeatureStore
from barrett.training.loops import fit_neural, predict_neural
M = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm"
CFG = {"img": ("image_only", None, dict(hidden_dim=256, attn_dim=128, dropout=0.1, lr=1e-4, weight_decay=0.01, batch_size=8, max_epochs=20, patience=5)),
       "early": ("early_fusion", None, dict(hidden_dim=512, dropout=0.2, lr=1e-4, weight_decay=0.01, batch_size=8, max_epochs=20, patience=5)),
       "inter": ("intermediate_fusion", None, dict(img_hidden=256, cnv_hidden=128, attn_dim=128, fusion_hidden=256, dropout=0.2, lr=1e-4, weight_decay=0.01, batch_size=8, max_epochs=20, patience=5))}
model = os.environ["MODEL"]; seed = int(os.environ.get("SEED", "0")); ch = int(os.environ.get("CHUNK_ID", "0")); nch = int(os.environ.get("N_CHUNKS", "1"))
kind, src = (model.split("_") + ["their"])[:2]; family, _, config = CFG[kind]
os.makedirs(M + "/neural", exist_ok=True); outf = f"{M}/neural/{model}_s{seed}_c{ch:02d}.csv"
if os.path.exists(outf): print("exists", outf); sys.exit(0)
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2"))); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C = pd.read_csv(M + "/set_C.csv", dtype=str); frame = pd.DataFrame({"sample_id": C.Sample, "patient_id": C.Patient, "y_progressor": (C.Status == "P").astype(int)})
cnv = pd.read_csv(f"{M}/cnv_{src}_C.csv", dtype={"sample_id": str}); feats = [c for c in cnv.columns if c != "sample_id"]
store = CanonicalFeatureStore(pd.read_csv(M + "/bag_index.csv", dtype=str), cnv, feats)
val = pd.read_csv(M + "/neural_val_patients.csv", dtype=str)
pats = sorted(frame.patient_id.unique(), key=lambda s: (len(s), s)); mine = [p for i, p in enumerate(pats) if i % nch == ch]; out = []
for p in mine:
    t0 = time.time(); vp = set(val[val.left_out == p].val_patient); te = frame[frame.patient_id == p]; va = frame[frame.patient_id.isin(vp)]; tr = frame[~frame.patient_id.isin(vp | {p})]
    fit = fit_neural(family, tr, va, store, config, device, seed)
    pr = predict_neural(fit.model, family, te, store, device, int(config["batch_size"]), fit.cnv_median, fit.cnv_mean, fit.cnv_std)
    pr["model"] = model; pr["seed"] = seed; pr["best_epoch"] = fit.best_epoch; out.append(pr)
    print(f"patient {p} train {len(tr)} val {len(va)} test {len(te)} best_epoch {fit.best_epoch} secs {time.time() - t0:.0f} device {device}", flush=True)
pd.concat(out).to_csv(outf, index=False); print("NEURAL CHUNK DONE", model, seed, ch, flush=True)
