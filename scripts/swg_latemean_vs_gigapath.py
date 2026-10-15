"""1.13b residual contrast: uni2 late-mean (headline) vs gigapath histology-only.

GigaPath image-only is the best unimodal arm by AUPRC point estimate (0.6093) in
multiencoder_metrics.csv, but the released paired tables never contrast the
headline fusion against it. Same methodology as the release: patient-level max
aggregation, paired bootstrap over patients, percentile CIs, 5,000 resamples.

REPRODUCTION GATE: aborts unless it reproduces the released numbers for both
arms (uni2_late_mean 0.6296/0.7742/0.1842; gigapath_image 0.6093/0.7332) to
±0.002 — if the gate fails, the join is wrong and no new number is emitted.

Run on cluster login/epyc node (CPU, seconds). Output: JSON next to OUTDIR.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

H = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_scientific_hardening_20260727"
OUT = os.environ.get("OUTDIR", H)
N_BOOT, SEED = 5000, 20260825

# headline late-mean OOF: frozen column in the hardening multimodal OOF table
oof = pd.read_csv(os.path.join(H, "clinical_multimodal_oof.csv"))
assert {"patient_id", "outer_fold", "y_true", "cnv_image_latemean_frozen"} <= set(oof.columns)

# gigapath image-only OOF: per-fold outer test predictions
gp_parts = []
for f in sorted(glob.glob(os.path.join(H, "gigapath_training/image_only/fold*/outer_test_predictions.csv"))):
    d = pd.read_csv(f)
    d["outer_fold"] = int(os.path.basename(os.path.dirname(f)).replace("fold", ""))
    gp_parts.append(d)
gp = pd.concat(gp_parts, ignore_index=True)
prob_col = next(c for c in ("y_prob", "probability", "prob", "y_pred_prob") if c in gp.columns)
pid_col = next(c for c in ("patient_id", "PatientID", "patient") if c in gp.columns)
true_col = next(c for c in ("y_true", "label", "target") if c in gp.columns)


def patient_max(frame, pid, prob, true):
    return frame.groupby([pid], as_index=False).agg(y_true=(true, "max"), y_prob=(prob, "max"))


lm = patient_max(oof, "patient_id", "cnv_image_latemean_frozen", "y_true")
gi = patient_max(gp, pid_col, prob_col, true_col)
m = lm.merge(gi, on="patient_id" if pid_col == "patient_id" else pid_col,
             suffixes=("_lm", "_gp"))
assert len(m) == 150, f"expected 150 patients, got {len(m)}"
assert (m["y_true_lm"] == m["y_true_gp"]).all(), "label mismatch across arms"
y = m["y_true_lm"].to_numpy()
p_lm, p_gp = m["y_prob_lm"].to_numpy(), m["y_prob_gp"].to_numpy()


def metrics(y, p):
    return (average_precision_score(y, p), roc_auc_score(y, p), brier_score_loss(y, p))


rep_lm, rep_gp = metrics(y, p_lm), metrics(y, p_gp)
released = {"lm": (0.6296, 0.7742, 0.1842), "gp": (0.6093, 0.7332, None)}
for name, rep, rel in (("uni2_late_mean", rep_lm, released["lm"]),
                       ("gigapath_image", rep_gp, released["gp"])):
    for got, want in zip(rep, rel):
        if want is not None and abs(got - want) > 0.002:
            raise SystemExit(f"REPRODUCTION GATE FAILED {name}: got {rep}, released {rel}")
print(f"gate passed: lm={rep_lm} gp={rep_gp}")

rng = np.random.default_rng(SEED)
deltas = {"auprc": [], "roc": [], "brier": []}
n = len(y)
for _ in range(N_BOOT):
    idx = rng.integers(0, n, n)
    if len(np.unique(y[idx])) < 2:
        continue
    a, b = metrics(y[idx], p_lm[idx]), metrics(y[idx], p_gp[idx])
    deltas["auprc"].append(a[0] - b[0])
    deltas["roc"].append(a[1] - b[1])
    deltas["brier"].append(a[2] - b[2])

point = {k: v for k, v in zip(("auprc", "roc", "brier"),
                              (rep_lm[0] - rep_gp[0], rep_lm[1] - rep_gp[1], rep_lm[2] - rep_gp[2]))}
res = {"contrast": "uni2_late_mean(headline) - gigapath_image", "n_patients": n,
       "n_boot_valid": len(deltas["auprc"]), "seed": SEED,
       "reproduced": {"uni2_late_mean": rep_lm, "gigapath_image": rep_gp}}
for k, d in deltas.items():
    d = np.asarray(d)
    res[k] = {"delta": point[k], "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
              "sign_prob": float((np.sign(d) != np.sign(point[k])).mean())}
path = os.path.join(OUT, "latemean_vs_gigapath_paired.json")
json.dump(res, open(path, "w"), indent=2)
print(json.dumps(res, indent=2))
print(f"wrote {path}")
