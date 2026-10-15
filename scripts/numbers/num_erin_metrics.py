"""NUMBERS E34: AUROC/AUPRC/F1 with patient-clustered CIs for both supervision schemes (case-max vs
section-resolved), binary (2.38) and six-class (2.38b), from the saved unit predictions."""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, balanced_accuracy_score
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
POS = {"LGD", "HGD", "CANCER"}; CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLASSES)}
lab = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str); m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
lab = lab.merge(m[["h5", "anon_id"]], on="h5").set_index("h5"); pat_of = lab.anon_id.to_dict()
def load(d, field):
    acc = {"case": {}, "slide": {}}
    for f in glob.glob(d + "/*.npz"):
        arm = os.path.basename(f).split("_")[0]; z = np.load(f, allow_pickle=True)
        for k, p in zip(z["keys"], z[field]): acc[arm].setdefault(str(k), []).append(p)
    return acc
def cboot(fn, pats, n):
    up = sorted(set(pats)); idx = {}; 
    for i, p in enumerate(pats): idx.setdefault(p, []).append(i)
    rng = np.random.RandomState(0); out = []
    for _ in range(NB):
        s = np.concatenate([idx[up[j]] for j in rng.randint(0, len(up), len(up))]); v = fn(s)
        if v is not None: out.append(v)
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]
res = {"task_binary": "NDBE/IND/normal vs LGD+ (LGD, HGD, CANCER) on the slide's OWN section grade (truth) ; models trained on case-max vs section label",
       "task_sixclass": "NORMAL_OTHER<NDBE<IND<LGD<HGD<CANCER, same truth", "split": "5-fold patient-disjoint, frozen (patient hash seed 0), 3 seeds averaged, 1,538 slides / 1,155 patients"}
acc = load(T + "/feasibility/svc_units", "preds"); keys = sorted(set(acc["case"]) & set(acc["slide"]) & set(lab.index))
y = np.array([lab.loc[k, "worst_grade"] in POS for k in keys], int); pats = [pat_of[k] for k in keys]
res["binary"] = {"n_slides": len(keys), "n_pos": int(y.sum()), "arms": {}}
for arm in ("case", "slide"):
    p = np.array([np.mean(acc[arm][k]) for k in keys]); fpr, tpr, thr = roc_curve(y, p); j = np.argmax(tpr - fpr); t = thr[j]
    res["binary"]["arms"][f"trained_on_{arm}_labels"] = {
        "auroc": round(float(roc_auc_score(y, p)), 4), "auroc_ci": cboot(lambda s: roc_auc_score(y[s], p[s]) if len(set(y[s])) > 1 else None, pats, len(keys)),
        "auprc": round(float(average_precision_score(y, p)), 4), "auprc_ci": cboot(lambda s: average_precision_score(y[s], p[s]) if len(set(y[s])) > 1 else None, pats, len(keys)),
        "f1_at_0.5": round(float(f1_score(y, p >= 0.5)), 4), "f1_at_youden": round(float(f1_score(y, p >= t)), 4), "youden_threshold": round(float(t), 4),
        "sens_spec_at_youden": [round(float(tpr[j]), 4), round(float(1 - fpr[j]), 4)],
        "spec_at_sens_0.95": round(float(max(1 - fpr[tpr >= 0.95 - 1e-9])), 4), "spec_at_sens_1.0": round(float(max(1 - fpr[tpr >= 1.0 - 1e-9])), 4),
        "f1_at_sens_0.95": round(float(f1_score(y, p >= thr[np.where(tpr >= 0.95 - 1e-9)[0][np.argmax(1 - fpr[tpr >= 0.95 - 1e-9])]])), 4)}
acc5 = load(T + "/feasibility/svc5_units", "probs"); keys5 = sorted(set(acc5["case"]) & set(acc5["slide"]) & set(lab.index))
yt = np.array([C_OF[lab.loc[k, "worst_grade"]] for k in keys5]); pats5 = [pat_of[k] for k in keys5]
def macro_auc(yy, P): return float(np.mean([roc_auc_score((yy == c).astype(int), P[:, c]) for c in range(6) if 0 < (yy == c).sum() < len(yy)]))
res["sixclass"] = {"n_slides": len(keys5), "class_counts": {CLASSES[i]: int((yt == i).sum()) for i in range(6)}, "arms": {}}
for arm in ("case", "slide"):
    P = np.stack([np.mean(acc5[arm][k], axis=0) for k in keys5]); pred = P.argmax(1)
    res["sixclass"]["arms"][f"trained_on_{arm}_labels"] = {
        "macro_auroc_ovr": round(macro_auc(yt, P), 4), "macro_auroc_ci": cboot(lambda s: macro_auc(yt[s], P[s]) if len(set(yt[s])) > 2 else None, pats5, len(keys5)),
        "macro_auprc_ovr": round(float(np.mean([average_precision_score((yt == c).astype(int), P[:, c]) for c in range(6) if (yt == c).sum() > 0])), 4),
        "macro_f1": round(float(f1_score(yt, pred, average="macro")), 4), "macro_f1_ci": cboot(lambda s: f1_score(yt[s], pred[s], average="macro"), pats5, len(keys5)),
        "balanced_accuracy": round(float(balanced_accuracy_score(yt, pred)), 4), "accuracy": round(float((pred == yt).mean()), 4),
        "per_class_f1": {CLASSES[i]: round(float(v), 3) for i, v in enumerate(f1_score(yt, pred, average=None, labels=list(range(6))))}}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2); print(json.dumps(res, indent=1))
