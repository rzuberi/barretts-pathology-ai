"""Check: does the SWG-trained image arm itself improve on the 0.5 um/px re-extraction (median 1,224 tiles vs the release's
256 level-2 tiles)? If yes, "ERIN grade head (0.756) beats the SWG image arm (0.731)" is partly a feature-quality artefact.
ABMIL on SWG 0.5um bags, release folds (fold_id_rep01), 3 seeds, 25 epochs; OOF saved. Env: OUTDIR."""
import glob, json, os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, h5py
from sklearn.metrics import roc_auc_score
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; sys.path.insert(0, T + "/scripts"); from abmil_clf import ABMIL
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
FE = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort/features_uni2h_05um"; OUT = os.environ.get("OUTDIR", "."); DEV = "cuda" if torch.cuda.is_available() else "cpu"
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); ui = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str).set_index("sample_id")
rs = np.random.RandomState(0); bags = {}
for s in man.sample_id:
    p = os.path.join(FE, os.path.splitext(ui.image_basename[s])[0] + ".h5")
    with h5py.File(p) as h: X = np.asarray(h["features"], np.float16)
    bags[s] = X[rs.choice(len(X), 1500, replace=False)] if len(X) > 1500 else X
y = dict(zip(man.sample_id, man.y_progressor.astype(int))); fold = dict(zip(man.sample_id, man.fold_id_rep01.astype(int))); keys = list(man.sample_id)
def train(tr, seed):
    rng = np.random.RandomState(seed); torch.manual_seed(seed); m = ABMIL(d_in=1536).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-5)
    pos = sum(y[k] for k in tr); w = (len(tr) - pos) / max(pos, 1); lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(w), device=DEV))
    bag = lambda k: torch.tensor(bags[k][rng.choice(len(bags[k]), min(800, len(bags[k])), replace=False)], dtype=torch.float32, device=DEV)
    for ep in range(25):
        order = rng.permutation(tr)
        for i in range(0, len(order), 32):
            ch = order[i:i + 32]; loss = lossf(torch.stack([m(bag(k))[0] for k in ch]).view(-1), torch.tensor([float(y[k]) for k in ch], device=DEV)); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()
oof = {k: [] for k in keys}
for seed in (0, 1, 2):
    for f in range(1, 6):
        te = [k for k in keys if fold[k] == f]; tr = [k for k in keys if fold[k] != f]; m = train(tr, seed)
        with torch.no_grad():
            for k in te: oof[k].append(float(torch.sigmoid(m(torch.tensor(bags[k], dtype=torch.float32, device=DEV))[0]).item()))
        print("seed", seed, "fold", f, flush=True)
d = man.assign(p=[np.mean(oof[k]) for k in keys], y=[y[k] for k in keys]); d[["sample_id", "patient_id", "fold_id_rep01", "y", "p"]].to_csv(os.path.join(OUT, "swg_img05_oof.csv"), index=False)
g = d.groupby("patient_id"); yp = g.y.max().values; pp = g.p.max().values
rel = pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(F + "/training_final_nested_cv_v1/image_only/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(d.sample_id).values
d["rel"] = rel; pr = d.groupby("patient_id").rel.max().values
rng = np.random.RandomState(0); B = [rng.choice(len(yp), len(yp)) for _ in range(2000)]; B = [s for s in B if len(set(yp[s])) > 1]
ci = lambda a, b=None: [round(float(np.percentile([roc_auc_score(yp[s], a[s]) - (roc_auc_score(yp[s], b[s]) if b is not None else 0) for s in B], q)), 4) for q in (2.5, 97.5)]
res = {"n_patients": int(len(yp)), "pos": int(yp.sum()), "image_05um_auroc": round(float(roc_auc_score(yp, pp)), 4), "ci": ci(pp), "release_image_auroc": round(float(roc_auc_score(yp, pr)), 4), "delta_05um_minus_release": ci(pp, pr), "sample_level_05um": round(float(roc_auc_score(d.y, d.p)), 4)}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1); print(json.dumps(res))
