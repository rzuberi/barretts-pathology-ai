"""P32 first pass: predict P31 report fields from the H&E image (one UNI2 slide per imaged ERIN case).
For each binary target with >= 30 positives and >= 30 negatives: gated-attention ABMIL (abmil_clf.ABMIL), patient-disjoint
5-fold (seed 0), 3 seeds averaged, 25 epochs, no tuning -> OOF AUROC with patient-clustered CI. The fold models are kept
in memory and ALSO applied to the 707 SWG release slides (UNI2 npz, 256 tiles at ~0.88 um/px: a scale shift, flagged) to
impute the field on a cohort that has outcomes; the imputed grade is sanity-checked against the SWG pathologist grade.
Env: FIELDS (comma list of target names, default all), P31DIR (dir with p31_fields.csv), OUTDIR."""
import json, os, sys, glob, numpy as np, pandas as pd, torch, torch.nn as nn, h5py
from sklearn.metrics import roc_auc_score
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; sys.path.insert(0, T + "/scripts"); from abmil_clf import ABMIL, patient_folds
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
OUT = os.environ.get("OUTDIR", "."); P31 = os.environ["P31DIR"]; FOLD_SEED = int(os.environ.get("FOLD_SEED", "0")); PERM_SEED = os.environ.get("PERM_SEED"); DEV = "cuda" if torch.cuda.is_available() else "cpu"; SEEDS = [0, 1, 2]; EPOCHS = 25; MAXT = 1500
f = pd.read_csv(P31 + "/p31_fields.csv", dtype=str); m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("CaseName")
d = m.merge(f, on="CaseName")
EXCL = os.environ.get("EXCLUDE_PATIENTS")   # file of ERIN anon_ids to drop from training (e.g. patients also in SWG)
if EXCL:
    ex = set(l.strip() for l in open(EXCL) if l.strip()); n0 = len(d); d = d[~d.anon_id.isin(ex)].reset_index(drop=True); print(f"EXCLUDE_PATIENTS: dropped {n0-len(d)} cases from {len(ex)} listed patients", flush=True)
print("cases with fields + slide:", len(d), flush=True)
TARGETS = {"grade_LGDplus": (d.grade.isin(["LGD", "HGD", "CANCER"]), d.grade.isin(["NDBE", "IND", "LGD", "HGD", "CANCER"])),
           "grade_HGDplus": (d.grade.isin(["HGD", "CANCER"]), d.grade.isin(["NDBE", "IND", "LGD", "HGD", "CANCER"])),
           "im_present": (d.intestinal_metaplasia.eq("present"), d.intestinal_metaplasia.isin(["present", "absent"])),
           "goblet_present": (d.goblet_cells.eq("present"), d.goblet_cells.isin(["present", "absent"])),
           "inflammation_mod_severe": (d.inflammation.isin(["moderate", "severe"]), d.inflammation.isin(["none", "mild", "moderate", "severe"])),
           "inflammation_any": (d.inflammation.isin(["mild", "moderate", "severe"]), d.inflammation.isin(["none", "mild", "moderate", "severe"])),
           "ulceration": (d.ulceration_or_erosion.eq("yes"), d.ulceration_or_erosion.isin(["yes", "no"])),
           "squamous_only": (d.squamous_only.eq("yes"), d.squamous_only.isin(["yes", "no"])),
           "gastric_present": (d.gastric_mucosa_present.eq("yes"), d.gastric_mucosa_present.isin(["yes", "no"])),
           "treatment_effect": (d.treatment_effect.eq("yes"), d.treatment_effect.isin(["yes", "no"])),
           "p53_abnormal": (d.p53.eq("abnormal"), d.p53.isin(["abnormal", "normal"])),
           "certainty_not_definite": (d.diagnostic_certainty.isin(["probable", "uncertain"]), d.diagnostic_certainty.isin(["definite", "probable", "uncertain"])),
           "site_goj_or_stomach": (d.site.isin(["goj", "stomach", "mixed"]), d.site.isin(["oesophagus", "goj", "stomach", "mixed"])),
           "specimen_resection": (d.specimen_type.eq("resection"), d.specimen_type.isin(["biopsy", "emr_esd", "resection"]))}
want = [t for t in (os.environ.get("FIELDS", "").split(",") if os.environ.get("FIELDS") else TARGETS) if t in TARGETS]
print("loading bags", flush=True); rs = np.random.RandomState(0); bags = {}
for k, p in zip(d.CaseName, d.h5):
    with h5py.File(p) as h: X = np.asarray(h["features"], np.float16)
    bags[k] = X[rs.choice(len(X), MAXT, replace=False)] if len(X) > MAXT else X
ui = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID"); man = pd.read_csv(F + "/training_manifest.csv", dtype=str)
SWG_FEATS = os.environ.get("SWG_FEATS")   # dir of <slide_basename_stem>.h5 at 0.5 um/px (P32 pass 2); default = release npz (0.88 um/px)
if SWG_FEATS:
    swg = {}
    for sid, b in zip(ui.sample_id, ui.image_basename):
        h5 = os.path.join(SWG_FEATS, os.path.splitext(b)[0] + ".h5")
        if os.path.exists(h5):
            with h5py.File(h5) as h: X = np.asarray(h["features"], np.float16)
            swg[sid] = X[rs.choice(len(X), MAXT, replace=False)] if len(X) > MAXT else X
    res_note = f"SWG bags from {SWG_FEATS} (0.5 um/px re-extraction)"
else:
    swg = {s: np.load(p, allow_pickle=True)["embeddings"].astype(np.float16) for s, p in zip(ui.sample_id, ui.npz_path)}; res_note = "SWG bags = release npz, 256 tiles at ~0.88 um/px (scale shift)"
print("SWG bags", len(swg), res_note, flush=True)
def train(keys_tr, y, seed):
    rng = np.random.RandomState(seed); torch.manual_seed(seed); model = ABMIL(d_in=1536).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    pos = sum(y[k] for k in keys_tr); w = (len(keys_tr) - pos) / max(pos, 1); lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(w), device=DEV))
    def bag(k):
        X = bags[k]; idx = rng.choice(len(X), min(800, len(X)), replace=False); return torch.tensor(X[idx], dtype=torch.float32, device=DEV)
    for ep in range(EPOCHS):
        order = rng.permutation(keys_tr)
        for i in range(0, len(order), 32):
            chunk = order[i:i + 32]; logits = torch.stack([model(bag(k))[0] for k in chunk]); yy = torch.tensor([float(y[k]) for k in chunk], device=DEV)
            loss = lossf(logits.view(-1), yy); opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()
def predict(model, X):
    with torch.no_grad(): return float(torch.sigmoid(model(torch.tensor(np.asarray(X), dtype=torch.float32, device=DEV))[0]).item())
def ci(v): return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
res = {"_meta": {"n_cases": int(len(d)), "epochs": EPOCHS, "seeds": SEEDS, "folds": f"patient_folds seed {FOLD_SEED}", "perm_seed": PERM_SEED, "excluded_patients_file": os.environ.get("EXCLUDE_PATIENTS"), "swg_bags": res_note}, "fields": {}}
swg_imp = pd.DataFrame(index=sorted(swg))
for t in want:
    pos, ok = TARGETS[t]; sub = d[ok.values].copy(); yv = pos[ok.values].astype(int).values
    if yv.sum() < 30 or (1 - yv).sum() < 30: res["fields"][t] = {"skipped": f"pos {int(yv.sum())} neg {int((1-yv).sum())}"}; print(t, "skipped", flush=True); continue
    keys = list(sub.CaseName); pat = dict(zip(keys, sub.anon_id))
    if PERM_SEED is not None: yv = np.random.RandomState(int(PERM_SEED)).permutation(yv)   # CONTROL: labels shuffled across cases -> a "random ERIN head"
    y = dict(zip(keys, yv)); folds = patient_folds(keys, pat, y, 5, seed=FOLD_SEED)
    oof = {k: [] for k in keys}; imp = {s: [] for s in swg}
    for s in SEEDS:
        for fi, te in enumerate(folds):
            tr = [k for j, fl in enumerate(folds) if j != fi for k in fl]; model = train(tr, y, s)
            for k in te: oof[k].append(predict(model, bags[k]))
            if s == 0:
                for sid in swg: imp[sid].append(predict(model, swg[sid]))
                torch.save(model.state_dict(), os.path.join(OUT, f"model_{t}_f{fi}.pt"))
            print(t, "seed", s, "fold", fi, flush=True)
    p = np.array([np.mean(oof[k]) for k in keys]); g = np.array([pat[k] for k in keys]); up = np.unique(g); idx_of = {u: np.where(g == u)[0] for u in up}; rng = np.random.RandomState(0); B = []
    while len(B) < 1000:
        s_ = np.concatenate([idx_of[u] for u in rng.choice(up, len(up))])
        if len(set(yv[s_])) < 2: continue
        B.append(roc_auc_score(yv[s_], p[s_]))
    swg_imp[t] = pd.Series({sid: np.mean(v) for sid, v in imp.items()})
    res["fields"][t] = {"n": int(len(keys)), "pos": int(yv.sum()), "auroc": round(float(roc_auc_score(yv, p)), 4), "ci": ci(B)}; print(t, res["fields"][t], flush=True)
    pd.DataFrame({"CaseName": keys, "y": yv, "p": p}).to_csv(os.path.join(OUT, f"oof_{t}.csv"), index=False)
swg_imp.to_csv(os.path.join(OUT, "swg_imputed_fields.csv"))
# sanity: imputed grade vs SWG pathologist grade
if "grade_LGDplus" in swg_imp:
    lab = pd.to_numeric(coh.Label.reindex(swg_imp.index), errors="coerce").fillna(0).values; ok = ~np.isnan(swg_imp.grade_LGDplus.values)
    res["swg_imputed_grade_vs_pathologist_LGDplus_auroc"] = round(float(roc_auc_score((lab[ok] >= 2).astype(int), swg_imp.grade_LGDplus.values[ok])), 4) if len(set(lab[ok] >= 2)) > 1 else None
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1); print(json.dumps(res, indent=None))
