"""Items 37-38 (24 Sep 2026): a THIRD modality for SWG - the clinical report text of the same biopsy, from the Barrett's
database (swg_matched_reports_v2), matched to release samples by accession stem. Two targets:
  (38) progression (release endpoint y_progressor, patient level = max over samples) - text arm alone, and 3-way late
       fusion text + image_only + cnv_only (fold-local z, release fold_id_rep01) on the matched subset, against 2-way
       image + cnv on the SAME subset;
  (37) pathologist grade (release Label >= 2, a label NOT derived from report text by our jury) - text arm (nomic
       embedding of the clinical report), image arm (ABMIL on the release UNI2 tiles), and their late fusion.
Embeddings: nomic-embed-text via a local ollama server (GPU). Folds: release fold_id_rep01; 3 seeds for ABMIL."""
import json, os, re, subprocess, sys, time, urllib.request, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts"); from abmil_clf import train_abmil_clf_fold
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; OUT = os.environ.get("OUTDIR", "."); JOB = os.environ.get("SLURM_JOB_ID", "0"); NB = 2000
def stem(x): x = str(x).lower().strip(); x = re.sub(r"[\s_].*$", "", x); return re.sub(r"[^a-z0-9]", "", x)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID")
man["stem"] = coh.BiopsyID_real.reindex(man.sample_id).map(stem).values; man["grade"] = pd.to_numeric(pd.Series(coh.Label.reindex(man.sample_id).values), errors="coerce").fillna(0).astype(int).values
man["y"] = man.y_progressor.astype(int); man["fold"] = man.fold_id_rep01.astype(int)
sm = pd.read_parquet(E + "/swg_matched_reports_v2.parquet"); rows = []
for r in sm.itertuples():
    for pid in str(r.swg_path_ids).split(";"):
        if pid.strip(): rows.append({"stem": stem(pid), "rid": str(r.pathology_text_id), "text": str(r.reporttext)})
rep = pd.DataFrame(rows).drop_duplicates("stem", keep="first")
d = man.merge(rep, on="stem", how="inner"); print(f"matched samples {len(d)} / {len(man)}; patients {d.patient_id.nunique()}; progressor samples {d.y.sum()}", flush=True)
# ---- embeddings (GPU ollama) ----
os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models"); port = 20000 + int(JOB) % 20000 if JOB.isdigit() else 21234
os.environ["OLLAMA_HOST"] = f"127.0.0.1:{port}"; base = f"http://127.0.0.1:{port}"; log = open(os.path.join(OUT, "ollama.log"), "w")
srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=log, stderr=log)
for _ in range(60):
    try: urllib.request.urlopen(base + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
EMB = {}
for rid, txt in d.drop_duplicates("rid")[["rid", "text"]].itertuples(index=False):
    body = json.dumps({"model": "nomic-embed-text", "prompt": txt[:6000]}).encode()
    for _ in range(3):
        try: EMB[rid] = np.asarray(json.loads(urllib.request.urlopen(urllib.request.Request(base + "/api/embeddings", data=body, headers={"Content-Type": "application/json"}), timeout=120).read())["embedding"], np.float32); break
        except Exception: time.sleep(2)
srv.terminate(); print(f"embedded {len(EMB)} reports", flush=True)
d = d[d.rid.isin(EMB)].reset_index(drop=True); X = np.stack([EMB[r] for r in d.rid])
# ---- release OOF for image and cnv on the matched samples ----
import glob
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{F}/training_final_nested_cv_v1/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob
d["img"] = oof("image_only").reindex(d.sample_id).values; d["cnv"] = oof("cnv_only").reindex(d.sample_id).values
def cv_logit(X, y, folds, C=1.0):
    p = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f; tr = ~te
        if len(set(y[tr])) < 2: p[te] = y[tr].mean(); continue
        p[te] = LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p
def zfold(v, folds):
    o = np.zeros(len(v))
    for f in np.unique(folds): te = folds == f; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
def patient(d, cols):
    g = d.groupby("patient_id"); return g.y.max().values.astype(int), {c: g[c].max().values for c in cols}
def boot_ci(y, a, b=None):
    rng = np.random.RandomState(0); n = len(y); out = []
    while len(out) < NB:
        s = rng.choice(n, n)
        if len(set(y[s])) < 2: continue
        out.append(roc_auc_score(y[s], a[s]) - (roc_auc_score(y[s], b[s]) if b is not None else 0))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]
res = {"_meta": {"matched_samples": int(len(d)), "patients": int(d.patient_id.nunique()), "progressor_samples": int(d.y.sum()), "embedding": "nomic-embed-text 768-d", "folds": "release fold_id_rep01", "n_boot": NB}}
folds = d.fold.values
# (38) progression
d["txt"] = cv_logit(X, d.y.values, folds, C=0.5)
d["fuse2"] = (zfold(d.img.values, folds) + zfold(d.cnv.values, folds)) / 2; d["fuse3"] = (zfold(d.img.values, folds) + zfold(d.cnv.values, folds) + zfold(d.txt.values, folds)) / 3; d["fuse_it"] = (zfold(d.img.values, folds) + zfold(d.txt.values, folds)) / 2
yp, P = patient(d, ["txt", "img", "cnv", "fuse2", "fuse3", "fuse_it"])
res["progression_matched_subset_patient_level"] = {"n_patients": int(len(yp)), "pos": int(yp.sum()), **{k: {"auroc": round(float(roc_auc_score(yp, v)), 4), "ci": boot_ci(yp, v)} for k, v in P.items()},
    "delta_fuse3_minus_fuse2": boot_ci(yp, P["fuse3"], P["fuse2"]), "delta_fuse3_minus_img": boot_ci(yp, P["fuse3"], P["img"]), "delta_txt_minus_img": boot_ci(yp, P["txt"], P["img"])}
res["progression_matched_subset_sample_level"] = {k: round(float(roc_auc_score(d.y, d[k])), 4) for k in ["txt", "img", "cnv", "fuse2", "fuse3"]}
# (37) pathologist grade LGD+ as target
d["yg"] = (d.grade >= 2).astype(int); print("grade target pos:", int(d.yg.sum()), flush=True)
d["txt_g"] = cv_logit(X, d.yg.values, folds, C=0.5)
uidx = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype={"sample_id": str}).set_index("sample_id")
bags = {s: np.load(uidx.npz_path[s], allow_pickle=True)["embeddings"].astype(np.float32) for s in d.sample_id}
keys = list(d.sample_id); yg = dict(zip(d.sample_id, d.yg.astype(int))); fold_of = dict(zip(d.sample_id, d.fold))
img_g = np.zeros(len(d))
for f in np.unique(folds):
    te = [k for k in keys if fold_of[k] == f]; tr = [k for k in keys if fold_of[k] != f]; acc = np.zeros(len(te))
    for seed in (0, 1, 2):
        out = train_abmil_clf_fold(bags, tr, te, yg, seed, epochs=25); pr = out[0] if isinstance(out, tuple) else out
        acc += np.array([pr[k] for k in te]) if isinstance(pr, dict) else np.asarray(pr)
    img_g[[keys.index(k) for k in te]] = acc / 3
d["img_g"] = img_g; d["cnv_g"] = cv_logit(np.zeros((len(d), 1)) + 0, d.yg.values, folds)  # placeholder (no CNV grade arm); replaced below if CNV features are available
d["fuse_g"] = (zfold(d.txt_g.values, folds) + zfold(d.img_g.values, folds)) / 2
res["pathologist_grade_LGDplus_sample_level"] = {"n": int(len(d)), "pos": int(d.yg.sum()), **{k: {"auroc": round(float(roc_auc_score(d.yg, d[k])), 4), "ci": boot_ci(d.yg.values, d[k].values)} for k in ["txt_g", "img_g", "fuse_g"]},
    "delta_fuse_minus_best_single": boot_ci(d.yg.values, d.fuse_g.values, d[["txt_g", "img_g"]].values[:, int(roc_auc_score(d.yg, d.img_g) > roc_auc_score(d.yg, d.txt_g))]),
    "jury_grade_reference": "jury two-tier agreement with the same pathologist grade is 0.82 (anchor_eval.json)"}
d[["sample_id", "patient_id", "fold", "y", "grade", "txt", "img", "cnv", "fuse3", "txt_g", "img_g"]].to_csv(os.path.join(OUT, "swg_text_arm_oof.csv"), index=False)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2); print(json.dumps(res, indent=1))
