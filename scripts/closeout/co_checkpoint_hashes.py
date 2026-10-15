"""Closeout item N: sha256 of every frozen model file the ACE-B plan will apply. No analysis. Writes results/closeout/aceb_checkpoint_hashes.json."""
import glob, hashlib, json, os
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", T + "/results/closeout")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
files = sorted(glob.glob(F + "/training_final_nested_cv_v1/image_only/fold*/model.pt") + glob.glob(F + "/training_final_nested_cv_v1/cnv_only/fold*/model.joblib") + glob.glob(F + "/training_final_nested_cv_v1/cnv_only/fold*/resolved_config.yaml")
              + glob.glob(F + "/training_final_nested_cv_v1/*/fold*/platt_calibrator.joblib") + glob.glob(T + "/feasibility/runs/p32_head_noov/output/model_*.pt")
              + [T + "/results/numbers/swg_operating_points_patient.json", T + "/results/numbers/swg_robustness.json", T + "/campaigns/allslides/extract_worker.py", T + "/scripts/projects/p32_fields_from_image.py", T + "/scripts/abmil_cox.py", T + "/scripts/abmil_clf.py",
                 F + "/feature_views/cnv/features_arms.csv", F + "/training_manifest.csv"])
res = {p.replace(T + "/", "thesis/").replace(F, "<release>"): {"sha256": sha(p), "bytes": os.path.getsize(p)} for p in files if os.path.exists(p)}
res["_late_mean_note"] = "late_mean has no model file: it is the plain mean of image_only and cnv_only probabilities (release construction, verified in results/numbers/swg_rep02.json)"
res["_release_dir"] = F; res["_missing"] = [p for p in files if not os.path.exists(p)]
os.makedirs(OUT, exist_ok=True); json.dump(res, open(OUT + "/aceb_checkpoint_hashes.json", "w"), indent=1); print(json.dumps(res, indent=1))
