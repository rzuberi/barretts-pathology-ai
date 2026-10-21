"""Exploration E5 (docs/paper_explore_E5.md): borrow knowledge from ERIN. Step 0 (CPU): (i) which pre-event slides of the analysed cohort already have
0.5 um/px-class UNI2-h features (SWGCohort/features_uni2h_05um, the ERIN UNI2-h pipeline); manifest of the rest for the two-phase extraction;
(ii) the ERIN patients to exclude from ERIN training = union of feasibility/erin_fusion/erin_swg_overlap_anon_ids.txt, the ISBI part-3 bridge
(release accessions + Barrett's-DB participant bridge) and the accessions of every sheet row / slide_matching row (all SWG patients, not only the release).
Row-level outputs -> killcoyne_mm/explore_E5/ (cluster only); prints aggregates only."""
import os, re, json, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; E5 = M + "/explore_E5"; os.makedirs(E5, exist_ok=True)
SWG = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; FE = SWG + "/features_uni2h_05um"; NEWF = E5 + "/features_uni2h_05um"
REL = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
def acc_norm(s):   # isbi_rerun/code/part3_swg.py, unchanged
    s = re.sub(r"[\s_][A-Za-z]{1,2}\d{0,2}\s*(\(.*\))?\s*$", "", str(s).strip()); m = re.match(r"^\s*([A-Za-z]{1,3})\s*(\d{2})\s*[-./]?\s*(\d{3,6})\s*$", s)
    return f"{m.group(1).upper()}{m.group(2)}-{int(m.group(3)):05d}" if m else None
SM = pd.read_csv(M + "/horizons/samples.csv", dtype=str); pre = SM.pre.astype(str).isin(["1", "True", "1.0"]).values
mp = pd.read_csv(T + "/feasibility/paper_plan/killcoyne/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(SM.Sample[pre])].drop_duplicates(["cnv", "base"]).copy()
mp["stem"] = mp.base.map(lambda b: os.path.splitext(b)[0]); have = set(os.path.splitext(f)[0] for f in os.listdir(FE) if f.endswith(".h5"))
mp["h5_shared"] = mp.stem.isin(have); mp["slide"] = mp.base.map(lambda b: f"{SWG}/slides/{b}"); mp["slide_exists"] = mp.slide.map(os.path.exists)
mp["h5"] = np.where(mp.h5_shared, FE + "/" + mp.stem + ".h5", NEWF + "/" + mp.stem + ".h5")
need = mp[~mp.h5_shared]; open(E5 + "/manifest_missing.txt", "w").write("\n".join(sorted(set(need[need.slide_exists].slide))) + "\n")
mp[["cnv", "base", "h5", "h5_shared", "slide", "slide_exists"]].to_csv(E5 + "/sample_slides.csv", index=False)
info = {"pre_event_samples": int(pre.sum()), "pre_event_slide_rows": int(len(mp)), "samples_with_a_slide_row": int(mp.cnv.nunique()),
        "slides_with_shared_05um_features": int(mp.h5_shared.sum()), "slides_to_extract": int((~mp.h5_shared).sum()),
        "slides_to_extract_on_disk": int(need.slide_exists.sum()), "samples_fully_covered_after_extraction": int(mp[mp.h5_shared | mp.slide_exists].cnv.nunique())}
# ---------------- ERIN patients to exclude
ex = set(l.strip() for l in open(T + "/feasibility/erin_fusion/erin_swg_overlap_anon_ids.txt") if l.strip()); info["exclude_list_file"] = len(ex)
rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv", dtype=str, low_memory=False).drop_duplicates("CaseName")
racc = pd.Series(rep.anon_id.values, index=rep.CaseName.map(acc_norm)); racc = racc[racc.index.notna()]; racc = racc[~racc.index.duplicated()]
coh = pd.read_csv(REL + "/pre_event_cohort.csv", dtype=str)
fd = pd.read_csv(SWG + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]
smf = pd.read_csv(SWG + "/slide_matching.csv", dtype=str)
accs = set(coh.BiopsyID_real.dropna().map(acc_norm)) | set(fd["Path ID"].dropna().map(acc_norm)) | set(smf.PathCaseID.dropna().map(acc_norm)); accs.discard(None)
b1 = set(racc.reindex(sorted(accs)).dropna())
pid = set(coh.participant_id.dropna().map(lambda v: re.sub(r"\.0$", "", str(v))))
db = pd.read_csv("/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export/pathology_text_normalised_full.csv", dtype=str, usecols=["participant_id", "specimennumber"])
db["acc"] = db.specimennumber.map(acc_norm); db["pid"] = db.participant_id.map(lambda v: re.sub(r"\.0$", "", str(v)) if pd.notna(v) else None)
b2 = set(racc.reindex(db[db.pid.isin(pid)].acc.dropna()).dropna())
pairs = pd.read_csv(T + "/feasibility/closeout/erin_swg_pairs.csv", dtype=str); b3 = set(pairs.erin_anon_id.dropna())
allx = ex | b1 | b2 | b3; open(E5 + "/erin_exclude_anon_ids.txt", "w").write("\n".join(sorted(allx)) + "\n")
info.update({"exclude_accession_bridge": len(b1), "exclude_db_bridge": len(b2), "exclude_pairs_file": len(b3), "exclude_union": len(allx), "union_minus_file": len(allx - ex)})
json.dump(info, open(E5 + "/prep_info.json", "w"), indent=1); print(json.dumps(info, indent=1)); print("E5 PREP DONE")
