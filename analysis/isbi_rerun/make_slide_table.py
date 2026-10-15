"""Part 2 step 0a: the slide table for the ISBI rerun. READ-ONLY on sources.
Oesophageal/GOJ H&E slides with features, own-section v3 grade, vote fraction, oesophageal case-max,
patient, human-holdout flags. Writes analysis/isbi_rerun/slides.csv + folds.json (patient-disjoint 5-fold, seed 0)."""
import json, os
import numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; L = T + "/labeller"
F = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/features/20x_224px/features_uni_v2_all"
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun"; HL = "/mnt/scratche/slow/fmlab/zuberi01/analysis/erin_handlabel"
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
os.makedirs(A, exist_ok=True); os.chmod(A, 0o700)
ORD = {"NORMAL_OTHER": -1, "NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
m = pd.read_csv(L + "/erin_slide_organ_v3.csv", dtype=str).fillna("")
s = m[(m.is_he == "True") & (m.organ_final == "OESOPHAGUS")].drop(columns=["n_jurors", "grades", "worst_grade", "site_frac", "cancer_subtypes"]).copy()
s["h5"] = s.file_uuid.map(lambda u: f"{F}/{u}.h5"); s = s[s.h5.map(os.path.exists)]
sec = pd.read_csv(L + "/erin_sections_v3_consensus.csv", dtype=str).fillna("")[["CaseName", "section", "grade_votes", "n_jurors", "worst_grade", "grades", "site"]].rename(columns={"site": "sec_site"})
s = s.merge(sec, on=["CaseName", "section"], how="left").fillna("")
s["vote_frac"] = [(json.loads(g).get(w, 0) / int(n)) if g and w and n else np.nan for g, w, n in zip(s.grade_votes, s.worst_grade, s.n_jurors)]
rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("").drop_duplicates("CaseName").set_index("CaseName")
s["anon_id"] = s.CaseName.map(rep.anon_id)
cm = s[s.worst_grade != ""].groupby("CaseName").worst_grade.agg(lambda g: max(g, key=lambda x: ORD[x]))
s["case_max"] = s.CaseName.map(cm).fillna("")
k = pd.read_csv(HL + "/key_DO_NOT_OPEN_UNTIL_DONE.csv", dtype=str).fillna("")
hum_pat = set(k.report_id.map(rep.anon_id)); hum_rep = set(k.report_id); hum_sec = set(zip(k.report_id, k.section))
s["human_patient"] = s.anon_id.isin(hum_pat); s["human_report"] = s.CaseName.isin(hum_rep); s["human_section"] = [(c, x) in hum_sec for c, x in zip(s.CaseName, s.section)]
# human label per section (worst grade; from labels table via key -> item ids)
import sqlite3
db = sqlite3.connect("file:/mnt/scratche/slow/fmlab/zuberi01/hand_grading/grading.db?mode=ro", uri=True, timeout=5)
lab = pd.read_sql('select item_id, grades, ambiguous from section_labels where grader="rehan"', db)
k["item_id"] = k.case_id + "#" + k.section; lab = lab.merge(k[["item_id", "report_id", "section"]], on="item_id")
lab["hworst"] = lab.grades.map(lambda g: (lambda S: max(S, key=lambda x: ORD[x]) if S else "")(json.loads(g)))
hw = lab.drop_duplicates(["report_id", "section"]).set_index(["report_id", "section"])
s["human_grade"] = [hw.hworst.get((c, x), "") if (c, x) in hw.index else "" for c, x in zip(s.CaseName, s.section)]
s["human_ambiguous"] = [int(hw.ambiguous.get((c, x), 0)) if (c, x) in hw.index else "" for c, x in zip(s.CaseName, s.section)]
# folds: patient-disjoint, stratified by case-max LGD+ at patient level, seed 0, over NON-human patients (human patients = holdout)
rng = np.random.RandomState(0)
pats = s[~s.human_patient].groupby("anon_id").case_max.agg(lambda g: int(any(ORD.get(x, -9) >= 2 for x in g))).reset_index()
fold_of = {}
for y in [0, 1]:
    p = pats[pats.case_max == y].anon_id.tolist(); rng.shuffle(p)
    for i, a in enumerate(p): fold_of[a] = i % 5
s["fold"] = s.anon_id.map(fold_of).fillna(-1).astype(int)        # -1 = human-patient holdout
s = s.sort_values(["fold", "CaseName", "section", "file_uuid"]).reset_index(drop=True)
s["idx"] = np.arange(len(s))
cols = ["idx", "file_uuid", "h5", "CaseName", "section", "anon_id", "fold", "worst_grade", "grades", "vote_frac", "case_max", "sec_site", "human_patient", "human_report", "human_section", "human_grade", "human_ambiguous"]
s[cols].to_csv(A + "/slides.csv", index=False); os.chmod(A + "/slides.csv", 0o600)
json.dump({"seed": 0, "n_folds": 5, "fold_of_patient": fold_of, "human_holdout_patients": sorted(hum_pat & set(s.anon_id))}, open(A + "/folds.json", "w")); os.chmod(A + "/folds.json", 0o600)
summary = {"slides": len(s), "patients": int(s.anon_id.nunique()), "cases": int(s.CaseName.nunique()),
           "with_section_label": int((s.worst_grade != "").sum()), "vote_frac_ge_0.75": int((s.vote_frac >= 0.75).sum()),
           "fold_sizes": s.fold.value_counts().sort_index().to_dict(), "human_holdout_slides": int(s.human_patient.sum()),
           "human_report_slides": int(s.human_report.sum()), "T2_human_section_slides": int(s.human_section.sum()),
           "T2_with_human_grade": int((s.human_grade != "").sum()),
           "grade_dist_all": s.worst_grade.replace("", "(none)").value_counts().to_dict(),
           "grade_dist_train_pool_patient_holdout": s[~s.human_patient].worst_grade.replace("", "(none)").value_counts().to_dict(),
           "own_section_ne_case_max": int(((s.worst_grade != s.case_max) & (s.worst_grade != "")).sum())}
json.dump(summary, open(A + "/slides_summary.json", "w"), indent=1); print(json.dumps(summary, indent=1))
