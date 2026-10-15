"""2.40 (Astra A14): cross-cohort identity crosswalk + split-hygiene audit.

ERIN, SWG and the Barrett's-DB export all carry Addenbrooke's pathology
accession numbers (ERIN CaseName 'PSyy-nnnnn', SWG BiopsyID_real 'psyy.nnnn',
DB specimennumber 'PSyy.nnnnn'). Normalise and intersect them to find shared
specimens and — via ERIN anon_id / SWG PatientID_real / DB participant_id —
shared PATIENTS across every training and evaluation set that is treated as
independent. Also re-derives the VLM train/val/test split and asserts
patient-disjointness, and checks the label-development samples (adjudications,
hand-label key) against the VLM test fold. Writes swg_overlap_patients.txt for
the VLM-SWG re-evaluation (excluding overlapping patients).
"""
import json, os, re
import pandas as pd, numpy as np

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
EXP = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
OUT = os.environ.get("OUTDIR", ".")

def acc_norm(s):
    """PSyy-nnnnn, PSyy.nnnnn, psyy.nnnn -> ('PS', yy, int(n)); None if not an accession."""
    m = re.match(r"^\s*([A-Za-z]{1,3})\s*(\d{2})\s*[-./]?\s*(\d{3,6})\s*$", str(s))
    return (m.group(1).upper(), m.group(2), int(m.group(3))) if m else None

# ---- ERIN: reports + slides + labels ----
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str)
rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv",
                  dtype=str, usecols=["anon_id", "CaseName"])
rep["acc"] = rep["CaseName"].map(acc_norm)
erin_acc2pat = {a: p for a, p in zip(rep["acc"], rep["anon_id"]) if a}
erin_pats = set(rep["anon_id"])
# ---- SWG release ----
man = pd.read_csv(F + "/training_manifest.csv", dtype=str)
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).merge(man, left_on="SampleID", right_on="sample_id")
coh["acc"] = coh["BiopsyID_real"].map(acc_norm)
swg_pats = set(coh["patient_id"])
swg_acc2pat = {a: p for a, p in zip(coh["acc"], coh["patient_id"]) if a}
# ---- DB export (bridge: accession -> participant) ----
db = pd.read_csv(EXP + "/pathology_text_normalised_full.csv", dtype=str,
                 usecols=["specimennumber", "participant_id"]).dropna()
db["acc"] = db["specimennumber"].map(acc_norm)
db = db[db["acc"].notna()]
acc2part = dict(zip(db["acc"], db["participant_id"]))
part2accs = db.groupby("participant_id")["acc"].agg(set).to_dict()

res = {"_meta": {"erin_reports": len(rep), "erin_patients": len(erin_pats),
                 "erin_acc_parsed": int(rep["acc"].notna().sum()),
                 "swg_samples": len(coh), "swg_patients": len(swg_pats),
                 "swg_acc_parsed": int(coh["acc"].notna().sum()),
                 "db_rows_with_acc": len(db), "db_participants": len(part2accs)}}

# 1. direct specimen overlap ERIN <-> SWG
shared_acc = set(erin_acc2pat) & set(swg_acc2pat)
direct_pairs = {(erin_acc2pat[a], swg_acc2pat[a]) for a in shared_acc}
# 2. bridged through DB participant: ERIN patient -> participant(s) -> any SWG accession
erin_pat2parts = {}
for a, p in erin_acc2pat.items():
    if a in acc2part: erin_pat2parts.setdefault(p, set()).add(acc2part[a])
swg_pat2parts = {}
for a, p in swg_acc2pat.items():
    if a in acc2part: swg_pat2parts.setdefault(p, set()).add(acc2part[a])
part2swg = {}
for p, parts in swg_pat2parts.items():
    for q in parts: part2swg.setdefault(q, set()).add(p)
bridged_pairs = set()
for ep, parts in erin_pat2parts.items():
    for q in parts:
        for sp in part2swg.get(q, ()):
            bridged_pairs.add((ep, sp))
all_pairs = direct_pairs | bridged_pairs
erin_overlap = {e for e, _ in all_pairs}; swg_overlap = {s for _, s in all_pairs}
res["erin_swg_overlap"] = {
    "shared_accessions_direct": len(shared_acc),
    "patient_pairs_direct": len(direct_pairs),
    "patient_pairs_bridged_via_db": len(bridged_pairs),
    "erin_patients_in_swg": len(erin_overlap),
    "swg_patients_in_erin": len(swg_overlap),
    "frac_swg_patients_in_erin": round(len(swg_overlap) / max(len(swg_pats), 1), 4),
    "erin_db_participant_match_rate": round(len(erin_pat2parts) / max(len(erin_pats), 1), 4),
    "swg_db_participant_match_rate": round(len(swg_pat2parts) / max(len(swg_pats), 1), 4),
    "multi_participant_erin_patients": int(sum(len(v) > 1 for v in erin_pat2parts.values())),
}
open(os.path.join(OUT, "swg_overlap_patients.txt"), "w").write("\n".join(sorted(swg_overlap)))

# 3. VLM split hygiene (re-derive exactly as task_vlm_pretrain.py does)
mm = m.dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
elig = mm[mm["label_status"].isin(["train_eligible", "adjudicated"])]
pats = elig["anon_id"].values
uniq = sorted(set(pats)); rng = np.random.RandomState(0)
fold_of = {a: i % 5 for i, a in enumerate(rng.permutation(uniq))}
fm = np.array([fold_of[a] for a in pats])
split_pats = {"train": set(pats[fm <= 2]), "val": set(pats[fm == 3]), "test": set(pats[fm == 4])}
split_cases = {k: set(elig["CaseName"].values[np.isin(pats, list(v))]) for k, v in split_pats.items()}
vlm_test_swg = {e for e in split_pats["test"]} & erin_overlap
vlm_train_swg = {e for e in split_pats["train"] | split_pats["val"]} & erin_overlap
res["vlm_split"] = {
    "n_slides": {"train": int((fm <= 2).sum()), "val": int((fm == 3).sum()),
                 "test": int((fm == 4).sum())},
    "patient_disjoint": bool(not (split_pats["train"] & split_pats["val"]) and
                             not (split_pats["train"] & split_pats["test"]) and
                             not (split_pats["val"] & split_pats["test"])),
    "report_disjoint": bool(not (split_cases["train"] & split_cases["test"]) and
                            not (split_cases["val"] & split_cases["test"])),
    "duplicate_h5_rows_in_master": int(m["h5"].duplicated().sum()),
    "vlm_trainval_patients_also_in_swg": len(vlm_train_swg),
    "vlm_test_patients_also_in_swg": len(vlm_test_swg),
}
# 4. label-development samples vs VLM test fold
def cases_of(path, col="CaseName"):
    return set(pd.read_csv(path, dtype=str)[col].dropna()) if os.path.exists(path) else set()
adj = cases_of(T + "/labeller/adjudications.csv")
hand = cases_of(T + "/labeller/handlabel_sample_key.csv")
res["label_dev_vs_vlm_test"] = {
    "adjudicated_cases": len(adj), "adjudicated_in_vlm_test": len(adj & split_cases["test"]),
    "handlabel_cases": len(hand), "handlabel_in_vlm_test": len(hand & split_cases["test"]),
    "note": "adjudicated cases carry human-checked labels; their presence in test is a labelling "
            "provenance note, not leakage of model training data"}
# 5. duplicate report text across patients (same CaseName under two anon_ids)
dup = rep.groupby("CaseName")["anon_id"].nunique()
res["report_integrity"] = {"casenames_with_multiple_anon_ids": int((dup > 1).sum()),
                           "anon_ids_per_casename_max": int(dup.max())}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
