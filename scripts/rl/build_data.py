"""Build the per-section datasets for projects 1-2 (CPU, once). READ-ONLY on sources.
Out: analysis/rl_grader/data/{train,val,test,test485,all_lowconf}.jsonl + data/summary.json
- items: one (report, section) with the v3 jury consensus worst grade and its vote fraction.
- excluded from train/val/test: every patient in the hand-label sample + SWG-shared patients (same rule as the sample builder).
- split by patient (seed 0): 80% train / 10% val / 10% jury-labelled test.
- train/val/test = vote fraction >= 0.75 (project 1); all_lowconf = train-patient sections with vf < 0.75 (project 2 adds them).
- test485 = the human-labelled sections (keyed by report + section; weight w and human worst grade from paperB_extras)."""
import json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
T = B + "/phd/thesis"; L = T + "/labeller"; HL = B + "/analysis/erin_handlabel"; OUT = B + "/analysis/rl_grader/data"
os.makedirs(OUT, exist_ok=True); os.chmod(os.path.dirname(OUT), 0o700)
rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("").drop_duplicates("CaseName").set_index("CaseName")
sec = pd.read_csv(L + "/erin_sections_v3_consensus.csv", dtype=str).fillna("")
sec = sec[sec.section.str.fullmatch(r"[A-J]|WHOLE") & sec.worst_grade.isin(GRADES) & sec.CaseName.isin(rep.index)].copy()
def vf(r):
    try: return json.loads(r.grade_votes).get(r.worst_grade, 0) / int(r.n_jurors)
    except Exception: return np.nan
sec["vf"] = sec.apply(vf, axis=1); sec["anon_id"] = sec.CaseName.map(rep.anon_id)
# exclusions: hand-label patients + SWG-shared (logic copied from build_handlabel_sample.py)
hl = pd.read_csv(HL + "/sample.csv", dtype=str); hl_pat = set(rep.loc[rep.index.intersection(hl.report_id.unique()), "anon_id"])
REL = B + "/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; EXP = B + "/barretts_db_export"
def acc_norm(s):
    s = re.sub(r"[\s_][A-Za-z]{1,2}\d{0,2}\s*(\(.*\))?\s*$", "", str(s).strip()); m = re.match(r"^\s*([A-Za-z]{1,3})\s*(\d{2})\s*[-./]?\s*(\d{3,6})\s*$", s)
    return f"{m.group(1).upper()}{m.group(2)}-{int(m.group(3)):05d}" if m else None
coh = pd.read_csv(REL + "/pre_event_cohort.csv", dtype=str); coh["acc"] = coh.BiopsyID_real.map(acc_norm)
coh["pid"] = coh.participant_id.map(lambda v: re.sub(r"\.0$", "", str(v)) if pd.notna(v) else None)
racc = pd.Series(rep.anon_id.values, index=rep.index.map(acc_norm)); swg = set(racc.reindex(coh.acc.dropna()).dropna())
db = pd.read_csv(EXP + "/pathology_text_normalised_full.csv", dtype=str, usecols=["participant_id", "specimennumber"])
db["acc"] = db.specimennumber.map(acc_norm); db["pid"] = db.participant_id.map(lambda v: re.sub(r"\.0$", "", str(v)) if pd.notna(v) else None)
br = db.dropna(subset=["acc", "pid"]); br = br[br.pid.isin(set(coh.pid.dropna()))]; swg |= set(racc.reindex(br.acc).dropna())
excl = hl_pat | swg
pool = sec[~sec.anon_id.isin(excl)].copy()
pats = np.array(sorted(pool.anon_id.unique())); rng = np.random.RandomState(0); perm = rng.permutation(pats)
n = len(perm); split_of = {p: ("train" if i < 0.8 * n else "val" if i < 0.9 * n else "test") for i, p in enumerate(perm)}
pool["split"] = pool.anon_id.map(split_of)
def item(r, extra=None):
    d = {"id": f"{r.CaseName}#{r.section}", "report_id": r.CaseName, "section": r.section, "anon_id": r.anon_id, "label": r.worst_grade,
         "vf": round(float(r.vf), 4) if pd.notna(r.vf) else None, "site": r.site, "report": report_text(rep.loc[r.CaseName])}
    if extra: d.update(extra)
    return d
def dump(df, name, extra=None):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in df.itertuples(): f.write(json.dumps(item(r, extra)) + "\n")
    os.chmod(f"{OUT}/{name}.jsonl", 0o600); return len(df)
hi = pool[pool.vf >= 0.75]; lo = pool[(pool.vf < 0.75) & (pool.split == "train")]
counts = {s: dump(hi[hi.split == s], s) for s in ("train", "val", "test")}; counts["all_lowconf"] = dump(lo, "all_lowconf")
# test485: human-labelled sections with weights, keyed report+section
lab = pd.read_csv(B + "/analysis/paperB_extras/triage/labelled_sections_with_signals.csv", dtype=str).fillna("")
lab["report_id"] = lab.section_id.str.split("#").str[0]; lab["section"] = lab.section_id.str.split("#").str[1]
with open(f"{OUT}/test485.jsonl", "w") as f:
    for r in lab.itertuples():
        f.write(json.dumps({"id": r.section_id, "report_id": r.report_id, "section": r.section, "anon_id": r.anon_id, "label": r.worst_grade,
                            "human": r.hworst, "w": float(r.w), "stratum": r.stratum, "ambiguous": int(float(r.ambiguous or 0)), "jury_vf": float(r.S1_votefrac or "nan"),
                            "report": report_text(rep.loc[r.report_id])}) + "\n")
os.chmod(f"{OUT}/test485.jsonl", 0o600); counts["test485"] = len(lab)
s = {"counts": counts, "excluded_patients": {"handlabel": len(hl_pat), "swg_shared": len(swg), "union": len(excl)},
     "patients": {k: int(sum(1 for p in perm if split_of[p] == k)) for k in ("train", "val", "test")},
     "label_dist_train": hi[hi.split == "train"].worst_grade.value_counts().to_dict(), "label_dist_test485_human": lab.hworst.value_counts().to_dict(),
     "leak_check_train_patients_in_test485": int(len(set(hi[hi.split == "train"].anon_id) & set(lab.anon_id))), "max_chars": MAXCHARS}
json.dump(s, open(f"{OUT}/summary.json", "w"), indent=1); print(json.dumps(s, indent=1))
