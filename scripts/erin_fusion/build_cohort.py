"""ERIN progression fusion — cohort build (docs/erin_progression_fusion_preregistration.md).
Index = each patient's FIRST train-eligible/adjudicated report graded NDBE or IND. Label y3 = HGD/CANCER report
within 1,095 days of index (1); no HGD/CANCER ever AND last report >= 1,095 days after index (0); else excluded
(censored early). y_any = any later HGD/CANCER (secondary). Index-case bag = ALL H&E slides of the index CaseName
with all-slides UNI2 features. Also: age at index, jury-v3 structured features of the index report (b2), and the
ERIN<->SWG patient overlap flag via the accession/DB crosswalk of task_overlap_audit.py."""
import glob, json, os, re
from collections import Counter, defaultdict
import numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", T + "/feasibility/erin_fusion"); os.makedirs(OUT, exist_ok=True)
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"; H = 1095
ORD = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
lab = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str).drop_duplicates("CaseName")
lab["d"] = pd.to_datetime(lab.CollectedOrOrdered, dayfirst=True, errors="coerce"); lab = lab[lab.d.notna()]
usable = lab[lab.label_status.isin(["train_eligible", "adjudicated"]) & lab.final_label.isin(ORD)].copy(); usable["g"] = usable.final_label.map(ORD)
rep = pd.read_csv(ERIN, dtype=str, usecols=["CaseName", "AgeAtInvestigation"]).drop_duplicates("CaseName").set_index("CaseName")
rows = []
for pid, g in usable.sort_values("d").groupby("anon_id"):
    idx = g[g.g <= 1]
    if idx.empty: continue
    i0 = idx.iloc[0]; later = g[g.d > i0.d]; allrep = lab[lab.anon_id == pid]; last = allrep.d.max()
    ev = later[later.g >= 3]; t_ev = (ev.d.min() - i0.d).days if len(ev) else None; fu = (last - i0.d).days
    if t_ev is not None and t_ev <= H: y3 = 1
    elif t_ev is None and fu >= H: y3 = 0
    else: y3 = -1
    rows.append({"anon_id": pid, "index_case": i0.CaseName, "index_date": i0.d.date(), "index_grade": i0.final_label, "index_status": i0.label_status,
                 "age": pd.to_numeric(rep.AgeAtInvestigation.get(i0.CaseName), errors="coerce"), "y3": y3, "y_any": int(t_ev is not None), "tte_days": t_ev, "followup_days": fu, "n_later_reports": len(later)})
coh = pd.DataFrame(rows); print("report-level index patients", len(coh), "y3 dist", coh.y3.value_counts().to_dict(), "y_any", coh.y_any.sum(), flush=True)
# ---- slides: all H&E slides of the index case with features ----
sl = pd.read_csv(T + "/campaigns/allslides/erin_slides_all.csv", dtype=str, low_memory=False)
he = sl[(sl.is_he.astype(str).str.lower() == "true") & sl.h5_new.notna()]; he = he[he.h5_new.map(os.path.exists)]
bags = he.groupby("CaseName").h5_new.agg(list); coh["h5_list"] = coh.index_case.map(lambda c: bags.get(c, [])); coh["n_slides"] = coh.h5_list.map(len)
img = coh[coh.n_slides > 0].copy()
# ---- SWG overlap (accession crosswalk + DB participant bridge, as task_overlap_audit) ----
def acc_norm(s):
    m = re.match(r"^\s*([A-Za-z]{1,3})\s*(\d{2})\s*[-./]?\s*(\d{3,6})\s*$", str(s)); return (m.group(1).upper(), m.group(2), int(m.group(3))) if m else None
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); swg = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).merge(man, left_on="SampleID", right_on="sample_id")
swg_acc = {acc_norm(b) for b in swg.BiopsyID_real} - {None}
db = pd.read_csv("/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export/pathology_text_normalised_full.csv", dtype=str, usecols=["specimennumber", "participant_id"]).dropna()
db["acc"] = db.specimennumber.map(acc_norm); db = db[db.acc.notna()]; acc2part = dict(zip(db.acc, db.participant_id)); swg_parts = {acc2part[a] for a in swg_acc if a in acc2part}
erin_acc = lab.assign(acc=lab.CaseName.map(acc_norm)).dropna(subset=["acc"])
pat_parts = defaultdict(set)
for a, p in zip(erin_acc.acc, erin_acc.anon_id):
    if a in acc2part: pat_parts[p].add(acc2part[a])
overlap = {p for p, parts in pat_parts.items() if parts & swg_parts} | set(erin_acc[erin_acc.acc.isin(swg_acc)].anon_id)
coh["swg_overlap"] = coh.anon_id.isin(overlap); img["swg_overlap"] = img.anon_id.isin(overlap)
# ---- jury v3 structured features of the index report (arm b2) ----
GR = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; SITES = ["OESOPHAGUS", "GOJ", "STOMACH", "DUODENUM", "OTHER"]; SPEC = ["BIOPSY", "RESECTION", "EMR", "OTHER"]
want = set(img.index_case); per_case = defaultdict(list)
for f in glob.glob(T + "/feasibility/runs/sec3full_*/output/sections_v3_*.csv"):
    d = pd.read_csv(f, dtype=str, on_bad_lines="skip"); d = d[d.CaseName.isin(want)]
    for c, js in zip(d.CaseName, d.sections_json):
        try: per_case[c].append(json.loads(js))
        except Exception: pass
def b2(case):
    js = per_case.get(case, [])
    if not js: return {"b2_n_jurors": 0}
    nsec = float(np.median([len(x) for x in js])); site = Counter(); spec = Counter(); grades = Counter(); ind_secs = []
    for x in js:
        site.update(str(s.get("site", "OTHER")).upper() for s in x); spec.update(str(s.get("specimen_type", "OTHER")).upper() for s in x); grades.update(g for s in x for g in s.get("grades", []))
        ind_secs.append(sum("IND" in s.get("grades", []) for s in x))
    n = len(js); out = {"b2_n_jurors": n, "b2_n_sections": nsec, "b2_frac_sections_IND": float(np.mean(ind_secs)) / max(nsec, 1)}
    tot = max(sum(site.values()), 1); out.update({f"b2_site_{s}": site.get(s, 0) / tot for s in SITES}); tot = max(sum(spec.values()), 1); out.update({f"b2_spec_{s}": spec.get(s, 0) / tot for s in SPEC})
    tot = max(sum(grades.values()), 1); out.update({f"b2_grade_{g}": grades.get(g, 0) / tot for g in GR}); return out
b2df = pd.DataFrame([{"index_case": c, **b2(c)} for c in img.index_case]); img = img.merge(b2df, on="index_case", how="left")
img["h5_list"] = img.h5_list.map(lambda l: "|".join(l)); img.to_csv(os.path.join(OUT, "cohort_index_cases.csv"), index=False)
a3 = img[img.y3 >= 0]
cnt = {"report_level": {"index_patients": len(coh), "y3_pos": int((coh.y3 == 1).sum()), "y3_neg": int((coh.y3 == 0).sum()), "y3_excluded_censored": int((coh.y3 == -1).sum()), "y_any_pos": int(coh.y_any.sum()), "index_grade": coh.index_grade.value_counts().to_dict()},
       "imaged_index_cases": {"n": len(img), "patients": int(img.anon_id.nunique()), "slides_total": int(img.n_slides.sum()), "slides_per_case": {"median": float(img.n_slides.median()), "min": int(img.n_slides.min()), "max": int(img.n_slides.max())},
                              "y3_pos": int((img.y3 == 1).sum()), "y3_neg": int((img.y3 == 0).sum()), "y3_excluded": int((img.y3 == -1).sum()), "y_any_pos": int(img.y_any.sum()),
                              "index_grade": img.index_grade.value_counts().to_dict(), "age_available": int(img.age.notna().sum()), "b2_available": int((img.b2_n_jurors > 0).sum()), "swg_overlap_patients": int(img.swg_overlap.sum()),
                              "swg_overlap_among_y3_pos": int(img[img.y3 == 1].swg_overlap.sum())},
       "analysis_set_y3": {"n": len(a3), "pos": int(a3.y3.sum()), "neg": int((a3.y3 == 0).sum()), "pos_frac": round(float(a3.y3.mean()), 4), "tte_days_pos_median": float(a3[a3.y3 == 1].tte_days.median()) if (a3.y3 == 1).any() else None, "followup_neg_median": float(a3[a3.y3 == 0].followup_days.median()) if (a3.y3 == 0).any() else None},
       "analysis_set_any": {"n": len(img), "pos": int(img.y_any.sum())}, "feasibility_flag_pos_lt_30": bool(a3.y3.sum() < 30), "horizon_days": H, "sex_available": False}
json.dump(cnt, open(os.path.join(OUT, "cohort_counts.json"), "w"), indent=2, default=str); print(json.dumps(cnt, indent=1, default=str))
