"""ERIN imminent-dysplasia task set (docs/erin_imminent_tasks_preregistration.md). Builds one sample table per task:
sample_id, anon_id, CaseName (report whose text/grade is the input), y, bag h5 list, age, grade, n_slides.
T1  first-index NDBE/IND -> HGD/CANCER within 365 d (feasibility only: <30 positives)
T2a landmark: every imaged NDBE/IND report with follow-up -> LGD+ within 365 d      (primary imminent task)
T2b landmark -> HGD+ within 365 d
T2c landmark -> next report is LGD+ (any gap)
T3a field effect: benign-section slide (NDBE/NORMAL_OTHER) -> case-max LGD+ elsewhere (slide-level bags, section labels v2)
T3b field effect -> case-max HGD+
T4  currently benign imaged report -> prior dysplasia (LGD+) in the patient's earlier reports (image + baseline only)"""
import json, os
import numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = T + "/feasibility/erin_fusion/tasks"; os.makedirs(OUT, exist_ok=True)
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"; ORD = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
lab = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str).drop_duplicates("CaseName"); lab["d"] = pd.to_datetime(lab.CollectedOrOrdered, dayfirst=True, errors="coerce"); lab = lab[lab.d.notna()]
u = lab[lab.label_status.isin(["train_eligible", "adjudicated"]) & lab.final_label.isin(ORD)].copy(); u["g"] = u.final_label.map(ORD)
rep = pd.read_csv(ERIN, dtype=str, usecols=["CaseName", "AgeAtInvestigation"]).drop_duplicates("CaseName").set_index("CaseName"); age = pd.to_numeric(rep.AgeAtInvestigation, errors="coerce")
sl = pd.read_csv(T + "/campaigns/allslides/erin_slides_all.csv", dtype=str, low_memory=False); he = sl[(sl.is_he.astype(str).str.lower() == "true") & sl.h5_new.notna()]; he = he[he.h5_new.map(os.path.exists)]
bags = he.groupby("CaseName").h5_new.agg(list); u["imaged"] = u.CaseName.isin(bags.index)
def row(pid, r, y, sid=None, h5s=None):
    h = h5s if h5s is not None else bags[r.CaseName]
    return {"sample_id": sid or r.CaseName, "anon_id": pid, "CaseName": r.CaseName, "y": int(y), "grade": r.final_label, "age": age.get(r.CaseName, np.nan), "n_slides": len(h), "h5_list": "|".join(h), "date": str(r.d.date())}
tasks = {k: [] for k in ("T1", "T2a", "T2b", "T2c", "T4")}
for pid, g in u.sort_values("d").groupby("anon_id"):
    g = g.reset_index(drop=True); last = lab[lab.anon_id == pid].d.max()
    idx = g[g.g <= 1]
    if len(idx):  # T1: first NDBE/IND report, 1-year horizon
        i0 = idx.iloc[0]
        if i0.imaged:
            later = g[g.d > i0.d]; ev = later[later.g >= 3]; t = (ev.d.min() - i0.d).days if len(ev) else None; fu = (last - i0.d).days
            if t is not None and t <= 365: tasks["T1"].append(row(pid, i0, 1))
            elif t is None and fu >= 365: tasks["T1"].append(row(pid, i0, 0))
    for i, r in g.iterrows():
        if not r.imaged or r.g > 1: continue
        later = g[g.d > r.d]; earlier = g[g.d < r.d]
        if len(earlier): tasks["T4"].append(row(pid, r, int((earlier.g >= 2).any()), sid=r.CaseName))
        if later.empty: continue
        fu = (later.d.max() - r.d).days; el = later[later.g >= 2]; eh = later[later.g >= 3]
        tl = (el.d.min() - r.d).days if len(el) else None; th = (eh.d.min() - r.d).days if len(eh) else None
        if tl is not None and tl <= 365: tasks["T2a"].append(row(pid, r, 1))
        elif tl is None and fu >= 365: tasks["T2a"].append(row(pid, r, 0))
        if th is not None and th <= 365: tasks["T2b"].append(row(pid, r, 1))
        elif th is None and fu >= 365: tasks["T2b"].append(row(pid, r, 0))
        tasks["T2c"].append(row(pid, r, int(later.iloc[0].g >= 2)))
# T3: slide-level field effect from the dual-labelled table (slide-specific h5 from the one-per-case set)
v2 = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str); m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
v2 = v2.merge(m[["h5", "anon_id"]], on="h5"); ben = v2[v2.worst_grade.isin(["NDBE", "NORMAL_OTHER"])]
for key, pos in (("T3a", {"LGD", "HGD", "CANCER"}), ("T3b", {"HGD", "CANCER"})):
    tasks[key] = [{"sample_id": os.path.basename(r.h5)[:8], "anon_id": r.anon_id, "CaseName": r.CaseName, "y": int(r.case_max in pos), "grade": r.worst_grade, "age": age.get(r.CaseName, np.nan), "n_slides": 1, "h5_list": r.h5, "date": ""} for r in ben.itertuples()]
counts = {}
for k, rows in tasks.items():
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, f"{k}.csv"), index=False)
    counts[k] = {"n": len(d), "pos": int(d.y.sum()), "neg": int((d.y == 0).sum()), "patients": int(d.anon_id.nunique()), "slides": int(d.n_slides.sum()), "grade_dist": d.grade.value_counts().to_dict(), "feasibility_only": bool(d.y.sum() < 30 or (d.y == 0).sum() < 30)}
    print(k, counts[k], flush=True)
json.dump(counts, open(os.path.join(OUT, "task_counts.json"), "w"), indent=2)
