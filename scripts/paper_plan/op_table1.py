"""One-paper Table 1 (cohort characteristics), progressors vs non-progressors. CPU, cluster only; writes aggregates only (no IDs, no dates).
Levels: 75 analysed patients (pre-event samples, the paper's population), 80 patients with a linked slide (set C), 88 discovery-sheet patients
(patient-level fields only: the 8 unlinked progressors have no set C rows).
Sources (read-only): killcoyne_mm/horizons/samples.csv (class, grade, p53, months before final, event time, pre-event flag, study number),
killcoyne_mm/set_C.csv (Endoscopy = endoscopy number within patient), SWGCohort/Demographics_full.csv (age at diagnosis, sex, Prague C/M, smoking,
weight, height), the sWGS sample sheet (88-patient key = mode of Hospital_Research_ID_updated, as hz_prep.py / dm_analyse.py; Path_class_per_OGD for
the endpoint type when no sequenced sample carries HGD/IMC).
Definitions: endoscopy = (patient, set_C Endoscopy) as tb_common.ENDO (288 pre-event endoscopies); first analysed endoscopy = the patient's pre-event
endoscopy with the largest months-before-final; grade there = highest sample grade (NDBE < ID < LGD); p53 there = aberrant if any sample aberrant,
normal if any sample stained and none aberrant, else not stained. Follow-up = first analysed endoscopy to the HGD/IMC endpoint (progressors) or to the
last sequenced endoscopy (non-progressors), years. Endpoint type = highest grade among the progressor's sequenced samples (kr_samples.csv, all 773) at the event time (IMC > HGD), else
the sheet's per-endoscopy class at that time, else 'not recorded'. BMI = weight / height^2 where both recorded. Cells < 5 written as '<5'.
Output: results/paper_final/one_paper/table1.json"""
import os, json, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
OUT = T + "/results/paper_final/one_paper"; os.makedirs(OUT, exist_ok=True)
SM = pd.read_csv(M + "/horizons/samples.csv", dtype={"Sample": str, "Patient": str, "study_number": str})
C = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample")
SM["endo"] = SM.Patient + "|" + C.Endoscopy.reindex(SM.Sample).values
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]
fdi = fd.drop_duplicates("combined_name").set_index("combined_name")
KR = pd.read_csv(T + "/feasibility/paper_plan/killcoyne/kr_samples.csv", dtype=str)   # all sequenced samples (773), incl. endpoint samples without a linked slide
KR["mbf"] = pd.to_numeric(KR.Sample.map(fdi["Months before final"]), errors="coerce"); KR["ogd"] = KR.Sample.map(fdi["Path_class_per_OGD"])
dm = pd.read_csv(S + "/Demographics_full.csv", dtype=str); dm.columns = [c.strip() for c in dm.columns]
for c in ("Study Number", "Alternate Study Number"): dm[c] = dm[c].str.strip()
dmi = {}
for _, r in dm.iterrows():
    for c in ("Study Number", "Alternate Study Number"):
        if isinstance(r[c], str) and r[c] and r[c] not in dmi: dmi[r[c]] = r
num = lambda v: pd.to_numeric(pd.Series([v]).replace("", np.nan), errors="coerce").iloc[0]
cnt = lambda k: "<5" if 0 < k < 5 else int(k)
def q(v):
    a = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(a): return {"n": 0}
    q1, md, q3 = np.percentile(a, [25, 50, 75]); return {"n": cnt(len(a)), "median": round(float(md), 1), "q1": round(float(q1), 1), "q3": round(float(q3), 1)}
def cat(vals, levels):
    s = pd.Series(vals, dtype=object); N = len(s); o = {}
    for lv in levels: k = int((s == lv).sum()); o[lv] = {"n": cnt(k), "pct": (round(100 * k / N) if k >= 5 else None)}
    return o
def demo(keys):
    rows = []
    for k in keys:
        r = dmi.get(k)
        if r is None: rows.append(dict(age=np.nan, sex=None, C=np.nan, M=np.nan, smk="missing", bmi=np.nan, linked=False)); continue
        w, h = num(r.get("Weight (kg)")), num(r.get("Height (cm)"))
        smk = r["Smoking Status"].strip() if isinstance(r["Smoking Status"], str) and r["Smoking Status"].strip() in ("Y", "N") else "missing"
        rows.append(dict(age=num(r["Age at diagnosis"]), sex=(r["Sex"].strip().upper()[:1] if isinstance(r["Sex"], str) and r["Sex"].strip() else None),
                         C=num(r["Circumference"]), M=num(r["Maximal"]), smk=smk, bmi=(w / (h / 100) ** 2 if np.isfinite(w) and np.isfinite(h) and h > 0 else np.nan), linked=True))
    return pd.DataFrame(rows)
def demo_block(D):
    N = len(D)
    return {"patients": N, "linked_to_demographics": cnt(int(D.linked.sum())), "age_at_diagnosis": q(D.age), "sex": cat(D.sex.fillna("missing"), ["M", "F", "missing"]),
            "prague_C": q(D.C), "prague_M": q(D.M), "prague_C_missing": cnt(int(D.C.isna().sum())), "prague_M_missing": cnt(int(D.M.isna().sum())),
            "smoking": cat(D.smk, ["Y", "N", "missing"]), "bmi": q(D.bmi), "bmi_missing": cnt(int(D.bmi.isna().sum()))}
GR = {"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}; GN = {0: "NDBE", 1: "ID", 2: "LGD", 3: "HGD", 4: "IMC"}
res = {"note": __doc__.split("Output:")[0].strip()}
for lev, rows in (("75_analysed", SM[SM.pre.astype(bool)]), ("80_linked_slide", SM)):
    out = {}
    for cls, yy in (("progressors", 1), ("non_progressors", 0)):
        R = rows[rows.y == yy]; pats = sorted(R.Patient.unique()); o = {}
        keys = [SM[SM.Patient == p].study_number.iloc[0] for p in pats]; o.update(demo_block(demo(keys)))
        E = R.groupby("endo").agg(mbf=("mbf", "max"), n=("Sample", "size"), pat=("Patient", "first"))
        o["endoscopies"] = int(len(E)); o["samples"] = int(len(R)); o["endoscopies_per_patient"] = q(E.groupby("pat").size().reindex(pats).values.astype(float))
        o["samples_per_endoscopy"] = q(E.n.values.astype(float)); o["samples_per_patient"] = q(R.groupby("Patient").size().reindex(pats).values.astype(float))
        g1, p1, fu, ep = [], [], [], []
        for p in pats:
            Rp = R[R.Patient == p]; Ep = E[E.pat == p]; first = Ep.mbf.idxmax(); f = Rp[Rp.endo == first]
            g1.append(GN[int(f.Pathology.map(GR).max())]); p53 = pd.to_numeric(f["P53 IHC"], errors="coerce")
            p1.append("aberrant" if (p53 == 1).any() else ("normal" if p53.notna().any() else "not stained"))
            A = KR[KR.Patient == p]
            if yy == 1:
                tev = float(SM[SM.Patient == p].tev.iloc[0]); fu.append((Ep.mbf.max() - tev) / 12.0)
                at = A[np.isclose(A.mbf.astype(float), tev)]; gmax = at.Pathology.map(GR).max() if len(at) else np.nan
                if np.isfinite(gmax) and gmax >= 3: ep.append(GN[int(gmax)])
                else:
                    og = " ".join(str(x) for x in at.ogd.dropna().unique()).upper()
                    ep.append("IMC" if "IMC" in og and "HGD" not in og else ("HGD" if "HGD" in og and "IMC" not in og else ("HGD/IMC (sheet)" if og else "not recorded")))
            else: fu.append(Ep.mbf.max() / 12.0)
        o["grade_first_endoscopy"] = cat(g1, ["NDBE", "ID", "LGD"]); o["p53_first_endoscopy"] = cat(p1, ["aberrant", "normal", "not stained"])
        p53s = pd.to_numeric(R["P53 IHC"], errors="coerce"); o["samples_p53_stained"] = int(p53s.notna().sum()); o["samples_p53_aberrant"] = cnt(int((p53s == 1).sum()))
        o["patients_any_aberrant_p53"] = cnt(int(R.assign(a=(p53s == 1)).groupby("Patient").a.any().sum()))
        o["grade_any_sample"] = cat([GN[int(R[R.Patient == p].Pathology.map(GR).max())] for p in pats], ["NDBE", "ID", "LGD", "HGD", "IMC"])
        o["follow_up_years"] = q(fu)
        if yy == 1: o["endpoint"] = cat(ep, ["HGD", "IMC", "HGD/IMC (sheet)", "not recorded"]); o["fallback_endpoint_patients"] = cnt(int(R.groupby("Patient").fallback_endpoint.first().astype(str).eq("True").sum()))
        out[cls] = o
    res[lev] = out
assert res["75_analysed"]["progressors"]["patients"] == 32 and res["75_analysed"]["non_progressors"]["patients"] == 43
assert res["75_analysed"]["progressors"]["endoscopies"] + res["75_analysed"]["non_progressors"]["endoscopies"] == 288
assert res["75_analysed"]["progressors"]["samples"] == 161 and res["75_analysed"]["non_progressors"]["samples"] == 410
# 88 discovery-sheet patients: patient-level demographic fields only
key = fd.groupby("PatientID")["Hospital_Research_ID_updated"].agg(lambda s: s.dropna().str.strip().mode().iloc[0] if s.dropna().size else None)
pcol = [c for c in fd.columns if c.startswith("Progressor")][0]
cls88 = fd.groupby("PatientID")[pcol].agg(lambda s: s.dropna().str.strip().mode().iloc[0] if s.dropna().size else None)
o88 = {}
for cls, lab in (("progressors", "P"), ("non_progressors", "NP")):
    ks = [key[p] for p in key.index if isinstance(cls88.get(p), str) and cls88[p].upper().startswith(lab) and not (lab == "P" and cls88[p].upper().startswith("NP"))]
    o88[cls] = demo_block(demo(ks))
res["88_discovery_sheet"] = o88; res["88_class_column"] = pcol; res["88_class_values"] = {str(k): int(v) for k, v in cls88.value_counts().items()}
json.dump(res, open(OUT + "/table1.json", "w"), indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)); print("OP TABLE1 DONE", json.dumps(res["88_class_values"]))
