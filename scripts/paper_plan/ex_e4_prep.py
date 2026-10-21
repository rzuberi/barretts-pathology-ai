"""Exploration E4 (docs/paper_explore_E4.md): per-sample time-to-event for the analysed set (676 samples), exact dates (pre_event_cohort.Date).
Progressors: sample date -> event (rv8_landscape.dates() reference = first HGD/IMC endoscopy); at/after-event samples 0.01 y.
Non-progressors (censored): study follow-up = to the last sequenced endoscopy (rv8 reference); extended = to the later of that and the last Barrett's-DB
record (pathology report or endoscopy) before any later HGD+ report / HGD-register entry (linkage and tables as r2_main.py, data2.md B).
Row-level output (cluster only): killcoyne_mm/explore_E4/surv.csv. Aggregates (no identifiers, no dates) -> results/paper_final/explore_E4/prep.json."""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.environ.setdefault("TASK", "none")
_s = open(HERE + "/rv8_landscape.py").read(); exec(compile(_s[:_s.index('res = {"task": TASK}')], HERE + "/rv8_landscape.py", "exec"))
OUTE = T + "/results/paper_final/explore_E4"; RE = M + "/explore_E4"
for d in (OUTE, RE): os.makedirs(d, exist_ok=True)
EPS = 0.01; W = 14; E_ = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
SD, REF, dinfo = dates()
SDs = pd.to_datetime(pd.Series(SD)); assert SDs.notna().all()
# ---- linkage (r2_main.py B: mode of Hospital_Research_ID_updated -> pre_event_cohort participant_id; then Demographics alternate number; then initial_history)
fd = sheet(); fd = fd[fd.combined_name.notna()].copy(); fd["Patient"] = fd.Patient.astype(str).str.replace(r"\.0$", "", regex=True)
pe = pd.read_csv(REL + "/pre_event_cohort.csv", dtype=str)
pid_of = pe.dropna(subset=["participant_id"]).groupby("PatientID_real").participant_id.agg(lambda s: s.str.replace(r"\.0$", "", regex=True).mode().iloc[0])
IH = pd.read_parquet(E_ + "/initial_history.parquet"); IH["pid"] = IH.participant_id.astype(str).str.replace(r"\.0$", "", regex=True)
dm = pd.read_csv(SWG + "/Demographics_full.csv", dtype=str)
for c_ in ("Study Number", "Alternate Study Number"): dm[c_] = dm[c_].fillna("").str.strip()
dmi = {}
for _, r in dm.iterrows():
    for c_ in ("Study Number", "Alternate Study Number"):
        if r[c_] and r[c_] not in dmi: dmi[r[c_]] = r
link = {}
for p in sorted(set(pat)):
    g = fd[fd.Patient == p]; ks = []
    for c_ in ("Hospital_Research_ID_updated", "Hospital Research ID"):
        m_ = g[c_].dropna().str.strip(); m_ = m_[m_ != ""]
        if len(m_): ks.append(m_.mode().iloc[0])
    ks = list(dict.fromkeys(ks)); pid = None
    for k in ks:
        if k in pid_of.index: pid = pid_of[k]; break
    if pid is None:
        d = next((dmi[k] for k in ks if k in dmi), None); ks2 = ks + ([d["Study Number"], d["Alternate Study Number"]] if d is not None else [])
        for k in ks2:
            if k and k in pid_of.index: pid = pid_of[k]; break
    if pid is None:
        for col in ("hospitalnumber", "patient_anonref"):
            m_ = IH[IH[col].astype(str).str.strip().isin(set(ks))]
            if len(m_): pid = m_.pid.iloc[0]; break
    link[p] = pid
RK = {2.0: 0, 3.0: 1, 4.0: 2, 5.0: 3, 8.0: 4, 6.0: 5}
pt = pd.read_csv(E_ + "/pathology_text_normalised_full.csv", dtype=str, usecols=["participant_id", "receiveddatetime", "highestgradedysconf"])
pt["pid"] = pt.participant_id.astype(str).str.replace(r"\.0$", "", regex=True); pt["d"] = pd.to_datetime(pt.receiveddatetime, errors="coerce"); pt["g"] = pd.to_numeric(pt.highestgradedysconf, errors="coerce").map(RK)
EN = pd.read_parquet(E_ + "/endoscopy.parquet", columns=["participant_id", "endoscopydate"]); EN["pid"] = EN.participant_id.astype(str).str.replace(r"\.0$", "", regex=True); EN["d"] = pd.to_datetime(EN.endoscopydate, errors="coerce")
ME = pd.read_parquet(E_ + "/mat_endoscopy_full.parquet", columns=["participant_id", "endoscopydate"]); ME["pid"] = ME.participant_id.astype(str); ME["d"] = pd.to_datetime(ME.endoscopydate, errors="coerce")
HG = pd.read_parquet(E_ + "/hgd_pathology_table.parquet"); HG["pid"] = HG.participant_id.astype(str).str.replace(r"\.0$", "", regex=True); HG["d"] = pd.to_datetime(HG.receiveddatetime, dayfirst=True, errors="coerce")
ENDOT = pd.concat([EN[["pid", "d"]], ME[["pid", "d"]]]).dropna().drop_duplicates(); ENDOT = ENDOT[ENDOT.d.dt.year >= 1980]
# ---- per patient reference / follow-up end
NPP = sorted(set(pat[y == 0])); FU = {}; stats = {"np_linked": 0, "np_extended": 0, "np_later_hgdplus_stopped": 0, "np_ext_years_added": []}
for p in NPP:
    last = pd.Timestamp(REF[p]); end = last; pid = link.get(p)
    if pid is not None and pid == pid:
        stats["np_linked"] += 1; a = pt[pt.pid == pid]; e = ENDOT[ENDOT.pid == pid]; h = HG[HG.pid == pid]
        bad = pd.concat([a[(a.g >= 3) & (a.d > last + pd.Timedelta(days=W))].d, h[h.d > last + pd.Timedelta(days=W)].d]).dropna()
        stop = bad.min() if len(bad) else pd.Timestamp.max
        if len(bad): stats["np_later_hgdplus_stopped"] += 1
        recs = pd.concat([a.d, e.d]).dropna(); recs = recs[(recs > last) & (recs < stop)]
        if len(recs): end = max(last, recs.max())
    if end > last: stats["np_extended"] += 1
    stats["np_ext_years_added"].append((end - last).days / 365.25); FU[p] = end
# ---- per sample times (years)
ty_s = np.full(n, np.nan); ty_x = np.full(n, np.nan); ev = y.copy()
for i in range(n):
    p = pat[i]
    if y[i] == 1:
        if p in REF and PRE[i]: ty_s[i] = max((pd.Timestamp(REF[p]) - SDs[i]).days / 365.25, EPS)
        else: ty_s[i] = EPS
        ty_x[i] = ty_s[i]
    else:
        ty_s[i] = max((pd.Timestamp(REF[p]) - SDs[i]).days / 365.25, EPS); ty_x[i] = max((FU[p] - SDs[i]).days / 365.25, EPS)
assert np.isfinite(ty_s).all() and np.isfinite(ty_x).all()
D = pd.DataFrame({"Sample": ids, "Patient": pat, "y": y, "pre": PRE, "event": ev, "t_study": ty_s, "t_ext": ty_x, "t_sheet": SM.time.values.astype(float)})
D.to_csv(RE + "/surv.csv", index=False); os.chmod(RE + "/surv.csv", 0o640)
q = lambda a: {"median": r3(np.median(a)), "q1": r3(np.percentile(a, 25)), "q3": r3(np.percentile(a, 75)), "min": r3(np.min(a)), "max": r3(np.max(a))}
pm = PRE & (y == 1); nm = y == 0
added = np.array(stats.pop("np_ext_years_added"))
R = {"samples": int(n), "pre_event_samples": int(PRE.sum()), "progressor_samples_not_pre_event_set_to_eps": int(((y == 1) & ~PRE).sum()),
     "progressor_pre_event_years_to_event": q(ty_s[pm]), "progressor_pre_event_gt8y": int((ty_s[pm] > 8).sum()),
     "np_years_study_followup": q(ty_s[nm]), "np_years_extended_followup": q(ty_x[nm]), "np_patients": len(NPP), **stats,
     "np_years_added_per_patient": q(added), "np_samples_ext_ge3": int((ty_x[nm] >= 3).sum()), "np_samples_ext_ge5": int((ty_x[nm] >= 5).sum()),
     "np_samples_study_ge3": int((ty_s[nm] >= 3).sum()), "np_samples_study_ge5": int((ty_s[nm] >= 5).sum()),
     "abs_diff_exact_vs_sheet_time_progressor_pre_event_years_median": r3(np.median(np.abs(ty_s[pm] - SM.time.values.astype(float)[pm]))),
     "progressor_samples_le3y": int(((y == 1) & (ty_s <= 3)).sum()), "progressor_samples_le5y": int(((y == 1) & (ty_s <= 5)).sum()),
     "dates_info_rv8": {k: dinfo[k] for k in ("reference_date_source_patients",) if k in dinfo}}
for k in ("np_linked", "np_extended", "np_later_hgdplus_stopped"): R[k] = "<5" if 0 < R[k] < 5 else R[k]
json.dump({"task": "prep", **R}, open(OUTE + "/prep.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("EXE4 PREP DONE", json.dumps(R)[:1500])
