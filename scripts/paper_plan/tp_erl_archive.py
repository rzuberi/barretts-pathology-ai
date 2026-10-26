"""ERL steps 2-4: (A) targeted-scan designs from the ERIN report history (tp_erl_cohort.py tables); (B) the same designs from the Cambridge
Barrett's database (longer histories, same pathology archive, PS accessions); (C) scanner/pixel size from the Philips .fic sidecars;
(D) power for NP cleared, progressors missed and the gastro safety bar. Scan-request lists + study-code key stay in killcoyne_mm/two_papers_ERL/
(cluster only). Aggregates (cells < 5 -> "<5") go to results/paper_final/two_papers_ERL/archive_designs.json. No model is scored."""
import glob, json, os, re, collections, numpy as np, pandas as pd
from scipy.stats import beta, chi2
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; W = T + "/feasibility/paper_plan/killcoyne_mm/two_papers_ERL"; R = T + "/results/paper_final/two_papers_ERL"
DBX = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; SWG = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
REL = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
def sup(n): n = int(n); return n if (n == 0 or n >= 5) else "<5"
def acc_norm(s):   # isbi_rerun/code/part3_swg.py, unchanged (as ex_e5_prep.py)
    s = re.sub(r"[\s_][A-Za-z]{1,2}\d{0,2}\s*(\(.*\))?\s*$", "", str(s).strip()); m = re.match(r"^\s*([A-Za-z]{1,3})\s*(\d{2})\s*[-./]?\s*(\d{3,6})\s*$", s)
    return f"{m.group(1).upper()}{m.group(2)}-{int(m.group(3)):05d}" if m else None
rx_ther = re.compile(r"\bRFA\b|radiofrequency|ablat|\bEMR\b|\bESD\b|mucosal resection|neosquam|argon|\bAPC\b|cryo", re.I)
out = {}
# ====================================================================== (A) ERIN report history
E = pd.read_csv(W + "/endoscopies.csv", parse_dates=["d"]); P = pd.read_csv(W + "/patients_primary.csv", parse_dates=["t_ev", "first_pre", "last_pre", "last_report", "first_sc", "last_sc"])
E["acc"] = E.cases
sc = E[E.scanned & (E.v3_oes_n > 0)]; ratio = float((sc.n_oes_slides / sc.v3_oes_n).median()) if len(sc) else 1.0
out["erin_scanned_slides_per_oes_section_median"] = round(ratio, 2); out["erin_oes_sections_per_endoscopy_median"] = float(E[E.barrett_endo].v3_oes_n.median())
v3 = pd.read_csv(T + "/labeller/erin_sections_v3_consensus.csv", dtype=str); oes_secs = v3[v3.site.isin(["OESOPHAGUS", "GOJ"])].groupby("CaseName").section.agg(lambda s: "".join(sorted(set(s))))
def endo_rows_erin(p, role, h):
    g = E[(E.anon_id == p.anon_id) & E.d.isin(pd.to_datetime([x for x in str(p.pre_dates).split("|") if x]))].sort_values("d")
    if role == "NP": g = g[g.d <= p.last_report - pd.Timedelta(days=int(365.25 * h))]
    return g
def design(Pt, Et, endo_fn, k, h, src):
    el = Pt[Pt.eligible]
    kth = pd.to_datetime(el.pre_list.map(lambda l: l[k - 1] if len(l) >= k else pd.NaT))
    fu_np = (el.last_fu - kth).dt.days / 365.25; tte = (el.t_ev - kth).dt.days / 365.25
    Pp = el[el.prog & (el.n_pre >= k)]; Pn = el[~el.prog & (el.n_pre >= k) & (fu_np >= h)]
    rows = []
    for role, S in (("P", Pp), ("NP", Pn)):
        for p in S.itertuples():
            g = endo_fn(p, role, h)
            for j, e in enumerate(g.itertuples()): rows.append({"src": src, "pid": p.pid, "role": role, "order": j + 1, "date": e.d, "acc": e.acc, "scanned": bool(getattr(e, "scanned", False)),
                                                               "sections": int(getattr(e, "v3_oes_n", 0) or 0)})
    L = pd.DataFrame(rows)
    if len(L) == 0: return {"progressors": 0, "non_progressors": 0}, L
    need = L[~L.scanned]; secs = need.sections.where(need.sections > 0, 2)   # unknown section count -> assume 2 (median of a surveillance endoscopy)
    yr = need.date.dt.year
    np_all_k = el[~el.prog & (el.n_pre >= k)]; fu_all = (np_all_k.last_fu - pd.to_datetime(np_all_k.pre_list.map(lambda l: l[k - 1]))).dt.days / 365.25
    pP = el[el.prog & (el.n_pre >= k)]; tte_P = (pP.t_ev - pd.to_datetime(pP.pre_list.map(lambda l: l[k - 1]))).dt.days / 365.25
    d = {"progressors": sup(len(Pp)), "non_progressors": sup(len(Pn)), "endoscopies_P": sup((L.role == "P").sum()), "endoscopies_NP": sup((L.role == "NP").sum()),
         "endoscopies_already_scanned": sup(L.scanned.sum()), "endoscopies_to_scan": sup(len(need)),
         "endoscopies_to_scan_by_era": {"before_2016": sup((yr < 2016).sum()), "2016_2021": sup(((yr >= 2016) & (yr <= 2021)).sum()), "2022_on": sup((yr >= 2022).sum())},
         "slides_all_oes_blocks": int(round(secs.sum() * (ratio if src == "ERIN" else 1.0))), "slides_cap2_per_endoscopy": int(np.minimum(secs, 2).sum()),
         "np_person_years_after_kth_all_eligible_NP": round(float(fu_all.clip(lower=0).sum()), 0), "np_all_with_k": sup(len(np_all_k)),
         "p_person_years_kth_to_event": round(float(tte_P.clip(lower=0).sum()), 0),
         "crude_incidence_after_kth_per_100py": round(100 * len(pP) / max(float(fu_all.clip(lower=0).sum() + tte_P.clip(lower=0).sum()), 1e-9), 2),
         "p_years_kth_to_event_median": round(float(tte_P.median()), 2) if len(pP) >= 5 else "<5 patients",
         "np_followup_after_kth_median_y": round(float(fu_np[Pn.index].median()), 2) if len(Pn) >= 5 else "<5 patients"}
    d["_n"] = {"P": int(len(Pp)), "NP": int(len(Pn)), "NP_all_k": int(len(np_all_k)), "PY_NP_all_k": float(fu_all.clip(lower=0).sum()), "P_all_k": int(len(pP))}
    return d, L
P["pid"] = P.anon_id; P["eligible"] = P.barretts & ~P.excluded_overlap; P["last_fu"] = P.last_report
P["pre_list"] = P.pre_dates.fillna("").map(lambda s: sorted(pd.to_datetime([x for x in s.split("|") if x])))
designs, lists = {}, {}
for k, h in ((2, 3), (2, 5), (3, 3), (3, 5)):
    d, L = design(P, E, endo_rows_erin, k, h, "ERIN"); designs[f"ERIN_k{k}_h{h}"] = d; lists[f"ERIN_k{k}_h{h}"] = L
# ====================================================================== (B) Barrett's database
pt = pd.read_csv(DBX + "/pathology_text_normalised_full.csv", dtype=str, usecols=["participant_id", "receiveddatetime", "highestgradedysconf", "specimennumber", "reporttext"]).fillna({"reporttext": ""})
pt["pid"] = pt.participant_id.map(lambda v: re.sub(r"\.0$", "", str(v)) if pd.notna(v) else None); pt = pt[pt.pid.notna()]
pt["d"] = pd.to_datetime(pt.receiveddatetime, errors="coerce").dt.normalize(); out["db_dates_unparsed"] = int(pt.d.isna().sum()); pt = pt[pt.d.notna()]
MAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 4}; pt["code"] = pd.to_numeric(pt.highestgradedysconf, errors="coerce"); pt["g"] = pt.code.map(MAP)
pt["ther"] = pt.reporttext.str.contains(rx_ther); pt["acc"] = pt.specimennumber.map(acc_norm)
HG = pd.read_parquet(DBX + "/hgd_pathology_table.parquet"); HG["pid"] = HG.participant_id.astype(str).str.replace(r"\.0$", "", regex=True); HG["d"] = pd.to_datetime(HG.receiveddatetime, dayfirst=True, errors="coerce")
en = pd.read_parquet(DBX + "/endoscopy.parquet"); en["pid"] = en.participant_id.astype(str).str.replace(r"\.0$", "", regex=True); en["d"] = pd.to_datetime(en.endoscopydate, errors="coerce")
# SWG/Killcoyne exclusion: release participant ids + any DB participant with an accession on the SWG sheet / slide_matching / pre-event cohort (as ex_e5_prep.py)
coh = pd.read_csv(REL + "/pre_event_cohort.csv", dtype=str)
fd = pd.read_csv(SWG + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]
smf = pd.read_csv(SWG + "/slide_matching.csv", dtype=str)
accs = set(coh.BiopsyID_real.dropna().map(acc_norm)) | set(fd["Path ID"].dropna().map(acc_norm)) | set(smf.PathCaseID.dropna().map(acc_norm)); accs.discard(None)
xpid = set(coh.participant_id.dropna().map(lambda v: re.sub(r"\.0$", "", str(v)))) | set(pt[pt.acc.isin(accs)].pid)
out["db_swg_excluded_participants"] = len(xpid)
erin_acc = set(pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv", dtype=str, usecols=["CaseName"]).CaseName.map(acc_norm)) - {None}
scanned_acc = set(E[E.scanned].cases.str.split("|").explode().map(acc_norm)) - {None}
DE = pt.groupby(["pid", "d"]).agg(g=("g", "max"), be=("code", lambda c: bool(c.isin([2, 3, 4]).any())), ther=("ther", "any"), acc=("acc", lambda a: "|".join(sorted(set(x for x in a if x))))).reset_index()
DE["hg"] = list(zip(DE.pid, DE.d)); hgset = set(zip(HG.pid, HG.d.dt.normalize())); DE["hgreg"] = DE.hg.map(lambda t: t in hgset); DE = DE.drop(columns="hg")
DE["ev"] = (DE.g >= 3) | DE.hgreg; DE["scanned"] = DE.acc.map(lambda a: any(x in scanned_acc for x in a.split("|") if x)); DE["in_erin"] = DE.acc.map(lambda a: any(x in erin_acc for x in a.split("|") if x))
DE["v3_oes_n"] = 0
lastfu = pd.concat([pt.groupby("pid").d.max(), en.groupby("pid").d.max()], axis=1).max(axis=1)
hg_first = HG.groupby("pid").d.min()
rows = []
for pid, g in DE.sort_values("d").groupby("pid"):
    t_ev = min([x for x in (g.d[g.ev].min() if g.ev.any() else pd.NaT, hg_first.get(pid, pd.NaT)) if pd.notna(x)], default=pd.NaT)
    t_th = g.d[g.ther].min() if g.ther.any() else pd.NaT
    stop = min([x for x in (t_ev, t_th) if pd.notna(x)], default=pd.NaT)
    pre = g[g.be & ~g.ev & ((g.d < stop) if pd.notna(stop) else True)]
    rows.append({"pid": pid, "prog": pd.notna(t_ev) and len(pre) > 0 and t_ev > pre.d.max(), "t_ev": t_ev, "n_pre": len(pre), "pre_dates": "|".join(pre.d.dt.strftime("%Y-%m-%d")),
                 "last_fu": lastfu.get(pid, g.d.max()), "excluded_overlap": pid in xpid, "barretts": len(pre) > 0, "ther_any": pd.notna(t_th)})
DP = pd.DataFrame(rows); DP["eligible"] = DP.barretts & ~DP.excluded_overlap; DP["pre_list"] = DP.pre_dates.fillna("").map(lambda s: sorted(pd.to_datetime([x for x in s.split("|") if x])))
DP["last_report"] = DP.last_fu
out["db_participants"] = {"with_pathology": int(DP.pid.nunique()), "barretts_pre_event": sup(DP.barretts.sum()), "eligible_nonoverlap": sup(DP.eligible.sum()),
                          "eligible_progressors_any": sup((DP.eligible & DP.prog).sum()), "pathology_year_min": int(pt.d.dt.year.min()), "pathology_year_max": int(pt.d.dt.year.max())}
def endo_rows_db(p, role, h):
    g = DE[(DE.pid == p.pid) & DE.d.isin(p.pre_list)].sort_values("d")
    if role == "NP": g = g[g.d <= p.last_fu - pd.Timedelta(days=int(365.25 * h))]
    return g
for k, h in ((2, 3), (2, 5), (3, 3), (3, 5)):
    d, L = design(DP, DE, endo_rows_db, k, h, "DB"); designs[f"DB_k{k}_h{h}"] = d; lists[f"DB_k{k}_h{h}"] = L
    if len(L): d["endoscopies_with_accession_in_erin_reports"] = sup(L.acc.map(lambda a: any(x in erin_acc for x in str(a).split("|") if x)).sum()); d["endoscopies_without_accession"] = sup((L.acc.fillna("") == "").sum())
# Prague length availability for the DB design participants
pr = en.assign(m=pd.to_numeric(en.barretts_maximum, errors="coerce")).groupby("pid").m.max()
L = lists["DB_k3_h5"]; out["db_k3_h5_prague_M_available"] = sup(L.drop_duplicates("pid").pid.map(pr).notna().sum()) if len(L) else 0
# ====================================================================== (C) scanner and pixel size from the Philips .fic sidecars
sl = pd.read_csv(T + "/labeller/erin_slide_organ_v3.csv", dtype=str).merge(pd.read_csv(T + "/campaigns/allslides/erin_slides_all.csv", dtype=str, low_memory=False)[["file_uuid", "path", "CollectedOrOrdered"]], on="file_uuid")
sl = sl[sl.is_he.str.lower().eq("true") & sl.organ_final.isin(["OESOPHAGUS", "GOJ"])]
man, mod, serial, lag, sf = collections.Counter(), collections.Counter(), set(), collections.Counter(), collections.Counter(); nofic = 0
rx = {k: re.compile(r'Name="%s"[^>]*>([^<]*)<' % k) for k in ("DICOM_MANUFACTURER", "DICOM_MANUFACTURERS_MODEL_NAME", "DICOM_DEVICE_SERIAL_NUMBER", "DICOM_ACQUISITION_DATETIME", "UFS_IMAGE_DIMENSION_SCALE_FACTOR")}
smp = sl.sample(n=min(1500, len(sl)), random_state=0)
for i, r in enumerate(smp.itertuples()):
    f = os.path.splitext(r.path)[0] + ".fic"
    if not os.path.exists(f): nofic += 1; continue
    with open(f, "rb") as fh: head = fh.read(20000 if i >= 300 else 400000).decode("latin-1")
    m = rx["DICOM_MANUFACTURER"].search(head); man[m.group(1) if m else "?"] += 1
    m = rx["DICOM_MANUFACTURERS_MODEL_NAME"].search(head); mod[m.group(1) if m else "?"] += 1
    m = rx["DICOM_DEVICE_SERIAL_NUMBER"].search(head); serial.add(m.group(1)) if m else None
    m = rx["DICOM_ACQUISITION_DATETIME"].search(head); cy = pd.to_datetime(r.CollectedOrOrdered, dayfirst=True, errors="coerce")
    if m and pd.notna(cy): lag[min(int(m.group(1)[:4]) - cy.year, 3)] += 1
    if i < 300:
        m = rx["UFS_IMAGE_DIMENSION_SCALE_FACTOR"].search(head); sf[round(float(m.group(1)), 4) if m else "?"] += 1
out["scanner"] = {"fic_checked": len(smp) - nofic, "fic_missing": nofic, "manufacturer": dict(man), "model": dict(mod), "distinct_devices": len(serial),
                  "scan_year_minus_case_year (3 = 3+)": {str(k): v for k, v in sorted(lag.items())}, "level0_um_per_px (first 300)": {str(k): v for k, v in sf.items()}}
# ====================================================================== (D) power
CLEAR, MISS = 16.4 / 43, 1.0 / 32   # no-sequencing tier (mean-logit + memory): 26.6/43 NP flagged, 31.0/32 progressors caught (WHAT_WE_TRIED.md)
def cp(x, n): return [round(float(beta.ppf(0.025, x, n - x + 1)) if x > 0 else 0.0, 3), round(float(beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0, 3)]
def pois_up(x): return float(chi2.ppf(0.975, 2 * x + 2) / 2)
pw = {"assumptions": {"np_cleared_fraction": round(CLEAR, 3), "progressor_miss_fraction": round(MISS, 3), "safety_bar_per_year": 0.002,
                      "py_needed_for_upper95_le_bar": {f"x={x}": int(round(pois_up(x) / 0.002)) for x in (0, 1, 2, 3)}}}
def power(nP, nNP, py_np_cohort, label):
    x_np = int(round(CLEAR * nNP)); xm = int(round(MISS * nP))
    d = {"nP": sup(nP), "nNP": sup(nNP), "np_cleared_ci": cp(x_np, nNP) if nNP else None, "np_cleared_ci_width": round(cp(x_np, nNP)[1] - cp(x_np, nNP)[0], 3) if nNP else None,
         "missed_expected": round(MISS * nP, 2), "missed_ci_if_0": cp(0, nP) if nP else None, "missed_ci_if_1": cp(1, nP) if nP > 1 else None,
         "deescalated_py_cohort": round(CLEAR * py_np_cohort, 0)}
    if py_np_cohort > 0:
        d["upper95_rate_if_0_missed_per_100py"] = round(100 * pois_up(0) / (CLEAR * py_np_cohort), 3)
        d["upper95_rate_if_expected_missed_per_100py"] = round(100 * pois_up(xm) / (CLEAR * py_np_cohort), 3)
        d["safety_bar_assessable_if_0_missed"] = bool(pois_up(0) / (CLEAR * py_np_cohort) <= 0.002)
        d["safety_bar_assessable_at_expected_misses"] = bool(pois_up(xm) / (CLEAR * py_np_cohort) <= 0.002)
    pw[label] = d
c1 = json.load(open(R + "/cohort_counts.json"))
pw["scanned_only_step1"] = {"note": "progressors with >= 2 scanned pre-event endoscopies <5, with >= 3: 0; non-progressors with >= 1 y follow-up after the last scanned endoscopy: 0 -> no CI is meaningful"}
for key in designs:
    n = designs[key]["_n"]; power(n["P"], n["NP"], n["PY_NP_all_k"], key)
    power(n["P"], min(n["NP"], 2 * n["P"]), n["PY_NP_all_k"], key + "_NP_2per_P")
for key in designs: designs[key].pop("_n", None)
out["designs"] = designs; out["power"] = pw
# ====================================================================== scan-request lists (cluster only): study codes, accessions, oesophageal sections; dates only in the key
for key in ("ERIN_k3_h5", "ERIN_k2_h3", "DB_k3_h5", "DB_k2_h3"):
    L = lists[key]
    if not len(L): continue
    L = L.copy(); codes = {}
    for role in ("P", "NP"):
        for i, p in enumerate(sorted(L[L.role == role].pid.unique())): codes[p] = f"{key.split('_')[0][0]}{role}{i + 1:03d}"
    L["study_code"] = L.pid.map(codes)
    L["oes_sections"] = L.acc.map(lambda a: "|".join(f"{c}:{oes_secs.get(c, '?')}" for c in str(a).split("|") if c) if key.startswith("ERIN") else "")
    L[["study_code", "role", "order", "acc", "oes_sections", "scanned", "sections"]].rename(columns={"acc": "accessions", "sections": "n_oes_sections_v3"}).to_csv(W + f"/scan_request_{key}.csv", index=False)
    L[["study_code", "pid", "role", "order", "date", "acc"]].to_csv(W + f"/scan_request_{key}_KEY.csv", index=False)
os.system(f"chmod 600 {W}/scan_request_*")
DP.drop(columns="pre_list").to_csv(W + "/db_patients.csv", index=False); DE.to_csv(W + "/db_endoscopies.csv", index=False)
json.dump(out, open(R + "/archive_designs.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
