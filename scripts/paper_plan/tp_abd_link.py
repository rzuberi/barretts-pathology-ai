"""ABD: outcome-blind linkage of ACE-B patients to the Cambridge Barrett's database (identifiers and dates only).

Reads (usecols only):
  ACE-B sample sheet: Batch, Name, PatientID.  ACE-B case list: Study Number, Date AFI, Date WLE.
  May-2026 ID files: study-number match table (study_number_raw, participant_ids_in_filter), PS->participant map (ps_id, mapped_ids).
  Barrett's DB, Aug-2026 API export: pathology (pathology_text_id, participant_id, receiveddatetime, rowupdatedatetime, specimennumber),
    endoscopy / mat_endoscopy_full (participant_id, endoscopydate, related_specimennumber).
  Barrett's DB, Feb-2026 API scrape: pathology (same id/date columns), view_patient_endoscopy_full (participant_id, endoscopydate,
    related_specimennumber), samples_filter_summary (participant_id, study_name), view_participant_basic (participant_id, list_study_name).
  Cross-cohort: release pre_event_cohort (PatientID, PatientID_real, participant_id), set_C (Patient), erin_swg_pairs (ids),
    ERIN reports (anon_id, CollectedOrOrdered, CaseName), ERL scan-request KEY files (pid / anon_id only).
Never read: Pathology/histology, grades, report text, progressor/outcome, treatment, date of death.
Writes: row-level link table to killcoyne_mm/two_papers_ABD/ (chmod 600); aggregates (cells 1-4 -> "<5") to results/paper_final/two_papers_ABD/.
"""
import json, os, re
from collections import Counter
import numpy as np, pandas as pd

Z = "/mnt/scratche/slow/fmlab/zuberi01"; META = f"{Z}/phd/aceb_meta"; DB = f"{Z}/barretts_db_export"
FEB = f"{Z}/phd/barretts_retraining/barretts_training/data/barretts_scrape_20260220_172725"
T = f"{Z}/phd/thesis"; KM = f"{T}/feasibility/paper_plan/killcoyne_mm"; W = f"{KM}/two_papers_ABD"; R = f"{T}/results/paper_final/two_papers_ABD"
REL = f"{Z}/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
os.makedirs(R, exist_ok=True)
def sup(n):
    n = int(n); return n if (n == 0 or n >= 5) else "<5"
def pid(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    s = re.sub(r"\.0$", "", str(v).strip()); return s if s and s.lower() not in ("nan", "none") else None
def acc_key(s):
    """Accession -> (prefix, yy, int number). AB12.345 / AB12-00345 / AB12-00345C / 'AB12-00345 x' all -> ('AB','12',345) (synthetic example)."""
    m = re.match(r"\s*([A-Za-z]{1,3})\s?(\d{2})[.\-/ ]?(\d{1,6})", str(s))
    return (m.group(1).upper(), m.group(2), int(m.group(3))) if m else None
def acc_key_alt(s):
    """Second sheet format, digits-letter-digits (e.g. 99A999999A9) -> (letter, yy, int number)."""
    m = re.match(r"\s*(\d{2})([A-Za-z])(\d{4,7})", str(s))
    return (m.group(2).upper(), m.group(1), int(m.group(3))) if m else None
def fmt(s): return re.sub(r"\d", "9", re.sub(r"[A-Za-z]", "A", str(s).strip()))
out = {"note": "aggregates only; cells 1-4 shown as <5"}

# ---------------- ACE-B identifiers
S = pd.read_csv(f"{META}/ACEB_samples_for Rehan.csv", usecols=["Batch", "Name", "PatientID"], dtype=str)
S["key"] = [acc_key(n) if fmt(n).startswith("AA") else acc_key_alt(n) for n in S.Name]
C = pd.read_csv(f"{META}/ACE-B cases for Rehan.csv", usecols=["Study Number", "Date AFI", "Date WLE"], dtype=str)
for c in ["Date AFI", "Date WLE"]: C[c] = pd.to_datetime(C[c], dayfirst=True, errors="coerce")
C = C.groupby("Study Number").agg({"Date AFI": "min", "Date WLE": "min"})
pats = sorted(S.PatientID.dropna().unique())
out["aceb_sheet"] = {"samples": len(S), "patients": len(pats), "batches": int(S.Batch.nunique()),
                     "name_formats": {f: sup(v) for f, v in Counter(fmt(n) for n in S.Name).items()},
                     "case_list_study_numbers": len(C), "sheet_patients_in_case_list": sup(sum(p in C.index for p in pats)),
                     "sheet_patients_with_case_list_date": sup(sum(p in C.index and (pd.notna(C.loc[p, "Date AFI"]) or pd.notna(C.loc[p, "Date WLE"])) for p in pats))}
gap = (C["Date WLE"] - C["Date AFI"]).dt.days.abs().dropna()
out["aceb_case_dates"] = {"afi_wle_gap_days_median": float(gap.median()), "gap_le_14d": sup((gap <= 14).sum()), "gap_gt_14d": sup((gap > 14).sum()),
                          "year_range": [int(min(C["Date AFI"].min(), C["Date WLE"].min()).year), int(max(C["Date AFI"].max(), C["Date WLE"].max()).year)]}

# ---------------- DB sources (ids + dates only)
def load_path(p, cols):
    d = pd.read_csv(p, usecols=cols, dtype=str)
    # the Feb scrape CSV has unquoted multi-line text: keep only well-formed rows (numeric report and participant ids)
    d = d[d.pathology_text_id.fillna("").str.fullmatch(r"\d+(\.0)?") & d.participant_id.fillna("").str.fullmatch(r"\d+(\.0)?")].copy(); d["pid"] = d.participant_id.map(pid); d["key"] = d.specimennumber.map(acc_key)
    d["d"] = pd.to_datetime(d.receiveddatetime, errors="coerce"); return d
PA = load_path(f"{DB}/pathology_text_normalised_full.csv", ["pathology_text_id", "participant_id", "receiveddatetime", "rowupdatedatetime", "specimennumber"])
PF = load_path(f"{FEB}/view_patient_pathology_text_normalised.csv", ["pathology_text_id", "participant_id", "receiveddatetime", "rowupdatedatetime", "specimennumber"])
def load_endo(p):
    cols = ["participant_id", "endoscopydate", "related_specimennumber"]
    d = pd.read_parquet(p, columns=cols) if p.endswith(".parquet") else pd.read_csv(p, usecols=cols, dtype=str)
    d["pid"] = d.participant_id.map(pid); d["d"] = pd.to_datetime(d.endoscopydate, errors="coerce", dayfirst=not p.endswith(".parquet"))
    d = d[d.participant_id.astype(str).str.fullmatch(r"\d+(\.0)?")].copy(); d.loc[d.d.dt.year < 1950, "d"] = pd.NaT; d["key"] = d.related_specimennumber.map(lambda s: acc_key(s) if pd.notna(s) else None); return d
EA = pd.concat([load_endo(f"{DB}/endoscopy.parquet"), load_endo(f"{DB}/mat_endoscopy_full.parquet")])
EF = load_endo(f"{FEB}/view_patient_endoscopy_full.csv")
out["db_extracts"] = {
    "aug2026_api_export": {"path": "barretts_db_export/ (scripts/export_barretts_db.py, commit b211413)", "files_written": "2026-08-21",
        "pathology_reports": len(PA), "participants_with_reports": int(PA.pid.nunique()),
        "report_received_max": str(PA.d.max().date()), "reports_received_from_2026-06-01": int((PA.d >= "2026-06-01").sum()),
        "report_rowupdate_max": str(pd.to_datetime(PA.rowupdatedatetime, errors="coerce").max().date()),
        "endoscopy_rows": len(EA), "endoscopy_date_max": str(EA.d.max().date())},
    "feb2026_api_scrape": {"path": "barretts_retraining/barretts_training/data/barretts_scrape_20260220_172725/", "scraped": "2026-02-20",
        "pathology_reports": len(PF), "participants_with_reports": int(PF.pid.nunique()), "report_received_max": str(PF.d.max().date()),
        "endoscopy_rows": len(EF), "endoscopy_date_max": str(EF.d.max().date())}}
both = set(PA.pathology_text_id) & set(PF.pathology_text_id)
out["db_extracts"]["report_ids_aug_only"] = len(set(PA.pathology_text_id) - both); out["db_extracts"]["report_ids_feb_only"] = len(set(PF.pathology_text_id) - both)

# ---------------- linkage routes: ACE-B patient -> DB participant
def key_index(*dfs):
    ix = {}
    for d in dfs:
        for k, p in zip(d.key, d.pid):
            if k is not None and p is not None: ix.setdefault(k, set()).add(p)
    return ix
IXA, IXF = key_index(PA, EA), key_index(PF, EF)
routes = {"R2_ps_aug2026": IXA, "R3_ps_feb2026": IXF}
link = {p: {} for p in pats}
for name, ix in routes.items():
    for p, k in zip(S.PatientID, S.key):
        if k in ix: link[p].setdefault(name, set()).update(ix[k])
M1 = pd.read_csv(f"{META}/aceb_rehan_study_number_match_table_20260527_1323.csv", usecols=["study_number_raw", "participant_ids_in_filter"], dtype=str)
for sn, ids in zip(M1.study_number_raw, M1.participant_ids_in_filter):
    if sn in link and pd.notna(ids): link[sn]["R1_studynumber_may2026"] = {pid(x) for x in re.split(r"[|;, ]+", ids) if pid(x)}
M4 = pd.read_csv(f"{META}/aceb_ps_to_participant_mapping_20260527_143713.csv", usecols=["ps_id", "mapped_ids"], dtype=str)
ix4 = {}
for a, ids in zip(M4.ps_id, M4.mapped_ids):
    k = acc_key(a)
    if k and pd.notna(ids): ix4.setdefault(k, set()).update(pid(x) for x in re.split(r"[|;, ]+", ids) if pid(x))
for p, k in zip(S.PatientID, S.key):
    if k in ix4: link[p].setdefault("R4_psmap_may2026", set()).update(ix4[k])
rnames = ["R1_studynumber_may2026", "R2_ps_aug2026", "R3_ps_feb2026", "R4_psmap_may2026"]
rows = []
for p in pats:
    r = link[p]; spec = set().union(*(r.get(n, set()) for n in ["R2_ps_aug2026", "R3_ps_feb2026"]))
    allp = set().union(*r.values()) if r else set()
    # primary participant: specimen routes (DB's own accession numbers); else May study-number route
    cand = spec if spec else r.get("R1_studynumber_may2026", set())
    if len(cand) > 1:  # pick the participant supported by most routes
        cand = {max(cand, key=lambda x: sum(x in r.get(n, set()) for n in rnames))}
    rows.append({"PatientID": p, **{n: "|".join(sorted(r.get(n, set()))) for n in rnames}, "n_candidates": len(allp),
                 "participant_id": next(iter(cand)) if cand else None})
L = pd.DataFrame(rows)
lk = L.participant_id.notna()
out["linkage"] = {"patients": len(L), "linked_any_route": sup(lk.sum()), "not_linked": sup((~lk).sum()),
    "by_route": {n: sup((L[n] != "").sum()) for n in rnames},
    "specimen_route_aug_or_feb": sup(((L.R2_ps_aug2026 != "") | (L.R3_ps_feb2026 != "")).sum()),
    "only_may_studynumber_route": sup(((L.R2_ps_aug2026 == "") & (L.R3_ps_feb2026 == "") & (L.R1_studynumber_may2026 != "")).sum()),
    "patients_with_>1_candidate_participant": sup((L.n_candidates > 1).sum()),
    "R1_vs_specimen_agree": sup(sum(bool(set(a.split("|")) & (set(b.split("|")) | set(c.split("|")))) for a, b, c in zip(L.R1_studynumber_may2026, L.R2_ps_aug2026, L.R3_ps_feb2026) if a and (b or c))),
    "R1_vs_specimen_disagree": sup(sum(not (set(a.split("|")) & (set(b.split("|")) | set(c.split("|")))) for a, b, c in zip(L.R1_studynumber_may2026, L.R2_ps_aug2026, L.R3_ps_feb2026) if a and (b or c))),
    "distinct_participants": int(L.participant_id.dropna().nunique())}
# sample-level presence of the ACE-B accession itself in each extract (import check)
KA, KF = key_index(PA), key_index(PF); inA = S.key.map(lambda k: k in KA); inF = S.key.map(lambda k: k in KF)
out["aceb_samples_not_in_either_extract_by_name_format"] = {f: sup(v) for f, v in Counter(fmt(n) for n, a, b in zip(S.Name, inA, inF) if not (a or b)).items()}
out["aceb_sample_accession_in_pathology"] = {"samples": len(S), "aug2026": sup(inA.sum()), "feb2026": sup(inF.sum()),
    "aug_only": sup((inA & ~inF).sum()), "feb_only": sup((~inA & inF).sum()), "both": sup((inA & inF).sum()), "neither": sup((~inA & ~inF).sum()),
    "patients_with_any_sample_accession_aug": sup(S[inA].PatientID.nunique()), "patients_with_any_sample_accession_feb": sup(S[inF].PatientID.nunique())}
# DB study enrolment (study-name lists; membership only)
sf = pd.read_csv(f"{FEB}/samples_filter_summary.csv", usecols=["participant_id", "study_name"], dtype=str)
vb = pd.read_csv(f"{FEB}/view_participant_basic.csv", usecols=["participant_id", "list_study_name"], dtype=str)
acep_s = set(sf[sf.study_name.fillna("").str.contains(r"ACE", case=False)].participant_id.map(pid)) - {None}
acep_v = set(vb[vb.list_study_name.fillna("").str.contains(r"ACE", case=False)].participant_id.map(pid)) - {None}
names = Counter(n for n in sf.study_name.dropna() if re.search("ACE", n, re.I)) + Counter(x.strip() for v in vb.list_study_name.dropna() for x in re.split(r"[,;|]", v) if re.search("ACE", x, re.I))
lp = set(L.participant_id.dropna())
lab = Counter(re.sub(r"^.*\(", "<study number> (", n) for n in names)  # labels are '<ACE-B study number> (ACE-B)': never output the numbers
out["db_aceb_study_enrolment_feb2026"] = {"distinct_ACE_study_labels": len(names), "label_pattern": sorted(lab), "participants_with_ACE_study_sample": sup(len(acep_s)),
    "participants_with_ACE_in_study_list": sup(len(acep_v)), "linked_aceb_patients_among_them": sup(len(lp & (acep_s | acep_v)))}

# ---------------- history of linked patients (dates only)
def episodes(dates, win=14):
    ds = sorted(set(pd.Timestamp(x).normalize() for x in dates if pd.notna(x))); ep = []
    for d in ds:
        if not ep or (d - ep[-1][-1]).days > win: ep.append([d])
        else: ep[-1].append(d)
    return [e[0] for e in ep]
ER = pd.read_csv(ERIN, usecols=["anon_id", "CollectedOrOrdered", "CaseName"], dtype=str)
ER["d"] = pd.to_datetime(ER.CollectedOrOrdered, dayfirst=True, errors="coerce"); ER["key"] = ER.CaseName.map(acc_key)
erin_ix = {}
out["erin_casename_formats"] = {f: sup(v) for f, v in Counter(fmt(x) for x in ER.CaseName.dropna()).most_common(6)}
for k, a in zip(ER.key, ER.anon_id):
    if k is not None: erin_ix.setdefault(k, set()).add(a)
spec_by_pid = {}
for d in (PA, PF, EA, EF):
    for k, p in zip(d.key, d.pid):
        if k is not None and p in lp: spec_by_pid.setdefault(p, set()).add(k)
recs = []
for _, r in L[lk].iterrows():
    p = r.participant_id
    eD = list(EA.d[EA.pid == p]) + list(EF.d[EF.pid == p]); pD = list(PA.d[PA.pid == p]) + list(PF.d[PF.pid == p])
    ks_all = spec_by_pid.get(p, set()) | {k for k in S.key[S.PatientID == r.PatientID] if k is not None}
    anon = set().union(*(erin_ix.get(k, set()) for k in ks_all))
    nD = list(ER.d[ER.anon_id.isin(anon)])
    ep_db = episodes(eD + pD); ep_all = episodes(eD + pD + nD)
    c = C.loc[r.PatientID] if r.PatientID in C.index else None
    a_lo = min([x for x in ([c["Date AFI"], c["Date WLE"]] if c is not None else []) if pd.notna(x)], default=pd.NaT)
    a_hi = max([x for x in ([c["Date AFI"], c["Date WLE"]] if c is not None else []) if pd.notna(x)], default=pd.NaT)
    src = "case_list"
    if pd.isna(a_lo):  # fallback: DB received date of the sample's own accession
        ks = set(S.key[S.PatientID == r.PatientID]); dd = [d for d, k in zip(PA.d, PA.key) if k in ks] + [d for d, k in zip(PF.d, PF.key) if k in ks]
        dd = [x for x in dd if pd.notna(x)]; a_lo, a_hi, src = (min(dd), max(dd), "db_accession") if dd else (pd.NaT, pd.NaT, "none")
    def ba(ep):
        if pd.isna(a_lo): return None, None
        return sum((a_lo - e).days > 14 for e in ep), sum((e - a_hi).days > 14 for e in ep)
    b_db, f_db = ba(ep_db); b_all, f_all = ba(ep_all)
    recs.append({"PatientID": r.PatientID, "participant_id": p, "n_endo_dates": len(set(pd.Timestamp(x).normalize() for x in eD if pd.notna(x))),
                 "n_report_dates": len(set(pd.Timestamp(x).normalize() for x in pD if pd.notna(x))), "n_episodes_db": len(ep_db),
                 "n_episodes_db_erin": len(ep_all), "erin_linked": bool(anon), "erin_anon_ids": "|".join(sorted(anon)),
                 "first": ep_db[0] if ep_db else pd.NaT, "last": ep_db[-1] if ep_db else pd.NaT,
                 "anchor_lo": a_lo, "anchor_hi": a_hi, "anchor_src": src, "before_db": b_db, "after_db": f_db, "before_all": b_all, "after_all": f_all})
H = pd.DataFrame(recs)
def dist(x):
    x = pd.Series(x).dropna().astype(int)
    return {"1": sup((x == 1).sum()), "2": sup((x == 2).sum()), "3": sup((x == 3).sum()), ">=4": sup((x >= 4).sum()), "0": sup((x == 0).sum()),
            "median": float(x.median()) if len(x) else None, "IQR": [float(x.quantile(.25)), float(x.quantile(.75))] if len(x) else None}
span = ((H["last"] - H["first"]).dt.days / 365.25).dropna()
out["history_linked"] = {"patients": len(H),
    "episodes_db (endoscopy+report dates, 14-day merge)": dist(H.n_episodes_db),
    "endoscopy_table_dates": dist(H.n_endo_dates), "report_dates": dist(H.n_report_dates),
    "episodes_db_plus_erin": dist(H.n_episodes_db_erin), "erin_linked_patients": sup(H.erin_linked.sum()),
    "erin_anon_ids_per_linked_patient": {k: sup(v) for k, v in Counter(H[H.erin_linked].erin_anon_ids.str.count("[|]") + 1).items()},
    "span_years_first_to_last_db": {"median": round(float(span.median()), 1), "IQR": [round(float(span.quantile(.25)), 1), round(float(span.quantile(.75)), 1)],
        "0": sup((span == 0).sum()), "(0,5]": sup(((span > 0) & (span <= 5)).sum()), "(5,10]": sup(((span > 5) & (span <= 10)).sum()), ">10": sup((span > 10).sum())},
    "first_year_range": [int(H["first"].min().year), int(H["first"].max().year)], "last_year_range": [int(H["last"].min().year), int(H["last"].max().year)],
    "anchor_source": {k: sup(v) for k, v in H.anchor_src.value_counts().items()}}
for tag in ["db", "all"]:
    b, f = H[f"before_{tag}"].dropna(), H[f"after_{tag}"].dropna()
    out["history_linked"][f"relative_to_aceb_date_{tag}"] = {"patients_with_anchor": len(b),
        "before>=1": sup((b >= 1).sum()), "before>=2": sup((b >= 2).sum()), "before>=3": sup((b >= 3).sum()),
        "after>=1": sup((f >= 1).sum()), "after>=2": sup((f >= 2).sum()), "after>=3": sup((f >= 3).sum()),
        "before>=1_and_after>=1": sup(((b >= 1) & (f >= 1)).sum()), "before>=2_or_(before>=1_and_after>=1)": sup(((b >= 2) | ((b >= 1) & (f >= 1))).sum()),
        "total_episodes>=3_incl_aceb": sup(((b + f) >= 2).sum())}
# import check for linked patients: reports updated/received after 2026-06-01, and Aug-only report ids
la = PA[PA.pid.isin(lp)]; lf = PF[PF.pid.isin(lp)]
out["import_check_linked"] = {"reports_aug": len(la), "reports_feb": len(lf), "reports_aug_only": len(set(la.pathology_text_id) - set(lf.pathology_text_id)),
    "reports_rowupdate_from_2026-06-01": sup((pd.to_datetime(la.rowupdatedatetime, errors="coerce") >= "2026-06-01").sum()),
    "reports_received_from_2026-06-01": sup((la.d >= "2026-06-01").sum())}

# ---------------- overlaps
pe = pd.read_csv(f"{REL}/pre_event_cohort.csv", usecols=["PatientID", "PatientID_real", "participant_id"], dtype=str)
pe["pid"] = pe.participant_id.map(pid)
sc = pd.read_csv(f"{KM}/set_C.csv", usecols=["Patient"], dtype=str); disc = set(sc.Patient.dropna())
m_disc = pe[pe.PatientID.isin(disc) | pe.PatientID_real.isin(disc)]
disc_pid = set(m_disc.pid.dropna()); rel_pid = set(pe.pid.dropna())
pairs = pd.read_csv(f"{T}/feasibility/closeout/erin_swg_pairs.csv", usecols=["erin_anon_id", "swg_patient_id"], dtype=str)
sw = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", nrows=0).columns
idc = [c for c in sw if re.search(r"patient|study|hospital|research|alternate", c, re.I)]
sw = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", usecols=idc, dtype=str)
pset = set(pairs.swg_patient_id.dropna())
out["pairs_id_probe"] = {"pairs_rows": len(pairs), "in_pre_event_PatientID": len(pset & set(pe.PatientID)), "in_pre_event_PatientID_real": len(pset & set(pe.PatientID_real)),
                         "in_setC_Patient": len(pset & disc), **{f"in_777_{c}": len(pset & set(sw[c].dropna())) for c in idc}}
# discovery patients -> study numbers through the 777 sheet rows that carry the set C Patient id
best = max(idc, key=lambda c: len(pset & set(sw[c].dropna())))
pcol = max(idc, key=lambda c: len(disc & set(sw[c].dropna())))
disc_sn = set(sw[sw[pcol].isin(disc)][best].dropna()) | (disc & pset)
out["pairs_id_probe"].update({"study_number_col": best, "setC_id_col": pcol, "smoke_discovery_in_pairs_via_777": len(pset & disc_sn)})
erin_lk = set(H.erin_anon_ids.str.split("|").explode().replace("", np.nan).dropna())
kd = pd.read_csv(f"{KM}/two_papers_ERL/scan_request_DB_final_KEY.csv", usecols=["pid"], dtype=str); kd_p = set(kd.pid.map(pid))
ke = pd.read_csv(f"{KM}/two_papers_ERL/scan_request_ERIN_final_KEY.csv", usecols=["anon_id"], dtype=str); ke_a = set(ke.anon_id)
hp = H.set_index("PatientID")
out["overlap"] = {
    "smoke_discovery_patients": len(disc), "smoke_discovery_with_participant_id": len(disc & (set(m_disc[m_disc.pid.notna()].PatientID) | set(m_disc[m_disc.pid.notna()].PatientID_real))),
    "smoke_discovery_in_erin_swg_pairs (via 777 sheet study number)": out["pairs_id_probe"]["smoke_discovery_in_pairs_via_777"],
    "release_distinct_participant_ids": len(rel_pid),
    "aceb_patientid_string_in_discovery_or_pairs": sup(len(set(pats) & (disc | set(pairs.swg_patient_id)))),
    "aceb_linked_in_discovery_80": sup(len(lp & disc_pid)), "aceb_linked_in_release_150_by_participant_id": sup(len(lp & rel_pid)),
    "aceb_erin_anon_in_erin_swg_pairs": sup(len(erin_lk & set(pairs.erin_anon_id))),
    "aceb_erin_anon_in_pairs_discovery80": sup(len(erin_lk & set(pairs[pairs.swg_patient_id.isin(disc_sn)].erin_anon_id))),
    "aceb_patients_linked_to_discovery80_any_route": sup(sum(bool((r.participant_id in disc_pid) or (set(r.erin_anon_ids.split("|")) & set(pairs[pairs.swg_patient_id.isin(disc_sn)].erin_anon_id))) for r in H.itertuples())),
    "aceb_patients_linked_to_release_any_route": sup(sum(bool((r.participant_id in rel_pid) or (set(r.erin_anon_ids.split("|")) & set(pairs.erin_anon_id))) for r in H.itertuples())),
    "aceb_linked_on_ERL_DB_scan_request_list": sup(len(lp & kd_p)), "ERL_DB_list_patients": len(kd_p),
    "aceb_linked_on_ERL_ERIN_scan_request_list": sup(len(erin_lk & ke_a)), "ERL_ERIN_list_patients": len(ke_a),
    "note": "list membership counted overall only; ERL role column not read"}
L.to_csv(f"{W}/aceb_db_link.csv", index=False); H.to_csv(f"{W}/aceb_db_history.csv", index=False)
for f in ("aceb_db_link.csv", "aceb_db_history.csv"): os.chmod(f"{W}/{f}", 0o600)
json.dump(out, open(f"{R}/aceb_db_linkage.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
