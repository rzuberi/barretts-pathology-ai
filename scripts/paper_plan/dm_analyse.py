"""Demographics completion D1.2-D1.4, D3, D4 (docs/demographics_completion.md @ 24b2972). Report only; no imputation.
PART=d1 | d3. Aggregates -> results/paper_final/demographics/<PART>.json; row-level tables on the cluster under feasibility/paper_plan/demographics/.
ACE-B: only identifier, sex and date-of-birth columns are read (pandas usecols), plus initial_history demographic fields; no pathology, outcome or
follow-up field is read for ACE-B (date_of_death, Pathology, Histology, dates of AFI/WLE, endoscopy reports are never loaded)."""
import os, re, json, glob, sqlite3, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
AM = "/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta"; K = T + "/feasibility/paper_plan/killcoyne"; M = K + "_mm"; D = T + "/feasibility/paper_plan/demographics"; O = T + "/results/paper_final/demographics"
os.makedirs(D, exist_ok=True); os.makedirs(O, exist_ok=True); PART = os.environ["PART"]
NULL = {"", "None", "nan", "NaN", "NULL", "null"}
isnull = lambda v: v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in NULL
r2 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 2)
def desc(v):
    a = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(a): return {"n": 0}
    q1, md, q3 = np.percentile(a, [25, 50, 75]); return {"n": int(len(a)), "mean": r2(a.mean()), "median": r2(md), "q1": r2(q1), "q3": r2(q3), "min": r2(a.min()), "max": r2(a.max())}
def missrep(series):
    s = pd.Series(series, dtype=object); return {"NaN_or_None_object": int(s.map(lambda v: v is None or (isinstance(v, float) and np.isnan(v))).sum()), "empty_string": int((s.astype(str) == "").sum()),
            "string_None": int((s.astype(str) == "None").sum()), "zero": int(s.astype(str).isin(["0", "0.0"]).sum()), "n": int(len(s))}
# ---------- shared sources
IH = pd.read_parquet(E + "/initial_history.parquet"); IH["pid"] = IH.participant_id.astype(str).str.replace(r"\.0$", "", regex=True)
pe = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str); pid_of = pe.dropna(subset=["participant_id"]).groupby("PatientID_real").participant_id.agg(lambda s: s.str.replace(r"\.0$", "", regex=True).mode().iloc[0])
dm = pd.read_csv(S + "/Demographics_full.csv", dtype=str)
for c in ("Study Number", "Alternate Study Number"): dm[c] = dm[c].str.strip()
dmi = {}
for _, r in dm.iterrows():
    for c in ("Study Number", "Alternate Study Number"):
        if isinstance(r[c], str) and r[c] and r[c] not in dmi: dmi[r[c]] = r
con = sqlite3.connect(f"file:{AM}/BE_Progression_Project.db?mode=ro", uri=True)
SQ = pd.read_sql("select ID, Gender, AgeAtDiagnosis, SmokingStatus, Height, Weight, DateInitialDiagnosis, Circumference, Maximal from Patient", con)   # Status, DateProgressed not read
ALT = pd.read_sql("select PatientID, AlternateID, AltIDStudy from AlternatePatientID", con); ALT["AlternateID"] = ALT.AlternateID.astype(str).str.strip()
sq_of = dict(zip(ALT.AlternateID, ALT.PatientID)); SQi = SQ.set_index("ID")
def ih_values(pid, col):
    if pid is None: return "no_participant_id", None
    rows = IH[IH.pid == str(pid)]
    if not len(rows): return "no_initial_history_row", None
    v = sorted(set(str(x).strip() for x in rows[col] if not isnull(x)))
    return ("blank", None) if not v else (("conflict", v) if len(v) > 1 else ("value", v[0]))
res = {"part": PART}
if PART == "d1":
    # ---- D1.2 cross-tab over every Demographics_full patient that links to the database
    rows = []
    for _, r in dm.iterrows():
        keys = [k for k in (r["Study Number"], r["Alternate Study Number"]) if isinstance(k, str) and k]
        pid = next((pid_of[k] for k in keys if k in pid_of.index), None); how = "pre_event_cohort.participant_id" if pid else None
        if pid is None:
            for col in ("hospitalnumber", "patient_anonref"):
                m = IH[IH[col].astype(str).str.strip().isin(keys)]
                if len(m): pid, how = m.pid.iloc[0], f"initial_history.{col}"; break
        st, v = ih_values(pid, "smoking"); yn = r["Smoking Status"].strip() if isinstance(r["Smoking Status"], str) and r["Smoking Status"].strip() in ("Y", "N") else "missing"
        sqid = next((sq_of[k] for k in keys if k in sq_of), None); sqs = SQi.SmokingStatus.get(sqid) if sqid is not None else None
        rows.append({"study": keys[0] if keys else None, "link": how or "not linked", "db_status": st, "db_code": v if st == "value" else ("conflict:" + "/".join(v) if st == "conflict" else st), "yn": yn,
                     "sqlite_smoking": "no sqlite link" if sqid is None else ("None" if isnull(sqs) else str(sqs))})
    X = pd.DataFrame(rows); X.to_csv(D + "/d1_crosstab_rows.csv", index=False)
    res["demographics_rows"] = int(len(X)); res["linkage"] = X.link.value_counts().to_dict()
    res["crosstab_dbcode_by_YN"] = pd.crosstab(X.db_code, X.yn).to_dict(); res["crosstab_n"] = int(((X.db_status == "value") & (X.yn != "missing")).sum())
    res["secondary_not_prespecified"] = {"crosstab_dbcode_by_sqlite_SmokingStatus": pd.crosstab(X.db_code, X.sqlite_smoking).to_dict(), "crosstab_YN_by_sqlite_SmokingStatus": pd.crosstab(X.yn, X.sqlite_smoking).to_dict(),
                                         "note": "BE_Progression_Project.db Patient.SmokingStatus (never/former/current) linked through AlternatePatientID.AlternateID = study number; not pre-specified"}
    # ---- D1.3 default-value test
    nn = IH.apply(lambda r: sum(not isnull(v) for v in r.values), axis=1); rk = nn.rank(method="first"); terc = pd.qcut(rk, 3, labels=["low", "mid", "high"])
    def share0(col, mask):
        s = IH.loc[mask, col].astype(str).str.strip(); return {"n": int(mask.sum()), "share_0": r2(100 * (s == "0").mean()), "share_null": r2(100 * s.isin(NULL).mean())}
    sm = {t: share0("smoking", terc == t) for t in ("low", "mid", "high")}
    res["d1_3_prespecified"] = {"nonnull_fields_per_record": desc(nn.values), "tercile_field_count_ranges": {t: [int(nn[terc == t].min()), int(nn[terc == t].max())] for t in ("low", "mid", "high")}, "smoking": sm,
                                "diff_low_minus_high_pp": r2(sm["low"]["share_0"] - sm["high"]["share_0"]), "zero_is_default_by_rule": bool(sm["low"]["share_0"] - sm["high"]["share_0"] >= 20)}
    rec = ~IH.smoking.astype(str).str.strip().isin(NULL)   # sensitivity, not pre-specified: records with a smoking value
    rk2 = nn[rec].rank(method="first"); t2 = pd.Series(pd.qcut(rk2, 3, labels=["low", "mid", "high"]), index=nn[rec].index)
    sm2 = {t: share0("smoking", IH.index.isin(t2[t2 == t].index)) for t in ("low", "mid", "high")}
    res["d1_3_sensitivity_records_with_smoking_value"] = {"n": int(rec.sum()), "smoking": sm2, "diff_low_minus_high_pp": r2(sm2["low"]["share_0"] - sm2["high"]["share_0"]), "note": "not pre-specified"}
    other = {}
    for c in IH.columns:
        if c in ("smoking", "pid") or re.search(r"_pk$|_fk$|participant|anonref|hospitalnumber|date|timestamp|^name$|surname|firstname", c): continue
        s = IH[c].astype(str).str.strip(); s = s[~s.isin(NULL)]
        if len(s) < 30 or pd.to_numeric(s, errors="coerce").notna().mean() < 0.9: continue
        a = {t: share0(c, terc == t) for t in ("low", "high")}; other[c] = {"n_nonnull": int(len(s)), "mode": s.mode().iloc[0], "zero_is_mode": bool(s.mode().iloc[0] in ("0", "0.0")), "share_0_low": a["low"]["share_0"], "share_0_high": a["high"]["share_0"], "share_0_overall_nonnull": r2(100 * s.isin(["0", "0.0"]).mean())}
    res["d1_3_other_numeric_columns"] = other
    # ---- D1.4 decision rule
    zdef = res["d1_3_prespecified"]["zero_is_default_by_rule"]
    Ov = X[(X.db_status == "value") & (X.yn != "missing")].copy()
    if zdef: Ov = Ov[Ov.db_code != "0"]
    mapping, tie = {}, False
    for c, g in Ov.groupby("db_code"):
        vc = g.yn.value_counts()
        if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]: tie = True
        else: mapping[c] = vc.index[0]
    agree = float((Ov.db_code.map(mapping) == Ov.yn).mean()) if len(Ov) else float("nan")
    accept = (len(Ov) >= 20) and (not tie) and (agree >= 0.90)
    res["d1_4_rule"] = {"overlap_n": int(len(Ov)), "zero_treated_as_blank": zdef, "candidate_mapping": mapping, "tie": tie, "agreement": r2(100 * agree) if len(Ov) else None, "accepted": bool(accept),
                        "status": "RESOLVED" if accept else "NOT RESOLVED"}
elif PART == "d3":
    # ---------- cohorts
    C = pd.read_csv(M + "/set_C.csv", dtype=str); fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]
    fdi = fd.drop_duplicates("combined_name").set_index("combined_name"); C["hrid"] = C.Sample.map(fdi["Hospital_Research_ID_updated"].str.strip()); C["mbf"] = pd.to_numeric(C.Sample.map(fdi["Months before final"]), errors="coerce")
    pe["cnv"] = pe.CNVAbsPath.map(lambda p: os.path.basename(str(p))); pe["d"] = pd.to_datetime(pe.Date, errors="coerce"); pdate = pe.dropna(subset=["d"]).drop_duplicates("cnv").set_index("cnv").d
    cf = C.sort_values(["Patient", "mbf"], ascending=[True, False]).groupby("Patient").first()
    disc = pd.DataFrame({"key": [C.loc[C.Patient == p, "hrid"].dropna().mode().iloc[0] for p in cf.index], "label": ["P" if s == "P" else "NP" for s in cf.Status], "first_date": [pdate.get(s, pd.NaT) for s in cf.Sample], "kpatient": cf.index})
    tm = pd.read_csv(F + "/training_manifest.csv", dtype=str); rel = pe[pe.SampleID.isin(tm.sample_id)].copy(); rel["y"] = rel.SampleID.map(tm.set_index("sample_id").y_progressor.astype(int))
    rel["d"] = pd.to_datetime(rel.Date, errors="coerce"); rg = rel.groupby("PatientID_real")
    relc = pd.DataFrame({"key": rg.y.max().index, "label": np.where(rg.y.max().values == 1, "P", "NP"), "first_date": rg.d.min().values})
    for X_ in (disc, relc): X_["pid"] = X_.key.map(pid_of)
    am = pd.read_csv(f"{AM}/ACEB_samples_for Rehan.csv", dtype=str, usecols=["PatientID"]); ac = pd.DataFrame({"key": sorted(am.PatientID.dropna().str.strip().unique())}); ac["label"] = "all"; ac["first_date"] = pd.NaT
    mt = pd.read_csv(sorted(glob.glob(f"{AM}/aceb_rehan_study_number_match_table_*.csv"))[-1], dtype=str, usecols=["study_number_raw", "study_number_normalized", "participant_ids_in_filter"])
    norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
    mp = {}
    for _, r in mt.iterrows():
        ids = [x for x in re.split(r"[;,| ]+", str(r.participant_ids_in_filter)) if x and x not in NULL]
        for k in (r.study_number_raw, r.study_number_normalized):
            if isinstance(k, str) and ids: mp[norm(k)] = ids
    ac["pids_all"] = ac.key.map(lambda k: mp.get(norm(k))); ac["pid"] = ac.pids_all.map(lambda v: v[0] if isinstance(v, list) and len(v) == 1 else None)
    ac["pid_status"] = ac.pids_all.map(lambda v: "none" if not isinstance(v, list) else ("one" if len(v) == 1 else "several"))
    part = pd.read_csv(sorted(glob.glob(f"{AM}/aceb_sample_counts_per_patient_*.csv"))[-1], dtype=str, usecols=["participant_id", "gender", "date_of_birth"]).drop_duplicates("participant_id").set_index("participant_id")
    EN = pd.read_parquet(E + "/endoscopy.parquet", columns=["participant_id", "endoscopydate", "barretts_circumference", "barretts_maximum", "hiatusherniapresent"]); EN["pid"] = EN.participant_id.astype(str).str.replace(r"\.0$", "", regex=True); EN["ed"] = pd.to_datetime(EN.endoscopydate, errors="coerce")
    num = lambda v: pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    def row(r, cohort):
        k, pid = r.key, (r.pid if isinstance(r.pid, str) else None); o = {"cohort": cohort, "label": r.label, "has_pid": pid is not None}
        d = dmi.get(k) if cohort != "aceb" else None; sqid = sq_of.get(k) if cohort != "aceb" else None; sq = SQi.loc[sqid] if sqid is not None and sqid in SQi.index else None
        st, dob = ih_values(pid, "dob"); dob_db = pd.to_datetime(dob, errors="coerce") if st == "value" else pd.NaT; o["dob_db_status"] = st
        st, ddx = ih_values(pid, "datediagnosed"); ddx = pd.to_datetime(ddx, errors="coerce") if st == "value" else pd.NaT; o["datediagnosed_status"] = st
        dob_sheet = pd.to_datetime(d["Date of birth"], errors="coerce") if d is not None else pd.NaT
        if cohort == "aceb" and pid is not None and pid in part.index:
            o["sex_participant_table"] = None if isnull(part.gender[pid]) else str(part.gender[pid]); pdob = pd.to_datetime(part.date_of_birth[pid], errors="coerce")
            if pd.isna(dob_db): dob_db = pdob; o["dob_source"] = "aceb_sample_counts_per_patient.date_of_birth"
        fdt = r.first_date
        o["age_first_sample_sheet"] = (fdt - dob_sheet).days / 365.25 if pd.notna(fdt) and pd.notna(dob_sheet) else np.nan
        o["age_first_sample_db"] = (fdt - dob_db).days / 365.25 if pd.notna(fdt) and pd.notna(dob_db) else np.nan
        o["age_dx_db"] = (ddx - dob_db).days / 365.25 if pd.notna(ddx) and pd.notna(dob_db) else np.nan
        o["age_dx_sheet"] = num(d["Age at diagnosis"]) if d is not None else np.nan
        o["age_dx_sqlite"] = num(sq.AgeAtDiagnosis) if sq is not None else np.nan
        o["sex_sheet"] = d["Sex"].strip() if d is not None and isinstance(d["Sex"], str) and d["Sex"].strip() else None
        o["sex_sqlite"] = None if sq is None or isnull(sq.Gender) else str(sq.Gender)
        o["prague_c_sheet"] = num(d["Circumference "]) if d is not None else np.nan; o["prague_m_sheet"] = num(d["Maximal"]) if d is not None else np.nan
        o["prague_c_sqlite"] = num(sq.Circumference) if sq is not None else np.nan; o["prague_m_sqlite"] = num(sq.Maximal) if sq is not None else np.nan
        for v, col in (("c", "barretts_circumference"), ("m", "barretts_maximum")):
            o[f"prague_{v}_db"] = np.nan; o[f"prague_{v}_gap"] = np.nan
            if pid is not None and pd.notna(fdt):
                e = EN[(EN.pid == pid) & EN.ed.notna()].copy(); e["val"] = pd.to_numeric(e[col], errors="coerce"); e = e[e.val.notna()]
                if len(e): e["gap"] = (e.ed - fdt).abs().dt.days; e = e.sort_values("gap"); o[f"prague_{v}_db"] = float(e.val.iloc[0]); o[f"prague_{v}_gap"] = float(e.gap.iloc[0])
            st, iv = ih_values(pid, f"prague{v}"); o[f"prague_{v}_ih"] = num(iv) if st == "value" else np.nan
        o["hiatal_hernia_db"] = None; o["hiatal_hernia_gap"] = np.nan
        if pid is not None and pd.notna(fdt):
            e = EN[(EN.pid == pid) & EN.ed.notna() & ~EN.hiatusherniapresent.astype(str).str.strip().isin(NULL)].copy()
            if len(e): e["gap"] = (e.ed - fdt).abs().dt.days; e = e.sort_values("gap"); o["hiatal_hernia_db"] = {"1": "yes", "true": "yes", "0": "no", "false": "no"}.get(str(e.hiatusherniapresent.iloc[0]).strip().lower(), str(e.hiatusherniapresent.iloc[0])); o["hiatal_hernia_gap"] = float(e.gap.iloc[0])
        for col, nm in (("smoking", "smoking_db_code"), ("packyears", "packyears_db"), ("bmi", "bmi_db"), ("heightm", "height_db"), ("weightkg", "weight_db"), ("alcoholcurrent", "alcohol_db"), ("ppi", "ppi_db_code"), ("fambarretts", "fam_be_db"), ("famoescancer", "fam_oac_db")):
            st, v = ih_values(pid, col); o[nm] = v if st == "value" else (f"conflict" if st == "conflict" else None); o[nm + "_status"] = st
        o["smoking_sheet"] = d["Smoking Status"].strip() if d is not None and isinstance(d["Smoking Status"], str) and d["Smoking Status"].strip() else None
        o["smoking_sqlite"] = None if sq is None or isnull(sq.SmokingStatus) else str(sq.SmokingStatus)
        o["height_sqlite"] = num(sq.Height) if sq is not None else np.nan; o["weight_sqlite"] = num(sq.Weight) if sq is not None else np.nan
        return o
    R = pd.DataFrame([row(r, "discovery") for r in disc.itertuples()] + [row(r, "release") for r in relc.itertuples()] + [row(r, "aceb") for r in ac.itertuples()])
    bmi = pd.to_numeric(R.bmi_db, errors="coerce"); R["bmi_db_num"] = bmi.where((bmi > 10) & (bmi < 80))   # implausible values (0, negatives) reported, not used
    R["bmi_sqlite"] = R.weight_sqlite / (R.height_sqlite / (100 if (R.height_sqlite > 3).any() else 1)) ** 2
    R.to_csv(D + "/d3_patient_rows.csv", index=False)
    res["aceb_linkage"] = {"patients": int(len(ac)), "pid_status": ac.pid_status.value_counts().to_dict(), "mapping_file": sorted(glob.glob(f"{AM}/aceb_rehan_study_number_match_table_*.csv"))[-1]}
    res["discovery_linkage"] = {"patients": int(len(disc)), "with_pid": int(disc.pid.notna().sum()), "in_sqlite": int(sum(k in sq_of for k in disc.key))}
    res["release_linkage"] = {"patients": int(len(relc)), "with_pid": int(relc.pid.notna().sum()), "in_sqlite": int(sum(k in sq_of for k in relc.key))}
    NUMV = ["age_first_sample_sheet", "age_first_sample_db", "age_dx_sheet", "age_dx_db", "age_dx_sqlite", "prague_c_sheet", "prague_c_db", "prague_c_ih", "prague_c_sqlite", "prague_m_sheet", "prague_m_db", "prague_m_ih", "prague_m_sqlite", "prague_c_gap", "prague_m_gap", "hiatal_hernia_gap", "bmi_db_num", "bmi_sqlite"]
    CATV = ["sex_sheet", "sex_sqlite", "sex_participant_table", "smoking_sheet", "smoking_db_code", "smoking_sqlite", "packyears_db", "ppi_db_code", "fam_be_db", "fam_oac_db", "hiatal_hernia_db"]
    out = {}
    for coh in ("discovery", "release", "aceb"):
        G = R[R.cohort == coh]; out[coh] = {}
        for lab in (("all", "P", "NP") if coh != "aceb" else ("all",)):
            g = G if lab == "all" else G[G.label == lab]; o = {"N": int(len(g)), "with_participant_id": int(g.has_pid.sum())}
            for v in NUMV:
                if v in g: o[v] = desc(pd.to_numeric(g[v], errors="coerce").values)
            for v in ("prague_c", "prague_m"): o[f"{v}_db_gap_gt_365"] = int((pd.to_numeric(g[f"{v}_gap"], errors="coerce") > 365).sum())
            for v in CATV:
                if v in g: o[v] = {"n": int(g[v].notna().sum()), "counts": g[v].dropna().astype(str).value_counts().head(15).to_dict()}
            o["alcohol_db_recorded"] = int(g.alcohol_db.notna().sum()); o["bmi_db_implausible"] = int((pd.to_numeric(g.bmi_db, errors="coerce").notna() & g.bmi_db_num.isna()).sum())
            o["conflicts"] = {c: int((g[c + "_status"] == "conflict").sum()) for c in ("smoking_db_code", "bmi_db", "ppi_db_code", "fam_be_db")}
            out[coh][lab] = o
    res["coverage"] = out
    sub = IH[IH.pid.isin(set(pd.concat([disc.pid, relc.pid, ac.pid]).dropna()))]
    res["missing_representation_initial_history_rows_of_cohort_patients"] = {c: missrep(sub[c].values) for c in ("dob", "datediagnosed", "smoking", "packyears", "praguec", "praguem", "bmi", "heightm", "weightkg", "currentbmi", "alcoholcurrent", "ppi", "fambarretts", "famoescancer")}
    res["missing_representation_endoscopy"] = {c: missrep(EN[c].values) for c in ("barretts_circumference", "barretts_maximum", "hiatusherniapresent")}
    res["missing_representation_sqlite_patient"] = {c: missrep(SQ[c].values) for c in ("Gender", "AgeAtDiagnosis", "SmokingStatus", "Height", "Weight", "Circumference", "Maximal")}
    res["sex_in_db_tables"] = "no sex/gender column in any Barrett's DB export table except OCCAMS masterlist _131_gender (OCCAMS cohort, no participant_id link); sex from BE_Progression_Project.db Patient.Gender (SWG) and aceb_sample_counts_per_patient.gender (participant table scrape, ACE-B)"
json.dump(res, open(f"{O}/{PART}.json", "w"), indent=1, default=str); print("DM ANALYSE DONE", PART)
