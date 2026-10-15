"""Dataset description (docs/dataset_description.md): aggregate descriptors of the frozen SWG release, the Killcoyne-protocol set C and the ACE-B manifest.
Report only: nothing is fitted. Usage: python dd_describe.py PART, PART in core | wsi_release | wsi_setc | swgs. Writes results/paper_final/dataset_description/PART.json (aggregates only)."""
import glob, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

PART = sys.argv[1]
TH = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training"
F = B + "/analysis/chapter1_lgd2_final_pre_event_20260713_final"
S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
K = TH + "/feasibility/paper_plan/killcoyne"; M = K + "_mm"
ST = "/mnt/scratche/slow/fmlab/zuberi01/envs/killcoyne_r/bin/samtools"
OUT = TH + "/results/paper_final/dataset_description"; os.makedirs(OUT, exist_ok=True)
S777 = S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv"; SVAL = S + "/sWGS_validation_cleaned_Leanne (4) (1).csv"
remap = lambda p: p.replace("/scratchc/", "/mnt/scratche/fast/") if isinstance(p, str) and p.startswith("/scratchc/") else p
truthy = lambda v: str(v).strip().lower() in {"1", "1.0", "true"}

def desc(v, nd=3):
    a = np.asarray([x for x in v if x is not None and pd.notna(x)], float)
    if len(a) == 0: return {"n": 0}
    q1, md, q3 = np.percentile(a, [25, 50, 75])
    return {"n": int(len(a)), "mean": round(float(a.mean()), nd), "median": round(float(md), nd), "q1": round(float(q1), nd), "q3": round(float(q3), nd), "min": round(float(a.min()), nd), "max": round(float(a.max()), nd)}

def by_groups(P, col, sheets=("all", "777", "268")):
    """P: patient table with label (P/NP), sheet; returns {sheet|label: desc(col)}."""
    out = {}
    for sh in sheets:
        for lab in ("all", "P", "NP"):
            d = P if sh == "all" else P[P.sheet == sh]
            d = d if lab == "all" else d[d.label == lab]
            out[f"{sh}|{lab}"] = desc(d[col]); out[f"{sh}|{lab}"]["single"] = int((d[col] == 1).sum())
    return out

# ---------------- shared: release rows, sheets, set C
tm = pd.read_csv(F + "/training_manifest.csv", dtype=str)
pe = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str)
mm = pd.read_csv(F + "/matched_manifest.csv", dtype=str)
rel = pe[pe.SampleID.isin(tm.sample_id)].copy(); assert len(rel) == len(tm) == 707, (len(rel), len(tm))
rel = rel.merge(mm[["sample_id", "biopsy_id", "slide_id", "cnv_id"]].drop_duplicates("sample_id"), left_on="SampleID", right_on="sample_id", how="left"); assert rel.cnv_id.notna().all() and rel.slide_id.notna().all() and len(rel) == 707
rel["y"] = rel.SampleID.map(tm.set_index("sample_id").y_progressor.astype(int)); rel["Date_dt"] = pd.to_datetime(rel.Date, errors="coerce")
fd = pd.read_csv(S777, dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]
va = pd.read_csv(SVAL, dtype=str); va.columns = [c.strip().replace("\n", " ") for c in va.columns]
ids777 = set(fd["Hospital_Research_ID_updated"].dropna().str.strip()) | set(fd["Hospital Research ID"].dropna().str.strip())
idsval = set(va.PatientID.dropna().str.strip())
def sheet_of(p):
    a, b = p in ids777, p in idsval
    return "777" if a and not b else "268" if b and not a else "both" if a and b else "none"
# sheet per sequencing sample: in the 777 sheet (combined_name), else in the validation CNV folders (val/); a patient takes its samples' sheet ("both" if mixed)
cnv777 = set(fd.combined_name.dropna().str.strip())
isval = lambda c: os.path.isdir(f"{S}/copy_number_hg38/val/500kb/{c}") or os.path.isdir(f"{S}/copy_number_hg38/val/perPatient/500kb/{c}")
rel["sheet_s"] = ["777" if c in cnv777 else "268" if isval(c) else "none" for c in rel.cnv_id]
RP = rel.groupby("PatientID_real").agg(y=("y", "max"), sheets=("sheet_s", lambda s: "/".join(sorted(set(s))))).reset_index().rename(columns={"PatientID_real": "patient_id"})
RP["label"] = np.where(RP.y == 1, "P", "NP"); RP["sheet"] = RP.sheets.map(lambda x: x if x in ("777", "268") else "both" if x == "268/777" else x)
RP["sheet_id"] = RP.patient_id.map(sheet_of)
st = pd.read_csv(TH + "/feasibility/paper_plan/f2_strata.csv", dtype=str).set_index("patient_id")
RP["sheet_f2"] = RP.patient_id.map(st.stratum).map(lambda s: "777" if str(s).startswith("discovery") else "268" if s == "validation_sheet" else "none")
C = pd.read_csv(M + "/set_C.csv", dtype=str); C["mbf"] = pd.to_numeric(C.Sample.map(fd.drop_duplicates("combined_name").set_index("combined_name")["Months before final"]), errors="coerce")
CP = C.groupby("Patient").agg(status=("Status", "first")).reset_index(); CP["label"] = np.where(CP.status == "P", "P", "NP"); CP["sheet"] = "777"
C["hrid"] = C.Sample.map(fd.drop_duplicates("combined_name").set_index("combined_name")["Hospital_Research_ID_updated"].str.strip())
res = {"part": PART}

if PART == "core":
    # ---------- item 1
    i1 = {"release": {}, "setC": {}}
    for sh in ("all", "777", "268", "both", "none"):
        d = RP if sh == "all" else RP[RP.sheet == sh]
        i1["release"][sh] = {"patients": int(len(d)), "P": int((d.label == "P").sum()), "NP": int((d.label == "NP").sum())}
    i1["release_sheet_vs_f2_strata"] = {"agree": int((RP.sheet == RP.sheet_f2).sum()), "n": int(len(RP))}
    i1["release_sheet_vs_patient_id_membership"] = {"agree": int((RP.sheet == RP.sheet_id).sum()), "n": int(len(RP)), "id_membership_counts": RP.sheet_id.value_counts().to_dict()}
    i1["release_samples_by_sheet"] = rel.sheet_s.value_counts().to_dict()
    plab = rel.groupby("PatientID_real").Progressor_label.apply(lambda s: int(any(truthy(v) for v in s)))
    i1["release_Progressor_label_patients"] = {"P": int(plab.sum()), "n": int(len(plab))}
    i1["setC"]["all"] = {"patients": int(len(CP)), "P": int((CP.label == "P").sum()), "NP": int((CP.label == "NP").sum())}
    i1["setC_samples_in_777_sheet"] = {"in": int(C.Sample.isin(set(fd.combined_name)).sum()), "n": int(len(C))}
    i1["setC_patients_with_release_rows"] = int(len(set(C.hrid.dropna()) & set(RP.patient_id)))
    res["item1"] = i1
    # ---------- item 2
    g = rel.groupby("PatientID_real").agg(rows=("SampleID", "nunique"), biopsies=("biopsy_id", "nunique"), slides=("slide_id", "nunique"), swgs=("cnv_id", "nunique")).reset_index().rename(columns={"PatientID_real": "patient_id"})
    RP2 = RP.merge(g, on="patient_id")
    i2 = {"release": {c: by_groups(RP2, c) for c in ("rows", "biopsies", "slides", "swgs")}}
    i2["release_single"] = {c: {"all": int((RP2[c] == 1).sum()), "P": int(((RP2[c] == 1) & (RP2.label == "P")).sum()), "NP": int(((RP2[c] == 1) & (RP2.label == "NP")).sum()), "n": int(len(RP2))} for c in ("rows", "biopsies", "slides", "swgs")}
    mp = pd.read_csv(K + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(C.Sample)]; mp["Patient"] = mp.cnv.map(C.drop_duplicates("Sample").set_index("Sample").Patient)
    gc = C.groupby("Patient").agg(swgs=("Sample", "nunique")).reset_index(); gc["slides"] = gc.Patient.map(mp.groupby("Patient").npz.nunique()); CP2 = CP.merge(gc, on="Patient")
    i2["setC"] = {c: by_groups(CP2, c, sheets=("all",)) for c in ("swgs", "slides")}
    i2["setC_single"] = {c: {"all": int((CP2[c] == 1).sum()), "P": int(((CP2[c] == 1) & (CP2.label == "P")).sum()), "NP": int(((CP2[c] == 1) & (CP2.label == "NP")).sum()), "n": int(len(CP2))} for c in ("swgs", "slides")}
    res["item2"] = i2
    # ---------- item 3 (years)
    span = rel.groupby("PatientID_real").Date_dt.agg(lambda s: (s.max() - s.min()).days / 365.25); first = rel.groupby("PatientID_real").Date_dt.min()
    pe["Date_dt"] = pd.to_datetime(pe.Date, errors="coerce"); pe["Next_dt"] = pd.to_datetime(pe.NextBiopsyDate, errors="coerce")
    ev = {}
    for p, d in pe.groupby("PatientID_real"):
        c1 = d.loc[d.is_lgd2_event_at_current.map(truthy), "Date_dt"].dropna().tolist(); c2 = d.loc[d.NextBiopsyProgression_LGD2plus.map(truthy), "Next_dt"].dropna().tolist()
        if c1 or c2: ev[p] = min(c1 + c2)
    RP3 = RP.copy(); RP3["first_to_last"] = RP3.patient_id.map(span)
    RP3["first_to_endpoint"] = [((ev[p] - first[p]).days / 365.25) if (lab == "P" and p in ev) else np.nan for p, lab in zip(RP3.patient_id, RP3.label)]
    i3 = {"release": {"first_to_last": by_groups(RP3, "first_to_last"), "first_to_endpoint": by_groups(RP3, "first_to_endpoint")},
          "release_P_with_event_date": {"n": int(sum(p in ev for p in RP3.patient_id[RP3.label == "P"])), "of": int((RP3.label == "P").sum())},
          "release_rows_with_date": int(rel.Date_dt.notna().sum())}
    A = pd.read_csv(K + "/kr_samples.csv", dtype=str).set_index("Sample"); fdi = fd.drop_duplicates("combined_name").set_index("combined_name")
    A["mbf"] = pd.to_numeric(A.index.map(fdi["Months before final"]), errors="coerce"); A["ogd"] = A.index.map(fdi["Path_class_per_OGD"])
    hg = A[(A.Pathology.isin(["HGD", "IMC"])) | (A.ogd.isin(["HGD", "IMC", "HGD/IMC"]))]; tev = hg.groupby("Patient").mbf.max()
    Ppats = sorted(CP.Patient[CP.label == "P"]); n_hg = int(sum(p in tev.index for p in Ppats)); tev = pd.concat([tev, pd.Series(0.0, index=[p for p in Ppats if p not in tev.index])])  # kc_merge.py fallback: final endoscopy
    cmax = C.groupby("Patient").mbf.max(); cmin = C.groupby("Patient").mbf.min()
    CP3 = CP.copy(); CP3["first_to_last"] = CP3.Patient.map((cmax - cmin) / 12.0)
    CP3["first_to_endpoint"] = [((cmax[p] - tev[p]) / 12.0) if (lab == "P" and p in tev.index) else np.nan for p, lab in zip(CP3.Patient, CP3.label)]
    i3["setC"] = {"first_to_last": by_groups(CP3, "first_to_last", ("all",)), "first_to_endpoint": by_groups(CP3, "first_to_endpoint", ("all",))}
    i3["setC_P_with_hgd_imc_sample"] = {"n": n_hg, "of": len(Ppats), "fallback_final_endoscopy": len(Ppats) - n_hg}
    i3["setC_P_endpoint_before_first_setC_sample"] = int((CP3.first_to_endpoint < 0).sum())
    i3["setC_samples_with_mbf"] = {"n": int(C.mbf.notna().sum()), "of": int(len(C))}
    res["item3"] = i3
    # ---------- item 4
    dm = pd.read_csv(S + "/Demographics_full.csv", dtype=str)
    for c in ("Study Number", "Alternate Study Number"): dm[c] = dm[c].str.strip()
    dmi = {}
    for _, r in dm.iterrows():
        for c in ("Study Number", "Alternate Study Number"):
            if isinstance(r[c], str) and r[c] and r[c] not in dmi: dmi[r[c]] = r
    ih = pd.read_parquet(E + "/initial_history.parquet"); en = pd.read_parquet(E + "/endoscopy.parquet")
    ih["pid"] = ih.participant_id.astype(str); en["pid"] = en.participant_id.astype(str); en["edate"] = pd.to_datetime(en.endoscopydate, errors="coerce")
    num = lambda v: pd.to_numeric(pd.Series([v]).replace("", np.nan), errors="coerce").iloc[0]
    pid_of = pe.dropna(subset=["participant_id"]).groupby("PatientID_real").participant_id.agg(lambda s: s.str.replace(r"\.0$", "", regex=True).mode().iloc[0])
    def demo_table(keys, first_dates, labels, base_grade, src_first):
        rows = []
        for k, fdt, lab, bg in zip(keys, first_dates, labels, base_grade):
            r = dmi.get(k); pid = pid_of.get(k)
            dob_r = pd.to_datetime(r["Date of birth"], errors="coerce") if r is not None else pd.NaT
            d = {"label": lab, "baseline_grade": bg, "sex_r": (r["Sex"].strip() if r is not None and isinstance(r["Sex"], str) else np.nan),
                 "smoking_r": (r["Smoking Status"].strip() if r is not None and isinstance(r["Smoking Status"], str) else np.nan),
                 "C_r": num(r["Circumference "]) if r is not None else np.nan, "M_r": num(r["Maximal"]) if r is not None else np.nan,
                 "age_r": ((fdt - dob_r).days / 365.25) if pd.notna(dob_r) and pd.notna(fdt) else np.nan, "in_dm": r is not None, "has_pid": pid is not None}
            dob_db = np.nan; smk_db = np.nan; C_db = np.nan; M_db = np.nan; csrc = msrc = None; gap = np.nan
            if pid is not None:
                h = ih[ih.pid == pid]
                if len(h):
                    dd = pd.to_datetime(h.dob, errors="coerce").dropna(); dob_db = dd.iloc[0] if len(dd) else np.nan
                    sm = h.smoking.dropna().astype(str); sm = sm[~sm.isin(["", "None"])]; smk_db = sm.iloc[0] if len(sm) else np.nan
                e = en[en.pid == pid].dropna(subset=["edate"])
                if len(e) and pd.notna(fdt):
                    e = e.assign(gap=(e.edate - fdt).abs().dt.days).sort_values("gap")
                    ec = e[pd.to_numeric(e.barretts_circumference, errors="coerce").notna()]; em = e[pd.to_numeric(e.barretts_maximum, errors="coerce").notna()]
                    if len(ec): C_db = float(pd.to_numeric(ec.barretts_circumference.iloc[0])); csrc = "endoscopy"; gap = float(ec.gap.iloc[0])
                    if len(em): M_db = float(pd.to_numeric(em.barretts_maximum.iloc[0])); msrc = "endoscopy"
                if len(h):
                    if pd.isna(C_db): v = pd.to_numeric(h.praguec.replace("", np.nan), errors="coerce").dropna(); C_db = float(v.iloc[0]) if len(v) else np.nan; csrc = "initial_history" if len(v) else None
                    if pd.isna(M_db): v = pd.to_numeric(h.praguem.replace("", np.nan), errors="coerce").dropna(); M_db = float(v.iloc[0]) if len(v) else np.nan; msrc = "initial_history" if len(v) else None
            age_db = ((fdt - dob_db).days / 365.25) if pd.notna(dob_db) and pd.notna(fdt) else np.nan
            d.update({"age_c": d["age_r"] if pd.notna(d["age_r"]) else age_db, "C_c": d["C_r"] if pd.notna(d["C_r"]) else C_db, "M_c": d["M_r"] if pd.notna(d["M_r"]) else M_db,
                      "C_src": "demographics" if pd.notna(d["C_r"]) else csrc, "M_src": "demographics" if pd.notna(d["M_r"]) else msrc, "C_gap_days": gap if (pd.isna(d["C_r"]) and csrc == "endoscopy") else np.nan,
                      "age_src": "demographics" if pd.notna(d["age_r"]) else ("initial_history" if pd.notna(age_db) else None), "smoking_db_raw": smk_db if pd.isna(d["smoking_r"]) else np.nan})
            rows.append(d)
        T = pd.DataFrame(rows); out = {"n_patients": int(len(T)), "in_demographics_sheet": int(T.in_dm.sum()), "with_participant_id": int(T.has_pid.sum()), "first_sample_date_source": src_first}
        for lab in ("all", "P", "NP"):
            d = T if lab == "all" else T[T.label == lab]; o = {"n": int(len(d))}
            for v in ("age", "C", "M"):
                o[v + "_release"] = desc(d[v + "_r"], 2); o[v + "_completed"] = desc(d[v + "_c"], 2)
                if v != "age": o[v + "_completed_sources"] = d[v + "_src"].value_counts().to_dict()
                else: o["age_completed_sources"] = d.age_src.value_counts().to_dict()
            o["C_endoscopy_gap_days"] = desc(d.C_gap_days, 0)
            o["sex_release"] = d.sex_r.value_counts().to_dict(); o["sex_n"] = int(d.sex_r.notna().sum())
            o["smoking_release"] = d.smoking_r.value_counts().to_dict(); o["smoking_n"] = int(d.smoking_r.notna().sum())
            o["smoking_db_raw_codes_for_missing"] = d.smoking_db_raw.value_counts().to_dict()
            o["baseline_grade"] = d.baseline_grade.value_counts().to_dict(); o["baseline_grade_n"] = int(d.baseline_grade.notna().sum())
            out[lab] = o
        return out
    lab_map = {"0": "NDBE", "1": "IND", "2": "LGD"}
    fr = rel.sort_values(["PatientID_real", "Date_dt"]).groupby("PatientID_real").first()
    i4 = {"release": demo_table(RP.patient_id.tolist(), [first[p] for p in RP.patient_id], RP.label.tolist(), [lab_map.get(str(fr.Label[p]).replace(".0", ""), str(fr.Label[p])) for p in RP.patient_id], "pre_event_cohort.Date (first release row)")}
    # set C: earliest sample = largest months-before-final; date = release Date where the sample is a pre_event_cohort CNV, else 1 July of Endoscopy Year
    pe["cnv"] = pe.CNVAbsPath.map(lambda p: os.path.basename(str(p))); pdate = pe.dropna(subset=["Date_dt"]).drop_duplicates("cnv").set_index("cnv").Date_dt
    C["date"] = [pdate.get(s, pd.Timestamp(f"{int(y)}-07-01") if str(y).isdigit() else pd.NaT) for s, y in zip(C.Sample, C.year)]
    C["date_src"] = ["pre_event_cohort.Date" if s in pdate.index else "Endoscopy Year (1 July)" for s in C.Sample]
    cf = C.sort_values(["Patient", "mbf"], ascending=[True, False]).groupby("Patient").first()
    hr = C.dropna(subset=["hrid"]).groupby("Patient").hrid.agg(lambda s: s.mode().iloc[0])
    i4["setC"] = demo_table([hr.get(p) for p in CP.Patient], [cf.date[p] for p in CP.Patient], CP.label.tolist(), [cf.Pathology[p] for p in CP.Patient], "first set C sample: " + json.dumps(cf.date_src.value_counts().to_dict()))
    i4["setC_patients_with_hrid"] = int(sum(hr.get(p) is not None for p in CP.Patient))
    res["item4"] = i4
    # ---------- item 7 (manifest only; Pathology and case-level files not read)
    A7 = glob.glob("/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta/ACEB_samples*.csv")[0]
    a = pd.read_csv(A7, dtype=str, usecols=["Batch", "PatientID", "SLX", "SampleID", "Mean_Coverage"])
    a["cov"] = pd.to_numeric(a.Mean_Coverage.astype(str).str.replace("x", "", case=False).str.strip(), errors="coerce")
    res["item7"] = {"manifest": A7, "samples": int(len(a)), "sample_ids_unique": int(a.SampleID.nunique()), "patients": int(a.PatientID.nunique()), "batches": a.Batch.value_counts().to_dict(),
                    "slx_pools": a.SLX.value_counts().to_dict(), "mean_coverage": desc(a["cov"], 2), "mean_coverage_by_batch": {b: desc(d["cov"], 2) for b, d in a.groupby("Batch")},
                    "samples_per_patient": desc(a.groupby("PatientID").size(), 2), "columns": pd.read_csv(A7, nrows=0).columns.tolist(),
                    "imaging_dirs_matching_ace": [d for d in os.listdir("/mnt/scratche/fast/fmlab/datasets/imaging") if re.search("ace", d, re.I)],
                    "slide_files_in_aceb_meta": [f for f in os.listdir(os.path.dirname(A7)) if re.search(r"\.(ndpi|svs|tif|tiff|mrxs)$", f, re.I)]}

elif PART in ("wsi_release", "wsi_setc"):
    import openslide
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    if PART == "wsi_release":
        ui = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str).set_index("sample_id")
        items = [(remap(p), ui.npz_path.get(s)) for s, p in zip(rel.SampleID, rel.ImageAbsPath)]
    else:
        mp = pd.read_csv(K + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(C.Sample)].drop_duplicates("npz")
        items = []
        for n in mp.npz:
            with np.load(n, allow_pickle=True) as z: items.append((remap(str(z["slide_path"])), n))
    def one(it):
        sp, npz = it; o = {"slide": sp}
        try:
            s = openslide.OpenSlide(sp); pr = s.properties
            o.update({"product": pr.get("hamamatsu.Product") or pr.get("tiff.Model") or pr.get("openslide.vendor"), "vendor": pr.get("openslide.vendor"), "mpp": float(pr.get("openslide.mpp-x", "nan")),
                      "objective": pr.get("openslide.objective-power"), "levels": s.level_count})
            with np.load(npz, allow_pickle=True) as z:
                lv = int(z["level"]); ts = int(z["tile_size"]); o.update({"level": lv, "tile_size": ts, "tiles_requested": int(z["tiles_requested"]), "tiles_ok": int(z["tiles_ok"]), "emb_dim": int(z["embeddings"].shape[1]),
                                                                          "distinct_sampled": int(len(np.unique(np.asarray(z["coords_level"]).reshape(-1, 2), axis=0)))})
            o["level_mpp"] = o["mpp"] * float(s.level_downsamples[lv]); W, H = s.level_dimensions[lv]
            md = os.path.join(os.path.dirname(os.path.dirname(sp)), "masks"); stem = os.path.splitext(os.path.basename(sp))[0]
            cand = sorted(glob.glob(os.path.join(md, glob.escape(stem) + "*")))
            if cand:
                mk = np.array(Image.open(cand[0]).convert("L")) > 0; mh, mw = mk.shape
                xs = (np.arange(W // ts) * ts + ts / 2) * mw / W; ys = (np.arange(H // ts) * ts + ts / 2) * mh / H
                xi = np.clip(xs.round().astype(int), 0, mw - 1); yi = np.clip(ys.round().astype(int), 0, mh - 1)
                o["tissue_grid_tiles"] = int(mk[np.ix_(yi, xi)].sum()); o["mask_pos_px"] = int(mk.sum()); o["with_replacement"] = bool(mk.sum() < o["tiles_requested"])
            s.close()
        except Exception as ex: o["error"] = repr(ex)[:200]
        return o
    with ThreadPoolExecutor(8) as ex: R = pd.DataFrame(list(ex.map(one, items)))
    R = R.drop_duplicates("slide")
    res["n_slides"] = int(len(R)); res["errors"] = int(R.error.notna().sum()) if "error" in R.columns else 0; res["error_examples"] = R.error.dropna().head(3).tolist() if "error" in R.columns else []
    ok = R[R.error.isna()] if "error" in R.columns else R
    res["scanners"] = ok["product"].value_counts().to_dict(); res["vendor"] = ok.vendor.value_counts().to_dict(); res["objective"] = ok.objective.value_counts().to_dict()
    res["mpp_native"] = desc(ok.mpp, 4); res["mpp_native_by_scanner"] = {k: desc(d.mpp, 4) for k, d in ok.groupby("product")}
    res["level"] = ok.level.value_counts().to_dict(); res["tile_size"] = ok.tile_size.value_counts().to_dict(); res["level_mpp"] = desc(ok.level_mpp, 4)
    res["level_mpp_by_scanner"] = {k: desc(d.level_mpp, 4) for k, d in ok.groupby("product")}
    res["tiles_requested"] = desc(ok.tiles_requested, 1); res["tiles_ok"] = desc(ok.tiles_ok, 1); res["distinct_sampled"] = desc(ok.distinct_sampled, 1)
    res["tissue_grid_tiles"] = desc(ok.get("tissue_grid_tiles", pd.Series(dtype=float)), 1); res["with_replacement"] = int(ok.get("with_replacement", pd.Series(dtype=bool)).fillna(False).sum())
    res["emb_dim"] = ok.emb_dim.value_counts().to_dict(); res["slides_tissue_tiles_lt_requested"] = int((ok.tissue_grid_tiles < ok.tiles_requested).sum())
    if PART == "wsi_release":
        res["patients_by_scanner"] = {}
        sp2pat = dict(zip(rel.ImageAbsPath.map(remap), rel.PatientID_real))
        ok = ok.assign(pat=ok.slide.map(sp2pat)); res["patients_by_scanner"] = {k: int(d.pat.nunique()) for k, d in ok.groupby("product")}
        views = {}
        for v in ("uni2", "virchow2", "gigapath"):
            x = pd.read_csv(F + f"/feature_views/{v}/{v}_index.csv", dtype=str); views[v] = {"rows": int(len(x)), "feat_dim": x.feat_dim.value_counts().to_dict(), "n_instances": x.n_instances.value_counts().head(3).to_dict()}
        res["feature_views"] = views
    else:
        bi = pd.read_csv(M + "/bag_index.csv", dtype=str); bi = bi[bi.sample_id.isin(C.Sample)]
        res["bag_tiles_per_sample"] = desc(bi.n_tiles.astype(float), 1); res["bag_slides_per_sample"] = bi.n_slides.value_counts().to_dict(); res["bag_samples"] = int(len(bi))

elif PART == "swgs":
    def rcs(cid, kb):
        for d in (f"{S}/copy_number_hg38/train/perPatient/{kb}kb/{cid}", f"{S}/copy_number_hg38/val/perPatient/{kb}kb/{cid}", f"{S}/copy_number_hg38/val/{kb}kb/{cid}"):
            f = f"{d}/{kb}.readCountSummary.txt"
            if os.path.exists(f):
                x = pd.read_csv(f, sep="\t"); return float(x["total.reads"].iloc[0]), float(x["used.reads"].iloc[0]), d.split("/copy_number_hg38/")[1].split("/")[0]
        return np.nan, np.nan, None
    def bam(cid):
        b = f"{S}/dna_seq_bam/{cid}.bam"
        if not os.path.exists(b): return {"bam": False}
        h = subprocess.run([ST, "view", "-H", b], capture_output=True, text=True).stdout.splitlines()
        sq = {l.split("\tSN:")[1].split("\t")[0]: int(l.split("\tLN:")[1].split("\t")[0]) for l in h if l.startswith("@SQ")}
        glen = sum(v for k, v in sq.items() if re.fullmatch(r"chr([0-9]+|X|Y)", k))
        pg = [l for l in h if l.startswith("@PG") and "ID:bwa" in l]; ref = re.search(r"CL:bwa \S+ (\S+)", pg[0]).group(1) if pg else None
        L = subprocess.run(f"{ST} view {b} | head -2000 | cut -f10 | awk '{{print length($0)}}'", shell=True, capture_output=True, text=True).stdout.split()
        L = np.array([int(x) for x in L]) if L else np.array([0])
        return {"bam": True, "glen": glen, "ref": ref, "bwa": (pg[0].split("\tVN:")[1].split("\t")[0] if pg else None), "aligner_cmd": (pg[0].split("\tCL:")[1].split(" ")[1] if pg else None), "rl_mode": int(np.bincount(L).argmax()), "rl_max": int(L.max()), "rl_min": int(L.min())}
    def run(ids, kb):
        def f(cid):
            t, u, split = rcs(cid, kb); o = {"cid": cid, "total": t, "used": u, "split": split}; o.update(bam(cid)); return o
        with ThreadPoolExecutor(8) as ex: R = pd.DataFrame(list(ex.map(f, ids)))
        R["depth_total"] = R.total * R.rl_mode / R.glen; R["depth_used"] = R.used * R.rl_mode / R.glen
        return R
    def summ(R):
        return {"samples": int(len(R)), "with_readcount": int(R.total.notna().sum()), "with_bam": int(R.bam.sum()), "split": R.split.value_counts().to_dict(),
                "total_reads": desc(R.total, 0), "used_reads": desc(R.used, 0), "depth_total": desc(R.depth_total, 4), "depth_used": desc(R.depth_used, 4),
                "read_length_mode": R.rl_mode.value_counts().to_dict(), "read_length_range": [int(R.rl_min.min()), int(R.rl_max.max())] if R.bam.any() else None,
                "reference": R.ref.value_counts().to_dict(), "bwa": R.bwa.value_counts().to_dict(), "aligner_cmd": R.aligner_cmd.value_counts().to_dict(), "genome_len_chr1_22_X_Y": R.glen.value_counts().to_dict()}
    rids = sorted(rel.cnv_id.dropna().unique()); Rr = run(rids, 500); sh = dict(zip(rel.cnv_id, rel.PatientID_real.map(dict(zip(RP.patient_id, RP.sheet)))))
    Rr["sheet"] = Rr.cid.map(sh); res["release"] = {"all": summ(Rr), **{s: summ(d) for s, d in Rr.groupby("sheet")}}
    Rc = run(sorted(C.Sample.unique()), 50); res["setC"] = summ(Rc)
    fdi = fd.drop_duplicates("combined_name").set_index("combined_name"); nr = pd.to_numeric(C.Sample.drop_duplicates().map(fdi["Number of reads"]).astype(str).str.replace(",", ""), errors="coerce")
    res["setC_sheet_number_of_reads"] = desc(nr, 0)
    res["release_cnv_build"] = json.load(open(B + "/data/killcoyne_repro_strict_500kb_slurm_v2/run_config.json"))
    res["release_cnv_build"] = {k: v for k, v in res["release_cnv_build"].items() if re.search("pcf|qc|alpha|lambda|seg|version|assembly|centromere_file$|input", k)}

json.dump(res, open(f"{OUT}/{PART}.json", "w"), indent=1, default=str)
print("DD_DONE", PART)
