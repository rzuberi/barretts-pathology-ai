"""NUMBERS A/F for OCCAMS: cases/patients behind the slides, specimen mix, years, sites, treatment,
outcome-field completeness (TRG, response, recurrence, survival), genomics coverage, candidate-label positives."""
import glob, json, os, re
import numpy as np, pandas as pd
OUT = os.environ.get("OUTDIR", ".")
FEAT = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/slides/features/20x_224px/features_uni_v2"
TSV = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/genomics/clinical_data_wgs_cases_therapy_tp53status_ploidy_wgd_status.tsv"
MASTER = os.path.expanduser("~/occams_work/occams_master_20260511.csv")
def norm_occ(s):
    s = str(s).strip().upper().replace("/", "-"); m = re.search(r"(?:OCCAMS|OC)[-_ ]?([A-Z]{2})[-_ ]?0*([0-9]+)", s); return f"{m.group(1)}{int(m.group(2)):04d}" if m else s
def mr(s): s = pd.Series(s).dropna(); return {"median": float(s.median()), "min": float(s.min()), "max": float(s.max()), "n": int(len(s))} if len(s) else None
h5 = sorted(glob.glob(FEAT + "/*.h5")); bn = [os.path.basename(f) for f in h5]
cid = [norm_occ(b.split("_")[0]) for b in bn]
spec = ["NOT-HE" if "NOT-HE" in b.upper() else ("RES" if "_RES" in b.upper() else ("OGD" if "_OGD" in b.upper() else "OTHER")) for b in bn]
df = pd.DataFrame({"cid": cid, "spec": spec, "bn": bn})
res = {"slides_h5": len(h5), "cases_with_slides": int(df.cid.nunique()), "specimen_mix_slides": df.spec.value_counts().to_dict(),
       "slides_per_case": mr(df.groupby("cid").size()), "cases_with_RES": int(df[df.spec == "RES"].cid.nunique()), "cases_with_OGD": int(df[df.spec == "OGD"].cid.nunique()),
       "cases_with_both": int(len(set(df[df.spec == "RES"].cid) & set(df[df.spec == "OGD"].cid)))}
ma = pd.read_csv(MASTER, dtype=str, low_memory=False)
idcol = next((c for c in ma.columns if re.search(r"occams.?id|^id$|patient", c, re.I)), ma.columns[0]); ma["cid"] = ma[idcol].map(norm_occ); ma = ma.drop_duplicates("cid")
res["master"] = {"rows_dedup": len(ma), "id_col": idcol}
ms = ma[ma.cid.isin(set(cid))]; res["master_rows_for_slide_cases"] = len(ms)
def nn(d, c): return int(d[c].notna().sum()) if c in d else None
dx = pd.to_datetime(ms.get("date_of_ogcdiagnosis"), errors="coerce")
res["slide_cases"] = {"diagnosis_years": {"min": int(dx.min().year) if dx.notna().any() else None, "max": int(dx.max().year) if dx.notna().any() else None, "by_year": dx.dt.year.value_counts().sort_index().to_dict()},
                      "cancer_site": ms.cancer_site.value_counts().to_dict() if "cancer_site" in ms else None,
                      "neoadjuvant_grouped": ms.neoadjuvant_grouped.value_counts().to_dict() if "neoadjuvant_grouped" in ms else None,
                      "neoadjuvant_broadcategory": ms.neoadjuvant_broadcategory.value_counts().to_dict() if "neoadjuvant_broadcategory" in ms else None,
                      "trg_score": ms.trg_score.value_counts().to_dict() if "trg_score" in ms else None,
                      "tumour_response": ms.tumour_response.value_counts().to_dict() if "tumour_response" in ms else None,
                      "pretreatment_T": ms.ps_tstage_primary_tumour_final_pretreatment_staging.value_counts().to_dict(), "pretreatment_N": ms.nstage_primary_tumour_final_pretreatment_staging_tnm7.value_counts().to_dict(),
                      "resection_pT": ms.resection_path_tstage_primary_tumour.value_counts().to_dict(), "resection_pN": ms.resection_path_nstage_rp_tnm7.value_counts().to_dict(),
                      "recurrence_date_nonnull": nn(ms, "date_original_disease_reoccurred"), "adjuvant_regimen_nonnull": nn(ms, "adjuvant_broadregimen")}
surv_cols = [c for c in ma.columns if re.search(r"surviv|deceas|death|died|alive|last_known|vital|status", c, re.I)]
res["survival_columns_found"] = {c: nn(ms, c) for c in surv_cols}
res["completeness_all_master_rows"] = {c: nn(ma, c) for c in ["trg_score", "resection_path_mandard_score_for_response", "tumour_response", "neoadjuvant_grouped", "date_original_disease_reoccurred", "resection_path_nstage_rp_tnm7", "date_of_operation"] if c in ma}
th = pd.read_csv(TSV, sep="\t", dtype=str); th["cid"] = th.OCCAMS_ID.map(norm_occ); th = th.drop_duplicates("cid")
gcols = [c for c in ["TP53_SNV", "TP53_indel", "TP53_deletion", "TP53_knockout", "ploidy", "WGD"] if c in th]
ts = th[th.cid.isin(set(cid))]
res["genomics"] = {"tsv_cases": len(th), "tsv_cases_with_slides": len(ts), "columns": gcols, "nonnull_in_slide_cases": {c: int(ts[c].notna().sum()) for c in gcols},
                   "WGD_dist_slide_cases": ts.WGD.value_counts().to_dict() if "WGD" in ts else None,
                   "TP53_any_slide_cases": int((ts[["TP53_SNV", "TP53_indel", "TP53_deletion", "TP53_knockout"]].apply(lambda r: any(str(v).strip().lower() in ("1", "true", "yes", "y") for v in r), axis=1)).sum()) if "TP53_SNV" in ts else None}
trg = pd.to_numeric(ms.trg_score.astype(str).str.extract(r"(\d)")[0], errors="coerce") if "trg_score" in ms else pd.Series(dtype=float)
ogd = set(df[df.spec == "OGD"].cid); mo = ms[ms.cid.isin(ogd)]; trg_o = pd.to_numeric(mo.trg_score.astype(str).str.extract(r"(\d)")[0], errors="coerce")
res["candidate_labels_positives"] = {
    "TRG1-3_responders_among_OGD_slide_cases": {"pos": int((trg_o <= 3).sum()), "n": int(trg_o.notna().sum())},
    "TRG1-2_among_OGD_slide_cases": {"pos": int((trg_o <= 2).sum()), "n": int(trg_o.notna().sum())},
    "recurrence_recorded_among_slide_cases": {"pos": nn(ms, "date_original_disease_reoccurred"), "n": len(ms), "note": "date present = recurred; absence != no recurrence"},
    "node_positive_pN_among_slide_cases": {"pos": int(ms.resection_path_nstage_rp_tnm7.astype(str).str.contains(r"N[1-3]", regex=True).sum()), "n": int(ms.resection_path_nstage_rp_tnm7.notna().sum())},
    "node_positive_pN_among_OGD_slide_cases": {"pos": int(mo.resection_path_nstage_rp_tnm7.astype(str).str.contains(r"N[1-3]", regex=True).sum()), "n": int(mo.resection_path_nstage_rp_tnm7.notna().sum())},
    "deaths_in_fusion_cohort": {"pos": 58, "n": 87, "source": "results/occams_v3.json"},
    "WGD_positive_slide_cases": {"pos": int(ts.WGD.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"]).sum()) if "WGD" in ts else None, "n": len(ts)},
    "rule_of_thumb": "fewer than ~30 positives -> not trainable"}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=str); print(json.dumps(res, indent=1, default=str)[:3500])
