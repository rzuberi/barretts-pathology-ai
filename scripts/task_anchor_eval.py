"""A: human-grade anchors. (1) 658 SWG specimen pairs with pathologist grade: jury consensus, each juror, MedGemma-27B.
(2) ACE-B Seattle grades vs DB-report grade nearest the trial date (jury; MedGemma-27B if run)."""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; A = "/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta"; OUT = os.environ.get("OUTDIR", ".")
ORD = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}; SWG = {"BE": "NDBE", "ID": "IND", "LGD": "LGD", "HGD": "HGD", "IMC": "CANCER"}; SEA = {"IM": "NDBE", "ID": "IND", "LGD": "LGD", "HGD": "HGD", "IMC": "CANCER"}
def metrics(truth, pred):
    m = pd.DataFrame({"t": truth, "p": pred}).dropna(); m = m[m.p.isin(ORD) & m.t.isin(ORD)]
    if len(m) < 10: return {"n": len(m)}
    t, p = m.t.map(ORD), m.p.map(ORD)
    return {"n": len(m), "exact": round(float((t == p).mean()), 4), "two_tier_LGDplus": round(float(((t >= 2) == (p >= 2)).mean()), 4), "qwk": round(float(cohen_kappa_score(t, p, weights="quadratic")), 4),
            "within_one_rung": round(float(((t - p).abs() <= 1).mean()), 4), "confusion_truth_rows": pd.crosstab(m.t, m.p).reindex(index=ORD, columns=ORD, fill_value=0).values.tolist()}
res = {}
sp = pd.read_parquet(E + "/specimen_pairs.parquet"); sp["truth"] = sp.swg_grade_code.map(SWG); sp = sp.set_index("CaseName")
jv = pd.read_csv(E + "/jury_specimen/jury_votes.csv", dtype=str).set_index("CaseName")
res["swg_pathologist_grade"] = {"n_pairs": len(sp), "truth_dist": sp.truth.value_counts().to_dict(), "jury_consensus": metrics(sp.truth, jv.jury_label.reindex(sp.index)),
                                "jury_consensus_confident_only(frac>=0.75)": metrics(sp.truth[jv.reindex(sp.index).jury_frac.astype(float) >= 0.75], jv.jury_label.reindex(sp.index)[jv.reindex(sp.index).jury_frac.astype(float) >= 0.75])}
per = {}
for f in glob.glob(E + "/jury_specimen/llm_grades_*.csv"):
    mo = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]
    try: d = pd.read_csv(f, dtype=str, on_bad_lines="skip").drop_duplicates("CaseName").set_index("CaseName")
    except Exception: continue
    per.setdefault(mo, pd.Series(dtype=str)); per[mo] = pd.concat([per[mo], d.llm_grade]); per[mo] = per[mo][~per[mo].index.duplicated()]
res["swg_pathologist_grade"]["per_juror"] = {mo: metrics(sp.truth, v.reindex(sp.index)) for mo, v in per.items()}
mg = glob.glob(T + "/feasibility/runs/jspec_medgemma27b/output/llm_grades_*.csv")
if mg:
    d = pd.concat([pd.read_csv(f, dtype=str, on_bad_lines="skip") for f in mg]).drop_duplicates("CaseName").set_index("CaseName")
    res["swg_pathologist_grade"]["medgemma27b"] = metrics(sp.truth, d.llm_grade.reindex(sp.index)); res["swg_pathologist_grade"]["medgemma27b_parse_fail"] = int((d.llm_grade == "PARSE_FAIL").sum())
# ---- ACE-B ----
ac = pd.read_csv(A + "/aceb_official_case_presence_in_barretts_db_20260527_150857.csv", dtype=str)
pt = pd.read_csv(E + "/pathology_text_normalised_full.csv", dtype=str, usecols=["pathology_text_id", "participant_id", "receiveddatetime"]); pt["d"] = pd.to_datetime(pt.receiveddatetime, dayfirst=True, errors="coerce")
def corpus_labels(pattern):
    votes = {}
    for f in glob.glob(pattern):
        d = pd.read_csv(f, dtype=str, on_bad_lines="skip"); d = d[d.llm_grade.isin(ORD)]
        for c, g in zip(d.CaseName, d.llm_grade): votes.setdefault(str(c), []).append(g)
    return {c: max(set(v), key=v.count) for c, v in votes.items() if len(v) >= max(1, min(4, len(v)))}
jury_db = corpus_labels(T + "/feasibility/runs/jury_full_*/output/llm_grades_*.csv"); mg_db = corpus_labels(T + "/feasibility/runs/jdb_medgemma27b_s*/output/llm_grades_*.csv")
rows = []
for _, r in ac.iterrows():
    pids = [x.strip() for x in str(r.participant_ids).replace(";", ",").split(",") if x.strip() and x.strip().lower() != "nan"]
    d0 = min([x for x in (pd.to_datetime(r["Date AFI"], dayfirst=True, errors="coerce"), pd.to_datetime(r["Date WLE"], dayfirst=True, errors="coerce")) if pd.notna(x)], default=pd.NaT)
    if not pids or pd.isna(d0) or r.Histology_Seattle_Protocol not in SEA: continue
    cand = pt[pt.participant_id.isin(pids) & pt.d.notna()].copy(); cand["gap"] = (cand.d - d0).abs().dt.days; cand = cand[cand.gap <= 120].sort_values("gap")
    if cand.empty: continue
    pid_ = str(cand.iloc[0].pathology_text_id)
    # item 26 (24 Sep 2026): a Seattle-protocol HGD can sit in a DIFFERENT report of the same visit than the one
    # closest in time; also score the MAX jury grade over every report in the 120-day window
    def mx(lab):
        g = [lab.get(str(x)) for x in cand.pathology_text_id]; g = [x for x in g if x in ORD]; return max(g, key=lambda k: ORD[k]) if g else None
    rows.append({"case": r.study_number_normalized, "truth": SEA[r.Histology_Seattle_Protocol], "gap_days": int(cand.iloc[0].gap), "jury": jury_db.get(pid_), "medgemma27b": mg_db.get(pid_),
                 "n_reports_in_window": int(len(cand)), "jury_max_window": mx(jury_db), "medgemma27b_max_window": mx(mg_db)})
ad = pd.DataFrame(rows)
res["aceb_seattle_grade"] = {"cases_with_report_within_120d": len(ad), "truth_dist": ad.truth.value_counts().to_dict() if len(ad) else {}, "gap_days_median": float(ad.gap_days.median()) if len(ad) else None,
                             "jury": metrics(ad.truth, ad.jury) if len(ad) else {}, "medgemma27b": metrics(ad.truth, ad.medgemma27b) if len(ad) and ad.medgemma27b.notna().any() else {"n": 0, "note": "DB-corpus MedGemma run not finished"}}
res["aceb_seattle_grade_maxwindow"] = {"rule": "max jury grade over all DB reports within 120 d of trial entry (vs closest report above)", "cases": len(ad),
    "reports_in_window_dist": ad.n_reports_in_window.value_counts().sort_index().to_dict() if len(ad) else {},
    "jury": metrics(ad.truth, ad.jury_max_window) if len(ad) else {}, "medgemma27b": metrics(ad.truth, ad.medgemma27b_max_window) if len(ad) else {}}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=str); print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_juror"} for k, v in res.items()}, indent=1, default=str)[:4000])
