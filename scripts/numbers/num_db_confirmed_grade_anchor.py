#!/usr/bin/env python3
"""Barrett's-database CODED grade as a human anchor for the LLM jury (found 24 Sep 2026).
`pathology_text_normalised_full.csv` carries `highestgradedysconf` (highest grade of dysplasia, confirmed) on 99.8 % of
13,645 reports, coded with the database's own lookup (`query_dysplasia_types.csv`): 1 Normal squamous, 10 Gastric
metaplasia, 9 Cardia IM, 2 Barrett's no dysplasia, 3 Indefinite, 4 LGD, 5 HGD, 6 IMC, 8 invasive adenocarcinoma,
7 Not specified (plus squamous-system codes 11-19 and rarer codes). Mapping to our five-rung ladder:
2->NDBE, 3->IND, 4->LGD, 5->HGD, 6/8->CANCER; 1/9/10 (non-Barrett's mucosa, no dysplasia) -> NDBE bucket in a secondary
analysis; 7 and others excluded. Also: the DB code vs the SWG spreadsheet code on the 313 reports that have both, and
the jury's provisional-code column for completeness. Provenance of the field (human transcription vs pathologist
consensus) is a question for the database team; it is NOT an independent re-read of the slides."""
import glob, json, os, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", ".")
S = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/data/barretts_scrape_20260220_172725"
ORD = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}; LADDER = ["NDBE", "IND", "LGD", "HGD", "CANCER"]
lk = pd.read_csv(S + "/query_dysplasia_types.csv", dtype=str); lookup = dict(zip(pd.to_numeric(lk.dysplasia_grade_id), lk.dysplasia_grade))
MAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 4}; MAP_WIDE = {**MAP, 1: 0, 9: 0, 10: 0}
pt = pd.read_csv(E + "/pathology_text_normalised_full.csv", dtype=str, usecols=["pathology_text_id", "participant_id", "receiveddatetime", "highestgradedysconf", "highestgradedysprov"])
votes = {}
for f in glob.glob(T + "/feasibility/runs/jury_full_*/output/llm_grades_*.csv"):
    d = pd.read_csv(f, dtype=str, on_bad_lines="skip"); d = d[d.llm_grade.isin(ORD)]
    for c, g in zip(d.CaseName, d.llm_grade): votes.setdefault(str(c), []).append(g)
lab = {c: max(set(v), key=v.count) for c, v in votes.items() if len(v) >= 4}; frac = {c: max(v.count(g) for g in set(v)) / len(v) for c, v in votes.items() if len(v) >= 4}
pt["jury"] = pt.pathology_text_id.astype(str).map(lab).map(ORD); pt["jfrac"] = pt.pathology_text_id.astype(str).map(frac)
pt["conf"] = pd.to_numeric(pt.highestgradedysconf, errors="coerce"); pt["prov"] = pd.to_numeric(pt.highestgradedysprov, errors="coerce")
def met(t, p):
    t = np.asarray(t, int); p = np.asarray(p, int)
    return {"n": int(len(t)), "exact": round(float((t == p).mean()), 4), "two_tier_LGDplus": round(float(((t >= 2) == (p >= 2)).mean()), 4), "qwk": round(float(cohen_kappa_score(t, p, weights="quadratic")), 4),
            "within_one_rung": round(float((np.abs(t - p) <= 1).mean()), 4), "overcall_rate_on_NDBE": round(float((p[t == 0] >= 1).mean()), 4), "undercall_rate_on_LGDplus": round(float((p[t >= 2] < 2).mean()), 4),
            "LGDplus_sensitivity": round(float((p[t >= 2] >= 2).mean()), 4), "LGDplus_specificity": round(float((p[t < 2] < 2).mean()), 4),
            "confusion_truth_rows": pd.crosstab(pd.Series(t), pd.Series(p)).reindex(index=range(5), columns=range(5), fill_value=0).values.tolist()}
res = {"_meta": {"source": E + "/pathology_text_normalised_full.csv", "field": "highestgradedysconf", "lookup": S + "/query_dysplasia_types.csv", "code_names": {int(k): v for k, v in lookup.items() if k == k},
                 "reports_total": int(len(pt)), "reports_with_conf_code": int(pt.conf.notna().sum()), "reports_with_jury_grade": int(pt.jury.notna().sum()), "participants": int(pt.participant_id.nunique()),
                 "ladder": LADDER, "mapping_core": {str(k): LADDER[v] for k, v in MAP.items()}, "mapping_wide_extra": "1 Normal squamous, 9 Cardia IM, 10 Gastric metaplasia -> NDBE bucket",
                 "caveat": "the DB code is a human coding of the clinical report inside the Barrett's database, not an independent slide re-read; provenance to be confirmed with the database team"}}
m = pt[pt.jury.notna() & pt.conf.notna()]
core = m[m.conf.isin(MAP)]; res["confirmed_code_core_barretts"] = {"code_distribution": {lookup.get(int(k), str(k)): int(v) for k, v in core.conf.value_counts().items()}, "participants": int(core.participant_id.nunique()),
    "jury_all": met(core.conf.map(MAP), core.jury), "jury_confident_frac_ge_0.75": met(core[core.jfrac >= 0.75].conf.map(MAP), core[core.jfrac >= 0.75].jury)}
wide = m[m.conf.isin(MAP_WIDE)]; res["confirmed_code_wide_incl_nonbarretts_mucosa"] = {"n": int(len(wide)), "jury_all": met(wide.conf.map(MAP_WIDE), wide.jury)}
res["other_codes_jury_distribution"] = {f"{int(k)} {lookup.get(int(k), '?')}": {LADDER[int(j)]: int(n) for j, n in g.jury.value_counts().items()} for k, g in m[~m.conf.isin(MAP_WIDE)].groupby("conf") if len(g) >= 5}
pv = m[m.prov.isin(MAP)]; res["provisional_code_core"] = {"n": int(len(pv)), "jury_all": met(pv.prov.map(MAP), pv.jury), "note": "provisional grade, 74 % filled"}
pc = m[m.conf.isin(MAP) & m.prov.isin(MAP)]; res["provisional_vs_confirmed_code"] = met(pc.conf.map(MAP), pc.prov.map(MAP)); res["provisional_vs_confirmed_code"]["note"] = "two human codings of the same report"
# DB code vs SWG spreadsheet code (per report, report max of SWG codes)
SWG = {"BE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}
sm = pd.read_parquet(E + "/swg_matched_reports_v2.parquet"); sp = pd.read_parquet(E + "/specimen_pairs.parquet"); sp["s"] = sp.swg_grade_code.map(SWG); mx = sp.groupby("report_id").s.max()
sm["conf"] = pd.to_numeric(sm.highestgradedysconf, errors="coerce"); sm["swgmax"] = sm.pathology_text_id.astype(str).map(mx); x = sm[sm.conf.isin(MAP) & sm.swgmax.notna()]
res["db_confirmed_code_vs_swg_spreadsheet"] = met(x.conf.map(MAP), x.swgmax.astype(int)); res["db_confirmed_code_vs_swg_spreadsheet"]["note"] = "two human codings; SWG code per report = max over its specimens"
jx = sm.assign(jury=sm.pathology_text_id.astype(str).map(lab).map(ORD)); jx = jx[jx.conf.isin(MAP) & jx.jury.notna()]
res["jury_vs_db_confirmed_code_on_swg_reports"] = met(jx.conf.map(MAP), jx.jury)
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "db_confirmed_grade_anchor.json"), "w"), indent=1)
for k in ("confirmed_code_core_barretts", "confirmed_code_wide_incl_nonbarretts_mucosa", "provisional_code_core", "provisional_vs_confirmed_code", "db_confirmed_code_vs_swg_spreadsheet", "jury_vs_db_confirmed_code_on_swg_reports"):
    v = res[k]; v = v.get("jury_all", v); print(k, {kk: vv for kk, vv in v.items() if kk != "confusion_truth_rows"})
