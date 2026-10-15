"""P31 validation (no human labels): (1) parse rates; (2) field vs keyword-regex agreement on the same report text
(treatment, p53, inflammation, IM/goblet, squamous-only, ulceration); (3) grade vs the 8-model jury final label and vs
the SpecimenProtocol for specimen_type; (4) field marginals; (5) a 50-report hand-check pack (review_local only).
Env: RUNDIR (dir holding fields_*_shard*.jsonl), OUTDIR."""
import glob, json, os, re, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
RUN = os.environ["RUNDIR"]; OUT = os.environ.get("OUTDIR", ".")
rows = [json.loads(l) for rd in RUN.split(":") for f in glob.glob(rd + "/fields_*_shard*.jsonl") for l in open(f)]; d = pd.DataFrame(rows).drop_duplicates("CaseName")
rep = pd.read_csv(ERIN, dtype=str).fillna(""); rep["text"] = rep.ClinicalInformation_redacted + " " + rep.GrossDescription_redacted + " " + rep.MicroscopicDescription_redacted + " " + rep.FinalDiagnosis_redacted + " " + rep.Addendum1_redacted
d = d.merge(rep[["CaseName", "text", "SpecimenProtocol"]], on="CaseName", how="left")
lab = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str).drop_duplicates("CaseName"); d = d.merge(lab[["CaseName", "final_label", "label_status"]], on="CaseName", how="left")
F = ["grade", "site", "specimen_type", "intestinal_metaplasia", "goblet_cells", "inflammation", "ulceration_or_erosion", "squamous_only", "gastric_mucosa_present", "treatment_effect", "p53", "diagnostic_certainty"]
res = {"n_reports": int(len(d)), "parse_fail_rate": {f: round(float((d[f] == "PARSE_FAIL").mean()), 4) for f in F}, "marginals": {f: d[f].value_counts().to_dict() for f in F}, "n_specimens_median": float(pd.to_numeric(d.n_specimens, errors="coerce").median())}
rx = {"treatment_effect": r"\bRFA\b|radiofrequency|ablat|\bEMR\b|\bESD\b|mucosal resection|neosquam|argon|\bAPC\b|cryo|buried", "p53": r"\bp53\b", "inflammation": r"inflamm|oesophagitis|esophagitis|gastritis", "intestinal_metaplasia": r"intestinal metaplasia|goblet|barrett", "ulceration_or_erosion": r"ulcer|erosion|eroded"}
def kw(f): return d.text.str.contains(rx[f], case=False, regex=True)
agree = {}
k = kw("treatment_effect"); agree["treatment_effect_yes_vs_keyword"] = {"kw_present": int(k.sum()), "field_yes": int((d.treatment_effect == "yes").sum()), "agreement": round(float(((d.treatment_effect == "yes") == k).mean()), 4), "yes_without_keyword": int(((d.treatment_effect == "yes") & ~k).sum()), "keyword_without_yes": int((k & (d.treatment_effect != "yes")).sum())}
k = kw("p53"); agree["p53_mentioned_vs_keyword"] = {"kw_present": int(k.sum()), "field_not_not_stated": int(d.p53.isin(["abnormal", "normal", "not_done"]).sum()), "agreement": round(float((d.p53.isin(["abnormal", "normal", "not_done"]) == k).mean()), 4)}
k = kw("inflammation"); agree["inflammation_stated_vs_keyword"] = {"kw_present": int(k.sum()), "field_stated": int((~d.inflammation.isin(["not_stated", "PARSE_FAIL"])).sum()), "agreement": round(float(((~d.inflammation.isin(["not_stated", "PARSE_FAIL"])) == k).mean()), 4)}
k = kw("intestinal_metaplasia"); agree["IM_present_vs_keyword"] = {"kw_present": int(k.sum()), "field_present": int((d.intestinal_metaplasia == "present").sum()), "agreement": round(float(((d.intestinal_metaplasia == "present") == k).mean()), 4)}
k = kw("ulceration_or_erosion"); agree["ulceration_yes_vs_keyword"] = {"kw_present": int(k.sum()), "field_yes": int((d.ulceration_or_erosion == "yes").sum()), "agreement": round(float(((d.ulceration_or_erosion == "yes") == k).mean()), 4)}
res["keyword_agreement"] = agree
res_ = d.SpecimenProtocol.str.upper().str.contains("RESECT|ECTOMY"); res["specimen_type_vs_protocol"] = {"protocol_resection": int(res_.sum()), "field_resection": int((d.specimen_type == "resection").sum()), "agreement_resection_flag": round(float(((d.specimen_type == "resection") == res_).mean()), 4), "crosstab": pd.crosstab(res_.map({True: "RESECTION", False: "OESOPHAGUS"}), d.specimen_type).to_dict()}
el = d[d.label_status.eq("train_eligible") & d.grade.isin(["NDBE", "IND", "LGD", "HGD", "CANCER"])]
res["grade_vs_jury_final"] = {"n": int(len(el)), "exact": round(float((el.grade == el.final_label).mean()), 4), "two_tier": round(float((el.grade.isin(["LGD", "HGD", "CANCER"]) == el.final_label.isin(["LGD", "HGD", "CANCER"])).mean()), 4), "crosstab_jury_rows": pd.crosstab(el.final_label, el.grade).to_dict()}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "p31_validation.json"), "w"), indent=1, default=str); print(json.dumps(res, indent=None, default=str)[:3000])
d[["CaseName"] + F + ["n_specimens"]].to_csv(os.path.join(OUT, "p31_fields.csv"), index=False)
# hand-check pack: 50 reports stratified over grade, redacted, laptop/cluster only
rs = np.random.RandomState(0); pick = pd.concat([g.sample(min(10, len(g)), random_state=0) for _, g in d[d.grade != "PARSE_FAIL"].groupby("grade")]).head(50)
def redact(t): t = re.sub(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", "[DATE]", str(t)); t = re.sub(r"\b(19|20)\d{2}\b", "[YEAR]", t); return re.sub(r"\b(Dr|Prof|Professor|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-zA-Z'-]+", "[CLINICIAN]", t)
os.makedirs(T + "/review_local", exist_ok=True); lines = ["# P31 hand-check pack: 50 reports with extracted fields (mark each field right/wrong)", ""]
for i, r in enumerate(pick.itertuples(), 1):
    lines += [f"## {i}. extracted: " + ", ".join(f"{f}={getattr(r, f)}" for f in F) + f", n_specimens={r.n_specimens}", "", "```", redact(r.text)[:5000], "```", ""]
open(T + "/review_local/p31_handcheck_pack.md", "w").write("\n".join(lines)); os.chmod(T + "/review_local/p31_handcheck_pack.md", 0o600); print("hand-check pack written (review_local)")
