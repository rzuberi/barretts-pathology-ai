"""SWG sWGS-sample grade vs LLM jury on the Barrett's-DB report x block set.
READ-ONLY on all sources. Outputs -> /mnt/scratche/slow/fmlab/zuberi01/analysis/erin_swg_linkage/

Steps: tiers (accession+block / report-level; dates via receiveddatetime), section-level
mappability, metrics on the common subset (all 8 jurors parsed), rule-based baseline,
jury variants with a 50/50 selection/report split, disagreement export for manual
adjudication, summary JSON. Nothing here adjudicates.
"""
import glob, json, os, re, hashlib
from collections import Counter, defaultdict
import numpy as np, pandas as pd

EXP = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
XL = "/mnt/scratche/slow/fmlab/zuberi01/phd/Leanne_shared_docs/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3).xlsx"
REL = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
OUT = "/mnt/scratche/slow/fmlab/zuberi01/analysis/erin_swg_linkage"
os.makedirs(OUT, exist_ok=True); os.chmod(OUT, 0o700)
ORD = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
GR = list(ORD)
MAP = {"BE": "NDBE", "ID": "IND", "LGD": "LGD", "HGD": "HGD", "IMC": "CANCER", "OAC": "CANCER"}
rng = np.random.RandomState(20261001)

def acc_norm(s):
    s = re.sub(r"[\s_][A-Za-z]{1,2}\d{0,2}\s*(\(.*\))?\s*$", "", str(s).strip())
    m = re.match(r"^\s*([A-Za-z]{1,3})\s*(\d{2})\s*[-./]?\s*(\d{3,6})\s*$", s)
    return f"{m.group(1).upper()}{m.group(2)}-{int(m.group(3)):05d}" if m else None

# ---------------- linked set ----------------
sp = pd.read_parquet(EXP + "/specimen_pairs.parquet")          # CaseName = path_id@report_id
sp["acc"] = sp.path_id.map(acc_norm)
sp["swg"] = sp.swg_grade_code.str.upper().str.strip().map(MAP)
rep = pd.read_parquet(EXP + "/swg_matched_reports_v2.parquet")
rep["report_id"] = rep.pathology_text_id.astype(float).astype(int).astype(str)
rep["db_date"] = pd.to_datetime(rep.receiveddatetime, errors="coerce")
sp = sp.merge(rep[["report_id", "db_date", "participant_id", "highestgradedysconf", "highestgradedysprov"]], on="report_id", how="left")
x = pd.read_excel(XL, dtype=str); x["acc"] = x["Path ID"].map(acc_norm)
x["path_key"] = x["Path ID"].str.strip(); x = x.drop_duplicates("path_key")
sp = sp.merge(x[["path_key", "PatientID", "Hospital Research ID", "Endoscopy Year", "Block", "Esophageal location", "Pathology", "notes on label review", "consistency after review"]]
              .rename(columns={"path_key": "path_id"}), on="path_id", how="left")
coh = pd.read_csv(REL + "/pre_event_cohort.csv", dtype=str); coh["acc"] = coh.BiopsyID_real.map(acc_norm)
coh["swg_date"] = pd.to_datetime(coh.Date, errors="coerce")
cdate = coh.dropna(subset=["acc"]).drop_duplicates("acc").set_index("acc")[["swg_date", "GradeSource", "DateSource", "CurrentGradeNorm"]]
sp = sp.join(cdate, on="acc")
sp["dd"] = (sp.db_date - sp.swg_date).dt.days

tiers = {"A_accession_block_pairs": int(len(sp)), "A_distinct_swg_samples": int(sp.path_id.nunique()),
         "A_distinct_reports": int(sp.report_id.nunique()), "A_patients_xlsx": int(sp.PatientID.nunique()),
         "samples_matching_more_than_one_report": int((sp.groupby("path_id").report_id.nunique() > 1).sum()),
         "reports_with_more_than_one_sample": int((sp.groupby("report_id").path_id.nunique() > 1).sum()),
         "date_available_both_sides": int(sp.dd.notna().sum()),
         "B_same_day": int((sp.dd == 0).sum()), "C_within_7d_not_same": int(((sp.dd.abs() <= 7) & (sp.dd != 0)).sum()),
         "date_offset_gt_7d": int((sp.dd.abs() > 7).sum()),
         "date_offsets_days_dist": {str(k): int(v) for k, v in sp.dd.dropna().astype(int).value_counts().sort_index().items()},
         "swg_grade_dist_all_pairs": sp.swg.value_counts().to_dict(),
         "xlsx_vs_release_grade_agree": int(((sp.CurrentGradeNorm.map(lambda g: MAP.get(g, g)) == sp.swg) & sp.CurrentGradeNorm.notna()).sum()),
         "xlsx_vs_release_grade_compared": int(sp.CurrentGradeNorm.notna().sum())}

# ---------------- section-level mappability ----------------
MARK = re.compile(r"(?:(?<=^)|(?<=\s)|(?<=\n))([A-J])\s?(?:[.)\]:]|-(?=\s))\s*(?=\S)")
NUMMARK = re.compile(r"(?:^|\n)\s*\(?(\d{1,2})\s?[.):\]]\s*(?=\S)")       # DB house style: "1.", "2)" at line start
PARTREF = re.compile(r"(?i)\b(?:part|specimen|pot|biopsy)s?\s*(\d{1,2})\b")
def body(t): return re.sub(r"^SPECIMEN OF INTEREST.*?\n\n", "", str(t), flags=re.S)
sp["body"] = sp.reporttext.map(body)
sp["letters"] = sp.body.map(lambda t: "".join(sorted(set(MARK.findall(t)))))
sp["parts"] = sp.body.map(lambda t: sorted(set(int(n) for n in NUMMARK.findall(t)) | set(int(n) for n in PARTREF.findall(t))))
sp["n_parts"] = sp.parts.map(len)
sp["suffix_kind"] = sp.suffix.map(lambda s: "block_number" if re.fullmatch(r"[A-Za-z]\d{1,2}", str(s)) else ("letters" if re.fullmatch(r"[A-Za-z]{2}", str(s)) else "none"))
sp["suffix_num"] = sp.suffix.map(lambda s: int(re.sub(r"\D", "", str(s))) if re.fullmatch(r"[A-Za-z]\d{1,2}", str(s)) else None)
sp["digit_in_parts"] = sp.apply(lambda r: r.suffix_num is not None and r.suffix_num in r.parts, axis=1)
sp["section_mappable"] = sp.digit_in_parts | ((sp.suffix_kind == "letters") & sp.apply(lambda r: str(r.suffix)[-1].upper() in r.letters, axis=1))
sec_counts = {"pairs": int(len(sp)), "reports_single_part_or_unnumbered": int((sp.n_parts <= 1).sum()),
              "reports_multi_part": int((sp.n_parts >= 2).sum()),
              "suffix_kind": sp.suffix_kind.value_counts().to_dict(),
              "suffix_digit_present_among_report_parts": int(sp.digit_in_parts.sum()),
              "suffix_digit_and_report_multi_part": int((sp.digit_in_parts & (sp.n_parts >= 2)).sum()),
              "mappable_to_a_part_or_section": int(sp.section_mappable.sum()),
              "whole_report_only": int((~sp.section_mappable).sum()),
              "note": "DB reports number specimens ('1.', 'Part 2'); hypothesis tested below: SWG block suffix digit (B1/B2) = report part number. Lettered sections are rare in this corpus."}

# ---------------- juror votes ----------------
def load_votes(d):
    v = {}
    for f in glob.glob(d + "/llm_grades_*.csv"):
        m = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]
        try: df = pd.read_csv(f, dtype=str)
        except Exception:
            recs = []
            for line in open(f).read().splitlines()[1:]:
                n, g = line.rsplit(",", 1); recs.append({"CaseName": n.strip('"'), "llm_grade": g})
            df = pd.DataFrame(recs)
        v.setdefault(m, {}).update(dict(zip(df.CaseName, df.llm_grade)))
    return v
spec = load_votes(EXP + "/jury_specimen")            # keyed by path_id@report_id
whole = load_votes(EXP + "/jury")                    # keyed by report_id
JUR = sorted(spec)
for m in JUR:
    sp[f"spec_{m}"] = sp.CaseName.map(spec[m]); sp[f"whole_{m}"] = sp.report_id.map(whole.get(m, {}))
def clean(s): return s.where(s.isin(GR))
spec_cols = [f"spec_{m}" for m in JUR]; whole_cols = [f"whole_{m}" for m in JUR]
for c in spec_cols + whole_cols: sp[c] = clean(sp[c])
common = sp[sp[spec_cols].notna().all(axis=1) & sp.swg.notna()].copy()
common_whole = sp[sp[whole_cols].notna().all(axis=1) & sp.swg.notna()].copy()

# ---------------- rule-based baseline ----------------
NEG = re.compile(r"\b(no|not|without|negative for|free of|absence of|no evidence of|insufficient for)\b", re.I)
LADDER = [("NDBE", [r"\bbarrett", r"\bintestinal metaplasia\b", r"\bnegative for dysplasia\b", r"\bno (?:evidence of )?dysplasia\b", r"\bcolumnar(?:[- ]lined)? (?:mucosa|epithelium)\b"]),
          ("IND", [r"\bindefinite for dysplasia\b", r"\bindefinite\b", r"\batypia[^.]{0,60}(?:indefinite|uncertain|cannot exclude)"]),
          ("LGD", [r"\blow[- ]grade (?:glandular )?dysplasia\b", r"\bLGD\b"]),
          ("HGD", [r"\bhigh[- ]grade (?:glandular )?dysplasia\b", r"\bHGD\b"]),
          ("CANCER", [r"\badenocarcinoma\b", r"\bcarcinoma\b", r"\bmalignan(?:t|cy)\b", r"\bintramucosal (?:adeno)?carcinoma\b"])]
def ladder_grade(text):
    best = None
    for sent in re.split(r"(?<=[.;:\n])\s+", str(text)):
        for li, (g, pats) in enumerate(LADDER):
            for p in pats:
                for m in re.finditer(p, sent, re.I):
                    pre = sent[:m.start()]
                    if any(m.start() - n.end() <= 130 for n in NEG.finditer(pre)): continue
                    if best is None or li > best: best = li
    return GR[best] if best is not None else None
def split_sections(text):
    ms = list(MARK.finditer(text)); out = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.setdefault(m.group(1), []).append(text[m.end():end])
    return {k: " ".join(v) for k, v in out.items()}
def split_parts(text):
    ms = list(NUMMARK.finditer(text)); out = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.setdefault(int(m.group(1)), []).append(text[m.end():end])
    return {k: " ".join(v) for k, v in out.items()}
def rule_section_grade(r):
    secs = split_sections(r.body); parts = split_parts(r.body)
    if r.suffix_kind == "letters" and str(r.suffix)[-1].upper() in secs: return ladder_grade(secs[str(r.suffix)[-1].upper()])
    if r.suffix_num is not None and r.suffix_num in parts and len(parts) >= 2: return ladder_grade(parts[r.suffix_num])
    return ladder_grade(r.body)          # unmappable: whole-report worst (explicit fallback)
def rule_other_part_grade(r):
    parts = split_parts(r.body)
    if r.suffix_num is not None and r.suffix_num in parts and len(parts) >= 2:
        others = [ladder_grade(t) for k, t in parts.items() if k != r.suffix_num]; others = [g for g in others if g]
        return max(others, key=lambda g: ORD[g]) if others else None
    return None
sp["rule_whole"] = sp.body.map(ladder_grade)
sp["rule_section"] = sp.apply(rule_section_grade, axis=1)
sp["rule_other_parts"] = sp.apply(rule_other_part_grade, axis=1)
common["rule_whole"] = sp.loc[common.index, "rule_whole"]; common["rule_section"] = sp.loc[common.index, "rule_section"]

# ---------------- metrics ----------------
def qwk(a, b):
    a, b = np.asarray([ORD[v] for v in a]), np.asarray([ORD[v] for v in b]); n = 5
    O = np.zeros((n, n));
    for i, j in zip(a, b): O[i, j] += 1
    W = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() else float("nan")
def metrics(pred, truth):
    d = pd.DataFrame({"p": pred, "t": truth}).dropna(); d = d[d.p.isin(GR) & d.t.isin(GR)]
    if not len(d): return {"n": 0}
    cm = pd.crosstab(d.p, d.t).reindex(index=GR, columns=GR, fill_value=0)
    per = {}
    for g in GR:
        tp = cm.loc[g, g]; fp = cm.loc[g].sum() - tp; fn = cm[g].sum() - tp
        pr = tp / (tp + fp) if tp + fp else float("nan"); rc = tp / (tp + fn) if tp + fn else float("nan")
        per[g] = {"n_truth": int(cm[g].sum()), "precision": round(pr, 3) if pr == pr else None, "recall": round(rc, 3) if rc == rc else None,
                  "f1": round(2 * pr * rc / (pr + rc), 3) if pr == pr and rc == rc and pr + rc else None}
    dis = d[d.p != d.t]; over = (dis.p.map(ORD) > dis.t.map(ORD)).sum()
    return {"n": int(len(d)), "exact": round(float((d.p == d.t).mean()), 4), "qwk": round(qwk(d.p, d.t), 4),
            "binary_LGDplus_agreement": round(float(((d.p.map(ORD) >= 2) == (d.t.map(ORD) >= 2)).mean()), 4),
            "over_graded_frac_all": round(float((d.p.map(ORD) > d.t.map(ORD)).mean()), 4),
            "under_graded_frac_all": round(float((d.p.map(ORD) < d.t.map(ORD)).mean()), 4),
            "disagreements": int(len(dis)), "frac_of_disagreements_over": round(float(over / len(dis)), 4) if len(dis) else None,
            "per_class": per, "confusion_pred_rows_truth_cols": cm.to_dict()}
def majority(df, cols, weights=None):
    out = []
    for _, r in df[cols].iterrows():
        c = Counter()
        for col in cols:
            if r[col] in GR: c[r[col]] += (weights or {}).get(col, 1.0)
        out.append(max(c.items(), key=lambda kv: (kv[1], -ORD[kv[0]]))[0] if c else None)
    return pd.Series(out, index=df.index)

R = {"common_subset_specimen": {"n": int(len(common)), "swg_dist": common.swg.value_counts().to_dict()},
     "common_subset_whole": {"n": int(len(common_whole))}}
R["specimen_scoped"] = {m: metrics(common[f"spec_{m}"], common.swg) for m in JUR}
R["specimen_scoped"]["JURY_majority_8"] = metrics(majority(common, spec_cols), common.swg)
R["specimen_scoped"]["JURY_excl_llama3.1_deepseek"] = metrics(majority(common, [c for c in spec_cols if "llama3.1" not in c and "deepseek" not in c]), common.swg)
R["whole_report"] = {m: metrics(common_whole[f"whole_{m}"], common_whole.swg) for m in JUR}
R["whole_report"]["JURY_majority_8"] = metrics(majority(common_whole, whole_cols), common_whole.swg)
mp = common[sp.loc[common.index, "digit_in_parts"] & (sp.loc[common.index, "n_parts"] >= 2)]
R["rule_based"] = {"whole_report_ladder": metrics(common.rule_whole, common.swg), "part_or_whole_ladder": metrics(common.rule_section, common.swg),
                   "coverage_whole": int(common.rule_whole.notna().sum()), "coverage_section": int(common.rule_section.notna().sum()),
                   "suffix_digit_hypothesis": {"n_pairs_multi_part_with_digit": int(len(mp)),
                       "grade_of_part_matching_suffix_digit_vs_record": metrics(mp.rule_section, mp.swg) if len(mp) else {"n": 0},
                       "worst_grade_of_OTHER_parts_vs_record": metrics(sp.loc[mp.index, "rule_other_parts"], mp.swg) if len(mp) else {"n": 0},
                       "whole_report_ladder_vs_record_same_pairs": metrics(mp.rule_whole, mp.swg) if len(mp) else {"n": 0}}}

# ---------------- selection on 50%, report on other 50% (split by SWG patient) ----------------
pats = common.PatientID.fillna(common.path_id).unique(); rng.shuffle(pats)
sel_p = set(pats[: len(pats) // 2]); S = common[common.PatientID.fillna(common.path_id).isin(sel_p)]; Tt = common[~common.PatientID.fillna(common.path_id).isin(sel_p)]
acc_S = {m: float((S[f"spec_{m}"] == S.swg).mean()) for m in JUR}
top3 = sorted(acc_S, key=acc_S.get, reverse=True)[:3]
w = {f"spec_{m}": acc_S[m] for m in JUR}
R["held_out_variants"] = {"split": "random 50/50 by SWG patient, seed 20261001", "n_select": int(len(S)), "n_report": int(len(Tt)),
                          "juror_accuracy_on_selection_half": {m: round(a, 3) for m, a in acc_S.items()}, "top3_selected": top3,
                          "report_half": {"JURY_majority_8": metrics(majority(Tt, spec_cols), Tt.swg),
                                          "top3_majority": metrics(majority(Tt, [f"spec_{m}" for m in top3]), Tt.swg),
                                          "accuracy_weighted_vote": metrics(majority(Tt, spec_cols, w), Tt.swg),
                                          "excl_llama3.1_deepseek": metrics(majority(Tt, [c for c in spec_cols if "llama3.1" not in c and "deepseek" not in c]), Tt.swg),
                                          "best_single_on_selection_half": {top3[0]: metrics(Tt[f"spec_{top3[0]}"], Tt.swg)},
                                          "rule_section_or_whole": metrics(Tt.rule_section, Tt.swg)}}

# ---------------- exports ----------------
common["jury_majority_8"] = majority(common, spec_cols); common["jury_excl_llama_deepseek"] = majority(common, [c for c in spec_cols if "llama3.1" not in c and "deepseek" not in c])
common["whole_report_jury"] = sp.loc[common.index, whole_cols].apply(lambda r: majority(pd.DataFrame([r]), whole_cols).iloc[0], axis=1)
link_cols = ["path_id", "report_id", "suffix", "suffix_kind", "suffix_num", "parts", "n_parts", "digit_in_parts", "swg", "Pathology", "CurrentGradeNorm", "GradeSource", "PatientID", "Endoscopy Year", "Block", "Esophageal location",
             "swg_date", "db_date", "dd", "letters", "section_mappable", "highestgradedysconf", "highestgradedysprov", "consistency after review", "notes on label review"]
sp[link_cols].to_csv(OUT + "/linkage_swg_db.csv", index=False)
exp_cols = link_cols + ["jury_majority_8", "jury_excl_llama_deepseek", "whole_report_jury", "rule_whole", "rule_section"] + spec_cols + whole_cols
dis = common[common.jury_majority_8 != common.swg].copy()
dis["block_section_text"] = dis.apply(lambda r: (split_sections(r.body).get(str(r.suffix)[-1].upper(), "") if r.suffix_kind == "letters" else (split_parts(r.body).get(r.suffix_num, "") if r.suffix_num is not None else "")), axis=1)
dis["report_text"] = dis.body
dis["rehan_adjudication"] = ""          # LLM_wrong | record_wrong | ambiguous
dis["rehan_note"] = ""
dis = dis.sort_values(["swg", "jury_majority_8"])
dis[exp_cols + ["block_section_text", "report_text", "rehan_adjudication", "rehan_note"]].to_csv(OUT + "/disagreements_for_adjudication.csv", index=False)
common[exp_cols].to_csv(OUT + "/common_subset_with_all_grades.csv", index=False)
for f in os.listdir(OUT):
    if os.path.isfile(os.path.join(OUT, f)): os.chmod(os.path.join(OUT, f), 0o600)
R["tiers"] = tiers; R["section_level"] = sec_counts; R["disagreements_exported"] = int(len(dis))
R["disagreement_pattern"] = {"jury_over_record": int((dis.jury_majority_8.map(ORD) > dis.swg.map(ORD)).sum()), "jury_under_record": int((dis.jury_majority_8.map(ORD) < dis.swg.map(ORD)).sum()),
                             "in_single_part_or_unnumbered_reports": int((dis.n_parts <= 1).sum()), "in_multi_part_reports": int((dis.n_parts >= 2).sum()), "suffix_digit_mapped_to_a_part": int(dis.digit_in_parts.sum()),
                             "whole_report_jury_equals_record": int((dis.whole_report_jury == dis.swg).sum()),
                             "record_grade_appears_as_rule_grade_of_some_part": int(dis.apply(lambda r: r.swg in {ladder_grade(t) for t in list(split_parts(r.body).values()) + list(split_sections(r.body).values())}, axis=1).sum())}
json.dump(R, open(OUT + "/summary_metrics.json", "w"), indent=1, default=lambda o: None)
print(json.dumps({k: R[k] for k in ["tiers", "section_level", "common_subset_specimen", "disagreements_exported", "disagreement_pattern"]}, indent=1))
def row(name, m): return f"| {name} | {m['n']} | {m['exact']:.3f} | {m['qwk']:.3f} | {m['binary_LGDplus_agreement']:.3f} | {m['over_graded_frac_all']:.3f} | {m['under_graded_frac_all']:.3f} |"
print("\n| model (specimen-scoped, common n) | n | exact | QWK | LGD+ agree | over | under |\n|---|---|---|---|---|---|---|")
for k, m in R["specimen_scoped"].items(): print(row(k, m))
print(row("rule: whole-report ladder", R["rule_based"]["whole_report_ladder"])); print(row("rule: part-or-whole ladder", R["rule_based"]["part_or_whole_ladder"])); print("suffix-digit hypothesis:", json.dumps({k: (v.get("n"), v.get("exact"), v.get("qwk")) for k, v in R["rule_based"]["suffix_digit_hypothesis"].items() if isinstance(v, dict)}))
print("\n| whole-report jury vs block grade | n | exact | QWK | LGD+ | over | under |\n|---|---|---|---|---|---|---|")
for k in ["JURY_majority_8"]: print(row(k, R["whole_report"][k]))
print("\nHELD-OUT (report half):", {k: (v["exact"], v["qwk"]) for k, v in R["held_out_variants"]["report_half"].items() if "exact" in v}, "top3:", top3)
