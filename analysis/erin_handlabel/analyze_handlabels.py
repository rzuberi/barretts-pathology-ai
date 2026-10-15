"""Analysis of Rehan's completed section labels vs the jury-v3 key. Read-only on sources.
Writes analysis/erin_handlabel/human_vs_jury_labelled.csv + handlabel_results.json."""
import sqlite3, json, hashlib, os
import pandas as pd, numpy as np
H = "/mnt/scratche/slow/fmlab/zuberi01/hand_grading"; A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/erin_handlabel"
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
ORD = {"NORMAL_OTHER": -1, "NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
R = {}
db = sqlite3.connect(f"file:{H}/grading.db?mode=ro", uri=True, timeout=5)
L = pd.read_sql('select * from section_labels where grader="rehan"', db); L["ts"] = pd.to_datetime(L.ts)
items = json.load(open(H + "/items_rehan.json")); I = pd.DataFrame(items)
M = L.set_index("item_id")
R["completion"] = {"labels": len(L), "items": len(I), "last": str(L.ts.max()), "sessions": int(L.session_no.max()),
                   "labelling_minutes_sum_lt600s": round(L.seconds[L.seconds < 600].sum() / 60, 1), "median_seconds": float(L.seconds.median())}
k = pd.read_csv(A + "/key_DO_NOT_OPEN_UNTIL_DONE.csv", dtype=str).fillna(""); k["item_id"] = k.case_id + "#" + k.section

# 1. intended repeats
rep_map = k[k.repeat_case_id != ""].drop_duplicates("case_id")[["repeat_case_id", "case_id"]].sort_values("repeat_case_id")
mm = dict(zip(rep_map.repeat_case_id, rep_map.case_id))
user = {"ERIN-HL-147": "ERIN-HL-091", "ERIN-HL-148": "ERIN-HL-105", "ERIN-HL-149": "ERIN-HL-106", "ERIN-HL-150": "ERIN-HL-128", "ERIN-HL-151": "ERIN-HL-112", "ERIN-HL-153": "ERIN-HL-072"}
R["repeats"] = {"intended_mapping": mm, "user_spotted_correct": sum(mm.get(a) == b for a, b in user.items()), "user_spotted": len(user),
                "not_spotted": [r for r in mm if r not in user]}
# 2. unintended duplicates
I["h"] = I.html.map(lambda s: hashlib.md5(s.encode()).hexdigest()); I["num"] = I.case_id.str.extract(r"(\d+)$").astype(int)
dup = I[I.num <= 143].groupby("h").case_id.agg(lambda s: sorted(set(s))); dup = dup[dup.map(len) > 1]
rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("").drop_duplicates("CaseName").set_index("CaseName")
rk = k.drop_duplicates("case_id")[["case_id", "report_id"]].copy(); rk["anon"] = rk.report_id.map(rep.anon_id)
pc = rk.groupby("anon").case_id.agg(list); pc = pc[pc.map(len) > 1]
R["unintended_duplicates"] = {"identical_text_groups": dup.tolist(), "patients_with_multiple_reports_in_sample": int(len(pc)), "examples": pc.tolist()[:5]}
# 3. intra-rater
rows = []
for r, o in mm.items():
    for sec in I[I.case_id == r].section:
        a, b = M.grades.get(f"{o}#{sec}"), M.grades.get(f"{r}#{sec}")
        if a is None or b is None: continue
        ga, gb = set(json.loads(a)), set(json.loads(b))
        wa = max(ga, key=lambda x: ORD[x]) if ga else "NONE"; wb = max(gb, key=lambda x: ORD[x]) if gb else "NONE"
        rows.append({"orig": o, "rep": r, "sec": sec, "same_set": ga == gb, "same_worst": wa == wb, "wa": wa, "wb": wb,
                     "amb_same": int(M.loc[f"{o}#{sec}", "ambiguous"]) == int(M.loc[f"{r}#{sec}", "ambiguous"])})
RR = pd.DataFrame(rows)
R["intra_rater"] = {"pairs": len(RR), "same_grade_set": int(RR.same_set.sum()), "same_worst": int(RR.same_worst.sum()),
                    "ambiguous_consistent": int(RR.amb_same.sum()), "disagreements_orig_to_repeat": [(x.wa, x.wb) for x in RR[~RR.same_worst].itertuples()]}
# 4. human vs jury
k["human"] = k.item_id.map(lambda i: M.grades.get(i)); k = k[k.human.notna()].copy()
k["hset"] = k.human.map(lambda s: set(json.loads(s))); k["hworst"] = k.hset.map(lambda g: max(g, key=lambda x: ORD[x]) if g else "NONE")
for f in ["ambiguous", "p53", "imc"]: k["h_" + f] = k.item_id.map(lambda i: int(M.loc[i, f]))
k["jset"] = k.grades.map(lambda s: set(s.split("|")))
k["agree_worst"] = k.hworst == k.worst_grade; k["agree_set"] = k.hset == k.jset; k["human_in_jset"] = [h in j for h, j in zip(k.hworst, k.jset)]
k["bin_h"] = k.hworst.map(lambda g: ORD.get(g, -9) >= 2); k["bin_j"] = k.worst_grade.map(lambda g: ORD[g] >= 2)
k["jury_higher"] = k.worst_grade.map(ORD) > k.hworst.map(lambda g: ORD.get(g, -1)); k["jury_lower"] = k.worst_grade.map(ORD) < k.hworst.map(lambda g: ORD.get(g, -1))
def qwk(a, b):
    a = np.array([ORD.get(x, -1) for x in a]) + 1; b = np.array([ORD.get(x, -1) for x in b]) + 1; n = 6
    O = np.zeros((n, n))
    for i, j in zip(a, b): O[i, j] += 1
    W = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)]); E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() else float("nan")
def block(d, w=None):
    if not len(d): return {"n": 0}
    av = (lambda x: float(np.average(x, weights=w))) if w is not None else (lambda x: float(np.mean(x)))
    return {"n": int(len(d)), "exact_worst": round(av(d.agree_worst), 3), "exact_set": round(av(d.agree_set), 3), "human_in_jury_set": round(av(d.human_in_jset), 3),
            "LGDplus_agree": round(av(d.bin_h == d.bin_j), 3), "qwk": round(qwk(d.hworst, d.worst_grade), 3),
            "jury_higher": round(av(d.jury_higher), 3), "jury_lower": round(av(d.jury_lower), 3)}
T = k[k.target == "target"].copy(); T["w"] = pd.to_numeric(T.weight)
res = {"targets_unweighted": block(T), "targets_reweighted_to_population": block(T, T.w.values), "extras": block(k[k.target == "extra"]),
       "all_labelled": block(k), "random_slice": block(T[T.stratum == "random"]), "stratified": block(T[T.stratum == "stratified"]),
       "special": block(T[T.stratum == "special"])}
res["by_jury_grade"] = {g: block(k[k.worst_grade == g]) for g in ORD}
res["by_human_grade"] = {g: block(k[k.hworst == g]) for g in list(ORD) + ["NONE"]}
res["by_jury_agreement"] = {a: block(k[k.agreement == a]) for a in ["unanimous", "majority", "split"]}
res["by_human_ambiguous"] = {str(a): dict(block(k[k.h_ambiguous == a]), jury_unanimous=round(float((k[k.h_ambiguous == a].agreement == "unanimous").mean()), 3)) for a in [0, 1]}
res["by_special_tag"] = {t: dict(block(k[k.special_tags.str.contains(t)]), human_ambiguous_rate=round(float(k[k.special_tags.str.contains(t)].h_ambiguous.mean()), 3))
                         for t in ["conclusion_without_part", "cross_reference", "hedged_wording", "imc_only", "p53_in_section", "section_vs_whole_differs"]}
res["confusion_human_rows_jury_cols"] = pd.crosstab(k.hworst, k.worst_grade).reindex(index=["NONE"] + list(ORD), columns=list(ORD)).fillna(0).astype(int).to_dict()
k["rep_p53"] = k.report_id.map(lambda c: "p53" in (rep.loc[c, "FinalDiagnosis_redacted"] + rep.loc[c, "MicroscopicDescription_redacted"]).lower())
k["rep_imc"] = k.report_id.map(lambda c: "intramucosal" in (rep.loc[c, "FinalDiagnosis_redacted"] + rep.loc[c, "MicroscopicDescription_redacted"]).lower())
res["flags_vs_text"] = {"p53": pd.crosstab(k.h_p53, k.rep_p53).to_dict(), "imc": pd.crosstab(k.h_imc, k.rep_imc).to_dict()}
k["af"] = pd.to_numeric(k.agree_frac); k["afbin"] = pd.cut(k.af, [0, 0.5, 0.75, 0.99, 1.0], labels=["<=0.5", "0.5-0.75", "0.75-<1", "1.0"])
res["calibration_vote_fraction"] = k.groupby("afbin", observed=True).agg(n=("item_id", "size"), human_agree=("agree_worst", "mean"), human_ambiguous=("h_ambiguous", "mean")).round(3).reset_index().astype(str).to_dict("records")
res["in_1538_targets"] = block(T[T.in_1538 == "True"])
R["human_vs_jury"] = res
k.to_csv(A + "/human_vs_jury_labelled.csv", index=False); os.chmod(A + "/human_vs_jury_labelled.csv", 0o600)
json.dump(R, open(A + "/handlabel_results.json", "w"), indent=1, default=str); os.chmod(A + "/handlabel_results.json", 0o600)
print(json.dumps(R, indent=1, default=str))
