"""Follow-up B: second-rater agreement (Rehan vs Jonathan) and models vs each rater. Reads the live grading DB READ-ONLY.
Out: rl_followups/B/results.json. Partial data are reported with n."""
import json, os, sqlite3, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
H = B + "/hand_grading"; HL = B + "/analysis/erin_handlabel"; R = B + "/analysis/rl_grader"; O = B + "/analysis/rl_followups/B"
con = sqlite3.connect(f"file:{H}/grading.db?mode=ro", uri=True, timeout=5)
L = pd.read_sql("select grader, item_id, grades from section_labels where grader in ('rehan','jonathan')", con)
key = pd.read_csv(HL + "/key_DO_NOT_OPEN_UNTIL_DONE.csv", dtype=str).fillna(""); key["item_id"] = key.case_id + "#" + key.section   # first-occurrence items only (repeats excluded)
def worst(s):
    try: g = json.loads(s)
    except Exception: g = str(s).split("|")
    g = [x.strip() for x in g if x.strip() in ORD]; return max(g, key=lambda x: ORD[x]) if g else None
L["worst"] = L.grades.map(worst); L = L.dropna(subset=["worst"])
P = L.pivot_table(index="item_id", columns="grader", values="worst", aggfunc="first").reset_index()
P = P.merge(key[["item_id", "section_id"]], on="item_id", how="inner")
lab = {json.loads(l)["id"]: json.loads(l) for l in open(R + "/data/test485.jsonl")}
P = P[P.section_id.isin(lab)].copy()
n_j = int(P.get("jonathan", pd.Series(dtype=str)).notna().sum()) if "jonathan" in P else 0
res = {"n_rehan": int(P["rehan"].notna().sum()), "n_jonathan": n_j, "status": "complete" if n_j >= len(lab) else ("partial" if n_j else "no Jonathan grades yet")}
if n_j:
    both = P.dropna(subset=["rehan", "jonathan"]).copy(); both["w"] = both.section_id.map(lambda i: lab[i]["w"])
    def qwk(a, b):
        a = np.array([ORD[x] for x in a]); b = np.array([ORD[x] for x in b]); k = len(GRADES); Oc = np.zeros((k, k))
        for i, j in zip(a, b): Oc[i, j] += 1
        W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]); E = np.outer(Oc.sum(1), Oc.sum(0)) / Oc.sum()
        return float(1 - (W * Oc).sum() / (W * E).sum())
    wm = lambda m: float((m * both.w).sum() / both.w.sum())
    res["raters"] = {"n_both": len(both), "exact": round(float((both.rehan == both.jonathan).mean()), 4), "exact_weighted": round(wm(both.rehan == both.jonathan), 4),
                     "dysplasia_agree": round(float((both.rehan.isin(POS) == both.jonathan.isin(POS)).mean()), 4), "qwk": round(qwk(both.rehan, both.jonathan), 4),
                     "confusion_rows_rehan": pd.crosstab(both.rehan, both.jonathan).reindex(index=GRADES, columns=GRADES, fill_value=0).to_dict("index")}
    preds = {"jury": {i: lab[i]["label"] for i in lab}}
    for t, f in (("RL model", R + "/preds/grpo_sft__checkpoint-300__test485.jsonl"), ("MedGemma-27B", R + "/preds/medgemma27b_zs__test485.jsonl")):
        preds[t] = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(f)}
    agree = both[both.rehan == both.jonathan]
    res["models"] = {}
    for m, p in preds.items():
        g = both.section_id.map(p); ga = agree.section_id.map(p)
        res["models"][m] = {"vs_rehan": round(wm(g == both.rehan), 4), "vs_jonathan": round(wm(g == both.jonathan), 4),
                            "vs_agreed": round(float(((ga == agree.rehan) * agree.section_id.map(lambda i: lab[i]["w"])).sum() / agree.section_id.map(lambda i: lab[i]["w"]).sum()), 4) if len(agree) else None,
                            "n_agreed": int(len(agree))}
os.makedirs(O, exist_ok=True); json.dump(res, open(O + "/results.json", "w"), indent=1); print(json.dumps(res, indent=1)[:1500])
