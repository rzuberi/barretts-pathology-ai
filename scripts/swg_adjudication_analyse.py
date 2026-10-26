"""Agreement of record / encoder / jury with Rehan's blind adjudication (grader `swg` in the grading app).
Rehan's grade per item = worst grade ticked (NORMAL_OTHER folded into NDBE, as in swg_encoder.py). Exact agreement with
patient-bootstrap 95% CIs, by batch, by stratum (enc_disagree / jury_only_disagree / agree) and by mappable vs whole-report.
Reads the live db read-only. Usage: python swg_adjudication_analyse.py [batch]   -> prints a table, writes OUT/agreement.json
Note: strata are selected ON disagreement, so pooled numbers are not population agreement rates; the per-stratum rows are the result."""
import json, sqlite3, sys
import numpy as np, pandas as pd

B = "/mnt/scratche/slow/fmlab/zuberi01"; OUT = B + "/analysis/erin_swg_adjudication"
O5 = {"NDBE": 0, "NORMAL_OTHER": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}; R5 = {0: "NDBE", 1: "IND", 2: "LGD", 3: "HGD", 4: "CANCER"}
key = pd.read_csv(OUT + "/key_DO_NOT_OPEN_UNTIL_DONE.csv", dtype=str).fillna("")
if len(sys.argv) > 1: key = key[key.batch == sys.argv[1]]
db = sqlite3.connect(f"file:{B}/hand_grading/grading.db?mode=ro", uri=True, timeout=5)
lab = pd.read_sql("SELECT item_id, grades, ambiguous, note FROM section_labels WHERE grader='swg'", db)
lab["rehan"] = lab.grades.map(lambda s: R5[max(O5[g] for g in json.loads(s))] if json.loads(s) else "")
d = key.merge(lab[["item_id", "rehan", "ambiguous", "note"]], on="item_id", how="inner"); d = d[d.rehan != ""]
d["jury"] = np.where(d.mappable == "True", d.jury_majority_8, d.whole_report_jury)   # jury at the level Rehan graded
print(f"{len(d)} graded pairs ({d.item_id.nunique()} items) of {len(key)} | ambiguous {int(d.ambiguous.sum())}")

def ci(d, col, n_boot=2000, seed=0):
    """Exact agreement of `col` with Rehan; bootstrap over patients (pairs of a patient move together)."""
    hit = (d[col] == d.rehan).astype(float).values; pat = d.PatientID.values; ups = np.unique(pat)
    rng = np.random.RandomState(seed); est = hit.mean()
    bs = [np.concatenate([hit[pat == p] for p in rng.choice(ups, len(ups))]).mean() for _ in range(n_boot)]
    return round(100 * est, 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)

res = {}; rows = []
for name, sub in [("all", d)] + [(f"stratum={s}", d[d.stratum == s]) for s in sorted(d.stratum.unique())] + \
                 [(f"mappable={m}", d[d.mappable == m]) for m in ("True", "False")] + \
                 [(f"{s}/mappable={m}", d[(d.stratum == s) & (d.mappable == m)]) for s in sorted(d.stratum.unique()) for m in ("True", "False")]:
    if len(sub) < 3: continue
    res[name] = {"n": len(sub), "patients": int(sub.PatientID.nunique())}
    for col in ("swg", "enc_aligned", "jury"):
        res[name][col] = ci(sub, col); rows.append((name, col, len(sub)) + res[name][col])
print(pd.DataFrame(rows, columns=["rows", "vs_rehan", "n", "exact%", "lo", "hi"]).to_string(index=False))
# who is right when encoder and record disagree: Rehan sides with record / encoder / neither
ed = d[d.stratum == "enc_disagree"]
if len(ed): res["enc_disagree_verdict"] = {"record": int((ed.rehan == ed.swg).sum()), "encoder": int((ed.rehan == ed.enc_aligned).sum()), "neither": int(((ed.rehan != ed.swg) & (ed.rehan != ed.enc_aligned)).sum())}
jd = d[d.stratum == "jury_only_disagree"]
if len(jd): res["jury_disagree_verdict"] = {"record": int((jd.rehan == jd.swg).sum()), "jury": int((jd.rehan == jd.jury).sum()), "neither": int(((jd.rehan != jd.swg) & (jd.rehan != jd.jury)).sum())}
print(json.dumps({k: v for k, v in res.items() if k.endswith("verdict")}))
json.dump(res, open(OUT + "/agreement.json", "w"), indent=1)
