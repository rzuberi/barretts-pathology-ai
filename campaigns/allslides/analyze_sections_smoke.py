"""Jury v3 smoke gate: per-candidate parse rate on the 80 adjudicated cases
(pre-registered gate: >=95% parse, never agreement), plus soft checks —
sections/report, site distribution, worst-grade CANCER vs the adjudication.
Usage: python analyze_sections_smoke.py <runs_root>  (globs sec3smoke_*/output/sections_v3_*.csv)
"""
import glob, json, os, sys
from collections import Counter
import pandas as pd

ROOT = sys.argv[1]
ADJ = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/labeller/adjudications.csv"
adj = pd.read_csv(ADJ, dtype=str).drop_duplicates("CaseName")
adj = adj[adj.decision.isin(["yes", "no"])].set_index("CaseName").decision
ORD = {"NORMAL_OTHER": -1, "NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
res = {}
for f in sorted(glob.glob(os.path.join(ROOT, "sec3smoke_*/output/sections_v3_*.csv"))):
    model = os.path.basename(f).replace("sections_v3_", "").rsplit("_shard", 1)[0]
    d = pd.read_csv(f, dtype=str).fillna("").drop_duplicates("CaseName")
    ok = ~d.sections_json.str.startswith("PARSE_FAIL")
    fails = Counter(x.split(":")[1] if ":" in x else x for x in d.sections_json[~ok])
    nsec, sites, sp, worst = [], Counter(), Counter(), {}
    for _, r in d[ok].iterrows():
        secs = json.loads(r.sections_json); nsec.append(len(secs))
        for s in secs: sites[s["site"]] += 1; sp[s["specimen_type"]] += 1
        worst[r.CaseName] = max((ORD.get(g, -9) for s in secs for g in s["grades"]), default=-9)
    both = [c for c in worst if c in adj.index]
    acc = sum((worst[c] == 4) == (adj[c] == "yes") for c in both) / len(both) if both else None
    secs_time = pd.to_numeric(d.secs, errors="coerce")
    res[model] = {"n": int(len(d)), "parse_rate": round(float(ok.mean()), 4), "fail_reasons": dict(fails),
                  "PASS_GATE_95": bool(ok.mean() >= 0.95),
                  "sections_per_report": round(sum(nsec) / max(len(nsec), 1), 2),
                  "site_dist": dict(sites.most_common()), "specimen_dist": dict(sp.most_common()),
                  "cancer_acc_vs_adjudication": (round(acc, 4) if acc is not None else None), "n_adj": len(both),
                  "sec_per_report_median": (round(float(secs_time.median()), 1) if secs_time.notna().any() else None)}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, "sections_v3_smoke_analysis.json"), "w"), indent=1)
print("\nGATE:", {m: ("PASS" if v["PASS_GATE_95"] else "FAIL") for m, v in res.items()})
