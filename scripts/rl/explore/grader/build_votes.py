"""Jury vote shares per (report, section) for the soft-label reward (CPU, once). READ-ONLY on sources.
Out: analysis/rl_explore/grader/votes.json  {id: {grade: share}} for every id in train.jsonl + all_lowconf.jsonl (no report text)."""
import json, os, sys
import pandas as pd
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl"); from common import *
D = B + "/analysis/rl_grader/data"; O = B + "/analysis/rl_explore/grader"; os.makedirs(O, exist_ok=True); os.chmod(O, 0o700)
ids = {}
for f in ("train", "all_lowconf"):
    for l in open(f"{D}/{f}.jsonl"):
        r = json.loads(l); ids[r["id"]] = (r["label"], r.get("vf"))
sec = pd.read_csv(B + "/phd/thesis/labeller/erin_sections_v3_consensus.csv", dtype=str).fillna("")
sec["id"] = sec.CaseName + "#" + sec.section
out = {}; mism = 0
for r in sec[sec.id.isin(ids)].itertuples():
    try: v = json.loads(r.grade_votes); n = int(r.n_jurors)
    except Exception: continue
    sh = {g: k / n for g, k in v.items() if g in ORD}
    lab, vf = ids[r.id]
    if vf is not None and abs(sh.get(lab, 0) - vf) > 0.01: mism += 1
    out[r.id] = sh
json.dump(out, open(O + "/votes.json", "w")); os.chmod(O + "/votes.json", 0o600)
cov = sum(i in out for i in ids); print(f"ids {len(ids)}, with votes {cov}, vf mismatches {mism}")
import numpy as np
sh_lab = [out[i].get(ids[i][0], 0) for i in ids if i in out]; print("share of consensus label: mean %.3f, min %.2f, frac<0.75 %.3f" % (np.mean(sh_lab), min(sh_lab), np.mean(np.array(sh_lab) < 0.75)))
