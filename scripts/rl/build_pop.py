"""Project 2: the hand-label POPULATION (4,929 oesophageal/GOJ sections with an H&E slide, SWG-shared excluded) as a scoring set,
so review thresholds come from the population, exactly as in the Paper B triage estimator v2. Evaluation only, never training.
Out: rl_grader/data/pop.jsonl (id, report_id, section, anon_id, label = jury worst grade, jury_vf, report)."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
pop = pd.read_csv(B + "/analysis/paperB_extras/triage/population_signals.csv", dtype=str).fillna("")
rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("").drop_duplicates("CaseName").set_index("CaseName")
out = B + "/analysis/rl_grader/data/pop.jsonl"
with open(out, "w") as f:
    for r in pop.itertuples():
        f.write(json.dumps({"id": r.sid, "report_id": r.CaseName, "section": r.section, "anon_id": r.anon_id, "label": r.worst_grade,
                            "jury_vf": float(r.S1_votefrac) if r.S1_votefrac else None, "report": report_text(rep.loc[r.CaseName])}) + "\n")
os.chmod(out, 0o600); print(len(pop), "population sections,", pop.CaseName.nunique(), "reports ->", out)
