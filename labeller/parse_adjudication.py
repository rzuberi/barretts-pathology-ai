"""Ingest Rehan's transcribed adjudications -> adjudications.csv.

Accepts messy transcript text: finds every "case <n> ... yes|no|unsure" (tolerates
filler words between). Later mentions of the same case override earlier ones.

Usage: python parse_adjudication.py transcript.txt
"""
import re, sys
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
idx = pd.read_csv(f"{HERE}/adjudication_index.csv").set_index("case_no")
text = open(sys.argv[1]).read().lower()

decisions = {}
for m in re.finditer(r"case\W+(\d+)\W+(?:\w+\W+){0,4}?(yes|no|unsure)", text):
    decisions[int(m.group(1))] = m.group(2)
print(f"parsed {len(decisions)} decisions")

missing = sorted(set(idx.index) - set(decisions))
if missing:
    print(f"NOT answered ({len(missing)}): {missing}")
rows = [{"case_no": n, "CaseName": idx.loc[n, "CaseName"], "decision": d}
        for n, d in sorted(decisions.items()) if n in idx.index]
pd.DataFrame(rows).to_csv(f"{HERE}/adjudications.csv", index=False)
print(f"wrote adjudications.csv ({len(rows)} rows) — now rerun build_erin_consensus_labels.py")
