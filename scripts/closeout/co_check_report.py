"""Final check for docs/closeout_for_review.md: every item A-N has a section with Status and Sources lines, the summary
table has a status for every item, no 'pending' placeholders remain, and every numeric table row lies inside a section
that carries a Sources line (so every number has a source)."""
import re, sys
t = open("docs/closeout_for_review.md").read(); ok = True
items = "ABCDEFGHIJKLMN"
summ = {m.group(1): m.group(2) for m in re.finditer(r"^\| ([A-N]) \| (DONE|PARTIAL|NOT AVAILABLE|PENDING) \|", t, re.M)}
for i in items:
    if summ.get(i) not in ("DONE", "PARTIAL", "NOT AVAILABLE"): print("summary status missing/invalid for", i, summ.get(i)); ok = False
secs = re.split(r"^### ([A-N])\. ", t, flags=re.M)[1:]
sec = dict(zip(secs[0::2], secs[1::2]))
for i in items:
    s = sec.get(i)
    if s is None: print("no section", i); ok = False; continue
    if "**Status.**" not in s: print("no Status in", i); ok = False
    if "**Sources.**" not in s: print("no Sources in", i); ok = False
    if re.search(r"\|\s*-?\d", s) and "**Sources.**" not in s: print("numbers without sources in", i); ok = False
body = [l for l in t.splitlines() if not l.startswith(">") and '"nothing pending"' not in l]   # skip the verbatim quoted pre-specification and the quoted digest phrase
if any(re.search(r"\bpending\b", l, re.I) for l in body): print("'pending' placeholder remains:", [l[:80] for l in body if re.search(r"\bpending\b", l, re.I)]); ok = False
print("sections:", len(sec), "summary rows:", len(summ), "OK" if ok else "FAILED"); sys.exit(0 if ok else 1)
