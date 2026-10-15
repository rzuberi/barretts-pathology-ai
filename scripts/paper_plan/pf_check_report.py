"""Final check for docs/paper_plan_followup.md: status table rows F1-F9 with a status, each item section with Status and
Sources, no 'pending' outside quoted pre-specification text."""
import re, sys
t = open("docs/paper_plan_followup.md").read(); ok = True; body = [l for l in t.splitlines() if not l.startswith(">")]
rows = [l for l in body if re.match(r"^\| F\d ", l)]
for l in rows:
    if not re.search(r"\| (DONE|PARTIAL|NOT AVAILABLE) \|", l): print("status missing:", l[:80]); ok = False
if len(rows) != 9: print("status rows:", len(rows)); ok = False
secs = re.split(r"^### F(\d)\. ", t, flags=re.M)[1:]; sec = dict(zip(secs[0::2], secs[1::2]))
for i in map(str, range(1, 10)):
    s = sec.get(i)
    if s is None: print("no section F", i); ok = False; continue
    if "**Status.**" not in s: print("no Status in F", i); ok = False
    if "**Sources.**" not in s: print("no Sources in F", i); ok = False
if any(re.search(r"\bpending\b", l, re.I) for l in body): print("pending remains:", [l[:80] for l in body if re.search(r"\bpending\b", l, re.I)][:5]); ok = False
print("OK" if ok else "FAILED"); sys.exit(0 if ok else 1)
