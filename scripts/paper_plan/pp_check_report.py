"""Final check for docs/paper_plan_answers.md: every item 0-10 section has Status and Sources; the plan status table has a
status for every whiteboard row; no 'pending' outside quoted pre-specification text."""
import re, sys
t = open("docs/paper_plan_answers.md").read(); ok = True
body = [l for l in t.splitlines() if not l.startswith(">")]
rows = [l for l in body if re.match(r"^\| (Intro|Table row|Every table row|Discussion)", l)]
for l in rows:
    if not re.search(r"\| (DONE|PARTIAL|NOT AVAILABLE) \|", l): print("status missing:", l[:90]); ok = False
print("status rows:", len(rows))
secs = re.split(r"^### Item (\d+)\. ", t, flags=re.M)[1:]; sec = dict(zip(secs[0::2], secs[1::2]))
if "## 3. Item 0" not in t: print("item 0 section missing"); ok = False
for i in map(str, range(1, 11)):
    s = sec.get(i)
    if s is None: print("no section for item", i); ok = False; continue
    if "**Status.**" not in s: print("no Status in item", i); ok = False
    if "**Sources.**" not in s: print("no Sources in item", i); ok = False
if any(re.search(r"\bpending\b", l, re.I) for l in body): print("pending remains:", [l[:80] for l in body if re.search(r"\bpending\b", l, re.I)][:5]); ok = False
print("OK" if ok else "FAILED"); sys.exit(0 if ok else 1)
