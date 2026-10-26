"""Cluster side of the Paper A code sync (private repo rzuberi/erin-jury-distillation). Adapted from
barretts-pathology-ai/tools/publish/stage_and_scan.py: copies an ALLOWLIST of code files into a staging tree, masks accession-number
examples, then runs the same checks (size, extension, secrets, real report numbers, patient pseudonyms, e-mails, 8-word runs shared
with fewer than 10 real reports). Exits 1 if anything is blocked; nothing is copied out of the cluster by this script."""
import csv, json, os, re, shutil, sys
from collections import Counter
B = "/mnt/scratche/slow/fmlab/zuberi01"; T = B + "/phd/thesis"; STAGE = B + "/paperA_repo_stage"
SOURCES = [(T + "/scripts/rl/paperA", "paperA", False), (T + "/scripts/rl/explore/clf", "explore_clf", False),
           (T + "/scripts/rl", "rl_deps", False)]
RL_DEPS = {"common.py", "metrics.py", "p2_metrics.py", "evaluate.py", "train.py", "build_data.py", "build_pop.py", "race3.sh", "merge.py", "eleventh_juror.py", "task_b_second_rater.py"}
EXT = {".py", ".sh"}; MAXB = 1_000_000
if os.path.isdir(STAGE): shutil.rmtree(STAGE)
os.makedirs(STAGE); os.chmod(STAGE, 0o700); copied = []
for src, dst, rec in SOURCES:
    for f in sorted(os.listdir(src)):
        p = os.path.join(src, f); ext = os.path.splitext(f)[1]
        if not os.path.isfile(p) or ext not in EXT or (dst == "rl_deps" and f not in RL_DEPS) or os.path.getsize(p) > MAXB: continue
        rel = os.path.join(dst, f); out = os.path.join(STAGE, rel); os.makedirs(os.path.dirname(out), exist_ok=True)
        txt = open(p, errors="replace").read()
        txt = re.sub(r"\b(PS|ps)(\d{2})([-.]?)(\d{4,6})\b", lambda m: m.group(1) + "yy" + m.group(3) + "n" * len(m.group(4)), txt)
        open(out, "w").write(txt); copied.append(rel)
rep = list(csv.DictReader(open("/mnt/scratche/slow/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv", errors="ignore")))
cases = {r["CaseName"] for r in rep if r.get("CaseName")}; anons = {r["anon_id"] for r in rep if r.get("anon_id")}
def words(s): return re.findall(r"[a-z]{2,}", s.lower())
nrep = Counter()
for r in rep:
    w = words(" ".join(r.get(k, "") for k in ("FinalDiagnosis_redacted", "MicroscopicDescription_redacted", "GrossDescription_redacted", "ClinicalInformation_redacted")))
    nrep.update({hash(" ".join(w[i:i + 8])) for i in range(len(w) - 7)})
REVIEWED_GENERIC = ["barrett oesophagus with intestinal metaplasia and no dysplasia"]
shingles = {h for h, c in nrep.items() if c < 10} - {hash(s) for s in REVIEWED_GENERIC}
SECRET = re.compile(r"(barretts-[0-9a-f]{6,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}"); ALLOWED_EMAIL = {"56508673+rzuberi@users.noreply.github.com"}; PATID = re.compile(r"FMOESOPH_\d+")
problems = {}
for rel in copied:
    t = open(os.path.join(STAGE, rel), errors="replace").read(); bad = []
    if SECRET.search(t): bad.append("secret-like token")
    toks = set(re.findall(r"[A-Za-z0-9_-]{6,}", t))
    if toks & cases: bad.append(f"{len(toks & cases)} real ERIN report number(s)")
    if toks & anons or PATID.search(t): bad.append("patient pseudonym")
    em = set(EMAIL.findall(t)) - ALLOWED_EMAIL
    if em: bad.append(f"e-mail address(es): {len(em)}")
    w = words(t); hit = sum(1 for i in range(len(w) - 7) if hash(" ".join(w[i:i + 8])) in shingles)
    if hit: bad.append(f"{hit} 8-word run(s) shared with real report text")
    if bad: problems[rel] = bad
json.dump({"files_staged": len(copied), "blocked_files": problems}, open(B + "/paperA_repo_scan.json", "w"), indent=1); os.chmod(B + "/paperA_repo_scan.json", 0o600)
print(f"staged {len(copied)} files; {len(problems)} with problems")
for k, v in sorted(problems.items()): print("  BLOCK", k, "->", "; ".join(v))
sys.exit(1 if problems else 0)
