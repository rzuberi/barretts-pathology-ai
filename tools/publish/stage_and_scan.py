"""Cluster side of the public-repo sync. Copies an ALLOWLIST of code files from the live project folders into a staging tree,
masks accession-number examples in comments, then scans every staged file. Exits 1 (nothing gets published) if any check fails.

Checks: (1) size < 1 MB; (2) extension allowlist (.py .sh .js .R .toml .cfg, plus DESIGN*.md design docs); (3) secrets (tokens, keys, the grading-app
passphrase pattern); (4) real ERIN report numbers and patient pseudonyms (compared against the report table on the cluster);
(5) verbatim report text: any 8-word run shared with fewer than 10 real reports (stock phrasing found in many reports is allowed); (6) e-mail addresses other than the noreply address.
Data never leaves the cluster: only file *contents that pass* are copied out, by tools/publish/publish.sh."""
import hashlib, json, os, re, shutil, sys
B = "/mnt/scratche/slow/fmlab/zuberi01"; T = B + "/phd/thesis"; STAGE = B + "/public_repo_stage"
SOURCES = [  # (source dir, destination in repo, recursive)
    (T + "/labeller", "labeller", False), (T + "/campaigns/allslides", "campaigns/allslides", False), (T + "/scripts", "scripts", True),
    (T + "/feasibility", "feasibility", False),
] + [(f"{B}/analysis/{s}/code", f"analysis/{s}", False) for s in
     ("isbi_rerun", "juror_benchmark", "erin_handlabel", "erin_swg_linkage", "paperB_extras", "tilebudget", "outreach_panakeia", "outreach_tilebio")]
EXT = {".py", ".sh", ".js", ".R", ".toml", ".cfg"}
SKIP_DIR = {"__pycache__", ".git", "runs", "output", "logs"}
MAXB = 1_000_000
if os.path.isdir(STAGE): shutil.rmtree(STAGE)
os.makedirs(STAGE); os.chmod(STAGE, 0o700)
copied = []
for src, dst, rec in SOURCES:
    if not os.path.isdir(src): continue
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR and not d.startswith(".")] if rec else []
        for f in files:
            p = os.path.join(root, f); ext = os.path.splitext(f)[1]
            if (ext not in EXT and not (ext == ".md" and f.startswith("DESIGN"))) or ".bak" in f or f.endswith("~"): continue  # design docs only among .md
            if os.path.getsize(p) > MAXB: continue
            rel = os.path.join(dst, os.path.relpath(p, src)); out = os.path.join(STAGE, rel); os.makedirs(os.path.dirname(out), exist_ok=True)
            txt = open(p, errors="replace").read()
            txt = re.sub(r"\b(PS|ps)(\d{2})([-.]?)(\d{4,6})\b", lambda m: m.group(1) + "yy" + m.group(3) + "n" * len(m.group(4)), txt)  # mask accession examples
            open(out, "w").write(txt); copied.append(rel)
# ---------------- scan ----------------
import csv
rep = list(csv.DictReader(open("/mnt/scratche/slow/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv", errors="ignore")))
cases = {r["CaseName"] for r in rep if r.get("CaseName")}; anons = {r["anon_id"] for r in rep if r.get("anon_id")}
def words(s): return re.findall(r"[a-z]{2,}", s.lower())
from collections import Counter
nrep = Counter()   # in how many distinct reports each 8-word run occurs
for r in rep:
    w = words(" ".join(r.get(k, "") for k in ("FinalDiagnosis_redacted", "MicroscopicDescription_redacted", "GrossDescription_redacted", "ClinicalInformation_redacted")))
    nrep.update({hash(" ".join(w[i:i + 8])) for i in range(len(w) - 7)})
STOCK = 10   # runs found in >= 10 different reports are stock diagnostic phrasing; rarer runs could identify a report and block
# phrases reviewed by hand (2026-10-15): generic diagnostic wording written into prompts, not copied from a report
REVIEWED_GENERIC = ["barrett oesophagus with intestinal metaplasia and no dysplasia"]
shingles = {h for h, c in nrep.items() if c < STOCK} - {hash(s) for s in REVIEWED_GENERIC}
SECRET = re.compile(r"(barretts-[0-9a-f]{6,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}"); ALLOWED_EMAIL = {"56508673+rzuberi@users.noreply.github.com", "noreply@anthropic.com"}
PATID = re.compile(r"FMOESOPH_\d+")
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
rep_ = {"files_staged": len(copied), "blocked_files": problems}
json.dump(rep_, open(B + "/public_repo_scan.json", "w"), indent=1); os.chmod(B + "/public_repo_scan.json", 0o600)
print(f"staged {len(copied)} files; {len(problems)} with problems")
for k, v in sorted(problems.items()): print("  BLOCK", k, "->", "; ".join(v))
sys.exit(1 if problems else 0)
