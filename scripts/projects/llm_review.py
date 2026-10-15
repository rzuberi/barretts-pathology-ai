"""Local-LLM review panel (Track C). Builds a review pack from the repo's AGGREGATE docs only (no report text, no
patient-level rows): the three project MDs, the 24 Sep status ledger, the digest's summary table and the relevant claims.
Each model in MODELS is asked, via local ollama on the job GPU, to act as a hostile senior reviewer and return a
structured critique. Output: one markdown file with every model's verbatim answer. Env: MODELS (comma), OUTDIR, NUM_CTX."""
import json, os, re, subprocess, time, urllib.request
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", "."); MODELS = os.environ.get("MODELS", "qwen3:32b,gemma3:27b,medgemma:27b").split(","); NUM_CTX = int(os.environ.get("NUM_CTX", "24576"))
os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models"); PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000; os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"; BASE = f"http://127.0.0.1:{PORT}"
srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=open(os.path.join(OUT, "ollama.log"), "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try: urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
def rd(p, n=None): s = open(p).read(); return s[:n] if n else s
def section(md, start, end=None):
    i = md.find(start); j = md.find(end, i + 1) if end else -1; return md[i:j] if i >= 0 else ""
digest = rd(T + "/docs/results_digest_2026-09-21_22.md"); ledger = rd(T + "/docs/status_ledger_2026-09-24.md"); claims = rd(T + "/docs/claims_register.md")
pack = "\n\n".join([
    "# CONTEXT: PhD thesis (Barrett's oesophagus / oesophageal adenocarcinoma), histology-anchored multimodal deep learning. Cohorts: SWG (150 patients, H&E + shallow WGS copy number, progression endpoint), ERIN (2,537 patients, 7,149 pathology reports, ~9,500 H&E slides, labels derived from report text by an 8-model local-LLM jury), OCCAMS (OAC resections). Everything below is aggregate; no patient data.",
    "# CLAIMS REGISTER (excerpt)\n" + section(claims, "- **C1", "- **C4") + section(claims, "- **C26", "- **C27") + section(claims, "- **C27", "- **C15") + section(claims, "- **C14.", "- **C15"),
    "# RESULTS DIGEST 21-22 SEP (summary table)\n" + section(digest, "| # | Run |", "\n---"),
    "# 24 SEP LEDGER: results section\n" + section(ledger, "## Results appended when jobs landed", "## Commits"),
    "# PROJECT P31 (pre-registration + status log)\n" + rd(T + "/docs/projects/P31_multifield_extraction.md"),
    "# PROJECT P32 (pre-registration + status log)\n" + rd(T + "/docs/projects/P32_image_to_fields.md"),
    "# PROJECT P33 (pre-registration + status log)\n" + rd(T + "/docs/projects/P33_patch_level_grading.md")])
pack = re.sub(r"\n{3,}", "\n\n", pack)[:90000]; open(os.path.join(OUT, "review_pack.md"), "w").write(pack); print("pack chars", len(pack), flush=True)
PROMPT = """You are a hostile but fair senior reviewer (computational pathology + clinical statistics) reading a PhD student's
internal research log. Read the whole pack, then write a structured review with exactly these headings:

## 1. What is actually established (max 5 bullets, each citing the specific number that supports it)
## 2. Strongest alternative explanations / confounds not yet excluded (be specific: which result, which mechanism, which check would kill it)
## 3. Statistical concerns (multiplicity, selection, sample size, circularity of LLM-derived labels, bootstrap/permutation design)
## 4. Experiments the student should run next, ranked, each with the expected outcome if the claim is true vs false
## 5. Missing related work or standard baselines a reviewer would demand
## 6. Ideas: anything in this data the student has not thought to use
## 7. Verdict in three sentences: what this work can honestly claim today

Be concrete, quote numbers from the pack, do not flatter, do not invent results that are not in the pack.

=== PACK ===
"""
out_md = [f"# Local-LLM review panel — {time.strftime('%Y-%m-%d %H:%M')}", "", f"Pack: `review_pack.md` ({len(pack):,} chars), aggregate docs only. Models run locally on the CRUK CI cluster; no data left the institution.", ""]
for M in MODELS:
    t0 = time.time(); body = json.dumps({"model": M, "prompt": PROMPT + pack, "stream": False, "think": False, "options": {"temperature": 0.3, "num_ctx": NUM_CTX, "num_predict": 3500}}).encode()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/api/generate", data=body, headers={"Content-Type": "application/json"}), timeout=3600).read())
        txt = r.get("response", "").strip(); meta = f"prompt tokens {r.get('prompt_eval_count')}, output tokens {r.get('eval_count')}, {time.time()-t0:.0f} s"
    except Exception as e: txt = f"(failed: {e})"; meta = f"{time.time()-t0:.0f} s"
    out_md += [f"---\n\n## Reviewer: `{M}`  ({meta})", "", txt, ""]; print(M, meta, flush=True)
    open(os.path.join(OUT, "llm_review.md"), "w").write("\n".join(out_md))
srv.terminate(); print("done")
