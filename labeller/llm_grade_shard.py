"""EXECUTION_PLAN 2.7 (extended 2026-08-20): local-LLM grading of ERIN report text.

Runs entirely on-cluster (report text never leaves). Starts an ollama server on the
job's GPU, grades a shard of reports onto the oesophageal ladder with a strict JSON
prompt (site-scoped: oesophagus/GOJ findings only — the site-mismatch lesson from
the adjudication audit), writes a per-report CSV shard.

Env: SHARD (0-based), N_SHARDS, MODEL (default qwen3:14b), SMOKE=adjudicated|all
"""
import json, os, subprocess, sys, time, urllib.request
import pandas as pd

ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
IDX = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/runs/adjpack/output/adjudication_index.csv"
OUT = os.environ.get("OUTDIR", ".")
MODEL = os.environ.get("MODEL", "qwen3:14b")
SHARD = int(os.environ.get("SHARD", "0")); N_SHARDS = int(os.environ.get("N_SHARDS", "1"))
SMOKE = os.environ.get("SMOKE", "")
TEXT = ["FinalDiagnosis_redacted", "MicroscopicDescription_redacted",
        "Addendum1_redacted", "Addendum2_redacted", "Addendum3_redacted"]

os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models")
CONC = int(os.environ.get("CONC", "6"))
os.environ.setdefault("OLLAMA_NUM_PARALLEL", str(CONC))
# unique port per job: multiple jobs can share a node; colliding on 11434 cross-wires servers
PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000
os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
BASE = f"http://127.0.0.1:{PORT}"
_slog = open(os.path.join(os.environ.get("OUTDIR", "."), f"ollama_server_{os.environ.get('SLURM_JOB_ID','x')}.log"), "w")
srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"],
                       stdout=_slog, stderr=_slog)
for _ in range(60):
    try:
        urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
pull = subprocess.run([os.path.expanduser("~/.local/bin/ollama"), "pull", MODEL])
if pull.returncode != 0:  # offline node: proceed if weights already cached
    have = subprocess.run([os.path.expanduser("~/.local/bin/ollama"), "list"],
                          capture_output=True, text=True).stdout
    if MODEL.split(":")[0] not in have:
        raise SystemExit(f"pull failed and {MODEL} not in local cache")
    print(f"pull failed (offline node?) but {MODEL} cached — continuing", flush=True)

SPECIMEN_PROMPT = """You grade UK oesophageal surveillance pathology reports.
The text begins with a SPECIMEN OF INTEREST header naming one laboratory sample
(its suffix denotes the specimen/block, e.g. B1 ~ specimen B, block 1). Reports
list specimens as lettered sections (A., B., C., ...). Grade ONLY the findings of
the section corresponding to the specimen of interest. If the report does not
distinguish specimens, grade the report as a whole. All other rules below apply.

Consider ONLY findings in the oesophagus or gastro-oesophageal junction (GOJ/cardia).
Ignore other-site findings, historical mentions, and negated findings.

Grade on this ladder, choosing the WORST current finding FOR THAT SPECIMEN:
NDBE, IND, LGD, HGD, CANCER, or NA (not gradeable / specimen not oesophageal).

Reply with ONLY a JSON object: {"grade":"NDBE|IND|LGD|HGD|CANCER|NA"}

REPORT:
"""

PROMPT = """You grade UK oesophageal surveillance pathology reports.
Consider ONLY findings in the oesophagus or gastro-oesophageal junction (GOJ/cardia).
Ignore stomach-body, duodenal, colorectal or other-site findings entirely.
Ignore historical mentions ("known/previous/history of carcinoma") — grade only what
is diagnosed as CURRENTLY PRESENT in this specimen. Negated findings ("no dysplasia",
"no evidence of malignancy") do not count.

Grade on this ladder, choosing the WORST current oesophageal/GOJ finding:
NDBE (Barrett's/intestinal metaplasia without dysplasia, or benign/normal),
IND (indefinite for dysplasia), LGD (low-grade dysplasia),
HGD (high-grade dysplasia), CANCER (carcinoma of any type, incl. intramucosal),
NA (report not gradeable on this ladder).

Reply with ONLY a JSON object: {"grade":"NDBE|IND|LGD|HGD|CANCER|NA"}

REPORT:
"""

ACTIVE_PROMPT = SPECIMEN_PROMPT if os.environ.get("PROMPT_MODE") == "specimen" else PROMPT

def grade(text):
    body = json.dumps({"model": MODEL, "prompt": ACTIVE_PROMPT + text[:6000] + "\n/no_think",
                       "stream": False, "format": "json", "think": False,
                       "options": {"temperature": 0, "num_predict": 200}}).encode()
    r = urllib.request.Request(BASE + "/api/generate", data=body,
                               headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(r, timeout=int(os.environ.get("REQ_TIMEOUT", "600"))).read())["response"]
    grades = ("NDBE", "IND", "LGD", "HGD", "CANCER", "NA")
    try:
        g = json.loads(resp).get("grade", "")
        if g in grades: return g, resp
    except Exception:
        pass
    import re as _re
    hits = [g for g in grades if _re.search(rf'\b{g}\b', resp)]
    return (hits[0] if len(hits) == 1 else "PARSE_FAIL"), resp

INPUT = os.environ.get("INPUT", "erin")
if INPUT == "erin":
    rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("")
    rep["_text"] = rep[TEXT].agg(" ".join, axis=1)
else:  # a CSV/parquet with reporttext (e.g. barretts db export)
    rep = (pd.read_parquet(INPUT) if INPUT.endswith(".parquet")
           else pd.read_csv(INPUT, dtype=str, low_memory=False)).astype(str).fillna("")
    if "CaseName" not in rep.columns:
        rep["CaseName"] = rep["pathology_text_id"]
    rep["_text"] = rep["reporttext"]
if SMOKE == "adjudicated":
    idx = pd.read_csv(IDX)
    rep = rep[rep["CaseName"].isin(idx["CaseName"])]
rep = rep.reset_index(drop=True)
rep = rep[rep.index % N_SHARDS == SHARD]
if os.environ.get("LIMIT"): rep = rep.head(int(os.environ["LIMIT"]))
print(f"model={MODEL} shard={SHARD}/{N_SHARDS} reports={len(rep)}", flush=True)

out = os.path.join(OUT, f"llm_grades_{MODEL.replace(':','_').replace('/','_')}_shard{SHARD}.csv")
done = set()
if os.path.exists(out):  # resume after preemption
    done = set(pd.read_csv(out)["CaseName"])
    print(f"resuming: {len(done)} already graded", flush=True)
todo = [(r["CaseName"], r["_text"]) for _, r in rep.iterrows() if r["CaseName"] not in done]

from concurrent.futures import ThreadPoolExecutor
import threading
lock = threading.Lock()
t0, count = time.time(), 0
if not os.path.exists(out):
    open(out, "w").write("CaseName,llm_grade\n")
def work(item):
    global count
    name, text = item
    g, raw = grade(text)
    with lock:
        import csv as _csv
        with open(out, "a", newline="") as fh:
            _csv.writer(fh).writerow([name, g])
        # Item 28 (24 Sep 2026): keep the model's raw JSON string, not only the parsed grade, one line per report
        with open(out.replace(".csv", "_raw_responses.jsonl"), "a") as rf:
            rf.write(json.dumps({"CaseName": name, "model": MODEL, "raw": raw, "parsed_grade": g, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        count += 1
        if count <= 3: print(f"RAW: {raw[:200]!r}", flush=True)
        if count % 50 == 0: print(f"{count}/{len(todo)} elapsed={time.time()-t0:.0f}s", flush=True)
with ThreadPoolExecutor(max_workers=CONC) as ex:
    list(ex.map(work, todo))
print("wrote", out, f"({(time.time()-t0)/max(len(todo),1):.2f}s/report at CONC={CONC})")
srv.terminate()
