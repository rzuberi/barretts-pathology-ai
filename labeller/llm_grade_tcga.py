"""2.33 shard worker: jury grading of TCGA pathology reports, pan-cancer.

Extracts HISTOLOGIC GRADE from TCGA report text: one of G1|G2|G3|G4|HIGH|LOW|GX.
INPUT env = csv with columns CaseName,_text (prepared per study). Same ollama
mechanics as llm_grade_shard (unique port, resume, quoted CSV).
"""
import csv, json, os, subprocess, time, urllib.request
import pandas as pd

MODEL = os.environ["MODEL"]
INPUT = os.environ["INPUT"]
OUT = os.environ.get("OUTDIR", ".")
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
CONC = int(os.environ.get("CONC", 2))
REQ_TIMEOUT = int(os.environ.get("REQ_TIMEOUT", 600))
VALID = {"G1", "G2", "G3", "G4", "HIGH", "LOW", "GX"}

os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models")
PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000
os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
BASE = f"http://127.0.0.1:{PORT}"
srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(60):
    try:
        urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception:
        time.sleep(2)
subprocess.run([os.path.expanduser("~/.local/bin/ollama"), "pull", MODEL],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

PROMPT = """You are a pathologist. Read this cancer pathology report and output the
HISTOLOGIC GRADE of the tumour as JSON: {"grade": "<G1|G2|G3|G4|HIGH|LOW|GX>"}.
Use G1-G4 when a numeric grade is stated (well=G1, moderately=G2, poorly=G3,
undifferentiated=G4). Use HIGH/LOW only when the report uses a two-tier system.
Use GX if no grade is determinable. Output ONLY the JSON.

REPORT:
"""

rep = pd.read_csv(INPUT, dtype=str).fillna("")
rep = rep.iloc[SHARD::N_SHARDS]
tag = MODEL.replace(":", "_").replace(".", "_")
out = os.path.join(OUT, f"llm_grades_{tag}_shard{SHARD}.csv")
done = set()
if os.path.exists(out):
    done = set(pd.read_csv(out)["CaseName"])
todo = [(r["CaseName"], r["_text"]) for _, r in rep.iterrows() if r["CaseName"] not in done]
print(f"{MODEL} shard {SHARD}: {len(todo)} to do", flush=True)
if not os.path.exists(out):
    with open(out, "w", newline="") as f:
        csv.writer(f).writerow(["CaseName", "llm_grade"])

def grade_one(cid, text):
    body = json.dumps({"model": MODEL, "prompt": PROMPT + text[:8000] + "\n/no_think",
                       "stream": False, "format": "json", "think": False,
                       "options": {"temperature": 0.0, "num_predict": 200, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(BASE + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=REQ_TIMEOUT).read())["response"]
        g = str(json.loads(resp).get("grade", "")).upper().strip()
        return g if g in VALID else "PARSE_FAIL"
    except Exception:
        return "PARSE_FAIL"

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=CONC) as ex:
    for i in range(0, len(todo), CONC * 4):
        chunk = todo[i:i + CONC * 4]
        grades = list(ex.map(lambda t: grade_one(*t), chunk))
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            for (cid, _), g in zip(chunk, grades):
                w.writerow([cid, g])
        if i % 200 == 0: print(f"{i}/{len(todo)}", flush=True)
print("shard complete", flush=True)
srv.terminate()
