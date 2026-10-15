"""ERIN MDT (Rehan 2026-09-02): LLM panel deliberation over hard cases,
following the published MDT-agent pattern (MDTeamGPT / MDAT / GI-oncology
multi-agent MDT): independent reads -> evidence-grounded rebuttal round ->
chair synthesis. Cases: the 80 adjudicated (binary CANCER truth available)
+ the 204 unsure-holdout reports (where independent voting failed).
Outputs per case: all round-1/2 opinions, chair verdict + confidence +
dissent, and a full JSON transcript for the conformity-flip analysis.
"""
import csv, json, os, subprocess, time, urllib.request
import pandas as pd

INPUT = os.environ["INPUT"]
OUT = os.environ.get("OUTDIR", ".")
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
REQ_TIMEOUT = int(os.environ.get("REQ_TIMEOUT", 900))
CONSULTANTS = ["qwen3:14b", "gemma3:27b", "phi4:14b"]
CHAIR = "qwen3:32b"
GRADES = ["NDBE", "IND", "LGD", "HGD", "CANCER"]

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
for m in CONSULTANTS + [CHAIR]:
    subprocess.run([os.path.expanduser("~/.local/bin/ollama"), "pull", m],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

LADDER = """Grades (Barrett's ladder): NDBE (no dysplasia), IND (indefinite for
dysplasia), LGD (low grade dysplasia), HGD (high grade dysplasia), CANCER
(invasive adenocarcinoma). Grade the WORST finding in this oesophageal report."""

def gen(model, prompt, n_predict=400):
    body = json.dumps({"model": model, "prompt": prompt + "\n/no_think",
                       "stream": False, "format": "json", "think": False,
                       "options": {"temperature": 0.2, "num_predict": n_predict,
                                   "num_ctx": 16384}}).encode()
    req = urllib.request.Request(BASE + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(json.loads(
            urllib.request.urlopen(req, timeout=REQ_TIMEOUT).read())["response"])
    except Exception:
        return {}

def valid_grade(d):
    g = str(d.get("grade", "")).upper().strip()
    return g if g in GRADES else None

rep = pd.read_csv(INPUT, dtype=str).fillna("")
rep = rep.iloc[SHARD::N_SHARDS]
out_csv = os.path.join(OUT, f"mdt_shard{SHARD}.csv")
tdir = os.path.join(OUT, "transcripts"); os.makedirs(tdir, exist_ok=True)
done = set()
if os.path.exists(out_csv):
    done = set(pd.read_csv(out_csv)["CaseName"])
if not os.path.exists(out_csv):
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["CaseName", "group"] +
            [f"r1_{m.split(':')[0]}" for m in CONSULTANTS] +
            [f"r2_{m.split(':')[0]}" for m in CONSULTANTS] +
            ["chair_grade", "chair_conf", "dissent"])

todo = [r for _, r in rep.iterrows() if r["CaseName"] not in done]
print(f"shard {SHARD}: {len(todo)} cases", flush=True)
for n, r in enumerate(todo):
    text = r["_text"][:8000]
    tr = {"case": r["CaseName"], "r1": {}, "r2": {}, "chair": {}}
    # Round 1: independent opinion + quoted evidence
    for m in CONSULTANTS:
        d = gen(m, f"""You are a consultant GI pathologist in an MDT. {LADDER}
Give YOUR independent read. Output ONLY JSON:
{{"grade": "<{'|'.join(GRADES)}>", "evidence": "<short quote from the report supporting it>"}}

REPORT:
{text}""")
        tr["r1"][m] = d
    r1_summary = "\n".join(
        f"- Consultant {i+1} ({m.split(':')[0]}): {valid_grade(tr['r1'][m]) or '?'} — "
        f"evidence: {str(tr['r1'][m].get('evidence', ''))[:200]}"
        for i, m in enumerate(CONSULTANTS))
    # Round 2: see colleagues, concur or rebut with evidence
    for m in CONSULTANTS:
        d = gen(m, f"""You are a consultant GI pathologist in an MDT. {LADDER}
Your initial read was: {valid_grade(tr['r1'][m]) or '?'}.
Your colleagues' reads:
{r1_summary}
Re-examine the report. If a colleague's evidence convinces you, change; otherwise
defend yours with a quote. Output ONLY JSON:
{{"grade": "<{'|'.join(GRADES)}>", "argument": "<one sentence citing report text>"}}

REPORT:
{text}""")
        tr["r2"][m] = d
    r2_summary = "\n".join(
        f"- Consultant {i+1}: {valid_grade(tr['r2'][m]) or '?'} — "
        f"{str(tr['r2'][m].get('argument', ''))[:200]}"
        for i, m in enumerate(CONSULTANTS))
    # Round 3: chair synthesis
    ch = gen(CHAIR, f"""You chair a GI pathology MDT. {LADDER}
Round-1 independent reads:
{r1_summary}
Round-2 positions after discussion:
{r2_summary}
Read the report yourself and issue the MDT conclusion. Output ONLY JSON:
{{"grade": "<{'|'.join(GRADES)}>", "confidence": <0.0-1.0>,
  "dissent": "<'none' or one sentence on unresolved disagreement>"}}

REPORT:
{text}""", n_predict=300)
    tr["chair"] = ch
    json.dump(tr, open(os.path.join(tdir, f"{r['CaseName']}.json"), "w"), indent=1)
    with open(out_csv, "a", newline="") as f:
        csv.writer(f).writerow(
            [r["CaseName"], r["group"]] +
            [valid_grade(tr["r1"][m]) or "PARSE_FAIL" for m in CONSULTANTS] +
            [valid_grade(tr["r2"][m]) or "PARSE_FAIL" for m in CONSULTANTS] +
            [valid_grade(ch) or "PARSE_FAIL", ch.get("confidence", ""),
             str(ch.get("dissent", ""))[:200]])
    if n % 10 == 0: print(f"{n}/{len(todo)}", flush=True)
print("shard complete", flush=True)
srv.terminate()
