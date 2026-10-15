"""SQ.3 pilot (Rehan 2026-09-02): open-vocabulary phenotype discovery.
One strong juror enumerates ALL distinct diagnoses/findings per report over
1,000 random ERIN reports -> vocabulary size/shape decides the full-corpus run.
"""
import csv, json, os, subprocess, time, urllib.request
import pandas as pd

MODEL = os.environ.get("MODEL", "gemma3:27b")
OUT = os.environ.get("OUTDIR", ".")
N = int(os.environ.get("N_REPORTS", 1000))
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

PROMPT = """You are a GI pathologist building a findings index. List EVERY distinct
pathological diagnosis or finding in this report as short canonical terms
(lowercase, singular, no negated findings, no normal findings). Examples of the
style: "barrett's oesophagus", "low grade dysplasia", "eosinophilic oesophagitis",
"candida oesophagitis", "h pylori gastritis", "signet ring cell carcinoma",
"post-treatment effect", "fundic gland polyp", "granulation tissue".
Output ONLY JSON: {"findings": ["...", "..."]}

REPORT:
"""

rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv",
                  dtype=str, low_memory=False).fillna("")
rep["_text"] = ("FINAL DIAGNOSIS:\n" + rep["FinalDiagnosis_redacted"] +
                "\n\nMICROSCOPIC DESCRIPTION:\n" + rep["MicroscopicDescription_redacted"])
rep = rep[rep["_text"].str.len() > 60].drop_duplicates("CaseName").sample(N, random_state=0)
out = os.path.join(OUT, "phenotypes_pilot.csv")
done = set()
if os.path.exists(out):
    done = set(pd.read_csv(out)["CaseName"])
if not os.path.exists(out):
    with open(out, "w", newline="") as f:
        csv.writer(f).writerow(["CaseName", "findings_json"])
todo = [(r["CaseName"], r["_text"]) for _, r in rep.iterrows() if r["CaseName"] not in done]
print(f"{len(todo)} to do", flush=True)
for i, (cid, text) in enumerate(todo):
    body = json.dumps({"model": MODEL, "prompt": PROMPT + text[:8000] + "\n/no_think",
                       "stream": False, "format": "json", "think": False,
                       "options": {"temperature": 0.0, "num_predict": 400, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(BASE + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.loads(json.loads(urllib.request.urlopen(req, timeout=900).read())["response"])
        fs = [str(x).lower().strip()[:80] for x in d.get("findings", [])][:30]
        row = json.dumps(fs)
    except Exception:
        row = "PARSE_FAIL"
    with open(out, "a", newline="") as f:
        csv.writer(f).writerow([cid, row])
    if i % 50 == 0: print(f"{i}/{len(todo)}", flush=True)
print("pilot complete", flush=True)
srv.terminate()
