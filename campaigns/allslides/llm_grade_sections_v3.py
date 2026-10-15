"""Per-section jury v3 (ERIN all-slides campaign, 2026-09-16).

v2 -> v3: every lettered section now also gets `site` (organ) and
`specimen_type`, so the organ filter is a downstream choice instead of a prompt
assumption. Same ollama mechanics as v2 (local server per job, unique port,
resume-by-CaseName, sharded). SMOKE=adjudicated restricts to the 80-case
adjudication set for the parse-rate gate.

Env: MODEL (required), SHARD, N_SHARDS, CONC, REQ_TIMEOUT, SMOKE, OUTDIR,
     JSON_MODE=format|none (gpt-oss failed under format:json in v2 -> try none)
"""
import csv, json, os, subprocess, time, urllib.request
import pandas as pd

MODEL = os.environ["MODEL"]
OUT = os.environ.get("OUTDIR", ".")
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
CONC = int(os.environ.get("CONC", 2))
REQ_TIMEOUT = int(os.environ.get("REQ_TIMEOUT", 900))
SMOKE = os.environ.get("SMOKE", "")
JSON_MODE = os.environ.get("JSON_MODE", "format")
GRADES = {"NDBE", "IND", "LGD", "HGD", "CANCER", "NORMAL_OTHER"}
SUBTYPES = {"ADENOCARCINOMA", "SQUAMOUS", "SIGNET_RING", "POST_NEOADJUVANT_TX_EFFECT", "OTHER_CANCER"}
SITES = {"OESOPHAGUS", "GOJ", "STOMACH", "DUODENUM", "COLORECTAL", "LYMPH_NODE", "OTHER", "UNSTATED"}
SPECIMENS = {"BIOPSY", "EMR_ESD", "RESECTION", "LYMPH_NODE", "OTHER", "UNSTATED"}
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
IDX = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/runs/adjpack/output/adjudication_index.csv"

os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models")
PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000
os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
BASE = f"http://127.0.0.1:{PORT}"
slog = open(os.path.join(OUT, f"ollama_server_{os.environ.get('SLURM_JOB_ID','x')}.log"), "w")
srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=slog, stderr=slog)
for _ in range(60):
    try: urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
pull = subprocess.run([os.path.expanduser("~/.local/bin/ollama"), "pull", MODEL], capture_output=True, text=True)
if pull.returncode != 0:
    have = subprocess.run([os.path.expanduser("~/.local/bin/ollama"), "list"], capture_output=True, text=True).stdout
    if MODEL.split(":")[0] not in have: raise SystemExit(f"pull failed and {MODEL} not cached")
    print("pull failed (offline node?) but cached — continuing", flush=True)

PROMPT = """You are a GI pathologist reading a UK upper-GI pathology report. The report may
contain several lettered specimen sections (A, B, C, ...). The GROSS DESCRIPTION
lists what each lettered specimen is and where it was taken from; the FINAL
DIAGNOSIS and MICROSCOPIC DESCRIPTION give the findings. If the report has no
lettered sections, use one section called "WHOLE".

For EACH section report:
- "site": where the tissue came from. One of:
  OESOPHAGUS | GOJ (gastro-oesophageal junction / cardia / squamocolumnar junction)
  | STOMACH (body, antrum, fundus, pylorus) | DUODENUM | COLORECTAL | LYMPH_NODE
  | OTHER | UNSTATED
- "specimen_type": BIOPSY | EMR_ESD (endoscopic mucosal/submucosal resection)
  | RESECTION (oesophagectomy/gastrectomy specimen) | LYMPH_NODE | OTHER | UNSTATED
- "grades": EVERY grade present in that section (a section can hold several):
  NDBE (Barrett's / intestinal metaplasia, no dysplasia) | IND (indefinite for
  dysplasia) | LGD | HGD | CANCER (invasive or intramucosal carcinoma, current) |
  NORMAL_OTHER (no Barrett's / dysplasia / cancer: normal, gastritis, oesophagitis,
  reactive change, polyp, lymph node without tumour, etc.)
  Historical mentions and negated findings do NOT count.
- "cancer_subtypes": only if CANCER is present: ADENOCARCINOMA | SQUAMOUS |
  SIGNET_RING | POST_NEOADJUVANT_TX_EFFECT | OTHER_CANCER

Output ONLY JSON:
{"sections": [{"section": "A", "site": "...", "specimen_type": "...",
               "grades": ["..."], "cancer_subtypes": ["..."]}, ...]}

REPORT:
"""

rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("")
rep["_text"] = ("GROSS DESCRIPTION:\n" + rep["GrossDescription_redacted"] +
                "\n\nFINAL DIAGNOSIS:\n" + rep["FinalDiagnosis_redacted"] +
                "\n\nMICROSCOPIC DESCRIPTION:\n" + rep["MicroscopicDescription_redacted"])
rep = rep[(rep["FinalDiagnosis_redacted"] + rep["MicroscopicDescription_redacted"]).str.len() > 40].drop_duplicates("CaseName")
if SMOKE == "adjudicated":
    rep = rep[rep["CaseName"].isin(pd.read_csv(IDX)["CaseName"])]
rep = rep.iloc[SHARD::N_SHARDS]
tag = MODEL.replace(":", "_").replace(".", "_").replace("/", "_")
out = os.path.join(OUT, f"sections_v3_{tag}_shard{SHARD}.csv")
done = set(pd.read_csv(out)["CaseName"]) if os.path.exists(out) else set()
todo = [(r["CaseName"], r["_text"]) for _, r in rep.iterrows() if r["CaseName"] not in done]
print(f"{MODEL} shard {SHARD}/{N_SHARDS} smoke={SMOKE!r} json={JSON_MODE}: {len(todo)} to do", flush=True)
if not os.path.exists(out):
    with open(out, "w", newline="") as f: csv.writer(f).writerow(["CaseName", "sections_json", "raw_len", "secs"])

def one(cid, text):
    body = {"model": MODEL, "prompt": PROMPT + text[:10000] + "\n/no_think", "stream": False, "think": False,
            "options": {"temperature": 0.0, "num_predict": 900, "num_ctx": 12288}}
    if JSON_MODE == "format": body["format"] = "json"
    req = urllib.request.Request(BASE + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=REQ_TIMEOUT).read())["response"]
    except Exception as e:
        return f"PARSE_FAIL:{type(e).__name__}", 0, round(time.time() - t0, 1)
    raw = resp
    try:
        d = json.loads(resp)
    except Exception:
        s, e = resp.find("{"), resp.rfind("}")           # tolerate prose around the JSON
        try: d = json.loads(resp[s:e + 1]) if s >= 0 and e > s else None
        except Exception: d = None
    if not isinstance(d, dict): return "PARSE_FAIL:json", len(raw), round(time.time() - t0, 1)
    clean = []
    for sec in (d.get("sections") or []):
        if not isinstance(sec, dict): continue
        gs = [g for g in map(str.upper, map(str, sec.get("grades") or [])) if g in GRADES]
        if not gs: continue
        st = [t for t in map(str.upper, map(str, sec.get("cancer_subtypes") or [])) if t in SUBTYPES]
        site = str(sec.get("site", "UNSTATED")).upper().replace(" ", "_"); site = site if site in SITES else "OTHER"
        sp = str(sec.get("specimen_type", "UNSTATED")).upper().replace(" ", "_"); sp = sp if sp in SPECIMENS else "OTHER"
        clean.append({"section": str(sec.get("section", "?")).upper()[:12], "site": site, "specimen_type": sp,
                      "grades": gs, "cancer_subtypes": st})
    return (json.dumps(clean) if clean else "PARSE_FAIL:empty"), len(raw), round(time.time() - t0, 1)

from concurrent.futures import ThreadPoolExecutor
n = 0; t_start = time.time()
with ThreadPoolExecutor(max_workers=CONC) as ex:
    for i in range(0, len(todo), CONC * 4):
        chunk = todo[i:i + CONC * 4]
        res = list(ex.map(lambda t: one(*t), chunk))
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            for (cid, _), (sj, rl, secs) in zip(chunk, res): w.writerow([cid, sj, rl, secs])
        n += len(chunk)
        if i == 0: print("FIRST:", res[0][0][:300], flush=True)
        if (i // (CONC * 4)) % 10 == 0: print(f"{n}/{len(todo)} {(time.time()-t_start)/max(n,1):.1f}s/report", flush=True)
print("shard complete", flush=True)
srv.terminate()
