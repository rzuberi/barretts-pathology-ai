"""Jury v3 per-section prompt applied to the 327 Barrett's-DB reports that carry SWG
sWGS samples (swg_matched_reports_v2.parquet). Same prompt/mechanics as
campaigns/allslides/llm_grade_sections_v3.py; only the input differs.
Env: MODEL (required), OUTDIR, CONC, REQ_TIMEOUT, JSON_MODE."""
import csv, json, os, re, subprocess, sys, time, urllib.request
import pandas as pd
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/campaigns/allslides")
src = open("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/campaigns/allslides/llm_grade_sections_v3.py").read()
PROMPT = re.search(r'PROMPT = """(.*?)"""', src, re.S).group(1)          # reuse the exact v3 prompt text

MODEL = os.environ["MODEL"]; OUT = os.environ.get("OUTDIR", ".")
CONC = int(os.environ.get("CONC", 2)); REQ_TIMEOUT = int(os.environ.get("REQ_TIMEOUT", 900)); JSON_MODE = os.environ.get("JSON_MODE", "format")
GRADES = {"NDBE", "IND", "LGD", "HGD", "CANCER", "NORMAL_OTHER"}
SUBTYPES = {"ADENOCARCINOMA", "SQUAMOUS", "SIGNET_RING", "POST_NEOADJUVANT_TX_EFFECT", "OTHER_CANCER"}
SITES = {"OESOPHAGUS", "GOJ", "STOMACH", "DUODENUM", "COLORECTAL", "LYMPH_NODE", "OTHER", "UNSTATED"}
SPECIMENS = {"BIOPSY", "EMR_ESD", "RESECTION", "LYMPH_NODE", "OTHER", "UNSTATED"}
os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models")
PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000
os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"; BASE = f"http://127.0.0.1:{PORT}"
slog = open(os.path.join(OUT, f"ollama_server_{os.environ.get('SLURM_JOB_ID','x')}.log"), "w")
srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=slog, stderr=slog)
for _ in range(60):
    try: urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
subprocess.run([os.path.expanduser("~/.local/bin/ollama"), "pull", MODEL], capture_output=True)

rep = pd.read_parquet("/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export/swg_matched_reports_v2.parquet")
rep["CaseName"] = rep.pathology_text_id.astype(float).astype(int).astype(str)
rep["_text"] = rep.reporttext.astype(str).str.replace("\r", "\n")
tag = MODEL.replace(":", "_").replace(".", "_")
out = os.path.join(OUT, f"sections_v3_{tag}_shard0.csv")
done = set(pd.read_csv(out)["CaseName"].astype(str)) if os.path.exists(out) else set()
todo = [(r.CaseName, r._text) for r in rep.itertuples() if r.CaseName not in done]
print(f"{MODEL}: {len(todo)} DB reports to do", flush=True)
if not os.path.exists(out):
    with open(out, "w", newline="") as f: csv.writer(f).writerow(["CaseName", "sections_json", "raw_len", "secs"])

def one(cid, text):
    body = {"model": MODEL, "prompt": PROMPT + text[:10000] + "\n/no_think", "stream": False, "think": False,
            "options": {"temperature": 0.0, "num_predict": 900, "num_ctx": 12288}}
    if JSON_MODE == "format": body["format"] = "json"
    req = urllib.request.Request(BASE + "/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try: resp = json.loads(urllib.request.urlopen(req, timeout=REQ_TIMEOUT).read())["response"]
    except Exception as e: return f"PARSE_FAIL:{type(e).__name__}", 0, round(time.time() - t0, 1)
    try: d = json.loads(resp)
    except Exception:
        s, e = resp.find("{"), resp.rfind("}")
        try: d = json.loads(resp[s:e + 1]) if s >= 0 and e > s else None
        except Exception: d = None
    if not isinstance(d, dict): return "PARSE_FAIL:json", len(resp), round(time.time() - t0, 1)
    clean = []
    for sec in (d.get("sections") or []):
        if not isinstance(sec, dict): continue
        gs = [g for g in map(str.upper, map(str, sec.get("grades") or [])) if g in GRADES]
        if not gs: continue
        st = [t for t in map(str.upper, map(str, sec.get("cancer_subtypes") or [])) if t in SUBTYPES]
        site = str(sec.get("site", "UNSTATED")).upper().replace(" ", "_"); site = site if site in SITES else "OTHER"
        sp = str(sec.get("specimen_type", "UNSTATED")).upper().replace(" ", "_"); sp = sp if sp in SPECIMENS else "OTHER"
        clean.append({"section": str(sec.get("section", "?")).upper()[:12], "site": site, "specimen_type": sp, "grades": gs, "cancer_subtypes": st})
    return (json.dumps(clean) if clean else "PARSE_FAIL:empty"), len(resp), round(time.time() - t0, 1)

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=CONC) as ex:
    for i in range(0, len(todo), CONC * 4):
        chunk = todo[i:i + CONC * 4]; res = list(ex.map(lambda t: one(*t), chunk))
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            for (cid, _), (sj, rl, secs) in zip(chunk, res): w.writerow([cid, sj, rl, secs])
        print(f"{i + len(chunk)}/{len(todo)}", flush=True)
print("complete", flush=True); srv.terminate()
