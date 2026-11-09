"""Structured extraction from BEST2 NHS pathology reports with a local LLM (report text never leaves the cluster).
Env: MODEL, N (sample size, default 100), LIMIT (smoke), OUTDIR. Random sample seed 0 of reports with text."""
import json, os, re, subprocess, time, urllib.request, concurrent.futures as cf
import pandas as pd
B = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export/best2_live_20261104/best2_pathology_reports_graded.csv"
OUT = os.environ.get("OUTDIR", "."); MODEL = os.environ.get("MODEL", "medgemma:27b"); N = int(os.environ.get("N", "100"))
os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models"); os.environ.setdefault("OLLAMA_NUM_PARALLEL", "6")
PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000; os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"; BASE = f"http://127.0.0.1:{PORT}"
srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=open(f"{OUT}/ollama.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try: urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
PROMPT = """You read one UK NHS pathology report from a patient in an oesophageal (Barrett's) study.
Extract facts stated in the report. Do not guess; use "not_stated" when the report does not say.

Return ONLY a JSON object with these keys:
"sample_type": one of "endoscopic_biopsy", "cytosponge", "brushing_cytology", "EMR", "ESD", "resection", "mixed", "other"
"n_specimens": integer number of separately labelled specimens/pots/parts in the report (e.g. A, B, C or 1, 2, 3); 1 if only one
"specimens": list, one per specimen, each {"label": str, "site": "oesophagus"|"GOJ_cardia"|"stomach"|"duodenum"|"other"|"not_stated",
              "grade": "NDBE"|"IND"|"LGD"|"HGD"|"CANCER"|"no_barretts"|"not_stated",
              "intestinal_metaplasia": "yes"|"no"|"not_stated"}
"report_intestinal_metaplasia": "yes"|"no"|"not_stated"   (IM anywhere in oesophagus/GOJ)
"multiple_slides_or_levels": "yes"|"no"|"not_stated"      (report says several slides, levels or blocks were examined)
"mentions_cytosponge": true|false
"mentions_tff3": true|false
"mentions_p53": true|false
"barretts_length_stated": true|false

Grade meanings: NDBE = Barrett's/columnar mucosa without dysplasia; no_barretts = squamous/gastric/other mucosa with no Barrett's.

REPORT:
"""
def ask(text):
    body = json.dumps({"model": MODEL, "prompt": PROMPT + text[:8000] + "\n/no_think", "stream": False, "format": "json",
                       "options": {"temperature": 0, "num_ctx": 8192}}).encode()
    for a in range(3):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/api/generate", body, {"Content-Type": "application/json"}), timeout=600).read())["response"]
            r = re.sub(r"<think>.*?</think>", "", r, flags=re.S).strip()
            return json.loads(r[r.find("{"): r.rfind("}") + 1]), r
        except Exception as e: err = repr(e)[:200]; time.sleep(3)
    return None, err
b = pd.read_csv(B, dtype=str).fillna(""); b = b[b.reporttext.str.strip() != ""].sample(N, random_state=0)
if os.environ.get("LIMIT"): b = b.head(int(os.environ["LIMIT"]))
print(f"model={MODEL} reports={len(b)}", flush=True)
res = []
with cf.ThreadPoolExecutor(6) as ex:
    for (pid, _), (js, raw) in zip(b[["pathology_text_id", "reporttext"]].values, ex.map(ask, b.reporttext)):
        res.append({"pathology_text_id": pid, "ok": js is not None, "json": json.dumps(js) if js else "", "raw": raw})
        if len(res) % 10 == 0: print(len(res), flush=True)
pd.DataFrame(res).to_csv(f"{OUT}/extract_{MODEL.replace(':','_')}.csv", index=False)
ok = sum(r["ok"] for r in res); print("parsed ok:", ok, "of", len(res)); srv.terminate()
if ok == 0: raise SystemExit("no reply parsed")
