"""2.37: per-SECTION jury grading of ERIN reports (label-space v2, Rehan
2026-09-02). For each lettered specimen section: every grade present
(multi-label) + cancer subtype(s). Joined to slides via Shiv Sakthivel's
matched_image_pathology.csv this upgrades ERIN supervision from case-max to
slide-level. Same ollama mechanics as the corpus jury; resume-capable.
"""
import csv, json, os, subprocess, time, urllib.request
import pandas as pd

MODEL = os.environ["MODEL"]
OUT = os.environ.get("OUTDIR", ".")
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
CONC = int(os.environ.get("CONC", 2))
REQ_TIMEOUT = int(os.environ.get("REQ_TIMEOUT", 900))
GRADES = {"NDBE", "IND", "LGD", "HGD", "CANCER", "NORMAL_OTHER"}
SUBTYPES = {"ADENOCARCINOMA", "SQUAMOUS", "SIGNET_RING", "POST_NEOADJUVANT_TX_EFFECT", "OTHER_CANCER"}

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

PROMPT = """You are a GI pathologist. This oesophageal pathology report may contain
multiple lettered specimen sections (A, B, C, ...). For EACH section, list EVERY
grade present (a section can hold several, e.g. NDBE and LGD). If the report has
no lettered sections, use one section called "WHOLE".
Grades: NDBE (Barrett's, no dysplasia), IND (indefinite for dysplasia),
LGD, HGD, CANCER (invasive malignancy), NORMAL_OTHER (no Barrett's/dysplasia/
cancer finding, e.g. normal, gastritis, oesophagitis only).
If CANCER is present in a section, also give its subtype(s):
ADENOCARCINOMA | SQUAMOUS | SIGNET_RING | POST_NEOADJUVANT_TX_EFFECT | OTHER_CANCER.
Output ONLY JSON:
{"sections": [{"section": "A", "grades": ["..."], "cancer_subtypes": ["..."]}, ...]}

REPORT:
"""

rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv",
                  dtype=str, low_memory=False).fillna("")
rep["_text"] = ("FINAL DIAGNOSIS:\n" + rep["FinalDiagnosis_redacted"] +
                "\n\nMICROSCOPIC DESCRIPTION:\n" + rep["MicroscopicDescription_redacted"])
rep = rep[rep["_text"].str.len() > 40].drop_duplicates("CaseName")
rep = rep.iloc[SHARD::N_SHARDS]
tag = MODEL.replace(":", "_").replace(".", "_")
out = os.path.join(OUT, f"sections_{tag}_shard{SHARD}.csv")
done = set()
if os.path.exists(out):
    done = set(pd.read_csv(out)["CaseName"])
todo = [(r["CaseName"], r["_text"]) for _, r in rep.iterrows() if r["CaseName"] not in done]
print(f"{MODEL} shard {SHARD}: {len(todo)} to do", flush=True)
if not os.path.exists(out):
    with open(out, "w", newline="") as f:
        csv.writer(f).writerow(["CaseName", "sections_json"])

def one(cid, text):
    body = json.dumps({"model": MODEL, "prompt": PROMPT + text[:8000] + "\n/no_think",
                       "stream": False, "format": "json", "think": False,
                       "options": {"temperature": 0.0, "num_predict": 600, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(BASE + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.loads(json.loads(urllib.request.urlopen(req, timeout=REQ_TIMEOUT).read())["response"])
        secs = d.get("sections")
        clean = []
        for sec in (secs or []):
            gs = [g for g in map(str.upper, map(str, sec.get("grades", []))) if g in GRADES]
            st = [t for t in map(str.upper, map(str, sec.get("cancer_subtypes", []))) if t in SUBTYPES]
            if gs:
                clean.append({"section": str(sec.get("section", "?")).upper()[:12],
                              "grades": gs, "cancer_subtypes": st})
        return json.dumps(clean) if clean else "PARSE_FAIL"
    except Exception:
        return "PARSE_FAIL"

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=CONC) as ex:
    for i in range(0, len(todo), CONC * 4):
        chunk = todo[i:i + CONC * 4]
        res = list(ex.map(lambda t: one(*t), chunk))
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            for (cid, _), sj in zip(chunk, res):
                w.writerow([cid, sj])
        if i % 200 == 0: print(f"{i}/{len(todo)}", flush=True)
print("shard complete", flush=True)
srv.terminate()
