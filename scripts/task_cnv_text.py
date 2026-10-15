"""C2: CNV-as-text zero-shot progression risk (docs/llm_extensions_preregistration.md).
Serialise each SWG sample's arm-level relative copy number as gains/losses, ask a local LLM for a 0-100
risk, score against the release endpoint. Env: MODEL (ollama tag), OUTDIR."""
import json, os, re, subprocess, time, urllib.request
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
OUT = os.environ.get("OUTDIR", "."); MODEL = os.environ.get("MODEL", "medgemma:27b"); CONC = int(os.environ.get("CONC", "6")); NO_GRADE = bool(os.environ.get("NO_GRADE")); REP_SEED = int(os.environ.get("REP_SEED", "0")); TEMP = float(os.environ.get("TEMP", "0"))  # repeat runs: seed + temperature > 0  # CNV-only prompt: fair comparison with the CNV-only model
os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models"); os.environ.setdefault("OLLAMA_NUM_PARALLEL", str(CONC))
PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000; os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"; BASE = f"http://127.0.0.1:{PORT}"
slog = open(os.path.join(OUT, "ollama_server.log"), "w"); srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=slog, stderr=slog)
for _ in range(60):
    try: urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); arms = pd.read_csv(F + "/feature_views/cnv/features_arms.csv", dtype=str)
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str); cxv = dict(zip(cx.sample_id, pd.to_numeric(cx.cx, errors="coerce")))
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID")
acols = [c for c in arms.columns if c.startswith("chr")]; A = arms.set_index("sample_id")[acols].apply(pd.to_numeric, errors="coerce")
mu, sd = A.stack().mean(), A.stack().std()
GR = {"0": "non-dysplastic Barrett's", "1": "indefinite for dysplasia", "2": "low-grade dysplasia", "3": "high-grade dysplasia", "4": "intramucosal carcinoma"}
def describe(sid):
    v = A.loc[sid]; z = (v - mu) / sd; gains = [f"{c[3:]} (+{v[c]:.2f})" for c in acols if z[c] > 1]; losses = [f"{c[3:]} ({v[c]:.2f})" for c in acols if z[c] < -1]
    g = GR.get(str(coh.loc[sid, "Label"]).strip(), "unknown") if sid in coh.index else "unknown"
    return (("" if NO_GRADE else f"Current biopsy histology: {g}. ") + f"Shallow whole-genome sequencing relative copy number by chromosome arm "
            f"(values are log-ratio-like, 0 = neutral). Arm gains: {', '.join(gains) or 'none'}. Arm losses: {', '.join(losses) or 'none'}. "
            f"Genome-wide copy-number complexity score: {cxv.get(sid, float('nan')):.3f} (cohort median {np.nanmedian(list(cxv.values())):.3f}).")
PROMPT = """You are an expert in Barrett's oesophagus genomics (Killcoyne et al. 2020: copy-number changes, especially losses of 17p (TP53) and 9p (CDKN2A), gains of 8q, and overall genomic complexity, predict progression to high-grade dysplasia or cancer years in advance).
Given one patient's biopsy below, estimate the probability (0-100) that this patient's NEXT surveillance biopsy will show low-grade dysplasia confirmed on two reads, high-grade dysplasia or cancer.
Reply with ONLY a JSON object: {"risk": <integer 0-100>}
BIOPSY:
"""
def ask(text):
    body = json.dumps({"model": MODEL, "prompt": PROMPT + text, "stream": False, "format": "json", "think": False, "options": {"temperature": TEMP, "seed": REP_SEED, "num_predict": 60}}).encode()
    r = urllib.request.Request(BASE + "/api/generate", data=body, headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(r, timeout=300).read())["response"]; m = re.search(r'"risk"\s*:\s*"?(\d+)', resp); return int(m.group(1)) if m else None
    except Exception as e: return None
sids = [s for s in man.sample_id if s in A.index]
out = os.path.join(OUT, f"cnv_text_{MODEL.replace(':', '_')}{'_nograde' if NO_GRADE else ''}{f'_rep{REP_SEED}' if REP_SEED else ''}.csv"); done = set(pd.read_csv(out, dtype=str).sample_id) if os.path.exists(out) else set()
if not os.path.exists(out): open(out, "w").write("sample_id,risk\n")
import threading; lock = threading.Lock()
def work(sid):
    r = ask(describe(sid))
    with lock: open(out, "a").write(f"{sid},{'' if r is None else r}\n")
with ThreadPoolExecutor(CONC) as ex: list(ex.map(work, [s for s in sids if s not in done]))
d = pd.read_csv(out, dtype={"sample_id": str}); d["risk"] = pd.to_numeric(d.risk, errors="coerce"); d = d.merge(man, on="sample_id"); d["y"] = d.y_progressor.astype(int)
from sklearn.metrics import roc_auc_score
ok = d.risk.notna(); pat = d[ok].groupby("patient_id").agg(y=("y", "max"), r=("risk", "max"))
res = {"model": MODEL, "prompt_includes_current_grade": not NO_GRADE, "rep_seed": REP_SEED, "temperature": TEMP, "n_samples": len(d), "parsed": int(ok.sum()), "sample_auroc": round(float(roc_auc_score(d.y[ok], d.risk[ok])), 4), "patient_auroc_maxrisk": round(float(roc_auc_score(pat.y, pat.r)), 4),
       "risk_dist": {k: round(float(v), 2) for k, v in d.risk.describe().items()}, "reference_cnv_only_model": {"patient": 0.663, "sample": 0.6195}}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2); print(json.dumps(res, indent=1)); srv.terminate()
