"""2.28c: ERIN-trained VLM evaluated ZERO-SHOT on SWG (third cohort).

Pairs: SWG slides <-> Barrett's-DB matched reports (specimen-number join, same
normalisation as the slide-CSV backfill). Readouts:
  (a) slide->report retrieval among matched pairs vs chance;
  (b) zero-shot grading vs the release PATHOLOGIST grade (NDBE vs LGD+) —
      human truth, not jury, on a cohort the VLM never saw.
Eval-only: one forward per pair; CPU-capable.
"""
import json, os, re, subprocess, time, urllib.request
import numpy as np, pandas as pd
import torch, torch.nn as nn

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
EXP = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
CK = T + "/feasibility/runs/vlm_pretrain/output/vlm_best.pt"
OUT = os.environ.get("OUTDIR", ".")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = {"NDBE": "Barrett's oesophagus with intestinal metaplasia, negative for dysplasia.",
           "LGDplus": "Barrett's oesophagus with dysplasia or adenocarcinoma."}

os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models")
PORT = 21000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000
os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
BASE = f"http://127.0.0.1:{PORT}"
srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(60):
    try:
        urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception:
        time.sleep(2)

def embed(text):
    body = json.dumps({"model": "nomic-embed-text", "prompt": text[:6000]}).encode()
    req = urllib.request.Request(BASE + "/api/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    return np.array(json.loads(urllib.request.urlopen(req, timeout=120).read())["embedding"],
                    dtype=np.float32)

class ImgTower(nn.Module):
    def __init__(self, d_in=1536, d_out=768):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(d_in, 512), nn.GELU())
        self.att = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.proj = nn.Linear(512, d_out)
    def forward(self, bag):
        h = self.emb(bag)
        a = self.att(h).softmax(dim=0)
        z = self.proj((a * h).sum(0))
        return z / z.norm()

ck = torch.load(CK, map_location=DEV)
img = ImgTower().to(DEV); img.load_state_dict(ck["img"]); img.eval()
txt = nn.Linear(768, 768).to(DEV); txt.load_state_dict(ck["txt"]); txt.eval()
def enc_txt(v):
    z = txt(torch.tensor(v, device=DEV))
    return (z / z.norm(dim=-1, keepdim=True)).detach().cpu().numpy()

# --- match SWG samples to DB reports by specimen number in the slide filename ---
def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
rep = pd.read_csv(EXP + "/pathology_text_normalised_full.csv", dtype=str, low_memory=False).fillna("")
rep["_sn"] = rep["specimennumber"].map(norm)
rep = rep[rep["_sn"].str.len() >= 6].drop_duplicates("_sn").set_index("_sn")
man = pd.read_csv(F + "/training_manifest.csv", dtype=str)
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).merge(
    man, left_on="SampleID", right_on="sample_id")
uidx = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str)
npz_of = dict(zip(uidx[uidx["status"] == "ok"]["sample_id"], uidx[uidx["status"] == "ok"]["npz_path"]))
NUMG = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
pairs = []
sns = list(rep.index)
# 2.40 follow-up: optionally drop SWG patients that overlap ERIN (VLM training cohort)
EXCL_F = os.environ.get("EXCLUDE_PATIENTS", "")
if EXCL_F and not os.path.exists(EXCL_F):
    raise SystemExit(f"EXCLUDE_PATIENTS set but missing: {EXCL_F}")
EXCL = set(open(EXCL_F).read().split()) if EXCL_F else set()
n_excl = 0
for _, r in coh.iterrows():
    if r["sample_id"] not in npz_of: continue
    if EXCL and (str(r.get("patient_id")) in EXCL or str(r.get("PatientID_real")) in EXCL):
        n_excl += 1; continue
    stem_n = norm(os.path.basename(str(r["ImageAbsPath"])))
    hit = next((sn for sn in sns if sn in stem_n), None)
    if hit is None: continue
    text = str(rep.loc[hit, "reporttext"])
    if len(text) < 60: continue
    pairs.append({"sample": r["sample_id"], "npz": npz_of[r["sample_id"]],
                  "text": text, "grade": NUMG.get(str(r["Label"]).strip(), None)})
print(f"matched slide-report pairs: {len(pairs)}", flush=True)

zi, zt = [], []
for i, p in enumerate(pairs):
    z = np.load(p["npz"])
    bag = torch.tensor(np.asarray(z["embeddings"]), dtype=torch.float32, device=DEV)
    with torch.no_grad():
        zi.append(img(bag).cpu().numpy())
    zt.append(enc_txt(embed(p["text"])))
    if i % 25 == 0: print(i, flush=True)
zi, zt = np.stack(zi), np.stack(zt)
sims = zi @ zt.T
rank = (-sims).argsort(1)
hit = np.array([np.where(rank[i] == i)[0][0] for i in range(len(pairs))])
zp = np.stack([enc_txt(embed(PROMPTS["NDBE"])), enc_txt(embed(PROMPTS["LGDplus"]))])
zs = zi @ zp.T
score = zs[:, 1] - zs[:, 0]
y = np.array([p["grade"] for p in pairs], dtype=float)
ok = ~np.isnan(y)
from sklearn.metrics import roc_auc_score
np.savez(os.path.join(OUT, "pair_preds.npz"), sample=np.array([p["sample"] for p in pairs]),
         rank_of_true=hit, zeroshot_score=score, grade=y)
res = {"_meta": {"n_pairs": len(pairs), "checkpoint": CK, "graded": int(ok.sum()),
                 "excluded_overlap_samples": n_excl, "exclude_file": EXCL_F},
       "retrieval": {"recall_at_1": round(float((hit == 0).mean()), 4),
                     "recall_at_5": round(float((hit < 5).mean()), 4),
                     "chance_at_1": round(1 / len(pairs), 5)},
       "zeroshot_grade_vs_pathologist": {
           "auc_ndbe_vs_lgdplus": round(float(roc_auc_score((y[ok] >= 2).astype(int), score[ok])), 4)
           if 0 < (y[ok] >= 2).sum() < ok.sum() else None,
           "n_pos": int((y[ok] >= 2).sum())}}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
srv.terminate()
