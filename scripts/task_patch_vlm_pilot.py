"""B: image-only zero-shot grading pilot (docs/llm_extensions_preregistration.md).
100 dual-labelled ERIN slides stratified by section grade; 8 highest-attention tiles each (section-trained
six-class MIL weights from erin_tilemaps); tiles read from the slide at the extraction level (coords are in
that level's pixel space) and sent as PNG to a vision LLM via ollama. Slide score = max tile grade and mean
P(LGD+). Env: MODEL (medgemma | medgemma:27b), N_SLIDES (100), OUTDIR."""
import base64, io, json, os, re, subprocess, time, urllib.request, threading
import h5py, numpy as np, pandas as pd, torch, torch.nn as nn, openslide
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", "."); MODEL = os.environ.get("MODEL", "medgemma:27b")
NS = int(os.environ.get("N_SLIDES", "100")); K = 8; CONC = int(os.environ.get("CONC", "4"))
CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLASSES)}
WTS = T + "/feasibility/runs/erin_tilemaps/output/mcmil_section.pt"
os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models"); os.environ.setdefault("OLLAMA_NUM_PARALLEL", str(CONC))
PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000; os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"; BASE = f"http://127.0.0.1:{PORT}"
slog = open(os.path.join(OUT, "ollama_server.log"), "w"); srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=slog, stderr=slog)
for _ in range(60):
    try: urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
lab = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str); lab = lab[lab.worst_grade.isin(C_OF)]
rs = np.random.RandomState(7); per = max(1, NS // 6); chosen = []
for c in CLASSES:
    pool = lab[lab.worst_grade == c]; chosen += list(pool.sample(min(per, len(pool)), random_state=rs).h5)
lab = lab.set_index("h5"); print("slides", len(chosen), flush=True)
class MC_MIL(nn.Module):
    def __init__(self, d_in=1536, n_cls=6):
        super().__init__(); self.emb = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.1))
        self.att_v = nn.Linear(512, 128); self.att_u = nn.Linear(512, 128); self.att_w = nn.Linear(128, 1); self.head = nn.Linear(512, n_cls)
    def tiles(self, bag):
        h = self.emb(bag); a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).softmax(0); return h, a.squeeze(-1)
net = MC_MIL(); net.load_state_dict(torch.load(WTS, map_location="cpu")); net.eval()
PROMPT = ("This is a 224x224 pixel haematoxylin-and-eosin histology tile at 20x from an oesophageal (Barrett's surveillance) biopsy. "
          "Classify the tissue in this tile using exactly one label: NDBE (Barrett's/intestinal metaplasia or benign/normal mucosa without dysplasia), "
          "IND (indefinite for dysplasia), LGD (low-grade dysplasia), HGD (high-grade dysplasia), CANCER (adenocarcinoma). "
          'Reply with ONLY a JSON object: {"grade":"NDBE|IND|LGD|HGD|CANCER"}')
ORD5 = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
def ask(png_b64):
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "images": [png_b64], "stream": False, "format": "json", "options": {"temperature": 0, "num_predict": 40}}).encode()
    try:
        resp = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/api/generate", data=body, headers={"Content-Type": "application/json"}), timeout=300).read())["response"]
        hits = [g for g in ORD5 if re.search(rf"\b{g}\b", resp)]; return hits[0] if len(hits) == 1 else None
    except Exception: return None
out = os.path.join(OUT, f"tiles_{MODEL.replace(':', '_')}.csv"); done = set()
if os.path.exists(out): done = set(pd.read_csv(out, dtype=str).h5)
else: open(out, "w").write("h5,tile_idx,x,y,attn,grade\n")
lock = threading.Lock()
def work(h5p):
    try:
        with h5py.File(h5p) as h: X = np.asarray(h["features"], np.float32); coords = np.asarray(h["coords"]); attrs = dict(h.attrs)
        with torch.no_grad(): _, a = net.tiles(torch.tensor(X)); a = a.numpy()
        top = np.argsort(-a)[:K]; sl = openslide.OpenSlide(attrs["slide_path"]); lvl = int(attrs.get("level", 0)); ds = sl.level_downsamples[lvl]
        rows = []
        for i in top:
            x, y = int(coords[i][0]), int(coords[i][1]); img = sl.read_region((int(x * ds), int(y * ds)), lvl, (224, 224)).convert("RGB").resize((448, 448), Image.BICUBIC)
            buf = io.BytesIO(); img.save(buf, format="PNG"); g = ask(base64.b64encode(buf.getvalue()).decode())
            rows.append(f"{h5p},{i},{x},{y},{a[i]:.5f},{g or 'PARSE_FAIL'}\n")
        with lock: open(out, "a").writelines(rows)
    except Exception as e:
        with lock: open(out, "a").write(f"{h5p},-1,0,0,0,ERROR:{str(e)[:40].replace(',', ';')}\n")
with ThreadPoolExecutor(CONC) as ex: list(ex.map(work, [h for h in chosen if h not in done]))
d = pd.read_csv(out, dtype=str); d = d[d.grade.isin(ORD5)]; d["o"] = d.grade.map(ORD5)
sl_ = d.groupby("h5").agg(max_o=("o", "max"), mean_o=("o", "mean"), frac_lgdplus=("o", lambda v: float((v >= 2).mean())), n=("o", "size"))
sl_["truth"] = lab.loc[sl_.index, "worst_grade"].values; sl_["y"] = sl_.truth.isin(["LGD", "HGD", "CANCER"]).astype(int)
res = {"model": MODEL, "slides_scored": len(sl_), "tiles_parsed": int(len(d)), "parse_fail_tiles": int((pd.read_csv(out, dtype=str).grade == "PARSE_FAIL").sum()),
       "tile_grade_dist": d.grade.value_counts().to_dict(), "truth_dist": sl_.truth.value_counts().to_dict()}
if 0 < sl_.y.sum() < len(sl_):
    res["auroc_LGDplus_max_tile_grade"] = round(float(roc_auc_score(sl_.y, sl_.max_o)), 4); res["auroc_LGDplus_frac_tiles_LGDplus"] = round(float(roc_auc_score(sl_.y, sl_.frac_lgdplus)), 4); res["auroc_LGDplus_mean_grade"] = round(float(roc_auc_score(sl_.y, sl_.mean_o)), 4)
    res["gate_0.70"] = bool(max(res["auroc_LGDplus_max_tile_grade"], res["auroc_LGDplus_frac_tiles_LGDplus"]) >= 0.70)
sl_.to_csv(os.path.join(OUT, f"slides_{MODEL.replace(':', '_')}.csv")); json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2); print(json.dumps(res, indent=1)); srv.terminate()
