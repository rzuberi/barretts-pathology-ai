"""2.28 full build: CLIP-style report-slide model on ERIN, zero-shot transfer
to TCGA. (Signal test passed: R@1 16x chance, zero-shot grade AUC 0.823.)

Image tower: gated-attention pooling over UNI2 tile features (trained).
Text tower: frozen nomic-embed-text + trainable projection. InfoNCE both ways.
Train on ERIN train-eligible pairs (patient-disjoint val fold for early stop
+ a held-out TEST fold never seen). Readouts:
  (a) ERIN test fold: retrieval R@1/@5 + zero-shot grading vs supervised 0.90;
  (b) TCGA transfer (446 slides, reports from TCGA-Reports): slide->report
      retrieval + zero-shot OAC-vs-STAD site classification — cross-cohort
      generalisation without a single TCGA training pair.
Note: full 9,517-pan-cancer pretraining is NOT feasible (features exist for
446 TCGA slides only); this ERIN->TCGA transfer design is the honest scaled
version and is recorded as such in the plan.
"""
import glob, json, os, re, subprocess, time, urllib.request
import h5py, numpy as np, pandas as pd
import torch, torch.nn as nn

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TILES, EPOCHS, BATCH = 192, 25, 96
POS = {"LGD", "HGD", "CANCER"}
PROMPTS = {"NDBE": "Barrett's oesophagus with intestinal metaplasia, negative for dysplasia.",
           "LGDplus": "Barrett's oesophagus with dysplasia or adenocarcinoma.",
           "OAC_site": "Oesophageal adenocarcinoma resection specimen.",
           "STAD_site": "Gastric adenocarcinoma, stomach resection specimen."}

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

# ---- ERIN pairs ----
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv",
                  dtype=str, low_memory=False).fillna("")
rep["_text"] = rep[["FinalDiagnosis_redacted", "MicroscopicDescription_redacted"]].agg(" ".join, axis=1)
m = m.merge(rep[["CaseName", "_text"]].drop_duplicates("CaseName"), on="CaseName", how="inner")
elig = m[m["label_status"].isin(["train_eligible", "adjudicated"])
         & m["final_label"].isin(["NDBE", "LGD", "HGD", "CANCER"])
         & (m["_text"].str.len() > 40)].reset_index(drop=True)
print(f"ERIN pairs: {len(elig)}", flush=True)

tcache = os.path.join(OUT, "erin_text_emb.npz")
if os.path.exists(tcache):
    z = np.load(tcache, allow_pickle=True); text_emb = dict(zip(z["k"], z["v"]))
else:
    text_emb = {}
    for i, r in elig.iterrows():
        if r["CaseName"] not in text_emb:
            text_emb[r["CaseName"]] = embed(r["_text"])
        if i % 300 == 0: print("emb", i, flush=True)
    np.savez(tcache, k=np.array(list(text_emb)), v=np.stack(list(text_emb.values())))
prompt_emb = {k: embed(v) for k, v in PROMPTS.items()}

pats = elig["anon_id"].values
uniq = sorted(set(pats)); rng = np.random.RandomState(0)
fold_of = {a: i % 5 for i, a in enumerate(rng.permutation(uniq))}
fm = np.array([fold_of[a] for a in pats])
TR, VA, TE = np.where(fm <= 2)[0], np.where(fm == 3)[0], np.where(fm == 4)[0]
Yt = torch.tensor(np.stack([text_emb[c] for c in elig["CaseName"]]))
Yt = Yt / Yt.norm(dim=1, keepdim=True)
y_grade = elig["final_label"].isin(POS).astype(int).values

def load_tiles(h5p, k=MAX_TILES):
    with h5py.File(h5p) as h:
        X = np.asarray(h["features"])
    if len(X) > k:
        X = X[np.random.RandomState(0).choice(len(X), k, replace=False)]
    return torch.tensor(X, dtype=torch.float32)

class ImgTower(nn.Module):
    def __init__(self, d_in=1536, d_out=768):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(d_in, 512), nn.GELU())
        self.att = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.proj = nn.Linear(512, d_out)
    def forward(self, bags):          # list of (n_i, d_in)
        outs = []
        for b in bags:
            h = self.emb(b)
            a = self.att(h).softmax(dim=0)
            outs.append(self.proj((a * h).sum(0)))
        z = torch.stack(outs)
        return z / z.norm(dim=1, keepdim=True)

txt_proj = nn.Linear(768, 768).to(DEV)
img = ImgTower().to(DEV)
opt = torch.optim.AdamW(list(img.parameters()) + list(txt_proj.parameters()), lr=3e-4, weight_decay=1e-4)
logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=DEV))
opt.add_param_group({"params": [logit_scale]})

def encode_text(Y):
    z = txt_proj(Y.to(DEV))
    return z / z.norm(dim=1, keepdim=True)

def epoch_pass(idx, train=True):
    perm = np.random.permutation(idx) if train else idx
    tot, nb = 0.0, 0
    for b in range(0, len(perm), BATCH):
        ids = perm[b:b + BATCH]
        if len(ids) < 8: continue
        bags = [load_tiles(elig.loc[i, "h5"]).to(DEV) for i in ids]
        zi = img(bags); zt = encode_text(Yt[ids])
        logits = logit_scale.exp().clamp(max=100) * zi @ zt.T
        tgt = torch.arange(len(ids), device=DEV)
        loss = (nn.functional.cross_entropy(logits, tgt)
                + nn.functional.cross_entropy(logits.T, tgt)) / 2
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss) * len(ids); nb += len(ids)
    return tot / max(nb, 1)

best, patience = 9e9, 0
for ep in range(EPOCHS):
    tr_loss = epoch_pass(TR, True)
    with torch.no_grad():
        va_loss = epoch_pass(VA, False)
    print(f"ep{ep} train={tr_loss:.3f} val={va_loss:.3f}", flush=True)
    if va_loss < best - 1e-3:
        best, patience = va_loss, 0
        torch.save({"img": img.state_dict(), "txt": txt_proj.state_dict()},
                   os.path.join(OUT, "vlm_best.pt"))
    else:
        patience += 1
        if patience >= 4: break
ck = torch.load(os.path.join(OUT, "vlm_best.pt"))
img.load_state_dict(ck["img"]); txt_proj.load_state_dict(ck["txt"])
img.eval(); txt_proj.eval()

from sklearn.metrics import roc_auc_score
res = {"_meta": {"n_train": len(TR), "n_val": len(VA), "n_test": len(TE),
                 "epochs_ran": ep + 1, "max_tiles": MAX_TILES}}
with torch.no_grad():
    zi = torch.cat([img([load_tiles(elig.loc[i, "h5"]).to(DEV) for i in TE[b:b+64]]).cpu()
                    for b in range(0, len(TE), 64)])
    zt = encode_text(Yt[TE]).cpu()
    sims = (zi @ zt.T).numpy()
    rank = (-sims).argsort(1)
    hit = np.array([np.where(rank[i] == i)[0][0] for i in range(len(TE))])
    zp = encode_text(torch.tensor(np.stack([prompt_emb["NDBE"], prompt_emb["LGDplus"]]))).cpu()
    zs = (zi @ zp.T).numpy()
    res["erin_test"] = {
        "recall_at_1": round(float((hit == 0).mean()), 4),
        "recall_at_5": round(float((hit < 5).mean()), 4),
        "chance_at_1": round(1 / len(TE), 5),
        "zeroshot_grade_auc": round(float(roc_auc_score(
            y_grade[TE], zs[:, 1] - zs[:, 0])), 4)}
print("ERIN test:", res["erin_test"], flush=True)

# ---- TCGA transfer ----
treports = pd.read_csv(T + "/feasibility/runs/tcga_reports/output/TCGA_Reports.csv")
tcol = next(c for c in treports.columns if "text" in c.lower())
pcol = next(c for c in treports.columns if "patient" in c.lower() or "id" in c.lower())
treports["bc"] = treports[pcol].astype(str).str.extract(r"(TCGA-\w{2}-\w{4})")
FEATS = {"OAC": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2",
         "GEJ": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/features/20x_224px/features_uni_v2"}
tc_rows = []
for grp, fd in FEATS.items():
    for f in sorted(glob.glob(os.path.join(fd, "*.h5"))):
        mm = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
        if not mm: continue
        rr = treports[treports["bc"] == mm.group(1)]
        if rr.empty: continue
        tc_rows.append({"bc": mm.group(1), "h5": f, "grp": grp, "text": str(rr.iloc[0][tcol])})
tc = pd.DataFrame(tc_rows).drop_duplicates("bc").reset_index(drop=True)
print(f"TCGA transfer pairs: {len(tc)} ({tc['grp'].value_counts().to_dict()})", flush=True)
if len(tc) > 50:
    t_emb = np.stack([embed(t) for t in tc["text"]])
    with torch.no_grad():
        zi = torch.cat([img([load_tiles(h).to(DEV) for h in tc["h5"][b:b+64]]).cpu()
                        for b in range(0, len(tc), 64)])
        zt = encode_text(torch.tensor(t_emb)).cpu()
        sims = (zi @ zt.T).numpy()
        rank = (-sims).argsort(1)
        hit = np.array([np.where(rank[i] == i)[0][0] for i in range(len(tc))])
        zp = encode_text(torch.tensor(np.stack(
            [prompt_emb["OAC_site"], prompt_emb["STAD_site"]]))).cpu()
        zs = (zi @ zp.T).numpy()
        y_site = (tc["grp"] == "GEJ").astype(int).values
        res["tcga_transfer"] = {
            "n_pairs": len(tc),
            "recall_at_1": round(float((hit == 0).mean()), 4),
            "recall_at_5": round(float((hit < 5).mean()), 4),
            "chance_at_1": round(1 / len(tc), 5),
            "zeroshot_site_auc": round(float(roc_auc_score(
                y_site, zs[:, 1] - zs[:, 0])), 4)}
    print("TCGA transfer:", res["tcga_transfer"], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
srv.terminate()
