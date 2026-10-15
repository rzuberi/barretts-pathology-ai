"""2.28 signal test: report-slide vision-language alignment on ERIN.

Lightweight version of the VLM rebuild: a 2-layer projection maps pooled UNI2
slide embeddings into a text-embedding space (nomic-embed-text via local
ollama), trained with InfoNCE on ERIN slide-report pairs, patient-disjoint
folds. Readouts: (a) slide->report retrieval recall@1/@5 on held-out pairs vs
chance; (b) ZERO-SHOT grading — classify slides by similarity to canonical
grade sentences, AUC NDBE-vs-LGD+ compared against the supervised probe
(~0.90) and chance. Pass = retrieval well above chance AND zero-shot AUC
meaningfully above 0.5; then the full 9,517-pair TCGA pretraining is justified.
"""
import json, os, subprocess, sys, time, urllib.request
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
CACHE = os.environ.get("POOLED", T + "/feasibility/runs/erin_probes/output/erin_pooled_uni2.npz")
EMB_MODEL = "nomic-embed-text"
POS = {"LGD", "HGD", "CANCER"}
PROMPTS = {
    "NDBE": "Barrett's oesophagus with intestinal metaplasia, negative for dysplasia.",
    "IND": "Barrett's oesophagus, indefinite for dysplasia.",
    "LGD": "Barrett's oesophagus with low grade dysplasia.",
    "HGD": "Barrett's oesophagus with high grade dysplasia.",
    "CANCER": "Invasive adenocarcinoma of the oesophagus.",
}

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
    body = json.dumps({"model": EMB_MODEL, "prompt": text[:6000]}).encode()
    req = urllib.request.Request(BASE + "/api/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    return np.array(json.loads(urllib.request.urlopen(req, timeout=120).read())["embedding"])

d = np.load(CACHE, allow_pickle=True)
X_all, h5s = d["X"], list(d["h5s"])
slide_emb = {h: X_all[i] for i, h in enumerate(h5s)}
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv",
                  dtype=str, low_memory=False).fillna("")
rep["_text"] = rep[["FinalDiagnosis_redacted", "MicroscopicDescription_redacted"]].agg(" ".join, axis=1)
m = m.merge(rep[["CaseName", "_text"]].drop_duplicates("CaseName"), on="CaseName", how="inner")
m = m[m["h5"].isin(slide_emb) & (m["_text"].str.len() > 40)]
elig = m[m["label_status"].isin(["train_eligible", "adjudicated"])
         & m["final_label"].isin(["NDBE", "LGD", "HGD", "CANCER"])].reset_index(drop=True)
print(f"paired slides: {len(elig)}", flush=True)

txt_cache = os.path.join(OUT, "text_emb.npz")
if os.path.exists(txt_cache):
    z = np.load(txt_cache, allow_pickle=True)
    text_emb = dict(zip(z["k"], z["v"]))
else:
    text_emb = {}
    for i, r in elig.iterrows():
        if r["CaseName"] not in text_emb:
            text_emb[r["CaseName"]] = embed(r["_text"])
        if i % 200 == 0: print("embedded", i, flush=True)
    np.savez(txt_cache, k=np.array(list(text_emb)), v=np.stack(list(text_emb.values())))
prompt_emb = {g: embed(t) for g, t in PROMPTS.items()}
print("text embeddings ready", flush=True)

import torch, torch.nn as nn
Xs = np.stack([slide_emb[h] for h in elig["h5"]]).astype(np.float32)
Xt = np.stack([text_emb[c] for c in elig["CaseName"]]).astype(np.float32)
Xt = Xt / np.linalg.norm(Xt, axis=1, keepdims=True)
y = elig["final_label"].isin(POS).astype(int).values
pats = elig["anon_id"].values
uniq = sorted(set(pats)); rng = np.random.RandomState(0)
fold_of = {a: i % 5 for i, a in enumerate(rng.permutation(uniq))}
fm = np.array([fold_of[a] for a in pats])
P = np.stack([prompt_emb[g] for g in PROMPTS])
P = P / np.linalg.norm(P, axis=1, keepdims=True)

res = {"_meta": {"n_pairs": len(elig), "emb_model": EMB_MODEL,
                 "text_dim": int(Xt.shape[1]), "slide_dim": int(Xs.shape[1])}}
r1s, r5s, zs_auc = [], [], []
zero_scores = np.zeros(len(elig))
for f in range(5):
    tr, te = np.where(fm != f)[0], np.where(fm == f)[0]
    torch.manual_seed(f)
    proj = nn.Sequential(nn.Linear(Xs.shape[1], 512), nn.GELU(), nn.Linear(512, Xt.shape[1]))
    opt = torch.optim.Adam(proj.parameters(), lr=1e-4, weight_decay=1e-4)
    Xtr = torch.tensor(Xs[tr]); Ttr = torch.tensor(Xt[tr])
    for ep in range(30):
        perm = torch.randperm(len(tr))
        for b in range(0, len(tr), 256):
            idx = perm[b:b + 256]
            if len(idx) < 8: continue
            z = proj(Xtr[idx]); z = z / z.norm(dim=1, keepdim=True)
            logits = z @ Ttr[idx].T / 0.07
            loss = nn.functional.cross_entropy(logits, torch.arange(len(idx)))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = proj(torch.tensor(Xs[te])); z = (z / z.norm(dim=1, keepdim=True)).numpy()
    sims = z @ Xt[te].T
    rank = (-sims).argsort(1)
    hit = np.array([np.where(rank[i] == i)[0][0] for i in range(len(te))])
    r1s.append(float((hit == 0).mean())); r5s.append(float((hit < 5).mean()))
    zsim = z @ P.T
    grades = list(PROMPTS)
    pos_ix = [grades.index(g) for g in ("LGD", "HGD", "CANCER")]
    zero_scores[te] = zsim[:, pos_ix].max(1) - zsim[:, grades.index("NDBE")]
    from sklearn.metrics import roc_auc_score
    zs_auc.append(float(roc_auc_score(y[te], zero_scores[te])))
    print(f"fold {f}: R@1={r1s[-1]:.3f} R@5={r5s[-1]:.3f} (chance {1/len(te):.4f}) "
          f"zeroshot AUC={zs_auc[-1]:.3f}", flush=True)

from sklearn.metrics import roc_auc_score
res["retrieval"] = {"recall_at_1": round(float(np.mean(r1s)), 4),
                    "recall_at_5": round(float(np.mean(r5s)), 4),
                    "chance_at_1": round(float(np.mean([1 / (fm == f).sum() for f in range(5)])), 5)}
res["zeroshot_grade"] = {"auc": round(float(roc_auc_score(y, zero_scores)), 4),
                         "per_fold": [round(a, 4) for a in zs_auc],
                         "supervised_probe_reference": 0.90}
res["signal_test"] = "PASS" if (res["retrieval"]["recall_at_5"] > 20 * res["retrieval"]["chance_at_1"]
                                and res["zeroshot_grade"]["auc"] > 0.65) else "FAIL"
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
srv.terminate()
