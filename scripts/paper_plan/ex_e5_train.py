"""E5: ERIN dysplasia models for transfer to the analysed SWG cohort (GPU; one job per ERIN patient fold, env FOLD 0-4).
Data, model and training exactly as analysis/isbi_rerun/code/train_fold.py arm B (own-section jury-v3 grade; GPU-resident pack of <= 1,200 UNI2-h tiles
per slide at 0.5 um/px; thesis gated-attention ABMIL 1536 -> 384, attention 128, dropout 0.25; Adam 1e-4, wd 1e-5, 25 epochs, minibatch 32 bags, 800
tiles resampled per bag per step; seeds 0/1/2; patient-level holdout folds of folds.json, the 130 hand-labelled patients never trained on).
Change: every slide of the ERIN patients in explore_E5/erin_exclude_anon_ids.txt (70 patients also in the SWG cohort) is dropped from training and test.
Tasks: lgd = NORMAL_OTHER/NDBE vs LGD/HGD/CANCER (IND excluded); hgd = NORMAL_OTHER/NDBE/LGD vs HGD/CANCER (IND excluded); six = six-class.
Writes explore_E5/erin_models/<task>_seed<s>_fold<F>.pt and explore_E5/erin_preds/<task>_seed<s>_fold<F>.npz (ERIN test-fold + human-holdout idx, prob)."""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun"; E5 = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm/explore_E5"
FOLD = int(os.environ["FOLD"]); MD = E5 + "/erin_models"; PD_ = E5 + "/erin_preds"; os.makedirs(MD, exist_ok=True); os.makedirs(PD_, exist_ok=True)
CFG = {"epochs": 25, "lr": 1e-4, "weight_decay": 1e-5, "mb": 32, "tiles_per_step": 800, "seeds": [0, 1, 2], "tasks": ["lgd", "hgd", "six"]}
CLS = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLS)}
s = pd.read_csv(A + "/slides.csv", dtype={"idx": int}).fillna("")
excl = set(pd.read_csv(A + "/qc/exclusions.csv").idx) if os.path.getsize(A + "/qc/exclusions.csv") > 0 else set()
EXP = set(l.strip() for l in open(E5 + "/erin_exclude_anon_ids.txt") if l.strip())
idxf = json.load(open(A + "/pack/index.json")); ntiles = np.array(idxf["n_tiles"]); N, CAP, D = len(s), idxf["cap"], idxf["dim"]
s["ok"] = (~s.idx.isin(excl)) & (s.worst_grade != "") & (ntiles[s.idx.values] > 0) & (~s.anon_id.isin(EXP))
print(f"slides ok {int(s.ok.sum())}; dropped as SWG-overlap patients {int(s.anon_id.isin(EXP).sum())} slides of {s[s.anon_id.isin(EXP)].anon_id.nunique()} patients", flush=True)
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; t0 = time.time()
X = torch.empty((N, CAP, D), dtype=torch.float16, device=dev)
for c in idxf["chunks"]:
    arr = np.load(A + "/pack/" + c["file"], mmap_mode="r")
    for a in range(0, c["hi"] - c["lo"], 256):
        e = min(a + 256, c["hi"] - c["lo"]); X[c["lo"] + a:c["lo"] + e] = torch.from_numpy(np.ascontiguousarray(arr[a:e])).to(dev)
NT = torch.tensor(ntiles, device=dev); print(f"resident ERIN pack {X.numel() * 2 / 1e9:.1f} GB in {time.time() - t0:.0f}s", flush=True)
class ABMIL(nn.Module):   # train_fold.py, unchanged
    def __init__(self, d_in=1536, d=384, d_attn=128, p_drop=0.25, n_out=1):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.attn_v = nn.Sequential(nn.Linear(d, d_attn), nn.Tanh()); self.attn_u = nn.Sequential(nn.Linear(d, d_attn), nn.Sigmoid())
        self.attn_w = nn.Linear(d_attn, 1); self.head = nn.Linear(d, n_out)
    def forward(self, bag, mask):
        h = self.emb(bag) * mask.unsqueeze(-1); a = self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf")).softmax(dim=1); return self.head((a.unsqueeze(-1) * h).sum(1))
def gather(ids, g, k=800):
    B = len(ids); n = NT[ids]; out = torch.zeros((B, k, D), dtype=torch.float16, device=dev); mask = torch.zeros((B, k), dtype=torch.bool, device=dev)
    for b, i in enumerate(ids.tolist()):
        ni = int(n[b]); kk = min(k, ni); sel = torch.randperm(ni, generator=g, device="cpu")[:kk].to(dev) if ni > k else torch.arange(kk, device=dev)
        out[b, :kk] = X[i, sel]; mask[b, :kk] = True
    return out, mask
def labels(task, d):
    g = d.worst_grade
    if task == "lgd": keep = g.isin(["NORMAL_OTHER", "NDBE", "LGD", "HGD", "CANCER"]); y = g.isin(["LGD", "HGD", "CANCER"]).astype(int)
    elif task == "hgd": keep = g.isin(["NORMAL_OTHER", "NDBE", "LGD", "HGD", "CANCER"]); y = g.isin(["HGD", "CANCER"]).astype(int)
    else: keep = g.isin(CLS); y = g.map(C_OF).fillna(-1).astype(int)
    return keep.values, y.values
def train(ids, y, task, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); rng = np.random.RandomState(seed)
    model = ABMIL(n_out=6 if task == "six" else 1).to(dev); opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    yt = torch.tensor(y, device=dev, dtype=torch.long if task == "six" else torch.float32)
    lossf = nn.CrossEntropyLoss() if task == "six" else nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((len(y) - y.sum()) / max(y.sum(), 1)), device=dev))
    model.train(); skipped = 0; ep_losses = []
    for ep in range(CFG["epochs"]):
        order = rng.permutation(len(ids)); tot = 0.0; nb = 0
        for i in range(0, len(order), CFG["mb"]):
            ch = order[i:i + CFG["mb"]]; bags, mask = gather(torch.tensor(ids[ch], device=dev), g, CFG["tiles_per_step"]); logit = model(bags.float(), mask)
            loss = lossf(logit, yt[ch]) if task == "six" else lossf(logit.squeeze(-1), yt[ch])
            if not torch.isfinite(loss): skipped += 1; opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); tot += float(loss); nb += 1
        ep_losses.append(round(tot / max(nb, 1), 4))
    return model, {"epoch_losses": ep_losses, "skipped_steps": skipped}
@torch.inference_mode()
def predict(model, ids, task):
    model.eval(); g = torch.Generator().manual_seed(0); out = []
    for i in range(0, len(ids), 64):
        bags, mask = gather(torch.tensor(ids[i:i + 64], device=dev), g, CAP); lg = model(bags.float(), mask)
        out.append(torch.softmax(lg, 1).cpu().numpy() if task == "six" else torch.sigmoid(lg.squeeze(-1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,))
log = []; d = s[s.ok]
for task in CFG["tasks"]:
    keep, y = labels(task, d); trm = keep & (d.fold.values != FOLD) & (d.fold.values >= 0); tr = d[trm]; ytr = y[trm]; te = d[d.fold == FOLD]; ho = d[d.fold == -1]
    for seed in CFG["seeds"]:
        t1 = time.time(); model, info = train(tr.idx.values, ytr, task, seed); tag = f"{task}_seed{seed}_fold{FOLD}"
        np.savez_compressed(f"{PD_}/{tag}.npz", test_idx=te.idx.values, test_prob=predict(model, te.idx.values, task), holdout_idx=ho.idx.values, holdout_prob=predict(model, ho.idx.values, task))
        torch.save(model.state_dict(), f"{MD}/{tag}.pt")
        log.append({"tag": tag, "n_train": int(len(tr)), "train_pos": int(ytr.sum()) if task != "six" else -1, "sec": round(time.time() - t1, 1), "last_loss": info["epoch_losses"][-1], "skipped": info["skipped_steps"]}); print(log[-1], flush=True)
json.dump(log, open(f"{E5}/erin_train_log_fold{FOLD}.json", "w"), indent=1); print(f"E5 TRAIN DONE fold {FOLD} in {(time.time() - t0) / 60:.1f} min", flush=True)
