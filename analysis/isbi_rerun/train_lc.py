"""Learning curve (follow-up 4): arm B labels, patient holdout, random NESTED subsets of training patients per fold
(sizes from env SIZES, e.g. 150,500,1000), same folds / model / seeds as train_fold.py; "all" = existing arm-B models.
Writes preds/lc_B_<task>_n<size>_seed<s>_fold<F>.npz (test fold + human holdout). Derived from train_fold.py — Part 2 GPU job: one fold, ALL arms x tasks x seeds x holdouts on a GPU-resident dataset.
Env: FOLD (0-4). Reads analysis/isbi_rerun/{slides.csv,pack/bags_fp16.dat,pack/index.json,qc/exclusions.csv,qc/summary.json}.
Writes preds/<holdout>_<arm>_<task>_seed<s>_fold<F>.npz (idx + prob[s]) and models/*.pt; config.json.
Model = the thesis ABMIL (abmil_cox.ABMIL: 1536->384 ReLU drop .25, gated attention 128, linear head), batched with masks;
25 epochs, Adam 1e-4, wd 1e-5, minibatch 32 bags, 800 tiles resampled per bag per step (as abmil_clf), seeds 0/1/2.
Binary task = NDBE-or-below (NORMAL_OTHER, NDBE) vs LGD+ (LGD, HGD, CANCER); IND excluded from the binary task.
Six-class = NORMAL_OTHER/NDBE/IND/LGD/HGD/CANCER, plain cross-entropy. Stops if qc/summary.json STOP_RULE is set."""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn

A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun"; FOLD = int(os.environ["FOLD"])
os.makedirs(A + "/preds", exist_ok=True); os.makedirs(A + "/models_lc", exist_ok=True)
SIZES = [int(x) for x in os.environ.get("SIZES", "150,500,1000").split(",")]
CFG = {"fold": FOLD, "epochs": 25, "lr": 1e-4, "weight_decay": 1e-5, "mb": 32, "tiles_per_step": 800, "cap_resident": 1200, "seeds": [0, 1, 2],
       "arms": {"A": "case_max", "B": "own_section", "C": "own_section_votefrac>=0.75"}, "tasks": ["bin", "six"], "holdouts": ["patient", "report"],
       "binary": "NORMAL_OTHER+NDBE vs LGD+HGD+CANCER (IND excluded)", "classes": ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"],
       "precision": "fp32 with TF32 matmul; grad clip 5.0; non-finite steps skipped (counted)", "torch": torch.__version__}
qsum = json.load(open(A + "/qc/summary.json"))
if qsum.get("STOP_RULE_gt5pct"): print("STOP RULE: QC excluded >5% — not training", flush=True); sys.exit(2)
s = pd.read_csv(A + "/slides.csv", dtype={"idx": int}).fillna("")
excl = set(pd.read_csv(A + "/qc/exclusions.csv").idx) if os.path.getsize(A + "/qc/exclusions.csv") > 0 else set()
idxf = json.load(open(A + "/pack/index.json")); ntiles = np.array(idxf["n_tiles"]); N, CAP, D = len(s), idxf["cap"], idxf["dim"]
CLS = CFG["classes"]; C_OF = {c: i for i, c in enumerate(CLS)}
s["ok"] = (~s.idx.isin(excl)) & (s.worst_grade != "") & (ntiles[s.idx.values] > 0)
# report-level holdout: human patients' NON-human reports rejoin training folds (patient hash), human reports stay held out
rng0 = np.random.RandomState(1)
hp = sorted(s[s.human_patient].anon_id.unique()); fold_hp = {a: i % 5 for i, a in enumerate(rng0.permutation(hp))}
s["fold_report"] = np.where(s.human_report, -1, np.where(s.fold >= 0, s.fold, s.anon_id.map(fold_hp)))
s["fold_patient"] = s.fold

dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
t0 = time.time()
X = torch.empty((N, CAP, D), dtype=torch.float16, device=dev)
for c in idxf["chunks"]:
    arr = np.load(A + "/pack/" + c["file"], mmap_mode="r")
    for a in range(0, c["hi"] - c["lo"], 256):
        e = min(a + 256, c["hi"] - c["lo"]); X[c["lo"] + a:c["lo"] + e] = torch.from_numpy(np.ascontiguousarray(arr[a:e])).to(dev)
NT = torch.tensor(ntiles, device=dev); print(f"resident dataset on GPU: {X.numel() * 2 / 1e9:.1f} GB in {time.time() - t0:.0f}s", flush=True)

class ABMIL(nn.Module):
    def __init__(self, d_in=1536, d=384, d_attn=128, p_drop=0.25, n_out=1):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.attn_v = nn.Sequential(nn.Linear(d, d_attn), nn.Tanh()); self.attn_u = nn.Sequential(nn.Linear(d, d_attn), nn.Sigmoid())
        self.attn_w = nn.Linear(d_attn, 1); self.head = nn.Linear(d, n_out)
    def forward(self, bag, mask):                       # bag (B,n,d_in), mask (B,n) True=valid
        h = self.emb(bag) * mask.unsqueeze(-1); a = self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf")).softmax(dim=1)
        return self.head((a.unsqueeze(-1) * h).sum(1))

def gather(ids, g, k=800):
    """(B,k,D) bags: k tiles sampled without replacement from each slide's resident tiles (all tiles if fewer)."""
    B = len(ids); n = NT[ids]; out = torch.zeros((B, k, D), dtype=torch.float16, device=dev)   # zeros, not empty: uninitialised padding leaked NaN/inf through 0*x
    mask = torch.zeros((B, k), dtype=torch.bool, device=dev)
    for b, i in enumerate(ids.tolist()):
        ni = int(n[b]); kk = min(k, ni)
        sel = torch.randperm(ni, generator=g, device="cpu")[:kk].to(dev) if ni > k else torch.arange(kk, device=dev)
        out[b, :kk] = X[i, sel]; mask[b, :kk] = True
    return out, mask

def labels(arm, task, d):
    g = d.case_max if arm == "A" else d.worst_grade
    if task == "bin":
        keep = g.isin(["NORMAL_OTHER", "NDBE", "LGD", "HGD", "CANCER"]); y = g.isin(["LGD", "HGD", "CANCER"]).astype(int)
    else:
        keep = g.isin(CLS); y = g.map(C_OF).fillna(-1).astype(int)
    if arm == "C": keep &= pd.to_numeric(d.vote_frac, errors="coerce") >= 0.75
    return keep.values, y.values

def train(ids, y, task, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); rng = np.random.RandomState(seed)
    model = ABMIL(n_out=1 if task == "bin" else 6).to(dev); opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    yt = torch.tensor(y, device=dev, dtype=torch.float32 if task == "bin" else torch.long)
    if task == "bin": lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((len(y) - y.sum()) / max(y.sum(), 1)), device=dev))
    else: lossf = nn.CrossEntropyLoss()
    model.train(); skipped = 0; ep_losses = []
    for ep in range(CFG["epochs"]):
        order = rng.permutation(len(ids)); tot = 0.0; nb = 0
        for i in range(0, len(order), CFG["mb"]):
            ch = order[i:i + CFG["mb"]]; bags, mask = gather(torch.tensor(ids[ch], device=dev), g, CFG["tiles_per_step"])
            logit = model(bags.float(), mask)                                   # fp32 (TF32 matmul); bf16 autocast diverged to NaN
            loss = lossf(logit.squeeze(-1), yt[ch]) if task == "bin" else lossf(logit, yt[ch])
            if not torch.isfinite(loss): skipped += 1; opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += float(loss); nb += 1
        ep_losses.append(round(tot / max(nb, 1), 4))
    return model, (ep_losses[-1] if ep_losses else float("nan")), {"epoch_losses": ep_losses, "skipped_steps": skipped}

@torch.inference_mode()
def predict(model, ids, task):
    model.eval(); g = torch.Generator().manual_seed(0); out = []
    for i in range(0, len(ids), 64):
        bags, mask = gather(torch.tensor(ids[i:i + 64], device=dev), g, CAP)      # all resident tiles at test time
        lg = model(bags.float(), mask)
        out.append(torch.sigmoid(lg.squeeze(-1)).cpu().numpy() if task == "bin" else torch.softmax(lg, 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,))

json.dump({**CFG, "sizes": SIZES, "subset": "nested random subsets of training PATIENTS, permutation seeded by 1000*seed+fold"}, open(A + f"/config_lc_fold{FOLD}_n{'-'.join(map(str, SIZES))}.json", "w"), indent=1)
log = []; fcol = "fold_patient"; arm = "B"
for task in CFG["tasks"]:
    d = s[s.ok]; keep, y = labels(arm, task, d)
    trmask = keep & (d[fcol].values != FOLD) & (d[fcol].values >= 0); tr_all = d[trmask]; y_all = y[trmask]
    te = d[d[fcol] == FOLD]; ho = d[d[fcol] == -1]
    for seed in CFG["seeds"]:
        pats = np.random.RandomState(1000 * seed + FOLD).permutation(sorted(tr_all.anon_id.unique()))
        for size in SIZES:
            if size >= len(pats): print(f"size {size} >= {len(pats)} training patients: skipped (use arm-B full models)", flush=True); continue
            sub = tr_all.anon_id.isin(set(pats[:size])).values; tr = tr_all[sub]; ytr = y_all[sub]
            t1 = time.time(); model, last, info = train(tr.idx.values, ytr, task, seed)
            p_te = predict(model, te.idx.values, task); p_ho = predict(model, ho.idx.values, task)
            tag = f"lc_B_{task}_n{size}_seed{seed}_fold{FOLD}"
            np.savez_compressed(A + f"/preds/{tag}.npz", test_idx=te.idx.values, test_prob=p_te, holdout_idx=ho.idx.values, holdout_prob=p_ho, n_train=len(tr), n_train_patients=size, train_pos=int(ytr.sum()) if task == "bin" else -1)
            torch.save(model.state_dict(), A + f"/models_lc/{tag}.pt")
            log.append({"tag": tag, "n_train_patients": size, "n_train": int(len(tr)), "n_test": int(len(te)), "n_holdout": int(len(ho)), "last_loss": round(last, 4), "sec": round(time.time() - t1, 1), "converged": bool(np.isfinite(last) and last < 5.0 and info["skipped_steps"] == 0), **info})
            print(log[-1], flush=True)
json.dump(log, open(A + f"/train_log_lc_fold{FOLD}_n{'-'.join(map(str, SIZES))}.json", "w"), indent=1)
print(f"fold {FOLD} learning curve complete: {len(log)} models in {(time.time() - t0) / 60:.1f} min", flush=True)
