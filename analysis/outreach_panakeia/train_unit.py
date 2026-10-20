"""GPU unit: one (task, split, model, repeat) -> out-of-fold predictions preds/{task}_{split}_{model}_r{rep}.csv.
task: wgd | cin | aneu | msi (CV over folds.csv column {task}_{split}_r{rep}) or wgdx (train all STAD, test ESCA-adeno; rep = seed).
model: abmil | meanlr | shuf (ABMIL on labels permuted within the training fold).
ABMIL, its training loop (800 tiles resampled per bag per step) and the mean-pool logistic head are copied from
analysis/tilebudget/code/train_eval_unit.py; hyperparameters fixed in DESIGN.md. Usage: train_unit.py TASK SPLIT MODEL REP"""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn

A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia"; D = 1536
dev = "cuda" if torch.cuda.is_available() else "cpu"   # GPU when free; epyc twins otherwise (features precomputed)
if dev == "cpu": torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
TASK, SPLIT, MODEL, REP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
PACK = os.environ.get("PACK", A + "/pack"); PRED = os.environ.get("PRED", A + "/preds")
os.makedirs(PRED, exist_ok=True); os.makedirs(A + "/attn", exist_ok=True)
tag = f"{TASK}_{SPLIT}_{MODEL}_r{REP}"; out = f"{PRED}/{tag}.csv"
if os.path.exists(out): print("done already"); sys.exit(0)
HP = {"epochs": 25, "lr": 1e-4, "weight_decay": 1e-5, "mb": 32, "tiles_per_step": 800}
REG = TASK == "aneu"; COL = {"wgd": "wgd", "wgdx": "wgd", "cin": "cin", "aneu": "aneuploidy", "msi": "msi"}[TASK]
torch.backends.cuda.matmul.allow_tf32 = True
# ---------------- data ----------------
ix = json.load(open(PACK + "/index.json")); P = ix["patients"]; row = {p: i for i, p in enumerate(P)}
t0 = time.time(); X = np.load(PACK + "/X.npy", mmap_mode="r"); XU = torch.empty(X.shape, dtype=torch.float16, device=dev)
for a in range(0, len(P), 32): XU[a:a + 32] = torch.from_numpy(np.ascontiguousarray(X[a:a + 32])).to(dev)
NT = np.array(ix["n_tiles"]); print(f"pack {tuple(XU.shape)} on {dev} in {time.time() - t0:.0f}s", flush=True)
c = pd.read_csv(A + "/cohort.csv").set_index("patient"); folds = pd.read_csv(A + "/folds.csv").set_index("patient")
c = c.loc[[p for p in c.index if p in row and NT[row[p]] > 0]]
lab = c[c[COL].notna()]
# ---------------- model (tile-budget copy) ----------------
class ABMIL(nn.Module):
    def __init__(self, d_in=1536, d=384, d_attn=128, p_drop=0.25, n_out=1):
        super().__init__(); self.emb = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.attn_v = nn.Sequential(nn.Linear(d, d_attn), nn.Tanh()); self.attn_u = nn.Sequential(nn.Linear(d, d_attn), nn.Sigmoid()); self.attn_w = nn.Linear(d_attn, 1); self.head = nn.Linear(d, n_out)
    def forward(self, bag, mask, attn=False):
        h = self.emb(bag) * mask.unsqueeze(-1); a = self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(1)
        out = self.head((a.unsqueeze(-1) * h).sum(1))
        return (out, a) if attn else out
def bags_from_rows(rows_list):
    k = max(1, max(len(r) for r in rows_list)); out = torch.zeros((len(rows_list), k, D), dtype=torch.float16, device=dev); mask = torch.zeros((len(rows_list), k), dtype=torch.bool, device=dev)
    for b, r in enumerate(rows_list): out[b, :len(r)] = r; mask[b, :len(r)] = True
    return out, mask
def train_bag(i, g):
    ni = int(NT[i]); kk = min(HP["tiles_per_step"], ni)
    selp = torch.randperm(ni, generator=g)[:kk].to(dev) if ni > kk else torch.arange(kk, device=dev); return XU[i, selp]
def lossf_for(y):
    if REG: return nn.MSELoss()
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((len(y) - y.sum()) / max(y.sum(), 1)), device=dev))
def train_abmil(ids, y, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); rng = np.random.RandomState(seed)
    model = ABMIL().to(dev); opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    yt = torch.tensor(y, device=dev, dtype=torch.float32); lossf = lossf_for(y); model.train(); last = float("nan")
    for ep in range(HP["epochs"]):
        order = rng.permutation(len(ids)); tot = 0.0; nb = 0
        for i in range(0, len(order), HP["mb"]):
            ch = order[i:i + HP["mb"]]; bags, mask = bags_from_rows([train_bag(int(ids[j]), g) for j in ch])
            loss = lossf(model(bags.float(), mask).squeeze(-1), yt[ch])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); tot += float(loss); nb += 1
        last = tot / max(nb, 1)
    return model, last
@torch.inference_mode()
def predict_abmil(model, ids):
    model.eval(); sc, at = [], {}
    for a in range(0, len(ids), 16):
        ch = [int(i) for i in ids[a:a + 16]]; bags, mask = bags_from_rows([XU[i, :int(NT[i])] for i in ch])
        out, att = model(bags.float(), mask, attn=True); out = out.squeeze(-1)
        sc.append((out if REG else torch.sigmoid(out)).cpu().numpy())
        for b, i in enumerate(ch): at[P[i]] = att[b, :int(NT[i])].cpu().numpy().astype(np.float32)
    return np.concatenate(sc), at
def mean_feats(ids):
    with torch.no_grad(): return torch.stack([XU[int(i), :int(NT[int(i)])].float().mean(0) for i in ids])
def train_meanlr(ids, y, seed):
    torch.manual_seed(seed); Xf = mean_feats(ids); mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Xf = (Xf - mu) / sd
    head = nn.Linear(D, 1).to(dev); opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-3)
    yt = torch.tensor(y, device=dev, dtype=torch.float32); lossf = lossf_for(y)
    for it in range(600):
        opt.zero_grad(); loss = lossf(head(Xf).squeeze(-1), yt); loss.backward(); opt.step()
    return head, mu, sd, float(loss)
@torch.inference_mode()
def predict_meanlr(m, ids):
    head, mu, sd, _ = m; o = head((mean_feats(ids) - mu) / sd).squeeze(-1); return (o if REG else torch.sigmoid(o)).cpu().numpy()
# ---------------- splits ----------------
if TASK == "wgdx":
    tr = lab[lab.cohort == "stad"]; te = lab[lab.cohort == "esca"]; splits = [(0, tr.index, te.index)]
else:
    fc = folds.loc[lab.index, f"{TASK}_{SPLIT}_r{REP}"]; assert (fc >= 0).all()
    splits = [(k, fc.index[fc != k], fc.index[fc == k]) for k in range(5)]
rows, attn, log = [], {}, []
for k, trp, tep in splits:
    ytr = lab.loc[trp, COL].values.astype(np.float32)
    if REG: mu_y, sd_y = ytr.mean(), ytr.std() + 1e-6; ytr = (ytr - mu_y) / sd_y
    if MODEL == "shuf": ytr = np.random.RandomState(1000 + 100 * REP + k).permutation(ytr)
    itr = np.array([row[p] for p in trp]); ite = np.array([row[p] for p in tep]); seed = 10 * REP + k; t1 = time.time()
    if MODEL in ("abmil", "shuf"):
        m, last = train_abmil(itr, ytr, seed); s, at = predict_abmil(m, ite)
        if MODEL == "abmil" and SPLIT in ("strat", "x") and REP == 0: attn.update(at)
    else:
        m = train_meanlr(itr, ytr, seed); last = m[3]; s = predict_meanlr(m, ite)
    if REG: s = s * sd_y + mu_y
    rows += [{"patient": p, "fold": k, "score": float(v), "y": float(lab.loc[p, COL]), "cohort": lab.loc[p, "cohort"]} for p, v in zip(tep, s)]
    log.append({"fold": k, "n_train": len(itr), "n_test": len(ite), "last_loss": round(last, 4), "sec": round(time.time() - t1, 1)}); print(log[-1], flush=True)
    assert np.isfinite(s).all(), "non-finite predictions"
if attn: np.savez_compressed(f"{A}/attn/{tag}.npz", **attn)
pd.DataFrame(rows).to_csv(out + ".part", index=False); os.replace(out + ".part", out)
json.dump(log, open(f"{PRED}/{tag}.log.json", "w"), indent=1); print("unit complete", tag, flush=True)
