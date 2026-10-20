"""Phase 4 (GPU unit): DS x MODEL x FOLD x ARM -> train ABMIL (3 seeds) + mean-pool LR, evaluate every test slide under every
(rule x budget). Adapted from analysis/tilebudget/code/train_eval_unit.py (same ABMIL, hyperparameters, class-balanced BCE, mean-pool LR head).
Env: DS (tcga|c16), MODEL (uni2h|virchow2), FOLD (tcga 0-4; c16 0 = official split), ARM (all|r256).
Rules: all; rand0/rand1/rand2 (independent random draws, fixed per slide, shared across models/arms); attn_top / attn_bot (the model's own
attention over all tiles) at m in {1,4,16,64,256}. Output: preds/{DS}_{MODEL}_{ARM}_f{FOLD}_s{seed}.npz with keys "{agg}|{rule}|{m}" -> P(y=1)
for test_slides; for c16 + ARM=all also attention scores per test slide (figure 3, hit rate)."""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"
DS, MODEL, F, ARM = os.environ["DS"], os.environ["MODEL"], int(os.environ["FOLD"]), os.environ["ARM"]; dev = "cuda"
HP = {"epochs": 25, "lr": 1e-4, "weight_decay": 1e-5, "mb": 16, "seeds": [0, 1, 2], "k_r256": 256}
MS = [1, 4, 16, 64, 256]; SMOKE = os.environ.get("SMOKE") == "1"; SFX = "_smoke" if SMOKE else ""   # smoke: own folders, 2 epochs, train = test = all slides
PREDS, MODELS_DIR, PACKS = A + "/preds" + SFX, A + "/models" + SFX, A + "/data/packs" + SFX; os.makedirs(PREDS, exist_ok=True); os.makedirs(MODELS_DIR, exist_ok=True)
if SMOKE: HP["epochs"] = 2
tags = [f"{DS}_{MODEL}_{ARM}_f{F}_s{sd}" for sd in HP["seeds"]]
if all(os.path.exists(f"{PREDS}/{t}.npz") for t in tags): print("done already"); sys.exit(0)
torch.backends.cuda.matmul.allow_tf32 = True
# ---------------- data ----------------
s = pd.read_csv(A + "/manifests/slides.csv"); s["row"] = np.arange(len(s)); s = s[s.ds == DS].reset_index(drop=True)
ix = json.load(open(f"{PACKS}/{MODEL}_{DS}_index.json")); s = s[s.slide.isin(ix["slides"])].reset_index(drop=True)
if DS == "tcga": tr, te = s[s.fold != F], s[s.fold == F]
else: tr, te = s[s.split == "train"], s[s.split == "test"]
if SMOKE: tr = te = s
print(f"{DS}/{MODEL}/f{F}/{ARM}: train {len(tr)} ({int(tr.label.sum())} pos), test {len(te)} ({int(te.label.sum())} pos); {len(ix['missing'])} slides missing features", flush=True)
t0 = time.time(); path = f"{PACKS}/{MODEL}_{DS}.npy"
with open(path, "rb") as f:   # sequential read of the whole pack into host RAM (no mmap)
    v = np.lib.format.read_magic(f); shape, _, dt = np.lib.format._read_array_header(f, v); X = np.empty(shape, dt); buf = X.reshape(-1); o = 0
    while o < buf.size: n = f.readinto(memoryview(buf[o:o + (1 << 28)])); assert n > 0; o += n
D = X.shape[1]; print(f"pack {X.shape} ({X.nbytes / 1e9:.1f} GB) in RAM in {time.time() - t0:.0f}s", flush=True)
def span(sl): return ix["slides"][sl]
def bag(sl, sel=None):
    st, en = span(sl); x = X[st:en] if sel is None else X[st + sel]; return torch.from_numpy(np.ascontiguousarray(x)).to(dev).float()
def rand_perm(sl_row, n, d): return np.random.RandomState(1000 * d + int(sl_row)).permutation(n)   # draw d, fixed per slide (shared across models/arms)
fixed = {}
if ARM == "r256":
    for r in tr.itertuples(): st, en = span(r.slide); n = en - st; fixed[r.slide] = np.sort(rand_perm(r.row, n, 9)[:HP["k_r256"]])
# ---------------- model (as train_eval_unit.py) ----------------
class ABMIL(nn.Module):
    def __init__(self, d_in, d=384, d_attn=128, p_drop=0.25, n_out=1):
        super().__init__(); self.emb = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.attn_v = nn.Sequential(nn.Linear(d, d_attn), nn.Tanh()); self.attn_u = nn.Sequential(nn.Linear(d, d_attn), nn.Sigmoid()); self.attn_w = nn.Linear(d_attn, 1); self.head = nn.Linear(d, n_out)
    def parts(self, x): h = self.emb(x); return h, self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1)
    def pool(self, h, a): return self.head((a.softmax(0).unsqueeze(-1) * h).sum(0)).squeeze(-1)
    def forward(self, x): h, a = self.parts(x); return self.pool(h, a)
def train(seed):
    torch.manual_seed(seed); rng = np.random.RandomState(seed); y = tr.label.values.astype(np.float32); ids = tr.slide.values
    model = ABMIL(D).to(dev); opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((len(y) - y.sum()) / max(y.sum(), 1)), device=dev)); model.train(); last = float("nan")
    for ep in range(HP["epochs"]):
        order = rng.permutation(len(ids)); tot = 0.0
        for i in range(0, len(order), HP["mb"]):
            ch = order[i:i + HP["mb"]]; opt.zero_grad(set_to_none=True); lsum = 0.0
            for c in ch:   # gradient accumulation over the minibatch = mean loss over 16 bags (full bags differ in size)
                loss = lossf(model(bag(ids[c], fixed.get(ids[c]))).view(1), torch.tensor([y[c]], device=dev)) / len(ch); loss.backward(); lsum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); tot += lsum
        last = tot / max(1, (len(order) + HP["mb"] - 1) // HP["mb"])
    return model.eval(), last
def meanlr(seed):
    torch.manual_seed(seed); y = tr.label.values.astype(np.float32)
    with torch.no_grad(): Xf = torch.stack([bag(sl, fixed.get(sl)).mean(0) for sl in tr.slide.values])
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Xf = (Xf - mu) / sd; head = nn.Linear(D, 1).to(dev); opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-3)
    yt = torch.tensor(y, device=dev); lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((len(y) - y.sum()) / max(y.sum(), 1)), device=dev))
    for _ in range(600): opt.zero_grad(); loss = lossf(head(Xf).squeeze(-1), yt); loss.backward(); opt.step()
    return head, mu, sd
@torch.inference_mode()
def evaluate(model, head, mu, sd):
    out = {}; att = {}
    def put(k, v): out.setdefault(k, []).append(float(torch.sigmoid(v)))
    for r in te.itertuples():
        x = bag(r.slide); n = len(x); h, a = model.parts(x); o = torch.argsort(a, descending=True).cpu().numpy()
        put("abmil|all|0", model.pool(h, a)); put("meanlr|all|0", head(((x.mean(0) - mu) / sd)).squeeze(-1))
        sels = {f"rand{d}": rand_perm(r.row, n, d) for d in range(3)}; sels["attn_top"] = o; sels["attn_bot"] = o[::-1].copy()
        for rule, order in sels.items():
            for m in MS:
                i_ = torch.from_numpy(order[:m].copy()).to(dev); put(f"abmil|{rule}|{m}", model.pool(h[i_], a[i_])); put(f"meanlr|{rule}|{m}", head(((x[i_].mean(0) - mu) / sd)).squeeze(-1))
        if DS == "c16" and ARM == "all": att[r.slide] = a.half().cpu().numpy()
    return {k: np.array(v, np.float32) for k, v in out.items()}, att
log = []
for sd_, tag in zip(HP["seeds"], tags):
    if os.path.exists(f"{PREDS}/{tag}.npz"): continue
    t1 = time.time(); model, last = train(sd_); torch.save(model.state_dict(), f"{MODELS_DIR}/{tag}.pt"); t_tr = time.time() - t1
    head, mu, sd = meanlr(sd_); P, att = evaluate(model, head, mu, sd)
    np.savez_compressed(f"{PREDS}/{tag}.part.npz", test_slides=te.slide.values.astype(str), labels=te.label.values.astype(np.int8), **P, **{f"attn|{k}": v for k, v in att.items()})
    os.replace(f"{PREDS}/{tag}.part.npz", f"{PREDS}/{tag}.npz")
    log.append({"tag": tag, "n_train": len(tr), "n_test": len(te), "last_loss": round(last, 4), "train_s": round(t_tr, 1), "total_s": round(time.time() - t1, 1), "gpu": torch.cuda.get_device_name(0)}); print(log[-1], flush=True)
json.dump(log, open(f"{A}/logs/train{SFX}_{DS}_{MODEL}_{ARM}_f{F}.json", "w"), indent=1); print("unit complete", flush=True)
