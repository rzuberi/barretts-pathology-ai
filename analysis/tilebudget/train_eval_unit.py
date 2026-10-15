"""P2 (GPU unit): fold F x training config CFG -> train 3 seeds x 2 tasks (unless CFG=rk1200, which reuses the reference arm-B
models) and evaluate every model on every (test rule x budget m x aggregator). Env: FOLD, CFG.
CFG: rk4|rk16|rk64|rk256 (fixed random k tiles from U1200, rand_u_0 prefix), rk1200 (reference: 800 resampled per step from U1200),
     ak16|ak64|ak256 (two-stage: attention-top k from attn_top_u), all (800 resampled per step from ALL tiles; pack_all in host RAM).
Test rules (U1200): rand_u (mean of 3 draws), grid_u, attn_top_u, attn_bot_u, mixed_u, conch_u, diverse_u at m in {1,4,16,64,256,1200};
ALL rules (only rk1200 and all units): all_tiles, attn_top_all, grid_all (m as above; all_tiles = every tile).
Aggregators: abmil (the trained model), meanlr (logistic head on mean-pooled tile features, trained on the same training bags),
max (max over tiles of the ABMIL instance-level logits). Output: preds/fold{F}_{CFG}_{task}_seed{s}.npz with keys rule|m|agg -> probs."""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
B = "/mnt/scratche/slow/fmlab/zuberi01"; ISBI = B + "/analysis/isbi_rerun"; A = B + "/analysis/tilebudget"; PA = A + "/pack_all"
F = int(os.environ["FOLD"]); CFG = os.environ["CFG"]; dev = "cuda"; CAP = 1200; D = 1536
os.makedirs(A + "/preds", exist_ok=True); os.makedirs(A + "/models", exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
HP = {"epochs": 25, "lr": 1e-4, "weight_decay": 1e-5, "mb": 32, "tiles_per_step": 800, "seeds": [0, 1, 2]}
CLS = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLS)}
MS = [1, 4, 16, 64, 256, 1200]; U_RULES = ["rand_u", "grid_u", "attn_top_u", "attn_bot_u", "mixed_u", "conch_u", "diverse_u"]; ALL_RULES = ["all_tiles", "attn_top_all", "grid_all"]
NEED_ALL = CFG in ("all", "rk1200")
tags = [f"fold{F}_{CFG}_{t}_seed{sd}" for t in ("bin", "six") for sd in HP["seeds"]]
if all(os.path.exists(f"{A}/preds/{t}.npz") for t in tags): print("done already"); sys.exit(0)
# ---------------- data ----------------
s = pd.read_csv(ISBI + "/slides.csv", dtype={"idx": int}).fillna(""); excl = set(pd.read_csv(ISBI + "/qc/exclusions.csv").idx); s = s[~s.idx.isin(excl)].reset_index(drop=True)
s["ok"] = (s.worst_grade != ""); _z = np.load(f"{A}/sel_fold{F}.npz"); sel = {k: _z[k] for k in _z.files}; _z.close(); assert (sel["idx"] == s.idx.values).all()   # NpzFile re-reads on every access: materialise once
pos_of = {int(i): j for j, i in enumerate(s.idx.values)}
# U1200 resident on GPU (ISBI pack; tile order = sorted subset indices, i.e. position p in the pack row == sub[p])
idxf = json.load(open(ISBI + "/pack/index.json")); NP = len(idxf["n_tiles"]); t0 = time.time()
XU = torch.empty((NP, CAP, D), dtype=torch.float16, device=dev)
for c in idxf["chunks"]:
    arr = np.load(ISBI + "/pack/" + c["file"], mmap_mode="r")
    for a in range(0, c["hi"] - c["lo"], 256): e = min(a + 256, c["hi"] - c["lo"]); XU[c["lo"] + a:c["lo"] + e] = torch.from_numpy(np.ascontiguousarray(arr[a:e])).to(dev)
NTU = np.array(idxf["n_tiles"]); print(f"U1200 pack on GPU in {time.time() - t0:.0f}s", flush=True)
# map h5 tile index -> row position inside the U1200 pack (the pack holds the sorted subset)
posmap = {}   # h5 tile index -> row position in the U1200 pack (vectorised lookups; -1 = not in the subset)
for j, idx in enumerate(s.idx.values):
    n = int(sel["n_all"][j]); sub = np.sort(np.random.RandomState(int(idx)).choice(n, CAP, replace=False)) if n > CAP else np.arange(n)
    pm = np.full(n, -1, np.int64); pm[sub] = np.arange(len(sub)); posmap[int(idx)] = pm
XA = None; span = None
if NEED_ALL:
    ix = json.load(open(PA + "/index_all.json")); chunks = ix["chunks"]; g = 0
    for c in chunks: c["goff"] = g; g += c["n_tiles"]
    t0 = time.time(); XA = np.empty((g, D), np.float16)
    for c in chunks: arr = np.load(f"{PA}/{c['file']}"); XA[c["goff"]:c["goff"] + c["n_tiles"]] = arr
    def span(idx): e = ix["slides"][str(idx)]; c = chunks[e["chunk"]]; st = c["goff"] + e["off"]; return st, st + e["n"]
    print(f"pack_all in host RAM: {XA.nbytes / 1e9:.1f} GB in {time.time() - t0:.0f}s", flush=True)
# ---------------- model ----------------
class ABMIL(nn.Module):
    def __init__(self, d_in=1536, d=384, d_attn=128, p_drop=0.25, n_out=1):
        super().__init__(); self.emb = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.attn_v = nn.Sequential(nn.Linear(d, d_attn), nn.Tanh()); self.attn_u = nn.Sequential(nn.Linear(d, d_attn), nn.Sigmoid()); self.attn_w = nn.Linear(d_attn, 1); self.head = nn.Linear(d, n_out)
    def forward(self, bag, mask, inst=False):
        h = self.emb(bag) * mask.unsqueeze(-1); a = self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(1)
        out = self.head((a.unsqueeze(-1) * h).sum(1))
        return (out, self.head(h)) if inst else out
def bags_from_rows(rows_list):
    """rows_list: list of fp16 tensors (n_i, D) on GPU -> padded (B, kmax, D) + mask"""
    k = max(1, max(len(r) for r in rows_list)); out = torch.zeros((len(rows_list), k, D), dtype=torch.float16, device=dev); mask = torch.zeros((len(rows_list), k), dtype=torch.bool, device=dev)
    for b, r in enumerate(rows_list): out[b, :len(r)] = r; mask[b, :len(r)] = True
    return out, mask
def rows_u(idx, tiles):
    """rows of the U1200 pack for h5 tile indices `tiles` (must be in the subset)"""
    p = posmap[idx][np.asarray(tiles, np.int64)]; assert (p >= 0).all(); return XU[idx, torch.from_numpy(p).to(dev)]   # pack row == slide idx (pack built over the unfiltered slides.csv)
def rows_all(idx, tiles):
    st, en = span(idx); return torch.from_numpy(np.ascontiguousarray(XA[st + np.asarray(tiles, np.int64)])).to(dev)
def train_bag(idx, g, k_fixed=None, mode="u"):
    if mode == "all":
        st, en = span(idx); n = en - st; sel_ = np.sort(np.random.RandomState(int(g.initial_seed() % 2**31) ^ idx ^ int(torch.randint(0, 2**30, (1,), generator=g))).choice(n, min(800, n), replace=False)); return rows_all(idx, sel_)
    if k_fixed is not None: return rows_u(idx, k_fixed)
    ni = int(NTU[idx]); kk = min(800, ni); selp = torch.randperm(ni, generator=g)[:kk].to(dev) if ni > 800 else torch.arange(kk, device=dev); return XU[idx, selp]
def labels(task, d):
    g = d.worst_grade
    if task == "bin": keep = g.isin(["NORMAL_OTHER", "NDBE", "LGD", "HGD", "CANCER"]); y = g.isin(["LGD", "HGD", "CANCER"]).astype(int)
    else: keep = g.isin(CLS); y = g.map(C_OF).fillna(-1).astype(int)
    return keep.values, y.values
fixed = {}
if CFG.startswith("rk") and CFG != "rk1200": k = int(CFG[2:]); fixed = {int(i): sel["rand_u_0"][j, :k][sel["rand_u_0"][j, :k] >= 0] for j, i in enumerate(s.idx.values)}
if CFG.startswith("ak"): k = int(CFG[2:]); fixed = {int(i): sel["attn_top_u"][j, :k][sel["attn_top_u"][j, :k] >= 0] for j, i in enumerate(s.idx.values)}
def train(ids, y, task, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); rng = np.random.RandomState(seed)
    model = ABMIL(n_out=1 if task == "bin" else 6).to(dev); opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    yt = torch.tensor(y, device=dev, dtype=torch.float32 if task == "bin" else torch.long)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((len(y) - y.sum()) / max(y.sum(), 1)), device=dev)) if task == "bin" else nn.CrossEntropyLoss()
    model.train(); skipped = 0; last = float("nan")
    for ep in range(HP["epochs"]):
        order = rng.permutation(len(ids)); tot = 0.0; nb = 0
        for i in range(0, len(order), HP["mb"]):
            ch = order[i:i + HP["mb"]]
            bags, mask = bags_from_rows([train_bag(int(ids[c]), g, fixed.get(int(ids[c])) if fixed else None, "all" if CFG == "all" else "u") for c in ch])
            logit = model(bags.float(), mask); loss = lossf(logit.squeeze(-1), yt[ch]) if task == "bin" else lossf(logit, yt[ch])
            if not torch.isfinite(loss): skipped += 1; opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); tot += float(loss); nb += 1
        last = tot / max(nb, 1)
    return model, last, skipped
def meanlr_train(ids, y, task, seed):
    """logistic head on mean-pooled training bags (same tiles as the ABMIL training config; for resampling configs: the full U1200/ALL mean)"""
    torch.manual_seed(seed); feats = []
    with torch.no_grad():
        for i in ids:
            i = int(i)
            if CFG == "all": st, en = span(i); f = torch.from_numpy(XA[st:en].astype(np.float32)).to(dev).mean(0)
            elif fixed: f = rows_u(i, fixed[i]).float().mean(0)
            else: f = XU[i, :int(NTU[i])].float().mean(0)
            feats.append(f)
    Xf = torch.stack(feats); mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Xf = (Xf - mu) / sd
    head = nn.Linear(D, 1 if task == "bin" else 6).to(dev); opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-3)
    yt = torch.tensor(y, device=dev, dtype=torch.float32 if task == "bin" else torch.long)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((len(y) - y.sum()) / max(y.sum(), 1)), device=dev)) if task == "bin" else nn.CrossEntropyLoss()
    for it in range(600):
        opt.zero_grad(); out = head(Xf); loss = lossf(out.squeeze(-1), yt) if task == "bin" else lossf(out, yt); loss.backward(); opt.step()
    return head, mu, sd
@torch.inference_mode()
def predict(model, head, mu, sd, ids, task, rule, m):
    """returns dict agg -> probs for the slides `ids` under test rule/budget"""
    model.eval(); outs = {"abmil": [], "meanlr": [], "max": []}
    draws = ["rand_u_0", "rand_u_1", "rand_u_2"] if rule == "rand_u" else [rule]
    for i0 in range(0, len(ids), 32):
        ch = [int(i) for i in ids[i0:i0 + 32]]; acc = {k_: 0 for k_ in outs}
        for dr in draws:
            rows = []
            for i in ch:
                j = pos_of[i]
                if dr == "all_tiles": st, en = span(i); rows.append(torch.from_numpy(np.ascontiguousarray(XA[st:en])).to(dev))
                elif dr in ("attn_top_all", "grid_all"): t_ = sel[dr][j]; t_ = t_[t_ >= 0][:m]; rows.append(rows_all(i, t_))
                else: t_ = sel[dr][j]; t_ = t_[t_ >= 0][:m]; rows.append(rows_u(i, t_))
            bags, mask = bags_from_rows(rows); logit, inst = model(bags.float(), mask, inst=True)
            fm = (bags.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1); lm = head((fm - mu) / sd)
            inst = inst.masked_fill(~mask.unsqueeze(-1), float("-inf")); lmax = inst.max(1).values
            for k_, lg in (("abmil", logit), ("meanlr", lm), ("max", lmax)):
                acc[k_] = acc[k_] + (torch.sigmoid(lg.squeeze(-1)) if task == "bin" else torch.softmax(lg, 1)).cpu().numpy() / len(draws)
        for k_ in outs: outs[k_].append(acc[k_])
    return {k_: np.concatenate(v) for k_, v in outs.items()}
# ---------------- run ----------------
d = s[s.ok]; te = d[d.fold == F]; ho = d[d.fold == -1]; log = []
for task in ("bin", "six"):
    keep, y = labels(task, d); trm = keep & (d.fold.values != F) & (d.fold.values >= 0); tr = d[trm]; ytr = y[trm]
    for sd_ in HP["seeds"]:
        tag = f"fold{F}_{CFG}_{task}_seed{sd_}"
        if os.path.exists(f"{A}/preds/{tag}.npz"): continue
        t1 = time.time()
        if CFG == "rk1200":
            model = ABMIL(n_out=1 if task == "bin" else 6).to(dev); model.load_state_dict(torch.load(f"{ISBI}/models/patient_B_{task}_seed{sd_}_fold{F}.pt", map_location=dev)); last, skipped = float("nan"), 0
        else:
            model, last, skipped = train(tr.idx.values, ytr, task, sd_); torch.save(model.state_dict(), f"{A}/models/{tag}.pt")
        head, mu, sd = meanlr_train(tr.idx.values, ytr, task, sd_)
        out = {"test_idx": te.idx.values, "holdout_idx": ho.idx.values, "n_train": len(tr), "last_loss": last, "skipped": skipped}
        rules = [(r, m) for r in U_RULES for m in MS] + ([(r, m) for r in ALL_RULES[1:] for m in MS] + [("all_tiles", 0)] if NEED_ALL else [])
        for rule, m in rules:
            for setname, ids in (("test", te.idx.values), ("holdout", ho.idx.values)):
                P = predict(model, head, mu, sd, ids, task, rule, m)
                for agg, p in P.items(): out[f"{setname}|{rule}|{m}|{agg}"] = p.astype(np.float32)
        np.savez_compressed(f"{A}/preds/{tag}.part.npz", **out); os.replace(f"{A}/preds/{tag}.part.npz", f"{A}/preds/{tag}.npz")
        log.append({"tag": tag, "n_train": int(len(tr)), "last_loss": last, "skipped": skipped, "sec": round(time.time() - t1, 1)}); print(log[-1], flush=True)
json.dump(log, open(f"{A}/train_log_fold{F}_{CFG}.json", "w"), indent=1); print("unit complete", flush=True)
