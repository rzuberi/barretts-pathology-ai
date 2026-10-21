"""Fine-tune one encoder as a 6-way section grader. Env: KEY (mbb/mbl/cbert), SEED, EPOCHS (3), LR, BS, N (smoke cap), TAG.
Selects on validation exact agreement with the jury (evaluated every half epoch); then predicts test, test485, pop with the best weights.
Out: O/preds/<TAG>__<set>.jsonl (pred, p_grade = max softmax, probs) + timing json + O/runs/<TAG>/log.json"""
import json, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
os.environ["HF_HOME"] = "/mnt/scratche/slow/fmlab/zuberi01/hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
from clf_common import *
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
KEY = os.environ["KEY"]; SEED = int(os.environ.get("SEED", 0)); EP = float(os.environ.get("EPOCHS", 3)); N = int(os.environ.get("N", 0))
LR = float(os.environ.get("LR", 3e-5)); BS = int(os.environ.get("BS", 16)); TAG = os.environ.get("TAG", f"{KEY}_s{SEED}"); EVAL_EVERY = float(os.environ.get("EVAL_EVERY", 0.5))
name, maxlen = MODELS[KEY]; P = O + "/preds"; R = f"{O}/runs/{TAG}"; os.makedirs(P, exist_ok=True); os.makedirs(R, exist_ok=True)
if all(os.path.exists(f"{P}/{TAG}__{s}.jsonl") for s in ["test", "test485", "pop"]): print("done already"); sys.exit(0)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); dev = "cuda"
T = {s: torch.load(f"{O}/tok/{KEY}__{s}.pt") for s in SETS}
if N:
    for s in SETS: T[s] = {k: v[:N] for k, v in T[s].items()}
pad = 0 if KEY == "cbert" else 50283   # [PAD] for BERT vocab; ModernBERT tokenizer pad id
from transformers import AutoTokenizer; pad = AutoTokenizer.from_pretrained(name).pad_token_id
def batches(d, bs, shuffle, rng=None):
    idx = list(range(len(d["input_ids"])))
    if shuffle: rng.shuffle(idx)
    else: idx.sort(key=lambda i: len(d["input_ids"][i]))   # length-sorted for fast inference
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]; L = max(len(d["input_ids"][j]) for j in b)
        ids = torch.full((len(b), L), pad, dtype=torch.long); am = torch.zeros((len(b), L), dtype=torch.long)
        for k, j in enumerate(b): x = d["input_ids"][j]; ids[k, :len(x)] = torch.tensor(x); am[k, :len(x)] = 1
        yield b, ids.to(dev), am.to(dev), torch.tensor([d["label"][j] for j in b], device=dev)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=6, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev)
nparam = sum(p.numel() for p in model.parameters())
@torch.no_grad()
def predict(d, bs=64):
    model.eval(); out = [None] * len(d["input_ids"]); t0 = time.time()
    for b, ids, am, _ in batches(d, bs, False):
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(input_ids=ids, attention_mask=am).logits.float()
        pr = F.softmax(lg, -1).cpu().numpy()
        for k, j in enumerate(b): out[j] = pr[k]
    torch.cuda.synchronize(); return np.stack(out), time.time() - t0
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01); rng = random.Random(SEED)
nb = math.ceil(len(T["train"]["input_ids"]) / BS); total = int(nb * EP); warm = int(0.06 * total); sch = get_linear_schedule_with_warmup(opt, warm, total)
eval_at = max(1, int(nb * EVAL_EVERY)); step = 0; best = (-1, None); log = {"tag": TAG, "model": name, "params": nparam, "lr": LR, "bs": BS, "epochs": EP, "seed": SEED, "evals": []}; t0 = time.time()
def do_eval():
    global best
    pr, _ = predict(T["val"]); acc = float((pr.argmax(1) == np.array(T["val"]["label"])).mean())
    log["evals"].append({"step": step, "epoch": round(step / nb, 2), "val_exact": round(acc, 4), "minutes": round((time.time() - t0) / 60, 1)}); print(log["evals"][-1], flush=True)
    if acc > best[0]: best = (acc, {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
    model.train()
model.train(); done = False
while not done:
    for b, ids, am, y in batches(T["train"], BS, True, rng):
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(input_ids=ids, attention_mask=am).logits.float()
        loss = F.cross_entropy(lg, y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); step += 1
        if step % 50 == 0: print(f"step {step}/{total} loss {loss.item():.3f} {(time.time()-t0)/60:.1f} min", flush=True)
        if step % eval_at == 0 or step == total: do_eval()
        if step >= total: done = True; break
log["best_val_exact"] = best[0]; log["train_minutes"] = round((time.time() - t0) / 60, 1); model.load_state_dict(best[1])
for s in ["val", "test", "test485", "pop"]:
    pr, dt = predict(T[s]); rs = rows(s); ids = T[s]["ids"]; assert [r["id"] for r in rs][:len(ids)] == ids
    with open(f"{P}/{TAG}__{s}.jsonl", "w") as f:
        for r, p in zip(rs, pr):
            d = {k: v for k, v in r.items() if k != "report"}; g = GRADES[int(p.argmax())]; d.update({"pred": g, "p_grade": float(p.max()), "probs": [round(float(x), 5) for x in p]}); f.write(json.dumps(d) + "\n")
    nrep = len({r["report_id"] for r in rs[:len(ids)]}); json.dump({"tag": TAG, "set": s, "n": len(ids), "reports": nrep, "seconds": dt, "params": nparam}, open(f"{P}/{TAG}__{s}.timing.json", "w"))
    key = "human" if s == "test485" else "label"; acc = float(np.mean([GRADES[int(p.argmax())] == r[key] for r, p in zip(rs, pr)]))
    log[f"{s}_exact_unweighted"] = round(acc, 4); log[f"{s}_s_per_report"] = round(dt / max(nrep, 1), 4); print(f"{TAG} {s}: n={len(ids)} exact(unw)={acc:.4f} {dt:.1f}s = {dt/max(nrep,1):.3f} s/report", flush=True)
json.dump(log, open(f"{R}/log.json", "w"), indent=1); print("TRAIN DONE", flush=True)
