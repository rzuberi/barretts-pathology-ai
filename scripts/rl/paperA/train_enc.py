"""Paper A encoder training: explore/clf/train_clf.py (BioClinical-ModernBERT-base, same recipe) generalised to
  FOLD   (-1 = all train patients; k = drop train rows of fold-k patients, see make_folds.py)
  TEACHER jury (train.jsonl vf>=0.75 labels) | juror1 | top3 | soft (labels from O/labels/*.json over train + all_lowconf rows)
  SAVE=1 (write best weights to O/runs/TAG/model.pt). Env also: SEED, TAG, EPOCHS (3), LR (3e-5), BS (16), N (smoke cap), EVAL_EVERY (0.5).
Checkpoint chosen on val.jsonl exact agreement with the jury, every half epoch. Predicts val, test, test485, pop -> O/preds/TAG__set.jsonl."""
import json, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
os.environ["HF_HOME"] = "/mnt/scratche/slow/fmlab/zuberi01/hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
from pa_common import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
SEED = int(os.environ.get("SEED", 0)); TAG = os.environ["TAG"]; FOLD = int(os.environ.get("FOLD", -1)); TEACHER = os.environ.get("TEACHER", "jury"); SAVE = os.environ.get("SAVE") == "1"
EP = float(os.environ.get("EPOCHS", 3)); LR = float(os.environ.get("LR", 3e-5)); BS = int(os.environ.get("BS", 16)); N = int(os.environ.get("N", 0)); EVAL_EVERY = float(os.environ.get("EVAL_EVERY", 0.5))
P = mkdir(O + "/preds"); R = mkdir(f"{O}/runs/{TAG}")
if all(os.path.exists(f"{P}/{TAG}__{s}.jsonl") for s in ["test", "test485", "pop"]) and not (SAVE and not os.path.exists(R + "/model.pt")): print("done already"); sys.exit(0)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); dev = "cuda"
SETS = ["train", "val", "test", "test485", "pop"]; Tk = {s: torch.load(f"{CLF}/tok/mbb__{s}.pt") for s in SETS}
meta = {s: {r["id"]: r for r in rows(s)} for s in ["train"]}
if TEACHER != "jury":
    lc = torch.load(O + "/tok/mbb__all_lowconf.pt"); meta["train"].update({r["id"]: r for r in rows("all_lowconf")})
    for k in ("ids", "input_ids", "label"): Tk["train"][k] = Tk["train"][k] + lc[k]
    lab = json.load(open(f"{O}/labels/{TEACHER}.json"))
    keep = [i for i, x in enumerate(Tk["train"]["ids"]) if x in lab]
    Tk["train"] = {"ids": [Tk["train"]["ids"][i] for i in keep], "input_ids": [Tk["train"]["input_ids"][i] for i in keep],
                   "label": [(lab[Tk["train"]["ids"][i]] if TEACHER == "soft" else ORD[lab[Tk["train"]["ids"][i]]]) for i in keep]}
if FOLD >= 0:
    fold = json.load(open(O + "/folds.json"))["fold"]; keep = [i for i, x in enumerate(Tk["train"]["ids"]) if fold[meta["train"][x]["anon_id"]] != FOLD]
    Tk["train"] = {k: [v[i] for i in keep] for k, v in Tk["train"].items()}
if N:
    for s in SETS: Tk[s] = {k: v[:N] for k, v in Tk[s].items()}
ntrain = len(Tk["train"]["ids"]); print(f"{TAG}: teacher {TEACHER} fold {FOLD} seed {SEED}: {ntrain} training rows", flush=True)
pad = AutoTokenizer.from_pretrained(MODEL).pad_token_id; SOFT = TEACHER == "soft"
def batches(d, bs, shuffle, rng=None):
    idx = list(range(len(d["input_ids"])))
    if shuffle: rng.shuffle(idx)
    else: idx.sort(key=lambda i: len(d["input_ids"][i]))
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]; L = max(len(d["input_ids"][j]) for j in b)
        ids = torch.full((len(b), L), pad, dtype=torch.long); am = torch.zeros((len(b), L), dtype=torch.long)
        for k, j in enumerate(b): x = d["input_ids"][j]; ids[k, :len(x)] = torch.tensor(x); am[k, :len(x)] = 1
        y = torch.tensor([d["label"][j] for j in b], device=dev, dtype=torch.float32 if (SOFT and shuffle) else torch.long)
        yield b, ids.to(dev), am.to(dev), y
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=6, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev)
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
nb = math.ceil(ntrain / BS); total = int(nb * EP); warm = int(0.06 * total); sch = get_linear_schedule_with_warmup(opt, warm, total)
eval_at = max(1, int(nb * EVAL_EVERY)); step = 0; best = (-1, None); t0 = time.time()
log = {"tag": TAG, "model": MODEL, "params": nparam, "lr": LR, "bs": BS, "epochs": EP, "seed": SEED, "fold": FOLD, "teacher": TEACHER, "n_train": ntrain, "evals": []}
def do_eval():
    global best
    pr, _ = predict(Tk["val"]); acc = float((pr.argmax(1) == np.array(Tk["val"]["label"])).mean())
    log["evals"].append({"step": step, "epoch": round(step / nb, 2), "val_exact": round(acc, 4), "minutes": round((time.time() - t0) / 60, 1)}); print(log["evals"][-1], flush=True)
    if acc > best[0]: best = (acc, {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
    model.train()
model.train(); done = False
while not done:
    for b, ids, am, y in batches(Tk["train"], BS, True, rng):
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(input_ids=ids, attention_mask=am).logits.float()
        loss = F.cross_entropy(lg, y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); step += 1
        if step % 50 == 0: print(f"step {step}/{total} loss {loss.item():.3f} {(time.time()-t0)/60:.1f} min", flush=True)
        if step % eval_at == 0 or step == total: do_eval()
        if step >= total: done = True; break
log["best_val_exact"] = best[0]; log["train_minutes"] = round((time.time() - t0) / 60, 1); model.load_state_dict(best[1])
if SAVE: torch.save(best[1], R + "/model.pt"); os.chmod(R + "/model.pt", 0o600)
for s in ["val", "test", "test485", "pop"]:
    pr, dt = predict(Tk[s]); rs = rows(s); ids = Tk[s]["ids"]; assert [r["id"] for r in rs][:len(ids)] == ids
    with open(f"{P}/{TAG}__{s}.jsonl", "w") as f:
        for r, p in zip(rs, pr):
            d = {k: v for k, v in r.items() if k != "report"}; g = GRADES[int(p.argmax())]; d.update({"pred": g, "p_grade": float(p.max()), "probs": [round(float(x), 5) for x in p]}); f.write(json.dumps(d) + "\n")
    os.chmod(f"{P}/{TAG}__{s}.jsonl", 0o600)
    nrep = len({r["report_id"] for r in rs[:len(ids)]}); json.dump({"tag": TAG, "set": s, "n": len(ids), "reports": nrep, "seconds": dt, "params": nparam}, open(f"{P}/{TAG}__{s}.timing.json", "w"))
    key = "human" if s == "test485" else "label"; acc = float(np.mean([GRADES[int(p.argmax())] == r[key] for r, p in zip(rs, pr)]))
    log[f"{s}_exact_unweighted"] = round(acc, 4); print(f"{TAG} {s}: n={len(ids)} exact(unw)={acc:.4f} {dt:.1f}s", flush=True)
json.dump(log, open(f"{R}/log.json", "w"), indent=1); print("TRAIN DONE", flush=True)
