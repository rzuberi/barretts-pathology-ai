"""Probe strand: hidden states of the project-1 RL model at the grade position. One forward pass per section, no generation:
prompt (project-1 make_messages) + assistant turn + forced prefix '{"grade": "'. The last prefix token's hidden state is stored
(layers in LAYERS, fp16 .npy) with per-section meta (no report text): forced-choice probabilities over the six grade first-tokens,
unrestricted top-1 probability and the forced argmax grade.
Env: SETS (comma list), N (cap, smoke), OUTD (default rl_explore/probe/feats), LAYERS (default 17,26,34), BS (default 8), MODEL."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); from common import *
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL = os.environ.get("MODEL", B + "/analysis/rl_grader/runs/p1_best_merged"); D = B + "/analysis/rl_grader/data"
OUTD = os.environ.get("OUTD", B + "/analysis/rl_explore/probe/feats"); os.makedirs(OUTD, exist_ok=True)
SETS = os.environ.get("SETS", "val").split(","); N = int(os.environ.get("N", 0)); LAYERS = [int(x) for x in os.environ.get("LAYERS", "17,26,34").split(",")]; BS = int(os.environ.get("BS", 8))
PREFIX = '{"grade": "'
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
# first token of each grade word in context (after the forced prefix); must be distinct
pre0 = tok.apply_chat_template(make_messages("x", "A"), add_generation_prompt=True, tokenize=False) + PREFIX
b0 = tok(pre0, add_special_tokens=False).input_ids; GID = []
for g in GRADES:
    ids = tok(pre0 + g + '"}', add_special_tokens=False).input_ids; assert ids[:len(b0)] == b0, g; GID.append(ids[len(b0)])
assert len(set(GID)) == len(GRADES), ("grade first tokens not distinct", GID); GID_T = torch.tensor(GID).cuda()
print("grade first tokens", dict(zip(GRADES, [tok.decode([t]) for t in GID])), flush=True)
def encode(r):
    s = tok.apply_chat_template(make_messages(r["report"], r["section"]), add_generation_prompt=True, tokenize=False) + PREFIX
    ids = tok(s, add_special_tokens=False).input_ids
    return ids[-3072:] if len(ids) > 3072 else ids   # keep the tail: the task text and prefix are at the end
for s in SETS:
    done = f"{OUTD}/{s}__meta.jsonl"
    if os.path.exists(done): print(s, "done already"); continue
    rows = [json.loads(l) for l in open(f"{D}/{s}.jsonl")]
    if N: rows = rows[:N]
    enc = [encode(r) for r in rows]; order = np.argsort([-len(e) for e in enc]); t0 = time.time()
    H = {L: np.zeros((len(rows), model.config.get_text_config().hidden_size), np.float16) for L in LAYERS}
    PF = np.zeros((len(rows), len(GRADES)), np.float32); PT = np.zeros(len(rows), np.float32); TOP = [None] * len(rows)
    with torch.no_grad():
        for b in range(0, len(rows), BS):
            ix = order[b:b + BS]; L = max(len(enc[i]) for i in ix)
            inp = torch.full((len(ix), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(ix), L), dtype=torch.long)
            for k, i in enumerate(ix): inp[k, :len(enc[i])] = torch.tensor(enc[i]); att[k, :len(enc[i])] = 1
            out = model(input_ids=inp.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
            last = att.sum(1) - 1; ar = torch.arange(len(ix))
            for Ly in LAYERS: H[Ly][ix] = out.hidden_states[Ly][ar, last].float().cpu().numpy().astype(np.float16)
            lg = out.logits[ar, last].float(); p = torch.softmax(lg, -1)
            PF[ix] = torch.softmax(lg[:, GID_T], -1).cpu().numpy(); top = p.max(-1); PT[ix] = top.values.cpu().numpy()
            for k, i in enumerate(ix): TOP[i] = tok.decode([int(top.indices[k])])
            if (b // BS) % 50 == 0: print(f"{s} {b}/{len(rows)} {time.time() - t0:.0f}s maxlen {L}", flush=True)
    for Ly in LAYERS: np.save(f"{OUTD}/{s}__L{Ly}.npy", H[Ly])
    with open(done + ".part", "w") as f:
        for i, r in enumerate(rows):
            d = {k: r[k] for k in r if k != "report"}; d.update({"ntok": len(enc[i]), "forced_probs": [round(float(x), 6) for x in PF[i]], "forced_pred": GRADES[int(PF[i].argmax())],
                                                              "p_top1": round(float(PT[i]), 6), "top1_tok": TOP[i]})
            f.write(json.dumps(d) + "\n")
    os.replace(done + ".part", done)
    print(f"{s}: {len(rows)} sections in {time.time() - t0:.0f}s; forced pred == jury label {np.mean([GRADES[int(PF[i].argmax())] == r['label'] for i, r in enumerate(rows)]):.3f}; top1 is a grade token {np.mean([t in [tok.decode([g]) for g in GID] for t in TOP]):.3f}", flush=True)
print("EXTRACT DONE", flush=True)
