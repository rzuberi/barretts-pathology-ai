import os
"""Shared ABMIL binary-classification machinery for Ch3/Ch4 (mirrors abmil_cox)."""
import numpy as np
import torch
import torch.nn as nn
from abmil_cox import ABMIL, MAX_TILES, RNG


def _bag(bags, key, rng):
    F = bags[key]
    if len(F) > MAX_TILES:
        F = F[rng.choice(len(F), MAX_TILES, replace=False)]
    return torch.tensor(F, dtype=torch.float32)


def train_abmil_clf_fold(bags, keys_tr, keys_te, y, seed, epochs=25, lr=1e-4,
                         mb=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = RNG(seed); torch.manual_seed(seed)
    model = ABMIL(d_in=next(iter(bags.values())).shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pos = sum(y[k] for k in keys_tr); w = (len(keys_tr) - pos) / max(pos, 1)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(w), device=device))
    for ep in range(epochs):
        order = rng.permutation(keys_tr)
        for i in range(0, len(order), mb):
            chunk = order[i:i + mb]
            logits = torch.stack([model(_bag(bags, k, rng).to(device))[0] for k in chunk])
            loss = lossf(logits, torch.tensor([float(y[k]) for k in chunk], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    out = {}
    attn_store = {}
    with torch.inference_mode():
        for k in keys_te:
            logit, attn = model(_bag(bags, k, RNG(0)).to(device))
            out[k] = float(torch.sigmoid(logit))
            if os.environ.get("SAVE_INTERP"):
                a = attn.cpu().numpy()
                top = a.argsort()[-10:][::-1]
                attn_store[k] = {"top_tile_idx": top.tolist(),
                                 "top_tile_attn": a[top].round(5).tolist()}
    if attn_store:
        import json as _json
        d = os.environ.get("OUTDIR", ".")
        fn = os.path.join(d, f"attention_seed{seed}.jsonl")
        with open(fn, "a") as fh:
            for k, v in attn_store.items():
                fh.write(_json.dumps({"key": k, **v}) + "\n")
    return out


def bootstrap_auc(prob_a, y, n_boot=2000, prob_b=None, seed=0):
    from sklearn.metrics import roc_auc_score
    keys = list(prob_a)
    pa = np.array([prob_a[k] for k in keys]); yy = np.array([y[k] for k in keys])
    pb = np.array([prob_b[k] for k in keys]) if prob_b else None
    rng, aucs, deltas = RNG(seed), [], []
    for _ in range(n_boot):
        b = rng.randint(0, len(keys), len(keys))
        if len(set(yy[b])) < 2: continue
        a = roc_auc_score(yy[b], pa[b]); aucs.append(a)
        if pb is not None: deltas.append(a - roc_auc_score(yy[b], pb[b]))
    out = {"auc": round(float(roc_auc_score(yy, pa)), 4),
           "auc_ci": [round(float(np.percentile(aucs, q)), 4) for q in (2.5, 97.5)]}
    if pb is not None:
        out["delta_vs_ref"] = {"mean": round(float(np.mean(deltas)), 4),
                               "ci": [round(float(np.percentile(deltas, q)), 4) for q in (2.5, 97.5)]}
    return out


def patient_folds(keys, patient_of, y, n_folds=5, seed=0):
    """Patient-disjoint, event-stratified folds over bag keys."""
    rng = RNG(seed)
    pats = {}
    for k in keys: pats.setdefault(patient_of[k], []).append(k)
    ppos = [(p, max(y[k] for k in ks)) for p, ks in pats.items()]
    pos = [p for p, v in ppos if v]; neg = [p for p, v in ppos if not v]
    rng.shuffle(pos); rng.shuffle(neg)
    folds = [[] for _ in range(n_folds)]
    for i, p in enumerate(pos + neg): folds[i % n_folds].extend(pats[p])
    return folds
