"""Shared ABMIL + Cox machinery for OCCAMS v3 (1.1) and TCGA ABMIL fusion (1.2).

Gated-attention MIL over tile features with a Cox partial-likelihood head, plus a
linear-Cox trainer for tabular arms, Harrell's C, and patient-level bootstrap.
CPU-friendly (bags capped, small model). Protocol: patient-disjoint folds,
event-stratified; OOF risk scores evaluated with Harrell's C over all pairs.
"""
import numpy as np
import torch
import torch.nn as nn

MAX_TILES = 800
RNG = np.random.RandomState


class ABMIL(nn.Module):
    def __init__(self, d_in=1536, d=384, d_attn=128, p_drop=0.25):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.attn_v = nn.Sequential(nn.Linear(d, d_attn), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(d, d_attn), nn.Sigmoid())
        self.attn_w = nn.Linear(d_attn, 1)
        self.head = nn.Linear(d, 1)

    def forward(self, bag):                     # bag: (n_tiles, d_in)
        h = self.emb(bag)
        a = self.attn_w(self.attn_v(h) * self.attn_u(h)).softmax(dim=0)  # (n,1)
        z = (a * h).sum(dim=0)
        return self.head(z).squeeze(), a.squeeze()


class LinearCox(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)

    def forward(self, X):
        return self.lin(X).squeeze(-1)


def cox_loss(risk, time, event):
    """Breslow negative partial log-likelihood over a batch."""
    order = torch.argsort(time, descending=True)   # risk sets by decreasing time
    risk, event = risk[order], event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    ll = ((risk - log_cumsum) * event).sum()
    return -ll / event.sum().clamp(min=1)


def cindex(risk, time, event):
    """Harrell's C: higher risk should mean shorter survival."""
    n, conc, comp = len(risk), 0.0, 0
    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if time[j] > time[i]:
                comp += 1
                if risk[i] > risk[j]: conc += 1
                elif risk[i] == risk[j]: conc += 0.5
    return conc / comp if comp else float("nan")


def _bag(bags, key, rng):
    F = bags[key]
    if len(F) > MAX_TILES:
        F = F[rng.choice(len(F), MAX_TILES, replace=False)]
    return torch.tensor(F, dtype=torch.float32)


def train_abmil_fold(bags, keys_tr, keys_te, time, event, seed, epochs=30,
                     lr=1e-4, mb=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = RNG(seed)
    torch.manual_seed(seed)
    model = ABMIL(d_in=next(iter(bags.values())).shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    t = {k: time[k] for k in keys_tr}; e = {k: event[k] for k in keys_tr}
    for ep in range(epochs):
        order = rng.permutation(keys_tr)
        for i in range(0, len(order), mb):
            chunk = order[i:i + mb]
            if sum(e[k] for k in chunk) == 0:
                continue
            risks = torch.stack([model(_bag(bags, k, rng).to(device))[0] for k in chunk])
            loss = cox_loss(risks,
                            torch.tensor([t[k] for k in chunk], dtype=torch.float32, device=device),
                            torch.tensor([float(e[k]) for k in chunk], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        return {k: float(model(_bag(bags, k, RNG(0)).to(device))[0]) for k in keys_te}


def train_linear_cox_fold(X, keys_tr, keys_te, time, event, seed, epochs=400, lr=5e-3):
    torch.manual_seed(seed)
    idx = {k: i for i, k in enumerate(X["keys"])}
    Xt = torch.tensor(X["X"], dtype=torch.float32)
    mu, sd = Xt[[idx[k] for k in keys_tr]].mean(0), Xt[[idx[k] for k in keys_tr]].std(0).clamp(min=1e-6)
    Xt = (Xt - mu) / sd
    model = LinearCox(Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    tr = [idx[k] for k in keys_tr]
    tt = torch.tensor([time[k] for k in keys_tr], dtype=torch.float32)
    ee = torch.tensor([float(event[k]) for k in keys_tr])
    for _ in range(epochs):
        loss = cox_loss(model(Xt[tr]), tt, ee)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        return {k: float(model(Xt[idx[k]].unsqueeze(0))) for k in keys_te}


def zscore_oof(oof):
    v = np.array(list(oof.values()))
    mu, sd = v.mean(), v.std() or 1.0
    return {k: (x - mu) / sd for k, x in oof.items()}


def bootstrap_c(risk_a, time, event, n_boot=1000, risk_b=None, seed=0):
    """CI for C-index of arm a, and for delta (a-b) if b given."""
    keys = list(risk_a)
    ra = np.array([risk_a[k] for k in keys])
    tt = np.array([time[k] for k in keys]); ee = np.array([event[k] for k in keys])
    rb = np.array([risk_b[k] for k in keys]) if risk_b else None
    rng, cs, ds = RNG(seed), [], []
    for _ in range(n_boot):
        b = rng.randint(0, len(keys), len(keys))
        if ee[b].sum() < 3: continue
        c = cindex(ra[b], tt[b], ee[b])
        cs.append(c)
        if rb is not None:
            ds.append(c - cindex(rb[b], tt[b], ee[b]))
    out = {"c": round(float(cindex(ra, tt, ee)), 4),
           "c_ci": [round(float(np.percentile(cs, q)), 4) for q in (2.5, 97.5)]}
    if rb is not None:
        out["delta_ci"] = [round(float(np.percentile(ds, q)), 4) for q in (2.5, 97.5)]
        out["delta_mean"] = round(float(np.mean(ds)), 4)
    return out


def stratified_folds(keys, event, n_folds=5, seed=0):
    rng = RNG(seed)
    pos = [k for k in keys if event[k]]; neg = [k for k in keys if not event[k]]
    rng.shuffle(pos); rng.shuffle(neg)
    folds = [[] for _ in range(n_folds)]
    for i, k in enumerate(pos + neg):
        folds[i % n_folds].append(k)
    return folds
