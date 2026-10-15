"""PORPOISE step 2 (1.4): run the PUBLISHED architecture on our TCGA pool.

Uses their own model classes (PorpoiseMMF bilinear fusion; PorpoiseAMIL as the
published path-only baseline) and their discrete-survival nll_loss, on our
frozen patient-disjoint splits from step 1. Decisive readout: does the
published fusion beat the published unimodal on this data — i.e. can our
in-house fusion nulls be attributed to architecture?
"""
import json, os, sys
import numpy as np, pandas as pd
import torch

P = "/mnt/scratche/slow/fmlab/zuberi01/phd/mahmood_lab/PORPOISE"
D = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/runs/porpoise_data/output"
OUT = os.environ.get("OUTDIR", ".")
sys.path.insert(0, P)
from models.model_porpoise import PorpoiseMMF, PorpoiseAMIL  # noqa: E402

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """Verbatim from PORPOISE utils/utils.py (Chen et al. 2022, MIT licence) —
    vendored because their utils module imports torch_geometric, which the
    erin env does not carry and the loss does not need."""
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    return ((1 - alpha) * neg_l + alpha * uncensored_loss).mean()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1]
EPOCHS, N_BINS = 20, 4

df = pd.read_csv(os.path.join(D, "tcga_esca_stad_all_clean.csv"), low_memory=False)
omic_cols = [c for c in df.columns if c.endswith(("_rnaseq", "_cnv", "_mut"))]
X_omic = df[omic_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
time_m = df["survival_months"].values.astype(float)
cens = df["censorship"].values.astype(int)   # 1 = censored (their convention)
event = 1 - cens
qbins = np.quantile(time_m[event == 1], [0.25, 0.5, 0.75])
ybin = np.digitize(time_m, qbins)
fold_of = {}
for k in range(5):
    sp = pd.read_csv(os.path.join(D, "splits", f"splits_{k}.csv"))
    for c in sp["val"].dropna():
        if str(c).strip(): fold_of[str(c)] = k
fm = np.array([fold_of.get(c, -1) for c in df["case_id"]])
print(f"slides={len(df)} omic_dim={len(omic_cols)} events={event.sum()} folds ok={np.sum(fm >= 0)}", flush=True)

def load_pt(sid):
    return torch.load(os.path.join(D, "pt_files", sid + ".pt"), map_location="cpu")

def cindex(risk, t, e):
    n = d = 0
    for i in range(len(t)):
        if not e[i]: continue
        later = t > t[i]
        d += later.sum()
        n += (risk[later] < risk[i]).sum() + 0.5 * (risk[later] == risk[i]).sum()
    return n / d if d else 0.5

def run_arm(kind, seed):
    risks = np.zeros(len(df))
    for k in range(5):
        tr = np.where((fm != k) & (fm >= 0))[0]
        te = np.where(fm == k)[0]
        mu, sd = X_omic[tr].mean(0), X_omic[tr].std(0) + 1e-6
        Xo = (X_omic - mu) / sd
        torch.manual_seed(seed)
        if kind == "mmf":
            net = PorpoiseMMF(omic_input_dim=len(omic_cols), path_input_dim=1536,
                              fusion="bilinear", n_classes=N_BINS).to(DEV)
        else:
            net = PorpoiseAMIL(n_classes=N_BINS)
            # their AMIL hard-codes 1024-dim path input; swap the first Linear
            # for our 1536-dim features (weights re-init, architecture intact)
            for name, mod in net.named_children():
                if isinstance(mod, torch.nn.Sequential):
                    for j, layer in enumerate(mod):
                        if isinstance(layer, torch.nn.Linear) and layer.in_features == 1024:
                            mod[j] = torch.nn.Linear(1536, layer.out_features)
                            break
            net = net.to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-4, weight_decay=1e-5)
        for ep in range(EPOCHS):
            net.train()
            for i in np.random.RandomState(ep).permutation(tr):
                xp = load_pt(df.loc[i, "slide_id"]).to(DEV)
                kw = {"x_path": xp}
                if kind == "mmf":
                    kw["x_omic"] = torch.tensor(Xo[i][None]).to(DEV)
                out = net(**kw)
                logits = out[0] if isinstance(out, tuple) else out
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                loss = nll_loss(hazards=hazards, S=S,
                                Y=torch.tensor([[ybin[i]]]).to(DEV),
                                c=torch.tensor([[cens[i]]]).float().to(DEV))
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for i in te:
                xp = load_pt(df.loc[i, "slide_id"]).to(DEV)
                kw = {"x_path": xp}
                if kind == "mmf":
                    kw["x_omic"] = torch.tensor(Xo[i][None]).to(DEV)
                out = net(**kw)
                logits = out[0] if isinstance(out, tuple) else out
                S = torch.cumprod(1 - torch.sigmoid(logits), dim=1)
                risks[i] = float(-S.sum())
        print(f"{kind} seed{seed} fold{k} done", flush=True)
    return risks

res = {"_meta": {"n_slides": len(df), "omic_dim": len(omic_cols),
                 "events": int(event.sum()), "seeds": SEEDS, "bins": N_BINS,
                 "note": "published PORPOISE classes + nll_loss, frozen splits"}}
riskset = {}
for kind in ["amil", "mmf"]:
    per = [run_arm(kind, s) for s in SEEDS]
    r = np.mean(per, axis=0)
    riskset[kind] = r
    # patient-level: max risk per case
    pat = pd.DataFrame({"c": df["case_id"], "r": r, "t": time_m, "e": event}).groupby("c").agg(
        r=("r", "max"), t=("t", "first"), e=("e", "first"))
    res[kind] = {"cindex": round(float(cindex(pat["r"].values, pat["t"].values,
                                              pat["e"].values.astype(bool))), 4)}
    print(kind, res[kind], flush=True)
# paired bootstrap delta
pat = pd.DataFrame({"c": df["case_id"], "a": riskset["amil"], "m": riskset["mmf"],
                    "t": time_m, "e": event}).groupby("c").agg(
    a=("a", "max"), m=("m", "max"), t=("t", "first"), e=("e", "first"))
rng = np.random.RandomState(0); deltas = []
for b in range(500):
    idx = rng.randint(0, len(pat), len(pat))
    sub = pat.iloc[idx]
    if sub["e"].sum() < 10: continue
    deltas.append(cindex(sub["m"].values, sub["t"].values, sub["e"].values.astype(bool))
                  - cindex(sub["a"].values, sub["t"].values, sub["e"].values.astype(bool)))
res["mmf_minus_amil"] = {"delta_c": round(float(np.mean(deltas)), 4),
                         "ci": [round(float(np.percentile(deltas, 2.5)), 4),
                                round(float(np.percentile(deltas, 97.5)), 4)]}
print("delta:", res["mmf_minus_amil"], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
