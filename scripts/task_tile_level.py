"""2.47: TILE-LEVEL training (label propagation, Campanella-style) vs ABMIL on the 1,538 dual-labelled ERIN
slides. Every tile inherits its slide's label; an MLP tile classifier is trained; slide score = mean / max /
top-10% mean of tile probabilities. Same frozen patient folds and 3 seeds as 2.38/2.38b, both label schemes
(case-max, section), binary (LGD+) and six-class. Saves per-slide OOF scores for paired bootstrap against the
saved ABMIL unit predictions (feasibility/svc_units, svc5_units), and per-tile probabilities for the 12
display slides used by the tile-map job. Env: OUTDIR, EPOCHS (4), TILES_PER_SLIDE (2000)."""
import glob, json, os
import h5py, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", "."); DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", "4")); TPS = int(os.environ.get("TILES_PER_SLIDE", "2000")); SEEDS = [0, 1, 2]
CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLASSES)}; POS = {3, 4, 5}
lab = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str)
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
lab = lab.merge(m[["h5", "anon_id"]], on="h5"); lab = lab[lab.worst_grade.isin(C_OF) & lab.case_max.isin(C_OF)].reset_index(drop=True)
keys = sorted(lab.h5); lab = lab.set_index("h5").loc[keys]; pat = dict(zip(keys, lab.anon_id))
y6 = {"section": {k: C_OF[g] for k, g in zip(keys, lab.worst_grade)}, "case_max": {k: C_OF[g] for k, g in zip(keys, lab.case_max)}}
rng = np.random.RandomState(0); uniq = sorted(set(pat.values())); fold_of = {a: i % 5 for i, a in enumerate(rng.permutation(uniq))}
folds = [[k for k in keys if fold_of[pat[k]] == f] for f in range(5)]
print("loading bags", flush=True); bags = {}; rs = np.random.RandomState(0)
for k in keys:
    with h5py.File(k) as h: X = np.asarray(h["features"], np.float16)
    bags[k] = X[rs.choice(len(X), TPS, replace=False)] if len(X) > TPS else X
print(f"tiles total {sum(len(v) for v in bags.values()):,}", flush=True)
class TileMLP(nn.Module):
    def __init__(s, d=1536, n=6): super().__init__(); s.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, n))
    def forward(s, x): return s.net(x)
def train_fold(tr, ydict, seed):
    torch.manual_seed(seed); r = np.random.RandomState(seed)
    X = torch.tensor(np.concatenate([bags[k] for k in tr])); Y = torch.tensor(np.concatenate([[ydict[k]] * len(bags[k]) for k in tr]))
    cnt = np.bincount(Y.numpy(), minlength=6).astype(float); w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32, device=DEV)
    net = TileMLP().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4); n = len(X); B = 4096
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, B):
            idx = perm[i:i + B]; xb = X[idx].to(DEV).float(); yb = Y[idx].to(DEV)
            loss = nn.functional.cross_entropy(net(xb), yb, weight=w); opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()
def score(net, k):
    with torch.no_grad(): P = torch.softmax(net(torch.tensor(bags[k]).to(DEV).float()), -1).cpu().numpy()
    return P
res = {"_meta": {"n_slides": len(keys), "n_patients": len(uniq), "epochs": EPOCHS, "tiles_per_slide_cap": TPS, "seeds": SEEDS, "aggregations": ["mean", "max", "top10"]}}
oof = {}   # (scheme) -> agg -> key -> 6-vector
for scheme in ("case_max", "section"):
    acc = {a: {k: [] for k in keys} for a in ("mean", "max", "top10")}
    for s in SEEDS:
        for f in range(5):
            te = folds[f]; tr = [k for j, fl in enumerate(folds) if j != f for k in fl]; net = train_fold(tr, y6[scheme], s)
            for k in te:
                P = score(net, k); acc["mean"][k].append(P.mean(0)); acc["max"][k].append(P.max(0))
                t = max(1, len(P) // 10); acc["top10"][k].append(np.sort(P, 0)[-t:].mean(0))
            print(scheme, "seed", s, "fold", f, flush=True)
    oof[scheme] = {a: {k: np.mean(v, 0) for k, v in d.items()} for a, d in acc.items()}
    np.savez(os.path.join(OUT, f"tile_oof_{scheme}.npz"), keys=np.array(keys), **{a: np.stack([oof[scheme][a][k] for k in keys]) for a in oof[scheme]})
yt = np.array([y6["section"][k] for k in keys]); yb = np.isin(yt, list(POS)).astype(int)
def macro(P): return float(np.mean([roc_auc_score((yt == c).astype(int), P[:, c]) for c in range(6) if 0 < (yt == c).sum() < len(yt)]))
def cboot(fn, n=2000):
    up = sorted(set(pat.values())); idx = {}; [idx.setdefault(pat[k], []).append(i) for i, k in enumerate(keys)]; rg = np.random.RandomState(0); out = []
    for _ in range(n):
        sidx = np.concatenate([idx[up[j]] for j in rg.randint(0, len(up), len(up))]); v = fn(sidx)
        if v is not None: out.append(v)
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]
# ABMIL references from saved units
def load_units(d, field):
    acc = {"case": {}, "slide": {}}
    for f in glob.glob(d + "/*.npz"):
        arm = os.path.basename(f).split("_")[0]; z = np.load(f, allow_pickle=True)
        for k, p in zip(z["keys"], z[field]): acc[arm].setdefault(str(k), []).append(p)
    return acc
ab = load_units(T + "/feasibility/svc_units", "preds"); ab5 = load_units(T + "/feasibility/svc5_units", "probs")
for scheme, arm in (("case_max", "case"), ("section", "slide")):
    res[scheme] = {}
    pa = np.array([np.mean(ab[arm][k]) for k in keys]); Pa = np.stack([np.mean(ab5[arm][k], 0) for k in keys])
    res[scheme]["abmil_reference"] = {"binary_auc": round(float(roc_auc_score(yb, pa)), 4), "macro_auc": round(macro(Pa), 4)}
    for a in ("mean", "max", "top10"):
        P = np.stack([oof[scheme][a][k] for k in keys]); pb = P[:, 3:].sum(1)
        res[scheme][f"tile_{a}"] = {"binary_auc": round(float(roc_auc_score(yb, pb)), 4), "binary_ci": cboot(lambda i: roc_auc_score(yb[i], pb[i]) if len(set(yb[i])) > 1 else None),
                                    "macro_auc": round(macro(P), 4),
                                    "delta_binary_vs_abmil": cboot(lambda i: roc_auc_score(yb[i], pb[i]) - roc_auc_score(yb[i], pa[i]) if len(set(yb[i])) > 1 else None),
                                    "delta_macro_vs_abmil": cboot(lambda i: (np.mean([roc_auc_score((yt[i] == c).astype(int), P[i][:, c]) for c in range(6) if 0 < (yt[i] == c).sum() < len(i)]) - np.mean([roc_auc_score((yt[i] == c).astype(int), Pa[i][:, c]) for c in range(6) if 0 < (yt[i] == c).sum() < len(i)])) if len(set(yt[i])) > 2 else None)}
        print(scheme, a, res[scheme][f"tile_{a}"]["binary_auc"], res[scheme][f"tile_{a}"]["macro_auc"], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2); print(json.dumps(res, indent=1))
