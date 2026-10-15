"""2.38b: five/six-class slide-vs-casemax contrast — the follow-up to the
binary null. The 32% label disagreement lives in fine-grained distinctions the
binary target collapsed; here the classes are
NORMAL_OTHER < NDBE < IND < LGD < HGD < CANCER and we test whether
section-resolved supervision beats case-max where it should matter.
Metrics: macro one-vs-rest AUC + quadratic-weighted kappa on slide truth,
paired bootstrap. UNIT env trains one (arm, seed, fold); MODE=agg aggregates.
Self-contained multiclass gated-attention MIL (same shape as the binary
machinery; both arms share it, so the contrast is architecture-fair).
"""
import glob, json, os, sys
import h5py, numpy as np, pandas as pd
import torch, torch.nn as nn

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
UD = T + "/feasibility/svc5_units"
CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]
C_OF = {c: i for i, c in enumerate(CLASSES)}
DEV = "cuda" if torch.cuda.is_available() else "cpu"

lab = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str)
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
lab = lab.merge(m[["h5", "anon_id"]], on="h5")
lab = lab[lab["worst_grade"].isin(C_OF) & lab["case_max"].isin(C_OF)].reset_index(drop=True)
keys = sorted(lab["h5"])
lab = lab.set_index("h5").loc[keys]
pat = dict(zip(keys, lab["anon_id"]))
y_slide = {k: C_OF[g] for k, g in zip(keys, lab["worst_grade"])}
y_case = {k: C_OF[g] for k, g in zip(keys, lab["case_max"])}

# frozen folds identical to the binary run: patient-hash split, seed 0
rng = np.random.RandomState(0)
uniq = sorted(set(pat.values()))
fold_of = {a: i % 5 for i, a in enumerate(rng.permutation(uniq))}
folds = [[k for k in keys if fold_of[pat[k]] == f] for f in range(5)]

def qwk(a, b, n=6):
    a, b = np.asarray(a), np.asarray(b)
    O = np.zeros((n, n))
    for i, j in zip(a, b): O[i, j] += 1
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)], dtype=float) / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    return 1 - (W * O).sum() / max((W * E).sum(), 1e-9)

MODE = os.environ.get("MODE", "unit")
if MODE == "agg":
    from sklearn.metrics import roc_auc_score
    acc = {"case": {}, "slide": {}}
    units = glob.glob(UD + "/*.npz")
    print(f"units: {len(units)}/30", flush=True)
    for f in units:
        arm = os.path.basename(f).split("_")[0]
        z = np.load(f, allow_pickle=True)
        for k, p in zip(z["keys"], z["probs"]):
            acc[arm].setdefault(str(k), []).append(p)
    res = {"_meta": {"units": len(units), "n_slides": len(keys), "classes": CLASSES}}
    yt = np.array([y_slide[k] for k in keys])
    probs, preds = {}, {}
    for arm in ("case", "slide"):
        P = np.stack([np.mean(acc[arm][k], axis=0) for k in keys])
        probs[arm] = P; preds[arm] = P.argmax(1)
        aucs = [roc_auc_score((yt == c).astype(int), P[:, c])
                for c in range(6) if 0 < (yt == c).sum() < len(yt)]
        res[f"trained_{arm}"] = {"macro_auc": round(float(np.mean(aucs)), 4),
                                 "qwk_vs_slide_truth": round(float(qwk(yt, preds[arm])), 4)}
    boot_a, boot_q = [], []
    rngb = np.random.RandomState(0)
    for _ in range(1000):
        idx = rngb.randint(0, len(yt), len(yt))
        ys = yt[idx]
        aa = []
        for arm in ("case", "slide"):
            aucs = [roc_auc_score((ys == c).astype(int), probs[arm][idx][:, c])
                    for c in range(6) if 0 < (ys == c).sum() < len(ys)]
            aa.append(np.mean(aucs))
        boot_a.append(aa[1] - aa[0])
        boot_q.append(qwk(ys, preds["slide"][idx]) - qwk(ys, preds["case"][idx]))
    res["delta_slide_minus_case"] = {
        "macro_auc": {"mean": round(float(np.mean(boot_a)), 4),
                      "ci": [round(float(np.percentile(boot_a, 2.5)), 4),
                             round(float(np.percentile(boot_a, 97.5)), 4)]},
        "qwk": {"mean": round(float(np.mean(boot_q)), 4),
                "ci": [round(float(np.percentile(boot_q, 2.5)), 4),
                       round(float(np.percentile(boot_q, 97.5)), 4)]}}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    sys.exit(0)

# ---- unit mode ----
UNIT = os.environ["UNIT"]
arm, seed, fold = UNIT.split("_")
seed, fold = int(seed), int(fold)
ydict = y_case if arm == "case" else y_slide

bags = {}
for h5p in keys:
    with h5py.File(h5p) as h:
        bags[h5p] = np.asarray(h["features"])

class MC_MIL(nn.Module):
    def __init__(self, d_in=1536, n_cls=6):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.1))
        self.att_v = nn.Linear(512, 128); self.att_u = nn.Linear(512, 128)
        self.att_w = nn.Linear(128, 1)
        self.head = nn.Linear(512, n_cls)
    def forward(self, bag):
        h = self.emb(bag)
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).softmax(0)
        return self.head((a * h).sum(0))

te = folds[fold]; tr = [k for j, f in enumerate(folds) if j != fold for k in f]
torch.manual_seed(seed)
net = MC_MIL().to(DEV)
cnt = np.bincount([ydict[k] for k in tr], minlength=6).astype(float)
w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32, device=DEV)
opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
rng2 = np.random.RandomState(seed)
MAXT = 2000
for ep in range(30):
    net.train()
    for k in rng2.permutation(tr):
        b = bags[k]
        if len(b) > MAXT:
            b = b[rng2.choice(len(b), MAXT, replace=False)]
        logit = net(torch.tensor(b, dtype=torch.float32, device=DEV))
        loss = nn.functional.cross_entropy(logit.unsqueeze(0),
                                           torch.tensor([ydict[k]], device=DEV), weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    print("epoch", ep, flush=True)
net.eval()
out_k, out_p = [], []
with torch.no_grad():
    for k in te:
        p = torch.softmax(net(torch.tensor(bags[k], dtype=torch.float32, device=DEV)), 0)
        out_k.append(k); out_p.append(p.cpu().numpy())
os.makedirs(UD, exist_ok=True)
np.savez(os.path.join(UD, f"{arm}_{seed}_{fold}.npz"),
         keys=np.array(out_k), probs=np.stack(out_p))
json.dump({"unit": UNIT, "n_test": len(out_k)}, open(os.path.join(OUT, "results.json"), "w"))
print("unit done", UNIT, flush=True)
