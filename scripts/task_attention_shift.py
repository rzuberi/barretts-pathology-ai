"""Attention-shift study (2.20, Rehan 2026-08-23): does conditioning histologic
attention on genomics change WHERE the model looks?

Arms on the TCGA pool (Cox survival, identical folds/seeds):
  A  plain ABMIL (no context)
  B  conditional ABMIL: genomic vector (tp53, ploidy, wgd) projected and added to
     every tile embedding before attention
  C  control: conditional ABMIL with PERMUTED genomics (any shift = noise floor)
Outputs: per-slide attention vectors for A/B/C + summary (Spearman(A,B) vs
Spearman(A,C), top-10 Jaccard, shift vs WGD status, C-indices).
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd, h5py, torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_cox import ABMIL, cox_loss, cindex, stratified_folds, MAX_TILES, RNG

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
FEATS = {"OAC": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2",
         "GEJ": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/features/20x_224px/features_uni_v2"}
LABELS = {"OAC": T + "/data/tcga_oac_labels.csv", "GEJ": T + "/data/tcga_stad_labels.csv"}
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class CondABMIL(ABMIL):
    def __init__(self, d_in=1536, d_ctx=3, **kw):
        super().__init__(d_in=d_in, **kw)
        self.ctx = nn.Linear(d_ctx, 384)

    def forward(self, bag, g=None):
        h = self.emb(bag)
        if g is not None:
            h = h + self.ctx(g).unsqueeze(0)
        a = self.attn_w(self.attn_v(h) * self.attn_u(h)).softmax(dim=0)
        z = (a * h).sum(dim=0)
        return self.head(z).squeeze(), a.squeeze()


COHORT = os.environ.get("COHORT", "tcga_pool")
bags, G, time_, event = {}, {}, {}, {}
if COHORT == "occams":
    OCC_FEAT = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/slides/features/20x_224px/features_uni_v2"
    OCC_TSV = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data/genomics/clinical_data_wgs_cases_therapy_tp53status_ploidy_wgd_status.tsv"
    MASTER = "/home/zuberi01/occams_work/occams_master_20260511.csv"
    def norm_occ(x):
        x = str(x).strip().upper().replace("/", "-")
        mm = re.search(r"(?:OCCAMS|OC)[-_ ]?([A-Z]{2})[-_ ]?0*([0-9]+)", x)
        return f"{mm.group(1)}{int(mm.group(2)):04d}" if mm else x
    th = pd.read_csv(OCC_TSV, sep="\t", dtype=str)
    th["cid"] = th["OCCAMS_ID"].map(norm_occ)
    th = th.drop_duplicates("cid").set_index("cid")
    mast = pd.read_csv(MASTER, dtype=str, low_memory=False)
    mast["cid"] = mast["occams_id"].map(norm_occ)
    dsd = pd.to_numeric(mast["deceased_survival_days"], errors="coerce")
    lkd = pd.to_numeric(mast["last_known_survival_days"], errors="coerce")
    mast["time"] = dsd.fillna(lkd); mast["event"] = dsd.notna().astype(int)
    mast = mast[mast["time"] > 0].drop_duplicates("cid").set_index("cid")
    def fl(v): return float(str(v).strip().lower() in ("1", "true", "yes", "y"))
    for f in sorted(glob.glob(os.path.join(OCC_FEAT, "*.h5"))):
        c = norm_occ(os.path.basename(f).split("_")[0])
        if c in bags or c not in th.index or c not in mast.index: continue
        with h5py.File(f) as h:
            bags[c] = np.asarray(h["features"])
        r = th.loc[c]
        tp53 = max(fl(r["TP53_SNV"]), fl(r["TP53_indel"]), fl(r["TP53_deletion"]), fl(r["TP53_knockout"]))
        G[c] = np.array([tp53,
                         (float(r["ploidy"]) - 2.0) if pd.notna(r["ploidy"]) else 0.0,
                         fl(r["WGD"]) if pd.notna(r["WGD"]) else 0.0], dtype=np.float32)
        time_[c] = float(mast.loc[c, "time"]); event[c] = int(mast.loc[c, "event"])
    FEATS = {}
for grp, fd in FEATS.items():
    lab = pd.read_csv(LABELS[grp])
    lab = lab[lab["os_days"] > 0].drop_duplicates("barcode").set_index("barcode")
    for f in sorted(glob.glob(os.path.join(fd, "*.h5"))):
        m = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
        if not m or m.group(1) not in lab.index or m.group(1) in bags: continue
        with h5py.File(f) as h:
            bags[m.group(1)] = np.asarray(h["features"])
        r = lab.loc[m.group(1)]
        G[m.group(1)] = np.array([float(r["tp53_mut"]) if pd.notna(r["tp53_mut"]) else 0.0,
                                  (float(r["ploidy"]) - 2.0) if pd.notna(r["ploidy"]) else 0.0,
                                  float(r["wgd"]) if pd.notna(r["wgd"]) else 0.0], dtype=np.float32)
        time_[m.group(1)] = float(r["os_days"]); event[m.group(1)] = int(r["os_event"])
cases = sorted(bags)
print(f"n={len(cases)}", flush=True)

perm = dict(zip(cases, [G[k] for k in RNG(7).permutation(cases)]))
folds = stratified_folds(cases, event, 5, seed=0)


def bag_t(k, rng):
    F = bags[k]
    if len(F) > MAX_TILES: F = F[rng.choice(len(F), MAX_TILES, replace=False)]
    return torch.tensor(F, dtype=torch.float32, device=DEV)


def train(arm, gmap, seed):
    torch.manual_seed(seed); rng = RNG(seed)
    oof_risk, attn_out = {}, {}
    for i in range(5):
        te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
        model = (ABMIL() if arm == "A" else CondABMIL()).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        for ep in range(25):
            order = rng.permutation(tr)
            for b in range(0, len(order), 32):
                chunk = order[b:b + 32]
                if sum(event[k] for k in chunk) == 0: continue
                risks = []
                for k in chunk:
                    g = None if arm == "A" else torch.tensor(gmap[k], device=DEV)
                    risks.append(model(bag_t(k, rng), g)[0] if arm != "A" else model(bag_t(k, rng))[0])
                loss = cox_loss(torch.stack(risks),
                                torch.tensor([time_[k] for k in chunk], device=DEV),
                                torch.tensor([float(event[k]) for k in chunk], device=DEV))
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            for k in te:
                # fixed rng for attention comparability across arms
                bt = bag_t(k, RNG(0))
                if arm == "A":
                    r, a = model(bt)
                else:
                    r, a = model(bt, torch.tensor(gmap[k], device=DEV))
                oof_risk[k] = float(r)
                attn_out[k] = a.cpu().numpy()
    return oof_risk, attn_out


res = {"_meta": {"n": len(cases), "seeds": SEEDS}}
attn = {}
for arm, gmap in [("A", None), ("B", G), ("C", perm)]:
    risks, attns = {}, {}
    for s in SEEDS:
        r, a = train(arm, gmap, s)
        for k in cases:
            risks.setdefault(k, []).append(r[k])
            attns.setdefault(k, []).append(a[k])
    oof = {k: float(np.mean(v)) for k, v in risks.items()}
    attn[arm] = {k: np.mean(np.stack(v), axis=0) for k, v in attns.items()}
    res[f"cindex_{arm}"] = round(float(cindex(np.array([oof[k] for k in cases]),
                                              np.array([time_[k] for k in cases]),
                                              np.array([event[k] for k in cases]))), 4)
    print(arm, res[f"cindex_{arm}"], flush=True)

from scipy.stats import spearmanr
def compare(a1, a2):
    rhos, jac = [], []
    for k in cases:
        x, y = attn[a1][k], attn[a2][k]
        n = min(len(x), len(y))
        rhos.append(spearmanr(x[:n], y[:n])[0])
        tx, ty = set(x[:n].argsort()[-10:]), set(y[:n].argsort()[-10:])
        jac.append(len(tx & ty) / len(tx | ty))
    return rhos, jac

for pair in (("A", "B"), ("A", "C")):
    rhos, jac = compare(*pair)
    res[f"attn_{pair[0]}vs{pair[1]}"] = {"spearman_mean": round(float(np.nanmean(rhos)), 3),
                                          "jaccard10_mean": round(float(np.mean(jac)), 3)}
rhos_b, _ = compare("A", "B")
wgd_pos = [i for i, k in enumerate(cases) if G[k][2] == 1.0]
wgd_neg = [i for i, k in enumerate(cases) if G[k][2] == 0.0]
res["shift_by_wgd"] = {"spearman_wgd_pos": round(float(np.nanmean([rhos_b[i] for i in wgd_pos])), 3),
                       "spearman_wgd_neg": round(float(np.nanmean([rhos_b[i] for i in wgd_neg])), 3)}
np.savez(os.path.join(OUT, "attention_vectors.npz"),
         **{f"{arm}_{k}": attn[arm][k] for arm in attn for k in list(cases)[:50]})
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
