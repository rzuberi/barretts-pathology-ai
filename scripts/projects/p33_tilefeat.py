"""P33 first pass: patch-level grading as a modality. (1) Tile MLP (1536->512->6) trained on SECTION labels of the 1,538
dual-labelled slides, 5 patient folds (the 2.47 fold rule), seed 0, 4 epochs -> OOF per-tile class probabilities for those
slides; a final model on all 1,538 for every other slide. (2) Applied to EVERY UNI2 feature file referenced by the ERIN
task tables (T2a/T2b/T3a/T3b case bags) and to all-slides features -> per-slide 13-d summary: 6 grade fractions (argmax),
6 mean probabilities, log tile count; a slide graded by the fold model whenever it was in the 1,538 (matched by uuid) so
no slide is scored by a model that saw its section label. Env: OUTDIR, EPOCHS (4), TPS (2000)."""
import glob, json, os, h5py, numpy as np, pandas as pd, torch, torch.nn as nn
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", "."); DEV = "cuda" if torch.cuda.is_available() else "cpu"; EPOCHS = int(os.environ.get("EPOCHS", "4")); TPS = int(os.environ.get("TPS", "2000")); SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLASSES)}
lab = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str); m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
lab = lab.drop(columns=[c for c in ("anon_id",) if c in lab.columns]).merge(m[["h5", "anon_id"]], on="h5"); lab = lab[lab.worst_grade.isin(C_OF)].reset_index(drop=True)
keys = sorted(lab.h5); lab = lab.set_index("h5").loc[keys]; pat = dict(zip(keys, lab.anon_id)); y6 = {k: C_OF[g] for k, g in zip(keys, lab.worst_grade)}
rng = np.random.RandomState(0); uniq = sorted(set(pat.values())); fold_of_pat = {a: i % 5 for i, a in enumerate(rng.permutation(uniq))}; fold_of = {k: fold_of_pat[pat[k]] for k in keys}
uuid_fold = {os.path.basename(k).replace(".h5", ""): fold_of[k] for k in keys}
# patient-level assignment: any slide of a training patient is scored by that patient's held-out fold model
_rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv", dtype=str, usecols=["CaseName", "anon_id"]).drop_duplicates("CaseName")
_sl = pd.read_csv(T + "/campaigns/allslides/erin_slides_all.csv", dtype=str)[["file_uuid", "CaseName"]].merge(_rep, on="CaseName", how="left")
uuid_patient = dict(zip(_sl.file_uuid, _sl.anon_id)); uuid_patient.update({os.path.basename(k).replace(".h5", ""): pat[k] for k in keys})
def fold_for(u): return uuid_fold.get(u, fold_of_pat.get(uuid_patient.get(u), None))
print("loading 1,538 training bags", flush=True); rs = np.random.RandomState(0); bags = {}
for k in keys:
    with h5py.File(k) as h: X = np.asarray(h["features"], np.float16)
    bags[k] = X[rs.choice(len(X), TPS, replace=False)] if len(X) > TPS else X
class TileMLP(nn.Module):
    def __init__(s, d=1536, n=6): super().__init__(); s.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, n))
    def forward(s, x): return s.net(x)
def train(tr, seed=0):
    torch.manual_seed(seed); X = torch.tensor(np.concatenate([bags[k] for k in tr])); Y = torch.tensor(np.concatenate([[y6[k]] * len(bags[k]) for k in tr]))
    cnt = np.bincount(Y.numpy(), minlength=6).astype(float); w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32, device=DEV)
    net = TileMLP().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4); n = len(X)
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]; loss = nn.functional.cross_entropy(net(X[idx].to(DEV).float()), Y[idx].to(DEV), weight=w); opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()
models = {}   # (fold or "final") -> list of nets over seeds
for f in list(range(5)) + ["final"]:
    models[f] = [train([k for k in keys if fold_of[k] != f] if f != "final" else keys, seed=sd) for sd in SEEDS]; print("models", f, "seeds", SEEDS, flush=True)
torch.save({f"{k}_s{i}": m.state_dict() for k, ms in models.items() for i, m in enumerate(ms)}, os.path.join(OUT, "tile_mlp_models.pt"))
def summarise(path, nets):
    with h5py.File(path) as h: X = np.asarray(h["features"], np.float32)
    with torch.no_grad(): P = np.mean([torch.softmax(n(torch.tensor(X).to(DEV)), -1).cpu().numpy() for n in nets], 0)
    am = np.bincount(P.argmax(1), minlength=6) / len(P); return np.concatenate([am, P.mean(0), [np.log1p(len(P))]]), P
# targets: every h5 in the task tables + the 1,538 + all-slides features
paths = set(keys)
for f in glob.glob(T + "/feasibility/erin_fusion/tasks/*.csv"):
    for hl in pd.read_csv(f, dtype=str).h5_list.dropna(): paths.update(hl.split("|"))
allslides = sorted(glob.glob("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/features/20x_224px/features_uni_v2_all/*.h5")); paths.update(allslides)
paths = sorted(p for p in paths if os.path.exists(p)); print("slides to summarise:", len(paths), flush=True)
rows = []; oof_slide = {}
for i, p in enumerate(paths):
    u = os.path.basename(p).replace(".h5", ""); ff = fold_for(u); nets = models[ff] if ff is not None else models["final"]
    try: v, P = summarise(p, nets)
    except Exception as e: print("[skip]", p, e, flush=True); continue
    rows.append([p, u, ("fold_slide" if u in uuid_fold else "fold_patient") if ff is not None else "final"] + v.tolist())
    if p in bags: oof_slide[p] = P.mean(0)
    if i % 500 == 0: print(i, flush=True)
cols = ["h5", "uuid", "model"] + [f"frac_{c}" for c in CLASSES] + [f"mean_{c}" for c in CLASSES] + ["log_ntiles"]
df = pd.DataFrame(rows, columns=cols); df.to_parquet(os.path.join(OUT, "tile_summary_all.parquet"), index=False); df.to_csv(os.path.join(OUT, "tile_summary_all.csv"), index=False)
# OOF slide-level AUROC on the 1,538 (sanity, matches 2.47 tile_mean ~0.836 section)
from sklearn.metrics import roc_auc_score
yt = np.array([y6[k] for k in keys]); pb = np.array([oof_slide[k][3:].sum() for k in keys]); yb = (yt >= 3).astype(int)
res = {"n_summarised": int(len(df)), "n_fold_scored_slide": int((df.model == "fold_slide").sum()), "n_fold_scored_patient": int((df.model == "fold_patient").sum()), "n_final": int((df.model == "final").sum()), "seeds": SEEDS, "oof_1538_binary_LGDplus_auroc_mean_prob": round(float(roc_auc_score(yb, pb)), 4), "epochs": EPOCHS}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1); print(json.dumps(res))
