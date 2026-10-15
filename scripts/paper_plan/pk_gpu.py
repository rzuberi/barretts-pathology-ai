"""Paper final K4 ablation (pre-specified in docs/paper_final_inputs.md @ e6e00c0): row-level held-out scores of each learned
fusion model under CNV / image ablations; deltas on all 150 and on the 82 discovery patients with patient-bootstrap CIs.
Outputs: results/paper_final/k4_ablation.json; rows feasibility/paper_plan/pk_ablation_rows.csv."""
import glob, json, os, sys, warnings, numpy as np, pandas as pd, torch
from scipy.stats import rankdata
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/multimodal-barretts-progression"; sys.path.insert(0, B + "/src")
from barrett.models.early_fusion import EarlyFusionMLP; from barrett.models.intermediate_fusion import IntermediateABMILCNV; from barrett.models.coattention import CoAttentionABMILCNV; from barrett.training.data import load_cnv_matrix
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; ROW = T + "/feasibility/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_final"); os.makedirs(AGG, exist_ok=True); DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); NB = 2000; SEED = 0; NREP = 50
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def r3(x): return round(float(x), 3)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); ids = list(man.index); y = man.y_progressor.astype(int).values; fold = man.fold_id_rep01.astype(int).values; pid = man.patient_id.values
ui = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str).set_index("sample_id").reindex(ids); cnv_df, feats = load_cnv_matrix(F + "/feature_views/cnv"); X_cnv = cnv_df.set_index("sample_id").loc[ids, feats].to_numpy(np.float64)
bags = {}
for s, p in zip(ids, ui.npz_path):
    with np.load(p, allow_pickle=False) as z: bags[s] = torch.from_numpy(np.asarray(z["embeddings"], np.float32))
def build(fam, cfg, cnv_dim):
    if fam == "early_fusion": return EarlyFusionMLP(1536, cnv_dim, int(cfg["hidden_dim"]), float(cfg["dropout"]))
    cls = IntermediateABMILCNV if fam == "intermediate_fusion" else CoAttentionABMILCNV; return cls(1536, cnv_dim, int(cfg["img_hidden"]), int(cfg["cnv_hidden"]), int(cfg["attn_dim"]), int(cfg["fusion_hidden"]), float(cfg["dropout"]))
def load(fam, k):
    ck = torch.load(f"{R}/{fam}/fold{k}/model.pt", map_location="cpu"); m = build(fam, ck["configuration"], len(feats)); m.load_state_dict(ck["state_dict"]); m.eval().to(DEV); med, mu, sd = ck.get("cnv_median"), ck.get("cnv_mean"), ck.get("cnv_std")
    Xs = np.where(np.isfinite(X_cnv), X_cnv, med) if med is not None else X_cnv; Xs = ((Xs - mu) / sd).astype(np.float32) if mu is not None else Xs.astype(np.float32); return m, torch.from_numpy(Xs).to(DEV)
def fwd(m, bl, Xc):
    with torch.no_grad(): return np.array([torch.sigmoid(m([b.to(DEV)], Xc[i:i + 1])).item() for i, b in enumerate(bl)])
mean_tile = {k: torch.stack([bags[s].mean(0) for s, f_ in zip(ids, fold) if f_ != k]).mean(0) for k in range(1, 6)}
rng = np.random.RandomState(SEED); rows = pd.DataFrame({"sample_id": ids, "patient_id": pid, "fold": fold, "y": y}); RES = {"_spec": "docs/paper_final_inputs.md @ e6e00c0 K4", "models": {}}
st = pd.read_csv(ROW + "/f2_strata.csv", dtype=str).set_index("patient_id"); pats = np.array(sorted(set(pid))); py = pd.Series(y, index=pid).groupby(level=0).max().reindex(pats).values; disc = st.stratum.reindex(pats).str.startswith("discovery").values
def pmax(s): return pd.Series(s, index=pid).groupby(level=0).max().reindex(pats).values
def bidx(yy):
    rr = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        s = rr.choice(len(yy), len(yy))
        if len(set(yy[s])) > 1: out.append(s)
    return out
B_all, B_disc = bidx(py), bidx(py[disc])
def delta(base, abl, mask, Bs): a, b = pmax(base)[mask], pmax(abl)[mask]; yy = py[mask]; v = [auc(yy[s], a[s]) - auc(yy[s], b[s]) for s in Bs]; return {"delta_auroc": r3(auc(yy, a) - auc(yy, b)), "ci": [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))], "baseline_auroc": r3(auc(yy, a)), "ablated_auroc": r3(auc(yy, b)), "n": int(mask.sum()), "events": int(yy.sum())}
for fam in ["early_fusion", "intermediate_fusion", "coattention_fusion"]:
    base = np.zeros(len(ids)); cp = np.zeros(len(ids)); cm = np.zeros(len(ids)); ip = np.zeros(len(ids)); im = np.zeros(len(ids))
    for k in range(1, 6):
        m, Xc = load(fam, k); te = np.where(fold == k)[0]; bl = [bags[ids[i]] for i in te]; base[te] = fwd(m, bl, Xc[te])
        acc = np.zeros(len(te))
        for r in range(NREP): acc += fwd(m, bl, Xc[te][torch.from_numpy(rng.permutation(len(te))).to(DEV)])
        cp[te] = acc / NREP; cm[te] = fwd(m, bl, torch.zeros_like(Xc[te]))
        acc = np.zeros(len(te))
        for r in range(NREP): perm = rng.permutation(len(te)); acc += fwd(m, [bl[j] for j in perm], Xc[te])
        ip[te] = acc / NREP; im[te] = fwd(m, [mean_tile[k][None, :] for _ in te], Xc[te]); print(fam, k, flush=True)
    for nm, arr in [("base", base), ("cnv_perm_mean", cp), ("cnv_mean", cm), ("img_perm_mean", ip), ("img_mean", im)]: rows[f"{fam}__{nm}"] = arr
    RES["models"][fam] = {pop: {"cnv_permuted_jointly_mean_of_50": delta(base, cp, mask, Bs), "cnv_replaced_by_training_mean": delta(base, cm, mask, Bs), "image_bags_permuted_mean_of_50": delta(base, ip, mask, Bs), "image_replaced_by_mean_tile": delta(base, im, mask, Bs)} for pop, mask, Bs in [("all_150", np.ones(len(pats), bool), B_all), ("discovery_82", disc, B_disc)]}
rows.to_csv(ROW + "/pk_ablation_rows.csv", index=False); RES["_method"] = "ablated score per row = mean over 50 permutations (or the deterministic replacement); delta = patient AUROC(baseline) - AUROC(ablated), patient = max over rows; CI = patient bootstrap (2,000, seed 0) of the paired difference"
json.dump(RES, open(AGG + "/k4_ablation.json", "w"), indent=1); print("K4 GPU DONE", flush=True)
