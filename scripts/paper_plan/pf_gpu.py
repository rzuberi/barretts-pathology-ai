"""Follow-up F6b/F6c, F7 and F8 (train-vs-held-out), pre-specified in docs/paper_plan_followup.md @ 0db4075.
F6c: ABMIL (release image config) retrained per outer fold on 0.44 um/px UNI2-h bags (<=2,048 tiles), fixed epochs = release
fold final_epochs. F6b: F6c fold model on 10 random 256-tile draws (seeds 0-9) of held-out slides. F7: modality ablations of
the learned fusion models on held-out rows. 8b noise floor. F8: training-fold vs held-out AUROC per fold and family.
Outputs: results/paper_plan/f6_image_tiles.json, f7_modality_ablation.json, f8_train_vs_heldout.json; rows -> feasibility/paper_plan/."""
import glob, json, os, sys, warnings, numpy as np, pandas as pd, torch, torch.nn as nn, joblib, h5py
from scipy.stats import rankdata
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/multimodal-barretts-progression"; sys.path.insert(0, B + "/src")
from barrett.models.image_mil import AttentionMIL; from barrett.models.early_fusion import EarlyFusionMLP; from barrett.models.intermediate_fusion import IntermediateABMILCNV; from barrett.models.coattention import CoAttentionABMILCNV; from barrett.training.data import load_cnv_matrix
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; ROW = T + "/feasibility/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_plan"); os.makedirs(AGG, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); NB = 2000; SEED = 0
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); ids = list(man.index); y = man.y_progressor.astype(int).values; fold = man.fold_id_rep01.astype(int).values; pid = man.patient_id.values
ui = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str).set_index("sample_id").reindex(ids); cnv_df, feats = load_cnv_matrix(F + "/feature_views/cnv"); X_cnv = cnv_df.set_index("sample_id").loc[ids, feats].to_numpy(np.float64)
pats = np.array(sorted(set(pid))); pfold = pd.Series(fold, index=pid).groupby(level=0).first().reindex(pats).values; py = pd.Series(y, index=pid).groupby(level=0).max().reindex(pats).values
def pmax(s): return pd.Series(s, index=pid).groupby(level=0).max().reindex(pats).values
def bidx(yy):
    rng = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        s = rng.choice(len(yy), len(yy))
        if len(set(yy[s])) > 1: out.append(s)
    return out
BI = bidx(py); ci = lambda a, b=None: [r3(np.percentile([auc(py[s], a[s]) - (auc(py[s], b[s]) if b is not None else 0) for s in BI], q)) for q in (2.5, 97.5)]
# ---------------------------------------------------------------- bags
bags = {}
for s, p in zip(ids, ui.npz_path):
    with np.load(p, allow_pickle=False) as z: bags[s] = torch.from_numpy(np.asarray(z["embeddings"], np.float32))
rs = np.random.RandomState(0); big = {}; kept = {}
for s, b in zip(ids, ui.image_basename):
    with h5py.File(f"{S}/features_uni2h_05um/{os.path.splitext(str(b))[0]}.h5") as h: Xh = np.asarray(h["features"], np.float16); kept[s] = int(len(Xh))
    big[s] = torch.from_numpy(Xh[rs.choice(len(Xh), 2048, replace=False)] if len(Xh) > 2048 else Xh)
print("bags loaded", flush=True)
# ---------------------------------------------------------------- F6c retrain on 0.44 um bags
def train_abmil(keys, ep, seed=0):
    torch.manual_seed(seed); rng = np.random.RandomState(seed); m = AttentionMIL(1536, 256, 128, 0.1).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=0.01); lossf = nn.BCEWithLogitsLoss(); yd = dict(zip(ids, y))
    for e in range(ep):
        m.train(); order = rng.permutation(keys)
        for i in range(0, len(order), 8):
            ch = order[i:i + 8]; logits = torch.stack([m.forward_bag(big[k].to(DEV).float()) for k in ch]).view(-1); loss = lossf(logits, torch.tensor([float(yd[k]) for k in ch], device=DEV)); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()
oof05 = np.zeros(len(ids)); trainsc = {}; models05 = {}
for k in range(1, 6):
    ep = int(torch.load(f"{R}/image_only/fold{k}/model.pt", map_location="cpu")["final_epochs"]); tr = [s for s, f_ in zip(ids, fold) if f_ != k]; te = [s for s, f_ in zip(ids, fold) if f_ == k]
    m = train_abmil(tr, ep); models05[k] = m
    with torch.no_grad():
        for s in te: oof05[ids.index(s)] = torch.sigmoid(m.forward_bag(big[s].to(DEV).float())).item()
        trainsc[k] = {s: torch.sigmoid(m.forward_bag(big[s].to(DEV).float())).item() for s in tr}
    torch.save(m.state_dict(), f"{ROW}/f6c_image05_fold{k}.pt"); print("F6c fold", k, "epochs", ep, flush=True)
P05 = pmax(oof05); img_rel = pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{R}/image_only/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(ids).values; Pimg = pmax(img_rel)
F6 = {"F6c_image_only_044um_le2048": {"n": int(len(py)), "events": int(py.sum()), "auroc": r3(auc(py, P05)), "ci": ci(P05), "delta_vs_release_image_only": [r3(auc(py, P05) - auc(py, Pimg))] + ci(P05, Pimg), "epochs_per_fold": {k: int(torch.load(f"{R}/image_only/fold{k}/model.pt", map_location="cpu")["final_epochs"]) for k in range(1, 6)}, "tiles_per_bag_median": r3(np.median([min(kept[s], 2048) for s in ids])), "note": "single seed 0; scale 0.44 um/px vs release 0.88 um/px; tiles capped at 2,048 random per slide (seed 0)"}}
# F6b: 10 random 256-tile draws
draws = np.zeros((10, len(ids)))
with torch.no_grad():
    for d in range(10):
        rd = np.random.RandomState(d)
        for i, s in enumerate(ids):
            Xb = big[s]; idx = rd.choice(len(Xb), 256, replace=False) if len(Xb) > 256 else np.arange(len(Xb)); draws[d, i] = torch.sigmoid(models05[fold[i]].forward_bag(Xb[idx].to(DEV).float())).item()
Pd = np.stack([pmax(draws[d]) for d in range(10)]); sd_pat = Pd.std(0)
def thr(k):
    tr = pfold != k; ytr = py[tr]; s_tr = pd.Series({p_: max(v for s_, v in trainsc[k].items() if pid[ids.index(s_)] == p_) for p_ in pats[tr]}).reindex(pats[tr]).values
    for t in np.sort(np.unique(s_tr))[::-1]:
        if (s_tr[ytr == 1] >= t).mean() >= 0.8: return t
    return np.min(s_tr)
TH = {k: thr(k) for k in range(1, 6)}; pred_d = np.stack([(Pd[d] >= np.array([TH[k] for k in pfold])).astype(int) for d in range(10)])
sp = pd.read_csv(ROW + "/patient_scores_predictions.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(pats); fn_late = (sp.pred_late_mean.values == 0) & (py == 1)
F6["F6b_256_tile_resampling"] = {"patient_score_sd_across_10_draws": {"median": r3(np.median(sd_pat)), "iqr": [r3(np.percentile(sd_pat, 25)), r3(np.percentile(sd_pat, 75))], "max": r3(sd_pat.max())}, "auroc_per_draw": [r3(auc(py, Pd[d])) for d in range(10)], "auroc_full_bag_F6c": r3(auc(py, P05)),
    "late_mean_FN_patients": int(fn_late.sum()), "FN_predicted_positive_draws_of_10": [int(v) for v in pred_d[:, fn_late].sum(0)], "FN_score_sd": [r3(v) for v in sd_pat[fn_late]], "patients_changing_class_across_draws": int(((pred_d.sum(0) > 0) & (pred_d.sum(0) < 10)).sum()), "thresholds_per_fold": {k: r3(v) for k, v in TH.items()}}
pd.DataFrame({"sample_id": ids, "image05_oof": oof05, **{f"draw{d}": draws[d] for d in range(10)}}).to_csv(ROW + "/f6_image05_rows.csv", index=False)
json.dump(F6, open(AGG + "/f6_image_tiles.json", "w"), indent=1); print("F6 done", F6["F6c_image_only_044um_le2048"]["auroc"], flush=True)
del big; torch.cuda.empty_cache()
# ---------------------------------------------------------------- F7 ablations + F8 train vs held-out
def build(fam, cfg, cnv_dim):
    if fam == "image_only": return AttentionMIL(1536, int(cfg["hidden_dim"]), int(cfg["attn_dim"]), float(cfg["dropout"]))
    if fam == "early_fusion": return EarlyFusionMLP(1536, cnv_dim, int(cfg["hidden_dim"]), float(cfg["dropout"]))
    cls = IntermediateABMILCNV if fam == "intermediate_fusion" else CoAttentionABMILCNV; return cls(1536, cnv_dim, int(cfg["img_hidden"]), int(cfg["cnv_hidden"]), int(cfg["attn_dim"]), int(cfg["fusion_hidden"]), float(cfg["dropout"]))
def load(fam, k):
    ck = torch.load(f"{R}/{fam}/fold{k}/model.pt", map_location="cpu"); m = build(fam, ck["configuration"], len(feats)); m.load_state_dict(ck["state_dict"]); m.eval().to(DEV); med, mu, sd = ck.get("cnv_median"), ck.get("cnv_mean"), ck.get("cnv_std")
    Xs = np.where(np.isfinite(X_cnv), X_cnv, med) if med is not None else X_cnv; Xs = ((Xs - mu) / sd).astype(np.float32) if mu is not None else Xs.astype(np.float32); return m, torch.from_numpy(Xs).to(DEV)
def fwd(m, fam, bag_list, Xc):
    with torch.no_grad():
        if fam == "image_only": return np.array([torch.sigmoid(m.forward_bag(b.to(DEV))).item() for b in bag_list])
        out = []
        for i, b in enumerate(bag_list): out.append(torch.sigmoid(m([b.to(DEV)], Xc[i:i + 1])).item())
        return np.array(out)
F7 = {}; F8 = {"train_vs_heldout_auroc_patient_level": {}}; rng = np.random.RandomState(SEED); NREP = 50
mean_tile = {k: torch.stack([bags[s].mean(0) for s, f_ in zip(ids, fold) if f_ != k]).mean(0) for k in range(1, 6)}
for fam in ["image_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]:
    base = np.zeros(len(ids)); cnv_perm = np.zeros((NREP, len(ids))); cnv_mean = np.zeros(len(ids)); img_perm = np.zeros((NREP, len(ids))); img_mean = np.zeros(len(ids)); tv = {}
    for k in range(1, 6):
        m, Xc = load(fam, k); te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]; bl = [bags[ids[i]] for i in te]
        base[te] = fwd(m, fam, bl, Xc[te]); ptr = fwd(m, fam, [bags[ids[i]] for i in tr], Xc[tr])
        gtr = pd.DataFrame({"p": pid[tr], "s": ptr, "y": y[tr]}).groupby("p"); gte = pd.DataFrame({"p": pid[te], "s": base[te], "y": y[te]}).groupby("p"); tv[k] = {"train_auroc": r3(auc(gtr.y.max().values, gtr.s.max().values)), "heldout_auroc": r3(auc(gte.y.max().values, gte.s.max().values)), "n_train_patients": int(len(gtr)), "n_heldout_patients": int(len(gte))}
        if fam != "image_only":
            for r in range(NREP): cnv_perm[r, te] = fwd(m, fam, bl, Xc[te][torch.from_numpy(rng.permutation(len(te))).to(DEV)])
            cnv_mean[te] = fwd(m, fam, bl, torch.zeros_like(Xc[te]))
        for r in range(NREP): perm = rng.permutation(len(te)); img_perm[r, te] = fwd(m, fam, [bl[j] for j in perm], Xc[te])
        img_mean[te] = fwd(m, fam, [mean_tile[k][None, :] for _ in te], Xc[te]); print("F7", fam, k, flush=True)
    F8["train_vs_heldout_auroc_patient_level"][fam] = tv; Pb = pmax(base); out = {"baseline_auroc": r3(auc(py, Pb)), "n": int(len(py)), "events": int(py.sum())}
    def abl(arr2d):
        per = np.array([auc(py, pmax(arr2d[r])) for r in range(arr2d.shape[0])]); Pm = pmax(arr2d.mean(0)); return {"delta_auroc_mean_over_repeats": r3(auc(py, Pb) - per.mean()), "delta_sd_over_repeats": r3(per.std()), "delta_of_mean_permuted_score_ci": [r3(-v) if v is not None else None for v in ci(Pm, Pb)][::-1]}
    if fam != "image_only": out["cnv_permuted_jointly"] = abl(cnv_perm); Pm = pmax(cnv_mean); out["cnv_replaced_by_training_mean"] = {"delta_auroc": r3(auc(py, Pb) - auc(py, Pm)), "ci": [r3(-v) for v in ci(Pm, Pb)][::-1]}
    out["image_bags_permuted"] = abl(img_perm); Pm = pmax(img_mean); out["image_replaced_by_training_mean_tile"] = {"delta_auroc": r3(auc(py, Pb) - auc(py, Pm)), "ci": [r3(-v) for v in ci(Pm, Pb)][::-1]}
    F7[fam] = out
# 8b noise floor: random 5-Mb window features
WIN = [i for i, f in enumerate(feats) if ":" in f]; ARM = [i for i, f in enumerate(feats) if ":" not in f]; rngw = np.random.RandomState(SEED); wsel = list(rngw.choice(WIN, 10, replace=False))
def pat_auc(idx, s): g = pd.DataFrame({"p": pid[idx], "s": s, "y": y[idx]}).groupby("p"); return auc(g.y.max().values, g.s.max().values)
NF = {}
for fam in ["cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]:
    vals = []; arm_fold = {feats[i]: [] for i in ARM}
    for k in range(1, 6):
        te = np.where(fold == k)[0]
        if fam == "cnv_only":
            pipe = joblib.load(f"{R}/cnv_only/fold{k}/model.joblib"); basek = pat_auc(te, pipe.predict_proba(X_cnv[te])[:, 1])
            for fi in wsel:
                for r in range(NREP): Xp = X_cnv[te].copy(); Xp[:, fi] = Xp[rng.permutation(len(te)), fi]; vals.append(basek - pat_auc(te, pipe.predict_proba(Xp)[:, 1]))
            for fi in ARM:
                dd = []
                for r in range(10): Xp = X_cnv[te].copy(); Xp[:, fi] = Xp[rng.permutation(len(te)), fi]; dd.append(basek - pat_auc(te, pipe.predict_proba(Xp)[:, 1]))
                arm_fold[feats[fi]].append(r3(np.mean(dd)))
        else:
            m, Xc = load(fam, k); bl = [bags[ids[i]] for i in te]; basek = pat_auc(te, fwd(m, fam, bl, Xc[te]))
            for fi in wsel:
                for r in range(NREP): Xp = Xc[te].clone(); Xp[:, fi] = Xp[torch.from_numpy(rng.permutation(len(te))).to(DEV), fi]; vals.append(basek - pat_auc(te, fwd(m, fam, bl, Xp)))
            for fi in ARM:
                dd = []
                for r in range(10): Xp = Xc[te].clone(); Xp[:, fi] = Xp[torch.from_numpy(rng.permutation(len(te))).to(DEV), fi]; dd.append(basek - pat_auc(te, fwd(m, fam, bl, Xp)))
                arm_fold[feats[fi]].append(r3(np.mean(dd)))
        print("noise floor", fam, k, flush=True)
    vals = np.array(vals); NF[fam] = {"random_window_delta_auroc": {"mean": r3(vals.mean()), "sd": r3(vals.std()), "q95": r3(np.percentile(vals, 95)), "q99": r3(np.percentile(vals, 99)), "n_values": int(len(vals)), "windows": [feats[i] for i in wsel]}, "arm_features_per_fold_delta_auroc_mean_of_10_repeats": arm_fold}
F7["_8b_noise_floor"] = NF; F7["_method"] = "held-out rows per fold; ablations: CNV rows permuted jointly (50 repeats) or replaced by the standardised training mean (zeros); image bags permuted across rows (50 repeats) or replaced by the training-fold mean tile; delta = baseline patient AUROC minus ablated; CI from patient bootstrap of (baseline, mean ablated score); noise floor = permuting one of 10 random 5-Mb window features, 50 repeats, per fold"
json.dump(F7, open(AGG + "/f7_modality_ablation.json", "w"), indent=1); json.dump(F8, open(AGG + "/f8_train_vs_heldout.json", "w"), indent=1); print("GPU DONE", flush=True)
