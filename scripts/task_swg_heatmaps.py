"""SWG attention / per-tile-risk maps, SAME slides for every release model family (Rehan, 21 Sep 2026).
For each chosen sample: load the fold-k models that never saw it (image_only AttentionMIL,
intermediate_fusion, coattention_fusion, early_fusion; state_dicts in the frozen release), compute
per-tile attention (image_only, intermediate, CNV-conditioned co-attention) and per-tile risk
(image_only classifier on each tile; early-fusion MLP fed one tile at a time with the sample's CNV),
render on the ndpi thumbnail, and tabulate Spearman agreement of tile rankings between models.
Cases: the 5 already-interpreted cases + top/missed progressors + top false positives by late_mean.
"""
import json, os, sys
import numpy as np, pandas as pd, torch, yaml
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training"
F = B + "/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"
sys.path.insert(0, B + "/multimodal-barretts-progression/src")
from barrett.models.image_mil import AttentionMIL
from barrett.models.intermediate_fusion import IntermediateABMILCNV
from barrett.models.coattention import CoAttentionABMILCNV
from barrett.models.early_fusion import EarlyFusionMLP
from barrett.training.data import load_cnv_matrix
OUT = os.environ.get("OUTDIR", "."); os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)
SLIDES = "/mnt/scratche/slow/fmlab/datasets/imaging/SWGCohort/slides"
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); fold_of = dict(zip(man.sample_id, man.fold_id_rep01.astype(int)))
uidx = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str); npz_of = dict(zip(uidx.sample_id, uidx.npz_path))
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID")
cnv_df, cnv_feats = load_cnv_matrix(F + "/feature_views/cnv"); cnv_df = cnv_df.set_index("sample_id")
oof = pd.concat([pd.read_csv(f, dtype={"sample_id": str, "patient_id": str}) for f in __import__("glob").glob(R + "/late_mean/fold*/outer_test_predictions.csv")])
oof = oof.set_index("sample_id")
# ---- choose samples ----
chosen = {"423": "prior", "712": "prior", "552": "prior", "533": "prior", "196": "prior"}
pos = oof[oof.y_true == 1].sort_values("y_prob"); neg = oof[oof.y_true == 0].sort_values("y_prob")
for s in pos.index[-3:]: chosen.setdefault(s, "progressor_top_prob")
for s in pos.index[:2]: chosen.setdefault(s, "progressor_missed_low_prob")
for s in neg.index[-2:]: chosen.setdefault(s, "nonprogressor_false_positive")
chosen = {s: t for s, t in chosen.items() if s in npz_of and s in fold_of}
print("samples:", chosen, flush=True)
def cfg(fam, k):
    y = yaml.safe_load(open(f"{R}/{fam}/fold{k}/resolved_config.yaml")); return y.get("selected_configuration", {})
def load_models(k):
    train_ids = man[man.fold_id_rep01.astype(int) != k].sample_id.astype(str)
    X = cnv_df.loc[[s for s in train_ids if s in cnv_df.index], cnv_feats].to_numpy(np.float32)
    med = np.nanmedian(X, 0); Xi = np.where(np.isfinite(X), X, med); mu, sd = Xi.mean(0), Xi.std(0); sd[sd == 0] = 1.0
    ms = {}
    for fam, cls in (("image_only", AttentionMIL), ("intermediate_fusion", IntermediateABMILCNV), ("coattention_fusion", CoAttentionABMILCNV), ("early_fusion", EarlyFusionMLP)):
        p = f"{R}/{fam}/fold{k}/model.pt"
        if not os.path.exists(p): print("missing", p); continue
        c = cfg(fam, k)
        if fam == "image_only": m = cls(in_dim=1536, hidden_dim=int(c.get("hidden_dim", 256)), attn_dim=int(c.get("attn_dim", 128)), dropout=float(c.get("dropout", 0.1)))
        elif fam == "early_fusion": m = cls(image_dim=1536, cnv_dim=len(cnv_feats), hidden_dim=int(c.get("hidden_dim", 512)), dropout=float(c.get("dropout", 0.2)))
        else: m = cls(image_dim=1536, cnv_dim=len(cnv_feats), img_hidden=int(c.get("img_hidden", 256)), cnv_hidden=int(c.get("cnv_hidden", 128)), attn_dim=int(c.get("attn_dim", 128)), fusion_hidden=int(c.get("fusion_hidden", 256)), dropout=float(c.get("dropout", 0.2)))
        sd_ = torch.load(p, map_location="cpu"); m.load_state_dict(sd_ if isinstance(sd_, dict) and "state_dict" not in sd_ else sd_["state_dict"]); ms[fam] = m.eval()
    return ms, (med, mu, sd)
try: import openslide
except Exception: openslide = None
models_cache = {}; rows = []; agree = []
for sid, why in chosen.items():
    k = fold_of[sid]
    if k not in models_cache: models_cache[k] = load_models(k)
    ms, (med, mu, sd) = models_cache[k]
    z = np.load(npz_of[sid], allow_pickle=True); bag = torch.tensor(np.asarray(z["embeddings"], np.float32)); coords = np.asarray(z["coords_level"]); lvl = int(z["level"]); ts = int(z["tile_size"])
    cnv = cnv_df.loc[sid, cnv_feats].to_numpy(np.float32); cnv = np.where(np.isfinite(cnv), cnv, med); cnv = torch.tensor(((cnv - mu) / sd).astype(np.float32)).unsqueeze(0)
    out = {}
    with torch.no_grad():
        if "image_only" in ms:
            m = ms["image_only"]; h = m.embed(bag); out["image_attention"] = m.attention_weights(bag).numpy(); out["image_tile_risk"] = torch.sigmoid(m.classifier(h)).squeeze(-1).numpy(); out["image_slide_prob"] = float(torch.sigmoid(m([bag])).item())
        if "intermediate_fusion" in ms: out["intermediate_attention"] = ms["intermediate_fusion"].attention_weights(bag).numpy(); out["intermediate_slide_prob"] = float(torch.sigmoid(ms["intermediate_fusion"]([bag], cnv)).item())
        if "coattention_fusion" in ms:
            m = ms["coattention_fusion"]; out["coattention_attention"] = m.attention_weights(bag, m.cnv_embed(cnv)[0]).numpy(); out["coattention_slide_prob"] = float(torch.sigmoid(m([bag], cnv)).item())
        if "early_fusion" in ms:
            m = ms["early_fusion"]; out["early_tile_risk"] = torch.sigmoid(m.classifier(torch.cat([bag, cnv.repeat(len(bag), 1)], 1))).squeeze(-1).numpy(); out["early_slide_prob"] = float(torch.sigmoid(m([bag], cnv)).item())
    rec = {"sample_id": sid, "why": why, "patient": str(coh.loc[sid, "PatientID_real"]).replace("/", "-") if sid in coh.index else "", "fold": k, "y_true": int(oof.loc[sid, "y_true"]), "late_mean_oof_prob": round(float(oof.loc[sid, "y_prob"]), 3),
           "grade_code": coh.loc[sid, "Label"] if sid in coh.index else "", "months_before_last_biopsy": coh.loc[sid, "MonthsBeforeLastBiopsy"] if sid in coh.index else "", "n_tiles": len(coords)}
    for kk, v in out.items():
        if kk.endswith("prob"): rec[kk] = round(v, 3)
        else: rec[kk + "_top10pct_share"] = round(float(np.sort(v)[-max(1, len(v) // 10):].sum() / v.sum()), 3)
    maps = {kk: v for kk, v in out.items() if not kk.endswith("prob")}; names = list(maps)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            agree.append({"sample_id": sid, "a": names[i], "b": names[j], "spearman": round(float(spearmanr(maps[names[i]], maps[names[j]])[0]), 3)})
    rows.append(rec)
    # render
    sp = os.path.join(SLIDES, os.path.basename(str(z["slide_path"]))); thumb = None
    if openslide is not None and os.path.exists(sp):
        try:
            sl = openslide.OpenSlide(sp); tl = sl.get_best_level_for_downsample(32); thumb = np.asarray(sl.read_region((0, 0), tl, sl.level_dimensions[tl]).convert("RGB"))
            scale = sl.level_downsamples[lvl] / sl.level_downsamples[tl]
        except Exception as e: print("openslide fail", sid, e, flush=True)
    if thumb is None:
        W, H = coords.max(0) + ts; scale = 1500.0 / max(W, H); thumb = np.full((int(H * scale) + 1, int(W * scale) + 1, 3), 255, np.uint8)
    xy = coords * scale; s_pt = max(4, (ts * scale) ** 2 / 3)
    panels = [(n, maps[n]) for n in names]; fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.2 * (len(panels) + 1), 4.4))
    axes[0].imshow(thumb); axes[0].set_title(f"{sid} {rec['patient']} y={rec['y_true']} late_mean p={rec['late_mean_oof_prob']}\n{why}", fontsize=8)
    for ax, (n, v) in zip(axes[1:], panels):
        ax.imshow(thumb, alpha=0.35); vv = v / v.max() if "attention" in n else v
        ax.scatter(xy[:, 0] + ts * scale / 2, xy[:, 1] + ts * scale / 2, c=vv, cmap="magma" if "attention" in n else "RdYlGn_r", vmin=0, vmax=1, s=s_pt, marker="s", linewidths=0)
        ax.set_title(f"{n}\nslide p={rec.get(n.split('_')[0] + '_slide_prob', '')}", fontsize=8)
    for ax in axes: ax.axis("off")
    fig.savefig(os.path.join(OUT, "maps", f"{sid}_{rec['patient']}_y{rec['y_true']}.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    np.savez(os.path.join(OUT, "maps", f"{sid}_tiles.npz"), coords=coords, **maps); print("rendered", sid, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "case_summary.csv"), index=False); ag = pd.DataFrame(agree); ag.to_csv(os.path.join(OUT, "tile_ranking_agreement.csv"), index=False)
summ = ag.groupby(["a", "b"]).spearman.agg(["mean", "min", "max"]).round(3).reset_index().to_dict("records") if len(ag) else []
json.dump({"samples": rows, "tile_ranking_spearman_by_model_pair": summ}, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=str); print(json.dumps(summ, indent=1))
