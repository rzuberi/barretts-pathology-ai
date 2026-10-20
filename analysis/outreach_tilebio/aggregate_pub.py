"""Phase 5 (CPU): aggregate preds/ -> results/*.json, figures 1-2. Bootstrap + wauc exactly as analysis/tilebudget/code/aggregate_tb.py
(vectorised weighted-rank AUROC over 2,000 resamples; no per-resample sklearn). One slide per patient in both datasets, so the
patient bootstrap = slide bootstrap. Primary endpoint fixed in DESIGN.md."""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; NB = 2000; MS = [1, 4, 16, 64, 256]
DSS, MODELS, ARMS, AGGS = ["tcga", "c16"], ["uni2h", "virchow2"], ["all", "r256"], ["abmil", "meanlr"]
RULES = ["rand", "attn_top", "attn_bot"]; NAMES = {"uni2h": "UNI2-h", "virchow2": "Virchow2", "tcga": "TCGA lung LUAD vs LUSC (diffuse)", "c16": "CAMELYON16 metastasis (focal)"}
SFX = "_smoke" if os.environ.get("SMOKE") == "1" else ""; PREDS, RES, FIG = A + "/preds" + SFX, A + "/results" + SFX, A + "/figures" + SFX
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
s = pd.read_csv(A + "/manifests/slides.csv").set_index("slide")
def load(ds, model, arm):
    """merge folds; average P over seeds -> {key: {slide: p}}"""
    acc = {}
    for f in sorted(glob.glob(f"{PREDS}/{ds}_{model}_{arm}_f*_s*.npz")):
        z = np.load(f); sl = z["test_slides"]
        for k in z.files:
            if k.count("|") == 2:
                for i, p in zip(sl, z[k]): acc.setdefault(k, {}).setdefault(str(i), []).append(float(p))
    return {k: {i: np.mean(v) for i, v in d.items()} for k, d in acc.items()}
def boot_weights(n, seed):
    rng = np.random.RandomState(seed); W = np.zeros((NB, n), np.float32)
    for b in range(NB): W[b] = np.bincount(rng.randint(0, n, n), minlength=n)
    return W
def wauc(score, y, W):
    """weighted AUROC for every bootstrap row of W (ties ignored); returns (NB,)  [from aggregate_tb.py]"""
    o = np.argsort(score, kind="stable"); yo = y[o].astype(np.float32); Wo = W[:, o]; neg = Wo * (1 - yo)
    cumneg = np.cumsum(neg, axis=1) - neg; num = (Wo * yo * cumneg).sum(1); den = (Wo * yo).sum(1) * neg.sum(1)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
def ci(v): v = np.asarray(v, float); return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
def score_sets(P, agg, rule, m):
    """list of score dicts: random = 3 independent draws (AUROC averaged over draws, never predictions)"""
    if rule == "all": return [P[f"{agg}|all|0"]]
    if rule == "rand": return [P[f"{agg}|rand{d}|{m}"] for d in range(3)]
    return [P[f"{agg}|{rule}|{m}"]]
def auc_and_boot(sets, ids, y, W):
    pts = [roc_auc_score(y, np.array([d[i] for i in ids])) for d in sets]; bs = [wauc(np.array([d[i] for i in ids]), y, W) for d in sets]
    return float(np.mean(pts)), np.mean(bs, 0)
R = {"design": "DESIGN.md", "n": {}, "curves": {}, "primary_endpoint": {}, "train_arm": {}, "encoder": {}, "lesion": {}, "hit_rate": {}, "cost": {}}
PRED = {(ds, mo, arm): load(ds, mo, arm) for ds in DSS for mo in MODELS for arm in ARMS}
PRED = {k: v for k, v in PRED.items() if v}; print("loaded", list(PRED), flush=True)
def entry(ds, sets, ref_sets, ids, y, W):
    p, b = auc_and_boot(sets, ids, y, W); out = {"auroc": round(p, 4), "ci": ci(b)}
    if ref_sets is not None:
        pr, br = auc_and_boot(ref_sets, ids, y, W); out["delta_vs_all"] = {"point": round(p - pr, 4), "ci": ci(b - br)}
    return out
for ds in DSS:
    keys = [k for k in PRED if k[0] == ds]
    if not keys: continue
    common = sorted(set.intersection(*[set(PRED[k]["abmil|all|0"]) for k in keys]))   # paired across models and arms
    y = s.loc[common, "label"].values.astype(int); W = boot_weights(len(common), 20261019 + DSS.index(ds)); R["n"][ds] = {"slides": len(common), "pos": int(y.sum())}
    for (_, mo, arm) in keys:
        P = PRED[(ds, mo, arm)]
        for agg in AGGS:
            ref = score_sets(P, agg, "all", 0); R["curves"][f"{ds}|{mo}|{arm}|{agg}|all|all"] = entry(ds, ref, None, common, y, W)
            for rule in RULES:
                for m in MS: R["curves"][f"{ds}|{mo}|{arm}|{agg}|{rule}|{m}"] = entry(ds, score_sets(P, agg, rule, m), ref, common, y, W)
        print("curves", ds, mo, arm, flush=True)
    # primary endpoint: ABMIL trained on ALL; smallest m whose delta-vs-all CI upper bound >= 0 (contains 0 or lies above it)
    for mo in MODELS:
        for rule in ("rand", "attn_top"):
            ok = [m for m in MS if (r := R["curves"].get(f"{ds}|{mo}|all|abmil|{rule}|{m}")) and r["delta_vs_all"]["ci"][1] >= 0]
            R["primary_endpoint"][f"{ds}|{mo}|{rule}"] = min(ok) if ok else "all"
    # training arm: r256 minus all at each test setting (ABMIL), paired
    for mo in MODELS:
        if (ds, mo, "r256") not in PRED or (ds, mo, "all") not in PRED: continue
        for rule, m in [("all", 0)] + [(r_, m_) for r_ in ("rand", "attn_top") for m_ in MS]:
            a_, ba = auc_and_boot(score_sets(PRED[(ds, mo, "r256")], "abmil", rule, m), common, y, W); b_, bb = auc_and_boot(score_sets(PRED[(ds, mo, "all")], "abmil", rule, m), common, y, W)
            R["train_arm"][f"{ds}|{mo}|{rule}|{m}"] = {"r256": round(a_, 4), "all": round(b_, 4), "delta": round(a_ - b_, 4), "ci": ci(ba - bb)}
    # encoder: Virchow2 minus UNI2-h (ABMIL, trained on all) at all tiles and at random 16 / 256
    if all((ds, mo, "all") in PRED for mo in MODELS):
        for rule, m in [("all", 0), ("rand", 16), ("rand", 256), ("attn_top", 16)]:
            a_, ba = auc_and_boot(score_sets(PRED[(ds, "virchow2", "all")], "abmil", rule, m), common, y, W); b_, bb = auc_and_boot(score_sets(PRED[(ds, "uni2h", "all")], "abmil", rule, m), common, y, W)
            R["encoder"][f"{ds}|{rule}|{m}"] = {"virchow2": round(a_, 4), "uni2h": round(b_, 4), "delta": round(a_ - b_, 4), "ci": ci(ba - bb)}
    # CAMELYON16: macro vs micro (each lesion class vs all normals)
    if ds == "c16":
        for les in ("macro", "micro"):
            ids = [i for i in common if s.loc[i, "label"] == 0 or s.loc[i, "lesion"] == les]; yy = s.loc[ids, "label"].values.astype(int); Wl = boot_weights(len(ids), 7 + len(les))
            R["n"][f"c16_{les}"] = {"slides": len(ids), "pos": int(yy.sum())}
            for mo in MODELS:
                if (ds, mo, "all") not in PRED: continue
                P = PRED[(ds, mo, "all")]; ref = score_sets(P, "abmil", "all", 0); R["lesion"][f"{les}|{mo}|all|all"] = entry(ds, ref, None, ids, yy, Wl)
                for rule in ("rand", "attn_top"):
                    for m in MS: R["lesion"][f"{les}|{mo}|{rule}|{m}"] = entry(ds, score_sets(P, "abmil", rule, m), ref, ids, yy, Wl)
                ok = [m for m in MS if R["lesion"][f"{les}|{mo}|rand|{m}"]["delta_vs_all"]["ci"][1] >= 0]; R["lesion"][f"{les}|{mo}|endpoint_rand"] = min(ok) if ok else "all"
        # attention-top-16 hit rate inside the tumour annotation (attention averaged over seeds) vs the random expectation (tumour fraction)
        les_js = RES + "/c16_lesion_tiles.json"
        if os.path.exists(les_js):
            L = json.load(open(les_js))
            for mo in MODELS:
                fs = sorted(glob.glob(f"{PREDS}/c16_{mo}_all_f0_s*.npz"))
                if not fs: continue
                zs = [np.load(f) for f in fs]; rows = []
                for sl, g in L.items():
                    if f"attn|{sl}" not in zs[0].files or g["n_tumour_tiles"] == 0: continue
                    a = np.mean([z[f"attn|{sl}"].astype(np.float32) for z in zs], 0); top = np.argsort(-a)[:16]; tum = set(g["tumour_idx"])
                    rows.append({"slide": sl, "lesion": g["lesion"], "tumour_frac": g["tumour_frac"], "n_tumour_tiles": g["n_tumour_tiles"], "hit16": float(np.mean([t in tum for t in top])), "any16": int(any(t in tum for t in top))})
                d = pd.DataFrame(rows)
                for les in ("macro", "micro", "all"):
                    q = d if les == "all" else d[d.lesion == les]
                    if len(q): R["hit_rate"][f"{mo}|{les}"] = {"n": len(q), "mean_hit16": round(q.hit16.mean(), 4), "slides_with_any_tumour_tile_in_top16": int(q.any16.sum()),
                                                               "median_tumour_frac": round(q.tumour_frac.median(), 5), "median_tumour_tiles": float(q.n_tumour_tiles.median()),
                                                               "p_random16_hits_any_median": round(float(np.median(1 - (1 - q.tumour_frac) ** 16)), 4)}
                d.to_csv(f"{RES}/c16_hit16_{mo}.csv", index=False)
# ---------------- cost ----------------
cr = []
for f in glob.glob(f"{A}/data/feats/cost/*/*.json"):
    c = json.load(open(f))
    for mo in MODELS:
        if mo in c and "gpu_s_all" in c[mo]: cr.append({"ds": c["ds"], "slide": c["slide"], "n": c["n_tiles"], "model": mo, "gpu": c[mo]["gpu"], **{k: c[mo][k] for k in ("gpu_s_16", "gpu_s_256", "gpu_s_all")}})
cd = pd.DataFrame(cr); cd.to_csv(RES + "/cost_per_slide.csv", index=False)
for (mo, gpu), q in cd.groupby(["model", "gpu"]):
    for ds in DSS + ["both"]:
        qq = q if ds == "both" else q[q.ds == ds]
        if len(qq) < 5: continue
        R["cost"][f"{mo}|{gpu}|{ds}"] = {"n_slides": len(qq), "median_tiles": float(qq.n.median()), "mean_tiles": round(float(qq.n.mean()), 1),
            **{f"{st}_{k}": round(float(getattr(qq[k], st)()), 4) for k in ("gpu_s_16", "gpu_s_256", "gpu_s_all") for st in ("median", "mean")},
            "tiles_per_s_all": round(float(qq.n.sum() / qq.gpu_s_all.sum()), 1), "saving_x_16_median": round(float((qq.gpu_s_all / qq.gpu_s_16).median()), 1),
            "saving_x_256_median": round(float((qq.gpu_s_all / qq.gpu_s_256).median()), 1)}
tl = []
for f in glob.glob(f"{A}/data/tiles/*/*.json"):
    m = json.load(open(f)); tl.append({"ds": m["ds"], "n": m["n_tiles"], "t_read_s": m["t_read_s"], "t_mask_s": m["t_mask_s"], "workers": m["read_workers"]})
tl = pd.DataFrame(tl)
for ds, q in tl.groupby("ds"): R["cost"][f"cpu_tiling|{ds}"] = {"n_slides": len(q), "median_tiles": float(q.n.median()), "mean_tiles": round(float(q.n.mean()), 1), "max_tiles": int(q.n.max()),
                                                              "total_tiles": int(q.n.sum()), "median_read_s": float(q.t_read_s.median()), "median_mask_s": float(q.t_mask_s.median()), "read_workers": int(q.workers.median())}
json.dump(R, open(RES + "/results.json", "w"), indent=1)
# ---------------- figure 1: AUROC vs budget ----------------
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6})
COL = {"uni2h": "#1f5fbf", "virchow2": "#d0571b"}; XS = MS + [1024]; XL = [str(m) for m in MS] + ["all"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
for ax, ds in zip(axes, DSS):
    for mo in MODELS:
        if (ds, mo, "all") not in PRED: continue
        ra = R["curves"][f"{ds}|{mo}|all|abmil|all|all"]
        for rule, ls, mk in (("rand", "-", "o"), ("attn_top", "--", "^")):
            pts = [R["curves"][f"{ds}|{mo}|all|abmil|{rule}|{m}"] for m in MS] + [ra]
            ax.plot(XS, [p["auroc"] for p in pts], ls=ls, marker=mk, ms=3, lw=1.1, color=COL[mo], label=f"{NAMES[mo]}, {'random' if rule == 'rand' else 'attention-top'}")
            ax.fill_between(XS, [p["ci"][0] for p in pts], [p["ci"][1] for p in pts], color=COL[mo], alpha=0.10 if rule == "rand" else 0.05, lw=0)
    ax.set_xscale("log", base=2); ax.set_xticks(XS); ax.set_xticklabels(XL); ax.minorticks_off(); ax.set_xlabel("tiles per slide at test time"); ax.set_ylabel("AUROC")
    n = R["n"].get(ds, {}); ax.set_title(f"{NAMES[ds]}\n(n = {n.get('slides')} slides, {n.get('pos')} positive)"); ax.grid(alpha=0.25, lw=0.4); ax.set_ylim(top=1.0)
axes[0].legend(loc="lower right", frameon=False)
plt.tight_layout(); [plt.savefig(f"{FIG}/fig1_auroc_vs_budget.{e}", dpi=300) for e in ("pdf", "png")]; plt.close()
# ---------------- figure 2: cost per slide ----------------
if len(cd):
    gpus = sorted(cd.gpu.unique(), key=lambda g: -(cd.gpu == g).sum()); g0 = gpus[0]; fig, ax = plt.subplots(figsize=(3.4, 2.4)); w = 0.38
    labs = ["16 tiles", "256 tiles", "all tiles\n(TCGA)", "all tiles\n(CAMELYON16)"]
    for k, mo in enumerate(MODELS):
        q = cd[(cd.model == mo) & (cd.gpu == g0)]
        if not len(q): continue
        v = [q.gpu_s_16.median(), q.gpu_s_256.median(), q[q.ds == "tcga"].gpu_s_all.median(), q[q.ds == "c16"].gpu_s_all.median()]
        b = ax.bar(np.arange(4) + (k - 0.5) * w, v, w, color=COL[mo], label=NAMES[mo])
        for x_, v_ in zip(np.arange(4) + (k - 0.5) * w, v): ax.text(x_, v_ * 1.12, f"{v_:.2g}", ha="center", fontsize=5.5)
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_ylabel(f"GPU-seconds per slide (median, {g0.replace('NVIDIA ', '')})"); ax.legend(frameon=False)
    ax.set_title("Embedding cost per slide (measured)"); ax.grid(axis="y", alpha=0.25, lw=0.4); plt.tight_layout(); [plt.savefig(f"{FIG}/fig2_cost_per_slide.{e}", dpi=300) for e in ("pdf", "png")]; plt.close()
print(json.dumps(R["primary_endpoint"], indent=1)); print(json.dumps({k: v for k, v in R["cost"].items()}, indent=1)[:3000])
