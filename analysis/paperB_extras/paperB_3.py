"""Paper B extras, Analysis 3 (CPU): CONCH zero-shot vs supervised arm B on the ISBI patient-holdout sets T1/T2, per-class,
and the label-scale curve (arm B at 150/500/1000/all training patients). READ-ONLY on inputs. Patient-clustered bootstrap (2,000).
Zero-shot slide score = aggregation of tile probabilities under the EXISTING prompt ensemble (no variants tried): six-class = mean
tile probability vector; binary P(LGD+) = mean (primary; also max and top-10%) of tile P(LGD)+P(HGD)+P(CANCER), as in task_conch_zeroshot.py."""
import glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/mnt/scratche/slow/fmlab/zuberi01"; ISBI = B + "/analysis/isbi_rerun"; Z = B + "/analysis/paperB_extras/zeroshot"; os.makedirs(Z, exist_ok=True)
CLS = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLS)}; NB = 2000; RNG = np.random.RandomState(20261007)
MODE = os.environ.get("MODE", "subset"); SIZES = [25, 50, 100, 150, 500, 1000]
s = pd.read_csv(ISBI + "/slides.csv", dtype={"idx": int}).fillna(""); s["vote_frac"] = pd.to_numeric(s.vote_frac, errors="coerce")
excl = set(pd.read_csv(ISBI + "/qc/exclusions.csv").idx); s = s[~s.idx.isin(excl)].set_index("idx")
log = []
def say(*a): t = " ".join(str(x) for x in a); print(t, flush=True); log.append(t)
# ---- zero-shot slide scores ----
ZS = {}
for f in glob.glob(f"{Z}/conch/*.npz"):
    i = int(os.path.basename(f).split(".")[0])
    if i not in s.index: continue
    z = np.load(f); m = z["is_subset"] if MODE == "subset" else z["is_attn64"]; P = z["prob"][m].astype(np.float32); pl = P[:, 3:].sum(1)
    if len(P) == 0: continue
    topk = lambda v, k: float(np.sort(v)[-max(1, min(k, len(v))):].mean())
    ZS[i] = {"six": P.mean(0), "bin_mean": float(pl.mean()), "bin_max": float(pl.max()), "bin_top10": float(np.sort(pl)[-max(1, len(pl) // 10):].mean()),
             "bin_top1": topk(pl, 1), "bin_top5": topk(pl, 5), "bin_top10pct": float(np.sort(pl)[-max(1, len(pl) // 10):].mean()),
             "six_top1": np.array([topk(P[:, c], 1) for c in range(6)]), "six_top5": np.array([topk(P[:, c], 5) for c in range(6)]), "six_top10pct": np.array([float(np.sort(P[:, c])[-max(1, len(P) // 10):].mean()) for c in range(6)]),
             "n_tiles": len(pl)}
say(f"zero-shot ({MODE}) slides scored: {len(ZS)} of {len(s)} QC-passed ({len(ZS)/len(s):.1%}); tiles/slide mean {np.mean([v['n_tiles'] for v in ZS.values()]):.0f}")
# ---- supervised preds ----
def load(pattern):
    te, ho = {}, {}
    for f in glob.glob(pattern):
        z = np.load(f)
        for i, p in zip(z["test_idx"], z["test_prob"]): te.setdefault(int(i), []).append(p)
        for i, p in zip(z["holdout_idx"], z["holdout_prob"]): ho.setdefault(int(i), []).append(p)
    return {k: np.mean(v, 0) for k, v in te.items()}, {k: np.mean(v, 0) for k, v in ho.items()}
SUP = {("all", t): load(ISBI + f"/preds/patient_B_{t}_seed*_fold*.npz") for t in ("bin", "six")}
for n in SIZES:
    for t in ("bin", "six"): SUP[(n, t)] = load(ISBI + f"/preds/lc_B_{t}_n{n}_seed*_fold*.npz")
say("learning-curve preds found:", {k: (len(v[0]), len(v[1])) for k, v in SUP.items()})
# ---- evaluation sets (identical to aggregate.py) ----
test = s[s.fold >= 0]; ho = s[(s.fold == -1) & (s.human_grade != "")]
SETS = {"T1": (test[(test.vote_frac >= 0.75) & (test.worst_grade != "")].index.tolist(), "worst_grade", 0), "T2": (ho.index.tolist(), "human_grade", 1)}
def truth(ids, task, col):
    g = s.loc[ids, col]
    if task == "bin": return g.isin(["NORMAL_OTHER", "NDBE", "LGD", "HGD", "CANCER"]).values, g.isin(["LGD", "HGD", "CANCER"]).astype(int).values
    return g.isin(CLS).values, g.map(C_OF).fillna(-1).astype(int).values
def macro(P, y): v = [roc_auc_score((y == c).astype(int), P[:, c]) for c in range(6) if 0 < (y == c).sum() < len(y)]; return float(np.mean(v))
def metric(task, P, y): return roc_auc_score(y, P) if task == "bin" else macro(P, y)
def groups(pats):
    up = np.unique(pats); g = {p: np.where(pats == p)[0] for p in up}
    for _ in range(NB): yield np.concatenate([g[p] for p in RNG.choice(up, len(up), replace=True)])
def ci(v): v = np.asarray(v, float); return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
def assemble(T, task, sup_key, zs_key="bin_mean"):
    ids, col, src = SETS[T]; sup = SUP[sup_key][src]
    common = [i for i in ids if i in sup and i in ZS]; keep, y = truth(common, task, col); common = np.array(common)[keep]; y = y[keep]
    Ps = np.stack([sup[i] for i in common]); Pz = np.stack([ZS[i][zs_key if zs_key != "bin_mean" else "six"] for i in common]) if task == "six" else np.array([ZS[i][zs_key] for i in common])
    return y, s.loc[common, "anon_id"].values, Ps, Pz, common
R = {"mode": MODE, "n_zeroshot_slides": len(ZS), "prompts": "existing ensemble from task_conch_zeroshot.py, unchanged; no variants tried", "aggregation": "six-class: mean tile probabilities; binary: mean tile P(LGD+) (primary), max and top-10% reported", "zeroshot_vs_B": {}, "per_class": {}, "label_scale": {}}
for T in ("T1", "T2"):
    for task in ("bin", "six"):
        y, pats, Ps, Pz, common = assemble(T, task, ("all", task)); ms, mz = metric(task, Ps, y), metric(task, Pz, y); d = []
        for ix in groups(pats): d.append(metric(task, Ps[ix], y[ix]) - metric(task, Pz[ix], y[ix]))
        r = {"n": int(len(y)), "n_patients": int(len(np.unique(pats))), "arm_B": round(ms, 4), "zeroshot": round(mz, 4), "B_minus_zeroshot": {"point": round(ms - mz, 4), "ci": ci(d)}}
        variants = ("bin_mean", "bin_top1", "bin_top5", "bin_top10pct", "bin_max") if task == "bin" else ("six", "six_top1", "six_top5", "six_top10pct")
        r["zeroshot_variants"] = {}
        for k in variants:
            Pv = np.array([ZS[i][k] for i in common]) if task == "bin" else np.stack([ZS[i][k] for i in common]); mv = metric(task, Pv, y); vv = [metric(task, Pv[ix], y[ix]) for ix in groups(pats)]
            r["zeroshot_variants"][k] = {"point": round(float(mv), 4), "ci": ci(vv)}
        R["zeroshot_vs_B"][f"{T}_{task}"] = r; say(T, task, r)
        if task == "six":
            pc = {}
            for c in range(6):
                yy = (y == c).astype(int)
                if not 0 < yy.sum() < len(yy): continue
                a_s, a_z = roc_auc_score(yy, Ps[:, c]), roc_auc_score(yy, Pz[:, c]); dd = []
                for ix in groups(pats):
                    y2 = yy[ix]; dd.append(roc_auc_score(y2, Ps[ix, c]) - roc_auc_score(y2, Pz[ix, c]) if 0 < y2.sum() < len(y2) else np.nan)
                pc[CLS[c]] = {"n": int(yy.sum()), "arm_B": round(a_s, 4), "zeroshot": round(a_z, 4), "B_minus_zeroshot": {"point": round(a_s - a_z, 4), "ci": ci(dd)}}
            R["per_class"][T] = pc
# ---- label-scale curve ----
NPAT = {25: 25, 50: 50, 100: 100, 150: 150, 500: 500, 1000: 1000, "all": 1168}   # "all" = mean training patients per fold job (1,166-1,171)
for T in ("T1", "T2"):
    for task in ("bin", "six"):
        rows = []; yA, patsA, PA, _, cA = assemble(T, task, ("all", task)); allmap = dict(zip(cA, PA))
        for n in SIZES + ["all"]:
            if not SUP[(n, task)][0]: continue
            y, pats, Ps, Pz, common = assemble(T, task, (n, task)); m = metric(task, Ps, y); mz = metric(task, Pz, y); v, d, da = [], [], []
            Pall = np.stack([allmap[i] for i in common]); ma = metric(task, Pall, y)
            for ix in groups(pats): a = metric(task, Ps[ix], y[ix]); v.append(a); d.append(a - metric(task, Pz[ix], y[ix])); da.append(a - metric(task, Pall[ix], y[ix]))
            rows.append({"n_train_patients": NPAT[n] if n != "all" else NPAT["all"], "label": str(n), "n_eval": int(len(y)), "auroc": round(m, 4), "ci": ci(v), "zeroshot": round(mz, 4), "sup_minus_zs": {"point": round(m - mz, 4), "ci": ci(d)}, "minus_all": {"point": round(m - ma, 4), "ci": ci(da)}})
        first = next((r["label"] for r in rows if r["sup_minus_zs"]["ci"][0] > 0), None)
        below_all = [r["label"] for r in rows if r["label"] != "all" and r["minus_all"]["ci"][1] < 0]
        first_indist = next((r["label"] for r in rows if r["label"] != "all" and r["minus_all"]["ci"][1] >= 0), None)
        R["label_scale"][f"{T}_{task}"] = {"rows": rows, "smallest_size_beating_zeroshot_CI_excl_0": first, "sizes_significantly_below_all": below_all, "first_size_indistinguishable_from_all": first_indist}
        say("label scale", T, task, [(r["label"], r["auroc"], r["minus_all"]) for r in rows], "| below all:", below_all, "| first ≈ all:", first_indist)
pd.DataFrame([{"set_task": k, **{kk: (vv if not isinstance(vv, dict) else json.dumps(vv)) for kk, vv in v.items()}} for k, v in R["zeroshot_vs_B"].items()]).to_csv(f"{Z}/zeroshot_vs_B_{MODE}.csv", index=False)
pd.DataFrame([{"set": T, "class": c, **{kk: (vv if not isinstance(vv, dict) else json.dumps(vv)) for kk, vv in v.items()}} for T, pc in R["per_class"].items() for c, v in pc.items()]).to_csv(f"{Z}/per_class_{MODE}.csv", index=False)
pd.DataFrame([{"set_task": k, **{kk: (vv if not isinstance(vv, dict) else json.dumps(vv)) for kk, vv in r.items()}} for k, v in R["label_scale"].items() for r in v["rows"]]).to_csv(f"{Z}/label_scale_{MODE}.csv", index=False)
json.dump(R, open(f"{Z}/results_{MODE}.json", "w"), indent=1)
# ---- figures ----
if "T2" in R["per_class"]:
    pc = R["per_class"]["T2"]; names = list(pc); x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [pc[c]["zeroshot"] for c in names], w, color="#9ca3af", label=f"CONCH zero-shot ({MODE})"); ax.bar(x + w / 2, [pc[c]["arm_B"] for c in names], w, color="#2563eb", label="supervised arm B (per-section labels)")
    for i, c in enumerate(names): ax.text(i, 0.3, f"n={pc[c]['n']}", ha="center", fontsize=7, color="w")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(0.3, 1.0); ax.set_ylabel("one-vs-rest AUROC"); ax.set_title("T2 (human-labelled holdout): zero-shot vs supervised, per class", fontsize=10); ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{Z}/zeroshot_vs_supervised_{MODE}.png", dpi=160); plt.savefig(f"{Z}/zeroshot_vs_supervised_{MODE}.pdf"); plt.close()
fig, axes = plt.subplots(1, 2, figsize=(10, 4)); cols = {"bin": "#2563eb", "six": "#f59e0b"}
for ax, T in zip(axes, ("T1", "T2")):
    for task in ("bin", "six"):
        rr = R["label_scale"].get(f"{T}_{task}", {}).get("rows", [])
        if not rr: continue
        xs = [r["n_train_patients"] for r in rr]; ys = [r["auroc"] for r in rr]; lo = [r["ci"][0] for r in rr]; hi = [r["ci"][1] for r in rr]
        ax.plot(xs, ys, "o-", color=cols[task], label=f"arm B, {'binary LGD+' if task=='bin' else 'six-class macro'}"); ax.fill_between(xs, lo, hi, color=cols[task], alpha=0.15)
        ax.axhline(rr[0]["zeroshot"], color=cols[task], ls="--", lw=1, label=f"zero-shot, {'binary' if task=='bin' else 'six-class'} ({rr[0]['zeroshot']:.3f})")
    ax.set_xscale("log"); ax.set_xlabel("labelled training patients per fold (log)"); ax.set_ylabel("AUROC"); ax.set_title(f"{T}: {'own-section jury truth' if T=='T1' else 'human truth'}", fontsize=10); ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(f"{Z}/label_scale_curve_{MODE}.png", dpi=160); plt.savefig(f"{Z}/label_scale_curve_{MODE}.pdf"); plt.close()
open(f"{Z}/run3_{MODE}_log.txt", "w").write("\n".join(log))
for f in os.listdir(Z):
    p = os.path.join(Z, f)
    if os.path.isfile(p): os.chmod(p, 0o600)
