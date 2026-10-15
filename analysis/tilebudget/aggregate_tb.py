"""P3 (CPU): tile-budget study aggregation. Reads tilebudget/preds/fold{F}_{cfg}_{task}_seed{s}.npz, builds
(a) inference-budget surface: reference model (rk1200) x test rule x m x aggregator, T1 and T2, binary + six-class, with
    patient-clustered bootstrap CIs and paired deltas vs the reference setting (rand_u, m=1200, abmil);
(b) training-budget curve: cfg x matched test budget (random m = k for rk*, attn_top m = k for ak*, all_tiles for all) + at m=1200;
(c) aggregator comparison. Writes results.json, long CSVs, figures, SUMMARY.md. Primary endpoint fixed in DESIGN.md."""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/mnt/scratche/slow/fmlab/zuberi01"; ISBI = B + "/analysis/isbi_rerun"; A = B + "/analysis/tilebudget"; NB = 2000; RNG = np.random.RandomState(20261008)
CLS = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLS)}
MS = [1, 4, 16, 64, 256, 1200]; U_RULES = ["rand_u", "grid_u", "attn_top_u", "attn_bot_u", "mixed_u", "conch_u", "diverse_u"]; ALL_RULES = ["attn_top_all", "grid_all"]
CFGS = ["rk4", "rk16", "rk64", "rk256", "rk1200", "ak16", "ak64", "ak256", "all"]; AGGS = ["abmil", "meanlr", "max"]
s = pd.read_csv(ISBI + "/slides.csv", dtype={"idx": int}).fillna(""); s["vote_frac"] = pd.to_numeric(s.vote_frac, errors="coerce")
excl = set(pd.read_csv(ISBI + "/qc/exclusions.csv").idx); s = s[~s.idx.isin(excl)].set_index("idx")
test = s[s.fold >= 0]; ho = s[(s.fold == -1) & (s.human_grade != "")]
T1 = test[(test.vote_frac >= 0.75) & (test.worst_grade != "")].index.tolist(); T2 = ho.index.tolist()
def truth(ids, task, col):
    g = s.loc[ids, col]
    if task == "bin": return g.isin(["NORMAL_OTHER", "NDBE", "LGD", "HGD", "CANCER"]).values, g.isin(["LGD", "HGD", "CANCER"]).astype(int).values
    return g.isin(CLS).values, g.map(C_OF).fillna(-1).astype(int).values
def metric(task, P, y):
    if task == "bin": return roc_auc_score(y, P) if 0 < y.sum() < len(y) else np.nan
    v = [roc_auc_score((y == c).astype(int), P[:, c]) for c in range(6) if 0 < (y == c).sum() < len(y)]; return float(np.mean(v))
# ---- load: per (cfg, task) merge folds: test preds by fold model (OOF), holdout preds = mean over 5 folds x 3 seeds ----
def load(cfg, task):
    te, hoP, keys = {}, {}, None
    for f in sorted(glob.glob(f"{A}/preds/fold*_{cfg}_{task}_seed*.npz")):
        z = np.load(f); ks = [k for k in z.files if "|" in k]; keys = keys or sorted(set(k.split("|", 1)[1] for k in ks))
        for k in ks:
            setname, rest = k.split("|", 1); ids = z["test_idx"] if setname == "test" else z["holdout_idx"]; tgt = te if setname == "test" else hoP
            for i, p in zip(ids, z[k]): tgt.setdefault(rest, {}).setdefault(int(i), []).append(p)
    return {k: {i: np.mean(v, 0) for i, v in d.items()} for k, d in te.items()}, {k: {i: np.mean(v, 0) for i, v in d.items()} for k, d in hoP.items()}, keys
_W = {}
def boot_weights(pats):
    """NB patient-cluster bootstrap resamples as per-slide count weights (NB, n); cached per patient vector."""
    key = hash(pats.tobytes())
    if key not in _W:
        up, inv = np.unique(pats, return_inverse=True); P = len(up); W = np.zeros((NB, len(pats)), np.float32)
        for b in range(NB):
            cnt = np.bincount(RNG.randint(0, P, P), minlength=P); W[b] = cnt[inv]
        _W[key] = W
    return _W[key]
def wauc(score, y, W):
    """weighted AUROC for every bootstrap row of W (ties ignored); returns (NB,)"""
    o = np.argsort(score, kind="stable"); yo = y[o].astype(np.float32); Wo = W[:, o]; neg = Wo * (1 - yo)
    cumneg = np.cumsum(neg, axis=1) - neg; num = (Wo * yo * cumneg).sum(1); den = (Wo * yo).sum(1) * neg.sum(1)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
def boot_metric(task, P, y, W):
    if task == "bin": return wauc(P, y, W)
    cols = [wauc(P[:, c], (y == c).astype(int), W) for c in range(6) if 0 < (y == c).sum() < len(y)]; return np.nanmean(np.stack(cols), 0)
def ci(v): v = np.asarray(v, float); return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
R = {"design": "see DESIGN.md", "inference_budget": {}, "training_budget": {}, "n": {}}; rows = []
PRED = {}
for cfg in CFGS:
    for task in ("bin", "six"):
        te, hp, keys = load(cfg, task)
        if keys: PRED[(cfg, task)] = (te, hp, keys); print("loaded", cfg, task, len(keys), "settings", flush=True)
def evalset(cfg, task, T, key, boot=True, ref=None):
    te, hp, _ = PRED[(cfg, task)]; src = te if T == "T1" else hp; ids = T1 if T == "T1" else T2; col = "worst_grade" if T == "T1" else "human_grade"
    common = [i for i in ids if i in src.get(key, {}) and (ref is None or i in ref[0].get(ref[1], {}))]
    keep, y = truth(common, task, col); common = np.array(common)[keep]; y = y[keep]
    if len(common) < 10: return None
    P = np.stack([src[key][i] for i in common]); pats = s.loc[common, "anon_id"].values; m = metric(task, P, y); out = {"n": int(len(y)), "point": round(float(m), 4)}
    W = boot_weights(pats); v = boot_metric(task, P, y, W); out["ci"] = ci(v)
    if ref is not None:
        Pr = np.stack([ref[0][ref[1]][i] for i in common]); d = v - boot_metric(task, Pr, y, W)
        out["delta_vs_ref"] = {"point": round(float(m - metric(task, Pr, y)), 4), "ci": ci(d)}
    return out
REFKEY = "rand_u|1200|abmil"
# (a) inference-budget surface for the reference model
for task in ("bin", "six"):
    if ("rk1200", task) not in PRED: continue
    te, hp, keys = PRED[("rk1200", task)]
    for T in ("T1", "T2"):
        ref = (te if T == "T1" else hp, REFKEY)
        for key in keys:
            rule, m, agg = key.split("|"); r = evalset("rk1200", task, T, key, ref=ref)
            if r is None: continue
            R["inference_budget"][f"{task}|{T}|{key}"] = r; rows.append({"block": "inference", "cfg": "rk1200", "task": task, "set": T, "rule": rule, "m": int(m), "agg": agg, **{k: (v if not isinstance(v, (list, dict)) else json.dumps(v)) for k, v in r.items()}})
        print("inference surface", task, T, "done", flush=True)
# (b) training-budget: each cfg at its matched test budget and at the full U1200 random test, abmil
MATCH = {"rk4": ("rand_u", 4), "rk16": ("rand_u", 16), "rk64": ("rand_u", 64), "rk256": ("rand_u", 256), "rk1200": ("rand_u", 1200), "ak16": ("attn_top_u", 16), "ak64": ("attn_top_u", 64), "ak256": ("attn_top_u", 256), "all": ("all_tiles", 0)}
for task in ("bin", "six"):
    if ("rk1200", task) not in PRED: continue
    te_r, hp_r, _ = PRED[("rk1200", task)]
    for T in ("T1", "T2"):
        ref = (te_r if T == "T1" else hp_r, REFKEY)
        for cfg in CFGS:
            if (cfg, task) not in PRED: continue
            for lab, (rule, m) in {"matched": MATCH[cfg], "test_rand1200": ("rand_u", 1200)}.items():
                for agg in AGGS:
                    key = f"{rule}|{m}|{agg}"; r = evalset(cfg, task, T, key, ref=ref)
                    if r is None: continue
                    R["training_budget"][f"{task}|{T}|{cfg}|{lab}|{agg}"] = r; rows.append({"block": "training", "cfg": cfg, "task": task, "set": T, "rule": rule, "m": int(m), "agg": agg, "test_setting": lab, **{k: (v if not isinstance(v, (list, dict)) else json.dumps(v)) for k, v in r.items()}})
        print("training curve", task, T, "done", flush=True)
pd.DataFrame(rows).to_csv(f"{A}/results_long.csv", index=False); json.dump(R, open(f"{A}/results.json", "w"), indent=1)
# ---- primary endpoint: smallest m (random, abmil, reference model, T1 binary) whose delta-vs-1200 CI contains 0 ----
def smallest_ok(task, T, rule, agg="abmil"):
    for m in MS:
        r = R["inference_budget"].get(f"{task}|{T}|{rule}|{m}|{agg}")
        if r and r["delta_vs_ref"]["ci"][1] >= 0 and r["delta_vs_ref"]["ci"][0] <= 0: return m
        if r and r["delta_vs_ref"]["ci"][0] > 0: return m   # better than reference also counts
    return None
R["primary_endpoint"] = {f"{task}|{T}|{rule}": smallest_ok(task, T, rule) for task in ("bin", "six") for T in ("T1", "T2") for rule in U_RULES + ALL_RULES}
json.dump(R, open(f"{A}/results.json", "w"), indent=1)
# ---- figures ----
cols = {"rand_u": "#2563eb", "grid_u": "#0891b2", "attn_top_u": "#dc2626", "attn_bot_u": "#9ca3af", "mixed_u": "#f59e0b", "conch_u": "#7c3aed", "diverse_u": "#059669", "attn_top_all": "#991b1b", "grid_all": "#155e75"}
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ai, (task, T) in enumerate([("bin", "T1"), ("bin", "T2"), ("six", "T1"), ("six", "T2")]):
    ax = axes[ai // 2][ai % 2]
    for rule in U_RULES + ALL_RULES:
        pts = [(m, R["inference_budget"].get(f"{task}|{T}|{rule}|{m}|abmil")) for m in MS]; pts = [(m, r) for m, r in pts if r]
        if not pts: continue
        ax.plot([m for m, _ in pts], [r["point"] for _, r in pts], "o-", ms=3, color=cols[rule], label=rule); ax.fill_between([m for m, _ in pts], [r["ci"][0] for _, r in pts], [r["ci"][1] for _, r in pts], color=cols[rule], alpha=0.08)
    ra = R["inference_budget"].get(f"{task}|{T}|all_tiles|0|abmil")
    if ra: ax.axhline(ra["point"], color="k", ls="--", lw=1, label=f"all tiles ({ra['point']:.3f})")
    ax.set_xscale("log"); ax.set_xlabel("tiles per slide at test time"); ax.set_ylabel("AUROC" if task == "bin" else "macro AUROC"); ax.set_title(f"{'binary LGD+' if task=='bin' else 'six-class'} — {T} ({'jury truth' if T=='T1' else 'human truth'}), reference model", fontsize=10)
    if ai == 0: ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(f"{A}/inference_budget.png", dpi=160); plt.savefig(f"{A}/inference_budget.pdf"); plt.close()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, task in zip(axes, ("bin", "six")):
    for T, ls in (("T1", "-"), ("T2", "--")):
        for fam, cfgs, col in (("random k", ["rk4", "rk16", "rk64", "rk256", "rk1200"], "#2563eb"), ("attention-top k", ["ak16", "ak64", "ak256"], "#dc2626")):
            pts = [(int(c[2:]), R["training_budget"].get(f"{task}|{T}|{c}|matched|abmil")) for c in cfgs]; pts = [(k, r) for k, r in pts if r]
            if pts: ax.plot([k for k, _ in pts], [r["point"] for _, r in pts], "o" + ls, color=col, ms=4, label=f"{fam}, {T}")
        ra = R["training_budget"].get(f"{task}|{T}|all|matched|abmil")
        if ra: ax.axhline(ra["point"], color="k", ls=ls, lw=0.8, label=f"all tiles train+test, {T} ({ra['point']:.3f})")
    ax.set_xscale("log"); ax.set_xlabel("tiles per slide used for training (and testing, matched)"); ax.set_ylabel("AUROC" if task == "bin" else "macro AUROC"); ax.set_title("training budget — " + ("binary LGD+" if task == "bin" else "six-class"), fontsize=10); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{A}/training_budget.png", dpi=160); plt.savefig(f"{A}/training_budget.pdf"); plt.close()
# aggregator heatmap-ish table figure
fig, ax = plt.subplots(figsize=(9, 3.6)); task, T = "bin", "T1"; width = 0.27
for ai, agg in enumerate(AGGS):
    pts = [R["inference_budget"].get(f"{task}|{T}|rand_u|{m}|{agg}") for m in MS]; ax.bar(np.arange(len(MS)) + (ai - 1) * width, [r["point"] if r else 0 for r in pts], width, label=agg)
ax.set_xticks(np.arange(len(MS))); ax.set_xticklabels([str(m) for m in MS]); ax.set_ylim(0.5, 1.0); ax.set_xlabel("random tiles per slide at test time"); ax.set_ylabel("AUROC (T1, binary)"); ax.set_title("aggregator comparison, reference model", fontsize=10); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{A}/aggregators.png", dpi=160); plt.savefig(f"{A}/aggregators.pdf"); plt.close()
# ---- summary ----
def f(r): return f"{r['point']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" if r and "ci" in r else (f"{r['point']:.3f}" if r else "–")
def fd(r): return f"{r['delta_vs_ref']['point']:+.3f} [{r['delta_vs_ref']['ci'][0]:+.3f}, {r['delta_vs_ref']['ci'][1]:+.3f}]" if r and "delta_vs_ref" in r else "–"
L = ["# Tile-budget study — results (auto-generated)", "", f"Reference = arm-B ABMIL, random 1,200 tiles train and test. Primary endpoint (smallest test budget m whose paired difference to m=1,200 has a CI containing 0): {json.dumps(R['primary_endpoint'])}", ""]
for task in ("bin", "six"):
    for T in ("T1", "T2"):
        L += [f"## Inference budget — {task} {T} (reference model, ABMIL unless stated)", "", "| rule | " + " | ".join(f"m={m}" for m in MS) + " | all tiles |", "|---|" + "---|" * (len(MS) + 1)]
        for rule in U_RULES + ALL_RULES:
            L.append(f"| {rule} | " + " | ".join(f(R["inference_budget"].get(f"{task}|{T}|{rule}|{m}|abmil")) for m in MS) + f" | {f(R['inference_budget'].get(f'{task}|{T}|all_tiles|0|abmil'))} |")
        L += ["", "Δ vs reference (rand_u, 1,200):", "", "| rule | " + " | ".join(f"m={m}" for m in MS) + " |", "|---|" + "---|" * len(MS)]
        for rule in U_RULES + ALL_RULES: L.append(f"| {rule} | " + " | ".join(fd(R["inference_budget"].get(f"{task}|{T}|{rule}|{m}|abmil")) for m in MS) + " |")
        L += ["", "| aggregator (rand_u) | " + " | ".join(f"m={m}" for m in MS) + " |", "|---|" + "---|" * len(MS)]
        for agg in AGGS: L.append(f"| {agg} | " + " | ".join(f(R["inference_budget"].get(f"{task}|{T}|rand_u|{m}|{agg}")) for m in MS) + " |")
        L += ["", f"## Training budget — {task} {T}", "", "| config | matched test (ABMIL) | Δ vs ref | test rand 1,200 (ABMIL) | matched, mean+LR | matched, max |", "|---|---|---|---|---|---|"]
        for cfg in CFGS:
            g = lambda lab, agg: R["training_budget"].get(f"{task}|{T}|{cfg}|{lab}|{agg}")
            L.append(f"| {cfg} | {f(g('matched','abmil'))} | {fd(g('matched','abmil'))} | {f(g('test_rand1200','abmil'))} | {f(g('matched','meanlr'))} | {f(g('matched','max'))} |")
        L.append("")
open(f"{A}/SUMMARY.md", "w").write("\n".join(L)); print("\n".join(L[:40]))
for fn in os.listdir(A):
    p = os.path.join(A, fn)
    if os.path.isfile(p): os.chmod(p, 0o600)
