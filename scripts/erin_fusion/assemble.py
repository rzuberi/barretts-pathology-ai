"""Assembler for the ERIN imminent task set: OOF per arm, fold-local late fusion, metrics with patient-clustered
bootstrap, paired deltas, permutation p (T2a), tile-count confound, three figures, report. Idempotent."""
import glob, json, os, subprocess, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; Q = T + "/feasibility/erin_fusion/queue"; TK = T + "/feasibility/erin_fusion/tasks"
R = T + "/results/erin_progression_fusion"; FIG = R + "/figures"; os.makedirs(FIG, exist_ok=True)
TASKS = ["T1", "T2a", "T2b", "T2c", "T3a", "T3b", "T4"]; TEXT = {"T1", "T2a", "T2b", "T2c"}; NB = 2000
import sys; sys.path.insert(0, T + "/scripts"); from abmil_clf import patient_folds
def J(p):
    try: return json.load(open(p))
    except Exception: return None
def zfold(vals, folds_idx):
    out = np.zeros_like(vals)
    for te in folds_idx: v = vals[te]; out[te] = (v - v.mean()) / (v.std() + 1e-9)
    return out
def cboot(fn, pats, n=NB):
    up = sorted(set(pats)); idx = {}; [idx.setdefault(p, []).append(i) for i, p in enumerate(pats)]; rng = np.random.RandomState(0); out = []
    for _ in range(n):
        s = np.concatenate([idx[up[j]] for j in rng.randint(0, len(up), len(up))]); v = fn(s)
        if v is not None and np.isfinite(v): out.append(v)
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)] if out else [None, None]
def ops(y, s):
    fpr, tpr, thr = roc_curve(y, s); o = {}
    for se in (0.95, 1.0): k = np.where(tpr >= se - 1e-9)[0]; k = k[np.argmax(1 - fpr[k])]; o[f"spec_at_sens_{se}"] = round(float(1 - fpr[k]), 4)
    k = np.where(1 - fpr >= 0.80)[0]; k = k[np.argmax(tpr[k])]; o["sens_at_spec_0.80"] = round(float(tpr[k]), 4); return o
report = {"tasks": {}, "compute": {}}; forest = []
for t in TASKS:
    d = pd.read_csv(f"{TK}/{t}.csv", dtype={"sample_id": str, "anon_id": str}); keys = list(d.sample_id); pats = list(d.anon_id); y = d.y.astype(int).values; idx = {k: i for i, k in enumerate(keys)}
    folds = patient_folds(keys, dict(zip(keys, pats)), dict(zip(keys, y)), 5, seed=0); folds_idx = [[idx[k] for k in f] for f in folds]
    arms = {}; tab = J(f"{Q}/results/tab_{t}.json")
    if tab:
        for a, pv in tab["arms"].items(): arms[a] = np.array([pv[k] for k in keys])
    img_files = {f: [J(f"{Q}/results/img_{t}_f{f}_s{s}.json") for s in range(3)] for f in range(5)}
    n_img = sum(1 for f in range(5) for r in img_files[f] if r); ntiles = {}
    if n_img == 15:
        c = np.zeros(len(keys))
        for f in range(5):
            for k in folds[f]: c[idx[k]] = np.mean([r["preds"][k] for r in img_files[f]]); ntiles[k] = img_files[f][0]["ntiles"].get(k)
        arms["c"] = c
    if "b" in arms and "c" in arms: arms["d"] = (zfold(arms["b"], folds_idx) + zfold(arms["c"], folds_idx)) / 2
    res = {"n": len(keys), "pos": int(y.sum()), "neg": int((1 - y).sum()), "patients": len(set(pats)), "img_units_done": f"{n_img}/15", "arms": {}, "deltas": {}}
    for a, s in arms.items():
        sb = 1 / (1 + np.exp(-s)) if a == "d" else s
        res["arms"][a] = {"auroc": round(float(roc_auc_score(y, s)), 4), "auroc_ci": cboot(lambda i: roc_auc_score(y[i], s[i]) if len(set(y[i])) > 1 else None, pats),
                          "auprc": round(float(average_precision_score(y, s)), 4), "auprc_ci": cboot(lambda i: average_precision_score(y[i], s[i]) if len(set(y[i])) > 1 else None, pats),
                          "brier": round(float(brier_score_loss(y, np.clip(sb, 0, 1))), 4), **ops(y, s)}
    def delta(a1, a2, name):
        if a1 in arms and a2 in arms:
            res["deltas"][name] = {"mean": round(float(roc_auc_score(y, arms[a1]) - roc_auc_score(y, arms[a2])), 4), "ci": cboot(lambda i: roc_auc_score(y[i], arms[a1][i]) - roc_auc_score(y[i], arms[a2][i]) if len(set(y[i])) > 1 else None, pats)}
            forest.append((f"{t}: {name}", res["deltas"][name]["mean"], *res["deltas"][name]["ci"]))
    if "c" in arms and "a" in arms: delta("c", "a", "image - baseline")
    if "b" in arms and "a" in arms: delta("b", "a", "text - baseline")
    if "d" in arms and "b" in arms and "c" in arms:
        best = "b" if roc_auc_score(y, arms["b"]) >= roc_auc_score(y, arms["c"]) else "c"; delta("d", best, f"late_mean - best single ({best})")
    if ntiles and "c" in arms:
        tn = np.array([ntiles.get(k, np.nan) for k in keys]); ok = np.isfinite(tn); res["tilecount_confound_auroc"] = round(float(roc_auc_score(y[ok], tn[ok])), 4) if len(set(y[ok])) > 1 else None
    if t == "T2a":   # permutation null for c and d
        perms = {}
        for f in glob.glob(f"{Q}/results/perm_T2a_p*_f*.json"):
            r = J(f); m = re.search(r"p(\d+)_f(\d+)", f)
            if r: perms.setdefault(int(m.group(1)), {})[int(m.group(2))] = r["preds"]
        tp = J(f"{Q}/results/tabperm_T2a.json"); nullc, nulld = [], []
        for p, fd in perms.items():
            if len(fd) < 5: continue
            cp = np.zeros(len(keys)); yp = None
            for f in range(5):
                for k in folds[f]: cp[idx[k]] = fd[f][k]
            if tp:
                pr = next((x for x in tp["perms"] if x["perm"] == p), None)
                if pr: yp = np.array([pr["y_used"][k] for k in keys]); bp = np.array([pr["arms"]["b"][k] for k in keys]); nulld.append(roc_auc_score(yp, (zfold(bp, folds_idx) + zfold(cp, folds_idx)) / 2))
            if yp is None: yp = np.random.RandomState(1000 + p).permutation(y)
            nullc.append(roc_auc_score(yp, cp))
        if nullc and "c" in arms: res["perm_null_c"] = {"n": len(nullc), "mean": round(float(np.mean(nullc)), 4), "p95": round(float(np.percentile(nullc, 95)), 4), "p_empirical": round((1 + sum(v >= res["arms"]["c"]["auroc"] for v in nullc)) / (len(nullc) + 1), 4)}
        if nulld and "d" in arms: res["perm_null_d"] = {"n": len(nulld), "mean": round(float(np.mean(nulld)), 4), "p95": round(float(np.percentile(nulld, 95)), 4), "p_empirical": round((1 + sum(v >= res["arms"]["d"]["auroc"] for v in nulld)) / (len(nulld) + 1), 4)}
        if len(arms) >= 3:   # figures 1 and 3
            fig, ax = plt.subplots(figsize=(5, 5))
            for a, lab in (("a", "baseline (grade+age)"), ("b", "text (nomic)"), ("c", "image (ABMIL)"), ("d", "late-mean fusion")):
                if a in arms: fpr, tpr, _ = roc_curve(y, arms[a]); ax.plot(fpr, tpr, label=f"{lab} AUC {res['arms'][a]['auroc']:.3f}"); pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(f"{FIG}/roc_T2a_{a}.csv", index=False)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.set_title(f"T2a: LGD+ within 1 year of a benign report (n={len(y)}, pos={y.sum()})", fontsize=9); ax.legend(fontsize=8); fig.savefig(f"{FIG}/fig1_roc_T2a.png", dpi=150, bbox_inches="tight"); plt.close(fig)
            if "d" in arms:
                pr = 1 / (1 + np.exp(-arms["d"])); bins = pd.qcut(pr, 8, duplicates="drop"); tab_ = pd.DataFrame({"p": pr, "y": y, "b": bins}).groupby("b", observed=True).agg(mean_pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
                tab_.to_csv(f"{FIG}/reliability_T2a_d.csv"); fig, ax = plt.subplots(figsize=(4.5, 4.5)); ax.plot(tab_.mean_pred, tab_.obs, "o-"); ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("predicted (sigmoid of fused z)"); ax.set_ylabel("observed rate"); ax.set_title("T2a late-mean reliability (octiles)", fontsize=9); fig.savefig(f"{FIG}/fig3_reliability_T2a_d.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    report["tasks"][t] = res; print(t, json.dumps({k: v for k, v in res.items() if k in ("n", "pos", "img_units_done")}), {a: v["auroc"] for a, v in res["arms"].items()}, flush=True)
if forest:
    pd.DataFrame(forest, columns=["contrast", "mean", "lo", "hi"]).to_csv(f"{FIG}/fig2_forest_deltas.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(forest) + 1)); yy = np.arange(len(forest))[::-1]
    for yi, (lab, m, lo, hi) in zip(yy, forest): col = "#2a9d5c" if lo > 0 else ("#c44e52" if hi < 0 else "#4c72b0"); ax.plot([lo, hi], [yi, yi], color=col); ax.plot(m, yi, "o", color=col)
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(yy); ax.set_yticklabels([f[0] for f in forest], fontsize=8); ax.set_xlabel("paired AUROC difference (patient-clustered 95% CI)"); fig.savefig(f"{FIG}/fig2_forest_deltas.png", dpi=150, bbox_inches="tight"); plt.close(fig)
# compute accounting
try:
    out = subprocess.run(["sacct", "-u", "zuberi01", "-S", "2026-09-21T19:30", "-X", "-n", "-P", "-o", "JobName,Partition,AllocTRES,ElapsedRaw,State"], capture_output=True, text=True).stdout
    gpu_h = {}; 
    for l in out.splitlines():
        n, p, tres, el, st = l.split("|")[:5]
        if not n.startswith("efw"): continue
        g = re.search(r"gres/gpu=(\d+)", tres); gpu_h[p] = gpu_h.get(p, 0) + (int(g.group(1)) if g else 0) * float(el) / 3600
    report["compute"] = {"gpu_hours_by_partition": {k: round(v, 2) for k, v in gpu_h.items()}, "note": "allocated GPUs x wall time of efw_* worker jobs incl. idle time"}
except Exception as e: report["compute"] = {"error": str(e)}
report["task_partition_log"] = {os.path.basename(f)[:-5]: (J(f) or {}).get("_job") for f in glob.glob(f"{Q}/results/*.json")}
json.dump(report, open(f"{R}/results.json", "w"), indent=2, default=str); print("wrote", f"{R}/results.json")
