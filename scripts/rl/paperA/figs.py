"""Paper A draft figures (CPU, aggregates only) -> O/figs/*.png. Reads clf/metrics.json, O/sft/results.json, O/enc_metrics.json,
O/curve_metrics.json, O/perclass.json, O/triage_oof_*.json. Seconds per report: jury 106 s (ten models, sequential Ollama;
analysis/rl_grader/RESULTS.md), MedGemma-27B 8.5 s (Ollama), RL 0.05 s (vLLM); encoder / masked SFT from the timing files."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pa_common import *
F = mkdir(O + "/figs"); J = lambda p: json.load(open(p)) if os.path.exists(p) else {}
C = {"jury": "#2a78d6", "enc": "#eb6834", "large": "#1baf7a", "sft": "#eda100", "rl": "#e87ba4", "27b": "#4a3aa7", "tfidf": "#008300", "oof": "#e34948", "grey": "#9a9a9a"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True})
clf = J(CLF + "/metrics.json")["test485"]; sft = J(O + "/sft/results.json").get("seeds", {}); enc = J(O + "/enc_metrics.json").get("test485", {}); cur = J(O + "/curve_metrics.json").get("tags", {})
pc = J(O + "/perclass.json").get("models", {}); tri = {k: J(O + f"/triage_oof_{k}.json") for k in ("none", "train_patients")}
def spr(p):
    f = p + "__test485.timing.json"; return J(f)["seconds"] / J(f)["reports"] if os.path.exists(f) else None
def pts(ax, x, ys, cis, color, label, ms=7):   # seeds as small hollow markers, mean as a filled one with the seed-0 CI
    for y, c in zip(ys, cis): ax.plot([x], [100 * y], "o", mfc="none", mec=color, ms=4, alpha=0.7)
    m = 100 * np.mean(ys); ax.errorbar([x], [m], yerr=[[m - 100 * cis[0][0]], [100 * cis[0][1] - m]], fmt="o", color=color, ms=ms, capsize=2, lw=1)
    ax.annotate(f"{label}\n{m:.1f}", (x, m), textcoords="offset points", xytext=(7, -3), fontsize=8, color="#333")

# 1. cost-accuracy frontier
fig, ax = plt.subplots(figsize=(6, 4))
pts(ax, 106, [clf["jury"]["exact_w"]], [clf["jury"]["exact_w_ci"]], C["jury"], "ten-model jury")
pts(ax, 8.5, [clf["medgemma27b_zs"]["exact_w"]], [clf["medgemma27b_zs"]["exact_w_ci"]], C["27b"], "MedGemma-27B zero-shot")
pts(ax, 0.05, [clf[f"rl_s{s}"]["exact_w"] for s in range(3)], [clf[f"rl_s{s}"]["exact_w_ci"] for s in range(3)], C["rl"], "4B SFT→GRPO")
if sft: pts(ax, spr(O + "/sft/preds/sftm_s0") or 0.086, [sft[str(s)]["sft_masked"]["exact_w"] for s in range(3)], [sft[str(s)]["sft_masked"]["exact_w_ci"] for s in range(3)], C["sft"], "4B masked SFT")
pts(ax, spr(CLF + "/preds/mbb32_s0") or 0.003, [clf[f"mbb32_s{s}"]["exact_w"] for s in range(3)], [clf[f"mbb32_s{s}"]["exact_w_ci"] for s in range(3)], C["enc"], "150M encoder")
L = [cur[f"large_s{s}"] for s in range(3) if f"large_s{s}" in cur]
if L: pts(ax, L[0].get("sec_per_report", 0.008), [x["exact_w"] for x in L], [x["exact_w_ci"] for x in L], C["large"], "400M encoder")
pts(ax, 0.006, [clf["tfidf_lr"]["exact_w"]], [clf["tfidf_lr"]["exact_w_ci"]], C["tfidf"], "TF-IDF + LR")
ax.set_xscale("log"); ax.set_xlim(1.5e-3, 1.5e3); ax.set_xlabel("seconds per report (one GPU, batched; log scale)"); ax.set_ylabel("weighted exact vs pathologist, % (n = 479)")
ax.set_title("Cost-accuracy frontier on the held-out human-graded sections", fontsize=10); fig.tight_layout(); fig.savefig(F + "/fig1_frontier.png", dpi=200); plt.close(fig)

# 2. triage curves (out-of-fold encoder vs jury vote fraction), full population and training patients removed
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, (k, t) in zip(axs, tri.items()):
    if not t: continue
    for m, lab_, col in (("jury_votefrac", "jury vote fraction", C["jury"]), ("enc_oof_s0", "encoder softmax (out-of-fold)", C["oof"]), ("p1_tokenprob", "RL token probability", C["rl"]), ("p1_random", "random order", C["grey"])):
        c = t["curves"].get(m)
        if c: ax.plot([100 * x for x in c["frac_reports"]], [100 * y for y in c["acc"]], color=col, lw=2, label=lab_)
    ax.axvline(20, color="#bbb", lw=0.8, ls="--"); ax.set_xlim(0, 50); ax.set_xlabel("reports sent for human review, %")
    ax.set_title({"none": f"full population ({t['n_reports']} reports)", "train_patients": f"training patients removed ({t['n_reports']} reports)"}[k], fontsize=9)
axs[0].set_ylabel("weighted accuracy on the 479, %"); axs[0].legend(frameon=False, fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(F + "/fig2_triage.png", dpi=200); plt.close(fig)

# 3. per-stratum bars: exact by jury vote-fraction band (weighted) and per-class recall (unweighted)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
bands = ["unanimous", "0.75-0.99", "<0.75"]; models = [("jury", "jury", C["jury"]), ("enc", "encoder (3 seeds)", C["enc"]), ("cf_oof_s0", "encoder out-of-fold", C["oof"]), ("rl_s0", "4B SFT→GRPO", C["rl"]), ("medgemma27b", "MedGemma-27B", C["27b"])]
def val(m, f):
    if m == "enc": return [np.mean([f(pc[f"enc_s{s}"]) for s in range(3)])]
    return [f(pc[m])] if m in pc else [np.nan]
ax = axs[0]; W = 0.16
for j, (m, lab_, col) in enumerate(models):
    ax.bar(np.arange(3) + (j - 2) * W, [100 * val(m, lambda d: d["band"][b]["exact_w"])[0] for b in bands], W - 0.02, color=col, label=lab_)
n = pc["jury"]["band"]; ax.set_xticks(range(3)); ax.set_xticklabels([f"{b}\n(n = {n[b]['n']})" for b in bands]); ax.set_xlabel("jury vote fraction on the section"); ax.set_ylabel("weighted exact agreement, %"); ax.set_ylim(30, 100); ax.legend(frameon=False, fontsize=7, loc="lower left")
ax = axs[1]
for j, (m, lab_, col) in enumerate(models):
    ax.bar(np.arange(6) + (j - 2) * W, [100 * val(m, lambda d: d["per_class"][g]["recall_unw"])[0] for g in GRADES], W - 0.02, color=col)
ax.set_xticks(range(6)); ax.set_xticklabels([f"{g}\n({pc['jury']['per_class'][g]['n_human']})" for g in GRADES], fontsize=7); ax.set_ylabel("recall, % (unweighted)"); ax.set_ylim(50, 101)
fig.tight_layout(); fig.savefig(F + "/fig3_strata.png", dpi=200); plt.close(fig)

# 4. teacher ablation
fig, ax = plt.subplots(figsize=(5.5, 3.6))
arms = [("juror1", "best single juror\n(qwen3-32b)"), ("top3", "top-3 consensus"), ("mbb32", "full ten-juror\nconsensus"), ("soft", "soft vote shares\n(+ low-agreement)")]
for j, (a, lab_) in enumerate(arms):
    src = clf if a == "mbb32" else enc; ys = [100 * src[f"{a}_s{s}"]["exact_w"] for s in range(3) if f"{a}_s{s}" in src]
    ax.bar(j, np.mean(ys), 0.6, color=C["enc"], alpha=0.85); ax.plot([j] * len(ys), ys, "o", mfc="none", mec="#333", ms=4); ax.text(j - 0.36, np.mean(ys) + 0.1, f"{np.mean(ys):.1f}", ha="left", fontsize=8)
ax.axhline(100 * clf["jury"]["exact_w"], color=C["jury"], ls="--", lw=1); ax.text(3.45, 100 * clf["jury"]["exact_w"] + 0.15, "jury", color=C["jury"], fontsize=8, ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels([a[1] for a in arms], fontsize=8); ax.set_ylim(88, 97); ax.set_ylabel("weighted exact on the 479, % (mean of 3 seeds)"); ax.set_title("Which teacher does the student need?", fontsize=10)
fig.tight_layout(); fig.savefig(F + "/fig4_teacher.png", dpi=200); plt.close(fig)

# 5. learning curve (training rows, by patient) and 6. temporal split
lc = {t: r for t, r in cur.items() if t.startswith("lc")}
if lc:
    import re
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6)); arms = {"e3": ("3 epochs (recipe)", C["enc"]), "e12": ("12 epochs", C["large"]), "const": ("constant ~2,700 steps", C["27b"])}
    for ax, key, yl in ((axs[0], "exact_w", "weighted exact vs pathologist, % (479)"), (axs[1], "test_agree", "agreement with the jury, % (test fold)")):
        G = {}   # arm -> frac -> [(n_train, value)]
        for t, r in lc.items():
            m = re.match(r"lc([0-9.]+)(?:e(\d+))?_s", t); frac, ep = float(m.group(1)), m.group(2)
            arm = "e3" if ep is None else ("e12" if ep == "12" and frac != 0.25 else "const")
            G.setdefault(arm, {}).setdefault(frac, []).append((r["train_log"]["n_train"], 100 * r[key]))
        for s_ in range(3):
            t = f"mbb32_s{s_}"
            if t in cur: G.setdefault("e3", {}).setdefault(1.0, []).append((14380, 100 * cur[t][key]))
        for arm, (lab_, col) in arms.items():
            if arm not in G: continue
            fr = sorted(G[arm]); xs = [np.mean([n for n, _ in G[arm][f]]) for f in fr]; ys = [np.mean([v for _, v in G[arm][f]]) for f in fr]
            ax.plot(xs, ys, "-o", color=col, lw=2, ms=6, label=lab_)
            for x, f in zip(xs, fr): ax.plot([x] * len(G[arm][f]), [v for _, v in G[arm][f]], "o", mfc="none", mec=col, ms=4, alpha=0.6)
        ax.set_xscale("log"); ax.set_xlabel("jury-labelled training sections (5-100% of patients, log)"); ax.set_ylabel(yl)
    axs[0].axhline(100 * clf["jury"]["exact_w"], color=C["jury"], ls="--", lw=1, label="jury"); axs[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(F + "/fig5_learning_curve.png", dpi=200); plt.close(fig)
tm = {t: r for t, r in cur.items() if t.startswith("y21") or t.startswith("lc0.5_") or t.startswith("mbb32")}
if any(t.startswith("y21") for t in tm):
    fig, ax = plt.subplots(figsize=(6, 3.6)); grp = [("y21", "trained on 2014-21 only"), ("lc0.5_", "random 50% of patients, all years"), ("mbb32", "all training rows, all years")]
    keys = [("exact_w", "479 human sections\n(all 2023-25)"), ("test_agree_>=2022", "jury test fold\n2022-25"), ("test_agree_<=2021", "jury test fold\n2014-21")]
    for j, (p, lab_) in enumerate(grp):
        rs = [r for t, r in tm.items() if t.startswith(p)]
        for k, (key, _) in enumerate(keys):
            ys = [100 * r[key] for r in rs]; ax.bar(k + (j - 1) * 0.27, np.mean(ys), 0.25, color=[C["oof"], C["grey"], C["enc"]][j], label=lab_ if k == 0 else None); ax.plot([k + (j - 1) * 0.27] * len(ys), ys, "o", mfc="none", mec="#333", ms=3)
    ax.set_xticks(range(3)); ax.set_xticklabels([k[1] for k in keys], fontsize=8); ax.set_ylim(85, 100); ax.set_ylabel("exact agreement, % (3 seeds)"); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3); ax.set_title("Temporal split", fontsize=10)
    fig.tight_layout(); fig.savefig(F + "/fig6_temporal.png", dpi=200); plt.close(fig)
print("figures:", sorted(os.listdir(F)))
