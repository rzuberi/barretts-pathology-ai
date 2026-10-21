"""E7 figures (local, not committed): paper/figures_explore_E7/. Reads the aggregate JSONs only."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = H + "/../../results/paper_final/explore_E7"; FD = H + "/../../paper/figures_explore_E7"; os.makedirs(FD, exist_ok=True)
B = json.load(open(R + "/base.json")); MG = json.load(open(R + "/merge.json"))
plt.rcParams.update({"font.size": 10})
COL = {"A1": "#4C78A8", "A12": "#72B7B2", "final": "#E45756", "final_infold": "#B279A2"}; LAB = {"A1": "memory (A1)", "A12": "memory + certificate (A12)", "final": "final (cn90 bar)", "final_infold": "final (bar re-chosen in fold)"}
# ---- 1. forest of Δ NP flagged vs R49 across stress tests
rows = [("All endoscopies (reference)", {k: B["all_patients"][k]["vs_R49"]["np_flagged"] for k in ("A1", "A12", "final", "final_infold")})]
T1 = B["test1_exposure_frozen"]
for nm, lab in (("np_matched_to_progressor_counts", "NP truncated to progressor counts (200 draws)"), ("first_2_everyone", "First 2 endoscopies, everyone"), ("first_3_everyone", "First 3 endoscopies, everyone")):
    rows.append((lab, {k: T1[nm][k]["vs_R49"]["np_flagged"] for k in ("A1", "A12", "final") if k in T1[nm]}))
FR = MG.get("test1_exposure_full_rerun", {})
if "matched" in FR: rows.append(("Matched truncation, selection rerun", {k: FR["matched"][k]["vs_R49"]["np_flagged"] for k in ("A1", "A12", "final", "final_infold")}))
for var, lab in (("both", "Order shuffled, selection rerun (100)"), ("deploy", "Order shuffled at deployment (100)")):
    if var in MG.get("test2_order_shuffle", {}): rows.append((lab, {k: MG["test2_order_shuffle"][var][k]["vs_R49"]["np_flagged"] for k in ("A1", "A12", "final", "final_infold")}))
for p in ("20pct", "40pct"):
    if p in MG.get("test6_missing_records", {}): rows.append((f"{p[:2]}% records missing at deployment (100)", {k: MG["test6_missing_records"][p]["deploy"][k]["vs_R49"]["np_flagged"] for k in ("A1", "A12", "final", "final_infold")}))
S3 = B["test3_prevalent_exclusion"]["drop_within_1y_of_event_and_np_within_1y_of_last"]; rows.append(("Endoscopies within 1 y of reference dropped", {k: S3[k]["vs_R49"]["np_flagged"] for k in ("A1", "A12", "final")}))
fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(rows) + 1.2)); yy = np.arange(len(rows))[::-1]
for i, (lab, d) in enumerate(rows):
    for q, k in enumerate(("A1", "A12", "final", "final_infold")):
        if k not in d: continue
        v = d[k]; yv = yy[i] + 0.27 - 0.18 * q; ax.plot(v["ci95"], [yv, yv], color=COL[k], lw=1.6); ax.plot(v["delta"], yv, "o", color=COL[k], ms=4.5, label=LAB[k] if i == 0 else None)
ax.axvline(0, color="grey", lw=0.8); ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlabel("Δ non-progressor patients ever flagged vs today's gate (R49), 95% CI")
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=9, frameon=False); fig.tight_layout(); fig.savefig(FD + "/e7_forest.png", dpi=200); plt.close(fig)
# ---- 2. check bar sweep
T5 = B["test5_bar_sweep"]; fig, ax = plt.subplots(figsize=(6.2, 4.4)); xs, ys, ls = [], [], []
for k, v in T5.items():
    if k.startswith("bar_"): xs.append(v["values"]["np_flagged"]); ys.append(v["values"]["caught"]); ls.append(k[4:])
ax.plot(xs, ys, "o-", color="#E45756", label="fixed bar (share of training progressor endoscopies above it)")
for i, (x, y_, l) in enumerate(zip(xs, ys, ls)): ax.annotate(f"{float(l):.0%}" if float(l) * 1000 % 10 == 0 else f"{float(l):.1%}", (x, y_), textcoords="offset points", xytext=(-6, 9) if i % 2 else (3, -16), fontsize=9)
for k, mk in (("infold_bar_slack0", "s"), ("infold_bar_slack1", "D")):
    v = T5[k]["values"]; ax.plot(v["np_flagged"], v["caught"], mk, color="#B279A2", ms=8, label={"infold_bar_slack0": "bar re-chosen in fold (no training loss)", "infold_bar_slack1": "bar re-chosen in fold (≤ 1 training loss)"}[k])
for k, lab, c in (("A12", "A12 (no check)", "#72B7B2"),):
    v = B["check"][k]; ax.plot(v["np_flagged"], v["caught"], "^", color=c, ms=8, label=lab)
ax.set_xlabel("Non-progressor patients ever flagged (of 43)"); ax.set_ylabel("Progressor patients caught (of 32)"); ax.set_ylim(27.8, 30.9); r = B["check"]["R49"]; ax.set_title(f"Today's gate (R49): {r['caught']} caught, {r['np_flagged']} NP flagged (off scale)", fontsize=9); ax.legend(fontsize=8.5, frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(FD + "/e7_bar_sweep.png", dpi=200); plt.close(fig)
# ---- 3. permutation null of the excess
T7 = MG.get("test7_permutation", {})
if "free" in T7:
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, kind, ttl in ((axs[0], "free", "labels permuted freely"), (axs[1], "strat", "permuted within endoscopy-count strata")):
        if kind not in T7: continue
        for k, c in (("final", "#E45756"), ("best_of_grid_excess", "black")):
            v = T7[kind][k]; ax.hist(v["null_excess_values"], bins=15, color=c, alpha=0.45 if k == "final" else 0.25, label=("nested final rule" if k == "final" else "best of 243 cells (optimistic)") + " under the null")
            ax.axvline(v["real_excess"], color=c, lw=2, ls="-" if k == "final" else "--", label=("real labels: nested final rule" if k == "final" else "real labels: best of 243 cells") if kind == "free" else None)
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("Δ share NP flagged − Δ share progressors caught")
    axs[0].set_ylabel("permutations"); axs[0].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.12, 1.0)); fig.tight_layout(); fig.savefig(FD + "/e7_null.png", dpi=200); plt.close(fig)
print("figures in", os.path.abspath(FD))
