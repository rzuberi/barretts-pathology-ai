"""Review round 9 figures (local, not committed): PNG 200 dpi, fonts >= 9 pt, written to paper/figures_r9/ from the aggregate JSONs of results/paper_final/review_round9/
(merge.json, f0.json, icc.json, curves.json). RV9_RES overrides the input directory (smoke tests).
  r9_forest.png    ranked forest of the change in non-progressor patients ever flagged against R49 (nested rows), change in progressors caught annotated
  r9_scatter.png   progressors caught against non-progressors flagged for every method, with R49, P1, P3 and grade marked
  r9_f0.png        F0: later dysplasia among flagged and unflagged non-progressors
  r9_icc.png       per-patient ICC of endoscopy scores
  r9_curves.png    cumulative false-flag curves by endoscopy index"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
T = "/Users/zuberi01/Documents/thesis"; RES = os.environ.get("RV9_RES", T + "/results/paper_final/review_round9"); OUTD = os.environ.get("RV9_FIGS", T + "/paper/figures_r9"); os.makedirs(OUTD, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
J = lambda n: json.load(open(f"{RES}/{n}.json"))
GC = {"A1 memory, H&E alone": "#4C78A8", "A1 memory, gate score": "#F58518", "A2/A2' certificate": "#54A24B", "A3 agreement": "#B279A2", "B1/B2 stacked": "#E45756", "B3 abstention": "#72B7B2",
      "A4 confirm-by-sequencing": "#8C564B", "B4 refit": "#9D755D", "sample aggregation": "#BAB0AC", "combined (A1+A2, B1+A1+A2)": "#D4A017", "control (threshold only / re-tuned)": "#000000"}
def group(m):
    f = m["family"]
    if f in ("C1", "B1:img") or f.endswith(":none"): return "control (threshold only / re-tuned)"
    if f.startswith("A4"): return "A4 confirm-by-sequencing"
    if f.startswith("A1:img:"): return "A1 memory, H&E alone"
    if f.startswith(("A1:gate", )): return "A1 memory, gate score"
    if f.startswith("A1:b1_al") or f.startswith("B1:"): return "B1/B2 stacked"
    if f.startswith("A2"): return "A2/A2' certificate"
    if f == "A3": return "A3 agreement"
    if f == "B3": return "B3 abstention"
    if f.startswith("B4"): return "B4 refit"
    if f.startswith("G:"): return "sample aggregation"
    if f in ("A12", "A12p", "B2A2"): return "combined (A1+A2, B1+A1+A2)"
    return "sample aggregation"
def short(m):
    n = m["name"].replace(" [nested]", "").replace(" [nested, pAUC selection]", " [pAUC]").replace("exponentially weighted mean", "EW mean").replace("gate score, R49-aligned", "gate (aligned)").replace("gate score, literal", "gate (literal)")
    n = n.replace("A4 confirm-by-sequencing | gate with the high band also sequenced (score 0 = flag boundary)", "A4 confirm gate (high band sequenced)")
    n = n.replace("A1 | ", "").replace("Sample aggregation | ", "").replace("stacked B1 gate score, aligned", "B1 gate (aligned)").replace("consecutive-2 or t_high", "consec-2 or t_high").replace("A4 confirm-by-sequencing | gate with the high band also sequenced (score 0 = flag boundary)", "A4 confirm gate (high band sequenced)")
    return n if len(n) <= 78 else n[:76] + ".."
def fig_forest(M):
    rows = [m for m in M["methods"] if m["kind"] in ("nested", "nested_pauc")]; key = lambda m: (0 if m["vs_R49"]["caught"]["delta"] >= -0.5 else 1, m["vs_R49"]["np_flagged"]["delta"]); rows = sorted(rows, key=key)[:48]
    fig, ax = plt.subplots(figsize=(11.5, 0.27 * len(rows) + 1.8)); yy = np.arange(len(rows))[::-1]
    for y, m in zip(yy, rows):
        d = m["vs_R49"]["np_flagged"]; lo, hi = d["ci95"]; l99, h99 = d["ci99_bonferroni"]; ok = m["vs_R49"]["caught"]["delta"] >= -0.5; c = GC[group(m)] if ok else "#B0B0B0"
        ax.plot([l99, h99], [y, y], color=c, lw=1.0, alpha=0.6, zorder=1); ax.plot([lo, hi], [y, y], color=c, lw=3.0, zorder=2); ax.plot(d["delta"], y, "o", color=c, ms=6, zorder=3, mec="white", mew=0.6)
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(yy); ax.set_yticklabels([short(m) for m in rows], fontsize=9); ax.set_xlabel("Change in non-progressor patients ever flagged vs R49 (of 43; negative = fewer)")
    x1 = ax.get_xlim()[1]; xr = ax.get_xlim()[1] - ax.get_xlim()[0]; ax.set_xlim(ax.get_xlim()[0], x1 + 0.34 * xr); xt = x1 + 0.02 * xr
    for y, m in zip(yy, rows): ax.text(xt, y, f"Δ caught {m['vs_R49']['caught']['delta']:+.1f}   seq {m['values']['seq']:.2f}", va="center", fontsize=9, color="#333333")
    ax.set_ylim(-0.8, len(rows) - 0.2); h = [plt.Line2D([0], [0], color=c, lw=3) for c in GC.values()] + [plt.Line2D([0], [0], color="#B0B0B0", lw=3)]
    fig.legend(h, list(GC) + ["Δ caught < -0.5 (ranked last)"], loc="lower center", ncol=4, frameon=False, fontsize=9)
    ax.set_title("Nested change in flagged non-progressors vs R49 (thick 95% CI, thin 99.2%)"); fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{OUTD}/r9_forest.png", dpi=200); plt.close(fig)
def fig_scatter(M):
    R8 = json.load(open(T + "/results/paper_final/review_round8/l1.json"))["L1"]["cells"]; ref = {}
    for c in R8:
        if (c["family"], c["label"]) == ("P3", "90%"): ref["P3 90%"] = (c["non_progressor_patients_ever_flagged"], c["progressor_patients_caught"])
        if (c["family"], c["label"]) == ("P10", "grade ID/LGD (all pre-event endoscopies)"): ref["grade"] = (c["non_progressor_patients_ever_flagged"], c["progressor_patients_caught"])
    refs = {m["family"]: m["values"] for m in M["methods"] if m["kind"] == "reference"}; ref["R49"] = (refs["R49"]["np_flagged"], refs["R49"]["caught"]); ref["P1 90%"] = (refs["P1 90%"]["np_flagged"], refs["P1 90%"]["caught"])
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1, 1.9]})
    for ax, zoom in zip(axs, (False, True)):
        for kd, mk, al, sz in (("grid_point", ".", 0.35, 14), ("matched", "x", 0.5, 18), ("nested_pauc", "^", 0.8, 24), ("nested", "o", 0.95, 34)):
            for m in M["methods"]:
                if m["kind"] != kd: continue
                ax.scatter(m["values"]["np_flagged"], m["values"]["caught"], s=sz, marker=mk, color=GC[group(m)], alpha=al, linewidths=0.8, zorder=2 if kd != "nested" else 3)
        for nm, (x, yv), mk in (("P1 90%", ref["P1 90%"], "s"), ("P3 90%", ref["P3 90%"], "D"), ("grade", ref["grade"], "v"), ("R49", ref["R49"], "*")):
            ax.scatter(x, yv, s=260 if nm == "R49" else 80, marker=mk, color="white", edgecolor="k", linewidths=1.6, zorder=5)
            if (not zoom and nm in ("grade",)) or (zoom and nm != "grade"): ax.annotate(nm, (x, yv), textcoords="offset points", xytext={"R49": (-34, 9), "P3 90%": (9, -14), "P1 90%": (9, 6), "grade": (7, 6)}[nm], fontsize=9)
        ax.set_xlabel("Non-progressor patients ever flagged (of 43)"); ax.set_ylabel("Progressor patients caught (of 32)")
        if zoom:
            ax.set_xlim(17, 41); ax.set_ylim(28.0, 32.2); ax.axhline(ref["R49"][1] - 0.5, color="k", lw=0.6, ls=":"); ax.text(17.3, ref["R49"][1] - 0.46, "R49 caught - 0.5", fontsize=9); ax.set_title("Zoom: the region of the ranking")
        else: ax.set_title("Full range (grade rule for scale)")
        ax.grid(alpha=0.2)
    h = [plt.Line2D([0], [0], marker="o", ls="", color=c, ms=7) for c in GC.values()] + [plt.Line2D([0], [0], marker=".", ls="", color="grey", ms=7), plt.Line2D([0], [0], marker="x", ls="", color="grey", ms=7), plt.Line2D([0], [0], marker="^", ls="", color="grey", ms=7)]
    fig.legend(h, list(GC) + ["grid point (apparent)", "evaluation-matched (optimistic)", "nested, pAUC selection"], loc="lower center", fontsize=9, ncol=5, framealpha=0.95)
    axs[0].set_xlim(15, 42); axs[0].set_ylim(15.5, 32.5); fig.suptitle("Round 9: every method, held-out patients (hollow markers: R49, P1 90%, P3 90%, grade)", fontsize=11); fig.tight_layout(rect=(0, 0.14, 1, 0.97)); fig.savefig(f"{OUTD}/r9_scatter.png", dpi=200); plt.close(fig)
def num(v): return np.nan if isinstance(v, str) or v is None else v
def fig_f0(F):
    rules = list(F["rules"]); fig, axs = plt.subplots(1, 3, figsize=(14, 4.8)); titles = [("all_43:later_IND_or_worse", "Later IND or worse (all 43)"), ("all_43:later_LGD_or_worse", "Later LGD or worse (all 43)"), ("any", "Any later database record")]
    for ax, (tk, tt) in zip(axs, titles):
        w = 0.36; x = np.arange(len(rules))
        for q, (lab, col, kf) in enumerate((("flagged", "#E45756", "share_later_among_flagged"), ("not flagged", "#4C78A8", "share_later_among_not_flagged"))):
            vals = []; labs = []
            for r in rules:
                o = F["rules"][r]
                if tk == "any": v = o["any_later_record"]["share_flagged" if q == 0 else "share_not_flagged"]; a = o["any_later_record"]["flagged_with_record" if q == 0 else "not_flagged_with"]; b = o["any_later_record"]["flagged_without" if q == 0 else "not_flagged_without"]
                else: t = o["tables"][tk]; v = t[kf]; a = t["flagged_later" if q == 0 else "not_flagged_later"]; b = t["flagged_not_later" if q == 0 else "not_flagged_not_later"]
                vals.append(num(v)); labs.append("<5\nlater" if isinstance(v, str) else f"{a}/{(a if not isinstance(a, str) else 0) + (b if not isinstance(b, str) else 0)}")
            xs = x + (q - 0.5) * w; ax.bar(xs, np.nan_to_num(vals), w, color=col, label=lab)
            for xx, v, lb in zip(xs, vals, labs): ax.text(xx, (0 if np.isnan(v) else v) + 0.015, lb, ha="center", fontsize=9, rotation=0)
        ax.set_xticks(x); ax.set_xticklabels(rules); ax.set_ylim(0, 1.05); ax.set_title(tt); ax.set_ylabel("Share of non-progressors")
        p = [F["rules"][r]["tables"][tk]["fisher_p_two_sided"] if tk != "any" else F["rules"][r]["any_later_record"]["fisher_p_two_sided"] for r in rules]; ax.text(0.02, 0.95, "Fisher p: " + ", ".join(f"{pp:.2f}" for pp in p), transform=ax.transAxes, fontsize=9, va="top")
    axs[0].legend(loc="upper right"); fig.suptitle("F0: later dysplasia among flagged and unflagged non-progressors (R49, P1 90%, P3 90%)", fontsize=11); fig.tight_layout(); fig.savefig(f"{OUTD}/r9_f0.png", dpi=200); plt.close(fig)
def fig_icc(I):
    S = I["scores"]; names = list(S); fig, ax = plt.subplots(figsize=(8.5, 4.2)); x = np.arange(len(names)); w = 0.36
    for q, (cl, col) in enumerate((("non_progressor", "#4C78A8"), ("progressor", "#E45756"))):
        v = [S[n][cl]["icc1_mean_over_repeats"] for n in names]; lo = [S[n][cl]["icc1_mean_over_repeats"] - S[n][cl]["ci95"][0] for n in names]; hi = [S[n][cl]["ci95"][1] - S[n][cl]["icc1_mean_over_repeats"] for n in names]
        ax.bar(x + (q - 0.5) * w, v, w, yerr=[lo, hi], color=col, label=cl.replace("_", "-"), capsize=3)
        for xx, vv in zip(x + (q - 0.5) * w, v): ax.text(xx, 0.02, f"{vv:.2f}", ha="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([n.replace(", R49-aligned (0 = R49 flag boundary)", ", aligned").replace("logit max L-IMG", "logit max\nL-IMG").replace("gate score, ", "gate score,\n") for n in names]); ax.set_ylabel("ICC(1) across a patient's endoscopies"); ax.set_ylim(0, 1)
    ax.legend(); ax.set_title("Per-patient intraclass correlation of held-out endoscopy scores"); fig.tight_layout(); fig.savefig(f"{OUTD}/r9_icc.png", dpi=200); plt.close(fig)
def fig_curves(C):
    cur = C["curves"]; names = list(cur); cols = {"R49": "#000000", "P1 90%": "#4C78A8"}; fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=False)
    for ax, (cl, ttl) in zip(axs, (("np", "Non-progressors ever flagged (false flags)"), ("prog", "Progressors caught"))):
        for nm in names:
            for rst, ls in (("all_patients_first_k_endoscopies", "-"), ("restricted_to_patients_with_ge_k_endoscopies", "--")):
                ks = [k for k in range(1, 7) if not isinstance(cur[nm][f"k={k}"][rst][cl]["share"], str)]
                v = [cur[nm][f"k={k}"][rst][cl]["share"] for k in ks]; ax.plot(ks, v, ls, marker="o", ms=4, color=cols.get(nm, "#F58518"), label=f"{nm.replace('A1 | ', '').replace('exponentially weighted mean', 'EW mean').replace(' [nested]', '')[:50]} ({'all' if ls == '-' else '>= k endoscopies'})")
        ax.set_xlabel("Endoscopy index k (flag at >= 1 of the first k)"); ax.set_ylabel("Share of patients"); ax.set_title(ttl); ax.grid(alpha=0.2); ax.set_ylim(0, 1.02)
    axs[0].legend(fontsize=8, loc="lower right"); fig.suptitle("Cumulative flag curves by endoscopy index (held-out predictions)", fontsize=11); fig.tight_layout(); fig.savefig(f"{OUTD}/r9_curves.png", dpi=200); plt.close(fig)
if __name__ == "__main__":
    M = J("merge")["merge"]; fig_forest(M); fig_scatter(M)
    for nm, fn, key in (("f0", fig_f0, "F0"), ("icc", fig_icc, "ICC"), ("curves", fig_curves, "curves")):
        if os.path.exists(f"{RES}/{nm}.json"): fn(J(nm)[key])
    print("figures ->", OUTD)
