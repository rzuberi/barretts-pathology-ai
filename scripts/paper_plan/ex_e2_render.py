"""Exploration E2: render the aggregate JSONs (results/paper_final/explore_E2/) into markdown tables printed to stdout and figures in
paper/figures_explore_E2/ (local, never committed). Usage: python ex_e2_render.py [results_dir]"""
import os, sys, json, glob
import numpy as np
RD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/paper_final/explore_E2")
FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../paper/figures_explore_E2"); os.makedirs(FD, exist_ok=True)
L = lambda f: json.load(open(os.path.join(RD, f))) if os.path.exists(os.path.join(RD, f)) else None
def ci(d, nd=1):
    return f"{d['delta']:+.{nd}f} [{d['ci95'][0]:+.{nd}f}, {d['ci95'][1]:+.{nd}f}]" if d and d.get("delta") is not None and d["ci95"][0] is not None else "n/a"
def cells():
    out = []
    for f in sorted(glob.glob(os.path.join(RD, "*.json"))):
        j = json.load(open(f)); src = os.path.basename(f)[:-5]
        for c in j.get("cells", []): c["_src"] = src; out.append(c)
    return out
def row(c):
    R, P = c["contrasts"]["R49"], c["contrasts"]["P1"]; ms = c["contrasts"].get("P1_matched_sens_d_endo_spec")
    return (f"| {c['label']} | {c['progressor_patients_caught']} | {c['non_progressor_patients_ever_flagged']} | {c['samples_sequenced_per_patient']:.2f} | {c['endoscopy_sensitivity']:.3f} / {c['endoscopy_specificity']:.3f} | "
            f"{ci(R['d_np_patients_flagged'])} | {ci(R['d_progressor_patients_caught'])} | {ci(P['d_np_patients_flagged'])} | {ci(P['d_progressor_patients_caught'])} | {ci(ms, 3)} |")
HDR = ("| Rule | Caught / 32 | NP flagged / 43 | Seq / patient | Endo sens / spec | Δ NP flagged vs R49 [CI] | Δ caught vs R49 [CI] | Δ NP flagged vs P1 [CI] | Δ caught vs P1 [CI] | Δ endo spec vs P1 at matched sens [CI] |\n"
       "|---|---|---|---|---|---|---|---|---|---|")
def ranked(C, key="R49", tol=-0.5, top=40):
    ok = [c for c in C if c["contrasts"][key]["d_progressor_patients_caught"]["delta"] >= tol]
    return sorted(ok, key=lambda c: c["contrasts"][key]["d_np_patients_flagged"]["delta"])[:top]
if __name__ == "__main__":
    C = cells(); print(f"{len(C)} nested cells\n")
    for key in ("R49", "P1"):
        print(f"### Ranked against {key} (Δ caught ≥ −0.5)\n"); print(HDR)
        for c in ranked(C, key): print(row(c))
        print()
    # E2 cells not dominated (caught, NP flagged, sequencing) by any round-8 policy cell or by another E2 cell
    l8 = os.path.join(RD, "../review_round8/l1.json"); R8 = [c for c in json.load(open(l8))["L1"]["cells"] if c.get("params", {}).get("subset") is None] if os.path.exists(l8) else []
    K = lambda c: (c["progressor_patients_caught"], -c["non_progressor_patients_ever_flagged"], -c["samples_sequenced_per_patient"])
    dom = lambda a, b: all(x >= y for x, y in zip(K(b), K(a))) and any(x > y for x, y in zip(K(b), K(a)))
    allc = C + R8; fr = [c for c in C if not any(dom(c, b) for b in allc if b is not c)]
    print("### E2 cells on the joint frontier with round 8 (caught, NP flagged, sequencing)\n"); print(HDR)
    for c in sorted(fr, key=lambda c: -c["progressor_patients_caught"]): print(row(c))
    print("\nround-8 cells still on the joint frontier:", ", ".join(f"{c['family']} {c['label']}" for c in R8 if not any(dom(c, b) for b in allc if b is not c)), "\n")
    # figure: caught vs NP flagged, all nested cells, by base
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10}); fig, ax = plt.subplots(figsize=(7.5, 5.5)); cols = {"P1": "#1f77b4", "R49h": "#2ca02c", "R49hm": "#d62728"}
    for b, col in cols.items():
        cc = [c for c in C if c["params"].get("base") == b]
        ax.scatter([c["non_progressor_patients_ever_flagged"] for c in cc], [c["progressor_patients_caught"] for c in cc], s=[10 + 12 * c["samples_sequenced_per_patient"] for c in cc], alpha=0.45, color=col, label=f"veto on {b}")
    cc = [c for c in C if str(c["params"].get("base", "")).startswith("mem:")]
    ax.scatter([c["non_progressor_patients_ever_flagged"] for c in cc], [c["progressor_patients_caught"] for c in cc], s=[10 + 12 * c["samples_sequenced_per_patient"] for c in cc], alpha=0.6, color="#9467bd", label="round-9 memory rules ± veto")
    l8 = os.path.join(RD, "../review_round8/l1.json")
    if os.path.exists(l8):
        fr = [c for c in json.load(open(l8))["L1"]["cells"] if c["pareto_caught_falseflags_sequencing"] and c["family"] != "P10"]
        ax.scatter([c["non_progressor_patients_ever_flagged"] for c in fr], [c["progressor_patients_caught"] for c in fr], marker="D", s=40, facecolors="none", edgecolors="k", label="round-8 frontier (P1, P3, P4, P6-P8)", zorder=4)
    ax.scatter([31.8], [31.1], marker="*", s=220, color="k", label="R49", zorder=5); ax.scatter([36.0], [31.1], marker="X", s=140, color="grey", label="P1 90%", zorder=5)
    ax.set_xlim(14, 41); ax.set_xlabel("Non-progressor patients ever flagged (of 43)"); ax.set_ylabel("Progressor patients caught (of 32)"); ax.set_ylim(24, 32.3); ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    ax.set_title("Copy-number vetoes, nested (marker size = samples sequenced per patient)", fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(FD, "e2_veto_scatter.png"), dpi=200); print("figure", os.path.join(FD, "e2_veto_scatter.png"))
