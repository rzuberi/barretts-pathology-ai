"""Two papers, agent ART: figure from the aggregate JSON only (results/paper_final/two_papers_ART/score.json, gate.json) -> paper/two_papers/figs_ART/ (local)."""
import json, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/Users/zuberi01/Documents/thesis/results/paper_final/two_papers_ART"; F = "/Users/zuberi01/Documents/thesis/paper/two_papers/figs_ART"; os.makedirs(F, exist_ok=True)
S = json.load(open(f"{R}/score.json"))["score"]; G = json.load(open(f"{R}/gate.json"))["gate"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6), dpi=200, gridspec_kw={"width_ratios": [1.15, 1]})
rows = [("overall", "All pairs"), ("within_batch", "Within sequencing batch"), ("within_scanner", "Within scanner"), ("ndbe", "NDBE only")]
mods = [("L-IMG stored", "L-IMG, all tiles (published)", "#555555"), ("L-IMG random-subset control", "L-IMG, random tiles removed", "#9a9a9a"),
        ("L-IMG clean, blur kept", "L-IMG, glass/edge/pen removed", "#4c9be8"), ("L-IMG", "L-IMG, all artefact removed", "#1f4e9c"), ("L-CNV", "L-CNV", "#d9822b")]
yt, yl = [], []
for i, (k, lab) in enumerate(rows):
    for j, (m, ml, c) in enumerate(mods):
        yy = -(i * (len(mods) + 1.5) + j); v = S[k]["auroc"][m]
        a1.errorbar(v["value"], yy, xerr=[[v["value"] - v["ci95"][0]], [v["ci95"][1] - v["value"]]], fmt="o", color=c, ms=4.5, capsize=2, label=ml if i == 0 else None)
    yt.append(-(i * (len(mods) + 1.5) + 2)); yl.append(lab)
a1.set_yticks(yt); a1.set_yticklabels(yl, fontsize=9); a1.set_xlabel("AUROC (fold-stratified, 95% patient-bootstrap CI)", fontsize=9); a1.axvline(0.5, color="#cccccc", lw=0.8)
a1.set_xlim(0.6, 1.0); a1.tick_params(labelsize=9); a1.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False); a1.set_title("Sample-level AUROC, pre-event samples", fontsize=10)
rules = [("R49", "R49 gate"), ("memory (gate score, EW)", "Memory"), ("light tier (A12)", "Light tier"), ("tier 0: mean-logit + EW memory", "Tier 0 (no sequencing)")]
for i, (k, lab) in enumerate(rules):
    for j, (nm, c) in enumerate((("stored", "#555555"), ("rand", "#9a9a9a"), ("clean", "#1f4e9c"))):
        v = G[k][nm]["np_flagged"]; yy = -(i * 4 + j)
        a2.errorbar(v["value"], yy, xerr=[[v["value"] - v["ci95"][0]], [v["ci95"][1] - v["value"]]], fmt="o", color=c, ms=4.5, capsize=2,
                    label={"stored": "all tiles (published)", "rand": "random tiles removed", "clean": "artefact removed"}[nm] if i == 0 else None)
        a2.text(44.5, yy, f"{G[k][nm]['caught']['value']:.1f}", fontsize=8, va="center", ha="right", color=c)
a2.set_yticks([-(i * 4 + 1) for i in range(len(rules))]); a2.set_yticklabels([l for _, l in rules], fontsize=9); a2.set_xlim(14, 45)
a2.set_xlabel("Non-progressor patients flagged of 43 (95% CI)", fontsize=9); a2.tick_params(labelsize=9); a2.set_title("Paper 2 rules (right column: progressors caught of 32)", fontsize=10)
a2.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(f"{F}/art_clean_tile_forest.png", dpi=200); print("saved")
