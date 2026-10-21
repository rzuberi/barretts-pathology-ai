"""Exploration E3 figure (local, not committed): change in non-progressor patients ever flagged against R49 (paired patient bootstrap 95% CI) for the
main E3 rules, from the aggregate JSON in results/paper_final/explore_E3/. Output: paper/figures_explore_E3/e3_np_flagged_vs_R49.png (200 dpi)."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RES = ROOT + "/results/paper_final/explore_E3"; FIG = ROOT + "/paper/figures_explore_E3"; os.makedirs(FIG, exist_ok=True)
J = lambda f: json.load(open(f"{RES}/{f}.json"))
def sc(v, rule):   # score JSON: rule of the variant against R49
    r = J(v); k = {"P1": "P1_90_new_vs_R49", "R49": "R49_new_vs_R49"}[rule]; c = r["contrasts"][k]; d = c["delta_new_minus_ref"]
    return c["new"]["progressor_patients_caught"], c["new"]["non_progressor_patients_flagged"], d["non_progressor_patients_flagged"], c["new"]["samples_sequenced_per_patient"]
def mem(v, fam):
    m = [x for x in J(v)["methods"] if x["family"] == fam][0]; return m["values"]["caught"], m["values"]["np_flagged"], m["vs_R49"]["np_flagged"], m["values"]["seq"]
ROWS = [("H&E alone (P1 90%), current max over samples", sc("v_stored", "P1")),
        ("H&E alone, H-optimus-0 20x (best non-UNI2 extractor)", sc("v_ho20_mean", "P1")),
        ("H&E alone, max-pooled tiles", sc("v_u2l2_max", "P1")),
        ("H&E alone, mean sample logit per endoscopy", sc("v_mlogit--base", "P1")),
        ("H&E alone, mean logit + EW memory", mem("mem_mlogit--base", "A1:img:ew")),
        ("H&E alone, current + EW memory (round 9)", mem("mem_stored", "A1:img:ew")),
        ("R49 gate, mean sample logit per endoscopy", sc("v_mlogit--base", "R49")),
        ("R49 gate + EW memory (round 9)", mem("mem_stored", "A1:gate_al:ew")),
        ("R49 gate + EW memory, mean logit", mem("mem_mlogit--base", "A1:gate_al:ew")),
        ("R49 gate + EW memory, mean logit, UNI2+GigaPath (post hoc)", mem("mem_mlogit--cat_u2gp2_mean", "A1:gate_al:ew"))]
plt.rcParams.update({"font.size": 9}); fig, ax = plt.subplots(figsize=(9.6, 4.8)); BLUE, ORANGE = "#2a78d6", "#eb6834"
for i, (lab, (ca, npf, d, seq)) in enumerate(ROWS[::-1]):
    col = ORANGE if seq > 0 else BLUE; lo, hi = d["ci95"]
    ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round"); ax.plot(d["delta"], i, "o", ms=8, color=col, mec="white", mew=2)
    ax.annotate(f"{ca:.1f} / {npf:.1f}", (1.02, i), xycoords=("axes fraction", "data"), va="center", ha="left", fontsize=9, color="#52514e")
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS[::-1]], fontsize=9); ax.axvline(0, color="#8a8984", lw=1, zorder=0)
ax.set_xlim(-16, 16); ax.set_xlabel("Δ non-progressor patients ever flagged vs R49 (of 43), 95% CI"); ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.annotate("caught / NP", (1.02, len(ROWS) - 0.55), xycoords=("axes fraction", "data"), fontsize=9, color="#52514e", ha="left", va="bottom", annotation_clip=False)
ax.plot([], [], "o", color=BLUE, label="no sequencing"); ax.plot([], [], "o", color=ORANGE, label="sequences the middle band"); ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_title(
    "R49 reference: 31.1 of 32 progressors caught, 31.8 of 43 non-progressors flagged", fontsize=9, loc="left", pad=16)
fig.tight_layout(); fig.savefig(FIG + "/e3_np_flagged_vs_R49.png", dpi=200); print(FIG + "/e3_np_flagged_vs_R49.png")
