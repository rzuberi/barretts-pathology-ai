"""E5 figure (local, not committed): forest plot of paired differences against R49 (non-progressor patients flagged, progressor patients caught) for the
primary and key E5 rules, from results/paper_final/explore_E5/*.json -> paper/figures_explore_E5/e5_forest.png (200 dpi)."""
import json, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = H + "/../../results/paper_final/explore_E5"; FD = H + "/../../paper/figures_explore_E5"; os.makedirs(FD, exist_ok=True)
J = lambda n: json.load(open(f"{R}/{n}.json"))
A, B, C, MM = J("e5_a")["A"], J("e5_b")["B"], J("e5_c")["C"], J("e5_mem")["MEM"]
ROWS = [("P1 90% (H&E alone)", None), ("(a) ERIN LGD+ alone, 90%", A["erin_lgd 90%"]), ("(a) P32 ERIN head alone, 90%", A["erin_p32 90%"]),
        ("(b) stack(L-IMG, ERIN LGD+), 90%", B["stack(erin_lgd) P1 90%"]), ("(b) R49 banded by stack", B["R49 banding by stack(erin_lgd)"]),
        ("(c1) R49 + ERIN veto (q95)", C["c1 erin_lgd q95"]), ("(c1) R49 + P32-head veto (q95)", C["c1 erin_p32 q95"]), ("(c3) weak-ERIN high band sequenced", C["c3 erin_lgd q95"]),
        ("(c4) strong-ERIN low band sequenced", C["c4 erin_lgd"]), ("(m) EW memory (round 9)", MM["A1:gate_al:ew (round-9 row 2)"]),
        ("(m) EW memory + ERIN veto", MM["EW memory + erin_lgd veto q95"]), ("(m) EW memory + P32-head veto", MM["EW memory + erin_p32 veto q95"]), ("(m) EW memory over stack(gate, ERIN)", MM["EW memory over stack(gate, erin_lgd)"])]
p1 = A["erin_lgd 90%"]["delta_vs_P1 90%"]   # P1 vs R49 = -(rule vs P1 - rule vs R49): take from any rule
def dv(o, k):
    if o is None:   # P1 90% minus R49 from the paired deltas of the same rule
        a, b = A["erin_lgd 90%"]["delta_vs_R49"][k], A["erin_lgd 90%"]["delta_vs_P1 90%"][k]; return a["delta"] - b["delta"], None
    d = o["delta_vs_R49"][k]; return d["delta"], d["ci95"]
plt.rcParams.update({"font.size": 10}); fig, axs = plt.subplots(1, 2, figsize=(10, 6.0), sharey=True); ink = "#2a5b8c"
for ax, k, ttl in zip(axs, ("non_progressor_patients_flagged", "progressor_patients_caught"), ("Δ non-progressor patients flagged (of 43)", "Δ progressor patients caught (of 32)")):
    for i, (lab, o) in enumerate(ROWS):
        v, ci = dv(o, k); yy = len(ROWS) - 1 - i
        if ci: ax.plot(ci, [yy, yy], color=ink, lw=2, solid_capstyle="round")
        ax.plot([v], [yy], "o", color=ink, ms=7, mfc="white" if ci is None else ink)
    ax.axvline(0, color="#888888", lw=1); ax.set_title(ttl, fontsize=10); ax.grid(axis="x", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_yticks(range(len(ROWS))); axs[0].set_yticklabels([r[0] for r in ROWS][::-1])
fig.suptitle("E5: ERIN dysplasia-likeness against the R49 gate (paired, 95% patient-bootstrap CI; P1 point only)", fontsize=10)
fig.tight_layout(); fig.savefig(FD + "/e5_forest.png", dpi=200); print("saved", FD + "/e5_forest.png")
