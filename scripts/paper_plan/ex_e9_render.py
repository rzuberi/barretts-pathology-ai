"""Exploration E9: figures (paper/figures_explore_E9/, local, never committed) and markdown tables (stdout) from results/paper_final/explore_E9/frontier.json.
Usage: python ex_e9_render.py [frontier.json]"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
J = json.load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT + "/results/paper_final/explore_E9/frontier.json")); FIG = ROOT + "/paper/figures_explore_E9"; os.makedirs(FIG, exist_ok=True)
C = J["curves"]; OP = J["operating_points_target_D"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
GREY = "#7a7974"; INK = "#0b0b0b"; INK2 = "#52514e"; SURF = "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
STY = {"gate": (GREY, "--", "Today's gate (swept)"), "hne_max": (BLUE, ":", "H&E max, single endoscopy"), "hne_mean": (MAGENTA, ":", "H&E mean-logit, single endoscopy"),
       "memory": (ORANGE, "-", "Memory"), "memcert": (YELLOW, "-", "Memory + certificate"), "final": (RED, "-", "Final rule (memory + certificate + check)"), "ml_mem": (AQUA, "-", "Mean-logit + memory (no sequencing)"),
       "grade_none": (GREEN, ":", "Grade, single endoscopy"), "grade_mean": (GREEN, "--", "Grade, running mean"), "grade_ew": (GREEN, "-", "Grade, EW memory"),
       "g53_none": (INK, ":", "Grade + p53, single endoscopy"), "g53_mean": (INK, "--", "Grade + p53, running mean"), "g53_ew": (INK, "-", "Grade + p53, EW memory"),
       "ml_final": (VIOLET, "-", "Mean-logit full stack (gate + memory + certificate + check)")}
FXP = {"R49": ("s", INK, "Today's gate (R49)"), "P1": ("^", INK, "H&E alone P1 90%"), "grade": ("D", INK2, "Grade IND/LGD"), "lgd": ("v", INK2, "LGD"), "p53": ("P", INK2, "p53 aberrant"), "g53": ("o", INK2, "Grade or p53")}
def curve(ax, k, band=False, lw=2.0, label=True):
    if k not in C: return
    col, ls, lab = STY[k]; x = np.array(C[k]["band_x"], float); yo = np.array([np.nan if v is None else v for v in C[k]["band_obs"]], float)
    if band:
        lo = np.array([np.nan if v is None else v for v in C[k]["band_lo"]], float); hi = np.array([np.nan if v is None else v for v in C[k]["band_hi"]], float); ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0)
    ax.plot(x, yo, color=col, ls=ls, lw=lw, label=lab if label else None, solid_capstyle="round")
    d = C[k]["points"][0]; ax.plot(d["np_flagged"], d["caught"], "o", ms=8, mfc=col, mec=SURF, mew=2, zorder=5)
def fixed(ax, keys, text=True, dx=None):
    for k in keys:
        if k not in OP: continue
        m, col, lab = FXP[k]; x, y = OP[k]["np_flagged"], OP[k]["caught"]; ax.plot(x, y, m, ms=9, mfc=col if k in ("R49", "P1") else "white", mec=col, mew=1.6, zorder=6)
        if text: ox, oy = (dx or {}).get(k, (0.6, -0.9)); ax.annotate(lab, (x, y), xytext=(x + ox, y + oy), fontsize=9, color=INK2)
fig, axs = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1.05, 1.0, 1.05]}); fig.patch.set_facecolor(SURF)
for ax in axs: ax.set_facecolor(SURF); ax.grid(True, color="#e6e5e0", lw=0.6); ax.set_xlabel("Non-progressor patients ever flagged (of 43)")
axs[0].set_ylabel("Progressor patients caught (of 32)")
ax = axs[0]
for k in ("gate", "hne_max", "hne_mean", "memory", "memcert", "final", "ml_mem", "ml_final"): curve(ax, k, band=k in ("gate", "final", "ml_mem"))
fixed(ax, ("R49", "P1"), text=False); fixed(ax, ("grade", "lgd", "p53", "g53"), dx={ "grade": (0.7, -0.3), "lgd": (0.7, -0.4), "p53": (0.7, -0.4), "g53": (0.7, -0.4)})
ax.set_xlim(0, 43); ax.set_ylim(0, 32.8); ax.set_title("A  H&E rules, full range", loc="left", fontsize=11, color=INK)
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
ax = axs[1]
for k in ("gate", "hne_max", "hne_mean", "memory", "memcert", "final", "ml_mem", "ml_final"): curve(ax, k, band=k in ("gate", "final", "ml_mem"))
fixed(ax, ("R49", "P1"), dx={"R49": (-6.2, 0.2), "P1": (0.4, -0.3)}); ax.set_xlim(8, 43); ax.set_ylim(26, 32.3)
ax.set_title("B  Zoom on high sensitivity (bands: 95% patient bootstrap)", loc="left", fontsize=11, color=INK)
ax = axs[2]
for k in ("grade_none", "grade_mean", "grade_ew", "g53_none", "g53_mean", "g53_ew"): curve(ax, k, lw=1.8)
curve(ax, "ml_mem", lw=1.6); curve(ax, "final", lw=1.6)
fixed(ax, ("grade", "lgd", "p53", "g53"), dx={"grade": (0.9, -0.2), "lgd": (-3.5, 0.8), "p53": (0.7, 0.4), "g53": (-9.0, 0.6)})
ax.set_xlim(0, 43); ax.set_ylim(0, 32.8); ax.set_title("C  The same memory applied to pathology", loc="left", fontsize=11, color=INK); ax.legend(loc="lower right", fontsize=8.5, frameon=False)
fig.text(0.01, 0.005, "Held-out patients; every threshold and half-life chosen on training patients (round-9 rule at a sweep of training detection targets). Large dots: each rule at the R49 gate's training detection (the reported operating points). "
         "Curves: monotone envelope of the swept points (training progressor-detection targets and training non-progressor-share targets); shading: vertical 95% bands (2,000 patient-bootstrap draws). "
         "The grey curve re-tunes the gate score's single threshold, so its dot differs from the R49 square.", fontsize=8.5, color=INK2, wrap=True)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG + "/fig_e9_tradeoff.png", dpi=200, facecolor=SURF); plt.close(fig)
# ---- prevalence figure: PPV and non-progressors never flagged per 1,000 at 3%
PV = J["per_1000_patients"]["prevalences"]; rules = [r for r in ("g53", "R49", "memory", "final", "ml_mem", "ml_final") if r in PV["3%"]]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8)); fig.patch.set_facecolor(SURF); names = {"g53": "Grade or p53", "R49": "Today's gate", "memory": "Memory", "final": "Final rule", "ml_mem": "Mean-logit + memory", "ml_final": "Mean-logit full stack"}
cols = {"g53": INK, "R49": GREY, "memory": ORANGE, "final": RED, "ml_mem": AQUA, "ml_final": VIOLET}; prevs = ["1.5%", "3%", "5%", "10%"]; xs = np.array([1.5, 3, 5, 10])
for ax, key, lab in ((axs[0], "ppv", "PPV of a patient flag"), (axs[1], "non_progressors_never_flagged", "Non-progressors never flagged per 1,000"), (axs[2], "progressors_missed", "Progressors missed per 1,000")):
    ax.set_facecolor(SURF); ax.grid(True, color="#e6e5e0", lw=0.6)
    for r in rules:
        v = np.array([PV[p][r][key]["value"] for p in prevs], float); ax.plot(xs, v, "-o", color=cols[r], lw=2, ms=7, mec=SURF, mew=1.5, label=names[r])
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(prevs); ax.minorticks_off(); ax.set_xlabel("Share of patients progressing over the cohort's follow-up"); ax.set_ylabel(lab)
axs[0].legend(fontsize=8.5, frameon=False); fig.tight_layout(); fig.savefig(FIG + "/fig_e9_prevalence.png", dpi=200, facecolor=SURF); plt.close(fig)
# ---- tables (markdown) to stdout
def f(v, d=1): return "–" if v is None else (f"{v:+.{d}f}" if isinstance(v, float) and False else f"{v:.{d}f}")
def ci(c, d=1): return "" if not c or c[0] is None else f" [{c[0]:.{d}f}, {c[1]:.{d}f}]"
def dl(o, d=1): return "–" if o is None or o.get("delta") is None else f"{o['delta']:+.{d}f}{ci(o.get('ci95'), d)}"
print("## Reproduction\n"); print(json.dumps(J["reproduction"])); print()
print("## Operating points (target D)\n\n| Rule | Caught /32 | NP flagged /43 | Samples/patient | Cost/patient £ | ΔNP vs R49 | Δcaught vs R49 | ΔNP vs P1 | Δcaught vs P1 | Years first flag → event |\n|---|---|---|---|---|---|---|---|---|---|")
for k, o in OP.items():
    print(f"| {o['label']} | {f(o['caught'])} | {f(o['np_flagged'])} | {f(o['seq_per_patient'], 2)} | {f(o['cost_per_patient'], 0)} | {dl(o['vs_R49']['np_flagged'])} | {dl(o['vs_R49']['caught'])} | {dl(o['vs_P1']['np_flagged'])} | {dl(o['vs_P1']['caught'])} | {f(o['years_first_flag_to_event_median'], 1)} |")
print("\n## Matched comparisons on the curves\n\n| Rule | NP at 31.1 caught | Δ vs gate | NP at 30.5 caught | Δ vs gate | NP at R49's caught − R49 | Caught at 23.6 NP | Δ vs gate | Caught at 17.2 NP | Δ vs gate |\n|---|---|---|---|---|---|---|---|---|---|")
for k, m in J["matched"].items():
    a, b = m["np_flagged_at_caught_31.1"], m["np_flagged_at_caught_30.5"]; c, d = m["caught_at_np_23.6"], m["caught_at_np_17.2"]
    print(f"| {C[k]['label']} | {f(a['value'])}{ci(a['ci95'])} | {dl(a['vs_gate'])} | {f(b['value'])}{ci(b['ci95'])} | {dl(b['vs_gate'])} | {dl(m['np_flagged_at_R49_caught_minus_R49'])} | {f(c['value'])}{ci(c['ci95'])} | {dl(c['vs_gate'])} | {f(d['value'])}{ci(d['ci95'])} | {dl(d['vs_gate'])} |")
print("\n## Tiered menu\n\n| Rule | Caught | NP flagged | Samples/patient | Cost/patient £ | Δcost vs ML+memory | ΔNP vs ML+memory | Δcaught vs ML+memory | £ per NP flag avoided (vs ML+mem) | £ per progressor lost (vs ML+mem) | Δcost vs R49 | ΔNP vs R49 | Δcaught vs R49 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, o in J["tiered_menu"].items():
    a = o["vs_ml_mem"]; b = o["vs_R49"]
    print(f"| {o['label']} | {f(o['caught'])} | {f(o['np_flagged'])} | {f(o['seq_per_patient'], 2)} | {f(o['cost_per_patient'], 0)} | {dl(a['d_cost_per_patient'], 0)} | {dl(a['d_np_flagged'])} | {dl(a['d_caught'])} | {f(a['gbp_per_np_flag_avoided_cohort_75'], 0)} | {f(a['gbp_per_progressor_lost_cohort_75'], 0)} | {dl(b['d_cost_per_patient'], 0)} | {dl(b['d_np_flagged'])} | {dl(b['d_caught'])} |")
for p in ("1.5%", "3%", "5%", "10%"):
    print(f"\n## Per 1,000 patients, {p} progressing\n\n| Rule | Patients flagged | Extra endoscopies (per flagged endoscopy) | Samples sequenced | PPV | NPV | Progressors caught | Progressors missed | NP never flagged (de-escalation) | Flagged per progressor caught |\n|---|---|---|---|---|---|---|---|---|---|")
    for k, o in PV[p].items():
        nm = OP[k]["label"] if k in OP else k; g = lambda kk, d=0: f"{f(o[kk]['value'], d)}{ci(o[kk]['ci95'], d)}"
        print(f"| {nm} | {g('patients_flagged')} | {g('flagged_endoscopies')} | {g('samples_sequenced')} | {g('ppv', 3)} | {g('npv', 4)} | {g('progressors_caught', 1)} | {g('progressors_missed', 1)} | {g('non_progressors_never_flagged')} | {g('flagged_per_progressor_caught', 1)} |")
BE = J["break_even"]["results"]
for p in ("cohort (43%)", "3%"):
    for conv in ("per_flagged_endoscopy", "once_per_flagged_patient"):
        print(f"\n## Break-even, {p}, flag costed {conv.replace('_', ' ')}\n\n| Rule | vs | Δsamples /1000 | Δflags /1000 | Δmissed /1000 | Break-even £/sample (miss £0) | (miss £20k) | (miss £40k) | Break-even £/flag (miss £0) | Δcost £/1000 at £85 (miss £0) | at £85 (miss £20k) |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for k, o in BE[p].items():
            for ref in ("vs_R49", "vs_g53"):
                r = o[ref]; c_ = r[conv]; g = lambda d_: f"{f(d_['value'], 0)}{ci(d_['ci95'], 0)}" if d_["value"] is not None else "–"
                print(f"| {OP[k]['label'] if k in OP else k} | {ref[3:]} | {g(r['d_samples'])} | {g(c_['d_flags'])} | {f(r['d_missed']['value'], 1)} | {g(c_['miss_0']['breakeven_seq_price_gbp'])} | {g(c_['miss_20000']['breakeven_seq_price_gbp'])} | {g(c_['miss_40000']['breakeven_seq_price_gbp'])} | {g(c_['miss_0']['breakeven_flag_cost_gbp'])} | {g(c_['miss_0']['d_cost_per_1000_gbp_at_seq_price']['85'])} | {g(c_['miss_20000']['d_cost_per_1000_gbp_at_seq_price']['85'])} |")
