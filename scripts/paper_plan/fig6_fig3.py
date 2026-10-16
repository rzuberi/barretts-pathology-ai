"""Figure 3, H&E-first triage (docs/paper_figures_v6.md @ d46b9a5), drawn on the cluster from the stored held-out predictions (tb_common: triage(),
comparators(), boots_for()). (a) flow of the 571 pre-event biopsies by class into the modal band over the 10 repeats and the voted final call (committed
flow, re-derived here and asserted equal); (b) share sequenced against single-model sensitivity for the H&E gate, the H-optimus-0 gate, the permuted-label
gate (5 permutations, range) and the random gate (100 seeds, 2.5-97.5% range), with the comparators and the 9 cut-off grid cells (committed values);
(c) per-patient outcomes per strategy: mean and range over the 10 repeats of progressor patients missed and non-progressor patients flagged;
(d) projected high-band PPV, low-band NPV and share sequenced against progressor-biopsy prevalence with 95% patient-bootstrap ribbons (class-wise band
rates per draw, 2,000 draws); (e) band stability from the committed R35 distribution. Source data: aggregate values only."""
import os, sys, glob, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig6_style import *
from tb_common import *
O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; yr = y[rows]; pr_ = pat[rows]
band, call, info = triage(O, I, FO, popm); comp, thr = comparators(O, I, FO, popm); CALLS = {"Triage": call, "(A)": comp["A"], "(B)": comp["B"], "(C)": comp["C"]}
src = {"n_samples": int(len(rows)), "n_progressor_samples": int(yr.sum()), "n_patients": int(len(set(pr_))), "n_progressor_patients": int(len(set(pr_[yr == 1])))}
# ------------------------------------------------------------------ (a) flow: modal band (ties -> middle) and the >= 6-of-10 vote, checked against the committed flow
Bq = band[rows]; cnt = np.stack([(Bq == g).sum(1) for g in range(3)], 1); modal = cnt.argmax(1); tie = (cnt == cnt.max(1, keepdims=True)).sum(1) > 1; modal[tie] = 1; V = vote(call)[rows]
flow = {}
for g, b in ((0, "low"), (1, "mid"), (2, "high")):
    m = modal == g; flow[b] = {"progressor": int((m & (yr == 1)).sum()), "non_progressor": int((m & (yr == 0)).sum()), "called_positive": {"progressor": int((m & V & (yr == 1)).sum()), "non_progressor": int((m & V & (yr == 0)).sum())}}
ref = J("triage/their_pre.json")["flow"]["triage_LATE"]; assert flow == ref, (flow, ref); src["flow_modal_band_voted_call"] = flow
fig = plt.figure(figsize=(183 * MM, 165 * MM))
ax = fig.add_axes([0.01, 0.56, 0.60, 0.40]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
P, N = int(yr.sum()), int((yr == 0).sum())
box(ax, 1, 40, 24, 20, f"Pre-event biopsies\nn = {P + N}\n{P} progressor\n{N} non-progressor", "#F2F2F2", "#444444", fs=6.3)
ys = {"low": 14, "mid": 50, "high": 86}; names = {"low": "Low band: cleared,\nnot sequenced", "mid": "Middle band:\nsequenced, L-LATE call", "high": "High band: flagged,\nnot sequenced"}
for b in ("low", "mid", "high"):
    f = flow[b]; nb = f["progressor"] + f["non_progressor"]; box(ax, 34, ys[b] - 10, 28, 20, f"{names[b]}\nn = {nb}\n{f['progressor']} P / {f['non_progressor']} NP", BAND[b] + "55", BAND[b], fs=6.3); arrow(ax, 25, 50, 34, ys[b])
    cp, cn = f["called_positive"]["progressor"], f["called_positive"]["non_progressor"]; outs = [("Flagged", cp, cn, "#CC6677"), ("Cleared", f["progressor"] - cp, f["non_progressor"] - cn, "#88CCEE")]
    outs = [o for o in outs if o[1] + o[2] > 0]; off = [0] if len(outs) == 1 else [9, -9]
    for (t, a, c, col), dy in zip(outs, off): box(ax, 72, ys[b] + dy - 7, 27, 14, f"{t}: n = {a + c}\n{a} P / {c} NP", col + "33", col, fs=6.1); arrow(ax, 62, ys[b], 72, ys[b] + dy)
ax.text(1, 4, "Band: most frequent of the 10 repeats (ties to middle); call: positive in ≥ 6 of 10 repeats.\nP, progressor; NP, non-progressor.", fontsize=5.4, color="#333333", va="top")
letter(fig, ax, "a", dx=0.01, dy=-0.02)
# ------------------------------------------------------------------ (b) share sequenced vs sensitivity (single model)
bx = fig.add_axes([0.70, 0.60, 0.29, 0.35]); pr0 = J("review_round4/r33_base_primary.json")["R33_base"]["claim_T"]; r33 = J("review_round4/r33_merge.json")["R33"]["claim_T_triage_vs_Bstar_training_matched_single_model"]
f5 = J("review_round2/f5_r28.json")["F5_R28"]["pre"]; f6 = J("review_round2/f6_r25b.json")["F6_R25b"]; r14 = J("review_round2/r14_their_pre.json")["R14"]["b_no_vote_paired"]; r24 = J("review_round2/r24.json")["R24"]["comparator_D"]["D1_grade_ID_or_LGD"]
seeds = [json.load(open(f)) for f in sorted(glob.glob(f"{M}/review_round2/randgate/seed_*.json"))]; assert len(seeds) == 100
rs = np.array([s["share_sequenced"] for s in seeds]); rsn = np.array([s["sensitivity"] for s in seeds]); pm = f6["per_permutation"]; ps = np.array([pm[k]["no_vote"]["share_sequenced"] for k in pm]); psn = np.array([pm[k]["no_vote"]["sensitivity"] for k in pm])
gates = {"H&E gate (UNI2-h)": (pr0["share_sequenced"]["per_repeat_mean"], pr0["share_sequenced"]["ci95"], pr0["triage_no_vote"]["sensitivity"], pr0["triage_no_vote"]["sens_ci95"], MODEL["L-IMG"], "*", 10),
         "H-optimus-0 gate": (f5["share_sequenced"]["per_repeat_mean"], f5["share_sequenced"]["ci95"], f5["strategies"]["no_vote"]["triage"]["sensitivity"], f5["strategies"]["no_vote"]["triage"]["sens_ci95"], "#8E6C8A", "P", 6),
         "permuted-label gate (5 permutations, range)": (float(ps.mean()), [float(ps.min()), float(ps.max())], float(psn.mean()), [float(psn.min()), float(psn.max())], "#AA4499", "X", 5),
         "random gate (100 seeds, 2.5–97.5%)": (float(rs.mean()), [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))], float(rsn.mean()), [float(np.percentile(rsn, 2.5)), float(np.percentile(rsn, 97.5))], GREY, "o", 4)}
for k, (s, sc, se, sec, col, mk, ms) in gates.items():
    bx.plot([100 * sc[0], 100 * sc[1]], [se, se], color=col, lw=0.8); bx.plot([100 * s, 100 * s], [sec[0], sec[1]], color=col, lw=0.8); bx.plot([100 * s], [se], marker=mk, color=col, ms=ms, lw=0, mec="black", mew=0.4, label=k, zorder=5)
comps = {"(A)": (100, r14["A"]["comparator_sensitivity"], MODEL["L-CNV"], "o"), "A*": (100, r14["A*"]["comparator_sensitivity"], MODEL["L-CNV"], "s"), "(B)": (100, pr0["no_vote_vs_B"]["comparator_sensitivity"], MODEL["L-LATE"], "o"),
         "B*": (100, pr0["no_vote_vs_Bstar_training_matched"]["comparator_sensitivity"], MODEL["L-LATE"], "s"), "(C)": (0, pr0["no_vote_vs_C"]["comparator_sensitivity"], MODEL["L-IMG"], "o"),
         "C*": (0, pr0["no_vote_vs_Cstar_training_matched"]["comparator_sensitivity"], MODEL["L-IMG"], "s"), "(D)": (0, r24["sample"]["sensitivity"], MODEL["grade"], "D")}
OFF = {"(A)": (3, 0.004), "A*": (3, 0), "(B)": (3, 0.0), "B*": (3, 0.0), "(C)": (3, -0.014), "C*": (3, 0.012), "(D)": (3, 0)}
for k, (s, se, col, mk) in comps.items(): bx.plot([s], [se], marker=mk, color=col, ms=4, lw=0, mec="black", mew=0.4, zorder=5); bx.text(s + OFF[k][0], se + OFF[k][1], k, fontsize=5.8, ha="left", va="center")
grid = [r for r in r33 if r["item"] == 8]
for r in grid:
    bx.plot([100 * r["share_sequenced"]], [r["triage_sens_spec"][0]], marker="o", color="white", mec=MODEL["L-IMG"], mew=0.7, ms=3.2, lw=0, zorder=4)
bx.plot([], [], marker="o", color="white", mec=MODEL["L-IMG"], mew=0.7, ms=3.2, lw=0, label="H&E gate, cut-off grid (t_low / t_high)")
bx.set_xlabel("share of biopsies sequenced (%)"); bx.set_ylabel("sensitivity (single model, mean over repeats)"); bx.set_xlim(-6, 118); bx.set_ylim(0.3, 1.0); bx.legend(loc="center left", fontsize=5.2, bbox_to_anchor=(0.08, 0.50))
src["gates_share_sensitivity"] = {k: {"share": v[0], "share_range": v[1], "sensitivity": v[2], "sensitivity_range": v[3]} for k, v in gates.items()}; src["comparators_share_sensitivity"] = {k: {"share": v[0] / 100, "sensitivity": v[1]} for k, v in comps.items()}
src["grid_cells"] = [{"analysis": r["analysis"], "share": r["share_sequenced"], "sensitivity": r["triage_sens_spec"][0]} for r in grid]
letter(fig, bx, "b", dx=-0.07, dy=-0.02)
# ------------------------------------------------------------------ (c) per-patient outcomes by strategy (per repeat; mean and range)
pp = pd.Series(pr_); up = pp.unique(); isP = pd.Series(yr).groupby(pp).max().reindex(up).values == 1; pats = {}
for k, C in CALLS.items():
    miss, flag = [], []
    for j in range(10):
        anyp = pd.Series(C[rows, j]).groupby(pp).any().reindex(up).values; miss.append(int((~anyp & isP).sum())); flag.append(int((anyp & ~isP).sum()))
    pats[k] = {"progressor_patients_missed": {"mean": float(np.mean(miss)), "range": [min(miss), max(miss)]}, "non_progressor_patients_flagged": {"mean": float(np.mean(flag)), "range": [min(flag), max(flag)]}}
D = r24["patient"]; pats["(D)"] = {"progressor_patients_missed": {"mean": float(D["progressor_missed"]), "range": [D["progressor_missed"], D["progressor_missed"]]}, "non_progressor_patients_flagged": {"mean": float(D["nonprogressor_flagged"]), "range": [D["nonprogressor_flagged"], D["nonprogressor_flagged"]]}}
src["per_patient_by_strategy"] = pats; order = ["Triage", "(A)", "(B)", "(C)", "(D)"]; SC = {"Triage": "black", "(A)": MODEL["L-CNV"], "(B)": MODEL["L-LATE"], "(C)": MODEL["L-IMG"], "(D)": MODEL["grade"]}
for q, key in enumerate(("progressor_patients_missed", "non_progressor_patients_flagged")):
    a_ = fig.add_axes([0.07 + q * 0.155, 0.10, 0.11, 0.33])
    for i, k in enumerate(order):
        v = pats[k][key]; yy = len(order) - 1 - i; a_.plot(v["range"], [yy, yy], color=SC[k], lw=1.6, alpha=0.45, solid_capstyle="butt"); a_.plot([v["mean"]], [yy], marker="o", color=SC[k], ms=4, lw=0)
    a_.set_yticks(range(len(order))); a_.set_yticklabels(order[::-1] if q == 0 else [""] * len(order)); a_.set_ylim(-0.6, len(order) - 0.4)
    a_.set_xlabel("progressor\npatients missed\n(of 32)" if q == 0 else "non-progressor\npatients flagged\n(of 43)", fontsize=6.2); a_.set_xlim(-0.5, 32.5 if q == 0 else 43.5)
    if q == 0: cxa = a_
letter(fig, cxa, "c", dx=-0.055)
# ------------------------------------------------------------------ (d) predictive values against prevalence
dx_ = fig.add_axes([0.42, 0.10, 0.25, 0.33]); boots = boots_for(rows); piv = np.linspace(0.005, 0.30, 60)
def rates(b):
    Bb = band[b]; yb = y[b]; return np.array([(Bb[yb == 1] == g).mean(0).mean() for g in range(3)]), np.array([(Bb[yb == 0] == g).mean(0).mean() for g in range(3)])
def curves(r1, r0):
    ppv = piv * r1[2] / (piv * r1[2] + (1 - piv) * r0[2]); npv = (1 - piv) * r0[0] / ((1 - piv) * r0[0] + piv * r1[0]); share = piv * r1[1] + (1 - piv) * r0[1]; return ppv, npv, share
obs = curves(*rates(rows)); BS = np.array([np.stack(curves(*rates(b))) for b in boots]); lo = np.percentile(BS, 2.5, axis=0); hi = np.percentile(BS, 97.5, axis=0)
for q, (lab, col) in enumerate((("high-band PPV (post-test risk if flagged)", BAND["high"]), ("low-band NPV", BAND["low"]), ("share sequenced (middle band)", BAND["mid"]))):
    dx_.fill_between(100 * piv, lo[q], hi[q], color=col, alpha=0.35, lw=0); dx_.plot(100 * piv, obs[q], color=col, lw=1.2, label=lab, path_effects=None)
dx_.axvline(2, color=LIGHT, lw=0.6, ls=":"); i2 = int(np.argmin(np.abs(piv - 0.02)))
dx_.text(29.5, 0.03, f"at 2% prevalence: PPV {100 * obs[0][i2]:.1f}%,\nNPV {100 * obs[1][i2]:.1f}%, sequenced {100 * obs[2][i2]:.1f}%", fontsize=5.4, va="bottom", ha="right")
dx_.set_xlabel("progressor-biopsy prevalence (%)"); dx_.set_ylabel("proportion"); dx_.set_xlim(0, 30); dx_.set_ylim(0, 1.02); dx_.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=5.2)
r1o, r0o = rates(rows); src["class_wise_band_rates"] = {"progressor": r1o.tolist(), "non_progressor": r0o.tolist()}
src["prevalence_curves"] = {"prevalence": piv.tolist(), "ppv_high": obs[0].tolist(), "ppv_high_ci": [lo[0].tolist(), hi[0].tolist()], "npv_low": obs[1].tolist(), "npv_low_ci": [lo[1].tolist(), hi[1].tolist()], "share_sequenced": obs[2].tolist(), "share_ci": [lo[2].tolist(), hi[2].tolist()], "draws": len(boots)}
letter(fig, dx_, "d", dx=-0.06)
# ------------------------------------------------------------------ (e) band stability (committed R35)
ex = fig.add_axes([0.765, 0.10, 0.185, 0.33]); st = J("review_round4/r35.json")["R35"]["c_band_stability"]["repeats_in_modal_band_distribution"]; keys = [f"{c}_{b}" for c in ("progressor", "non_progressor") for b in ("low", "mid", "high")]
greys = {10: "#222222", 9: "#555555", 8: "#808080", 7: "#A6A6A6", 6: "#CCCCCC", 5: "#E8E8E8"}; src["band_stability"] = st
for i, k in enumerate(keys):
    d = st[k]; tot = sum(d.values()); left = 0; yy = len(keys) - 1 - i
    for r_ in (10, 9, 8, 7, 6, 5):
        v = d.get(str(r_), 0); ex.barh(yy, 100 * v / tot, left=left, color=greys[r_], edgecolor="white", lw=0.3, height=0.7); left += 100 * v / tot
    ex.text(102, yy, f"n {tot}", fontsize=5.4, va="center", clip_on=False)
ex.set_yticks(range(len(keys))); ex.set_yticklabels([f"{'P' if k.startswith('progressor') else 'NP'}, {k.split('_')[-1]} band" for k in keys][::-1]); ex.set_xlabel("biopsies (%)"); ex.set_xlim(0, 100)
for r_ in (10, 9, 8, 7, 6, 5): ex.barh([-10], [0], color=greys[r_], label=f"{r_} of 10")
ex.set_ylim(-0.6, len(keys) - 0.4); ex.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=5.2, title="repeats in the modal band", title_fontsize=5.4, columnspacing=0.8)
letter(fig, ex, "e", dx=-0.09)
save(fig, "Fig3"); dump("Fig3", src)
