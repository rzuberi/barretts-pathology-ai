"""Figure 3 (manuscript v7), H&E-first triage, drawn on the cluster from the stored held-out predictions (tb_common: triage(), comparators(); the
training-matched comparators as in the round-4/6 scripts). (a) single-model band flow: per-repeat mean counts of pre-event biopsies by class in each band
and call. (b) share sequenced against single-model sensitivity, marker colour = specificity, for the H&E, H-optimus-0, permuted-label and random gates,
the comparators and the 9 cut-off cells; (b, right) the same at endoscopy level with one random biopsy per endoscopy. (c) per-patient outcomes per strategy
with 95% CIs (patient bootstrap for model rules; exact for grade and grade + p53). (d) projected predictive values against prevalence. (e) band stability
(modal band = most frequent band over the 10 repeats, ties to the lower band, as in the stored stability analysis) with the unanimity of the voted call."""
import os, sys, glob, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig7_style import *
from tb_common import *
import matplotlib as mpl
O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; yr = y[rows]; pr_ = pat[rows]
band, call, info = triage(O, I, FO, popm); comp, thr = comparators(O, I, FO, popm)
def matched_calls(O, I, FO, popm):   # training-matched A*, B*, C* (round-4 rule)
    MM = {"A*": "L-CNV", "B*": "L-LATE", "C*": "L-IMG"}; mc = {a: np.zeros((n, 10), bool) for a in MM}
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            si = I["L-IMG"][(r, k)]; tri = si.index.values[popm[si.index.values]]; s_tr = si.reindex(tri).values; yc = y[tri] == 1
            tl = thr_sens(s_tr[yc], 0.95); cs = c_star(s_tr[~yc], 0.90); li = I["L-LATE"][(r, k)].reindex(tri).values
            md = (s_tr >= tl) & (s_tr <= cs) if tl <= cs else np.zeros(len(tri), bool); sel = md & yc; ts = thr_sens(li[sel], 0.8) if sel.sum() >= 5 else thr_sens(li[yc], 0.8)
            tcall = np.where(s_tr < tl, False, np.where(s_tr > cs, True, li >= ts)) if tl <= cs else s_tr > cs; s_star = 1 - tcall[~yc].mean(); te = FO[:, j] == k
            for a, mn in MM.items():
                xc = np.sort(I[mn][(r, k)].reindex(tri).values[~yc]); q = int(np.ceil(s_star * len(xc))); thr_ = xc[q - 1] if q > 0 else -np.inf; mc[a][te, j] = O[mn][te, j] > thr_
    return mc
mc = matched_calls(O, I, FO, popm); CALLS = {"Triage": call, "(A)": comp["A"], "(B)": comp["B"], "(C)": comp["C"], "A*": mc["A*"], "B*": mc["B*"], "C*": mc["C*"]}
src = {"n_samples": int(len(rows)), "n_progressor_samples": int(yr.sum()), "n_patients": int(len(set(pr_))), "n_progressor_patients": int(len(set(pr_[yr == 1])))}
pr0 = J("review_round4/r33_base_primary.json")["R33_base"]["claim_T"]
assert abs(pr0["no_vote_vs_Bstar_training_matched"]["comparator_sensitivity"] - CALLS["B*"][rows][yr == 1].mean(0).mean()) < 0.002, "B* reproduction"
fig = plt.figure(figsize=(183 * MM, 170 * MM))
# ------------------------------------------------------------------ (a) single-model flow (per-repeat mean counts)
Bq = band[rows]; Cq = call[rows]; P = int(yr.sum()); N = int((yr == 0).sum())
def mcount(mask_mat, cls): return float(mask_mat[yr == cls].sum(0).mean())
flow = {}
for g, b in ((0, "low"), (1, "mid"), (2, "high")):
    flow[b] = {"progressor": mcount(Bq == g, 1), "non_progressor": mcount(Bq == g, 0), "flagged": {"progressor": mcount((Bq == g) & Cq, 1), "non_progressor": mcount((Bq == g) & Cq, 0)}}
src["flow_single_model_per_repeat_mean_counts"] = flow
ax = fig.add_axes([0.01, 0.57, 0.60, 0.39]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
box(ax, 1, 40, 24, 20, f"Pre-event biopsies\nn = {P + N}\n{P} progressor\n{N} non-progressor", "#F2F2F2", "#444444", fs=6.2)
ys = {"low": 14, "mid": 50, "high": 86}; names = {"low": "Low band: cleared,\nnot sequenced", "mid": "Middle band:\nsequenced, L-LATE call", "high": "High band: flagged,\nnot sequenced"}; f1 = lambda v: f"{v:.1f}"
for b in ("low", "mid", "high"):
    f = flow[b]; nb = f["progressor"] + f["non_progressor"]; box(ax, 34, ys[b] - 10, 28, 20, f"{names[b]}\nn = {f1(nb)}\n{f1(f['progressor'])} P / {f1(f['non_progressor'])} NP", BAND[b] + "55", BAND[b], fs=6.1); arrow(ax, 25, 50, 34, ys[b])
    cp, cn = f["flagged"]["progressor"], f["flagged"]["non_progressor"]; outs = [("Flagged", cp, cn, "#CC6677"), ("Cleared", f["progressor"] - cp, f["non_progressor"] - cn, "#88CCEE")]
    outs = [o for o in outs if o[1] + o[2] > 0.05]; off = [0] if len(outs) == 1 else [9, -9]
    for (t, a, c, col), dy in zip(outs, off): box(ax, 72, ys[b] + dy - 7, 27, 14, f"{t}: n = {f1(a + c)}\n{f1(a)} P / {f1(c)} NP", col + "33", col, fs=6.0); arrow(ax, 62, ys[b], 72, ys[b] + dy)
ax.text(1, 4, "Single model: counts are means over the 10 repeats (one fitted model per repeat).\nP, progressor; NP, non-progressor.", fontsize=5.4, color="#333333", va="top")
letter(fig, ax, "a", dx=0.01, dy=-0.02)
# ------------------------------------------------------------------ (b) share vs sensitivity, colour = specificity (sample level) and endoscopy level
r33 = J("review_round4/r33_merge.json")["R33"]["claim_T_triage_vs_Bstar_training_matched_single_model"]; f5 = J("review_round2/f5_r28.json")["F5_R28"]["pre"]; f6 = J("review_round2/f6_r25b.json")["F6_R25b"]
r14 = J("review_round2/r14_their_pre.json")["R14"]["b_no_vote_paired"]; r24 = J("review_round2/r24.json")["R24"]["comparator_D"]["D1_grade_ID_or_LGD"]; r41 = J("review_round6/r41.json")["R41"]
seeds = [json.load(open(f)) for f in sorted(glob.glob(f"{M}/review_round2/randgate/seed_*.json"))]; assert len(seeds) == 100
rs = np.array([s["share_sequenced"] for s in seeds]); rsn = np.array([s["sensitivity"] for s in seeds]); rsp = np.array([s["specificity"] for s in seeds]); pm = f6["per_permutation"]
ps = np.array([pm[k]["no_vote"]["share_sequenced"] for k in pm]); psn = np.array([pm[k]["no_vote"]["sensitivity"] for k in pm]); psp = np.array([pm[k]["no_vote"]["specificity"] for k in pm])
cmap = mpl.colormaps["viridis"]; norm = mpl.colors.Normalize(vmin=0.45, vmax=0.85); col = lambda sp: cmap(norm(sp))
pts = {"H&E gate (UNI2-h)": dict(x=pr0["share_sequenced"]["per_repeat_mean"], xr=pr0["share_sequenced"]["ci95"], y=pr0["triage_no_vote"]["sensitivity"], yr=pr0["triage_no_vote"]["sens_ci95"], sp=pr0["triage_no_vote"]["specificity"], mk="*", ms=10),
       "H-optimus-0 gate": dict(x=f5["share_sequenced"]["per_repeat_mean"], xr=f5["share_sequenced"]["ci95"], y=f5["strategies"]["no_vote"]["triage"]["sensitivity"], yr=f5["strategies"]["no_vote"]["triage"]["sens_ci95"], sp=f5["strategies"]["no_vote"]["triage"]["specificity"], mk="P", ms=6),
       "permuted-label gate (5 permutations, range)": dict(x=float(ps.mean()), xr=[float(ps.min()), float(ps.max())], y=float(psn.mean()), yr=[float(psn.min()), float(psn.max())], sp=float(psp.mean()), mk="X", ms=5.5),
       "random gate (100 seeds, 2.5–97.5% range)": dict(x=float(rs.mean()), xr=[float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))], y=float(rsn.mean()), yr=[float(np.percentile(rsn, 2.5)), float(np.percentile(rsn, 97.5))], sp=float(rsp.mean()), mk="o", ms=4.5),
       "(A)": dict(x=1, y=r14["A"]["comparator_sensitivity"], sp=r14["A"]["comparator_specificity"], mk="o", ms=4), "A*": dict(x=1, y=r14["A*"]["comparator_sensitivity"], sp=r14["A*"]["comparator_specificity"], mk="s", ms=4),
       "(B)": dict(x=1, y=pr0["no_vote_vs_B"]["comparator_sensitivity"], sp=pr0["no_vote_vs_B"]["comparator_specificity"], mk="o", ms=4), "B*": dict(x=1, y=pr0["no_vote_vs_Bstar_training_matched"]["comparator_sensitivity"], sp=pr0["no_vote_vs_Bstar_training_matched"]["comparator_specificity"], mk="s", ms=4),
       "(C)": dict(x=0, y=pr0["no_vote_vs_C"]["comparator_sensitivity"], sp=pr0["no_vote_vs_C"]["comparator_specificity"], mk="o", ms=4), "C*": dict(x=0, y=pr0["no_vote_vs_Cstar_training_matched"]["comparator_sensitivity"], sp=pr0["no_vote_vs_Cstar_training_matched"]["comparator_specificity"], mk="s", ms=4),
       "(D)": dict(x=0, y=r24["sample"]["sensitivity"], sp=r24["sample"]["specificity"], mk="D", ms=4)}
bx = fig.add_axes([0.655, 0.60, 0.19, 0.35]); LABOFF = {"(A)": (3, 0.004), "A*": (3, 0), "(B)": (3, 0), "B*": (3, 0), "(C)": (3, -0.014), "C*": (3, 0.012), "(D)": (3, 0)}
for k, d in pts.items():
    if "xr" in d: bx.plot([100 * d["xr"][0], 100 * d["xr"][1]], [d["y"], d["y"]], color=GREY, lw=0.7, zorder=2); bx.plot([100 * d["x"], 100 * d["x"]], [d["yr"][0], d["yr"][1]], color=GREY, lw=0.7, zorder=2)
    bx.plot([100 * d["x"]], [d["y"]], marker=d["mk"], color=col(d["sp"]), ms=d["ms"], lw=0, mec="black", mew=0.4, zorder=5)
    if k in LABOFF: bx.text(100 * d["x"] + LABOFF[k][0], d["y"] + LABOFF[k][1], k, fontsize=5.6, ha="left", va="center")
grid = [r for r in r33 if r["item"] == 8]
for r in grid: bx.plot([100 * r["share_sequenced"]], [r["triage_sens_spec"][0]], marker="o", color=col(r["triage_sens_spec"][1]), mec="black", mew=0.3, ms=3, lw=0, zorder=4, alpha=0.9)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=d["mk"], color="white", mec="black", mew=0.5, ms=min(d["ms"], 6), lw=0, label=k.split(" (")[0]) for k, d in pts.items() if k in ("H&E gate (UNI2-h)", "H-optimus-0 gate", "permuted-label gate (5 permutations, range)", "random gate (100 seeds, 2.5–97.5% range)")]
h.append(Line2D([], [], marker="o", color="white", mec="black", mew=0.3, ms=3, lw=0, label="H&E gate, 9 cut-off settings")); h.append(Line2D([], [], marker="s", color="white", mec="black", mew=0.5, ms=4, lw=0, label="matched A*, B*, C*"))
bx.legend(handles=h, loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=4.8, handletextpad=0.3); bx.set_xlabel("share of biopsies sequenced (%)"); bx.set_ylabel("sensitivity (single model, per biopsy)"); bx.set_xlim(-6, 118); bx.set_ylim(0.3, 1.0)
cax = fig.add_axes([0.655, 0.535, 0.19, 0.012]); cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal"); cb.set_label("specificity (marker colour)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
letter(fig, bx, "b", dx=-0.095, dy=-0.01)
src["b_sample_level"] = {k: {kk: v for kk, v in d.items() if kk not in ("mk", "ms")} for k, d in pts.items()}; src["b_grid_cells"] = [{"analysis": r["analysis"], "share": r["share_sequenced"], "sensitivity": r["triage_sens_spec"][0], "specificity": r["triage_sens_spec"][1]} for r in grid]
# endoscopy level (R41)
ex_ = fig.add_axes([0.915, 0.60, 0.075, 0.35]); E = r41["levels"]["endoscopy"]; shr = r41["share_sequenced"]
ept = {"triage": (shr["triage_per_repeat_mean"], "sens_triage", "spec_triage", "*", 9), "one random\nbiopsy": (shr["random_biopsy"], "sens_random_biopsy", "spec_random_biopsy", "o", 4.5), "random biopsy\n+ H&E on rest": (shr["random_biopsy"], "sens_random_biopsy_rest_HE", "spec_random_biopsy_rest_HE", "P", 5.5), "(B) all\nsequenced": (1.0, "sens_B_all_sequenced", "spec_B_all_sequenced", "o", 4)}
src["b_endoscopy_level"] = {}
for k, (x, ks, kp, mk, ms) in ept.items():
    se, sp = E[ks], E[kp]; ex_.plot([100 * x, 100 * x], se["ci95_bootstrap_of_seed_mean"], color=GREY, lw=0.7, zorder=2); ex_.plot([100 * x], [se["mean_over_seeds"]], marker=mk, color=col(sp["mean_over_seeds"]), ms=ms, lw=0, mec="black", mew=0.4, zorder=5)
    src["b_endoscopy_level"][k.replace("\n", " ")] = {"share": x, "sensitivity": se, "specificity": sp}
ex_.text(100 * shr["triage_per_repeat_mean"] - 4, E["sens_triage"]["mean_over_seeds"] + 0.03, "triage", fontsize=5.2, ha="right"); ex_.text(100 * shr["random_biopsy"] + 3, E["sens_random_biopsy"]["mean_over_seeds"] - 0.03, "1 random\nbiopsy", fontsize=5.0, va="top")
ex_.text(100 * shr["random_biopsy"] + 3, E["sens_random_biopsy_rest_HE"]["mean_over_seeds"] + 0.005, "+ H&E\non rest", fontsize=5.0, va="bottom"); ex_.text(97, E["sens_B_all_sequenced"]["mean_over_seeds"] - 0.03, "(B)", fontsize=5.2, ha="right", va="top")
ex_.set_xlim(30, 112); ex_.set_ylim(0.7, 1.0); ex_.set_xlabel("sequenced (%)", fontsize=6); ex_.set_ylabel("sensitivity per endoscopy", fontsize=6, labelpad=1); ex_.set_title("endoscopy level", fontsize=6, pad=8, loc="right")
# ------------------------------------------------------------------ (c) per-patient outcomes with CIs
pp = pd.Series(pr_); up = pp.unique(); isP = pd.Series(yr).groupby(pp).max().reindex(up).values == 1; nP, nN = int(isP.sum()), int((~isP).sum())
def pat_stats(b, tag):   # per draw: per-repeat mean share of progressor patients missed and non-progressor patients flagged (units = patient x tag)
    key = pd.Series(pat[b]).astype(str).values + "#" + tag.astype(str); lab = pd.Series(y[b]).groupby(key).max(); out = {}
    for s, C in CALLS.items():
        A = pd.DataFrame(C[b]).groupby(key).max(); out[f"miss_{s}"] = float(1 - A[lab.values == 1].mean(0).mean()); out[f"flag_{s}"] = float(A[lab.values == 0].mean(0).mean())
    return out
def boots_tagged(rows_, seed=0, nb=NB):
    pp_ = pat[rows_]; up_ = np.unique(pp_); ro = {q: rows_[pp_ == q] for q in up_}; rng = np.random.RandomState(seed); out = []
    while len(out) < nb:
        dr = rng.choice(len(up_), len(up_)); b = np.concatenate([ro[up_[i]] for i in dr]); t = np.concatenate([np.full(len(ro[up_[i]]), q) for q, i in enumerate(dr)])
        if 0 < y[b].sum() < len(b): out.append((b, t))
    return out
obs = pat_stats(rows, np.zeros(len(rows), int)); BT = boots_tagged(rows); D = pd.DataFrame([pat_stats(b, t) for b, t in BT])
pats = {s: {"progressor_patients_missed": {"mean": obs[f"miss_{s}"] * nP, "ci95": [float(np.percentile(D[f"miss_{s}"], 2.5)) * nP, float(np.percentile(D[f"miss_{s}"], 97.5)) * nP], "ci": "patient bootstrap"},
            "non_progressor_patients_flagged": {"mean": obs[f"flag_{s}"] * nN, "ci95": [float(np.percentile(D[f"flag_{s}"], 2.5)) * nN, float(np.percentile(D[f"flag_{s}"], 97.5)) * nN], "ci": "patient bootstrap"}} for s in CALLS}
r37f = J("review_round6/r37.json")["R37"]["f_missing_as_wild_type_all_pre_event"]["rules"]
for s, k in (("(D)", "D"), ("(D)+p53", "D+p53")):
    q = r37f[k]["patient"]; pats[s] = {"progressor_patients_missed": {"mean": (1 - q["sensitivity"]) * nP, "ci95": [(1 - q["sens_ci95_exact"][1]) * nP, (1 - q["sens_ci95_exact"][0]) * nP], "ci": "exact (Clopper-Pearson)"},
                                       "non_progressor_patients_flagged": {"mean": (1 - q["specificity"]) * nN, "ci95": [(1 - q["spec_ci95_exact"][1]) * nN, (1 - q["spec_ci95_exact"][0]) * nN], "ci": "exact (Clopper-Pearson)"}}
src["per_patient_by_strategy"] = pats; order = ["Triage", "(A)", "(B)", "(C)", "A*", "B*", "C*", "(D)", "(D)+p53"]
SC = {"Triage": "black", "(A)": MODEL["L-CNV"], "(B)": MODEL["L-LATE"], "(C)": MODEL["L-IMG"], "A*": MODEL["L-CNV"], "B*": MODEL["L-LATE"], "C*": MODEL["L-IMG"], "(D)": MODEL["grade"], "(D)+p53": MODEL["p53"]}
for q_, key in enumerate(("progressor_patients_missed", "non_progressor_patients_flagged")):
    a_ = fig.add_axes([0.07 + q_ * 0.155, 0.08, 0.11, 0.36])
    for i, k in enumerate(order):
        v = pats[k][key]; yy = len(order) - 1 - i; a_.plot(v["ci95"], [yy, yy], color=SC[k], lw=0.9, solid_capstyle="butt"); a_.plot([v["mean"]], [yy], marker="s" if k.endswith("*") else "o", color=SC[k], ms=3.6, lw=0)
    a_.set_yticks(range(len(order))); a_.set_yticklabels(order[::-1] if q_ == 0 else [""] * len(order)); a_.set_ylim(-0.6, len(order) - 0.4)
    a_.set_xlabel("progressor\npatients missed\n(of 32)" if q_ == 0 else "non-progressor\npatients flagged\n(of 43)", fontsize=6.2); a_.set_xlim(-0.5, 32.5 if q_ == 0 else 43.5)
    if q_ == 0: cxa = a_
letter(fig, cxa, "c", dx=-0.055)
# ------------------------------------------------------------------ (d) predictive values against prevalence
dx_ = fig.add_axes([0.42, 0.10, 0.25, 0.33]); boots = boots_for(rows); piv = np.linspace(0.005, 0.30, 60)
def rates(b):
    Bb = band[b]; yb = y[b]; return np.array([(Bb[yb == 1] == g).mean(0).mean() for g in range(3)]), np.array([(Bb[yb == 0] == g).mean(0).mean() for g in range(3)])
def curves(r1, r0):
    ppv = piv * r1[2] / (piv * r1[2] + (1 - piv) * r0[2]); npv = (1 - piv) * r0[0] / ((1 - piv) * r0[0] + piv * r1[0]); share = piv * r1[1] + (1 - piv) * r0[1]; return ppv, npv, share
obsc = curves(*rates(rows)); BS = np.array([np.stack(curves(*rates(b))) for b in boots]); lo = np.percentile(BS, 2.5, axis=0); hi = np.percentile(BS, 97.5, axis=0)
for q_, (lab, c_) in enumerate((("high-band PPV (post-test risk if flagged)", BAND["high"]), ("low-band NPV", BAND["low"]), ("share sequenced (middle band)", BAND["mid"]))):
    dx_.fill_between(100 * piv, lo[q_], hi[q_], color=c_, alpha=0.35, lw=0); dx_.plot(100 * piv, obsc[q_], color=c_, lw=1.2, label=lab)
prev = float(yr.mean()); dx_.axvline(2, color=LIGHT, lw=0.6, ls=":"); dx_.axvline(100 * prev, color=GREY, lw=0.6, ls="--"); i2 = int(np.argmin(np.abs(piv - 0.02)))
dx_.text(100 * prev - 0.5, 0.30, f"cohort prevalence {100 * prev:.0f}%\n(case–control; curves\nare projections)", fontsize=5.2, ha="right", va="top", color="#333333")
dx_.text(29.5, 0.03, f"at 2% (surveillance-like): PPV {100 * obsc[0][i2]:.1f}%,\nNPV {100 * obsc[1][i2]:.1f}%, sequenced {100 * obsc[2][i2]:.1f}%", fontsize=5.2, va="bottom", ha="right")
dx_.set_xlabel("progressor-biopsy prevalence (%)"); dx_.set_ylabel("proportion"); dx_.set_xlim(0, 30); dx_.set_ylim(0, 1.02); dx_.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=5.2)
r1o, r0o = rates(rows); src["class_wise_band_rates"] = {"progressor": r1o.tolist(), "non_progressor": r0o.tolist()}; src["cohort_prevalence"] = prev
src["prevalence_curves"] = {"prevalence": piv.tolist(), "ppv_high": obsc[0].tolist(), "ppv_high_ci": [lo[0].tolist(), hi[0].tolist()], "npv_low": obsc[1].tolist(), "npv_low_ci": [lo[1].tolist(), hi[1].tolist()], "share_sequenced": obsc[2].tolist(), "share_ci": [lo[2].tolist(), hi[2].tolist()], "draws": len(boots)}
letter(fig, dx_, "d", dx=-0.06)
# ------------------------------------------------------------------ (e) band stability + unanimity of the voted call (modal band: argmax, ties to the lower band)
ex = fig.add_axes([0.775, 0.10, 0.18, 0.33]); modal = np.array([np.bincount(r_, minlength=3).argmax() for r_ in Bq]); nmod = np.array([(r_ == m).sum() for r_, m in zip(Bq, modal)])
V = Cq.sum(1) >= 6; agree = (Cq == V[:, None]).sum(1); st = J("review_round6/r39_merge.json")["R39"]["stability_of_voted_call"]["by_modal_band_and_class"]
keys = [(c, v, g, bn) for c, v in (("progressor", 1), ("non_progressor", 0)) for g, bn in ((0, "low"), (1, "mid"), (2, "high"))]; greys = {10: "#222222", 9: "#555555", 8: "#808080", 7: "#A6A6A6", 6: "#CCCCCC", 5: "#E8E8E8"}; src["band_stability"] = {}
for i, (c, v, g, bn) in enumerate(keys):
    mk = (yr == v) & (modal == g); tot = int(mk.sum()); left = 0; yy = len(keys) - 1 - i; dist = {int(k): int((nmod[mk] == k).sum()) for k in range(10, 4, -1)}
    for r_ in (10, 9, 8, 7, 6, 5): ex.barh(yy, 100 * dist[r_] / tot, left=left, color=greys[r_], edgecolor="white", lw=0.3, height=0.62); left += 100 * dist[r_] / tot
    u10 = float((agree[mk] == 10).mean()); u8 = float((agree[mk] >= 8).mean()); ref = st[c][bn]; assert ref["n"] == tot and abs(ref["unanimous_10_of_10"] - u10) < 0.002, (c, bn, tot, ref, u10)
    ex.plot([100 * u8], [yy - 0.42], marker="D", color="white", mec=MODEL["L-IMG"], mew=0.7, ms=3, lw=0, zorder=4); ex.plot([100 * u10], [yy - 0.42], marker="D", color=MODEL["L-IMG"], ms=3, lw=0, zorder=5)
    ex.text(102, yy, f"n {tot}", fontsize=5.4, va="center", clip_on=False); src["band_stability"][f"{c}_{bn}"] = {"n": tot, "repeats_in_modal_band": dist, "voted_call_unanimous_10_of_10": u10, "voted_call_ge_8_of_10": u8}
ex.set_yticks(range(len(keys))); ex.set_yticklabels([f"{'P' if c == 'progressor' else 'NP'}, {bn} band" for c, v, g, bn in keys][::-1]); ex.set_xlabel("biopsies (%)"); ex.set_xlim(0, 100); ex.set_ylim(-0.8, len(keys) + 4.6)
for r_ in (10, 9, 8, 7, 6, 5): ex.barh([-10], [0], color=greys[r_], label=f"{r_} of 10")
ex.plot([], [], marker="D", color=MODEL["L-IMG"], ms=3, lw=0, label="vote: all 10 agree"); ex.plot([], [], marker="D", color="white", mec=MODEL["L-IMG"], mew=0.7, ms=3, lw=0, label="vote: ≥ 8 of 10 agree")
ex.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=4.8, title="bars: repeats in the modal band\ndiamonds: repeats agreeing with the vote", title_fontsize=4.9, columnspacing=0.8, handletextpad=0.3, alignment="left")
letter(fig, ex, "e", dx=-0.09, dy=0.008)
save(fig, "Fig3"); dump("Fig3", src)
