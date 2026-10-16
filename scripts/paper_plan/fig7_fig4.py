"""Figure 4 (manuscript v7), patients over time, drawn on the cluster. (a) swimmer plot: one anonymous row per patient (75), ordered by class and then
by the span of pre-event sampling (first to last pre-event endoscopy, as in Table 1); pre-event biopsies placed by time before the first HGD/IMC
endoscopy (progressors) or the last sequenced endoscopy (non-progressors), coloured by modal band (most frequent band over the 10 repeats, ties to the
lower band) and shaped by grade; endoscopy-level decision = any biopsy positive under the 10-model vote (>= 6 of 10 repeats), flagged endoscopies
ticked and the first flagged endoscopy ringed. (b) fold-stratified IPCW time-dependent AUROC at 1, 3 and 5 years (committed). Aggregate source data only."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig7_style import *
from tb_common import *
from matplotlib.lines import Line2D
O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]; yr = y[rows]; band, call, info = triage(O, I, FO, popm)
Bq = band[rows]; modal = np.array([np.bincount(r_, minlength=3).argmax() for r_ in Bq]); V = call[rows].sum(1) >= 6
TIME = SM.time.values.astype(float)[rows]; GR = SM.Pathology.values[rows]; PT = pat[rows]; EN = ENDO[rows]
D = pd.DataFrame({"pat": PT, "y": yr, "t": TIME, "grade": GR, "band": modal, "vote": V, "endo": EN})
E = D.groupby(["pat", "endo"]).agg(t=("t", "max"), flag=("vote", "any"), y=("y", "max")).reset_index()
span = D.groupby("pat").agg(y=("y", "max"), tmax=("t", "max"), tmin=("t", "min"), n=("t", "size")); span["span"] = span.tmax - span.tmin; span = span.sort_values(["y", "span"], ascending=[False, False]); order = {p: i for i, p in enumerate(span.index)}
first_flag = E[E.flag].groupby("pat").t.max()   # largest time before the event = earliest flagged endoscopy
fig = plt.figure(figsize=(183 * MM, 160 * MM)); ax = fig.add_axes([0.05, 0.08, 0.60, 0.89]); BN = {0: "low", 1: "mid", 2: "high"}; MK = {"NDBE": "o", "ID": "^", "LGD": "s"}
nrow = len(span); yrow = lambda p: nrow - 1 - order[p]
for p, r in span.iterrows():
    yy = yrow(p); ax.plot([-r.tmax, -r.tmin], [yy, yy], color="#CCCCCC", lw=0.9, zorder=1)
    if r.tmin > 0: ax.plot([-r.tmin, 0], [yy, yy], color="#DDDDDD", lw=0.5, ls=":", zorder=1)
    if r.y == 1: ax.plot([0], [yy], marker="x", color="black", ms=3.5, mew=0.8, lw=0, zorder=4)
    else: ax.plot([0], [yy], marker="|", color=GREY, ms=4, mew=0.7, lw=0, zorder=4)
for _, e in E[E.flag].iterrows(): ax.plot([-e.t], [yrow(e.pat) + 0.36], marker="v", color="#CC3311", ms=2.2, lw=0, zorder=4)
for p, t in first_flag.items(): ax.plot([-t], [yrow(p)], marker="o", color="none", mec="#CC3311", mew=0.7, ms=5.5, lw=0, zorder=5)
for (p, t), g in D.groupby(["pat", "t"]):
    k = len(g); off = (np.arange(k) - (k - 1) / 2) * 0.28 if k > 1 else np.zeros(1)
    for (idx, s), o in zip(g.iterrows(), off): ax.plot([-s.t], [yrow(p) + o], marker=MK.get(s.grade, "o"), color=BAND[BN[s.band]], ms=2.6, mec="#333333", mew=0.25, lw=0, zorder=3)
nP = int(span.y.sum()); ax.axhline(nrow - nP - 0.5, color="black", lw=0.5, ls=":")
ax.text(-13.8, nrow - nP / 2 - 0.5, f"progressors (n {nP})", rotation=90, va="center", ha="center", fontsize=6.5); ax.text(-13.8, (nrow - nP) / 2 - 0.5, f"non-progressors (n {nrow - nP})", rotation=90, va="center", ha="center", fontsize=6.5)
ax.set_xlim(-14.5, 0.8); ax.set_ylim(-1, nrow); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel("years before the first HGD/IMC endoscopy (progressors)\nor the last sequenced endoscopy (non-progressors)")
h = [Line2D([], [], marker="o", color=BAND[b], mec="#333333", mew=0.25, lw=0, ms=4, label=BAND_NAME[b] + " (modal band)") for b in ("low", "mid", "high")]
h += [Line2D([], [], marker=MK[g], color="white", mec="#333333", mew=0.6, lw=0, ms=4, label=f"grade {g}") for g in ("NDBE", "ID", "LGD")]
h += [Line2D([], [], marker="v", color="#CC3311", lw=0, ms=3, label="endoscopy flagged (10-model vote, any biopsy)"), Line2D([], [], marker="o", color="none", mec="#CC3311", mew=0.7, lw=0, ms=5.5, label="first flagged endoscopy"),
      Line2D([], [], color="#CCCCCC", lw=0.9, label="span of pre-event sampling"), Line2D([], [], marker="x", color="black", lw=0, ms=4, mew=0.8, label="first HGD/IMC endoscopy"), Line2D([], [], marker="|", color=GREY, lw=0, ms=5, mew=0.7, label="last sequenced endoscopy")]
fig.legend(handles=h, loc="upper left", fontsize=5.6, ncol=1, bbox_to_anchor=(0.70, 0.50))
letter(fig, ax, "a", dx=-0.035, dy=-0.01)
# ------------------------------------------------------------------ (b) time-dependent AUROC
bx = fig.add_axes([0.74, 0.60, 0.25, 0.35]); H = (1, 3, 5); arms = {"grade": ("GRADE", MODEL["grade"]), "L-CNV": ("L-CNV", MODEL["L-CNV"]), "L-IMG": ("L-IMG", MODEL["L-IMG"]), "L-LATE": ("L-LATE", MODEL["L-LATE"])}; td = {}
for i, (nm, (key, c_)) in enumerate(arms.items()):
    td[nm] = {}
    for hz in H:
        if nm == "grade": g = J(f"horizon_grade_row/hg_their_{hz}.json")["arms"]["GRADE"]["foldstrat"]; v, ci = g["auroc"], g["ci95"]
        else: g = J(f"horizon_foldstrat/fs_their_pre_{hz}.json")["arms"][key]; v, ci = g["ipcw"], g["ipcw_ci95"]
        td[nm][hz] = {"auroc": v, "ci95": ci}; ci_point(bx, hz + (i - 1.5) * 0.28, v, ci, c_, ms=3.5, label=MODEL_NAME[nm] if hz == 1 else None)
bx.axhline(0.5, color=LIGHT, lw=0.6, ls=":"); bx.set_xticks(H); bx.set_xticklabels([f"{h_} year{'s' if h_ > 1 else ''}" for h_ in H]); bx.set_ylabel("time-dependent AUROC (fold-stratified, IPCW)"); bx.set_ylim(0.3, 1.04); bx.set_xlim(0, 6)
bx.legend(loc="lower left", fontsize=5.6); cases = {hz: J(f"horizon_foldstrat/fs_their_pre_{hz}.json") for hz in H}
for hz in H: bx.text(hz, 1.035, f"{cases[hz]['n_cases']} cases /\n{cases[hz]['n_controls']} controls", ha="center", va="top", fontsize=5.2, color="#333333")
letter(fig, bx, "b", dx=-0.07)
save(fig, "Fig4")
ffP = first_flag.reindex(span.index[span.y == 1]); ffN = first_flag.reindex(span.index[span.y == 0])
agg = {"n_patients": int(nrow), "n_progressor_patients": nP, "biopsies_by_class_band_grade": {CLASS_NAME[c]: {BN[b]: {g: int(((D.y == v) & (D.band == b) & (D.grade == g)).sum()) for g in ("NDBE", "ID", "LGD")} for b in range(3)} for c, v in (("P", 1), ("N", 0))},
       "span_of_pre_event_sampling_years": {CLASS_NAME[c]: {"median": float(span.span[span.y == v].median()), "max": float(span.span[span.y == v].max())} for c, v in (("P", 1), ("N", 0))},
       "endoscopies": {"n": int(len(E)), "flagged_voted": int(E.flag.sum()), "progressor_flagged": int((E.flag & (E.y == 1)).sum()), "non_progressor_flagged": int((E.flag & (E.y == 0)).sum())},
       "patients_with_any_flagged_endoscopy_voted": {"progressor": int(ffP.notna().sum()), "non_progressor": int(ffN.notna().sum())}, "first_flag_years_before_event_progressors_voted": {"median": float(ffP.dropna().median()), "iqr": [float(ffP.dropna().quantile(0.25)), float(ffP.dropna().quantile(0.75))]},
       "modal_band_rule": "most frequent band over the 10 repeats; ties to the lower band", "time_dependent_auroc": td}
dump("Fig4", agg)
