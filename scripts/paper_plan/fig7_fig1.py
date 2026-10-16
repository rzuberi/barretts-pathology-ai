"""Figure 1 (manuscript v7), study design: (a) CONSORT-style flow with biopsy, patient and endoscopy counts; (b) pipeline (published hg19 matrix as the
primary copy-number input, re-segmented variants as sensitivity analyses); (c) the three-band rule with per-repeat mean band shares; (d) timing of
pre-event biopsies before the first HGD/IMC endoscopy (progressors) or the last sequenced endoscopy (non-progressors). Drawn on the cluster: counts are
cross-checked against the sample sheet (killcoyne_mm/horizons/samples.csv, set_C.csv Endoscopy)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig7_style import *
from tb_common import SM, y, pat, PRE, NDBE, ENDO
rows = np.where(PRE)[0]; nd = np.where(PRE & NDBE)[0]; TIME = SM.time.values.astype(float); allr = np.arange(len(SM))
def cnt(idx): return {"samples": int(len(idx)), "patients": int(len(set(pat[idx]))), "endoscopies": int(len(set(ENDO[idx]))), "progressor_patients": int(len(set(pat[idx][y[idx] == 1]))), "progressor_samples": int(y[idx].sum()),
                      "progressor_endoscopies": int(len(set(ENDO[idx][y[idx] == 1])))}
chk = {"all": cnt(allr), "pre_event": cnt(rows), "ndbe_pre_event": cnt(nd)}
exp = {"all": (676, 80, 37, 266), "pre_event": (571, 75, 32, 161), "ndbe_pre_event": (438, 71, 28, 104)}
for k, (a, b, c, d) in exp.items(): assert (chk[k]["samples"], chk[k]["patients"], chk[k]["progressor_patients"], chk[k]["progressor_samples"]) == (a, b, c, d), (k, chk[k])
assert chk["pre_event"]["endoscopies"] == 288 and chk["pre_event"]["progressor_endoscopies"] == 88, chk["pre_event"]
fig = plt.figure(figsize=(183 * MM, 150 * MM))
# ------------------------------------------------------------------ (a) flow
ax = fig.add_axes([0.02, 0.04, 0.46, 0.93]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
FC, EC = "#F2F2F2", "#444444"; XC, XE = "#FFFFFF", "#999999"; A, Pq, Nq = chk["all"], chk["pre_event"], chk["ndbe_pre_event"]
flow = [("Discovery cohort (Killcoyne et al. 2020)\n88 patients, 777 sequenced biopsies\n45 progressors, 43 non-progressors", 90),
        (f"Biopsies with a matched scanned H&E slide\n80 patients, {A['endoscopies']} endoscopies, 676 biopsies (683 slides)\n37 progressors (266 biopsies, {A['progressor_endoscopies']} endoscopies)\n43 non-progressors (410 biopsies, {A['endoscopies'] - A['progressor_endoscopies']} endoscopies)", 63),
        (f"Primary population: pre-event biopsies\n75 patients, {Pq['endoscopies']} endoscopies, 571 biopsies\n32 progressors (161 biopsies, {Pq['progressor_endoscopies']} endoscopies)\n43 non-progressors (410 biopsies, {Pq['endoscopies'] - Pq['progressor_endoscopies']} endoscopies)", 36),
        (f"Secondary population: NDBE pre-event biopsies\n71 patients, {Nq['endoscopies']} endoscopies, 438 biopsies\n28 progressors (104 biopsies, {Nq['progressor_endoscopies']} endoscopies)\n43 non-progressors (334 biopsies, {Nq['endoscopies'] - Nq['progressor_endoscopies']} endoscopies)", 9)]
excl = [("No matched slide:\n101 biopsies; 8 patients\nwithout any slide\n(all progressors)", 76.5),
        (f"At or after the first\nHGD/IMC endoscopy:\n105 biopsies ({A['endoscopies'] - Pq['endoscopies']} endoscopies);\n5 progressors with no\npre-event biopsy", 49.5),
        (f"Graded ID or LGD:\n133 biopsies; 4 progressors\nwith no NDBE\npre-event biopsy", 22.5)]
H = 15; W = 64
for i, (t, yc) in enumerate(flow):
    box(ax, 0.5, yc - H / 2, W, H, t, FC, EC, fs=5.3, weight="bold" if i == 2 else "normal")
    if i < 3: arrow(ax, 32, yc - H / 2, 32, flow[i + 1][1] + H / 2)
for t, yc in excl:
    box(ax, 68.5, yc - 6.5, 30, 13, t, XC, XE, fs=5.3); arrow(ax, 32, yc, 68.5, yc, color="#999999")
letter(fig, ax, "a", dx=0.005)
# ------------------------------------------------------------------ (b) pipeline
bx = fig.add_axes([0.51, 0.60, 0.48, 0.37]); bx.set_xlim(0, 100); bx.set_ylim(0, 100); bx.axis("off")
def chain(y0, items, col):
    xs = [1, 22, 43, 64]; w = 19; h = 28
    for (t, x) in zip(items, xs): box(bx, x, y0 - h / 2, w, h, t, col + "22", col, fs=5.3)
    for x in xs[:-1]: arrow(bx, x + w, y0, x + 21, y0)
chain(76, ["Biopsy:\nH&E slide", "256 tiles\n224 px,\n0.88 µm/px", "UNI2-h\n1,536 features,\nmean over\ntiles", "L-IMG\nelastic-net\nlogistic"], MODEL["L-IMG"])
chain(24, ["Biopsy:\nDNA", "sWGS\n~0.3× depth", "Copy number:\npublished matrix\n(hg19, joint segm.)\nprimary; rebuilt joint,\npast-only, per-sample\nas sensitivity", "L-CNV\nelastic-net\nlogistic"], MODEL["L-CNV"])
box(bx, 85, 38, 14, 24, "L-LATE\n= mean of\nL-IMG and\nL-CNV", MODEL["L-LATE"] + "22", MODEL["L-LATE"], fs=5.4)
arrow(bx, 83, 76, 92, 62.5); arrow(bx, 83, 24, 92, 37.5)
bx.text(1, 99, "Patient-grouped 10 × 10-fold CV, one fitted model per repeat;\nevaluation on held-out pre-event biopsies", fontsize=5.6, va="top", color="#333333")
letter(fig, bx, "b", dx=-0.01)
# ------------------------------------------------------------------ (c) three-band rule
cx = fig.add_axes([0.53, 0.38, 0.45, 0.20]); cx.set_xlim(0, 100); cx.set_ylim(0, 100); cx.axis("off")
tp = J("triage/their_pre.json"); sh = {b: tp["bands"][b]["all"]["value"] for b in ("low", "mid", "high")}; cut = J("review_round4/r33_base_primary.json")["R33_base"]["cutoffs_median"]
x0 = 2; wtot = 96; edges = [x0, x0 + wtot * sh["low"], x0 + wtot * (sh["low"] + sh["mid"]), x0 + wtot]
for b, (l, r) in zip(("low", "mid", "high"), zip(edges[:-1], edges[1:])):
    cx.add_patch(plt.Rectangle((l, 62), r - l, 14, fc=BAND[b], ec="none")); cx.text((l + r) / 2, 69, f"{b} band\n{100 * sh[b]:.0f}% of biopsies", ha="center", va="center", fontsize=6)
cx.annotate("", (x0 + wtot + 1, 62), (x0 - 1, 62), arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7, mutation_scale=7)); cx.text(x0 + wtot / 2, 56, "H&E score (L-IMG); band widths = per-repeat mean share of biopsies, not the score scale", ha="center", va="top", fontsize=5.2, color="#333333")
for e, lab, rule in ((edges[1], f"t_low = {cut['t_low']:.3f}", "95% training sensitivity"), (edges[2], f"t_high = {cut['c_star']:.3f}", "90% training specificity")):
    cx.plot([e, e], [60, 80], color="black", lw=0.8); cx.text(e, 82, lab, ha="center", va="bottom", fontsize=6, fontweight="bold"); cx.text(e, 94, rule, ha="center", va="bottom", fontsize=5.6, color="#333333")
txt = {"low": "cleared without\nsequencing", "mid": f"sequenced; L-LATE ≥ t_seq = {cut['t_seq']:.3f}\n(80% training sensitivity in the band)\nflags, otherwise cleared", "high": "flagged\nwithout\nsequencing"}
for b, (l, r) in zip(("low", "mid", "high"), zip(edges[:-1], edges[1:])): cx.text((l + r) / 2, 40, txt[b], ha="center", va="top", fontsize=5.8)
cx.text(x0, 6, "Cut-offs are chosen in each fold on inner out-of-fold training predictions;\nvalues shown are the medians over the 100 folds (the frozen cut-offs).", fontsize=5.4, color="#333333", va="top")
letter(fig, cx, "c", dx=-0.03, dy=-0.01)
# ------------------------------------------------------------------ (d) timeline
dx_ = fig.add_axes([0.57, 0.11, 0.41, 0.21]); bins = np.arange(0, 15, 1.0); hist = {}
for cls, v, dxo in (("P", 1, -0.2), ("N", 0, 0.2)):
    t = TIME[rows][y[rows] == v]; h, _ = np.histogram(t, bins); hist[cls] = h
    dx_.bar(bins[:-1] + 0.5 + dxo, h, width=0.38, color=CLASS[cls], label=f"{CLASS_NAME[cls]} (n {len(t)}, median {np.median(t):.1f} y)")
dx_.set_xlabel("years before the first HGD/IMC endoscopy (P)\nor the last sequenced endoscopy (NP)", fontsize=6.3); dx_.set_ylabel("pre-event biopsies"); dx_.set_xlim(0, 14); dx_.legend(loc="upper right")
letter(fig, dx_, "d", dx=-0.08)
save(fig, "Fig1")
dump("Fig1", {"counts_checked_against_sample_sheet": chk, "flow_text": [t for t, _ in flow], "exclusions_text": [t for t, _ in excl], "band_shares_per_repeat_mean": sh, "frozen_cutoffs_median": cut,
              "timeline_bins_years": bins.tolist(), "timeline_counts": {CLASS_NAME[c]: hist[c].tolist() for c in hist},
              "timeline_median_years": {CLASS_NAME[c]: float(np.median(TIME[rows][y[rows] == v])) for c, v in (("P", 1), ("N", 0))}})
