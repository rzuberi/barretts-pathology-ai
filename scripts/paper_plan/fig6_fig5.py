"""Figure 5, copy-number segmentation (docs/paper_figures_v6.md @ d46b9a5), drawn on the cluster from the R36 re-segmentation outputs
(killcoyne_mm/review_round4/reseg/tiles_{joint_stored,persample,past}_analysed.csv, final blocks cnv_pkg_C.csv and r36_cnv_pkgps_C.csv).
(a) genome-wide 5-Mb segmented profiles of one progressor's earliest pre-event biopsy under joint, per-sample and past-only segmentation, with the
patient's later HGD/IMC biopsy (joint segmentation, as stored) for reference; the patient is the progressor with a stored HGD/IMC biopsy whose earliest
pre-event biopsy has the largest joint-versus-per-sample difference (1 - Pearson r over the 587 raw 5-Mb tiles); anonymous. (b) committed L-CNV AUROC by
copy-number version against L-IMG, with delta and one-sided bound (R33 Claim H rows). (c) per-biopsy Pearson r between joint and per-sample features (raw
5-Mb tiles; final 632-feature block) for pre-event biopsies, by class and by whether the patient has a stored later HGD/IMC biopsy. No identifiers."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig6_style import *
from tb_common import *
RS = M + "/review_round4/reseg"; rd = lambda f: pd.read_csv(f, dtype={"Sample": str}).set_index("Sample").reindex(ids)
TJ, TP, TQ = rd(RS + "/tiles_joint_stored_analysed.csv"), rd(RS + "/tiles_persample_analysed.csv"), rd(RS + "/tiles_past_analysed.csv")
tc = [c for c in TJ.columns if ":" in c]; assert len(tc) == 587 + 44; five = tc[:-44]; chrom = np.array([c.split(":")[0] for c in five])
XJ = pd.read_csv(M + "/cnv_pkg_C.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(ids); XP = pd.read_csv(M + "/review_round4/r36_cnv_pkgps_C.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(ids)
assert list(XJ.columns) == list(XP.columns) and XJ.notna().all().all() and XP.notna().all().all() and TP[five].notna().all().all()
def rowcorr(A, B): A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True); return (A * B).sum(1) / np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
r_tiles = rowcorr(TJ[five].values, TP[five].values); r_block = rowcorr(XJ.values, XP.values)
MBF = SM.mbf.values.astype(float); GR = SM.Pathology.values; post = (~PRE) & np.isin(GR, ["HGD", "IMC"]); has_post = np.isin(pat, np.unique(pat[post]))
# ------------------------------------------------------------------ (a) choose the patient
cands = []
for p in np.unique(pat[(y == 1) & has_post & PRE]):
    pre_idx = np.where((pat == p) & PRE)[0]; i0 = pre_idx[np.argmax(MBF[pre_idx])]; cands.append((1 - r_tiles[i0], p, i0))
cands.sort(reverse=True); diff, p, i0 = cands[0]; post_idx = np.where((pat == p) & post)[0]; i1 = post_idx[np.argmax(MBF[post_idx])]
tracks = [("joint segmentation (stored)", TJ[five].values[i0], MODEL["L-CNV-joint"]), ("per-sample segmentation", TP[five].values[i0], MODEL["L-CNV-ps"]), ("past-only segmentation", TQ[five].values[i0], MODEL["L-CNV-past"]),
          (f"later HGD/IMC biopsy ({GR[i1]}), joint segmentation", TJ[five].values[i1], "#333333")]
fig = plt.figure(figsize=(183 * MM, 150 * MM)); x = np.arange(len(five)); bounds = np.where(chrom[1:] != chrom[:-1])[0] + 0.5; centres = [np.mean(x[chrom == c]) for c in dict.fromkeys(chrom)]
vmin, vmax = min(t[1].min() for t in tracks), max(t[1].max() for t in tracks); pad = 0.08 * (vmax - vmin); axes = []
for k, (lab, v, col) in enumerate(tracks):
    a = fig.add_axes([0.06, 0.545 + (3 - k) * 0.10, 0.60, 0.09]); axes.append(a)
    for b in bounds: a.axvline(b, color="#E6E6E6", lw=0.4, zorder=0)
    a.axhline(1.0, color=LIGHT, lw=0.5, ls=":"); a.step(x, v, where="mid", color=col, lw=0.8); a.set_ylim(vmin - pad, vmax + pad); a.set_xlim(-1, len(five))
    a.text(0.005, 0.95, lab, transform=a.transAxes, fontsize=6, va="top", color=col if k < 3 else "black")
    a.set_xticks([]); a.spines["bottom"].set_visible(k == 3)
    if k == 3: a.set_xticks(centres); a.set_xticklabels([c for c in dict.fromkeys(chrom)], fontsize=5.2); a.set_xlabel("chromosome (5-Mb windows, hg38)")
    if k == 1: a.set_ylabel("segmented relative copy number", labelpad=2)
t0 = SM.time.values.astype(float)[i0]; axes[0].set_title(f"One progressor: earliest pre-event biopsy ({GR[i0]}, {t0:.1f} years before the first HGD/IMC endoscopy)", fontsize=6.5, pad=4)
letter(fig, axes[0], "a", dx=-0.045)
# ------------------------------------------------------------------ (b) AUROC by copy-number version
bx = fig.add_axes([0.74, 0.60, 0.24, 0.35]); H = {r["analysis"]: r for r in J("review_round4/r33_merge.json")["R33"]["claim_H_LIMG_vs_LCNV_dAUROC"]}; tp = J("triage/their_pre.json")["section1"]["auroc"]; r36 = J("review_round4/r36.json")["R36"]
vers = [("shipped matrix", "L-CNV", tp["L-CNV"], H["primary: all pre-event samples"]), ("joint package", "L-CNV-joint", r36["joint_package_reference"]["pre"]["discrimination"]["auroc"]["L-CNV"], H["package features"]),
        ("past-only", "L-CNV-past", r36["past"]["pre"]["discrimination"]["auroc"]["L-CNV"], H["past-only re-segmented copy number (R36, extra row)"]), ("per-sample", "L-CNV-ps", r36["persample"]["pre"]["discrimination"]["auroc"]["L-CNV"], H["per-sample re-segmented copy number (R36)"])]
src_b = {}
for i, (lab, key, a, row) in enumerate(vers):
    yy = len(vers) - i; ci_point(bx, a["value"], yy, a["ci95"], MODEL[key], horizontal=True, ms=4); assert abs(row["L-CNV"] - a["value"]) < 1e-9 and row["L-IMG"] == tp["L-IMG"]["value"]
    bx.text(0.36, yy + 0.12, f"Δ {signed(row['d_auroc'])}, bound {mi(row['one_sided_lower'])}", fontsize=5.6, va="bottom", color="#333333" if row["within_-0.05"] else MODEL["L-CNV"]); src_b[lab] = {"L-CNV": a, "IMG_minus_CNV": row["d_auroc"], "ci95": row["ci95"], "one_sided_lower": row["one_sided_lower"]}
li = tp["L-IMG"]; bx.axvspan(li["ci95"][0], li["ci95"][1], color=MODEL["L-IMG"], alpha=0.12, lw=0); bx.axvline(li["value"], color=MODEL["L-IMG"], lw=1.0); bx.text(li["value"], len(vers) + 0.75, f"L-IMG {li['value']:.3f}", color=MODEL["L-IMG"], fontsize=6, ha="center", va="bottom")
bx.set_yticks(range(1, len(vers) + 1)); bx.set_yticklabels([v[0] for v in vers][::-1]); bx.set_xlim(0.35, 1.0); bx.set_ylim(0.4, len(vers) + 1.2); bx.set_xlabel("L-CNV AUROC, fold-stratified [95% CI]")
letter(fig, bx, "b", dx=-0.13)
# ------------------------------------------------------------------ (c) per-biopsy correlation joint vs per-sample
rows = np.where(PRE)[0]; grp = np.where(y == 0, 0, np.where(has_post, 2, 1)); GN = {0: "non-progressor", 1: "progressor,\nno HGD/IMC\nbiopsy stored", 2: "progressor,\nHGD/IMC\nbiopsy stored"}; GC = {0: CLASS["N"], 1: "#C994C7", 2: CLASS["P"]}
src_c = {}; rng = np.random.RandomState(0)
for q, (lab, r_) in enumerate((("raw 5-Mb tiles", r_tiles), ("final feature block (632 features)", r_block))):
    a = fig.add_axes([0.08 + q * 0.33, 0.08, 0.27, 0.36]); src_c[lab] = {}
    for g in (0, 1, 2):
        v = r_[rows][grp[rows] == g]; a.scatter(g + rng.uniform(-0.22, 0.22, len(v)), v, s=4, color=GC[g], alpha=0.5, lw=0, zorder=2)
        a.boxplot([v], positions=[g], widths=0.55, showfliers=False, medianprops=dict(color="black", lw=1), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), zorder=3, manage_ticks=False)
        src_c[lab][GN[g].replace("\n", " ")] = {"n": int(len(v)), "median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))], "min": float(v.min())}
        a.text(g, 1.03, f"n {len(v)}\nmedian {np.median(v):.2f}", ha="center", va="bottom", fontsize=5.6)
    a.set_xticks([0, 1, 2]); a.set_xticklabels([GN[g] for g in (0, 1, 2)], fontsize=5.6); a.set_ylim(min(0, r_[rows].min() - 0.05), 1.18); a.set_title(lab, fontsize=6.5, pad=14)
    if q == 0: a.set_ylabel("Pearson r, joint vs per-sample\n(per pre-event biopsy)"); ca = a
letter(fig, ca, "c", dx=-0.06)
save(fig, "Fig5")
dump("Fig5", {"a_patient": {"class": "progressor", "n_pre_event_biopsies": int(((pat == p) & PRE).sum()), "n_stored_post_event_hgd_imc_biopsies": int(len(post_idx)), "earliest_pre_event_biopsy_grade": str(GR[i0]), "years_before_event": float(t0),
                            "reference_biopsy_grade": str(GR[i1]), "r_joint_vs_persample_raw_tiles": float(r_tiles[i0]), "r_joint_vs_past_raw_tiles": float(rowcorr(TJ[five].values[[i0]], TQ[five].values[[i0]])[0]),
                            "selection_rule": "largest 1 - r (joint vs per-sample raw 5-Mb tiles) of the earliest pre-event biopsy among progressors with a stored HGD/IMC biopsy", "n_candidates": len(cands), "cx_raw": {"joint": float(TJ.cx_raw.values[i0]), "persample": float(TP.cx_raw.values[i0]), "past": float(TQ.cx_raw.values[i0])}},
              "a_tracks_5mb": {lab: np.round(v, 4).tolist() for lab, v, _ in tracks}, "a_chromosome_per_tile": chrom.tolist(), "b_auroc_by_version": src_b, "b_L_IMG": li, "c_correlations": src_c})
