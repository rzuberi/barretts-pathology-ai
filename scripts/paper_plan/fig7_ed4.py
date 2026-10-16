"""Extended Data Figure 4 (manuscript v7), copy-number versions, drawn on the cluster from the re-segmentation outputs (killcoyne_mm/review_round4/
reseg/tiles_*_analysed.csv, feature blocks cnv_pkg_C.csv and r36_cnv_pkgps_C.csv, kr_samples.csv for the number of samples segmented together).
(a) one progressor's earliest pre-event biopsy: joint and per-sample 5-Mb profiles overlaid, then the per-sample - joint and past-only - joint difference
tracks (the biopsy with the largest joint-vs-per-sample difference among progressors with a stored HGD/IMC biopsy; anonymous). (b) L-CNV AUROC by version
against L-IMG. (c) paired differences between versions with Holm-adjusted p (committed). (d) per-biopsy correlation of joint vs per-sample final features
by class and by tertile of the number of samples segmented together (pre-event biopsies). No identifiers."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig7_style import *
from tb_common import *
RS = M + "/review_round4/reseg"; rd = lambda f: pd.read_csv(f, dtype={"Sample": str}).set_index("Sample").reindex(ids)
TJ, TP, TQ = rd(RS + "/tiles_joint_stored_analysed.csv"), rd(RS + "/tiles_persample_analysed.csv"), rd(RS + "/tiles_past_analysed.csv")
tc = [c for c in TJ.columns if ":" in c]; assert len(tc) == 587 + 44; five = tc[:-44]; chrom = np.array([c.split(":")[0] for c in five])
XJ = pd.read_csv(M + "/cnv_pkg_C.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(ids); XP = pd.read_csv(M + "/review_round4/r36_cnv_pkgps_C.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(ids)
assert list(XJ.columns) == list(XP.columns) and XJ.notna().all().all() and XP.notna().all().all()
def rowcorr(A, B): A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True); return (A * B).sum(1) / np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
r_tiles = rowcorr(TJ[five].values, TP[five].values); r_block = rowcorr(XJ.values, XP.values)
MBF = SM.mbf.values.astype(float); GR = SM.Pathology.values; post = (~PRE) & np.isin(GR, ["HGD", "IMC"]); has_post = np.isin(pat, np.unique(pat[post])); rows = np.where(PRE)[0]
r40 = J("review_round6/r40.json")["R40"]; SI = r40["sampling_intensity"]
KR = pd.read_csv(T + "/feasibility/paper_plan/killcoyne/kr_samples.csv", dtype=str); NJ = KR.groupby("Patient").size().reindex(pat).values.astype(float); assert np.isfinite(NJ).all()
e1, e2 = SI["n_segmented_together_joint"]["tertile_edges_pre_event"]; tert = np.where(NJ <= e1, 0, np.where(NJ <= e2, 1, 2))
# (a) patient
cands = []
for p in np.unique(pat[(y == 1) & has_post & PRE]):
    pre_idx = np.where((pat == p) & PRE)[0]; i0 = pre_idx[np.argmax(MBF[pre_idx])]; cands.append((1 - r_tiles[i0], p, i0))
cands.sort(reverse=True); diff, p, i0 = cands[0]; t0 = SM.time.values.astype(float)[i0]
vj, vp, vq = TJ[five].values[i0], TP[five].values[i0], TQ[five].values[i0]
fig = plt.figure(figsize=(183 * MM, 150 * MM)); x = np.arange(len(five)); bounds = np.where(chrom[1:] != chrom[:-1])[0] + 0.5; centres = [np.mean(x[chrom == c]) for c in dict.fromkeys(chrom)]
tracks = [("profiles: joint (brown) and per-sample (orange) segmentation", None), ("per-sample − joint", vp - vj), ("past-only − joint", vq - vj)]; axes = []
for k, (lab, v) in enumerate(tracks):
    a = fig.add_axes([0.06, 0.60 + (2 - k) * 0.12, 0.60, 0.11]); axes.append(a)
    for b in bounds: a.axvline(b, color="#E6E6E6", lw=0.4, zorder=0)
    if v is None: a.axhline(1.0, color=LIGHT, lw=0.5, ls=":"); a.step(x, vj, where="mid", color=MODEL["L-CNV-joint"], lw=0.9); a.step(x, vp, where="mid", color=MODEL["L-CNV-ps"], lw=0.7, alpha=0.9); a.set_ylabel("relative\ncopy number", fontsize=6)
    else: a.axhline(0.0, color=LIGHT, lw=0.5, ls=":"); a.step(x, v, where="mid", color=MODEL["L-CNV-ps"] if "per-sample" in lab else MODEL["L-CNV-past"], lw=0.8); a.set_ylabel("difference", fontsize=6); a.set_ylim(-0.6, 0.6)
    a.text(0.005, 0.06 if v is None else 0.95, lab, transform=a.transAxes, fontsize=6, va="bottom" if v is None else "top"); a.set_xlim(-1, len(five)); a.set_xticks([]); a.spines["bottom"].set_visible(k == 2)
    if k == 2: a.set_xticks(centres); a.set_xticklabels([c if (int(c) <= 16 or int(c) % 2 == 0) else "" for c in dict.fromkeys(chrom)], fontsize=5.0); a.set_xlabel("chromosome (5-Mb windows, hg38)")
axes[0].set_title(f"One progressor, earliest pre-event biopsy ({GR[i0]}, {t0:.1f} y before the first HGD/IMC endoscopy); joint vs per-sample r = {r_tiles[i0]:.2f}", fontsize=6, pad=10)
letter(fig, axes[0], "a", dx=-0.045)
# (b) AUROC by version
bx = fig.add_axes([0.76, 0.70, 0.22, 0.22]); A = r40["auroc_L_CNV"]; tp = J("triage/their_pre.json")["section1"]["auroc"]; H = {r["analysis"]: r for r in J("review_round4/r33_merge.json")["R33"]["claim_H_LIMG_vs_LCNV_dAUROC"]}
vers = [("published matrix", "shipped_matrix", "L-CNV", H["primary: all pre-event samples"]), ("joint package", "joint_package", "L-CNV-joint", H["package features"]), ("past-only", "past_only", "L-CNV-past", H["past-only re-segmented copy number (R36, extra row)"]), ("per-sample", "per_sample", "L-CNV-ps", H["per-sample re-segmented copy number (R36)"])]
for i, (lab, key, ck, row) in enumerate(vers):
    yy = len(vers) - i; ci_point(bx, A[key]["value"], yy, A[key]["ci95"], MODEL[ck], horizontal=True, ms=4); bx.text(0.36, yy + 0.12, f"L-IMG − L-CNV {signed(row['d_auroc'])}, bound {mi(row['one_sided_lower'])}", fontsize=5.2, va="bottom", color="#333333" if row["within_-0.05"] else MODEL["L-CNV"])
li = tp["L-IMG"]; bx.axvspan(li["ci95"][0], li["ci95"][1], color=MODEL["L-IMG"], alpha=0.12, lw=0); bx.axvline(li["value"], color=MODEL["L-IMG"], lw=1.0); bx.text(li["value"], len(vers) + 0.75, f"L-IMG {li['value']:.3f}", color=MODEL["L-IMG"], fontsize=6, ha="center", va="bottom")
bx.set_yticks(range(1, len(vers) + 1)); bx.set_yticklabels([v[0] for v in vers][::-1], fontsize=6); bx.set_xlim(0.35, 1.0); bx.set_ylim(0.4, len(vers) + 1.2); bx.set_xlabel("L-CNV AUROC, fold-stratified [95% CI]", fontsize=6); letter(fig, bx, "b", dx=-0.09, dy=0.03)
# (c) pairwise differences
cx = fig.add_axes([0.80, 0.47, 0.18, 0.16]); PW = r40["pairwise"]; NM = {"shipped_matrix": "published", "joint_package": "joint pkg", "past_only": "past-only", "per_sample": "per-sample"}; keys = list(PW)
for i, k in enumerate(keys):
    d = PW[k]; yy = len(keys) - 1 - i; a_, b_ = k.split("_minus_"); ci_point(cx, d["delta"], yy, d["ci95"], "#333333", horizontal=True, ms=3)
cx.axvline(0, color=LIGHT, lw=0.6); cx.set_yticks(range(len(keys))); cx.set_yticklabels([f"{NM[k.split('_minus_')[0]]} − {NM[k.split('_minus_')[1]]}\n(p {PW[k]['p_two_sided']:.2f}, Holm {PW[k]['p_holm']:.2f})" for k in keys][::-1], fontsize=4.8); cx.set_xlim(-0.15, 0.16); cx.set_xlabel("Δ AUROC, paired [95% CI]", fontsize=6); letter(fig, cx, "c", dx=-0.13, dy=0.01)
# (d) correlation by class and tertile
dx_ = fig.add_axes([0.07, 0.08, 0.90, 0.34]); rng = np.random.RandomState(0); groups = [("all", np.ones(n, bool))] + [(f"tertile {t + 1}", tert == t) for t in range(3)]; src_d = {}
TR = SI["per_sample_r_joint_vs_per_sample_final_features"]["by_tertile"]; CD = SI["class_difference_median_r_progressor_minus_non_progressor"]
for gi, (glab, gmask) in enumerate(groups):
    src_d[glab] = {}
    for ci_, (cls, v) in enumerate((("N", 0), ("P", 1))):
        m = PRE & gmask & (y == v); vals = r_block[m]; xpos = gi * 2.6 + (ci_ - 0.5) * 0.9
        dx_.scatter(xpos + rng.uniform(-0.3, 0.3, len(vals)), vals, s=4, color=CLASS[cls], alpha=0.5, lw=0, zorder=2)
        dx_.boxplot([vals], positions=[xpos], widths=0.7, showfliers=False, medianprops=dict(color="black", lw=1), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), zorder=3, manage_ticks=False)
        dx_.text(xpos, 1.03, f"n {len(vals)}\nmedian {np.median(vals):.2f}", ha="center", va="bottom", fontsize=5.2); src_d[glab][CLASS_NAME[cls]] = {"n": int(len(vals)), "median": float(np.median(vals)), "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))]}
    key = "all_pre_event" if glab == "all" else glab.replace(" ", "_"); cd = CD[key]; src_d[glab]["class_difference_median_committed"] = cd
    dx_.text(gi * 2.6, -0.02, f"P − NP median {signed(cd['delta'], 2)}\n[{mi(cd['ci95'][0])}, {mi(cd['ci95'][1])}]", ha="center", va="top", fontsize=5.2, color="#333333")
    if glab != "all": rngs = TR[key]["n_segmented_range"]; src_d[glab]["n_segmented_together_range"] = rngs
tl = ["all pre-event biopsies"] + [f"tertile {t + 1}: {TR[f'tertile_{t + 1}']['n_segmented_range'][0]}–{TR[f'tertile_{t + 1}']['n_segmented_range'][1]} samples\nsegmented together" for t in range(3)]
dx_.set_xticks([gi * 2.6 for gi in range(4)]); dx_.set_xticklabels(tl, fontsize=5.6); dx_.tick_params(axis="x", pad=18); dx_.set_ylim(-0.12, 1.22); dx_.set_ylabel("Pearson r, joint vs per-sample final features\n(per pre-event biopsy)")
dx_.scatter([], [], color=CLASS["N"], s=12, label="non-progressor"); dx_.scatter([], [], color=CLASS["P"], s=12, label="progressor"); dx_.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.6); letter(fig, dx_, "d", dx=-0.05)
save(fig, "ED4")
dump("ED4", {"a_patient": {"class": "progressor", "n_pre_event_biopsies": int(((pat == p) & PRE).sum()), "earliest_pre_event_biopsy_grade": str(GR[i0]), "years_before_event": float(t0), "r_joint_vs_persample_raw_tiles": float(r_tiles[i0]), "n_candidates": len(cands),
                           "selection_rule": "largest 1 - r (joint vs per-sample raw 5-Mb windows) of the earliest pre-event biopsy among progressors with a stored HGD/IMC biopsy"},
             "a_tracks_5mb": {"joint": np.round(vj, 4).tolist(), "per_sample": np.round(vp, 4).tolist(), "past_only": np.round(vq, 4).tolist()}, "a_chromosome_per_window": chrom.tolist(), "b_auroc_by_version": A, "b_L_IMG": li, "c_pairwise": PW, "d_correlation_by_class_and_tertile": src_d, "tertile_edges": [e1, e2]})
