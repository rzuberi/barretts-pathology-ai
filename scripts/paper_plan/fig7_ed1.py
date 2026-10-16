"""Extended Data Figure 1 (manuscript v7), tissue amount, drawn on the cluster. (a) tissue tiles per slide by class; (b) repeat-mean L-IMG score against
log10 tissue tiles with within-class Spearman correlations (committed values; recomputed here); (c) fold-stratified AUROCs of the three models by tissue
stratum and within tissue tertiles, with biopsy and patient counts; (d) tissue tiles and biopsy year as raw scores, and L-IMG before and after adjustment."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig7_style import *
from tb_common import *
O, I, FO = load("their"); rows = np.where(popmask("pre"))[0]; yr = y[rows]; img = O["L-IMG"].mean(1)[rows]
TILES = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids).tissue_tiles.values.astype(float)[rows]; lt = np.log10(TILES)
r5 = J("review_round1/r5.json")["R5"]; r32 = J("review_round4/r32a.json")["R32a"]; tp = J("triage/their_pre.json")["section1"]["auroc"]; npat_all = int(len(set(pat[rows])))
fig = plt.figure(figsize=(183 * MM, 120 * MM)); src = {"n_samples": int(len(rows)), "n_patients": npat_all}
ax = fig.add_axes([0.06, 0.60, 0.40, 0.36]); bins = np.linspace(np.floor(lt.min() * 2) / 2, np.ceil(lt.max() * 2) / 2, 25); src["tiles_hist"] = {"log10_bins": bins.tolist()}
for cls, v in (("P", 1), ("N", 0)):
    h, _ = np.histogram(lt[yr == v], bins); src["tiles_hist"][CLASS_NAME[cls]] = h.tolist(); ax.hist(lt[yr == v], bins, histtype="stepfilled", color=CLASS[cls], alpha=0.2, lw=0); ax.hist(lt[yr == v], bins, histtype="step", color=CLASS[cls], lw=1.1, label=f"{CLASS_NAME[cls]} (n {int((yr == v).sum())}, median {int(np.median(TILES[yr == v]))} tiles)")
ax.axvline(np.log10(256), color="black", lw=0.7, ls="--"); ax.text(np.log10(256) + 0.03, ax.get_ylim()[1] * 0.95, "256 tiles: below this the\n256 sampled tiles overlap", fontsize=5.6, va="top")
ax.set_xlabel("tissue tiles per slide (log10)"); ax.set_ylabel("pre-event biopsies"); ax.legend(loc="upper right", fontsize=5.6, bbox_to_anchor=(1.0, 0.75)); letter(fig, ax, "a", dx=-0.05)
bx = fig.add_axes([0.57, 0.60, 0.40, 0.31]); src["spearman_log_tiles_vs_LIMG"] = {}
for cls, v in (("N", 0), ("P", 1)):
    m = yr == v; ref0 = r5[f"c_spearman_LIMG_log_tiles_{'progressor' if v else 'nonprogressor'}"]; rho = float(spearmanr(lt[m], img[m]).correlation); src["spearman_log_tiles_vs_LIMG"][CLASS_NAME[cls]] = {"computed": round(rho, 3), "committed": ref0}
    bx.scatter(lt[m], img[m], s=6, color=CLASS[cls], alpha=0.6, lw=0, label=f"{CLASS_NAME[cls]}: Spearman ρ {mi(ref0['value'])} [{mi(ref0['ci95'][0])}, {mi(ref0['ci95'][1])}]")
bx.axvline(np.log10(256), color="black", lw=0.7, ls="--"); bx.set_xlabel("tissue tiles per slide (log10)"); bx.set_ylabel("L-IMG score (repeat mean)"); bx.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=5.8); letter(fig, bx, "b", dx=-0.06, dy=0.06)
cx = fig.add_axes([0.06, 0.10, 0.52, 0.38]); S = r32["strata"]
groups = [(f"all pre-event\n({src['n_samples']} biopsies,\n{npat_all} patients)", {m: tp[m] for m in ("L-CNV", "L-IMG", "L-LATE")}), (f"< 256 tiles\n({S['lt256']['n_samples']} biopsies,\n{S['lt256']['n_patients']} patients)", S["lt256"]["claim_H"]["auroc"]),
          (f"≥ 256 tiles\n({S['ge256']['n_samples']} biopsies,\n{S['ge256']['n_patients']} patients)", S["ge256"]["claim_H"]["auroc"]), (f"tissue-matched pairs\n(same tertile; {src['n_samples']} biopsies,\n{npat_all} patients)", r32["tissue_matched_pairs_claim_H"]["auroc"])]
src["auroc_by_stratum"] = {}
for i, (g, d) in enumerate(groups):
    src["auroc_by_stratum"][g.replace("\n", " ")] = d
    for k, m in enumerate(("L-CNV", "L-IMG", "L-LATE")): ci_point(cx, i + (k - 1) * 0.22, d[m]["value"], d[m]["ci95"], MODEL[m], ms=3.5, label=MODEL_NAME[m] if i == 0 else None)
for i, d in ((1, S["lt256"]["claim_H"]["IMG_minus_CNV"]), (2, S["ge256"]["claim_H"]["IMG_minus_CNV"])): cx.text(i, 0.62, f"L-IMG − L-CNV\nΔ {signed(d['delta'])}\nbound {mi(d['one_sided_95_lower'])}", ha="center", fontsize=5.4, color="#333333")
cx.set_xticks(range(len(groups))); cx.set_xticklabels([g for g, _ in groups], fontsize=5.6); cx.set_ylabel("AUROC, fold-stratified [95% CI]"); cx.set_ylim(0.6, 1.0); cx.legend(loc="upper left", ncol=3, fontsize=5.6); letter(fig, cx, "c", dx=-0.05)
dx_ = fig.add_axes([0.70, 0.10, 0.27, 0.38]); items = [("tissue tiles alone", r5["a_auroc_tissue_tiles"], GREY), ("biopsy year alone", r5["a_auroc_year"], GREY), ("L-IMG, unadjusted", {"value": r5["d_LIMG_residualised"]["unadjusted"], "ci95": tp["L-IMG"]["ci95"]}, MODEL["L-IMG"]),
                                                     ("L-IMG, tissue- and\nyear-adjusted", {"value": r5["d_LIMG_residualised"]["value"], "ci95": r5["d_LIMG_residualised"]["ci95"]}, MODEL["L-IMG"])]
for i, (lab, d, c_) in enumerate(items): ci_point(dx_, d["value"], len(items) - 1 - i, d["ci95"], c_, horizontal=True, ms=4)
dx_.axvline(0.5, color=LIGHT, lw=0.6, ls=":"); dx_.set_yticks(range(len(items))); dx_.set_yticklabels([it[0] for it in items][::-1], fontsize=6); dx_.set_xlim(0.4, 1.0); dx_.set_xlabel("AUROC, fold-stratified [95% CI]"); src["raw_scores_and_adjusted"] = {lab: d for lab, d, _ in items}
letter(fig, dx_, "d", dx=-0.16)
save(fig, "ED1"); dump("ED1", src)
