"""Paper figures v3: re-rendering only, from committed result JSONs (results/paper_final/figs/*.json, final_checks.json,
round3_main.json, round3_extra.json) and, for the supplementary UMAP, the saved embeddings. No new analysis. Saves
results/paper_final/figs/<name>_v3.{png,pdf,json} and F_S1_umap_v3."""
import json, os, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore")
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; FIG = T + "/results/paper_final/figs"; RP = T + "/results/paper_plan"; F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
J = lambda p: json.load(open(p)); FI = J(f"{FIG}/F_intro_forest_v2.json"); FT = J(f"{FIG}/F_table_forest_v2.json"); K = J(T + "/results/paper_final/final_checks.json"); R3 = J(RP + "/round3_main.json"); EX = J(RP + "/round3_extra.json"); D6 = J(f"{FIG}/F_D6_false_negatives.json")
NAME = {"C2_grade_maxsofar": "Clinical", "cnv_only": "CNV (release RF)", "cnv_km": "CNV (Killcoyne method)", "image_only": "WSI", "early_fusion": "Early fusion", "intermediate_fusion": "Intermediate fusion", "late_mean": "Late fusion", "coattention_fusion": "Co-attention (suppl.)", "late_stack_logit": "Late stack (suppl.)"}
ARMS = list(NAME); NE = FT["n_events"]
def savefig(fig, name, data): fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.pdf", dpi=300, bbox_inches="tight"); plt.close(fig); json.dump(data, open(f"{FIG}/{name}.json", "w"), indent=1, default=str)
def eb(ax, v, yy, **kw): v = np.asarray(v, float); ax.errorbar(v[:, 0], yy, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=2, ms=4, **kw)
# ------------------------------------------------------------- F_intro v3
fig, axes = plt.subplots(2, 2, figsize=(10, 8.8), gridspec_kw={"width_ratios": [1.3, 1]}); cols = {"image_only": "#1f77b4", "cnv_only": "#d62728", "cnv_km": "#ff7f0e"}
for ri, how in enumerate(["max", "mean"]):
    rows = FI["rows"][how]; yy_ = np.arange(len(rows))[::-1]; labels = [f"{r['population']} / {'LGD2+' if r['endpoint'] == 'LGD2plus' else 'E-HGD (post hoc)'}\nn {r['n_events'][0]}, events {r['n_events'][1]}" for r in rows]; ax0, ax1 = axes[ri]
    for j, a in enumerate(["image_only", "cnv_only", "cnv_km"]): eb(ax0, [r[a] for r in rows], yy_ + (j - 1) * 0.22, fmt="o", color=cols[a], label=NAME[a])
    ax0.axvline(0.5, ls=":", c="grey"); ax0.set_yticks(yy_); ax0.set_yticklabels(labels); ax0.set_xlabel("AUROC (95 % CI)"); ax0.set_xlim(0.25, 1.0); ax0.set_title(f"AUROC, {how} over rows" + (" (convention)" if how == "max" else " (sensitivity)"), fontsize=9)
    for j, (k_, c_, lab) in enumerate([("WSI_minus_cnv_only", "#d62728", "WSI − CNV (release RF)"), ("WSI_minus_cnv_km", "#ff7f0e", "WSI − CNV (Killcoyne method)")]): eb(ax1, [r[k_] for r in rows], yy_ + (j - 0.5) * 0.25, fmt="s", color=c_, label=lab)
    ax1.axvline(0, c="grey"); ax1.axvline(-0.05, ls="--", c="k", label="non-inferiority margin −0.05"); ax1.set_yticks(yy_); ax1.set_yticklabels([]); ax1.set_xlabel("ΔAUROC WSI − CNV (95 % CI)"); ax1.set_xlim(-0.25, 0.32); ax1.set_title(f"Difference, {how} over rows", fontsize=9)
    if ri == 0: ax0.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False); ax1.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=1, frameon=False)
fig.tight_layout(); savefig(fig, "F_intro_forest_v3", FI)
# ------------------------------------------------------------- F_table v3
fig, axes = plt.subplots(2, 3, figsize=(14, 9.5)); yy_ = np.arange(len(ARMS))[::-1]
for ri, how in enumerate(["max", "mean"]):
    for ci_, (key, ttl) in enumerate([("A", f"A. Discovery, LGD2+ (n {NE['discovery']['LGD2plus'][0]}, events {NE['discovery']['LGD2plus'][1]}), {how} over rows"), ("B", f"B. Discovery, E-HGD post hoc (n {NE['discovery']['E_HGD'][0]}, events {NE['discovery']['E_HGD'][1]}), {how} over rows")]):
        ax = axes[ri, ci_]
        for i, a in enumerate(ARMS): v = FT[how][key][a]; ax.errorbar(v[0], yy_[i], xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", color="grey" if "suppl" in NAME[a] else "#1f77b4", capsize=2)
        ax.set_yticks(yy_); ax.set_yticklabels([NAME[a] for a in ARMS]); ax.axvline(0.5, ls=":", c="grey"); ax.set_xlabel("AUROC (95 % CI)"); ax.set_title(ttl, fontsize=8); ax.set_xlim(0.25, 1.0)
    ax = axes[ri, 2]
    for j, (pop, c_, lab) in enumerate([("pooled", "#bbbbbb", f"pooled, confounded (n 150, events {NE['pooled']['LGD2plus'][1]})"), ("stratified", "#ff7f0e", f"stratified (n 150, events {NE['stratified']['LGD2plus'][1]})"), ("discovery", "#1f77b4", f"discovery (n 82, events {NE['discovery']['LGD2plus'][1]})")]): eb(ax, [FT[how]["C"][a][pop] for a in ARMS], yy_ + (j - 1) * 0.25, fmt="o", color=c_, label=lab)
    ax.set_yticks(yy_); ax.set_yticklabels([NAME[a] for a in ARMS]); ax.axvline(0.5, ls=":", c="grey"); ax.set_xlabel("AUROC (95 % CI), LGD2+"); ax.set_xlim(0.25, 1.0); ax.set_title(f"C. Pooled vs stratified vs discovery, LGD2+, {how} over rows", fontsize=8)
    if ri == 0: ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=1, frameon=False)
fig.tight_layout(); savefig(fig, "F_table_forest_v3", FT)
# ------------------------------------------------------------- F_D1 v3
K5 = K["K5"]["LGD2plus"]; arms4 = ["C2_grade_maxsofar", "cnv_only", "image_only", "late_mean"]; nd, ed = NE["discovery"]["LGD2plus"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2, 1]}); w = 0.25
for j, gn in enumerate(["low", "moderate", "high"]):
    vals = np.array([K5[a]["groups"][gn]["rate"] or 0 for a in arms4]); lo = np.array([K5[a]["groups"][gn]["wilson"][0] or 0 for a in arms4]); hi = np.array([K5[a]["groups"][gn]["wilson"][1] or 0 for a in arms4]); xs = np.arange(4) + (j - 1) * w
    axes[0].bar(xs, vals, w, yerr=[vals - lo, hi - vals], capsize=2, label=f"{gn} tertile", color=["#1f77b4", "#ff7f0e", "#d62728"][j])
    for x_, a in zip(xs, arms4): axes[0].text(x_, (K5[a]["groups"][gn]["wilson"][1] or 0) + 0.02, f"{K5[a]['groups'][gn]['events']}/{K5[a]['groups'][gn]['n']}", ha="center", fontsize=6)
axes[0].set_xticks(np.arange(4)); axes[0].set_xticklabels([NAME[a] for a in arms4]); axes[0].set_ylabel("progression rate (Wilson 95 % CI)"); axes[0].set_ylim(0, 1.05); axes[0].set_title(f"Training-fold tertiles, discovery, LGD2+ (n {nd}, events {ed})", fontsize=8); axes[0].legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
ors = [K5[a]["OR_high_vs_low_haldane"] for a in arms4]; eb(axes[1], ors, np.arange(4)[::-1], fmt="o", color="k"); axes[1].set_xscale("log"); axes[1].axvline(1, ls=":", c="grey"); axes[1].set_yticks(np.arange(4)[::-1]); axes[1].set_yticklabels([NAME[a] for a in arms4]); axes[1].set_xlabel("OR high vs low tertile (Haldane, 95 % CI)"); axes[1].set_title(f"Discovery, LGD2+ (n {nd}, events {ed})", fontsize=8)
for i, a in enumerate(arms4): axes[1].text(ors[i][2] * 1.15, 3 - i, f"trend p {K5[a]['cochran_armitage_z_p'][1]}", va="center", fontsize=6)
fig.tight_layout(); savefig(fig, "F_D1_risk_groups_v3", {"tertiles": K5, "n_events": [nd, ed]})
# ------------------------------------------------------------- F_D2 v3 (A deltas, B stratum probes)
K3 = K["K3"]; pr = R3["R1"]["probes_stratum_from_inputs_nonprogressors"]; fams = ["early_fusion", "intermediate_fusion", "coattention_fusion"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.4, 1]}); yy_ = np.arange(3)[::-1]
for j, (ref, c_, lab) in enumerate([("image_only", "#1f77b4", "fused − WSI representation"), ("cnv_only", "#d62728", "fused − CNV representation")]): eb(axes[0], [K3[f][f"probe_delta_vs_{ref}_discovery"] for f in fams], yy_ + (j - 0.5) * 0.25, fmt="o", color=c_, label=lab)
axes[0].axvline(0, c="grey"); axes[0].set_yticks(yy_); axes[0].set_yticklabels([NAME[f] for f in fams]); axes[0].set_xlabel("Δ linear-probe AUROC (95 % CI)"); axes[0].set_title(f"A. Probe gain of fused over unimodal embeddings, discovery held-out (n {K3['early_fusion']['n']}, events {K3['early_fusion']['events']})", fontsize=8); axes[0].legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
pb = {"WSI embedding": pr["image_patient_embedding_256d"], "CNV features (PCA-20)": pr["cnv_features_pca20"], "CNV QC (no read count, post hoc)": EX["cnv_qc_without_reads"]}
eb(axes[1], [[v["auroc"]] + v["ci"] for v in pb.values()], np.arange(3)[::-1], fmt="o", color="k"); axes[1].axvline(0.5, ls=":", c="grey"); axes[1].set_yticks(np.arange(3)[::-1]); axes[1].set_yticklabels(list(pb)); axes[1].set_xlabel("AUROC for predicting stratum (95 % CI)"); axes[1].set_xlim(0.4, 1.0); axes[1].set_title(f"B. Stratum predicted from inputs, non-progressors only (n {pr['image_patient_embedding_256d']['n_nonprogressors']}; discovery {pr['image_patient_embedding_256d']['discovery']}, validation {pr['image_patient_embedding_256d']['validation']})", fontsize=8)
fig.tight_layout(); savefig(fig, "F_D2_latent_v3", {"A": {f: {r: K3[f][f"probe_delta_vs_{r}_discovery"] for r in ["image_only", "cnv_only"]} for f in fams}, "B": pb})
# ------------------------------------------------------------- F_S1 UMAP (supplementary), fold 1 per model
import umap; from sklearn.preprocessing import StandardScaler
Lat = T + "/feasibility/paper_plan/latent"; idx = pd.read_csv(Lat + "/index.csv", dtype={"sample_id": str, "patient_id": str}); pid = idx.patient_id.values; fold = idx.fold.values; y = idx.y.values
pats = np.array(sorted(set(pid))); pfold = pd.Series(fold, index=pid).groupby(level=0).first().reindex(pats).values; py = pd.Series(y, index=pid).groupby(level=0).max().reindex(pats).values
st = pd.read_csv(T + "/feasibility/paper_plan/f2_strata.csv", dtype=str).set_index("patient_id").stratum.reindex(pats).str.startswith("discovery").values
fig, axes = plt.subplots(2, 2, figsize=(10, 9)); S1 = {}
for r_, fam in enumerate(["image_only", "intermediate_fusion"]):
    Pm = pd.DataFrame(np.load(f"{Lat}/emb_{fam}_fold1.npy"), index=pid).groupby(level=0).mean().reindex(pats).values; tr = pfold != 1; te = ~tr; sc = StandardScaler().fit(Pm[tr]); Z = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit(sc.transform(Pm[tr])).transform(sc.transform(Pm[te]))
    S1[fam] = {"held_out_n": int(te.sum()), "discovery": int((te & st).sum()), "events_discovery": int(py[te & st].sum()), "events_all": int(py[te].sum())}
    ax = axes[r_, 0]; sv = st[te]; ax.scatter(Z[~sv, 0], Z[~sv, 1], s=22, c="#999999", label=f"validation (n {int((~sv).sum())})"); ax.scatter(Z[sv, 0], Z[sv, 1], s=22, c="#ff7f0e", label=f"discovery (n {int(sv.sum())})"); ax.set_title(f"{NAME[fam]} embedding, UMAP, held-out fold 1 (n {int(te.sum())}), by stratum", fontsize=8); ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False); ax.set_xticks([]); ax.set_yticks([])
    ax = axes[r_, 1]; m = st[te]; yd = py[te]; ax.scatter(Z[m & (yd == 0), 0], Z[m & (yd == 0), 1], s=22, c="#1f77b4", label=f"non-progressor (n {int((m & (yd == 0)).sum())})"); ax.scatter(Z[m & (yd == 1), 0], Z[m & (yd == 1), 1], s=22, c="#d62728", label=f"progressor (n {int((m & (yd == 1)).sum())})"); ax.set_title(f"{NAME[fam]} embedding, discovery patients only (n {int(m.sum())}, events {int(yd[m].sum())}), by LGD2+ label", fontsize=8); ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); savefig(fig, "F_S1_umap_v3", {"fold": 1, "projection": "UMAP n_neighbors 15, min_dist 0.1, seed 0, fitted on fold-1 training patients", "panels": S1})
# ------------------------------------------------------------- F_D5 v3
K1 = K["K1"]["per_stratum"]; fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
for ax, a in zip(axes, ["late_mean", "image_only", "cnv_only", "C2_grade_maxsofar"]):
    for j, l in enumerate(["discovery", "validation"]):
        for i, grp in enumerate(["FP", "TN"]):
            d_ = K1[a][l]; rate = d_[f"rate_{grp}"] or 0; wl = d_[f"wilson_{grp}"]; x_ = j * 2.4 + i; ax.bar(x_, rate, 0.8, yerr=[[rate - (wl[0] or 0)], [(wl[1] or 0) - rate]], capsize=2, color="#d62728" if grp == "FP" else "#1f77b4"); ax.text(x_, (wl[1] or 0) + 0.03, f"{d_['laterHGD_' + grp]}/{d_[grp]}", ha="center", fontsize=6)
    ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(["FP\ndiscovery", "TN\ndiscovery", "FP\nvalidation", "TN\nvalidation"], fontsize=7); ax.set_ylim(0, 1.0); ax.set_title(f"{NAME[a]}\nnon-progressors: discovery n {K1[a]['discovery']['FP'] + K1[a]['discovery']['TN']}, validation n {K1[a]['validation']['FP'] + K1[a]['validation']['TN']}\nFisher p discovery {K1[a]['discovery']['fisher_p']}, validation {K1[a]['validation']['fisher_p']}", fontsize=7)
axes[0].set_ylabel("later HGD+ rate (Wilson 95 % CI)"); fig.tight_layout(); savefig(fig, "F_D5_false_positives_v3", K1)
# ------------------------------------------------------------- F_D6 v3 (rows per patient instead of max grade)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); sp = pd.read_csv(T + "/feasibility/paper_plan/followup_patient_scores.csv", dtype={"patient_id": str}).set_index("patient_id"); pt = pd.read_csv(T + "/feasibility/paper_plan/round3_patient_table.csv", dtype={"patient_id": str}).set_index("patient_id"); PT = pd.read_csv(T + "/feasibility/closeout/swg_patient_table.csv", dtype=str).set_index("patient_id")
pats2 = pt.index.values; y2 = pt.y.values.astype(int); disc2 = (pt.stratum2 == "discovery").values; nrows = man.groupby("patient_id").size().reindex(pats2).values; pred = sp.pred_late_mean.reindex(pats2).values.astype(int); pos = (y2 == 1) & disc2; fn = pos & (pred == 0); tp = pos & (pred == 1)
cxv = pd.to_numeric(PT.cx_max.reindex(pats2), errors="coerce").values; fte = pd.to_numeric(PT.days_first_to_event.reindex(pats2), errors="coerce").values; kept = pt.kept_tiles.values
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID").loc[man.sample_id]; nl = pd.Series(pd.to_numeric(coh.NextBiopsyLabel, errors="coerce").values, index=man.patient_id.values); ypos = pd.Series(man.y_progressor.astype(int).values, index=man.patient_id.values); et = nl[ypos == 1].groupby(level=0).max().reindex(pats2).map({2: "second LGD", 3: "HGD", 4: "IMC/cancer", 5: "IMC/cancer"})
fig, axes = plt.subplots(1, 5, figsize=(14, 3.8)); D6v3 = {"n_FN": int(fn.sum()), "n_TP": int(tp.sum()), "boxes": {}}
for ax, (c, v, lab) in zip(axes[:4], [("kept_tiles", kept, "tissue tiles kept (0.44 µm/px)"), ("cx", cxv, "CNV complexity cx (patient max)"), ("first_to_event", fte, "days first row → endpoint biopsy"), ("rows_per_patient", nrows.astype(float), "pre-event rows per patient")]):
    a_, b_ = v[fn][~np.isnan(v[fn])], v[tp][~np.isnan(v[tp])]; p_ = mannwhitneyu(a_, b_).pvalue; ax.boxplot([a_, b_], labels=[f"FN (n {len(a_)})", f"TP (n {len(b_)})"], showfliers=True); ax.set_title(f"{lab}\nMann-Whitney p {'<0.001' if p_ < 0.001 else round(float(p_), 3)}", fontsize=7); D6v3["boxes"][c] = {"FN": [float(x) for x in a_], "TP": [float(x) for x in b_], "mannwhitney_p": float(p_)}
t = pd.crosstab(et.values[fn | tp], np.where(fn[fn | tp], "FN", "TP")).reindex(["second LGD", "HGD", "IMC/cancer"]).fillna(0).astype(int); D6v3["endpoint_type"] = {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in t.to_dict().items()}
xs = np.arange(3); axes[4].bar(xs - 0.2, t["FN"].values, 0.4, color="#d62728", label=f"FN (n {int(fn.sum())})"); axes[4].bar(xs + 0.2, t["TP"].values, 0.4, color="#1f77b4", label=f"TP (n {int(tp.sum())})"); axes[4].set_xticks(xs); axes[4].set_xticklabels(t.index, fontsize=7); axes[4].set_ylabel("patients"); axes[4].set_title("endpoint type", fontsize=7); axes[4].legend(fontsize=7, frameon=False)
fig.suptitle(f"Late fusion false negatives vs true positives, discovery progressors only (n {int(pos.sum())}: FN {int(fn.sum())}, TP {int(tp.sum())}), LGD2+", fontsize=8); fig.tight_layout(); savefig(fig, "F_D6_false_negatives_v3", D6v3); print("V3 DONE", flush=True)
