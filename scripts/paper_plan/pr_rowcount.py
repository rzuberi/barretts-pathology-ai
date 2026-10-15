"""Row count, aggregation and label (pre-specified in docs/paper_plan_rowcount.md @ 9ac8165). Frozen release; nothing retrained.
Outputs results/paper_final/rowcount.json and figures F_intro_forest_v2 / F_table_forest_v2 (+ json)."""
import glob, json, os, warnings, numpy as np, pandas as pd
from scipy.stats import rankdata, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; ROW = T + "/feasibility/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_final"); FIG = T + "/results/paper_final/figs"; os.makedirs(AGG, exist_ok=True); os.makedirs(FIG, exist_ok=True); NB = 2000; SEED = 0
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
def pf(p): return None if p is None or (isinstance(p, float) and np.isnan(p)) else ("<0.001" if p < 0.001 else round(float(p), 3))
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID").loc[man.index]; ids = np.array(man.index); pid = man.patient_id.values; folds = man.fold_id_rep01.astype(int).values; yrow = man.y_progressor.astype(int).values; dates = pd.to_datetime(coh.Date).values
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{R}/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(ids).values
ROWS = {f: oof(f) for f in ["cnv_only", "image_only", "early_fusion", "intermediate_fusion", "late_mean", "coattention_fusion", "late_stack_logit"]}; ROWS["cnv_km"] = pd.read_csv(ROW + "/f1_cnv_km_oof.csv", dtype={"sample_id": str}).set_index("sample_id").cnv_km.reindex(ids).values
inner = {k: pd.read_csv(f"{R}/image_only/fold{k}/inner_fold_assignments.csv", dtype=str).set_index("patient_id").inner_fold.astype(int) for k in range(1, 6)}
def nested_logistic(X, yv, fl, pids, Cs=(0.01, 0.1, 1.0, 10.0)):
    p = np.zeros(len(yv))
    for k in np.unique(fl):
        te = fl == k; tr = ~te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xtr = (X[tr] - mu) / sd; inn = inner[int(k)].reindex(pids[tr]).values; best, bestC = -1, 1.0
        for C in Cs:
            pv = np.zeros(tr.sum())
            for j in np.unique(inn): v = inn == j; pv[v] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], yv[tr][~v]).predict_proba(Xtr[v])[:, 1]
            g = pd.DataFrame({"p": pids[tr], "s": pv, "y": yv[tr]}).groupby("p"); a = auc(g.y.max().values, g.s.max().values)
            if a > best: best, bestC = a, C
        p[te] = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, yv[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return p
ROWS["C2_grade_maxsofar"] = nested_logistic(np.column_stack([pd.to_numeric(coh.Label, errors="coerce").fillna(0).values, pd.to_numeric(coh.MaxPathologySoFar, errors="coerce").fillna(0).values]).astype(float), yrow, folds, pid)
ARMS = ["C2_grade_maxsofar", "cnv_only", "cnv_km", "image_only", "early_fusion", "intermediate_fusion", "late_mean", "coattention_fusion", "late_stack_logit"]; FUS = ["early_fusion", "intermediate_fusion", "late_mean", "coattention_fusion", "late_stack_logit"]
pt = pd.read_csv(ROW + "/round3_patient_table.csv", dtype={"patient_id": str}).set_index("patient_id"); pats = pt.index.values; y = pt.y.values.astype(int); yH = pt.y_hgd.values.astype(int); strat2 = pt.stratum2.values; disc = strat2 == "discovery"
nrows = pd.Series(1, index=pid).groupby(level=0).sum().reindex(pats).values; order = pd.DataFrame({"p": pid, "d": dates, "s": ids}).sort_values(["d", "s"]); last_id = order.groupby("p").s.last().reindex(pats).values; last_idx = pd.Series(np.arange(len(ids)), index=ids).reindex(last_id).values
def agg(s, how):
    if how == "last": return np.asarray(s)[last_idx]
    g = pd.Series(s, index=pid).groupby(level=0); return (g.max() if how == "max" else g.mean()).reindex(pats).values
def zf(v):
    v = np.asarray(v, float); o = np.zeros(len(v))
    for f_ in np.unique(folds): te = folds == f_; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
COMBO = {"C2+image": ["image_only"], "C2+cnv_only": ["cnv_only"], "C2+cnv_km": ["cnv_km"], "C2+late_mean": ["late_mean"], "C2+image+cnv_only": ["image_only", "cnv_only"]}
for name, mods in COMBO.items(): ROWS[name] = (zf(ROWS["C2_grade_maxsofar"]) + sum(zf(ROWS[m_]) for m_ in mods)) / (1 + len(mods))
def strat_auc(yy, s, g2):
    num = den = 0.0
    for l in np.unique(g2):
        m = g2 == l; n1 = yy[m].sum(); n0 = m.sum() - n1
        if n1 > 0 and n0 > 0: num += n1 * n0 * auc(yy[m], s[m]); den += n1 * n0
    return num / den
def bidx(yy, strata=None):
    rng = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        s = rng.choice(len(yy), len(yy)) if strata is None else np.concatenate([rng.choice(np.where(strata == l)[0], (strata == l).sum()) for l in np.unique(strata)])
        if (len(set(yy[s])) > 1) and (strata is None or all(len(set(yy[s][strata[s] == l])) > 1 for l in np.unique(strata))): out.append(s)
    return out
def perms(yy, strata=None):
    rng = np.random.RandomState(SEED); out = []
    for _ in range(NB):
        if strata is None: out.append(rng.permutation(len(yy)))
        else:
            ix = np.arange(len(yy))
            for l in np.unique(strata): m = np.where(strata == l)[0]; ix[m] = m[rng.permutation(len(m))]
            out.append(ix)
    return out
RES = {"_spec": "docs/paper_plan_rowcount.md @ 9ac8165"}
# ---------------- 1 rows per patient by label
R1 = {}
for pop, m in [("discovery", disc), ("validation", ~disc), ("all_150", np.ones(len(pats), bool))]:
    R1[pop] = {}
    for ep, yy in [("LGD2plus", y), ("E_HGD", yH)]:
        a_, b_ = nrows[m & (yy == 1)], nrows[m & (yy == 0)]; Bm = bidx(yy[m]); v = [auc(yy[m][s], nrows[m][s]) for s in Bm]
        R1[pop][ep] = {"n": int(m.sum()), "events": int(yy[m].sum()), "rows_progressors_median_iqr": [r3(np.median(a_)), r3(np.percentile(a_, 25)), r3(np.percentile(a_, 75))], "rows_nonprogressors_median_iqr": [r3(np.median(b_)), r3(np.percentile(b_, 25)), r3(np.percentile(b_, 75))], "mannwhitney_p": pf(mannwhitneyu(a_, b_).pvalue), "auroc_rowcount_more_rows_higher": r3(auc(yy[m], nrows[m])), "ci": [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]}
RES["item1_rows_per_patient"] = R1; print("1", flush=True)
# ---------------- 2 aggregation by row-count group
R2 = {"by_arm": {}, "spearman_rowcount_vs_score_discovery_nonprogressors": {}}; few = disc & (nrows <= 2); many = disc & (nrows >= 3)
for a in ARMS:
    R2["by_arm"][a] = {}
    for gname, m in [("all_82", disc), ("rows_1_2", few), ("rows_ge3", many)]: R2["by_arm"][a][gname] = {"n": int(m.sum()), "events": int(y[m].sum()), **{how: r3(auc(y[m], agg(ROWS[a], how)[m])) for how in ["max", "mean", "last"]}}
    mn = disc & (y == 0); R2["spearman_rowcount_vs_score_discovery_nonprogressors"][a] = {how: r3(spearmanr(nrows[mn], agg(ROWS[a], how)[mn]).correlation) for how in ["max", "mean"]}
R2["row_count_groups"] = {"rows_1_2": {"n": int(few.sum()), "events": int(y[few].sum())}, "rows_ge3": {"n": int(many.sum()), "events": int(y[many].sum())}}; RES["item2_aggregation_by_rowcount"] = R2; print("2", flush=True)
# ---------------- 3 mean aggregation differences
PM = {a: agg(ROWS[a], "mean") for a in ROWS}; PX = {a: agg(ROWS[a], "max") for a in ROWS}; R3 = {}
for pop in ["discovery", "stratified"]:
    for ep, yy in [("LGD2plus", y), ("E_HGD", yH)]:
        if pop == "discovery": m = disc; yy2 = yy[m]; g2 = None; f = lambda yv, sv, gv=None: auc(yv, sv); Bs = bidx(yy2); Ps = perms(yy2)
        else: m = np.ones(len(pats), bool); yy2 = yy; g2 = strat2; f = lambda yv, sv, gv: strat_auc(yv, sv, gv); Bs = bidx(yy2, g2); Ps = perms(yy2, g2)
        S = {a: PM[a][m] for a in ROWS}; out = {"n": int(len(yy2)), "events": int(yy2.sum()), "arms_mean_auroc": {a: r3(f(yy2, S[a], g2)) for a in ARMS + list(COMBO)}, "differences": {}}
        def dif(a, b):
            obs = f(yy2, S[a], g2) - f(yy2, S[b], g2); v = [f(yy2[q], S[a][q], g2[q] if g2 is not None else None) - f(yy2[q], S[b][q], g2[q] if g2 is not None else None) for q in Bs]; null = np.array([f(yy2[ix], S[a], g2) - f(yy2[ix], S[b], g2) for ix in Ps]); return {"delta": r3(obs), "ci": [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))], "perm_p": round(float((1 + (null >= obs).sum()) / (NB + 1)), 4)}, null
        for a, b in [("image_only", "cnv_only"), ("image_only", "cnv_km"), ("late_mean", "cnv_only")] + [(c, "C2_grade_maxsofar") for c in COMBO]: out["differences"][f"{a}_minus_{b}"], _ = dif(a, b)
        nulls = {}
        for f_ in FUS: out["differences"][f"{f_}_minus_image_only"], nulls[f_] = dif(f_, "image_only")
        mx = np.max(np.stack([nulls[f_] for f_ in FUS]), 0)
        for f_ in FUS: out["differences"][f"{f_}_minus_image_only"]["perm_p_selection_adjusted_max_over_5_fusion_arms"] = round(float((1 + (mx >= (f(yy2, S[f_], g2) - f(yy2, S["image_only"], g2))).sum()) / (NB + 1)), 4)
        R3[f"{pop}__{ep}"] = out; print("3", pop, ep, flush=True)
RES["item3_mean_aggregation_differences"] = R3
# ---------------- 4 figures v2
def est(pop, ep, a, how, b=None):
    yy = y if ep == "LGD2plus" else yH; P_ = PX if how == "max" else PM
    if pop == "discovery": m = disc; Bs = bidx(yy[m]); pt_ = auc(yy[m], P_[a][m]) - (auc(yy[m], P_[b][m]) if b else 0); v = [auc(yy[m][q], P_[a][m][q]) - (auc(yy[m][q], P_[b][m][q]) if b else 0) for q in Bs]
    elif pop == "pooled": Bs = bidx(yy); pt_ = auc(yy, P_[a]) - (auc(yy, P_[b]) if b else 0); v = [auc(yy[q], P_[a][q]) - (auc(yy[q], P_[b][q]) if b else 0) for q in Bs]
    else: Bs = bidx(yy, strat2); pt_ = strat_auc(yy, P_[a], strat2) - (strat_auc(yy, P_[b], strat2) if b else 0); v = [strat_auc(yy[q], P_[a][q], strat2[q]) - (strat_auc(yy[q], P_[b][q], strat2[q]) if b else 0) for q in Bs]
    return [r3(pt_), r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]
NE = {"discovery": {"LGD2plus": [int(disc.sum()), int(y[disc].sum())], "E_HGD": [int(disc.sum()), int(yH[disc].sum())]}, "stratified": {"LGD2plus": [150, int(y.sum())], "E_HGD": [150, int(yH.sum())]}, "pooled": {"LGD2plus": [150, int(y.sum())], "E_HGD": [150, int(yH.sum())]}}
NAME = {"C2_grade_maxsofar": "Clinical (C2)", "cnv_only": "CNV (release RF)", "cnv_km": "CNV (Killcoyne method)", "image_only": "WSI", "early_fusion": "Early fusion", "intermediate_fusion": "Intermediate fusion", "late_mean": "Late fusion (mean)", "coattention_fusion": "Co-attention (suppl.)", "late_stack_logit": "Late stack (suppl.)"}
def savefig(fig, name, data): fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.pdf", dpi=300, bbox_inches="tight"); plt.close(fig); json.dump(data, open(f"{FIG}/{name}.json", "w"), indent=1, default=str)
FI = {"rows": {}}; fig, axes = plt.subplots(2, 2, figsize=(9, 8.4), gridspec_kw={"width_ratios": [1.3, 1]}); cols = {"image_only": "#1f77b4", "cnv_only": "#d62728", "cnv_km": "#ff7f0e"}
for ri, how in enumerate(["max", "mean"]):
    rows = []; labels = []
    for ep in ["LGD2plus", "E_HGD"]:
        for pop in ["discovery", "stratified", "pooled"]:
            rec = {"population": pop, "endpoint": ep, "n_events": NE[pop][ep], **{a: est(pop, ep, a, how) for a in ["image_only", "cnv_only", "cnv_km"]}, "WSI_minus_cnv_only": est(pop, ep, "image_only", how, "cnv_only"), "WSI_minus_cnv_km": est(pop, ep, "image_only", how, "cnv_km")}; rows.append(rec); labels.append(f"{pop} / {'LGD2+' if ep == 'LGD2plus' else 'E-HGD (post hoc)'}\nn {rec['n_events'][0]}, ev {rec['n_events'][1]}")
    FI["rows"][how] = rows; yy_ = np.arange(len(rows))[::-1]; ax0, ax1 = axes[ri]
    for j, a in enumerate(["image_only", "cnv_only", "cnv_km"]): v = np.array([r[a] for r in rows]); ax0.errorbar(v[:, 0], yy_ + (j - 1) * 0.22, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=4, color=cols[a], label=NAME[a], capsize=2)
    ax0.axvline(0.5, ls=":", c="grey"); ax0.set_yticks(yy_); ax0.set_yticklabels(labels); ax0.set_xlabel("AUROC (95 % CI)"); ax0.set_title(f"WSI vs CNV arms, {how} over rows" + (" (convention)" if how == "max" else " (sensitivity)")); ax0.legend(fontsize=7, loc="lower right"); ax0.set_xlim(0.35, 0.95)
    for j, (k_, c_) in enumerate([("WSI_minus_cnv_only", "#d62728"), ("WSI_minus_cnv_km", "#ff7f0e")]): v = np.array([r[k_] for r in rows]); ax1.errorbar(v[:, 0], yy_ + (j - 0.5) * 0.25, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="s", ms=4, color=c_, label=k_.replace("_minus_", " − ").replace("cnv_only", "CNV RF").replace("cnv_km", "CNV KM"), capsize=2)
    ax1.axvline(0, c="grey"); ax1.axvline(-0.05, ls="--", c="k", label="non-inferiority margin −0.05"); ax1.set_yticks(yy_); ax1.set_yticklabels([]); ax1.set_xlabel("ΔAUROC WSI − CNV (95 % CI)"); ax1.set_title(f"Difference, {how} over rows"); ax1.legend(fontsize=7, loc="lower right"); ax1.set_xlim(-0.25, 0.3)
fig.tight_layout(); savefig(fig, "F_intro_forest_v2", FI); print("fig intro", flush=True)
FT = {"n_events": NE}; fig, axes = plt.subplots(2, 3, figsize=(13, 9)); yy_ = np.arange(len(ARMS))[::-1]
for ri, how in enumerate(["max", "mean"]):
    FT[how] = {"A": {a: est("discovery", "LGD2plus", a, how) for a in ARMS}, "B": {a: est("discovery", "E_HGD", a, how) for a in ARMS}, "C": {a: {pop: est(pop, "LGD2plus", a, how) for pop in ["pooled", "stratified", "discovery"]} for a in ARMS}}
    for ax, key, ttl in [(axes[ri, 0], "A", f"A. Discovery, LGD2+ (n {NE['discovery']['LGD2plus'][0]}, events {NE['discovery']['LGD2plus'][1]}), {how}"), (axes[ri, 1], "B", f"B. Discovery, E-HGD post hoc (events {NE['discovery']['E_HGD'][1]}), {how}")]:
        for i, a in enumerate(ARMS): v = FT[how][key][a]; ax.errorbar(v[0], yy_[i], xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", color="grey" if a in ("coattention_fusion", "late_stack_logit") else "#1f77b4", capsize=2)
        ax.set_yticks(yy_); ax.set_yticklabels([NAME[a] for a in ARMS]); ax.axvline(0.5, ls=":", c="grey"); ax.set_xlabel("AUROC (95 % CI)"); ax.set_title(ttl, fontsize=8); ax.set_xlim(0.3, 1.0)
    ax = axes[ri, 2]
    for j, (pop, c_) in enumerate([("pooled", "#bbbbbb"), ("stratified", "#ff7f0e"), ("discovery", "#1f77b4")]): v = np.array([FT[how]["C"][a][pop] for a in ARMS]); ax.errorbar(v[:, 0], yy_ + (j - 1) * 0.25, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=4, color=c_, label=f"{pop}" + (" (confounded)" if pop == "pooled" else ""), capsize=2)
    ax.set_yticks(yy_); ax.set_yticklabels([]); ax.axvline(0.5, ls=":", c="grey"); ax.set_xlabel("AUROC (95 % CI), LGD2+"); ax.set_title(f"C. Pooled vs stratified vs discovery, {how} over rows", fontsize=8); ax.legend(fontsize=7, loc="lower right"); ax.set_xlim(0.3, 1.0)
fig.tight_layout(); savefig(fig, "F_table_forest_v2", FT); print("fig table", flush=True)
json.dump(RES, open(AGG + "/rowcount.json", "w"), indent=1, default=str); print("ROWCOUNT DONE", flush=True)
