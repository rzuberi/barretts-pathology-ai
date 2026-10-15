"""Paper final checks K1-K5 and figures F-intro, F-table, F-D1..F-D6 (pre-specified in docs/paper_final_inputs.md @ e6e00c0).
Frozen release only; nothing retrained. Outputs results/paper_final/final_checks.json, figures results/paper_final/figs/<name>.{png,pdf,json}."""
import glob, json, os, warnings, numpy as np, pandas as pd, h5py
from scipy.stats import rankdata, spearmanr, mannwhitneyu, fisher_exact, norm
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import umap
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; ROW = T + "/feasibility/paper_plan"; RP = T + "/results/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_final"); FIG = T + "/results/paper_final/figs"; os.makedirs(AGG, exist_ok=True); os.makedirs(FIG, exist_ok=True)
NB = 2000; SEED = 0; plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
def pf(p): return None if p is None or (isinstance(p, float) and np.isnan(p)) else ("<0.001" if p < 0.001 else round(float(p), 3))
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [r3(c - h), r3(c + h)]
def bidx(y):
    rng = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        s = rng.choice(len(y), len(y))
        if len(set(y[s])) > 1: out.append(s)
    return out
def cis(y, B, a, b=None): v = [auc(y[s], a[s]) - (auc(y[s], b[s]) if b is not None else 0) for s in B]; return [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]
def savefig(fig, name, data):
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.pdf", dpi=300, bbox_inches="tight"); plt.close(fig); json.dump(data, open(f"{FIG}/{name}.json", "w"), indent=1, default=str)
RES = {"_spec": "docs/paper_final_inputs.md @ e6e00c0", "figures": {}}
# ------------------------------------------------------------- data
pt = pd.read_csv(ROW + "/round3_patient_table.csv", dtype={"patient_id": str}).set_index("patient_id"); pats = pt.index.values; y = pt.y.values.astype(int); yH = pt.y_hgd.values.astype(int); disc = (pt.stratum2 == "discovery").values; strat2 = pt.stratum2.values; kept = pt.kept_tiles.values
ARMS = ["C2_grade_maxsofar", "cnv_only", "cnv_km", "image_only", "early_fusion", "intermediate_fusion", "late_mean", "coattention_fusion", "late_stack_logit"]; MAIN = ["C2_grade_maxsofar", "cnv_only", "cnv_km", "image_only", "early_fusion", "intermediate_fusion", "late_mean"]
NAME = {"C2_grade_maxsofar": "Clinical (C2)", "cnv_only": "CNV (release RF)", "cnv_km": "CNV (Killcoyne method)", "image_only": "WSI", "early_fusion": "Early fusion", "intermediate_fusion": "Intermediate fusion", "late_mean": "Late fusion (mean)", "coattention_fusion": "Co-attention (suppl.)", "late_stack_logit": "Late stack (suppl.)"}
P = {a: pt[a].values for a in ARMS}
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID").loc[man.index]; ids = list(man.index); pid = man.patient_id.values; folds = man.fold_id_rep01.astype(int).values
man["y"] = man.y_progressor.astype(int); man["next_label"] = pd.to_numeric(coh.NextBiopsyLabel, errors="coerce"); man["maxsofar"] = pd.to_numeric(coh.MaxPathologySoFar, errors="coerce").fillna(0); pfold = pd.Series(folds, index=pid).groupby(level=0).first().reindex(pats).values
sp = pd.read_csv(ROW + "/followup_patient_scores.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(pats); LT = pd.read_csv(ROW + "/later_disease_patient.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(pats); PT = pd.read_csv(T + "/feasibility/closeout/swg_patient_table.csv", dtype=str).set_index("patient_id").reindex(pats)
R3 = json.load(open(RP + "/round3_main.json")); DM = json.load(open(RP + "/discovery_main.json")); L6 = json.load(open(RP + "/latent_item6.json"))
neg = y == 0; laterH = ((LT.db_max_grade_after >= 3) | (LT.release_excluded_max_grade >= 3) | (LT.slidematch_max_grade_after >= 3) | (LT.hgd_table_entries_after > 0)).values; laterL = ((LT.db_max_grade_after >= 2) | (LT.release_excluded_max_grade >= 2) | (LT.slidematch_max_grade_after >= 2)).values
bgr = pd.to_numeric(PT.baseline_grade, errors="coerce").fillna(0).values; mxg = pd.to_numeric(PT.max_grade, errors="coerce").fillna(0).values
Bd = bidx(y[disc]); Ba = bidx(y)
# ------------------------------------------------------------- K1
PRED = {"late_mean": sp.pred_late_mean.values.astype(int), "image_only": sp.pred_image_only.values.astype(int), "cnv_only": sp.pred_cnv_only.values.astype(int), "C2_grade_maxsofar": sp.pred_C2_grade_maxsofar.values.astype(int)}
K1 = {"per_stratum": {}, "validation_logistic": {}, "spearman_tiles_vs_laterHGD_validation_nonprogressors": None, "discovery_nonprogressors": {}}
for a, pred in PRED.items():
    K1["per_stratum"][a] = {}
    for l in ["discovery", "validation"]:
        m = neg & (strat2 == l); fp = m & (pred == 1); tn = m & (pred == 0); K1["per_stratum"][a][l] = {"FP": int(fp.sum()), "TN": int(tn.sum()), "laterHGD_FP": int(laterH[fp].sum()), "laterHGD_TN": int(laterH[tn].sum()), "rate_FP": r3(laterH[fp].mean()) if fp.sum() else None, "rate_TN": r3(laterH[tn].mean()) if tn.sum() else None, "wilson_FP": wilson(int(laterH[fp].sum()), int(fp.sum())), "wilson_TN": wilson(int(laterH[tn].sum()), int(tn.sum())), "fisher_p": pf(fisher_exact([[laterH[fp].sum(), fp.sum() - laterH[fp].sum()], [laterH[tn].sum(), tn.sum() - laterH[tn].sum()]])[1]) if fp.sum() and tn.sum() else None}
    m = neg & (strat2 == "validation"); fp = (pred == 1)[m]
    try:
        Xd = pd.DataFrame({"FP": fp.astype(float), "baseline_grade": bgr[m], "log_kept_tiles": np.log(kept[m])}); mres = sm.Logit(laterH[m].astype(int), sm.add_constant(Xd)).fit(disp=0, maxiter=300); ci_ = np.exp(mres.conf_int().loc["FP"]); K1["validation_logistic"][a] = {"OR_FP": r3(np.exp(mres.params["FP"])), "ci": [r3(ci_[0]), r3(ci_[1])], "p": pf(mres.pvalues["FP"]), "OR_log_tiles": r3(np.exp(mres.params["log_kept_tiles"])), "OR_baseline_grade": r3(np.exp(mres.params["baseline_grade"])), "n": int(m.sum()), "events": int(laterH[m].sum()), "separated": bool(ci_[1] > 1e4 or not np.isfinite(ci_).all())}
    except Exception as e: K1["validation_logistic"][a] = {"error": str(e)[:80], "fisher_only": True}
mv = neg & (strat2 == "validation"); rho = spearmanr(kept[mv], laterH[mv].astype(int)); K1["spearman_tiles_vs_laterHGD_validation_nonprogressors"] = {"rho": r3(rho.correlation), "p": pf(rho.pvalue), "n": int(mv.sum()), "events": int(laterH[mv].sum())}
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id").reindex(ids); fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
sheet_status = pd.Series(cx.cnv_id.map(fd.Status).values, index=pid).dropna().groupby(level=0).agg(lambda s: s.mode().iloc[0]).reindex(pats)
md = neg & disc; fu = LT.db_followup_days.fillna(0).values
K1["discovery_nonprogressors"] = {"n": int(md.sum()), "db_followup_days_median_iqr": [r3(np.median(fu[md])), r3(np.percentile(fu[md], 25)), r3(np.percentile(fu[md], 75))], "share_with_any_later_db_report": r3((LT.db_reports_after.values[md] > 0).mean()), "killcoyne_sheet_controls_NP": int((sheet_status.values[md] == "NP").sum()), "killcoyne_sheet_cases_P_labelled_nonprogressor_by_us": int((sheet_status.values[md] == "P").sum()), "later_HGDplus": int(laterH[md].sum()), "later_LGDplus": int(laterL[md].sum())}
RES["K1"] = K1; print("K1", flush=True)
# ------------------------------------------------------------- K2 FN vs TP within discovery
qc = pd.read_csv(T + "/feasibility/closeout/swg_cnv_qc.csv"); man["cnv_id"] = cx.cnv_id.values; man = man.join(qc.set_index("cnv_id")[["noise_mapd", "n_segments", "frac_altered_0p15"]], on="cnv_id"); man["reads"] = pd.to_numeric(man.cnv_id.map(fd["Number of reads"]).str.replace(",", ""), errors="coerce")
ui = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str).set_index("sample_id").reindex(ids); kk = []; gg = []
for b in ui.image_basename:
    with h5py.File(f"{S}/features_uni2h_05um/{os.path.splitext(str(b))[0]}.h5") as h: kk.append(float(h.attrs["kept_tiles"])); gg.append(float(h.attrs["grid_tiles"]))
man["tfrac"] = np.array(kk) / np.array(gg); slm = pd.read_csv(T + "/feasibility/closeout/swg_slide_meta.csv", dtype=str).set_index("sample_id").reindex(ids); man["scanner"] = slm["tiff.Model"].values
g = man.groupby("patient_id"); PF = pd.DataFrame({"first_to_event": pd.to_numeric(PT.days_first_to_event, errors="coerce"), "endpoint_label": man[man.y == 1].groupby("patient_id").next_label.max().reindex(pats), "n_rows": g.size().reindex(pats), "baseline_grade": bgr, "max_sofar": g.maxsofar.max().reindex(pats), "cx": pd.to_numeric(PT.cx_max, errors="coerce"), "noise": g.noise_mapd.mean().reindex(pats), "segments": g.n_segments.mean().reindex(pats), "frac_alt": g.frac_altered_0p15.mean().reindex(pats), "reads": g.reads.mean().reindex(pats), "kept_tiles": kept, "tissue_frac": g.tfrac.mean().reindex(pats), "scanner": g.scanner.agg(lambda s: s.mode().iloc[0]).reindex(pats), "p53": PT.p53_ihc_any_aberrant}, index=pats)
PF["endpoint_type"] = PF.endpoint_label.map({2: "second LGD", 3: "HGD", 4: "IMC/cancer", 5: "IMC/cancer"})
def cmp(m1, m2, l1, l2):
    out = {}
    for c in ["first_to_event", "n_rows", "baseline_grade", "max_sofar", "cx", "noise", "segments", "frac_alt", "reads", "kept_tiles", "tissue_frac"]:
        a_, b_ = PF[c].values[m1].astype(float), PF[c].values[m2].astype(float); a_, b_ = a_[~np.isnan(a_)], b_[~np.isnan(b_)]; out[c] = {l1: [r3(np.median(a_)) if len(a_) else None, int(len(a_))], l2: [r3(np.median(b_)) if len(b_) else None, int(len(b_))], "mannwhitney_p": pf(mannwhitneyu(a_, b_).pvalue) if len(a_) > 1 and len(b_) > 1 else None}
    for c in ["endpoint_type", "scanner", "p53"]:
        t = pd.crosstab(PF[c].fillna("missing").values[m1 | m2], np.where(m1[m1 | m2], l1, l2)); out[c] = {"table": {str(k): {str(kk_): int(vv) for kk_, vv in v.items()} for k, v in t.to_dict().items()}, "fisher_p": pf(fisher_exact(t.values)[1]) if t.shape == (2, 2) else None}
    return out
K2 = {}
for a in ["late_mean", "image_only"]:
    pos = (y == 1) & disc; fn = pos & (PRED[a] == 0); tp = pos & (PRED[a] == 1); K2[a] = {"n_FN": int(fn.sum()), "n_TP": int(tp.sum()), "comparison": cmp(fn, tp, "FN", "TP")}
RES["K2"] = K2; print("K2", flush=True)
# ------------------------------------------------------------- K3 latent within discovery
Lat = T + "/feasibility/paper_plan/latent"; inner = {k: pd.read_csv(f"{R}/image_only/fold{k}/inner_fold_assignments.csv", dtype=str).set_index("patient_id").inner_fold.astype(int) for k in range(1, 6)}
def pmean(M): return pd.DataFrame(M, index=pid).groupby(level=0).mean().reindex(pats).values
K3 = {}; PROBE = {}; KNN = {}; EMBP = {}
for fam in ["image_only", "cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]:
    E = {k: np.load(f"{Lat}/emb_{fam}_fold{k}.npy") for k in range(1, 6)}; EMBP[fam] = E; probe = np.zeros(len(pats)); knn = np.zeros(len(pats))
    for k in range(1, 6):
        Pm = pmean(E[k]); tr = pfold != k; te = ~tr; sc = StandardScaler().fit(Pm[tr]); Xtr, Xte = sc.transform(Pm[tr]), sc.transform(Pm[te]); ytr = y[tr]; inn = inner[k].reindex(pats[tr]).values; best, bestC = -1, 1.0
        for C in [0.01, 0.1, 1.0, 10.0]:
            pv = np.zeros(tr.sum())
            for j in np.unique(inn): v = inn == j; pv[v] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], ytr[~v]).predict_proba(Xtr[v])[:, 1]
            a_ = auc(ytr, pv)
            if a_ > best: best, bestC = a_, C
        probe[te] = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]; knn[te] = KNeighborsClassifier(n_neighbors=10).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    PROBE[fam] = probe; KNN[fam] = knn; K3[fam] = {"n": int(disc.sum()), "events": int(y[disc].sum()), "probe_auroc_discovery": r3(auc(y[disc], probe[disc])), "probe_ci": cis(y[disc], Bd, probe[disc]), "knn10_auroc_discovery": r3(auc(y[disc], knn[disc])), "knn_ci": cis(y[disc], Bd, knn[disc]), "probe_auroc_pooled_item6": L6[fam]["probe_auroc"]}
for fam in ["early_fusion", "intermediate_fusion", "coattention_fusion"]:
    for ref in ["image_only", "cnv_only"]: K3[fam][f"probe_delta_vs_{ref}_discovery"] = [r3(auc(y[disc], PROBE[fam][disc]) - auc(y[disc], PROBE[ref][disc]))] + cis(y[disc], Bd, PROBE[fam][disc], PROBE[ref][disc]); K3[fam][f"knn_delta_vs_{ref}_discovery"] = [r3(auc(y[disc], KNN[fam][disc]) - auc(y[disc], KNN[ref][disc]))] + cis(y[disc], Bd, KNN[fam][disc], KNN[ref][disc])
RES["K3"] = K3; print("K3", flush=True)
# ------------------------------------------------------------- K4b 8a within discovery
def groups_tertile(a):
    gq = np.zeros(len(y), int)
    for k in range(1, 6):
        tr = pfold != k; te = ~tr; q1, q2 = np.quantile(P[a][tr], [1 / 3, 2 / 3]); gq[te] = np.where(P[a][te] > q2, 2, np.where(P[a][te] > q1, 1, 0))
    return gq
GC, GL = groups_tertile("cnv_only"), groups_tertile("late_mean"); yd = y[disc]; pc = rankdata(P["cnv_only"][disc]) / disc.sum() * 100; pl = rankdata(P["late_mean"][disc]) / disc.sum() * 100; dr = pl - pc
def nri(yy, gc_, gl_): up = gl_ > gc_; dn = gl_ < gc_; ev = yy == 1; return (up[ev].mean() - dn[ev].mean()) + (dn[~ev].mean() - up[~ev].mean())
nb = [nri(yd[s], GC[disc][s], GL[disc][s]) for s in Bd]
K4b = {"n": int(disc.sum()), "events": int(yd.sum()), "spearman_cnv_vs_late": r3(spearmanr(P["cnv_only"][disc], P["late_mean"][disc]).correlation), "progressors_up_gt20": int(((dr > 20) & (yd == 1)).sum()), "progressors_down_gt20": int(((dr < -20) & (yd == 1)).sum()), "nonprogressors_up_gt20": int(((dr > 20) & (yd == 0)).sum()), "nonprogressors_down_gt20": int(((dr < -20) & (yd == 0)).sum()), "categorical_NRI": [r3(nri(yd, GC[disc], GL[disc])), r3(np.percentile(nb, 2.5)), r3(np.percentile(nb, 97.5))]}
K4a = json.load(open(AGG + "/k4_ablation.json")) if os.path.exists(AGG + "/k4_ablation.json") else (json.load(open(T + "/feasibility/runs/pk_gpu/output/k4_ablation.json")) if os.path.exists(T + "/feasibility/runs/pk_gpu/output/k4_ablation.json") else None)
RES["K4"] = {"ablation": K4a["models"] if K4a else "pending", "rank_shift_discovery": K4b}; print("K4", flush=True)
# ------------------------------------------------------------- K5 risk groups within discovery
def ca_trend(ns, rs):
    N, Rr = sum(ns), sum(rs); s = np.array([0, 1, 2.]); ns, rs = np.array(ns, float), np.array(rs, float); num = (s * (rs - ns * Rr / N)).sum(); var = (Rr / N) * (1 - Rr / N) * ((s ** 2 * ns).sum() - (s * ns).sum() ** 2 / N); z = num / np.sqrt(var) if var > 0 else np.nan; return r3(z), pf(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else None
K5 = {}
for ep, yy in [("LGD2plus", y), ("E_HGD", yH)]:
    K5[ep] = {}
    for a in ["C2_grade_maxsofar", "cnv_only", "image_only", "late_mean"]:
        gq = groups_tertile(a); out = {"groups": {}}; ns = []; rs = []
        for gi, gn in enumerate(["low", "moderate", "high"]): m = disc & (gq == gi); out["groups"][gn] = {"n": int(m.sum()), "events": int(yy[m].sum()), "rate": r3(yy[m].mean()) if m.sum() else None, "wilson": wilson(int(yy[m].sum()), int(m.sum()))}; ns.append(int(m.sum())); rs.append(int(yy[m].sum()))
        hi, lo = disc & (gq == 2), disc & (gq == 0); a_, b_, c_, d_ = yy[hi].sum() + .5, (1 - yy[hi]).sum() + .5, yy[lo].sum() + .5, (1 - yy[lo]).sum() + .5; lor = np.log((a_ * d_) / (b_ * c_)); se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
        out["OR_high_vs_low_haldane"] = [r3(np.exp(lor)), r3(np.exp(lor - 1.96 * se)), r3(np.exp(lor + 1.96 * se))]; out["cochran_armitage_z_p"] = ca_trend(ns, rs); K5[ep][a] = out
RES["K5"] = K5; print("K5", flush=True)
# ------------------------------------------------------------- stratified helpers for figures
def strat_auc(yy, s, g2):
    num = den = 0.0
    for l in np.unique(g2):
        m = g2 == l; n1 = yy[m].sum(); n0 = m.sum() - n1
        if n1 > 0 and n0 > 0: num += n1 * n0 * auc(yy[m], s[m]); den += n1 * n0
    return num / den
def boots_strat(yy):
    rng = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        s = np.concatenate([rng.choice(np.where(strat2 == l)[0], (strat2 == l).sum()) for l in ["discovery", "validation"]])
        if all(len(set(yy[s][strat2[s] == l])) > 1 for l in ["discovery", "validation"]): out.append(s)
    return out
BS = boots_strat(y); BSH = boots_strat(yH); BaH = bidx(yH); BdH = bidx(yH[disc])
def est(pop, ep, a, b=None):
    yy = y if ep == "LGD2plus" else yH
    if pop == "discovery": m = disc; Bs = Bd if ep == "LGD2plus" else BdH; f = lambda yv, s: auc(yv, s); yy2, sa, sb = yy[m], P[a][m], (P[b][m] if b else None)
    elif pop == "pooled": Bs = Ba if ep == "LGD2plus" else BaH; f = auc; yy2, sa, sb = yy, P[a], (P[b] if b else None)
    else: Bs = BS if ep == "LGD2plus" else BSH; f = lambda yv, s, g=None: None; yy2, sa, sb = yy, P[a], (P[b] if b else None)
    if pop == "stratified":
        v = [strat_auc(yy2[s], sa[s], strat2[s]) - (strat_auc(yy2[s], sb[s], strat2[s]) if b else 0) for s in Bs]; pt_ = strat_auc(yy2, sa, strat2) - (strat_auc(yy2, sb, strat2) if b else 0)
    else:
        v = [f(yy2[s], sa[s]) - (f(yy2[s], sb[s]) if b else 0) for s in Bs]; pt_ = f(yy2, sa) - (f(yy2, sb) if b else 0)
    return [r3(pt_), r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]
NE = {"discovery": {"LGD2plus": [int(disc.sum()), int(y[disc].sum())], "E_HGD": [int(disc.sum()), int(yH[disc].sum())]}, "stratified": {"LGD2plus": [150, int(y.sum())], "E_HGD": [150, int(yH.sum())]}, "pooled": {"LGD2plus": [150, int(y.sum())], "E_HGD": [150, int(yH.sum())]}}
# ------------------------------------------------------------- F-intro
FI = {"panels": {}}; fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1.3, 1]}); rows = []; labels = []
for ep in ["LGD2plus", "E_HGD"]:
    for pop in ["discovery", "stratified", "pooled"]:
        rec = {"population": pop, "endpoint": ep, "n_events": NE[pop][ep]}
        for a in ["image_only", "cnv_only", "cnv_km"]: rec[a] = est(pop, ep, a)
        rec["WSI_minus_cnv_only"] = est(pop, ep, "image_only", "cnv_only"); rec["WSI_minus_cnv_km"] = est(pop, ep, "image_only", "cnv_km"); rows.append(rec); labels.append(f"{pop} / {'LGD2+' if ep == 'LGD2plus' else 'E-HGD (post hoc)'}\nn {rec['n_events'][0]}, ev {rec['n_events'][1]}")
FI["rows"] = rows; yy_ = np.arange(len(rows))[::-1]; cols = {"image_only": "#1f77b4", "cnv_only": "#d62728", "cnv_km": "#ff7f0e"}
for j, a in enumerate(["image_only", "cnv_only", "cnv_km"]):
    v = np.array([r[a] for r in rows]); axes[0].errorbar(v[:, 0], yy_ + (j - 1) * 0.22, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=4, color=cols[a], label=NAME[a], capsize=2)
axes[0].axvline(0.5, ls=":", c="grey"); axes[0].set_yticks(yy_); axes[0].set_yticklabels(labels); axes[0].set_xlabel("AUROC (95 % CI)"); axes[0].legend(fontsize=7, loc="lower right"); axes[0].set_title("WSI vs CNV arms")
for j, (k_, c_) in enumerate([("WSI_minus_cnv_only", "#d62728"), ("WSI_minus_cnv_km", "#ff7f0e")]):
    v = np.array([r[k_] for r in rows]); axes[1].errorbar(v[:, 0], yy_ + (j - 0.5) * 0.25, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="s", ms=4, color=c_, label=k_.replace("_minus_", " − ").replace("cnv_only", "CNV RF").replace("cnv_km", "CNV KM"), capsize=2)
axes[1].axvline(0, c="grey"); axes[1].axvline(-0.05, ls="--", c="k", label="non-inferiority margin −0.05"); axes[1].set_yticks(yy_); axes[1].set_yticklabels([]); axes[1].set_xlabel("ΔAUROC WSI − CNV (95 % CI)"); axes[1].legend(fontsize=7, loc="lower right"); axes[1].set_title("Difference")
fig.tight_layout(); savefig(fig, "F_intro_forest", FI); RES["figures"]["F_intro"] = "results/paper_final/figs/F_intro_forest"
# ------------------------------------------------------------- F-table
FT = {"A_discovery_LGD2plus": {}, "B_discovery_EHGD": {}, "C_pooled_stratified_discovery_LGD2plus": {}}
for a in ARMS: FT["A_discovery_LGD2plus"][a] = est("discovery", "LGD2plus", a); FT["B_discovery_EHGD"][a] = est("discovery", "E_HGD", a); FT["C_pooled_stratified_discovery_LGD2plus"][a] = {pop: est(pop, "LGD2plus", a) for pop in ["pooled", "stratified", "discovery"]}
FT["n_events"] = NE; fig, axes = plt.subplots(1, 3, figsize=(13, 4.5)); yy_ = np.arange(len(ARMS))[::-1]
for ax, key, ttl in [(axes[0], "A_discovery_LGD2plus", f"A. Discovery, LGD2+ (n {NE['discovery']['LGD2plus'][0]}, events {NE['discovery']['LGD2plus'][1]})"), (axes[1], "B_discovery_EHGD", f"B. Discovery, E-HGD post hoc (n {NE['discovery']['E_HGD'][0]}, events {NE['discovery']['E_HGD'][1]})")]:
    for i, a in enumerate(ARMS): v = FT[key][a]; ax.errorbar(v[0], yy_[i], xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", color="grey" if a in ("coattention_fusion", "late_stack_logit") else "#1f77b4", capsize=2)
    ax.set_yticks(yy_); ax.set_yticklabels([NAME[a] for a in ARMS]); ax.axvline(0.5, ls=":", c="grey"); ax.set_xlabel("AUROC (95 % CI)"); ax.set_title(ttl, fontsize=8); ax.set_xlim(0.3, 1.0)
ax = axes[2]
for j, (pop, c_) in enumerate([("pooled", "#bbbbbb"), ("stratified", "#ff7f0e"), ("discovery", "#1f77b4")]):
    v = np.array([FT["C_pooled_stratified_discovery_LGD2plus"][a][pop] for a in ARMS]); ax.errorbar(v[:, 0], yy_ + (j - 1) * 0.25, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=4, color=c_, label=f"{pop}" + (" (confounded)" if pop == "pooled" else ""), capsize=2)
ax.set_yticks(yy_); ax.set_yticklabels([]); ax.axvline(0.5, ls=":", c="grey"); ax.set_xlabel("AUROC (95 % CI), LGD2+"); ax.set_title("C. Pooled vs stratified vs discovery", fontsize=8); ax.legend(fontsize=7, loc="lower right"); ax.set_xlim(0.3, 1.0)
fig.tight_layout(); savefig(fig, "F_table_forest", FT); RES["figures"]["F_table"] = "results/paper_final/figs/F_table_forest"
# ------------------------------------------------------------- F-D1 risk groups
D1 = {"tertile_rates_discovery_LGD2plus": K5["LGD2plus"], "MH_OR_two_strata": {a: R3["R4"]["pooled_rates_and_ORs"][a]["MH_two_strata"] for a in ["C2_grade_maxsofar", "cnv_only", "image_only", "late_mean"]}}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]}); arms4 = ["C2_grade_maxsofar", "cnv_only", "image_only", "late_mean"]; w = 0.25
for j, gn in enumerate(["low", "moderate", "high"]):
    vals = [K5["LGD2plus"][a]["groups"][gn]["rate"] or 0 for a in arms4]; lo = [K5["LGD2plus"][a]["groups"][gn]["wilson"][0] or 0 for a in arms4]; hi = [K5["LGD2plus"][a]["groups"][gn]["wilson"][1] or 0 for a in arms4]
    xs = np.arange(4) + (j - 1) * w; axes[0].bar(xs, vals, w, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], capsize=2, label=f"{gn} tertile", color=["#1f77b4", "#ff7f0e", "#d62728"][j])
    for x_, a in zip(xs, arms4): axes[0].text(x_, 0.02, f"{K5['LGD2plus'][a]['groups'][gn]['events']}/{K5['LGD2plus'][a]['groups'][gn]['n']}", ha="center", fontsize=6, rotation=90, color="white")
axes[0].set_xticks(np.arange(4)); axes[0].set_xticklabels([NAME[a] for a in arms4], fontsize=7); axes[0].set_ylabel("progression rate (Wilson 95 % CI)"); axes[0].set_title(f"Discovery, LGD2+ (n {int(disc.sum())}, events {int(y[disc].sum())}); training-fold tertiles", fontsize=8); axes[0].legend(fontsize=7)
mh = [D1["MH_OR_two_strata"][a] for a in arms4]; axes[1].errorbar([m_["OR"] for m_ in mh], np.arange(4)[::-1], xerr=[[m_["OR"] - m_["ci"][0] for m_ in mh], [m_["ci"][1] - m_["OR"] for m_ in mh]], fmt="o", capsize=2, color="k"); axes[1].set_xscale("log"); axes[1].axvline(1, ls=":", c="grey"); axes[1].set_yticks(np.arange(4)[::-1]); axes[1].set_yticklabels([NAME[a] for a in arms4], fontsize=7); axes[1].set_xlabel("Mantel–Haenszel OR high vs low (two strata, all 150)"); axes[1].set_title("Stratified OR", fontsize=8)
fig.tight_layout(); savefig(fig, "F_D1_risk_groups", D1); RES["figures"]["F_D1"] = "results/paper_final/figs/F_D1_risk_groups"
# ------------------------------------------------------------- F-D2 latent
D2 = {"A_probe_auroc": {fam: {"pooled_item6": L6[fam]["probe_auroc"], "pooled_ci": L6[fam]["probe_ci"], "discovery": K3[fam]["probe_auroc_discovery"], "discovery_ci": K3[fam]["probe_ci"]} for fam in K3}, "B_C_umap": {"folds": "1-5, projection fitted on each fold's training patients, held-out patients shown", "n_heldout_per_fold": {}}}
fig = plt.figure(figsize=(16, 9)); gs = fig.add_gridspec(3, 6, height_ratios=[1.1, 1, 1]); ax = fig.add_subplot(gs[0, :])
fams = list(K3); xs = np.arange(len(fams))
for j, (key, c_, lab) in enumerate([("pooled_item6", "#bbbbbb", "pooled (all 150, item 6)"), ("discovery", "#1f77b4", "discovery held-out (82)")]):
    v = np.array([[D2["A_probe_auroc"][f][key]] + D2["A_probe_auroc"][f][key.replace("item6", "ci") if key == "pooled_item6" else "discovery_ci"] for f in fams]); ax.errorbar(xs + (j - 0.5) * 0.2, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", color=c_, capsize=2, label=lab)
ax.set_xticks(xs); ax.set_xticklabels(fams); ax.axhline(0.5, ls=":", c="grey"); ax.set_ylabel("linear-probe AUROC (95 % CI)"); ax.legend(fontsize=7); ax.set_title("A. Linear probe on patient-mean embeddings: pooled vs discovery-only evaluation", fontsize=9)
for r_, fam in enumerate(["image_only", "intermediate_fusion"]):
    for k in range(1, 6):
        Pm = pmean(EMBP[fam][k]); tr = pfold != k; te = ~tr; sc = StandardScaler().fit(Pm[tr]); Z = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit(sc.transform(Pm[tr])).transform(sc.transform(Pm[te])); D2["B_C_umap"]["n_heldout_per_fold"][f"{fam}_fold{k}"] = {"n": int(te.sum()), "discovery": int((te & disc).sum()), "events_discovery": int(y[te & disc].sum())}
        axb = fig.add_subplot(gs[1 + r_, k - 1]); s_ = strat2[te]; axb.scatter(Z[s_ == "validation", 0], Z[s_ == "validation", 1], s=10, c="#999999", label="validation"); axb.scatter(Z[s_ == "discovery", 0], Z[s_ == "discovery", 1], s=10, c="#ff7f0e", label="discovery"); axb.set_xticks([]); axb.set_yticks([]); axb.set_title(f"B. {fam} UMAP fold {k}, by stratum", fontsize=7)
        if k == 1: axb.legend(fontsize=6)
        yd_ = y[te]; dd_ = disc[te]; axb2 = axb.inset_axes([0.62, 0.62, 0.38, 0.38]); axb2.scatter(Z[dd_ & (yd_ == 0), 0], Z[dd_ & (yd_ == 0), 1], s=6, c="#1f77b4"); axb2.scatter(Z[dd_ & (yd_ == 1), 0], Z[dd_ & (yd_ == 1), 1], s=6, c="#d62728"); axb2.set_xticks([]); axb2.set_yticks([]); axb2.set_title("C. discovery, by label", fontsize=5)
fig.tight_layout(); savefig(fig, "F_D2_latent", D2); RES["figures"]["F_D2"] = "results/paper_final/figs/F_D2_latent"
# ------------------------------------------------------------- F-D3 attention
A7 = pd.read_csv(RP + "/attention_item7_rows.csv"); ids_l = list(np.load(Lat + "/ids.npy")); fold_l = pd.Series(folds, index=ids).reindex(ids_l).values; mass = {}
for fam in ["image_only", "intermediate_fusion", "coattention_fusion"]:
    v = []
    for k in range(1, 6): A = np.load(f"{Lat}/attn_{fam}_fold{k}.npy")[fold_l == k]; v += list((-np.sort(-A, 1))[:, :13].sum(1))
    mass[fam] = np.array(v)
D3 = {"A_spearman_rows": {fam: {"n_rows": int((A7.fusion_model == fam).sum()), "median": r3(A7.spearman[A7.fusion_model == fam].median()), "iqr": [r3(A7.spearman[A7.fusion_model == fam].quantile(.25)), r3(A7.spearman[A7.fusion_model == fam].quantile(.75))]} for fam in ["intermediate_fusion", "coattention_fusion"]}, "B_top5pct_mass": {fam: {"n_rows": int(len(v)), "median": r3(np.median(v)), "iqr": [r3(np.percentile(v, 25)), r3(np.percentile(v, 75))]} for fam, v in mass.items()}, "uniform_top5": 0.05, "montage_paths_cluster": "feasibility/paper_plan/figs/attention/M01-M20.png; manifest feasibility/paper_plan/montage_manifest_SECRET.csv"}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6)); bins = np.linspace(-1, 1, 41)
for fam, c_ in [("intermediate_fusion", "#1f77b4"), ("coattention_fusion", "#ff7f0e")]: axes[0].hist(A7.spearman[A7.fusion_model == fam], bins=bins, alpha=.6, color=c_, label=f"image-only vs {fam.replace('_fusion', '')} (n {int((A7.fusion_model == fam).sum())} rows)")
axes[0].set_xlabel("Spearman correlation of tile attention (per slide)"); axes[0].set_ylabel("slides"); axes[0].legend(fontsize=7); axes[0].set_title("A. Attention agreement", fontsize=8)
axes[1].boxplot([mass[f] for f in mass], labels=[f.replace("_fusion", "") for f in mass], showfliers=False); axes[1].axhline(0.05, ls="--", c="k", label="uniform (13 of 256 tiles)"); axes[1].set_ylabel("attention mass on top 5 % of tiles"); axes[1].legend(fontsize=7); axes[1].set_title("B. Attention concentration (707 slides)", fontsize=8)
fig.tight_layout(); savefig(fig, "F_D3_attention", D3); RES["figures"]["F_D3"] = "results/paper_final/figs/F_D3_attention"
# ------------------------------------------------------------- F-D4 CNV change
D4 = {"A_scatter_discovery": {"n": int(disc.sum()), "events": int(yd.sum()), "spearman": K4b["spearman_cnv_vs_late"], "points": [{"cnv_pct": r3(a_), "late_pct": r3(b_), "y": int(c_)} for a_, b_, c_ in zip(pc, pl, yd)]}, "B_ablation": K4a["models"] if K4a else "pending"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2)); axes[0].scatter(pc[yd == 0], pl[yd == 0], s=18, c="#1f77b4", label=f"non-progressor (n {int((yd == 0).sum())})"); axes[0].scatter(pc[yd == 1], pl[yd == 1], s=18, c="#d62728", label=f"progressor (n {int(yd.sum())})"); axes[0].plot([0, 100], [0, 100], c="grey", lw=.8); axes[0].set_xlabel("CNV-only percentile (discovery)"); axes[0].set_ylabel("late-fusion percentile (discovery)"); axes[0].legend(fontsize=7); axes[0].set_title(f"A. Rank shift, discovery (Spearman {K4b['spearman_cnv_vs_late']})", fontsize=8)
if K4a:
    fams3 = ["early_fusion", "intermediate_fusion", "coattention_fusion"]; xs = np.arange(3); w = 0.2
    for j, (pop, abl, c_, lab) in enumerate([("discovery_82", "cnv_permuted_jointly_mean_of_50", "#d62728", "CNV destroyed, discovery"), ("all_150", "cnv_permuted_jointly_mean_of_50", "#f4a6a6", "CNV destroyed, pooled"), ("discovery_82", "image_bags_permuted_mean_of_50", "#1f77b4", "image destroyed, discovery"), ("all_150", "image_bags_permuted_mean_of_50", "#9ecae1", "image destroyed, pooled")]):
        v = np.array([[K4a["models"][f][pop][abl]["delta_auroc"]] + K4a["models"][f][pop][abl]["ci"] for f in fams3]); axes[1].bar(xs + (j - 1.5) * w, v[:, 0], w, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=2, color=c_, label=lab)
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f.replace("_fusion", "") for f in fams3]); axes[1].axhline(0, c="grey"); axes[1].set_ylabel("ΔAUROC (baseline − ablated), 95 % CI"); axes[1].legend(fontsize=6); axes[1].set_title("B. Modality ablation", fontsize=8)
else: axes[1].text(0.1, 0.5, "ablation pending"); 
fig.tight_layout(); savefig(fig, "F_D4_cnv_change", D4); RES["figures"]["F_D4"] = "results/paper_final/figs/F_D4_cnv_change"
# ------------------------------------------------------------- F-D5 FP by stratum
D5 = K1["per_stratum"]; fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
for ax, a in zip(axes, ["late_mean", "image_only", "cnv_only", "C2_grade_maxsofar"]):
    for j, l in enumerate(["discovery", "validation"]):
        for i, grp in enumerate(["FP", "TN"]):
            d_ = D5[a][l]; rate = d_[f"rate_{grp}"] or 0; wl = d_[f"wilson_{grp}"]; x_ = j * 2.4 + i; ax.bar(x_, rate, 0.8, yerr=[[rate - (wl[0] or 0)], [(wl[1] or 0) - rate]], capsize=2, color="#d62728" if grp == "FP" else "#1f77b4"); ax.text(x_, rate + 0.02, f"{d_['laterHGD_' + grp]}/{d_[grp]}", ha="center", fontsize=6)
    ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(["FP\ndisc.", "TN\ndisc.", "FP\nval.", "TN\nval."], fontsize=7); ax.set_title(f"{NAME[a]}\nFisher p disc {D5[a]['discovery']['fisher_p']}, val {D5[a]['validation']['fisher_p']}", fontsize=7)
axes[0].set_ylabel("later HGD+ rate (Wilson 95 % CI)"); fig.tight_layout(); savefig(fig, "F_D5_false_positives", D5); RES["figures"]["F_D5"] = "results/paper_final/figs/F_D5_false_positives"
# ------------------------------------------------------------- F-D6 FN vs TP discovery
pos = (y == 1) & disc; fn = pos & (PRED["late_mean"] == 0); tp = pos & (PRED["late_mean"] == 1); D6 = {"n_FN": int(fn.sum()), "n_TP": int(tp.sum()), "boxes": {}, "endpoint_type": {}}
fig, axes = plt.subplots(1, 5, figsize=(13, 3.6))
for ax, c, lab in zip(axes[:4], ["kept_tiles", "cx", "first_to_event", "max_sofar"], ["tissue tiles kept (0.44 µm)", "CNV complexity cx (max)", "days first row → endpoint", "max grade so far (0/1/2)"]):
    a_, b_ = PF[c].values[fn].astype(float), PF[c].values[tp].astype(float); a_, b_ = a_[~np.isnan(a_)], b_[~np.isnan(b_)]; ax.boxplot([a_, b_], labels=[f"FN (n {len(a_)})", f"TP (n {len(b_)})"], showfliers=True); ax.set_title(f"{lab}\nMann-Whitney p {pf(mannwhitneyu(a_, b_).pvalue) if len(a_) > 1 and len(b_) > 1 else 'n/a'}", fontsize=7); D6["boxes"][c] = {"FN_values": [r3(v) for v in a_], "TP_values": [r3(v) for v in b_], "mannwhitney_p": pf(mannwhitneyu(a_, b_).pvalue) if len(a_) > 1 and len(b_) > 1 else None}
et = pd.crosstab(PF.endpoint_type.values[fn | tp], np.where(fn[fn | tp], "FN", "TP")).reindex(["second LGD", "HGD", "IMC/cancer"]).fillna(0); D6["endpoint_type"] = {str(k): {str(kk_): int(vv) for kk_, vv in v.items()} for k, v in et.to_dict().items()}
et.plot.bar(ax=axes[4], color=["#d62728", "#1f77b4"]); axes[4].set_title("endpoint type", fontsize=7); axes[4].set_ylabel("patients"); axes[4].tick_params(axis="x", rotation=20, labelsize=7)
fig.suptitle(f"Late-fusion false negatives vs true positives, discovery progressors only (FN {int(fn.sum())}, TP {int(tp.sum())})", fontsize=8); fig.tight_layout(); savefig(fig, "F_D6_false_negatives", D6); RES["figures"]["F_D6"] = "results/paper_final/figs/F_D6_false_negatives"
json.dump(RES, open(AGG + "/final_checks.json", "w"), indent=1, default=str); print("PK MAIN DONE", flush=True)
