"""Part 2 aggregation: metrics with patient-clustered paired bootstrap, T1/T2/T3, slide + patient level,
confusion on T2, label-disagreement fractions, one-slide-per-case sanity check -> results.json + SUMMARY.md."""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun"
ORD = {"NORMAL_OTHER": -1, "NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}; CLS = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]; C_OF = {c: i for i, c in enumerate(CLS)}
s = pd.read_csv(A + "/slides.csv", dtype={"idx": int}).fillna(""); s["vote_frac"] = pd.to_numeric(s.vote_frac, errors="coerce")
excl = set(pd.read_csv(A + "/qc/exclusions.csv").idx) if os.path.getsize(A + "/qc/exclusions.csv") > 0 else set()
s = s[~s.idx.isin(excl)].set_index("idx")
rng0 = np.random.RandomState(1); hp = sorted(s[s.human_patient].anon_id.unique()); fold_hp = {a: i % 5 for i, a in enumerate(rng0.permutation(hp))}
s["fold_report"] = np.where(s.human_report, -1, np.where(s.fold >= 0, s.fold, s.anon_id.map(fold_hp))); s["fold_patient"] = s.fold
NB = 2000

def load(hold, arm, task):
    """Mean over seeds of per-fold test predictions (each slide predicted by exactly one fold model) and of
    holdout predictions (ensemble over the 5 fold models x seeds)."""
    te, ho = {}, {}
    for f in glob.glob(A + f"/preds/{hold}_{arm}_{task}_seed*_fold*.npz"):
        z = np.load(f)
        for i, p in zip(z["test_idx"], z["test_prob"]): te.setdefault(int(i), []).append(p)
        for i, p in zip(z["holdout_idx"], z["holdout_prob"]): ho.setdefault(int(i), []).append(p)
    return {k: np.mean(v, 0) for k, v in te.items()}, {k: np.mean(v, 0) for k, v in ho.items()}

def truth(ids, task, col):
    g = s.loc[ids, col]
    if task == "bin":
        keep = g.isin(["NORMAL_OTHER", "NDBE", "LGD", "HGD", "CANCER"]).values; return keep, g.isin(["LGD", "HGD", "CANCER"]).astype(int).values
    keep = g.isin(CLS).values; return keep, g.map(C_OF).fillna(-1).astype(int).values

def qwk(a, b, n=6):
    O = np.zeros((n, n))
    for i, j in zip(a, b): O[i, j] += 1
    W = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)]); E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() else float("nan")
def macro_auc(P, y):
    aucs = []
    for c in range(P.shape[1]):
        yy = (y == c).astype(int)
        if 0 < yy.sum() < len(yy): aucs.append(roc_auc_score(yy, P[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")
def metric(task, P, y):
    if task == "bin": return roc_auc_score(y, P) if 0 < y.sum() < len(y) else float("nan")
    return macro_auc(P, y)
def boot(task, Pa, y, pats, Pb=None, seed=0):
    """patient-clustered bootstrap of metric(a) and, if Pb, of metric(b) - metric(a)."""
    rng = np.random.RandomState(seed); up = np.unique(pats); groups = {p: np.where(pats == p)[0] for p in up}
    ma = metric(task, Pa, y); md = metric(task, Pb, y) - ma if Pb is not None else None
    va, vd = [], []
    for _ in range(NB):
        pick = rng.choice(up, len(up), replace=True); ix = np.concatenate([groups[p] for p in pick])
        a = metric(task, Pa[ix], y[ix]); va.append(a)
        if Pb is not None: vd.append(metric(task, Pb[ix], y[ix]) - a)
    out = {"point": round(ma, 4), "ci": [round(float(np.nanpercentile(va, 2.5)), 4), round(float(np.nanpercentile(va, 97.5)), 4)]}
    if Pb is not None: out["delta_b_minus_a"] = {"point": round(md, 4), "ci": [round(float(np.nanpercentile(vd, 2.5)), 4), round(float(np.nanpercentile(vd, 97.5)), 4)]}
    return out

def eval_set(hold, task, ids, col, name):
    """ids: slide idx of the test set; col: truth column. Returns per-arm metrics + paired deltas B-A, C-A, C-B."""
    res = {"name": name, "n_slides": 0}
    preds = {arm: load(hold, arm, task) for arm in "ABC"}
    src = 0 if name != "T2" else 1
    common = [i for i in ids if all(i in preds[a][src] for a in "ABC")]
    keep, y = truth(common, task, col); common = np.array(common)[keep]; y = y[keep]
    if len(common) < 10: return res
    pats = s.loc[common, "anon_id"].values
    P = {a: np.stack([preds[a][src][i] for i in common]) for a in "ABC"}
    res.update({"n_slides": int(len(common)), "n_patients": int(len(np.unique(pats))), "class_counts": (dict(zip(*np.unique(y, return_counts=True))) if task == "six" else {"pos": int(y.sum()), "neg": int(len(y) - y.sum())})})
    res["class_counts"] = {str(k): int(v) for k, v in res["class_counts"].items()}
    for a in "ABC": res[a] = boot(task, P[a], y, pats)
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]: res[f"{b}_minus_{a}"] = boot(task, P[a], y, pats, P[b])["delta_b_minus_a"]
    if task == "six":
        for a in "ABC": res[a]["qwk"] = round(qwk(P[a].argmax(1), y), 4)
        if name == "T2": res["confusion_truth_rows_pred_cols"] = {a: pd.crosstab(pd.Series(y, name="truth").map(dict(enumerate(CLS))), pd.Series(P[a].argmax(1), name="pred").map(dict(enumerate(CLS)))).reindex(index=CLS, columns=CLS, fill_value=0).to_dict() for a in "ABC"}
    # patient level: max over slides (binary) / argmax of max-prob (six -> use max over slides of class probs for macro AUC)
    df = pd.DataFrame({"pat": pats, "y": y}); pt = {}
    for a in "ABC":
        if task == "bin": pp = pd.Series(P[a]).groupby(pats).max(); yy = df.groupby("pat").y.max()
        else: pp = pd.DataFrame(P[a]).groupby(pats).max(); yy = df.groupby("pat").y.max()
        pt[a] = round(metric(task, pp.values if task == "bin" else pp.values, yy.values), 4)
    res["patient_level_point"] = pt; res["n_patients_eval"] = int(len(np.unique(pats)))
    return res

R = {"n_slides_after_qc": int(len(s)), "qc": json.load(open(A + "/qc/summary.json"))}
for hold in ["patient", "report"]:
    fcol = f"fold_{hold}"; R[hold] = {}
    for task in ["bin", "six"]:
        test = s[s[fcol] >= 0]; ho = s[(s[fcol] == -1) & (s.human_grade != "")]
        T1 = test[(test.vote_frac >= 0.75) & (test.worst_grade != "")].index.tolist(); T3 = test[test.worst_grade != ""].index.tolist(); T2 = ho.index.tolist()
        R[hold][task] = {"T1": eval_set(hold, task, T1, "worst_grade", "T1"), "T2": eval_set(hold, task, T2, "human_grade", "T2"), "T3": eval_set(hold, task, T3, "worst_grade", "T3")}
# label disagreement A vs B in the training population (own-section vs oesophageal case-max)
lab = s[s.worst_grade != ""]
R["label_disagreement_A_vs_B"] = {"all": round(float((lab.worst_grade != lab.case_max).mean()), 4), "n": int(len(lab)),
                                   "per_own_section_class": {g: round(float((lab[lab.worst_grade == g].case_max != g).mean()), 4) for g in CLS}}
# sanity: old one-slide-per-case picks (largest file) -> own-section == case-max?
sl = pd.read_csv("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/campaigns/allslides/erin_slides_all.csv", dtype=str).fillna(""); sl["size_mb"] = pd.to_numeric(sl.size_mb)
big = set(sl.sort_values("size_mb", ascending=False).drop_duplicates("CaseName").file_uuid)
bj = lab[lab.file_uuid.isin(big)]
R["sanity_largest_slide_equals_case_max"] = {"n": int(len(bj)), "frac": round(float((bj.worst_grade == bj.case_max).mean()), 4)}
R["train_log"] = [l for f in glob.glob(A + "/train_log_fold*.json") for l in json.load(open(f))]
R["nonconverged"] = [l["tag"] for l in R["train_log"] if not l["converged"]]
json.dump(R, open(A + "/results.json", "w"), indent=1, default=str)

def fmt(m): return f"{m['point']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]" if m and "point" in m else "–"
md = [f"# ISBI rerun — per-section vs case-max supervision ({R['n_slides_after_qc']} oesophageal/GOJ H&E slides after QC)\n",
      f"QC excluded {R['qc']['excluded']} ({R['qc']['excluded_frac']:.1%}): {R['qc']['by_reason']}. Non-converged models: {R['nonconverged'] or 'none'}.\n",
      f"Training-label disagreement A vs B (own-section ≠ oesophageal case-max): {R['label_disagreement_A_vs_B']['all']:.1%} of {R['label_disagreement_A_vs_B']['n']} slides; per own-section class: {R['label_disagreement_A_vs_B']['per_own_section_class']}.",
      f"Sanity check of the earlier one-slide-per-case runs: the largest slide's own-section grade equals the case-max in {R['sanity_largest_slide_equals_case_max']['frac']:.1%} of {R['sanity_largest_slide_equals_case_max']['n']} such slides.\n"]
for hold in ["patient", "report"]:
    md.append(f"## Human holdout = {hold} level")
    for task, lab_ in [("bin", "Binary LGD+ (slide AUROC)"), ("six", "Six-class (macro one-vs-rest AUROC; QWK)")]:
        md += [f"### {lab_}", "| test set | n slides (patients) | class counts | A case-max | B per-section | C conf-filtered | B−A | C−A | C−B | patient-level A/B/C |", "|---|---|---|---|---|---|---|---|---|---|"]
        for T in ["T1", "T2", "T3"]:
            r = R[hold][task][T]
            if r.get("n_slides", 0) < 10: md.append(f"| {T} | too few | | | | | | | | |"); continue
            q = (lambda a: f" (QWK {r[a]['qwk']:.3f})") if task == "six" else (lambda a: "")
            pl = r.get("patient_level_point", {})
            md.append(f"| {T}{' (PRIMARY)' if (T=='T1' and task=='bin' and hold=='patient') else ''} | {r['n_slides']} ({r['n_patients']}) | {r['class_counts']} | {fmt(r['A'])}{q('A')} | {fmt(r['B'])}{q('B')} | {fmt(r['C'])}{q('C')} | {fmt(r['B_minus_A'])} | {fmt(r['C_minus_A'])} | {fmt(r['C_minus_B'])} | {pl.get('A','–')}/{pl.get('B','–')}/{pl.get('C','–')} |")
        md.append("")
    if "confusion_truth_rows_pred_cols" in R[hold]["six"]["T2"]:
        md.append(f"T2 six-class confusion (truth rows × predicted cols), {hold} holdout: " + json.dumps(R[hold]["six"]["T2"]["confusion_truth_rows_pred_cols"]) + "\n")
open(A + "/SUMMARY_results.md", "w").write("\n".join(md)); print("\n".join(md))
for f in glob.glob(A + "/*.json") + glob.glob(A + "/*.md"): os.chmod(f, 0o600)
