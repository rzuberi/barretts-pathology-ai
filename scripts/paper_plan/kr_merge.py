"""Killcoyne reconciliation merge (docs/paper_plan_killcoyne_reconcile.md @ a655e08): plan A (frozen model on our counts), plans B/C
(LOPO grid), plan D (step table). Aggregates only -> results/paper_final/killcoyne_reconcile.json."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; O = T + "/feasibility/paper_plan/killcoyne"; OUT = os.environ.get("OUTDIR", T + "/results/paper_final")
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
smp = pd.read_csv(O + "/kr_samples.csv", dtype=str).set_index("Sample"); smp["k_prob"] = smp.k_prob.astype(float); smp["y"] = (smp.Status == "P").astype(int)
rel = set(pd.read_csv(O + "/rel_rows.csv", dtype=str).Sample); disc = set(smp.loc[list(rel)].Patient)
def metrics(d, sc, boot=False, perm=False):   # d: Sample-indexed frame with Patient, y and score column sc
    g = d.groupby("Patient"); yp = g.y.max(); out = {"n_samples": int(len(d)), "n_patients": int(len(yp)), "n_P_patients": int(yp.sum()),
          "per_sample": r3(auc(d.y, d[sc])), "per_sample_4dp": round(auc(d.y, d[sc]), 4), "patient_mean": r3(auc(yp, g[sc].mean())), "patient_max": r3(auc(yp, g[sc].max()))}
    kp = smp.k_prob.reindex(d.index); ok = kp.notna(); out["spearman_vs_published"] = r3(spearmanr(d[sc][ok], kp[ok]).correlation)
    if boot or perm:
        pats = np.array(yp.index); rows = {p: np.where(d.Patient.values == p)[0] for p in pats}; ys, ss = d.y.values, d[sc].values; ypv = yp.values; pm = g[sc].mean().reindex(pats).values; px = g[sc].max().reindex(pats).values
    if boot:
        rng = np.random.RandomState(0); bs = []
        while len(bs) < 2000:
            ix = rng.choice(len(pats), len(pats))
            if 0 < ypv[ix].sum() < len(ix): ii = np.concatenate([rows[pats[i]] for i in ix]); bs.append((auc(ys[ii], ss[ii]), auc(ypv[ix], pm[ix]), auc(ypv[ix], px[ix])))
        bs = np.array(bs); out["ci95"] = {k: [r3(np.percentile(bs[:, j], 2.5)), r3(np.percentile(bs[:, j], 97.5))] for j, k in enumerate(["per_sample", "patient_mean", "patient_max"])}
    if perm:
        rng = np.random.RandomState(0); pr = []
        for _ in range(2000):
            yq = rng.permutation(ypv); mp = dict(zip(pats, yq)); pr.append((auc(d.Patient.map(mp).values, ss), auc(yq, pm), auc(yq, px)))
        pr = np.array(pr); obs = [auc(ys, ss), auc(ypv, pm), auc(ypv, px)]; out["perm_p"] = {k: round(float((1 + (pr[:, j] >= obs[j]).sum()) / 2001), 4) for j, k in enumerate(["per_sample", "patient_mean", "patient_max"])}
    return out
res = {"_spec": "docs/paper_plan_killcoyne_reconcile.md @ a655e08", "A0": json.load(open(O + "/kr_a0.json"))}
a0_ps = res["A0"]["all_773"]["per_sample"]["auroc_4dp"]
# plan A
A = pd.read_csv(O + "/kr_A_pred.csv").set_index("Sample").join(smp[["Patient", "y", "Pathology"]]); A["pass"] = A["pass"].astype(bool)
res["A"] = {"segmented": int(len(A)), "qc_pass": int(A["pass"].sum()), "all_segmented": metrics(A, "probA", True, True), "qc_pass_only": metrics(A[A["pass"]], "probA"),
            "release_rows": metrics(A[A.index.isin(rel)], "probA", True), "note": "frozen model trained on these samples: fidelity check, not a performance estimate"}
ins = pd.read_csv(O + "/their_insample_aligned.csv").set_index("Sample").join(smp[["Patient", "y"]]); res["A"]["shipped_matrix_insample"] = metrics(ins, "insample")
res["alignment"] = json.load(open(O + "/kr_align.json"))
# plans B, C
grid = pd.read_csv(O + "/grid.csv"); fs = sorted(glob.glob(O + "/lopo/cfg_*_chunk_*.csv")); L = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in fs]) if fs else pd.DataFrame()
rows = []
for (cfg, rule), d in L.groupby(["cfg", "rule"]):
    d = d.drop_duplicates("Sample").set_index("Sample"); gr = grid.set_index("cfg").loc[cfg].to_dict()
    exp_rows = {"s773": 773, "s711": int((~smp.Pathology.isin(["HGD", "IMC"])).sum())}.get(gr["set"])
    rec = {"cfg": int(cfg), **{k: gr[k] for k in ["set", "feat", "std", "alpha", "stdz", "nlam", "tag"]}, "rule": rule, "complete": bool(d.Patient.nunique() == (88 if gr["set"] == "s773" else d.Patient.nunique())),
           "own": metrics(d, "pred"), "disc82": metrics(d[d.Patient.isin(disc)], "pred"), "rel500": metrics(d[d.index.isin(rel)], "pred"), "median_lambda": r3(d["lambda"].median())}
    if gr["set"] == "s773": rec["reproduces"] = bool(abs(rec["own"]["per_sample_4dp"] - a0_ps) <= 0.010 and (rec["own"]["spearman_vs_published"] or 0) >= 0.80)
    rows.append(rec)
res["grid"] = rows; res["criterion"] = f"per-sample AUROC on 773 within +/-0.010 of {a0_ps} and Spearman with published >= 0.80"
def pick(**kw):
    for r in rows:
        if all((r[k] == v) if not isinstance(v, float) else abs(r[k] - v) < 1e-9 for k, v in kw.items()): return r
b = pick(set="s773", feat="pkg", std="cohort", alpha=0.9, stdz=True, nlam=100, rule="class_min")
if b is not None:
    d = L[(L.cfg == b["cfg"]) & (L.rule == "class_min")].drop_duplicates("Sample").set_index("Sample")
    res["B"] = {"cfg": b["cfg"], "own": metrics(d, "pred", True, True), "rel500": metrics(d[d.index.isin(rel)], "pred", True)}
steps = [("0 our R2 retrain (sklearn elastic net, release features, 504 release rows, C by patient AUROC)", None),
         ("1 R glmnet, release features, release rows, whole-set standardisation, alpha 0.9, class min", dict(set="rel504", feat="rel", std="cohort", alpha=0.9, stdz=True, nlam=100, rule="class_min")),
         ("2 package features from our counts (same rows)", dict(set="rel504", feat="pkg", std="cohort", alpha=0.9, stdz=True, nlam=100, rule="class_min")),
         ("3 train on all 773 published samples (HGD/IMC and post-event included)", dict(set="s773", feat="pkg", std="cohort", alpha=0.9, stdz=True, nlam=100, rule="class_min")),
         ("4 their shipped training matrix instead of our features [post hoc]", dict(set="s773", feat="their", std="cohort", alpha=0.9, stdz=True, nlam=100, rule="class_min")),
         ("5 glmnet standardize = FALSE, as in their fit call [post hoc]", dict(set="s773", feat="their", std="cohort", alpha=0.9, stdz=False, nlam=100, rule="class_min")),
         ("6 lambda.1se (class), as in their LOO coefficient tables [post hoc]", dict(set="s773", feat="their", std="cohort", alpha=0.9, stdz=False, nlam=100, rule="class_1se")),
         ("7 1,000-value lambda path [post hoc]", dict(set="s773", feat="their", std="cohort", alpha=0.9, stdz=False, nlam=1000, rule="class_1se")),
         ("8 one fixed lambda for every left-out patient = the published model's lambda (not fold-honest) [post hoc]", dict(set="s773", feat="their", std="cohort", alpha=0.9, stdz=False, nlam=100, rule="published")),
         ("9 as 8 but trained on the release rows only (their matrix) [post hoc]", dict(set="rel504", feat="their", std="cohort", alpha=0.9, stdz=False, nlam=100, rule="published")),
         ("10 as 8 with our counts through the package (pkg features) [post hoc]", dict(set="s773", feat="pkg", std="cohort", alpha=0.9, stdz=False, nlam=100, rule="published"))]
r2 = json.load(open(T + "/results/paper_plan/round3_lopo.json")); D = []
for lab, kw in steps:
    if kw is None: D.append({"step": lab, "rel500_patient_max": r2["lopo_vs_sheet_status_matched_rows"]["patient_auroc_max"], "rel500_per_sample": r2["lopo_vs_sheet_status_matched_rows"]["row_auroc"], "s773_per_sample": None, "spearman_vs_published": r2["spearman_lopo_vs_published_rows"]}); continue
    r = pick(**kw)
    if r is None: D.append({"step": lab, "status": "NOT RUN"}); continue
    d = L[(L.cfg == r["cfg"]) & (L.rule == kw["rule"])].drop_duplicates("Sample").set_index("Sample"); m5 = metrics(d[d.index.isin(rel)], "pred", True)
    D.append({"step": lab, "cfg": r["cfg"], "rel500_patient_max": m5["patient_max"], "rel500_patient_max_ci": m5["ci95"]["patient_max"], "rel500_per_sample": m5["per_sample"],
              "s773_per_sample": r["own"]["per_sample"] if kw["set"] == "s773" else None, "spearman_vs_published": r["own"]["spearman_vs_published"]})
D.append({"step": "published LOPO probabilities (MOESM4)", "rel500_patient_max": res["A0"]["release_rows_matched"]["patient_max"]["auroc"], "rel500_per_sample": res["A0"]["release_rows_matched"]["per_sample"]["auroc"], "s773_per_sample": res["A0"]["all_773"]["per_sample"]["auroc"], "spearman_vs_published": 1.0})
res["D"] = D
os.makedirs(OUT, exist_ok=True); json.dump(res, open(OUT + "/killcoyne_reconcile.json", "w"), indent=1)
print(json.dumps({"A": {k: res["A"][k] for k in ["segmented", "qc_pass"]}, "A_all": res["A"]["all_segmented"], "D": D}, indent=1)); print("MERGE DONE")
