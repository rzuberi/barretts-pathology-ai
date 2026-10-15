"""C1 evaluation: zero-shot LLM prognosis vs progression (ERIN v3), with index-grade baseline and bootstrap CIs."""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", ".")
coh = pd.read_csv(T + "/labeller/erin_progression_cohort_v3.csv", dtype=str); coh["y"] = coh.progressed_to_HGDplus.map(lambda v: 1 if str(v).lower() in ("true", "1") else 0)
GR = {"NDBE": 0, "IND": 1, "LGD": 2}; coh["g"] = coh.index_grade.map(GR).fillna(0)
def boot(y, s, n=2000):
    rng = np.random.RandomState(0); out = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(set(y[i])) > 1: out.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]
res = {"n_patients": len(coh), "progressors": int(coh.y.sum()), "baseline_index_grade_auroc": round(float(roc_auc_score(coh.y, coh.g)), 4), "baseline_ci": boot(coh.y.values, coh.g.values), "models": {}}
for tag in ("medgemma_27b", "qwen3_32b", "gemma3_27b"):
    fs = glob.glob(f"{T}/feasibility/runs/prog_*/output/prognosis_{tag}_shard*.csv")
    if not fs: continue
    d = pd.concat([pd.read_csv(f, dtype=str) for f in fs]).drop_duplicates("anon_id"); d["risk"] = pd.to_numeric(d.risk, errors="coerce")
    m = coh.merge(d, on="anon_id", how="left"); ok = m.risk.notna()
    res["models"][tag] = {"n_scored": int(ok.sum()), "parse_fail": int((~ok).sum()), "auroc": round(float(roc_auc_score(m.y[ok], m.risk[ok])), 4), "auroc_ci": boot(m.y[ok].values, m.risk[ok].values),
                          "auroc_within_index_grade_NDBE_only": round(float(roc_auc_score(m.y[ok & (m.g == 0)], m.risk[ok & (m.g == 0)])), 4) if (ok & (m.g == 0)).sum() > 20 and 0 < m.y[ok & (m.g == 0)].sum() else None,
                          "risk_by_outcome_median": {"progressors": float(m.risk[ok & (m.y == 1)].median()), "non": float(m.risk[ok & (m.y == 0)].median())},
                          "reports_shown_median": float(pd.to_numeric(m.n_reports_shown[ok], errors="coerce").median())}
res["reference_trained_text_model_db_corpus"] = 0.696
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2); print(json.dumps(res, indent=1))
