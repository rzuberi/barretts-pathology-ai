"""One paper, Extended Data Fig. 3 (Killcoyne et al. Fig. 4b) on the 288-endoscopy unit of Table 1 and Fig. 4b.
tp_kcf_figs.py groups samples by patient x months-before-endpoint (its ENDO2), which gives 25 progressors with two pre-event endoscopies.
Here an endoscopy is the `Endoscopy` field of the cohort sheet (tb_common.ENDO), restricted to pre-event samples, as in Fig. 4b (22 progressors at k = 2).
Same rule, same median risk classes (MC) and same grade / p53 mapping as tp_kcf_figs.recs(). Counts only (no bootstrap) -> ed3_endo.json.
Runs on the cluster (row-level inputs); writes aggregates only. CPU, a few minutes."""
import os, json, numpy as np
PP = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"
os.environ["TASK"] = "none"   # definitions only: no TASK branch of tp_kcf_figs matches
exec(compile(open(PP + "/tp_kcf_figs.py").read(), PP + "/tp_kcf_figs.py", "exec"))
# ------------------------------------------------------------------ the 288-endoscopy unit (pre-event samples only)
rows = np.where(PRE)[0]; eu8 = pd.unique(ENDO[rows]); R8 = [rows[ENDO[rows] == e] for e in eu8]
P8 = np.array([pidx[pat[r[0]]] for r in R8]); M8 = np.array([MBF[r].max() for r in R8])   # larger months-before-final = earlier
O8 = {i: [e for e in np.where(P8 == i)[0][np.argsort(-M8[P8 == i])]] for i in range(NPAT)}
CLS8 = {m: np.array([MC[m][r].max() for r in R8]) for m in ALLM}
G8 = np.array([G[r].max() for r in R8]); Q8 = np.array([np.nanmax(P53[r]) if np.isfinite(P53[r]).any() else np.nan for r in R8])
def recs8(m):
    out = {}
    for i in range(NPAT):
        es = O8[i]
        if len(es) < 2: continue
        if m == "GRADE":
            cc = [{0: 0, 1: 1, 2: 2}.get(int(G8[e]), 2) for e in es[:2]]; out[i] = rule(cc[0], cc[1], any(Q8[e] == 1 for e in es[:2]))
        else: out[i] = rule(int(CLS8[m][es[0]]), int(CLS8[m][es[1]]))
    return out
def counts(R):
    return {lab: [sum(1 for i, v in R.items() if P_y[i] == st and v == k) for k in range(4)] for lab, st in (("P", 1), ("NP", 0))}
res = {"unit": "cohort-sheet Endoscopy field, pre-event samples (Table 1 / Fig. 4b)", "n_endoscopies": len(R8),
       "n_endoscopies_P": int(sum(P_y[P8] == 1)), "n_endoscopies_NP": int(sum(P_y[P8] == 0)), "fig4b": {}}
for m in ("CN", "HE", "COMB", "GRADE"):
    res["fig4b"][m] = counts(recs8(m))
# checks: the old unit must reproduce the committed figs.json counts; the new unit must give 22 / 43 patients
res["old_unit_check"] = {m: counts(REC_OF[m]) for m in ("CN", "HE", "COMB", "GRADE")}
res["n_patients"] = {"P": int(sum(res["fig4b"]["CN"]["P"])), "NP": int(sum(res["fig4b"]["CN"]["NP"]))}
res["note"] = ("Recommendation after the first two pre-event endoscopies (BarrettsProgressionRisk rxRules; endoscopy class = highest median sample class); "
               "recommendation order: immediate RFA, 6-12-month recheck, 12-24-month recheck, 3-5-year surveillance.")
json.dump(res, open(OUTR + "/ed3_endo.json", "w"), indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
print("ED3 DONE", res["n_endoscopies"], res["n_patients"], flush=True)
