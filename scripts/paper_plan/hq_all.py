"""Q2 sample-level breakdown at 1/3/5 years (docs/paper_horizon_q2_all_horizons.md). Reads the existing Q2 per-sample list written by ha_q2.py
(afa0278; feasibility/paper_plan/killcoyne_mm/horizon_answers/q2_patient_list.csv; L-LATE vs L-CNV, their matrix, all pre-event samples) - no
new predictions or thresholds. Adds means of raw cx and tissue tiles, and Killcoyne's published probability (MOESM4 Fig 2a `Probability`, checked
against set_C k_prob). Aggregates -> results/paper_final/horizon_q2_all/q2_all.json; patient list (study numbers) cluster-only."""
import json, os, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HA = M + "/horizon_answers"; OUT = T + "/results/paper_final/horizon_q2_all"; os.makedirs(OUT, exist_ok=True)
L = pd.read_csv(HA + "/q2_patient_list.csv", dtype={"Sample": str, "study_number": str}); L = L[(L.src == "their") & (L.comparison == "L-LATE vs L-CNV")]
X = pd.read_excel("/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper/41591_2020_1033_MOESM4_ESM.xlsx", sheet_name="Supporting data for Figure 2a", header=1)
pr = pd.to_numeric(X.set_index(X.Samplename.astype(str)).Probability, errors="coerce"); C = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample"); kp = pd.to_numeric(C.k_prob, errors="coerce")
L["pub_prob"] = L.Sample.map(pr); L["k_prob"] = L.Sample.map(kp)
q2 = json.load(open(T + "/results/paper_final/horizon_answers/q2.json"))["results"]["their"]
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
def ms(v): v = pd.to_numeric(pd.Series(v), errors="coerce").dropna(); return {"n": int(len(v)), "mean": r3(v.mean()) if len(v) else None, "sd": r3(v.std(ddof=1)) if len(v) > 1 else None}
res = {"source_list": HA + "/q2_patient_list.csv", "prob_match_moesm4_vs_kprob_max_abs_diff": r3(float((L.pub_prob - L.k_prob).abs().max())), "horizons": {}}
for t in (1, 3, 5):
    g = {k: L[(L.horizon == t) & (L.group == k)] for k in ("captured_cases_CNVneg_bestpos", "cases_both_pos", "cleared_controls_CNVpos_bestneg", "controls_both_pos")}
    o = {k: {"samples": int(len(v)), "patients": int(v.study_number.nunique())} for k, v in g.items()}
    for k in ("captured_cases_CNVneg_bestpos", "cases_both_pos"):
        o[k].update({"cx_raw_pkg": ms(g[k].cx_raw_pkg), "tissue_tiles": ms(g[k].tissue_tiles), "pub_prob_le_0.3": int((g[k].pub_prob <= 0.3).sum()), "pub_prob_missing": int(g[k].pub_prob.isna().sum())})
    rc = q2[f"{t}|sample"]["L-LATE_vs_L-CNV"]["reclassification"]
    o["check_vs_q2json"] = {"captured_samples": rc["cases"]["CNVneg_bestpos"] == o["captured_cases_CNVneg_bestpos"]["samples"], "cleared_samples": rc["controls"]["CNVpos_bestneg"] == o["cleared_controls_CNVpos_bestneg"]["samples"],
                            "captured_patients": rc["cases_patients"]["CNVneg_bestpos"] == o["captured_cases_CNVneg_bestpos"]["patients"], "cleared_patients": rc["controls_patients"]["CNVpos_bestneg"] == o["cleared_controls_CNVpos_bestneg"]["patients"]}
    res["horizons"][str(t)] = o; print(t, json.dumps(o)[:600], flush=True)
L[L.group.isin(["captured_cases_CNVneg_bestpos", "cleared_controls_CNVpos_bestneg", "cases_both_pos", "controls_both_pos"])].to_csv(HA + "/q2_all_horizons_patients.csv", index=False)
json.dump(res, open(OUT + "/q2_all.json", "w"), indent=1); print("HQ DONE")
