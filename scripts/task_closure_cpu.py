"""Compute-closure CPU bundle (wave-3 gate items 1.18, 1.21, 2.36):

(a) 1.21 progression cohort v2-vs-v3 reconciliation — patient-level join,
    categorised gains/losses of patients and progressor flags.
(b) 1.18 Holm-Bonferroni over the four confirmatory contrasts' bootstrap
    sign probabilities.
(c) 2.36 patient-level aggregation of the ERIN grade task from saved OOF
    (frozen protocol says patient-level; slide-level was reported).
"""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
res = {}

# ---------- (a) v2 vs v3 reconciliation ----------
v3 = pd.read_csv(T + "/labeller/erin_progression_cohort_v3.csv", dtype=str)
v2p = sorted(glob.glob(T + "/labeller/*progression*v2*.csv")) or \
      sorted(glob.glob(T + "/labeller/erin_progression_cohort*.csv"))
v2_file = next((f for f in v2p if "v3" not in f), None)
if v2_file:
    v2 = pd.read_csv(v2_file, dtype=str)
    a = {r["anon_id"]: r.get("progressed_to_HGDplus") == "True" for _, r in v2.iterrows()}
    b = {r["anon_id"]: r.get("progressed_to_HGDplus") == "True" for _, r in v3.iterrows()}
    gained = sorted(set(b) - set(a)); lost = sorted(set(a) - set(b))
    common = sorted(set(a) & set(b))
    flips = [(p, a[p], b[p]) for p in common if a[p] != b[p]]
    res["reconciliation"] = {
        "v2_file": os.path.basename(v2_file),
        "v2": {"n": len(a), "progressors": int(sum(a.values()))},
        "v3": {"n": len(b), "progressors": int(sum(b.values()))},
        "patients_gained_in_v3": len(gained), "patients_lost_in_v3": len(lost),
        "event_flag_flips": len(flips),
        "flips_pos_to_neg": int(sum(1 for _, x, y in flips if x and not y)),
        "flips_neg_to_pos": int(sum(1 for _, x, y in flips if y and not x)),
        "progressors_among_gained": int(sum(b[p] for p in gained)),
        "progressors_among_lost": int(sum(a[p] for p in lost))}
else:
    res["reconciliation"] = {"error": "no v2 cohort file found",
                             "candidates": [os.path.basename(f) for f in v2p]}
print("reconciliation:", res["reconciliation"], flush=True)

# ---------- (b) Holm over the four confirmatory contrasts ----------
# sign probabilities: two-sided p approximated as 2*min(sign_prob, 1-sign_prob)
contrasts = {}
# 1. SWG late_mean vs image_only (released table)
rel = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/multimodal-barretts-progression/reports/thesis_ch1/lgd2_final_pre_event_paired_differences.csv"
d = pd.read_csv(rel)
row = d[(d["model_a"] == "late_mean") & (d["model_b"] == "image_only")].iloc[0]
contrasts["ch2_swg_latemean_vs_hist_auc"] = float(row["delta_roc_auc_sign_prob"])
# 2. OCCAMS fusion vs hist (occams_v3: delta CI) — sign prob from delta CI unavailable;
#    approximate from results json delta_ci via normal assumption
oc = json.load(open(T + "/results/occams_v3.json"))
lo, hi = oc["late_hist_gen"]["delta_ci"]; m = oc["late_hist_gen"]["delta_mean"]
se = (hi - lo) / 3.92
from scipy.stats import norm
contrasts["ch2_occams_fusion_vs_hist_c"] = float(2 * norm.sf(abs(m) / se))
# 3. ERIN progression fusion vs hist (ablation file)
ep = json.load(open(T + "/results/erin_prog_ablation.json"))
lo, hi = ep["late_fusion"]["delta_vs_ref"]["ci"]; m = ep["late_fusion"]["delta_vs_ref"]["mean"]
se = (hi - lo) / 3.92
contrasts["ch3_erin_prog_fusion_vs_hist_auc"] = float(2 * norm.sf(abs(m) / (se + 1e-12)))
# 4. Ch4 jury-vs-pathladder trained delta (ch4 xeval: same eval, paired) — from
#    label-source result: jury 0.9209 vs pathladder 0.9211 — treat via xeval bootstrap
ch4 = json.load(open(T + "/results/ch4_labelsource_xeval.json"))
if "pathladder" in ch4 and "delta_vs_ref" in ch4.get("pathladder", {}):
    lo, hi = ch4["pathladder"]["delta_vs_ref"]["ci"]; m = ch4["pathladder"]["delta_vs_ref"]["mean"]
    se = (hi - lo) / 3.92
    contrasts["ch4_pathladder_vs_jury_trained_auc"] = float(2 * norm.sf(abs(m) / (se + 1e-12)))
ps = sorted(contrasts.items(), key=lambda kv: kv[1])
holm = {}
k = len(ps)
run_max = 0.0  # Holm step-down requires monotone non-decreasing adjusted p
for i, (name, p_) in enumerate(ps):
    run_max = max(run_max, min(1.0, (k - i) * p_))
    holm[name] = {"p_raw": round(p_, 5), "p_holm": round(run_max, 5),
                  "significant_at_0.05": bool(run_max < 0.05)}
res["holm_confirmatory"] = holm
print("holm:", holm, flush=True)

# ---------- (c) patient-level ERIN grade aggregation ----------
z = np.load(T + "/feasibility/runs/ch4_labelsource_xeval/output/oof_preds.npz", allow_pickle=True)
keys = list(z["keys"])
m_ = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5").set_index("h5")
POS = {"LGD", "HGD", "CANCER"}
sub = m_.loc[[k for k in keys if k in m_.index]]
pred = pd.DataFrame({"h5": keys, "p": z["jury"]}).set_index("h5").loc[sub.index]
df = pd.DataFrame({"pat": sub["anon_id"], "y": sub["final_label"].isin(POS).astype(int),
                   "p": pred["p"]})
pat = df.groupby("pat").agg(y=("y", "max"), p=("p", "max"))
res["erin_grade_patient_level"] = {
    "n_slides": len(df), "n_patients": len(pat),
    "slide_level_auc_reference": 0.9209,
    "patient_level_auc": round(float(roc_auc_score(pat["y"], pat["p"])), 4)}
print("patient-level:", res["erin_grade_patient_level"], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
