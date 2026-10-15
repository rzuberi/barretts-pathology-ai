"""Killcoyne final merge (docs/paper_plan_killcoyne_final.md @ 8342ac0): parts of kf_stats.py, kf_power.py and kf_aceb_size.py ->
results/paper_final/killcoyne_final.json (aggregates only)."""
import json, glob, os, numpy as np
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; F = T + "/feasibility/paper_plan/killcoyne_mm/final"
parts = {os.path.basename(f)[:-5]: json.load(open(f))["result"] for f in glob.glob(F + "/parts/*.json")}
assert len(parts) == 9, sorted(parts)
res = {"_spec": "docs/paper_plan_killcoyne_final.md @ 8342ac0", "item1_fold_stratified": {}, "item2_max_T": parts["maxt"], "aceb_size": json.load(open(F + "/aceb_size.json")), "item3_power": {}}
for k, v in parts.items():
    if k.startswith("fs_"): _, src, sub = k.split("_", 2); res["item1_fold_stratified"].setdefault(src, {})[sub] = v
for S in ["S1", "S2", "S3", "S1m", "S2m"]:
    for E in (["E1", "E2"] if not S.endswith("m") else ["E1"]):
        P = [json.load(open(f)) for f in sorted(glob.glob(f"{F}/power/{S}_{E}_*.json"))]; assert len(P) == 8, (S, E, len(P))
        sims = [s for p in P for s in p["sims"]]; assert len({s["sim"] for s in sims}) == 2000
        lo = np.array([s["lower95"] for s in sims]); d = np.array([s["delta"] for s in sims]); ns = np.array([s["n_samples"] for s in sims])
        res["item3_power"][f"{S}_{E}"] = {"n_P": P[0]["n_P"], "n_NP": P[0]["n_NP"], "blend_w": round(P[0]["w"], 4), "set_C_NDBE_delta": round(P[0]["set_C_delta_at_w"], 3), "simulations": 2000,
                                          "power_lower95_above_0": round(float((lo > 0).mean()), 3), "power_95ci": [round(float((lo > 0).mean() - 1.96 * np.sqrt((lo > 0).mean() * (1 - (lo > 0).mean()) / 2000)), 3), round(float((lo > 0).mean() + 1.96 * np.sqrt((lo > 0).mean() * (1 - (lo > 0).mean()) / 2000)), 3)],
                                          "power_E2_shift_post_hoc": round(float((lo - (0.059 - 0.021) > 0).mean()), 3) if E == "E1" else None,
                                          "median_simulated_delta": round(float(np.median(d)), 3), "median_lower95": round(float(np.median(lo)), 3), "median_simulated_samples": int(np.median(ns)), "samples_range": [int(ns.min()), int(ns.max())]}
os.makedirs(T + "/results/paper_final", exist_ok=True); json.dump(res, open(T + "/results/paper_final/killcoyne_final.json", "w"), indent=1)
print(json.dumps(res["item3_power"], indent=1)); print("KF MERGE DONE")
