"""Aggregate the OAC response campaign: real arms + 50-permutation null.

Reads feasibility/runs/oacr_*/output/*.json and writes results/oac_response.json.
The null is the point of this script: occams_reference §3.1 records that a single
shuffle sitting off 0.5 was mistaken for a leak, and only a 50-permutation spread
resolved it. Primary claim requires BOTH the bootstrap CI gate and the
permutation p-value.
"""
import glob, json, os, sys
import numpy as np

T = os.environ.get("THESIS", "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis")
RUNS = os.path.join(T, "feasibility", "runs")
ENCODERS = ["uni_v2", "virchow2", "gigapath", "phikon2", "hoptimus0"]

def load(pat):
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, pat, "output", "*.json"))):
        try:
            out.append((d, json.load(open(d))))
        except Exception as e:
            print(f"  [skip] {d}: {e}")
    return out

res = {"_meta": {"design": "docs/oac_response_preregistration.md",
                 "primary": "uni_v2, TRG1-3 vs TRG4-5, AUC lower CI > 0.5 and point >= 0.60"}}

for enc in ENCODERS:
    got = load(f"oacr_{enc}")
    if not got:
        print(f"[pending] {enc}"); continue
    _, r = got[0]
    res[enc] = {k: v for k, v in r.items() if not k.startswith("_")}
    if enc == "uni_v2":
        res["_meta"].update({k: v for k, v in r.get("_meta", {}).items()})

def perm_summary(perms, observed):
    p = np.array(perms)
    return {"n_permutations": len(p),
            "null_mean": round(float(p.mean()), 4),
            "null_sd": round(float(p.std(ddof=1)), 4),
            "null_p2.5_p97.5": [round(float(np.percentile(p, 2.5)), 4),
                                round(float(np.percentile(p, 97.5)), 4)],
            "observed_auc": observed,
            "p_empirical": (round(float((p >= observed).sum() + 1) / (len(p) + 1), 4)
                            if observed is not None else None)}

# Per-encoder permutation nulls. UNI2 shards were named oacr_perm_<i> (first
# campaign); the secondary encoders' shards are oacr_perm_<enc>_<i>. Every
# encoder gets its own null -- comparing hoptimus0's AUC against UNI2's null
# would be the wrong reference distribution.
for enc in ENCODERS:
    if enc not in res:
        continue
    pat = "oacr_perm_[0-9]*" if enc == "uni_v2" else f"oacr_perm_{enc}_*"
    perms = [r["hist"]["auc"] for _, r in load(pat) if "hist" in r]
    if perms:
        res[enc]["permutation_null"] = perm_summary(perms, res[enc]["hist"]["auc"])
    else:
        print(f"[pending] permutation shards for {enc}")
    ecf = load(f"oacr_{enc}_ecf")           # pre-registered single-regimen sensitivity
    if ecf:
        _, r = ecf[0]
        res[enc]["ecf_only"] = {"n": r.get("_meta", {}).get("n"),
                                "hist": r["hist"], "clin": r["clin"]}
    else:
        print(f"[pending] ECF-only arm for {enc}")
if "uni_v2" in res and "permutation_null" in res["uni_v2"]:
    res["permutation_null"] = res["uni_v2"]["permutation_null"]   # primary, kept top-level

if "uni_v2" in res:
    h = res["uni_v2"]["hist"]
    ci = h.get("auc_ci", [None, None])
    gate = ci[0] is not None and ci[0] > 0.5 and h["auc"] >= 0.60
    pn = res.get("permutation_null", {})
    res["_meta"]["primary_gate_passed"] = bool(gate)
    res["_meta"]["verdict"] = (
        "POSITIVE — attention analysis unlocked" if gate and (pn.get("p_empirical") or 1) < 0.05
        else "NULL — attention analysis stays locked (§3.4 precedent)")
    # Secondary encoders that would have passed the gate had they been primary.
    # Reported, NOT promoted: the primary was pre-registered as UNI2 and stays so.
    sec = {}
    for enc in ENCODERS[1:]:
        if enc in res:
            hh = res[enc]["hist"]; cc = hh.get("auc_ci", [None, None])
            passes = cc[0] is not None and cc[0] > 0.5 and hh["auc"] >= 0.60
            sec[enc] = {"auc": hh["auc"], "auc_ci": cc, "would_pass_gate": passes,
                        "p_empirical": res[enc].get("permutation_null", {}).get("p_empirical")}
    res["_meta"]["secondary_encoders"] = sec
    res["_meta"]["n_encoders_tested"] = len([e for e in ENCODERS if e in res])

# ---- specimen-mix audit of the pre-registered Ch2 cohort ----
# occams_reference §1 calls OCCAMS a resection cohort, but task_occams_v3's
# RES-preference only fires when a RES slide exists, so OGD-only cases enter as
# biopsies (21/87 = 24%). These arms re-run the identical pipeline split by
# specimen. Published all-specimen values from results/occams_v3.json.
PUBLISHED = {"hist_abmil": 0.627, "gen_cox": 0.521, "clin_cox": 0.536,
             "late_hist_gen": 0.589, "late_hist_clin": 0.613}
audit = {}
for arm in ("res_only", "ogd_only"):
    got = load(f"occv3_{arm}")
    if not got:
        print(f"[pending] specimen audit {arm}"); continue
    _, r = got[0]
    meta = r.get("_meta", {})
    audit[arm] = {"n": meta.get("n"), "events": meta.get("events"),
                  "specimen_mix": meta.get("specimen_mix")}
    for k, pub in PUBLISHED.items():
        if k in r:
            v = r[k]
            c = v.get("c") if isinstance(v, dict) else v
            audit[arm][k] = {"c": c, "published_all_specimen": pub,
                             "delta_vs_published": (round(c - pub, 4)
                                                    if isinstance(c, (int, float)) else None)}
if audit:
    res["specimen_audit"] = audit

out = os.path.join(T, "results", "oac_response.json")
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1)); print("wrote", out)
