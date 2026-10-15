"""2.23 (joint, Rehan 2026-08-25): power map — minimum detectable fusion delta
per cohort (GLM-5.3's proposal, wave-2 consultation).

For each cohort's (n, positives/events) and observed unimodal performance,
simulate paired arm predictions: a base risk score achieving the observed
AUC/C, and a fusion score = base signal + injected delta, correlated at rho.
Detection = paired bootstrap 95% CI on the delta excludes zero. 200 replicates
per (cohort, delta, rho). Readout: detection probability surface + the minimum
delta detectable with 80% power — 'a fusion benefit below X was structurally
undetectable in cohort Y'.

Binary-endpoint cohorts use the biGaussian AUC model. Survival cohorts use
exponential event times with risk-proportional hazard and censoring matched to
the observed event fraction, scored by Harrell's C.
"""
import json, os
import numpy as np
from scipy.stats import norm

OUT = os.environ.get("OUTDIR", ".")
REPS, N_BOOT = 200, 500
# v2 (Astra A9, 2026-09-14): d=0 added for a type-I check; baselines updated to
# the FINAL unimodal results; erin_grade uses the patient-level baseline that
# matches its patient n; deltas that would push the true AUC/C above CAP are
# reported as infeasible instead of being silently clipped.
DELTAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
RHOS = [0.6, 0.8, 0.9]
CAP = 0.99
ALPHA_HOLM = 0.05 / 4   # worst-case Holm threshold for the 4 confirmatory contrasts

COHORTS = {
    "swg":        {"kind": "binary",   "n": 150,  "pos": 50,  "base": 0.731},
    "occams_v3":  {"kind": "survival", "n": 87,   "events": 58,  "base": 0.627},
    "tcga_oac":   {"kind": "survival", "n": 65,   "events": 36,  "base": 0.679},
    "tcga_pool":  {"kind": "survival", "n": 399,  "events": 176, "base": 0.547},
    "erin_grade": {"kind": "binary",   "n": 1574, "pos": 528, "base": 0.960},
    "erin_prog":  {"kind": "binary",   "n": 153,  "pos": 28,  "base": 0.819},
}

def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 3), round(float(c + h), 3)]

def cindex(risk, time, event):
    order = np.argsort(time)
    risk, time, event = risk[order], time[order], event[order]
    num = den = 0
    for i in range(len(time)):
        if not event[i]: continue
        later = time > time[i]
        den += later.sum()
        num += (risk[later] < risk[i]).sum() + 0.5 * (risk[later] == risk[i]).sum()
    return num / den if den else 0.5

def auc_fast(y, s):
    r = np.argsort(np.argsort(s))
    return (r[y == 1].sum() - (y == 1).sum() * ((y == 1).sum() + 1) / 2) / ((y == 1).sum() * (y == 0).sum())

def make_binary(rng, n, pos, base_auc, delta_auc, rho):
    """Two correlated scores whose true AUCs are base and base+delta."""
    y = np.zeros(n, int); y[rng.choice(n, pos, replace=False)] = 1
    mu_a = norm.ppf(base_auc) * np.sqrt(2)          # biGaussian separation for AUC
    mu_b = norm.ppf(min(base_auc + delta_auc, 0.999)) * np.sqrt(2)
    z1 = rng.normal(size=n); z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.normal(size=n)
    a = z1 + mu_a * y
    b = z2 + mu_b * y
    return y, a, b

_ALPHA_CACHE = {}
def _alpha_for_c(target_c, events_frac):
    """Calibrate corr(score, true risk) needed to reach target C under this
    censoring level, on one large reference draw (deterministic seed)."""
    key = (round(target_c, 4), round(events_frac, 3))
    if key in _ALPHA_CACHE: return _ALPHA_CACHE[key]
    r = np.random.RandomState(12345)
    N = 20000
    u = r.normal(size=N)
    t_lat = r.exponential(np.exp(-1.5 * u))
    cens = np.quantile(t_lat, events_frac)
    time = np.minimum(t_lat, cens); event = t_lat <= cens
    sub = r.choice(N, 3000, replace=False)
    noise = r.normal(size=N)
    lo, hi = 0.0, 1.0
    for _ in range(14):
        mid = (lo + hi) / 2
        s_ = mid * u + np.sqrt(1 - mid**2) * noise
        c = cindex(s_[sub], time[sub], event[sub])
        lo, hi = (mid, hi) if c < target_c else (lo, mid)
    _ALPHA_CACHE[key] = (lo + hi) / 2
    return _ALPHA_CACHE[key]

def make_survival(rng, n, events, base_c, delta_c, rho):
    """True risk u drives exponential times; arm scores are noisy views of u
    whose fidelity (alpha) is calibrated so C(a)=base, C(b)=base+delta; the two
    arms' noise components are correlated at rho."""
    ef = events / n
    aa = _alpha_for_c(base_c, ef)
    ab = _alpha_for_c(min(base_c + delta_c, 0.99), ef)
    u = rng.normal(size=n)
    e1 = rng.normal(size=n); e2 = rho * e1 + np.sqrt(1 - rho**2) * rng.normal(size=n)
    a = aa * u + np.sqrt(1 - aa**2) * e1
    b = ab * u + np.sqrt(1 - ab**2) * e2
    t_lat = rng.exponential(np.exp(-1.5 * u))
    cens = np.quantile(t_lat, ef)
    time = np.minimum(t_lat, cens); event = t_lat <= cens
    return time, event, a, b

res = {"_meta": {"reps": REPS, "n_boot": N_BOOT, "deltas": DELTAS, "rhos": RHOS,
                 "cohorts": COHORTS}}
for cname, c in COHORTS.items():
    res[cname] = {}
    for rho in RHOS:
        for d in DELTAS:
            if c["base"] + d > CAP:
                res[cname][f"rho{rho}_d{d}"] = {"infeasible": True,
                                                "reason": f"base {c['base']} + delta {d} > cap {CAP}"}
                continue
            detect = 0; detect_holm = 0; runs = 0
            for rep in range(REPS):
                rng = np.random.RandomState(rep * 7919 + int(d * 1000) + int(rho * 100))
                if c["kind"] == "binary":
                    y, a, b = make_binary(rng, c["n"], c["pos"], c["base"], d, rho)
                    stats = []
                    for _ in range(N_BOOT):
                        idx = rng.randint(0, c["n"], c["n"])
                        if y[idx].sum() in (0, c["n"]): continue
                        stats.append(auc_fast(y[idx], b[idx]) - auc_fast(y[idx], a[idx]))
                else:
                    time, event, a, b = make_survival(rng, c["n"], c["events"], c["base"], d, rho)
                    stats = []
                    for _ in range(N_BOOT // 5):   # C-index bootstrap is costlier
                        idx = rng.randint(0, c["n"], c["n"])
                        if event[idx].sum() < 5: continue
                        stats.append(cindex(b[idx], time[idx], event[idx])
                                     - cindex(a[idx], time[idx], event[idx]))
                if len(stats) < 50: continue
                lo = np.percentile(stats, 2.5)
                lo_h = np.percentile(stats, 100 * ALPHA_HOLM / 2)
                runs += 1; detect += int(lo > 0); detect_holm += int(lo_h > 0)
            if runs:
                res[cname][f"rho{rho}_d{d}"] = {
                    "power": round(detect / runs, 3), "power_ci95": wilson(detect, runs),
                    "power_holm_alpha": round(detect_holm / runs, 3), "runs": runs,
                    "is_type1_check": d == 0.0}
        # minimum detectable delta at 80% power for this rho (nominal and Holm-alpha)
        for tag, fld in (("", "power"), ("_holm", "power_holm_alpha")):
            mdd = next((d for d in DELTAS if d > 0 and
                        res[cname].get(f"rho{rho}_d{d}", {}).get(fld, 0) >= 0.8), None)
            res[cname][f"rho{rho}_min_detectable_80{tag}"] = mdd
        t1 = res[cname].get(f"rho{rho}_d0.0", {}).get("power")
        print(cname, "rho", rho, "typeI:", t1, "MDD80:", res[cname][f"rho{rho}_min_detectable_80"],
              "MDD80_holm:", res[cname][f"rho{rho}_min_detectable_80_holm"], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
