"""2.44 (Astra A13): hardening of the pan-cancer jury validation + juror-count
replay.

(a) Per study: confusion matrix (jury majority vs registry grade), balanced
    accuracy, macro-F1, coverage, majority-class baseline agreement, Wilson
    interval on two-tier agreement; BLCA tier-mapping audit (raw registry
    values vs jury outputs) to explain the 0.58 -> 0.99 exact-to-two-tier jump.
(b) ERIN juror-count replay on SAVED votes: the deployed 8-juror majority rule
    vs the 5-juror subset used for TCGA — label agreement and eligibility
    (jury_frac >= 0.75) Jaccard, so the TCGA validation's procedure is shown
    to be a faithful proxy for the ERIN one.
"""
import glob, json, os
from collections import Counter
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
PREP = T + "/data/pancancer"
OUT = os.environ.get("OUTDIR", ".")
VALID = {"G1", "G2", "G3", "G4", "HIGH", "LOW"}
STUDIES = ["esca_tcga", "stad_tcga", "kirc_tcga", "blca_tcga"]
NUMMAP = {"G1": "G1", "G2": "G2", "G3": "G3", "G4": "G4",
          "HIGH GRADE": "HIGH", "LOW GRADE": "LOW", "HIGH": "HIGH", "LOW": "LOW"}
TWO_TIER = {"G1": "LOW", "G2": "LOW", "G3": "HIGH", "G4": "HIGH", "HIGH": "HIGH", "LOW": "LOW"}
FIVE = ["gemma3_12b", "gemma3_27b", "phi4_14b", "qwen3_14b", "qwen3_32b"]

def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]

res = {}
# ---------------- (a) pan-cancer ----------------
votes = {}
for f in glob.glob(T + "/feasibility/runs/pancancer_jury_*/output/llm_grades_*.csv"):
    model = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]
    d = pd.read_csv(f, dtype=str); d = d[d["llm_grade"].isin(VALID)]
    votes.setdefault(model, {}).update(dict(zip(d["CaseName"], d["llm_grade"])))
maj, frac = {}, {}
for cid in set.union(*(set(v) for v in votes.values())):
    vs = [v[cid] for v in votes.values() if cid in v]
    if len(vs) < 4: continue
    top = max(set(vs), key=vs.count); maj[cid] = top; frac[cid] = vs.count(top) / len(vs)
res["pancancer"] = {"_jurors": sorted(votes), "n_graded": len(maj)}
for st in STUDIES:
    cp, rp = os.path.join(PREP, st + "_clinical.txt"), os.path.join(PREP, st + "_pan_can_atlas_2018_reports.csv")
    if not (os.path.exists(cp) and os.path.exists(rp)): continue
    clin = pd.read_csv(cp, sep="\t", comment="#")
    gcol = next((c for c in clin.columns if c.upper() in ("GRADE", "NEOPLASM_HISTOLOGIC_GRADE", "TUMOR_GRADE")), None)
    if gcol is None: continue
    raw = clin[gcol].astype(str).str.upper().str.strip()
    clin["_g"] = raw.map(lambda v: NUMMAP.get(v, v if v in VALID else None))
    truth = dict(zip(clin["PATIENT_ID"], clin["_g"]))
    reps = pd.read_csv(rp, dtype=str)
    all_cases = [c for c in reps["CaseName"] if truth.get(c) in VALID]
    rows = [(c, maj[c], truth[c]) for c in all_cases if c in maj]
    if len(rows) < 30: continue
    yj = [TWO_TIER[r[1]] for r in rows]; yt = [TWO_TIER[r[2]] for r in rows]
    ej = [r[1] for r in rows]; et = [r[2] for r in rows]
    labs = sorted(set(et) | set(ej))
    cm = confusion_matrix(et, ej, labels=labs)
    majority_class = Counter(yt).most_common(1)[0][0]
    agree = sum(a == b for a, b in zip(yj, yt))
    res["pancancer"][st] = {
        "n_with_registry_grade": len(all_cases), "n_jury_graded": len(rows),
        "coverage": round(len(rows) / len(all_cases), 4),
        "registry_value_dist": dict(Counter(et)), "jury_value_dist": dict(Counter(ej)),
        "registry_raw_values": dict(Counter(raw[raw.isin(list(NUMMAP) + list(VALID))]).most_common(8)),
        "exact_confusion": {"labels": labs, "matrix": cm.tolist()},
        "two_tier": {"agreement": round(agree / len(rows), 4),
                     "wilson95": wilson(agree, len(rows)),
                     "majority_class": majority_class,
                     "majority_class_baseline_agreement": round(float(np.mean([t == majority_class for t in yt])), 4),
                     "balanced_accuracy": round(float(balanced_accuracy_score(yt, yj)), 4),
                     "macro_f1": round(float(f1_score(yt, yj, average="macro")), 4),
                     "minority_class_recall": round(float(np.mean([j == t for j, t in zip(yj, yt) if t != majority_class])), 4)
                     if any(t != majority_class for t in yt) else None},
        "exact": {"agreement": round(float(np.mean([a == b for a, b in zip(ej, et)])), 4),
                  "macro_f1": round(float(f1_score(et, ej, average="macro")), 4)},
        "mapping_note": "exact agreement compares raw values; registry may code HIGH/LOW while jury emits G1-G4 "
                        "(or vice versa) — see registry_value_dist vs jury_value_dist"}
    print(st, res["pancancer"][st]["two_tier"], flush=True)

# ---------------- (b) juror-count replay: 8-juror deployed rule vs 5-juror TCGA subset ----------------
# ERIN per-model votes live in labeller/llm_full/ (accession CaseNames);
# feasibility/runs/jury_full_* is the SEPARATE Barrett's-DB corpus jury (integer ids) — replayed too, labelled as such.
def replay(pattern, tag):
    ev = {}
    for f in glob.glob(pattern):
        model = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]
        d = pd.read_csv(f, dtype=str)
        d = d[d["llm_grade"].isin(["NDBE", "IND", "LGD", "HGD", "CANCER"])]
        ev.setdefault(model, {}).update(dict(zip(d["CaseName"], d["llm_grade"])))
    if not ev: return {"error": "no vote files for " + pattern}
    def rule(models):
        lab, fr = {}, {}
        common = set.intersection(*(set(ev[m]) for m in models))
        for c in common:
            vs = [ev[m][c] for m in models]
            top = max(set(vs), key=vs.count); lab[c] = top; fr[c] = vs.count(top) / len(vs)
        return lab, fr
    eight = sorted(ev); five = [m for m in FIVE if m in ev]
    l8, f8 = rule(eight); l5, f5 = rule(five)
    common = sorted(set(l8) & set(l5))
    e8 = {c for c in common if f8[c] >= 0.75}; e5 = {c for c in common if f5[c] >= 0.75}
    out = {"corpus": tag, "all_jurors": eight, "five_jurors": five, "n_common_reports": len(common),
           "label_agreement_all_vs_5": round(float(np.mean([l8[c] == l5[c] for c in common])), 4),
           "eligible_all": len(e8), "eligible_5": len(e5),
           "eligible_jaccard": round(len(e8 & e5) / max(len(e8 | e5), 1), 4),
           "label_agreement_on_both_eligible": round(float(np.mean([l8[c] == l5[c] for c in e8 & e5])), 4) if e8 & e5 else None,
           "label_dist_all": dict(Counter(l8[c] for c in e8)), "label_dist_5": dict(Counter(l5[c] for c in e5))}
    print("replay", tag, out, flush=True)
    return out
res["erin_juror_replay"] = replay(T + "/labeller/llm_full/llm_grades_*.csv", "ERIN (7,149 reports)")
res["barretts_db_juror_replay"] = replay(T + "/feasibility/runs/jury_full_*/output/llm_grades_*.csv",
                                         "Barrett's-DB export corpus (13,645 reports; NOT ERIN)")
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
