"""2.33 analysis (after pan-cancer jury shards): jury grade vs cBioPortal
structured GRADE per study — the human-independent external check on
report-derived supervision, across cancer types the jury never saw.
"""
import glob, json, os
import pandas as pd

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
PREP = T + "/data/pancancer"
OUT = os.environ.get("OUTDIR", ".")
VALID = {"G1", "G2", "G3", "G4", "HIGH", "LOW"}
# provisional studies carry GRADE; pan_can_atlas files do not
STUDIES = ["esca_tcga", "stad_tcga", "kirc_tcga", "blca_tcga"]
NUMMAP = {"G1": "G1", "G2": "G2", "G3": "G3", "G4": "G4",
          "HIGH GRADE": "HIGH", "LOW GRADE": "LOW", "HIGH": "HIGH", "LOW": "LOW"}
TWO_TIER = {"G1": "LOW", "G2": "LOW", "G3": "HIGH", "G4": "HIGH",
            "HIGH": "HIGH", "LOW": "LOW"}

votes = {}
for f in glob.glob(T + "/feasibility/runs/pancancer_jury_*/output/llm_grades_*.csv"):
    model = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]
    d = pd.read_csv(f, dtype=str)
    d = d[d["llm_grade"].isin(VALID)]
    votes.setdefault(model, {}).update(dict(zip(d["CaseName"], d["llm_grade"])))
print("jurors:", {k: len(v) for k, v in votes.items()}, flush=True)

maj, frac = {}, {}
for cid in set.union(*(set(v) for v in votes.values())):
    vs = [v[cid] for v in votes.values() if cid in v]
    if len(vs) < 4: continue
    top = max(set(vs), key=vs.count)
    maj[cid] = top; frac[cid] = vs.count(top) / len(vs)

res = {"_meta": {"jurors": sorted(votes), "n_graded": len(maj)}}
for st in STUDIES:
    cp = os.path.join(PREP, st + "_clinical.txt")
    rp = os.path.join(PREP, st + "_pan_can_atlas_2018_reports.csv")
    if not (os.path.exists(cp) and os.path.exists(rp)): continue
    clin = pd.read_csv(cp, sep="\t", comment="#")
    gcol = next((c for c in clin.columns if c.upper() in ("GRADE", "NEOPLASM_HISTOLOGIC_GRADE",
                                                          "TUMOR_GRADE")), None)
    if gcol is None:
        res[st] = {"skip": f"no grade column; cols={list(clin.columns)[:10]}"}; continue
    clin["_g"] = clin[gcol].astype(str).str.upper().str.strip().map(
        lambda v: NUMMAP.get(v, v if v in VALID else None))
    truth = dict(zip(clin["PATIENT_ID"], clin["_g"]))
    reps = pd.read_csv(rp, dtype=str)
    rows = [(c, maj[c], frac[c], truth.get(c)) for c in reps["CaseName"]
            if c in maj and truth.get(c) in VALID]
    if len(rows) < 30:
        res[st] = {"skip": f"only {len(rows)} matched"}; continue
    df = pd.DataFrame(rows, columns=["c", "jury", "frac", "truth"])
    df["jury2"], df["truth2"] = df["jury"].map(TWO_TIER), df["truth"].map(TWO_TIER)
    conf = df[df["frac"] >= 0.8]
    res[st] = {"n": len(df),
               "exact_agreement": round(float((df["jury"] == df["truth"]).mean()), 4),
               "two_tier_agreement": round(float((df["jury2"] == df["truth2"]).mean()), 4),
               "n_confident": len(conf),
               "two_tier_agreement_confident": round(float(
                   (conf["jury2"] == conf["truth2"]).mean()), 4) if len(conf) else None}
    print(st, res[st], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")
