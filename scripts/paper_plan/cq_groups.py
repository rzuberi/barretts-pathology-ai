"""CNV-quiet morphology (docs/paper_cnvquiet_morphology.md), step 1: groups (a), (b), (c) from stored outputs only; no model is fitted.
(a) newly caught cases and (b) cases caught by both, L-LATE vs L-CNV, their matrix, union over the 3- and 5-year horizons, from the Q2 per-sample
list of ha_q2.py (afa0278); (b) excludes any sample in (a). (c) non-progressor pre-event samples matched to (a): exact on scanner and on grade class
(NDBE vs ID/LGD), nearest raw cx (package block), 3 per (a) sample, without replacement, 3 round-robin rounds over (a) in a seeded order
(numpy default_rng(20261012)); ties in |delta cx| broken by a seeded random key. Checks the per-horizon counts against docs/paper_horizon_q2_all_horizons.md.
Row-level outputs (cluster only): cnvquiet_morphology/groups.csv, slides.csv. Aggregates: results/paper_final/cnvquiet_morphology/groups.json."""
import os, json, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; K = T + "/feasibility/paper_plan/killcoyne"; M = K + "_mm"; W = M + "/cnvquiet_morphology"
R = T + "/results/paper_final/cnvquiet_morphology"; os.makedirs(W, exist_ok=True); os.makedirs(R, exist_ok=True)
SEED, K_MATCH = 20261012, 3
sa = pd.read_csv(M + "/horizons/samples.csv"); sd = pd.read_csv(M + "/horizons/slide_desc.csv")
cx = pd.read_csv(M + "/cnv_pkg_C.csv", usecols=["sample_id", "cx"]).rename(columns={"sample_id": "Sample"})
sc = pd.read_csv(T + "/models/killcoyne_frozen_pkg_v1/cnv_scaling_setC.csv"); r = sc[sc.kind == "cx"].iloc[0]
d = sa.merge(sd, on="Sample").merge(cx, on="Sample"); d["cx_raw"] = d.cx * r["sd"] + r["mean"]; d["grade_cls"] = np.where(d.Pathology == "NDBE", "NDBE", "ID/LGD")
assert len(d) == 676 and d.Patient.nunique() == 80
q = pd.read_csv(M + "/horizon_answers/q2_patient_list.csv"); q = q[(q.src == "their") & (q.comparison == "L-LATE vs L-CNV")]
EXPECT = {(3, "captured_cases_CNVneg_bestpos"): (11, 6), (5, "captured_cases_CNVneg_bestpos"): (13, 8), (3, "cases_both_pos"): (73, 26), (5, "cases_both_pos"): (91, 28)}
chk = {}
for (h, g), (ns, npat) in EXPECT.items():
    s = q[(q.horizon == h) & (q.group == g)]; got = (s.Sample.nunique(), s.study_number.nunique()); chk[f"{h}y {g}"] = {"expected": [ns, npat], "got": list(got), "ok": got == (ns, npat)}
    assert got == (ns, npat), (h, g, got)
q35 = q[q.horizon.isin([3, 5])]
A = sorted(set(q35[q35.group == "captured_cases_CNVneg_bestpos"].Sample)); B = sorted(set(q35[q35.group == "cases_both_pos"].Sample) - set(A))
hz = q35.groupby("Sample").apply(lambda g: ";".join(f"{h}y:{gr}" for h, gr in sorted(zip(g.horizon, g.group)))).rename("horizons")
assert d.set_index("Sample").loc[A, "y"].eq(1).all() and d.set_index("Sample").loc[A, "pre"].all()
# (c) matching
rng = np.random.default_rng(SEED); a_df = d.set_index("Sample").loc[A]; order = list(rng.permutation(A))
pool = d[(d.y == 0) & d.pre].copy(); pool["tie"] = rng.random(len(pool)); pool = pool.set_index("Sample"); used, match = set(), []
for rnd in range(K_MATCH):
    for s in order:
        ar = a_df.loc[s]; c = pool[(pool.scanner == ar.scanner) & (pool.grade_cls == ar.grade_cls) & ~pool.index.isin(used)].copy()
        assert len(c), ("no candidate", s); c["dcx"] = (c.cx_raw - ar.cx_raw).abs(); best = c.sort_values(["dcx", "tie"]).index[0]
        used.add(best); match.append({"Sample": best, "matched_to": s, "round": rnd + 1, "dcx": float(c.loc[best, "dcx"])})
mt = pd.DataFrame(match); C = list(mt.Sample)
G = pd.concat([pd.DataFrame({"Sample": A, "group": "a"}), pd.DataFrame({"Sample": B, "group": "b"}), pd.DataFrame({"Sample": C, "group": "c"})])
G = G.merge(d[["Sample", "Patient", "study_number", "y", "Pathology", "grade_cls", "scanner", "cx_raw", "tissue_tiles", "n_slides", "time", "event"]], on="Sample", how="left")
G = G.merge(hz, left_on="Sample", right_index=True, how="left").merge(mt[["Sample", "matched_to", "dcx"]], on="Sample", how="left")
G["a_3y"] = (G.group == "a") & G.horizons.fillna("").str.contains("3y:captured"); G.to_csv(W + "/groups.csv", index=False)
mp = pd.read_csv(K + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(G.Sample)].drop_duplicates(["cnv", "npz"])
mp["slide_key"] = mp.npz.map(lambda p: os.path.splitext(os.path.basename(p))[0]); sl = mp.rename(columns={"cnv": "Sample"})[["slide_key", "npz", "Sample"]]
sl = sl.merge(G[["Sample", "group"]], on="Sample"); sl.to_csv(W + "/slides.csv", index=False)
assert set(sl.Sample) == set(G.Sample) and sl.groupby("Sample").size().reindex(G.Sample).values.tolist() == G.n_slides.tolist()
def summ(g):
    return {"samples": int(len(g)), "patients": int(g.Patient.nunique()), "slides": int(g.n_slides.sum()), "cx_raw_mean": round(float(g.cx_raw.mean()), 2), "cx_raw_sd": round(float(g.cx_raw.std()), 2),
            "cx_raw_median": float(g.cx_raw.median()), "grade": g.Pathology.value_counts().to_dict(), "scanner": g.scanner.value_counts().to_dict(),
            "tissue_tiles_median": float(g.tissue_tiles.median()), "tissue_tiles_mean": round(float(g.tissue_tiles.mean()), 1), "samples_per_patient_max": int(g.groupby("Patient").size().max())}
out = {"seed": SEED, "k_match": K_MATCH, "count_checks": chk, "groups": {k: summ(G[G.group == k]) for k in "abc"},
       "a_in_3y_newly_caught": int(G[(G.group == "a")].horizons.str.contains("3y:captured").sum()),
       "a_patients_also_in_b": int(len(set(G[G.group == "a"].Patient) & set(G[G.group == "b"].Patient))),
       "c_abs_dcx": {"mean": round(float(mt.dcx.mean()), 2), "max": float(mt.dcx.max()), "n_exact": int((mt.dcx == 0).sum())}}
json.dump(out, open(R + "/groups.json", "w"), indent=1); print(json.dumps(out, indent=1)); print("CQ GROUPS DONE")
