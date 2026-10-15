"""Part 2 step 2: merge pack shards -> pack/index.json; merge QC shards -> qc/qc.csv and apply the
pre-fixed exclusion rule; write qc/exclusions.csv + qc/summary.json. Flags the >5% stop rule."""
import glob, json, os
import numpy as np, pandas as pd
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun"; P, Q = A + "/pack", A + "/qc"
s = pd.read_csv(A + "/slides.csv", dtype={"idx": int})
shards = sorted((json.load(open(f)) for f in glob.glob(P + "/meta_shard*.json")), key=lambda d: d["shard"])
meta = [m for d in shards for m in d["rows"]]; mi = pd.DataFrame(meta).sort_values("idx"); n_err = int((mi.n == 0).sum())
assert len(mi) == len(s) and all(os.path.exists(f"{P}/chunk_{d['shard']:02d}.npy") for d in shards), f"pack incomplete: {len(mi)}/{len(s)}"
json.dump({"n_slides": len(s), "cap": 1200, "dim": 1536, "dtype": "float16", "n_tiles": mi.n.tolist(), "n_total": mi.n_total.fillna(0).astype(int).tolist(),
           "chunks": [{"file": f"chunk_{d['shard']:02d}.npy", "lo": d["lo"], "hi": d["hi"]} for d in shards],
           "pack_errors": n_err, "capped_slides": int((mi.n_total > 1200).sum())}, open(P + "/index.json", "w"))
qc = pd.concat([pd.read_csv(f) for f in glob.glob(Q + "/qc_shard*.csv")]).sort_values("idx")
assert len(qc) == len(s), f"qc incomplete: {len(qc)}/{len(s)}"
qc = qc.merge(s[["idx", "file_uuid", "CaseName", "section", "worst_grade"]], on="idx")
thr_lap = float(np.nanpercentile(qc.lapvar_median, 1))
qc["excl_reason"] = ""
qc.loc[qc.kept_tiles.fillna(0) < 50, "excl_reason"] = "kept_tiles<50"
qc.loc[(qc.white_frac > 0.98) & (qc.excl_reason == ""), "excl_reason"] = "white_frac>0.98"
qc.loc[(qc.lapvar_median < thr_lap) & (qc.excl_reason == ""), "excl_reason"] = f"lapvar<p1({thr_lap:.2f})"
qc.loc[(qc.error.fillna("") != "") & (qc.excl_reason == ""), "excl_reason"] = "qc_error:" + qc.error.fillna("")
qc.loc[qc.idx.isin(mi[mi.n == 0].idx) & (qc.excl_reason == ""), "excl_reason"] = "pack_error"
qc.to_csv(Q + "/qc.csv", index=False); ex = qc[qc.excl_reason != ""]; ex.to_csv(Q + "/exclusions.csv", index=False)
summ = {"slides": len(qc), "excluded": int(len(ex)), "excluded_frac": round(len(ex) / len(qc), 4), "by_reason": ex.excl_reason.value_counts().to_dict(),
        "excluded_by_grade": ex.worst_grade.fillna("(none)").value_counts().to_dict(), "lapvar_p1_threshold": thr_lap,
        "kept_tiles_median": float(qc.kept_tiles.median()), "kept_tiles_min": float(qc.kept_tiles.min()),
        "STOP_RULE_gt5pct": bool(len(ex) / len(qc) > 0.05), "pack_errors": n_err}
json.dump(summ, open(Q + "/summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
for f in glob.glob(A + "/**/*", recursive=True):
    if os.path.isfile(f): os.chmod(f, 0o600)
