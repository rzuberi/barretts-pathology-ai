"""Two papers, agent HTS (docs/two_papers_HTS.md): do only some tiles drive the artefact-free H&E score?  Step 1 of 3, CPU only, no slide pixels.
Exports the embeddings of every sample's kept (artefact-free) tiles, using exactly the op_fig3_tiles.py flag and keep rule (its module body is executed
with TASK=none, so flags / keep_of / S / BI are the verified Fig. 3 ones), into one float32 matrix for tp_hts_fit.R:
  killcoyne_mm/two_papers_HTS/emb_kept.f32   column-major (tiles x 1536; tile rows grouped by sample in samples.csv order)
  killcoyne_mm/two_papers_HTS/kept_index.csv  sample, start (0-based), n_kept           (row-level, cluster only)
Check: the mean of a sample's exported tiles equals the clean-tile feature row used by the stored refit (two_papers_ART/feats/clean.csv.gz)."""
import os, json, numpy as np, pandas as pd
PP = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"
os.environ["TASK"] = "none"; ns = {"__file__": PP + "/op_fig3_tiles.py", "__name__": "hts_fig3"}
exec(compile(open(PP + "/op_fig3_tiles.py").read(), PP + "/op_fig3_tiles.py", "exec"), ns)
M, A, S, BI, flags, keep_of = ns["M"], ns["A"], ns["S"], ns["BI"], ns["flags"], ns["keep_of"]
W = M + "/two_papers_HTS"; os.makedirs(W, exist_ok=True)
Q = pd.read_parquet(A + "/qc_all.parquet"); F, LT = flags(Q); Q = pd.concat([Q, F], axis=1); QI = Q.groupby("sample").indices
CL = pd.read_csv(A + "/feats/clean.csv.gz", dtype={"Sample": str}).set_index("Sample")
ids = list(S.index); blocks, idx, start, dmax, dt = [], [], 0, 0.0, set()
for s in ids:
    ii = np.sort(QI[s]); q = Q.iloc[ii]; E_ = np.load(BI.loc[s, "npz_path"])["embeddings"]; dt.add(str(E_.dtype)); assert len(E_) == len(q)
    assert E_.dtype in (np.float16, np.float32), E_.dtype   # float32 export is then exact
    keep = keep_of(q.art.values.copy(), q); Ek = E_[keep].astype(np.float32); blocks.append(Ek)
    dmax = max(dmax, float(np.abs(Ek.astype(np.float64).mean(0) - CL.loc[s].values).max()))
    idx.append({"sample": s, "start": start, "n_kept": int(keep.sum())}); start += int(keep.sum())
E = np.vstack(blocks); E.T.tofile(W + "/emb_kept.f32")   # E.T in C order = E in column-major
pd.DataFrame(idx).to_csv(W + "/kept_index.csv", index=False)
I = pd.DataFrame(idx); chk = {"samples": len(ids), "kept_tiles": int(len(E)), "dims": int(E.shape[1]), "npz_dtypes": sorted(dt),
                              "kept_mean_vs_clean_feats_max_abs": dmax, "n_kept_min": int(I.n_kept.min()), "n_kept_lt64": int((I.n_kept < 64).sum())}
json.dump(chk, open(W + "/prep_check.json", "w"), indent=1); print("HTS PREP DONE", json.dumps(chk), flush=True)
assert dmax < 1e-6, chk
