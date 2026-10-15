"""Paper plan item 7 montages (pre-specified @ db236a0): for the 20 pre-drawn rows (feasibility/paper_plan/montage_rows.csv),
top-16 attention tiles per model (image_only, intermediate_fusion, coattention_fusion) side by side, read from the .ndpi at
the release tile level using the npz coords. Files named by opaque montage id (no outcome). Output:
results/paper_plan/figs/attention/<montage_id>.png ; manifest with outcomes stays on the cluster (montage_manifest_SECRET.csv)."""
import os, numpy as np, pandas as pd, openslide
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; L = T + "/feasibility/paper_plan/latent"; OUT = T + "/results/paper_plan/figs/attention"; os.makedirs(OUT, exist_ok=True)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); ui = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str).set_index("sample_id"); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID")
ids = list(np.load(L + "/ids.npy")); rows = pd.read_csv(T + "/feasibility/paper_plan/montage_rows.csv", dtype=str)
ATT = {fam: {k: np.load(f"{L}/attn_{fam}_fold{k}.npy") for k in range(1, 6)} for fam in ["image_only", "intermediate_fusion", "coattention_fusion"]}
manifest = []
for r in rows.itertuples():
    sid = r.sample_id; i = ids.index(sid); k = int(man.fold_id_rep01[sid]); z = np.load(ui.npz_path[sid], allow_pickle=False); coords = z["coords_level"]; lvl = int(z["level"]); ts = int(z["tile_size"])
    sp = coh.ImageAbsPath[sid].replace("/scratchc/fmlab", "/mnt/scratche/fast/fmlab"); sl = openslide.OpenSlide(sp); ds = sl.level_downsamples[lvl]
    tl = np.clip(coords - ts // 2, 0, np.array(sl.level_dimensions[lvl]) - ts)   # coords_level are tile centres: top-left = centre - ts/2, clamped (extraction rule)
    fig, axes = plt.subplots(3, 16, figsize=(24, 5))
    for row_i, fam in enumerate(["image_only", "intermediate_fusion", "coattention_fusion"]):
        a = ATT[fam][k][i]; top = np.argsort(a)[::-1][:16]
        for col, t in enumerate(top):
            x, y = [int(v) for v in tl[t]]; img = sl.read_region((int(x * ds), int(y * ds)), lvl, (ts, ts)).convert("RGB"); ax = axes[row_i, col]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{a[t]:.3f}", fontsize=6)
        axes[row_i, 0].set_ylabel(fam.replace("_fusion", ""), fontsize=8)
    fig.suptitle(f"{r.montage_id}: top-16 attention tiles per model (attention weight above each tile; level {lvl}, {ts}px)", fontsize=9); fig.tight_layout(); fig.savefig(f"{OUT}/{r.montage_id}.png", dpi=110); plt.close(fig)
    manifest.append({"montage_id": r.montage_id, "sample_id": sid, "fold": k, "level": lvl, "tile_px": ts, "file": f"results/paper_plan/figs/attention/{r.montage_id}.png"}); print(r.montage_id, flush=True)
pd.DataFrame(manifest).to_csv(T + "/feasibility/paper_plan/montage_files.csv", index=False); print("montages done", len(manifest))
