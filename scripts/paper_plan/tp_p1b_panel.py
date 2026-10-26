"""Two papers, agent P1B (docs/two_papers_P1B.md), item 10: a blinded pathologist review panel of the tiles that push the L-IMG score most up and most down.
Tile contribution u_t = w . z(e_t) of the outer (held-out) fit of each pre-event sample, mean over the 10 repeats (E1 tile scores, killcoyne_mm/explore_E1/u,
checked there against the stored logits). Single-slide samples only, so a bag row is the slide's stored tile. Tiles are read with the extraction rule of
extract_foundation_from_masks.py (coords_level = tile centres at level 2; top-left = centre - 112, clamped; as sr_tiles.py), here at level 1 for the same field.
Only CPU jobs open slides. Row-level files and the answer key stay on the cluster (killcoyne_mm/two_papers_P1B/panel/); montages carry numbers only.
TASK = pick | crop_<i>of<n> | montage"""
import os, sys, glob, json, numpy as np, pandas as pd
PP = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"; sys.path.insert(0, PP)
from tb_common import T, M, SM, n, ids, y, pat, PRE, load
TASK = os.environ["TASK"]; KD = T + "/feasibility/paper_plan/killcoyne"; PD_ = M + "/two_papers_P1B/panel"; os.makedirs(PD_ + "/crops", exist_ok=True)
remap = lambda p: p.replace("/scratchc/", "/mnt/scratche/fast/") if isinstance(p, str) and p.startswith("/scratchc/") else p
NPER = 40; NRES = 120; PERPAT = 2
if TASK == "pick":
    FO = load("their")[2]; bi = pd.read_csv(M + "/bag_index.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(ids)
    mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(set(ids))].drop_duplicates(["cnv", "npz"]); one = mp.groupby("cnv").npz.agg(list)
    rows = []
    for s in np.where(PRE)[0]:
        if bi.n_slides.iloc[s] != 1: continue
        U = np.load(f"{M}/explore_E1/u/{ids[s]}.npz")["U"].astype(float); u = np.mean([U[:, j, FO[s, j] - 1] for j in range(10)], 0)
        for t, v in enumerate(u): rows.append((s, t, v))
    D = pd.DataFrame(rows, columns=["s", "tile", "u"]); D["sample"] = ids[D.s]; D["patient"] = pat[D.s]; D["y"] = y[D.s]; D["grade"] = SM.Pathology.values[D.s]
    D["npz"] = D["sample"].map(lambda x: one[x][0])
    def take(order):
        out, pc, sc = [], {}, set()
        for r in D.iloc[order].itertuples():
            if pc.get(r.patient, 0) >= PERPAT or r.s in sc: continue
            out.append(r.Index); pc[r.patient] = pc.get(r.patient, 0) + 1; sc.add(r.s)
            if len(out) == NRES: break
        return out
    o = np.argsort(D.u.values); top = take(o[::-1]); bot = take(o)
    C = pd.concat([D.loc[top].assign(group="top", rank=np.arange(1, len(top) + 1)), D.loc[bot].assign(group="bottom", rank=np.arange(1, len(bot) + 1))]).reset_index(drop=True)
    C["cid"] = [f"c{i:04d}" for i in range(len(C))]; C.to_csv(PD_ + "/candidates_SECRET.csv", index=False)
    print("P1B PANEL pick", len(C), "tiles scored", len(D), "u range", round(D.u.min(), 3), round(D.u.max(), 3), flush=True)
elif TASK.startswith("crop_"):
    import openslide
    from PIL import Image
    i, k = map(int, TASK[5:].split("of")); C = pd.read_csv(PD_ + "/candidates_SECRET.csv", dtype=str).iloc[i::k]; out = []
    for r in C.itertuples():
        f = f"{PD_}/crops/{r.cid}.png"
        with np.load(r.npz, allow_pickle=True) as z: c = np.asarray(z["coords_level"])[int(r.tile)]; lv = int(z["level"]); ts = int(z["tile_size"]); sp = remap(str(z["slide_path"]))
        s = openslide.OpenSlide(sp); W, H = s.level_dimensions[lv]; ds = float(s.level_downsamples[lv]); half = ts // 2
        x0 = max(0, min(int(c[0] - half), W - ts)); y0 = max(0, min(int(c[1] - half), H - ts)); X0, Y0 = int(round(x0 * ds)), int(round(y0 * ds))
        l1 = 1 if s.level_count > 1 else 0; d1 = float(s.level_downsamples[l1]); sz = int(round(ts * ds / d1))
        im = s.read_region((X0, Y0), l1, (sz, sz)).convert("RGB"); mpp = float(s.properties.get("openslide.mpp-x", "nan")); s.close()
        a = np.asarray(im).astype(np.float32); wh = (a > 220).all(-1); mx = a.max(-1); mn = a.min(-1); sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
        rr, gg, bb = a[..., 0], a[..., 1], a[..., 2]; dd = np.maximum(mx - mn, 1e-6)
        hue = np.where(mx == rr, (60 * (gg - bb) / dd) % 360, np.where(mx == gg, 60 * (bb - rr) / dd + 120, 60 * (rr - gg) / dd + 240)); ink = (hue >= 80) & (hue <= 250) & (sat > 0.25) & ~wh
        im.save(f); out.append({"cid": r.cid, "white": float(wh.mean()), "ink": float(ink.mean()), "dark": float((a.mean(-1) < 50).mean()), "px": sz, "level": l1, "um_per_px": mpp * d1, "field_um": mpp * d1 * sz})
    pd.DataFrame(out).to_csv(f"{PD_}/crop_qc_{i:02d}.csv", index=False); print("P1B PANEL crop", TASK, len(out), flush=True)
elif TASK == "montage":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    C = pd.read_csv(PD_ + "/candidates_SECRET.csv", dtype={"sample": str}); Q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(PD_ + "/crop_qc_*.csv"))]); C = C.merge(Q, on="cid"); assert len(C) == 2 * NRES
    def tissue(cid):   # share of stained pixels (HSV saturation > 0.15, not white): rejects blank, out-of-focus edge and glass tiles
        a = np.asarray(Image.open(f"{PD_}/crops/{cid}.png").convert("RGB")).astype(np.float32); mx = a.max(-1); mn = a.min(-1); sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
        return float(((sat > 0.15) & ~(a > 220).all(-1)).mean())
    C["tissue"] = [tissue(c) for c in C.cid]
    C["ok"] = (C.white < 0.30) & (C.ink < 0.02) & (C.dark < 0.30) & (C.tissue >= 0.15)
    S = pd.concat([C[(C.group == g) & C.ok].sort_values("rank").head(NPER) for g in ("top", "bottom")]); assert (S.group.value_counts() == NPER).all(), S.group.value_counts()
    rng = np.random.RandomState(20261021); S = S.iloc[rng.permutation(len(S))].reset_index(drop=True); S["code"] = np.arange(1, len(S) + 1)
    os.makedirs(PD_ + "/montages", exist_ok=True); per = 16; letters = "ABCDEFGH"
    for m in range(int(np.ceil(len(S) / per))):
        G = S.iloc[m * per:(m + 1) * per]; fig, axs = plt.subplots(4, 4, figsize=(9, 9.6), dpi=200)
        for ax, r in zip(axs.ravel(), list(G.itertuples()) + [None] * (per - len(G))):
            ax.set_xticks([]); ax.set_yticks([])
            if r is None: ax.axis("off"); continue
            ax.imshow(Image.open(f"{PD_}/crops/{r.cid}.png")); ax.set_title(f"{r.code}", fontsize=11, pad=3)
        fig.suptitle(f"Panel {letters[m]}: tiles {G.code.min()}-{G.code.max()} (each square about {S.field_um.median():.0f} µm across)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f"{PD_}/montages/panel_{letters[m]}.png", dpi=200); plt.close(fig)
    S[["code", "group", "rank", "u", "y", "grade", "sample", "patient", "tile", "white", "ink", "tissue"]].to_csv(PD_ + "/ANSWER_KEY_SECRET.csv", index=False)
    summ = {"n_tiles": int(len(S)), "per_group": NPER, "max_tiles_per_patient_per_group": PERPAT, "candidates_failing_qc": {g: int((~C.ok & (C.group == g)).sum()) for g in ("top", "bottom")},
            "failing_qc_within_ranks_used": {g: int((~C.ok & (C.group == g) & (C["rank"] <= S[S.group == g]["rank"].max())).sum()) for g in ("top", "bottom")},
            "artefact_share_tissue_lt_0.15_among_120_candidates": {g: round(float((C[C.group == g].tissue < 0.15).mean()), 3) for g in ("top", "bottom")},
            "artefact_share_tissue_lt_0.15_among_first_40_candidates": {g: round(float((C[(C.group == g) & (C["rank"] <= 40)].tissue < 0.15).mean()), 3) for g in ("top", "bottom")},
            "field_um_median": round(float(S.field_um.median()), 1), "n_montages": int(np.ceil(len(S) / per)), "patients_represented": int(S.patient.nunique())}
    json.dump(summ, open(PD_ + "/montage_summary.json", "w"), indent=1); print("P1B PANEL montage", json.dumps(summ), flush=True)
