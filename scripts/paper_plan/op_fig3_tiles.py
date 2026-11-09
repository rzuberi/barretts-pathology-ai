"""One-paper Fig. 3a-b: per-tile contributions of the artefact-free H&E model (clean-tile L-IMG, agent ART). CPU only; the only step that reads
slide pixels (TASK=make). The model is an elastic net on the mean embedding of a sample's kept tiles, so with raw-scale coefficients (c0, w) of the
outer fit that held the sample out, the sample's log-odds = mean over kept tiles of (c0 + w . e_t). Tile score u_t = c0 + w . e_t, averaged over
the 10 cross-validation repeats (each repeat uses the fold model that held the sample out). Coefficients: tp_art_refit.R FEAT=clean_coef COEF=1
(clean_coef.csv.gz = symlink to clean.csv.gz), checked against the stored clean refit predictions.
TASK=u     tile scores for every pre-event sample -> killcoyne_mm/one_paper_fig3/u_tiles.parquet (row-level, cluster only) + smoke numbers
TASK=make  examples + montage -> killcoyne_mm/one_paper_fig3/out/ (generic names, no IDs): ex_p_thumb.png / ex_np_thumb.png (tiled tissue region,
           label never included), ex_p_tiles.csv / ex_np_tiles.csv (tile squares in thumbnail px, u, artefact flag), top_XX.png / bot_XX.png
           (224 px tiles at level 2), montage.csv (group, rank, u, class, grade); aggregates -> results/paper_final/one_paper/fig3_tiles.json
Example rule (fixed before any image was viewed): pre-event NDBE sample, one slide of 256 tiles, < 5% artefact tiles, not the Fig. 1b sample;
in each class, the sample whose held-out H&E probability (repeat mean) is closest to that class's median among these candidates.
Montage rule: kept (artefact-free) tiles of single-slide pre-event samples with tissue share >= 0.5; the 16 highest and 16 lowest u_t, at most
one tile per sample and two per patient (as P1B's panel)."""
import os, glob, json, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; A = M + "/two_papers_ART"
W0 = M + "/one_paper_fig3"; OD = W0 + "/out"; os.makedirs(OD, exist_ok=True); OUTR = T + "/results/paper_final/one_paper"; os.makedirs(OUTR, exist_ok=True)
FLAG = dict(tis_min=0.15, line_d=0.50, ink=0.02, blur_q=0.05)
assert "FLAG = dict(tis_min=0.15, line_d=0.50, ink=0.02, blur_q=0.05)" in open(T + "/scripts/paper_plan/tp_art_main.py").read()
def flags(Q):   # = op_fig1_example.flags = tp_art_main.flags
    F = pd.DataFrame(index=Q.index); F["bg"] = Q.tissue.values < FLAG["tis_min"]; F["edge"] = ~F.bg & (Q.line_d.values >= FLAG["line_d"])
    F["pen"] = Q.ink.values > FLAG["ink"]; ok = ~F.bg & np.isfinite(Q.lapvar_t.values); lt = float(np.nanquantile(Q.lapvar_t.values[ok], FLAG["blur_q"]))
    F["blur"] = ~F.bg & (np.nan_to_num(Q.lapvar_t.values, nan=0.0) < lt); F["art"] = F.bg | F.edge | F.pen | F.blur; return F, lt
def keep_of(bad, q, MINC=32):   # = tp_art_main feats rule (top-up to 32 with the least-artefactual flagged tiles)
    keep = ~bad; nc = int(keep.sum())
    if nc < MINC:
        order = np.lexsort((-np.nan_to_num(q.lapvar_t.values), -q.tissue.values)); add = [i for i in order if not keep[i]][:MINC - nc]; keep[add] = True
    return keep
remap = lambda p: p.replace("/scratchc/", "/mnt/scratche/fast/") if isinstance(p, str) and p.startswith("/scratchc/") else p
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
TASK = os.environ["TASK"]
S = pd.read_csv(M + "/horizons/samples.csv", dtype={"Sample": str, "Patient": str}).set_index("Sample")
BI = pd.read_csv(M + "/bag_index.csv", dtype={"sample_id": str}).set_index("sample_id")
RD = A + "/refit/clean_coef"; RS = A + "/refit/clean"
if TASK == "u":
    Q = pd.read_parquet(A + "/qc_all.parquet"); F, LT = flags(Q); Q = pd.concat([Q, F], axis=1); QI = Q.groupby("sample").indices
    CO = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(RD + "/coef_rep_*.csv.gz"))]); CO = CO[CO.inner_fold == 0]; assert len(CO) == 100
    wc = [c for c in CO.columns if c.startswith("w")]; Wm = {(int(r), int(k)): (c0, w) for r, k, c0, w in zip(CO.rep, CO.fold, CO.c0, CO[wc].values)}
    Oc = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(RD + "/outer_rep_*.csv"))])
    Os = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(RS + "/outer_rep_*.csv"))])
    mg = Oc.merge(Os, on=["Sample", "rep"], suffixes=("", "_s")); assert len(mg) == len(Os) == len(Oc)
    smoke = {"coef_refit_vs_stored_clean_max_abs_pred": float(np.abs(mg.pred - mg.pred_s).max()), "folds_equal": bool((mg.fold == mg.fold_s).all())}
    fold = Oc.pivot_table(index="Sample", columns="rep", values="fold"); pred = Oc.pivot_table(index="Sample", columns="rep", values="pred")
    pre = [s for s in S.index if bool(S.loc[s, "pre"])]; rows = []; dmax = 0.0
    for s in pre:
        ii = np.sort(QI[s]); q = Q.iloc[ii]; E_ = np.load(BI.loc[s, "npz_path"])["embeddings"].astype(np.float64); assert len(E_) == len(q)
        keep = keep_of(q.art.values.copy(), q); U = np.zeros((len(q), 10))
        for j, r in enumerate(range(1, 11)):
            c0, w = Wm[(r, int(fold.loc[s, r]))]; U[:, j] = c0 + E_ @ w
            dmax = max(dmax, abs(U[keep, j].mean() - lg(pred.loc[s, r])))
        rows.append(pd.DataFrame({"sample": s, "patient": S.loc[s, "Patient"], "y": int(S.loc[s, "y"]), "grade": S.loc[s, "Pathology"], "npz": q.npz.values,
                                  "tile": q.tile.values, "x0": q.x0.values, "y0": q.y0.values, "tissue": q.tissue.values, "art": q.art.values, "kept": keep,
                                  "bg": q.bg.values, "edge": q.edge.values, "pen": q.pen.values, "blur": q.blur.values, "u": U.mean(1),
                                  "p_sample": float(pred.loc[s].mean()), "nslides": int(BI.loc[s, "n_slides"])}))
    D = pd.concat(rows, ignore_index=True); D.to_parquet(W0 + "/u_tiles.parquet")
    smoke["tile_mean_vs_stored_logit_max_abs"] = dmax; smoke["pre_samples"] = len(pre); smoke["tiles"] = int(len(D))
    json.dump(smoke, open(W0 + "/smoke.json", "w"), indent=1); print("OP FIG3 TILES u DONE", json.dumps(smoke), flush=True)
    assert smoke["coef_refit_vs_stored_clean_max_abs_pred"] < 1e-6 and dmax < 1e-3, smoke
elif TASK == "make":
    import openslide
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    D = pd.read_parquet(W0 + "/u_tiles.parquet"); smoke = json.load(open(W0 + "/smoke.json"))
    f1 = open(M + "/one_paper_fig1/selection_SECRET.txt").read().split(":")[1].strip()
    g = D.groupby("sample"); Sx = pd.DataFrame({"n": g.size(), "art": g.art.mean(), "y": g.y.first(), "grade": g.grade.first(), "p": g.p_sample.first(),
                                                "nslides": g.nslides.first()})
    cand = Sx[(Sx.grade == "NDBE") & (Sx.nslides == 1) & (Sx.n == 256) & (Sx.art < 0.05) & (Sx.index != f1)]
    ex = {}
    for lab, yy in (("p", 1), ("np", 0)):
        c = cand[cand.y == yy].sort_index(); med = c.p.median(); s = (c.p - med).abs().idxmin(); ex[lab] = s
    open(W0 + "/selection_SECRET.txt", "w").write(json.dumps(ex) + "\n")
    info = {}
    for lab, s in ex.items():
        q = D[D["sample"] == s].sort_values("tile"); npz = q.npz.iloc[0]
        with np.load(npz, allow_pickle=True) as z: sp = remap(str(z["slide_path"])); lv = int(z["level"]); ts = int(z["tile_size"])
        sl = openslide.OpenSlide(sp); ds = float(sl.level_downsamples[lv]); mpp = float(sl.properties.get("openslide.mpp-x", "nan"))
        x0, y0 = q.x0.min() - ts, q.y0.min() - ts; x1, y1 = q.x0.max() + 2 * ts, q.y0.max() + 2 * ts
        Wl, Hl = sl.level_dimensions[lv]; x0, y0, x1, y1 = max(0, x0), max(0, y0), min(Wl, x1), min(Hl, y1)
        target = 2000.0; want = (x1 - x0) * ds / target; L2 = sl.get_best_level_for_downsample(want); d2 = float(sl.level_downsamples[L2])
        w2, h2 = int((x1 - x0) * ds / d2), int((y1 - y0) * ds / d2)
        im = sl.read_region((int(x0 * ds), int(y0 * ds)), L2, (w2, h2)).convert("RGB")
        sc = target / w2 if w2 > target else 1.0; im = im.resize((int(w2 * sc), int(h2 * sc)), Image.LANCZOS); im.save(f"{OD}/ex_{lab}_thumb.png"); sl.close()
        f = ds / d2 * sc
        pd.DataFrame({"x": (q.x0.values - x0) * f, "y": (q.y0.values - y0) * f, "w": ts * f, "u": q.u.values.round(4), "art": q.art.values,
                      "kept": q.kept.values}).to_csv(f"{OD}/ex_{lab}_tiles.csv", index=False)
        k = q[q.kept]; info[lab] = {"class": "progressor" if lab == "p" else "non-progressor", "grade": "NDBE", "tiles": int(len(q)), "kept": int(q.kept.sum()),
                                   "artefact": int(q.art.sum()), "p_heldout_repeat_mean": round(float(q.p_sample.iloc[0]), 3),
                                   "logodds_from_tiles": round(float(k.u.mean()), 3), "u_kept_min": round(float(k.u.min()), 2), "u_kept_max": round(float(k.u.max()), 2),
                                   "u_kept_median": round(float(k.u.median()), 2), "thumb_um_per_px": round(mpp * d2 / sc, 2), "thumb_size": list(im.size),
                                   "class_median_p_among_candidates": round(float(cand[cand.y == (1 if lab == "p" else 0)].p.median()), 3),
                                   "candidates": int((cand.y == (1 if lab == "p" else 0)).sum())}
    # montage
    K = D[D.kept & ~D.art & (D.nslides == 1) & (D.tissue >= 0.5)].copy()
    def take(order, nmax=16):
        out, pc, sc_ = [], {}, set()
        for i in order:
            r = K.loc[i]
            if r["sample"] in sc_ or pc.get(r.patient, 0) >= 2: continue
            out.append(i); sc_.add(r["sample"]); pc[r.patient] = pc.get(r.patient, 0) + 1
            if len(out) == nmax: break
        return out
    o = K.sort_values("u").index.values; top = take(o[::-1]); bot = take(o)
    mrows = []
    for grp, idx in (("top", top), ("bot", bot)):
        for rk, i in enumerate(idx):
            r = K.loc[i]
            with np.load(r.npz, allow_pickle=True) as z: sp = remap(str(z["slide_path"])); lv = int(z["level"]); ts = int(z["tile_size"])
            sl = openslide.OpenSlide(sp); ds = float(sl.level_downsamples[lv])
            sl.read_region((int(r.x0 * ds), int(r.y0 * ds)), lv, (ts, ts)).convert("RGB").save(f"{OD}/{grp}_{rk:02d}.png"); sl.close()
            mrows.append({"group": grp, "rank": rk, "u": round(float(r.u), 3), "class": "P" if r.y == 1 else "NP", "grade": r.grade})
    MC = pd.DataFrame(mrows); MC.to_csv(OD + "/montage.csv", index=False)
    Kd = K.groupby("y").u
    res = {"note": __doc__.split("TASK=u")[0].strip(), "smoke": smoke, "examples": info,
           "montage": {"eligible_tiles": int(len(K)), "eligible_samples": int(K["sample"].nunique()),
                       "top16_from_progressor_samples": int((MC[MC.group == "top"]["class"] == "P").sum()), "bottom16_from_progressor_samples": int((MC[MC.group == "bot"]["class"] == "P").sum()),
                       "top16_grades": MC[MC.group == "top"].grade.value_counts().to_dict(), "bottom16_grades": MC[MC.group == "bot"].grade.value_counts().to_dict(),
                       "u_top16_range": [float(MC[MC.group == "top"].u.min()), float(MC[MC.group == "top"].u.max())],
                       "u_bottom16_range": [float(MC[MC.group == "bot"].u.min()), float(MC[MC.group == "bot"].u.max())]},
           "kept_tile_u_by_class": {("progressor" if k == 1 else "non_progressor"): {"median": round(float(v.median()), 2), "q1": round(float(v.quantile(0.25)), 2),
                                                                                       "q3": round(float(v.quantile(0.75)), 2), "tiles": int(v.size)} for k, v in Kd}}
    json.dump(res, open(OUTR + "/fig3_tiles.json", "w"), indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)); print("OP FIG3 TILES make DONE", flush=True)
