"""Two papers, agent ART (docs/two_papers_ART.md): does L-IMG read slide artefact?  Phase 2 (no slide pixels except TASK=montage_*, a CPU job).
Inputs, read-only: per-tile QC from tp_art_qc.py (correct tile-centre positions), the UNI2-h bags (killcoyne_mm/bags), stored predictions (tb_common),
E1 tile scores u_t (killcoyne_mm/explore_E1/u; checked there against the stored logits), E1 pixel statistics at the old (top-left) positions
(killcoyne_mm/explore_E1/px), the refits of tp_art_refit.R. Metric helpers (fsa, boots, hblock, patw, batch / scanner labels) come from tp_p1b_main.py,
whose module body is executed unchanged with a task name matching none of its branches.
Row-level files only under killcoyne_mm/two_papers_ART/ on the cluster; aggregates (no identifiers, no dates, cells < 5 as "<5") ->
results/paper_final/two_papers_ART/<TASK>.json.  Montages: tiles with numbers only.
TASK = merge | describe | montage_<set> | feats | frac | score | e1          ART_NB=k: smoke mode (k bootstrap draws; JSON written aside)."""
import os, sys, json, glob, numpy as np, pandas as pd
PP = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, PP)
ATASK = os.environ["TASK"]; os.environ["TASK"] = "art_none"
exec(compile(open(PP + "/tp_p1b_main.py").read(), PP + "/tp_p1b_main.py", "exec"))   # definitions only (no P1B branch matches)
os.environ["TASK"] = ATASK
from tp_art_qc import bag_slides
A = M + "/two_papers_ART"; OUTA = T + "/results/paper_final/two_papers_ART"; NBA = NB
if os.environ.get("ART_NB"): NBA = int(os.environ["ART_NB"]); OUTA = A + "/smoke_json"
NBX = NBA
for d in (A, OUTA, A + "/feats", A + "/montages"): os.makedirs(d, exist_ok=True)
def adump(res, name=None):
    res["agent"] = "ART"; json.dump(res, open(f"{OUTA}/{name or ATASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("ART DONE", name or ATASK, flush=True)
r4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
# ------------------------------------------------------------------ the artefact flag (fixed after the describe / calibration montages; see the doc)
FLAG = dict(tis_min=0.15, line_d=0.50, ink=0.02, blur_q=0.05)
# calibration (montage_calib, viewed 21 Oct): tissue 0.15-0.25 = mostly genuine tissue-edge tiles -> glass cut 0.15 (P1B's panel rule); a straight
# band of 'all non-tissue' pixels (line_w) flags ordinary tissue borders -> dropped; a straight band of dark unstained pixels (line_d) = coverslip edges,
# bubble rims, folds; ink > 0.02 = blue/green marker or margin dye; lowest 5% tissue-pixel Laplacian variance = visibly out of focus (5-10% mostly sharp)
def flags(Q, lap_thr=None):
    """per-tile artefact components (boolean columns) and the union 'art'. blur threshold = blur_q quantile of lapvar_t over tissue tiles (all set C tiles)"""
    F = pd.DataFrame(index=Q.index)
    F["bg"] = Q.tissue.values < FLAG["tis_min"]                                             # glass (white or grey), coverslip, empty / faint areas
    F["edge"] = ~F.bg & (Q.line_d.values >= FLAG["line_d"])   # straight dark unstained band across the tile (coverslip edge, bubble rim, fold)
    F["pen"] = Q.ink.values > FLAG["ink"]
    ok = ~F.bg & np.isfinite(Q.lapvar_t.values)
    if lap_thr is None: lap_thr = float(np.nanquantile(Q.lapvar_t.values[ok], FLAG["blur_q"]))
    F["blur"] = ~F.bg & (np.nan_to_num(Q.lapvar_t.values, nan=0.0) < lap_thr)
    F["art"] = F.bg | F.edge | F.pen | F.blur; return F, lap_thr
def qc_table():
    return pd.read_parquet(A + "/qc_all.parquet")
res = {"task": ATASK}
# ================================================================== merge: QC chunks -> one table in bag order (sample, slide, tile)
if ATASK == "merge":
    Q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(A + "/qc/chunk_??.csv"))]); SL = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(A + "/qc/chunk_??.slides.csv"))])
    mp = bag_slides(); BI = pd.read_csv(M + "/bag_index.csv", dtype={"sample_id": str}).set_index("sample_id")
    assert Q.npz.nunique() == mp.npz.nunique(), (Q.npz.nunique(), mp.npz.nunique()); Qg = {k: g.sort_values("tile") for k, g in Q.groupby("npz")}
    parts_ = []
    for s in ids:
        for npz in mp[mp.cnv == s].npz: g = Qg[npz].copy(); g.insert(0, "sample", s); parts_.append(g)
    QA = pd.concat(parts_).reset_index(drop=True); QA["bag_pos"] = QA.groupby("sample").cumcount()
    nt = QA.groupby("sample").size().reindex(ids); assert (nt.values == BI.loc[ids, "n_tiles"].values).all(), "bag length mismatch"
    # E1 pixel table (old positions) in the same order, for the E1 recheck
    E1 = pd.concat([pd.read_csv(f, dtype={"key": str}) for f in sorted(glob.glob(M + "/explore_E1/px/chunk_*.csv"))]).rename(columns={"key": "sample"})
    QA = QA.merge(E1[["sample", "npz", "tile", "white", "lapvar", "mask_nb", "ink"]].rename(columns={"white": "white_e1", "lapvar": "lapvar_e1", "mask_nb": "mask_nb_e1", "ink": "ink_e1"}),
                  on=["sample", "npz", "tile"], how="left", validate="many_to_one")
    assert QA.white_e1.notna().all() and len(QA) == nt.sum()
    QA.to_parquet(A + "/qc_all.parquet", index=False); SL.to_csv(A + "/qc_slides.csv", index=False)
    res["merge"] = {"tiles": int(len(QA)), "slides": int(QA.npz.nunique()), "samples": int(QA["sample"].nunique()), "slide_products": SL["product"].value_counts().to_dict(),
                    "level_mpp_median": r3(SL.level_mpp.median()), "tile_size": SL.tile_size.unique().tolist()}
    adump(res)
# ================================================================== describe: metric distributions (all tiles), to set the flag
elif ATASK == "describe":
    Q = qc_table(); cols = ["white", "tissue", "glass", "dlow", "ink", "dark", "sat_t", "grey_t", "lapvar", "lapvar_t", "ten_t", "line", "line_d", "line_w", "step", "mask_own", "mask_nb"]
    qs = [0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1]
    res["quantiles"] = {c: dict(zip([str(q) for q in qs], [r4(v) for v in np.nanquantile(Q[c].values.astype(float), qs)])) for c in cols}
    res["share_tissue_lt"] = {str(t): r4((Q.tissue < t).mean()) for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)}
    res["share_line_d_ge"] = {str(t): r4((Q.line_d >= t).mean()) for t in (0.5, 0.7, 0.8, 0.9, 0.95)}
    res["share_ink_gt"] = {str(t): r4((Q.ink > t).mean()) for t in (0.005, 0.01, 0.02, 0.05, 0.1)}
    F, lt = flags(Q); res["flag_params"] = FLAG; res["lapvar_t_threshold"] = r4(lt); res["flag_shares"] = {c: r4(F[c].mean()) for c in F}
    SC = SDS.scanner.reindex(Q["sample"].values).values
    res["lapvar_t_by_scanner_tissue_tiles"] = {k: [r4(v) for v in np.nanquantile(Q.lapvar_t.values[(SC == k) & ~F.bg.values], [0.05, 0.25, 0.5, 0.75])] for k in pd.unique(SC)}
    res["flag_shares_by_scanner"] = {k: {c: r4(F[c].values[SC == k].mean()) for c in F} for k in pd.unique(SC)}
    adump(res)
# ================================================================== montages (CPU; reads slide pixels at the model's tile: level 2, 224 px, centre rule)
elif ATASK.startswith("montage_"):
    import openslide, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from tp_art_qc import tile_top_left, remap
    Q = qc_table(); F, lt = flags(Q); Q = pd.concat([Q, F], axis=1); SET = ATASK[8:]; rng = np.random.RandomState(20261021); per = 48
    def pick(m, k): idx = np.where(m)[0]; return list(rng.choice(idx, min(k, len(idx)), replace=False)) if len(idx) else []
    if SET == "calib":   # bins of each metric, to calibrate thresholds by eye
        G = {"tissue 0.15-0.25": pick((Q.tissue >= 0.15) & (Q.tissue < 0.25), per), "tissue 0.25-0.35": pick((Q.tissue >= 0.25) & (Q.tissue < 0.35), per),
             "tissue 0.10-0.15": pick((Q.tissue >= 0.10) & (Q.tissue < 0.15), per), "line_d 0.3-0.5, tissue >= 0.15": pick((Q.line_d >= 0.3) & (Q.line_d < 0.5) & (Q.tissue >= 0.15), per),
             "tissue < 0.15": pick(Q.tissue < 0.15, per),
             "line_d >= 0.9, tissue >= 0.25": pick((Q.line_d >= 0.9) & (Q.tissue >= 0.25), per), "line_d 0.6-0.9, tissue >= 0.25": pick((Q.line_d >= 0.6) & (Q.line_d < 0.9) & (Q.tissue >= 0.25), per),
             "line_w >= 0.97, tissue 0.25-0.6": pick((Q.line_w >= 0.97) & (Q.tissue >= 0.25) & (Q.tissue < 0.6), per),
             "lapvar_t lowest 5% (tissue >= 0.25)": pick(F.blur.values, per), "lapvar_t 5-10% (tissue >= 0.25)": pick(~F.bg.values & (Q.lapvar_t >= lt) & (Q.lapvar_t < np.nanquantile(Q.lapvar_t[~F.bg], 0.10)), per),
             "ink > 0.02": pick(Q.ink > 0.02, per), "ink 0.005-0.02": pick((Q.ink > 0.005) & (Q.ink <= 0.02), per)}
    elif SET == "valid":   # random flagged vs random unflagged tiles (all set C tiles), and each component
        G = {"flagged (random)": pick(Q.art.values, 96), "unflagged (random)": pick(~Q.art.values, 96)}
        for c in ("bg", "edge", "pen", "blur"): G[f"component {c} (random)"] = pick(Q[c].values, per)
        G["unflagged, tissue 0.15-0.25 (borderline)"] = pick(~Q.art.values & (Q.tissue < 0.25), per)
    elif SET == "blind":   # 40 flagged + 40 unflagged pre-event tiles, shuffled, numbers only; the key is written aside (montages/blind_key.json)
        pre_ = np.isin(Q["sample"].values, ids[PRE]); L = pick(Q.art.values & pre_, 40) + pick(~Q.art.values & pre_, 40); L = [L[i] for i in rng.permutation(len(L))]
        G = {"blind check: 80 tiles, half flagged": L}
    elif SET == "p1b":   # P1B panel candidates (top / bottom L-IMG tiles): flag vs P1B's tissue < 0.15 rule
        C = pd.read_csv(M + "/two_papers_P1B/panel/candidates_SECRET.csv", dtype={"sample": str}); C["tile"] = C.tile.astype(int)
        key = pd.Series(np.arange(len(Q)), index=pd.MultiIndex.from_arrays([Q["sample"].values, Q.bag_pos.values])); C["q"] = key.reindex(pd.MultiIndex.from_arrays([C["sample"].values, C.tile.values])).values
        assert C.q.notna().all(); C["q"] = C.q.astype(int)
        top = C[(C.group == "top") & (C["rank"] <= 40)]; bot = C[(C.group == "bottom") & (C["rank"] <= 40)]
        G = {"P1B top-40 L-IMG tiles": list(top.q.values), "P1B bottom-40 L-IMG tiles": list(bot.q.values)}
        tab = {g: {"n": int(len(d)), "flagged_art": int(Q.art.values[d.q.values].sum()), **{c: int(Q[c].values[d.q.values].sum()) for c in ("bg", "edge", "pen", "blur")}} for g, d in (("top40", top), ("bottom40", bot))}
        T120 = C[C.group == "top"]; Tt = C[C.group == "bottom"]
        tab["top120"] = {"n": int(len(T120)), "flagged_art": int(Q.art.values[T120.q.values].sum())}; tab["bottom120"] = {"n": int(len(Tt)), "flagged_art": int(Q.art.values[Tt.q.values].sum())}
        res["p1b_candidates"] = tab
    else: raise SystemExit("unknown set " + SET)
    cache = {}
    def crop(q):
        r = Q.iloc[q]
        if r.npz not in cache:
            with np.load(r.npz, allow_pickle=True) as z: cache[r.npz] = (remap(str(z["slide_path"])), int(z["level"]), int(z["tile_size"]))
        sp, lv, ts = cache[r.npz]; s = openslide.OpenSlide(sp); ds = float(s.level_downsamples[lv]); x0, y0 = int(r.x0), int(r.y0)
        im = s.read_region((int(round(x0 * ds)), int(round(y0 * ds))), lv, (ts, ts)).convert("RGB"); s.close(); return im
    meta = []
    for gi, (g, L) in enumerate(G.items()):
        L = list(L); nm = f"{SET}_{gi:02d}"; ncol = 8; nrow = int(np.ceil(len(L) / ncol)) if L else 1
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 2.1 + 0.6), dpi=200, squeeze=False)
        for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
        for i, q in enumerate(L):
            ax = axs.ravel()[i]; ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.imshow(crop(q)); r = Q.iloc[q]
            tag = "".join(k for k, c in (("B", "bg"), ("E", "edge"), ("P", "pen"), ("F", "blur")) if r[c])
            ax.set_title(f"{i + 1}" if SET == "blind" else f"{i + 1} t{r.tissue:.2f} {tag or '-'}", fontsize=9, pad=1.5)
            meta.append({"set": nm, "pos": i + 1, "art": bool(r.art), "tissue": r4(r.tissue), "line_d": r4(r.line_d), "lapvar_t": r4(r.lapvar_t), "ink": r4(r.ink)})
        fig.suptitle(f"{g}  (n = {len(L)})" + ("" if SET == "blind" else "; title: number, tissue share, flags B=glass/background E=edge P=pen F=blur"), fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f"{A}/montages/{nm}.png", dpi=200); plt.close(fig)
    json.dump({k: v for k, v in res.items() if k != "task"}, open(f"{A}/montages/{SET}_summary.json", "w"), indent=1, default=str)
    if SET == "blind": json.dump([{"pos": m_["pos"], "art": m_["art"]} for m_ in meta], open(f"{A}/montages/blind_key.json", "w"))
    res["groups"] = {f"{SET}_{gi:02d}": g for gi, g in enumerate(G)}; res["lapvar_t_threshold"] = r4(lt); adump(res)
# ================================================================== feats: clean-tile and control mean embeddings per sample
elif ATASK == "feats":
    Q = qc_table(); F, lt = flags(Q); BI = pd.read_csv(M + "/bag_index.csv", dtype={"sample_id": str}).set_index("sample_id"); MINC = 32
    IMG = pd.read_csv(M + "/img_mean.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids)
    rng = np.random.RandomState(20261021); rows = {k: [] for k in ("all", "clean", "rand", "clean_noblur")}; info = []
    def keep_of(bad, q):
        clean = ~bad; nc = int(clean.sum()); keep = clean.copy(); topped = 0
        if nc < MINC:   # stated rule: top up to 32 with the least-artefactual flagged tiles (highest tissue share, then sharpest)
            order = np.lexsort((-np.nan_to_num(q.lapvar_t.values), -q.tissue.values)); add = [i for i in order if not keep[i]][:MINC - nc]; keep[add] = True; topped = len(add)
        return keep, nc, topped
    QI = Q.groupby("sample").indices
    for s in ids:
        ii = np.sort(QI[s]); q = Q.iloc[ii]; f = F.iloc[ii]; E_ = np.load(BI.loc[s, "npz_path"])["embeddings"].astype(np.float64); assert len(E_) == len(q) and (q.bag_pos.values == np.arange(len(q))).all()
        keep, nc, topped = keep_of(f.art.values, q); keep2, nc2, _ = keep_of((f.bg | f.edge | f.pen).values, q)
        rnd = np.zeros(len(q), bool); rnd[rng.choice(len(q), int(keep.sum()), replace=False)] = True   # control: random subset of the same size
        rows["all"].append(E_.mean(0)); rows["clean"].append(E_[keep].mean(0)); rows["rand"].append(E_[rnd].mean(0)); rows["clean_noblur"].append(E_[keep2].mean(0))
        info.append({"sample": s, "n_tiles": len(q), "n_clean": nc, "topped_up": topped, "n_used": int(keep.sum()), "n_clean_noblur": nc2})
    I = pd.DataFrame(info); dmax = float(np.abs(np.vstack(rows["all"]) - IMG.values).max())
    for k, v in rows.items():
        X = pd.DataFrame(np.vstack(v), columns=[f"u{i}" for i in range(1536)]); X.insert(0, "Sample", ids); X.to_csv(f"{A}/feats/{k}.csv.gz", index=False, float_format="%.8g")
    I.to_csv(A + "/feats/clean_info.csv", index=False)
    # per-slide clean counts
    Qs = Q.assign(clean=~F.art.values).groupby(["sample", "npz"]).clean.agg(["sum", "size"])
    pre_s = set(ids[PRE]); Qs_pre = Qs[[i[0] in pre_s for i in Qs.index]]
    res["feats"] = {"check_all_tiles_mean_vs_img_mean_csv_max_abs": dmax, "lap_thr": r4(lt), "flag": FLAG, "min_clean": MINC,
                    "samples_lt32_clean": {"set_C": cnt((I.n_clean < MINC).sum()), "pre_event": cnt((I.n_clean.values[PRE] < MINC).sum()),
                                           "pre_event_progressor": cnt(((I.n_clean.values < MINC) & PRE & (y == 1)).sum()), "pre_event_np": cnt(((I.n_clean.values < MINC) & PRE & (y == 0)).sum())},
                    "slides_lt32_clean": {"set_C": cnt((Qs["sum"] < MINC).sum()), "pre_event_samples_slides": cnt((Qs_pre["sum"] < MINC).sum()), "n_slides_set_C": int(len(Qs))},
                    "samples_with_0_clean": cnt((I.n_clean == 0).sum()), "clean_share_median": r3((I.n_clean / I.n_tiles).median()),
                    "rule": "sample mean over its clean tiles (all its slides); a sample with < 32 clean tiles is topped up to 32 with its flagged tiles of highest tissue share (then highest lapvar_t); control 'rand' = random tiles, same number as used"}
    assert dmax < 1e-4, dmax; adump(res)
# ================================================================== frac: artefact fraction per sample by class, scanner, batch; artefact-only probe
elif ATASK == "frac":
    from sklearn.linear_model import LogisticRegression
    Q = qc_table(); F, lt = flags(Q); comps = ["art", "bg", "edge", "pen", "blur"]
    FR = F.groupby(Q["sample"].values)[comps].mean().reindex(ids); FR["lapvar_t_med"] = Q.groupby("sample").lapvar_t.median().reindex(ids).values
    rows = ROWS; bb = boots(rows); cls = np.where(y == 1, "progressor", "non_progressor")
    def mdiff(v, m1, m0):   # difference in means (sample level) with patient bootstrap
        o = np.nanmean(v[rows][m1[rows]]) - np.nanmean(v[rows][m0[rows]]); d = [np.nanmean(v[b][m1[b]]) - np.nanmean(v[b][m0[b]]) for b in bb]; return dstat(o, d)
    def desc(v, m): v = v[m]; return {"n": cnt(m.sum()), "mean": r4(np.nanmean(v)), "median": r4(np.nanmedian(v)), "p75": r4(np.nanpercentile(v, 75)), "p90": r4(np.nanpercentile(v, 90))}
    R = {"by_class": {}, "by_scanner": {}, "by_batch": {}}
    for c in comps:
        v = FR[c].values; R["by_class"][c] = {"progressor": desc(v, PRE & (y == 1)), "non_progressor": desc(v, PRE & (y == 0)), "P_minus_NP": mdiff(v, y == 1, y == 0)}
        R["by_scanner"][c] = {sc_: {"all": desc(v, PRE & (SCAN == sc_)), "progressor": desc(v, PRE & (SCAN == sc_) & (y == 1)), "non_progressor": desc(v, PRE & (SCAN == sc_) & (y == 0))} for sc_ in pd.unique(SCAN[PRE])}
        sc2 = pd.unique(SCAN[PRE]); R["by_scanner"][c]["difference_first_minus_second"] = {"scanners": list(map(str, sc2[:2])), **mdiff(v, SCAN == sc2[0], SCAN == sc2[1])}
    # batch: anonymised labels B1.. by size (pre-event samples)
    bl = pd.Series(BATCH[PRE]).value_counts(); bmap = {b: f"B{i + 1}" for i, b in enumerate(bl.index)}; BL = np.array([bmap.get(b, "other") for b in BATCH])
    for c in comps:
        v = FR[c].values; R["by_batch"][c] = {b: {"n_samples": cnt((PRE & (BL == b)).sum()), "n_progressor_samples": cnt((PRE & (BL == b) & (y == 1)).sum()), "mean": r4(np.nanmean(v[PRE & (BL == b)]))} for b in bmap.values()}
        # between-batch variance: eta^2 over pre-event samples, permutation p (batch labels permuted across patients' samples, 2,000)
        vv = v[PRE]; gg = BL[PRE]; e2 = lambda g_: sum(((vv[g_ == b].mean() - vv.mean()) ** 2) * (g_ == b).sum() for b in np.unique(g_)) / ((vv - vv.mean()) ** 2).sum()
        o = e2(gg); rng = np.random.RandomState(0); perm = [e2(rng.permutation(gg)) for _ in range(2000)]
        R["by_batch"][c]["_eta2"] = r3(o); R["by_batch"][c]["_perm_p_sample_level"] = r4((1 + np.sum(np.array(perm) >= o)) / 2001)
        # within class
        for cl_, mm in (("non_progressor", y == 0), ("progressor", y == 1)):
            m2 = PRE & mm; vv2 = v[m2]; g2 = BL[m2]; e2b = lambda g_: sum(((vv2[g_ == b].mean() - vv2.mean()) ** 2) * (g_ == b).sum() for b in np.unique(g_)) / max(((vv2 - vv2.mean()) ** 2).sum(), 1e-12)
            o2 = e2b(g2); pp2 = [e2b(rng.permutation(g2)) for _ in range(2000)]; R["by_batch"][c][f"_eta2_{cl_}"] = r3(o2); R["by_batch"][c][f"_perm_p_{cl_}"] = r4((1 + np.sum(np.array(pp2) >= o2)) / 2001)
    # artefact-only probes. (a) single fraction, direction fixed in advance (more artefact = progressor): no fitting
    PR = {}
    for c in comps:
        X = np.repeat(FR[c].values[:, None], 10, 1); PR[f"{c}_fraction (no fit)"] = X
    # (b) nested logistic probe on all artefact features (fractions + median focus), fitted on the training rows of each outer fold (set C rows, as L-IMG)
    feats = comps + ["lapvar_t_med"]; Xf = FR[feats].values.astype(float); Xf[:, -1] = np.log(np.maximum(Xf[:, -1], 1e-6)); P = np.full((n, 10), np.nan)
    for j in range(10):
        for k in range(1, 11):
            tr = FO0[:, j] != k; te = ~tr; mu = Xf[tr].mean(0); sd = Xf[tr].std(0); sd[sd == 0] = 1
            lr = LogisticRegression(C=1.0, max_iter=2000).fit((Xf[tr] - mu) / sd, y[tr]); P[te, j] = lr.predict_proba((Xf[te] - mu) / sd)[:, 1]
    PR["probe: logistic on artefact features (nested)"] = P
    obs = {k: fsa(v, rows) for k, v in PR.items()}; B = pd.DataFrame([{k: fsa(v, b) for k, v in PR.items()} for b in bb])
    cl_ = pd.factorize(BATCH)[0]; obs_wb = {k: fsa(v, rows, cell=cl_) for k, v in PR.items()}; Bwb = pd.DataFrame([{k: fsa(v, b, cell=cl_) for k, v in PR.items()} for b in bb])
    R["artefact_only_auroc"] = {k: {"auroc": r3(obs[k]), "ci95": ci2(B[k]), "within_batch_auroc": r3(obs_wb[k]), "within_batch_ci95": ci2(Bwb[k])} for k in PR}
    # correlation of artefact fraction with the stored L-IMG score within class (repeat-mean held-out logit)
    from scipy.stats import spearmanr
    LG = np.log(np.clip(O0["L-IMG"], 1e-6, 1 - 1e-6) / (1 - np.clip(O0["L-IMG"], 1e-6, 1 - 1e-6))).mean(1)
    R["spearman_art_fraction_vs_stored_LIMG_logit"] = {cl: r3(spearmanr(FR.art.values[PRE & m], LG[PRE & m]).correlation) for cl, m in (("all", np.ones(n, bool)), ("progressor", y == 1), ("non_progressor", y == 0))}
    R["note"] = "pre-event samples; artefact fraction = share of the sample's L-IMG tiles flagged; differences = sample-level means with 2,000-draw patient bootstrap; batch labels anonymised by size; AUROC = fold-stratified (fsa), repeat mean"
    res["frac"] = R; adump(res)
# ================================================================== score: clean-tile L-IMG refit vs stored
elif ATASK == "score":
    def rd(name):
        fo = sorted(glob.glob(f"{A}/refit/{name}/outer_rep_*.csv")); assert len(fo) == 10, (name, len(fo)); D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo])
        Of = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int); assert (Of == FO0).all(); return D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values
    base = rd("base"); chk = float(np.nanmax(np.abs(base - O0["L-IMG"]))); res["check_base_refit_vs_stored_max_abs"] = chk; assert chk < 1e-6, chk
    CL = rd("clean"); RN = rd("rand"); NB_ = rd("clean_noblur"); CNV = O0["L-CNV"]
    O = {"L-CNV": CNV, "L-IMG": CL, "L-LATE": (CNV + CL) / 2, "L-IMG stored": O0["L-IMG"], "L-LATE stored": O0["L-LATE"], "L-IMG random-subset control": RN, "L-LATE random-subset control": (CNV + RN) / 2,
         "L-IMG clean, blur kept": NB_, "L-LATE clean, blur kept": (CNV + NB_) / 2}
    EXTRA = (("clean_minus_stored_IMG", "L-IMG", "L-IMG stored"), ("clean_minus_stored_LATE", "L-LATE", "L-LATE stored"), ("clean_minus_random_IMG", "L-IMG", "L-IMG random-subset control"),
             ("stored_IMG_minus_CNV", "L-IMG stored", "L-CNV"), ("stored_LATE_minus_IMG", "L-LATE stored", "L-IMG stored"),
             ("random_IMG_minus_CNV", "L-IMG random-subset control", "L-CNV"), ("random_minus_stored_IMG", "L-IMG random-subset control", "L-IMG stored"),
             ("noblur_minus_stored_IMG", "L-IMG clean, blur kept", "L-IMG stored"), ("noblur_IMG_minus_CNV", "L-IMG clean, blur kept", "L-CNV"), ("noblur_LATE_minus_IMG", "L-LATE clean, blur kept", "L-IMG clean, blur kept"))
    INFO = pd.read_csv(A + "/feats/clean_info.csv", dtype={"sample": str}).set_index("sample").reindex(ids); LOW = (INFO.n_clean.values < 32)
    rows = ROWS; bb = boots(rows); W = patw(rows); bc = pd.factorize(BATCH)[0]; scn = pd.factorize(SCAN)[0]; R = {}
    def blk(**kw):
        r, o, B = hblock(O, rows, bb, keep=True, **kw)
        for nm, a, c in EXTRA: r[nm] = dstat(o[a] - o[c], B[a] - B[c])
        return r
    R["overall"] = blk(); R["patient_weighted"] = blk(w=W); R["within_batch"] = blk(cell=bc); R["within_batch_patient_weighted"] = blk(w=W, cell=bc); R["within_scanner"] = blk(cell=scn)
    R["excluding_samples_lt32_clean"] = blk(mask=~LOW); R["ndbe"] = blk(mask=NDBE)
    R["note"] = ("L-IMG = clean-tile refit (same recipe, folds, nested lambda); L-LATE = (stored L-CNV + clean L-IMG) / 2; 'stored' = the published L-IMG; random-subset control = "
                 "same number of tiles drawn at random per sample. fsa AUROC, 2,000-draw patient bootstrap, paired deltas on the same draws.")
    res["score"] = R; adump(res)
# ================================================================== e1: recheck of E1's background conclusion with corrected positions
elif ATASK == "e1":
    Q = qc_table(); F, lt = flags(Q)
    # E1 flag definitions at old (E1) and corrected positions
    def e1flags(white, lapvar, mask_nb, ink):
        bg = white > 0.5; edge = ~bg & ((np.nan_to_num(mask_nb, nan=1.0) < 0.8) | (white > 0.25)); lq = np.nanpercentile(lapvar[~bg], 10); blur = ~bg & (lapvar < lq); return {"bg": bg, "edge": edge, "ink": ink > 0.02, "blur": blur}
    OLD = e1flags(Q.white_e1.values, Q.lapvar_e1.values, Q.mask_nb_e1.values, Q.ink_e1.values); NEW = e1flags(Q.white.values, Q.lapvar.values, Q.mask_nb.values, Q.ink.values)
    R = {"tile_flag_shares_all_tiles": {k: {"E1_old_positions": r4(OLD[k].mean()), "E1_definition_corrected_positions": r4(NEW[k].mean())} for k in OLD}}
    R["tile_flag_shares_all_tiles"]["artefact_flag_ART"] = {c: r4(F[c].mean()) for c in F}
    R["agreement_old_vs_corrected_bg"] = {"both": int((OLD["bg"] & NEW["bg"]).sum()), "old_only": int((OLD["bg"] & ~NEW["bg"]).sum()), "new_only": int((~OLD["bg"] & NEW["bg"]).sum())}
    R["ART_art_among_E1_old_bg"] = r4(F.art.values[OLD["bg"]].mean()); R["E1_old_bg_among_ART_art"] = r4(OLD["bg"][F.art.values].mean())
    # tile scores: outer (held-out) model of each sample, per repeat
    BI = pd.read_csv(M + "/bag_index.csv", dtype={"sample_id": str}).set_index("sample_id"); SETS = {"E1 bg (old positions)": OLD["bg"], "E1 bg (corrected)": NEW["bg"],
            "E1 bg+edge (old)": OLD["bg"] | OLD["edge"], "E1 bg+edge (corrected)": NEW["bg"] | NEW["edge"], "ART artefact": F.art.values, "ART glass/background": F.bg.values}
    contrib = {k: np.full((n, 10), np.nan) for k in SETS}; sub = {k: np.full((n, 10), np.nan) for k in SETS}; full = np.full((n, 10), np.nan); share = {k: np.zeros(n) for k in SETS}
    meanu = {k: {"art": [], "clean": []} for k in ("ART artefact",)}; ucls = {"art_P": [], "art_NP": [], "clean_P": [], "clean_NP": []}
    pos_q = Q.groupby("sample").indices
    for s_ in range(n):
        if not PRE[s_]: continue
        z = np.load(f"{M}/explore_E1/u/{ids[s_]}.npz"); U = z["U"].astype(np.float64); B0 = z["B0"]; qi = np.sort(pos_q[ids[s_]]); qi = qi[np.argsort(Q.bag_pos.values[qi])]; assert len(qi) == U.shape[0]
        for j in range(10):
            k = FO0[s_, j] - 1; u = U[:, j, k]; b0 = B0[j, k]; full[s_, j] = b0 + u.mean()
            for nm, fl in SETS.items():
                f = fl[qi]; share[nm][s_] = f.mean(); contrib[nm][s_, j] = u[f].sum() / len(u); sub[nm][s_, j] = b0 + (u[~f].mean() if (~f).any() else u.mean())
        ua = U[:, np.arange(10), FO0[s_] - 1].mean(1); fa = F.art.values[qi]; tag = "P" if y[s_] == 1 else "NP"
        if fa.any(): ucls[f"art_{tag}"].append(ua[fa].mean())
        if (~fa).any(): ucls[f"clean_{tag}"].append(ua[~fa].mean())
    LG = np.log(np.clip(O0["L-IMG"], 1e-6, 1 - 1e-6) / (1 - np.clip(O0["L-IMG"], 1e-6, 1 - 1e-6))); okc = PRE[:, None] & (O0["L-IMG"] > 1e-6) & (O0["L-IMG"] < 1 - 1e-6); R["check_decomposition_max_abs_logit"] = float(np.nanmax(np.abs(full - LG)[okc])); R["check_cells"] = int(okc.sum())
    rows = ROWS; bb = boots(rows)
    cd = lambda v, m1, m0: dstat(np.nanmean(v[rows][m1[rows]]) - np.nanmean(v[rows][m0[rows]]), [np.nanmean(v[b][m1[b]]) - np.nanmean(v[b][m0[b]]) for b in bb])
    T_ = {}
    for nm in SETS:
        c = contrib[nm].mean(1); ch = (sub[nm] - full).mean(1)
        T_[nm] = {"tile_share_pre_event": r4(share[nm][PRE].mean()), "contribution_logit_progressor": r3(np.nanmean(c[PRE & (y == 1)])), "contribution_logit_np": r3(np.nanmean(c[PRE & (y == 0)])),
                  "contribution_P_minus_NP": cd(c, y == 1, y == 0), "score_change_if_removed_progressor": r3(np.nanmean(ch[PRE & (y == 1)])), "score_change_if_removed_np": r3(np.nanmean(ch[PRE & (y == 0)])),
                  "score_change_P_minus_NP": cd(ch, y == 1, y == 0), "abs_score_change_median": r3(np.nanmedian(np.abs(ch[PRE]))), "abs_score_change_p90": r3(np.nanpercentile(np.abs(ch[PRE]), 90))}
    # AUROC of the substituted (no refit) scores vs the full mean, paired
    OS = {"full mean (stored)": full, **{f"mean without {nm}": sub[nm] for nm in SETS}}; ob = {k: fsa(v, rows) for k, v in OS.items()}; Bs = pd.DataFrame([{k: fsa(v, b) for k, v in OS.items()} for b in bb])
    T_["auroc_no_refit"] = {k: {"auroc": r3(ob[k]), "ci95": ci2(Bs[k]), "delta_vs_full": dstat(ob[k] - ob["full mean (stored)"], Bs[k] - Bs["full mean (stored)"])} for k in OS}
    T_["mean_tile_score_u (repeat-mean held-out model)"] = {k: r3(np.mean(v)) for k, v in ucls.items()}
    R["contributions"] = T_; R["note"] = ("pre-event samples; u_t from E1 (held-out fit of each sample); contribution = sum of u_t over flagged tiles / N (logit); score change = b0 + mean over unflagged tiles "
                                          "- b0 - mean over all tiles (same fitted model, no refit); E1 definitions: bg white > 0.5, edge mask_nb < 0.8 or white > 0.25, blur < p10 lapvar of non-bg")
    res["e1"] = R; adump(res)
