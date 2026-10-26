"""ERL step 2: scanner model and pyramid of ERIN oesophageal H&E slides (headers only, no pixels), plus the stored all-slides feature attributes,
compared with the L-IMG input (SWG Hamamatsu NDPI, 0.2204 um/px at level 0, level 2 = 0.88 um/px, 256 mask-sampled 224 px tiles).
Random 200 oesophageal H&E slides (seed 0). Only vendor/model strings, pyramid shapes and counts are saved; ImageDescription is reduced to
known scanner tokens and key=value fields that carry no identifiers (AppMag, MPP)."""
import json, os, re, collections, numpy as np, pandas as pd, openslide, h5py
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; R = T + "/results/paper_final/two_papers_ERL"; os.makedirs(R, exist_ok=True)
sl = pd.read_csv(T + "/labeller/erin_slide_organ_v3.csv", dtype=str).merge(pd.read_csv(T + "/campaigns/allslides/erin_slides_all.csv", dtype=str, low_memory=False)[["file_uuid", "path", "h5_new"]], on="file_uuid")
sl = sl[sl.is_he.str.lower().eq("true") & sl.organ_final.isin(["OESOPHAGUS", "GOJ"])]
smp = sl.sample(n=min(200, len(sl)), random_state=0)
TOK = ["aperio", "leica", "hamamatsu", "philips", "3dhistech", "ventana", "roche", "zeiss", "grundium", "olympus", "huron", "motic", "kfbio", "sakura", "dp200", "dp600", "gt450", "at2", "s360", "s60", "nanozoomer", "ultrafast", "pannoramic", "isyntax"]
props = collections.Counter(); vend = collections.Counter(); make = collections.Counter(); model = collections.Counter(); soft = collections.Counter(); toks = collections.Counter()
kv = collections.Counter(); lv = collections.Counter(); ds = collections.Counter(); mpp = collections.Counter(); dims = []; fail = 0
for p in smp.path:
    try: s = openslide.OpenSlide(p)
    except Exception: fail += 1; continue
    pr = dict(s.properties); props.update(pr.keys())
    vend[pr.get("openslide.vendor", "")] += 1; make[pr.get("tiff.Make", "")] += 1; model[pr.get("tiff.Model", "")] += 1; soft[re.sub(r"\d+(\.\d+)*", "#", pr.get("tiff.Software", ""))] += 1
    desc = pr.get("tiff.ImageDescription", "").lower(); toks.update(t for t in TOK if t in desc)
    for k in ("appmag", "mpp", "magnification", "objective"):
        m = re.search(k + r"\s*[=:]\s*([0-9.]+)", desc)
        if m: kv[f"{k}={m.group(1)}"] += 1
    lv[s.level_count] += 1; ds[tuple(round(x, 2) for x in s.level_downsamples)] += 1; mpp[pr.get("openslide.mpp-x", "none")] += 1; dims.append(s.level_dimensions[0])
# stored all-slides feature attributes (what the existing ERIN features are)
fa = collections.Counter(); ntiles = []
for h in smp.h5_new.dropna()[:60]:
    if not os.path.exists(h): continue
    with h5py.File(h, "r") as f:
        fa.update(f"{k}={f.attrs[k]}" for k in f.attrs.keys() if k in ("level", "mpp_used", "tile_size", "target_mpp", "mpp", "patch_size"))
        ntiles.append(int(f["features"].shape[0]))
out = {"n_sampled": len(smp), "open_failed": fail, "openslide_vendor": dict(vend), "tiff_make": dict(make), "tiff_model": dict(model), "tiff_software": dict(soft),
       "image_description_scanner_tokens": dict(toks), "image_description_fields": dict(kv), "level_count": {str(k): v for k, v in lv.items()},
       "level_downsamples": {str(k): v for k, v in ds.most_common(6)}, "openslide_mpp_x": dict(mpp), "property_keys_top": [k for k, _ in props.most_common(40)],
       "level0_width_median": int(np.median([d[0] for d in dims])) if dims else None, "level0_height_median": int(np.median([d[1] for d in dims])) if dims else None,
       "stored_feature_attrs": dict(fa), "stored_feature_tiles_per_slide_median": float(np.median(ntiles)) if ntiles else None, "n_feature_files_checked": len(ntiles)}
json.dump(out, open(R + "/scanner_and_scale.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
