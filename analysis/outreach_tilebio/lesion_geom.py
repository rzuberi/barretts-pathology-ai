"""CPU: CAMELYON16 test tumour slides -> which tissue tiles lie inside the tumour annotation (tile centre inside a Tumor polygon
and not inside an Exclusion polygon). Writes results/c16_lesion_tiles.json: slide -> {n_tiles, n_tumour_tiles, tumour_frac,
tumour_idx}. ASAP XML groups: Tumor/_0/_1 = tumour, Exclusion/_2 = cut-outs."""
import json, os, xml.etree.ElementTree as ET
import numpy as np, pandas as pd
from matplotlib.path import Path
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; M = A + "/data/c16_meta/annotations"
s = pd.read_csv(A + "/manifests/slides.csv"); s = s[(s.ds == "c16") & (s.split == "test") & (s.label == 1)]
out = {}
for sl in s.slide:
    js = f"{A}/data/tiles/c16/{sl}.json"
    if not os.path.exists(js) or not os.path.exists(f"{M}/{sl}.xml"): print("missing", sl); continue
    m = json.load(open(js)); C = np.array(m["coords_level0_topleft"], float).reshape(-1, 2) + m["tile_side_level0"] / 2
    tum, exc = [], []
    for an in ET.parse(f"{M}/{sl}.xml").getroot().iter("Annotation"):
        g = an.get("PartOfGroup", ""); pts = sorted(((int(c.get("Order")), float(c.get("X")), float(c.get("Y"))) for c in an.iter("Coordinate")))
        if len(pts) < 3: continue
        (exc if g in ("Exclusion", "_2") else tum).append(Path(np.array([(x, y) for _, x, y in pts])))
    inside = np.zeros(len(C), bool)
    for p in tum: inside |= p.contains_points(C)
    for p in exc: inside &= ~p.contains_points(C)
    out[sl] = {"n_tiles": int(len(C)), "n_tumour_tiles": int(inside.sum()), "tumour_frac": float(inside.mean()) if len(C) else 0.0,
               "tumour_idx": np.flatnonzero(inside).tolist(), "n_polygons": len(tum), "lesion": s.set_index("slide").loc[sl, "lesion"]}
    print(sl, out[sl]["lesion"], out[sl]["n_tumour_tiles"], "/", len(C), flush=True)
os.makedirs(A + "/results", exist_ok=True); json.dump(out, open(A + "/results/c16_lesion_tiles.json", "w"))
