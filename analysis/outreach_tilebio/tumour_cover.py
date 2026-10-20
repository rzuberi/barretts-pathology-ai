"""QC: fraction of annotated tumour area (sampled points) that falls inside a kept tissue tile. usage: tumour_cover.py slide [slide...]"""
import json, sys, xml.etree.ElementTree as ET
import numpy as np
from matplotlib.path import Path
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"
for sl in sys.argv[1:]:
    m = json.load(open(f"{A}/data/tiles/c16/{sl}.json")); C = np.array(m["coords_level0_topleft"]); S = m["tile_side_level0"]
    kept = set(map(tuple, (C // S).astype(int).tolist())); pts = []
    for an in ET.parse(f"{A}/data/c16_meta/annotations/{sl}.xml").getroot().iter("Annotation"):
        if an.get("PartOfGroup", "") in ("Exclusion", "_2"): continue
        P = np.array([(float(c.get("X")), float(c.get("Y"))) for c in sorted(an.iter("Coordinate"), key=lambda c: int(c.get("Order")))])
        if len(P) < 3: continue
        lo, hi = P.min(0), P.max(0); g = np.mgrid[lo[0]:hi[0]:S / 4, lo[1]:hi[1]:S / 4].reshape(2, -1).T
        pts.append(g[Path(P).contains_points(g)] if len(g) else np.zeros((0, 2)))
    q = np.vstack(pts) if pts else np.zeros((0, 2)); cell = (q // S).astype(int)
    cov = np.mean([tuple(c) in kept for c in cell.tolist()]) if len(q) else float("nan")
    print(sl, f"tumour points {len(q)}, covered by kept tiles {cov:.3f}, otsu {m['otsu']}")
