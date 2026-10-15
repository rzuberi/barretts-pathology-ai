import time, numpy as np, h5py, openslide, pandas as pd, os
from PIL import Image
s = pd.read_csv("/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun/slides.csv", dtype={"idx": int})
for row in s.iloc[[5, 1000]].itertuples():
    with h5py.File(row.h5) as h: coords = np.asarray(h["coords"]); a = dict(h.attrs)
    path = a["slide_path"]; print("slide", row.idx, "path realpath:", os.path.realpath(path)[:60], "n tiles", len(coords))
    t = time.time(); sl = openslide.OpenSlide(path); lvl = int(a["level"]); ds = sl.level_downsamples[lvl]; print(" open %.2fs" % (time.time() - t), "level", lvl, "ds", ds, "dims", sl.level_dimensions[lvl], "vendor", sl.properties.get("openslide.vendor"), "tile", sl.properties.get("openslide.level[1].tile-width"))
    sel = np.sort(np.random.RandomState(0).choice(len(coords), 200, replace=False)); t = time.time()
    for x, y in coords[sel]: im = sl.read_region((int(x * ds), int(y * ds)), lvl, (224, 224)).convert("RGB")
    dt = time.time() - t; print(" random 200 tiles read_region: %.1f tiles/s" % (200 / dt))
    sel2 = sel[:200]; t = time.time()
    for x, y in coords[np.sort(sel2)]: im = sl.read_region((int(x * ds), int(y * ds)), lvl, (224, 224)).convert("RGB")
    print(" sorted order 200: %.1f tiles/s" % (200 / (time.time() - t)))
    W, H = sl.level_dimensions[lvl]; t = time.time(); band = sl.read_region((0, int(2000 * ds)), lvl, (W, 2000)).convert("RGB"); dt = time.time() - t
    print(" band %dx2000 read: %.2fs -> %.0f Mpx/s; equiv tiles in band %d" % (W, dt, W * 2000 / dt / 1e6, (W // 224) * (2000 // 224)))
