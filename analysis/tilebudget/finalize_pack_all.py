"""P0 finalizer: index_all.json with global offsets per slide (chunk file, local offset, n)."""
import glob, json, os
P = "/mnt/scratche/slow/fmlab/zuberi01/analysis/tilebudget/pack_all"
metas = sorted(glob.glob(P + "/meta_*.json")); idx = {}; chunks = []; tot = 0
for m in metas:
    d = json.load(open(m)); f = f"chunk_{d['shard']:03d}.npy"; chunks.append({"file": f, "n_tiles": d["n_tiles"], "lo": d["lo"], "hi": d["hi"]})
    for r in d["rows"]:
        if r["n"] > 0: idx[str(r["idx"])] = {"chunk": d["shard"], "off": r["off"], "n": r["n"]}
    tot += d["n_tiles"]
json.dump({"n_shards": len(metas), "chunks": chunks, "slides": idx, "total_tiles": tot, "dim": 1536}, open(P + "/index_all.json", "w"))
print("shards", len(metas), "slides", len(idx), "tiles", tot, f"({tot * 1536 * 2 / 1e9:.1f} GB fp16)")
