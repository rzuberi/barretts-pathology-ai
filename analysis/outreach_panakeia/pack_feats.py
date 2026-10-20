"""CPU unit: pack feats/{patient}.h5 into one padded fp16 array pack/X.npy (N, CAP, 1536) + pack/index.json, in cohort.csv order.
Patients without features are listed in index.json["missing"] (and excluded downstream)."""
import json, os
import h5py, numpy as np, pandas as pd

A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia"; CAP, D = 2000, 1536
os.makedirs(A + "/pack", exist_ok=True)
c = pd.read_csv(A + "/cohort.csv")
have = [p for p in c.patient if os.path.exists(f"{A}/feats/{p}.h5")]
X = np.lib.format.open_memmap(A + "/pack/X.part.npy", mode="w+", dtype=np.float16, shape=(len(have), CAP, D))
n = []
for i, p in enumerate(have):
    with h5py.File(f"{A}/feats/{p}.h5") as h: F = h["features"][:]
    X[i, :len(F)] = F[:CAP]; X[i, len(F):] = 0; n.append(int(min(len(F), CAP)))
X.flush(); del X; os.replace(A + "/pack/X.part.npy", A + "/pack/X.npy")
missing = sorted(set(c.patient) - set(have))
json.dump({"patients": have, "n_tiles": n, "missing": missing}, open(A + "/pack/index.json", "w"))
print(f"packed {len(have)} slides, {sum(n)} tiles; missing {len(missing)}: {missing[:20]}; zero-tile slides {sum(v == 0 for v in n)}")
