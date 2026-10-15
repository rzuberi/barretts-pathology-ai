"""Scanner robustness, item 4 prep (docs/paper_scanner_robustness.md @ abe96e9): write the switch extractor's sample representation (sr_index.py
sample_mean_<m>.npy, rows in sample-table order) as a CSV in the img_mean.csv layout for the R fits. Row-level, cluster only. Idempotent, atomic."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_common import ROW, M
KEY = {"H-optimus-0": "hoptimus0", "Virchow2": "virchow2"}; dec = json.load(open(ROW + "/decision.json")); assert dec["switch_to"] in KEY, dec["decision"]
m = KEY[dec["switch_to"]]; out = f"{ROW}/img_mean_{m}.csv"
if not os.path.exists(out):
    X = np.load(f"{ROW}/sample_mean_{m}.npy"); C = pd.read_csv(M + "/set_C.csv", dtype=str); assert X.shape[0] == len(C)
    D = pd.DataFrame(X.astype(np.float64), columns=[f"u{i}" for i in range(X.shape[1])]); D.insert(0, "Sample", C.Sample.values)
    tmp = out + f".tmp{os.getpid()}"; D.to_csv(tmp, index=False); os.replace(tmp, out)
print("SR CV PREP DONE", m, out, flush=True)
