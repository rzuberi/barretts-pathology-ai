"""Shared pieces for docs/paper_scanner_robustness.md (pre-specification abe96e9): paths, slide list in map order, smoke slides."""
import os, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; K = T + "/feasibility/paper_plan/killcoyne"; M = K + "_mm"; ROW = M + "/scanner_robustness"
OUT = T + "/results/paper_final/scanner_robustness"; HF = "/mnt/scratche/fast/fmlab/zuberi01/hf_cache"
remap = lambda p: p.replace("/scratchc/", "/mnt/scratche/fast/") if p.startswith("/scratchc/") else p
gp_of = lambda f: f.replace("uni2_tile224_lvl2", "gigapath_tile224_lvl2")
sha_of = lambda f: os.path.basename(f)[:-4]
def slide_map():
    """map rows (sample, npz) of the 676 samples in km_prep.py order (sorted by sample, npz); scanner per sample from horizons/slide_desc.csv."""
    C = pd.read_csv(M + "/set_C.csv", dtype=str); mp = pd.read_csv(K + "/uni2_npz_map.csv", dtype=str)
    mp = mp[mp.cnv.isin(C.Sample)].sort_values(["cnv", "npz"]).reset_index(drop=True)
    SD = pd.read_csv(M + "/horizons/slide_desc.csv", dtype=str).set_index("Sample"); mp["scanner"] = SD.scanner.reindex(mp.cnv).values
    assert len(mp) == 683 and mp.npz.is_unique and mp.scanner.isin(["C13239-01", "C13210"]).all(); return mp
def chunk(mp, task):
    """task = 'smoke' (first slide of each scanner in map order) or 'c<i>of<n>' (every n-th slide from i)."""
    if task == "smoke": return [mp[mp.scanner == s].npz.iloc[0] for s in ("C13239-01", "C13210")]
    i, n = map(int, task[1:].split("of")); return list(mp.npz.values[i::n])
