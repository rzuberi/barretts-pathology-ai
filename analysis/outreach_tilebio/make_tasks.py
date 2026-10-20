"""Write campaign task lists. usage: make_tasks.py PHASE {smoke|full} OUT.tsv   (PHASE = tile | embed)"""
import sys, pandas as pd
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; ph, which, out = sys.argv[1:4]
s = pd.read_csv(A + "/manifests/slides.csv")
if which == "smoke":
    t = s[s.ds == "tcga"]; sm = pd.concat([t[t.label == 0].head(3), t[t.label == 1].head(2)])
    c = s[s.ds == "c16"].set_index("slide"); sm = pd.concat([sm, c.loc[["tumor_001", "normal_001", "test_001", "test_004", "test_003"]].reset_index()])
else: sm = s
env, cmd = ("pathology", "python " + A + "/code/tile_slide.py") if ph == "tile" else ("virchow2", "python " + A + "/code/embed_slide.py")
with open(out, "w") as f:
    for r in sm.itertuples(): f.write(f"{r.ds}_{r.slide.replace('.', '_')}\t{env}\t{cmd} {r.ds} {r.slide}\n")
print(len(sm), "tasks ->", out)
