"""Pre-registered compute rule (DESIGN.md): measured phase-2 cost for all 956 TCGA patients + CAMELYON16 is ~25 L40S-hours (> 20),
so TCGA is subsampled to 250 LUAD + 250 LUSC (stratified random, seed 20261019; fold labels unchanged). CAMELYON16 is never subsampled.
Adds column `use` to manifests/slides.csv."""
import numpy as np, pandas as pd
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; s = pd.read_csv(A + "/manifests/slides.csv"); s["use"] = s.ds == "c16"
t = s[s.ds == "tcga"]; keep = pd.concat([g.sample(250, random_state=20261019) for _, g in t.groupby("label")]).index; s.loc[keep, "use"] = True
s.to_csv(A + "/manifests/slides.csv", index=False); u = s[s.use]
print(u.groupby(["ds", "label"]).size().to_string()); print(u[u.ds == "tcga"].groupby(["fold", "label"]).size().unstack().to_string())
