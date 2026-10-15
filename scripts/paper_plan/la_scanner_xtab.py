"""Descriptive, NOT pre-specified (docs/paper_latent_attention.md results): scanner by progressor status, samples and patients, pre-event and NDBE pre-event. Aggregates only."""
import json, pandas as pd
M = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm"
SM = pd.read_csv(M + "/horizons/samples.csv", dtype={"Sample": str, "Patient": str}).set_index("Sample"); SD = pd.read_csv(M + "/horizons/slide_desc.csv", dtype={"Sample": str}).set_index("Sample")
SM["scanner"] = SD.scanner.reindex(SM.index); res = {}
for p, m in (("pre", SM.pre.astype(bool)), ("pre_ndbe", SM.pre.astype(bool) & SM.ndbe.astype(bool))):
    d = SM[m]; res[p] = {"samples": {s: {"progressor": int(((d.scanner == s) & (d.y == 1)).sum()), "non_progressor": int(((d.scanner == s) & (d.y == 0)).sum())} for s in sorted(d.scanner.unique())},
                         "patients_with_any_sample_on": {s: {"progressor": int(d[(d.scanner == s) & (d.y == 1)].Patient.nunique()), "non_progressor": int(d[(d.scanner == s) & (d.y == 0)].Patient.nunique())} for s in sorted(d.scanner.unique())},
                         "patients_on_both_scanners": int((d.groupby("Patient").scanner.nunique() > 1).sum())}
json.dump(res, open("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/results/paper_final/latent_attention/scanner_xtab.json", "w"), indent=1); print("XTAB DONE")
