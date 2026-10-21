"""Review round 9, F0 input: per analysed patient, the highest Barrett's-database confirmed grade after the last sequenced endoscopy (round-7 lookup B2 convention:
records more than 14 days after the latest exact sample date; grades NDBE 0, IND 1, LGD 2, HGD 3, IMC 4, OAC 5). Read-only use of the round-7 lookup code
(/mnt/scratche/slow/fmlab/zuberi01/analysis/clinical_lookup/r2_main.py executed up to its per-patient table `O2`; nothing in that folder is written).
Row-level output (patient id, class, flags) stays on the cluster: killcoyne_mm/review_round9/later.csv. Run in the `pathology` conda env."""
import io, contextlib, os, sys
import numpy as np, pandas as pd
M9 = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm/review_round9"; os.makedirs(M9, exist_ok=True)
src = open("/mnt/scratche/slow/fmlab/zuberi01/analysis/clinical_lookup/r2_main.py").read(); cut = src.index('O2 = pd.DataFrame(out).set_index("Patient")')
with contextlib.redirect_stdout(io.StringIO()):
    g = {"__name__": "x"}; exec(compile(src[:cut] + 'O2 = pd.DataFrame(out).set_index("Patient")\n', "r2_main_prefix", "exec"), g)
O2 = g["O2"]; cols = [c for c in ("cls", "linked", "path_after", "endo_after", "any_after", "max_after", "yrs_to_last_record") if c in O2.columns]
O2[cols].reset_index().to_csv(M9 + "/later.csv", index=False)
np_ = O2[O2.cls == "NP"]; print("NP", len(np_), "linked", int(np_.linked.sum()), "any_after", int(np_.any_after.sum()), "max_after counts", np_.max_after.value_counts(dropna=False).to_dict())
