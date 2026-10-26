"""Cross-fitting folds (CPU, once): every patient in train.jsonl or pop.jsonl gets one of 5 folds (seed 0). Fold-k encoder trains on
train.jsonl rows of patients NOT in fold k and scores the pop / test485 sections of fold-k patients, so every population score is
out-of-fold and the 479 human sections are never trained on (their patients contribute no training rows anyway). Out: O/folds.json"""
import json, numpy as np
from pa_common import *
mkdir(O); tr = rows("train"); pop = rows("pop"); t485 = rows("test485")
pats = sorted({r["anon_id"] for r in tr} | {r["anon_id"] for r in pop} | {r["anon_id"] for r in t485})
rng = np.random.RandomState(0); perm = rng.permutation(pats); fold = {p: int(i % 5) for i, p in enumerate(perm)}
trp = {r["anon_id"] for r in tr}
summary = {"n_patients": len(pats), "train_patients": len(trp), "train_rows_per_fold_heldout": {k: sum(fold[r["anon_id"]] == k for r in tr) for k in range(5)},
           "pop_sections_per_fold": {k: sum(fold[p["anon_id"]] == k for p in pop) for k in range(5)}, "test485_per_fold": {k: sum(fold[t["anon_id"]] == k for t in t485) for k in range(5)}}
json.dump({"fold": fold, "summary": summary}, open(O + "/folds.json", "w")); os.chmod(O + "/folds.json", 0o600); print(json.dumps(summary))
