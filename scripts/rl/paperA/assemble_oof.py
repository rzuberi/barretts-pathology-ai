"""Item 2: stitch the five fold models' predictions into one out-of-fold prediction file per set (each section scored by the model that
did not train on its patient's fold). Env: SEED. Out: O/preds/cf_oof_s<SEED>__{pop,test485,val,test}.jsonl (+ timing)."""
import json, os
from pa_common import *
S = int(os.environ.get("SEED", 0)); fold = json.load(open(O + "/folds.json"))["fold"]
for s in ("pop", "test485", "val", "test"):
    F = [load_preds(f"{O}/preds/cf_s{S}_f{k}", s) for k in range(5)]; assert all(F), "missing fold preds"
    rs = rows(s); n_oof = 0
    with open(f"{O}/preds/cf_oof_s{S}__{s}.jsonl", "w") as f:
        for r in rs:
            k = fold.get(r["anon_id"], 0); d = F[k][r["id"]]; d = dict(d); d["fold"] = k; n_oof += r["anon_id"] in fold; f.write(json.dumps(d) + "\n")
    os.chmod(f"{O}/preds/cf_oof_s{S}__{s}.jsonl", 0o600)
    tm = json.load(open(f"{O}/preds/cf_s{S}_f0__{s}.timing.json")); tm["tag"] = f"cf_oof_s{S}"; json.dump(tm, open(f"{O}/preds/cf_oof_s{S}__{s}.timing.json", "w"))
    print(s, len(rs), "rows,", n_oof, "with a fold assignment (val/test patients default to fold 0's model: never trained on)")
