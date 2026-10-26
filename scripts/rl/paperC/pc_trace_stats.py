"""Paper C item 5, qualitative half: what the thinking traces do. For the retrieval-prompted thinking run, measure how often the
model's answer simply follows the retrieved majority outcome, how long the traces are, and whether traces that disagree with the
retrieval are any better than the retrieval. CPU, reads only the saved preds (which carry the capped trace in 'think').
Out: $O/trace_stats.json
"""
import json, re
import numpy as np
from pc_common import O, X, read_jsonl

TAG = X + "/preds/pc2_think_r__test_sub.jsonl"
WORDS = ("up", "down", "none")


def retrieval_majority(ret):
    """The outcome the retrieval sentence reports most often: '... went up after A, down after B and was unchanged after C of the N.'"""
    m = re.search(r"went up after (\d+), down after (\d+) and was unchanged after (\d+)", ret or "")
    if not m: return None
    u, d, n = (int(x) for x in m.groups())
    if u == d == n == 0: return None
    return WORDS[int(np.argmax([u, d, n]))]


if __name__ == "__main__":
    rows = read_jsonl(TAG)
    sub = {(r["cell"], r["pert"], r["gene"]): r for r in read_jsonl(X + "/data/test_sub.jsonl")}
    out = {"n": len(rows)}
    ans = [max(WORDS, key=lambda w: r["p_" + w]) for r in rows]
    maj = [retrieval_majority(sub[(r["cell"], r["pert"], r["gene"])].get("ret")) for r in rows]
    lab = [r["label"] for r in rows]
    ok = [i for i, m in enumerate(maj) if m]
    follow = [ans[i] == maj[i] for i in ok]
    out["follows_retrieval_majority"] = float(np.mean(follow))
    out["acc_overall"] = float(np.mean([a == l for a, l in zip(ans, lab)]))
    out["acc_retrieval_majority_itself"] = float(np.mean([maj[i] == lab[i] for i in ok]))
    agree = [i for i in ok if ans[i] == maj[i]]; dis = [i for i in ok if ans[i] != maj[i]]
    out["n_agree"], out["n_disagree"] = len(agree), len(dis)
    out["acc_when_following"] = float(np.mean([ans[i] == lab[i] for i in agree])) if agree else None
    out["acc_when_departing"] = float(np.mean([ans[i] == lab[i] for i in dis])) if dis else None
    out["acc_of_retrieval_when_model_departs"] = float(np.mean([maj[i] == lab[i] for i in dis])) if dis else None
    th = [r.get("think", "") for r in rows]
    out["trace_chars_mean"] = float(np.mean([len(t) for t in th])); out["trace_chars_median"] = float(np.median([len(t) for t in th]))
    out["trace_closed_fraction"] = float(np.mean([t.rstrip().endswith("</think>") for t in th]))
    out["trace_mentions_training_counts"] = float(np.mean([bool(re.search(r"training data|unchanged after|went up after|neighbou?r", t, re.I)) for t in th]))
    json.dump(out, open(O + "/trace_stats.json", "w"), indent=1); print(json.dumps(out, indent=1))
