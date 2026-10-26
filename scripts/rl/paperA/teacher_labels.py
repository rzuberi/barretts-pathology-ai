"""Teacher ablation labels (CPU, once). From the v3 per-juror outputs: (a) best single juror, (b) top-3 jurors' consensus, (d) soft vote
shares (share of returning jurors whose WORST grade is g), over train.jsonl + all_lowconf.jsonl (= all train-patient sections).
Juror ranking uses the validation split only: agreement of each juror's worst grade with the leave-one-out consensus of the other
jurors on val.jsonl (jury labels, no human labels). Out: O/labels/{juror1,top3,soft}.json (id -> label | probs), O/labels/selection.json"""
import json
from pa_common import *
L = mkdir(O + "/labels"); tr = rows("train") + rows("all_lowconf"); va = rows("val")
votes = juror_votes({r["report_id"] for r in tr + va}); jurors = sorted({j for v in votes.values() for j in v})
def jw(r, j): v = votes.get((r["report_id"], r["section"]), {}); return worst(v[j]) if j in v else None
sel = {}
for j in jurors:
    ok = n = 0
    for r in va:
        v = votes.get((r["report_id"], r["section"]), {})
        if j not in v: continue
        loo, _ = consensus({k: g for k, g in v.items() if k != j})
        if loo is None: continue
        n += 1; ok += worst(v[j]) == loo
    sel[j] = {"n_val": n, "loo_agreement": round(ok / max(n, 1), 4), "agree_stored_label": round(float(np.mean([jw(r, j) == r["label"] for r in va if jw(r, j)])), 4)}
rank = sorted(jurors, key=lambda j: -sel[j]["loo_agreement"]); best, top3 = rank[0], rank[:3]
lab = {"juror1": {}, "top3": {}, "soft": {}}
for r in tr:
    v = votes.get((r["report_id"], r["section"]), {})
    g = jw(r, best)
    if g: lab["juror1"][r["id"]] = g
    g, _ = consensus({j: v[j] for j in top3 if j in v})
    if g: lab["top3"][r["id"]] = g
    if v:
        c = Counter(worst(gs) for gs in v.values()); lab["soft"][r["id"]] = [round(c[g] / len(v), 4) for g in GRADES]
for k, d in lab.items(): json.dump(d, open(f"{L}/{k}.json", "w")); os.chmod(f"{L}/{k}.json", 0o600)
hard_dist = {k: dict(Counter(lab[k].values())) for k in ("juror1", "top3")}
agree_jury = {k: round(float(np.mean([lab[k].get(r["id"]) == r["label"] for r in rows("train") if r["id"] in lab[k]])), 4) for k in ("juror1", "top3")}
out = {"jurors": sel, "ranking": rank, "best": best, "top3": top3, "n_train_rows": {k: len(d) for k, d in lab.items()}, "n_candidate_rows": len(tr),
       "label_dist": hard_dist, "agreement_with_jury_on_train_hi": agree_jury}
json.dump(out, open(L + "/selection.json", "w"), indent=1); print(json.dumps(out, indent=1))
