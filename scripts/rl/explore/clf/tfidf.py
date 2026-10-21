"""TF-IDF (word 1-2g + char 3-5g) + logistic regression, CPU. C chosen on validation exact agreement. Same output format as train_clf.py."""
import json, os, sys, time
import numpy as np
from clf_common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
TAG = "tfidf_lr"; P = O + "/preds"; os.makedirs(P, exist_ok=True); os.makedirs(f"{O}/runs/{TAG}", exist_ok=True)
X = {s: [text_for(r) for r in rows(s)] for s in SETS}; y = {s: np.array([ORD[r["human"] if s == "test485" else r["label"]] for r in rows(s)]) for s in SETS}
t0 = time.time(); vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300000); vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=300000)
Xtr = hstack([vw.fit_transform(X["train"]), vc.fit_transform(X["train"])]).tocsr(); feat = lambda s: hstack([vw.transform(X[s]), vc.transform(X[s])]).tocsr()
Xv = feat("val"); best = (-1, None, None)
for C in [1, 3, 10, 30, 100]:
    clf = LogisticRegression(C=C, max_iter=3000, class_weight=None); clf.fit(Xtr, y["train"]); acc = float((clf.predict(Xv) == y["val"]).mean()); print(f"C={C} val exact {acc:.4f}", flush=True)
    if acc > best[0]: best = (acc, C, clf)
acc, C, clf = best; log = {"tag": TAG, "C": C, "best_val_exact": acc, "params": int(clf.coef_.size + clf.intercept_.size), "train_minutes": round((time.time() - t0) / 60, 1)}
for s in ["val", "test", "test485", "pop"]:
    rs = rows(s); t1 = time.time(); pr = clf.predict_proba(feat(s)); dt = time.time() - t1; nrep = len({r["report_id"] for r in rs})
    with open(f"{P}/{TAG}__{s}.jsonl", "w") as f:
        for r, p in zip(rs, pr):
            d = {k: v for k, v in r.items() if k != "report"}; d.update({"pred": GRADES[int(p.argmax())], "p_grade": float(p.max()), "probs": [round(float(x), 5) for x in p]}); f.write(json.dumps(d) + "\n")
    json.dump({"tag": TAG, "set": s, "n": len(rs), "reports": nrep, "seconds": dt, "params": log["params"]}, open(f"{P}/{TAG}__{s}.timing.json", "w"))
    log[f"{s}_exact_unweighted"] = round(float((pr.argmax(1) == y[s]).mean()), 4); log[f"{s}_s_per_report"] = round(dt / nrep, 4); print(s, log[f"{s}_exact_unweighted"], f"{dt/nrep:.4f} s/report", flush=True)
json.dump(log, open(f"{O}/runs/{TAG}/log.json", "w"), indent=1); print("TFIDF DONE")
