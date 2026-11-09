import sys; sys.path.insert(0, sys.path[0])
import pandas as pd
from prep import *
sec, rep = sections(); e = endoscopies(sec)
print("sections", len(sec), "oes", sec.oes.sum(), "patients", sec.anon_id.nunique(), "endoscopies(oes)", len(e))
print("date range", e.date.min().date(), e.date.max().date())
# idea 1
rows = []
for pid, d in e.groupby("anon_id"):
    d = d.reset_index(drop=True)
    idx = d.index[d.any_lgd_bx & ~d.hgdplus]
    if not len(idx): continue
    i = idx[0]
    if d.loc[:i, "hgdplus"].any(): continue  # prior HGD+
    if i == len(d) - 1: rows.append((pid, d.loc[i, "n_lgd_bx"], False, 0, None)); continue
    later = d.loc[i + 1:]
    ev = later[later.hgdplus]
    t = ((ev.date.iloc[0] if len(ev) else later.date.iloc[-1]) - d.loc[i, "date"]).days
    rows.append((pid, d.loc[i, "n_lgd_bx"], True, t, len(ev) > 0))
c = pd.DataFrame(rows, columns=["pid", "nl", "fu", "t", "ev"])
print("idea1 index-LGD patients", len(c), "with later endoscopy", c.fu.sum())
f = c[c.fu]
print(" multifocal", (f.nl >= 2).sum(), "events", f[f.nl >= 2].ev.sum(), "| unifocal", (f.nl == 1).sum(), "events", f[f.nl == 1].ev.sum())
print(" events <183d", (f.ev & (f.t < 183)).sum(), "median fu days", f.t.median())
# idea 2
sl = slide_scores()
sl = sl.merge(rep[["CaseName", "date"]], on="CaseName", how="left")
print("slides", len(sl), "n_pred min", sl.n_pred.min(), "years", sl.date.dt.year.value_counts().sort_index().to_dict())
r = sl.groupby(["anon_id", "date"]).p_dys.max().reset_index()
m = r.merge(e, on=["anon_id", "date"], how="inner")
nxt = e[["anon_id", "date", "dysplasia", "worst"]].rename(columns={"date": "d2", "dysplasia": "dys2", "worst": "w2"})
pairs = []
for pid, d in m.groupby("anon_id"):
    for _, row in d.sort_values("date").iterrows():
        n2 = nxt[(nxt.anon_id == pid) & (nxt.d2 > row.date)].sort_values("d2")
        if len(n2): pairs.append((pid, GRADES[row.worst], row.p_dys, n2.dys2.iloc[0], (n2.d2.iloc[0] - row.date).days))
p = pd.DataFrame(pairs, columns=["pid", "g", "p", "dys2", "gap"])
print("slide-endoscopies with a later endoscopy", len(p), "patients", p.pid.nunique())
print(p.groupby("g").agg(n=("pid", "size"), pts=("pid", "nunique"), next_dys=("dys2", "sum")).to_string())
