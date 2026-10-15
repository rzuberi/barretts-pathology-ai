"""Item 6b, ACE-B power for the triage criterion (docs/paper_triage_robustness.md @ 32653f1). TASK = S2 (15 P / 67 NP patients) | owner (11 P / 93 NP).
Frozen cut-offs (frozen_cutoffs_their.json) on each internal pre-event sample's repeat-mean held-out L-IMG / L-LATE; Delta = sensitivity(triage) - sensitivity(B).
E1 = internal Delta; E2 = its one-sided 95% patient-bootstrap lower bound, imposed by turning triage-positive / B-negative progressor samples negative with prob q.
2,000 simulations (seed = simulation index), each tested with 2,000 patient-bootstrap draws (RandomState(simulation index)); power = P(lower bound > -0.05)."""
import os, json, numpy as np
from tb_common import *
TASK = os.environ["TASK"]; NPN = {"S2": (15, 67), "owner": (11, 93)}[TASK]; NS = 2000
F = json.load(open(f"{OUT}/frozen_cutoffs_their.json"))["cutoffs"]; O, I, FO = load("their"); popm = popmask("pre"); rows = np.where(popm)[0]
img = O["L-IMG"].mean(1); late = O["L-LATE"].mean(1)
tri = np.where(img < F["t_low"], False, np.where(img > F["t_high"], True, late >= F["t_seq"])); bpos = late >= F["t_B"]
pp = pat[rows]; up = np.unique(pp); ip = {q: rows[pp == q] for q in up}; Pp = [q for q in up if y[ip[q]].max() == 1]; Np = [q for q in up if y[ip[q]].max() == 0]
st = {q: (len(ip[q]), int(tri[ip[q]].sum()), int(bpos[ip[q]].sum()), int((tri[ip[q]] & ~bpos[ip[q]]).sum())) for q in Pp}   # progressor patients: n samples, triage+, B+, discordant
def delta_from(stats): s = np.asarray(stats, float); return (s[:, 1].sum() - s[:, 2].sum()) / s[:, 0].sum()
E1 = delta_from([st[q] for q in Pp]); rng = np.random.RandomState(0); bl = []
while len(bl) < 2000:
    dr = rng.choice(len(up), len(up)); dp = [up[i] for i in dr if up[i] in st]
    if dp and len(dp) < len(dr): bl.append(delta_from([st[q] for q in dp]))
E2 = float(np.percentile(bl, 5)); disc = sum(st[q][3] for q in Pp); ntot = sum(st[q][0] for q in Pp); q_flip = (E1 - E2) * ntot / disc if disc else 0.0
res = {"task": TASK, "prespec": "32653f1", "n_P": NPN[0], "n_NP": NPN[1], "simulations": NS, "frozen_cutoffs": F, "internal_delta_E1": r3(E1), "internal_lower_E2": r3(E2), "q_flip_E2": r3(q_flip),
       "internal_sens_triage": r3(tri[rows][y[rows] == 1].mean()), "internal_sens_B": r3(bpos[rows][y[rows] == 1].mean()), "internal_spec_triage": r3(1 - tri[rows][y[rows] == 0].mean()), "internal_spec_B": r3(1 - bpos[rows][y[rows] == 0].mean()),
       "internal_share_sequenced": r3(((img >= F["t_low"]) & (img <= F["t_high"]))[rows].mean()), "effects": {}}
for eff, q in (("E1", 0.0), ("E2", q_flip)):
    succ, dl, lbs = [], [], []
    for sim in range(NS):
        r = np.random.RandomState(sim); dP = [Pp[i] for i in r.choice(len(Pp), NPN[0])]; dN = r.choice(len(Np), NPN[1])
        S = np.array([st[qq] for qq in dP], float)
        if q > 0: fl = r.binomial(S[:, 3].astype(int), q); S[:, 1] -= fl                         # E2: discordant triage-positive progressor samples turned negative
        Z = np.vstack([S[:, :3], np.zeros((NPN[1], 3))]); d0 = (Z[:, 1].sum() - Z[:, 2].sum()) / Z[:, 0].sum(); rb = np.random.RandomState(sim)
        idx = rb.randint(0, len(Z), (2000, len(Z))); sums = Z[idx].sum(1); ok = sums[:, 0] > 0; d = (sums[ok, 1] - sums[ok, 2]) / sums[ok, 0]   # draws with no progressor sample dropped
        lb = np.percentile(d, 5); succ.append(lb > -0.05); dl.append(d0); lbs.append(lb)
    pw = float(np.mean(succ)); se = np.sqrt(pw * (1 - pw) / NS); z = 1.96; den = 1 + z * z / NS; cen = (pw + z * z / (2 * NS)) / den; hw = z * np.sqrt(pw * (1 - pw) / NS + z * z / (4 * NS * NS)) / den
    res["effects"][eff] = {"target_delta": r3(E1 if eff == "E1" else E2), "power": r3(pw), "power_ci95_wilson": [r3(cen - hw), r3(cen + hw)], "median_simulated_delta": r3(np.median(dl)), "median_lower_bound": r3(np.median(lbs))}
json.dump(res, open(f"{OUT}/power_{TASK}.json", "w"), indent=1); print("TB POWER DONE", TASK, res["effects"], flush=True)
