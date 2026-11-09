"""One-paper Fig. 4a-b on the artefact-free (clean-tile) H&E model. CPU, cluster only; aggregates only.
Runs rv9_falseflags.py unchanged (module executed with TASK=none, so only its definitions run), then executes the body of its TASK=icc block
verbatim and the R49 part of its TASK=curves block (same setup(), refs(), curve(), pdraws(), dump9; the curves block's extra 'best memory rule'
is skipped because it needs the round-9 merge files). The only change is which L-IMG predictions setup() loads:
  MODE=alltile  stored L-IMG (must reproduce review_round9/icc.json and curves.json R49)
  MODE=clean    agent ART clean-tile L-IMG (two_papers_ART/refit/clean, same folds); every gate cut-off re-chosen on the clean training folds
  MODE=cleanA   clean L-IMG scores, but the gate cut-offs (t_low, c*, t_seq) taken from the all-tile training folds (TC.triage replaced by the all-tile INFO)
L-LATE = mean of stored L-CNV and the L-IMG in use, as in rv9 load_refit_var. TASK=icc | curves.
Output: results/paper_final/one_paper/fig4/<MODE>/<TASK>.json"""
import os, sys, glob, json, textwrap
MODE = os.environ["MODE"]; TASK_ = os.environ["TASK"]; assert MODE in ("alltile", "clean", "cleanA") and TASK_ in ("icc", "curves")
HERE = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"; sys.path.insert(0, HERE)
SRC = open(HERE + "/rv9_falseflags.py").read()
os.environ["TASK"] = "none"; __file__ = HERE + "/rv9_falseflags.py"
exec(compile(SRC, HERE + "/rv9_falseflags.py", "exec"))
TASK = TASK_
OUT = T + "/results/paper_final/one_paper/fig4/" + MODE; os.makedirs(OUT, exist_ok=True)
ART = M + "/two_papers_ART/refit/clean"


def load_clean(src):
    O, I, FO = _load0("their")
    oo, fo = TC._outer(sorted(glob.glob(ART + "/outer_rep_*.csv"))); assert (fo == FO).all()
    ii = TC._inner(sorted(glob.glob(ART + "/inner_rep_*.csv")))
    O["L-IMG"] = oo; O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2; I["L-IMG"] = ii
    I["L-LATE"] = {k: (I["L-CNV"][k] + ii[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    for k in I["L-CNV"]: assert ii[k].index.sort_values().equals(I["L-CNV"][k].index.sort_values()), k
    return O, I, FO


if MODE == "cleanA":   # all-tile cut-offs: INFO from the stored model, then clean scores
    S0 = setup(); INFO_ALL = TC.triage(S0["O"], S0["I"], S0["FO"], S0["popm"], 0.95, 0.90)[2]; del S0
    _tri = TC.triage; TC.triage = lambda *a, **k: (None, None, INFO_ALL)
if MODE in ("clean", "cleanA"):
    load = load_clean
res = {"task": TASK, "mode": MODE}
if TASK == "icc":
    a = SRC.index('elif TASK == "icc":\n') + len('elif TASK == "icc":\n'); b = SRC.index("# ================================================================== cumulative")
    exec(compile(textwrap.dedent(SRC[a:b]), "rv9_icc_block", "exec"))
else:
    S = setup(); D = pdraws(S); RF = refs(S); out = {}
    F, NSr = RF["R49"]; cv = curve(S, F, D); out["R49"] = {}
    for (kk, rst), r in cv.items():
        e = {}
        for cl in ("np", "prog"):
            ob, dr, npat = r[cl]; ok = npat >= 5
            e[cl] = {"patients": m5(npat), "share": r3(ob) if ok else "<5", "ci95": pct(dr[np.isfinite(dr)]) if ok else "<5"}
        out["R49"].setdefault(f"k={kk}", {})["restricted_to_patients_with_ge_k_endoscopies" if rst else "all_patients_first_k_endoscopies"] = e
    A_ = patient_arrays(S, F, NSr); isP = S["plab"] == 1
    out["R49_all_follow_up"] = {"np_patients_ever_flagged": r2((A_["NF"][~isP] > 0).sum(0).mean())}
    res["curves"] = {"curves": out, "note": "rv9 curves block, R49 only; " + MODE}; dump9(res)
