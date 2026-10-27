"""External temporal test (GPU, ~15 min): the completion-masked SFT MedGemma-4B LoRA adapters (analysis/paperA/sft/runs/sftm_s*/final,
Paper A §3, 3 seeds) on the SAME pre-Oct-2014 Barrett's-DB report set the frozen encoders were scored on (db_encoder.py), vs the
database's confirmed grade and the whole-report jury. READ-ONLY on sources.
Report set, exclusion, record/jury labels, year and the multi-part split are taken from external_db/grades.csv written by db_encoder.py
(not re-derived); the text helpers (body / mapped / lettered / parts_of) are copied verbatim from db_encoder.py and the part count is
asserted equal to grades.csv's n_parts. The model is prompted exactly as in training (common.make_messages, greedy, parse_grade):
once with section WHOLE on the ERIN-layout text and once per re-lettered part for multi-part reports; report grade = worst part, else
WHOLE (the encoder's rule). NORMAL_OTHER folded into NDBE; a parse failure is kept as None (counted wrong).
Env: SEEDS (default 0,1,2), N (smoke cap on reports), OUTSUB (output subfolder), VLLM_MEM, MAXLEN (default 8192).
Out: O/external_db[/OUTSUB]/{sft_results.json, sft_grades.csv}. Prints aggregates only."""
import json, os, re, sys, time
from vllm import LLM, SamplingParams                      # vLLM before any HF CUDA init
from vllm.lora.request import LoRARequest
import numpy as np, pandas as pd
os.environ["HF_HOME"] = "/mnt/scratche/slow/fmlab/zuberi01/hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
from pa_common import *
from common import make_messages, parse_grade
from sklearn.metrics import cohen_kappa_score
# ---- text helpers, verbatim from db_encoder.py ----
def body(t): return re.sub(r"^SPECIMEN OF INTEREST.*?\n\n", "", t, flags=re.S)
HEAD = ["CLINICAL DETAILS", "NATURE OF SPECIMEN", "MACROSCOPIC DESCRIPTION", "MICROSCOPY FINDINGS", "MICROSCOPY COMMENT"]
HRX = re.compile(r"(?m)^\s*(" + "|".join(HEAD) + r")\s*:?\s*$")
def split_heads(t):
    parts = {}; cur = "_pre"; parts[cur] = []
    for line in t.split("\n"):
        m = HRX.match(line)
        if m: cur = m.group(1); parts.setdefault(cur, []); continue
        parts.setdefault(cur, []).append(line)
    return {k: "\n".join(v).strip() for k, v in parts.items()}
def mapped(t):
    p = split_heads(t)
    if not any(h in p for h in HEAD[1:]): return "GROSS DESCRIPTION:\n\n\nFINAL DIAGNOSIS:\n" + t.strip() + "\n\nMICROSCOPIC DESCRIPTION:\n"
    gross = "\n".join(x for x in [p.get("CLINICAL DETAILS", ""), p.get("NATURE OF SPECIMEN", ""), p.get("MACROSCOPIC DESCRIPTION", "")] if x)
    return f"GROSS DESCRIPTION:\n{gross}\n\nFINAL DIAGNOSIS:\n{p.get('MICROSCOPY COMMENT', '')}\n\nMICROSCOPIC DESCRIPTION:\n{p.get('MICROSCOPY FINDINGS', '')}{(chr(10) + p['_pre']) if p.get('_pre') else ''}"
NUMMARK = re.compile(r"(?m)(^[ \t]*\(?)(\d{1,2})(\s?[.):\]]\s*)(?=\S)"); PARTREF = re.compile(r"(?i)\b(part|specimen|pot|biopsy)(s?\s*)(\d{1,2})\b")
L = lambda n: chr(64 + int(n)) if 1 <= int(n) <= 10 else str(n)
def lettered(t): return PARTREF.sub(lambda m: m.group(1) + m.group(2) + L(m.group(3)), NUMMARK.sub(lambda m: m.group(1) + L(m.group(2)) + m.group(3), t))
def parts_of(t):
    ns = {int(m.group(2)) for m in NUMMARK.finditer(t)}; ns = sorted(n for n in ns if 1 <= n <= 10)
    return ns if len(ns) >= 2 and ns[0] == 1 and ns[1] == 2 else []
# ---- the encoder's report set (db_encoder.py output) + text ----
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]; N = int(os.environ.get("N", 0)); MAXLEN = int(os.environ.get("MAXLEN", 8192))
S = mkdir(O + "/external_db" + (("/" + os.environ["OUTSUB"]) if os.environ.get("OUTSUB") else "")); EXP = B + "/barretts_db_export"
GR = ["NDBE", "IND", "LGD", "HGD", "CANCER"]; O5 = {g: i for i, g in enumerate(GR)}; fold = lambda g: "NDBE" if g == "NORMAL_OTHER" else g
ENC = [c for c in ("full_s0", "full_s1", "full_s2")]
ev = pd.read_csv(O + "/external_db/grades.csv", dtype={"report_id": str, "participant_id": str})
pt = pd.read_csv(EXP + "/pathology_text_normalised_full.csv", dtype=str); pt["report_id"] = pt.pathology_text_id.astype(str)
ev = ev.merge(pt[["report_id", "reporttext"]].drop_duplicates("report_id"), on="report_id", how="left"); assert ev.reporttext.notna().all()
if N: ev = ev.head(N).copy()
items = []
for r in ev.itertuples():
    t = mapped(body(str(r.reporttext).replace("\r", "\n"))); items.append((f"{r.report_id}|whole", "WHOLE", t)); ps = parts_of(t)
    assert len(ps) == r.n_parts, (r.report_id, len(ps), r.n_parts)   # same sectioning as the encoder run
    if ps:
        lt = lettered(t)
        for n in ps: items.append((f"{r.report_id}|part{n}", L(n), lt))
print(f"{len(ev)} reports, {len(items)} prompts, {int((ev.n_parts > 0).sum())} multi-part reports", flush=True)
# ---- vLLM, one base, three LoRA adapters ----
BASE = os.environ["BASE"]; AD = {s: f"{O}/sft/runs/sftm_s{s}/final" for s in SEEDS}
llm = LLM(model=BASE, max_model_len=MAXLEN, gpu_memory_utilization=float(os.environ.get("VLLM_MEM", 0.85)), dtype="bfloat16", seed=0, enable_lora=True, max_lora_rank=16,
          enforce_eager=True, limit_mm_per_prompt={"image": 0})   # CUDA graphs corrupt Gemma 3 on this driver (evaluate.py)
tok = llm.get_tokenizer(); msgs = [make_messages(t, s) for _, s, t in items]
ntok = [len(tok.apply_chat_template(m, add_generation_prompt=True)) for m in msgs]; keep = [i for i, n in enumerate(ntok) if n + 384 <= MAXLEN]
R = {"n_prompts": len(items), "prompt_tokens_max": int(max(ntok)), "prompts_over_3584_tokens": int(sum(n + 384 > 3584 for n in ntok)), "prompts_dropped_too_long": len(items) - len(keep)}
greedy = SamplingParams(temperature=0.0, max_tokens=384)
for s in SEEDS:
    t0 = time.time(); res = llm.chat([msgs[i] for i in keep], greedy, use_tqdm=False, lora_request=LoRARequest(f"sftm_s{s}", s + 1, AD[s])); dt = time.time() - t0
    pred = {items[i][0]: parse_grade(o.outputs[0].text) for i, o in zip(keep, res)}
    tag = f"sftm_s{s}"; R[f"{tag}_timing"] = {"prompts": len(keep), "seconds": round(dt, 1)}
    R[f"{tag}_parse_fail"] = {"prompts": int(sum(v is None for v in pred.values())), "raw_examples_len": [len(o.outputs[0].text) for o in res[:3]]}
    ev[f"{tag}_whole"] = ev.report_id.map(lambda r: fold(pred.get(f"{r}|whole")))
    ev[f"{tag}_parts"] = [worst([g for n in range(1, 11) if (g := fold(pred.get(f"{r.report_id}|part{n}"))) is not None]) if r.n_parts else None for r in ev.itertuples()]
    ev[tag] = ev[f"{tag}_parts"].where(ev[f"{tag}_parts"].notna(), ev[f"{tag}_whole"])   # worst lettered section, else whole report
    print(tag, "scored", f"{dt:.0f}s, parse fails {R[f'{tag}_parse_fail']['prompts']}", flush=True)
tags = [f"sftm_s{s}" for s in SEEDS]
# ---- metrics, as db_encoder.py ----
TIER = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 3}
def qwk(p, t):
    ok = pd.Series(p).isin(GR).values; return float(cohen_kappa_score([O5[x] for x in t[ok]], [O5[x] for x in p[ok]], weights="quadratic")) if len(set(t)) > 1 else float("nan")
def metrics(d, col, bs):
    p, t = d[col].values, d.record.values; ex = (p == t).astype(float); tier = (d[col].map(TIER).values == d.record.map(TIER).values).astype(float)
    rec = {g: (p[t == g] == g).astype(float) for g in GR}; po, to = d[col].map(O5), d.record.map(O5)
    return {"n": int(len(d)), "patients": int(d.participant_id.nunique()), "exact": round(float(ex.mean()), 4), "exact_ci": ci([ex[ix].mean() for ix in bs]), "qwk": round(qwk(p, t), 4),
            "tier_exact": round(float(tier.mean()), 4), "tier_ci": ci([tier[ix].mean() for ix in bs]), "LGDplus_sensitivity": round(float((po >= 2)[to >= 2].mean()), 4),
            "over": round(float((po > to).mean()), 4), "under": round(float((po < to).mean()), 4), "unparsed_reports": int(d[col].isna().sum()),
            "recall": {g: [int(len(rec[g])), round(float(rec[g].mean()), 4)] for g in GR},
            "recall_ci": {g: ci([(p[ix][t[ix] == g] == g).mean() for ix in bs]) for g in ("LGD", "HGD")},
            "confusion_truth_rows_pred_cols": pd.crosstab(d.record, d[col]).reindex(index=GR, columns=GR, fill_value=0).values.tolist()}
def paired(d, a, b, bs):
    x = (d[a].values == d.record.values).astype(float) - (d[b].values == d.record.values).astype(float); return {"n": int(len(d)), "delta": round(float(x.mean()), 4), "ci": ci([x[ix].mean() for ix in bs])}
bs_all = patient_bootstrap(ev.participant_id.values); wj = ev[ev.jury.notna()]; bs_j = patient_bootstrap(wj.participant_id.values)
R["all_kept"] = {c: metrics(ev, c, bs_all) for t in tags for c in (t, f"{t}_whole")}
R["jury_common"] = {c: metrics(wj, c, bs_j) for c in tags + ["jury"]}
R["paired_minus_jury"] = {t: paired(wj, t, "jury", bs_j) for t in tags}
R["paired_minus_encoder_same_seed"] = {t: paired(ev, t, f"full_s{t[-1]}", bs_all) for t in tags}
R["paired_minus_encoder_all"] = {f"{t}-{e}": paired(ev, t, e, bs_all) for t in tags for e in ENC}
R["paired_parts_minus_whole"] = {t: paired(ev, t, f"{t}_whole", bs_all) for t in tags}
R["by_year"] = {int(y): {"n": int(len(g)), "n_jury": int(g.jury.notna().sum()), **{t: round(float((g[t] == g.record).mean()), 4) for t in tags},
                         **{e: round(float((g[e] == g.record).mean()), 4) for e in ENC}, "jury": round(float((g.jury == g.record)[g.jury.notna()].mean()), 4) if g.jury.notna().any() else None}
                for y, g in ev.groupby("year")}
mp = ev[ev.n_parts > 0]; R["multi_part_reports"] = {"n": int(len(mp)), "exact_parts_vs_whole": {t: [round(float((mp[t] == mp.record).mean()), 4), round(float((mp[f"{t}_whole"] == mp.record).mean()), 4)] for t in tags}}
json.dump(R, open(S + "/sft_results.json", "w"), indent=1, default=lambda o: None); os.chmod(S + "/sft_results.json", 0o600)
ev[["report_id", "participant_id", "year", "record", "jury", "n_parts"] + ENC + [c for t in tags for c in (t, f"{t}_whole", f"{t}_parts")]].to_csv(S + "/sft_grades.csv", index=False); os.chmod(S + "/sft_grades.csv", 0o600)
def row(nm, m): return f"{nm:14s} n={m['n']:5d} exact {100*m['exact']:.1f} {[round(100*x,1) for x in m['exact_ci']]} QWK {m['qwk']:.3f} tier {100*m['tier_exact']:.1f} LGD rec {100*m['recall']['LGD'][1]:.1f} HGD rec {100*m['recall']['HGD'][1]:.1f} over {100*m['over']:.1f} under {100*m['under']:.1f} unparsed {m['unparsed_reports']}"
print(json.dumps({k: R[k] for k in R if k not in ("all_kept", "jury_common", "by_year")}))
for blk in ("all_kept", "jury_common"):
    print("==", blk)
    for nm, m in R[blk].items(): print(row(nm, m))
print("== by year"); [print(y, v) for y, v in R["by_year"].items()]
print("DBSFT DONE")
