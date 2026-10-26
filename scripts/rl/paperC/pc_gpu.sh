#!/bin/bash
# Paper C GPU stages (item 5: Qwen3-8B in thinking mode, zero-shot, with and without the retrieval sentence).
# Usage: bash pc_gpu.sh STAGE  (think_smoke | think_plain | think_r). Submit via race3.sh with prefix pC2_. Eval code: explore/pert/xpert_eval.py
# (THINK=1: capped thinking trace, then answer-word probabilities). Data: $X/data/test_sub.jsonl (1 in 4 test perturbations, seed 0).
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl/explore/pert
BASE=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
export THINK=1 MAXTHINK=${MAXTHINK:-400} MAXMODEL=1280 BASE
case $1 in
  think_smoke) PROMPT=r TAG=pc2_smoke_think_r SETS=test_sub N=40 python $C/xpert_eval.py
               python - <<'PY'
import json
r = [json.loads(l) for l in open("/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_explore/pert/preds/pc2_smoke_think_r__test_sub.jsonl")]
acc = sum(max(("up", "down", "none"), key=lambda w: x["p_" + w]) == x["label"] for x in r) / len(r); mass = sum(x["mass"] for x in r) / len(r)
closed = sum(x["think"].rstrip().endswith("</think>") for x in r) / len(r)
print("smoke n", len(r), "3-class acc", round(acc, 3), "mass", round(mass, 3), "trace closed by itself", round(closed, 2)); print(r[0]["think"][:600])
PY
               echo PC2_SMOKE_OK ;;
  think_plain) PROMPT=plain TAG=pc2_think_plain SETS=test_sub python $C/xpert_eval.py; echo PC2_THINK_PLAIN_OK ;;
  think_r)     PROMPT=r TAG=pc2_think_r SETS=test_sub python $C/xpert_eval.py; echo PC2_THINK_R_OK ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
