#!/bin/bash
# Probe strand stages (explore/probe). Never edit while jobs run (bash reads scripts lazily).
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 30000))
B=/mnt/scratche/slow/fmlab/zuberi01; C=$B/phd/thesis/scripts/rl; P=$C/explore/probe; R=$B/analysis/rl_grader; O=$B/analysis/rl_explore/probe
P1=$R/runs/p1_best_merged; D=$R/data; mkdir -p $O; chmod 700 $O
case $1 in
  smoke)    SETS=val N=24 OUTD=$O/smoke python $P/extract.py ;;
  extract)  SETS=$2 python $P/extract.py ;;
  # part 2: own 8-sample accuracy of the p1 model on the training items (vLLM, T=1), then GRPO with the self-knowledge reward
  sample)   SETS=$2 OUTD=$O/selfk python $P/self_sample.py ;;
  sk_train) REWARD=selfk MODE=grpo BASE=$P1 OUT=$O/selfk/runs/sk STEPS=300 EXTRA=$D/all_lowconf.jsonl KEEP_EXTRA=1 OWNACC=$O/selfk/ownacc.json python $P/train_selfk.py ;;
  sk_ckpt)  UNSURE=1 LOGPROBS=1 BASE=$P1 ADAPTER=$O/selfk/runs/sk/$2 TAG=sk__$2 SETS=val PREDS=$O/selfk/preds python $C/evaluate.py ;;
  sk_best)  UNSURE=1 LOGPROBS=1 BASE=$P1 ADAPTER=$O/selfk/runs/sk/$2 TAG=sk__$2 SETS=val,pop,test485 PREDS=$O/selfk/preds python $C/evaluate.py ;;
  sk_smoke) SM=$O/selfk/smoke; rm -rf $SM; mkdir -p $SM
            SETS=val N=40 OUTD=$SM python $P/self_sample.py 2>&1 | tail -3
            REWARD=selfk MODE=grpo BASE=$P1 OUT=$SM/sk STEPS=5 N_TRAIN=40 OWNACC=$SM/ownacc.json python $P/train_selfk.py 2>&1 | grep -E "selfk|'reward'|TRAIN DONE|Error|Traceback" | tail -5
            echo SK_SMOKE_OK ;;
esac
